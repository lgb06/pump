from data_provider.data_factory import data_provider
from utils.tools import adjust_learning_rate, cal_accuracy, adjustment
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from utils.metrics import metric
from utils.losses import mape_loss, mase_loss, smape_loss,FocalLoss
from utils.dataloader import BalancedDataLoaderIterator
from utils.layer_decay import param_groups_lrd
from utils.ddp import get_world_size, is_main_process, gather_tensors_from_all_gpus
import torch.nn.functional as F
from models.layers.LoRA import LinearWithLoRAMerged
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import pandas as pd
from exp.exp_pretrain import custom_print_decorator
import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist
from tqdm import tqdm
import os
import time
import warnings
import numpy as np
import yaml
# import wandb
import sys
import copy

warnings.filterwarnings('ignore')
 

def apply_random_mask_for_imputation(x, patch_len, mask_rate):
    """
    Apply a random mask to the input tensor.

    Parameters:
    x (torch.Tensor): The input tensor with shape [B, T, N].
    patch_len (int): The length of each patch.
    mask_rate (float): The proportion of the tensor to be masked.

    Returns:
    torch.Tensor: The masked input tensor.
    torch.Tensor: The mask tensor.
    """
    B, T, N = x.shape
    num_keep = int((T // patch_len) * (1 - mask_rate))

    # Generate random noise and sort it
    noise = torch.rand(B, T // patch_len, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Select indices to keep
    ids_keep = ids_shuffle[:, :num_keep]
    mask = torch.zeros([B, T], device=x.device)
    mask[:, :num_keep] = 1
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # Expand the mask to the original shape
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_len).view(B, T)
    mask = mask.unsqueeze(-1).repeat(1, 1, N)

    # Apply the mask
    x_masked = x.masked_fill(mask == 0, 0)

    return x_masked, mask


# Replace print to save all print into log files
print = custom_print_decorator(print)


def read_task_data_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    task_dataset_config = config.get('task_dataset', {})
    return task_dataset_config


def get_task_data_config_list(task_data_config, default_batch_size=None):
    task_data_config_list = []

    for task_name, task_config in task_data_config.items():
        task_config['max_batch'] = default_batch_size
        task_data_config_list.append([task_name, task_config])

    return task_data_config_list



def get_loss_by_name(loss_name):
    if loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'MAPE':
        return mape_loss()
    elif loss_name == 'MASE':
        return mase_loss()
    elif loss_name == 'SMAPE':
        return smape_loss()
    elif loss_name == 'CE':
        return nn.CrossEntropyLoss()
    elif loss_name == 'Focal':
        return FocalLoss()
    else:
        print("no loss function found!")
        exit()


def init_and_merge_datasets(data_loader_list):
    dataloader = BalancedDataLoaderIterator(data_loader_list)
    train_steps = dataloader.__len__()
    return dataloader, train_steps


class Exp_All_Task(object):
    def __init__(self, args):
        super(Exp_All_Task, self).__init__()
        self.args = args
        self.task_data_config = read_task_data_config(
            self.args.task_data_config_path)
        #
        # self.test_task_data_config = read_task_data_config(
        #     "data_provider/data_config/pump/NLNEMP.yaml" )
        # #        
        self.task_data_config_list = get_task_data_config_list(
            self.task_data_config, default_batch_size=self.args.batch_size)
        if args.ddp:
            device_id = dist.get_rank() % torch.cuda.device_count()
            print("this device_id:", device_id)
        else:
            device_id = args.device
        self.device_id = device_id
        print("device id", self.device_id)
        self.model = self._build_model()
        # Initialize training history storage
        self.training_history = {'train_loss': [], 'train_loss_PHM_ROT': [], 
                                'eval_CLS-acc_PHM_ROT': [], 'val_acc': []}

    def _build_model(self, ddp=True):
        if ddp==False: 
            ddp = False
        else:
            ddp = self.args.ddp
        import importlib
        module = importlib.import_module("models."+self.args.model)
        # print(self.args.__dict__) # 会打印出对象的属性及其值
        model = module.Model(
            self.args, self.task_data_config_list).to(self.device_id)
        if ddp:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device_id],
                                                        find_unused_parameters=True, gradient_as_bucket_view=True, static_graph=False)        
        #删除掉model的pretrain_head
        # if 'pretrain_head' in model.__dict__:
        try:
            del model.pretrain_head
        except:
            pass
        return model

    def _get_data(self, flag):
        ddp = self.args.ddp

        # 不可以，因为args也会影响数据集读取 /*TODO*/
        # if flag == 'test':
        #     this_task_data_config = self.test_task_data_config
        # else:
        this_task_data_config = self.task_data_config
        #
            
        data_set_list = []
        data_loader_list = []

        for task_data_name, task_config in this_task_data_config.items():
            if task_config['task_name'] == 'classification' and flag == 'val':
                # TODO strange that no val set is used for classification. Set to test set for val
                flag = 'test'
            else:
                data_set, data_loader = data_provider(
                    self.args, task_config, flag, ddp=ddp)
                data_set_list.append(data_set)
                data_loader_list.append(data_loader)
                print(task_data_name, len(data_set))
        return data_set_list, data_loader_list

    def _select_optimizer(self):
        if self.args.ddp:
            world_size = get_world_size()
        else:
            world_size = 1    
        eff_batch_size = self.args.batch_size * self.args.acc_it * get_world_size()
        real_learning_rate = self.args.learning_rate * eff_batch_size / 32
        # self.real_learning_rate = min(real_learning_rate,0.0003)
        self.real_learning_rate = real_learning_rate
        print("base lr: %.2e" % (self.args.learning_rate * 32 / eff_batch_size))
        print("actual lr: %.2e" % real_learning_rate)

        print("accumulate grad iterations: %d" % self.args.acc_it)
        print("effective batch size: %d" % eff_batch_size)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found. Check tuning settings.")
        if self.args.layer_decay is not None:
            print("layer decay: %.2f" % self.args.layer_decay)
            if self.args.ddp:
                model_without_ddp = self.model.module
            else:
                model_without_ddp = self.model
            param_groups = param_groups_lrd(model_without_ddp, self.args.weight_decay,
                                            no_weight_decay_list=[
                                                'prompts', 'mask_tokens', 'cls_tokens', 'category_tokens'],
                                            layer_decay=self.args.layer_decay
                                            )
            model_optim = optim.Adam(param_groups, lr=real_learning_rate)
        else:
            model_optim = optim.Adam(trainable_params,
            lr=real_learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self, config_list):
        criterion_list = []
        for each_config in config_list:
            if 'loss' in each_config[1]:
                loss_name = each_config[1]['loss']
            else:
                if each_config[1]['task_name'] == 'long_term_forecast':
                    loss_name = 'MSE'
                elif 'classification' in each_config[1]['task_name']:
                    loss_name = 'CE'
                elif each_config[1]['task_name'] == 'imputation':
                    loss_name = 'MSE'
                elif each_config[1]['task_name'] == 'anomaly_detection':
                    loss_name = 'MSE'
                elif 'RUL' in each_config[1]['task_name'] :
                    loss_name = 'MSE'
                else:
                    print("this task has no loss now!", folder=self.path)
                    exit()
            criterion_list.append(get_loss_by_name(loss_name))

        return criterion_list

    def choose_training_parts(self, prompt_tune=False, head_tune=False):
        model_param = []
        trainable_param = []
        #通过控制可训练参数的梯度来控制训练部分
        for name, param in self.model.named_parameters():
            if head_tune:
                if any(keyword in name for keyword in ['cls_head', 'category', 'rul_head', 'global_token']):
                    param.requires_grad = True
                    print("head_tuning trainable:", name)
                else:
                    param.requires_grad = False
            elif prompt_tune:
                #仅仅训练网络中保存的任务参数
                if any(keyword in name for keyword in 
                       ['rul',  'category','prompt','lora','global','cls_head']):
                    param.requires_grad = True
                    if 'lora' not in name:
                        print("trainable:", name)
                else:
                    param.requires_grad = False
            else:
                if any(keyword in name for keyword in 
                       ['quantizer']) and self.args.tokenizer_path is not None:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_param.append(param.numel())
            model_param.append(param.numel())

        trainable_total_param = sum(trainable_param)
        model_total_params = sum(model_param)

        print("Parameters number for total RmGPT {} M, trainable number {} M, trainable ratio {:.2%}".format(
            model_total_params / 1e6, trainable_total_param / 1e6,
            trainable_total_param / model_total_params))
    
    def load_model_from_pretrain(self,setting):
        # Load pretrained weights (Optional),如果有预训练权重，加载预训练权重；读取相同名字的部分都权重
        if self.args.pretrained_weight is not None:
            if self.args.pretrained_weight == 'auto':
                pretrain_weight_path = os.path.join(
                    self.path, 'pretrain_checkpoint.pth')
            else:
                pretrain_weight_path = self.args.pretrained_weight
            print('loading pretrained model:',
                  pretrain_weight_path, folder=self.path)
            if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                state_dict = torch.load(
                    pretrain_weight_path, map_location='cpu', weights_only=False)['student']
                ckpt = {}
                for k, v in state_dict.items():
                    if not ('cls_prompts' in k) and not ('category' in k):
                        ckpt[k] = v
            else:
                ckpt = torch.load(pretrain_weight_path, map_location='cpu')
            msg = self.model.load_state_dict(ckpt, strict=False)
            print(msg, folder=self.path)

    def lora_transform(self, rank=4, alpha=8):
        #lora改造
        model = copy.deepcopy(self.model)
        for name, module in model.named_modules():
            if 'blocks' in name and isinstance(module, nn.Linear):
                parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]]
                setattr(parent_module, name.rsplit('.', 1)[-1], LinearWithLoRAMerged(module, rank=rank, alpha=alpha))
        self.model = model.to(self.device_id)
        print("LoRA transform done!", folder=self.path)

    def train_parameter_choice(self):
        if self.args.lora_transform:
            self.lora_transform()
        if self.args.lradj == 'head_tuning':
            self.choose_training_parts(head_tune=True)
        elif self.args.efficiency_tuning:
            self.choose_training_parts(prompt_tune=True)

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and is_main_process():
            os.makedirs(path)
        self.path = path
        self.load_model_from_pretrain(setting)
        self.train_parameter_choice()
        # Data
        train_data_list, train_loader_list = self._get_data(flag='train')
        # Since some datasets do not have val set, we use test set and report the performance of last epoch instead of the best epoch.
        test_data_list, test_loader_list = self._get_data(
            flag='test')
        data_loader_cycle, train_steps = init_and_merge_datasets(
            train_loader_list)

        # Optimizer and Criterion
        model_optim = self._select_optimizer()
        criterion_list = self._select_criterion(self.task_data_config_list)
        scaler = NativeScaler()

        # Set up batch size for each task
        if self.args.memory_check:
            self.memory_check(data_loader_cycle, criterion_list)
            torch.cuda.empty_cache()
        if self.args.ddp:
            torch.cuda.synchronize()
            dist.barrier()
        #先prompt后finetune，和args里的设置有关self.category_token[task_data_name]= nn.Parameter(category_token)
        for epoch in tqdm(range(self.args.train_epochs ), desc="Training Out Epochs"):
            adjust_learning_rate(model_optim, epoch,
                                 self.real_learning_rate, self.args)
            train_loss = self.train_one_epoch(
                model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler)

            # we report the results of last epoch and not find the best epoch based on val set, since some datasets do not have val set
            avg_cls_acc_traindata, _ , _ = self.test(
                setting, load_pretrain=False, test_data_list=train_data_list, test_loader_list=train_loader_list)
            
            avg_cls_acc, avg_forecast_mse, avg_forecast_mae = self.test(
                setting, load_pretrain=False, test_data_list=test_data_list, test_loader_list=test_loader_list)

            # save ckpt
            if is_main_process():
                torch.save(self.model.state_dict(),
                               os.path.join(self.path, 'checkpoint.pth'))
            #清空显存占用

            if is_main_process():
                # wandb.log({'Final_LF-mse': avg_forecast_mse,
                #            'Final_LF-mae': avg_forecast_mae, 'Final_CLS-acc': avg_cls_acc})
                print("For training data score: LF-mse: {}, LF-mae: {}, CLS-acc {}".format(avg_forecast_mse,
                                                                               avg_forecast_mae, avg_cls_acc_traindata), folder=self.path)
                print("Final testing data score: LF-mse: {}, LF-mae: {}, CLS-acc {}".format(avg_forecast_mse,
                                                                               avg_forecast_mae, avg_cls_acc), folder=self.path)
                # Store validation accuracy
                for task_id, (test_data, test_loader) in enumerate(zip(test_data_list, test_loader_list)):
                    task_name = self.task_data_config_list[task_id][1]['task_name']
                    if 'classification' in task_name:
                        data_task_name = self.task_data_config_list[task_id][0]
                        if 'eval_CLS-acc_'+data_task_name not in self.training_history:
                            self.training_history['eval_CLS-acc_'+data_task_name] = []
                        # Get accuracy from test results (stored in total_dict)
                        # This will be updated after test completes
                
                # Save training history to file
                import json
                history_file = os.path.join(self.path, 'training_history.json')
                with open(history_file, 'w') as f:
                    json.dump(self.training_history, f, indent=2)

        return self.model

    def train_one_epoch(self, model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler):
        current_device = torch.cuda.current_device()
        train_loss_set = []
        acc_it = max(self.args.acc_it,1.0)
        max_norm = self.args.clip_grad

        self.model.train()
        #如果args里有tokenzier_path，tokenizer 不训练
        try:
            if self.args.tokenizer_path is not None:
                self.model.tokenizer.requires_grad_(False)
                self.model.tokenizer.trainning = False
        except:
            pass
        epoch_time = time.time()
        self.model.zero_grad(set_to_none=True)
        loss_sum = 0
        print_epoch = 1 if self.args.task_name == 'few_shot' else 50

        for i, (sample_init, task_id) in enumerate(tqdm(data_loader_cycle, desc="Training Inner Epoch {}".format(epoch + 1))):
            task_name = self.task_data_config_list[task_id][1]['task_name']
            small_batch_size = self.task_data_config_list[task_id][1]['max_batch']
            if small_batch_size != self.args.batch_size:
                sample_list = self.split_batch(
                    sample_init, small_batch_size, task_name)
                len_sample_list = len(sample_list)
            else:
                sample_list = [sample_init]
                len_sample_list = 1
            len_sample_list = max(len_sample_list, 1.0)
            for sample_idx in range(len_sample_list):
                sample = sample_list[sample_idx]
                if  'classification' in task_name:
                    loss = self.train_classification(
                        self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss_scale = 1.0
                elif  'RUL' in task_name:
                    loss = self.train_RUL(
                            self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss_scale = 1.0

                loss /= acc_it
                loss /= len_sample_list
                if sample_idx < len_sample_list-1:
                    norm_value = scaler(loss*loss_scale, model_optim, clip_grad=max_norm,
                                        parameters=self.model.parameters(), create_graph=False, update_grad=False)
            loss_display = loss.item()*len_sample_list*acc_it
            train_loss_set.append(loss_display)
            #如果 loss 是 nan 或者 inf，跳过这次迭代
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss is invalid: {loss}. Skipping this iteration!--train_one_epoch")
                continue
            ## 当update_grad为True时，会更新梯度
            norm_value = scaler(loss*loss_scale, model_optim, clip_grad=max_norm,
                                parameters=self.model.parameters(), create_graph=False, update_grad=((i + 1) % acc_it == 0))
            if (i+1) % acc_it == 0:
                model_optim.zero_grad()
            torch.cuda.synchronize()

            loss_sum += loss_display
            loss_sum_display = loss_sum

            del sample_init
            del sample_list
            if torch.cuda.memory_reserved(current_device) > 30*1e9:
                torch.cuda.empty_cache()

            if is_main_process():
                # wandb.log(
                #     {'train_loss_'+self.task_data_config_list[task_id][0]: loss_display, 'norm_value': norm_value, "loss_sum": loss_sum_display/(i+1)})
                # Store training history
                task_name = self.task_data_config_list[task_id][0]
                if 'train_loss_'+task_name not in self.training_history:
                    self.training_history['train_loss_'+task_name] = []
                self.training_history['train_loss_'+task_name].append(loss_display)
            if (i + 1) % print_epoch == 0:
                if norm_value == None:
                    norm_value = -1
                if is_main_process():
                    print("\titers: {0}, epoch: {1} | norm: {2:.2f} | loss: {3:.7f} | current_loss: {4} |current task: {5}".format(
                        i + 1, epoch + 1, norm_value, loss_sum_display/(i+1), loss_display, task_name, folder=self.path))

        print("Epoch: {} cost time: {}".format(
            epoch + 1, time.time() - epoch_time), folder=self.path)
        train_loss = np.average(train_loss_set)
        torch.cuda.synchronize()
        if self.args.ddp:
            dist.barrier()

        return train_loss



    def train_classification(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']
        try:
            batch_x, label, padding_mask = this_batch
        except:
            batch_x, label = this_batch
            padding_mask = batch_x

        batch_x = batch_x.float().to(self.device_id)
        padding_mask = batch_x
        label = label.to(self.device_id)
        with torch.cuda.amp.autocast():
            outputs = model(batch_x, padding_mask, None,
                            None, task_id=task_id, task_name=task_name)
            if outputs.shape[0] == label.shape[0]:  #[B, num_class]     
                # loss = criterion(outputs, label.float().squeeze(-1)) 
                # ERROR? NotImplementedError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Float'    //NLLLoss 期望的输入是 对数概率，而它期望的 目标（标签） 通常是 类索引（整数类型），而不是浮点数
                print(f"--------==----------dataset_name:{config['dataset_name']}")
                print(f"------------------------------label.shape:{label.shape}")
                loss = criterion(outputs, label.float().squeeze(-1))
            else:       # B不同
                print(f"--------!!==----------dataset_name:{config['dataset_name']}")
                print(f"------------------------------output:{outputs.shape}")
                label = label.repeat(outputs.shape[0]//label.shape[0], 1)
                loss = criterion(outputs, label.float().squeeze(-1))
        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            print(f"Loss is invalid: {loss}. Skipping this iteration!--train_classification")
        return loss


    def get_feature(self, model, this_batch, criterion, config, task_id):
        task_name = 'get_feature'
        try:
             batch_x, label, padding_mask = this_batch
        except:
            batch_x, label = this_batch
            padding_mask = batch_x

        batch_x = batch_x.float().to(self.device_id)
        padding_mask = batch_x
        label = label.to(self.device_id)
        with torch.cuda.amp.autocast():
            outputs = model(batch_x, padding_mask, None,
                            None, task_id=task_id, task_name=task_name)
        return outputs

    def train_RUL(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']
        try:
            batch_x, label, padding_mask = this_batch
        except:
            batch_x, label = this_batch
            padding_mask = torch.zeros_like(batch_x)

        batch_x = batch_x.float().to(self.device_id)
        padding_mask = padding_mask.float().to(self.device_id)
        label = label.to(self.device_id)
        with torch.cuda.amp.autocast():
            outputs = model(batch_x, padding_mask, None,
                            None, task_id=task_id, task_name=task_name)
            if outputs.shape[0] == label.shape[0]:
                loss = criterion(outputs, label.float().squeeze(-1))
            else:
                label = label.repeat(outputs.shape[0]//label.shape[0], 1)
                loss = criterion(outputs, label.long().squeeze(-1))

        return loss



    def train_anomaly_detection(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']
        features = config['features']

        batch_x, _ = this_batch

        batch_x = batch_x.float().to(self.device_id)

        with torch.cuda.amp.autocast():
            outputs = model(batch_x, None, None,
                            None, task_id=task_id, task_name=task_name)
            f_dim = -1 if features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            loss = criterion(outputs, batch_x)

        return loss


    def test(self, setting, load_pretrain=False, test_data_list=None, test_loader_list=None):
        self.path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(self.path) and is_main_process():
            os.makedirs(self.path)
        if test_data_list is None or test_loader_list is None:
            test_data_list, test_loader_list = self._get_data(
                flag='test')
        if load_pretrain:
            # First try to use pretrained_weight if provided
            if self.args.pretrained_weight is not None and os.path.exists(self.args.pretrained_weight):
                pretrain_weight_path = self.args.pretrained_weight
                print('loading pretrained model:',
                      pretrain_weight_path, folder=self.path)
                if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                    state_dict = torch.load(
                        pretrain_weight_path, map_location='cpu', weights_only=False)['student']
                    ckpt = {}
                    for k, v in state_dict.items():
                        if not ('cls_prompts' in k):
                            ckpt[k] = v
                else:
                    ckpt = torch.load(pretrain_weight_path, map_location='cpu')
                msg = self.model.load_state_dict(ckpt, strict=False)
                print(msg)
            # If no pretrained_weight provided, try to load from checkpoint directory
            elif os.path.exists(os.path.join(self.path, 'checkpoint.pth')):
                checkpoint_path = os.path.join(self.path, 'checkpoint.pth')
                print('loading checkpoint from:', checkpoint_path, folder=self.path)
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                msg = self.model.load_state_dict(ckpt, strict=False)
                print(msg)
            else:
                print("no ckpt found! Please provide --pretrained_weight or ensure checkpoint.pth exists in checkpoint directory.")
                exit()

        total_dict = {}
        avg_classification_acc = []
        avg_long_term_forecast_mse = []
        avg_long_term_forecast_mae = []
        avg_imputation_mse = []
        avg_imputation_mae = []
        avg_anomaly_f_score = []
        avg_rul_f_mse = []
        avg_rul_f_mae = []
        for task_id, (test_data, test_loader) in enumerate(zip(test_data_list, test_loader_list)):
            task_name = self.task_data_config_list[task_id][1]['task_name']
            if self.args.task_name =='DG':
                data_task_name = test_data.__class__.__name__
            else:
                data_task_name = self.task_data_config_list[task_id][0]
            if task_name == 'long_term_forecast':
                if self.args.zero_shot_forecasting_new_length=='unify':
                    mse, mae = self.test_long_term_forecast_offset_unify(
                        setting, test_data, test_loader, data_task_name, task_id)
                else:
                    mse, mae = self.test_long_term_forecast(
                        setting, test_data, test_loader, data_task_name, task_id)
                data_task_name = self.task_data_config_list[task_id][0]
                total_dict[data_task_name] = {'mse': mse, 'mae': mae}
                # if is_main_process():
                #     wandb.log({'eval_LF-mse_'+data_task_name: mse})
                #     wandb.log({'eval_LF-mae_'+data_task_name: mae})
                avg_long_term_forecast_mse.append(mse)
                avg_long_term_forecast_mae.append(mae)
            elif 'classification' in task_name:
                acc = self.test_classification(
                    setting, test_data, test_loader, data_task_name, task_id)
                total_dict[data_task_name] = {'acc': acc}
                # if is_main_process():
                #     wandb.log({'eval_CLS-acc_'+data_task_name: acc})
                #     # Store validation accuracy in training history
                #     if 'eval_CLS-acc_'+data_task_name not in self.training_history:
                #         self.training_history['eval_CLS-acc_'+data_task_name] = []
                #     self.training_history['eval_CLS-acc_'+data_task_name].append(acc)
                avg_classification_acc.append(acc)

            elif 'RUL' in task_name:
                mse,mae = self.test_RUL(
                    setting, test_data, test_loader, data_task_name, task_id)
                total_dict[data_task_name] = {'mse': mse,'mae':mae}
                # if is_main_process():
                #     wandb.log({'eval_RUL-mse_'+data_task_name: mse})
                #     wandb.log({'eval_RUL-mae_'+data_task_name: mae})
                avg_rul_f_mse.append(mse)
                avg_rul_f_mae.append(mae)
            elif task_name == 'imputation':
                mse, mae = self.test_imputation(
                    setting, test_data, test_loader, data_task_name, task_id)
                total_dict[data_task_name] = {'mse': mse, 'mae': mae}
                # if is_main_process():
                #     wandb.log({'eval_Imputation-mse_'+data_task_name: mse})
                #     wandb.log({'eval_Imputation-mae_'+data_task_name: mae})
                avg_imputation_mse.append(mse)
                avg_imputation_mae.append(mae)
            elif task_name == 'anomaly_detection':
                f_score = self.test_anomaly_detection(
                    setting, test_data, test_loader, data_task_name, task_id)
                total_dict[data_task_name] = {'f_score': f_score}
                # if is_main_process():
                #     wandb.log({'eval_Anomaly-f_score_' +
                #               data_task_name: f_score})
                avg_anomaly_f_score.append(f_score)

        avg_long_term_forecast_mse = np.average(avg_long_term_forecast_mse)
        avg_long_term_forecast_mae = np.average(avg_long_term_forecast_mae)

        avg_classification_acc = np.average(avg_classification_acc)

        avg_imputation_mse = np.average(avg_imputation_mse)
        avg_imputation_mae = np.average(avg_imputation_mae)

        avg_anomaly_f_score = np.average(avg_anomaly_f_score)

        # if is_main_process():
        #     wandb.log({'avg_eval_LF-mse': avg_long_term_forecast_mse, 'avg_eval_LF-mae': avg_long_term_forecast_mae,
        #                'avg_eval_CLS-acc': avg_classification_acc,
        #                'avg_eval_IMP-mse': avg_imputation_mse, 'avg_eval_IMP-mae': avg_imputation_mae,
        #                'avg_eval_Anomaly-f_score': avg_anomaly_f_score})
        #     print("Avg score: LF-mse: {}, LF-mae: {}, CLS-acc {}, IMP-mse: {}, IMP-mae: {}, Ano-F: {}".format(avg_long_term_forecast_mse,
        #                                                                                                       avg_long_term_forecast_mae, avg_classification_acc, avg_imputation_mse, avg_imputation_mae, avg_anomaly_f_score), folder=self.path)
        #     print(total_dict, folder=self.path)
        print("Avg score: LF-mse: {}, LF-mae: {}, CLS-acc {}, IMP-mse: {}, IMP-mae: {}, Ano-F: {}".format(avg_long_term_forecast_mse,
                                                                                                              avg_long_term_forecast_mae, avg_classification_acc, avg_imputation_mse, avg_imputation_mae, avg_anomaly_f_score), folder=self.path)
        print(total_dict, folder=self.path)
        
        torch.cuda.empty_cache()
        return avg_classification_acc, avg_long_term_forecast_mse, avg_long_term_forecast_mae



    def test_classification(self, setting, test_data, test_loader, data_task_name, task_id):
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, item in enumerate(test_loader):
                try:
                    batch_x, label, condition = item  # batch_x, label, padding_mask = this_batch                     
                except:
                    batch_x, label = item
                    condition = batch_x

                batch_x = batch_x.float().to(self.device_id)
                label = label.to(self.device_id)

                outputs = self.model(
                    batch_x, condition, None, None, task_id=task_id, task_name='classification')
                # outputs = torch.nn.functional.softmax(outputs)
                predictions = torch.argmax(outputs, dim=1)
                # print(f"predictions:{predictions}")
                true = torch.argmax(label, dim=1) # One-Hot 编码的标签
                # true = label nonononono!!
                # print(f"true:{true}")
                preds.append(predictions.detach())
                trues.append(true)
        if self.args.ddp:
            preds = gather_tensors_from_all_gpus(
                preds, self.device_id, to_numpy=False)
            trues = gather_tensors_from_all_gpus(
                trues, self.device_id, to_numpy=False)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        # if len(trues.shape) >= 2:
        #     trues = torch.argmax(trues, dim=1)
        predictions = preds.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        del predictions
        del trues
        torch.cuda.empty_cache()

        print('data_task_name: {} accuracy:{}'.format(
            data_task_name, accuracy), folder=self.path)

        return accuracy


    def test_RUL(self, setting, test_data, test_loader, data_task_name, task_id):
        preds = []
        trues = []
        self.model.eval()
        ids_pres ={}
        ids_trues = {}
        with torch.no_grad():
            for i, (batch_x, label,name_and_id) in enumerate(test_loader):
                name = name_and_id[0]
                padding_mask = torch.zeros_like(batch_x)
                batch_x = batch_x.float().to(self.device_id)
                padding_mask = padding_mask.float().to(self.device_id)
                label = label.to(self.device_id)

                outputs = self.model(
                    batch_x, padding_mask, None, None, task_id=task_id, task_name='PHM_RUL')
                #判断是否是新的id
                if name in ids_pres.keys():
                    ids_pres[name].append(outputs)
                    ids_trues[name].append(label)
                else:
                    ids_pres[name] = [outputs]
                    ids_trues[name] = [label]
        for key in ids_pres.keys():
            if self.args.ddp:
                ids_pres[key] = gather_tensors_from_all_gpus(
                    ids_pres[key], self.device_id, to_numpy=False)
                ids_trues[key] = gather_tensors_from_all_gpus(
                    ids_trues[key], self.device_id, to_numpy=False)
            
            ids_pres[key] = torch.cat(ids_pres[key], 0)
            ids_trues[key] = torch.cat(ids_trues[key], 0)
            ids_pres[key] = ids_pres[key].cpu().numpy()
            ids_trues[key] = ids_trues[key].cpu().numpy()
            if len(ids_pres[key].shape)>=2:
                ids_pres[key] = ids_pres[key].squeeze(-1)
            mae, mse, rmse, mape, mspe = metric(ids_pres[key] , ids_trues[key])
            print('data_task_name: {} id:{} mse:{}, mae:{}'.format(
                data_task_name,key, mse, mae), folder=self.path)
        torch.cuda.empty_cache()
        del ids_pres, ids_trues
        return mse, mae




    def split_batch(self, batch, small_batch_size, task_name):
        def split_tensor(tensor, size):
            return [tensor[i:min(i + size, tensor.size(0))] for i in range(0, tensor.size(0), size)]
        if task_name == 'classification':
            batch_x, label, padding_mask = batch
            split_batch_x = split_tensor(batch_x, small_batch_size)
            split_label = split_tensor(label, small_batch_size)
            split_padding_mask = split_tensor(padding_mask, small_batch_size)
            return list(zip(split_batch_x, split_label, split_padding_mask))
        elif task_name == 'long_term_forecast' or task_name == 'imputation':
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            split_batch_x = split_tensor(batch_x, small_batch_size)
            split_batch_y = split_tensor(batch_y, small_batch_size)
            split_batch_x_mark = split_tensor(batch_x_mark, small_batch_size)
            split_batch_y_mark = split_tensor(batch_y_mark, small_batch_size)
            return list(zip(split_batch_x, split_batch_y, split_batch_x_mark, split_batch_y_mark))
        elif task_name == 'anomaly_detection':
            batch_x, batch_y = batch
            split_batch_x = split_tensor(batch_x, small_batch_size)
            split_batch_y = split_tensor(batch_y, small_batch_size)
            return list(zip(split_batch_x, split_batch_y))

    def memory_check(self, data_loader_cycle, criterion_list, holdout_memory=3):
        """
        Checks the memory usage of the model by gradually increasing the batch size until it reaches the maximum batch size that can be supported without running out of memory.

        Args:
            data_loader_cycle (DataLoaderCycle): The data loader cycle object.
            holdout_memory (int): The amount of memory (in GB) to hold out for other operations.

        Returns:
            None
        """
        num_elements = holdout_memory * 1024 * 1024 * 1024 // 4
        extra_mem = torch.empty(
            num_elements, dtype=torch.float32, device=self.device_id)

        model_tmp = self._build_model(ddp=False)
        model_tmp.train()
        model_tmp.zero_grad(set_to_none=True)

        for data_loader_id in range(data_loader_cycle.num_dataloaders):
            batch_size = 1  # Initial batch size
            max_batch_size = 0  # Record the maximum batch size before OOM
            torch.cuda.synchronize()
            model_tmp.zero_grad(set_to_none=True)
            while True:
                try:
                    sample, task_id = data_loader_cycle.generate_fake_samples_for_batch(
                        data_loader_id, batch_size)  # 2 makes the memory larger
                    task_name = self.task_data_config_list[task_id][1]['task_name']
                    # Try running the function with the current batch size
                    print(task_id, task_name,
                          sample[0].shape, "max batch size", max_batch_size)
                    if task_name == 'long_term_forecast':
                        loss = self.train_long_term_forecast(
                            model_tmp, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    elif 'classification' in task_name:
                        loss = self.train_classification(
                            model_tmp, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    elif task_name == 'imputation':
                        loss = self.train_imputation(
                            model_tmp, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    elif task_name == 'anomaly_detection':
                        loss = self.train_anomaly_detection(
                            model_tmp, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    elif  'RUL' in task_name:
                        loss = self.train_RUL(
                            model_tmp, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss = loss * 0.0
                    loss.backward()
                    max_batch_size = batch_size  # Update the maximum batch size
                    batch_size *= 2  # Increase the batch size

                    if max_batch_size >= self.args.batch_size:
                        print("Can support default batch size:",
                              self.args.batch_size, max_batch_size)
                        self.task_data_config_list[task_id][1]['max_batch'] = max_batch_size
                        self.task_data_config_list[task_id][1]['checkpointing'] = False
                        break

                except Exception as e:
                    task_name = self.task_data_config_list[task_id][1]['task_name']
                    print(task_id,  "max batch size:", max_batch_size)
                    # If any exception occurs, break the loop
                    self.task_data_config_list[task_id][1]['max_batch'] = max_batch_size
                    del model_tmp
                    model_tmp = self._build_model(ddp=False)
                    print(f"An exception occurred: {e}")
                    break
        print(self.task_data_config_list)
        del model_tmp
        del extra_mem
        torch.cuda.empty_cache()
        return
