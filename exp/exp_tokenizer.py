from exp.exp_pretrain import Exp_All_Task,custom_print_decorator,init_and_merge_datasets
from models.layers.Quantizer import NormEMAVectorQuantizer
from models.layers.vq import ResidualTokenizer
import torch.nn as nn
from utils.ddp import is_main_process
import torch
import torch.nn as nn
import time
import numpy as np
import wandb
from tqdm import tqdm
from utils.tools import cosine_scheduler
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from utils.losses import DiscreteMaskRecLoss
from utils.ddp import is_main_process, get_world_size
import torch
import torch.nn as nn
import torch.distributed as dist
import os
import time
import numpy as np
import wandb
from tqdm import tqdm
print = custom_print_decorator(print)



class Exp_Tokenizer(Exp_All_Task):
    def __init__(self, args):
        super(Exp_Tokenizer, self).__init__(args)
    def _build_model(self,ddp=False):
        if ddp==False:
            ddp = False
        else:
            ddp = self.args.ddp
        model = NormEMAVectorQuantizer(
            self.args).to(self.device_id)
        if ddp:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.device_id], find_unused_parameters=True)
        return model.to(self.device_id)

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        # if self.args.train_quantizer:
        #     path = os.path.join(path, 'quantizer')
        if not os.path.exists(path) and is_main_process():
            os.makedirs(path)
        self.path = path
        if self.args.ddp:
            torch.cuda.synchronize()
            dist.barrier()

        # Data loader
        _, train_loader_list = self._get_data(flag='pretrain')
        # data_loader_cycle是BalancedDataLoaderIterator，train_steps是单个数据集最大的迭代次数和数据集数目的乘积
        data_loader_cycle, train_steps = init_and_merge_datasets(
            train_loader_list)

        # Set up batch size for each task
        if self.args.memory_check:
            self.memory_check(data_loader_cycle)
            torch.cuda.empty_cache()


        if self.args.ddp:
            torch.cuda.synchronize()
            dist.barrier()

        # Model
        self.model = self._build_model()
        # # if self.args.train_quantizer:
        # #     # self.model.eval()
        # #     self.model.quantizer.train()
        # #     self.model.quantizer.requires_grad_(True)
        # #     print('train quantizer')
        # # else:
        # #     self.model.train()
        # #     #指定加载目录下的quantizer.pth文件
        # #     if self.args.load_quantizer:
        # #         self.model.quantizer.load_state_dict(torch.load(self.args.load_quantizer)['quantizer'])
        # #         self.model.quantizer.requires_grad_(False)
        # #         # self.model.quantizer.eval()
        #         print('load quantizer from', self.args.load_quantizer)
        self.model.train()
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        if is_main_process():
            print("Parameters number {} M".format(
                pytorch_total_params/1e6), folder=self.path)
            print("{} steps for each epoch".format(train_steps), folder=self.path)

        # Optimizer
        model_optim = self._select_optimizer()
        lr_schedule = cosine_scheduler(
            self.real_learning_rate,
            self.args.min_lr,
            self.args.train_epochs, train_steps,
            warmup_epochs=self.args.warmup_epochs,
        )

        # Loss
        # if self.args.train_quantizer:
        #     criterion = QuantizerLoss().to(self.device_id)
        # else:
        criterion = DiscreteMaskRecLoss().to(self.device_id)
        scaler = NativeScaler()

        for epoch in tqdm(range(self.args.train_epochs), desc="Training Out Epochs"):
            train_loss = self.train_one_epoch(
                model_optim, data_loader_cycle, criterion, epoch, train_steps, scaler, lr_schedule)

            print("Epoch: {0}, Steps: {1} | Avg Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss), folder=self.path)
            if is_main_process():
                wandb.log({'train_loss_avg': train_loss})
                save_dict = {
                'student': self.model.state_dict(),
                'optimizer': model_optim.state_dict(),
                'epoch': epoch + 1,
                'args': self.args,
                }
                torch.save(save_dict, path + '/' + 'pretrain_checkpoint.pth')

        return self.model    
    def train_one_epoch(self, model_optim, data_loader_cycle, criterion, epoch, train_steps, scaler, lr_schedule):
        #一次epoch训练完所有的数据集，每个数据集出现一次batch
        current_device = self.device_id
        train_loss_set = []

        acc_it = self.args.acc_it
        max_norm = self.args.clip_grad

        self.model.train()
        epoch_time = time.time()
        self.model.zero_grad(set_to_none=True)
        loss_sum_display = 0
        loss = 0
        #外循环训练不同数据集,每次采样不同数据集的一个batch
        for i, (sample_init, task_id) in enumerate(tqdm(data_loader_cycle, desc="Training Inner Epoch {}".format(epoch + 1))):
            
            it = train_steps * epoch + i
            for _, param_group in enumerate(model_optim.param_groups):
                #调整学习率
                param_group["lr"] = lr_schedule[it]

            # Get batch data based on the real batch size of each task: avoid OOM for large samples
            task_name = self.task_data_config_list[task_id][1]['task_name']
            small_batch_size = self.task_data_config_list[task_id][1]['max_batch']
            #根据设备能够承受的最大batchsize，将一个batch分成多个小batch
            sample_list = self.get_multi_source_data(
                sample_init, task_name, small_batch_size, min_keep_ratio=None)
            len_sample_list = len(sample_list)

            # Accumulate gradients of mulitple samples
            for sample_idx in range(len_sample_list):
                sample = sample_list[sample_idx]
                x_enc, label, x_mark_enc = sample
                with torch.cuda.amp.autocast():
                    out,dict = self.model(
                        x_enc)
                loss = dict['loss_dict']+dict['loss_recon']
                loss /= acc_it
                loss /= len_sample_list
                if sample_idx < len_sample_list-1:
                    norm_value = scaler(loss, model_optim, clip_grad=max_norm,
                                        parameters=self.model.parameters(), create_graph=False, update_grad=False)

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print("loss is too large, skip this iteration!")
                continue
            loss_display = loss.item()*len_sample_list*acc_it
            train_loss_set.append(loss_display)

            norm_value = scaler(loss, model_optim, clip_grad=max_norm,
                                parameters=self.model.parameters(), create_graph=False, update_grad=((i + 1) % acc_it == 0))

            if (i+1) % acc_it == 0:
                model_optim.zero_grad()
            torch.cuda.synchronize()

            loss_sum_display += loss_display

            # release memory to avoid OOM
            del sample_init
            del sample_list
            if torch.cuda.memory_reserved(current_device) > 30*1e9:
                torch.cuda.empty_cache()
            #针对每个数据集都有两个pretrainloss
            if is_main_process():

                wandb_loss_dict = {
                    'norm': norm_value if norm_value is not None else 0,
                    'train_dict_loss_'+self.task_data_config_list[task_id][0]: dict['loss_dict'].item(),
                    'train_recon_loss'+self.task_data_config_list[task_id][0]: dict['loss_recon'].item(),
                    "loss_avg": loss_sum_display/(i+1)
                }
                wandb.log(wandb_loss_dict)

            if (i + 1) % 50 == 0 and is_main_process():
                print("\titers: {0}, epoch: {1} | lr: {2:.5} | loss_avg: {3} | current_loss: {4} |current data: {5}".format(
                    i + 1, epoch + 1, lr_schedule[it], loss_sum_display/(i+1), loss.item() * acc_it, task_name), folder=self.path)
        

        if is_main_process():
            print("Epoch: {} cost time: {}".format(
                epoch + 1, time.time() - epoch_time), folder=self.path)
        train_loss = np.average(train_loss_set)

        # if self.debug==True:
        #     for item in loss_dict:
        #         loss_dict[item] = loss_dict[item].detach().cpu().numpy()
        #     save_dict = { 'loss':loss_dict,
        #                  'pred_cls_phase': model_output[0][0].detach().cpu().numpy(),
        #                  'pred_cls_angle': model_output[0][1].detach().cpu().numpy(),
        #                  'pred_mask':model_output[1].detach().cpu().numpy(),
        #                 'target':x_enc.detach().cpu().numpy(),
        #                 'mask':model_output[2].detach().cpu().numpy()}
        #     save_path = 'debug_result'
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     torch.save(save_dict, save_path + '/' + 'pretrain_result_'+str(epoch+1)+'.pth')


        return train_loss

