from data_provider.data_factory import data_provider
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from exp.exp_sup import custom_print_decorator,Exp_All_Task,init_and_merge_datasets
from data_provider.data_factory import data_provider
from utils.tools import adjust_learning_rate
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from utils.ddp import is_main_process
from exp.exp_pretrain import custom_print_decorator
import torch
import torch.distributed as dist
from tqdm import tqdm
import os
import wandb
from models.layers.LoRA import LinearWithLoRAMerged
import torch.nn as nn
import copy

print = custom_print_decorator(print)
class Exp_FewShot(Exp_All_Task):
    def __init__(self, args):
        super(Exp_FewShot, self).__init__(args)

    # def lora_transform(self, rank=4, alpha=8):
    #     #lora改造
    #     model = copy.deepcopy(self.model)
    #     # Get all block linear layers
    #     block_linear_layers = []
    #     for name, module in model.named_modules():
    #         if 'blocks' in name and isinstance(module, nn.Linear):
    #             block_linear_layers.append((name, module))
        
    #     # Group by block number
    #     blocks = {}
    #     for name, module in block_linear_layers:
    #         block_num = int(name.split('blocks.')[1].split('.')[0])
    #         if block_num not in blocks:
    #             blocks[block_num] = []
    #         blocks[block_num].append((name, module))
        
    #     # Find the highest block number (last block)
    #     if blocks:
    #         last_block_num = max(blocks.keys())
    #         # Apply LoRA to all layers in the last block
    #         for name, module in blocks[last_block_num]:
    #             parent_module = dict(model.named_modules())[name.rsplit('.', 1)[0]]
    #             setattr(parent_module, name.rsplit('.', 1)[-1], 
    #                   LinearWithLoRAMerged(module, rank=rank, alpha=alpha))
    #             print(f"Applied LoRA to {name}")
        
    #     self.model = model.to(self.device_id)
    #     print("LoRA transform done!", folder=self.path)

    def train_parameter_choice(self):
        if self.args.lora_transform:
            self.lora_transform()
        if self.args.efficiency_tuning:
            self.choose_training_parts(prompt_tune=True)    
    def choose_training_parts(self, prompt_tune=False):
    
        model_param = []
        trainable_param = []
        #通过控制可训练参数的梯度来控制训练部分
        for name, param in self.model.named_parameters():
            if prompt_tune:
                #仅仅训练网络中保存的任务参数
                if any(keyword in name for keyword in 
                       ['rul',  'category','prompt','lora','global','cls_head']):
                    param.requires_grad = True
                    if 'lora' not in name:
                        print("trainable:", name)
                else:
                    param.requires_grad = False
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_param.append(param.numel())
            model_param.append(param.numel())

        trainable_total_param = sum(trainable_param)
        model_total_params = sum(model_param)

        print("Parameters number for total RmGPT {} M, trainable number {} M, trainable ratio {:.2%}".format(
            model_total_params / 1e6, trainable_total_param / 1e6,
            trainable_total_param / model_total_params))    
    def _get_data(self, flag):
        ddp = self.args.ddp
        this_task_data_config = self.task_data_config
        data_set_list_train = []
        data_loader_list_train = []
        data_set_list_fs = []
        data_loader_list_fs = []
        for task_data_name, task_config in this_task_data_config.items():
            if task_config['task_name'] == 'classification' and flag == 'val':
                # TODO strange that no val set is used for classification. Set to test set for val
                flag = 'test'
            data_set, data_loader = data_provider(
                self.args, task_config, flag, ddp=ddp)
            if task_config['few_shot']:
                data_set_list_fs.append(data_set)
                data_loader_list_fs.append(data_loader)
            else:
                data_set_list_train.append(data_set)
                data_loader_list_train.append(data_loader)
            print(task_data_name, len(data_set))
        return data_set_list_train, data_loader_list_train, data_set_list_fs, data_loader_list_fs
    
    def train(self, setting,model= None):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and is_main_process():
            os.makedirs(path)
        self.path = path

        if model is not None:
            msg = self.model.load_state_dict(model.state_dict(), strict=False)
            print(f"Loading model parameters: {msg}", folder=self.path)
            self.train_parameter_choice()
        else:
            self.load_model_from_pretrain(setting)
            self.train_parameter_choice()

        _, train_loader_list,_,fs_loader_list = self._get_data(flag='train')
        # Since some datasets do not have val set, we use test set and report the performance of last epoch instead of the best epoch.
        test_data_list, test_loader_list,test_data_list_fs,test_loader_list_fs = self._get_data(
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

        #进入few-shot训练   
        acc_dict = self.fs_train(self.model,fs_loader_list,test_loader_list_fs,test_data_list_fs,setting)
        #打印最终的准确率
        for key in acc_dict.keys():
            print(f"Final accuracy for {key}: {acc_dict[key]}", folder=self.path)
        #求平均准确率
        avg_acc = sum(acc_dict.values()) / len(acc_dict)
        #wandb记录
        wandb.log({'Final_CLS-acc': avg_acc})
        print("Final score:  CLS-acc {}".format(avg_acc), folder=self.path)
        return self.model
    

    def fs_train(self,model,fs_loader_list,test_loader_list_fs,test_data_list_fs,setting):
        model_optim = self._select_optimizer()
        criterion_list = self._select_criterion(self.task_data_config_list)
        scaler = NativeScaler()
        # 初始化早停相关变量

        best_acc_dict = {}
        best_epoch_dict = {}
        patience = 7  # 连续3个epoch准确率下降时早停
        patience_counter = 0
        history_accs_dict = {}

        #对每个数据集的few-shot进行分别训练
        for i in range(len(fs_loader_list)):
        #加载当前的模型参数

            self.model = model
            train_fs_loader = [fs_loader_list[i]]
            test_fs_loader = [test_loader_list_fs[i]]
            test_fs_data = [test_data_list_fs[i]]
            dataset_name = test_data_list_fs[i].__class__.__name__
            print('Current few-shot task:', dataset_name)
            best_acc_dict[dataset_name] = 0
            best_epoch_dict[dataset_name] = 0
            history_accs_dict[dataset_name] = []

            fs_loader_cycle, fs_train_steps = init_and_merge_datasets(train_fs_loader)
            for epoch in tqdm(range(self.args.train_epochs), desc="Training In Epochs"):
                adjust_learning_rate(model_optim, epoch,
                                        self.real_learning_rate, self.args)
                train_loss = self.train_one_epoch(
                    model_optim, fs_loader_cycle, criterion_list, epoch, fs_train_steps, scaler)
                avg_cls_acc, avg_forecast_mse, avg_forecast_mae = self.test(
                    setting, load_pretrain=False, test_data_list=test_fs_data, test_loader_list=test_fs_loader)
                # 记录当前准确率
                history_accs_dict[dataset_name].append(avg_cls_acc)
                # 输出当前训练信息
                if is_main_process():

                    # 检查是否为最佳模型
                    if avg_cls_acc > best_acc_dict[dataset_name] :
                        best_acc_dict[dataset_name]  = avg_cls_acc
                        best_epoch_dict[dataset_name] = epoch
                        patience_counter = 0  # 重置早停计数器
                        print(f"Epoch: {epoch+1}, Accuracy: {avg_cls_acc:.4f}, Best Accuracy: {best_acc_dict[dataset_name] :.4f} at epoch {best_epoch_dict[dataset_name]+1}", folder=self.path)
                        wandb.log({"epoch": epoch, "acc": avg_cls_acc, "best_acc_"+dataset_name: best_acc_dict[dataset_name] })                        
                        # 保存最佳模型
                        model_path = os.path.join(self.path, 'best_model.pth')
                        torch.save(self.model.state_dict(), model_path)
                        print(f"Best model saved at epoch {epoch+1} with acc: {best_acc_dict[dataset_name] :.4f}", folder=self.path)

                    else:
                        patience_counter += 1
                    # 检查是否触发早停
                    if patience_counter >= patience:
                        if is_main_process():
                            print(f"Early stopping triggered! No improvement for {patience} epochs.", folder=self.path)
                        break
        return best_acc_dict