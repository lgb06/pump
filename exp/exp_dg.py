import time

import numpy as np
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from exp.exp_sup import custom_print_decorator,Exp_All_Task,init_and_merge_datasets
from utils.tools import adjust_learning_rate
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from utils.ddp import is_main_process
from exp.exp_pretrain import custom_print_decorator
import torch
import torch.distributed as dist
from tqdm import tqdm
import os
import wandb

print = custom_print_decorator(print)
class Exp_DG(Exp_All_Task):
    def __init__(self, args):
        super(Exp_DG, self).__init__(args)
    def train(self, setting,model=None):
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
        # Data
        _, train_loader_list = self._get_data(flag='train')
        # Since some datasets do not have val set, we use test set and report the performance of last epoch instead of the best epoch.
        test_data_list, test_loader_list = self._get_data(
            flag='test')
        # For each dataset in test_data_list, remove the corresponding elements
        ACC_list = []
        ACC_dict = {}
        for i in range(len(test_data_list)):
            # Create temporary lists excluding the current dataset
            temp_train_loader_list = [loader for j, loader in enumerate(train_loader_list) if j != i]
            temp_test_loader_list = [loader for j, loader in enumerate(test_loader_list) if j == i]
            temp_test_data_list = [data for j, data in enumerate(test_data_list) if j == i]
            print(f"Test Current class: {test_data_list[i].__class__.__name__}")
            # Train the model without this dataset
            acc = self.train_DG(setting, temp_train_loader_list, temp_test_data_list, temp_test_loader_list)
            ACC_dict[test_data_list[i].__class__.__name__] = acc
            ACC_list.append(acc)
        avg_cls_acc = sum(ACC_list) / len(ACC_list)
        if is_main_process():
            wandb.log({'Final_CLS-acc': avg_cls_acc})
            print("Final score:  CLS-acc {}".format(avg_cls_acc), folder=self.path)
            #打印每个数据集的准确率
            for key in ACC_dict:
                print(f"{key} acc: {ACC_dict[key]}", folder=self.path)
        return self.model   
    
    
    def train_DG(self,setting,train_loader_list,test_data_list,test_loader_list):
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
        # 初始化早停相关变量
        best_acc = 0
        best_epoch = -1
        patience = 2  # 连续3个epoch准确率下降时早停
        patience_counter = 0
        history_accs = []

        for epoch in tqdm(range(self.args.train_epochs), desc="Training Out Epochs"):
            adjust_learning_rate(model_optim, epoch,
                                self.real_learning_rate, self.args)
            train_loss = self.train_one_epoch(
                model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler)
            avg_cls_acc, avg_forecast_mse, avg_forecast_mae = self.test(
                setting, load_pretrain=False, test_data_list=test_data_list, test_loader_list=test_loader_list)
            
            # 记录当前准确率
            history_accs.append(avg_cls_acc)
            
            # 检查是否为最佳模型
            if avg_cls_acc > best_acc:
                best_acc = avg_cls_acc
                best_epoch = epoch
                patience_counter = 0  # 重置早停计数器
                # 保存最佳模型
                if is_main_process():
                    model_path = os.path.join(self.path, 'best_model.pth')
                    torch.save(self.model.state_dict(), model_path)
                    print(f"Best model saved at epoch {epoch+1} with acc: {best_acc:.4f}", folder=self.path)
            else:
                patience_counter += 1
            # 输出当前训练信息
            if is_main_process():
                print(f"Epoch: {epoch+1}, Accuracy: {avg_cls_acc:.4f}, Best Accuracy: {best_acc:.4f} at epoch {best_epoch+1}", folder=self.path)
                wandb.log({"epoch": epoch, "acc": avg_cls_acc, "best_acc": best_acc})
            # 检查是否触发早停
            if patience_counter >= patience:
                if is_main_process():
                    print(f"Early stopping triggered! No improvement for {patience} epochs.", folder=self.path)
                break
        return best_acc  # 返回最佳准确率而不是最后一个epoch的准确率
    
    def train_one_epoch(self, model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler):
        current_device = torch.cuda.current_device()
        train_loss_set = []
        acc_it = max(self.args.acc_it,1.0)
        max_norm = self.args.clip_grad
        self.model.train()
        epoch_time = time.time()
        self.model.zero_grad(set_to_none=True)
        loss_sum = 0
        device_features = {}  # 存储不同设备的特征
        lambda_coral = 0.1  # CORAL Loss 系数
        
        for i, (sample_init, task_id) in enumerate(tqdm(data_loader_cycle, desc="Training Inner Epoch {}".format(epoch + 1))):
            task_name = self.task_data_config_list[task_id][1]['task_name']
            sample_list = [sample_init]
            len_sample_list = 1
            len_sample_list = max(len_sample_list, 1.0)
            for sample_idx in range(len_sample_list):
                sample = sample_list[sample_idx]
                if  'classification' in task_name:
                    loss = self.train_classification(self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss_scale = 1.0

                # **提取特征**
                device_id = task_id  # 假设 `task_id` 代表设备 ID
                features = self.get_feature(
                        self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id).detach()  
                if device_id not in device_features:
                    device_features[device_id] = []
                device_features[device_id].append(features)
                loss /= acc_it
                loss /= len_sample_list
                if sample_idx < len_sample_list-1:
                    norm_value = scaler(loss*loss_scale, model_optim, clip_grad=max_norm,
                                        parameters=self.model.parameters(), create_graph=False, update_grad=False)
            loss_display = loss.item()*len_sample_list*acc_it
            train_loss_set.append(loss_display)

            #如果 loss 是 nan 或者 inf，跳过这次迭代
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                print("loss is too large, skip this iteration!")
                continue

            # if (i + 1) % acc_it == 0 and len(device_features) > 1:
            #     coral_loss = self.compute_global_coral_loss(device_features)
            #     loss += lambda_coral * coral_loss  # 加入 CORAL Loss
            #     # **清空累积的设备数据**
            #     device_features.clear()

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
                wandb.log(
                    {'train_loss_'+self.task_data_config_list[task_id][0]: loss_display, 'norm_value': norm_value, "loss_sum": loss_sum_display/(i+1)})

            if (i + 1) % 50 == 0:
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

    def compute_global_coral_loss(self, device_feature_dict):
        """
        计算多个设备之间的 CORAL Loss，使所有设备的特征分布对齐到一个全局分布
        :param device_feature_dict: 字典，key 是设备 ID，value 是设备特征的列表，每个列表包含多个特征 Tensor
        :return: CORAL Loss
        """
        device_keys = list(device_feature_dict.keys())
        K = len(device_keys)  # 设备数量
        if K < 2:
            return torch.tensor(0.0).cuda()  # 设备不足，CORAL Loss 无需计算

        # 计算所有设备的均值和协方差
        C_list = []
        mean_list = []
        
        d =self.args.d_model

        for device in device_keys:
            if len(device_feature_dict[device]) > 0:
                # 将每个设备的所有特征张量合并为一个大张量
                features = torch.cat(device_feature_dict[device], dim=0)
                mean = torch.mean(features, dim=0, keepdim=True)
                # 添加小的正则化项以避免奇异矩阵
                cov = (features - mean).T @ (features - mean) / (features.shape[0] - 1 + 1e-8)
                C_list.append(cov)
                mean_list.append(mean)

        if len(C_list) < 2:
            return torch.tensor(0.0).cuda()  # 不足两个有效设备
            
        # 计算全局协方差矩阵
        C_global = sum(C_list) / len(C_list)

        # 计算所有设备到全局协方差的 Frobenius 距离
        coral_loss = sum(torch.norm(C - C_global, p='fro') ** 2 for C in C_list) / (len(C_list) * d * d)*1000

        return coral_loss