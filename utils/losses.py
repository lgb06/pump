# Part of the file is from https://github.com/thuml/Time-Series-Library/blob/main/utils/losses.py
"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb
import torch.nn as nn
import torch.nn.functional as F


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class TokenizerLoss(nn.Module):
    def __init__(self,alpha=0.5):
        super().__init__()
        self.alpha = alpha
    def forward(self, dict):
        loss_dict,loss_recon= dict['loss_dict'],dict['loss_recon']
        total_loss = self.alpha*loss_dict+(1-self.alpha)*loss_recon
        return total_loss


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class UnifiedMaskRecLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward_mim_loss(self, target, pred, pad_mask):
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per step,mean on dimension D

        combined_mask = pad_mask.bool()

        loss = (loss * combined_mask).sum() / combined_mask.sum()
        return loss
    
    def foward_full_loss(self, target, pred):
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = loss.sum()
        return loss

    def forward(self, outputs, target, pad_mask):
        student_cls, student_fore, _ = outputs

        mask_loss = self.forward_mim_loss(target, student_fore, pad_mask)

        if student_cls is not None:
            cls_loss = self.forward_mim_loss(target, student_cls, pad_mask)
        else:
            cls_loss = 0.0 * mask_loss

        total_loss = dict(cls_loss=cls_loss,
                          mask_loss=mask_loss, loss=mask_loss+cls_loss)
        return total_loss


class QuantizerLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_1 = nn.Parameter(t.tensor(1.0))  # 对应 recon
        self.alpha_2 = nn.Parameter(t.tensor(1.0))  # 对应 recon_middle
    def forward(self, outputs):
        loss_dict,loss_recon = outputs
        total_loss = self.alpha_1*loss_dict+loss_recon+self.alpha_2*loss_recon
        loss = dict(loss=total_loss,dict_loss=loss_dict,recon=loss_recon)
        return loss


class DiscreteMaskRecLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_1 = nn.Parameter(t.tensor(1.0))  # 对应 recon
        self.alpha_2 = nn.Parameter(t.tensor(1.0))  # 对应 recon_middle
        self.alpha_3 = nn.Parameter(t.tensor(1.0))  # 对应 dict_loss
    def forward(self,outputs):
        target, pred,pred_middle,mask,loss_dict = outputs
        B, V,N, D = pred.shape
        mask  = mask.flatten()
        # target = target.reshape(B, V * N, D)[mask]
        pred = pred.view(B*V*N,D)[mask]
        pred_middle = pred_middle.view(B*V*N,D)[mask]
        target = target[mask]
        loss =F.cross_entropy(pred,target)
        loss_middle = F.cross_entropy(pred_middle,target)
        
        total_loss = self.alpha_1*loss+self.alpha_2*loss_middle

        loss = dict(loss=total_loss,recon = loss,recon_middle=loss_middle,dict_loss=loss_dict)
        return loss   



class FreqTimeMaskRecLoss(nn.Module):
    def __init__(self,alpha =1.0,lamda=0.5):
        ##alpha: the weight of the time loss and frequency loss
        ##lamda: the weight of the frequency loss of amplitude and phase
        super().__init__()
        self.lamda = lamda
        self.alpha = alpha

    def forward_mim_loss(self, target, pred, pad_mask):
        loss = (pred - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per step,mean on dimension D
        # loss_feq = (t.fft.rfft(pred, dim=1) - t.fft.rfft(target, dim=1)).abs().mean()

        combined_mask = pad_mask.bool()

        loss = (loss * combined_mask).sum() / combined_mask.sum()
        return loss
    
    # def batch_std(self, x):
    #     mean = x.mean(dim=(0,1), keepdim=True) #在batch和时间维度上求均值，得到channel的均值
    #     std = x.std(dim=(0,1), keepdim=True)
    #     x = (x-mean)/std
    #     return x

    # # def freq_loss(self,pred,target):
    # #     fft_target = t.fft.fft(target,dim=1)
    # #     amplitude_pred = self.batch_std(pred[1])
    # #     amplitude_target = self.batch_std(t.abs(fft_target))
    # #     phase_pred =  self.batch_std(pred[0])
    # #     phase_target = self.batch_std(t.angle(fft_target))
    # #     #检查一下为什么会出错
    # #     _,len,_ = target.shape
    # #     loss_amp = (amplitude_pred[:,:len,:] - amplitude_target) ** 2
    # #     loss_pha = (phase_pred[:,:len,:] - phase_target) ** 2
    # #     loss = self.lamda*loss_amp +(1-self.lamda)*loss_pha
    # #     loss = loss.mean()
    # #     return loss

    def forward(self, outputs, target):
        mask_out, _, pad_mask = outputs

        mask_loss = self.forward_mim_loss(target, mask_out, pad_mask)

        # # if cls_out is not None:
        # #     cls_out = self.forward_mim_loss(target, cls_out, pad_mask)
        # # else:
        # #     cls_out = t.tensor(0.0)
        # freq_loss = self.freq_loss(cls_out,target)
        # loss = self.alpha*mask_loss + (1-self.alpha)*freq_loss
        total_loss = dict(cls_loss=mask_loss,
                          freq_loss=mask_loss, loss=mask_loss)
        return total_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 获取预测概率
        pt = t.exp(-BCE_loss)
        
        # 计算 Focal Loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


#测试函数
if  __name__ == '__main__':
    #测试loss函数
    loss = FreqTimeMaskRecLoss()
    target = t.rand(32,224, 111)
    pred = t.rand(32,224, 111)
    pad_mask = t.ones(32, 224)
    # print(loss.freq_loss([pred,pred], target))
    print(loss.forward([pred,pred,pred], target))