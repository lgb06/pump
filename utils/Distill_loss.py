import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
class SimDINOLoss(nn.Module):
    def __init__(self, d_model, alpha = 0.9 ,eps=0.05, gamma=0.05,
                                  teacher_temp=0.04,
                   student_temp=0.1):
        """
        d_model: 特征维度 D
        alpha: 损失函数中的权重超参数
        eps: coding rate 正则化中的量化超参数
        gamma: coding rate 正则化的强度超参数
        """
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.d_model = d_model
        # 构造单位矩阵 I_d (D, D)，用于后续计算 logdet(I + d/eps^2 * cov)
        self.register_buffer("I_d", torch.eye(d_model))
        self.tcr_max = 100
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def compute_TCR(self,Z, epsilon=0.05):
        Z = Z.reshape(-1, self.d_model)
        b, d = Z.shape  # 样本数 b，特征维度 d
        #进行l2norm
        Z = F.normalize(Z, p=2, dim=-1)
        I = torch.eye(d).to(Z.device)  # 单位矩阵
        cov_matrix = Z.T @ Z / b  # 计算协方差矩阵
        reg_matrix = I + (d / (b *epsilon**2)) * cov_matrix
        log_det = torch.logdet(reg_matrix)  # 计算 log-determinant
        TCR_value = 0.5 * log_det  # 计算 TCR
        # self.tcr_max = 10
        TCR_value = self.tcr_max-TCR_value
        return TCR_value    
    def compute_infoNCE_loss(self, student_features, teacher_features,temp=0.1):
        """
        计算InfoNCE损失用于模态对齐
        student_features: 学生模型特征 (B*V, D)
        teacher_features: 教师模型特征 (B*V, D)
        temperature: 温度参数
        """
        # L2 归一化
        student_features = F.normalize(student_features, p=2, dim=1)
        teacher_features = F.normalize(teacher_features, p=2, dim=1)
        
        # 计算相似度矩阵 (B*V, B*V)
        sim_matrix = torch.matmul(student_features, teacher_features.T) / temp
        
        # 对角线元素是正样本对
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=sim_matrix.device)
        # 计算交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def forward(self, dict):
        """
        输入：
          student_seq: (B, V, L, D)
          student_cls: (B, V, 1, D)
          teacher_seq: (B, V, L, D)
          teacher_cls: (B, V, 1, D)
        输出返回一个字典，包含 'seq_loss', 'cls_loss' 和 'total_loss'
        """
        student_seq = dict['st_seq']   # (B, V, L, D)
        student_cls = dict['st_cls']   # (B, V, 1, D)
        teacher_seq = dict['te_seq']   # (B, V, L, D)
        teacher_cls = dict['te_cls']   # (B, V, 1, D)
        mask_id = dict['mask_id'] # (B, V, L, PL)
        mask_id = mask_id.mean(dim=-1).unsqueeze(-1)
        B, V, L, D = student_seq.shape
        # 创建布尔掩码
        # 1. cls token 损失：
        # diff_cls = student_cls- teacher_cls
        # loss_cls = torch.abs(diff_cls).sum(dim=-1).mean()  # L1距离
        # loss_cls = self.compute_similarity_loss(student_cls, teacher_cls)

        loss_time_freq =self.compute_infoNCE_loss(student_seq.mean(dim=2).reshape(-1, D), 
                                            student_cls.reshape(-1, D),temp = self.student_temp)
        loss_freq_freq = self.compute_infoNCE_loss(student_cls.reshape(-1, D),
                                            teacher_cls.reshape(-1, D),temp=self.teacher_temp)
        loss_cls = (loss_time_freq+loss_freq_freq)/2
        bool_mask = mask_id > 0
        B, V, L, D = student_seq.shape
        student_seq_flat = student_seq.reshape(-1, D)  # (B*V*L, D)
        teacher_seq_flat = teacher_seq.reshape(-1, D)  # (B*V*L, D)

        # 然后将mask_id也展平为(B*V*L)
        bool_mask_flat = bool_mask.reshape(-1)  # (B*V*L)

        # 提取非零部分
        student_seq_nonzero = student_seq_flat[bool_mask_flat]  # (N_nonzero, D)
        teacher_seq_nonzero = teacher_seq_flat[bool_mask_flat]  # (N_nonzero, D)
        
        # loss_seq = self.compute_infoNCE_loss(student_seq_nonzero, teacher_seq_nonzero,temp=self.teacher_temp)

        # 2. 序列（patch token）损失：计算所有学生视图与教师视图的 patch 表示之间的 L1 距离
        diff_seq = student_seq_nonzero - teacher_seq_nonzero
        loss_seq = torch.abs(diff_seq).sum(dim=-1).mean()*0.1  # L1距离

        tcr_loss = self.compute_TCR(student_seq)
        total_loss = (loss_cls *self.alpha+(1-self.alpha)*loss_seq+self.gamma*tcr_loss)
        return {
            'seq_loss': loss_seq, 
            'cls_loss': loss_cls,
            'tcr_loss':tcr_loss,
            'total_loss': total_loss,
            'cls_ft':loss_time_freq,
            'cls_ff':loss_freq_freq
        }
    
    def compute_similarity_loss(self, student_out, teacher_out):
        """
        Computes KL divergence loss with improved numerical stability
        """
        teacher_out = teacher_out.detach()
        student_out = student_out.reshape(-1, self.d_model)
        teacher_out = teacher_out.reshape(-1, self.d_model)

        # Apply temperature scaling in log s    pace to avoid overflow
        student_logits = student_out / self.student_temp
        teacher_logits = teacher_out / self.teacher_temp
        
        # Use log_softmax which is more numerically stable than softmax + log
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Compute KL divergence using F.kl_div with 'batchmean' reduction
        # KL(T||S) = sum(T * (log(T) - log(S))) using the stable implementation
        loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        ) 
        
        return loss
class DistillLoss(nn.Module):
    def __init__(self, d_model,
                 teacher_temp=0.04,
                   student_temp=0.1, 
                   center_momentum=0.9, 
                   cls_weight=0.5,
                   tcr_weight=0.1,
                   tcr_max= 100,
                    ):
        """
        teacher_temp: 教师温度系数
        student_temp: 学生温度系数
        center_momentum: EMA 更新中心的权重
        seq_weight: 序列损失的权重
        cls_weight: 分类损失的权重
        """
        super().__init__()
        self.d_model = d_model
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.cls_weight = cls_weight
        self.tcr_weight = tcr_weight
        self.tcr_max= tcr_max
    def forward(self, dict):
        """
        student_seq: (B, V, L, D)
        student_cls: (B, V, 1, D)
        teacher_seq: (B, V, L, D)
        teacher_cls: (B, V, 1, D)
        """
        student_seq,student_cls,teacher_seq,teacher_cls = dict['st_seq'],dict['st_cls'],dict['te_seq'],dict['te_cls']


        # 计算 seq loss
        seq_loss = self.compute_similarity_loss(student_seq, 
                                                teacher_seq)
        
        # 计算 cls loss
        cls_loss = self.compute_similarity_loss(student_cls, 
                                                teacher_cls
                                                )
        tcr_loss = self.compute_TCR(student_seq)
        # 计算最终损失
        total_loss = (1-self.cls_weight)* seq_loss + self.cls_weight * cls_loss+self.tcr_weight*tcr_loss
        # total_loss = (1-self.cls_weight)* seq_loss + self.cls_weight * cls_loss
        # # 更新中心
        # self.update_center(teacher_seq, teacher_cls)

        dict = {
            'seq_loss': seq_loss,
            'cls_loss': cls_loss,
            'total_loss': total_loss,
            'tcr_loss':tcr_loss
        }

        return dict

    def compute_similarity_loss(self, student_out, teacher_out):
        """
        Computes KL divergence loss with improved numerical stability
        """
        teacher_out = teacher_out.detach()
        student_out = student_out.reshape(-1, self.d_model)
        teacher_out = teacher_out.reshape(-1, self.d_model)

        # Apply temperature scaling in log s    pace to avoid overflow
        student_logits = student_out / self.student_temp
        teacher_logits = teacher_out / self.teacher_temp
        
        # Use log_softmax which is more numerically stable than softmax + log
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Compute KL divergence using F.kl_div with 'batchmean' reduction
        # KL(T||S) = sum(T * (log(T) - log(S))) using the stable implementation
        loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        ) 
        
        return loss

    def compute_TCR(self,Z, epsilon=0.05):
        Z = Z.reshape(-1, self.d_model)
        b, d = Z.shape  # 样本数 b，特征维度 d
        #进行l2norm
        Z = F.normalize(Z, p=2, dim=-1)
        I = torch.eye(d).to(Z.device)  # 单位矩阵
        cov_matrix = Z.T @ Z / b  # 计算协方差矩阵
        reg_matrix = I + (d / (b *epsilon**2)) * cov_matrix
        log_det = torch.logdet(reg_matrix)  # 计算 log-determinant
        TCR_value = 0.5 * log_det  # 计算 TCR
        # self.tcr_max = 10
        TCR_value = self.tcr_max-TCR_value
        return TCR_value        


if __name__ == '__main__':
    torch.manual_seed(42)

    B, V, L, D = 2, 3, 4, 512 # Batch size, Views, Length, Dimension

    # 生成随机数据
    student_seq = torch.randn(B, V, L, D)
    student_cls = torch.randn(B, V, 1, D)
    teacher_seq = torch.randn(B, V, L, D)
    teacher_cls = torch.randn(B, V, 1, D)
    mask_id = torch.ones_like(student_seq)
    #加一层LN
    student_seq = F.layer_norm(student_seq, normalized_shape=(D,))
    student_cls = F.layer_norm(student_cls, normalized_shape=(D,))
    teacher_seq = F.layer_norm(teacher_seq, normalized_shape=(D,))
    teacher_cls = F.layer_norm(teacher_cls, normalized_shape=(D,))
    # 初始化损失函数
    loss_fn = SimDINOLoss(D)

    dict = {
        'st_seq': student_seq,
        'st_cls': student_cls,
        'te_seq': teacher_seq,
        'te_cls': teacher_cls,
        'mask_id':mask_id
    }
    # 计算损失
    loss = loss_fn(dict)

    print("Computed Loss:", loss)
