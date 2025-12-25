import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenMasker(nn.Module):
    def __init__(self, patch_len=256, stride=256):
        """
        初始化TokenMasker类
        
        参数:
        - patch_len: 每个patch的长度
        - stride: patch之间的步长
        """
        super(TokenMasker, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
    
    def forward(self, x):
        """
        对输入张量进行批量掩码处理，完全使用并行操作，没有显式循环
        
        参数:
        - x: 输入张量，形状为 [B, L, M]
        
        返回:
        - masked_x: 掩码后的序列，形状为 [B, L, M]
        - mask_indicators: 掩码标记
        """
        B, L, M = x.shape
        device = x.device
        
        # 计算patch数量，与unfold操作一致
        patch_numbers = (L - self.patch_len) // self.stride + 1
        
        # 先创建掩码后的序列（克隆原始序列）
        masked_x = x.clone()
        
        # 使用unfold操作将序列分割成patch
        # 转换维度顺序以便进行unfold操作
        x_transposed = x.transpose(1, 2)  # [B, M, L]
        
        # 使用unfold操作，形状变为 [B, M, patch_numbers, patch_len]
        patches = x_transposed.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # 创建掩码标记，初始为全0（0表示没有掩码）
        mask_indicators = torch.zeros((B, M, patch_numbers, self.patch_len), device=device)
        
        # 1. 掩码每个序列的最后一个时间块 - 并行实现
        # 计算最后一个patch在原始序列中的位置
        last_patch_start = (patch_numbers - 1) * self.stride
        last_patch_end = last_patch_start + self.patch_len
        
        # 创建一个掩码张量，初始为1
        time_mask = torch.ones_like(masked_x)
        
        # 将最后一个patch位置设置为0
        time_mask[:, last_patch_start:last_patch_end, :] = 0.0
        
        # 应用掩码
        masked_x = masked_x * time_mask
        
        # 更新掩码标记 - 所有批次的最后一个patch
        mask_indicators[:, :, -1, :] = 1.0
        
        # 2. 如果特征维度大于1，为每个批次随机选择一个特征维度进行掩码 - 并行实现
        if M > 1:
            # 为每个批次随机选择一个特征维度 [B]
            random_features = torch.randint(0, M, (B,), device=device)
            
            # 创建批次索引 [B]
            batch_indices = torch.arange(B, device=device)
            
            # 创建特征掩码
            # 初始全1张量
            feature_mask = torch.ones(B, M, device=device)
            
            # 使用scatter_将选定位置设为0
            feature_mask.scatter_(1, random_features.unsqueeze(1), 0)
            
            # 将特征掩码扩展到整个时间序列
            # [B, M] -> [B, 1, M] -> [B, L, M]
            expanded_feature_mask = feature_mask.unsqueeze(1).expand(-1, L, -1)
            
            # 应用特征掩码
            masked_x = masked_x * expanded_feature_mask
            
            # 更新掩码标记
            # [B, M] -> [B, M, 1, 1] -> [B, M, patch_numbers, patch_len]
            expanded_feature_indicators = (1 - feature_mask).unsqueeze(-1).unsqueeze(-1)
            expanded_feature_indicators = expanded_feature_indicators.expand(-1, -1, patch_numbers, self.patch_len)
            
            # 合并时间掩码和特征掩码的标记，确保不超过1
            mask_indicators = torch.clamp(mask_indicators + expanded_feature_indicators, 0, 1)
        
        return masked_x, mask_indicators


if __name__ == "__main__":
    # 参数设置
    B = 8       # 批次大小
    L = 1000    # 序列长度
    M = 3       # 特征维度
    patch_len = 100  # 每个patch的长度
    stride = 100      # patch之间的步长
    
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    
    # 创建随机输入
    x = torch.randn(B, L, M)
    
    # 创建掩码器实例
    masker = TokenMasker(patch_len=patch_len, stride=stride)
    
    # 应用掩码
    masked_x, mask_indicators = masker(x)
    
    # 计算出实际的patch数量（用于验证）
    patch_numbers = (L - patch_len) // stride + 1
    
    # 检查形状
    print(f"原始输入形状: {x.shape}")                 # 应为 [8, 1000, 3]
    print(f"掩码后输出形状: {masked_x.shape}")        # 应为 [8, 1000, 3]
    print(f"掩码标记形状: {mask_indicators.shape}")   # 应为 [8, 3, patch_numbers, patch_len]
    print(f"计算得到的patch数量: {patch_numbers}")
    
    # 验证掩码效果
    # 1. 检查最后一个时间块是否被掩码
    last_patch_start = (patch_numbers - 1) * stride
    last_patch_end = last_patch_start + patch_len
    last_block_masked = torch.all(masked_x[:, last_patch_start:last_patch_end, :] == 0).item()
    print(f"所有序列的最后一个时间块都被掩码: {last_block_masked}")
    
    # 2. 对于每个批次，验证是否有一个特征维度被完全掩码
    if M > 1:
        # 计算每个特征的总和，如果为0则说明该特征被完全掩码
        feature_sums = masked_x.sum(dim=1)  # [B, M]
        # 检查每个批次是否至少有一个特征维度的总和为0
        batches_with_masked_feature = torch.any(feature_sums == 0, dim=1).sum().item()
        print(f"有特征维度被完全掩码的批次数: {batches_with_masked_feature}/{B}")
    
    print("掩码操作完成!")