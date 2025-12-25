import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
from einops import rearrange, repeat
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_channel, hidden_dim):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_channel, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channel)
        )

    def forward(self, input):
        out = self.mlp(input)
        out += input
        return out


class Encoder(nn.Module):
    def __init__(self, L, n_res_block):
        super().__init__()
        
        blocks = [
            nn.Linear(L, L // 2),
            nn.ReLU(inplace=True),
            nn.Linear(L // 2, L // 4),
            nn.ReLU(inplace=True),
            nn.Linear(L // 4, L // 8)
        ]

        hidden_dim = L // 8

        for _ in range(n_res_block):
            blocks.append(ResBlock(hidden_dim, 4*hidden_dim))
        
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)
        self.shotcut = nn.Linear(L , L // 8)
    def forward(self, input):
        return self.blocks(input)+self.shotcut(input)


class Decoder(nn.Module):
    def __init__(self, L, n_res_block):
        super().__init__()
        
        blocks = []

        hidden_dim = L //8
        for _ in range(n_res_block):
            blocks.append(ResBlock(hidden_dim, 4*hidden_dim))        
        blocks.extend([
            nn.Linear(L // 8, L // 4),
            nn.ReLU(inplace=True),
            nn.Linear(L // 4, L // 2),
            nn.ReLU(inplace=True),
            nn.Linear(L // 2, L)
        ])

        self.blocks = nn.Sequential(*blocks)
        self.shotcut = nn.Linear( L // 8 ,L)

    def forward(self, input):
        return self.blocks(input)+self.shotcut(input)

# L2 归一化
def l2norm(t):
    return F.normalize(t, p=2, dim=-1)

# EMA 更新函数
def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

def norm_ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))
    moving_avg.data.copy_(l2norm(moving_avg.data))
# 从样本中采样向量
def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]

# K-Means 聚类
def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins

# EMA 嵌入类
class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5, kmeans_init=True, codebook_init_path=''):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps 
        if codebook_init_path == '':   
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = l2norm(weight)
            else:
                weight = torch.zeros(num_tokens, codebook_dim)
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            print(f"load init codebook weight from {codebook_init_path}")
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))
        # self.codebook_project = nn.Linear(codebook_dim, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        print("Performing K-Means init for codebook")
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim=True)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
        
    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)   

class PatchFFT(nn.Module):
    def __init__(self, patch_len, stride, padding, dropout):
        super(PatchFFT, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        assert self.patch_len == self.stride, "non-overlap"


    def forward(self, x):
        n_vars = x.shape[1]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        orin_x = x
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = torch.fft.fft(x).abs()
        return x, n_vars, orin_x

# Norm EMA 向量量化器
class NormEMAVectorQuantizer(nn.Module):
    def __init__(self, args, decay=0.99, eps=1e-5, 
                statistic_code_usage=True, kmeans_init=False, codebook_init_path=''):
        super().__init__()
        self.num_tokens = args.codebook_size
        self.codebook_dim = args.patch_len//8
        self.decay = decay
        self.patch_len = args.patch_len
        self.patch_embeddings = PatchFFT(
            args.patch_len, args.stride, args.stride, args.dropout)
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps, kmeans_init, codebook_init_path)
        self.encoder= Encoder(args.patch_len,2)
        self.decoder = Decoder(args.patch_len, 2)
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:            self.register_buffer('cluster_size', torch.zeros(self.num_tokens))
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()

    
    def reset_cluster_size(self, device):
        if self.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.num_tokens))
            self.cluster_size = self.cluster_size.to(device)
    def compute_infoNCE_loss(self, student_features, teacher_features,temp=0.1):
        """
        计算InfoNCE损失用于模态对齐
        student_features: 学生模型特征 (B*V, D)
        teacher_features: 教师模型特征 (B*V, D)
        temperature: 温度参数
        """
        # L2 归一化
        B,L,D = student_features.shape
        student_features = student_features.reshape(-1,D)
        teacher_features = teacher_features.reshape(-1,D)
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
    def signal_tokenize(self, x, mask=None):
        # Normalization from Non-stationary Transformer
        x = x.permute(0, 2, 1)
        remainder = x.shape[2] % self.patch_len
        if remainder != 0:
            padding = self.patch_len - remainder
            x = F.pad(x, (0, padding))
        else:
            padding = 0
        # orin_x = 
        x, n_vars,orin_x = self.patch_embeddings(x)
        x = torch.reshape(
        x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        #进行量化
        return x, n_vars, padding,orin_x
    def forward(self, x):
        
        # 输入 z 的形状为 (B, V, L, D)
        z_orin, n_vars, padding,orin_x = self.signal_tokenize(x)
        means = z_orin.mean(dim=(0, 1, 3), keepdim=True)  # 保持维度，得到shape [1, 1, 8, 1]
        stdev = z_orin.std(dim=(0, 1, 3), keepdim=True)  # 保持维度，得到shape [1, 1, 8, 1]

        z_norm = (z_orin - means) / (stdev + 1e-5)
        # 展平 V 和 L，得到 (B, L * V, D)
        b,v,l,d = z_norm.shape

        z_norm =rearrange(z_norm, 'b v l d -> b (v l) d')

        z_h = self.encoder(z_norm)


        z_h = l2norm(z_h)  # 归一化输入特征

        # 展平成二维 (B * L * V, D)
        z_flattened = z_h.reshape(-1, self.codebook_dim)
        
        # 初始化嵌入（如果没有初始化）
        self.embedding.init_embed_(z_flattened)

        # 计算 z 和嵌入向量之间的距离
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight) 
        
        # 获取最接近的嵌入向量索引
        encoding_indices = torch.argmin(d, dim=1)

        # 从嵌入表中查找量化后的 z
        z_hq= self.embedding(encoding_indices).view(z_h.shape)
        
        
        # 对量化后的编码进行 one-hot 编码
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z_hq.dtype)     


        # 保留梯度，计算 z_q 的梯度
        z_hq = z_h + (z_hq - z_h).detach()

        # # 将 z_q 重新 reshape 回 (B, V, L, D)
        # z_q = rearrange(z_q, 'b (l v) d -> b v l d', v=v, l=l)

        z_q_norm = self.decoder(z_hq)


        # 更新 EMA（如果在训练中）
        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)
        
        if self.training and self.embedding.update:
            bins = encodings.sum(0)
            self.all_reduce_fn(bins)

            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = z_flattened.t() @ encodings
            self.all_reduce_fn(embed_sum)
                            
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            
            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight,
                                           embed_normalized)
            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)

            # 计算量化损失
        if  self.training :
            loss_dict = (F.mse_loss(z_h.detach(), z_hq) +F.mse_loss(z_h, z_hq.detach()))
            loss_recon = self.compute_infoNCE_loss(z_q_norm,z_norm)
            # loss_recon = F.mse_loss(z_q_norm,z_norm)

        # z_orin_q = z_orin_q * (stdev + 1e-5) + means
        dict = {
            'encoding_indices': encoding_indices,
            'n_vars': n_vars,
            'padding': padding,

           **({'loss_dict': loss_dict,
                'loss_recon': loss_recon,
               'bins': bins,
               } if self.training else {})
                }
        # def plot_data(z_q_norm, z_norm):
        #     import matplotlib.pyplot as plt
            
        #     # Convert tensors to numpy arrays and take absolute values for magnitude spectrum
        #     z_q_norm_np = np.abs(z_q_norm.detach().cpu().numpy())
        #     z_norm_np = np.abs(z_norm.detach().cpu().numpy())
            
        #     # Create figure and subplots
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
        #     x = np.arange(z_norm_np.shape[-1])
            
        #     # Plot original spectrum as bar chart
        #     ax1.bar(x, z_norm_np[0,1,:], alpha=0.7)
        #     ax1.set_title('Original Frequency Spectrum')
        #     ax1.set_xlabel('Frequency Bin')
        #     ax1.set_ylabel('Magnitude')
        #     ax1.set_yscale('log')  # Use log scale for better visualization
            
        #     # Plot quantized spectrum as bar chart
        #     ax2.bar(x, z_q_norm_np[0,1,:], alpha=0.7)
        #     ax2.set_title('Quantized Frequency Spectrum')
        #     ax2.set_xlabel('Frequency Bin')
        #     ax2.set_yscale('log')
            
        #     plt.tight_layout()
        #     plt.savefig('frequency_spectrum.png', dpi=300, bbox_inches='tight')
        #     return fig
        # plot_data(z_q_norm,z_norm)

        z_q_norm = rearrange(z_q_norm, 'b (v l) d -> b v l d', v=v, l=l)
        z_q_all = z_q_norm * (stdev + 1e-5) + means

        return z_q_all, dict




                



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试NormEMAVectorQuantizer')
    parser.add_argument('--codebook_size', type=int, default=512, help='码本大小')
    parser.add_argument('--patch_len', type=int, default=256, help='补丁长度')
    parser.add_argument('--stride', type=int, default=256, help='步长')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout率')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--seq_len', type=int, default=1024, help='序列长度')
    parser.add_argument('--n_vars', type=int, default=3, help='变量数量')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化模型
    model = NormEMAVectorQuantizer(args, kmeans_init=True)
    model.to(device)
    model.train()
    
    # 生成随机输入数据
    # 形状为 [batch_size, n_vars, seq_len]
    x = torch.randn(args.batch_size, args.n_vars, args.seq_len).to(device)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    print("执行前向传播...")
    z_q, output_dict = model(x)
    
    # 打印输出信息
    print(f"量化后输出形状: {z_q.shape}")
    print(f"编码索引形状: {output_dict['encoding_indices'].shape}")
    
    if model.training:
        print(f"量化损失: {output_dict['loss_dict'].item():.4f}")
        print(f"重建损失: {output_dict['loss_recon'].item():.4f}")
        
        # 打印码本使用统计
        if 'bins' in output_dict:
            bins = output_dict['bins'].cpu().numpy()
            used_codes = (bins > 0).sum()
            print(f"使用的码本大小: {used_codes}/{args.codebook_size} ({used_codes/args.codebook_size*100:.2f}%)")
    
    
    print("测试完成!")
