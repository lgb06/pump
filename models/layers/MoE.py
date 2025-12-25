import sys
sys.path.append('/dataWYL/WYL/PHM-Large-Model/')
import torch
import torch.nn.functional as F
from torch import Tensor, nn
class DeepSeekMoE(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int,
                capacity_factor: float = 1.0, drop_tokens: bool = True):
        """
        Args:
            hidden_dim (int): 隐藏层维度，等于输入特征的维度，也是专家和共享网络的输入/输出维度。
            num_experts (int): 除共享专家之外的网络数量，即模型中并行处理数据的专家数目。
            top_k (int): 每个样本选择的前 k 个分数最高的专家，用于动态路由/专家激活。
            capacity_factor (float, 可选): 专家额外分配的容量因子，用来控制专家实际可处理的最大数据量，
                                            较大的值会增加每个专家的处理容量。默认值为 1.0。
            drop_tokens (bool, 可选): 是否丢弃未被选中的令牌（token）。如果为 True，
                                    则只处理被选中的专家对应的令牌，从而提升计算效率。默认值为 True。
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens

        # Shared expert
        self.shared_expert = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        # Routed experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating with auxiliary-loss-free balancing
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.bias = nn.Parameter(torch.zeros(num_experts))
        self.bias_update_speed = 0.001
        
        # Expert load tracking (initialized as a buffer)
        self.register_buffer('expert_load', torch.zeros(num_experts))

    def forward(self, x: Tensor) -> Tensor:
        bs,var,len,hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        combined = torch.zeros_like(x_flat)

        # Calculate scores with bias
        scores = self.gate(x_flat) + self.bias.unsqueeze(0)
        scores = F.sigmoid(scores)
        top_scores, top_indices = scores.topk(self.top_k, dim=-1)

        # Adjust top_indices to include all selected experts
        mask = F.one_hot(top_indices, self.num_experts).float().sum(dim=1)
        expert_load = mask.sum(dim=0)
        self.bias.data += self.bias_update_speed * (expert_load - self.bias.data)
        self.expert_load = 0.9 * self.expert_load + 0.1 * expert_load

        # Collect outputs and indices for selected experts
        expert_outputs = []
        indices = []

        for expert_idx in range(self.num_experts):
            selected = (top_indices == expert_idx).any(dim=-1)
            selected_indices = torch.where(selected)[0]
            if not selected.any():
                continue

            expert_in = x_flat[selected]
            expert_out = self.experts[expert_idx](expert_in)

            # Get scores and expand them
            positions = (top_indices[selected] == expert_idx).nonzero(as_tuple=True)[1]
            expert_weights = top_scores[selected, positions].unsqueeze(-1)
            weighted_out = expert_weights * expert_out

            expert_outputs.append(weighted_out)
            indices.append(selected_indices)

        # Scatter the expert outputs into combined
        if expert_outputs:
            outputs_cat = torch.cat(expert_outputs, dim=0)
            indices_cat = torch.cat(indices, dim=0)
            indices_expanded = indices_cat.unsqueeze(-1).expand(-1, outputs_cat.size(-1))
            combined.scatter_add_(0, indices_expanded, outputs_cat)

        # Add shared expert output
        shared_out = self.shared_expert(x)+ combined.view_as(x)
        return shared_out.view(bs,var,len,hidden_dim)