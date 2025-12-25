
import torch
from torch import Tensor
import math
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        '''
        in_dim: the input dimension of the layer we want to modify using LoRA
        out_dim: the respective output dimension of that layer
        rank: a hyperparameter that controls the inner dimension of the matrices A and B
        alpha: a scaling hyperparameter applied to the output of the low-rank adaptation
        '''
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        lora = self.lora.A @ self.lora.B # Combine LoRA matrices
        # Then combine LoRA with orig. weights
        combined_weight = self.linear.weight + self.lora.alpha*lora.T 
        return F.linear(x, combined_weight, self.linear.bias)