# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Residual vector quantizer implementation."""

from dataclasses import dataclass, field
import math
import typing as tp
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from .core_vq import ResidualVectorQuantization
from models.layers.Quantizer import l2norm

class PatchFFT_Tokenizer(nn.Module):
    def __init__(self, args):
        super(PatchFFT_Tokenizer, self).__init__()
        # Patching
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.input_len = args.input_len
        self.d_model = args.d_model
        self.patch_project = nn.Linear(self.patch_len, self.d_model)
        self.sequence_project = nn.Linear(self.input_len, self.d_model)
        assert self.patch_len == self.stride, "non-overlap"

    def patch_embeddings(self, x):
        # 现在的 FFT 特征是：全局的,不随 patch 变化
        # patch 提供的是“局部内容”  (x)   ;    x只做了「切片（unfold）」和「reshape」
        # FFT   提供的是“背景信息”  (seq) ;    频谱的幅值 + 相位, 前一半是幅度,后一半是相位
        bs,n_vars,seq_len = x.shape
        # Compute FFT once and store the result
        fft_result = torch.fft.fft(x)
        # Extract amplitude (abs) and phase (angle) in one go
        seq_fft = fft_result.abs()[:,:,:seq_len//2]
        seq_phase = fft_result.angle()[:,:,:seq_len//2]
        seq = torch.cat([seq_fft,seq_phase],dim=-1)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # (bs, n_vars, seq_len)  ==>  ( bs,  n_vars,  num_patches,  patch_len )  
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return x, n_vars, seq
    
    def forward(self, x):
        
        x = x.permute(0, 2, 1)  # 将输入张量的形状从 (batch_size, seq_length, channels) 转换为 (batch_size, channels, seq_length)
        
        remainder = x.shape[2] % self.patch_len
        if remainder != 0:
            padding = self.patch_len - remainder
            x = F.pad(x, (0, padding))
        else:
            padding = 0
        
        # orin_x = 
        x, n_vars,seq_x = self.patch_embeddings(x)
        x = torch.reshape(
        x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        
        x = self.patch_project(x)
        
        seq_x = self.sequence_project(seq_x).unsqueeze(2)
        
        #进行量化
        return x,seq_x


    # def forward(self, x):
    #     print(f"PatchFFT_Tokenizer: x.shape:{x.shape}")
    #     # x = x.permute(0, 2, 1)
    #     print(f"PatchFFT_Tokenizer: x.shape:{x.shape}")
    #     remainder = x.shape[2] % self.patch_len
    #     if remainder != 0:
    #         padding = self.patch_len - remainder
    #         x = F.pad(x, (0, padding))
    #     else:
    #         padding = 0
    #     print(f"PatchFFT_Tokenizer: x.shape:{x.shape}")
    #     # orin_x = 
    #     x, n_vars,seq_x = self.patch_embeddings(x)
    #     x = torch.reshape(
    #     x, (-1, n_vars, x.shape[-2], x.shape[-1]))
    #     print(f"x.shape, seq_x.shape:{x.shape, seq_x.shape}")
    #     x = self.patch_project(x)
    #     print(f"x.shape, seq_x.shape:{x.shape, seq_x.shape}")
    #     print(f"self.input_len, self.d_model:{self.input_len, self.d_model}")
    #     seq_x = self.sequence_project(seq_x).unsqueeze(2)
    #     print(f"PatchFFT_Tokenizer: x.shape, seq_x.shape:{x.shape, seq_x.shape}")
    #     #进行量化
    #     return x,seq_x