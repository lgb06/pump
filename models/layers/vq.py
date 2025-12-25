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

@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer.
    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
        )

    def forward(self, x: torch.Tensor, n_q: tp.Optional[int] = None, layers: tp.Optional[list] = None) -> QuantizedResult:
        """Residual vector quantization on the given input tensor.
        Args:
            x (torch.Tensor): Input tensor.
            n_q (int): Number of quantizer used to quantize. Default: All quantizers.
            layers (list): Layer that need to return quantized. Defalt: None.
        Returns:
            QuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated numbert quantizers and layer quantized required to return.
        """
        n_q = n_q if n_q else self.n_q
        if layers and max(layers) >= n_q:
            raise ValueError(f'Last layer index in layers: A {max(layers)}. Number of quantizers in RVQ: B {self.n_q}. A must less than B.')
        quantized, codes, commit_loss, quantized_list = self.vq(x, n_q=n_q, layers=layers)
        return quantized, codes, torch.mean(commit_loss), quantized_list


    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None, st: tp.Optional[int] = None) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        Args:
            x (torch.Tensor): Input tensor.
            n_q (int): Number of quantizer used to quantize. Default: All quantizers.
            st (int): Start to encode input from which layers. Default: 0.
        """
        n_q = n_q if n_q else self.n_q
        st = st or 0
        codes = self.vq.encode(x, n_q=n_q, st=st)
        return codes

    def decode(self, codes: torch.Tensor, st: int = 0) -> torch.Tensor:
        """Decode the given codes to the quantized representation.
        Args:
            codes (torch.Tensor): Input indices for each quantizer.
            st (int): Start to decode input codes from which layers. Default: 0.
        """
        quantized = self.vq.decode(codes, st=st)
        return quantized


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
class ResidualTokenizer(nn.Module):
    def __init__(self, args, decay=0.99, eps=1e-5, 
            statistic_code_usage=True, kmeans_init=False, codebook_init_path=''):
            super().__init__()
            self.num_tokens = args.codebook_size
            self.codebook_dim = args.patch_len
            self.decay = decay
            self.patch_len = args.patch_len
            self.patch_embeddings = PatchFFT(
            args.patch_len, args.stride, args.stride, args.dropout)
            # self.encoder= Encoder(args.patch_len,2)
            self.encoder =nn.Linear(args.patch_len, args.patch_len)
            self.decoder = nn.Linear(args.patch_len, args.patch_len)
            # self.decoder = Decoder(args.patch_len, 2)
            self.vq = ResidualVectorQuantizer(dimension=self.codebook_dim,bins = self.num_tokens,
                                              decay=self.decay, kmeans_init=kmeans_init)

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
    def forward(self,x):
        z_orin, n_vars, padding,orin_x = self.signal_tokenize(x)
        means = z_orin.mean(dim=(0, 1, 3), keepdim=True)  # 保持维度，得到shape [1, 1, 8, 1]
        stdev = z_orin.std(dim=(0, 1, 3), keepdim=True)  # 保持维度，得到shape [1, 1, 8, 1]

        z_norm = (z_orin - means) / (stdev + 1e-5)
        # 展平 V 和 L，得到 (B, L * V, D)
        b,v,l,d = z_norm.shape

        z_norm =rearrange(z_norm, 'b v l d -> b (v l) d')

        z_h = self.encoder(z_norm).transpose(1, 2)
        z_h =l2norm(z_h) 
        z_hq, encodings, loss_dict, quantized_list = self.vq(z_h)
        # z_hq  = z_h+z_hq
        z_q_norm = self.decoder(z_hq.transpose(1, 2))

        if self.training:
            loss_recon = F.mse_loss(z_q_norm, z_norm)

        z_q_norm = rearrange(z_q_norm, 'b (v l) d -> b v l d', v=v, l=l)

        z_q = z_q_norm * (stdev + 1e-5) + means


        dict = {
            'encoding_indices': encodings,
            'n_vars': n_vars,
            'padding': padding,

           **({'loss_dict': loss_dict,
                'loss_recon': loss_recon,
            #    'bins': bins,
               } if self.training else {})
                }
        return z_q, dict