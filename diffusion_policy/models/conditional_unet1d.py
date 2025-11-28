"""
1D Conditional U-Net for Diffusion Policy

Based on the diffusion policy architecture for robot manipulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timestep encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    """1D downsampling layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """1D upsampling layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """1D convolution block with group norm and activation."""

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int,
                 n_groups: int = 8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Conditional residual block with FiLM (Feature-wise Linear Modulation)."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
                 kernel_size: int = 3, n_groups: int = 8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM conditioning
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )

        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            out: (B, out_channels, T)
        """
        out = self.blocks[0](x)

        # FiLM conditioning
        cond_emb = self.cond_encoder(cond)
        scale, shift = cond_emb.chunk(2, dim=1)
        out = out * (scale[:, :, None] + 1) + shift[:, :, None]

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    1D Conditional U-Net for diffusion policy.

    Predicts noise given noisy action sequence and condition (observations).
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
        """
        Args:
            input_dim: Dimension of action space
            global_cond_dim: Dimension of observation/condition
            diffusion_step_embed_dim: Dimension of timestep embedding
            down_dims: Channel dimensions for downsampling path
            kernel_size: Kernel size for convolutions
            n_groups: Number of groups for GroupNorm
        """
        super().__init__()

        # Timestep embedding
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Condition dimension (timestep + global condition)
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        # Input projection
        self.input_proj = Conv1dBlock(input_dim, down_dims[0], kernel_size, n_groups)

        # Encoder (downsampling)
        down_modules = []
        in_dim = down_dims[0]
        for i, out_dim in enumerate(down_dims[1:]):
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(in_dim, out_dim, cond_dim, kernel_size, n_groups),
                Downsample1d(out_dim)
            ]))
            in_dim = out_dim

        self.down_modules = nn.ModuleList(down_modules)

        # Bottleneck
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], cond_dim,
                                       kernel_size, n_groups),
            ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], cond_dim,
                                       kernel_size, n_groups),
        ])

        # Decoder (upsampling)
        up_modules = []
        up_dims = list(reversed(down_dims))
        for i in range(len(up_dims) - 1):
            out_dim = up_dims[i + 1]
            in_dim = up_dims[i] * 2  # Skip connection doubles channels

            up_modules.append(nn.ModuleList([
                Upsample1d(up_dims[i]),
                ConditionalResidualBlock1D(in_dim, out_dim, cond_dim, kernel_size, n_groups),
            ]))

        self.up_modules = nn.ModuleList(up_modules)

        # Output projection
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None
    ):
        """
        Args:
            sample: (B, T, input_dim) - noisy action sequence
            timestep: (B,) - diffusion timestep
            global_cond: (B, global_cond_dim) - observation condition
        Returns:
            noise_pred: (B, T, input_dim) - predicted noise
        """
        # Transpose for 1D conv: (B, T, C) -> (B, C, T)
        sample = sample.transpose(1, 2)

        # Encode timestep
        timestep_emb = self.diffusion_step_encoder(timestep)

        # Concatenate with global condition
        if global_cond is not None:
            global_feature = torch.cat([timestep_emb, global_cond], dim=-1)
        else:
            global_feature = timestep_emb

        # Input projection
        x = self.input_proj(sample)

        # Encoder
        h = []
        for resnet, downsample in self.down_modules:
            x = resnet(x, global_feature)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Decoder with skip connections
        for upsample, resnet in self.up_modules:
            x = upsample(x)
            x = torch.cat([x, h.pop()], dim=1)
            x = resnet(x, global_feature)

        # Output
        x = self.final_conv(x)

        # Transpose back: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        return x
