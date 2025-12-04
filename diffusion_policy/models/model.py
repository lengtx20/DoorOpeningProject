# MIT License

# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copied and adapted from Diffusion Policy

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
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

class VisionEncoder(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*(list(resnet.children())[:-1]))
        self.projection = nn.Linear(512, embedding_dim)
        self.act = nn.Mish()

    def forward(self, images):
        B, T, C, H, W = images.shape
        flat_images = images.view(B * T, C, H, W)
        
        features = self.backbone(flat_images).flatten(1)
        features = self.act(self.projection(features))
        
        return features.view(B, T, -1).flatten(1)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.GroupNorm(math.gcd(n_groups, out_channels), out_channels),
                nn.Mish()
            ),
            nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.GroupNorm(math.gcd(n_groups, out_channels), out_channels),
                nn.Mish()
            )
        ])

        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2)
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        scale, shift = embed.chunk(2, dim=1)
        out = out * (1 + scale) + shift
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256, down_dims=[256, 512, 1024]):
        super().__init__()
        start_dim = down_dims[0]
        self.mid_proj = nn.Conv1d(input_dim, start_dim, kernel_size=1)
        
        all_dims = [start_dim] + down_dims

        self.step_proj = SinusoidalPosEmb(diffusion_step_embed_dim)
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        self.down_modules = nn.ModuleList([])
        for i in range(len(down_dims)):
            in_dim = all_dims[i]
            out_dim = all_dims[i + 1]
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(in_dim, out_dim, cond_dim),
                ConditionalResidualBlock1D(out_dim, out_dim, cond_dim),
                nn.Conv1d(out_dim, out_dim, 3, stride=2, padding=1)
            ]))

        self.up_modules = nn.ModuleList([])
        for i in reversed(range(len(down_dims))):
            in_dim = all_dims[i + 1]
            out_dim = all_dims[i]
            skip_dim = all_dims[i + 1]
            
            self.up_modules.append(nn.ModuleList([
                nn.ConvTranspose1d(in_dim, out_dim, 4, stride=2, padding=1),

                ConditionalResidualBlock1D(out_dim + skip_dim, out_dim, cond_dim),
                ConditionalResidualBlock1D(out_dim, out_dim, cond_dim),
            ]))

        self.final_conv = nn.Sequential(
            nn.Conv1d(start_dim, start_dim, 5, padding=2, groups=math.gcd(8, start_dim)),
            nn.Mish(),
            nn.Conv1d(start_dim, input_dim, 1)
        )

    def forward(self, sample, timestep, global_cond):
        sample = sample.transpose(1, 2)
        sample = self.mid_proj(sample)
        
        timesteps_emb = self.step_proj(timestep)
        cond = torch.cat([timesteps_emb, global_cond], dim=-1)

        h = []
        for res1, res2, down in self.down_modules:
            sample = res1(sample, cond)
            sample = res2(sample, cond)
            h.append(sample)
            sample = down(sample)

        for up, res1, res2 in self.up_modules:
            sample = up(sample)
            skip = h.pop()
            if sample.shape[-1] != skip.shape[-1]:
                sample = F.interpolate(sample, size=skip.shape[-1], mode='nearest')
            sample = torch.cat((sample, skip), dim=1)
            sample = res1(sample, cond)
            sample = res2(sample, cond)

        sample = self.final_conv(sample)
        return sample.transpose(1, 2)