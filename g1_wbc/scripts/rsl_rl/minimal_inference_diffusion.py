"""
Minimal self-contained diffusion policy.

This single file implements a tiny policy capable of:
 - constructing a minimal vision+unet-style model
 - loading a checkpoint (best-effort)
 - accepting image + proprio inputs and returning predicted actions

The file only depends on torch and numpy. Inference is intentionally
naive: it calls the learned model once with a fixed timestep and
returns the model output. This is sufficient to validate shape
management and checkpoint wiring.

Usage:
  python diffusion_policy/minimal_policy_onefile.py --model /path/to/checkpoint.pt

If no checkpoint is given the model stays at random initialization and
the script still prints the predicted shape.
"""
from pathlib import Path
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TinyVisionEncoder(nn.Module):
    """Small conv-based vision encoder to avoid torchvision dependency."""
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, embedding_dim)

    def forward(self, images):
        # images: (B, T, C, H, W)
        B, T, C, H, W = images.shape
        flat = images.view(B * T, C, H, W)
        feat = self.conv(flat).view(B * T, -1)
        feat = self.proj(feat)
        return feat.view(B, T, -1).flatten(1)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, kernel_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(),
        )
        self.cond = nn.Linear(cond_dim, out_ch * 2)
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        out = self.net(x)
        embed = self.cond(cond).unsqueeze(-1)
        scale, shift = embed.chunk(2, dim=1)
        out = out * (1 + scale) + shift
        return out + self.res_conv(x)


class SimpleUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, hidden=128):
        super().__init__()
        self.mid = nn.Conv1d(input_dim, hidden, 1)
        self.res1 = ConditionalResidualBlock1D(hidden, hidden, global_cond_dim)
        self.res2 = ConditionalResidualBlock1D(hidden, hidden, global_cond_dim)
        self.final = nn.Conv1d(hidden, input_dim, 1)

    def forward(self, sample, timestep, global_cond):
        # sample: (B, T, action_dim)
        x = sample.transpose(1, 2)
        x = self.mid(x)
        x = self.res1(x, global_cond)
        x = self.res2(x, global_cond)
        x = self.final(x)
        return x.transpose(1, 2)


class MinimalPolicy(nn.Module):
    """A single-file policy that can load a checkpoint and predict actions.

    Prediction is intentionally simple: it computes global condition and
    runs the unet once with a fixed timestep; the returned tensor has
    shape (B, pred_horizon, action_dim).
    """
    def __init__(self,
                 action_dim=10,
                 obs_horizon=2,
                 pred_horizon=16,
                 vision_feat=64,
                 proprio_dim=29,
                 use_proprio=True,
                 timestep_emb_dim=64):
        super().__init__()
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.use_proprio = use_proprio

        self.vision = TinyVisionEncoder(embedding_dim=vision_feat)
        vision_out = obs_horizon * vision_feat

        if use_proprio:
            self.proprio_dim = proprio_dim
            self.proprio_mlp = nn.Sequential(
                nn.Linear(obs_horizon * proprio_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
            )
            global_cond_dim = vision_out + 64 + timestep_emb_dim
        else:
            global_cond_dim = vision_out + timestep_emb_dim

        self.timestep_proj = SinusoidalPosEmb(timestep_emb_dim)
        self.unet = SimpleUnet1D(input_dim=action_dim, global_cond_dim=vision_out + (64 if use_proprio else 0) + timestep_emb_dim)

    def get_global_cond(self, image, agent_pos=None):
        # image: (B, obs_horizon, C, H, W)
        vision = self.vision(image)
        if self.use_proprio and agent_pos is not None:
            B = agent_pos.shape[0]
            p = agent_pos.view(B, -1)
            pfeat = self.proprio_mlp(p)
            return torch.cat([vision, pfeat], dim=-1)
        return vision

    def load(self, path: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        ckpt = torch.load(str(path), map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt

        # attempt direct load, otherwise strip common prefixes
        try:
            self.load_state_dict(state)
            return
        except Exception:
            new = {}
            for k, v in state.items():
                if k.startswith('module.'):
                    new[k.replace('module.', '')] = v
                elif k.startswith('model.'):
                    new[k.replace('model.', '')] = v
                else:
                    new[k] = v
            self.load_state_dict(new, strict=False)

    @torch.no_grad()
    def predict(self, image: torch.Tensor, agent_pos: torch.Tensor = None):
        """Run a naive inference pass.

        image: (B, obs_horizon, C, H, W)
        agent_pos: (B, obs_horizon, proprio_dim) or None
        returns: (B, pred_horizon, action_dim)
        """
        device = image.device
        B = image.shape[0]
        global_cond = self.get_global_cond(image, agent_pos)
        # create a timestep embedding (zeros)
        t = torch.zeros((B,), dtype=torch.long, device=device)
        t_emb = self.timestep_proj(t.float()) if isinstance(self.timestep_proj, SinusoidalPosEmb) else torch.zeros((B, 64), device=device)
        # append timestep to global_cond for unet
        # unet expects global_cond to be provided inside forward; our simple unet uses global_cond in blocks
        full_cond = torch.cat([global_cond, t_emb], dim=-1)

        # dummy initial sample (gaussian noise)
        sample = torch.randn((B, self.pred_horizon, self.action_dim), device=device)
        out = self.unet(sample, t, full_cond)
        return out


def _demo(model_path=None):
    device = torch.device('cpu')
    policy = MinimalPolicy()
    if model_path is not None:
        try:
            policy.load(model_path)
            print('Loaded checkpoint:', model_path)
        except Exception as e:
            print('Failed to load checkpoint (continuing with random weights):', e)

    policy.to(device)

    B = 1
    obs_horizon = policy.obs_horizon
    C, H, W = 3, 64, 64
    img = torch.randn((B, obs_horizon, C, H, W), device=device)
    proprio = torch.randn((B, obs_horizon, policy.proprio_dim), device=device) if policy.use_proprio else None

    out = policy.predict(img, proprio)
    print('image:', img.shape)
    if proprio is not None:
        print('proprio:', proprio.shape)
    print('actions:', out.shape)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default=None)
    args = p.parse_args()
    _demo(args.model)
