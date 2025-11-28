"""
Diffusion Policy with DDPM/DDIM sampling

Implements the full diffusion policy model with noise scheduling and sampling.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .conditional_unet1d import ConditionalUnet1D


class DiffusionUNet(nn.Module):
    """
    Diffusion Policy using U-Net noise predictor with DDPM/DDIM sampling.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        num_diffusion_iters: int = 100,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        obs_encoder_layers: Tuple[int, ...] = (256, 256),
        **kwargs
    ):
        """
        Args:
            obs_dim: Observation dimension (e.g., 47 for G1)
            action_dim: Action dimension (e.g., 29 for joint positions)
            obs_horizon: Number of observation frames to condition on
            pred_horizon: Number of action steps to predict
            action_horizon: Number of action steps to execute
            num_diffusion_iters: Number of diffusion denoising steps
            down_dims: U-Net downsampling dimensions
            obs_encoder_layers: MLP layers for encoding observations
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.num_diffusion_iters = num_diffusion_iters

        # Observation encoder (MLP)
        obs_encoder = []
        in_dim = obs_dim * obs_horizon
        for hidden_dim in obs_encoder_layers:
            obs_encoder.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.obs_encoder = nn.Sequential(*obs_encoder)
        obs_feature_dim = obs_encoder_layers[-1]

        # Noise prediction network (U-Net)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_feature_dim,
            down_dims=down_dims,
            **kwargs
        )

        # DDPM noise schedule (cosine schedule)
        self.register_buffer(
            'betas',
            self._cosine_beta_schedule(num_diffusion_iters)
        )
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observation sequence.

        Args:
            obs: (B, obs_horizon, obs_dim)
        Returns:
            obs_feature: (B, obs_feature_dim)
        """
        B, T, D = obs.shape
        obs_flat = obs.reshape(B, -1)
        return self.obs_encoder(obs_flat)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor,
                timestep: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass: predict noise.

        Args:
            obs: (B, obs_horizon, obs_dim)
            actions: (B, pred_horizon, action_dim) - noisy actions
            timestep: (B,) - diffusion timestep
        Returns:
            noise_pred: (B, pred_horizon, action_dim)
        """
        obs_feature = self.encode_obs(obs)
        noise_pred = self.noise_pred_net(actions, timestep, obs_feature)
        return noise_pred

    @torch.no_grad()
    def conditional_sample(
        self,
        obs: torch.Tensor,
        use_ddim: bool = False,
        ddim_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample action sequence given observations using DDPM or DDIM.

        Args:
            obs: (B, obs_horizon, obs_dim)
            use_ddim: Use DDIM sampling (faster)
            ddim_steps: Number of DDIM steps (if None, use num_diffusion_iters)
        Returns:
            actions: (B, pred_horizon, action_dim)
        """
        B = obs.shape[0]
        device = obs.device

        # Encode observations
        obs_feature = self.encode_obs(obs)

        # Start from random noise
        actions = torch.randn(B, self.pred_horizon, self.action_dim, device=device)

        if use_ddim and ddim_steps is not None:
            # DDIM sampling
            actions = self._ddim_sample(actions, obs_feature, ddim_steps)
        else:
            # DDPM sampling
            actions = self._ddpm_sample(actions, obs_feature)

        return actions

    def _ddpm_sample(self, x: torch.Tensor, obs_feature: torch.Tensor) -> torch.Tensor:
        """
        DDPM sampling (full reverse diffusion process).

        Args:
            x: (B, pred_horizon, action_dim) - initial noise
            obs_feature: (B, obs_feature_dim) - encoded observations
        Returns:
            x: (B, pred_horizon, action_dim) - denoised actions
        """
        for t in reversed(range(self.num_diffusion_iters)):
            timestep = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

            # Predict noise
            noise_pred = self.noise_pred_net(x, timestep, obs_feature)

            # Compute denoising step
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

            # Predicted x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_t_prev) * noise_pred

            # Add noise if not final step
            if t > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + torch.sqrt(self.betas[t]) * noise
            else:
                x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

        return x

    def _ddim_sample(self, x: torch.Tensor, obs_feature: torch.Tensor,
                     num_steps: int, eta: float = 0.0) -> torch.Tensor:
        """
        DDIM sampling (deterministic, faster).

        Args:
            x: (B, pred_horizon, action_dim) - initial noise
            obs_feature: (B, obs_feature_dim) - encoded observations
            num_steps: Number of sampling steps
            eta: Stochasticity parameter (0 = deterministic)
        Returns:
            x: (B, pred_horizon, action_dim) - denoised actions
        """
        # Create timestep schedule
        step_size = self.num_diffusion_iters // num_steps
        timesteps = torch.arange(0, self.num_diffusion_iters, step_size, device=x.device)
        timesteps = torch.flip(timesteps, dims=[0])

        for i, t in enumerate(timesteps):
            timestep_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

            # Predict noise
            noise_pred = self.noise_pred_net(x, timestep_batch, obs_feature)

            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            if i < len(timesteps) - 1:
                alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=x.device)

            # Predicted x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # DDIM update
            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * noise_pred

            if i < len(timesteps) - 1:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt + sigma_t * noise
            else:
                x = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

        return x

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss.

        Args:
            obs: (B, obs_horizon, obs_dim)
            actions: (B, pred_horizon, action_dim) - clean actions
        Returns:
            loss_dict: Dictionary with 'loss' and other metrics
        """
        B = obs.shape[0]
        device = obs.device

        # Sample random timesteps
        timesteps = torch.randint(0, self.num_diffusion_iters, (B,), device=device)

        # Sample noise
        noise = torch.randn_like(actions)

        # Add noise to actions (forward diffusion)
        alpha_t = self.alphas_cumprod[timesteps]
        noisy_actions = (
            torch.sqrt(alpha_t[:, None, None]) * actions +
            torch.sqrt(1 - alpha_t[:, None, None]) * noise
        )

        # Predict noise
        noise_pred = self.forward(obs, noisy_actions, timesteps)

        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        return {
            'loss': loss,
            'mse': loss.item(),
        }
