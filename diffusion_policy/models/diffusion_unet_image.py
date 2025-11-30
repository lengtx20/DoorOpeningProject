"""
Diffusion Policy with Image + 29 DOF observations and 10-dim actions.

Input:
- Images (visual observations)
- 29 DOF state (joint positions + optionally velocities)

Output:
- 10-dim actions (left hand pos, right hand pos, base vel cmd, grasp state)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, Tuple
from .conditional_unet1d import ConditionalUnet1D


class ImageEncoder(nn.Module):
    """CNN encoder for image observations."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        feature_dim: int = 256,
        backbone: str = 'resnet18',
    ):
        """
        Args:
            image_size: (height, width) of input images
            feature_dim: Output feature dimension
            backbone: Backbone architecture ('resnet18', 'resnet34', 'resnet50')
        """
        super().__init__()
        
        # Load pretrained ResNet backbone
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            backbone_dim = 512
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)
            backbone_dim = 512
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Project to desired feature dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, T, C, H, W) or (B*T, C, H, W) image sequence
        Returns:
            features: (B, T, feature_dim) or (B*T, feature_dim)
        """
        original_shape = images.shape
        if images.ndim == 5:
            # (B, T, C, H, W) -> (B*T, C, H, W)
            B, T = images.shape[:2]
            images = images.reshape(B * T, *images.shape[2:])
            reshape_back = True
        else:
            reshape_back = False
        
        # Extract features
        features = self.backbone(images)  # (B*T, backbone_dim, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B*T, backbone_dim)
        features = self.projection(features)  # (B*T, feature_dim)
        
        if reshape_back:
            features = features.reshape(B, T, -1)  # (B, T, feature_dim)
        
        return features


class DiffusionUNetImage(nn.Module):
    """
    Diffusion Policy with image + 29 DOF observations.
    
    Uses CNN encoder for images and MLP encoder for 29 DOF state.
    Outputs 10-dim actions.
    """

    def __init__(
        self,
        dof_dim: int = 58,  # 29 (positions) or 58 (positions + velocities)
        action_dim: int = 10,  # 10-dim actions
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        num_diffusion_iters: int = 100,
        image_size: Tuple[int, int] = (224, 224),
        image_feature_dim: int = 256,
        dof_encoder_layers: Tuple[int, ...] = (256, 256),
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        image_backbone: str = 'resnet18',
        **kwargs
    ):
        """
        Args:
            dof_dim: Dimension of DOF state (29 or 58)
            action_dim: Action dimension (10 for door opening)
            obs_horizon: Number of observation frames to condition on
            pred_horizon: Number of action steps to predict
            action_horizon: Number of action steps to execute
            num_diffusion_iters: Number of diffusion denoising steps
            image_size: (height, width) of input images
            image_feature_dim: Feature dimension from image encoder
            dof_encoder_layers: MLP layers for encoding DOF state
            down_dims: U-Net downsampling dimensions
            image_backbone: Image encoder backbone ('resnet18', 'resnet34', 'resnet50')
        """
        super().__init__()

        self.dof_dim = dof_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.num_diffusion_iters = num_diffusion_iters

        # Image encoder (CNN)
        self.image_encoder = ImageEncoder(
            image_size=image_size,
            feature_dim=image_feature_dim,
            backbone=image_backbone,
        )

        # DOF state encoder (MLP)
        dof_encoder = []
        in_dim = dof_dim * obs_horizon
        for hidden_dim in dof_encoder_layers:
            dof_encoder.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        self.dof_encoder = nn.Sequential(*dof_encoder)
        dof_feature_dim = dof_encoder_layers[-1]

        # Combine image and DOF features
        # Option 1: Concatenate
        # Option 2: Add (requires same dimension)
        # We'll use concatenation for flexibility
        obs_feature_dim = image_feature_dim + dof_feature_dim

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
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def encode_obs(
        self,
        images: torch.Tensor,
        dof_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode observation sequence (images + DOF state).

        Args:
            images: (B, obs_horizon, C, H, W) image sequence
            dof_state: (B, obs_horizon, dof_dim) DOF state sequence
        Returns:
            obs_feature: (B, obs_feature_dim) combined feature
        """
        B, T, C, H, W = images.shape
        
        # Encode images: (B, T, C, H, W) -> (B, T, image_feature_dim)
        image_features = self.image_encoder(images)  # (B, T, image_feature_dim)
        
        # Average over time or use last frame
        # Option: Use temporal average
        image_feature = image_features.mean(dim=1)  # (B, image_feature_dim)
        # Alternative: Use last frame
        # image_feature = image_features[:, -1, :]
        
        # Encode DOF state: (B, T, dof_dim) -> (B, dof_feature_dim)
        dof_flat = dof_state.reshape(B, -1)  # (B, T * dof_dim)
        dof_feature = self.dof_encoder(dof_flat)  # (B, dof_feature_dim)
        
        # Concatenate features
        obs_feature = torch.cat([image_feature, dof_feature], dim=-1)  # (B, obs_feature_dim)
        
        return obs_feature

    def forward(
        self,
        images: torch.Tensor,
        dof_state: torch.Tensor,
        actions: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Training forward pass: predict noise.

        Args:
            images: (B, obs_horizon, C, H, W)
            dof_state: (B, obs_horizon, dof_dim)
            actions: (B, pred_horizon, action_dim) - noisy actions
            timestep: (B,) - diffusion timestep
        Returns:
            noise_pred: (B, pred_horizon, action_dim)
        """
        obs_feature = self.encode_obs(images, dof_state)
        noise_pred = self.noise_pred_net(actions, timestep, obs_feature)
        return noise_pred

    @torch.no_grad()
    def conditional_sample(
        self,
        images: torch.Tensor,
        dof_state: torch.Tensor,
        use_ddim: bool = False,
        ddim_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample action sequence given observations using DDPM or DDIM.

        Args:
            images: (B, obs_horizon, C, H, W)
            dof_state: (B, obs_horizon, dof_dim)
            use_ddim: Use DDIM sampling (faster)
            ddim_steps: Number of DDIM steps (if None, use num_diffusion_iters)
        Returns:
            actions: (B, pred_horizon, action_dim)
        """
        B = images.shape[0]
        device = images.device

        # Encode observations
        obs_feature = self.encode_obs(images, dof_state)

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
        """DDPM sampling (full reverse diffusion process)."""
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

    def _ddim_sample(
        self,
        x: torch.Tensor,
        obs_feature: torch.Tensor,
        num_steps: int,
        eta: float = 0.0
    ) -> torch.Tensor:
        """DDIM sampling (deterministic, faster)."""
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
        images: torch.Tensor,
        dof_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss.

        Args:
            images: (B, obs_horizon, C, H, W)
            dof_state: (B, obs_horizon, dof_dim)
            actions: (B, pred_horizon, action_dim) - clean actions
        Returns:
            loss_dict: Dictionary with 'loss' and other metrics
        """
        B = images.shape[0]
        device = images.device

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
        noise_pred = self.forward(images, dof_state, noisy_actions, timesteps)

        # MSE loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        return {
            'loss': loss,
            'mse': loss.item(),
        }

