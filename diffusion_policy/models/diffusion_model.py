"""
Diffusion Policy Implementation

Based on: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
Paper: https://arxiv.org/abs/2303.04137
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Conv1dBlock(nn.Module):
    """1D Convolutional block with GroupNorm and Mish activation."""

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int = 8,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ):
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

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
            cond: [B, cond_dim]
        """
        out = self.blocks[0](x)

        # FiLM conditioning
        cond_emb = self.cond_encoder(cond)
        scale, shift = torch.chunk(cond_emb, 2, dim=1)
        out = out * (scale[:, :, None] + 1) + shift[:, :, None]

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """
    1D U-Net for noise prediction in diffusion.

    Predicts the noise added to actions conditioned on observations and diffusion timestep.
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: list = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        """
        Args:
            input_dim: Dimension of input (action_dim)
            global_cond_dim: Dimension of global conditioning (obs_embedding_dim)
            diffusion_step_embed_dim: Dimension of timestep embedding
            down_dims: Channel dimensions for downsampling
            kernel_size: Kernel size for convolutions
            n_groups: Number of groups for GroupNorm
        """
        super().__init__()

        all_dims = [input_dim] + down_dims
        start_dim = down_dims[0]

        # Diffusion timestep embedding
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Combine diffusion timestep and global conditioning
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        # Initial projection
        self.input_proj = Conv1dBlock(input_dim, start_dim, kernel_size, n_groups=n_groups)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_dims)):
            in_ch = all_dims[i]
            out_ch = all_dims[i + 1]
            self.down_blocks.append(
                ConditionalResidualBlock1D(
                    in_ch, out_ch, cond_dim, kernel_size, n_groups=n_groups
                )
            )

        # Middle block
        self.mid_blocks = nn.ModuleList([
            ConditionalResidualBlock1D(
                down_dims[-1], down_dims[-1], cond_dim, kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                down_dims[-1], down_dims[-1], cond_dim, kernel_size, n_groups=n_groups
            ),
        ])

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(down_dims))):
            in_ch = all_dims[i + 1]
            out_ch = all_dims[i]
            self.up_blocks.append(
                ConditionalResidualBlock1D(
                    in_ch * 2, out_ch, cond_dim, kernel_size, n_groups=n_groups  # *2 for skip connection
                )
            )

        # Final projection
        self.final_proj = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy actions [B, action_dim, pred_horizon]
            timestep: Diffusion timestep [B]
            global_cond: Observation embedding [B, global_cond_dim]

        Returns:
            Predicted noise [B, action_dim, pred_horizon]
        """
        # Encode timestep
        timestep_emb = self.diffusion_step_encoder(timestep)

        # Combine conditioning
        global_feature = torch.cat([timestep_emb, global_cond], dim=-1)

        # Initial projection
        x = self.input_proj(x)

        # Downsampling with skip connections
        skip_connections = [x]
        for down_block in self.down_blocks:
            x = down_block(x, global_feature)
            skip_connections.append(x)

        # Middle blocks
        for mid_block in self.mid_blocks:
            x = mid_block(x, global_feature)

        # Upsampling with skip connections
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = up_block(x, global_feature)

        # Final projection
        x = self.final_proj(x)
        return x


class VisionEncoder(nn.Module):
    """Vision encoder using ResNet18 backbone."""

    def __init__(self, output_dim: int = 512):
        super().__init__()
        from torchvision import models

        # Use ResNet18 as backbone with updated API
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # ResNet18 outputs 512 features
        self.fc = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, obs_horizon, 3, H, W] image tensor

        Returns:
            features: [batch, obs_horizon, output_dim] image features
        """
        batch_size, obs_horizon, c, h, w = x.shape

        # Reshape to process all images in batch
        x = x.view(batch_size * obs_horizon, c, h, w)

        # Extract features
        features = self.backbone(x)  # [batch*obs_horizon, 512, 1, 1]
        features = features.view(batch_size * obs_horizon, -1)  # [batch*obs_horizon, 512]

        # Project to output dimension
        features = self.fc(features)  # [batch*obs_horizon, output_dim]

        # Reshape back
        features = features.view(batch_size, obs_horizon, -1)  # [batch, obs_horizon, output_dim]

        return features


class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy for robotic manipulation.

    Architecture:
    1. Observation encoder: Maps observations to latent space
    2. Noise prediction network (U-Net): Predicts noise given noisy actions and observations
    3. Diffusion process: Iteratively denoises random noise to generate actions
    """

    def __init__(
        self,
        proprio_dim: int = 29,
        stage_dim: int = 5,
        action_dim: int = 10,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        obs_embedding_dim: int = 256,
        diffusion_step_embed_dim: int = 128,
        down_dims: list = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
        num_diffusion_iters: int = 100,
        num_inference_steps: int = 10,
        beta_schedule: str = "squaredcos_cap_v2",
        use_image: bool = False,
        vision_feature_dim: int = 512,
    ):
        """
        Args:
            proprio_dim: Dimension of proprioceptive input
            stage_dim: Dimension of stage one-hot encoding
            action_dim: Dimension of action output
            obs_horizon: Number of observation steps
            pred_horizon: Number of action steps to predict
            action_horizon: Number of actions to execute before replanning
            obs_embedding_dim: Dimension of observation embedding
            diffusion_step_embed_dim: Dimension of timestep embedding
            down_dims: U-Net downsampling dimensions
            kernel_size: Convolution kernel size
            n_groups: Group normalization groups
            num_diffusion_iters: Training denoising steps
            num_inference_steps: Inference denoising steps (DDIM)
            beta_schedule: Noise schedule type
            use_image: Whether to use image input
            vision_feature_dim: Dimension of vision features from ResNet
        """
        super().__init__()

        self.proprio_dim = proprio_dim
        self.stage_dim = stage_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_embedding_dim = obs_embedding_dim
        self.use_image = use_image

        # Vision encoder (if using images)
        if use_image:
            self.vision_encoder = VisionEncoder(output_dim=vision_feature_dim)
            vision_input_dim = vision_feature_dim * obs_horizon
        else:
            self.vision_encoder = None
            vision_input_dim = 0

        # Observation encoder
        proprio_input_dim = proprio_dim * obs_horizon
        stage_input_dim = stage_dim
        total_input_dim = proprio_input_dim + stage_input_dim + vision_input_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, obs_embedding_dim),
            nn.ReLU(),
        )

        # Noise prediction network
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_embedding_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

        # Noise schedulers
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule=beta_schedule,
            clip_sample=True,
            prediction_type='epsilon',  # Predict noise
        )

        self.num_inference_steps = num_inference_steps

    def encode_obs(
        self,
        proprio: torch.Tensor,
        stage: torch.Tensor,
        image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode multimodal observations (images + proprio + stage).

        Args:
            proprio: [B, obs_horizon, proprio_dim]
            stage: [B, stage_dim] (one-hot encoding)
            image: [B, obs_horizon, 3, H, W] RGB images (optional)

        Returns:
            obs_embedding: [B, obs_embedding_dim]
        """
        B = proprio.shape[0]

        # Flatten proprioception
        proprio_flat = proprio.reshape(B, -1)  # [B, obs_horizon * proprio_dim]

        # Process images if available
        if self.use_image and image is not None:
            vision_features = self.vision_encoder(image)  # [B, obs_horizon, vision_feature_dim]
            vision_flat = vision_features.reshape(B, -1)  # [B, obs_horizon * vision_feature_dim]

            # Concatenate vision + proprio + stage
            obs_flat = torch.cat([vision_flat, proprio_flat, stage], dim=-1)
        else:
            # Concatenate proprio + stage only
            obs_flat = torch.cat([proprio_flat, stage], dim=-1)

        return self.obs_encoder(obs_flat)

    def forward(self, batch: dict, training: bool = True) -> dict:
        """
        Forward pass for training or inference.

        Args:
            batch: Dictionary containing:
                - 'proprio': [B, obs_horizon, proprio_dim]
                - 'stage': [B, stage_dim] (one-hot encoding)
                - 'image': [B, obs_horizon, 3, H, W] (optional)
                - 'action': [B, pred_horizon, action_dim] (training only)
            training: Whether in training mode

        Returns:
            Dictionary with 'action' and optionally 'loss'
        """
        proprio = batch['proprio']  # [B, obs_horizon, proprio_dim]
        stage = batch['stage']  # [B, stage_dim]
        image = batch.get('image', None) if self.use_image else None
        B = proprio.shape[0]

        # Encode observations
        obs_cond = self.encode_obs(proprio, stage, image)  # [B, obs_embedding_dim]

        if training:
            # Training: predict noise
            actions = batch['action']  # [B, pred_horizon, action_dim]
            actions = actions.transpose(1, 2)  # [B, action_dim, pred_horizon]

            # Sample random timestep
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=actions.device
            ).long()

            # Add noise to actions
            noise = torch.randn_like(actions)
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

            # Predict noise
            noise_pred = self.noise_pred_net(noisy_actions, timesteps, obs_cond)

            return {
                'noise': noise,
                'noise_pred': noise_pred,
            }
        else:
            # Inference: denoise from random noise
            return self.conditional_sample(obs_cond)

    def conditional_sample(self, obs_cond: torch.Tensor) -> dict:
        """
        Sample actions by denoising from random noise.

        Args:
            obs_cond: Observation conditioning [B, obs_embedding_dim]

        Returns:
            Dictionary with sampled 'action'
        """
        B = obs_cond.shape[0]
        device = obs_cond.device

        # Initialize with random noise
        noisy_action = torch.randn(
            (B, self.action_dim, self.pred_horizon),
            device=device
        )

        # Set up inference scheduler (DDIM for faster sampling)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # Denoising loop
        for t in self.noise_scheduler.timesteps:
            # Predict noise
            timesteps = t.unsqueeze(0).repeat(B).to(device)
            noise_pred = self.noise_pred_net(noisy_action, timesteps, obs_cond)

            # Denoise
            noisy_action = self.noise_scheduler.step(
                noise_pred, t, noisy_action
            ).prev_sample

        # Transpose back: [B, action_dim, pred_horizon] -> [B, pred_horizon, action_dim]
        actions = noisy_action.transpose(1, 2)

        return {'action': actions}

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Compute training loss (MSE between predicted and actual noise).

        Args:
            batch: Batch with 'proprio' and 'action'

        Returns:
            loss: MSE loss
        """
        output = self.forward(batch, training=True)
        loss = F.mse_loss(output['noise_pred'], output['noise'])
        return loss


if __name__ == "__main__":
    # Test diffusion model
    print("Testing Diffusion Policy...")

    batch_size = 4
    obs_horizon = 2
    pred_horizon = 16
    proprio_dim = 29
    stage_dim = 5
    action_dim = 10
    image_size = 224

    # Test 1: Proprio + Stage mode (no images)
    print("\n=== Test 1: Proprio + Stage Mode ===")
    model_proprio = DiffusionPolicy(
        proprio_dim=proprio_dim,
        stage_dim=stage_dim,
        action_dim=action_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        action_horizon=8,
        obs_embedding_dim=256,
        down_dims=[256, 512, 1024],
        num_diffusion_iters=100,
        num_inference_steps=10,
        use_image=False,
    )

    # Test training forward pass
    batch = {
        'proprio': torch.randn(batch_size, obs_horizon, proprio_dim),
        'stage': torch.zeros(batch_size, stage_dim),
        'action': torch.randn(batch_size, pred_horizon, action_dim),
    }
    batch['stage'][:, 0] = 1.0  # Set to stage 0

    output = model_proprio(batch, training=True)
    print(f"Training noise shape: {output['noise'].shape}")
    print(f"Training noise_pred shape: {output['noise_pred'].shape}")

    # Test loss computation
    loss = model_proprio.compute_loss(batch)
    print(f"Loss: {loss.item():.6f}")

    # Test inference
    model_proprio.eval()
    with torch.no_grad():
        batch_inference = {
            'proprio': torch.randn(batch_size, obs_horizon, proprio_dim),
            'stage': torch.zeros(batch_size, stage_dim),
        }
        batch_inference['stage'][:, 0] = 1.0
        output_inference = model_proprio(batch_inference, training=False)
        print(f"Inference action shape: {output_inference['action'].shape}")

    # Count parameters
    num_params = sum(p.numel() for p in model_proprio.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")

    # Test 2: Vision + Proprio + Stage mode
    print("\n=== Test 2: Vision + Proprio + Stage Mode ===")
    model_vision = DiffusionPolicy(
        proprio_dim=proprio_dim,
        stage_dim=stage_dim,
        action_dim=action_dim,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        action_horizon=8,
        obs_embedding_dim=256,
        down_dims=[256, 512, 1024],
        num_diffusion_iters=100,
        num_inference_steps=10,
        use_image=True,
        vision_feature_dim=512,
    )

    # Test training forward pass with images
    batch_vision = {
        'proprio': torch.randn(batch_size, obs_horizon, proprio_dim),
        'stage': torch.zeros(batch_size, stage_dim),
        'image': torch.randn(batch_size, obs_horizon, 3, image_size, image_size),
        'action': torch.randn(batch_size, pred_horizon, action_dim),
    }
    batch_vision['stage'][:, 0] = 1.0

    output_vision = model_vision(batch_vision, training=True)
    print(f"Training noise shape: {output_vision['noise'].shape}")
    print(f"Training noise_pred shape: {output_vision['noise_pred'].shape}")

    # Test loss computation
    loss_vision = model_vision.compute_loss(batch_vision)
    print(f"Loss: {loss_vision.item():.6f}")

    # Test inference
    model_vision.eval()
    with torch.no_grad():
        batch_vision_inference = {
            'proprio': torch.randn(batch_size, obs_horizon, proprio_dim),
            'stage': torch.zeros(batch_size, stage_dim),
            'image': torch.randn(batch_size, obs_horizon, 3, image_size, image_size),
        }
        batch_vision_inference['stage'][:, 0] = 1.0
        output_vision_inference = model_vision(batch_vision_inference, training=False)
        print(f"Inference action shape: {output_vision_inference['action'].shape}")

    # Count parameters
    num_params_vision = sum(p.numel() for p in model_vision.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params_vision:,}")

    print("\n✓ Diffusion Policy tests passed!")
