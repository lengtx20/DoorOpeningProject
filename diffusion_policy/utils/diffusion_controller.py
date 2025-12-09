import torch
import numpy as np
from collections import deque
from typing import Dict, Optional
from data.dataset import get_stage_encoding


class DiffusionController:
    """
    Diffusion Policy Controller for inference.

    Manages observation history, stage, and action chunk execution.
    """

    def __init__(
        self,
        model,
        normalizer,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        device: str = "cuda:0",
        dt: float = 0.02,
        use_image: bool = False,
    ):
        """
        Args:
            model: Diffusion policy model
            normalizer: Data normalizer
            obs_horizon: Number of observation steps to keep
            pred_horizon: Number of actions predicted per query
            action_horizon: Number of actions to execute before replanning
            device: Device to run inference on
            dt: Timestep duration for timestamp tracking
            use_image: Whether to use image input
        """
        self.model = model.to(device)
        self.model.eval()
        self.normalizer = normalizer
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.device = torch.device(device)
        self.dt = dt
        self.use_image = use_image

        # Observation history buffer
        self.obs_history = deque(maxlen=obs_horizon)

        # Image history buffer (if using images)
        if use_image:
            self.image_history = deque(maxlen=obs_horizon)
        else:
            self.image_history = None

        # Stage tracking
        self.current_stage = None
        self.timestamp = 0.0

        # Action chunk buffer
        self.action_buffer = []
        self.action_idx = 0

        print(f"[DiffusionController] Initialized with:")
        print(f"  obs_horizon: {obs_horizon}")
        print(f"  pred_horizon: {pred_horizon}")
        print(f"  action_horizon: {action_horizon}")
        print(f"  device: {device}")
        print(f"  dt: {dt}s")
        print(f"  use_image: {use_image}")

    def reset(self):
        """Reset the controller state."""
        self.obs_history.clear()
        if self.image_history is not None:
            self.image_history.clear()
        self.action_buffer = []
        self.action_idx = 0
        self.current_stage = None
        self.timestamp = 0.0

    def _get_stage_encoding(self, timestamp: float) -> np.ndarray:
        frame_idx = int(timestamp * 50)
        return get_stage_encoding(frame_idx)

    def add_observation(
        self,
        proprio: torch.Tensor,
        image: Optional[torch.Tensor] = None
    ):
        """
        Add a new observation to the history.

        Args:
            proprio: Proprioceptive observation [proprio_dim] or [1, proprio_dim]
            image: RGB image [3, H, W] or [1, 3, H, W] (optional)
        """
        if proprio.ndim == 2:
            proprio = proprio[0]

        proprio = proprio.detach().cpu().float()
        self.obs_history.append(proprio)

        # Pad history if not full yet
        while len(self.obs_history) < self.obs_horizon:
            self.obs_history.append(proprio.clone())

        # Add image to history if using images
        if self.use_image and image is not None:
            if image.ndim == 4:
                image = image[0]
            image = image.detach().cpu().float()
            self.image_history.append(image)

            # Pad image history if not full yet
            while len(self.image_history) < self.obs_horizon:
                self.image_history.append(image.clone())

        # Update timestamp and stage
        self.timestamp += self.dt
        self.current_stage = self._get_stage_encoding(self.timestamp)

    def get_action(
        self,
        proprio: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        stage: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get the next action to execute.

        Args:
            proprio: Current proprioceptive observation
            image: Current RGB image (optional)
            stage: External stage one-hot encoding [5] (optional, overrides computed stage)

        Returns:
            action: Action to execute [action_dim]
        """
        if proprio is not None:
            self.add_observation(proprio, image)

        # Replan if buffer is empty or we've executed all actions in the chunk
        if len(self.action_buffer) == 0 or self.action_idx >= self.action_horizon:
            self._replan(external_stage=stage)

        action = self.action_buffer[self.action_idx]
        self.action_idx += 1

        return action

    def _replan(self, external_stage: Optional[torch.Tensor] = None):
        """
        Query the diffusion model for a new action chunk.

        Args:
            external_stage: Optional external stage one-hot encoding [5]. If provided, uses this instead of computing from timestamp.
        """
        if len(self.obs_history) < self.obs_horizon:
            raise RuntimeError(
                f"Not enough observations in history: {len(self.obs_history)} < {self.obs_horizon}"
            )

        # Stack observations
        obs_stack = torch.stack(list(self.obs_history), dim=0)  # [obs_horizon, proprio_dim]
        obs_batch = obs_stack.unsqueeze(0).to(self.device)  # [1, obs_horizon, proprio_dim]

        # Use external stage if provided, otherwise compute from timestamp
        if external_stage is not None:
            stage_batch = external_stage.unsqueeze(0).to(self.device)  # [1, 5]
        else:
            if self.current_stage is None:
                self.current_stage = self._get_stage_encoding(self.timestamp)
            stage_batch = torch.from_numpy(self.current_stage).unsqueeze(0).to(self.device)  # [1, 5]

        # Prepare batch for normalization
        batch_to_normalize = {'proprio': obs_batch}

        # Stack images if using vision
        image_batch = None
        if self.use_image and self.image_history is not None and len(self.image_history) > 0:
            image_stack = torch.stack(list(self.image_history), dim=0)  # [obs_horizon, 3, H, W]
            image_batch = image_stack.unsqueeze(0).to(self.device)  # [1, obs_horizon, 3, H, W]

        # Normalize
        normalized_batch = self.normalizer.normalize(batch_to_normalize)
        normalized_obs = normalized_batch['proprio']

        # Query model
        with torch.no_grad():
            batch = {
                'proprio': normalized_obs,
                'stage': stage_batch,
            }
            if image_batch is not None:
                batch['image'] = image_batch

            output = self.model(batch, training=False)
            normalized_actions = output['action']  # [1, pred_horizon, action_dim]

        # Denormalize
        actions = self.normalizer.denormalize(normalized_actions[0], 'action')  # [pred_horizon, action_dim]

        # Store action chunk
        self.action_buffer = actions.cpu()
        self.action_idx = 0

    def get_statistics(self) -> Dict[str, float]:
        """
        Get controller statistics for debugging.

        Returns:
            stats: Dictionary of statistics
        """
        stats = {
            'obs_history_len': len(self.obs_history),
            'action_buffer_len': len(self.action_buffer),
            'action_idx': self.action_idx,
        }

        if len(self.obs_history) > 0:
            last_obs = self.obs_history[-1]
            stats.update({
                'obs_mean': last_obs.mean().item(),
                'obs_std': last_obs.std().item(),
                'obs_min': last_obs.min().item(),
                'obs_max': last_obs.max().item(),
            })

        if len(self.action_buffer) > 0:
            last_action = self.action_buffer[self.action_idx - 1] if self.action_idx > 0 else self.action_buffer[0]
            stats.update({
                'action_mean': last_action.mean().item(),
                'action_std': last_action.std().item(),
                'action_min': last_action.min().item(),
                'action_max': last_action.max().item(),
            })

        return stats


def load_diffusion_policy_and_normalizer(checkpoint_path: str, device: str = "cuda:0"):
    """
    Load diffusion policy and normalizer from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded diffusion model
        normalizer: Loaded normalizer
        config: Training configuration
    """
    import sys
    import os
    from pathlib import Path

    # Add project root to path
    project_root = Path(checkpoint_path).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from diffusion_policy.models.diffusion_model import DiffusionPolicy
    from data.dataset import Normalizer

    print(f"[INFO] Loading Diffusion policy from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    print(f"[INFO] Checkpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")

    # Create model
    model = DiffusionPolicy(
        proprio_dim=config['proprio_dim'],
        stage_dim=config.get('stage_dim', 5),
        action_dim=config['action_dim'],
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        action_horizon=config['action_horizon'],
        obs_embedding_dim=config.get('obs_embedding_dim', 256),
        diffusion_step_embed_dim=config.get('diffusion_step_embed_dim', 128),
        down_dims=config.get('down_dims', [256, 512, 1024]),
        kernel_size=config.get('kernel_size', 5),
        n_groups=config.get('n_groups', 8),
        num_diffusion_iters=config.get('num_diffusion_iters', 100),
        num_inference_steps=config.get('num_inference_steps', 10),
        beta_schedule=config.get('beta_schedule', 'squaredcos_cap_v2'),
        use_image=config.get('use_image', False),
        vision_feature_dim=config.get('vision_feature_dim', 512),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"[INFO] Model loaded successfully")

    # Load normalizer
    checkpoint_dir = Path(checkpoint_path).parent
    normalizer_path = checkpoint_dir / 'normalizer.pth'

    if normalizer_path.exists():
        normalizer = Normalizer.load(str(normalizer_path))
        print(f"[INFO] Normalizer loaded from: {normalizer_path}")
    elif 'stats' in checkpoint:
        print(f"[WARNING] Normalizer file not found, using stats from checkpoint")
        normalizer = Normalizer(checkpoint['stats'])
    else:
        raise ValueError(f"No normalizer found at {normalizer_path} or in checkpoint")

    return model, normalizer, config


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/jason/DoorOpeningProject')

    from diffusion_policy.models import DiffusionPolicy
    from data.dataset import Normalizer

    print("Testing Diffusion Controller...")

    # Create dummy model and normalizer
    model = DiffusionPolicy(
        proprio_dim=29,
        action_dim=10,
        obs_horizon=2,
        pred_horizon=16,
        action_horizon=8,
    )

    stats = {
        'proprio': {
            'min': torch.zeros(29),
            'max': torch.ones(29),
            'mean': torch.zeros(29),
            'std': torch.ones(29),
        },
        'action': {
            'min': torch.zeros(10),
            'max': torch.ones(10),
            'mean': torch.zeros(10),
            'std': torch.ones(10),
        }
    }
    normalizer = Normalizer(stats)

    controller = DiffusionController(
        model=model,
        normalizer=normalizer,
        obs_horizon=2,
        pred_horizon=16,
        action_horizon=8,
        device='cpu',
    )

    print("\nSimulating episode...")
    for t in range(20):
        proprio = torch.randn(29)
        action = controller.get_action(proprio)
        print(f"Step {t}: action shape {action.shape}")

        if t == 0 or t == 8 or t == 16:
            stats = controller.get_statistics()
            print(f"  Statistics: {stats}")

    print("\n✓ Diffusion Controller test passed!")
