import torch
import numpy as np
from collections import deque
from typing import Dict, Optional
from data.dataset import get_stage_encoding

class ACTController:
    def __init__(
        self,
        model,
        normalizer,
        obs_horizon: int = 1,
        pred_horizon: int = 16,
        exec_horizon: int = 8,
        device: str = "cuda:0",
        temporal_ensemble: bool = False,
        dt: float = 0.02,  # Timestep duration (50 fps)
        use_image: bool = False,
    ):
 
        self.model = model.to(device)
        self.model.eval()
        self.normalizer = normalizer
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.exec_horizon = exec_horizon
        self.device = torch.device(device)
        self.temporal_ensemble = temporal_ensemble
        self.dt = dt
        self.use_image = use_image


        self.obs_pos_history = deque(maxlen=obs_horizon) 
        self.obs_vel_history = deque(maxlen=obs_horizon)  

        if use_image:
            self.image_history = deque(maxlen=obs_horizon)
        else:
            self.image_history = None

       
        self.action_buffer = []
        self.action_idx = 0
        self.frame_idx = 0

        self._warned_missing_image = False

        print(f"[ACTController] Initialized with:")
        print(f"  obs_horizon: {obs_horizon}")
        print(f"  pred_horizon: {pred_horizon}")
        print(f"  exec_horizon: {exec_horizon}")
        print(f"  device: {device}")
        print(f"  temporal_ensemble: {temporal_ensemble}")
        print(f"  dt: {dt}")
        print(f"  use_image: {use_image}")

    def reset(self):
        self.obs_pos_history.clear()
        self.obs_vel_history.clear()
        if self.image_history is not None:
            self.image_history.clear()
        self.action_buffer = []
        self.action_idx = 0
        self.frame_idx = 0

    def add_observation(
        self,
        proprio_pos: torch.Tensor,
        proprio_vel: torch.Tensor,
        image: Optional[torch.Tensor] = None
    ):
        if proprio_pos.ndim == 2:
            proprio_pos = proprio_pos[0]
        if proprio_vel.ndim == 2:
            proprio_vel = proprio_vel[0]

        proprio_pos = proprio_pos.detach().cpu().float()
        proprio_vel = proprio_vel.detach().cpu().float()

        self.obs_pos_history.append(proprio_pos)
        self.obs_vel_history.append(proprio_vel)

        while len(self.obs_pos_history) < self.obs_horizon:
            self.obs_pos_history.append(proprio_pos.clone())
            self.obs_vel_history.append(proprio_vel.clone())

        if self.use_image:
            if image is not None:
                if image.ndim == 4:
                    image = image[0]
                image = image.detach().cpu().float()
                self.image_history.append(image)

                while len(self.image_history) < self.obs_horizon:
                    self.image_history.append(image.clone())
            else:
                if not self._warned_missing_image:
                    print("[WARNING] ACT controller expects images but none provided. Using zero placeholder.")
                    self._warned_missing_image = True

                placeholder = torch.zeros(3, 224, 224, dtype=torch.float32)
                self.image_history.append(placeholder)
                while len(self.image_history) < self.obs_horizon:
                    self.image_history.append(placeholder.clone())

    def get_action(
        self,
        proprio_pos: Optional[torch.Tensor] = None,
        proprio_vel: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        stage: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if proprio_pos is not None and proprio_vel is not None:
            self.add_observation(proprio_pos, proprio_vel, image)

        if len(self.action_buffer) == 0 or self.action_idx >= self.exec_horizon:
            self._replan(external_stage=stage)

        action = self.action_buffer[self.action_idx]
        self.action_idx += 1

        self.frame_idx += 1

        return action

    def _replan(self, external_stage: Optional[torch.Tensor] = None):
        if len(self.obs_pos_history) < self.obs_horizon:
            raise RuntimeError(
                f"Not enough observations in history: {len(self.obs_pos_history)} < {self.obs_horizon}"
            )

        pos_stack = torch.stack(list(self.obs_pos_history), dim=0)  
        vel_stack = torch.stack(list(self.obs_vel_history), dim=0) 

        proprio_stack = torch.cat([pos_stack, vel_stack], dim=-1)
        proprio_batch = proprio_stack.unsqueeze(0).to(self.device)

        if external_stage is not None:
            stage_batch = external_stage.unsqueeze(0).to(self.device)
        else:
            stage_encoding = get_stage_encoding(self.frame_idx)
            stage_encoding = torch.from_numpy(stage_encoding).float()
            stage_batch = stage_encoding.unsqueeze(0).to(self.device)  

        normalized_obs = self.normalizer.normalize({'proprio': proprio_batch})['proprio']

        image_batch = None
        if self.use_image and self.image_history is not None and len(self.image_history) > 0:
            image_stack = torch.stack(list(self.image_history), dim=0) 
            image_batch = image_stack.unsqueeze(0).to(self.device) 

        with torch.no_grad():
            batch = {
                'proprio': normalized_obs,
                'stage': stage_batch,
            }
            if image_batch is not None:
                batch['image'] = image_batch

            output = self.model(batch, inference=True)
            normalized_actions = output['action']

        actions = self.normalizer.denormalize(normalized_actions[0], 'action') 

        self.action_buffer = actions.cpu()
        self.action_idx = 0


    def get_statistics(self) -> Dict[str, float]:
        stats = {
            'obs_pos_history_len': len(self.obs_pos_history),
            'obs_vel_history_len': len(self.obs_vel_history),
            'action_buffer_len': len(self.action_buffer),
            'action_idx': self.action_idx,
            'frame_idx': self.frame_idx,
        }

        if len(self.obs_pos_history) > 0:
            last_pos = self.obs_pos_history[-1]
            last_vel = self.obs_vel_history[-1]
            stats.update({
                'pos_mean': last_pos.mean().item(),
                'pos_std': last_pos.std().item(),
                'vel_mean': last_vel.mean().item(),
                'vel_std': last_vel.std().item(),
            })

        if len(self.action_buffer) > 0:
            last_action = self.action_buffer[self.action_idx - 1] if self.action_idx > 0 else self.action_buffer[0]
            stats.update({
                'action_mean': last_action.mean().item(),
                'action_std': last_action.std().item(),
                'action_min': last_action.min().item(),
                'action_max': last_action.max().item(),
            })

        stage_encoding = get_stage_encoding(self.frame_idx)
        stage_encoding = torch.from_numpy(stage_encoding)
        stage_idx = torch.argmax(stage_encoding).item()
        stats['current_stage'] = stage_idx

        return stats


def load_act_policy_and_normalizer(checkpoint_path: str, device: str = "cuda:0"):
    import sys
    import os
    from pathlib import Path

    project_root = Path(checkpoint_path).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from act_policy.models.act_model import ACTPolicy
    from data.dataset import Normalizer

    print(f"[INFO] Loading ACT policy from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    print(f"[INFO] Checkpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")

    model = ACTPolicy(
        proprio_dim=config['proprio_dim'],
        stage_dim=config['stage_dim'],
        action_dim=config['action_dim'],
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        hidden_dim=config['hidden_dim'],
        nheads=config['nheads'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        latent_dim=config['latent_dim'],
        dropout=config['dropout'],
        use_vae=config['use_vae'],
        use_image=config.get('use_image', False),
        vision_feature_dim=config.get('vision_feature_dim', 512),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"[INFO] Model loaded successfully")

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

