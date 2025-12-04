
import torch
import numpy as np
from collections import deque
from typing import Dict, Optional
class BCController:
    def __init__(
        self,
        model,
        normalizer,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        exec_horizon: int = 8,
        device: str = "cuda:0",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.normalizer = normalizer
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.exec_horizon = exec_horizon
        self.device = torch.device(device)
        self.obs_history = deque(maxlen=obs_horizon)
        self.action_buffer = []
        self.action_idx = 0
        print(f"[BCController] Initialized with:")
        print(f"  obs_horizon: {obs_horizon}")
        print(f"  pred_horizon: {pred_horizon}")
        print(f"  exec_horizon: {exec_horizon}")
        print(f"  device: {device}")
    def reset(self):
        self.obs_history.clear()
        self.action_buffer = []
        self.action_idx = 0
    def add_observation(self, proprio: torch.Tensor):
        if proprio.ndim == 2:
            proprio = proprio[0]  
        proprio = proprio.detach().cpu().float()
        self.obs_history.append(proprio)
        while len(self.obs_history) < self.obs_horizon:
            self.obs_history.append(proprio.clone())
    def get_action(self, proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        if proprio is not None:
            self.add_observation(proprio)
        if len(self.action_buffer) == 0 or self.action_idx >= self.exec_horizon:
            self._replan()
        action = self.action_buffer[self.action_idx]
        self.action_idx += 1
        return action
    def _replan(self):
        if len(self.obs_history) < self.obs_horizon:
            raise RuntimeError(
                f"Not enough observations in history: {len(self.obs_history)} < {self.obs_horizon}"
            )
        obs_stack = torch.stack(list(self.obs_history), dim=0)
        obs_batch = obs_stack.unsqueeze(0).to(self.device)
        normalized_obs = self.normalizer.normalize({'proprio': obs_batch})['proprio']
        with torch.no_grad():
            normalized_actions = self.model(normalized_obs)
        actions = self.normalizer.denormalize(normalized_actions, 'action')
        self.action_buffer = actions[0].cpu()  
        self.action_idx = 0
    def get_statistics(self) -> Dict[str, float]:
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
def load_bc_policy_and_normalizer(checkpoint_path: str, device: str = "cuda:0"):
    import sys
    import os
    from pathlib import Path
    bc_root = Path(checkpoint_path).parent.parent
    if str(bc_root) not in sys.path:
        sys.path.insert(0, str(bc_root))
    from bc_policy.models import BCPolicy
    from bc_policy.utils import Normalizer
    print(f"[INFO] Loading BC policy from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    print(f"[INFO] Checkpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    model = BCPolicy(
        proprio_dim=config['proprio_dim'],
        action_dim=config['action_dim'],
        obs_horizon=config['obs_horizon'],
        pred_horizon=config['pred_horizon'],
        hidden_dims=config['hidden_dims'],
        activation=config['activation'],
        use_layer_norm=config['use_layer_norm'],
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
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/jason/DoorOpeningProject')
    from bc_policy.models import BCPolicy
    from bc_policy.utils import Normalizer
    print("Testing BC Controller...")
    model = BCPolicy(
        proprio_dim=29,
        action_dim=10,
        obs_horizon=2,
        pred_horizon=16,
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
    controller = BCController(
        model=model,
        normalizer=normalizer,
        obs_horizon=2,
        pred_horizon=16,
        exec_horizon=8,
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
    print("\n✓ BC Controller test passed!")
