
import torch
import torch.nn as nn
from typing import Dict, List
class BCPolicy(nn.Module):
    def __init__(
        self,
        proprio_dim: int = 29,
        action_dim: int = 10,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        hidden_dims: List[int] = [512, 512, 512],
        activation: str = 'relu',
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        input_dim = obs_horizon * proprio_dim
        output_dim = pred_horizon * action_dim
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        batch_size = proprio.shape[0]
        proprio_flat = proprio.reshape(batch_size, -1)
        action_flat = self.mlp(proprio_flat)
        action = action_flat.reshape(batch_size, self.pred_horizon, self.action_dim)
        return action
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        proprio = batch['proprio']  
        action_gt = batch['action']  
        action_pred = self.forward(proprio)  
        loss = nn.functional.mse_loss(action_pred, action_gt)
        return loss
    def predict(self, proprio: torch.Tensor) -> torch.Tensor:
        is_single = (proprio.ndim == 2)
        if is_single:
            proprio = proprio.unsqueeze(0)  
        with torch.no_grad():
            action = self.forward(proprio)
        if is_single:
            action = action.squeeze(0)  
        return action
class BCPolicyWithHistory(nn.Module):
    def __init__(
        self,
        proprio_dim: int = 29,
        action_dim: int = 10,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        hidden_dims: List[int] = [512, 512, 512],
        activation: str = 'relu',
        use_layer_norm: bool = True,
        temporal_mode: str = 'conv',  
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.temporal_mode = temporal_mode
        if temporal_mode == 'conv':
            self.temporal = nn.Sequential(
                nn.Conv1d(proprio_dim, 128, kernel_size=obs_horizon, stride=1),
                nn.ReLU(),
            )
            temporal_output_dim = 128
        elif temporal_mode == 'gru':
            self.temporal = nn.GRU(proprio_dim, 128, num_layers=1, batch_first=True)
            temporal_output_dim = 128
        else:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode}")
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        layers = []
        prev_dim = temporal_output_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        output_dim = pred_horizon * action_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        batch_size = proprio.shape[0]
        if self.temporal_mode == 'conv':
            proprio_t = proprio.transpose(1, 2)  
            temporal_feat = self.temporal(proprio_t)  
            temporal_feat = temporal_feat.squeeze(-1)  
        elif self.temporal_mode == 'gru':
            _, h_n = self.temporal(proprio)  
            temporal_feat = h_n.squeeze(0)  
        action_flat = self.mlp(temporal_feat)  
        action = action_flat.reshape(batch_size, self.pred_horizon, self.action_dim)
        return action
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        proprio = batch['proprio']
        action_gt = batch['action']
        action_pred = self.forward(proprio)
        loss = nn.functional.mse_loss(action_pred, action_gt)
        return loss
    def predict(self, proprio: torch.Tensor) -> torch.Tensor:
        is_single = (proprio.ndim == 2)
        if is_single:
            proprio = proprio.unsqueeze(0)
        with torch.no_grad():
            action = self.forward(proprio)
        if is_single:
            action = action.squeeze(0)
        return action
if __name__ == "__main__":
    print("Testing BCPolicy...")
    policy = BCPolicy(
        proprio_dim=29,
        action_dim=10,
        obs_horizon=2,
        pred_horizon=16,
        hidden_dims=[512, 512, 512],
    )
    batch_proprio = torch.randn(32, 2, 29)
    batch_action = torch.randn(32, 16, 10)
    output = policy(batch_proprio)
    print(f"Output shape: {output.shape}")  
    batch = {'proprio': batch_proprio, 'action': batch_action}
    loss = policy.compute_loss(batch)
    print(f"Loss: {loss.item():.4f}")
    single_proprio = torch.randn(2, 29)
    single_action = policy.predict(single_proprio)
    print(f"Single prediction shape: {single_action.shape}")  
    print("\nTesting BCPolicyWithHistory (conv)...")
    policy_conv = BCPolicyWithHistory(
        proprio_dim=29,
        action_dim=10,
        obs_horizon=2,
        pred_horizon=16,
        temporal_mode='conv',
    )
    output_conv = policy_conv(batch_proprio)
    print(f"Output shape: {output_conv.shape}")
    print("\nTesting BCPolicyWithHistory (gru)...")
    policy_gru = BCPolicyWithHistory(
        proprio_dim=29,
        action_dim=10,
        obs_horizon=2,
        pred_horizon=16,
        temporal_mode='gru',
    )
    output_gru = policy_gru(batch_proprio)
    print(f"Output shape: {output_gru.shape}")
    print("\nAll tests passed!")
