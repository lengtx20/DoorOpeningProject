
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
class BCDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        mode: str = 'train',
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        sample_stride: int = 1,
        noise_std: float = 0.0,
    ):
        self.dataset_root = dataset_root
        self.mode = mode
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.sample_stride = sample_stride
        self.noise_std = noise_std
        split_path = os.path.join(dataset_root, 'split.json')
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        episode_ids = split_data[mode]
        print(f"Loading {mode} dataset with {len(episode_ids)} episodes")
        self.episodes = []
        for ep_id in episode_ids:
            ep_path = os.path.join(dataset_root, ep_id, 'log_dict.npy')
            if not os.path.exists(ep_path):
                print(f"Warning: Episode {ep_id} not found, skipping...")
                continue
            log_dict = np.load(ep_path, allow_pickle=True).item()
            self.episodes.append(log_dict)
        print(f"Loaded {len(self.episodes)} episodes for {mode}")
        self.indices = []
        for ep_idx, log_dict in enumerate(self.episodes):
            ep_len = len(log_dict['q_0'])  
            valid_indices = list(range(0, ep_len, sample_stride))
            for i in range(len(valid_indices) - obs_horizon - pred_horizon + 1):
                self.indices.append((ep_idx, valid_indices[i + obs_horizon - 1]))
        print(f"Total samples: {len(self.indices)}")
    def __len__(self) -> int:
        return len(self.indices)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, end_obs_idx = self.indices[idx]
        log_dict = self.episodes[ep_idx]
        proprio_keys = [f'q_{i}' for i in range(29)]
        proprio_list = []
        for t in range(end_obs_idx - self.obs_horizon + 1, end_obs_idx + 1):
            proprio_t = np.array([log_dict[key][t] for key in proprio_keys], dtype=np.float32)
            proprio_list.append(proprio_t)
        proprio = np.stack(proprio_list, axis=0)  
        if self.mode == 'train' and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, proprio.shape).astype(np.float32)
            proprio = proprio + noise
        action_keys = [
            'left_wrist_torso_pos_x', 'left_wrist_torso_pos_y', 'left_wrist_torso_pos_z',
            'right_wrist_torso_pos_x', 'right_wrist_torso_pos_y', 'right_wrist_torso_pos_z',
            'vel_body_x', 'vel_body_y',
            'yaw_speed',
            'p_pressed'
        ]
        action_list = []
        for t in range(end_obs_idx + 1, end_obs_idx + self.pred_horizon + 1):
            action_t = np.array([log_dict[key][t] for key in action_keys], dtype=np.float32)
            action_list.append(action_t)
        action = np.stack(action_list, axis=0)  
        return {
            'proprio': torch.from_numpy(proprio),  
            'action': torch.from_numpy(action),    
        }
def get_data_stats(dataloader: torch.utils.data.DataLoader) -> Dict[str, Dict[str, torch.Tensor]]:
    print("Computing normalization statistics...")
    proprio_data = []
    action_data = []
    for batch in dataloader:
        proprio_data.append(batch['proprio'].reshape(-1, 29))  
        action_data.append(batch['action'].reshape(-1, 10))    
    proprio_data = torch.cat(proprio_data, dim=0)  
    action_data = torch.cat(action_data, dim=0)    
    stats = {
        'proprio': {
            'min': proprio_data.min(dim=0)[0],
            'max': proprio_data.max(dim=0)[0],
            'mean': proprio_data.mean(dim=0),
            'std': proprio_data.std(dim=0),
        },
        'action': {
            'min': action_data.min(dim=0)[0],
            'max': action_data.max(dim=0)[0],
            'mean': action_data.mean(dim=0),
            'std': action_data.std(dim=0),
        }
    }
    print("Statistics computed successfully")
    print(f"  Proprio range: [{stats['proprio']['min'].min():.3f}, {stats['proprio']['max'].max():.3f}]")
    print(f"  Action range: [{stats['action']['min'].min():.3f}, {stats['action']['max'].max():.3f}]")
    return stats
class Normalizer:
    def __init__(self, stats: Dict[str, Dict[str, torch.Tensor]]):
        self.stats = stats
    def normalize(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized = {}
        for key in ['proprio', 'action']:
            if key in batch:
                data = batch[key]
                data_min = self.stats[key]['min'].to(data.device)
                data_max = self.stats[key]['max'].to(data.device)
                normalized[key] = 2 * (data - data_min) / (data_max - data_min + 1e-8) - 1
        return normalized
    def denormalize(self, data: torch.Tensor, key: str) -> torch.Tensor:
        data_min = self.stats[key]['min'].to(data.device)
        data_max = self.stats[key]['max'].to(data.device)
        return (data + 1) / 2 * (data_max - data_min + 1e-8) + data_min
    def save(self, path: str):
        torch.save(self.stats, path)
        print(f"Normalizer saved to {path}")
    @classmethod
    def load(cls, path: str):
        stats = torch.load(path)
        return cls(stats)
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = BCDataset(
        dataset_root="/home/jason/DoorOpeningProject/diffusion_policy/data/torso_rgb_logs",
        mode='train',
        obs_horizon=2,
        pred_horizon=16,
        sample_stride=1,
    )
    print(f"\nDataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  proprio shape: {sample['proprio'].shape}")
    print(f"  action shape: {sample['action'].shape}")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  proprio shape: {batch['proprio'].shape}")
    print(f"  action shape: {batch['action'].shape}")
    stats = get_data_stats(dataloader)
    normalizer = Normalizer(stats)
    normalized_batch = normalizer.normalize(batch)
    print(f"\nNormalized batch:")
    print(f"  proprio range: [{normalized_batch['proprio'].min():.3f}, {normalized_batch['proprio'].max():.3f}]")
    print(f"  action range: [{normalized_batch['action'].min():.3f}, {normalized_batch['action'].max():.3f}]")
