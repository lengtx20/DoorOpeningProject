import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from torchvision import transforms


def smooth_array(arr: np.ndarray, window_size: int = 5) -> np.ndarray:

    smoothed = np.copy(arr)
    half_window = window_size // 2

    for i in range(len(arr)):
        start_idx = i - half_window
        end_idx = i + half_window + 1

        # If we have sufficient data on both sides, apply smoothing
        if start_idx >= 0 and end_idx <= len(arr):
            smoothed[i] = np.mean(arr[start_idx:end_idx])
        # Otherwise keep the raw value 

    return smoothed


def compute_stage_from_frame_idx(frame_idx: int) -> int:
    if frame_idx < 75:
        return 0
    elif frame_idx < 400:
        return 1
    elif frame_idx < 450:
        return 2
    elif frame_idx < 750:
        return 3
    else:
        return 4


def get_stage_encoding(frame_idx: int) -> np.ndarray:
    one_hot = np.zeros(5, dtype=np.float32)

    if frame_idx < 75:
        one_hot[0] = 1.0
    elif frame_idx < 400:
        one_hot[1] = 1.0
    elif frame_idx < 450:
        one_hot[2] = 1.0
    elif frame_idx < 750:
        one_hot[3] = 1.0
    else:
        one_hot[4] = 1.0

    return one_hot


class G1Dataset(Dataset):
    def __init__(self,
                 dataset_root,
                 mode='train',
                 obs_horizon=2,
                 pred_horizon=16,
                 use_proprio=True,
                 use_image=True,
                 image_resize_size=None,
                 sample_stride=5,
                 noise_std=0.0,
                 temporal_dropout=0.0,
                 action_noise_std=0.0,
                 smooth_window=0):

        self.root = dataset_root
        self.mode = mode
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.use_proprio = use_proprio
        self.use_image = use_image
        self.sample_stride = sample_stride
        self.noise_std = noise_std
        self.temporal_dropout = temporal_dropout
        self.action_noise_std = action_noise_std
        self.smooth_window = smooth_window

        if self.use_image:
            if image_resize_size is not None:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(image_resize_size, antialias=True),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = None

        split_path = os.path.join(self.root, "split.json")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found at {split_path}")
            
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        self.episode_ids = split_data[mode]
        
        self.data_cache = {} 
        self.indices = [] 

        print(f"Loading {mode} data ({len(self.episode_ids)} episodes)...")

        for ep_id in self.episode_ids:
            ep_path = os.path.join(self.root, ep_id)
            log_path = os.path.join(ep_path, "log_dict.npy")
            
            raw_dict = np.load(log_path, allow_pickle=True).item()

            action_vec = np.concatenate([
                raw_dict['left_wrist_pos_x'].reshape(-1, 1),
                raw_dict['left_wrist_pos_y'].reshape(-1, 1),
                raw_dict['left_wrist_pos_z'].reshape(-1, 1),
                raw_dict['right_wrist_pos_x'].reshape(-1, 1),
                raw_dict['right_wrist_pos_y'].reshape(-1, 1),
                raw_dict['right_wrist_pos_z'].reshape(-1, 1),
                raw_dict['vel_x'].reshape(-1, 1),
                raw_dict['vel_y'].reshape(-1, 1),
                raw_dict['yaw_speed'].reshape(-1, 1),
            ], axis=1).astype(np.float32)  

     
            if self.smooth_window > 0:
                action_smoothed = np.zeros_like(action_vec)
                for dim in range(action_vec.shape[1]):  
                    action_smoothed[:, dim] = smooth_array(action_vec[:, dim], self.smooth_window)
                action_vec = action_smoothed

            if self.use_proprio:
                q_cols = [f'q_{i}' for i in range(29)]
                proprio_pos_vec = np.stack([raw_dict[c] for c in q_cols], axis=-1).astype(np.float32)

                dq_cols = [f'dq_{i}' for i in range(29)]
                proprio_vel_vec = np.stack([raw_dict[c] for c in dq_cols], axis=-1).astype(np.float32)
            else:
                proprio_pos_vec = None
                proprio_vel_vec = None

            world_pose_vec = np.concatenate([
                raw_dict['pos_x'].reshape(-1, 1),
                raw_dict['pos_y'].reshape(-1, 1),
                raw_dict['pos_z'].reshape(-1, 1),
                raw_dict['quat_x'].reshape(-1, 1),
                raw_dict['quat_y'].reshape(-1, 1),
                raw_dict['quat_z'].reshape(-1, 1),
                raw_dict['quat_w'].reshape(-1, 1),
            ], axis=1).astype(np.float32)

 
            timestamp_vec = raw_dict['timestamp'].astype(np.float32)
            timestamp_vec = timestamp_vec - timestamp_vec[0]  

            self.data_cache[ep_id] = {
                'action_traj': action_vec,
                'proprio_pos_traj': proprio_pos_vec,
                'proprio_vel_traj': proprio_vel_vec,
                'pose_traj': world_pose_vec,
                'timestamp_traj': timestamp_vec,
                'length': len(action_vec)
            }

            L = len(action_vec)
            max_start = L - (self.obs_horizon + self.pred_horizon) + 1
            
            for t in range(0, max_start, self.sample_stride):
                self.indices.append((ep_id, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_id, start_t = self.indices[idx]
        data = self.data_cache[ep_id]

        if self.use_image:
            images_list = []
            for i in range(self.obs_horizon):
                frame_idx = start_t + i
                fname = f"{frame_idx:06d}.npy"
                fpath = os.path.join(self.root, ep_id, fname)

                img_arr = np.load(fpath)
                img_t = self.transform(img_arr)
                images_list.append(img_t)

            image_tensor = torch.stack(images_list)
        else:
            image_tensor = torch.empty(0)  

        act_start = start_t + (self.obs_horizon - 1)
        act_end = act_start + self.pred_horizon
        action = data['action_traj'][act_start : act_end].copy()

        if self.mode == 'train' and self.action_noise_std > 0:
            action_noise = np.random.normal(0, self.action_noise_std, action.shape).astype(np.float32)
            action = action + action_noise

        if self.use_proprio:
            agent_pos = data['proprio_pos_traj'][start_t : start_t + self.obs_horizon].copy()
            agent_vel = data['proprio_vel_traj'][start_t : start_t + self.obs_horizon].copy()

            if self.mode == 'train':
                if self.noise_std > 0:
                    pos_noise = np.random.normal(0, self.noise_std, agent_pos.shape).astype(np.float32)
                    agent_pos = agent_pos + pos_noise
                    vel_noise = np.random.normal(0, self.noise_std * 0.5, agent_vel.shape).astype(np.float32)
                    agent_vel = agent_vel + vel_noise

                if self.temporal_dropout > 0 and self.obs_horizon > 1:
                    for t in range(1, self.obs_horizon):
                        if np.random.rand() < self.temporal_dropout:
                            agent_pos[t] = agent_pos[t-1]
                            agent_vel[t] = agent_vel[t-1]

            agent_proprio = np.concatenate([agent_pos, agent_vel], axis=-1)
            agent_proprio_tensor = torch.from_numpy(agent_proprio)
        else:
            agent_proprio_tensor = torch.empty(0)

        current_pose_idx = start_t + self.obs_horizon - 1
        start_pose = data['pose_traj'][current_pose_idx]

        obs_timestamps = data['timestamp_traj'][start_t : start_t + self.obs_horizon]
        action_timestamps = data['timestamp_traj'][act_start : act_end]


        stage_one_hot = get_stage_encoding(start_t)

        return {
            'image': image_tensor,
            'action': torch.from_numpy(action),
            'proprio': agent_proprio_tensor, 
            'stage': torch.from_numpy(stage_one_hot),  
            'start_pose': torch.from_numpy(start_pose),
            'frame_idx': torch.tensor(start_t, dtype=torch.long),
            'obs_timestamps': torch.from_numpy(obs_timestamps),
            'action_timestamps': torch.from_numpy(action_timestamps),
        }




def get_data_stats(dataloader) -> dict:
    from tqdm import tqdm

    print("Computing normalization statistics...")
    proprio_data = []
    action_data = []

    for batch in tqdm(dataloader, desc="Computing stats"):
        proprio_dim = batch['proprio'].shape[-1]
        proprio_data.append(batch['proprio'].reshape(-1, proprio_dim))

        action_dim = batch['action'].shape[-1]
        action_data.append(batch['action'].reshape(-1, action_dim))

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
    print(f"  Proprio shape: {proprio_data.shape}, dim: {proprio_dim}")
    print(f"  Proprio range: [{stats['proprio']['min'].min():.3f}, {stats['proprio']['max'].max():.3f}]")
    print(f"  Action range: [{stats['action']['min'].min():.3f}, {stats['action']['max'].max():.3f}]")

    return stats


class Normalizer:

    def __init__(self, stats: dict):
        self.stats = stats

    def normalize(self, batch: dict) -> dict:
        normalized = {}
        for key in ['proprio', 'action']:
            if key in batch:
                data = batch[key]
                data_min = self.stats[key]['min'].to(data.device)
                data_max = self.stats[key]['max'].to(data.device)
                normalized[key] = 2 * (data - data_min) / (data_max - data_min + 1e-8) - 1

        for key in batch:
            if key not in ['proprio', 'action']:
                normalized[key] = batch[key]

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