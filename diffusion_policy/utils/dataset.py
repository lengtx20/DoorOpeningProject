import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from torchvision import transforms

class G1Dataset(Dataset):
    def __init__(self, 
                 dataset_root, 
                 mode='train', 
                 obs_horizon=2, 
                 pred_horizon=16,
                 use_proprio=True,
                 image_resize_size=None,
                 sample_stride=5):
      
        self.root = dataset_root
        self.mode = mode
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.use_proprio = use_proprio
        self.sample_stride = sample_stride

        if image_resize_size is not None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(image_resize_size, antialias=True)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

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
            
            grasp_key = 'p_pressed' if 'p_pressed' in raw_dict else 'q_19'
            
            action_vec = np.concatenate([
                raw_dict['left_wrist_torso_pos_x'].reshape(-1, 1),
                raw_dict['left_wrist_torso_pos_y'].reshape(-1, 1),
                raw_dict['left_wrist_torso_pos_z'].reshape(-1, 1),
                raw_dict['right_wrist_torso_pos_x'].reshape(-1, 1),
                raw_dict['right_wrist_torso_pos_y'].reshape(-1, 1),
                raw_dict['right_wrist_torso_pos_z'].reshape(-1, 1),
                raw_dict['vel_body_x'].reshape(-1, 1),
                raw_dict['vel_body_y'].reshape(-1, 1),
                raw_dict['yaw_speed'].reshape(-1, 1),
                raw_dict[grasp_key].reshape(-1, 1)
            ], axis=1).astype(np.float32)

 
            proprio_vec = None
            if self.use_proprio:
                q_cols = [f'q_{i}' for i in range(29)]
                
                q_vec = np.stack([raw_dict[c] for c in q_cols], axis=-1).astype(np.float32) 
                proprio_vec = q_vec 

            world_pose_vec = np.concatenate([
                raw_dict['pos_x'].reshape(-1, 1),
                raw_dict['pos_y'].reshape(-1, 1),
                raw_dict['pos_z'].reshape(-1, 1),
                raw_dict['quat_x'].reshape(-1, 1),
                raw_dict['quat_y'].reshape(-1, 1),
                raw_dict['quat_z'].reshape(-1, 1),
                raw_dict['quat_w'].reshape(-1, 1),
            ], axis=1).astype(np.float32)

            self.data_cache[ep_id] = {
                'action_traj': action_vec,
                'proprio_traj': proprio_vec,
                'pose_traj': world_pose_vec, 
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

        images_list = []
        for i in range(self.obs_horizon):
            frame_idx = start_t + i
            fname = f"{frame_idx:06d}.npy"
            fpath = os.path.join(self.root, ep_id, fname)
            
            img_arr = np.load(fpath)
            img_t = self.transform(img_arr)
            images_list.append(img_t)
        
        image_tensor = torch.stack(images_list)

        act_start = start_t + (self.obs_horizon - 1)
        act_end = act_start + self.pred_horizon
        action = data['action_traj'][act_start : act_end]

        if self.use_proprio:
            agent_pos = data['proprio_traj'][start_t : start_t + self.obs_horizon]
            agent_pos_tensor = torch.from_numpy(agent_pos)
        else:
            agent_pos_tensor = torch.empty(0)
    
        current_pose_idx = start_t + self.obs_horizon - 1
        start_pose = data['pose_traj'][current_pose_idx]

        return {
            'image': image_tensor,
            'action': torch.from_numpy(action),
            'agent_pos': agent_pos_tensor, 
            'start_pose': torch.from_numpy(start_pose)
        }

    def get_normalizer(self):
        all_actions = []
        all_proprios = []
        
        for ep_id in self.episode_ids:
            all_actions.append(self.data_cache[ep_id]['action_traj'])
            if self.use_proprio:
                all_proprios.append(self.data_cache[ep_id]['proprio_traj'])
        
        all_actions = np.concatenate(all_actions, axis=0)
        
        stats = {
            'action': {
                'min': torch.from_numpy(all_actions.min(axis=0)),
                'max': torch.from_numpy(all_actions.max(axis=0)),
                'mean': torch.from_numpy(all_actions.mean(axis=0)),
                'std': torch.from_numpy(all_actions.std(axis=0))
            }
        }
        
        if self.use_proprio and len(all_proprios) > 0:
            all_proprios = np.concatenate(all_proprios, axis=0)
            stats['agent_pos'] = {
                'min': torch.from_numpy(all_proprios.min(axis=0)),
                'max': torch.from_numpy(all_proprios.max(axis=0)),
                'mean': torch.from_numpy(all_proprios.mean(axis=0)),
                'std': torch.from_numpy(all_proprios.std(axis=0))
            }
            
        return stats