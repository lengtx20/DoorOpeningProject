"""
Dataset for G1 door opening task with image observations + 29 DOF state.

Input:
- Images (visual observations)
- 29 DOF (joint positions q_0 to q_28, and optionally velocities dq_0 to dq_28)

Output:
- 10-dim actions:
  - left_hand_pos_torso: [x, y, z] (3)
  - right_hand_pos_torso: [x, y, z] (3)
  - base_vel_cmd: [vx, vy, yaw_rate] (3)
  - left_hand_grasp: [grasp_state] (1)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation
import warnings
import os


class G1ImageDoorOpeningDataset(Dataset):
    """
    Dataset for G1 door opening task with image + 29 DOF observations.
    
    Input:
    - Images: (obs_horizon, C, H, W) - visual observations
    - 29 DOF state: (obs_horizon, 29 or 58) - joint positions (and optionally velocities)
    
    Output:
    - 10-dim actions: (pred_horizon, 10) - task-specific actions
    """

    def __init__(
        self,
        data_dir: str,
        image_dir: Optional[str] = None,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        include_velocities: bool = True,
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        use_grasp_from_pressure: bool = True,
    ):
        """
        Args:
            data_dir: Directory containing CSV files
            image_dir: Directory containing images (if None, looks for images/ subdirectory)
            obs_horizon: Number of observation frames to use
            pred_horizon: Number of action frames to predict
            action_horizon: Number of action frames to execute
            include_velocities: Include joint velocities (29 DOF -> 58 dims)
            image_size: (height, width) for image resizing
            augment: Apply data augmentation
            use_grasp_from_pressure: Use p_pressed column for grasp state
        """
        self.data_dir = Path(data_dir)
        if image_dir is None:
            self.image_dir = self.data_dir / "images"
        else:
            self.image_dir = Path(image_dir)
        
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.include_velocities = include_velocities
        self.image_size = image_size
        self.augment = augment
        self.use_grasp_from_pressure = use_grasp_from_pressure
        
        # DOF dimension: 29 (positions only) or 58 (positions + velocities)
        self.dof_dim = 58 if include_velocities else 29
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.augment_transform = None

        # Load all trajectories
        self.trajectories = self._load_trajectories()

        # Create indices for sampling
        self.indices = self._create_indices()

        print(f"Loaded {len(self.trajectories)} trajectories")
        print(f"Total samples: {len(self.indices)}")
        print(f"Image directory: {self.image_dir}")
        print(f"DOF state dim: {self.dof_dim} ({'positions + velocities' if include_velocities else 'positions only'})")
        print(f"Action dim: 10")

    def _world_to_torso_frame(
        self,
        world_pos: np.ndarray,
        base_pos: np.ndarray,
        base_quat: np.ndarray
    ) -> np.ndarray:
        """Transform world position to torso (base) frame."""
        single_input = world_pos.ndim == 1

        if single_input:
            world_pos = world_pos.reshape(1, 3)
            base_pos = base_pos.reshape(1, 3)
            base_quat = base_quat.reshape(1, 4)

        rel_pos_world = world_pos - base_pos
        quat_scipy = np.concatenate([base_quat[:, 1:], base_quat[:, 0:1]], axis=1)
        R_world_to_base = Rotation.from_quat(quat_scipy).inv().as_matrix()
        rel_pos_base = np.einsum('bij,bj->bi', R_world_to_base, rel_pos_world)

        if single_input:
            return rel_pos_base[0]
        return rel_pos_base

    def _load_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess an image."""
        try:
            if not image_path.exists():
                # Return black image if not found
                return torch.zeros(3, self.image_size[0], self.image_size[1])
            
            img = Image.open(image_path).convert('RGB')
            
            if self.augment_transform is not None:
                img = self.augment_transform(img)
            
            img = self.image_transform(img)
            return img
        except Exception as e:
            warnings.warn(f"Error loading image {image_path}: {e}")
            return torch.zeros(3, self.image_size[0], self.image_size[1])

    def _load_trajectories(self) -> List[Dict[str, np.ndarray]]:
        """Load all CSV files and compute task-specific actions."""
        csv_files = sorted(self.data_dir.glob("merged_50hz_log*.csv"))

        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        trajectories = []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Extract required columns
                required_cols = [
                    'pos_x', 'pos_y', 'pos_z',
                    'quat_w', 'quat_x', 'quat_y', 'quat_z',
                    'vel_x', 'vel_y', 'yaw_speed',
                ]

                # Check for wrist positions
                wrist_cols = [
                    'left_wrist_pos_x', 'left_wrist_pos_y', 'left_wrist_pos_z',
                    'right_wrist_pos_x', 'right_wrist_pos_y', 'right_wrist_pos_z',
                ]

                # Check for grasp state
                if self.use_grasp_from_pressure and 'p_pressed' in df.columns:
                    grasp_col = 'p_pressed'
                else:
                    grasp_col = 'q_19'

                missing_required = [c for c in required_cols if c not in df.columns]
                missing_wrist = [c for c in wrist_cols if c not in df.columns]

                if missing_required:
                    warnings.warn(f"Skipping {csv_file.name}: missing {missing_required}")
                    continue

                if missing_wrist:
                    warnings.warn(f"Skipping {csv_file.name}: missing wrist data. Run preprocessing first!")
                    continue

                n_samples = len(df)

                # Extract 29 DOF state (joint positions and optionally velocities)
                dof_state = []
                for i in range(29):
                    dof_state.append(df[f'q_{i}'].values.astype(np.float32))
                    if self.include_velocities:
                        if f'dq_{i}' in df.columns:
                            dof_state.append(df[f'dq_{i}'].values.astype(np.float32))
                        else:
                            # Use zeros if velocities not available
                            dof_state.append(np.zeros(n_samples, dtype=np.float32))
                
                dof_state = np.stack(dof_state, axis=1)  # (n_samples, dof_dim)

                # Base state for transformations
                base_pos = df[['pos_x', 'pos_y', 'pos_z']].values.astype(np.float32)
                base_quat = df[['quat_w', 'quat_x', 'quat_y', 'quat_z']].values.astype(np.float32)

                # Wrist positions (world frame)
                left_wrist_world = df[['left_wrist_pos_x', 'left_wrist_pos_y',
                                       'left_wrist_pos_z']].values.astype(np.float32)
                right_wrist_world = df[['right_wrist_pos_x', 'right_wrist_pos_y',
                                        'right_wrist_pos_z']].values.astype(np.float32)

                # Transform wrists to torso frame
                left_wrist_torso = self._world_to_torso_frame(
                    left_wrist_world, base_pos, base_quat
                )
                right_wrist_torso = self._world_to_torso_frame(
                    right_wrist_world, base_pos, base_quat
                )

                # Base velocity command
                base_vel_cmd = df[['vel_x', 'vel_y', 'yaw_speed']].values.astype(np.float32)

                # Grasp state
                if grasp_col in df.columns:
                    grasp_state = df[grasp_col].values.astype(np.float32).reshape(-1, 1)
                else:
                    grasp_state = np.zeros((n_samples, 1), dtype=np.float32)

                # Concatenate actions: [left_hand(3), right_hand(3), vel_cmd(3), grasp(1)]
                actions = np.concatenate([
                    left_wrist_torso,    # 3
                    right_wrist_torso,   # 3
                    base_vel_cmd,        # 3
                    grasp_state,         # 1
                ], axis=1)  # Total: 10 dimensions

                # Skip trajectories that are too short
                if len(dof_state) < self.obs_horizon + self.pred_horizon:
                    warnings.warn(f"Skipping {csv_file.name}: too short ({len(dof_state)} frames)")
                    continue

                trajectories.append({
                    'dof_state': dof_state,
                    'actions': actions,
                    'filename': csv_file.name,
                    'n_samples': n_samples,
                })

            except Exception as e:
                warnings.warn(f"Error loading {csv_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return trajectories

    def _create_indices(self) -> List[Tuple[int, int]]:
        """Create list of (trajectory_idx, start_idx) pairs for sampling."""
        indices = []

        for traj_idx, traj in enumerate(self.trajectories):
            traj_len = len(traj['dof_state'])
            max_start = traj_len - self.obs_horizon - self.pred_horizon + 1

            for start_idx in range(max_start):
                indices.append((traj_idx, start_idx))

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
                - 'images': (obs_horizon, C, H, W) image sequence
                - 'dof_state': (obs_horizon, dof_dim) DOF state sequence
                - 'actions': (pred_horizon, 10) action sequence
        """
        traj_idx, start_idx = self.indices[idx]
        traj = self.trajectories[traj_idx]

        # Extract observation sequence
        obs_end = start_idx + self.obs_horizon
        dof_state_seq = traj['dof_state'][start_idx:obs_end]

        # Load images for this sequence
        images = []
        filename_base = traj['filename'].replace('.csv', '')
        for i in range(start_idx, obs_end):
            # Try different image naming conventions
            image_paths = [
                self.image_dir / f"{filename_base}_{i:06d}.jpg",
                self.image_dir / f"{filename_base}_{i:06d}.png",
                self.image_dir / f"{filename_base}/frame_{i:06d}.jpg",
                self.image_dir / f"{filename_base}/frame_{i:06d}.png",
            ]
            
            image_loaded = False
            for img_path in image_paths:
                if img_path.exists():
                    images.append(self._load_image(img_path))
                    image_loaded = True
                    break
            
            if not image_loaded:
                # Use black image if not found
                images.append(torch.zeros(3, self.image_size[0], self.image_size[1]))
        
        images = torch.stack(images, dim=0)  # (obs_horizon, C, H, W)

        # Extract action sequence
        action_start = start_idx + self.obs_horizon - 1
        action_end = action_start + self.pred_horizon
        action_seq = traj['actions'][action_start:action_end]

        # Data augmentation for DOF state and actions
        if self.augment:
            dof_noise = np.random.normal(0, 0.01, dof_state_seq.shape).astype(np.float32)
            action_noise = np.random.normal(0, 0.005, action_seq.shape).astype(np.float32)
            dof_state_seq = dof_state_seq + dof_noise
            action_seq = action_seq + action_noise

        return {
            'images': images.float(),
            'dof_state': torch.from_numpy(dof_state_seq).float(),
            'actions': torch.from_numpy(action_seq).float(),
        }

    def get_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute statistics for normalization."""
        all_dof_state = []
        all_actions = []

        for traj in self.trajectories:
            all_dof_state.append(traj['dof_state'])
            all_actions.append(traj['actions'])

        all_dof_state = np.concatenate(all_dof_state, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)

        stats = {
            'dof_state': {
                'min': np.min(all_dof_state, axis=0),
                'max': np.max(all_dof_state, axis=0),
                'mean': np.mean(all_dof_state, axis=0),
                'std': np.std(all_dof_state, axis=0),
            },
            'actions': {
                'min': np.min(all_actions, axis=0),
                'max': np.max(all_actions, axis=0),
                'mean': np.mean(all_actions, axis=0),
                'std': np.std(all_actions, axis=0),
            }
        }

        return stats

