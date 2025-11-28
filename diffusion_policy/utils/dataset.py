"""
Dataset for loading G1 robot trajectory data.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings


class G1TrajectoryDataset(Dataset):
    """
    Dataset for G1 robot trajectories from preprocessed CSV files.

    Loads demonstration data and creates training samples with observation
    and action sequences for diffusion policy.
    """

    def __init__(
        self,
        data_dir: str,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        obs_keys: Optional[List[str]] = None,
        action_keys: Optional[List[str]] = None,
        augment: bool = False,
    ):
        """
        Args:
            data_dir: Directory containing CSV files
            obs_horizon: Number of observation frames to use
            pred_horizon: Number of action frames to predict
            action_horizon: Number of action frames to execute
            obs_keys: List of observation column names (if None, auto-detect)
            action_keys: List of action column names (if None, use q_0 to q_28)
            augment: Apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.augment = augment

        # Default action keys (joint positions)
        if action_keys is None:
            self.action_keys = [f'q_{i}' for i in range(29)]
        else:
            self.action_keys = action_keys

        # Default observation keys
        if obs_keys is None:
            self.obs_keys = self._get_default_obs_keys()
        else:
            self.obs_keys = obs_keys

        # Load all trajectories
        self.trajectories = self._load_trajectories()

        # Create indices for sampling
        self.indices = self._create_indices()

        print(f"Loaded {len(self.trajectories)} trajectories")
        print(f"Total samples: {len(self.indices)}")
        print(f"Observation dim: {len(self.obs_keys)}")
        print(f"Action dim: {len(self.action_keys)}")

    def _get_default_obs_keys(self) -> List[str]:
        """
        Get default observation keys based on G1 state.

        Returns a subset of available data that's useful for policy learning.
        """
        obs_keys = []

        # Base state
        obs_keys.extend(['pos_x', 'pos_y', 'pos_z'])  # Base position
        obs_keys.extend(['quat_w', 'quat_x', 'quat_y', 'quat_z'])  # Base orientation
        obs_keys.extend(['vel_x', 'vel_y', 'vel_z'])  # Base linear velocity
        obs_keys.extend(['gyro_x', 'gyro_y', 'gyro_z'])  # Angular velocity

        # Joint positions and velocities
        for i in range(29):
            obs_keys.append(f'q_{i}')
            obs_keys.append(f'dq_{i}')

        # Wrist positions (if available in data)
        obs_keys.extend(['left_wrist_pos_x', 'left_wrist_pos_y', 'left_wrist_pos_z'])
        obs_keys.extend(['right_wrist_pos_x', 'right_wrist_pos_y', 'right_wrist_pos_z'])

        return obs_keys

    def _load_trajectories(self) -> List[Dict[str, np.ndarray]]:
        """
        Load all CSV files as trajectories.

        Returns:
            List of trajectory dictionaries with 'obs' and 'actions' arrays
        """
        csv_files = sorted(self.data_dir.glob("merged_50hz_log*.csv"))

        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        trajectories = []

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Filter observation keys that exist in the dataframe
                available_obs_keys = [k for k in self.obs_keys if k in df.columns]
                if len(available_obs_keys) < len(self.obs_keys):
                    missing = set(self.obs_keys) - set(available_obs_keys)
                    warnings.warn(f"Missing obs keys in {csv_file.name}: {missing}")

                # Extract observations and actions
                obs = df[available_obs_keys].values.astype(np.float32)
                actions = df[self.action_keys].values.astype(np.float32)

                # Skip trajectories that are too short
                if len(obs) < self.obs_horizon + self.pred_horizon:
                    warnings.warn(f"Skipping {csv_file.name}: too short ({len(obs)} frames)")
                    continue

                trajectories.append({
                    'obs': obs,
                    'actions': actions,
                    'filename': csv_file.name,
                })

            except Exception as e:
                warnings.warn(f"Error loading {csv_file.name}: {e}")
                continue

        return trajectories

    def _create_indices(self) -> List[Tuple[int, int]]:
        """
        Create list of (trajectory_idx, start_idx) pairs for sampling.

        Returns:
            List of valid sampling indices
        """
        indices = []

        for traj_idx, traj in enumerate(self.trajectories):
            traj_len = len(traj['obs'])

            # We need obs_horizon frames for observation and pred_horizon for actions
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
                - 'obs': (obs_horizon, obs_dim) observation sequence
                - 'actions': (pred_horizon, action_dim) action sequence
        """
        traj_idx, start_idx = self.indices[idx]
        traj = self.trajectories[traj_idx]

        # Extract observation sequence
        obs_end = start_idx + self.obs_horizon
        obs_seq = traj['obs'][start_idx:obs_end]

        # Extract action sequence
        action_start = start_idx + self.obs_horizon - 1  # Overlap by 1
        action_end = action_start + self.pred_horizon
        action_seq = traj['actions'][action_start:action_end]

        # Data augmentation (optional)
        if self.augment:
            obs_seq, action_seq = self._augment(obs_seq, action_seq)

        return {
            'obs': torch.from_numpy(obs_seq).float(),
            'actions': torch.from_numpy(action_seq).float(),
        }

    def _augment(self, obs: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation.

        Args:
            obs: (obs_horizon, obs_dim)
            actions: (pred_horizon, action_dim)
        Returns:
            Augmented obs and actions
        """
        # Add small noise
        obs_noise = np.random.normal(0, 0.01, obs.shape).astype(np.float32)
        action_noise = np.random.normal(0, 0.005, actions.shape).astype(np.float32)

        obs = obs + obs_noise
        actions = actions + action_noise

        return obs, actions

    def get_stats(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute statistics for normalization.

        Returns:
            Dictionary with 'obs' and 'actions' statistics (min, max, mean, std)
        """
        all_obs = []
        all_actions = []

        for traj in self.trajectories:
            all_obs.append(traj['obs'])
            all_actions.append(traj['actions'])

        all_obs = np.concatenate(all_obs, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)

        stats = {
            'obs': {
                'min': np.min(all_obs, axis=0),
                'max': np.max(all_obs, axis=0),
                'mean': np.mean(all_obs, axis=0),
                'std': np.std(all_obs, axis=0),
            },
            'actions': {
                'min': np.min(all_actions, axis=0),
                'max': np.max(all_actions, axis=0),
                'mean': np.mean(all_actions, axis=0),
                'std': np.std(all_actions, axis=0),
            }
        }

        return stats
