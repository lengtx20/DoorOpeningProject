"""
Dataset for G1 door opening task with task-specific observations.

Observations:
- Left hand position in torso frame (3D)
- Right hand position in torso frame (3D)
- Base velocity command (vx, vy, yaw_rate)
- Left hand grasp state (1D)

Total: 10 dimensions
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation
import warnings


class G1DoorOpeningDataset(Dataset):
    """
    Dataset for G1 door opening task with task-relevant observations.

    Observations (10-dim):
    - left_hand_pos_torso: [x, y, z] in torso frame (3)
    - right_hand_pos_torso: [x, y, z] in torso frame (3)
    - base_vel_cmd: [vx, vy, yaw_rate] (3)
    - left_hand_grasp: [grasp_state] (1)
    """

    def __init__(
        self,
        data_dir: str,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        action_keys: Optional[List[str]] = None,
        augment: bool = False,
        use_grasp_from_pressure: bool = True,
    ):
        """
        Args:
            data_dir: Directory containing CSV files
            obs_horizon: Number of observation frames to use
            pred_horizon: Number of action frames to predict
            action_horizon: Number of action frames to execute
            action_keys: List of action column names (if None, use q_0 to q_28)
            augment: Apply data augmentation
            use_grasp_from_pressure: Use p_pressed column for grasp state
        """
        self.data_dir = Path(data_dir)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.augment = augment
        self.use_grasp_from_pressure = use_grasp_from_pressure

        # Default action keys (joint positions)
        if action_keys is None:
            self.action_keys = [f'q_{i}' for i in range(29)]
        else:
            self.action_keys = action_keys

        # Observation dimension is fixed: 10
        self.obs_dim = 10

        # Load all trajectories
        self.trajectories = self._load_trajectories()

        # Create indices for sampling
        self.indices = self._create_indices()

        print(f"Loaded {len(self.trajectories)} trajectories")
        print(f"Total samples: {len(self.indices)}")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {len(self.action_keys)}")

    def _world_to_torso_frame(
        self,
        world_pos: np.ndarray,
        base_pos: np.ndarray,
        base_quat: np.ndarray
    ) -> np.ndarray:
        """
        Transform world position to torso (base) frame.

        Args:
            world_pos: Position in world frame (3,) or (N, 3)
            base_pos: Base position in world frame (3,) or (N, 3)
            base_quat: Base orientation quaternion [w, x, y, z] (4,) or (N, 4)

        Returns:
            Position in torso frame (3,) or (N, 3)
        """
        # Handle both single and batch inputs
        single_input = world_pos.ndim == 1

        if single_input:
            world_pos = world_pos.reshape(1, 3)
            base_pos = base_pos.reshape(1, 3)
            base_quat = base_quat.reshape(1, 4)

        # Relative position in world frame
        rel_pos_world = world_pos - base_pos

        # Convert quaternion to rotation matrix
        # base_quat is [w, x, y, z], scipy expects [x, y, z, w]
        quat_scipy = np.concatenate([base_quat[:, 1:], base_quat[:, 0:1]], axis=1)

        # Get rotation from world to base frame (inverse rotation)
        R_world_to_base = Rotation.from_quat(quat_scipy).inv().as_matrix()

        # Transform to base frame
        rel_pos_base = np.einsum('bij,bj->bi', R_world_to_base, rel_pos_world)

        if single_input:
            return rel_pos_base[0]
        return rel_pos_base

    def _load_trajectories(self) -> List[Dict[str, np.ndarray]]:
        """
        Load all CSV files and compute task-specific observations.

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

                # Extract required columns
                required_cols = [
                    'pos_x', 'pos_y', 'pos_z',
                    'quat_w', 'quat_x', 'quat_y', 'quat_z',
                    'vel_x', 'vel_y', 'yaw_speed',
                ]

                # Check for wrist positions (from preprocessing)
                wrist_cols = [
                    'left_wrist_pos_x', 'left_wrist_pos_y', 'left_wrist_pos_z',
                    'right_wrist_pos_x', 'right_wrist_pos_y', 'right_wrist_pos_z',
                ]

                # Check for grasp state
                if self.use_grasp_from_pressure and 'p_pressed' in df.columns:
                    grasp_col = 'p_pressed'
                else:
                    # Use left wrist roll joint as proxy (joint 19)
                    grasp_col = 'q_19'

                # Verify all required columns exist
                missing_required = [c for c in required_cols if c not in df.columns]
                missing_wrist = [c for c in wrist_cols if c not in df.columns]

                if missing_required:
                    warnings.warn(f"Skipping {csv_file.name}: missing {missing_required}")
                    continue

                if missing_wrist:
                    warnings.warn(f"Skipping {csv_file.name}: missing wrist data. "
                                  f"Run preprocessing first!")
                    continue

                # Extract data
                n_samples = len(df)

                # Base state
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
                    # Default: not grasping
                    grasp_state = np.zeros((n_samples, 1), dtype=np.float32)

                # Concatenate observations: [left_hand(3), right_hand(3), vel_cmd(3), grasp(1)]
                obs = np.concatenate([
                    left_wrist_torso,    # 3
                    right_wrist_torso,   # 3
                    base_vel_cmd,        # 3
                    grasp_state,         # 1
                ], axis=1)  # Total: 10 dimensions

                # Extract actions
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
                import traceback
                traceback.print_exc()
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
                - 'obs': (obs_horizon, 10) observation sequence
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
            obs: (obs_horizon, 10)
            actions: (pred_horizon, action_dim)
        Returns:
            Augmented obs and actions
        """
        # Add small noise to positions and velocities
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
