#!/usr/bin/env python3
"""
Preprocessing script for robot trajectory data with forward kinematics.

This script:
1. Loads trajectory data from backup_55_traj/ folder
2. Computes forward kinematics for end effectors and wrist links
3. Extracts position and orientation (rotation matrix) for:
   - Left wrist yaw link
   - Right wrist yaw link
4. Saves processed data to data/ folder
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation
from tqdm import tqdm


class RobotFKProcessor:
    """Process robot trajectory data with forward kinematics."""

    def __init__(self, urdf_path: str):
        """
        Initialize the FK processor.

        Args:
            urdf_path: Path to the robot URDF file
        """
        self.urdf_path = urdf_path
        self.physics_client = None
        self.robot_id = None
        self.joint_indices = {}
        self.link_indices = {}

    def setup_pybullet(self):
        """Setup PyBullet simulation environment."""
        # Connect to PyBullet in DIRECT mode (no GUI)
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load robot URDF
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=False
        )

        # Build joint name to index mapping
        num_joints = p.getNumJoints(self.robot_id)
        joint_names = []

        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            link_name = joint_info[12].decode('utf-8')

            # Store joint index
            self.joint_indices[joint_name] = i
            # Store link index
            self.link_indices[link_name] = i

            if joint_info[2] == p.JOINT_REVOLUTE or joint_info[2] == p.JOINT_PRISMATIC:
                joint_names.append(joint_name)

        print(f"Loaded robot with {num_joints} joints")
        print(f"Revolute/Prismatic joints: {len(joint_names)}")

        # Print available links for debugging
        print("\nAvailable links:")
        for link_name in sorted(self.link_indices.keys()):
            if 'wrist' in link_name.lower():
                print(f"  - {link_name} (index: {self.link_indices[link_name]})")

    def get_joint_order(self) -> list:
        """
        Get the ordered list of joint names matching q_0, q_1, ..., q_28.

        Returns:
            List of joint names in the correct order
        """
        # Based on g1_29dof URDF, the typical joint order is:
        # 0-5: legs (left: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
        # 6-11: legs (right: same as left)
        # 12-14: waist (yaw, roll, pitch)
        # 15-21: left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
        # 22-28: right arm (same as left)

        joint_order = [
            # Left leg (0-5)
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            # Right leg (6-11)
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            # Waist (12-14)
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            # Left arm (15-21)
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            # Right arm (22-28)
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        return joint_order

    def compute_fk(self, joint_positions: np.ndarray, base_pos: np.ndarray,
                   base_quat: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute forward kinematics for specified links.

        Args:
            joint_positions: Array of 29 joint positions
            base_pos: Base position [x, y, z]
            base_quat: Base quaternion [w, x, y, z]

        Returns:
            Dictionary with link names as keys and pos/rot as values
        """
        # Set base pose
        p.resetBasePositionAndOrientation(
            self.robot_id,
            base_pos,
            [base_quat[1], base_quat[2], base_quat[3], base_quat[0]]  # PyBullet uses [x,y,z,w]
        )

        # Set joint positions
        joint_order = self.get_joint_order()
        for joint_name, joint_pos in zip(joint_order, joint_positions):
            if joint_name in self.joint_indices:
                p.resetJointState(
                    self.robot_id,
                    self.joint_indices[joint_name],
                    joint_pos
                )

        # Compute FK for desired links
        link_states = {}
        links_to_track = [
            "left_wrist_yaw_link",
            "right_wrist_yaw_link"
        ]

        for link_name in links_to_track:
            if link_name in self.link_indices:
                link_state = p.getLinkState(
                    self.robot_id,
                    self.link_indices[link_name],
                    computeForwardKinematics=True
                )

                # Extract position and orientation
                pos = np.array(link_state[0])  # World position
                quat = np.array(link_state[1])  # World orientation [x, y, z, w]

                # Convert quaternion to rotation matrix
                rot = Rotation.from_quat(quat).as_matrix()

                link_states[link_name] = {
                    'position': pos,
                    'rotation': rot,
                    'quaternion': quat
                }

        return link_states

    def process_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Process a single CSV file and add FK data.

        Args:
            csv_path: Path to input CSV file

        Returns:
            DataFrame with original data plus FK data
        """
        # Load CSV
        df = pd.read_csv(csv_path)

        # Extract joint positions (q_0 to q_28)
        joint_cols = [f'q_{i}' for i in range(29)]
        joint_positions = df[joint_cols].values

        # Extract base pose
        base_pos = df[['pos_x', 'pos_y', 'pos_z']].values
        base_quat = df[['quat_w', 'quat_x', 'quat_y', 'quat_z']].values

        # Initialize arrays for FK data
        num_samples = len(df)

        # Left wrist
        left_wrist_pos = np.zeros((num_samples, 3))
        left_wrist_rot = np.zeros((num_samples, 9))  # Flattened 3x3 rotation matrix
        left_wrist_quat = np.zeros((num_samples, 4))

        # Right wrist
        right_wrist_pos = np.zeros((num_samples, 3))
        right_wrist_rot = np.zeros((num_samples, 9))
        right_wrist_quat = np.zeros((num_samples, 4))

        # Process each timestep
        for i in tqdm(range(num_samples), desc=f"Processing {Path(csv_path).name}"):
            link_states = self.compute_fk(
                joint_positions[i],
                base_pos[i],
                base_quat[i]
            )

            # Store left wrist data
            if 'left_wrist_yaw_link' in link_states:
                left_wrist_pos[i] = link_states['left_wrist_yaw_link']['position']
                left_wrist_rot[i] = link_states['left_wrist_yaw_link']['rotation'].flatten()
                left_wrist_quat[i] = link_states['left_wrist_yaw_link']['quaternion']

            # Store right wrist data
            if 'right_wrist_yaw_link' in link_states:
                right_wrist_pos[i] = link_states['right_wrist_yaw_link']['position']
                right_wrist_rot[i] = link_states['right_wrist_yaw_link']['rotation'].flatten()
                right_wrist_quat[i] = link_states['right_wrist_yaw_link']['quaternion']

        # Add FK data to dataframe
        # Left wrist
        df['left_wrist_pos_x'] = left_wrist_pos[:, 0]
        df['left_wrist_pos_y'] = left_wrist_pos[:, 1]
        df['left_wrist_pos_z'] = left_wrist_pos[:, 2]
        for i in range(9):
            df[f'left_wrist_rot_{i}'] = left_wrist_rot[:, i]
        df['left_wrist_quat_x'] = left_wrist_quat[:, 0]
        df['left_wrist_quat_y'] = left_wrist_quat[:, 1]
        df['left_wrist_quat_z'] = left_wrist_quat[:, 2]
        df['left_wrist_quat_w'] = left_wrist_quat[:, 3]

        # Right wrist
        df['right_wrist_pos_x'] = right_wrist_pos[:, 0]
        df['right_wrist_pos_y'] = right_wrist_pos[:, 1]
        df['right_wrist_pos_z'] = right_wrist_pos[:, 2]
        for i in range(9):
            df[f'right_wrist_rot_{i}'] = right_wrist_rot[:, i]
        df['right_wrist_quat_x'] = right_wrist_quat[:, 0]
        df['right_wrist_quat_y'] = right_wrist_quat[:, 1]
        df['right_wrist_quat_z'] = right_wrist_quat[:, 2]
        df['right_wrist_quat_w'] = right_wrist_quat[:, 3]

        return df

    def cleanup(self):
        """Disconnect from PyBullet."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)


def main():
    """Main preprocessing function."""
    # Paths
    project_root = Path(__file__).parent.parent
    urdf_path = project_root / "unitree_rl_gym/resources/robots/g1_description/g1_29dof.urdf"
    input_dir = project_root / "backup_55_traj"
    output_dir = project_root / "data"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print(f"URDF path: {urdf_path}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Initialize FK processor
    print("\nInitializing FK processor...")
    processor = RobotFKProcessor(str(urdf_path))
    processor.setup_pybullet()

    # Find all CSV files
    csv_files = sorted(input_dir.glob("merged_50hz_log*.csv"))
    print(f"\nFound {len(csv_files)} CSV files to process")

    # Process each file
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"Processing: {csv_file.name}")
        print(f"{'='*60}")

        # Process CSV
        df_processed = processor.process_csv(str(csv_file))

        # Save to output directory
        output_file = output_dir / csv_file.name
        df_processed.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}")
        print(f"Output shape: {df_processed.shape}")

    # Cleanup
    processor.cleanup()
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Processed {len(csv_files)} files")
    print(f"Output saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
