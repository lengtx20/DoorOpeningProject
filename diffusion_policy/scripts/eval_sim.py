#!/usr/bin/env python3
"""
Evaluation script for diffusion policy in unitree_rl_gym simulation.

This script loads a trained diffusion policy and evaluates it in the
G1 robot simulation environment for door opening task.
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from tqdm import tqdm
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from diffusion_policy.models.diffusion_unet import DiffusionUNet
from diffusion_policy.utils.normalizer import LinearNormalizer

# Import unitree_rl_gym
sys.path.insert(0, str(project_root / "unitree_rl_gym"))
from legged_gym.utils import task_registry, get_args
from legged_gym import LEGGED_GYM_ROOT_DIR


class DiffusionPolicyController:
    """Controller that uses diffusion policy for action generation."""

    def __init__(
        self,
        model_path: str,
        normalizer_path: str,
        device: str = 'cuda:0',
        use_ddim: bool = True,
        ddim_steps: int = 10,
    ):
        """
        Initialize diffusion policy controller.

        Args:
            model_path: Path to trained model checkpoint
            normalizer_path: Path to normalizer statistics
            device: Device to run on
            use_ddim: Use DDIM sampling (faster)
            ddim_steps: Number of DDIM steps
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        # Load normalizer
        self.normalizer = self._load_normalizer(normalizer_path)

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Observation history buffer
        self.obs_history = []

        print(f"Loaded diffusion policy controller")
        print(f"  Device: {self.device}")
        print(f"  Sampling: {'DDIM' if use_ddim else 'DDPM'}")
        print(f"  DDIM steps: {ddim_steps if use_ddim else 'N/A'}")

    def _load_normalizer(self, path: str) -> LinearNormalizer:
        """Load normalization statistics."""
        data = np.load(path)
        normalizer = LinearNormalizer(mode='limits')

        normalizer.params['obs'] = {
            'min': data['obs_min'],
            'max': data['obs_max'],
            'scale': data['obs_max'] - data['obs_min'],
        }
        normalizer.params['actions'] = {
            'min': data['actions_min'],
            'max': data['actions_max'],
            'scale': data['actions_max'] - data['actions_min'],
        }

        return normalizer

    def _load_model(self, path: str) -> DiffusionUNet:
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)

        # Extract model dimensions from checkpoint
        state_dict = checkpoint['model_state_dict']

        # Infer dimensions from saved weights
        # obs_encoder first layer: (obs_dim * obs_horizon, hidden_dim)
        obs_encoder_weight = state_dict['obs_encoder.0.weight']
        obs_dim_total = obs_encoder_weight.shape[1]

        # noise_pred_net input: action_dim
        input_proj_weight = state_dict['noise_pred_net.input_proj.block.0.weight']
        action_dim = input_proj_weight.shape[1]

        # Create model (use defaults that match training)
        model = DiffusionUNet(
            obs_dim=obs_dim_total // 2,  # Assuming obs_horizon=2
            action_dim=action_dim,
            obs_horizon=2,
            pred_horizon=16,
            action_horizon=8,
            num_diffusion_iters=100,
            down_dims=(256, 512, 1024),
            obs_encoder_layers=(256, 256),
        ).to(self.device)

        model.load_state_dict(state_dict)

        return model

    def reset(self):
        """Reset observation history."""
        self.obs_history = []
        self.action_buffer = None
        self.action_counter = 0

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get action from observation.

        Args:
            obs: Current observation (obs_dim,)
        Returns:
            action: Action to execute (action_dim,)
        """
        # Add to history
        self.obs_history.append(obs)

        # Keep only last obs_horizon observations
        if len(self.obs_history) > self.model.obs_horizon:
            self.obs_history.pop(0)

        # Need enough history
        if len(self.obs_history) < self.model.obs_horizon:
            # Return zero action until we have enough history
            return np.zeros(self.model.action_dim, dtype=np.float32)

        # Generate new action sequence every action_horizon steps
        if self.action_buffer is None or self.action_counter >= self.model.action_horizon:
            self.action_buffer = self._generate_actions()
            self.action_counter = 0

        # Get current action from buffer
        action = self.action_buffer[self.action_counter]
        self.action_counter += 1

        return action

    def _generate_actions(self) -> np.ndarray:
        """
        Generate action sequence using diffusion policy.

        Returns:
            actions: (pred_horizon, action_dim) array
        """
        # Stack observation history
        obs_seq = np.stack(self.obs_history, axis=0)  # (obs_horizon, obs_dim)

        # Normalize
        obs_tensor = torch.from_numpy(obs_seq).float().to(self.device)
        obs_tensor = obs_tensor.unsqueeze(0)  # (1, obs_horizon, obs_dim)

        B, T, D = obs_tensor.shape
        obs_flat = obs_tensor.reshape(B * T, D)
        obs_norm = self.normalizer.normalize(obs_flat, 'obs').reshape(B, T, D)

        # Sample actions
        with torch.no_grad():
            action_pred = self.model.conditional_sample(
                obs_norm,
                use_ddim=self.use_ddim,
                ddim_steps=self.ddim_steps,
            )  # (1, pred_horizon, action_dim)

        # Denormalize
        B, T_act, D_act = action_pred.shape
        action_flat = action_pred.reshape(B * T_act, D_act)
        action_denorm = self.normalizer.denormalize(action_flat, 'actions')
        action_denorm = action_denorm.reshape(B, T_act, D_act)

        # Convert to numpy
        actions = action_denorm[0].cpu().numpy()

        return actions


def build_observation(env, env_id: int = 0) -> np.ndarray:
    """
    Build observation from environment state.

    This should match the observation keys used in training dataset.

    Args:
        env: Gym environment
        env_id: Environment index
    Returns:
        obs: Observation array
    """
    obs_list = []

    # Base state
    obs_list.extend(env.base_pos[env_id].cpu().numpy())  # pos_x, pos_y, pos_z
    obs_list.extend(env.base_quat[env_id].cpu().numpy())  # quat_w, quat_x, quat_y, quat_z
    obs_list.extend(env.base_lin_vel[env_id].cpu().numpy())  # vel_x, vel_y, vel_z
    obs_list.extend(env.base_ang_vel[env_id].cpu().numpy())  # gyro_x, gyro_y, gyro_z

    # Joint positions and velocities
    for i in range(env.num_dof):
        obs_list.append(env.dof_pos[env_id, i].item())
        obs_list.append(env.dof_vel[env_id, i].item())

    # Wrist positions (if available)
    # Note: This would require FK computation or access to link states
    # For now, use zeros as placeholder
    obs_list.extend([0.0] * 6)  # left_wrist_pos, right_wrist_pos

    return np.array(obs_list, dtype=np.float32)


def evaluate_policy(
    controller: DiffusionPolicyController,
    env,
    num_episodes: int = 10,
    max_steps: int = 1000,
    render: bool = True,
):
    """
    Evaluate diffusion policy in simulation.

    Args:
        controller: Diffusion policy controller
        env: Gym environment
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Render simulation
    Returns:
        stats: Dictionary with evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for episode in range(num_episodes):
        # Reset environment
        env.reset()
        controller.reset()

        episode_reward = 0.0
        episode_length = 0

        for step in tqdm(range(max_steps), desc=f"Episode {episode + 1}/{num_episodes}"):
            # Get observation (use first environment)
            obs = build_observation(env, env_id=0)

            # Get action from policy
            action = controller.get_action(obs)

            # Convert action to tensor and expand to all environments
            action_tensor = torch.from_numpy(action).float().to(env.device)
            action_tensor = action_tensor.unsqueeze(0).repeat(env.num_envs, 1)

            # Step environment
            obs_buf, privileged_obs, rewards, resets, infos = env.step(action_tensor)

            # Track statistics (for first environment)
            episode_reward += rewards[0].item()
            episode_length += 1

            # Check if episode done
            if resets[0]:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"\nEpisode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    # Compute statistics
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / num_episodes,
    }

    return stats


def main(args):
    """Main evaluation function."""
    # Load environment
    print("Loading simulation environment...")

    # Create custom args for environment
    class EnvArgs:
        task = args.task
        sim_device = args.sim_device
        rl_device = args.rl_device
        headless = args.headless
        num_envs = args.num_envs

    env_args = EnvArgs()

    # Create environment
    env, env_cfg = task_registry.make_env(name=args.task, args=env_args)

    print(f"Environment: {args.task}")
    print(f"Num envs: {env.num_envs}")
    print(f"Observation dim: {env.num_obs}")
    print(f"Action dim: {env.num_actions}")

    # Load diffusion policy controller
    print(f"\nLoading diffusion policy from {args.model_path}...")
    controller = DiffusionPolicyController(
        model_path=args.model_path,
        normalizer_path=args.normalizer_path,
        device=args.rl_device,
        use_ddim=args.use_ddim,
        ddim_steps=args.ddim_steps,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("Starting evaluation...")
    print("=" * 60)

    stats = evaluate_policy(
        controller=controller,
        env=env,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render=not args.headless,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print(f"Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Mean episode length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print("=" * 60)

    # Save results
    if args.save_results:
        results_path = Path(args.model_path).parent / "eval_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(stats, f)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate diffusion policy in simulation")

    # Model
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--normalizer_path', type=str, required=True,
                        help='Path to normalizer statistics')

    # Sampling
    parser.add_argument('--use_ddim', action='store_true', default=True,
                        help='Use DDIM sampling')
    parser.add_argument('--ddim_steps', type=int, default=10,
                        help='Number of DDIM steps')

    # Environment
    parser.add_argument('--task', type=str, default='g1',
                        help='Task name')
    parser.add_argument('--num_envs', type=int, default=1,
                        help='Number of parallel environments')
    parser.add_argument('--sim_device', type=str, default='cuda:0',
                        help='Simulation device')
    parser.add_argument('--rl_device', type=str, default='cuda:0',
                        help='RL device')
    parser.add_argument('--headless', action='store_true',
                        help='Run without rendering')

    # Evaluation
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per episode')
    parser.add_argument('--save_results', action='store_true',
                        help='Save evaluation results')

    args = parser.parse_args()

    main(args)
