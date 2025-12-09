import argparse
import sys
from pathlib import Path
from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description="Open-loop evaluation: Precompute ACT actions from GT observations, then replay in sim.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during replay.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-Wbc-Play-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--traj_path",
    type=str,
    required=True,
    help="Path to trajectory directory with log_dict.npy (required for GT observations)"
)
parser.add_argument(
    "--act_checkpoint",
    type=str,
    required=True,
    help="Path to ACT policy checkpoint"
)
parser.add_argument("--obs_horizon", type=int, default=2, help="Observation history length for ACT")
parser.add_argument("--exec_horizon", type=int, default=8, help="Action execution horizon for ACT")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import time
import g1_wbc.tasks
import gymnasium as gym
import isaaclab_tasks
import torch
import numpy as np
from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

PROJECT_ROOT = Path(__file__).resolve().parents[3]
act_root = PROJECT_ROOT / "act_policy"

if str(act_root) not in sys.path:
    sys.path.insert(0, str(act_root))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import compute_stage_from_frame_idx, get_stage_encoding

def load_gt_trajectory(traj_path: str, load_images: bool = False):
    log_path = Path(traj_path) / "log_dict.npy"
    if not log_path.exists():
        raise FileNotFoundError(f"GT trajectory file not found: {log_path}")

    print(f"[INFO] Loading GT trajectory from: {log_path}")
    log_dict = np.load(log_path, allow_pickle=True).item()
    T = len(log_dict['timestamp'])

    gt_commands = {
        'left_hand_pos': np.stack([
            log_dict['left_wrist_pos_x'],
            log_dict['left_wrist_pos_y'],
            log_dict['left_wrist_pos_z']
        ], axis=1),
        'right_hand_pos': np.stack([
            log_dict['right_wrist_pos_x'],
            log_dict['right_wrist_pos_y'],
            log_dict['right_wrist_pos_z']
        ], axis=1),
        'base_vel_xy': np.stack([
            log_dict['vel_x'],
            log_dict['vel_y']
        ], axis=1),
        'yaw_speed': np.array(log_dict['yaw_speed']),
        'timestamp': np.array(log_dict['timestamp']),
    }

    print(f"[INFO] Loading full GT state trajectory for open-loop precomputation...")
    gt_joint_pos_traj = np.stack([
        np.array([log_dict[f'q_{i}'][t] for i in range(29)], dtype=np.float32)
        for t in range(T)
    ], axis=0)

    gt_joint_vel_traj = np.stack([
        np.array([log_dict[f'dq_{i}'][t] for i in range(29)], dtype=np.float32)
        for t in range(T)
    ], axis=0)

    gt_base_pos_traj = np.stack([
        np.array([log_dict['pos_x'][t], log_dict['pos_y'][t], log_dict['pos_z'][t]], dtype=np.float32)
        for t in range(T)
    ], axis=0)

    gt_base_quat_traj = np.stack([
        np.array([
            log_dict['quat_x'][t],
            log_dict['quat_y'][t],
            log_dict['quat_z'][t],
            log_dict['quat_w'][t]
        ], dtype=np.float32)
        for t in range(T)
    ], axis=0)

    gt_commands['joint_pos_traj'] = gt_joint_pos_traj
    gt_commands['joint_vel_traj'] = gt_joint_vel_traj
    gt_commands['base_pos_traj'] = gt_base_pos_traj
    gt_commands['base_quat_traj'] = gt_base_quat_traj

    from data.dataset import compute_stage_from_frame_idx
    if 'frame_idx' in log_dict:
        frame_indices = log_dict['frame_idx']
        stage = np.array([compute_stage_from_frame_idx(int(idx)) for idx in frame_indices], dtype=np.int32)
        gt_commands['stage'] = stage
        print(f"[INFO] Computed stage from frame_idx")
    else:
        stage = np.array([compute_stage_from_frame_idx(t) for t in range(T)], dtype=np.int32)
        gt_commands['stage'] = stage
        print(f"[WARNING] frame_idx not found, using fallback")

    if load_images:
        traj_dir = Path(traj_path)
        image_files = sorted(traj_dir.glob("*.npy"))
        image_files = [f for f in image_files if f.name != "log_dict.npy"]
        if image_files:
            gt_commands['image_path'] = traj_path
            gt_commands['num_images'] = len(image_files)
            print(f"[INFO] Found {len(image_files)} image files for ACT input")
        else:
            print(f"[WARNING] No image files found in {traj_path}")

    gt_commands['initial_state'] = {
        'joint_pos': gt_joint_pos_traj[0],
        'joint_vel': gt_joint_vel_traj[0],
        'base_pos': gt_base_pos_traj[0],
        'base_quat': gt_base_quat_traj[0],
    }

    return gt_commands


def load_real_image(traj_path: str, timestep: int, device: str = "cuda:0") -> torch.Tensor:
    image_file = Path(traj_path) / f"{timestep:06d}.npy"
    if not image_file.exists():
        return None

    image_np = np.load(image_file)

    image_tensor = torch.from_numpy(image_np).float() / 255.0

    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

    import torchvision.transforms as T
    transform = T.Resize((224, 224))
    image_tensor = transform(image_tensor)

    return image_tensor.to(device)


def precompute_actions(
    gt_commands: dict,
    model,
    normalizer,
    obs_horizon: int,
    pred_horizon: int,
    exec_horizon: int,
    device: str,
    use_image: bool = False
) -> np.ndarray:

    T = len(gt_commands['timestamp'])
    action_dim = 9

    actions_precomputed = np.zeros((T, action_dim), dtype=np.float32)

    print(f"\n[PRECOMPUTATION] Starting offline action generation...")
    print(f"  Total timesteps: {T}")
    print(f"  Observation horizon: {obs_horizon}")
    print(f"  Prediction horizon: {pred_horizon}")
    print(f"  Execution horizon: {exec_horizon}")
    print(f"  Using images: {use_image}")

    from data.dataset import get_stage_encoding

    t = 0
    while t < T:
        obs_indices = []
        for i in range(obs_horizon):
            obs_t = t - (obs_horizon - 1 - i)
            obs_t = max(0, obs_t)
            obs_indices.append(obs_t)

        pos_window = []
        vel_window = []
        for idx in obs_indices:
            pos_window.append(torch.from_numpy(gt_commands['joint_pos_traj'][idx]))
            vel_window.append(torch.from_numpy(gt_commands['joint_vel_traj'][idx]))

        pos_stack = torch.stack(pos_window, dim=0)
        vel_stack = torch.stack(vel_window, dim=0)
        proprio_stack = torch.cat([pos_stack, vel_stack], dim=-1)
        proprio_batch = proprio_stack.unsqueeze(0).to(device)


        image_batch = None
        if use_image and 'image_path' in gt_commands:
            image_window = []
            for idx in obs_indices:
                img = load_real_image(gt_commands['image_path'], idx, device)
                if img is None:
                    img = torch.zeros(1, 3, 224, 224, device=device)
                image_window.append(img[0])

            image_stack = torch.stack(image_window, dim=0)
            image_batch = image_stack.unsqueeze(0).to(device)

        if 'stage' in gt_commands:
            gt_stage_label = int(gt_commands['stage'][t])
            stage_encoding = get_stage_encoding(gt_stage_label)
        else:
            stage_encoding = get_stage_encoding(t)

        stage_encoding = torch.from_numpy(stage_encoding).float()
        stage_batch = stage_encoding.unsqueeze(0).to(device)

        normalized_obs = normalizer.normalize({'proprio': proprio_batch})['proprio']

        with torch.no_grad():
            batch = {
                'proprio': normalized_obs,
                'stage': stage_batch,
            }
            if image_batch is not None:
                batch['image'] = image_batch

            output = model(batch, inference=True)
            normalized_actions = output['action']

        actions = normalizer.denormalize(normalized_actions[0], 'action')
        actions_np = actions.cpu().numpy()[:, :9]

        num_to_copy = min(exec_horizon, T - t, len(actions_np))
        actions_precomputed[t:t+num_to_copy] = actions_np[:num_to_copy]

        t += exec_horizon

    print(f"  Actions shape: {actions_precomputed.shape}")

    return actions_precomputed


def inject_policy_command(env, action: torch.Tensor, device):
    if action.ndim == 1:
        action = action.unsqueeze(0)

    cmd_manager = env.unwrapped.command_manager

    left_hand_term = cmd_manager.get_term("target_left_hand_pos_in_base")
    left_hand_term.command[:, :3] = action[:, 0:3].to(device)

    right_hand_term = cmd_manager.get_term("target_right_hand_pos_in_base")
    right_hand_term.command[:, :3] = action[:, 3:6].to(device)

    base_vel_term = cmd_manager.get_term("target_base_velocity")
    base_vel_term.command[:, :3] = action[:, 6:9].to(device)


def replay_openloop(env, policy, actions_precomputed: np.ndarray, gt_commands: dict, device: str):

    T = len(actions_precomputed)
    robot = env.unwrapped.scene["robot"]

    print(f"\n[OPEN-LOOP REPLAY] Starting replay...")
    print(f"  Total timesteps: {T}")

    if 'initial_state' in gt_commands:
        init_joint_pos = torch.from_numpy(gt_commands['initial_state']['joint_pos']).to(device)
        init_joint_vel = torch.from_numpy(gt_commands['initial_state']['joint_vel']).to(device)
        init_base_pos = torch.from_numpy(gt_commands['initial_state']['base_pos']).to(device)
        init_base_quat = torch.from_numpy(gt_commands['initial_state']['base_quat']).to(device)

        base_pose = torch.cat([init_base_pos, init_base_quat]).unsqueeze(0)
        robot.write_root_pose_to_sim(base_pose)
        robot.write_joint_state_to_sim(
            init_joint_pos.unsqueeze(0),
            init_joint_vel.unsqueeze(0),
            joint_ids=torch.arange(29, device=device)
        )
        robot.update(dt=0.0)
        print(f"[INFO] Initialized robot to GT starting state")

    obs, _ = env.get_observations()


    hand_errors_left = []
    hand_errors_right = []

    for timestep in range(T):
        action_np = actions_precomputed[timestep]
        action_torch = torch.from_numpy(action_np).to(device)

        inject_policy_command(env, action_torch, device)

        if timestep % 50 == 0:
            robot = env.unwrapped.scene["robot"]
            current_joint_pos = robot.data.joint_pos[0, :29].cpu().numpy()

            gt_idx = min(timestep, len(gt_commands['left_hand_pos']) - 1)
            gt_left = gt_commands['left_hand_pos'][gt_idx]
            gt_right = gt_commands['right_hand_pos'][gt_idx]
            gt_joint_pos = gt_commands['joint_pos_traj'][gt_idx]

            pred_left = action_np[0:3]
            pred_right = action_np[3:6]

            left_error = np.linalg.norm(pred_left - gt_left)
            right_error = np.linalg.norm(pred_right - gt_right)
            joint_error = np.linalg.norm(current_joint_pos - gt_joint_pos)

            hand_errors_left.append(left_error)
            hand_errors_right.append(right_error)

            print(f"\n[Step {timestep}] OPEN-LOOP REPLAY")
            print(f"  Precomputed Left Hand:  {pred_left}")
            print(f"  GT Left Hand:           {gt_left}")
            print(f"  Hand Position Error:    {left_error:.4f}m")
            print(f"  Joint Position Error:   {joint_error:.4f} (divergence from GT state)")

        with torch.inference_mode():
            wbc_actions = policy(obs)
            obs, _, _, _ = env.step(wbc_actions)

        if timestep >= len(gt_commands['timestamp']) - 1:
            print(f"\n[INFO] Finished replaying all GT data")
            break

    print(f"\n[OPEN-LOOP REPLAY] Completed!")
    print(f"  Average left hand error:  {np.mean(hand_errors_left):.4f}m")
    print(f"  Final left hand error:    {hand_errors_left[-1]:.4f}m")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):

    task_name = args_cli.task.split(":")[-1]
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    print(f"\n{'='*80}")
    print(f"OPEN-LOOP EVALUATION: ACT Policy with Precomputed Actions")
    print(f"{'='*80}\n")

    print(f"[INFO] Loading GT trajectory for precomputation...")
    gt_commands = load_gt_trajectory(args_cli.traj_path, load_images=True)

    default_checkpoint = os.path.join(PROJECT_ROOT, "g1_wbc", "logs", "rsl_rl", "stable_eef", "for_door_hist_10.pt")
    default_checkpoint = os.path.abspath(default_checkpoint)

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif os.path.exists(default_checkpoint):
        resume_path = default_checkpoint
        print(f"[INFO] Using default WBC checkpoint: {resume_path}")
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading WBC policy from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "openloop_replay"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during replay.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading WBC tracking policy from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    from act_policy.utils import load_act_policy_and_normalizer
    act_checkpoint_path = Path(args_cli.act_checkpoint).expanduser()
    if not act_checkpoint_path.is_file():
        raise FileNotFoundError(f"ACT checkpoint not found: {act_checkpoint_path}")

    model, normalizer, config = load_act_policy_and_normalizer(str(act_checkpoint_path), str(env.unwrapped.device))

    obs_horizon = args_cli.obs_horizon or config.get('obs_horizon', 1)
    pred_horizon = config['pred_horizon']
    exec_horizon = args_cli.exec_horizon or 8
    use_image = config.get('use_image', False)
    dt = env.unwrapped.step_dt

    print(f"[INFO] Loaded ACT policy from: {act_checkpoint_path}")
    print(f"[INFO] ACT Configuration:")
    print(f"  - Timestep dt: {dt}s")
    print(f"  - Observation horizon: {obs_horizon}")
    print(f"  - Prediction horizon: {pred_horizon}")
    print(f"  - Execution horizon: {exec_horizon}")
    print(f"  - Using images: {use_image}")

    print(f"\n{'='*80}")
    print(f"PHASE 1: PRECOMPUTATION - Generate Actions from GT Observations")
    print(f"{'='*80}")


    actions_precomputed = precompute_actions(
        gt_commands=gt_commands,
        model=model,
        normalizer=normalizer,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        exec_horizon=exec_horizon,
        device=str(env.unwrapped.device),
        use_image=use_image
    )

    traj_dir = Path(args_cli.traj_path)
    save_path = traj_dir / "actions_precomputed_openloop.npy"
    np.save(save_path, actions_precomputed)
    print(f"\n[INFO] Saved precomputed actions to: {save_path}")

    print(f"\n{'='*80}")
    print(f"PHASE 2: OPEN-LOOP REPLAY - Execute Precomputed Actions in Simulation")
    print(f"{'='*80}")

    replay_openloop(
        env=env,
        policy=policy,
        actions_precomputed=actions_precomputed,
        gt_commands=gt_commands,
        device=str(env.unwrapped.device)
    )

    print(f"\n{'='*80}")
    print(f"OPEN-LOOP EVALUATION COMPLETED")
    print(f"{'='*80}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
