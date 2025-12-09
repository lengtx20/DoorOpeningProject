
import argparse
import sys
from pathlib import Path
from isaaclab.app import AppLauncher
import cli_args
parser = argparse.ArgumentParser(description="Experiment 6: Inject ALL ACT predictions (raw, no smoothing) to tracking policy.")
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
    help="Path to trajectory directory with log_dict.npy (required for GT images)"
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

    from data.dataset import compute_stage_from_frame_idx
    if 'frame_idx' in log_dict:
        frame_indices = np.array(log_dict['frame_idx'], dtype=np.int32)
        gt_commands['frame_idx'] = frame_indices
        stage = np.array([compute_stage_from_frame_idx(int(idx)) for idx in frame_indices], dtype=np.int32)
        gt_commands['stage'] = stage
        print(f"[INFO] Computed stage from frame_idx")
        print(f"[INFO] Frame index range: {frame_indices[0]} to {frame_indices[-1]}")
    else:
        frame_indices = np.arange(T, dtype=np.int32)
        gt_commands['frame_idx'] = frame_indices
        stage = np.array([compute_stage_from_frame_idx(t) for t in range(T)], dtype=np.int32)
        gt_commands['stage'] = stage
        print(f"[WARNING] frame_idx not found, using fallback sequential indices")

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

    initial_joint_pos = np.array([log_dict[f'q_{i}'][0] for i in range(29)], dtype=np.float32)
    initial_joint_vel = np.array([log_dict[f'dq_{i}'][0] for i in range(29)], dtype=np.float32)
    initial_base_pos = np.array([
        log_dict['pos_x'][0],
        log_dict['pos_y'][0],
        log_dict['pos_z'][0]
    ], dtype=np.float32)
    initial_base_quat = np.array([
        log_dict['quat_x'][0],
        log_dict['quat_y'][0],
        log_dict['quat_z'][0],
        log_dict['quat_w'][0]
    ], dtype=np.float32)

    gt_commands['initial_state'] = {
        'joint_pos': initial_joint_pos,
        'joint_vel': initial_joint_vel,
        'base_pos': initial_base_pos,
        'base_quat': initial_base_quat,
    }

    print(f"[INFO] Loaded {T} timesteps of GT data")
    print(f"[INFO] Duration: {gt_commands['timestamp'][-1] - gt_commands['timestamp'][0]:.2f} seconds")
    print(f"[INFO] Extracted initial state from timestep 0")
    print(f"[INFO]   Base pos: {initial_base_pos}")
    print(f"[INFO]   Base quat: {initial_base_quat}")
    print(f"[INFO]   Joint pos[0:5]: {initial_joint_pos[0:5]}")
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


def inject_raw_act_command(env, act_action: torch.Tensor, device):
    action = act_action[0:9]

    cmd_manager = env.unwrapped.command_manager

    left_hand_term = cmd_manager.get_term("target_left_hand_pos_in_base")
    left_hand_term.command[:, :3] = action[0:3].unsqueeze(0)

    right_hand_term = cmd_manager.get_term("target_right_hand_pos_in_base")
    right_hand_term.command[:, :3] = action[3:6].unsqueeze(0)

    base_vel_term = cmd_manager.get_term("target_base_velocity")
    base_vel_term.command[:, :3] = action[6:9].unsqueeze(0)

    return action.cpu().numpy()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    gt_commands = load_gt_trajectory(args_cli.traj_path, load_images=True)

    default_checkpoint = os.path.join(PROJECT_ROOT, "g1_wbc", "logs", "rsl_rl", "stable_eef", "for_door_hist_10.pt")
    print(f"[INFO] Default checkpoint: {default_checkpoint}")
    default_checkpoint = os.path.abspath(default_checkpoint)

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif os.path.exists(default_checkpoint):
        resume_path = default_checkpoint
        print(f"[INFO] Using default checkpoint: {resume_path}")
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "exp6_inject_pred_all"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during replay.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    from act_policy.utils import ACTController, load_act_policy_and_normalizer
    act_checkpoint_path = Path(args_cli.act_checkpoint).expanduser()
    if not act_checkpoint_path.is_file():
        raise FileNotFoundError(f"ACT checkpoint not found: {act_checkpoint_path}")

    model, normalizer, config = load_act_policy_and_normalizer(str(act_checkpoint_path), str(env.unwrapped.device))

    dt = env.unwrapped.step_dt
    act_controller = ACTController(
        model=model,
        normalizer=normalizer,
        obs_horizon=args_cli.obs_horizon or config.get('obs_horizon', 1),
        pred_horizon=config['pred_horizon'],
        exec_horizon=args_cli.exec_horizon or 8,
        device=str(env.unwrapped.device),
        dt=dt,
        use_image=config.get('use_image', False),
    )
    print(f"[INFO] Loaded ACT policy from: {act_checkpoint_path}")
    print(f"[INFO] Timestep dt: {dt}s")
    print(f"[INFO] Using images: {config.get('use_image', False)}")
    print(f"[INFO] Mode: Experiment 6 - Inject ALL ACT predictions (raw, no smoothing)")

    if gt_commands and 'initial_state' in gt_commands:
        robot = env.unwrapped.scene["robot"]
        device = env.unwrapped.device

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

        print(f"[INFO] Initialized robot to GT starting pose (EXACT MATCH)")
        print(f"[INFO]   Base: {init_base_pos.cpu().numpy()}, quat: {init_base_quat.cpu().numpy()}")
        print(f"[INFO]   Joints[0:5]: {init_joint_pos[:5].cpu().numpy()}")

    obs, _ = env.get_observations()
    timestep = 0

    print(f"\n[INFO] Starting Experiment 6: All ACT Predictions (Raw)")
    print(f"[INFO] Replaying {len(gt_commands['timestamp'])} timesteps")
    print(f"[INFO] ACT receives: Sim state (robot.data), GT images, calculated stage")
    print(f"[INFO] Tracking policy receives: ALL ACT predictions (hand positions + velocities, raw)")

    while simulation_app.is_running():
        start_time = time.time()

        robot = env.unwrapped.scene["robot"]
        joint_pos = robot.data.joint_pos[:, :29]
        joint_vel = robot.data.joint_vel[:, :29]

        gt_idx = min(timestep, len(gt_commands['timestamp']) - 1)
        actual_frame_idx = int(gt_commands['frame_idx'][gt_idx]) if 'frame_idx' in gt_commands else timestep

        image = None
        if act_controller.use_image:
            if 'image_path' in gt_commands:
                image = load_real_image(gt_commands['image_path'], actual_frame_idx, str(env.unwrapped.device))
                if image is None and timestep % 50 == 0:
                    print(f"[WARNING] Could not load real image for frame_idx {actual_frame_idx}, using sim camera")

            if image is None:
                camera = env.unwrapped.scene["camera"]
                rgb_data = camera.data.output["rgb"]
                import torchvision.transforms as T
                transform = T.Compose([T.Resize((224, 224))])
                image = rgb_data.permute(0, 3, 1, 2).float() / 255.0
                image = transform(image)

        from data.dataset import compute_stage_from_frame_idx
        stage_label = compute_stage_from_frame_idx(actual_frame_idx)
        stage_encoding = torch.zeros(5, dtype=torch.float32)
        stage_encoding[stage_label] = 1.0

        act_action = act_controller.get_action(joint_pos, joint_vel, image, stage_encoding)

        action = inject_raw_act_command(env, act_action, env.unwrapped.device)

        if timestep == 0 and 'initial_state' in gt_commands:
            robot = env.unwrapped.scene["robot"]
            actual_joint_pos = robot.data.joint_pos[0, :29].cpu().numpy()
            gt_joint_pos = gt_commands['initial_state']['joint_pos']

            joint_diff = np.abs(actual_joint_pos - gt_joint_pos)
            print(f"\n[VERIFICATION] Joint position error at t=0:")
            print(f"  Max error: {joint_diff.max():.6f}")
            print(f"  Mean error: {joint_diff.mean():.6f}")
            print(f"  Joints[0:5] actual: {actual_joint_pos[0:5]}")
            print(f"  Joints[0:5] GT: {gt_joint_pos[0:5]}")

            if joint_diff.max() > 0.01:
                print(f"[WARNING] Large initialization error! Robot may not match GT trajectory")

        if timestep % 50 == 0:
            stats = act_controller.get_statistics()
            img_source = "Real" if 'image_path' in gt_commands else "Sim"
            gt_stage = gt_commands['stage'][gt_idx] if 'stage' in gt_commands else None
            print(f"\n[Step {timestep}] Experiment 6: All ACT Predictions (Raw)")
            print(f"  Injected ACT left hand: {action[0:3]}")
            print(f"  Injected ACT right hand: {action[3:6]}")
            print(f"  Injected ACT velocity: {action[6:9]}")
            print(f"  GT left hand: {gt_commands['left_hand_pos'][gt_idx]}")
            print(f"  GT right hand: {gt_commands['right_hand_pos'][gt_idx]}")
            print(f"  GT velocity: xy={gt_commands['base_vel_xy'][gt_idx]}, yaw={gt_commands['yaw_speed'][gt_idx]:.3f}")
            print(f"  ACT stage: {stage_label} (actual frame_idx={actual_frame_idx})")
            if gt_stage is not None:
                print(f"  GT stage: {gt_stage}")
            print(f"  Image source: {img_source}")
            print(f"  Buffer: {stats['action_idx']}/{stats['action_buffer_len']}")

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        timestep += 1

        if timestep >= len(gt_commands['timestamp']):
            print(f"\n[INFO] Finished replaying all GT data")
            break

        if args_cli.video and timestep >= args_cli.video_length:
            print(f"[INFO] Video length reached")
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
