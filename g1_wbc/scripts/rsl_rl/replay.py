
import argparse
import sys
from pathlib import Path
from isaaclab.app import AppLauncher
import cli_args  
parser = argparse.ArgumentParser(description="Replay with ground truth commands, BC policy, or ACT policy.")
parser.add_argument(
    "--mode",
    type=str,
    default="gt",
    choices=["gt", "bc", "act", "diffusion"],
    help="Replay mode: 'gt' (ground truth), 'bc' (behavioral cloning), 'act' (action chunking transformer), 'diffusion' (diffusion policy)"
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during replay.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-Wbc-Play-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--traj_path",
    type=str,
    default=None,
    help="Path to trajectory directory with log_dict.npy (required for gt mode)"
)
parser.add_argument(
    "--bc_checkpoint",
    type=str,
    default=None,
    help="Path to BC policy checkpoint (required for bc mode)"
)
parser.add_argument(
    "--act_checkpoint",
    type=str,
    default=None,
    help="Path to ACT policy checkpoint (required for act mode)"
)
parser.add_argument(
    "--diffusion_checkpoint",
    type=str,
    default=None,
    help="Path to Diffusion policy checkpoint (required for diffusion mode)"
)
parser.add_argument("--obs_horizon", type=int, default=2, help="Observation history length for BC")
parser.add_argument("--exec_horizon", type=int, default=2, help="Action execution horizon for BC")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.mode == "gt" and not args_cli.traj_path:
    parser.error("--traj_path is required for gt mode")
if args_cli.mode == "bc" and not args_cli.bc_checkpoint:
    parser.error("--bc_checkpoint is required for bc mode")
if args_cli.mode == "act" and not args_cli.act_checkpoint:
    parser.error("--act_checkpoint is required for act mode")
if args_cli.mode == "diffusion" and not args_cli.diffusion_checkpoint:
    parser.error("--diffusion_checkpoint is required for diffusion mode")
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
bc_root = PROJECT_ROOT / "bc_policy"
act_root = PROJECT_ROOT / "act_policy"
diffusion_root = PROJECT_ROOT / "diffusion_policy"
if str(bc_root) not in sys.path:
    sys.path.insert(0, str(bc_root))
if str(act_root) not in sys.path:
    sys.path.insert(0, str(act_root))
if str(diffusion_root) not in sys.path:
    sys.path.insert(0, str(diffusion_root))

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

    # Compute stage from frame_idx (never use precomputed stage)
    from data.dataset import compute_stage_from_frame_idx
    if 'frame_idx' in log_dict:
        frame_indices = log_dict['frame_idx']
        stage = np.array([compute_stage_from_frame_idx(int(idx)) for idx in frame_indices], dtype=np.int32)
        gt_commands['stage'] = stage
        print(f"[INFO] Computed stage from frame_idx")
    else:
        # Fallback: assume sequential frames
        stage = np.array([compute_stage_from_frame_idx(t) for t in range(T)], dtype=np.int32)
        gt_commands['stage'] = stage
        print(f"[WARNING] frame_idx not found, using fallback")

    # Load images if requested
    if load_images:
        traj_dir = Path(traj_path)
        image_files = sorted(traj_dir.glob("*.npy"))
        # Filter out log_dict.npy
        image_files = [f for f in image_files if f.name != "log_dict.npy"]
        if image_files:
            gt_commands['image_path'] = traj_path
            gt_commands['num_images'] = len(image_files)
            print(f"[INFO] Found {len(image_files)} image files for real image feeding")
        else:
            print(f"[WARNING] No image files found in {traj_path}")

    # Extract initial state from timestep 0
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

    print(f"[INFO] Loaded {T} timesteps of GT commands")
    print(f"[INFO] Duration: {gt_commands['timestamp'][-1] - gt_commands['timestamp'][0]:.2f} seconds")
    print(f"[INFO] Extracted initial state from timestep 0")
    return gt_commands

def load_real_image(traj_path: str, timestep: int, device: str = "cuda:0") -> torch.Tensor:
    """Load real image from trajectory directory and preprocess for model input."""
    image_file = Path(traj_path) / f"{timestep:06d}.npy"
    if not image_file.exists():
        return None

    # Load image [H, W, 3] uint8
    image_np = np.load(image_file)

    # Convert to torch tensor and normalize to [0, 1]
    image_tensor = torch.from_numpy(image_np).float() / 255.0

    # Convert from [H, W, 3] to [1, 3, H, W]
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

    # Resize to 224x224 for model input
    import torchvision.transforms as T
    transform = T.Resize((224, 224))
    image_tensor = transform(image_tensor)

    return image_tensor.to(device)
def inject_gt_command(env, step_idx: int, gt_commands: dict, device):
    T = len(gt_commands['timestamp'])
    idx = min(step_idx, T - 1)
    cmd_manager = env.unwrapped.command_manager
    left_hand_term = cmd_manager.get_term("target_left_hand_pos_in_base")
    left_hand_cmd = torch.from_numpy(gt_commands['left_hand_pos'][idx]).float().to(device)
    left_hand_term.command[:, :3] = left_hand_cmd.unsqueeze(0)
    right_hand_term = cmd_manager.get_term("target_right_hand_pos_in_base")
    right_hand_cmd = torch.from_numpy(gt_commands['right_hand_pos'][idx]).float().to(device)
    right_hand_term.command[:, :3] = right_hand_cmd.unsqueeze(0)
    base_vel_term = cmd_manager.get_term("target_base_velocity")
    base_vel_xy = torch.from_numpy(gt_commands['base_vel_xy'][idx]).float().to(device)
    yaw_speed = torch.tensor([gt_commands['yaw_speed'][idx]], dtype=torch.float32, device=device)
    base_vel_cmd = torch.cat([base_vel_xy, yaw_speed])
    base_vel_term.command[:, :3] = base_vel_cmd.unsqueeze(0)
    return idx
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

def stage_to_onehot(stage_label: int, num_stages: int = 5) -> torch.Tensor:
    """Convert stage label to one-hot encoding."""
    onehot = torch.zeros(num_stages)
    if 0 <= stage_label < num_stages:
        onehot[stage_label] = 1.0
    return onehot
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    gt_commands = None
    if args_cli.mode == "gt":
        gt_commands = load_gt_trajectory(args_cli.traj_path, load_images=False)
    elif args_cli.traj_path:
        # Load GT trajectory for stage/image information even in policy modes
        # Load images if ACT or Diffusion mode uses vision
        load_images = args_cli.mode in ["act", "diffusion"]
        gt_commands = load_gt_trajectory(args_cli.traj_path, load_images=load_images)
        print(f"[INFO] Loaded GT trajectory for stage/image injection")
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
            "video_folder": os.path.join(log_dir, "videos", "replay"),
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
    bc_controller = None
    act_controller = None
    diffusion_controller = None

    if args_cli.mode == "bc":
        from bc_policy.utils import BCController, load_bc_policy_and_normalizer
        bc_checkpoint_path = Path(args_cli.bc_checkpoint).expanduser()
        if not bc_checkpoint_path.is_file():
            raise FileNotFoundError(f"BC checkpoint not found: {bc_checkpoint_path}")
        model, normalizer, config = load_bc_policy_and_normalizer(str(bc_checkpoint_path), str(env.unwrapped.device))
        bc_controller = BCController(
            model=model,
            normalizer=normalizer,
            obs_horizon=args_cli.obs_horizon or config.get('obs_horizon', 2),
            pred_horizon=config['pred_horizon'],
            exec_horizon=args_cli.exec_horizon,
            device=str(env.unwrapped.device),
        )
        print(f"[INFO] Loaded BC policy from: {bc_checkpoint_path}")

    elif args_cli.mode == "act":
        from act_policy.utils import ACTController, load_act_policy_and_normalizer
        act_checkpoint_path = Path(args_cli.act_checkpoint).expanduser()
        if not act_checkpoint_path.is_file():
            raise FileNotFoundError(f"ACT checkpoint not found: {act_checkpoint_path}")
        model, normalizer, config = load_act_policy_and_normalizer(str(act_checkpoint_path), str(env.unwrapped.device))
        # Get dt from environment for timestamp tracking
        dt = env.unwrapped.step_dt
        act_controller = ACTController(
            model=model,
            normalizer=normalizer,
            obs_horizon=args_cli.obs_horizon or config.get('obs_horizon', 1),
            pred_horizon=config['pred_horizon'],
            exec_horizon=args_cli.exec_horizon or 8,  # Default to 8 for ACT
            device=str(env.unwrapped.device),
            dt=dt,  # Pass timestep duration for timestamp tracking
            use_image=config.get('use_image', False),
        )
        print(f"[INFO] Loaded ACT policy from: {act_checkpoint_path}")
        print(f"[INFO] Timestep dt: {dt}s")
        print(f"[INFO] Using images: {config.get('use_image', False)}")

    elif args_cli.mode == "diffusion":
        from diffusion_policy.utils import DiffusionController, load_diffusion_policy_and_normalizer
        diffusion_checkpoint_path = Path(args_cli.diffusion_checkpoint).expanduser()
        if not diffusion_checkpoint_path.is_file():
            raise FileNotFoundError(f"Diffusion checkpoint not found: {diffusion_checkpoint_path}")
        model, normalizer, config = load_diffusion_policy_and_normalizer(str(diffusion_checkpoint_path), str(env.unwrapped.device))
        # Get dt from environment for timestamp tracking
        dt = env.unwrapped.step_dt
        diffusion_controller = DiffusionController(
            model=model,
            normalizer=normalizer,
            obs_horizon=args_cli.obs_horizon or config.get('obs_horizon', 2),
            pred_horizon=config['pred_horizon'],
            action_horizon=args_cli.exec_horizon or config.get('action_horizon', 8),
            device=str(env.unwrapped.device),
            dt=dt,  # Pass timestep duration for timestamp tracking
            use_image=config.get('use_image', False),
        )
        print(f"[INFO] Loaded Diffusion policy from: {diffusion_checkpoint_path}")
        print(f"[INFO] Timestep dt: {dt}s")
        print(f"[INFO] Using images: {config.get('use_image', False)}")
    dt = env.unwrapped.step_dt

    # Set robot to EXACT GT initial state (all aspects) if GT trajectory is loaded
    if gt_commands and 'initial_state' in gt_commands:
        robot = env.unwrapped.scene["robot"]
        device = env.unwrapped.device

        # Convert to torch tensors
        init_joint_pos = torch.from_numpy(gt_commands['initial_state']['joint_pos']).to(device)
        init_joint_vel = torch.from_numpy(gt_commands['initial_state']['joint_vel']).to(device)
        init_base_pos = torch.from_numpy(gt_commands['initial_state']['base_pos']).to(device)
        init_base_quat = torch.from_numpy(gt_commands['initial_state']['base_quat']).to(device)

        # Prepare poses [num_envs, 7] where 7 = [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        base_pose = torch.cat([init_base_pos, init_base_quat]).unsqueeze(0)

        # Write to simulation (overwrites default initialization)
        robot.write_root_pose_to_sim(base_pose)
        robot.write_joint_state_to_sim(
            init_joint_pos.unsqueeze(0),
            init_joint_vel.unsqueeze(0),
            joint_ids=torch.arange(29, device=device)
        )

        # Important: reset buffers to sync with new state
        robot.update(dt=0.0)

        print(f"[INFO] Initialized robot to GT starting pose (EXACT MATCH)")
        print(f"[INFO]   Base: {init_base_pos.cpu().numpy()}, quat: {init_base_quat.cpu().numpy()}")
        print(f"[INFO]   Joints[0:5]: {init_joint_pos[:5].cpu().numpy()}")

    obs, _ = env.get_observations()
    timestep = 0
    print(f"\n[INFO] Starting replay in {args_cli.mode} mode...")
    if args_cli.mode == "gt":
        print(f"[INFO] Replaying {len(gt_commands['timestamp'])} timesteps")
    while simulation_app.is_running():
        start_time = time.time()
        if args_cli.mode == "gt":
            cmd_idx = inject_gt_command(env, timestep, gt_commands, env.unwrapped.device)

            # Verification at timestep 0
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

            if timestep % 100 == 0:
                print(f"[Step {timestep}] GT Mode - Command {cmd_idx}/{len(gt_commands['timestamp'])}")
                print(f"  Injected GT left hand: {gt_commands['left_hand_pos'][cmd_idx]}")
                print(f"  Injected GT right hand: {gt_commands['right_hand_pos'][cmd_idx]}")
                print(f"  Injected GT base velocity: xy={gt_commands['base_vel_xy'][cmd_idx]}, yaw={gt_commands['yaw_speed'][cmd_idx]:.3f}")
                if 'stage' in gt_commands:
                    print(f"  GT stage: {gt_commands['stage'][cmd_idx]}")

        elif args_cli.mode == "bc":
            robot = env.unwrapped.scene["robot"]
            joint_pos = robot.data.joint_pos[:, :29]
            bc_action = bc_controller.get_action(joint_pos)
            inject_policy_command(env, bc_action, env.unwrapped.device)

            # Verification at timestep 0
            if timestep == 0 and gt_commands and 'initial_state' in gt_commands:
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
                stats = bc_controller.get_statistics()
                print(f"[Step {timestep}] BC Policy")
                print(f"  Injected BC action: {bc_action.cpu().numpy()}")
                print(f"  Buffer: {stats['action_idx']}/{stats['action_buffer_len']}")
                if gt_commands is not None:
                    gt_idx = min(timestep, len(gt_commands['timestamp']) - 1)
                    print(f"  GT left hand: {gt_commands['left_hand_pos'][gt_idx]}")
                    print(f"  GT right hand: {gt_commands['right_hand_pos'][gt_idx]}")
                    print(f"  GT velocity: xy={gt_commands['base_vel_xy'][gt_idx]}, yaw={gt_commands['yaw_speed'][gt_idx]:.3f}")
                    if 'stage' in gt_commands:
                        print(f"  GT stage: {gt_commands['stage'][gt_idx]}")

        elif args_cli.mode == "act":
            robot = env.unwrapped.scene["robot"]
            joint_pos = robot.data.joint_pos[:, :29]
            joint_vel = robot.data.joint_vel[:, :29]

            # Get image - use real images from trajectory if available, otherwise sim camera
            image = None
            if act_controller.use_image:
                if gt_commands is not None and 'image_path' in gt_commands:
                    # Load real image from trajectory
                    image = load_real_image(gt_commands['image_path'], timestep, str(env.unwrapped.device))
                    if image is None and timestep % 50 == 0:
                        print(f"[WARNING] Could not load real image for timestep {timestep}, using sim camera")

                # Fall back to sim camera if real image not available
                if image is None:
                    camera = env.unwrapped.scene["camera"]
                    rgb_data = camera.data.output["rgb"]
                    import torchvision.transforms as T
                    transform = T.Compose([T.Resize((224, 224))])
                    image = rgb_data.permute(0, 3, 1, 2).float() / 255.0
                    image = transform(image)

            # Get stage from GT if available
            stage = None
            stage_label = None
            if gt_commands is not None and 'stage' in gt_commands:
                stage_idx = min(timestep, len(gt_commands['stage']) - 1)
                stage_label = int(gt_commands['stage'][stage_idx])
                stage = stage_to_onehot(stage_label)

            act_action = act_controller.get_action(joint_pos, joint_vel, image, stage)
            inject_policy_command(env, act_action, env.unwrapped.device)

            # Verification at timestep 0
            if timestep == 0 and gt_commands and 'initial_state' in gt_commands:
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
                img_source = "Real" if (gt_commands and 'image_path' in gt_commands) else "Sim"
                print(f"[Step {timestep}] ACT Policy [Image: {img_source}]")
                print(f"  Injected ACT action: {act_action.cpu().numpy()}")
                print(f"  Buffer: {stats['action_idx']}/{stats['action_buffer_len']} (replan every {act_controller.exec_horizon} steps)")
                print(f"  ACT stage: {stats['current_stage']} (frame={act_controller.frame_idx})")
                if gt_commands is not None:
                    gt_idx = min(timestep, len(gt_commands['timestamp']) - 1)
                    print(f"  GT left hand: {gt_commands['left_hand_pos'][gt_idx]}")
                    print(f"  GT right hand: {gt_commands['right_hand_pos'][gt_idx]}")
                    print(f"  GT velocity: xy={gt_commands['base_vel_xy'][gt_idx]}, yaw={gt_commands['yaw_speed'][gt_idx]:.3f}")
                    if 'stage' in gt_commands:
                        print(f"  GT stage: {gt_commands['stage'][gt_idx]}")

        elif args_cli.mode == "diffusion":
            robot = env.unwrapped.scene["robot"]
            joint_pos = robot.data.joint_pos[:, :29]

            # Get image - use real images from trajectory if available, otherwise sim camera
            image = None
            if diffusion_controller.use_image:
                if gt_commands is not None and 'image_path' in gt_commands:
                    # Load real image from trajectory
                    image = load_real_image(gt_commands['image_path'], timestep, str(env.unwrapped.device))
                    if image is None and timestep % 50 == 0:
                        print(f"[WARNING] Could not load real image for timestep {timestep}, using sim camera")

                # Fall back to sim camera if real image not available
                if image is None:
                    camera = env.unwrapped.scene["camera"]
                    rgb_data = camera.data.output["rgb"]
                    import torchvision.transforms as T
                    transform = T.Compose([T.Resize((224, 224))])
                    image = rgb_data.permute(0, 3, 1, 2).float() / 255.0
                    image = transform(image)

            # Get stage from GT if available
            stage = None
            stage_label = None
            if gt_commands is not None and 'stage' in gt_commands:
                stage_idx = min(timestep, len(gt_commands['stage']) - 1)
                stage_label = int(gt_commands['stage'][stage_idx])
                stage = stage_to_onehot(stage_label)

            diffusion_action = diffusion_controller.get_action(joint_pos, image, stage)
            inject_policy_command(env, diffusion_action, env.unwrapped.device)

            # Verification at timestep 0
            if timestep == 0 and gt_commands and 'initial_state' in gt_commands:
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
                stats = diffusion_controller.get_statistics()
                img_source = "Real" if (gt_commands and 'image_path' in gt_commands) else "Sim"
                print(f"[Step {timestep}] Diffusion Policy [Image: {img_source}]")
                print(f"  Injected Diffusion action: {diffusion_action.cpu().numpy()}")
                print(f"  Buffer: {stats['action_idx']}/{stats['action_buffer_len']} (replan every {diffusion_controller.action_horizon} steps)")
                print(f"  Diffusion stage: {diffusion_controller.current_stage.argmax()} (t={diffusion_controller.timestamp:.2f}s)")
                if gt_commands is not None:
                    gt_idx = min(timestep, len(gt_commands['timestamp']) - 1)
                    print(f"  GT left hand: {gt_commands['left_hand_pos'][gt_idx]}")
                    print(f"  GT right hand: {gt_commands['right_hand_pos'][gt_idx]}")
                    print(f"  GT velocity: xy={gt_commands['base_vel_xy'][gt_idx]}, yaw={gt_commands['yaw_speed'][gt_idx]:.3f}")
                    if 'stage' in gt_commands:
                        print(f"  GT stage: {gt_commands['stage'][gt_idx]}")
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        timestep += 1
        if args_cli.mode == "gt" and timestep >= len(gt_commands['timestamp']):
            print(f"[INFO] Finished replaying all GT data")
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