
import argparse
import sys
from pathlib import Path
from isaaclab.app import AppLauncher
import cli_args  
parser = argparse.ArgumentParser(description="Replay with ground truth commands or BC policy.")
parser.add_argument(
    "--mode",
    type=str,
    default="gt",
    choices=["gt", "bc"],
    help="Replay mode: 'gt' (ground truth commands from demo), 'bc' (behavioral cloning policy)"
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
parser.add_argument("--obs_horizon", type=int, default=2, help="Observation history length for BC")
parser.add_argument("--exec_horizon", type=int, default=8, help="Action execution horizon for BC")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.mode == "gt" and not args_cli.traj_path:
    parser.error("--traj_path is required for gt mode")
if args_cli.mode == "bc" and not args_cli.bc_checkpoint:
    parser.error("--bc_checkpoint is required for bc mode")
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
if args_cli.mode == "bc":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    bc_root = PROJECT_ROOT / "bc_policy"
    if str(bc_root) not in sys.path:
        sys.path.insert(0, str(bc_root))
def load_gt_trajectory(traj_path: str):
    log_path = Path(traj_path) / "log_dict.npy"
    if not log_path.exists():
        raise FileNotFoundError(f"GT trajectory file not found: {log_path}")
    print(f"[INFO] Loading GT trajectory from: {log_path}")
    log_dict = np.load(log_path, allow_pickle=True).item()
    T = len(log_dict['timestamp'])
    gt_commands = {
        'left_hand_pos': np.stack([
            log_dict['left_wrist_torso_pos_x'],
            log_dict['left_wrist_torso_pos_y'],
            log_dict['left_wrist_torso_pos_z']
        ], axis=1),
        'right_hand_pos': np.stack([
            log_dict['right_wrist_torso_pos_x'],
            log_dict['right_wrist_torso_pos_y'],
            log_dict['right_wrist_torso_pos_z']
        ], axis=1),
        'base_vel_xy': np.stack([
            log_dict['vel_body_x'],
            log_dict['vel_body_y']
        ], axis=1),
        'yaw_speed': np.array(log_dict['yaw_speed']),
        'timestamp': np.array(log_dict['timestamp']),
    }
    print(f"[INFO] Loaded {T} timesteps of GT commands")
    print(f"[INFO] Duration: {gt_commands['timestamp'][-1] - gt_commands['timestamp'][0]:.2f} seconds")
    return gt_commands
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
        gt_commands = load_gt_trajectory(args_cli.traj_path)
    default_checkpoint = os.path.join("g1_wbc", "logs", "model_43400.pt")
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
    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()
    timestep = 0
    print(f"\n[INFO] Starting replay in {args_cli.mode} mode...")
    if args_cli.mode == "gt":
        print(f"[INFO] Replaying {len(gt_commands['timestamp'])} timesteps")
    while simulation_app.is_running():
        start_time = time.time()
        if args_cli.mode == "gt":
            cmd_idx = inject_gt_command(env, timestep, gt_commands, env.unwrapped.device)
            if timestep % 100 == 0:
                print(f"[Step {timestep}] GT command {cmd_idx}/{len(gt_commands['timestamp'])}")
                print(f"  L_hand: {gt_commands['left_hand_pos'][cmd_idx]}")
                print(f"  R_hand: {gt_commands['right_hand_pos'][cmd_idx]}")
                print(f"  Base: {gt_commands['base_vel_xy'][cmd_idx]}, yaw: {gt_commands['yaw_speed'][cmd_idx]:.3f}")
        elif args_cli.mode == "bc":
            robot = env.unwrapped.scene["robot"]
            joint_pos = robot.data.joint_pos[:, :29]  
            bc_action = bc_controller.get_action(joint_pos)
            inject_policy_command(env, bc_action, env.unwrapped.device)
            if timestep % 50 == 0:
                stats = bc_controller.get_statistics()
                print(f"[Step {timestep}] BC action: {bc_action.cpu().numpy()}")
                print(f"  Buffer: {stats['action_idx']}/{stats['action_buffer_len']}")
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
