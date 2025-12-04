
import argparse
import sys
import os
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Play a BC Policy checkpoint in Isaac Sim.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=200, help="Length of video (steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="G1-Wbc-Play-v0", help="Task name.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to BC policy checkpoint (default: bc_policy/checkpoints/best.pth).",
)
parser.add_argument("--policy_device", type=str, default="cuda:0", help="Device for BC Policy inference")
parser.add_argument("--obs_horizon", type=int, default=2, help="Observation history length")
parser.add_argument("--exec_horizon", type=int, default=8, help="Action execution horizon (how many steps before replanning)")
parser.add_argument(
    "--low_level_checkpoint",
    type=str,
    default=None,
    help="Path to low-level WBC policy checkpoint (default: g1_wbc/logs/rsl_rl/stable_eef/exported/policy.pt)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BC_CKPT_DIR = PROJECT_ROOT / "bc_policy" / "checkpoints"
DEFAULT_BC_CKPT = BC_CKPT_DIR / "best.pth"
WBC_CKPT_DIR = PROJECT_ROOT / "g1_wbc" / "logs" / "rsl_rl" / "stable_eef" / "exported"
DEFAULT_WBC_CKPT = WBC_CKPT_DIR / "policy.pt"
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import time
import torch
import numpy as np
import gymnasium as gym
from collections import deque
import g1_wbc.tasks
import isaaclab_tasks
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
bc_root = PROJECT_ROOT / "bc_policy"
if str(bc_root) not in sys.path:
    sys.path.insert(0, str(bc_root))
from bc_policy.utils import BCController, load_bc_policy_and_normalizer
def _resolve_checkpoint_path(cli_checkpoint: str | None, default_path: Path) -> Path:
    if cli_checkpoint:
        path = Path(cli_checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    elif default_path.is_file():
        return default_path
    else:
        raise FileNotFoundError(
            f"No checkpoint specified and default not found at {default_path}. "
            f"Use --checkpoint to specify a path."
        )
def load_low_level_policy(checkpoint_path: str, num_obs: int, num_actions: int, device: torch.device):
    print(f"[INFO] Loading low-level policy from: {checkpoint_path}")
    try:
        policy = torch.jit.load(checkpoint_path, map_location=device)
        policy.eval()
        print(f"[INFO] Loaded as TorchScript model")
    except Exception as e:
        print(f"[INFO] TorchScript load failed, trying state dict: {e}")
        from rsl_rl.modules import ActorCritic
        policy = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_obs,
            num_actions=num_actions,
        ).to(device)
        loaded_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy.load_state_dict(loaded_dict['model_state_dict'])
        policy.eval()
        print(f"[INFO] Loaded from state dict")
    print(f"[INFO] Low-level policy ready with {num_obs} obs dims and {num_actions} action dims")
    return policy
def extract_proprio_from_obs(obs_dict: dict, num_joint_pos: int = 29) -> torch.Tensor:
    if 'policy' in obs_dict:
        policy_obs = obs_dict['policy']
        if isinstance(policy_obs, dict):
            if 'joint_pos' in policy_obs:
                joint_pos = policy_obs['joint_pos']
            elif 'dof_pos' in policy_obs:
                joint_pos = policy_obs['dof_pos']
            else:
                obs_list = []
                for key, val in policy_obs.items():
                    if key != 'camera' and isinstance(val, torch.Tensor):
                        obs_list.append(val)
                if obs_list:
                    joint_pos = torch.cat(obs_list, dim=-1)[:, :num_joint_pos]
                else:
                    raise KeyError(f"Could not find joint positions in policy observations. Keys: {list(policy_obs.keys())}")
        else:
            joint_pos = policy_obs[:, :num_joint_pos]
    elif 'joint_pos' in obs_dict:
        joint_pos = obs_dict['joint_pos']
    else:
        raise KeyError(f"Could not extract proprioception from observations. Keys: {list(obs_dict.keys())}")
    if joint_pos.shape[-1] != num_joint_pos:
        print(f"[WARNING] Expected {num_joint_pos} joint positions, got {joint_pos.shape[-1]}. Truncating/padding.")
        if joint_pos.shape[-1] > num_joint_pos:
            joint_pos = joint_pos[:, :num_joint_pos]
        else:
            padding = torch.zeros(*joint_pos.shape[:-1], num_joint_pos - joint_pos.shape[-1], device=joint_pos.device)
            joint_pos = torch.cat([joint_pos, padding], dim=-1)
    return joint_pos
def decompose_bc_action_to_commands(action: torch.Tensor) -> dict:
    if action.ndim == 1:
        action = action.unsqueeze(0)  
    commands = {
        'left_hand_pos': action[:, 0:3],      
        'right_hand_pos': action[:, 3:6],     
        'base_vel_xy': action[:, 6:8],        
        'yaw_speed': action[:, 8:9],          
        'grasp': action[:, 9:10],             
    }
    return commands
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    print(f"[INFO] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("videos", "bc_policy", args_cli.task),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos to: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env)
    device = torch.device(args_cli.policy_device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    checkpoint_path = _resolve_checkpoint_path(args_cli.checkpoint, DEFAULT_BC_CKPT)
    model, normalizer, config = load_bc_policy_and_normalizer(str(checkpoint_path), str(device))
    obs_horizon = args_cli.obs_horizon if args_cli.obs_horizon else config['obs_horizon']
    pred_horizon = config['pred_horizon']
    exec_horizon = args_cli.exec_horizon
    bc_controller = BCController(
        model=model,
        normalizer=normalizer,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        exec_horizon=exec_horizon,
        device=str(device),
    )
    wbc_ckpt_path = Path(args_cli.low_level_checkpoint).expanduser() if args_cli.low_level_checkpoint else DEFAULT_WBC_CKPT
    if not wbc_ckpt_path.is_file():
        raise FileNotFoundError(
            f"Low-level policy checkpoint not found at '{wbc_ckpt_path}'. "
            f"Provide --low_level_checkpoint explicitly or place weights under '{WBC_CKPT_DIR}'."
        )
    obs_space = env.unwrapped.single_observation_space
    if isinstance(obs_space, gym.spaces.Dict):
        policy_obs_space = obs_space['policy']
        if isinstance(policy_obs_space, gym.spaces.Dict):
            num_obs = sum(
                space.shape[0] if hasattr(space, 'shape') else 1
                for key, space in policy_obs_space.spaces.items()
                if key != 'camera'
            )
        else:
            num_obs = policy_obs_space.shape[0]
    else:
        num_obs = obs_space.shape[0]
    num_actions = env.unwrapped.action_space.shape[0]
    print(f"[INFO] Environment dimensions: obs={num_obs}, actions={num_actions}")
    low_level_policy = load_low_level_policy(str(wbc_ckpt_path), num_obs, num_actions, device)
    obs, _ = env.get_observations()
    print(f"[INFO] Starting simulation...")
    print(f"[INFO] Hierarchical control:")
    print(f"  - BC Policy: Generates high-level commands every {exec_horizon} steps")
    print(f"  - WBC Policy: Tracks commands at every step")
    timestep = 0
    dt = env.unwrapped.step_dt
    replan_counter = 0
    bc_action_history = []
    proprio_history = []
    while simulation_app.is_running():
        try:
            robot = env.unwrapped.scene["robot"]
            joint_pos = robot.data.joint_pos  
            proprio = joint_pos[:, :config['proprio_dim']]  
        except Exception as e:
            print(f"[ERROR] Failed to extract proprioception: {e}")
            import traceback
            traceback.print_exc()
            break
        bc_action = bc_controller.get_action(proprio)  
        if timestep % 10 == 0:
            stats = bc_controller.get_statistics()
            print(f"[Step {timestep}] BC Controller Stats:")
            print(f"  Action: {bc_action.numpy()}")
            print(f"  Buffer idx: {stats['action_idx']}/{stats['action_buffer_len']}")
        with torch.inference_mode():
            joint_actions = low_level_policy(obs)
        step_result = env.step(joint_actions)
        if len(step_result) == 5:
            obs, _, terminated, truncated, _ = step_result
        else:
            obs, _, done, _ = step_result
            terminated = done
            truncated = torch.zeros_like(done)
        timestep += 1
        if terminated.any() or truncated.any():
            print(f"[INFO] Episode terminated/truncated at step {timestep}")
            obs, _ = env.get_observations()
            bc_controller.reset()
            timestep = 0
        if args_cli.video and timestep >= args_cli.video_length:
            print(f"[INFO] Video length reached ({args_cli.video_length} steps)")
            break
        if args_cli.real_time:
            time.sleep(dt)
    env.close()
    simulation_app.close()
    print("[INFO] Simulation closed")
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
