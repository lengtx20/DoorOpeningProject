import argparse
import sys
import os
import torch
import time
from pathlib import Path


from isaaclab.app import AppLauncher
import cli_args 

parser = argparse.ArgumentParser(description="Play Hierarchical Agent.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="G1-Wbc-Play-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False)


cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

script_path = Path(__file__).resolve()
g1_wbc_root = script_path.parents[2] 
project_root = g1_wbc_root.parent


source_path = g1_wbc_root / "source"
if str(source_path) not in sys.path:
    sys.path.insert(0, str(source_path))


if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gymnasium as gym
import g1_wbc.tasks
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.math import quat_rotate_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner


from g1_wbc.tasks.manager_based.g1_wbc.agents.diffusion_controller import HierarchicalController


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed if agent_cfg.seed is not None else 42
    
    env_cfg.commands.target_base_velocity.resampling_time_range = (1e9, 1e9)
    env_cfg.commands.target_left_hand_pos_in_base.resampling_time_range = (1e9, 1e9)
    env_cfg.commands.target_right_hand_pos_in_base.resampling_time_range = (1e9, 1e9)

    diff_output_dir = project_root / "diffusion_policy/scripts/outputs/door_opening_20251129_193711"
    diff_model_path = diff_output_dir / "best_model.pt"
    diff_norm_path = diff_output_dir / "normalizer.npz"
    
    if args_cli.checkpoint:
        rl_ckpt_path = Path(args_cli.checkpoint)
    else:
        rl_ckpt_path = g1_wbc_root / "logs/rsl_rl/stable_eef/model_43400.pt"

    print(f"[INFO] Diffusion Model: {diff_model_path}")
    print(f"[INFO] RL Policy: {rl_ckpt_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(str(rl_ckpt_path))
    rl_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    controller = HierarchicalController(
        env,
        model_path=str(diff_model_path), 
        normalizer_path=str(diff_norm_path),
        device="cuda"
    )

    unwrapped_env = env.unwrapped
    robot = unwrapped_env.scene["robot"]
    
    l_hand_idx = robot.find_bodies("left_wrist_yaw_link")[0][0]
    r_hand_idx = robot.find_bodies("right_wrist_yaw_link")[0][0]
    
    all_joint_names = robot.joint_names
    has_gripper = False
    l_grip_idxs = None

    gripper_matches = [name for name in all_joint_names if "left_gripper" in name]
    
    if len(gripper_matches) > 0:
        l_grip_idxs, _ = robot.find_joints("left_gripper_.*")
        has_gripper = True
        print(f"[INFO] Gripper joints found: {gripper_matches}")
    else:
        print("[INFO] No gripper joints found in robot asset. Grasp logic will be skipped.")


    obs, _ = env.get_observations()
    dt = unwrapped_env.step_dt
    
    print("[INFO] Starting Hierarchical Control Loop...")
    
    while simulation_app.is_running():
        start_time = time.time()
        
        base_pos = robot.data.root_pos_w[:, :3]
        base_quat = robot.data.root_quat_w[:, :]
        
        l_hand_pos_w = robot.data.body_pos_w[:, l_hand_idx, :]
        r_hand_pos_w = robot.data.body_pos_w[:, r_hand_idx, :]
        
        l_rel = l_hand_pos_w - base_pos
        r_rel = r_hand_pos_w - base_pos
        l_hand_torso = quat_rotate_inverse(base_quat, l_rel)
        r_hand_torso = quat_rotate_inverse(base_quat, r_rel)
        
        base_lin_vel = robot.data.root_lin_vel_b[:, :3]
        base_ang_vel = robot.data.root_ang_vel_b[:, :3]
        
        formatted_vel = torch.cat([base_lin_vel[:, 0:2], base_ang_vel[:, 2:3]], dim=1)
        
        if has_gripper:
            curr_finger_pos = robot.data.joint_pos[:, l_grip_idxs]
            avg_finger_pos = torch.mean(curr_finger_pos, dim=1, keepdim=True)
            is_grasping = (avg_finger_pos < 0.005).float()
        else:
            is_grasping = torch.zeros((robot.data.root_pos_w.shape[0], 1), device=robot.device)

        state_dict = {
            'l_hand': l_hand_torso[0],
            'r_hand': r_hand_torso[0],
            'vel': formatted_vel[0],
            'grasp': is_grasping[0]
        }
        
        controller.step(state_dict)

        with torch.inference_mode():
            actions = rl_policy(obs)
            obs, _, _, _ = env.step(actions)
            
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()