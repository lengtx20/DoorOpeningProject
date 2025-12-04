import argparse
import sys
from isaaclab.app import AppLauncher

# local imports
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Play Diffusion + RL Agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-Wbc-Play-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use pretrained checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
parser.add_argument("--diffusion_model", type=str, required=True, help="Path to diffusion ckpt")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time
import math
import copy

import gymnasium as gym
import torch
import torch.nn as nn

# Isaac Lab & RSL-RL imports
import g1_wbc.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

# ---------------- Diffusion Models (保持不变) ---------------- #

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class TinyVisionEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, embedding_dim)
    def forward(self, images):
        B, T, C, H, W = images.shape
        feat = self.conv(images.view(B * T, C, H, W))
        return self.proj(feat.view(B * T, -1)).view(B, T, -1).flatten(1)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(1, out_ch), nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(1, out_ch), nn.ReLU(),
        )
        self.cond = nn.Linear(cond_dim, out_ch * 2)
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, cond):
        scale, shift = self.cond(cond).unsqueeze(-1).chunk(2, dim=1)
        return self.net(x) * (1 + scale) + shift + self.res_conv(x)

class SimpleUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, hidden=128):
        super().__init__()
        self.mid = nn.Conv1d(input_dim, hidden, 1)
        self.res1 = ConditionalResidualBlock1D(hidden, hidden, global_cond_dim)
        self.res2 = ConditionalResidualBlock1D(hidden, hidden, global_cond_dim)
        self.final = nn.Conv1d(hidden, input_dim, 1)
    def forward(self, sample, t, global_cond):
        x = sample.transpose(1, 2)
        x = self.mid(x)
        x = self.res2(self.res1(x, global_cond), global_cond)
        return self.final(x).transpose(1, 2)

class MinimalPolicy(nn.Module):
    def __init__(self, action_dim=10, obs_horizon=2, pred_horizon=16):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.vision = TinyVisionEncoder(embedding_dim=64)
        self.timestep_proj = SinusoidalPosEmb(64)
        self.unet = SimpleUnet1D(input_dim=action_dim, global_cond_dim=(obs_horizon * 64) + 64)

    def forward(self, image):
        B = image.shape[0]
        vision_feat = self.vision(image)
        t = torch.zeros((B,), dtype=torch.long, device=image.device)
        t_emb = self.timestep_proj(t.float())
        sample = torch.randn((B, self.pred_horizon, 10), device=image.device)
        return self.unet(sample, t, torch.cat([vision_feat, t_emb], dim=-1))

# ---------------- Integrated Policy Logic ---------------- #

class IntegratedDiffusionPolicy:
    def __init__(self, rl_policy, diffusion_path, device):
        self.rl_policy = rl_policy
        self.device = device
        
        # Load Diffusion Model
        self.diffusion = MinimalPolicy()
        try:
            # 兼容不同的 checkpoint 格式
            ckpt = torch.load(diffusion_path, map_location=device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt['state_dict']
            self.diffusion.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"[ERROR] Failed to load diffusion checkpoint: {e}")
        
        self.diffusion.to(device).eval()
        self.camera_size = 64 * 64 * 3

    def __call__(self, obs):
        # 假设 Obs 结构: [Camera(12288) | RL_Obs(N)]
        # 1. 提取 Camera 数据
        raw_cam = obs[:, :self.camera_size].float()
        B = raw_cam.shape[0]

        # (B, H*W*C) -> (B, H, W, C) -> (B, C, H, W)
        img = raw_cam.view(B, 64, 64, 3).permute(0, 3, 1, 2)
        
        # 2. 构造 Temporal Horizon (复制当前帧)
        img_input = img.unsqueeze(1).repeat(1, 2, 1, 1, 1)

        # 3. Diffusion 推理
        with torch.no_grad():
            diffusion_out = self.diffusion(img_input)[:, 0, :]

        # 4. 解析指令
        left_hand = diffusion_out[:, 0:3]
        right_hand = diffusion_out[:, 3:6]
        base_vel = diffusion_out[:, 6:9]
        # grasp = diffusion_out[:, 9]

        # 5. 准备给 RL 的观测数据
        # 必须 clone，否则 modify 会影响下一帧或者原来的 buffer
        obs_rl = obs[:, self.camera_size:].clone()
        
        # 6. 覆盖高层指令
        # 注意：这里的索引偏移量需要严格根据 G1 任务的 Observation 定义
        # 假设原代码中的 offset 是正确的：
        # Base Vel (3) -> Left Hand (3) -> Right Hand (3)
        obs_rl[:, 3+3          : 3+3+3]               = base_vel
        obs_rl[:, 3+3+3+1+1    : 3+3+3+1+1+3]         = left_hand
        obs_rl[:, 3+3+3+1+1+3+4: 3+3+3+1+1+3+4+3]     = right_hand
        
        # 7. 调用 RL Policy (输入纯 RL obs)
        return self.rl_policy(obs_rl)

# ---------------- Helper for Loading RL Logic ---------------- #

class RLPolicyView:
    """
    这是一个代理类。
    它的作用是把一个包含 Camera 的大环境，伪装成一个不包含 Camera 的小环境。
    目的是为了让 RSL-RL 的 OnPolicyRunner 能够正常初始化并加载权重，
    因为它会检查 obs 的维度。
    """
    def __init__(self, real_env, camera_size):
        self.real_env = real_env
        self.camera_size = camera_size
        
        # 伪装关键属性
        self.num_envs = real_env.num_envs
        self.num_actions = real_env.num_actions
        self.device = real_env.device
        self.max_episode_length = real_env.max_episode_length
        
        # 这一步是关键：告诉 Runner 观测维度比实际的小
        # real_env.num_obs 包含 camera，我们需要减去它
        self.num_obs = real_env.num_obs - camera_size
        
        print(f"[INFO] RL View Created: Real Obs {real_env.num_obs} -> Fake Obs {self.num_obs}")

    def get_observations(self):
        # 如果 Runner 初始化时调用了这个，我们要返回切片后的数据
        obs, extras = self.real_env.get_observations()
        # 假设 Camera 在前，RL 在后
        return obs[:, self.camera_size:], extras

    @property
    def unwrapped(self):
        return self.real_env.unwrapped

# ---------------- Main Execution ---------------- #

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 1. Config Setup (Same as play.py)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 2. Checkpoint Logic
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # 3. Create Real Environment ONCE (With Camera)
    # 按照 play.py 的方式初始化，包括 Video Wrapper
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_diffusion"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 4. Load RL Policy
    # 我们不创建第二个环境，而是创建一个"视图"传给 Runner
    camera_size = 64 * 64 * 3
    env_proxy = RLPolicyView(env, camera_size)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # 使用 proxy 来欺骗 Runner 关于 obs 形状的信息
    ppo_runner = OnPolicyRunner(env_proxy, agent_cfg.to_dict(), log_dir=None, device=env.device)
    ppo_runner.load(resume_path)
    
    # 获取纯 RL 的 inference policy
    rl_policy_fn = ppo_runner.get_inference_policy(device=env.device)

    # 5. Integrate Diffusion + RL
    # 这里传入原始的 env（包含完整 obs），wrapper 内部会自己负责切分
    policy = IntegratedDiffusionPolicy(rl_policy_fn, retrieve_file_path(args_cli.diffusion_model), env.device)

    # 6. Play Loop (Strictly following play.py logic)
    obs, _ = env.get_observations()
    print("[INFO] Starting play...")
    print(f"[DEBUG] Full Observation shape: {obs.shape}")

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        
        with torch.inference_mode():
            # Integrated Policy 处理 Camera -> Diffusion -> Cmd -> RL
            actions = policy(obs)
            # Step 环境
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        
        # Real-time control
        if args_cli.real_time:
            # simple sleep
            time.sleep(0.01)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()