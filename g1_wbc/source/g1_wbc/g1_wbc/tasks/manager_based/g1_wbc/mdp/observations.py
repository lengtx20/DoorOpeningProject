from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv



def last_processed_action(env: ManagerBasedEnv) -> torch.Tensor:
    """Returns the last action processed by the action manager."""
    return env.action_manager.action

def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def hand_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_pitch_link"]),
    reference_frame: str = "base",
) -> torch.Tensor:
    """Compute hand positions relative to base or world frame."""
    asset: Articulation = env.scene[asset_cfg.name]

    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w

    # --- ensure body ids tensor ---
    if len(asset_cfg.body_ids) != 1:
        raise NotImplementedError("Only single body id is supported.")

    body_ids = torch.as_tensor(asset_cfg.body_ids, device=base_pos_w.device, dtype=torch.long)

    hand_pos_w = asset.data.body_pos_w[:, body_ids, :]  # (num_envs, 1, 3)

    base_pos_w = base_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
    base_quat_w = base_quat_w.unsqueeze(1)  # (num_envs, 1, 4)

    hand_pos_b = math_utils.quat_apply_inverse(base_quat_w, hand_pos_w - base_pos_w)

    if reference_frame == "world":
        return hand_pos_w.reshape(env.num_envs, -1)
    elif reference_frame == "base":
        return hand_pos_b.reshape(env.num_envs, -1)
    else:
        raise ValueError(f"Invalid reference_frame '{reference_frame}'. Must be 'base' or 'world'.")
