# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for the G1 WBC environment."""

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from pxr import UsdPhysics, PhysxSchema


def reset_object_root_state_uniform_wrt_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("door"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This is a custom version that can optionally position objects relative to the robot.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]

    # positions = root_states[:, 0:3] + env.scene["robot"]._data._root_link_pose_w.data[env_ids, :3] + rand_samples[:, 0:3]

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def disable_frame_collision(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    body_names: list[str] = ["frame", "Frame", "door_frame", "base_link"],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("door"),
):
    """Disable collision on the frame link of the door.

    This function disables physics collision on the frame body of the door asset.
    It tries multiple common frame link names to ensure compatibility.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply the changes to. Can be None for startup.
        body_names: List of potential frame body names to try disabling.
        asset_cfg: The asset configuration for the door.
    """
    from pxr import Usd
    import omni.usd

    # extract the asset
    asset: Articulation = env.scene[asset_cfg.name]

    # Get the stage
    stage = omni.usd.get_context().get_stage()

    # If env_ids is None (startup event), apply to all environments
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Try to disable collision for each potential frame body name
    for env_id in env_ids:
        for body_name in body_names:
            frame_prim_path = f"{asset.cfg.prim_path.replace('{ENV_REGEX_NS}', f'/World/envs/env_{env_id}')}/{body_name}"
            frame_prim = stage.GetPrimAtPath(frame_prim_path)

            if frame_prim.IsValid():
                # Disable collision by modifying the collision API
                collision_api = UsdPhysics.CollisionAPI.Get(stage, frame_prim_path)
                if collision_api:
                    collision_api.GetCollisionEnabledAttr().Set(False)
                    print(f"[INFO] Disabled collision on {frame_prim_path}")
                    break  # Found and disabled, move to next env_id
