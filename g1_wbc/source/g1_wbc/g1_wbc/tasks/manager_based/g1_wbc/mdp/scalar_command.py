"""Sub-module containing command generators for a 1-d scalar command."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands import UniformScalarCommandCfg


class UniformScalarCommand(CommandTerm):
    """Command generator for generating scalar commands uniformly.

    The generator samples a single scalar value uniformly within a specified range
    for each environment. It is designed to be general-purpose and can represent
    any scalar control input (e.g., gain, torque multiplier, or scale factor).
    """

    cfg: UniformScalarCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformScalarCommandCfg, env: ManagerBasedEnv):
        """Initialize the scalar command generator.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)

        # extract robot reference (optional: may not be used directly)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = None
        if cfg.body_name is not None:
            self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- command tensor: (num_envs, 1)
        self.scalar_command = torch.zeros(self.num_envs, 1, device=self.device)

        # -- metrics
        self.metrics["command_mean"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformScalarCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current scalar command tensor of shape (num_envs, 1)."""
        return self.scalar_command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        self.metrics["command_mean"] = self.scalar_command.mean(dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pcommand values uniformly within the specified range
        r = torch.empty(len(env_ids), device=self.device)
        self.scalar_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.value)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError("Debug visualization not implemented for UniformScalarCommand.")

    def _debug_vis_callback(self, event):
        raise NotImplementedError("Debug visualization callback not implemented for UniformScalarCommand.")
