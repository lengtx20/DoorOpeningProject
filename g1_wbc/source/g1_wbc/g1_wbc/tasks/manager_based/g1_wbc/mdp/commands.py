from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformPoseCommandCfg, UniformVelocityCommandCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from .scalar_command import UniformScalarCommand


# class template
@configclass
class UniformScalarCommandCfg(CommandTermCfg):
    """Configuration for uniform scalar command generator.

    This generates a single scalar command sampled uniformly from a given range.
    It can represent any scalar quantity (e.g., height, gain, torque multiplier, scaling factor).
    """

    class_type: type = UniformScalarCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = None
    """(Optional) Name of the body in the asset associated with the scalar command."""

    @configclass
    class Ranges:
        """Uniform distribution range for the scalar command."""

        value: tuple[float, float] = MISSING
        """Range (min, max) for the scalar command value."""

    ranges: Ranges = MISSING
    """Ranges for the scalar command."""

    visualize: bool = False
    """Whether to visualize the scalar command (e.g., as text or bar).
    Defaults to False since scalar has no spatial meaning.
    """

    visualization_cfg: VisualizationMarkersCfg | None = None
    """Optional configuration for visualizing scalar command (e.g., as numeric label or gauge)."""


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


@configclass
class UniformHandPoseCommandCfg(UniformPoseCommandCfg):
    limit_ranges: UniformPoseCommandCfg.Ranges = MISSING


@configclass
class UniformHeightCommandCfg(UniformScalarCommandCfg):
    limit_ranges: UniformScalarCommandCfg.Ranges = MISSING


@configclass
class UniformBaseRPYCommandCfg(UniformPoseCommandCfg):
    limit_ranges: UniformPoseCommandCfg.Ranges = MISSING


@configclass
class UniformBasePitchCommandCfg(UniformScalarCommandCfg):
    limit_ranges: UniformScalarCommandCfg.Ranges = MISSING
