from __future__ import annotations

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import active_suspension_mixed.tasks.suspension.basic_function.mdp as mdp
from active_suspension_mixed.tasks.suspension.basic_function.basic_function_env_cfg import ObservationsCfg
from active_suspension_mixed.tasks.suspension.mixed_terrains.mixed_terrains_env_cfg import (
    DeeproboticsM20MixedTerrainsLineEnvCfg,
    line_terrain_boundary,
)
from active_suspension_mixed.tasks.suspension.mixed_terrains_depth_camera_vae.mixed_terrains_depth_camera_vae_env_cfg import (
    MIXED_TERRAINS_DEPTH_CAMERA_VAE_RING_CFG,
)


@configclass
class MixedTerrainsHistoryVaeObservationsCfg:
    @configclass
    class HistoryCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        current_yaw = ObsTerm(func=mdp.current_yaw, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-math.pi, math.pi), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class VelocityTargetCfg(ObsGroup):
        current_planar_velocity = ObsTerm(
            func=mdp.base_velocity_xy_yaw,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


@configclass
class DeeproboticsM20MixedTerrainsHistoryVaeObservationsCfg(ObservationsCfg):
    history: MixedTerrainsHistoryVaeObservationsCfg.HistoryCfg = MixedTerrainsHistoryVaeObservationsCfg.HistoryCfg()
    velocity_target: MixedTerrainsHistoryVaeObservationsCfg.VelocityTargetCfg = (
        MixedTerrainsHistoryVaeObservationsCfg.VelocityTargetCfg()
    )


@configclass
class DeeproboticsM20MixedTerrainsHistoryVaeEnvCfg(DeeproboticsM20MixedTerrainsLineEnvCfg):
    """Mixed-terrains line task with policy-history VAE supervision and no depth-camera branch."""

    observations: DeeproboticsM20MixedTerrainsHistoryVaeObservationsCfg = DeeproboticsM20MixedTerrainsHistoryVaeObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.scene.terrain.terrain_generator = MIXED_TERRAINS_DEPTH_CAMERA_VAE_RING_CFG
        self.observations.policy.height_scan = None
        self.commands.base_velocity = mdp.UniformThresholdVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 10.0),
            rel_standing_envs=0.02,
            rel_heading_envs=0.0,
            heading_command=False,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=None,
            ),
        )
        self.events.randomize_reset_base.func = mdp.reset_root_state_uniform_randomized_terrain_type
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-3.2, -2.8),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.upside_down_l2 = RewTerm(
            func=mdp.upside_down_l2,
            weight=-5.0,
            params={"threshold": 0.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.world_lin_vel_y_l2 = RewTerm(
            func=mdp.world_lin_vel_y_l2,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.terrain_out_of_bounds = None
        self.terminations.line_terrain_boundary = DoneTerm(
            func=line_terrain_boundary,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "x_half_width": MIXED_TERRAINS_DEPTH_CAMERA_VAE_RING_CFG.size[0]
                * MIXED_TERRAINS_DEPTH_CAMERA_VAE_RING_CFG.num_rows
                * 0.5,
                "y_half_width": MIXED_TERRAINS_DEPTH_CAMERA_VAE_RING_CFG.size[1]
                * MIXED_TERRAINS_DEPTH_CAMERA_VAE_RING_CFG.num_cols
                * 0.5,
                "boundary_buffer": 0.0,
            },
        )
        self.terminations.line_goal_reached = None
        self.terminations.upside_down = DoneTerm(
            func=mdp.upside_down,
            params={"threshold": 0.2, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.curriculum.command_levels = CurrTerm(
            func=mdp.command_levels_vel,
            params={"reward_term_name": "track_lin_vel_xy_exp", "range_multiplier": (0.2, 1.0)},
        )
        self.disable_zero_weight_rewards()
