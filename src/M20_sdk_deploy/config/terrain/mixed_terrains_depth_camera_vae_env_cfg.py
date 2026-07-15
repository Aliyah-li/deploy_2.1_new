# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

import numpy as np
import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.utils import configclass

import active_suspension_mixed.tasks.suspension.basic_function.mdp as mdp
from active_suspension_mixed.tasks.suspension.depth_camera.depth_camera_env_cfg import DepthCameraObservationsCfg
from active_suspension_mixed.tasks.suspension.mixed_terrains.mixed_terrains_env_cfg import HfUnevenTerrainCfg
from active_suspension_mixed.tasks.suspension.mixed_terrains_depth_camera.mixed_terrains_depth_camera_env_cfg import (
    DeeproboticsM20MixedTerrainsDepthCameraEnvCfg,
)


class MixedTerrainsDepthCameraVaeRingTerrainGenerator(TerrainGenerator):
    """Arrange boxes/uneven/waves across the 8x8 grid so terrain changes along +X only."""

    def _generate_curriculum_terrains(self):
        sub_terrains_cfgs = self.cfg.sub_terrains
        boxes_cfg = sub_terrains_cfgs["boxes"]
        uneven_cfg = sub_terrains_cfgs["uneven"]
        waves_cfg = sub_terrains_cfgs["waves"]
        # random_rough_cfg = sub_terrains_cfgs["random_rough"]

        lower, upper = self.cfg.difficulty_range

        for sub_row in range(self.cfg.num_rows):
            difficulty = lower + (upper - lower) * ((sub_row + 0.5) / self.cfg.num_rows)
            tile_pattern = sub_row % 3
            if tile_pattern == 0:
                sub_cfg = boxes_cfg
            elif tile_pattern == 1:
                sub_cfg = uneven_cfg
            else:
                sub_cfg = waves_cfg

            for sub_col in range(self.cfg.num_cols):
                mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_cfg)


MIXED_TERRAINS_DEPTH_CAMERA_VAE_RING_CFG = TerrainGeneratorCfg(
    class_type=MixedTerrainsDepthCameraVaeRingTerrainGenerator,
    size=(8.0, 8.0),
    border_width=1.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=1.0,
        #     noise_range=(0.01, 0.12),
        #     noise_step=0.01,
        #     border_width=0.25,
        # ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.45,
            grid_height_range=(0.025, 0.16),
            platform_width=2.0,
            holes=False,
        ),
        "uneven": HfUnevenTerrainCfg(
            proportion=1.0,
            border_width=0.25,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
        ),
        "waves": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0,
            amplitude_range=(0.02, 0.12),
            num_waves=2,
            border_width=0.25,
        ),
    },
)


@configclass
class MixedTerrainsDepthCameraVaeObservationsCfg(DepthCameraObservationsCfg):
    """Depth-camera observations plus privileged velocity target for VAE supervision."""

    @configclass
    class VelocityTargetCfg(ObsGroup):
        current_lin_vel = ObsTerm(
            func=mdp.base_lin_vel_xy,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    velocity_target: VelocityTargetCfg = VelocityTargetCfg()


@configclass
class DeeproboticsM20MixedTerrainsDepthCameraVaeEnvCfg(DeeproboticsM20MixedTerrainsDepthCameraEnvCfg):
    """Mixed-terrains-line depth task whose actor also uses a VAE-estimated velocity latent."""

    observations: MixedTerrainsDepthCameraVaeObservationsCfg = MixedTerrainsDepthCameraVaeObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        if self.__class__.__name__ == "DeeproboticsM20MixedTerrainsDepthCameraVaeEnvCfg":
            self.scene.terrain.terrain_generator = MIXED_TERRAINS_DEPTH_CAMERA_VAE_RING_CFG
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
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.upside_down_l2 = RewTerm(
            func=mdp.upside_down_l2,
            weight=-5.0,
            params={"threshold": 0.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.world_lin_vel_y_l2.weight = 0.0
        self.terminations.line_terrain_boundary = None
        self.terminations.line_goal_reached = None
        self.terminations.upside_down = DoneTerm(
            func=mdp.upside_down,
            params={"threshold": 0.2, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.curriculum.command_levels = CurrTerm(
            func=mdp.command_levels_vel,
            params={
                "reward_term_name": "track_lin_vel_xy_exp",
                "range_multiplier": (0.2, 1.0),
            },
        )

        if self.__class__.__name__ == "DeeproboticsM20MixedTerrainsDepthCameraVaeEnvCfg":
            self.disable_zero_weight_rewards()
