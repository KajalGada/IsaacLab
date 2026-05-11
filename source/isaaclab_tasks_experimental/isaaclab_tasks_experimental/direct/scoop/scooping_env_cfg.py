# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.physics.implicit_mpm_manager_cfg import ImplicitMPMSolverCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg


@configclass
class ScoopingPhysicsCfg(PresetCfg):
    """Physics backend presets for scooping without a robot.

    Default: Newton + implicit MPM. Use ``+physics=physx`` to switch to PhysX for comparison.
    """

    default: NewtonCfg = NewtonCfg(
        solver_cfg=ImplicitMPMSolverCfg(
            voxel_size=0.02,
            grid_type="sparse",
            max_iterations=250,
            tolerance=1.0e-4,
            transfer_scheme="apic",
            collider_velocity_mode="forward",
        ),
        num_substeps=4,
        debug_mode=False,
        use_cuda_graph=False,
    )
    physx: PhysxCfg = PhysxCfg()


@configclass
class ScoopingEnvCfg(DirectRLEnvCfg):
    """Minimal direct env config: Newton MPM sandbox, no articulation yet."""

    episode_length_s = 20.0
    decimation = 2

    # Placeholders — must match what ScoopingEnv returns / expects (adjust together).
    action_space = 1
    observation_space = 8
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, -10.0),
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physics=ScoopingPhysicsCfg(),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
