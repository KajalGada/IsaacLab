# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.physics.implicit_mpm_manager_cfg import ImplicitMPMSolverCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class ScoopingPhysicsCfg(PresetCfg):
    """Physics backend presets for scooping task."""

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
class ScoopingContainerCfg:
    """Container geometry and collision settings."""

    width: float = 0.35
    depth: float = 0.35
    height: float = 0.08
    wall_thickness: float = 0.04
    wall_overlap: float = 0.01
    friction: float = 0.6
    gap: float = 0.01


@configclass
class ScoopingEmitterCfg:
    """Particle emitter settings."""

    emit_lo: tuple[float, float, float] = (-0.13, -0.13, 0.055)
    emit_hi: tuple[float, float, float] = (0.13, 0.13, 0.15)
    particles_per_cell: int = 3
    initial_jitter: float = 0.10
    density: float = 1100.0


@configclass
class ScoopingMaterialCfg:
    """Per-particle MPM material settings."""

    young_modulus: float = 5.0e3
    poisson_ratio: float = 0.25
    friction: float = 0.8
    damping: float = 3000.0
    yield_pressure: float = 50.0
    tensile_yield_ratio: float = 0.0
    yield_stress: float = 25.0
    hardening: float = 0.5
    dilatancy: float = 0.0
    viscosity: float = 0.0


@configclass
class ScoopingEnvCfg(DirectRLEnvCfg):
    """Direct RL config for scooping sandbox."""

    episode_length_s = 20.0
    decimation = 2

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

    container: ScoopingContainerCfg = ScoopingContainerCfg()
    emitter: ScoopingEmitterCfg = ScoopingEmitterCfg()
    material: ScoopingMaterialCfg = ScoopingMaterialCfg()

    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
