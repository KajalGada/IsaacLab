# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the kinetic-sand scooping RL environment.

The environment is a standalone gymnasium.Env that uses Newton directly:
  - SolverMuJoCo  : UR5 arm articulation (PD joint control)
  - SolverImplicitMPM : kinetic-sand particle simulation

Action space  : joint-position deltas Δq ∈ [-1,1]^6, scaled to ±action_scale rad.
Observation   : joint_pos(6) | joint_vel(6) | eef_pos(3) | eef_quat_wxyz(4) |
                sand_on_scoop_norm(1) | sand_in_target_norm(1) | last_action(6)  → 27 dims.
"""

from __future__ import annotations

import math
import os

ASSETS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


class SandScoopEnvCfg:
    """All hyper-parameters for the sand scooping environment."""

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------
    episode_length_s: float = 20.0
    """Maximum episode duration in seconds."""

    sim_dt: float = 1.0 / 60.0
    """Newton simulation frame dt (seconds)."""

    mpm_substeps: int = 4
    """MPM solver substeps per simulation frame. Robot solver uses the same count."""

    policy_decimation: int = 3
    """Number of simulation frames per RL policy step."""

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------
    num_actions: int = 6
    """Action dimension: joint-position deltas for 6 UR5 revolute joints."""

    num_observations: int = 27
    """Observation dimension (see module docstring)."""

    # ------------------------------------------------------------------
    # Robot
    # ------------------------------------------------------------------
    urdf_path: str = os.path.join(ASSETS_DIR, "ur5_with_scoop.urdf")
    """Absolute path to the UR5-with-scoop URDF (mesh paths already patched)."""

    # Robot base pose (between the two sand containers).
    robot_base_pos: tuple[float, float, float] = (-0.45, 0.0, 0.0)

    # Home joint angles [shoulder_pan, lift, elbow, wrist1, wrist2, wrist3] in rad.
    # Arm poised above source container, scoop pointing down.
    home_q: tuple[float, ...] = (
        -0.27, -1.50, 1.70,
        math.pi / 2, math.pi / 2, math.pi / 2,
    )

    # Joint position limits [rad] – used for clamping targets and limit penalties.
    joint_lo: tuple[float, ...] = (-6.28,) * 6
    joint_hi: tuple[float, ...] = (6.28,) * 6

    # PD gains applied to every joint.
    joint_kp: float = 2000.0
    joint_kd: float = 100.0

    # ------------------------------------------------------------------
    # Containers
    # ------------------------------------------------------------------
    # World-frame centre of each container's bottom face.
    source_pos: tuple[float, float, float] = (0.05, 0.0, 0.0)
    target_pos: tuple[float, float, float] = (0.60, 0.0, 0.0)

    # Container dimensions (all in metres).
    box_w: float = 0.35
    box_d: float = 0.35
    box_h: float = 0.12
    wall_t: float = 0.02
    wall_mu: float = 0.6   # friction on container walls / bottom

    # ------------------------------------------------------------------
    # Newton MPM – kinetic sand material
    # ------------------------------------------------------------------
    # Particle grid emitted above the source container bottom.
    # Coordinates are relative to source_pos.
    mpm_emit_lo: tuple[float, float, float] = (-0.12, -0.12, 0.04)
    mpm_emit_hi: tuple[float, float, float] = (0.12,  0.12,  0.16)
    mpm_voxel_size: float = 0.06          # coarser grid ≈ 200-300 particles
    mpm_particles_per_cell: int = 2
    mpm_grid_type: str = "sparse"         # "sparse" or "fixed"

    # Kinetic-sand material parameters (tuned in simulation_newton_sand_v2.py).
    sand_density: float = 1100.0          # kg/m³
    sand_young_modulus: float = 1.5e5     # Pa
    sand_poisson_ratio: float = 0.25
    sand_friction: float = 1.2            # high inter-grain locking
    sand_damping: float = 800.0           # kills elastic rebound
    sand_yield_pressure: float = 300.0    # Pa
    sand_yield_stress: float = 150.0      # Pa
    sand_hardening: float = 5.0
    sand_air_drag: float = 1.0

    # Number of gravity-only frames to run at startup so particles settle.
    mpm_settle_frames: int = 180          # 3 s at 60 fps

    # ------------------------------------------------------------------
    # Scoop detection region
    # ------------------------------------------------------------------
    # The scoop bowl is approximated as a sphere in world space.
    # Its centre is computed as:  scoop_link_world_pos + scoop_bowl_z_offset * scoop_z_axis
    scoop_bowl_z_offset: float = 0.06    # m along the scoop link's local Z
    scoop_detect_radius: float = 0.07    # m

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    action_scale: float = 0.05           # max joint-angle change per policy step [rad]

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------
    success_fraction: float = 0.30       # fraction of total particles that must land in target

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    w_approach: float = 0.30
    w_scoop: float = 2.00
    w_transport: float = 0.30
    w_pour: float = 5.00
    w_success: float = 20.0
    w_action_penalty: float = -0.005
    w_joint_limit: float = -1.00
