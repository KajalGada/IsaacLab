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

    # Robot base pose — original position facing +X toward the containers.
    robot_base_pos: tuple[float, float, float] = (-0.45, 0.0, 0.0)
    robot_base_yaw: float = 0.0  # rotation around Z [rad]

    # Home joint angles — hovers scoop at approx (−0.002, +0.251, 0.251) above source container.
    # Found by FK search for robot at (−0.45, 0, 0), source_pos=(0.0, +0.25, 0.0).
    home_q: tuple[float, ...] = (
        0.2974, -0.9734, 1.3006,
        1.4318, math.pi / 2, math.pi / 2,
    )

    # "In-sand" joint angles — scoop at approx (−0.001, +0.252, 0.041), at sand surface.
    # Found by FK search for robot at (−0.45, 0, 0), source_pos=(0.0, +0.25, 0.0).
    in_sand_q: tuple[float, ...] = (
        0.2974, -0.7364, 1.4695,
        0.9487, math.pi / 2, math.pi / 2,
    )

    # Joint position limits [rad] – used for clamping targets and limit penalties.
    joint_lo: tuple[float, ...] = (-6.28,) * 6
    joint_hi: tuple[float, ...] = (6.28,) * 6

    # PD gains applied to every joint.
    joint_kp: float = 2000.0
    joint_kd: float = 100.0

    # MuJoCo solver CCD iterations. Default (35) produces warnings with the
    # UR5 mesh complexity; 100 silences them without a meaningful perf cost.
    robot_ccd_iterations: int = 100

    # ------------------------------------------------------------------
    # Containers
    # ------------------------------------------------------------------
    # World-frame centre of each container's bottom face.
    # Side-by-side in Y with a clear gap (~0.15 m) between walls.
    source_pos: tuple[float, float, float] = (0.0,  0.25, 0.0)
    target_pos: tuple[float, float, float] = (0.0, -0.25, 0.0)

    # Container dimensions (all in metres).
    box_w: float = 0.35
    box_d: float = 0.35
    box_h: float = 0.06
    # Must be >= 2 * mpm_voxel_size so the MPM SDF resolves the wall (needs ≥2 grid cells).
    wall_t: float = 0.04
    wall_mu: float = 0.6   # friction on container walls / bottom

    # ------------------------------------------------------------------
    # Newton MPM – kinetic sand material
    # ------------------------------------------------------------------
    # Particle grid emitted above the source container bottom.
    # Coordinates are relative to source_pos.
    mpm_emit_lo: tuple[float, float, float] = (-0.12, -0.12, 0.02)
    mpm_emit_hi: tuple[float, float, float] = (0.12,  0.12,  0.16)
    # Voxel size used for both particle emission spacing and the MPM solver grid.
    # Must satisfy: particle_radius ≈ 0.25 * mpm_voxel_size for stable MPM.
    # Must be <= wall_t so the MPM grid can resolve container walls.
    mpm_voxel_size: float = 0.02
    mpm_particles_per_cell: int = 2
    mpm_grid_type: str = "sparse"         # "sparse" or "fixed"

    # Kinetic-sand material parameters — matched to simulation_newton_sand_v1.py.
    sand_density: float = 1100.0          # kg/m³
    sand_young_modulus: float = 5e3       # Pa
    sand_poisson_ratio: float = 0.25
    sand_friction: float = 0.8
    sand_damping: float = 3000.0
    sand_yield_pressure: float = 50.0     # Pa
    sand_yield_stress: float = 25.0       # Pa
    sand_hardening: float = 0.5
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
