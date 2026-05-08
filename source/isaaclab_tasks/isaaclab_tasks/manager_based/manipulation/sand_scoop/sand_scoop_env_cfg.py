# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclass for the Newton MPM kinetic-sand scooping environment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SandScoopEnvCfg:
    """Configuration for the kinetic-sand scooping RL environment.

    The environment uses Newton ``SolverImplicitMPM`` for particle dynamics and
    ``SolverMuJoCo`` for UR5 arm rigid-body dynamics.  Both solvers are run
    each frame: the robot solver advances joint positions under PD control,
    then the MPM solver resolves particle-collider interactions.

    Actions are 6-D joint-position deltas (one per UR5 DOF), scaled by
    ``action_scale``.  Observations are a 20-D vector: joint positions, joint
    velocities, EE position, sand-particle centroid, mean particle z, and the
    fraction of particles above the ``scoop_z_threshold``.

    The kinetic-sand material parameters are tuned from
    ``simulation_newton_sand_v1.py`` to produce sticky, mouldable behaviour:
    high friction, heavy damping, and low yield thresholds.
    """

    # ------------------------------------------------------------------
    # Asset paths
    # ------------------------------------------------------------------
    urdf_path: str = "/home/gmr/Downloads/NewtonSimulation/ur_urdf/ur5_with_scoop.urdf"

    # ------------------------------------------------------------------
    # Robot base pose  (orientation is always π around Z so the arm faces origin)
    # ------------------------------------------------------------------
    robot_base_pos: tuple[float, float, float] = (0.5, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Simulation timing
    # ------------------------------------------------------------------
    fps: float = 60.0
    """Frames (render ticks) per second.  One RL step = one frame."""

    sim_substeps: int = 4
    """MuJoCo substeps per frame; MPM runs once per frame at ``frame_dt``."""

    episode_length_s: float = 15.0
    """Maximum episode duration [s]."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -10.0)
    """Gravity vector [m/s²]."""

    # ------------------------------------------------------------------
    # PD joint-position control gains (applied to all UR5 joints)
    # ------------------------------------------------------------------
    pd_stiffness: float = 2000.0
    """Position stiffness [N·m/rad]."""

    pd_damping: float = 100.0
    """Velocity damping [N·m·s/rad]."""

    # ------------------------------------------------------------------
    # Container box geometry  (all shapes are static, body=-1)
    # ------------------------------------------------------------------
    box_width: float = 0.35
    """Box x-extent [m]."""

    box_depth: float = 0.35
    """Box y-extent [m]."""

    box_height: float = 0.05
    """Wall height above ground [m]."""

    wall_thickness: float = 0.02
    """Wall/floor thickness [m].  Must be ≥ ``voxel_size`` so the MPM grid
    can resolve the wall geometry."""

    box_mu: float = 0.6
    """Friction coefficient of box surfaces."""

    box_gap: float = 0.01
    """MPM contact gap [m].  Must be ≥ particle radius (~0.01 m at
    ``voxel_size`` = 0.06)."""

    # ------------------------------------------------------------------
    # Sand particle spawn volume
    # ------------------------------------------------------------------
    emit_lo: tuple[float, float, float] = (-0.15, -0.15, 0.02)
    """Lower corner of the initial sand volume [m]."""

    emit_hi: tuple[float, float, float] = (0.15, 0.15, 0.20)
    """Upper corner of the initial sand volume [m].  z capped at 0.20 so
    no robot link overlaps the spawn cloud at startup."""

    particles_per_cell: int = 1
    """Particles per MPM grid cell.  Reduce for faster training (fewer
    particles); increase for more physical accuracy."""

    initial_jitter: float = 0.5
    """Fraction of cell half-size used to jitter initial particle positions."""

    # ------------------------------------------------------------------
    # Kinetic-sand material parameters
    # Tuned from simulation_newton_sand_v1.py to reproduce sticky,
    # mouldable kinetic-sand behaviour.
    # ------------------------------------------------------------------
    density: float = 1100.0
    """Particle bulk density [kg/m³]."""

    young_modulus: float = 5e3
    """Young's modulus [Pa].  Low value → soft, deformable."""

    poisson_ratio: float = 0.25
    """Poisson's ratio (dimensionless)."""

    friction: float = 0.8
    """Inter-particle friction coefficient.  High value → grains lock
    together under compression but release under shear."""

    damping: float = 3000.0
    """Viscous damping coefficient.  Heavy damping kills elastic rebound
    so the material flows rather than bouncing."""

    yield_pressure: float = 50.0
    """Compressive yield threshold [Pa].  Low → skeleton yields at small
    loads, giving soft mouldable feel."""

    tensile_yield_ratio: float = 0.0
    """Tensile yield stress as a fraction of ``yield_pressure``."""

    yield_stress: float = 25.0
    """Shear yield stress [Pa]."""

    hardening: float = 0.5
    """Isotropic hardening coefficient (dimensionless)."""

    air_drag: float = 1.0
    """Air-drag coefficient applied to particle velocities."""

    critical_fraction: float = 0.0
    """Critical volume fraction for the density correction."""

    # ------------------------------------------------------------------
    # MPM solver options
    # ------------------------------------------------------------------
    voxel_size: float = 0.06
    """MPM grid cell size [m].  Larger → fewer cells, faster but less
    accurate.  Increase to 0.06 for RL training speed."""

    mpm_grid_type: str = "sparse"
    """Grid type: ``"sparse"`` | ``"fixed"`` | ``"dense"``.
    ``"fixed"`` enables CUDA graph capture for the MPM step."""

    mpm_max_iterations: int = 250
    """Maximum number of solver iterations per MPM step."""

    mpm_tolerance: float = 1e-6
    """Convergence tolerance for the MPM solver."""

    mpm_transfer_scheme: str = "apic"
    """Particle-grid transfer scheme: ``"apic"`` | ``"pic"``."""

    mpm_solver: str = "gauss-seidel"
    """Iterative solver variant:
    ``"gauss-seidel"`` | ``"jacobi"`` | ``"cg"`` | ``"cg+jacobi"`` |
    ``"cg+gauss-seidel"``."""

    mpm_strain_basis: str = "P0"
    """Strain interpolation basis."""

    mpm_collider_basis: str = "Q1"
    """Collider interpolation basis."""

    mpm_grid_padding: int = 0
    """Extra grid padding cells around the particle cloud."""

    mpm_max_active_cell_count: int = -1
    """Upper bound on active sparse cells; -1 = unlimited."""

    # ------------------------------------------------------------------
    # RL hyper-parameters
    # ------------------------------------------------------------------
    action_scale: float = 0.05
    """Joint-position delta scale [rad/step].  Raw action ∈ [-1, 1] is
    multiplied by this before adding to the current joint target."""

    joint_pos_limit: tuple[float, float] = (-3.14, 3.14)
    """Joint position clip range [rad] applied after adding the delta."""

    scoop_z_threshold: float = 0.12
    """Z height above which a particle is considered 'scooped' [m]."""

    rew_height_delta: float = 10.0
    """Reward weight for Δ(mean particle z) between consecutive steps."""

    rew_frac_above: float = 5.0
    """Reward weight for the fraction of particles above
    ``scoop_z_threshold``."""

    rew_action_penalty: float = 1e-3
    """Penalty weight applied to the squared L2 norm of the action."""

    rew_alive: float = 0.0
    """Per-step alive bonus."""
