# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-Scoop-Direct-Warp-v2: unified Newton model for true robot–sand collision.

Architecture difference vs v0/v1
---------------------------------
v0/v1: TWO separate Newton models.
  - Model 1 (IsaacLab Articulation / MJWarp): robot dynamics + USD box walls.
  - Model 2 (SandMPMHelper): kinematic proxy bodies + MPM particles.
  - Coupling: one-way pose copy each step → scoop passes through sand.

v2: ONE unified ModelBuilder.
  - Robot loaded from URDF (actual mesh collision on every link incl. scoop).
  - Box walls, ground, and particles all in the same model.
  - SolverMuJoCo drives robot dynamics; SolverImplicitMPM drives sand.
  - Both solvers share state_0.body_q → zero-lag, real bidirectional geometry.

Result: scoop, sand particles, box walls, and ground are all aware of each
other within the same physics timestep.
"""

from __future__ import annotations

import math
import os

import gymnasium as gym
import newton
import numpy as np
import torch
import warp as wp
from gymnasium import spaces
from newton.solvers import SolverImplicitMPM, SolverMuJoCo

from isaaclab.utils import configclass

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
_URDF_PATH = os.path.join(_ASSETS_DIR, "ur5_with_scoop.urdf")

# Hover pose — arm parked above sand, matches v0/v1 initial state.
_Q_HOVER = np.array([-0.27, -1.50, 1.70, math.pi / 2, math.pi / 2, math.pi / 2], dtype=np.float32)

# Sandbox geometry — identical to v0/v1 so environments look the same.
_BOX_W = 0.35
_BOX_D = 0.35
_BOX_WALL_H = 0.05
_BOX_FLOOR_T = 0.02

# (centre_xyz, half_extents_xyz) — all env-local, applied before box_offset.
_BOX_PIECES: list[tuple[tuple, tuple]] = [
    ((0.0, 0.0, _BOX_FLOOR_T / 2), (_BOX_W / 2, _BOX_D / 2, _BOX_FLOOR_T / 2)),
    ((0.0, -_BOX_D / 2, _BOX_WALL_H / 2), (_BOX_W / 2, _BOX_FLOOR_T / 2, _BOX_WALL_H / 2)),
    ((0.0, _BOX_D / 2, _BOX_WALL_H / 2), (_BOX_W / 2, _BOX_FLOOR_T / 2, _BOX_WALL_H / 2)),
    ((-_BOX_W / 2, 0.0, _BOX_WALL_H / 2), (_BOX_FLOOR_T / 2, _BOX_D / 2, _BOX_WALL_H / 2)),
    ((_BOX_W / 2, 0.0, _BOX_WALL_H / 2), (_BOX_FLOOR_T / 2, _BOX_D / 2, _BOX_WALL_H / 2)),
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@configclass
class ScoopWarpEnvCfgV2:
    """Configuration for Isaac-Scoop-Direct-Warp-v2 (unified Newton model)."""

    # --- Simulation timing ---
    fps: float = 60.0
    sim_substeps: int = 4  # robot solver substeps per control step
    max_episode_steps: int = 600

    # --- Robot ---
    robot_pos: tuple = (0.5, 0.0, 0.0)  # base position in world frame

    # --- Sandbox ---
    box_offset: tuple = (0.0, -0.2, 0.0)  # shift sandbox in env-local frame

    # --- Sand MPM material ---
    voxel_size: float = 0.02
    grid_type: str = "sparse"
    solver: str = "gauss-seidel"
    transfer_scheme: str = "apic"
    max_iterations: int = 250
    tolerance: float = 1e-6
    young_modulus: float = 5e3
    poisson_ratio: float = 0.25
    friction: float = 0.8
    damping: float = 6000.0
    yield_pressure: float = 50.0
    yield_stress: float = 25.0
    tensile_yield_ratio: float = 0.0
    hardening: float = 0.5
    air_drag: float = 1.0
    critical_fraction: float = 0.0

    # --- Particle spawn ---
    density: float = 1100.0
    particles_per_cell: int = 2
    emit_lo: tuple = (-0.15, -0.15, 0.02)
    emit_hi: tuple = (0.15, 0.15, 0.20)
    initial_jitter: float = 0.5
    particle_max_velocity: float = 1.0
    settle_steps: int = 120

    # --- Rewards ---
    dist_reward_scale: float = -0.5
    action_penalty_scale: float = -0.001
    alive_reward: float = 0.05
    elevation_reward_scale: float = 4.0
    lift_threshold: float = 0.03


# ---------------------------------------------------------------------------
# Compatibility shims so waypoint_agent.py can read joint limits unchanged
# ---------------------------------------------------------------------------


class _LimitsData:
    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        # Shape (1, num_dofs, 2) — matches env.unwrapped.robot.data.soft_joint_pos_limits
        limits = np.stack([lower, upper], axis=-1)[np.newaxis]
        self.soft_joint_pos_limits = torch.from_numpy(limits.astype(np.float32))


class _RobotShim:
    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        self.data = _LimitsData(lower, upper)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ScoopWarpEnvV2(gym.Env):
    """v2: unified Newton model — true scoop/sand/box collision.

    Unlike v0/v1 this env does NOT use IsaacLab's Articulation system.
    Robot dynamics are driven by Newton's SolverMuJoCo; sand by
    SolverImplicitMPM.  Both solvers operate on the SAME model state so
    body_q is always current with zero lag and no proxy bodies are needed.
    """

    metadata = {"render_modes": ["human"]}

    NUM_DOFS = 6  # UR5: 6 revolute joints; fixed joints have 0 DOF

    def __init__(self, cfg: ScoopWarpEnvCfgV2 | None = None, render_mode: str | None = None, **kwargs):
        super().__init__()
        self.cfg = cfg or ScoopWarpEnvCfgV2()
        self.render_mode = render_mode

        # Timing
        self._control_dt = 1.0 / self.cfg.fps
        self._sim_dt = self._control_dt / self.cfg.sim_substeps
        self._max_steps = self.cfg.max_episode_steps
        self._step_count = 0
        self._sim_time = 0.0

        # Public attrs expected by runner scripts
        self.device = "cuda" if wp.get_device().is_cuda else "cpu"
        self.num_envs = 1

        # Build unified Newton model
        self._build_model()

        # Settle sand and save snapshot
        self._settle()

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.NUM_DOFS,), dtype=np.float32)
        # obs: joint_q_norm(6) + joint_qd(6) + EE_pos(3) + sand_centroid(3) + elevated_frac(1) = 19
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(19,), dtype=np.float32)

        # Shim so waypoint_agent.py can call env.unwrapped.robot.data.soft_joint_pos_limits
        self.robot = _RobotShim(self._joint_lower, self._joint_upper)

        # Newton viewer (set externally via set_viewer() before the render loop)
        self._viewer = None

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        cfg = self.cfg
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)

        # ---- Robot — loaded from URDF (real mesh collision on every link) ----
        builder.add_urdf(
            _URDF_PATH,
            xform=wp.transform(
                wp.vec3(*cfg.robot_pos),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi),
            ),
            floating=False,
            enable_self_collisions=False,
        )

        # Initialise joint angles to hover pose
        builder.joint_q[: self.NUM_DOFS] = _Q_HOVER.tolist()

        # PD gains
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 2000.0
            builder.joint_target_kd[i] = 100.0

        # ---- Ground plane ----
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        # ---- Static sandbox (same builder — robot sees these for collision) ----
        off = np.array(cfg.box_offset, dtype=np.float32)
        box_shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.6, gap=0.01)
        for (cx, cy, cz), (hx, hy, hz) in _BOX_PIECES:
            builder.add_shape_box(
                body=-1,
                cfg=box_shape_cfg,
                xform=wp.transform(
                    wp.vec3(float(cx + off[0]), float(cy + off[1]), float(cz + off[2])),
                    wp.quat_identity(),
                ),
                hx=hx,
                hy=hy,
                hz=hz,
            )

        # ---- Sand particles ----
        emit_lo = np.array(cfg.emit_lo, dtype=np.float32) + off
        emit_hi = np.array(cfg.emit_hi, dtype=np.float32) + off
        particle_res = np.ceil(cfg.particles_per_cell * (emit_hi - emit_lo) / cfg.voxel_size).astype(int)
        cell_size = (emit_hi - emit_lo) / particle_res
        radius = float(np.max(cell_size) * 0.5)
        mass_p = float(np.prod(cell_size) * cfg.density)

        builder.add_particle_grid(
            pos=wp.vec3(*emit_lo.tolist()),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(particle_res[0]) + 1,
            dim_y=int(particle_res[1]) + 1,
            dim_z=int(particle_res[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass_p,
            jitter=cfg.initial_jitter * radius,
            radius_mean=radius,
        )

        # ---- Finalise ----
        self._model = builder.finalize()
        self._model.set_gravity([0.0, 0.0, -10.0])
        self._model.particle_max_velocity = cfg.particle_max_velocity

        # Set MPM material params
        mpm_fields = [
            "young_modulus", "poisson_ratio", "friction", "damping",
            "yield_pressure", "yield_stress", "tensile_yield_ratio",
            "hardening", "air_drag", "critical_fraction",
        ]
        for field in mpm_fields:
            if hasattr(self._model.mpm, field):
                getattr(self._model.mpm, field).fill_(getattr(cfg, field))

        # Joint limits from URDF (first NUM_DOFS entries = 6 revolute joints)
        self._joint_lower = self._model.joint_limit_lower.numpy()[: self.NUM_DOFS]
        self._joint_upper = self._model.joint_limit_upper.numpy()[: self.NUM_DOFS]

        # Double-buffered states for robot solver
        self._state0 = self._model.state()
        self._state1 = self._model.state()

        # Evaluate FK to populate body_q from hover pose
        newton.eval_fk(self._model, self._state0.joint_q, self._state0.joint_qd, self._state0)

        # ---- Solvers ----
        mpm_opts = SolverImplicitMPM.Config()
        mpm_opts.voxel_size = cfg.voxel_size
        mpm_opts.grid_type = cfg.grid_type
        mpm_opts.solver = cfg.solver
        mpm_opts.transfer_scheme = cfg.transfer_scheme
        mpm_opts.max_iterations = cfg.max_iterations
        mpm_opts.tolerance = cfg.tolerance
        mpm_opts.collider_velocity_mode = "finite_difference"

        self._mpm_solver = SolverImplicitMPM(self._model, mpm_opts)
        # All robot bodies treated as kinematic by MPM (mass=0 override).
        # SolverMuJoCo still sees their real masses for robot dynamics.
        self._mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self._model.body_mass),
            body_q=self._state0.body_q,
        )

        self._robot_solver = SolverMuJoCo(self._model)

        # Control buffer
        self._control = self._model.control()
        ctrl = self._control.joint_target_pos.numpy()
        ctrl[: self.NUM_DOFS] = _Q_HOVER
        self._control.joint_target_pos.assign(ctrl)

        # Diagnostics
        n_bodies = self._model.body_mass.shape[0]
        n_particles = self._state0.particle_q.shape[0]
        print(
            f"[ScoopWarpEnvV2] Unified model: {n_bodies} bodies, "
            f"{self.NUM_DOFS} DOFs, {n_particles} particles"
        )
        print(f"[ScoopWarpEnvV2] Joint lower: {np.round(self._joint_lower, 3)}")
        print(f"[ScoopWarpEnvV2] Joint upper: {np.round(self._joint_upper, 3)}")

        # Print body positions after FK so the user can verify the scoop index
        body_q_np = self._state0.body_q.numpy()
        print("[ScoopWarpEnvV2] Body positions after FK (index: xyz):")
        for i, bq in enumerate(body_q_np):
            print(f"  body[{i:2d}]  pos=({bq[0]:.3f}, {bq[1]:.3f}, {bq[2]:.3f})")

        # Scoop is the most distal body — highest distance from robot base
        base_pos = body_q_np[0, :3]
        dists = np.linalg.norm(body_q_np[:n_bodies, :3] - base_pos, axis=1)
        self._scoop_body_idx = int(np.argmax(dists))
        print(f"[ScoopWarpEnvV2] Scoop body index (auto): {self._scoop_body_idx}")

    # ------------------------------------------------------------------
    # Sand settling
    # ------------------------------------------------------------------

    def _settle(self) -> None:
        dt = self._control_dt
        for _ in range(self.cfg.settle_steps):
            self._mpm_solver.step(self._state0, self._state0, contacts=None, control=None, dt=dt)
        self._snapshot_q = self._state0.particle_q.numpy().copy()
        self._snapshot_qd = self._state0.particle_qd.numpy().copy()
        print(f"[ScoopWarpEnvV2] Sand settled over {self.cfg.settle_steps} steps.")

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Reset joints to hover pose
        q = self._state0.joint_q.numpy()
        q[: self.NUM_DOFS] = _Q_HOVER
        self._state0.joint_q.assign(q)

        qd = self._state0.joint_qd.numpy()
        qd[:] = 0.0
        self._state0.joint_qd.assign(qd)

        # Update body_q from new joint angles
        newton.eval_fk(self._model, self._state0.joint_q, self._state0.joint_qd, self._state0)

        # Restore sand snapshot
        self._state0.particle_q.assign(self._snapshot_q)
        self._state0.particle_qd.assign(self._snapshot_qd)

        # Re-init MPM collider so body_q_prev = current hover pose (no velocity spike)
        self._mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self._model.body_mass),
            body_q=self._state0.body_q,
        )

        # Reset control targets
        ctrl = self._control.joint_target_pos.numpy()
        ctrl[: self.NUM_DOFS] = _Q_HOVER
        self._control.joint_target_pos.assign(ctrl)

        self._step_count = 0
        self._sim_time = 0.0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # Scale action [-1, 1] → joint position targets [rad]
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        rng = self._joint_upper - self._joint_lower
        targets = 0.5 * (a + 1.0) * rng + self._joint_lower

        ctrl = self._control.joint_target_pos.numpy()
        ctrl[: self.NUM_DOFS] = targets
        self._control.joint_target_pos.assign(ctrl)

        # Robot dynamics — multiple substeps for stability
        for _ in range(self.cfg.sim_substeps):
            self._state0.clear_forces()
            self._robot_solver.step(self._state0, self._state1, self._control, contacts=None, dt=self._sim_dt)
            self._state0, self._state1 = self._state1, self._state0

        # MPM step — uses state0.body_q that SolverMuJoCo just updated (zero lag)
        self._mpm_solver.step(self._state0, self._state0, contacts=None, control=None, dt=self._control_dt)

        self._step_count += 1
        self._sim_time += self._control_dt

        obs = self._get_obs()
        reward = float(
            self.cfg.dist_reward_scale * float(np.linalg.norm(obs[12:15] - obs[15:18]))
            + self.cfg.elevation_reward_scale * float(obs[18])
            + self.cfg.alive_reward
        )
        terminated = False
        truncated = self._step_count >= self._max_steps

        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        joint_q = self._state0.joint_q.numpy()[: self.NUM_DOFS]
        joint_qd = self._state0.joint_qd.numpy()[: self.NUM_DOFS]

        # Normalise joint positions to [-1, 1]
        rng = np.where((self._joint_upper - self._joint_lower) > 1e-6, self._joint_upper - self._joint_lower, 1.0)
        joint_q_norm = (2.0 * joint_q - self._joint_upper - self._joint_lower) / rng

        # Scoop (EE) position from body_q — shape (N_bodies, 7): [px, py, pz, qx, qy, qz, qw]
        body_q_np = self._state0.body_q.numpy()
        ee_pos = body_q_np[self._scoop_body_idx, :3]

        # Sand centroid
        particle_q = self._state0.particle_q.numpy()  # (N_particles, 3)
        centroid = particle_q.mean(axis=0)

        # Elevated fraction
        elevated = float(np.mean(particle_q[:, 2] > self._snapshot_q[:, 2] + self.cfg.lift_threshold))

        obs = np.concatenate([
            joint_q_norm,        # 6
            joint_qd * 0.1,      # 6
            ee_pos,              # 3  [12:15]
            centroid,            # 3  [15:18]
            [elevated],          # 1  [18]
        ]).astype(np.float32)    # 19 total

        return obs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def set_viewer(self, viewer) -> None:
        """Attach a Newton ViewerGL. Call before the render loop."""
        self._viewer = viewer
        viewer.set_model(self._model)
        viewer.show_particles = True

    def render(self) -> None:
        if self._viewer is None:
            return
        self._viewer.begin_frame(self._sim_time)
        self._viewer.log_state(self._state0)
        self._viewer.end_frame()

    def close(self) -> None:
        pass
