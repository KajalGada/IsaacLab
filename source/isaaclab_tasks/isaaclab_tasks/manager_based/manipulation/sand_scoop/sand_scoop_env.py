# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinetic-sand scooping RL environment (Newton MPM + MuJoCo solver).

Architecture
------------
This is a *standalone* :class:`gymnasium.Env` that runs entirely inside
Newton — no Isaac Sim / USD stage is required.  Two Newton solvers run
side-by-side each simulation frame:

1. ``SolverMuJoCo``     – UR5 arm rigid-body dynamics + PD joint control.
2. ``SolverImplicitMPM`` – kinetic-sand particle simulation.

Newton manages robot↔particle collision automatically: after the robot
solver updates ``state.body_q``, the MPM solver reads those same body
transforms as kinematic colliders.

Training
--------
Single-environment mode (``num_envs=1``) is the current target.  For
multi-env parallel training wrap several instances with
``gymnasium.vector.AsyncVectorEnv`` or run them across processes with SB3 /
SKRL.  True GPU-parallel vectorisation (single Newton multi-world model)
can be added once Newton MPM supports ``begin_world()`` batching.

Observation (27-dim)
--------------------
joint_pos (6) | joint_vel (6) | eef_pos (3) | eef_quat_wxyz (4) |
sand_on_scoop_norm (1) | sand_in_target_norm (1) | last_action (6)

Action (6-dim)
--------------
Normalised joint-position deltas in [-1, 1], scaled by ``cfg.action_scale``
(radians).  IK-relative EEF control can be layered on top via an external
Jacobian controller; the low-level interface is intentionally joint-space so
it works out of the box.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp
import gymnasium as gym

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM, SolverMuJoCo

from .sand_scoop_env_cfg import SandScoopEnvCfg

# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def count_particles_in_box(
    particle_q: wp.array(dtype=wp.vec3),
    box_min: wp.vec3,
    box_max: wp.vec3,
    count: wp.array(dtype=wp.int32),
):
    """Count MPM particles whose position falls inside an axis-aligned box."""
    i = wp.tid()
    p = particle_q[i]
    if (
        p[0] >= box_min[0] and p[0] <= box_max[0]
        and p[1] >= box_min[1] and p[1] <= box_max[1]
        and p[2] >= box_min[2] and p[2] <= box_max[2]
    ):
        wp.atomic_add(count, 0, 1)


@wp.kernel
def count_particles_in_sphere(
    particle_q: wp.array(dtype=wp.vec3),
    center: wp.vec3,
    radius: float,
    count: wp.array(dtype=wp.int32),
):
    """Count MPM particles within a sphere (approximation for scoop bowl)."""
    i = wp.tid()
    d = wp.length(particle_q[i] - center)
    if d < radius:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def reset_particle_state(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    init_q: wp.array(dtype=wp.vec3),
):
    """Reset particle positions to initial (settled) values and zero velocities."""
    i = wp.tid()
    particle_q[i] = init_q[i]
    particle_qd[i] = wp.vec3(0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SandScoopEnv(gym.Env):
    """Kinetic-sand scooping environment backed by Newton MPM.

    Args:
        cfg: Environment configuration.  Defaults to :class:`SandScoopEnvCfg`.
        render_mode: ``"human"`` to open Newton's OpenGL viewer during
            ``step()`` / ``reset()``, ``None`` for headless training.
    """

    metadata: dict[str, Any] = {"render_modes": ["human", None]}

    def __init__(
        self,
        cfg: SandScoopEnvCfg | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or SandScoopEnvCfg()
        self.render_mode = render_mode

        # gymnasium spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.num_actions,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.num_observations,), dtype=np.float32,
        )

        self._build_simulation()
        self._settle_sand()

        # internal bookkeeping
        self._prev_action: np.ndarray = np.zeros(self.cfg.num_actions, dtype=np.float32)
        self._step_count: int = 0
        self._max_steps: int = int(
            self.cfg.episode_length_s / (self.cfg.sim_dt * self.cfg.policy_decimation)
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_simulation(self) -> None:
        """Construct the Newton model: UR5 robot, two containers, sand particles."""
        cfg = self.cfg
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)

        # -- Robot -------------------------------------------------------
        # Robot is at -X; containers are at +X → no Z rotation needed.
        base_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), cfg.robot_base_yaw)  # type: ignore[arg-type]
        builder.add_urdf(
            cfg.urdf_path,
            xform=wp.transform(wp.vec3(*cfg.robot_base_pos), base_rot),
            floating=False,
            enable_self_collisions=False,
        )
        self._robot_dof = 6

        # Capture scoop body index from builder NOW (body_label only exists on
        # ModelBuilder before finalize; the Model object drops it).
        self._scoop_body_idx: int = next(
            (i for i, n in enumerate(builder.body_label) if n.split("/")[-1] == "scoop_link"),
            len(builder.body_label) - 1,  # fallback: last body
        )
        print(f"[SandScoopEnv] scoop body index = {self._scoop_body_idx} "
              f"(label: {builder.body_label[self._scoop_body_idx]})")

        # Home joint angles as initial state
        builder.joint_q[: self._robot_dof] = list(cfg.home_q)

        # PD gains
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = cfg.joint_kp
            builder.joint_target_kd[i] = cfg.joint_kd

        # -- Disable COLLIDE_PARTICLES for all URDF mesh shapes ------------
        # High-polygon URDF meshes crash NanoVDB's PointsToGrid kernel in
        # setup_collider — replaced by the explicit scoop sphere below.
        # COLLIDE_SHAPES is kept so MuJoCo prevents the scoop from going
        # through the robot floor (added below at z=-0.05).
        for i in range(len(builder.shape_flags)):
            if builder.shape_type[i] == int(newton.GeoType.MESH):
                builder.shape_flags[i] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)

        # -- Scoop bowl collision sphere ---------------------------------
        # Sphere placed at scoop_bowl_z_offset along the scoop link's local
        # Z axis — matches the detection geometry used in _count_scoop().
        builder.add_shape_sphere(
            body=self._scoop_body_idx,
            xform=wp.transform(wp.vec3(0.0, 0.0, cfg.scoop_bowl_z_offset), wp.quat_identity()),
            radius=cfg.scoop_detect_radius,
            # has_shape_collision=False: MuJoCo ignores this sphere so the arm
            # is not pushed away from the ground; MPM still uses it for particles.
            cfg=newton.ModelBuilder.ShapeConfig(mu=cfg.wall_mu, has_shape_collision=False),
        )

        # -- Ground planes -----------------------------------------------
        # Particle floor at z=0: _project_outside keeps particles above it.
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, has_shape_collision=False)
        )
        # Invisible robot arm floor at z=-0.05: MuJoCo stops the scoop mesh
        # from going through the floor without a second visible surface.
        # Placed 5 cm below the particle floor so the arm can still reach
        # the sand at z≈0 without being blocked at z≈0.22.
        builder.add_shape_plane(
            plane=(0.0, 0.0, 1.0, 0.05),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, has_particle_collision=False, is_visible=False),
        )

        # -- Source container --------------------------------------------
        self._add_container(builder, cfg.source_pos)

        # -- Target container --------------------------------------------
        self._add_container(builder, cfg.target_pos)

        # -- Sand particles (in source container) -----------------------
        self._num_particles = self._add_particles(builder)

        # -- Finalise model ---------------------------------------------
        self.model = builder.finalize()
        self.model.set_gravity(wp.vec3(0.0, 0.0, -9.81))

        # -- MPM solver --------------------------------------------------
        mpm_opts = SolverImplicitMPM.Config()
        mpm_opts.collider_velocity_mode = "finite_difference"
        mpm_opts.voxel_size = cfg.mpm_voxel_size

        # Material parameters: some live on SolverImplicitMPM.Config, others on
        # model.mpm as per-particle Warp arrays.  Mirror the guard pattern from
        # simulation_newton_sand_v2.py so we hit the right location for each param.
        _material = {
            "density":        cfg.sand_density,
            "young_modulus":  cfg.sand_young_modulus,
            "poisson_ratio":  cfg.sand_poisson_ratio,
            "friction":       cfg.sand_friction,
            "damping":        cfg.sand_damping,
            "yield_pressure": cfg.sand_yield_pressure,
            "yield_stress":   cfg.sand_yield_stress,
            "hardening":      cfg.sand_hardening,
        }
        for key, val in _material.items():
            if hasattr(mpm_opts, key):
                setattr(mpm_opts, key, val)
            if hasattr(self.model.mpm, key):
                getattr(self.model.mpm, key).fill_(val)

        self.mpm_solver = SolverImplicitMPM(self.model, mpm_opts)

        # -- Robot (MuJoCo) solver ---------------------------------------
        self.robot_solver = SolverMuJoCo(self.model, ccd_iterations=cfg.robot_ccd_iterations)

        # -- States ------------------------------------------------------
        self.state   = self.model.state()
        self.state_1 = self.model.state()

        # Forward kinematics: populate body_q from joint_q
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

        # MPM collider setup (uses current body_q as reference).
        # Register static shapes (body=-1) with default projection_threshold so
        # _project_outside corrects slow boundary seepage on the ground/walls.
        # Register the scoop sphere (scoop_body_idx) with projection_threshold=0
        # so _project_outside NEVER violently ejects particles from inside the
        # sphere — the MPM grid contact force handles gradual scoop interaction.
        self.mpm_solver.setup_collider(
            collider_body_ids=[-1, self._scoop_body_idx],
            # Static shapes: default seepage correction (0.01 * voxel_size ≈ 0.2 mm).
            # Scoop sphere: 1 voxel threshold — only project deeply penetrating
            # particles, avoiding the violent ejection from tiny default threshold.
            collider_projection_threshold=[0.01 * cfg.mpm_voxel_size, cfg.mpm_voxel_size],
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state.body_q,
        )

        # -- PD control targets ------------------------------------------
        self.control = self.model.control()
        q_np = self.control.joint_target_pos.numpy()
        q_np[: self._robot_dof] = list(cfg.home_q)
        self.control.joint_target_pos.assign(q_np)


    def _add_container(
        self,
        builder: newton.ModelBuilder,
        pos: tuple[float, float, float],
    ) -> None:
        """Add a static open-top box (5 rigid planes) at the given position."""
        cfg = self.cfg
        cx, cy, cz = pos
        hw = cfg.box_w / 2
        hd = cfg.box_d / 2
        wt = cfg.wall_t
        bh = cfg.box_h
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=cfg.wall_mu,
            gap=0.01,
            # Particle-only: _project_outside keeps sand inside walls,
            # but the robot arm can pass through freely (no MuJoCo contact).
            has_shape_collision=False,
        )
        # No floor shape — the ground plane (analytical SDF) is the container floor.
        # Box SDF needs ≥2 MPM voxels to be resolved; the 2cm floor was only 1 voxel.
        # front wall (−y)
        builder.add_shape_box(body=-1, cfg=shape_cfg,
            xform=wp.transform(wp.vec3(cx, cy - hd, cz + bh * 0.5), wp.quat_identity()),
            hx=hw, hy=wt * 0.5, hz=bh * 0.5)
        # back wall (+y)
        builder.add_shape_box(body=-1, cfg=shape_cfg,
            xform=wp.transform(wp.vec3(cx, cy + hd, cz + bh * 0.5), wp.quat_identity()),
            hx=hw, hy=wt * 0.5, hz=bh * 0.5)
        # left wall (−x)
        builder.add_shape_box(body=-1, cfg=shape_cfg,
            xform=wp.transform(wp.vec3(cx - hw, cy, cz + bh * 0.5), wp.quat_identity()),
            hx=wt * 0.5, hy=hd, hz=bh * 0.5)
        # right wall (+x)
        builder.add_shape_box(body=-1, cfg=shape_cfg,
            xform=wp.transform(wp.vec3(cx + hw, cy, cz + bh * 0.5), wp.quat_identity()),
            hx=wt * 0.5, hy=hd, hz=bh * 0.5)

    def _add_particles(self, builder: newton.ModelBuilder) -> int:
        """Emit a grid of MPM particles inside the source container. Returns particle count."""
        cfg = self.cfg
        src = np.array(cfg.source_pos, dtype=np.float32)
        lo = src + np.array(cfg.mpm_emit_lo, dtype=np.float32)
        hi = src + np.array(cfg.mpm_emit_hi, dtype=np.float32)

        ppc = cfg.mpm_particles_per_cell
        dx  = cfg.mpm_voxel_size
        res = np.array(np.ceil(ppc * (hi - lo) / dx), dtype=int)
        cell = (hi - lo) / res
        radius = float(np.max(cell) * 0.5)
        mass   = float(np.prod(cell) * cfg.sand_density)

        builder.add_particle_grid(
            pos=wp.vec3(float(lo[0]), float(lo[1]), float(lo[2])),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(res[0]) + 1,
            dim_y=int(res[1]) + 1,
            dim_z=int(res[2]) + 1,
            cell_x=float(cell[0]),
            cell_y=float(cell[1]),
            cell_z=float(cell[2]),
            mass=mass,
            jitter=0.5 * radius,
            radius_mean=radius,
        )
        total = (int(res[0]) + 1) * (int(res[1]) + 1) * (int(res[2]) + 1)
        print(f"[SandScoopEnv] Emitted {total} sand particles "
              f"(radius={radius:.4f} m, mass/particle={mass:.5f} kg)")
        return total

    # ------------------------------------------------------------------
    # Settling
    # ------------------------------------------------------------------

    def _settle_sand(self) -> None:
        """Run gravity-only frames so particles form a natural pile before training."""
        cfg = self.cfg
        print(f"[SandScoopEnv] Settling sand ({cfg.mpm_settle_frames} frames)…")
        dt = cfg.sim_dt / cfg.mpm_substeps
        for _ in range(cfg.mpm_settle_frames):
            self._sim_frame(dt)
        # Save settled particle state for fast episode reset
        self._init_particle_q  = wp.clone(self.state.particle_q)
        self._init_particle_qd = wp.clone(self.state.particle_qd)
        print("[SandScoopEnv] Sand settled — environment ready.")

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Reset robot joints to home pose
        jq = self.state.joint_q.numpy()
        jq[: self._robot_dof] = list(self.cfg.home_q)
        self.state.joint_q.assign(jq)

        jqd = self.state.joint_qd.numpy()
        jqd[: self._robot_dof] = 0.0
        self.state.joint_qd.assign(jqd)

        # Reset PD control targets
        q_np = self.control.joint_target_pos.numpy()
        q_np[: self._robot_dof] = list(self.cfg.home_q)
        self.control.joint_target_pos.assign(q_np)

        # Reset particle positions & velocities to settled state
        wp.launch(
            reset_particle_state,
            dim=self._num_particles,
            inputs=[self.state.particle_q, self.state.particle_qd, self._init_particle_q],
        )
        wp.synchronize()

        # Re-run FK so body_q reflects the home joint angles
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

        self._prev_action = np.zeros(self.cfg.num_actions, dtype=np.float32)
        self._step_count  = 0

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        scaled = action * self.cfg.action_scale

        # Apply joint-position delta
        q_np = self.control.joint_target_pos.numpy()
        lo    = np.array(self.cfg.joint_lo, dtype=np.float32)
        hi    = np.array(self.cfg.joint_hi, dtype=np.float32)
        q_np[: self._robot_dof] = np.clip(
            q_np[: self._robot_dof] + scaled, lo, hi
        )
        self.control.joint_target_pos.assign(q_np)

        # Run several simulation frames
        dt = self.cfg.sim_dt / self.cfg.mpm_substeps
        for _ in range(self.cfg.policy_decimation):
            self._sim_frame(dt)

        obs     = self._get_obs()
        reward  = self._compute_reward(scaled)
        self._step_count += 1
        done    = self._is_done()

        self._prev_action = action.copy()
        return obs, reward, done, False, {}

    def render(self) -> None:
        # Newton viewer is managed externally when render_mode="human"
        pass

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------

    def _sim_frame(self, dt: float) -> None:
        """One Newton simulation frame: robot substeps then one MPM frame."""
        cfg = self.cfg
        for _ in range(cfg.mpm_substeps):
            self.state.clear_forces()
            self.robot_solver.step(
                self.state, self.state_1, self.control,
                contacts=None, dt=dt,
            )
            self.state, self.state_1 = self.state_1, self.state

        # MPM reads state.body_q (updated by robot solver) as colliders
        self.mpm_solver.step(
            self.state, self.state,
            contacts=None, control=None,
            dt=cfg.sim_dt,
        )
        # Project particles back outside all colliders to prevent gradual boundary seepage.
        # Grid-based MPM contact forces alone cannot maintain perfect separation over many steps.
        self.mpm_solver._project_outside(self.state, self.state, cfg.sim_dt)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the 27-dim observation vector on CPU."""
        # Joint state
        jq  = self.state.joint_q.numpy()[: self._robot_dof].astype(np.float32)
        jqd = self.state.joint_qd.numpy()[: self._robot_dof].astype(np.float32)

        # EEF pose from body transforms (FK already current after each solver step)
        body_q = self.state.body_q.numpy()          # [N_bodies, 7] (px,py,pz, qx,qy,qz,qw)
        sb     = self._scoop_body_idx
        eef_pos  = body_q[sb, :3].astype(np.float32)
        # Newton stores quat as (qx,qy,qz,qw); re-order to (qw,qx,qy,qz)
        q_xyzw   = body_q[sb, 3:7].astype(np.float32)
        eef_quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

        # Particle counts (GPU → scalar)
        sand_scoop  = float(self._count_scoop()) / max(self._num_particles, 1)
        sand_target = float(self._count_target()) / max(self._num_particles, 1)

        obs = np.concatenate([
            jq, jqd, eef_pos, eef_quat,
            [sand_scoop, sand_target],
            self._prev_action,
        ])
        return obs.astype(np.float32)

    def _count_scoop(self) -> int:
        """Count particles inside the scoop bowl (sphere approximation)."""
        body_q  = self.state.body_q.numpy()
        sb      = self._scoop_body_idx
        # Scoop-link world position
        pos     = body_q[sb, :3].astype(np.float64)
        # Approximate bowl centre: offset along scoop link's local +Z axis
        # (rotation from local to world via quaternion)
        q_xyzw  = body_q[sb, 3:7].astype(np.float64)
        qw, qx, qy, qz = q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]
        # local Z = (0,0,1) rotated by q
        local_z = np.array([
            2*(qx*qz + qw*qy),
            2*(qy*qz - qw*qx),
            1 - 2*(qx*qx + qy*qy),
        ])
        center = pos + local_z * self.cfg.scoop_bowl_z_offset

        count_arr = wp.zeros(1, dtype=wp.int32)
        wp.launch(
            count_particles_in_sphere,
            dim=self._num_particles,
            inputs=[
                self.state.particle_q,
                wp.vec3(float(center[0]), float(center[1]), float(center[2])),
                float(self.cfg.scoop_detect_radius),
                count_arr,
            ],
        )
        wp.synchronize()
        return int(count_arr.numpy()[0])

    def _count_target(self) -> int:
        """Count particles inside the target container AABB."""
        cfg = self.cfg
        cx, cy, cz = cfg.target_pos
        hw, hd = cfg.box_w / 2, cfg.box_d / 2
        box_min = wp.vec3(cx - hw, cy - hd, cz)
        box_max = wp.vec3(cx + hw, cy + hd, cz + cfg.box_h)

        count_arr = wp.zeros(1, dtype=wp.int32)
        wp.launch(
            count_particles_in_box,
            dim=self._num_particles,
            inputs=[self.state.particle_q, box_min, box_max, count_arr],
        )
        wp.synchronize()
        return int(count_arr.numpy()[0])

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, scaled_action: np.ndarray) -> float:
        cfg    = self.cfg
        reward = 0.0

        body_q = self.state.body_q.numpy()
        eef_pos = body_q[self._scoop_body_idx, :3]

        sand_scoop  = float(self._count_scoop())  / max(self._num_particles, 1)
        # sand_target = float(self._count_target()) / max(self._num_particles, 1)

        # Stage 1: approach source container
        src   = np.array(cfg.source_pos, dtype=np.float64)
        hover = src + np.array([0.0, 0.0, cfg.box_h * 0.5 + 0.05])
        dist_src = float(np.linalg.norm(eef_pos - hover))
        reward += cfg.w_approach * math.exp(-dist_src * 5.0)

        # Stage 2: scoop reward (particles on scoop)
        reward += cfg.w_scoop * sand_scoop

        # Stage 3: transport — disabled, focusing on scooping only
        # if sand_scoop > 0.05:
        #     tgt   = np.array(cfg.target_pos, dtype=np.float64)
        #     pour  = tgt + np.array([0.0, 0.0, cfg.box_h + 0.08])
        #     dist_tgt = float(np.linalg.norm(eef_pos - pour))
        #     reward += cfg.w_transport * math.exp(-dist_tgt * 5.0)

        # Stage 4: pour reward — disabled
        # reward += cfg.w_pour * sand_target

        # Stage 5: success bonus — disabled
        # if sand_target >= cfg.success_fraction:
        #     reward += cfg.w_success

        # Regularisation: penalise large actions
        reward += cfg.w_action_penalty * float(np.sum(scaled_action ** 2))

        # Joint-limit proximity penalty
        jq  = self.state.joint_q.numpy()[: self._robot_dof]
        lo  = np.array(cfg.joint_lo, dtype=np.float32)
        hi  = np.array(cfg.joint_hi, dtype=np.float32)
        vio = float(np.sum(
            np.maximum(jq - hi + 0.1, 0.0) + np.maximum(lo + 0.1 - jq, 0.0)
        ))
        reward += cfg.w_joint_limit * vio

        return float(reward)

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _is_done(self) -> bool:
        if self._step_count >= self._max_steps:
            return True
        sand_target = float(self._count_target()) / max(self._num_particles, 1)
        return sand_target >= self.cfg.success_fraction
