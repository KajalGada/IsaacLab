# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinetic-sand MPM simulation that runs alongside IsaacLab's MuJoCo-Warp robot model.

Architecture
------------
A separate Newton model holds only the sand particles and kinematic proxy bodies
(capsules/spheres/box approximating UR5+scoop links). Each control step:

  1. Robot body transforms are copied from IsaacLab → proxy body_q.
  2. SolverImplicitMPM.step() advances the particles.

Coupling is one-way: robot displaces sand; sand does not exert forces back on the robot.
"""

from __future__ import annotations

import dataclasses
import math

import newton
import numpy as np
import torch
import warp as wp
from newton.solvers import SolverImplicitMPM

from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Scene geometry constants (must match scoop_env_warp._BOX_PIECES)
# ---------------------------------------------------------------------------

_BOX_W = 0.35
_BOX_D = 0.35
_BOX_WALL_H = 0.05
_BOX_FLOOR_T = 0.02

# (name, half_extents_xyz, centre_xyz_env_local)
_BOX_PIECES: list[tuple[str, tuple, tuple]] = [
    ("floor", (0.175, 0.175, 0.01), (0.0, 0.0, 0.01)),
    ("wall_neg_y", (0.175, 0.01, 0.025), (0.0, -_BOX_D / 2, 0.025)),
    ("wall_pos_y", (0.175, 0.01, 0.025), (0.0, _BOX_D / 2, 0.025)),
    ("wall_neg_x", (0.01, 0.175, 0.025), (-_BOX_W / 2, 0.0, 0.025)),
    ("wall_pos_x", (0.01, 0.175, 0.025), (_BOX_W / 2, 0.0, 0.025)),
]

# ---------------------------------------------------------------------------
# Proxy body shapes for UR5+scoop links
# (link_name, shape, kwargs for builder.add_shape_*)
# ---------------------------------------------------------------------------

UR5_PROXY_LINK_NAMES: list[str] = [
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "scoop_link",
]

_NUM_PROXY_PER_ENV = len(UR5_PROXY_LINK_NAMES)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@configclass
class SandMPMCfg:
    """Configuration for the kinetic-sand MPM simulation."""

    # --- SolverImplicitMPM.Config fields ---
    voxel_size: float = 0.02  # [m] — coarser grid keeps particle count tractable for many envs
    grid_type: str = "sparse"
    solver: str = "gauss-seidel"
    transfer_scheme: str = "apic"
    max_iterations: int = 250
    tolerance: float = 1e-6
    # collider_velocity_mode is always "finite_difference" for FK robots; not user-exposed

    # --- model.mpm material fields ---
    young_modulus: float = 5e3
    poisson_ratio: float = 0.25
    friction: float = 0.8
    damping: float = 6000.0
    yield_pressure: float = 50.0
    yield_stress: float = 25.0
    tensile_yield_ratio: float = 0.0
    hardening: float = 0.5
    air_drag: float = 1.0  # particle air resistance (ref: 1.0)
    critical_fraction: float = 0.0  # fracture compression threshold (ref: 0.0)

    # --- Particle spawn ---
    density: float = 1100.0  # [kg/m³]
    particles_per_cell: int = 2  # controls spawn grid resolution; 2 gives ~18k particles/env at voxel_size=0.02
    emit_lo: tuple = (-0.15, -0.15, 0.02)  # z_lo=0.02 keeps particles above the floor
    emit_hi: tuple = (0.15, 0.15, 0.20)  # z_hi=0.20 keeps robot links above spawn volume
    initial_jitter: float = 0.5

    # --- Box and particle offset (env-local, applied on top of env origin) ---
    box_offset: tuple = (0.0, 0.0, 0.0)  # [m] shifts sandbox and particle spawn region

    # --- Startup settling ---
    settle_steps: int = 120  # MPM steps at episode init

    # --- Velocity caps ---
    # CFL condition: v_max * dt / voxel_size < 1.
    # With voxel_size=0.02 and dt=1/60: v_max < 1.2 m/s for stability.
    # particle_max_velocity clamps particle speed after each MPM step.
    # max_proxy_velocity clamps how fast the scoop collider appears to move
    # to the MPM finite_difference kernel — this is the dominant injection path
    # when a random RL agent drives joints aggressively (scoop tip at 5–15 m/s).
    particle_max_velocity: float = 1.0  # [m/s] — keep below CFL limit (1.2 m/s)
    max_proxy_velocity: float = 1.0  # [m/s] — cap collider velocity seen by MPM


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _update_proxy_body_q(
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    body_quat_w: wp.array2d(dtype=wp.quatf),
    robot_link_indices: wp.array(dtype=wp.int32),
    num_proxy_per_env: wp.int32,
    body_q: wp.array(dtype=wp.transformf),
):
    """Copy current UR5+scoop link transforms into the sand model's proxy body_q."""
    env_id, link_id = wp.tid()
    robot_body_idx = robot_link_indices[link_id]
    proxy_idx = env_id * num_proxy_per_env + link_id
    body_q[proxy_idx] = wp.transform(
        body_pos_w[env_id, robot_body_idx],
        body_quat_w[env_id, robot_body_idx],
    )


@wp.kernel
def _update_proxy_body_q_bounded(
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    body_quat_w: wp.array2d(dtype=wp.quatf),
    robot_link_indices: wp.array(dtype=wp.int32),
    num_proxy_per_env: wp.int32,
    max_disp: wp.float32,
    body_q: wp.array(dtype=wp.transformf),
):
    """Copy robot link transforms into proxy body_q, clamping translational displacement.

    Limits the velocity seen by the MPM finite_difference kernel to max_disp/dt.
    Without this a random RL agent can drive the scoop at 5–15 m/s, which the MPM
    imposes as a grid boundary condition → CFL violation → particles disappear.
    """
    env_id, link_id = wp.tid()
    robot_body_idx = robot_link_indices[link_id]
    proxy_idx = env_id * num_proxy_per_env + link_id

    new_pos = body_pos_w[env_id, robot_body_idx]
    new_quat = body_quat_w[env_id, robot_body_idx]
    old_pos = wp.transform_get_translation(body_q[proxy_idx])

    delta = new_pos - old_pos
    dist = wp.length(delta)
    if dist > max_disp and dist > wp.float32(1e-6):
        new_pos = old_pos + delta * (max_disp / dist)

    body_q[proxy_idx] = wp.transform(new_pos, new_quat)


@wp.kernel
def _restore_snapshot(
    snapshot_q: wp.array(dtype=wp.vec3f),
    snapshot_qd: wp.array(dtype=wp.vec3f),
    particle_env_id: wp.array(dtype=wp.int32),
    env_mask: wp.array(dtype=wp.bool),
    particle_q: wp.array(dtype=wp.vec3f),
    particle_qd: wp.array(dtype=wp.vec3f),
):
    """Reset particles to their settled snapshot for envs in the reset mask."""
    p_idx = wp.tid()
    env_id = particle_env_id[p_idx]
    if env_mask[env_id]:
        particle_q[p_idx] = snapshot_q[p_idx]
        particle_qd[p_idx] = snapshot_qd[p_idx]


@wp.kernel
def _compute_sand_obs(
    particle_q: wp.array(dtype=wp.vec3f),
    snapshot_q: wp.array(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    ee_pos_local: wp.array(dtype=wp.vec3f),
    starts: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    displaced_threshold: wp.float32,
    lift_threshold: wp.float32,
    observations: wp.array2d(dtype=wp.float32),
    obs_offset: wp.int32,
    centroid_out: wp.array(dtype=wp.vec3f),
):
    """Write 8 aggregated sand features into observations[:, obs_offset:obs_offset+8].

    Features (env-local):
        [0:3]  sand centroid xyz
        [3:6]  scoop → centroid vector (centroid - ee_pos_local)
        [6]    displaced_fraction  (particles moved > displaced_threshold from settled)
        [7]    elevated_fraction   (particles lifted > lift_threshold above settled z)
    """
    env_id = wp.tid()
    n = counts[env_id]
    start = starts[env_id]
    origin = env_origins[env_id]
    ee = ee_pos_local[env_id]

    cx = wp.float32(0.0)
    cy = wp.float32(0.0)
    cz = wp.float32(0.0)
    displaced = wp.int32(0)
    elevated = wp.int32(0)

    for i in range(n):
        idx = start + i
        p = particle_q[idx] - origin
        p0 = snapshot_q[idx] - origin
        cx += p[0]
        cy += p[1]
        cz += p[2]
        if wp.length(p - p0) > displaced_threshold:
            displaced += 1
        if p[2] > p0[2] + lift_threshold:
            elevated += 1

    fn = wp.float32(n)
    cent_x = cx / fn
    cent_y = cy / fn
    cent_z = cz / fn

    observations[env_id, obs_offset + 0] = cent_x
    observations[env_id, obs_offset + 1] = cent_y
    observations[env_id, obs_offset + 2] = cent_z
    observations[env_id, obs_offset + 3] = cent_x - ee[0]
    observations[env_id, obs_offset + 4] = cent_y - ee[1]
    observations[env_id, obs_offset + 5] = cent_z - ee[2]
    observations[env_id, obs_offset + 6] = wp.float32(displaced) / fn
    observations[env_id, obs_offset + 7] = wp.float32(elevated) / fn

    centroid_out[env_id] = wp.vec3f(cent_x, cent_y, cent_z)


@wp.kernel
def _compute_centroids(
    particle_q: wp.array(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    starts: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    centroid_out: wp.array(dtype=wp.vec3f),
):
    """Write env-local sand centroid positions to centroid_out."""
    env_id = wp.tid()
    n = counts[env_id]
    start = starts[env_id]
    origin = env_origins[env_id]
    cx = wp.float32(0.0)
    cy = wp.float32(0.0)
    cz = wp.float32(0.0)
    for i in range(n):
        p = particle_q[start + i] - origin
        cx += p[0]
        cy += p[1]
        cz += p[2]
    fn = wp.float32(n)
    centroid_out[env_id] = wp.vec3f(cx / fn, cy / fn, cz / fn)


@wp.kernel
def _add_scoop_rewards(
    particle_q: wp.array(dtype=wp.vec3f),
    snapshot_q: wp.array(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    ee_pos_local: wp.array(dtype=wp.vec3f),
    starts: wp.array(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    lift_threshold: wp.float32,
    capture_radius_xy: wp.float32,
    elevation_scale: wp.float32,
    captured_scale: wp.float32,
    rewards: wp.array(dtype=wp.float32),
):
    """Add elevation and captured-elevated reward components in-place.

    elevation reward: fraction of particles lifted > lift_threshold above settled z.
    captured_elevated reward: fraction of elevated particles within capture_radius_xy of the scoop.
    """
    env_id = wp.tid()
    n = counts[env_id]
    start = starts[env_id]
    origin = env_origins[env_id]
    ee = ee_pos_local[env_id]

    elevated = wp.int32(0)
    captured_elevated = wp.int32(0)

    for i in range(n):
        idx = start + i
        p = particle_q[idx] - origin
        p0 = snapshot_q[idx] - origin
        is_elevated = p[2] > p0[2] + lift_threshold
        if is_elevated:
            elevated += 1
            dx = p[0] - ee[0]
            dy = p[1] - ee[1]
            if dx * dx + dy * dy < capture_radius_xy * capture_radius_xy:
                captured_elevated += 1

    fn = wp.float32(n)
    rewards[env_id] += elevation_scale * wp.float32(elevated) / fn
    rewards[env_id] += captured_scale * wp.float32(captured_elevated) / fn


# ---------------------------------------------------------------------------
# Helper class
# ---------------------------------------------------------------------------


class SandMPMHelper:
    """Manages a standalone Newton MPM model for kinetic-sand simulation."""

    def build(
        self,
        num_envs: int,
        env_origins: torch.Tensor,
        robot_link_indices: list[int],
        cfg: SandMPMCfg,
        device: str,
        mpm_dt: float = 1.0 / 60.0,
    ) -> None:
        """Build the Newton sand model. Call after clone_environments() so env_origins are set.

        Args:
            num_envs: Number of parallel environments.
            env_origins: Tensor of shape (num_envs, 3), world-space env origin positions.
            robot_link_indices: Indices into robot.body_names for the 6 proxy links.
            cfg: Sand simulation configuration.
            device: Warp/CUDA device string (e.g. "cuda:0").
            mpm_dt: Duration of one MPM step [s] (= sim.dt × decimation). Used to convert
                cfg.max_proxy_velocity into a per-step displacement cap.
        """
        self._num_envs = num_envs
        self._device = device
        self._cfg = cfg
        # Max translational displacement per MPM step for the bounded proxy update.
        # Keeps the collider velocity seen by finite_difference below the CFL limit.
        self._max_proxy_disp = float(cfg.max_proxy_velocity * mpm_dt)
        origins_np = env_origins.cpu().numpy()  # (num_envs, 3)

        builder = newton.ModelBuilder()
        # Must be called before any bodies or particles are added
        SolverImplicitMPM.register_custom_attributes(builder)

        # Single ground plane covers all envs (all at z=0 for flat terrain)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        env_particle_starts: list[int] = []
        env_particle_counts: list[int] = []
        particle_env_ids: list[int] = []

        for env_i in range(num_envs):
            offset = origins_np[env_i]  # [x, y, z]

            # ---- Kinematic proxy bodies (one per UR10 link) ----------------
            # Bodies start at origin; transforms are overwritten every step via
            # _update_proxy_body_q before mpm_solver.step().
            # Newton capsules extend along Z by default; rotate 90° around Y to
            # align with the arm's local X axis for upper_arm and forearm links.
            _x_axis_capsule_xform = wp.transform(
                wp.vec3(0.0, 0.0, 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2),
            )
            for link_id in range(len(UR5_PROXY_LINK_NAMES)):
                body_idx = builder.add_body(mass=0.0)
                if link_id == 0:  # shoulder_link
                    builder.add_shape_sphere(body=body_idx, radius=0.055)
                elif link_id == 1:  # upper_arm_link — UR5 A2=0.425 m along X
                    builder.add_shape_capsule(
                        body=body_idx,
                        radius=0.04,
                        half_height=0.21,
                        xform=_x_axis_capsule_xform,
                    )
                elif link_id == 2:  # forearm_link — UR5 A3=0.39225 m along X
                    builder.add_shape_capsule(
                        body=body_idx,
                        radius=0.035,
                        half_height=0.20,
                        xform=_x_axis_capsule_xform,
                    )
                elif link_id == 3 or link_id == 4:  # wrist_1_link
                    builder.add_shape_sphere(body=body_idx, radius=0.038)
                elif link_id == 5:  # wrist_3_link
                    builder.add_shape_sphere(body=body_idx, radius=0.032)
                else:  # scoop_link — box matching the USD collision (ur5_scoop.stl bounds, mm→m):
                    # STL spans X:±3.75cm, Y:0→24.9cm, Z:±3.75cm → half-extents (0.0375, 0.1245, 0.0375),
                    # centre at (0, 0.1245, 0) in scoop_link frame.
                    # The fixed joint (rpy=π/2,0,π) maps scoop_link Y → wrist_3 Z (tool approach axis).
                    builder.add_shape_box(
                        body=body_idx,
                        hx=0.0375,
                        hy=0.1245,
                        hz=0.0375,
                        xform=wp.transform(wp.vec3(0.0, 0.1245, 0.0), wp.quat_identity()),
                    )

            # ---- Static box container --------------------------------------
            box_off = cfg.box_offset
            box_cfg = newton.ModelBuilder.ShapeConfig(mu=0.6, gap=0.01)
            for _, (hx, hy, hz), centre_local in _BOX_PIECES:
                cx, cy, cz = (
                    centre_local[0] + offset[0] + box_off[0],
                    centre_local[1] + offset[1] + box_off[1],
                    centre_local[2] + offset[2] + box_off[2],
                )
                builder.add_shape_box(
                    body=-1,
                    cfg=box_cfg,
                    xform=wp.transform(wp.vec3(cx, cy, cz), wp.quat_identity()),
                    hx=hx,
                    hy=hy,
                    hz=hz,
                )

            # ---- Particle grid ---------------------------------------------
            box_off_np = np.array(cfg.box_offset, dtype=np.float32)
            emit_lo = np.array(cfg.emit_lo, dtype=np.float32) + offset + box_off_np
            emit_hi = np.array(cfg.emit_hi, dtype=np.float32) + offset + box_off_np
            particle_res = np.ceil(cfg.particles_per_cell * (emit_hi - emit_lo) / cfg.voxel_size).astype(int)
            cell_size = (emit_hi - emit_lo) / particle_res
            radius_p = float(np.max(cell_size) * 0.5)
            mass_p = float(np.prod(cell_size) * cfg.density)
            n_particles = int((particle_res[0] + 1) * (particle_res[1] + 1) * (particle_res[2] + 1))

            env_particle_starts.append(len(particle_env_ids))
            env_particle_counts.append(n_particles)
            particle_env_ids.extend([env_i] * n_particles)

            builder.add_particle_grid(
                pos=wp.vec3(float(emit_lo[0]), float(emit_lo[1]), float(emit_lo[2])),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=int(particle_res[0]) + 1,
                dim_y=int(particle_res[1]) + 1,
                dim_z=int(particle_res[2]) + 1,
                cell_x=float(cell_size[0]),
                cell_y=float(cell_size[1]),
                cell_z=float(cell_size[2]),
                mass=mass_p,
                jitter=cfg.initial_jitter * radius_p,
                radius_mean=radius_p,
            )

        # ---- Finalize model ------------------------------------------------
        self._model = builder.finalize(device=device)
        self._model.set_gravity([0.0, 0.0, -10.0])
        self._model.particle_max_velocity = cfg.particle_max_velocity

        # Populate SolverImplicitMPM.Config from cfg fields
        opts = SolverImplicitMPM.Config()
        for f in dataclasses.fields(cfg):
            if hasattr(opts, f.name):
                setattr(opts, f.name, getattr(cfg, f.name))
        opts.collider_velocity_mode = "finite_difference"  # required for FK robots

        # Populate model.mpm material arrays
        for f in dataclasses.fields(cfg):
            if hasattr(self._model.mpm, f.name):
                getattr(self._model.mpm, f.name).fill_(getattr(cfg, f.name))

        self._state = self._model.state()
        self._solver = SolverImplicitMPM(self._model, opts)
        # Cache zero body-mass array used in every setup_collider() call.
        # All proxy bodies are kinematic (mass=0) from the MPM's perspective.
        self._zero_body_mass = wp.zeros_like(self._model.body_mass)
        self._solver.setup_collider(
            body_mass=self._zero_body_mass,
            body_q=self._state.body_q,
        )

        # ---- Warp bookkeeping arrays ---------------------------------------
        self._robot_link_indices = wp.array(robot_link_indices, dtype=wp.int32, device=device)
        total = sum(env_particle_counts)
        self._total_particles = total
        self._env_particle_start = wp.array(env_particle_starts, dtype=wp.int32, device=device)
        self._env_particle_count = wp.array(env_particle_counts, dtype=wp.int32, device=device)
        self._particle_env_id = wp.array(particle_env_ids, dtype=wp.int32, device=device)

        # Snapshot buffers (populated by settle())
        self._snapshot_q = wp.zeros(total, dtype=wp.vec3f, device=device)
        self._snapshot_qd = wp.zeros(total, dtype=wp.vec3f, device=device)

        print(f"[SandMPMHelper] Built: {num_envs} envs, {total} particles total ({total // num_envs} per env)")

    def settle(
        self,
        body_pos_w: wp.array,
        body_quat_w: wp.array,
        n_steps: int = 120,
    ) -> None:
        """Run MPM-only settling with the arm in its hover pose. Save snapshot.

        Args:
            body_pos_w: Warp view of robot.data.body_pos_w, shape (num_envs, num_bodies).
            body_quat_w: Warp view of robot.data.body_quat_w, shape (num_envs, num_bodies).
            n_steps: Number of MPM steps (~2 s at 60 Hz).
        """
        self.update_proxy_bodies(body_pos_w, body_quat_w, bounded=False)
        # Re-initialize body_q_prev to the hover pose so finite_difference velocity
        # starts at zero on the first MPM step. Without this, body_q_prev would be
        # the origin (set when setup_collider() was called in build()), and the first
        # step would compute a ~18 m/s impulse (origin → hover) that explodes the pile.
        self._solver.setup_collider(
            body_mass=self._zero_body_mass,
            body_q=self._state.body_q,
        )
        dt = 1.0 / 60.0
        for _ in range(n_steps):
            self._solver.step(self._state, self._state, contacts=None, control=None, dt=dt)

        wp.copy(self._snapshot_q, self._state.particle_q)
        wp.copy(self._snapshot_qd, self._state.particle_qd)
        print(f"[SandMPMHelper] Sand settled over {n_steps} steps.")

    def update_proxy_bodies(
        self,
        body_pos_w: wp.array,
        body_quat_w: wp.array,
        bounded: bool = True,
    ) -> None:
        """Overwrite proxy body transforms with current robot link poses.

        Args:
            bounded: If True, clamp translational displacement to cfg.max_proxy_velocity × dt
                so the finite_difference collider velocity stays within the MPM CFL limit.
                Pass False for initial placement (settle, reset) where a large jump is expected
                and body_q_prev will be re-initialised by reinit_collider() immediately after.
        """
        if bounded:
            wp.launch(
                _update_proxy_body_q_bounded,
                dim=(self._num_envs, _NUM_PROXY_PER_ENV),
                inputs=[
                    body_pos_w,
                    body_quat_w,
                    self._robot_link_indices,
                    _NUM_PROXY_PER_ENV,
                    self._max_proxy_disp,
                    self._state.body_q,
                ],
                device=self._device,
            )
        else:
            wp.launch(
                _update_proxy_body_q,
                dim=(self._num_envs, _NUM_PROXY_PER_ENV),
                inputs=[
                    body_pos_w,
                    body_quat_w,
                    self._robot_link_indices,
                    _NUM_PROXY_PER_ENV,
                    self._state.body_q,
                ],
                device=self._device,
            )

    def step(self, dt: float) -> None:
        """Advance the MPM simulation by one control step."""
        self._solver.step(self._state, self._state, contacts=None, control=None, dt=dt)

    def get_env_particle_q(self, env_id: int = 0) -> wp.array:
        """Return a CUDA Warp view of particle positions for one environment.

        Args:
            env_id: Environment index (default 0).

        Returns:
            Warp array of shape (n_particles,), dtype ``wp.vec3f``, on the simulation device.
        """
        start = int(self._env_particle_start.numpy()[env_id])
        count = int(self._env_particle_count.numpy()[env_id])
        particle_q: wp.array = self._state.particle_q
        return particle_q[start : start + count]

    def get_all_particle_q(self) -> wp.array:
        """Return the full CUDA particle position array across all environments.

        Returns:
            Warp array of shape (total_particles,), dtype ``wp.vec3f``, on the simulation device.
        """
        particle_q: wp.array = self._state.particle_q
        return particle_q

    def reset(
        self,
        env_mask: wp.array,
        env_origins: wp.array,
        body_pos_w: wp.array,
        body_quat_w: wp.array,
    ) -> None:
        """Restore settled snapshot for environments in the reset mask."""
        wp.launch(
            _restore_snapshot,
            dim=self._total_particles,
            inputs=[
                self._snapshot_q,
                self._snapshot_qd,
                self._particle_env_id,
                env_mask,
                self._state.particle_q,
                self._state.particle_qd,
            ],
            device=self._device,
        )
        # Re-sync proxy bodies so the restored particles see correct collider positions.
        # setup_collider() cannot be called here — this runs inside a CUDA graph and
        # setup_collider() does a GPU→CPU copy (shape_flags.numpy()) which is illegal
        # during graph capture. The caller handles reinit via reinit_collider().
        # Use bounded=False: the robot teleports to its reset pose, which is a large
        # discontinuous jump. reinit_collider() (called by the env one step later) will
        # zero body_q_prev, so the next MPM step sees velocity ≈ 0 regardless.
        self.update_proxy_bodies(body_pos_w, body_quat_w, bounded=False)

    def reinit_collider(self) -> None:
        """Reset the solver's internal body_q_prev to the current proxy body positions.

        Must be called **outside** any CUDA graph after proxy bodies have been updated.
        This prevents a finite_difference velocity spike when body positions jump
        discontinuously — e.g. the first MPM step after an episode reset, where the
        robot teleports from its end-of-episode pose to the new hover pose.

        Call pattern in scoop_env_warp._post_step_visualize():
            if reset_happened_last_step:
                self._sand.reinit_collider()   # body_q_prev = current hover pose
            self._sand.step(...)               # velocity = (hover+δ − hover)/dt ≈ 0 ✓
        """
        self._solver.setup_collider(
            body_mass=self._zero_body_mass,
            body_q=self._state.body_q,
        )

    def compute_obs(
        self,
        env_origins: wp.array,
        ee_pos_local: wp.array,
        observations: wp.array,
        centroid_out: wp.array,
        lift_threshold: float = 0.03,
        obs_offset: int = 25,
    ) -> None:
        """Write 8 sand feature dims into observations starting at obs_offset.

        Also writes env-local sand centroid to centroid_out.

        Features written (env-local):
            [0:3]  sand centroid xyz
            [3:6]  scoop → centroid vector
            [6]    displaced_fraction
            [7]    elevated_fraction
        """
        wp.launch(
            _compute_sand_obs,
            dim=self._num_envs,
            inputs=[
                self._state.particle_q,
                self._snapshot_q,
                env_origins,
                ee_pos_local,
                self._env_particle_start,
                self._env_particle_count,
                0.02,  # displaced_threshold [m]
                lift_threshold,
                observations,
                obs_offset,
                centroid_out,
            ],
            device=self._device,
        )

    def compute_centroids(self, env_origins: wp.array, centroid_out: wp.array) -> None:
        """Write env-local sand centroids to centroid_out (lightweight, no obs write)."""
        wp.launch(
            _compute_centroids,
            dim=self._num_envs,
            inputs=[
                self._state.particle_q,
                env_origins,
                self._env_particle_start,
                self._env_particle_count,
                centroid_out,
            ],
            device=self._device,
        )

    def add_scoop_rewards(
        self,
        env_origins: wp.array,
        ee_pos_local: wp.array,
        rewards: wp.array,
        lift_threshold: float,
        capture_radius_xy: float,
        elevation_scale: float,
        captured_scale: float,
    ) -> None:
        """Add elevation and captured-elevated reward components in-place to rewards."""
        wp.launch(
            _add_scoop_rewards,
            dim=self._num_envs,
            inputs=[
                self._state.particle_q,
                self._snapshot_q,
                env_origins,
                ee_pos_local,
                self._env_particle_start,
                self._env_particle_count,
                lift_threshold,
                capture_radius_xy,
                elevation_scale,
                captured_scale,
                rewards,
            ],
            device=self._device,
        )
