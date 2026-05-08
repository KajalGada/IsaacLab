# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton MPM + MuJoCo kinetic-sand scooping environment.

Physics backend
---------------
* ``SolverImplicitMPM``  — kinetic-sand particles (particle↔rigid collision)
* ``SolverMuJoCo``       — UR5 arm rigid-body dynamics (PD joint control)

Per-step flow
-------------
1. Scale raw action ∈ [-1,1]⁶ by ``action_scale`` and add to joint targets.
2. Run MuJoCo solver for ``sim_substeps`` sub-steps (robot dynamics).
3. Run MPM solver once at ``frame_dt`` (sand particle update).
4. Read observations from ``state_0``.

Episode reset
-------------
Particle positions/velocities and the MPM deformation-gradient tensors
(``particle_Jp``, ``particle_elastic_strain``) are restored to their
stored initial values.  The robot joint state is reset to ``_Q_ABOVE``.

Note: the Newton MPM documentation marks ``begin_world()``/``end_world()``
batching as "not yet available"; multi-environment GPU parallelism requires
Python ``multiprocessing`` (e.g., SB3 ``SubprocVecEnv``).
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverImplicitMPM, SolverMuJoCo

from .sand_scoop_env_cfg import SandScoopEnvCfg

# ---------------------------------------------------------------------------
# Robot initial pose (hover above sand pile while particles settle)
# joint order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
# ---------------------------------------------------------------------------
_Q_ABOVE = np.array([-0.27, -1.50, 1.70, math.pi / 2, math.pi / 2, math.pi / 2], dtype=np.float32)

# Joint position limits for clipping after applying delta actions [rad]
_JOINT_LO = np.full(6, -math.pi, dtype=np.float32)
_JOINT_HI = np.full(6, math.pi, dtype=np.float32)

# UR5 DOF count
_UR5_DOF = 6


class SandScoopEnv(gym.Env):
    """Kinetic-sand scooping RL environment backed by Newton MPM + MuJoCo.

    Observation space (20-D float32):
        * ``joint_q``          (6)  — joint positions [rad]
        * ``joint_qd``         (6)  — joint velocities [rad/s]
        * ``ee_pos``           (3)  — EE (scoop) position in world frame [m]
        * ``sand_centroid``    (3)  — mean particle position [m]
        * ``sand_z_mean``      (1)  — mean particle z [m]
        * ``sand_frac_above``  (1)  — fraction of particles above
          ``cfg.scoop_z_threshold``

    Action space (6-D float32 ∈ [-1, 1]):
        Joint-position deltas, scaled by ``cfg.action_scale`` [rad/step].

    Reward:
        ``rew = w_Δz · Δ(mean_z) + w_frac · frac_above``
        ``    − w_action · ‖action‖² + w_alive``

    Termination:
        Episode length exceeded (``cfg.episode_length_s · cfg.fps`` steps).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: SandScoopEnvCfg | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg or SandScoopEnvCfg()
        self.render_mode = render_mode

        self._frame_dt: float = 1.0 / self.cfg.fps
        self._sim_dt: float = self._frame_dt / self.cfg.sim_substeps
        self._max_steps: int = int(self.cfg.episode_length_s * self.cfg.fps)

        # Clip range for joint targets after applying action delta
        self._joint_lo = np.full(_UR5_DOF, self.cfg.joint_pos_limit[0], dtype=np.float32)
        self._joint_hi = np.full(_UR5_DOF, self.cfg.joint_pos_limit[1], dtype=np.float32)

        self._viewer = None
        self._sim_time: float = 0.0

        self._build_scene()

        if self.render_mode == "human":
            self._setup_viewer()

        self._setup_spaces()

        # Episode counters
        self._step_count: int = 0
        self._prev_sand_z_mean: float = 0.0

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def _build_scene(self) -> None:
        """Build Newton model, create states, and instantiate both solvers."""
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)

        # ---- UR5 + scoop -------------------------------------------------
        cfg = self.cfg
        builder.add_urdf(
            cfg.urdf_path,
            xform=wp.transform(
                wp.vec3(*cfg.robot_base_pos),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi),
            ),
            floating=False,
            enable_self_collisions=False,
        )

        builder.joint_q[:_UR5_DOF] = _Q_ABOVE.tolist()

        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = cfg.pd_stiffness
            builder.joint_target_kd[i] = cfg.pd_damping

        # Track the scoop body: it is the last body added by the URDF.
        self._ee_body_idx: int = len(builder.body_mass) - 1

        # ---- Ground plane ------------------------------------------------
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        # ---- Static container box ----------------------------------------
        self._add_container(builder)

        # ---- Sand particles ----------------------------------------------
        n_particles = self._add_sand(builder)
        self._n_particles: int = n_particles

        # ---- Finalize model ----------------------------------------------
        self.model = builder.finalize()
        self.model.set_gravity(wp.vec3(*cfg.gravity))

        # ---- Material properties on model.mpm ---------------------------
        _MATERIAL_KEYS = (
            "young_modulus",
            "poisson_ratio",
            "friction",
            "damping",
            "yield_pressure",
            "tensile_yield_ratio",
            "yield_stress",
            "hardening",
        )
        for key in _MATERIAL_KEYS:
            if hasattr(self.model.mpm, key):
                getattr(self.model.mpm, key).fill_(getattr(cfg, key))

        # ---- MPM solver Config -------------------------------------------
        mpm_cfg = SolverImplicitMPM.Config()
        _MPM_CONFIG_KEYS = {
            "voxel_size": cfg.voxel_size,
            "grid_type": cfg.mpm_grid_type,
            "max_iterations": cfg.mpm_max_iterations,
            "tolerance": cfg.mpm_tolerance,
            "transfer_scheme": cfg.mpm_transfer_scheme,
            "solver": cfg.mpm_solver,
            "strain_basis": cfg.mpm_strain_basis,
            "collider_basis": cfg.mpm_collider_basis,
            "grid_padding": cfg.mpm_grid_padding,
            "max_active_cell_count": cfg.mpm_max_active_cell_count,
            "air_drag": cfg.air_drag,
            "critical_fraction": cfg.critical_fraction,
            "collider_velocity_mode": "finite_difference",
        }
        for key, val in _MPM_CONFIG_KEYS.items():
            if hasattr(mpm_cfg, key):
                setattr(mpm_cfg, key, val)
        self._mpm_cfg = mpm_cfg

        # ---- Simulation states (ping-pong buffers) -----------------------
        self._state_buf_0 = self.model.state()
        self._state_buf_1 = self.model.state()
        self.state_0 = self._state_buf_0
        self.state_1 = self._state_buf_1

        # ---- Forward kinematics ------------------------------------------
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # ---- Snapshot initial state for fast reset -----------------------
        self._init_particle_q = self.state_0.particle_q.numpy().copy()
        self._init_particle_qd = np.zeros_like(self._init_particle_q)
        self._init_joint_q = self.state_0.joint_q.numpy().copy()
        self._init_joint_qd = np.zeros(len(self.state_0.joint_qd), dtype=np.float32)

        # Deformation-gradient tensors (particle_Jp = 1, elastic_strain = I)
        self._init_particle_Jp = self.model.mpm.particle_Jp.numpy().copy()
        self._init_particle_elastic_strain = self.model.mpm.particle_elastic_strain.numpy().copy()

        # ---- Instantiate solvers -----------------------------------------
        self.mpm_solver = SolverImplicitMPM(self.model, mpm_cfg)
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )
        self.robot_solver = SolverMuJoCo(self.model)

        # ---- PD control object -------------------------------------------
        self.control = self.model.control()
        pos_np = self.control.joint_target_pos.numpy()
        pos_np[:_UR5_DOF] = _Q_ABOVE
        self.control.joint_target_pos.assign(pos_np)
        self._joint_targets = _Q_ABOVE.copy()

    def _setup_viewer(self) -> None:
        """Create a Newton GL viewer and attach it to the current model."""
        from newton.viewer import ViewerGL

        self._viewer = ViewerGL(width=1280, height=720, vsync=True)
        self._viewer.set_model(self.model)
        self._viewer.show_particles = True
        # Camera convention: pitch and yaw are in DEGREES (Z-up coordinate system).
        # yaw=135° → camera faces the scene from the +x/−y quadrant.
        # pitch=−25° → looking slightly downward.
        # pos sits above and to the side of the robot + sand box.
        self._viewer.set_camera(
            pos=wp.vec3(2.0, -1.5, 1.2),
            pitch=-25.0,
            yaw=135.0,
        )

    # ------------------------------------------------------------------
    # Viewer property
    # ------------------------------------------------------------------
    @property
    def viewer_is_running(self) -> bool:
        """True while the GL window is open (always True when no viewer)."""
        if self._viewer is None:
            return True
        return self._viewer.is_running()

    def _add_container(self, builder: newton.ModelBuilder) -> None:
        """Add static box container (floor + 4 walls) to ``builder``."""
        cfg = self.cfg
        w = cfg.box_width
        d = cfg.box_depth
        h = cfg.box_height
        t = cfg.wall_thickness
        box_cfg = newton.ModelBuilder.ShapeConfig(mu=cfg.box_mu, gap=cfg.box_gap)

        def _box(xform, hx, hy, hz):
            builder.add_shape_box(body=-1, cfg=box_cfg, xform=xform, hx=hx, hy=hy, hz=hz)

        # Floor
        _box(wp.transform(wp.vec3(0.0, 0.0, t * 0.5), wp.quat_identity()), w * 0.5, d * 0.5, t * 0.5)
        # Front wall (−y)
        _box(wp.transform(wp.vec3(0.0, -d * 0.5, h * 0.5), wp.quat_identity()), w * 0.5, t * 0.5, h * 0.5)
        # Back wall (+y)
        _box(wp.transform(wp.vec3(0.0, d * 0.5, h * 0.5), wp.quat_identity()), w * 0.5, t * 0.5, h * 0.5)
        # Left wall (−x)
        _box(wp.transform(wp.vec3(-w * 0.5, 0.0, h * 0.5), wp.quat_identity()), t * 0.5, d * 0.5, h * 0.5)
        # Right wall (+x)
        _box(wp.transform(wp.vec3(w * 0.5, 0.0, h * 0.5), wp.quat_identity()), t * 0.5, d * 0.5, h * 0.5)

    def _add_sand(self, builder: newton.ModelBuilder) -> int:
        """Add the initial sand particle grid to ``builder``.

        Returns the total number of particles created.
        """
        cfg = self.cfg
        lo = np.array(cfg.emit_lo, dtype=np.float32)
        hi = np.array(cfg.emit_hi, dtype=np.float32)
        ppc = cfg.particles_per_cell
        vs = cfg.voxel_size

        res = np.array(np.ceil(ppc * (hi - lo) / vs), dtype=int)
        cell_size = (hi - lo) / res
        cell_volume = float(np.prod(cell_size))
        radius = float(np.max(cell_size) * 0.5)
        mass = cell_volume * cfg.density

        builder.add_particle_grid(
            pos=wp.vec3(lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(res[0]) + 1,
            dim_y=int(res[1]) + 1,
            dim_z=int(res[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass,
            jitter=cfg.initial_jitter * radius,
            radius_mean=radius,
        )
        return (res[0] + 1) * (res[1] + 1) * (res[2] + 1)

    # ------------------------------------------------------------------
    # Gymnasium spaces
    # ------------------------------------------------------------------
    def _setup_spaces(self) -> None:
        obs_dim = _UR5_DOF + _UR5_DOF + 3 + 3 + 1 + 1  # = 20
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(_UR5_DOF,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # ---- Restore particle state ------------------------------------
        for state in (self._state_buf_0, self._state_buf_1):
            state.particle_q.assign(self._init_particle_q)
            state.particle_qd.assign(self._init_particle_qd)

        # ---- Restore MPM deformation-gradient tensors ------------------
        self.model.mpm.particle_Jp.assign(self._init_particle_Jp)
        self.model.mpm.particle_elastic_strain.assign(self._init_particle_elastic_strain)

        # ---- Restore robot joint state ---------------------------------
        init_jq = self._init_joint_q.copy()
        init_jqd = self._init_joint_qd.copy()
        for state in (self._state_buf_0, self._state_buf_1):
            state.joint_q.assign(init_jq)
            state.joint_qd.assign(init_jqd)

        # ---- Reset working references ----------------------------------
        self.state_0 = self._state_buf_0
        self.state_1 = self._state_buf_1

        # ---- Re-run FK so body_q reflects the reset joint positions ----
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # ---- Reset PD targets ------------------------------------------
        pos_np = self.control.joint_target_pos.numpy()
        pos_np[:_UR5_DOF] = _Q_ABOVE
        self.control.joint_target_pos.assign(pos_np)
        self._joint_targets = _Q_ABOVE.copy()

        # ---- Re-initialize MPM collider with reset body positions ------
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )

        # ---- Episode counters ------------------------------------------
        self._step_count = 0
        particle_q = self.state_0.particle_q.numpy()
        self._prev_sand_z_mean = float(np.mean(particle_q[:, 2]))

        obs = self._get_obs()

        # Draw the reset state immediately so the window is not blank.
        self.render()

        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # ---- Apply action: joint-position delta -------------------------
        self._joint_targets = np.clip(
            self._joint_targets + action * self.cfg.action_scale,
            self._joint_lo,
            self._joint_hi,
        )
        pos_np = self.control.joint_target_pos.numpy()
        pos_np[:_UR5_DOF] = self._joint_targets
        self.control.joint_target_pos.assign(pos_np)

        # ---- Simulate robot (MuJoCo, multiple substeps) -----------------
        for _ in range(self.cfg.sim_substeps):
            self.state_0.clear_forces()
            if self._viewer is not None:
                self._viewer.apply_forces(self.state_0)
            self.robot_solver.step(self.state_0, self.state_1, self.control, contacts=None, dt=self._sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # ---- Simulate sand (MPM, one step at frame_dt) ------------------
        self.mpm_solver.step(
            self.state_0, self.state_0, contacts=None, control=None, dt=self._frame_dt
        )

        self._sim_time += self._frame_dt

        # ---- Observations, reward, termination --------------------------
        obs = self._get_obs()
        reward, info = self._compute_reward(action)
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self._max_steps

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._viewer is None:
            return
        self._viewer.begin_frame(self._sim_time)
        self._viewer.log_state(self.state_0)
        self._viewer.end_frame()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        joint_q = self.state_0.joint_q.numpy()[:_UR5_DOF]
        joint_qd = self.state_0.joint_qd.numpy()[:_UR5_DOF]

        # EE position: last robot body (scoop link), first 3 elements of transform
        body_q = self.state_0.body_q.numpy()
        ee_pos = body_q[self._ee_body_idx][:3].astype(np.float32)

        # Sand statistics
        particle_q = self.state_0.particle_q.numpy()  # (N, 3)
        centroid = np.mean(particle_q, axis=0).astype(np.float32)
        z_mean = np.mean(particle_q[:, 2], keepdims=True).astype(np.float32)
        frac_above = np.array(
            [np.mean(particle_q[:, 2] > self.cfg.scoop_z_threshold)], dtype=np.float32
        )

        return np.concatenate([joint_q, joint_qd, ee_pos, centroid, z_mean, frac_above])

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict]:
        particle_q = self.state_0.particle_q.numpy()
        z_mean = float(np.mean(particle_q[:, 2]))
        frac_above = float(np.mean(particle_q[:, 2] > self.cfg.scoop_z_threshold))

        delta_z = z_mean - self._prev_sand_z_mean
        self._prev_sand_z_mean = z_mean

        rew = (
            self.cfg.rew_height_delta * delta_z
            + self.cfg.rew_frac_above * frac_above
            - self.cfg.rew_action_penalty * float(np.sum(action ** 2))
            + self.cfg.rew_alive
        )
        info = {
            "sand_z_mean": z_mean,
            "sand_frac_above": frac_above,
            "delta_z": delta_z,
        }
        return float(rew), info
