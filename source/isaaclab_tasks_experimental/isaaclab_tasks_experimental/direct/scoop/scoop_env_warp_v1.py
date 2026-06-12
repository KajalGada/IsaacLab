# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-Scoop-Direct-Warp-v1: standalone self-contained environment.

The only intra-package dependency is sand_mpm.py.

Physics: Newton MuJoCo-Warp solver (robot) + separate Newton implicit MPM (sand),
one-way coupled — scoop poses are copied to kinematic proxy bodies each step.
Voxel size 0.02 m with default sand material parameters. Sandbox shifted via
box_offset=(1.0, 0.1, 0.0).
"""

from __future__ import annotations

import os

import warp as wp
from isaaclab_experimental.envs import DirectRLEnvWarp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .sand_mpm import UR5_PROXY_LINK_NAMES, SandMPMCfg, SandMPMHelper

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
_UR5_SCOOP_USD = os.path.join(_ASSETS_DIR, "ur5_with_scoop.usd")

# Sand-box geometry (matches simulation_newton_sand_v3.py + (x=+1.0, y=-0.1) scene offset)
_BOX_W = 0.35   # x full width
_BOX_D = 0.35   # y full depth
_BOX_WALL_H = 0.05   # z wall height
_BOX_FLOOR_T = 0.02  # floor / wall thickness
_BOX_X = 1.0    # box centre x
_BOX_Y = 0.1    # box centre y  (ref -0.2 + scene offset -0.1 + 0.4)

# Static box pieces: (name, full_size_xyz, centre_xyz) — env-local coords
_BOX_PIECES: list[tuple[str, tuple, tuple]] = [
    ("floor",     (_BOX_W,      _BOX_D,      _BOX_FLOOR_T), (_BOX_X,            _BOX_Y,              _BOX_FLOOR_T / 2)),
    ("wall_neg_y",(_BOX_W,      _BOX_FLOOR_T,_BOX_WALL_H),  (_BOX_X,            _BOX_Y - _BOX_D / 2, _BOX_WALL_H / 2)),
    ("wall_pos_y",(_BOX_W,      _BOX_FLOOR_T,_BOX_WALL_H),  (_BOX_X,            _BOX_Y + _BOX_D / 2, _BOX_WALL_H / 2)),
    ("wall_neg_x",(_BOX_FLOOR_T,_BOX_D,      _BOX_WALL_H),  (_BOX_X - _BOX_W / 2, _BOX_Y,            _BOX_WALL_H / 2)),
    ("wall_pos_x",(_BOX_FLOOR_T,_BOX_D,      _BOX_WALL_H),  (_BOX_X + _BOX_W / 2, _BOX_Y,            _BOX_WALL_H / 2)),
]

# UR5+scoop hovering above the sand box.
# Initial pose matches _Q_ABOVE from the Newton reference script.
# Robot base at (0.5, 0, 0), rotated 180° around Z so the arm faces the box.
UR5_SCOOP_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_UR5_SCOOP_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.0),
        rot=(0.0, 0.0, 1.0, 0.0),  # 180° around Z so arm faces the box at origin
        joint_pos={
            "shoulder_pan_joint": -0.27,
            "shoulder_lift_joint": -1.50,
            "elbow_joint": 1.70,
            "wrist_1_joint": 1.5708,
            "wrist_2_joint": 1.5708,
            "wrist_3_joint": 1.5708,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*_joint", "elbow_joint", "wrist_.*_joint"],
            stiffness=2000.0,
            damping=100.0,
            effort_limit_sim=150.0,
        ),
    },
)


# ---------------------------------------------------------------------------
# Warp helper kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _initialize_rng(seed: wp.int32, state: wp.array(dtype=wp.uint32)):
    env_id = wp.tid()
    state[env_id] = wp.rand_init(seed, wp.int32(env_id))


@wp.kernel
def _compute_ee_pos(
    body_pos_w: wp.array2d(dtype=wp.vec3f),
    env_origins: wp.array(dtype=wp.vec3f),
    scoop_body_idx: wp.int32,
    ee_pos_local: wp.array(dtype=wp.vec3f),
):
    env_id = wp.tid()
    ee_pos_local[env_id] = body_pos_w[env_id, scoop_body_idx] - env_origins[env_id]


@wp.kernel
def _scale_actions(
    actions: wp.array2d(dtype=wp.float32),
    joint_limits: wp.array2d(dtype=wp.vec2f),
    joint_targets: wp.array2d(dtype=wp.float32),
):
    """Map actions from [-1, 1] to each joint's [lower, upper] range."""
    env_id, dof_id = wp.tid()
    a = wp.clamp(actions[env_id, dof_id], wp.float32(-1.0), wp.float32(1.0))
    lower = joint_limits[env_id, dof_id][0]
    upper = joint_limits[env_id, dof_id][1]
    joint_targets[env_id, dof_id] = wp.float32(0.5) * (a + wp.float32(1.0)) * (upper - lower) + lower


@wp.kernel
def _compute_observations(
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    joint_limits: wp.array2d(dtype=wp.vec2f),
    actions: wp.array2d(dtype=wp.float32),
    ee_pos_local: wp.array(dtype=wp.vec3f),
    body_quat_w: wp.array2d(dtype=wp.quatf),
    scoop_body_idx: wp.int32,
    dof_vel_scale: wp.float32,
    num_dofs: wp.int32,
    observations: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    # [0:6] joint positions scaled to [-1, 1]
    for i in range(num_dofs):
        lower = joint_limits[env_id, i][0]
        upper = joint_limits[env_id, i][1]
        observations[env_id, i] = (wp.float32(2.0) * joint_pos[env_id, i] - upper - lower) / (upper - lower)
    # [6:12] joint velocities
    for i in range(num_dofs):
        observations[env_id, num_dofs + i] = joint_vel[env_id, i] * dof_vel_scale
    # [12:18] previous actions
    for i in range(num_dofs):
        observations[env_id, wp.int32(2) * num_dofs + i] = actions[env_id, i]
    # [18:21] EE position (env-local)
    base = wp.int32(3) * num_dofs
    observations[env_id, base + 0] = ee_pos_local[env_id][0]
    observations[env_id, base + 1] = ee_pos_local[env_id][1]
    observations[env_id, base + 2] = ee_pos_local[env_id][2]
    # [21:25] scoop quaternion components (4 dims — consistent ordering for the policy network)
    q = body_quat_w[env_id, scoop_body_idx]
    observations[env_id, base + 3] = q[0]
    observations[env_id, base + 4] = q[1]
    observations[env_id, base + 5] = q[2]
    observations[env_id, base + 6] = q[3]


@wp.kernel
def _compute_rewards(
    ee_pos_local: wp.array(dtype=wp.vec3f),
    sand_centroid: wp.array(dtype=wp.vec3f),
    actions: wp.array2d(dtype=wp.float32),
    num_dofs: wp.int32,
    dist_scale: wp.float32,
    action_penalty_scale: wp.float32,
    alive_reward: wp.float32,
    rewards: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    # Approach: reward getting the scoop close to the sand centroid
    dist = wp.length(ee_pos_local[env_id] - sand_centroid[env_id])
    action_penalty = wp.float32(0.0)
    for i in range(num_dofs):
        action_penalty += actions[env_id, i] * actions[env_id, i]
    rewards[env_id] = dist_scale * dist + action_penalty_scale * action_penalty + alive_reward


@wp.kernel
def _get_dones(
    episode_length_buf: wp.array(dtype=wp.int32),
    max_episode_length: wp.int32,
    time_out: wp.array(dtype=wp.bool),
    reset_buf: wp.array(dtype=wp.bool),
):
    env_id = wp.tid()
    time_out[env_id] = episode_length_buf[env_id] >= (max_episode_length - 1)
    reset_buf[env_id] = time_out[env_id]


@wp.kernel
def _reset_actions_for_mask(
    env_mask: wp.array(dtype=wp.bool),
    num_dofs: wp.int32,
    actions: wp.array2d(dtype=wp.float32),
):
    """Zero out the smoothed-action buffer for envs in the reset mask."""
    env_id = wp.tid()
    if env_mask[env_id]:
        for i in range(num_dofs):
            actions[env_id, i] = wp.float32(0.0)


@wp.kernel
def _smooth_actions(
    raw: wp.array2d(dtype=wp.float32),
    prev: wp.array2d(dtype=wp.float32),
    alpha: wp.float32,
    out: wp.array2d(dtype=wp.float32),
):
    """Exponential moving average: out = alpha * raw + (1 - alpha) * prev."""
    env_id, dof_id = wp.tid()
    out[env_id, dof_id] = alpha * raw[env_id, dof_id] + (wp.float32(1.0) - alpha) * prev[env_id, dof_id]


@wp.kernel
def _reset_joints(
    default_joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_vel: wp.array2d(dtype=wp.float32),
    reset_pos_noise: wp.float32,
    num_dofs: wp.int32,
    env_mask: wp.array(dtype=wp.bool),
    rng_state: wp.array(dtype=wp.uint32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
):
    env_id = wp.tid()
    if env_mask[env_id]:
        for i in range(num_dofs):
            noise = wp.randf(rng_state[env_id], -reset_pos_noise, reset_pos_noise)
            rng_state[env_id] += wp.uint32(1)
            joint_pos[env_id, i] = default_joint_pos[env_id, i] + noise
            joint_vel[env_id, i] = default_joint_vel[env_id, i]


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


@configclass
class ScoopWarpEnvCfgV1(DirectRLEnvCfg):
    # env
    episode_length_s: float = 10.0
    decimation: int = 2
    action_space: int = 6
    # 6 joint pos + 6 joint vel + 6 prev actions + 3 EE pos + 4 scoop quat
    # + 3 sand centroid + 3 scoop→centroid + 1 displaced_frac + 1 elevated_frac = 33
    observation_space: int = 33
    state_space: int = 0

    # Newton MuJoCo-Warp solver
    solver_cfg = MJWarpSolverCfg(
        njmax=80,
        nconmax=20,
        cone="pyramidal",
        integrator="implicitfast",
        impratio=1,
        save_to_mjcf="/tmp/scoop_newton_debug.xml",
    )
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
        ),
        physics=newton_cfg,
    )

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene — num_envs reduced from 512: MPM is significantly more expensive
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # robot
    robot: ArticulationCfg = UR5_SCOOP_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore[attr-defined]

    # sand MPM — sandbox shifted to the front of the robot (+x=1.0, +y=0.1)
    sand: SandMPMCfg = SandMPMCfg(box_offset=(1.0, 0.1, 0.0))

    # task
    scoop_body_name: str = "scoop_link"

    # reward scales
    dist_reward_scale: float = -0.5  # approach: penalises distance scoop → sand centroid
    action_penalty_scale: float = -0.001
    alive_reward: float = 0.05
    elevation_reward_scale: float = 4.0   # fraction of particles lifted > lift_threshold
    captured_elevated_scale: float = 3.0  # fraction of elevated particles near scoop
    lift_threshold: float = 0.03          # [m] above settled z to count as elevated
    capture_radius_xy: float = 0.08       # [m] horizontal radius for captured-elevated reward

    # observation scaling
    dof_vel_scale: float = 0.1

    # reset
    reset_dof_pos_noise: float = 0.05  # radians noise around default pose

    # action smoothing — exponential moving average applied in _pre_physics_step.
    # Limits how fast joints can move between steps: scoop velocity ≈ raw_velocity × alpha.
    # alpha=1.0 = no smoothing; alpha=0.2 = ~5-step lag, reduces 15 m/s → ~3 m/s.
    action_smoothing: float = 0.2


# ---------------------------------------------------------------------------
# Environment implementation
# ---------------------------------------------------------------------------


class ScoopWarpEnvV1(DirectRLEnvWarp):
    cfg: ScoopWarpEnvCfgV1

    def __init__(self, cfg: ScoopWarpEnvCfgV1, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Scoop body index in the articulation body list
        if self.cfg.scoop_body_name in self.robot.body_names:
            self._scoop_body_idx = self.robot.body_names.index(self.cfg.scoop_body_name)
        else:
            self._scoop_body_idx = self.robot.body_names.index("wrist_3_link")

        # Warp views into Newton simulation data (zero-copy)
        self._joint_pos = self.robot.data.joint_pos.warp
        self._joint_vel = self.robot.data.joint_vel.warp
        self._body_pos_w = self.robot.data.body_pos_w.warp
        self._body_quat_w = self.robot.data.body_quat_w.warp
        self._joint_limits = self.robot.data.soft_joint_pos_limits.warp

        # Env-local origins
        self._env_origins = wp.from_torch(self.scene.env_origins, dtype=wp.vec3f)

        # Sand MPM — build separate Newton model and settle particles
        robot_link_indices = [self.robot.body_names.index(name) for name in UR5_PROXY_LINK_NAMES]
        # MPM control-step dt = physics dt × decimation
        self._mpm_dt: float = self.cfg.sim.dt * self.cfg.decimation

        self._sand = SandMPMHelper()
        self._sand.build(
            num_envs=self.num_envs,
            env_origins=self.scene.env_origins,
            robot_link_indices=robot_link_indices,
            cfg=self.cfg.sand,
            device=self.device,
            mpm_dt=self._mpm_dt,
        )
        self._sand.settle(
            body_pos_w=self._body_pos_w,
            body_quat_w=self._body_quat_w,
            n_steps=self.cfg.sand.settle_steps,
        )

        # Pre-allocated CUDA arrays for log_points (avoids per-frame GPU allocation).
        # log_points requires wp.array for radii/colors — Python float/tuple cause a kernel error.
        _n_total = self._sand.get_all_particle_q().shape[0]
        _r = float(self.cfg.sand.voxel_size * 0.5)
        self._particle_radii = wp.full(_n_total, _r, dtype=wp.float32, device=self.device)
        self._particle_colors = wp.full(_n_total, wp.vec3(0.85, 0.75, 0.45), dtype=wp.vec3, device=self.device)

        # Persistent warp buffers
        self._ee_pos_local = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        self._actions = wp.zeros((self.num_envs, self.cfg.action_space), dtype=wp.float32, device=self.device)
        self._joint_targets = wp.zeros((self.num_envs, self.robot.num_joints), dtype=wp.float32, device=self.device)
        self._observations = wp.zeros((self.num_envs, self.cfg.observation_space), dtype=wp.float32, device=self.device)
        self._rewards = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)

        # Per-env RNG state for reset noise
        self._rng_state = wp.zeros(self.num_envs, dtype=wp.uint32, device=self.device)
        seed = self.cfg.seed if self.cfg.seed is not None else 0
        wp.launch(_initialize_rng, dim=self.num_envs, inputs=[seed, self._rng_state], device=self.device)

        # Torch-aliased buffers expected by DirectRLEnvWarp base class
        self.torch_obs_buf = wp.to_torch(self._observations)
        self.torch_reward_buf = wp.to_torch(self._rewards)
        self.torch_reset_terminated = wp.to_torch(self.reset_terminated)
        self.torch_reset_time_outs = wp.to_torch(self.reset_time_outs)

        # Smoothed action buffer — holds the EMA-filtered actions from the previous step.
        # Initialised to zeros (hover pose actions are near-zero after scaling).
        self._actions_prev = wp.zeros((self.num_envs, self.cfg.action_space), dtype=wp.float32, device=self.device)

        # Sand centroid buffer — filled by compute_centroids (rewards) and compute_obs (obs).
        self._sand_centroid = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)

        # Flag: reinit MPM collider body_q_prev the step AFTER a reset.
        # setup_collider() cannot run inside the CUDA graph (does GPU→CPU copies),
        # so we detect resets outside the graph and apply reinit one step later.
        self._collider_reinit_pending = False

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot)

        # Flat ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Sand container box — 5 static collision pieces spawned at env_0 before cloning
        self._spawn_sand_box()

        # Clone env_0 into all environments
        self.scene.clone_environments(copy_from_source=False)

        # Register robot with scene manager
        self.scene.articulations["robot"] = self.robot

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_sand_box(self) -> None:
        """Spawn 5 static cuboids forming the sand container at env_0."""
        box_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.6,
            dynamic_friction=0.4,
        )
        off = self.cfg.sand.box_offset
        for name, size, centre in _BOX_PIECES:
            piece_cfg = sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=box_material,
            )
            piece_cfg.func(
                f"/World/envs/env_0/SandBox/{name}",
                piece_cfg,
                translation=(centre[0] + off[0], centre[1] + off[1], centre[2] + off[2]),
            )

    # ------------------------------------------------------------------
    # Step callbacks
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: wp.array) -> None:
        # Exponential moving average smoothing: limits scoop velocity seen by the MPM.
        # Without smoothing a random agent can drive the scoop at ~15 m/s; the MPM
        # imposes that as a boundary condition on contact particles → they fly.
        # alpha=0.2 reduces effective scoop velocity to ~20% per step.
        wp.launch(
            _smooth_actions,
            dim=(self.num_envs, self.cfg.action_space),
            inputs=[actions, self._actions_prev, self.cfg.action_smoothing, self._actions],
            device=self.device,
        )
        wp.copy(self._actions_prev, self._actions)

    def _apply_action(self) -> None:
        wp.launch(
            _scale_actions,
            dim=(self.num_envs, self.robot.num_joints),
            inputs=[self._actions, self._joint_limits, self._joint_targets],
            device=self.device,
        )
        self.robot.set_joint_position_target_mask(target=self._joint_targets)

    def _get_observations(self) -> dict:
        self._update_ee_pos()
        wp.launch(
            _compute_observations,
            dim=self.num_envs,
            inputs=[
                self._joint_pos,
                self._joint_vel,
                self._joint_limits,
                self._actions,
                self._ee_pos_local,
                self._body_quat_w,
                self._scoop_body_idx,
                self.cfg.dof_vel_scale,
                self.robot.num_joints,
                self._observations,
            ],
            device=self.device,
        )
        # dims [25:33]: sand centroid (3), scoop→centroid (3), displaced_frac (1), elevated_frac (1)
        self._sand.compute_obs(
            env_origins=self._env_origins,
            ee_pos_local=self._ee_pos_local,
            observations=self._observations,
            centroid_out=self._sand_centroid,
            lift_threshold=self.cfg.lift_threshold,
            obs_offset=25,
        )
        return {"policy": self.torch_obs_buf}

    def _get_rewards(self) -> None:
        # Compute sand centroid for approach reward (obs not yet computed this step)
        self._sand.compute_centroids(
            env_origins=self._env_origins,
            centroid_out=self._sand_centroid,
        )
        wp.launch(
            _compute_rewards,
            dim=self.num_envs,
            inputs=[
                self._ee_pos_local,
                self._sand_centroid,
                self._actions,
                self.robot.num_joints,
                self.cfg.dist_reward_scale,
                self.cfg.action_penalty_scale,
                self.cfg.alive_reward,
                self._rewards,
            ],
            device=self.device,
        )
        # Add elevation and captured-elevated rewards
        self._sand.add_scoop_rewards(
            env_origins=self._env_origins,
            ee_pos_local=self._ee_pos_local,
            rewards=self._rewards,
            lift_threshold=self.cfg.lift_threshold,
            capture_radius_xy=self.cfg.capture_radius_xy,
            elevation_scale=self.cfg.elevation_reward_scale,
            captured_scale=self.cfg.captured_elevated_scale,
        )

    def _post_step_visualize(self) -> None:
        # If a reset happened last step, reinit the MPM collider's body_q_prev NOW
        # (before the MPM step) so finite_difference velocity = (hover − hover)/dt ≈ 0.
        # Without this, the first post-reset MPM step would compute velocity from the
        # end-of-episode pose to the new hover pose → huge impulse → particles fly.
        # setup_collider() cannot run inside the CUDA graph (GPU→CPU copy), so we
        # defer it here, one step after the actual reset.
        if self._collider_reinit_pending:
            self._sand.reinit_collider()
            self._collider_reinit_pending = False

        # Detect if any env was reset THIS step and schedule reinit for the next step.
        # reset_buf is a Warp bool array written by _get_dones() inside the graph;
        # reading it here (outside the graph) is safe and requires only a tiny D2H copy.
        if wp.to_torch(self.reset_buf).any():
            self._collider_reinit_pending = True

        # MPM step runs outside the CUDA graph (sparse grid needs dynamic allocation).
        # Proxy bodies were already updated inside _get_dones() during graph capture,
        # so they already reflect the current robot pose when this runs.
        self._sand.step(dt=self._mpm_dt)

        # Render env_0 particles via Newton viewer's log_points with a CUDA Warp array.
        # VisualizationMarkers routes through CPU numpy, causing a device-mismatch error
        # in the GL instancer's Warp kernel. log_points accepts CUDA wp.array directly.
        particle_q_all = self._sand.get_all_particle_q()
        for viz in self.sim.visualizers:
            viewer = getattr(viz, "_viewer", None)
            if viewer is not None and hasattr(viewer, "log_points"):
                viewer.log_points(
                    "/sand_particles",
                    particle_q_all,
                    radii=self._particle_radii,
                    colors=self._particle_colors,
                )

    def _get_dones(self) -> None:
        # Update proxy body transforms inside the graph (pure array writes — graph-safe).
        # mpm_solver.step() is called in _post_step_visualize() outside the graph.
        self._sand.update_proxy_bodies(self._body_pos_w, self._body_quat_w)

        self._update_ee_pos()
        wp.launch(
            _get_dones,
            dim=self.num_envs,
            inputs=[
                self._episode_length_buf_wp,
                self.max_episode_length,
                self.reset_time_outs,
                self.reset_buf,
            ],
            device=self.device,
        )
        # Fixed-base arm: no early termination, only timeout resets
        self.reset_terminated.zero_()

    def _reset_idx(self, mask: wp.array | None = None) -> None:
        if mask is None:
            mask = self._ALL_ENV_MASK

        super()._reset_idx(mask)

        wp.launch(
            _reset_joints,
            dim=self.num_envs,
            inputs=[
                self.robot.data.default_joint_pos.warp,
                self.robot.data.default_joint_vel.warp,
                self.cfg.reset_dof_pos_noise,
                self.robot.num_joints,
                mask,
                self._rng_state,
                self._joint_pos,
                self._joint_vel,
            ],
            device=self.device,
        )

        self._update_ee_pos()

        # Clear smoothed-action history for reset envs so stale end-of-episode
        # actions don't bleed into the new episode via the EMA filter.
        wp.launch(
            _reset_actions_for_mask,
            dim=self.num_envs,
            inputs=[mask, self.cfg.action_space, self._actions_prev],
            device=self.device,
        )

        # Reset sand particles to settled snapshot for the masked envs
        self._sand.reset(
            env_mask=mask,
            env_origins=self._env_origins,
            body_pos_w=self._body_pos_w,
            body_quat_w=self._body_quat_w,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_ee_pos(self) -> None:
        wp.launch(
            _compute_ee_pos,
            dim=self.num_envs,
            inputs=[
                self._body_pos_w,
                self._env_origins,
                self._scoop_body_idx,
                self._ee_pos_local,
            ],
            device=self.device,
        )
