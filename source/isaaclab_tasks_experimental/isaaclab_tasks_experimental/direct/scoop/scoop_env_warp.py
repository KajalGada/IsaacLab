# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10 direct RL environment for sand scooping using Newton MuJoCo-Warp physics.

Sand particles are excluded in this version; the task is wrist-tip reaching.
"""

from __future__ import annotations

import torch
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
from isaaclab_assets.robots.universal_robots import UR10_CFG

# Sand-box geometry (matches Newton reference script)
_BOX_W = 0.35  # x full width
_BOX_D = 0.35  # y full depth
_BOX_WALL_H = 0.05  # z wall height
_BOX_FLOOR_T = 0.02  # floor / wall thickness

# Static box pieces: (name, full_size_xyz, centre_xyz) — all env-local coords
_BOX_PIECES: list[tuple[str, tuple, tuple]] = [
    ("floor",      (_BOX_W,        _BOX_D,        _BOX_FLOOR_T),  (0.0,            0.0,            _BOX_FLOOR_T / 2)),
    ("wall_neg_y", (_BOX_W,        _BOX_FLOOR_T,  _BOX_WALL_H),   (0.0,           -_BOX_D / 2,     _BOX_WALL_H / 2)),
    ("wall_pos_y", (_BOX_W,        _BOX_FLOOR_T,  _BOX_WALL_H),   (0.0,            _BOX_D / 2,     _BOX_WALL_H / 2)),
    ("wall_neg_x", (_BOX_FLOOR_T,  _BOX_D,        _BOX_WALL_H),   (-_BOX_W / 2,    0.0,            _BOX_WALL_H / 2)),
    ("wall_pos_x", (_BOX_FLOOR_T,  _BOX_D,        _BOX_WALL_H),   ( _BOX_W / 2,    0.0,            _BOX_WALL_H / 2)),
]

# UR10 hovering above the sand box, arm facing -X (toward box at origin)
UR10_SCOOP_CFG = UR10_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.0),
        rot=(0.0, 0.0, 1.0, 0.0),  # 180° around Z so arm faces the box at origin
        joint_pos={
            "shoulder_pan_joint":  -0.27,
            "shoulder_lift_joint": -1.50,
            "elbow_joint":          1.70,
            "wrist_1_joint":        1.5708,
            "wrist_2_joint":        1.5708,
            "wrist_3_joint":        1.5708,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
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
    target_pos: wp.vec3f,
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
    # [21:24] vector EE → target
    observations[env_id, base + 3] = target_pos[0] - ee_pos_local[env_id][0]
    observations[env_id, base + 4] = target_pos[1] - ee_pos_local[env_id][1]
    observations[env_id, base + 5] = target_pos[2] - ee_pos_local[env_id][2]


@wp.kernel
def _compute_rewards(
    ee_pos_local: wp.array(dtype=wp.vec3f),
    actions: wp.array2d(dtype=wp.float32),
    target_pos: wp.vec3f,
    num_dofs: wp.int32,
    dist_scale: wp.float32,
    action_penalty_scale: wp.float32,
    alive_reward: wp.float32,
    rewards: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    dist = wp.length(ee_pos_local[env_id] - target_pos)
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
class ScoopWarpEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s: float = 10.0
    decimation: int = 2
    action_space: int = 6
    # 6 scaled joint pos + 6 joint vel + 6 prev actions + 3 EE pos + 3 vec-to-target
    observation_space: int = 24
    state_space: int = 0

    # Newton MuJoCo-Warp solver
    solver_cfg = MJWarpSolverCfg(
        njmax=80,
        nconmax=20,
        cone="pyramidal",
        integrator="implicitfast",
        impratio=1,
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # robot
    robot: ArticulationCfg = UR10_SCOOP_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # task
    scoop_body_name: str = "wrist_3_link"
    # Target in env-local coordinates: centre of sand box at 5 cm depth
    target_pos: tuple = (0.0, 0.0, 0.05)

    # reward scales
    dist_reward_scale: float = -1.0
    action_penalty_scale: float = -0.001
    alive_reward: float = 0.1

    # observation scaling
    dof_vel_scale: float = 0.1

    # reset
    reset_dof_pos_noise: float = 0.05  # radians noise around default pose


# ---------------------------------------------------------------------------
# Environment implementation
# ---------------------------------------------------------------------------


class ScoopWarpEnv(DirectRLEnvWarp):
    cfg: ScoopWarpEnvCfg

    def __init__(self, cfg: ScoopWarpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Scoop body index in the articulation body list
        if self.cfg.scoop_body_name in self.robot.body_names:
            self._scoop_body_idx = self.robot.body_names.index(self.cfg.scoop_body_name)
        else:
            # Fallback if scoop_link was merged during USD import
            self._scoop_body_idx = self.robot.body_names.index("wrist_3_link")

        # Warp views into Newton simulation data (zero-copy)
        self._joint_pos = self.robot.data.joint_pos.warp
        self._joint_vel = self.robot.data.joint_vel.warp
        self._body_pos_w = self.robot.data.body_pos_w.warp
        self._joint_limits = self.robot.data.soft_joint_pos_limits.warp

        # Env-local origins
        self._env_origins = wp.from_torch(self.scene.env_origins, dtype=wp.vec3f)

        # Persistent warp buffers
        self._ee_pos_local = wp.zeros(self.num_envs, dtype=wp.vec3f, device=self.device)
        self._actions = wp.zeros((self.num_envs, self.cfg.action_space), dtype=wp.float32, device=self.device)
        self._joint_targets = wp.zeros(
            (self.num_envs, self.robot.num_joints), dtype=wp.float32, device=self.device
        )
        self._observations = wp.zeros(
            (self.num_envs, self.cfg.observation_space), dtype=wp.float32, device=self.device
        )
        self._rewards = wp.zeros(self.num_envs, dtype=wp.float32, device=self.device)

        # Per-env RNG state for reset noise
        self._rng_state = wp.zeros(self.num_envs, dtype=wp.uint32, device=self.device)
        seed = self.cfg.seed if self.cfg.seed is not None else 0
        wp.launch(_initialize_rng, dim=self.num_envs, inputs=[seed, self._rng_state], device=self.device)

        # Target position constant (same for every env — env-local frame)
        self._target_pos_wp = wp.vec3f(*self.cfg.target_pos)

        # Torch-aliased buffers expected by DirectRLEnvWarp base class
        self.torch_obs_buf = wp.to_torch(self._observations)
        self.torch_reward_buf = wp.to_torch(self._rewards)

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
        for name, size, centre in _BOX_PIECES:
            piece_cfg = sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=box_material,
            )
            piece_cfg.func(
                f"/World/envs/env_0/SandBox/{name}",
                piece_cfg,
                translation=centre,
            )

    # ------------------------------------------------------------------
    # Step callbacks
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: wp.array) -> None:
        wp.copy(self._actions, actions)

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
                self._target_pos_wp,
                self.cfg.dof_vel_scale,
                self.robot.num_joints,
                self._observations,
            ],
            device=self.device,
        )
        return {"policy": self.torch_obs_buf}

    def _get_rewards(self) -> None:
        wp.launch(
            _compute_rewards,
            dim=self.num_envs,
            inputs=[
                self._ee_pos_local,
                self._actions,
                self._target_pos_wp,
                self.robot.num_joints,
                self.cfg.dist_reward_scale,
                self.cfg.action_penalty_scale,
                self.cfg.alive_reward,
                self._rewards,
            ],
            device=self.device,
        )

    def _get_dones(self) -> None:
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
