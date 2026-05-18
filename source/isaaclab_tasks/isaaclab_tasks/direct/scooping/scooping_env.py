# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

if TYPE_CHECKING:
    from .scooping_env_cfg import ScoopingEnvCfg


logger = logging.getLogger(__name__)


class ScoopingEnv(DirectRLEnv):
    """Direct env: Newton Implicit MPM sandbox for scooping task bring-up."""

    cfg: ScoopingEnvCfg

    def __init__(self, cfg: ScoopingEnvCfg, render_mode: str | None = None, **kwargs):
        self._physics_callback_handles = []
        super().__init__(cfg, render_mode, **kwargs)

    # ---------------------------------------------------------------------
    # Scene setup
    # ---------------------------------------------------------------------

    def _setup_scene(self) -> None:
        # Ground
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Visual-only container in source env (collision handled in builder callback)
        self._spawn_visual_container()

        # Clone env_0 to env_1..N
        self.scene.clone_environments(copy_from_source=False)

        # CPU collision filtering
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Register solver lifecycle callbacks before sim.reset()
        self._register_newton_mpm_callbacks()

    def _register_newton_mpm_callbacks(self) -> None:
        """Register Newton MODEL_INIT/PHYSICS_READY hooks."""
        physics = self.cfg.sim.physics
        if physics is None or type(physics).__name__ != "NewtonCfg":
            logger.info(
                "ScoopingEnv: skipping MPM callbacks (physics is %r, not Newton).",
                type(physics).__name__,
            )
            return

        voxel_size = float(physics.solver_cfg.voxel_size)  # type: ignore[attr-defined]
        env_origins = self.scene.env_origins.detach().cpu().numpy()

        def _on_model_init(_payload: object | None = None) -> None:
            import newton
            from newton.solvers import SolverImplicitMPM

            # Access _builder from the active physics manager class (e.g. NewtonImplicitMPMManager).
            # NewtonManager._builder would be None when a subclass is active, because Python stores
            # cls._builder on the subclass dict, shadowing the base-class attribute.
            builder = self.sim.physics_manager._builder
            if builder is None:
                logger.warning("ScoopingEnv MODEL_INIT: physics_manager._builder is None.")
                return

            # Register per-particle MPM attributes.
            SolverImplicitMPM.register_custom_attributes(builder)

            # Ground in solver model
            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

            # Inject one container + one particle emitter per env origin.
            # Particles added outside begin_world/end_world get world=-1, which makes
            # model.particle_world_start show an empty range for world 0 and prevents
            # rendering. Manually assign each particle to its env's world index.
            for world_idx, origin in enumerate(env_origins):
                origin_vec = np.asarray(origin, dtype=np.float32)
                self._add_container_mpm_colliders(builder, newton, origin_vec)

                start_particle_idx = builder.particle_count
                self._add_particle_grid(builder, voxel_size, origin_vec)
                end_particle_idx = builder.particle_count
                builder.particle_world[start_particle_idx:end_particle_idx] = [world_idx] * (
                    end_particle_idx - start_particle_idx
                )

        def _on_physics_ready(_payload: object | None = None) -> None:
            # Same subclass-shadowing issue as MODEL_INIT: use active physics manager.
            model = self.sim.physics_manager.get_model()
            if model is None:
                logger.warning("ScoopingEnv PHYSICS_READY: Newton model is None.")
                return

            mp = getattr(model, "mpm", None)
            if mp is None:
                logger.warning("ScoopingEnv PHYSICS_READY: model has no 'mpm' namespace.")
                return

            self._fill_mpm_material(mp)

        from isaaclab_newton.physics import NewtonManager

        h1 = NewtonManager.register_callback(_on_model_init, PhysicsEvent.MODEL_INIT, name="scooping_mpm_model_init")
        h2 = NewtonManager.register_callback(
            _on_physics_ready,
            PhysicsEvent.PHYSICS_READY,
            name="scooping_mpm_physics_ready",
        )
        self._physics_callback_handles.extend((h1, h2))

    # ---------------------------------------------------------------------
    # Newton builder helpers
    # ---------------------------------------------------------------------

    def _add_container_mpm_colliders(self, builder, newton, origin: np.ndarray) -> None:
        """Add static open-top container colliders into Newton builder."""
        c = self.cfg.container
        ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])

        box_cfg = newton.ModelBuilder.ShapeConfig(mu=float(c.friction), gap=float(c.gap))

        # Bottom
        builder.add_shape_box(
            body=-1,
            cfg=box_cfg,
            xform=wp.transform(wp.vec3(ox, oy, oz + c.wall_thickness * 0.5), wp.quat_identity()),
            hx=c.width * 0.5,
            hy=c.depth * 0.5,
            hz=c.wall_thickness * 0.5,
        )

        # Front / back
        wall_hx = (c.width + 2.0 * c.wall_overlap) * 0.5
        builder.add_shape_box(
            body=-1,
            cfg=box_cfg,
            xform=wp.transform(wp.vec3(ox, oy - c.depth * 0.5, oz + c.height * 0.5), wp.quat_identity()),
            hx=wall_hx,
            hy=c.wall_thickness * 0.5,
            hz=c.height * 0.5,
        )
        builder.add_shape_box(
            body=-1,
            cfg=box_cfg,
            xform=wp.transform(wp.vec3(ox, oy + c.depth * 0.5, oz + c.height * 0.5), wp.quat_identity()),
            hx=wall_hx,
            hy=c.wall_thickness * 0.5,
            hz=c.height * 0.5,
        )

        # Left / right
        wall_hy = (c.depth + 2.0 * c.wall_overlap) * 0.5
        builder.add_shape_box(
            body=-1,
            cfg=box_cfg,
            xform=wp.transform(wp.vec3(ox - c.width * 0.5, oy, oz + c.height * 0.5), wp.quat_identity()),
            hx=c.wall_thickness * 0.5,
            hy=wall_hy,
            hz=c.height * 0.5,
        )
        builder.add_shape_box(
            body=-1,
            cfg=box_cfg,
            xform=wp.transform(wp.vec3(ox + c.width * 0.5, oy, oz + c.height * 0.5), wp.quat_identity()),
            hx=c.wall_thickness * 0.5,
            hy=wall_hy,
            hz=c.height * 0.5,
        )

    def _add_particle_grid(self, builder, voxel_size: float, origin: np.ndarray) -> None:
        """Emit one particle grid volume at the given env origin."""
        e = self.cfg.emitter
        origin = np.asarray(origin, dtype=np.float32)

        particle_lo = np.asarray(e.emit_lo, dtype=np.float32) + origin
        particle_hi = np.asarray(e.emit_hi, dtype=np.float32) + origin
        ppc = int(e.particles_per_cell)

        particle_res = np.ceil(ppc * (particle_hi - particle_lo) / voxel_size).astype(int)
        cell_size = (particle_hi - particle_lo) / particle_res
        cell_volume = float(np.prod(cell_size))
        radius = float(np.max(cell_size) * 0.5)
        mass = cell_volume * float(e.density)
        total_particles = int((particle_res[0] + 1) * (particle_res[1] + 1) * (particle_res[2] + 1))

        logger.info(
            "ScoopingEnv MPM emit: origin=(%.3f, %.3f, %.3f) res=%s total=%d radius=%.6f mass=%.6f",
            float(origin[0]),
            float(origin[1]),
            float(origin[2]),
            particle_res.tolist(),
            total_particles,
            radius,
            mass,
        )

        builder.add_particle_grid(
            pos=wp.vec3(particle_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(particle_res[0] + 1),
            dim_y=int(particle_res[1] + 1),
            dim_z=int(particle_res[2] + 1),
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass,
            jitter=float(e.initial_jitter) * radius,
            radius_mean=radius,
        )

    def _fill_mpm_material(self, mp) -> None:
        """Fill per-particle MPM material arrays after PHYSICS_READY."""
        m = self.cfg.material
        mats = dict(
            young_modulus=float(m.young_modulus),
            poisson_ratio=float(m.poisson_ratio),
            friction=float(m.friction),
            damping=float(m.damping),
            yield_pressure=float(m.yield_pressure),
            tensile_yield_ratio=float(m.tensile_yield_ratio),
            yield_stress=float(m.yield_stress),
            hardening=float(m.hardening),
            dilatancy=float(m.dilatancy),
            viscosity=float(m.viscosity),
        )
        for name, value in mats.items():
            arr = getattr(mp, name, None)
            if arr is None:
                continue
            try:
                arr.fill_(wp.float32(value))
            except Exception:
                logger.debug("ScoopingEnv: skipped filling mpm.%s", name)

    # ---------------------------------------------------------------------
    # Visual helper
    # ---------------------------------------------------------------------

    def _spawn_visual_container(self) -> None:
        """Spawn visual-only static cuboid container in source env."""
        c = self.cfg.container
        source_env_path = self.scene.env_prim_paths[0]
        container_path = f"{source_env_path}/Container"
        sim_utils.create_prim(container_path, "Xform")

        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        )
        collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)

        bottom_cfg = sim_utils.CuboidCfg(
            size=(c.width, c.depth, c.wall_thickness),
            rigid_props=rigid_props,
            collision_props=collision_props,
        )
        front_cfg = sim_utils.CuboidCfg(
            size=(c.width + 2.0 * c.wall_overlap, c.wall_thickness, c.height),
            rigid_props=rigid_props,
            collision_props=collision_props,
        )
        back_cfg = sim_utils.CuboidCfg(
            size=(c.width + 2.0 * c.wall_overlap, c.wall_thickness, c.height),
            rigid_props=rigid_props,
            collision_props=collision_props,
        )
        left_cfg = sim_utils.CuboidCfg(
            size=(c.wall_thickness, c.depth + 2.0 * c.wall_overlap, c.height),
            rigid_props=rigid_props,
            collision_props=collision_props,
        )
        right_cfg = sim_utils.CuboidCfg(
            size=(c.wall_thickness, c.depth + 2.0 * c.wall_overlap, c.height),
            rigid_props=rigid_props,
            collision_props=collision_props,
        )

        bottom_cfg.func(
            f"{container_path}/bottom",
            bottom_cfg,
            translation=(0.0, 0.0, c.wall_thickness * 0.5),
        )
        front_cfg.func(
            f"{container_path}/front",
            front_cfg,
            translation=(0.0, -c.depth * 0.5, c.height * 0.5),
        )
        back_cfg.func(
            f"{container_path}/back",
            back_cfg,
            translation=(0.0, c.depth * 0.5, c.height * 0.5),
        )
        left_cfg.func(
            f"{container_path}/left",
            left_cfg,
            translation=(-c.width * 0.5, 0.0, c.height * 0.5),
        )
        right_cfg.func(
            f"{container_path}/right",
            right_cfg,
            translation=(c.width * 0.5, 0.0, c.height * 0.5),
        )

    # ---------------------------------------------------------------------
    # RL stubs (same pattern as direct envs, placeholder logic for now)
    # ---------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        return

    def _get_observations(self) -> dict:
        obs = torch.zeros(self.num_envs, int(self.cfg.observation_space), device=self.device)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.ones(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)
        return died, time_out

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        del debug_vis
