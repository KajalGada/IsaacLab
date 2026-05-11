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
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    from .scooping_env_cfg import ScoopingEnvCfg


logger = logging.getLogger(__name__)


class ScoopingEnv(DirectRLEnv):
    """Direct env: Isaac Lab scene + Newton implicit MPM sand (no robot yet)."""

    cfg: ScoopingEnvCfg

    def __init__(self, cfg: ScoopingEnvCfg, render_mode: str | None = None, **kwargs):
        self._physics_callback_handles = []
        self._particle_emit_cfg = dict(
            emit_lo=np.array([-0.15, -0.15, 0.02], dtype=np.float32),
            emit_hi=np.array([0.15, 0.15, 0.18], dtype=np.float32),
            particles_per_cell=3,
            initial_jitter=0.5,
            density=1100.0,
        )
        super().__init__(cfg, render_mode, **kwargs)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._register_newton_mpm_callbacks()

    def _register_newton_mpm_callbacks(self) -> None:
        """Register Newton MPM builder/model hooks before ``sim.reset()`` runs."""
        physics = self.cfg.sim.physics
        if physics is None or type(physics).__name__ != "NewtonCfg":
            logger.info(
                "ScoopingEnv: skipping MPM callbacks (physics is %r, not Newton).",
                type(physics).__name__,
            )
            return

        voxel_size = float(physics.solver_cfg.voxel_size)  # type: ignore[attr-defined]
        emit = self._particle_emit_cfg
        env_ref = self  # strong ref for the closure

        def _on_model_init(_payload: object | None = None) -> None:
            from isaaclab_newton.physics import NewtonManager
            import newton
            from newton.solvers import SolverImplicitMPM

            builder = NewtonManager._builder
            if builder is None:
                logger.warning("ScoopingEnv MODEL_INIT: NewtonManager._builder is None.")
                return

            logger.info("ScoopingEnv MODEL_INIT: registering MPM attributes and adding particles.")
            SolverImplicitMPM.register_custom_attributes(builder)

            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

            particle_lo = np.asarray(emit["emit_lo"], dtype=np.float32)
            particle_hi = np.asarray(emit["emit_hi"], dtype=np.float32)
            ppc = int(emit["particles_per_cell"])

            # env_origins is (num_envs, 3), populated after clone_environments()
            env_origins_np = env_ref.scene.env_origins.cpu().numpy()

            total_particles = 0
            for env_i, origin in enumerate(env_origins_np):
                lo = particle_lo + origin.astype(np.float32)
                hi = particle_hi + origin.astype(np.float32)

                particle_res = np.ceil(ppc * (hi - lo) / voxel_size).astype(int)
                cell_size = (hi - lo) / particle_res
                cell_volume = float(np.prod(cell_size))
                radius = float(np.max(cell_size) * 0.5)
                mass = cell_volume * float(emit["density"])
                n = int((particle_res[0] + 1) * (particle_res[1] + 1) * (particle_res[2] + 1))
                total_particles += n

                builder.add_particle_grid(
                    pos=wp.vec3(float(lo[0]), float(lo[1]), float(lo[2])),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    dim_x=int(particle_res[0] + 1),
                    dim_y=int(particle_res[1] + 1),
                    dim_z=int(particle_res[2] + 1),
                    cell_x=float(cell_size[0]),
                    cell_y=float(cell_size[1]),
                    cell_z=float(cell_size[2]),
                    mass=mass,
                    jitter=float(emit["initial_jitter"]) * radius,
                    radius_mean=radius,
                )

            logger.info(
                "ScoopingEnv MODEL_INIT: added %d particles across %d envs.",
                total_particles,
                len(env_origins_np),
            )

        def _on_physics_ready(_payload: object | None = None) -> None:
            from isaaclab_newton.physics import NewtonManager

            model = NewtonManager.get_model()
            if model is None:
                logger.warning("ScoopingEnv PHYSICS_READY: Newton model is None.")
                return

            mp = getattr(model, "mpm", None)
            if mp is None:
                logger.warning("ScoopingEnv PHYSICS_READY: model has no 'mpm' namespace.")
                return

            ScoopingEnv._fill_mpm_material(mp)

        from isaaclab_newton.physics import NewtonManager

        h1 = NewtonManager.register_callback(_on_model_init, PhysicsEvent.MODEL_INIT, name="scooping_mpm_model_init")
        h2 = NewtonManager.register_callback(
            _on_physics_ready,
            PhysicsEvent.PHYSICS_READY,
            name="scooping_mpm_physics_ready",
        )
        self._physics_callback_handles.extend((h1, h2))

    @staticmethod
    def _fill_mpm_material(mp) -> None:
        """Fill per-particle MPM fields on ``model.mpm``. Adjust keys to match your Newton build."""
        mats = dict(
            young_modulus=float(5e3),
            poisson_ratio=float(0.25),
            friction=float(0.8),
            damping=float(3000.0),
            yield_pressure=float(50.0),
            tensile_yield_ratio=float(0.0),
            yield_stress=float(25.0),
            hardening=float(0.5),
            dilatancy=float(0.0),
            viscosity=float(0.0),
        )
        for name, value in mats.items():
            arr = getattr(mp, name, None)
            if arr is None:
                continue
            try:
                arr.fill_(wp.float32(value))
            except Exception:
                logger.debug("ScoopingEnv: skipped filling mpm.%s", name)

    # ------------------------------------------------------------------
    # RL stubs (must match observation_space / action_space in cfg).
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        return

    def _get_observations(self) -> dict:
        dt = torch.zeros(self.num_envs, int(self.cfg.observation_space), device=self.device)
        return {"policy": dt}

    def _get_rewards(self) -> torch.Tensor:
        return torch.ones(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)
        return died, time_out

    # ------------------------------------------------------------------
    # Debug vis (base body raises unless overridden — keep noop).
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        del debug_vis


if TYPE_CHECKING:
    pass  # Silence unused-import linters when only used in forward refs.
