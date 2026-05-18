# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implicit MPM Newton manager."""

from __future__ import annotations

import logging
from dataclasses import fields

import numpy as np
import warp as wp

from newton import Model
from newton.solvers import SolverImplicitMPM

from isaaclab.physics import PhysicsManager

from .implicit_mpm_manager_cfg import ImplicitMPMSolverCfg
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)


class NewtonImplicitMPMManager(NewtonManager):
    """:class:`NewtonManager` specialization for :class:`SolverImplicitMPM`.

    Builds a :class:`SolverImplicitMPM.Config` from :class:`ImplicitMPMSolverCfg`,
    constructs :class:`SolverImplicitMPM`, and configures MPM colliders after the
    base solver initialization. Does not use Newton's :class:`CollisionPipeline`
    for stepping (contacts are ``None`` in :meth:`_step_solver`).
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: ImplicitMPMSolverCfg) -> None:
        """Construct :class:`SolverImplicitMPM` with a nested :class:`SolverImplicitMPM.Config`.

        Only keys that exist on :class:`SolverImplicitMPM.Config` are forwarded;
        metadata fields such as ``class_type`` / ``solver_type`` are dropped.
        """
        cfg_fields = {f.name for f in fields(SolverImplicitMPM.Config)}
        raw = solver_cfg.to_dict()
        mpm_kwargs = {k: v for k, v in raw.items() if k in cfg_fields}
        mpm_config = SolverImplicitMPM.Config(**mpm_kwargs)

        NewtonManager._solver = SolverImplicitMPM(
            model,
            mpm_config,
            temporary_store=None,
            verbose=None,
            enable_timers=False,
        )
        # In-place stepping: base calls step(state_0, state_0, ...) in single-state mode.
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = False

    @classmethod
    def _get_solver_convergence_steps(cls) -> dict[str, float | int]:
        """Return placeholder rheology iteration stats.

        Replace with real counters when the solver exposes a stable API for them.
        """
        return {"max": 0, "mean": 0.0, "min": 0, "std": 0.0}
