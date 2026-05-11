# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton implicit MPM physics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class ImplicitMPMSolverCfg(NewtonSolverCfg):
    """Parameters mapped to :class:`newton.solvers.SolverImplicitMPM.Config`.

    Field names and defaults match the installed Newton package. See the
    `Newton documentation`_ for physical meaning.

    .. _Newton documentation: https://newton-physics.github.io/newton/1.1.0/api/_generated/newton.solvers.SolverImplicitMPM.html
    """

    class_type: type[NewtonManager] | str = "{DIR}.implicit_mpm_manager:NewtonImplicitMPMManager"
    """Manager class for the implicit MPM solver."""

    solver_type: str = "implicit_mpm"
    """Solver type label for logging (dispatch uses :attr:`class_type`)."""

    # --- SolverImplicitMPM.Config (keep names aligned with Newton) ---

    max_iterations: int = 250
    """Maximum iterations for the rheology solver."""

    tolerance: float = 1.0e-4
    """Convergence tolerance for the rheology solver."""

    solver: Literal["gauss-seidel", "jacobi", "cg"] = "gauss-seidel"
    """Rheology linear subsolver."""

    warmstart_mode: Literal["none", "auto", "particles", "grid", "smoothed"] = "auto"
    """Warmstart mode for the rheology solver."""

    collider_velocity_mode: Literal[
        "forward", "backward", "instantaneous", "finite_difference"
    ] = "forward"
    """How collider velocities are estimated for MPM contact."""

    voxel_size: float = 0.1
    """Grid voxel size [m]."""

    grid_type: Literal["sparse", "dense", "fixed"] = "sparse"
    """Active grid storage type."""

    grid_padding: int = 0
    """Extra empty cells around particles when allocating the grid."""

    max_active_cell_count: int = -1
    """Cap on active cells for dense-grid subsets; ``-1`` means unlimited."""

    transfer_scheme: Literal["apic", "pic"] = "apic"
    """Particle-grid transfer scheme."""

    integration_scheme: Literal["pic", "gimp"] = "pic"
    """Shape-function support / integration scheme."""

    critical_fraction: float = 0.0
    """Yield-surface collapse fraction when particle fraction is low."""

    air_drag: float = 1.0
    """Numerical drag for the background air."""

    collider_normal_from_sdf_gradient: bool = False
    """If True, collider normals come from the SDF gradient."""

    collider_basis: str = "Q1"
    """Collider finite-element basis (see Newton docs for accepted strings)."""

    strain_basis: str = "P0"
    """Strain basis (see Newton docs for accepted strings)."""

    velocity_basis: str = "Q1"
    """Velocity basis (see Newton docs for accepted strings)."""
