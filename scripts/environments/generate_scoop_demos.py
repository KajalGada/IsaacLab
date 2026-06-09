# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate synthetic scoop demonstration trajectories for BC pre-training.

Executes a parametric 5-phase joint-space motion primitive in
``Isaac-Scoop-Direct-Warp-v3`` and saves successful trajectories as ``.npz``
files compatible with ``collect_demo_data.py``.

Motion phases
-------------
1. **Tilt**    — wrist_1 rotates so the scoop faces down into the sand pile.
2. **Descend** — shoulder_lift (and elbow) lower the arm toward the sand surface.
3. **Sweep**   — shoulder_pan sweeps the arm horizontally through the sand.
4. **Level**   — wrist_1 rotates back to trap sand inside the scoop.
5. **Lift**    — shoulder_lift and elbow return to the hover height.

Between keyframes, joint angles are linearly interpolated.  Each attempt
samples the five scalar parameters randomly.  A trajectory is accepted only
when ``--min_particles`` or more sand particles satisfy:

* lifted more than ``--lift_threshold`` metres above their settled z-position,
* AND within ``--capture_radius`` metres (horizontal) of the scoop EE.

Output .npz keys (compatible with collect_demo_data.py)
-------------------------------------------------------
* ``states``      — (T, 6) absolute joint angles [rad]
* ``joint_names`` — (6,)   joint name strings
* ``params``      — (5,)   [tilt, descend, sweep, level_frac, steps] for reproducibility
* ``n_captured``  — scalar, number of captured particles at end of lift

Usage::

    ./isaaclab.sh -p scripts/environments/generate_scoop_demos.py \\
        --task Isaac-Scoop-Direct-Warp-v3 \\
        --num_demos 50 \\
        --output_dir logs/scoop_demos \\
        --headless
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import numpy as np
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Generate synthetic scoop demo trajectories.")
parser.add_argument("--task",         type=str,   default="Isaac-Scoop-Direct-Warp-v3",
                    help="Gym task ID.")
parser.add_argument("--num_demos",    type=int,   default=50,
                    help="Target number of successful trajectories to save.")
parser.add_argument("--max_attempts", type=int,   default=300,
                    help="Hard limit on total attempts (to avoid infinite loops).")
parser.add_argument("--output_dir",   type=str,   default="logs/scoop_demos",
                    help="Directory for output .npz files.")
parser.add_argument("--seed",         type=int,   default=42,
                    help="NumPy RNG seed for reproducible sampling.")

# --- Motion primitive parameter ranges ---
parser.add_argument("--tilt_min",    type=float, default=0.3,
                    help="Min wrist_1 tilt delta [rad] (phase 1).")
parser.add_argument("--tilt_max",    type=float, default=0.8,
                    help="Max wrist_1 tilt delta [rad] (phase 1).")
parser.add_argument("--descend_min", type=float, default=0.2,
                    help="Min shoulder_lift descend delta [rad] (phase 2).")
parser.add_argument("--descend_max", type=float, default=0.55,
                    help="Max shoulder_lift descend delta [rad] (phase 2).")
parser.add_argument("--sweep_min",   type=float, default=0.15,
                    help="Min shoulder_pan sweep magnitude [rad] (phase 3).")
parser.add_argument("--sweep_max",   type=float, default=0.45,
                    help="Max shoulder_pan sweep magnitude [rad] (phase 3).")
parser.add_argument("--steps_min",   type=int,   default=20,
                    help="Min interpolation steps per phase.")
parser.add_argument("--steps_max",   type=int,   default=60,
                    help="Max interpolation steps per phase.")

# --- Success filter ---
parser.add_argument("--lift_threshold", type=float, default=0.2,
                    help="Particle z must exceed settled z + this value [m] to count as lifted.")
parser.add_argument("--capture_radius", type=float, default=0.10,
                    help="Horizontal radius [m] from scoop EE for 'inside scoop' check.")
parser.add_argument("--min_particles",  type=int,   default=30,
                    help="Minimum captured particles required to accept a trajectory.")

add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Initial joint angles from UR5_SCOOP_CFG (hover pose).
# Order matches UR5 URDF joint order: pan, lift, elbow, wrist_1, wrist_2, wrist_3.
_HOVER_CFG = {
    "shoulder_pan_joint":   -0.27,
    "shoulder_lift_joint":  -1.50,
    "elbow_joint":           1.70,
    "wrist_1_joint":         1.5708,
    "wrist_2_joint":         1.5708,
    "wrist_3_joint":         1.5708,
}


# ---------------------------------------------------------------------------
# Motion primitive helpers
# ---------------------------------------------------------------------------


def _lerp(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Linearly interpolate n steps from a toward b, excluding a itself."""
    t = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)[1:]  # (n,)
    return a + t[:, None] * (b - a)                           # (n, 6)


def make_scoop_waypoints(
    hover: np.ndarray,
    tilt_delta: float,
    descend_delta: float,
    sweep_delta: float,
    level_frac: float,
    steps: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Build a (T, 6) joint-angle trajectory for one scoop attempt.

    Joint index convention (UR5):
        0 shoulder_pan   1 shoulder_lift   2 elbow
        3 wrist_1        4 wrist_2         5 wrist_3

    Args:
        hover:         Starting (hover) joint angles [rad], shape (6,).
        tilt_delta:    wrist_1 increase in phase 1 [rad].
        descend_delta: shoulder_lift decrease in phase 2 [rad].
        sweep_delta:   shoulder_pan change in phase 3 [rad] (signed — direction).
        level_frac:    Fraction of tilt removed in phase 4 (0 = keep tilt, 1 = full level).
        steps:         Interpolation steps per phase.
        lower:         Joint lower limits [rad], shape (6,).
        upper:         Joint upper limits [rad], shape (6,).

    Returns:
        Waypoints array of shape (5*steps + 1, 6), clamped to joint limits.
    """
    # Phase 0 keyframe — hover
    p0 = hover.copy()

    # Phase 1 — Tilt: wrist_1 up so scoop faces sand
    p1 = p0.copy()
    p1[3] += tilt_delta

    # Phase 2 — Descend: shoulder_lift down, elbow adjusts to keep arm stable
    p2 = p1.copy()
    p2[1] -= descend_delta
    p2[2] -= descend_delta * 0.4   # partial elbow coupling keeps EE roughly level

    # Phase 3 — Sweep: shoulder_pan sweeps through sand
    p3 = p2.copy()
    p3[0] += sweep_delta

    # Phase 4 — Level: wrist_1 back toward horizontal to trap sand
    p4 = p3.copy()
    p4[3] -= tilt_delta * level_frac

    # Phase 5 — Lift: shoulder_lift and elbow return to hover height
    p5 = p4.copy()
    p5[1] = p0[1]
    p5[2] = p0[2]

    keyframes = [p0, p1, p2, p3, p4, p5]
    segments  = [_lerp(keyframes[i], keyframes[i + 1], steps) for i in range(len(keyframes) - 1)]
    waypoints = np.concatenate([p0[None], *segments], axis=0)  # (5*steps + 1, 6)

    return np.clip(waypoints, lower[None], upper[None])


# ---------------------------------------------------------------------------
# Success check
# ---------------------------------------------------------------------------


def count_captured_particles(
    raw_env,
    lift_threshold: float,
    capture_radius: float,
) -> int:
    """Count particles that are lifted AND horizontally near the scoop EE.

    Reads CUDA arrays directly — no CPU round-trip for the particle data.

    Args:
        raw_env:        Unwrapped ScoopWarpEnvV3 instance.
        lift_threshold: Minimum z rise above settled position [m].
        capture_radius: Horizontal distance from EE centre [m].

    Returns:
        Number of particles meeting both conditions (int).
    """
    import warp as wp

    sand        = raw_env._sand
    particle_q  = wp.to_torch(sand._state.particle_q)   # (N_total, 3) world coords
    snapshot_q  = wp.to_torch(sand._snapshot_q)          # (N_total, 3) settled world coords

    # Slice particles belonging to env 0 (num_envs=1, so this is all particles)
    start = int(sand._env_particle_start.numpy()[0])
    count = int(sand._env_particle_count.numpy()[0])
    p  = particle_q[start : start + count]   # (N, 3)
    p0 = snapshot_q[start : start + count]   # (N, 3)

    # EE world position — env_origin (torch) + ee_pos_local (warp→torch)
    env_origin = raw_env.scene.env_origins[0]                        # (3,)
    ee_local   = wp.to_torch(raw_env._ee_pos_local)[0]               # (3,)
    ee_world   = env_origin + ee_local.to(env_origin.device)         # (3,)

    # Lifted condition
    lifted = (p[:, 2] - p0[:, 2]) > lift_threshold                  # (N,) bool

    # Horizontal proximity to scoop EE
    dx       = p[:, 0] - ee_world[0]
    dy       = p[:, 1] - ee_world[1]
    in_scoop = (dx * dx + dy * dy) < (capture_radius * capture_radius)  # (N,) bool

    return int((lifted & in_scoop).sum().item())


# ---------------------------------------------------------------------------
# Action normalisation — inverse of env's _scale_actions kernel
# ---------------------------------------------------------------------------


def _to_normalised(joint_pos: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    rng = np.where((upper - lower) > 1e-6, upper - lower, 1.0)
    return np.clip(2.0 * (joint_pos - lower) / rng - 1.0, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import gymnasium as gym

    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env     = gym.make(args_cli.task, cfg=env_cfg)
        raw_env = env.unwrapped   # ScoopWarpEnvV3

        device  = raw_env.device

        # Joint limits from the live simulation
        limits = raw_env.robot.data.soft_joint_pos_limits   # (1, 6, 2)
        lower  = limits[0, :, 0].cpu().numpy()              # (6,)
        upper  = limits[0, :, 1].cpu().numpy()              # (6,)

        # Build hover array in the env's actual joint ordering
        actual_names = raw_env.robot.joint_names            # list[str]
        name_to_idx  = {n: i for i, n in enumerate(actual_names)}
        hover = np.zeros(len(actual_names), dtype=np.float32)
        for name, val in _HOVER_CFG.items():
            if name in name_to_idx:
                hover[name_to_idx[name]] = val
            else:
                print(f"[WARN] Joint '{name}' not found in robot — skipping.")

        print(f"[INFO] Robot joints  : {actual_names}")
        print(f"[INFO] Hover pose    : {np.round(hover, 4)}")
        print(f"[INFO] Joint lower   : {np.round(lower, 3)}")
        print(f"[INFO] Joint upper   : {np.round(upper, 3)}")
        print(f"[INFO] Target demos  : {args_cli.num_demos}  (max {args_cli.max_attempts} attempts)")
        print(f"[INFO] Success filter: >={args_cli.min_particles} particles, "
              f"lift>{args_cli.lift_threshold}m, radius<{args_cli.capture_radius}m")
        print()

        os.makedirs(args_cli.output_dir, exist_ok=True)
        rng = np.random.default_rng(args_cli.seed)

        saved    = 0
        attempts = 0

        while saved < args_cli.num_demos and attempts < args_cli.max_attempts:

            # ---- Sample motion parameters ----
            tilt_delta    = float(rng.uniform(args_cli.tilt_min,    args_cli.tilt_max))
            descend_delta = float(rng.uniform(args_cli.descend_min, args_cli.descend_max))
            sweep_mag     = float(rng.uniform(args_cli.sweep_min,   args_cli.sweep_max))
            sweep_delta   = sweep_mag * float(rng.choice([-1.0, 1.0]))   # random direction
            level_frac    = float(rng.uniform(0.4, 0.9))
            steps         = int(rng.integers(args_cli.steps_min, args_cli.steps_max + 1))

            # ---- Build waypoints ----
            waypoints = make_scoop_waypoints(
                hover, tilt_delta, descend_delta, sweep_delta, level_frac, steps, lower, upper
            )   # (T, 6)

            # ---- Execute trajectory ----
            env.reset()
            with torch.inference_mode():
                for joint_pos in waypoints:
                    action_norm = _to_normalised(joint_pos, lower, upper)
                    action = (
                        torch.tensor(action_norm, dtype=torch.float32, device=device)
                        .unsqueeze(0)   # (1, 6) for num_envs=1
                    )
                    env.step(action)

            # ---- Success check ----
            n_captured = count_captured_particles(raw_env, args_cli.lift_threshold, args_cli.capture_radius)
            attempts  += 1
            success    = n_captured >= args_cli.min_particles

            status = "SUCCESS" if success else "fail   "
            print(
                f"  [{attempts:4d}] {status}  captured={n_captured:4d}  "
                f"tilt={tilt_delta:.2f}  descend={descend_delta:.2f}  "
                f"sweep={sweep_delta:+.2f}  level={level_frac:.2f}  steps={steps}"
            )

            if success:
                out_path = os.path.join(args_cli.output_dir, f"demo_{saved:04d}.npz")
                np.savez_compressed(
                    out_path,
                    states      = waypoints,                                          # (T, 6) float32
                    joint_names = np.array(actual_names),                            # (6,) str
                    params      = np.array(                                           # (5,) float32
                        [tilt_delta, descend_delta, sweep_delta, level_frac, float(steps)],
                        dtype=np.float32,
                    ),
                    n_captured  = np.int32(n_captured),
                )
                saved += 1
                print(f"           → saved demo_{saved - 1:04d}.npz  ({saved}/{args_cli.num_demos})")

        env.close()

        success_rate = saved / max(attempts, 1) * 100
        print(f"\n[INFO] Finished: {saved} demos saved from {attempts} attempts "
              f"({success_rate:.1f}% success rate)")
        print(f"[INFO] Output directory: {os.path.abspath(args_cli.output_dir)}")

        if saved < args_cli.num_demos:
            print(
                f"[WARN] Only {saved}/{args_cli.num_demos} demos collected.\n"
                f"       Try: --min_particles {max(1, args_cli.min_particles // 2)}  "
                f"or --lift_threshold {args_cli.lift_threshold * 0.5:.2f}  "
                f"or --max_attempts {args_cli.max_attempts * 2}"
            )


if __name__ == "__main__":
    main()
