# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent script for the Newton MPM sand-scooping environment.

Three policy modes
------------------
``demo``   (default with viewer) — replay the waypoint scoop trajectory from
           ``simulation_newton_sand_v1.py`` so the robot performs the full
           back-and-forth sand sweep exactly as in the original Newton example.

``zero``   — hold the arm at its initial pose; useful for inspecting physics.

``random`` — sample random joint-delta actions each step.

Usage examples
--------------
GL viewer with demo trajectory (default)::

    ./isaaclab.sh -p scripts/environments/sand_scoop_zero_agent.py \\
        --task Isaac-Sand-Scoop-UR5-v0

GL viewer, finer MPM grid::

    ./isaaclab.sh -p scripts/environments/sand_scoop_zero_agent.py \\
        --task Isaac-Sand-Scoop-UR5-Play-v0

Hold pose (zero policy)::

    ./isaaclab.sh -p scripts/environments/sand_scoop_zero_agent.py \\
        --task Isaac-Sand-Scoop-UR5-v0 --policy zero

Random actions::

    ./isaaclab.sh -p scripts/environments/sand_scoop_zero_agent.py \\
        --task Isaac-Sand-Scoop-UR5-v0 --policy random

Headless throughput benchmark (requires stable-baselines3)::

    ./isaaclab.sh -p scripts/environments/sand_scoop_zero_agent.py \\
        --task Isaac-Sand-Scoop-UR5-v0 --num_envs 4 --num_steps 500
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

# Register all sand-scoop gym IDs before anything else.
import isaaclab_tasks.manager_based.manipulation.sand_scoop  # noqa: F401
import gymnasium as gym

# ---------------------------------------------------------------------------
# Waypoints — identical to simulation_newton_sand_v1.py so the demo produces
# the same scoop trajectory as the standalone Newton example.
# ---------------------------------------------------------------------------
_W3 = math.pi / 2
_Q_ABOVE = np.array([-0.27, -1.50, 1.70, math.pi / 2, math.pi / 2, _W3], dtype=np.float32)
_Q_SIDE_A = np.array([-0.55, -0.50, 1.00, math.pi / 2, math.pi / 2, _W3], dtype=np.float32)
_Q_SIDE_B = np.array([0.20, -0.50, 1.00, math.pi / 2, math.pi / 2, _W3], dtype=np.float32)

_WAYPOINTS: list[tuple[np.ndarray, float]] = [
    (_Q_ABOVE, 2.5),   # hover — let sand settle
    (_Q_SIDE_A, 1.5),  # descend to side A
    (_Q_SIDE_B, 2.5),  # sweep 1 → B
    (_Q_SIDE_A, 2.5),  # sweep 2 ← A
    (_Q_SIDE_B, 2.5),  # sweep 3 → B
    (_Q_SIDE_A, 2.5),  # sweep 4 ← A
    (_Q_SIDE_B, 2.5),
    (_Q_SIDE_A, 2.5),
    (_Q_SIDE_B, 2.5),
    (_Q_SIDE_A, 2.5),
    (_Q_ABOVE, 1.5),   # lift out — loop repeats
]


class WaypointPolicy:
    """Interpolates through the scoop waypoints and drives the UR5 arm.

    Rather than computing a delta action and fighting the action-scale
    clipping, this policy directly overwrites ``env.unwrapped._joint_targets``
    with the interpolated waypoint position before each ``step(zeros)``.
    Because ``step`` applies ``_joint_targets + action * scale`` and action
    is all-zeros, the PD controller receives exactly the waypoint target each
    frame — matching the behaviour of the original Newton simulation.
    """

    def __init__(self, fps: float = 60.0):
        self._frame_dt = 1.0 / fps
        self._wp_idx = 0
        self._elapsed = 0.0

    def reset(self) -> None:
        self._wp_idx = 0
        self._elapsed = 0.0

    def get_joint_target(self) -> np.ndarray:
        """Return interpolated joint target for the current frame."""
        q_target, duration = _WAYPOINTS[self._wp_idx]
        prev_idx = (self._wp_idx - 1) % len(_WAYPOINTS)
        q_prev, _ = _WAYPOINTS[prev_idx]
        t = min(self._elapsed / duration, 1.0)
        return (1.0 - t) * q_prev + t * q_target

    def step(self) -> None:
        """Advance internal timer by one frame."""
        _, duration = _WAYPOINTS[self._wp_idx]
        self._elapsed += self._frame_dt
        if self._elapsed >= duration:
            self._wp_idx = (self._wp_idx + 1) % len(_WAYPOINTS)
            self._elapsed = 0.0


# ---- CLI ---------------------------------------------------------------
parser = argparse.ArgumentParser(description="Agent script for sand-scooping Newton MPM environments.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Sand-Scoop-UR5-v0",
    help="Gym environment ID.",
)
parser.add_argument(
    "--policy",
    type=str,
    default="demo",
    choices=["demo", "zero", "random"],
    help=(
        "'demo' replays the waypoint scoop trajectory (default). "
        "'zero' holds the arm in place. "
        "'random' samples random actions."
    ),
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help=(
        "Number of parallel environments. "
        "When > 1 the environments run as headless SubprocVecEnv workers (no GL viewer). "
        "Newton MPM does not support GPU-batched multi-env; each worker is a separate process."
    ),
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=0,
    help="Max total steps before exit. 0 = run until the GL window is closed (single-env only).",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--voxel_size",
    type=float,
    default=0.01,
    help=(
        "MPM grid cell size [m] and particle-spacing basis. "
        "Matches simulation_newton_sand_v1.py default (0.01). "
        "Use 0.06 for fast RL training (fewer, larger particles)."
    ),
)
parser.add_argument(
    "--particles_per_cell",
    type=int,
    default=3,
    help=(
        "Particles per MPM cell. "
        "Matches simulation_newton_sand_v1.py default (3). "
        "Use 1 for faster simulation with fewer particles."
    ),
)
args = parser.parse_args()


# ---- Main --------------------------------------------------------------
def main():
    np.random.seed(args.seed)
    use_viewer = args.num_envs == 1

    # -- Build env(s) ----------------------------------------------------
    if args.num_envs == 1:
        # Directly instantiate so we can override voxel_size / particles_per_cell
        # without fighting gym.make's fixed registered kwargs.
        from isaaclab_tasks.manager_based.manipulation.sand_scoop.sand_scoop_env import SandScoopEnv
        from isaaclab_tasks.manager_based.manipulation.sand_scoop.sand_scoop_env_cfg import SandScoopEnvCfg

        cfg = SandScoopEnvCfg()
        cfg.voxel_size = args.voxel_size
        cfg.particles_per_cell = args.particles_per_cell
        env = SandScoopEnv(cfg, render_mode="human")
        raw_env = env
    else:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        env = SubprocVecEnv([lambda: gym.make(args.task, render_mode=None)] * args.num_envs)
        raw_env = None

    # -- Policy setup ----------------------------------------------------
    fps = raw_env.cfg.fps if raw_env is not None else 60.0
    waypoint_policy = WaypointPolicy(fps=fps)

    print(f"[INFO] Task:         {args.task}")
    print(f"[INFO] Num envs:     {args.num_envs}")
    print(f"[INFO] Policy:       {args.policy}")
    print(f"[INFO] Obs space:    {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")
    print()

    # -- Reset -----------------------------------------------------------
    if args.num_envs == 1:
        obs, _ = env.reset(seed=args.seed)
        waypoint_policy.reset()
    else:
        obs = env.reset()

    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    total_steps = 0
    episode = 0
    ep_return = 0.0
    t_start = time.perf_counter()

    # -- Run loop --------------------------------------------------------
    while True:
        # Exit conditions
        if args.num_steps > 0 and total_steps >= args.num_steps:
            break
        if use_viewer and not raw_env.viewer_is_running:
            break

        # Pause: keep rendering but skip physics when viewer is paused.
        if use_viewer and raw_env._viewer is not None and raw_env._viewer.is_paused():
            env.render()
            continue

        # -- Compute action ----------------------------------------------
        if args.num_envs == 1:
            assert raw_env is not None
            if args.policy == "demo":
                # Directly set the joint target so the PD controller tracks
                # the waypoint exactly, then pass zero delta action.
                raw_env._joint_targets = waypoint_policy.get_joint_target()
                waypoint_policy.step()
                action = zero_action
            elif args.policy == "random":
                action = env.action_space.sample()
            else:
                action = zero_action
        else:
            action = np.stack([zero_action] * args.num_envs)

        # -- Step --------------------------------------------------------
        if args.num_envs == 1:
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            ep_return += float(reward)
            total_steps += 1

            if terminated or truncated:
                elapsed = time.perf_counter() - t_start
                sps = total_steps / max(elapsed, 1e-9)
                print(
                    f"[Episode {episode:4d}]  steps={total_steps:6d}"
                    f"  return={ep_return:8.3f}"
                    f"  z_mean={info['sand_z_mean']:.3f}"
                    f"  frac_above={info['sand_frac_above']:.2f}"
                    f"  {sps:.0f} steps/s"
                )
                episode += 1
                ep_return = 0.0
                obs, _ = env.reset()
                waypoint_policy.reset()
        else:
            obs, rewards, dones, infos = env.step(action)
            total_steps += 1

            if total_steps % 100 == 0:
                elapsed = time.perf_counter() - t_start
                sps = args.num_envs * total_steps / max(elapsed, 1e-9)
                print(f"[Step {total_steps:6d}]  {sps:.0f} env-steps/s")

    # -- Summary ---------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    sps = args.num_envs * total_steps / max(elapsed, 1e-9)
    print(f"\n[INFO] Done.  {args.num_envs * total_steps} env-steps in {elapsed:.1f}s  →  {sps:.0f} steps/s")

    env.close()


if __name__ == "__main__":
    main()
