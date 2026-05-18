# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint-following agent for UR5 sand-scooping tasks.

Replays joint-position waypoints from a recorded demonstration (.npz) by
converting absolute joint angles to the environment's normalised [-1, 1]
action space, then broadcasting the same waypoint to all parallel environments.

Usage::

    ./isaaclab.sh -p scripts/environments/waypoint_agent.py \\
        --task Isaac-Scoop-Direct-Warp-v0 \\
        --num_envs 1 \\
        --waypoints /home/gmr/Downloads/ur_ws/dataset/demo_20260515_140725.npz

Add ``--loop`` to cycle the trajectory indefinitely.
Add ``--step N`` to replay every N-th waypoint (useful to speed up long demos).
"""

import argparse
import contextlib
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

parser = argparse.ArgumentParser(description="Waypoint-following agent for Isaac Lab scoop environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Scoop-Direct-Warp-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--waypoints",
    type=str,
    default="/home/gmr/Downloads/ur_ws/dataset/demo_20260515_140725.npz",
    help="Path to the .npz demo file with keys 'joint_names', 'states', 'actions'.",
)
parser.add_argument(
    "--loop",
    action="store_true",
    default=False,
    help="Loop the waypoints indefinitely instead of stopping at the end.",
)
parser.add_argument(
    "--step",
    type=int,
    default=1,
    help="Step size through the waypoint array (1 = full speed, 2 = 2x speed, etc.).",
)
# append AppLauncher cli args
add_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# pass remaining args to Hydra
sys.argv = [sys.argv[0]] + hydra_args

# PLACEHOLDER: Extension template (do not remove this comment)


def _to_normalised_actions(joint_pos_rad: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Convert absolute joint positions [rad] to the environment's [-1, 1] action space.

    The environment's ``_scale_actions`` kernel maps actions as:
        target = 0.5 * (a + 1) * (upper - lower) + lower

    The inverse is:
        a = 2 * (q - lower) / (upper - lower) - 1
    """
    range_ = upper - lower
    # Guard against degenerate joints with zero range (shouldn't happen for UR5)
    range_ = np.where(range_ > 1e-6, range_, 1.0)
    return np.clip(2.0 * (joint_pos_rad - lower) / range_ - 1.0, -1.0, 1.0)


def main():
    """Waypoint-following agent: replays a recorded UR5 joint trajectory."""

    # ------------------------------------------------------------------
    # Load demo
    # ------------------------------------------------------------------
    demo = np.load(args_cli.waypoints)
    joint_names: list[str] = demo["joint_names"].tolist()
    # states: (T, 6) absolute joint positions [rad]
    waypoints_rad: np.ndarray = demo["states"]
    total_steps = len(waypoints_rad)
    print(f"[INFO]: Loaded {total_steps} waypoints from '{args_cli.waypoints}'")
    print(f"[INFO]: Joint order: {joint_names}")

    # ------------------------------------------------------------------
    # Build environment
    # ------------------------------------------------------------------
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        env = gym.make(args_cli.task, cfg=env_cfg)

        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")

        env.reset()

        device = env.unwrapped.device
        num_envs = env.unwrapped.num_envs

        # ------------------------------------------------------------------
        # Retrieve joint limits from the running environment.
        # soft_joint_pos_limits: (num_envs, num_joints, 2) — (lower, upper) [rad]
        # All envs share the same limits when replicate_physics=True.
        # ------------------------------------------------------------------
        soft_limits = env.unwrapped.robot.data.soft_joint_pos_limits  # torch.Tensor
        lower = soft_limits[0, :, 0].cpu().numpy()  # (num_joints,)
        upper = soft_limits[0, :, 1].cpu().numpy()  # (num_joints,)
        print(f"[INFO]: Joint lower limits [rad]: {np.round(lower, 4)}")
        print(f"[INFO]: Joint upper limits [rad]: {np.round(upper, 4)}")

        # ------------------------------------------------------------------
        # Pre-compute normalised actions for every waypoint.
        # ------------------------------------------------------------------
        waypoints_norm = np.stack(
            [_to_normalised_actions(waypoints_rad[i], lower, upper) for i in range(total_steps)],
            axis=0,
        )  # (T, num_joints)

        # ------------------------------------------------------------------
        # Replay loop
        # ------------------------------------------------------------------
        sim = env.unwrapped.sim
        wp_idx = 0  # index into waypoints_norm

        print(f"[INFO]: Starting waypoint replay (loop={args_cli.loop}, step={args_cli.step})")

        while True:
            # Exit when the viewer window is closed (rendering mode)
            if sim.visualizers:
                if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break

            # Advance waypoint index
            if wp_idx >= total_steps:
                if args_cli.loop:
                    wp_idx = 0
                    print(f"[INFO]: Waypoints exhausted — looping back to start.")
                else:
                    print(f"[INFO]: Waypoints exhausted — done.")
                    break

            with torch.inference_mode():
                # Broadcast the same waypoint to all parallel environments
                action_np = waypoints_norm[wp_idx]  # (num_joints,)
                action = torch.tensor(action_np, dtype=torch.float32, device=device)
                action = action.unsqueeze(0).expand(num_envs, -1)  # (num_envs, num_joints)
                env.step(action)

            wp_idx += args_cli.step

        env.close()


if __name__ == "__main__":
    main()
