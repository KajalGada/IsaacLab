# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay demonstration trajectories through the scoop env to collect (obs, action) pairs for BC.

Each demo .npz file contains absolute joint angles (``states``, shape (T, 6)).  For each
timestep the corresponding normalised action is ``2*(q - lower)/(upper - lower) - 1``, which
is the position target the env's PD controller will track.  We replay every waypoint through
the live simulation so that the full 33-dim observation — including sand centroid, displaced /
elevated fractions — is recorded at each step.

Usage::

    ./isaaclab.sh -p scripts/environments/collect_demo_data.py \\
        --task Isaac-Scoop-Direct-Warp-v3 \\
        --headless \\
        --demo_dir /home/gmr/Downloads/ur_ws/dataset \\
        --output logs/bc_dataset/demo_data.npz
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import sys

import numpy as np
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

parser = argparse.ArgumentParser(description="Collect BC dataset by replaying demos through the scoop env.")
parser.add_argument("--task", type=str, default="Isaac-Scoop-Direct-Warp-v3", help="Gym task ID.")
parser.add_argument(
    "--demo_dir",
    type=str,
    default="/home/gmr/Downloads/ur_ws/dataset",
    help="Directory containing demo *.npz files.",
)
parser.add_argument(
    "--output",
    type=str,
    default="logs/bc_dataset/demo_data.npz",
    help="Output path for the (obs, action) dataset.",
)
parser.add_argument(
    "--glob",
    type=str,
    default="*.npz",
    help="Glob pattern relative to --demo_dir for selecting demo files (default: *.npz).",
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def _to_normalised(joint_pos_rad: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Map absolute joint angles [rad] → normalised action space [-1, 1]."""
    rng = upper - lower
    rng = np.where(rng > 1e-6, rng, 1.0)
    return np.clip(2.0 * (joint_pos_rad - lower) / rng - 1.0, -1.0, 1.0)


def main():
    import gymnasium as gym

    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env = gym.make(args_cli.task, cfg=env_cfg)

        device = env.unwrapped.device
        limits = env.unwrapped.robot.data.soft_joint_pos_limits  # (1, 6, 2)
        lower = limits[0, :, 0].cpu().numpy()  # (6,)
        upper = limits[0, :, 1].cpu().numpy()  # (6,)
        print(f"[INFO] Joint lower limits: {np.round(lower, 3)}")
        print(f"[INFO] Joint upper limits: {np.round(upper, 3)}")

        demo_files = sorted(glob.glob(os.path.join(args_cli.demo_dir, args_cli.glob)))
        if not demo_files:
            raise FileNotFoundError(f"No files matching '{args_cli.glob}' found in {args_cli.demo_dir}")
        print(f"[INFO] Found {len(demo_files)} demo file(s)")

        all_obs: list[np.ndarray] = []
        all_acts: list[np.ndarray] = []

        for demo_path in demo_files:
            demo = np.load(demo_path)
            states = demo["states"]  # (T, 6) absolute joint angles [rad]
            T = len(states)
            waypoints_norm = _to_normalised(states, lower, upper)  # (T, 6) float64

            obs_dict, _ = env.reset()
            obs_t = obs_dict["policy"][0].cpu().numpy()  # (33,)

            ep_obs = np.empty((T, obs_t.shape[0]), dtype=np.float32)
            ep_acts = waypoints_norm.astype(np.float32)

            with torch.inference_mode():
                for t in range(T):
                    ep_obs[t] = obs_t
                    action = torch.from_numpy(waypoints_norm[t]).float().unsqueeze(0).to(device)
                    obs_dict, *_ = env.step(action)
                    obs_t = obs_dict["policy"][0].cpu().numpy()

            all_obs.append(ep_obs)
            all_acts.append(ep_acts)
            print(f"[INFO]   {os.path.basename(demo_path)}: {T} steps collected")

        env.close()

        observations = np.concatenate(all_obs, axis=0)
        actions = np.concatenate(all_acts, axis=0)

        out_dir = os.path.dirname(os.path.abspath(args_cli.output))
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(args_cli.output, observations=observations, actions=actions)

        print(f"[INFO] Dataset saved → {args_cli.output}")
        print(f"[INFO] observations: {observations.shape}, actions: {actions.shape}")
        print(f"[INFO] obs  range  : [{observations.min():.3f}, {observations.max():.3f}]")
        print(f"[INFO] acts range  : [{actions.min():.3f}, {actions.max():.3f}]")


if __name__ == "__main__":
    main()
