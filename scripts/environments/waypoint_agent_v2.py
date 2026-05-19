# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Waypoint-following agent for Isaac-Scoop-Direct-Warp-v2.

v2 uses a unified Newton model (robot + sand + box walls in one ModelBuilder),
so robot–sand and robot–box collision both work correctly.  This runner uses
Newton's own viewer infrastructure instead of IsaacLab's launcher.

Usage::

    # With Newton viewer (requires display)
    python scripts/environments/waypoint_agent_v2.py \\
        --waypoints /home/gmr/Downloads/ur_ws/dataset/demo_20260515_140725.npz

    # Headless (no display required)
    python scripts/environments/waypoint_agent_v2.py \\
        --waypoints /path/to/demo.npz --headless --num_steps 500

    # Loop waypoints indefinitely
    python scripts/environments/waypoint_agent_v2.py \\
        --waypoints /path/to/demo.npz --loop
"""

import argparse
import contextlib
import sys

import numpy as np

# Register the task so gym.make works (not strictly needed here, but keeps parity)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

parser = argparse.ArgumentParser(description="Waypoint agent for Isaac-Scoop-Direct-Warp-v2.")
parser.add_argument(
    "--waypoints",
    type=str,
    default="/home/gmr/Downloads/ur_ws/dataset/demo_20260515_140725.npz",
    help="Path to .npz demo file with keys 'joint_names', 'states'.",
)
parser.add_argument("--loop", action="store_true", default=False, help="Loop waypoints continuously.")
parser.add_argument("--step", type=int, default=1, help="Step size through waypoints (1 = full speed).")
parser.add_argument("--headless", action="store_true", default=False, help="Run without a viewer.")
parser.add_argument("--num_steps", type=int, default=0, help="Stop after N steps (0 = run until done/closed).")
parser.add_argument("--settle_steps", type=int, default=120, help="MPM settling steps at startup.")
parser.add_argument(
    "--box_offset",
    type=float,
    nargs=3,
    default=[0.0, -0.2, 0.0],
    metavar=("X", "Y", "Z"),
    help="Shift sandbox by (x, y, z) metres.",
)
# Camera — sensible default for the v2 scene layout:
#   robot base at (0.5, 0, 0) facing -X, sandbox centred near (0, -0.2, 0.05).
#   Camera sits to the right-front of the robot, elevated, looking down into the sandbox.
parser.add_argument(
    "--camera_pos",
    type=float,
    nargs=3,
    default=[1.5, -1.5, 1.2],
    metavar=("X", "Y", "Z"),
    help="Camera world position.",
)
parser.add_argument("--camera_pitch", type=float, default=-30.0, help="Camera pitch in degrees (- = look down).")
parser.add_argument("--camera_yaw", type=float, default=135.0, help="Camera yaw in degrees.")
args = parser.parse_args()


def _to_normalised(joint_pos_rad: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Convert absolute joint angles [rad] → environment action space [-1, 1]."""
    rng = upper - lower
    rng = np.where(rng > 1e-6, rng, 1.0)
    return np.clip(2.0 * (joint_pos_rad - lower) / rng - 1.0, -1.0, 1.0)


def main():
    # ------------------------------------------------------------------
    # Load demo waypoints
    # ------------------------------------------------------------------
    demo = np.load(args.waypoints)
    joint_names = demo["joint_names"].tolist()
    waypoints_rad = demo["states"]  # (T, 6) absolute joint angles [rad]
    total_steps = len(waypoints_rad)
    print(f"[INFO] Loaded {total_steps} waypoints. Joints: {joint_names}")

    # ------------------------------------------------------------------
    # Build environment (unified Newton model)
    # ------------------------------------------------------------------
    from isaaclab_tasks_experimental.direct.scoop.scoop_env_warp_v2 import (
        ScoopWarpEnvCfgV2,
        ScoopWarpEnvV2,
    )

    cfg = ScoopWarpEnvCfgV2()
    cfg.settle_steps = args.settle_steps
    cfg.box_offset = tuple(args.box_offset)

    env = ScoopWarpEnvV2(cfg)
    env.reset()

    # Convert all waypoints to normalised actions using the model's joint limits
    lower = env._joint_lower
    upper = env._joint_upper
    waypoints_norm = np.stack(
        [_to_normalised(waypoints_rad[i], lower, upper) for i in range(total_steps)]
    )  # (T, 6)

    print(f"[INFO] Joint lower limits: {np.round(lower, 3)}")
    print(f"[INFO] Joint upper limits: {np.round(upper, 3)}")

    # ------------------------------------------------------------------
    # Optional Newton viewer
    # ------------------------------------------------------------------
    viewer = None
    if not args.headless:
        try:
            import warp as wp
            from newton.viewer import ViewerGL
            viewer = ViewerGL()
            env.set_viewer(viewer)
            viewer.set_camera(
                pos=wp.vec3(*args.camera_pos),
                pitch=args.camera_pitch,
                yaw=args.camera_yaw,
            )
            print(f"[INFO] Newton ViewerGL initialised. Camera: pos={args.camera_pos} pitch={args.camera_pitch} yaw={args.camera_yaw}")
        except Exception as exc:
            print(f"[WARN] Could not create Newton viewer ({exc}). Running headless.")

    # ------------------------------------------------------------------
    # Replay loop
    # ------------------------------------------------------------------
    wp_idx = 0
    global_step = 0
    print(f"[INFO] Starting waypoint replay (loop={args.loop}, step={args.step})")

    while True:
        # Viewer exit condition (rendering mode)
        if viewer is not None:
            if not viewer.is_running():
                break
            # Honour the viewer's pause button — render but don't step
            if viewer.is_paused():
                env.render()
                continue

        # Step limit (headless)
        if args.num_steps > 0 and global_step >= args.num_steps:
            print(f"[INFO] Reached {args.num_steps} steps — done.")
            break

        # Advance waypoint index
        if wp_idx >= total_steps:
            if args.loop:
                wp_idx = 0
                env.reset()
                print("[INFO] Waypoints exhausted — looping.")
            else:
                print("[INFO] Waypoints exhausted — done.")
                break

        action = waypoints_norm[wp_idx]  # (6,)
        env.step(action)

        if viewer is not None:
            env.render()

        wp_idx += args.step
        global_step += 1

    if viewer is not None:
        viewer.close()
    env.close()
    print("[INFO] Simulation finished.")


if __name__ == "__main__":
    main()
