# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive demo for the kinetic-sand scooping environment.

Runs the SandScoopEnv with Newton's OpenGL viewer and a random policy so
you can watch the physics before attaching a trained agent.

Usage
-----
    ./isaaclab.sh -p scripts/environments/sand_scoop_demo.py

Optional flags (passed through to Newton's example framework):

    --num_frames 2000   total frames to render (default: 2000)
    --fps 60            render / simulation frame-rate
    --substeps 4        MPM substeps per frame

To run a trained SB3 policy instead of random actions, pass ``--policy``::

    ./isaaclab.sh -p scripts/environments/sand_scoop_demo.py \\
        --policy sand_scoop_ppo.zip
"""

from __future__ import annotations

import argparse

import numpy as np
import newton
import newton.examples

# Register "Isaac-Sand-Scoop-UR5-v0" in the gym registry
import isaaclab_tasks.manager_based.manipulation.sand_scoop  # noqa: F401

from isaaclab_tasks.manager_based.manipulation.sand_scoop import SandScoopEnv, SandScoopEnvCfg


class SandScoopDemo:
    """Thin Newton-example adapter around :class:`SandScoopEnv`.

    Newton's ``run()`` helper calls ``step()`` then ``render()`` every frame,
    so we just delegate to the gymnasium env and forward the Newton state to
    the viewer.
    """

    def __init__(self, viewer: newton.viewer.ViewerBase, args: argparse.Namespace) -> None:
        self.viewer   = viewer
        self.sim_time = 0.0
        self._episode = 0

        cfg = SandScoopEnvCfg()
        # Override timing from CLI if provided
        if hasattr(args, "fps") and args.fps:
            cfg.sim_dt = 1.0 / args.fps
        if hasattr(args, "substeps") and args.substeps:
            cfg.mpm_substeps = args.substeps

        print("[SandScoopDemo] Building env (sand settling takes ~3 s)…")
        self.env = SandScoopEnv(cfg=cfg)
        self.obs, _ = self.env.reset()
        self._done = False

        # Side A / Side B: same lift/elbow/wrist as in_sand_q, only shoulder_pan
        # varies to sweep the scoop left↔right through the source container.
        #   pan=+0.10 → scoop y≈0.16 (front of container)
        #   pan=+0.55 → scoop y≈0.36 (back of container)
        _,lift,elbow,w1,w2,w3 = cfg.in_sand_q
        side_a = (0.10, lift, elbow, w1, w2, w3)
        side_b = (0.55, lift, elbow, w1, w2, w3)

        # Waypoints: (target_q, duration_in_demo_steps)
        # 1 demo step ≈ policy_decimation * sim_dt = 3/60 s → 20 steps/s
        # 2.5 s ≈ 50 steps,  1.5 s ≈ 30 steps
        self._waypoints = [
            (cfg.home_q, 50),   # hover above sand
            (side_a,     30),   # descend into sand at side A
            (side_b,     50),   # sweep → side B
            (side_a,     50),   # sweep ← side A
            (side_b,     50),   # sweep → side B
            (side_a,     50),   # sweep ← side A
            (side_b,     50),   # sweep → side B
            (side_a,     50),   # sweep ← side A
            (cfg.home_q, 30),   # lift out
        ]
        self._wp_idx    = 0
        self._wp_frame  = 0
        self._prev_q    = np.array(cfg.home_q, dtype=np.float64)

        # Wire Newton model/state into the viewer
        self.viewer.set_model(self.env.model)
        self.viewer.show_particles = True

        # Optional: load a trained SB3 policy
        self._policy = None
        if hasattr(args, "policy") and args.policy:
            from stable_baselines3 import PPO
            self._policy = PPO.load(args.policy)
            print(f"[SandScoopDemo] Loaded policy from {args.policy}")

    # ------------------------------------------------------------------
    # Newton example interface
    # ------------------------------------------------------------------

    def step(self) -> None:
        if self._done:
            self._episode += 1
            sand_frac = (
                self.env._count_target() / max(self.env._num_particles, 1)
            )
            print(
                f"[SandScoopDemo] Episode {self._episode} ended "
                f"(sand in target: {sand_frac:.1%}) — resetting."
            )
            self.obs, _ = self.env.reset()
            self._done = False

        # Policy: trained model, scripted waypoints, or random fallback
        if self._policy is not None:
            action, _ = self._policy.predict(self.obs, deterministic=True)
            self.obs, _reward, self._done, _trunc, _info = self.env.step(action)
        else:
            # Scripted waypoint control with linear interpolation (v1.py style).
            target_q, duration = self._waypoints[self._wp_idx]
            t = min(self._wp_frame / duration, 1.0)
            q_interp = (1.0 - t) * self._prev_q + t * np.array(target_q, dtype=np.float64)

            q_np = self.env.control.joint_target_pos.numpy()
            q_np[:self.env._robot_dof] = q_interp
            self.env.control.joint_target_pos.assign(q_np)

            dt = self.env.cfg.sim_dt / self.env.cfg.mpm_substeps
            for _ in range(self.env.cfg.policy_decimation):
                self.env._sim_frame(dt)
            self.obs = self.env._get_obs()

            self._wp_frame += 1
            if self._wp_frame >= duration:
                self._prev_q   = np.array(target_q, dtype=np.float64)
                self._wp_frame = 0
                self._wp_idx   = (self._wp_idx + 1) % len(self._waypoints)
        self.sim_time += self.env.cfg.sim_dt * self.env.cfg.policy_decimation

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.env.state)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Argument parser
    # ------------------------------------------------------------------

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=2000)
        parser.add_argument(
            "--policy",
            type=str,
            default=None,
            help="Path to a saved SB3 PPO .zip to run instead of random actions.",
        )
        return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = SandScoopDemo.create_parser()
    viewer, args = newton.examples.init(parser)
    demo = SandScoopDemo(viewer, args)
    newton.examples.run(demo, args)
