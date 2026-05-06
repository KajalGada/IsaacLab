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

        # Policy: trained model or small random exploration
        if self._policy is not None:
            action, _ = self._policy.predict(self.obs, deterministic=True)
        else:
            # Random policy with small magnitude so the arm moves gently
            action = self.env.action_space.sample() * 0.15

        self.obs, _reward, self._done, _trunc, _info = self.env.step(action)
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
