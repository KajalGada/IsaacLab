# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable-Baselines3 PPO hyperparameters for Isaac-Sand-Scoop-UR5-v0.

Usage (standalone, no Isaac Sim required):

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/ \\
        manipulation/sand_scoop/config/ur5/agents/sb3_ppo_cfg.py

Or import ``SB3_PPO_CFG`` into your own training script.
"""

from __future__ import annotations

SB3_PPO_CFG: dict = {
    # --- Algorithm ---
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,       # rollout steps per update (single env)
    "batch_size": 512,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    # --- Network ---
    "policy_kwargs": {
        "net_arch": [{"pi": [512, 256, 128], "vf": [512, 256, 128]}],
        "activation_fn": "torch.nn.ELU",
    },
    # --- Total timesteps ---
    "total_timesteps": 5_000_000,
}

# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    import isaaclab_tasks.manager_based.manipulation.sand_scoop  # registers env  # noqa: F401

    cfg = dict(SB3_PPO_CFG)
    total_ts = cfg.pop("total_timesteps")

    # Vectorise across CPU cores for parallel training
    n_envs = 4
    vec_env = make_vec_env(
        "Isaac-Sand-Scoop-UR5-v0",
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    model = PPO(env=vec_env, verbose=1, **cfg)
    model.learn(total_timesteps=total_ts)
    model.save("sand_scoop_ppo")
    print("Training complete — model saved to sand_scoop_ppo.zip")
