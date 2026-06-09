# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioural Cloning pre-training for Isaac-Scoop-Direct-Warp-v3.

Trains a policy (identical MLP architecture to the rl_games PPO actor) on the
(obs, action) dataset produced by ``collect_demo_data.py`` using MSE loss.
Saves an rl_games-compatible ``.pth`` checkpoint that can be loaded directly
via the ``--checkpoint`` argument of the rl_games training script for PPO
fine-tuning.

Usage::

    # Train BC policy
    python3 scripts/reinforcement_learning/bc_pretrain_scoop.py \\
        --dataset logs/bc_dataset/demo_data.npz \\
        --output  logs/bc_pretrain/scoop_bc.pth \\
        --epochs  100

    # Fine-tune with PPO starting from the BC checkpoint
    ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \\
        --task Isaac-Scoop-Direct-Warp-v3 \\
        --num_envs 16 \\
        --checkpoint logs/bc_pretrain/scoop_bc.pth
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser(description="BC pre-training for scoop env.")
parser.add_argument("--dataset",    type=str,   default="logs/bc_dataset/demo_data.npz")
parser.add_argument("--output",     type=str,   default="logs/bc_pretrain/scoop_bc.pth")
parser.add_argument("--epochs",     type=int,   default=100)
parser.add_argument("--batch_size", type=int,   default=256)
parser.add_argument("--lr",         type=float, default=3e-4)
parser.add_argument("--device",     type=str,   default="cuda:0")
args = parser.parse_args()

OBS_DIM = 33
ACT_DIM = 6


# ---------------------------------------------------------------------------
# Modules matching rl_games' model layout exactly
# ---------------------------------------------------------------------------


class RunningMeanStd(nn.Module):
    """Online obs normalizer whose buffer names match rl_games exactly.

    Initialised offline from the full dataset via :meth:`fit`, then frozen
    for the BC training loop.  PPO will continue updating it after loading.
    """

    def __init__(self, size: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(size))
        self.register_buffer("running_var",  torch.ones(size))
        self.register_buffer("count",        torch.ones(()))

    def fit(self, data: torch.Tensor) -> None:
        self.running_mean.copy_(data.mean(0))
        self.running_var.copy_(data.var(0).clamp(min=1e-8))
        self.count.fill_(float(data.shape[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.running_mean) / (self.running_var.sqrt() + self.epsilon)


class ScoopBCNet(nn.Module):
    """MLP actor + critic skeleton matching the rl_games a2c_continuous_logstd layout.

    Only the actor MLP and mu head are trained.  The value head is initialised
    randomly (rl_games will train it during PPO fine-tuning).
    The obs normalizer is fitted once from the demo dataset and frozen.
    """

    def __init__(self):
        super().__init__()
        # top-level normalizers (rl_games model state-dict prefix, no a2c_network. prefix)
        self.running_mean_std = RunningMeanStd(OBS_DIM)
        self.value_mean_std   = RunningMeanStd(1)

        # MLP layers at indices 0, 2, 4 in Sequential (ELU at 1, 3, 5)
        self.actor_mlp = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.ELU(),
            nn.Linear(256,     256), nn.ELU(),
            nn.Linear(256,     128), nn.ELU(),
        )
        self.mu    = nn.Linear(128, ACT_DIM)
        self.sigma = nn.Parameter(torch.zeros(ACT_DIM))  # log-std (unused in BC)
        self.value = nn.Linear(128, 1)                   # trained by PPO, not BC

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.running_mean_std(obs)
        x = self.actor_mlp(x)
        return torch.tanh(self.mu(x))


# ---------------------------------------------------------------------------
# Checkpoint builder
# ---------------------------------------------------------------------------


def build_rl_games_checkpoint(net: ScoopBCNet) -> dict:
    """Map ScoopBCNet state dict → rl_games model key layout.

    rl_games stores top-level normalizer buffers without a sub-module prefix
    (e.g. ``running_mean_std.running_mean``) and wraps the network layers under
    ``a2c_network.*``.  This function re-keys our state dict to match exactly.

    An empty Adam optimizer state is included so that rl_games' ``set_full_state_weights``
    (which unconditionally calls ``optimizer.load_state_dict``) does not raise KeyError.
    The state is empty (no momentum buffers) because rl_games will warm it up during the
    first PPO backward pass.  ``len(param_groups[0]['params'])`` must equal the number of
    trainable parameters in the rl_games model (11), which matches ``ScoopBCNet.parameters()``.
    """
    # Build a fresh Adam over all ScoopBCNet params to get a structurally valid
    # optimizer state dict (state={}, param_groups with 11 entries).
    dummy_opt = Adam(net.parameters(), lr=1e-4)
    opt_state = dummy_opt.state_dict()   # state is empty before any .step()

    sd = net.state_dict()
    model_state = {
        # obs normalizer
        "running_mean_std.running_mean": sd["running_mean_std.running_mean"],
        "running_mean_std.running_var":  sd["running_mean_std.running_var"],
        "running_mean_std.count":        sd["running_mean_std.count"],
        # value normalizer — defaults; PPO will update
        "value_mean_std.running_mean":   torch.zeros(1),
        "value_mean_std.running_var":    torch.ones(1),
        "value_mean_std.count":          torch.ones(()),
        # actor MLP (layers at sequential indices 0, 2, 4)
        "a2c_network.actor_mlp.0.weight": sd["actor_mlp.0.weight"],
        "a2c_network.actor_mlp.0.bias":   sd["actor_mlp.0.bias"],
        "a2c_network.actor_mlp.2.weight": sd["actor_mlp.2.weight"],
        "a2c_network.actor_mlp.2.bias":   sd["actor_mlp.2.bias"],
        "a2c_network.actor_mlp.4.weight": sd["actor_mlp.4.weight"],
        "a2c_network.actor_mlp.4.bias":   sd["actor_mlp.4.bias"],
        # action head
        "a2c_network.mu.weight": sd["mu.weight"],
        "a2c_network.mu.bias":   sd["mu.bias"],
        "a2c_network.sigma":     sd["sigma"],
        # value head (random init; PPO trains from here)
        "a2c_network.value.weight": sd["value.weight"],
        "a2c_network.value.bias":   sd["value.bias"],
    }
    return {
        "model":             model_state,
        "optimizer":         opt_state,   # required by rl_games set_full_state_weights
        "epoch":             0,
        "frame":             0,
        "last_mean_rewards": -1e8,
        "env_state":         None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load dataset
    data   = np.load(args.dataset)
    obs_np = data["observations"].astype(np.float32)
    act_np = data["actions"].astype(np.float32)
    print(f"[INFO] Dataset: {obs_np.shape[0]} samples  obs={obs_np.shape[1]}  act={act_np.shape[1]}")

    obs_t = torch.from_numpy(obs_np).to(device)
    act_t = torch.from_numpy(act_np).to(device)

    # Build network and fit obs normalizer from the full dataset
    net = ScoopBCNet().to(device)
    net.running_mean_std.fit(obs_t)
    # Freeze normalizer — only actor MLP + mu head are trained
    for p in net.running_mean_std.parameters():
        p.requires_grad_(False)
    for buf in net.running_mean_std.buffers():
        buf.requires_grad_(False)

    print(
        f"[INFO] Obs normalizer: mean [{obs_t.mean(0).min():.3f}, {obs_t.mean(0).max():.3f}]"
        f"  std [{obs_t.std(0).min():.3f}, {obs_t.std(0).max():.3f}]"
    )

    dataset = TensorDataset(obs_t, act_t)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    optimizer = Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"[INFO] Training for {args.epochs} epochs  batch={args.batch_size}  lr={args.lr}")
    for epoch in range(args.epochs):
        net.train()
        total_loss = 0.0
        for obs_b, act_b in loader:
            pred = net(obs_b)
            loss = F.mse_loss(pred, act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * obs_b.shape[0]
        scheduler.step()
        mean_loss = total_loss / len(dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:4d}/{args.epochs}  loss={mean_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    net.eval()
    ckpt    = build_rl_games_checkpoint(net)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    torch.save(ckpt, args.output)
    print(f"[INFO] BC checkpoint saved → {args.output}")

    # Quick sanity-check: verify all expected keys are present
    loaded = torch.load(args.output, weights_only=False)
    expected = {
        "running_mean_std.running_mean", "running_mean_std.running_var", "running_mean_std.count",
        "value_mean_std.running_mean",   "value_mean_std.running_var",   "value_mean_std.count",
        "a2c_network.actor_mlp.0.weight", "a2c_network.actor_mlp.0.bias",
        "a2c_network.actor_mlp.2.weight", "a2c_network.actor_mlp.2.bias",
        "a2c_network.actor_mlp.4.weight", "a2c_network.actor_mlp.4.bias",
        "a2c_network.mu.weight", "a2c_network.mu.bias",
        "a2c_network.sigma",
        "a2c_network.value.weight", "a2c_network.value.bias",
    }
    missing = expected - set(loaded["model"].keys())
    if missing:
        raise RuntimeError(f"Checkpoint is missing keys: {missing}")
    print("[INFO] Checkpoint key verification passed.")


if __name__ == "__main__":
    main()
