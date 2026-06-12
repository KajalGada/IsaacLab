# Isaac Lab — Scoop Task: Offboarding Guide

This document is an offboarding reference for the UR5 sand-scooping work built on top of the [IsaacLab](https://github.com/isaac-sim/IsaacLab) framework. It covers what was built, why decisions were made, and how to continue the work.

> **Fork:** [github.com/KajalGada/IsaacLab](https://github.com/KajalGada/IsaacLab) — a fork of `isaac-sim/IsaacLab`.

---

## What Was Built

A complete RL training pipeline for teaching a UR5 robot arm to scoop kinetic sand, using Isaac Lab's Newton/Warp physics engine. The pipeline covers:

- **Three task variants** (`v1` through `v3`) — progressive improvements to the simulation architecture
- **Real-robot demo collection** — replaying UR5 recordings through simulation to generate labeled training data
- **Behavior cloning (BC) pre-training** — initializing the policy from demonstrations before RL
- **PPO fine-tuning** — refining the BC policy with environment rewards

The production environment is `Isaac-Scoop-Direct-Warp-v3`.

---

## Repository Layout

Everything in this repo is upstream IsaacLab **except** the following:

### Custom additions

| Path | Description |
|------|-------------|
| `source/isaaclab_tasks_experimental/.../direct/scoop/` | All scoop task source (env, physics, assets, config) |
| `scripts/environments/collect_demo_data.py` | Step 1: replay real-robot demos through sim to collect (obs, action) pairs |
| `scripts/environments/generate_scoop_demos.py` | Fallback: synthesize demos using scripted motion primitives |
| `scripts/environments/waypoint_agent.py` | Scripted waypoint controller (v1) |
| `scripts/environments/waypoint_agent_v2.py` | Scripted waypoint controller (v2, improved) |
| `scripts/environments/zero_agent.py` | Baseline: always output zero actions |
| `scripts/environments/random_agent.py` | Baseline: random actions |
| `scripts/reinforcement_learning/bc_pretrain_scoop.py` | Step 2: BC pre-training from demo dataset |
| `logs/bc_dataset/demo_data.npz` | Collected BC dataset (obs, actions) |
| `logs/bc_pretrain/scoop_bc.pth` | BC-trained checkpoint |

### Scoop task directory

```
source/isaaclab_tasks_experimental/isaaclab_tasks_experimental/direct/scoop/
├── __init__.py                     # Gymnasium task registration (v1–v3 + MPM-v0)
├── README.md                       # v3 training pipeline quick-reference
├── scoop_env_warp_v1.py            # v1: standalone environment class + config
├── scoop_env_warp_v2.py            # v2: unified Newton model (research reference)
├── scoop_env_warp_v3.py            # v3: production environment
├── sand_mpm.py                     # SandMPMHelper — Warp kernels for MPM sand
├── scooping_env.py                 # Legacy MPM-only environment (Isaac-Scoop-MPM-Direct-v0)
├── scooping_env_cfg.py             # Legacy config
├── agents/
│   └── rl_games_ppo_cfg.yaml       # PPO hyperparameters for rl_games
└── assets/
    ├── ur5_with_scoop.usd          # Main robot asset (USD)
    ├── ur5_with_scoop.urdf         # URDF version (used by v2 Newton loader)
    ├── *.stl                       # Link meshes for scoop and arm
    └── create_ur5_scoop_usd.py     # Script to regenerate USD from URDF
```

---

## Task Evolution: v1 → v3

The task went through three iterations, each addressing a specific limitation.

### v1 — `Isaac-Scoop-Direct-Warp-v1` (baseline)

**Architecture:** Runs the UR5 through IsaacLab's standard Articulation API (Newton MJWarp solver). Sand is modeled separately as a `SandMPMHelper` (Newton implicit MPM). The scoop pose is copied to kinematic proxy bodies each step, which then collide with particles — one-way coupling only. Voxel size 0.02 m with default material parameters. Sandbox shifted via `box_offset=(1.0, 0.1, 0)`.

**Problem:** Coarse voxel grid (0.02 m) and rough default material parameters. Particles can tunnel through the scoop between control steps because the robot pose update and the MPM solve are separate.

**Status:** Superseded by v3.

### v2 — `Isaac-Scoop-Direct-Warp-v2` (research reference)

**Architecture:** Completely different from v1. Instead of IsaacLab's Articulation wrapper, the robot is loaded directly via Newton's `ModelBuilder` from the URDF. The robot (via `SolverMuJoCo`) and sand (via `SolverImplicitMPM`) live in the **same Newton model** and share the same physics state. This gives true bidirectional collision — sand pushes the scoop and the scoop pushes sand, resolved in the same timestep with no lag.

**Why it matters:**  Material parameters were tuned here to match a standalone reference Newton-sand script: voxel size 0.01 m, `young_modulus=8000`, `friction=1.4`, and so on.

**The limitation:** Newton's `ModelBuilder` does not yet support instancing multiple identical scenes. v2 can only run a **single environment**. Multi-env parallelism — the main throughput lever for PPO — is unavailable.

**Status:** Keep as a research reference. If Newton adds per-env scene instancing in a future release, v2's architecture is superior to v3 and should become the new production base.

### v3 — `Isaac-Scoop-Direct-Warp-v3` (production)

**Architecture:** Returns to the dual-model architecture of v1 (IsaacLab Articulation + separate MPM helper) to keep multi-env support, but adopts all improvements from v2:

- **Finer voxel grid:** `voxel_size=0.01 m` (was 0.02 m in v1)
- **Reference material params:** matches the v2-tuned values (see table below)
- **Particle tunneling fix:** `_project_outside=True` — before every MPM grid solve, Newton's `_project_outside` repairs particles that tunnelled through the scoop mesh since the last step. This mirrors lines 246-247 of the reference Newton-sand script.

**Status:** Use this for all new work.

---

## v3 Environment Reference

**Registered ID:** `Isaac-Scoop-Direct-Warp-v3`
**Entry point:** `scoop_env_warp_v3.py` → `ScoopWarpEnvV3`

### Key files

| File | Role |
|------|------|
| `scoop_env_warp_v3.py` | Env loop, Warp kernels (obs, reward, reset), proxy-body coupling |
| `sand_mpm.py` | `SandMPMHelper` — particle init, MPM step, sand obs computation |
| `agents/rl_games_ppo_cfg.yaml` | PPO algorithm config for rl_games |
| `assets/ur5_with_scoop.usd` | Robot + scoop asset loaded by IsaacLab |

### Observation space (33 dims)

| Slice | Content |
|---|---|
| `[0:6]` | Joint positions scaled to `[-1, 1]` |
| `[6:12]` | Joint velocities × `dof_vel_scale=0.1` |
| `[12:18]` | Previous smoothed actions |
| `[18:21]` | End-effector position (env-local) |
| `[21:25]` | Scoop quaternion (4 components) |
| `[25:28]` | Sand centroid (env-local) |
| `[28:31]` | Scoop → centroid vector |
| `[31]` | Displaced fraction of particles |
| `[32]` | Elevated fraction of particles |

### Action space (6 dims)

Normalized joint position targets in `[-1, 1]`, mapped to each joint's `[lower, upper]` range. An exponential moving average filter (`alpha=0.2`) is applied each step to limit scoop velocity and prevent MPM particle blow-up from CFL violations.

### Reward function

| Term | Scale | Description |
|---|---|---|
| Distance | `-0.5` | Euclidean distance from scoop EE to sand centroid |
| Elevation | `+4.0` | Fraction of particles lifted > 3 cm above settled z |
| Captured elevated | `+3.0` | Elevated particles within 8 cm (horizontal) of scoop EE |
| Action penalty | `-0.001` | L2 norm of actions — discourages jerky motion |
| Alive | `+0.05` | Per-step constant bonus |

### Physics configuration

| Parameter | Value |
|---|---|
| Control frequency | 60 Hz (sim runs at 120 Hz, decimation = 2) |
| Episode length | 10 s (600 control steps) |
| Num envs (PPO) | 16 (MPM is expensive) |
| Voxel size | 0.01 m |
| Sand density | 1400 kg/m³ |
| Young's modulus | 8000 Pa |
| Poisson ratio | 0.4 |
| Friction | 1.4 |
| Damping | 400 |
| Yield stress | 100 Pa |
| Hardening | 1.0 |
| Air drag | 6.0 |

**Proxy bodies:** The sand solver uses simplified collision shapes for the robot (spheres and capsules per link, plus a box for the scoop). Poses are copied from the IsaacLab Articulation each control step. Forces from sand are not fed back to the robot (one-way coupling).

---

## Training Pipeline

```
real_demos/*.npz          (UR5 recordings: joint angles)
        ↓
collect_demo_data.py  →  replay through sim  →  demo_data.npz (obs, actions)
        ↓
bc_pretrain_scoop.py  →  BC pre-training     →  scoop_bc.pth
        ↓
rl_games/train.py     →  PPO fine-tuning     →  logs/rl_games/scoop_direct_warp/
```

### Step 0 — Real robot demo collection

Demos are pre-recorded UR5 trajectories saved as `.npz` files, each containing:
- `states` — `(T, 6)` array of absolute joint angles in radians, one row per timestep

The demo directory used during development is `/home/gmr/Downloads/ur_ws/dataset`. Some raw recordings in that directory are duplicated — use the glob `demo_2026051[58]*.npz` to select only the deduplicated files (from May 15 and May 18).

### Step 1 — Collect BC dataset (~10–20 min, requires Isaac Sim)

[`scripts/environments/collect_demo_data.py`](scripts/environments/collect_demo_data.py) replays the joint-angle recordings through live simulation to record the full 33-dim observation at every timestep. The sand state (centroid, elevated/displaced fractions) only exists inside the sim, so pure offline processing is not possible.

```bash
./isaaclab.sh -p scripts/environments/collect_demo_data.py \
    --task Isaac-Scoop-Direct-Warp-v3 \
    --headless \
    --demo_dir /home/gmr/Downloads/ur_ws/dataset \
    --output logs/bc_dataset/demo_data.npz
```

**Output:** `demo_data.npz` — keys `observations (N, 33)` and `actions (N, 6)` in `float32`, actions normalized to `[-1, 1]`.

#### Fallback: synthetic demos (no real robot needed)

[`scripts/environments/generate_scoop_demos.py`](scripts/environments/generate_scoop_demos.py) synthesizes demos using a scripted 5-phase motion primitive:

1. **Tilt** — rotate `wrist_1` so the scoop faces down into the sand
2. **Descend** — lower `shoulder_lift` + `elbow` toward the sand surface
3. **Sweep** — pan `shoulder_pan` horizontally through the sand
4. **Level** — partially untilt `wrist_1` to trap sand inside the scoop
5. **Lift** — return `shoulder_lift` + `elbow` to hover height

A trajectory is accepted only if ≥ 30 particles are captured. The output format is identical to real-robot recordings.

```bash
./isaaclab.sh -p scripts/environments/generate_scoop_demos.py \
    --task Isaac-Scoop-Direct-Warp-v3 \
    --num_demos 50 \
    --output_dir logs/scoop_demos \
    --headless
```

Quality is lower than real-robot demos — prefer real data when available.

### Step 2 — BC pre-training (~2–5 min, no Isaac Sim needed)

[`scripts/reinforcement_learning/bc_pretrain_scoop.py`](scripts/reinforcement_learning/bc_pretrain_scoop.py) trains an MLP actor via supervised MSE loss on `(obs → action)` pairs. The architecture matches the rl_games PPO actor exactly so the checkpoint loads directly into PPO without modification.

**Network (`ScoopBCNet`):**

```
obs (33) → RunningMeanStd normalizer
         → Linear(33→256) → ELU
         → Linear(256→256) → ELU
         → Linear(256→128) → ELU
         → Linear(128→6)   → tanh → action (6)
```

The normalizer is fitted from the full dataset offline and frozen during BC training. PPO continues updating it after loading. The value head and log-std are randomly initialized by BC and trained from scratch by PPO.

```bash
python3 scripts/reinforcement_learning/bc_pretrain_scoop.py \
    --dataset logs/bc_dataset/demo_data.npz \
    --output  logs/bc_pretrain/scoop_bc.pth
```

Loss should drop from ~0.3 to below 0.05. The script verifies all checkpoint keys match the expected rl_games format before exiting.

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `logs/bc_dataset/demo_data.npz` | Input dataset |
| `--output` | `logs/bc_pretrain/scoop_bc.pth` | Output checkpoint |
| `--epochs` | `100` | Training epochs |
| `--batch_size` | `256` | Minibatch size |
| `--lr` | `3e-4` | Initial LR (cosine decay to `1e-5`) |

### Step 3 — PPO fine-tuning (from BC init)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Scoop-Direct-Warp-v3 \
    --num_envs 16 \
    --checkpoint logs/bc_pretrain/scoop_bc.pth
```

The learning rate in `rl_games_ppo_cfg.yaml` is `1e-4` (reduced from a typical `3e-4`) to avoid gradient steps large enough to destroy the BC initialization.

Checkpoints are saved every 50 epochs in `logs/rl_games/scoop_direct_warp/<timestamp>/nn/`.

**Key PPO hyperparameters** (`agents/rl_games_ppo_cfg.yaml`):

| Parameter | Value |
|---|---|
| Network | `[256, 256, 128]` ELU (matches BC) |
| `horizon_length` | 32 |
| `minibatch_size` | 512 |
| `mini_epochs` | 4 |
| `learning_rate` | `1e-4` (adaptive KL, threshold `0.008`) |
| `gamma` | 0.99 |
| `tau` (GAE λ) | 0.95 |
| `e_clip` | 0.2 |
| `max_epochs` | 3000 |

To train from scratch (without BC):

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Scoop-Direct-Warp-v3 \
    --num_envs 16
```

---

## Known Issues & Future Work

### PPO reward plateau

Training rewards plateau around -120 to -150 after 3000 epochs. Directions worth trying:
- Curriculum: start with more forgiving success criteria and tighten over training
- Domain randomization on sand material parameters to improve sim-to-real transfer
- Reward shaping: bonus for sustained elevation across multiple consecutive steps

---

## Development Tips

### Quick sanity checks

```bash
# Zero-action baseline — particles should settle; scoop should not move
./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Scoop-Direct-Warp-v3

# Random-action baseline — scoop should thrash around; MPM should not blow up
./isaaclab.sh -p scripts/environments/random_agent.py --task Isaac-Scoop-Direct-Warp-v3

# Scripted waypoint agent — should achieve modest particle elevation
./isaaclab.sh -p scripts/environments/waypoint_agent_v2.py --task Isaac-Scoop-Direct-Warp-v3
```