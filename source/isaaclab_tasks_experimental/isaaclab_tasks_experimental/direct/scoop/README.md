# Isaac-Scoop-Direct-Warp-v3: RL Pipeline

UR5 arm scooping sand (MPM particles) into a box. Training uses behavior cloning from real-robot demonstrations followed by PPO fine-tuning.

## Environment

**Registered ID:** `Isaac-Scoop-Direct-Warp-v3`

**Entry point:** [`scoop_env_warp_v3.py`](scoop_env_warp_v3.py) — `ScoopWarpEnvV3`

**Physics:** Newton MuJoCo-Warp solver with MPM sand simulation (`voxel_size=0.01 m`, kinetic-sand material parameters). `_project_outside` is called before every MPM grid solve to repair particles that tunnel through the scoop.

**Control:** 60 Hz (120 Hz sim, `decimation=2`). Episodes are 10 seconds (600 steps).

### Observation Space (33 dims)

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

### Action Space (6 dims)

Normalized joint position targets in `[-1, 1]`, mapped to each joint's `[lower, upper]` range. An exponential moving average filter (`alpha=0.2`) is applied every step to limit scoop velocity and prevent MPM particle blow-up.

### Reward

| Term | Scale | Description |
|---|---|---|
| Distance | `-0.5` | Distance from scoop EE to sand centroid |
| Elevation | `+4.0` | Fraction of particles lifted > 3 cm above settled z |
| Captured elevated | `+3.0` | Elevated particles within 8 cm (horizontal) of scoop EE |
| Action penalty | `-0.001` | L2 norm of actions |
| Alive | `+0.05` | Per-step bonus |

---

## Training Pipeline

### Overview

```
/path/to/real_demos/*.npz          (real robot recordings: joint angles)
            ↓
collect_demo_data.py     →  replay through sim  →  demo_data.npz (obs, actions)
            ↓
bc_pretrain_scoop.py     →  BC pre-training     →  scoop_bc.pth
            ↓
rl_games/train.py        →  PPO fine-tuning
```

---

### Step 1: Collect BC Dataset

[`scripts/environments/collect_demo_data.py`](../../../../../scripts/environments/collect_demo_data.py)

Takes pre-existing real-robot demo files and replays them through the live simulation to record the full 33-dim observation at every step. The MPM sand state (centroid, elevated/displaced fractions) only exists inside the sim, so replay is required to get complete observations.

**Input format:** `*.npz` files each containing:
- `states` — `(T, 6)` absolute joint angles in radians

**Default demo directory:** `/home/gmr/Downloads/ur_ws/dataset`

```bash
./isaaclab.sh -p scripts/environments/collect_demo_data.py \
    --task Isaac-Scoop-Direct-Warp-v3 \
    --headless \
    --demo_dir /home/gmr/Downloads/ur_ws/dataset \
    --output logs/bc_dataset/demo_data.npz
```

**Output:** `demo_data.npz` with keys:
- `observations` — `(N, 33)` float32
- `actions` — `(N, 6)` float32, normalized to `[-1, 1]`

#### Generating Synthetic Demos (optional fallback)

If real-robot data is unavailable, [`scripts/environments/generate_scoop_demos.py`](../../../../../scripts/environments/generate_scoop_demos.py) synthesizes demos using a scripted 5-phase motion primitive:

1. **Tilt** — rotate `wrist_1` so scoop faces down into sand
2. **Descend** — lower `shoulder_lift` + `elbow` toward sand surface
3. **Sweep** — pan `shoulder_pan` horizontally through sand
4. **Level** — partially untilt `wrist_1` to trap sand inside scoop
5. **Lift** — return `shoulder_lift` + `elbow` to hover height

A trajectory is accepted only if ≥ 30 particles are captured (lifted > 20 cm AND within 10 cm of EE). The output `.npz` format is identical to real-robot recordings.

```bash
./isaaclab.sh -p scripts/environments/generate_scoop_demos.py \
    --task Isaac-Scoop-Direct-Warp-v3 \
    --num_demos 50 \
    --output_dir logs/scoop_demos \
    --headless
```

---

### Step 2: Behavior Cloning Pre-training

[`scripts/reinforcement_learning/bc_pretrain_scoop.py`](../../../../../scripts/reinforcement_learning/bc_pretrain_scoop.py)

Trains an MLP actor via supervised MSE loss on `(obs → action)` pairs. The network architecture is identical to the rl_games PPO actor so the checkpoint loads directly into PPO without modification.

#### Network Architecture (`ScoopBCNet`)

```
obs (33) → RunningMeanStd → Linear(33→256) → ELU
                           → Linear(256→256) → ELU
                           → Linear(256→128) → ELU
                           → Linear(128→6)   → tanh → action (6)
```

- **Obs normalizer** (`RunningMeanStd`): fitted from the full dataset, frozen during BC training. PPO continues updating it after loading.
- **Value head** (`Linear(128→1)`): randomly initialized. Not trained by BC — PPO trains it.
- **Log-std** (`sigma`): initialized to zero. Used by PPO's stochastic policy.

#### Training

```bash
python3 scripts/reinforcement_learning/bc_pretrain_scoop.py \
    --dataset logs/bc_dataset/demo_data.npz \
    --output  logs/bc_pretrain/scoop_bc.pth \
    --epochs  100
```

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `logs/bc_dataset/demo_data.npz` | Input dataset |
| `--output` | `logs/bc_pretrain/scoop_bc.pth` | Output checkpoint |
| `--epochs` | `100` | Training epochs |
| `--batch_size` | `256` | Minibatch size |
| `--lr` | `3e-4` | Initial learning rate (cosine decay to `1e-5`) |

The checkpoint is saved in rl_games key format (`a2c_network.actor_mlp.0.weight`, etc.) and verified for key completeness before exit.

---

### Step 3: PPO Fine-tuning

Uses [rl_games](https://github.com/Denys88/rl_games) PPO with the config at [`agents/rl_games_ppo_cfg.yaml`](agents/rl_games_ppo_cfg.yaml).

#### Key Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | `a2c_continuous` (PPO) |
| Network | `[256, 256, 128]` + ELU (matches BC) |
| `horizon_length` | 32 |
| `minibatch_size` | 512 |
| `mini_epochs` | 4 |
| `learning_rate` | `1e-4` (adaptive KL, threshold `0.008`) |
| `gamma` | 0.99 |
| `tau` (GAE λ) | 0.95 |
| `e_clip` | 0.2 |
| `max_epochs` | 3000 |
| `num_envs` | 16 (MPM is expensive) |

#### Fine-tune from BC checkpoint

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Scoop-Direct-Warp-v3 \
    --num_envs 16 \
    --checkpoint logs/bc_pretrain/scoop_bc.pth
```

#### Train from scratch (no BC)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Scoop-Direct-Warp-v3 \
    --num_envs 16
```
