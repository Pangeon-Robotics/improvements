# Repository Structure Proposal

**Date**: 2026-02-10
**Purpose**: Define where HNN/RL training code and models live

---

## Problem

Current plan has training code in local `/Users/graham/code/robotics/training/` which is not in any GitHub repo. Everything must be in the Pangeon-Robotics GitHub organization.

---

## Proposed Structure

### New Repository: `training`

**Purpose**: Offline training for HNN dynamics models and RL locomotion policies

**URL**: `https://github.com/Pangeon-Robotics/training`

**Contains**:
```
training/
├── CLAUDE.md                    # Agent identity and workflows
├── README.md                    # Overview and quick start
├── requirements.txt             # jax, flax, torch, sb3, mujoco
├── pyproject.toml               # Package metadata
│
├── hnn/                         # Hamiltonian Neural Network training
│   ├── collect_trajectories.py # Data collection from layers_1_2
│   ├── train.py                 # Train HNN (JAX/Flax)
│   ├── validate.py              # Test HNN predictions
│   └── model.py                 # HNN architecture definition
│
├── rl/                          # Reinforcement learning training
│   ├── envs/
│   │   └── quadruped_env.py     # Gym environment wrapping layers
│   ├── train_ppo.py             # PPO training (stable-baselines3)
│   ├── validate_ppo.py          # Policy evaluation
│   └── callbacks.py             # Training callbacks, logging
│
├── data/                        # Training data (gitignored, use DVC or releases)
│   ├── .gitignore               # Ignore large files
│   └── README.md                # How to download datasets
│
├── configs/                     # Training configurations
│   ├── hnn_b2.yaml              # HNN config for B2 robot
│   └── ppo_locomotion.yaml      # PPO config
│
└── tests/
    ├── test_hnn.py
    └── test_rl_env.py
```

**Key Points**:
- Separate repo because training is NOT owned by any single layer
- Uses layers_1_2 (simulation), but doesn't belong in it
- Produces models for layers 4-5 to consume
- Has its own team (ML/training team)

**Layers in repos.json**: `[]` (uses layers, but doesn't implement any)

---

### Model Distribution: GitHub Releases

**Storage**: Trained models as release artifacts in `training` repo

**Release naming**: `v{model-version}-{robot}-{type}`
- Example: `v1.0.0-b2-hnn` → `hnn_b2_v1.0.0.pkl`
- Example: `v1.0.0-b2-ppo-gait` → `ppo_gait_v1.0.0.zip`

**Access pattern**:
```bash
# Layer 4 downloads HNN model
cd layer_4
curl -L https://github.com/Pangeon-Robotics/training/releases/download/v1.0.0-b2-hnn/hnn_b2_v1.0.0.pkl \
  -o models/hnn_b2_v1.0.0.pkl

# Layer 5 downloads RL policy
cd layer_5
curl -L https://github.com/Pangeon-Robotics/training/releases/download/v1.0.0-b2-ppo-gait/ppo_gait_v1.0.0.zip \
  -o models/ppo_gait_v1.0.0.zip
```

**Why GitHub Releases:**
- Versioned (semantic versioning)
- Immutable (can't accidentally overwrite)
- CDN-backed (fast downloads)
- No Git LFS needed (cleaner)
- Easy CI/CD integration

**Alternative: Separate `models` repo with Git LFS**
- More complex
- Only needed if models change frequently
- Releases are simpler for frozen models

---

### Alternative: DVC (Data Version Control)

If training data and models become large (>1 GB per file):

```bash
# In training repo
dvc remote add -d storage s3://pangeon-training-artifacts
dvc add data/trajectories.npz
dvc add models/hnn_b2_v1.pkl
dvc push
```

**Pros**: Handles large files well, tracks data lineage
**Cons**: Extra tool, S3 costs, more complexity

**Recommendation**: Start with GitHub Releases, move to DVC if needed

---

## Updated repos.json

```json
{
  "repos": [
    {
      "repo": "training",
      "url": "https://github.com/Pangeon-Robotics/training",
      "layers": [],
      "status": "active",
      "note": "Offline ML training (HNN, RL) - uses layers_1_2 sim, produces models for layers 4-5"
    }
  ]
}
```

---

## Dependency Flow

```
┌─────────────┐
│ training/   │  ← Offline training scripts
│ (new repo)  │  ← Uses layers_1_2 for simulation
└──────┬──────┘  ← Produces frozen models
       │
       │ (GitHub Releases)
       │
       ├──────────────┬─────────────┐
       ↓              ↓             ↓
  ┌─────────┐   ┌─────────┐   ┌─────────┐
  │ layer_4 │   │ layer_5 │   │ layer_6 │
  │ (HNN)   │   │ (RL)    │   │ (future)│
  └─────────┘   └─────────┘   └─────────┘
       ↓              ↓
   Deployment   Deployment
```

**Key property**: One-way dependency (training → layers, never layers → training)

---

## Where Things Live

| Component | Repo | Reason |
|-----------|------|--------|
| **HNN training code** | `training/hnn/` | Not layer-specific, uses multiple layers |
| **RL training code** | `training/rl/` | Not layer-specific, uses multiple layers |
| **Trained models** | `training` releases | Versioned artifacts |
| **Training data** | `training/data/` (ignored) or DVC | Too large for git, downloaded separately |
| **HNN inference class** | `layer_4/predictor.py` | Deployment code lives in layer |
| **RL policy loader** | `layer_5/policy.py` | Deployment code lives in layer |
| **Implementation plan** | `improvements/` | Architecture/planning docs |
| **Training configs** | `training/configs/` | Hyperparameters, reward weights |

---

## Repository Boundaries

### `training` repo:
**Does:**
- Collect simulation data from layers_1_2
- Train HNN models (JAX/Flax)
- Train RL policies (PyTorch/SB3)
- Validate models
- Publish models as releases

**Does NOT:**
- Deploy models to layers (layers pull from releases)
- Implement layer interfaces (just produces artifacts)
- Modify layer code (files issues via fix-request workflow)

### `layer_4` repo:
**Does:**
- Provide HNN inference class (`HNNPredictor`)
- Download HNN model from `training` releases
- Use frozen model for terrain estimation

**Does NOT:**
- Train models (that's `training` repo's job)
- Generate training data (uses existing MuJoCo sim)

### `layer_5` repo:
**Does:**
- Provide RL policy loader (`PolicyLoader`)
- Download policy from `training` releases
- Use frozen policy for gait selection

**Does NOT:**
- Train policies (that's `training` repo's job)

---

## Implementation Steps

### Step 1: Create `training` repo
```bash
# On GitHub: Create Pangeon-Robotics/training repo

# Locally
cd /Users/graham/code/robotics/
git clone https://github.com/Pangeon-Robotics/training.git
cd training

# Initialize structure
mkdir -p hnn rl/envs configs data tests
touch README.md CLAUDE.md requirements.txt
git add .
git commit -m "Initial training repo structure"
git push origin main
```

### Step 2: Update `philosophy/workflows/repos.json`
Add training repo entry (see above)

### Step 3: Move training code from IMPLEMENTATION_PLAN.md
- Create `hnn/train.py` based on LEARNING_ROADMAP.md code
- Create `rl/train_ppo.py` based on LEARNING_ROADMAP.md code
- Create configs

### Step 4: File enhancement issues to layers 4-5
Use prepared content from `/tmp/layer4_hnn_enhancement.md` and `/tmp/layer5_rl_enhancement.md`

---

## FAQ

**Q: Why not put HNN training in layer_4 repo?**
A: Training USES layers_1_2 simulation. It's not owned by layer 4. Layer 4 only does deployment (inference).

**Q: Why not put all training in layers_1_2 repo?**
A: layers_1_2 is firmware + physics simulation. Training is a separate concern that uses the simulation.

**Q: Should training data be in git?**
A: No. Use `.gitignore` and provide download instructions. GitHub Releases or DVC for large files.

**Q: How do layers know which model version to use?**
A: Layers document required model version in their README. Example: "Requires HNN v1.0.0+ from training repo releases"

**Q: Can we train online in deployment?**
A: Future work. Current plan is offline training only (frozen models). If online learning needed, Layer 5+ would own it.

---

## Summary

**New repo**: `training` (HNN + RL training code)
**Model storage**: GitHub Releases (versioned artifacts)
**Data storage**: Local with download instructions (or DVC if large)
**Layer repos**: Only contain deployment/inference code, not training

This structure:
- ✅ Everything in GitHub repos
- ✅ Clear separation of concerns
- ✅ One-way dependencies (training → layers)
- ✅ Versioned model distribution
- ✅ No Git LFS needed initially

---

**Next step**: Create `training` repo and initialize structure
