# Implementation Plan: HNN and RL Training

**Date**: 2026-02-10
**Status**: Architecture validated, ready for implementation
**Purpose**: Step-by-step plan with fix-requests for HNN dynamics and RL locomotion

---

## Architecture Review

### Current 8-Layer Stack

| Layer | Title | Nature | Current Status |
|-------|-------|--------|----------------|
| **1-2** | Firmware + Physics | Instant | ✅ Active |
| **3** | Inverse Kinematics | Instant | ✅ Active |
| **4** | Cartesian Positions | Instant | ✅ Active (rule-based terrain) |
| **5** | Locomotion | **Sequence** | ✅ Active (rule-based gaits) |
| **6** | Waypoints & Tasks | Sequence | ❌ Not needed yet |
| **7** | Mission Planning | Sequence | ❌ Not needed yet |
| **8** | Application UI | Event-driven | ❌ Not needed yet |

### Key Architectural Decisions

**1. Layers 1-4 are INSTANT (stateless)**
- No temporal state (`self.phase`, `self.time`)
- Pure translation, per-timestep
- Learning violates this boundary

**2. Layer 5+ are SEQUENCE (stateful)**
- Can maintain state
- Plan over time
- First place learning can live in deployment

**3. Training happens OFFLINE**
- Separate `training/` directory
- Not owned by any layer
- Produces frozen models for deployment

**4. Layer 5 is the right place for RL policy**
- Already stateful (sequence layer)
- Already does gait selection
- RL enhances this, doesn't bypass it
- Vocabulary: MotionCommand → GaitParams

**5. Layer 6 is NOT needed yet**
- Layer 6 = waypoint navigation, task planning
- RL policy is for gait selection (Layer 5's job)
- Can add Layer 6 later for path planning

---

## Phase 1: Training Infrastructure (Week 1-2)

**Owner**: ML team (independent, not a layer)
**Location**: `../training/`

### Tasks

Create training directory structure:
```bash
mkdir -p training training_data models

training/
├── collect_trajectories.py   # Collect MuJoCo data for HNN
├── train_hnn.py              # Train Hamiltonian NN (JAX/Flax)
├── train_ppo.py              # Train RL policy (PyTorch/SB3)
├── validate_hnn.py           # Test HNN predictions
├── validate_ppo.py           # Test RL policy
├── envs/
│   └── quadruped_env.py      # Gym environment wrapping MuJoCo
└── requirements.txt          # jax, flax, torch, sb3, mujoco
```

### Dependencies

```
jax[cuda]>=0.4.20
flax>=0.7.0
optax>=0.1.7
torch>=2.1.0
stable-baselines3>=2.1.0
gymnasium>=0.29.0
```

### Deliverables

- [ ] Training directory created
- [ ] `collect_trajectories.py` implemented (uses layers_1_2 MuJoCo)
- [ ] `train_hnn.py` stub (loads data, trains HNN)
- [ ] `train_ppo.py` stub (defines env, trains PPO)
- [ ] Requirements file

**No fix-requests needed** - training infrastructure is independent

---

## Phase 2: HNN Training and Layer 4 Deployment (Week 3-5)

### 2A: Collect Training Data (Week 3)

**Owner**: ML team
**Depends on**: layers_1_2 (MuJoCo simulation)

```bash
cd training
python collect_trajectories.py \
  --robot b2 \
  --episodes 10000 \
  --gaits walk,trot,bound \
  --terrain flat,rough,stairs \
  --output ../training_data/trajectories.npz
```

**Output**: `training_data/trajectories.npz` (~5 GB)

### 2B: Train HNN Model (Week 3-4)

**Owner**: ML team

```bash
python train_hnn.py \
  --data ../training_data/trajectories.npz \
  --epochs 1000 \
  --output ../models/hnn_b2_v1.pkl
```

**Output**: `models/hnn_b2_v1.pkl` (frozen model)

### 2C: Deploy HNN to Layer 4 (Week 4-5)

**Owner**: Layer 4 team
**Mechanism**: Fix-request from improvements to Layer 4

#### Fix-Request: Layer 4 HNN-Based Terrain Estimation

**Issue Title**: "Add HNN-based terrain estimation via frozen model inference"

**Summary**:
Layer 4 should load a frozen Hamiltonian Neural Network to predict robot dynamics and compute prediction errors for improved terrain estimation.

**Current State**:
Layer 4 uses simple sensor heuristics (issue #19):
- Force variance → roughness
- IMU vibration → stability

**Proposed Enhancement**:
Add HNN predictor class that:
1. Loads frozen model from `models/hnn_b2_v1.pkl`
2. Predicts next state given current (q, p)
3. Compares prediction to actual observation
4. Computes terrain properties from prediction errors

**Interface Addition** (public API enrichment):
```python
# layer_4/terrain.py

class HNNTerrainEstimator:
    """Enhanced terrain estimation using HNN prediction errors."""

    def __init__(self, model_path: str = None):
        if model_path and os.path.exists(model_path):
            self.predictor = HNNPredictor(model_path)
            self.use_hnn = True
        else:
            self.use_hnn = False  # Fall back to sensor heuristics

    def estimate(self, kinematic_state: KinematicState) -> TerrainEstimate:
        """
        Compute terrain properties from observations.

        Returns:
            TerrainEstimate with roughness, compliance, stability scalars
        """
        if self.use_hnn:
            return self._estimate_with_hnn(kinematic_state)
        else:
            return self._estimate_with_heuristics(kinematic_state)
```

**Implementation Notes**:
- HNN prediction is **stateless inference** (no training, no parameter updates)
- Load params in `__init__`, do forward pass in `estimate()`
- Graceful fallback if model file missing
- No JAX/Flax dependency in Layer 4's core - predictor is isolated module

**Vocabulary Check**: ✅ All Layer 4 vocabulary (positions, forces, terrain estimates)

**Temporal Check**: ✅ Instant layer - no `self.time`, just stateless prediction

**Success Criteria**:
- [ ] HNNPredictor class loads frozen pickle file
- [ ] `estimate()` runs at 100 Hz without blocking
- [ ] Prediction errors abstracted to scalars before exposing to Layer 5
- [ ] All tests pass with and without HNN model present
- [ ] Version bump (MINOR - new optional feature)

---

## Phase 3: RL Training (Week 6-9)

### 3A: Define Environment (Week 6)

**Owner**: ML team

Create `training/envs/quadruped_env.py`:

```python
import gymnasium as gym
import numpy as np

class QuadrupedLocomotionEnv(gym.Env):
    """RL environment for learning gait selection.

    Observation: Joint state, IMU, terrain estimate, velocity command
    Action: GaitParams (gait_type, step_height, freq, duty_cycle)
    Reward: Track velocity, maintain stability, minimize energy
    """

    def __init__(self):
        # Observation space (48D)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(48,),  # 12 joints + 4 quat + 3 vel + 3 terrain + 3 cmd
            dtype=np.float32
        )

        # Action space (GaitParams)
        # Discrete: gait_type (0=walk, 1=trot, 2=bound)
        # Continuous: step_height, freq, duty_cycle
        self.action_space = gym.spaces.Dict({
            "gait_type": gym.spaces.Discrete(3),
            "step_height": gym.spaces.Box(0.02, 0.12, (1,)),
            "freq": gym.spaces.Box(0.5, 3.0, (1,)),
            "duty_cycle": gym.spaces.Box(0.5, 0.8, (1,)),
        })

    def step(self, action):
        # Convert action to GaitParams
        gait_params = self._action_to_gait_params(action)

        # Layer 4: GaitParams → foot positions
        positions = self.layer4.compute(gait_params, self.t)

        # Layer 3: Positions → joint commands
        joint_cmd = self.layer3.ik(positions)

        # Step MuJoCo simulation
        obs = self.sim.step(joint_cmd)

        # Compute reward
        reward = self._compute_reward(obs)

        return obs, reward, done, truncated, {}
```

### 3B: Train PPO Policy (Week 7-9)

**Owner**: ML team

```bash
python train_ppo.py \
  --env quadruped_locomotion \
  --timesteps 10000000 \
  --parallel-envs 8 \
  --output ../models/ppo_gait_v1.zip
```

**Compute**: 8× parallel environments, RTX 4090 or better, 12-24 hours

**Output**: `models/ppo_gait_v1.zip` (frozen policy)

### 3C: Validate Policy (Week 9)

```bash
python validate_ppo.py \
  --model ../models/ppo_gait_v1.zip \
  --episodes 100 \
  --terrain varied
```

**Success Metrics**:
- Velocity tracking error < 0.1 m/s
- No falls in 100 episodes
- Generalizes to unseen terrain

---

## Phase 4: RL Deployment to Layer 5 (Week 10-11)

**Owner**: Layer 5 team
**Mechanism**: Fix-request from improvements to Layer 5

#### Fix-Request: Layer 5 RL-Enhanced Gait Selection

**Issue Title**: "Add optional RL policy for learned gait selection"

**Summary**:
Layer 5 should optionally load a frozen RL policy to select gait parameters instead of rule-based heuristics.

**Current State**:
Layer 5 uses if-then rules (issue #2):
- Rough terrain → walk gait
- Soft terrain → high duty cycle
- Fast command → trot gait

**Proposed Enhancement**:
Add RL policy loader that:
1. Loads frozen policy from `models/ppo_gait_v1.zip`
2. Maps (observations + terrain + velocity command) → GaitParams
3. Falls back to rule-based if policy unavailable
4. Maintains Layer 5's existing interface

**Interface Enhancement** (no breaking changes):
```python
# layer_5/locomotion.py

class GaitSelector:
    """Select gait parameters for locomotion."""

    def __init__(self, policy_path: str = None):
        self.use_learned = False

        if policy_path and os.path.exists(policy_path):
            import torch
            self.policy = torch.load(policy_path)
            self.use_learned = True

    def update(self,
               motion_cmd: MotionCommand,
               terrain: TerrainEstimate,
               state: RobotState,
               dt: float) -> GaitParams:
        """
        Convert motion command to gait parameters.

        Args:
            motion_cmd: Desired velocity (vx, vy, wz)
            terrain: From Layer 4 (roughness, compliance, stability)
            state: Current robot state (joints, IMU)
            dt: Timestep

        Returns:
            GaitParams for Layer 4
        """
        if self.use_learned:
            return self._policy_based_selection(motion_cmd, terrain, state)
        else:
            return self._rule_based_selection(motion_cmd, terrain)
```

**Vocabulary Check**: ✅ All Layer 5 vocabulary (motion commands, gait params)

**Temporal Check**: ✅ Sequence layer - CAN maintain state (policy has LSTM/history)

**Dependency Note**: PyTorch only imported if policy present (optional dependency)

**Success Criteria**:
- [ ] Policy loader isolates torch dependency
- [ ] Graceful fallback to rules if model missing
- [ ] Inference runs at 100 Hz
- [ ] All existing tests pass with policy=None
- [ ] New tests validate policy integration
- [ ] Version bump (MINOR - new optional feature)

---

## Phase 5: Evaluation and Iteration (Week 12+)

### Comparative Testing

Run side-by-side tests:

| Scenario | Rule-Based (Current) | HNN + RL (New) | Metrics |
|----------|---------------------|----------------|---------|
| Flat terrain walk | Baseline | Compare | Velocity error, energy |
| Rough terrain | Baseline | Compare | Stability, falls |
| Soft terrain | Baseline | Compare | Sinkage, slip |
| Terrain transitions | Baseline | Compare | Smoothness |

### Metrics to Track

- **Velocity tracking**: RMS error vs commanded velocity
- **Stability**: Base orientation deviation, falls per km
- **Energy**: Average torque magnitude
- **Adaptation**: Performance on unseen terrain
- **Latency**: Control loop timing (must stay < 10ms)

### Iteration

- Retrain HNN if prediction errors high
- Retrain RL if velocity tracking poor
- Tune reward weights
- Expand terrain variety in training data

---

## Fix-Request Summary

### Fix-Request #1: Layer 4 (HNN-Based Terrain Estimation)

**From**: improvements repo
**To**: layer_4 repo
**Type**: Enhancement (optional feature)
**Version Bump**: MINOR
**Dependencies**: Trained HNN model (`models/hnn_b2_v1.pkl`)

**Key Points**:
- Adds `HNNTerrainEstimator` class
- Loads frozen HNN for stateless prediction
- Graceful fallback to sensor heuristics
- No breaking changes to Layer 4 API

### Fix-Request #2: Layer 5 (RL-Enhanced Gait Selection)

**From**: improvements repo
**To**: layer_5 repo
**Type**: Enhancement (optional feature)
**Version Bump**: MINOR
**Dependencies**: Trained RL policy (`models/ppo_gait_v1.zip`)

**Key Points**:
- Adds policy loader to `GaitSelector`
- Falls back to rule-based if no policy
- Maintains existing Layer 5 interface
- Optional PyTorch dependency

---

## Directory Structure After Implementation

```
workspace root (../):
├── layers_1_2/          # Firmware + MuJoCo (unchanged)
├── layer_3/             # IK (unchanged)
├── layer_4/             # Cartesian + HNN terrain (enhanced)
│   ├── terrain.py       # HNNTerrainEstimator added
│   └── predictor.py     # HNNPredictor (new file)
├── layer_5/             # Locomotion + RL gait (enhanced)
│   └── locomotion.py    # GaitSelector with policy option
├── training/            # NEW - Training infrastructure
│   ├── collect_trajectories.py
│   ├── train_hnn.py
│   ├── train_ppo.py
│   ├── validate_hnn.py
│   ├── validate_ppo.py
│   └── envs/
│       └── quadruped_env.py
├── training_data/       # NEW - Training datasets
│   └── trajectories.npz
├── models/              # NEW - Trained models
│   ├── hnn_b2_v1.pkl
│   └── ppo_gait_v1.zip
├── improvements/        # Documentation (this plan)
└── philosophy/          # Architecture reference
```

---

## Key Architectural Wins

✅ **No Layer 6 needed**: RL policy enhances Layer 5, doesn't bypass it
✅ **Learning outside layers**: Training code independent of deployment
✅ **Boundary respect**: Layers 1-4 instant, Layer 5+ sequence preserved
✅ **Graceful degradation**: Rule-based fallback if models missing
✅ **Interface stability**: No breaking changes to existing APIs
✅ **Vocabulary consistency**: Each layer uses only its own terms

---

## Timeline Summary

| Week | Phase | Owner | Deliverable |
|------|-------|-------|-------------|
| 1-2 | Training infra | ML team | `training/` directory, scripts |
| 3 | Data collection | ML team | 10k episodes, 5 GB data |
| 3-4 | HNN training | ML team | `hnn_b2_v1.pkl` |
| 4-5 | HNN deployment | Layer 4 | Fix-request resolved, tested |
| 6 | RL environment | ML team | Gym environment definition |
| 7-9 | RL training | ML team | `ppo_gait_v1.zip` |
| 10-11 | RL deployment | Layer 5 | Fix-request resolved, tested |
| 12+ | Evaluation | All | Comparative metrics, iteration |

---

## Next Immediate Steps

1. **Create `training/` directory** and stub files
2. **File fix-request to Layer 4** for HNN terrain estimation
3. **File fix-request to Layer 5** for RL gait selection
4. **Begin data collection** in parallel

---

**Status**: Plan complete, ready to execute
**Last Updated**: 2026-02-10
**Architecture Validated**: ✅ Respects all boundaries from philosophy/
