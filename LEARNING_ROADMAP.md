# Learning Roadmap: HNN and RL Training

**Date**: 2026-02-10
**Status**: Future work
**Purpose**: Define where and how learning happens in the architecture

---

## Current State (What We Shipped)

✅ **Stateless, rule-based system**:
- Layer 4: Sensor heuristics (force variance → terrain estimate)
- Layer 5: If-then rules (rough terrain → walk, soft → high duty cycle)
- **No learning**: All parameters hand-tuned

**This works** and respects layer boundaries, but doesn't adapt or improve.

---

## Two Types of Learning to Add

### 1. HNN Training (Dynamics Model)
**Goal**: Learn to predict "what will happen" given current state and action

### 2. RL/PPO Training (Policy)
**Goal**: Learn to select actions that achieve objectives

---

## Architecture Constraint: Where Can Learning Live?

**Problem**: We established that:
- Layers 1-4 are **instant (stateless)** - no persistent state
- Learning requires state (model params, optimizer, replay buffer)
- **Therefore: Learning cannot live in Layers 1-4**

**Options**:

### Option A: Offline Training (Recommended)
- Train models **outside the layer stack** (separate training process)
- Deploy **frozen models** to layers
- No online learning during deployment
- **Pro**: Respects all boundaries, simple deployment
- **Con**: No online adaptation to new terrain

### Option B: Learning in Layer 5+
- Layer 5 or higher maintains learning state
- Updates model parameters online
- Provides updated params to Layer 4 (downward)
- **Pro**: Online adaptation
- **Con**: Requires Layer 5 to hold ML state (PyTorch/JAX models)

### Option C: Separate Learning Service
- External process outside the 8-layer stack
- Subscribes to observations via DDS
- Publishes updated models via DDS
- **Pro**: Doesn't violate any layer boundaries
- **Con**: Additional architectural complexity

---

## Recommended Approach: Offline Training (Option A)

Start with **offline training + frozen deployment**. Add online learning later if needed.

---

## HNN Training Pipeline

### Phase 1: Data Collection

**Where**: In simulation (`layers_1_2` MuJoCo)

```bash
# Collect trajectories with varied gaits and terrain
cd layers_1_2
python collect_trajectories.py \
  --robot b2 \
  --episodes 10000 \
  --gaits walk,trot,bound \
  --terrain flat,rough,stairs \
  --output ../training_data/trajectories.npz
```

**Data format**:
```python
trajectories.npz:
  q: (N, 12)       # Joint positions over time
  dq: (N, 12)      # Joint velocities
  tau: (N, 12)     # Joint torques
  contacts: (N, 4) # Foot contact forces
  dt: float        # Timestep (0.002s for MuJoCo)
```

### Phase 2: HNN Training

**Where**: Separate training script (not in any layer)

```python
# File: training/train_hnn.py

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

class HamiltonianNN(nn.Module):
    """Learns H(q, p) dynamics."""
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, q, p):
        qp = jnp.concatenate([q, p], axis=-1)
        x = nn.Dense(self.hidden_dim)(qp)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        H = nn.Dense(1)(x)  # Scalar Hamiltonian
        return H

    def predict_dynamics(self, q, p):
        """Hamilton's equations via autodiff."""
        H = lambda q, p: self(q[None, :], p[None, :])[0, 0]
        q_dot = jax.grad(H, argnums=1)(q, p)   # ∂H/∂p
        p_dot = -jax.grad(H, argnums=0)(q, p)  # -∂H/∂q
        return q_dot, p_dot

# Training loop
def train_hnn(data_file, epochs=1000):
    data = np.load(data_file)
    q, dq = data['q'], data['dq']

    # Convert velocities to momenta (assume unit mass)
    p = dq

    # Initialize model
    hnn = HamiltonianNN()
    params = hnn.init(jax.random.PRNGKey(0),
                     jnp.zeros(12), jnp.zeros(12))

    # Optimizer
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    # Training
    for epoch in range(epochs):
        for i in range(len(q) - 1):
            q_curr, p_curr = q[i], p[i]
            q_next_true, p_next_true = q[i+1], p[i+1]

            # Predict next state
            q_dot, p_dot = hnn.apply(params, q_curr, p_curr,
                                     method='predict_dynamics')
            q_pred = q_curr + 0.002 * q_dot  # dt = 0.002
            p_pred = p_curr + 0.002 * p_dot

            # Loss
            loss = jnp.mean((q_pred - q_next_true)**2 +
                           (p_pred - p_next_true)**2)

            # Update
            grads = jax.grad(lambda p: loss)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    # Save trained model
    with open('hnn_trained.pkl', 'wb') as f:
        pickle.dump(params, f)

    return params
```

**Run training**:
```bash
python training/train_hnn.py \
  --data ../training_data/trajectories.npz \
  --epochs 1000 \
  --output models/hnn_trained.pkl
```

**Validation**:
```bash
python training/validate_hnn.py \
  --model models/hnn_trained.pkl \
  --test-data ../training_data/test_trajectories.npz
```

### Phase 3: Deploy to Layer 4 (Stateless Inference)

**Where**: Layer 4 (prediction only, no training)

```python
# File: layer_4/dynamics.py

class HNNPredictor:
    """Stateless HNN prediction (no learning)."""

    def __init__(self, model_path):
        # Load frozen model
        with open(model_path, 'rb') as f:
            self.params = pickle.load(f)
        self.hnn = HamiltonianNN()

    def predict(self, q, p, dt=0.01):
        """Predict next state (stateless)."""
        q_dot, p_dot = self.hnn.apply(
            self.params, q, p,
            method='predict_dynamics'
        )
        q_next = q + dt * q_dot
        p_next = p + dt * p_dot
        return q_next, p_next
```

**Key**: Layer 4 only does **inference** with frozen params. No training, no state updates.

---

## RL Training Pipeline

### Phase 1: Define Observation and Action Space

**Observation** (what the policy sees):
```python
@dataclass
class Observation:
    # From Layer 3 (via Layer 4)
    joint_positions: np.ndarray  # (12,)
    joint_velocities: np.ndarray # (12,)
    base_orientation: np.ndarray # (4,) quaternion
    base_angular_vel: np.ndarray # (3,)

    # From Layer 4
    terrain_roughness: float     # 0-1
    terrain_compliance: float    # 0-1
    terrain_stability: float     # 0-1

    # Goal
    target_velocity: np.ndarray  # (3,) vx, vy, wz
```

**Action** (what the policy outputs):
```python
@dataclass
class Action:
    gait_type: str               # 'walk', 'trot', 'bound'
    step_height: float           # m
    step_length: float           # m
    gait_freq: float             # Hz
    duty_cycle: float            # 0-1
```

**Or simpler** (let existing Layer 5 handle gait params):
```python
@dataclass
class Action:
    velocity_command: np.ndarray # (3,) vx, vy, wz
    # Layer 5 converts this to GaitParams
```

### Phase 2: RL Training (PPO)

**Where**: Separate training script with MuJoCo simulation

```python
# File: training/train_ppo.py

import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

class QuadrupedEnv(gym.Env):
    """RL environment wrapping MuJoCo simulation."""

    def __init__(self):
        self.sim = MuJoCoSimulator('b2')

        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(28,),  # 12 joints + 4 quat + 3 ang_vel + 3 terrain + 3 target
            dtype=np.float32
        )

        # Action space (velocity commands)
        self.action_space = gym.spaces.Box(
            low=np.array([-2, -1, -2]),  # vx, vy, wz limits
            high=np.array([2, 1, 2]),
            dtype=np.float32
        )

    def step(self, action):
        # Action = velocity command
        vx, vy, wz = action

        # Layer 5 converts to GaitParams
        terrain = self.sim.get_terrain_estimate()
        gait_params = self.layer5.update(
            MotionCommand(vx, vy, wz),
            dt=0.01,
            terrain=terrain
        )

        # Layer 4 generates trajectories
        positions = self.sim.compute(gait_params, t)

        # Layer 3 does IK
        joint_cmd = self.layer3.ik(positions)

        # Step simulation
        obs = self.sim.step(joint_cmd)

        # Reward
        velocity_error = np.linalg.norm(obs.velocity - action)
        stability_reward = obs.base_stable * 1.0
        energy_penalty = -0.01 * np.sum(obs.joint_torques**2)

        reward = -velocity_error + stability_reward + energy_penalty

        done = obs.fallen or obs.time > 10.0

        return self._build_obs(obs), reward, done, {}

# Training
def train_ppo():
    # Create vectorized environments
    envs = SubprocVecEnv([
        lambda: QuadrupedEnv()
        for _ in range(8)
    ])

    # PPO agent
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1
    )

    # Train
    model.learn(total_timesteps=10_000_000)

    # Save
    model.save("models/ppo_locomotion")

    return model
```

**Run training**:
```bash
python training/train_ppo.py \
  --timesteps 10000000 \
  --envs 8 \
  --output models/ppo_locomotion.zip
```

### Phase 3: Deploy RL Policy

**Option 1: Replace Layer 5 with RL policy**
```python
# Layer 5 becomes a neural network inference layer
class RLLocomotion:
    def __init__(self, model_path):
        self.policy = torch.load(model_path)

    def update(self, command, dt, terrain):
        # Build observation
        obs = np.concatenate([
            self.get_joint_state(),
            self.get_base_state(),
            [terrain.roughness, terrain.compliance, terrain.stability],
            [command.vx, command.vy, command.wz]
        ])

        # Policy inference (no training)
        with torch.no_grad():
            action = self.policy(obs)

        # Convert action to GaitParams
        return self._action_to_gait_params(action)
```

**Option 2: RL as Layer 6+**
- Layer 5 stays rule-based (fallback)
- Layer 6 runs RL policy
- Layer 6 outputs velocity commands → Layer 5 converts to gaits

---

## Training Infrastructure

### Directory Structure
```
training/
├── collect_trajectories.py   # Data collection from sim
├── train_hnn.py              # HNN dynamics training
├── train_ppo.py              # RL policy training
├── validate_hnn.py           # HNN validation
├── validate_ppo.py           # RL policy validation
└── requirements.txt          # jax, flax, torch, sb3

training_data/
├── trajectories.npz          # HNN training data
├── test_trajectories.npz     # HNN test data
└── rl_episodes/              # RL training episodes

models/
├── hnn_trained.pkl           # Trained HNN (for Layer 4)
└── ppo_locomotion.zip        # Trained RL policy (for Layer 5/6)
```

### Compute Requirements

**HNN Training**:
- GPU: RTX 3090 or better
- Time: ~2-4 hours for 10k episodes
- Data: ~5 GB trajectory data

**RL Training**:
- GPU: RTX 4090 or better (or 8× RTX 3090)
- Time: ~12-24 hours for 10M steps
- Parallel envs: 8-16 workers

---

## Phased Rollout

### Phase 1: HNN Offline (2-3 weeks)
1. Collect trajectories in MuJoCo
2. Train HNN on dynamics
3. Deploy frozen HNN to Layer 4 for prediction
4. Use HNN predictions to improve terrain estimation

**Result**: Better terrain estimates from physics-based prediction errors

### Phase 2: RL Training (4-6 weeks)
1. Define observation/action spaces
2. Train PPO policy in simulation
3. Deploy frozen policy to Layer 5 or 6
4. Compare RL vs rule-based performance

**Result**: Learned locomotion policy

### Phase 3: Online Adaptation (8-10 weeks)
1. Design online learning architecture (Option B or C)
2. Implement continuous parameter updates
3. Safety guardrails (freeze if diverging)
4. Monitor and logging

**Result**: Adaptive system that improves with experience

---

## Key Architectural Decisions

### Decision 1: Where Does Trained Model Live?

**HNN**: Layer 4 inference class loads frozen params
**RL Policy**: Layer 5 or 6 inference class loads frozen params

**Both**: Stateless inference only, no training in deployment

### Decision 2: Online Learning?

**Phase 1-2**: No online learning (frozen models)
**Phase 3**: If needed, add learning in Layer 5+ or separate service

### Decision 3: Training vs Deployment Separation

**Training**: Separate `training/` directory, not part of any layer
**Deployment**: Layers load frozen models, do inference only

---

## Next Steps

1. **Immediate**: Set up `training/` directory structure
2. **Week 1**: Implement trajectory collection
3. **Week 2-3**: Train and validate HNN
4. **Week 4**: Deploy frozen HNN to Layer 4
5. **Week 5-8**: Train RL policy
6. **Week 9**: Deploy frozen policy
7. **Week 10**: Compare learned vs rule-based

---

**Key Insight**: Learning happens **offline** in a separate training process. Layers receive **frozen models** and do stateless inference. This respects all architectural boundaries while enabling sophisticated learning.

---

**Status**: Roadmap defined, ready to implement
**Next**: Create `training/` directory and begin data collection
