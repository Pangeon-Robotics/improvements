# HNN Placement and Role in FEP

**Question**: Where does the Hamiltonian Neural Network (HNN) live, and does the same network minimize surprise?

**Short Answer**:
- **Where**: Layer 4 (Dynamics/Trajectory)
- **Minimize surprise**: Yes, but via two different mechanisms (prediction + learning)

---

## Where the HNN Lives: Layer 4

### Why Layer 4?

**Layer 4 is the physics/dynamics layer** - it predicts trajectories in Cartesian space.

```
Layer 5: "I want to trot at 1 m/s"
    ↓ GaitParams
Layer 4: "Given this gait, the feet will follow this trajectory"
    ↓ CartesianPositions (predicted)
Layer 3: "Convert those positions to joint angles"
    ↓ Joint angles
Layer 2: "Apply torques to achieve those angles"
```

**The HNN's job**: Predict how the system evolves in phase space
```
H(q, p) → (q̇, ṗ)   (Hamilton's equations)
```

Where:
- `q`: Generalized positions (foot positions, base pose)
- `p`: Generalized momenta (related to velocities)
- `q̇`: Velocity predictions
- `ṗ`: Force/acceleration predictions

### Layer 4 Architecture with HNN

```python
# layer_4/predictor.py

class Layer4_DynamicsPredictor:
    """Predicts trajectories using HNN."""

    def __init__(self):
        self.hnn = HamiltonianNN()           # The physics model
        self.contact_model = ContactModel()  # Terrain interaction

    def predict(self, gait_params: GaitParams,
                current_state: KinematicState) -> DynamicState:
        """
        Forward prediction: What will happen?

        This is the HNN's PRIMARY use in FEP.
        """
        # Extract phase space state from kinematics
        q = current_state.foot_positions_world
        p = self.compute_momentum(current_state.foot_velocities_world)

        # HNN predicts dynamics
        q_dot, p_dot = self.hnn.predict_dynamics(q, p)

        # Integrate forward (predict next state)
        dt = 0.01  # 100 Hz
        q_next = q + dt * q_dot
        p_next = p + dt * p_dot

        # Predict contact forces from terrain model
        forces_pred = self.contact_model.predict(q_next, contact_states)

        return DynamicState(
            predicted_foot_positions=q_next,
            predicted_foot_velocities=self.momentum_to_velocity(p_next),
            predicted_contact_forces=forces_pred,
            actual_kinematics=current_state,
            # Errors computed by comparing prediction vs actual
            position_error=current_state.foot_positions_world - q_next,
            # ... etc
        )
```

### Not Layer 3 or Layer 5

**Why not Layer 3?**
- Layer 3 is **kinematic** (geometry only, no dynamics)
- IK is purely geometric: Cartesian positions → joint angles
- No prediction over time, just instantaneous mapping

**Why not Layer 5?**
- Layer 5 is **behavioral** (high-level gait selection)
- Too high-level for physics prediction
- Operates on gait-cycle timescales (1-10 sec), not dynamics timescales (0.1-1 sec)

---

## How the HNN Minimizes Surprise

The **same HNN** minimizes surprise via **two complementary mechanisms**:

### 1. Prediction (Forward Pass) - Active Inference

**Goal**: Take actions that make observations match predictions

```python
class Layer4_ActionSelection:
    """Use HNN to choose actions that minimize expected surprise."""

    def select_action(self, current_state: KinematicState,
                      goal: GaitParams) -> CartesianPositions:
        """
        Active Inference: Choose action that minimizes expected free energy.

        Expected Free Energy (EFE) = -E[log P(o|a)] - E[KL(Q||P)]
                                    = Pragmatic     + Epistemic
        """
        # Generate multiple candidate actions (gait variations)
        candidates = self.generate_candidates(goal)

        efe = []
        for action in candidates:
            # Use HNN to predict outcome
            predicted_state = self.hnn.rollout(current_state, action, horizon=10)

            # Pragmatic value: Will I achieve my goal?
            pragmatic = self.compute_goal_likelihood(predicted_state, goal)

            # Epistemic value: Will I learn something?
            epistemic = self.compute_information_gain(predicted_state)

            # Expected Free Energy
            efe.append(-pragmatic + 0.1 * epistemic)

        # Choose action with minimum EFE
        best_action = candidates[np.argmin(efe)]
        return best_action
```

**Key insight**: HNN predicts "if I take action A, observation O will occur". Choose A such that O matches goal.

---

### 2. Learning (Backward Pass) - Perceptual Inference

**Goal**: Update model parameters to better predict observations

```python
class Layer4_ModelUpdater:
    """Update HNN parameters from prediction errors (online learning)."""

    def update(self, prediction: DynamicState):
        """
        Perceptual Inference: Minimize prediction error by updating beliefs.

        Free Energy (variational) = E_Q[log Q(θ) - log P(o,θ)]
        """
        # Prediction error (surprise!)
        error = prediction.actual_kinematics.foot_positions_world - \
                prediction.predicted_foot_positions

        # Gradient descent on HNN parameters
        loss = np.mean(error ** 2)  # MSE prediction error
        grads = compute_gradients(self.hnn, loss)
        self.hnn.update_parameters(grads, lr=0.001)

        # Also update terrain model
        force_error = prediction.force_error
        self.contact_model.update(force_error)
```

**Key insight**: HNN parameters (θ) are "beliefs" about physics. Update them to minimize surprise.

---

## The Complete FEP Loop at Layer 4

```
┌─────────────────────────────────────────────────────────┐
│  Layer 4: Dynamics with HNN                             │
│                                                          │
│  1. PREDICT (Forward):                                  │
│     HNN: q,p → q_next, p_next                          │
│     "What will happen?"                                 │
│                                                          │
│  2. OBSERVE (from Layer 3):                            │
│     KinematicState: actual q, actual p                 │
│     "What actually happened?"                           │
│                                                          │
│  3. COMPUTE ERROR:                                      │
│     surprise = ||actual - predicted||²                 │
│     "How wrong was I?"                                  │
│                                                          │
│  4. UPDATE MODEL (Learning):                            │
│     θ_new = θ_old - α * ∇_θ(surprise)                 │
│     "Make better predictions"                           │
│                                                          │
│  5. SELECT ACTION (Active Inference):                   │
│     action = argmin_a E[surprise | a]                  │
│     "Act to confirm predictions"                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Two Uses of the Same HNN

| Use Case | Mechanism | Direction | Goal |
|----------|-----------|-----------|------|
| **Prediction** | Forward pass | Current state → Future state | Generate expectations |
| **Action Selection** | Forward rollout | Simulate actions → Pick best | Minimize expected error |
| **Learning** | Backward pass (gradients) | Errors → Parameter updates | Minimize actual error |

**Same network, three roles**:
1. **Generator**: HNN generates predictions
2. **Planner**: HNN evaluates action candidates
3. **Learner**: HNN parameters updated from errors

---

## Multi-Timescale Separation

Different layers minimize surprise at different rates:

| Layer | HNN Role | Update Frequency | What It Minimizes |
|-------|----------|------------------|-------------------|
| **Layer 4** | Core dynamics model | Every step (100 Hz) | Trajectory prediction error |
| **Layer 5** | Uses Layer 4's predictions | Every gait cycle (1-2 Hz) | Gait selection error |
| **Offline** | Pre-training | Hours/days | Generalization error on dataset |

**Online learning** (Layer 4): Fast adaptation to terrain
**Offline training** (Pre-deployment): Learn general physics

---

## Training vs Deployment

### Offline Training (Before Deployment)

```python
# Collect data in simulation
trajectories = collect_trajectories(sim, episodes=10000)

# Train HNN to predict dynamics
hnn = HamiltonianNN()
for epoch in range(1000):
    for (q, p, q_next, p_next) in trajectories:
        q_pred, p_pred = hnn.integrate_step(q, p, dt)
        loss = mse(q_pred, q_next) + mse(p_pred, p_next)
        hnn.update(loss)

# Save trained model
hnn.save("layer_4/models/hnn_pretrained.pkl")
```

### Online Deployment (On Robot)

```python
# Load pre-trained HNN
hnn = load_hnn("layer_4/models/hnn_pretrained.pkl")

# Online adaptation loop
for step in range(mission):
    # 1. Predict using HNN
    prediction = hnn.predict(current_state)

    # 2. Observe actual outcome
    actual = layer3.get_kinematic_state()

    # 3. Compute surprise
    error = actual - prediction

    # 4. Update HNN (online learning)
    if error > threshold:
        hnn.adapt(error, lr_online=0.0001)  # Smaller LR than offline

    # 5. Use updated HNN for next action
    action = select_action_with_hnn(hnn, goal)
```

**Key difference**:
- Offline: Large LR, many epochs, diverse data
- Online: Small LR, single-pass, recent data only

---

## Why Not Multiple HNNs?

**Question**: Should we have separate HNNs for prediction vs action selection?

**Answer**: No, use the **same HNN** because:

1. **Consistency**: Same model ensures predictions match what actions are based on
2. **Efficiency**: One model to train/update, not two
3. **FEP principle**: The generative model is both predictor and guide for action
4. **Sample efficiency**: Single model learns from all interactions

---

## HNN State Space: What Variables?

The HNN can operate in different state spaces:

### Option A: Joint Space (q = joint angles)
```python
H(q_joints, p_joints) → (q̇_joints, ṗ_joints)
```
**Pros**: Natural for robot, matches actuator space
**Cons**: Less interpretable, 12-dimensional

### Option B: Cartesian Space (q = foot positions)
```python
H(q_feet, p_feet) → (q̇_feet, ṗ_feet)
```
**Pros**: Interpretable, matches Layer 4 outputs, 12-dimensional (4 feet × 3D)
**Cons**: Doesn't capture full robot state (base pose missing)

### Option C: Hybrid (q = feet + base)
```python
H(q_feet, q_base, p_feet, p_base) → (q̇, ṗ)
```
**Pros**: Complete state, captures full dynamics
**Cons**: Higher dimensional (18-24D), more complex

**Recommendation**: **Option C (Hybrid)** for completeness, but start with **Option B (Cartesian feet)** for simplicity.

---

## Summary: HNN Placement & Roles

### Where: Layer 4 (Dynamics)
✅ Right abstraction level (trajectories in Cartesian space)
✅ Right timescale (0.1-1 sec, gait cycle prediction)
✅ Right information flow (receives kinematics from L3, sends predictions to L5)

### How It Minimizes Surprise: Two Mechanisms

**1. Active Inference (Action Selection)**
- Use HNN to simulate action candidates
- Choose action minimizing expected surprise
- "Act to make predictions come true"

**2. Perceptual Inference (Learning)**
- Compare HNN predictions with actual observations
- Update HNN parameters to reduce error
- "Improve model to better predict reality"

### One HNN, Three Roles
1. **Predictor**: Generate expected trajectories
2. **Planner**: Evaluate action candidates
3. **Learner**: Adapt parameters from errors

**Same network throughout** - this is the FEP principle in action.

---

## Implementation Checklist

- [ ] Implement `HamiltonianNN` in `layer_4/models/hamiltonian_nn.py`
- [ ] Add forward prediction: `predict_dynamics(q, p) → (q̇, ṗ)`
- [ ] Add action rollout: `rollout(state, action, horizon) → trajectory`
- [ ] Add online learning: `adapt(error) → update θ`
- [ ] Integrate with `DynamicState` publisher
- [ ] Pre-train offline on simulation data
- [ ] Deploy with online adaptation enabled
- [ ] Measure prediction accuracy vs surprise minimization

---

**Last updated**: 2026-02-10
**Status**: Design specification
**Next**: Implement HNN in Layer 4 as specified
