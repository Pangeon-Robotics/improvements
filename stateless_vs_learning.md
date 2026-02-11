# Stateless Layer 4 vs HNN Learning Requirements

**Problem**: Layer 4 is defined as stateless (instant translation), but HNN requires:
- Parameter persistence (θ)
- Online learning (updating θ from errors)
- Error history for terrain estimation
- Model state (terrain stiffness, friction coefficients)

**Question**: Can stateless Layer 4 support HNN?

---

## The Architectural Constraint

From `philosophy/architecture.md`:

```
Layers 1-4: Instant (per-timestep) — pure translation, no planning
Layers 5-8: Sequences (planning over time) — maintain state, generate plans
```

**Layer 4 must be stateless**: `compute(GaitParams, t) → CartesianPositions`

No memory between calls. Pure function. Same input → same output.

---

## What HNN Actually Needs

### 1. Model Parameters (θ)
```python
class HamiltonianNN:
    def __init__(self):
        self.params = initialize_params()  # Weight matrices
```
**Requirement**: Parameters must persist across calls

### 2. Online Learning
```python
def update(self, error):
    self.params -= learning_rate * gradient(error)
```
**Requirement**: Parameters change based on history (temporal dependency!)

### 3. Terrain Model State
```python
self.terrain_stiffness = 1e6  # Learned from force errors
self.error_history = []       # Last 100 errors
```
**Requirement**: Maintains history, adapts over time

### 4. Prediction Error Tracking
```python
self.prediction_errors.append(current_error)
if len(self.prediction_errors) > window:
    self.prediction_errors.pop(0)
```
**Requirement**: Memory of past predictions

**Conclusion**: HNN as typically implemented **violates statelessness**.

---

## Three Solutions

### Solution 1: Model State vs Computational State ⚠️

**Idea**: Distinguish two types of state:
- **Model state**: Parameters that persist but are "configuration" (like PD gains)
- **Computational state**: History-dependent sequences (violates statelessness)

**Layer 4 under this interpretation**:
```python
class Layer4:
    def __init__(self):
        # Model state (OK to persist)
        self.hnn = HamiltonianNN()  # θ persists
        self.terrain_model = TerrainModel()  # Parameters persist

    def compute(self, gait_params: GaitParams, t: float) -> CartesianPositions:
        """Stateless computation: same (gait_params, t) → same output."""
        # Use persistent models, but computation is pure
        positions = self.trajectory_generator(gait_params, t)
        # NO: Don't update models here (that would be stateful)
        return positions
```

**Problem**: Where does learning happen? Models never update!

**Verdict**: ❌ Doesn't solve the learning problem

---

### Solution 2: Learning Lives in Layer 5 ✅ (RECOMMENDED)

**Key insight**: Separate **prediction** (stateless) from **learning** (stateful).

**Layer 4**: Stateless predictor
```python
class Layer4_Predictor:
    """Stateless: Given model parameters, predict trajectory."""

    def compute(self, gait_params: GaitParams,
                current_state: KinematicState,
                model_params: ModelParams,  # From Layer 5!
                t: float) -> CartesianPositions:
        """
        Pure function: (gait_params, state, params, t) → positions

        No internal state. Model params passed from Layer 5.
        """
        # Use HNN with given parameters (no updates)
        hnn = HamiltonianNN(model_params.hnn_weights)
        predicted_trajectory = hnn.rollout(current_state, gait_params, t)
        return predicted_trajectory
```

**Layer 5**: Stateful learner
```python
class Layer5_Locomotion:
    """Stateful: Maintains model parameters, updates from errors."""

    def __init__(self):
        # STATEFUL: Maintain model parameters
        self.hnn_params = initialize_hnn()
        self.terrain_model = TerrainModel()
        self.error_history = []

    def update(self, dynamic_state: DynamicState):
        """
        Receive prediction errors from Layer 4.
        Update model parameters.
        """
        # Accumulate errors (stateful!)
        self.error_history.append(dynamic_state.position_error)
        if len(self.error_history) > 100:
            self.error_history.pop(0)

        # Update HNN parameters (online learning)
        if np.std(self.error_history) > threshold:
            grads = compute_gradients(dynamic_state.prediction_error)
            self.hnn_params -= learning_rate * grads

        # Update terrain model
        self.terrain_model.adapt(dynamic_state.force_error)

    def get_model_params(self) -> ModelParams:
        """Layer 4 requests current model parameters."""
        return ModelParams(
            hnn_weights=self.hnn_params,
            terrain_stiffness=self.terrain_model.stiffness,
            terrain_friction=self.terrain_model.friction
        )
```

**Information Flow**:
```
┌─────────────────────────────────────────────────────────┐
│ Layer 5 (STATEFUL)                                      │
│  - Maintains HNN parameters θ                           │
│  - Maintains terrain model                              │
│  - Updates from prediction errors                       │
│  - Passes current params to Layer 4                     │
└─────────────────────────────────────────────────────────┘
         ↓ ModelParams                    ↑ PredictionError
┌─────────────────────────────────────────────────────────┐
│ Layer 4 (STATELESS)                                     │
│  - Pure predictor: (state, params) → trajectory        │
│  - Uses HNN with given params (no updates)             │
│  - Computes prediction errors                           │
│  - Returns CartesianPositions + DynamicState            │
└─────────────────────────────────────────────────────────┘
```

**Benefits**:
- ✅ Layer 4 remains stateless (pure function)
- ✅ Layer 5 handles all learning (stateful updates)
- ✅ Clean separation: prediction vs learning
- ✅ Layer 5 is already defined as stateful (first sequence layer)

**Verdict**: ✅ **This is the solution**

---

### Solution 3: Redefine Layer 4 as Stateful ❌

**Idea**: Acknowledge that prediction+learning inherently requires state.

**Change architecture**:
```
Layers 1-3: Instant (stateless)
Layers 4-8: Sequences (stateful)
```

**Arguments for**:
- Physics prediction inherently temporal
- HNN naturally lives in Layer 4
- Simpler implementation (no param passing)

**Arguments against**:
- ❌ Breaks established architecture
- ❌ Blurs distinction between instant/sequence layers
- ❌ Layer 4 would do both prediction AND learning (violates single responsibility)
- ❌ Makes Layer 4 harder to test (stateful = harder to reproduce)

**Verdict**: ❌ Don't do this - Solution 2 is cleaner

---

## Recommended Architecture: Solution 2

### Layer 4: Stateless Physics Predictor

**Responsibility**: Given model parameters, predict trajectory

**Interface**:
```python
@dataclass
class ModelParams:
    """Model parameters from Layer 5."""
    hnn_weights: Dict[str, np.ndarray]
    terrain_stiffness: float
    terrain_friction: float

def compute(gait_params: GaitParams,
            kinematic_state: KinematicState,
            model_params: ModelParams,
            t: float) -> Tuple[CartesianPositions, DynamicState]:
    """
    Stateless prediction.

    Args:
        gait_params: Desired gait from Layer 5
        kinematic_state: Current state from Layer 3
        model_params: Model weights from Layer 5
        t: Current time

    Returns:
        positions: Predicted foot positions
        dynamic_state: Predictions + errors (sent to Layer 5)
    """
    # Instantiate HNN with given parameters
    hnn = HamiltonianNN(model_params.hnn_weights)

    # Predict trajectory (stateless computation)
    predicted = hnn.predict(kinematic_state, gait_params, t)

    # Compute errors (comparison, not learning)
    errors = compute_prediction_errors(predicted, kinematic_state)

    # Package results
    dynamic_state = DynamicState(
        predicted_foot_positions=predicted.positions,
        actual_kinematics=kinematic_state,
        position_error=errors.position,
        velocity_error=errors.velocity,
        terrain_stiffness=model_params.terrain_stiffness,
        # ... etc
    )

    return predicted.positions, dynamic_state
```

**Key**: No `self.params`, no `self.error_history`, no updates. Pure function.

---

### Layer 5: Stateful Model Manager

**Responsibility**: Maintain and update model parameters

**Interface**:
```python
class Layer5_Locomotion:
    def __init__(self):
        # STATEFUL: Model parameters
        self.hnn_params = load_pretrained_hnn()
        self.terrain_model = TerrainModel()
        self.error_buffer = deque(maxlen=100)

        # STATEFUL: Learning rate schedule
        self.learning_rate = 0.001
        self.adaptation_step = 0

    def update_from_dynamics(self, dynamic_state: DynamicState):
        """
        Called every timestep with Layer 4's prediction errors.
        Updates model parameters.
        """
        # Accumulate error history
        self.error_buffer.append(dynamic_state.position_error)

        # Online learning: Update HNN
        if self.adaptation_step % 10 == 0:  # Every 10 steps
            avg_error = np.mean(list(self.error_buffer), axis=0)
            grads = self.compute_gradients(avg_error)
            self.hnn_params = self.update_parameters(self.hnn_params, grads)

        # Update terrain model
        self.terrain_model.adapt(
            force_error=dynamic_state.force_error,
            contact_states=dynamic_state.actual_kinematics.foot_contact_states
        )

        self.adaptation_step += 1

    def get_model_params(self) -> ModelParams:
        """Layer 4 calls this to get current model parameters."""
        return ModelParams(
            hnn_weights=self.hnn_params,
            terrain_stiffness=self.terrain_model.stiffness,
            terrain_friction=self.terrain_model.friction
        )

    def select_gait(self, velocity_cmd: float, terrain: str) -> GaitParams:
        """Main Layer 5 function: high-level gait selection."""
        # Use model predictions to choose gait
        # ...
```

**Key**: All stateful operations (learning, history, adaptation) live here.

---

## DDS Topic Flow

### Downward (Commands + Params)
```
rt/l5/gait_params        # Layer 5 → Layer 4: GaitParams
rt/l5/model_params       # Layer 5 → Layer 4: ModelParams (NEW!)
rt/l4/cartesian_target   # Layer 4 → Layer 3: CartesianPositions
```

### Upward (Observations + Errors)
```
rt/l3/kinematic_state    # Layer 3 → Layer 4: KinematicState
rt/l4/dynamic_state      # Layer 4 → Layer 5: DynamicState (includes errors!)
rt/l5/behavioral_state   # Layer 5 → Layer 6: BehavioralState
```

**Key addition**: `rt/l5/model_params` topic
- Layer 5 publishes updated model parameters
- Layer 4 subscribes and uses for prediction
- Updated at slower rate than control (e.g., 1-10 Hz, not 100 Hz)

---

## Implementation Pattern

### Layer 4: Stateless Function
```python
class Layer4_Controller:
    """Stateless trajectory predictor."""

    def __init__(self):
        # Subscribe to model params from Layer 5
        self.model_params_sub = Subscriber("rt/l5/model_params", ModelParams)
        self.model_params_sub.init(self._on_model_params, 10)

        # Subscribe to kinematics from Layer 3
        self.kin_state_sub = Subscriber("rt/l3/kinematic_state", KinematicState)

        # Publishers
        self.positions_pub = Publisher("rt/l4/cartesian_target", CartesianPositions)
        self.dynamics_pub = Publisher("rt/l4/dynamic_state", DynamicState)

        # Current params (updated by Layer 5)
        self.current_model_params = None

    def _on_model_params(self, params: ModelParams):
        """Receive updated model from Layer 5."""
        self.current_model_params = params

    def control_loop(self, gait_params: GaitParams, t: float):
        """Main loop: stateless prediction."""
        if self.current_model_params is None:
            return  # Wait for model from Layer 5

        # Get current kinematic state
        kin_state = self.get_latest_kinematic_state()

        # STATELESS COMPUTATION
        positions, dynamics = compute(
            gait_params,
            kin_state,
            self.current_model_params,  # From Layer 5
            t
        )

        # Publish results
        self.positions_pub.write(positions)  # → Layer 3
        self.dynamics_pub.write(dynamics)    # → Layer 5
```

### Layer 5: Stateful Learner
```python
class Layer5_Controller:
    """Stateful model manager + gait selector."""

    def __init__(self):
        # STATEFUL: Model parameters
        self.model_manager = ModelManager()

        # Subscribe to dynamics from Layer 4
        self.dynamics_sub = Subscriber("rt/l4/dynamic_state", DynamicState)
        self.dynamics_sub.init(self._on_dynamics, 10)

        # Publisher for updated models
        self.model_params_pub = Publisher("rt/l5/model_params", ModelParams)

    def _on_dynamics(self, dynamics: DynamicState):
        """Receive prediction errors from Layer 4."""
        # STATEFUL UPDATE
        self.model_manager.update_from_errors(dynamics)

        # Publish updated model params (10 Hz, not every step)
        if self.should_publish_model_update():
            params = self.model_manager.get_current_params()
            self.model_params_pub.write(params)

    def control_loop(self):
        """Main loop: gait selection."""
        # Use model predictions for gait selection
        gait = self.select_gait_via_fep(
            velocity_cmd=self.velocity_cmd,
            terrain_model=self.model_manager.terrain_model
        )
        # Publish gait params to Layer 4
        # ...
```

---

## Comparison Table

|  | Solution 1: Model State | Solution 2: Learning in L5 | Solution 3: Stateful L4 |
|--|-------------------------|----------------------------|-------------------------|
| **Layer 4 stateless?** | ⚠️ Technically yes | ✅ Yes | ❌ No |
| **Learning supported?** | ❌ No mechanism | ✅ Yes, in Layer 5 | ✅ Yes, in Layer 4 |
| **Clean separation?** | ❌ No | ✅ Yes (predict vs learn) | ⚠️ Mixed |
| **Testability** | ✅ Easy | ✅ Easy | ❌ Harder (stateful) |
| **Aligns with architecture?** | ⚠️ Debatable | ✅ Yes | ❌ No |
| **Implementation complexity** | Low | Medium | Low |

**Winner**: **Solution 2**

---

## Summary: Layer 4 Stateless + Learning in Layer 5

### Layer 4: Pure Predictor
- **Stateless**: Same inputs → same outputs
- **No learning**: Uses parameters given by Layer 5
- **Role**: Physics prediction (HNN forward pass)
- **Interface**: `compute(gait_params, kin_state, model_params, t) → (positions, errors)`

### Layer 5: Model Manager
- **Stateful**: Maintains model parameters, error history
- **Learning**: Updates HNN and terrain model from errors
- **Role**: Model adaptation + gait selection
- **Interface**:
  - `update_from_dynamics(errors)` — receives Layer 4 errors
  - `get_model_params()` — provides params to Layer 4

### Benefits
✅ Preserves Layer 4 statelessness
✅ Enables online learning (in Layer 5)
✅ Clean separation of concerns
✅ Aligns with existing architecture (L1-4 instant, L5+ sequences)

---

## Answer to Your Question

**Question**: Can stateless Layer 4 support HNN?

**Answer**:
- **No**, if HNN includes learning (parameter updates require state)
- **Yes**, if we split responsibilities:
  - Layer 4: Stateless **prediction** using HNN
  - Layer 5: Stateful **learning** of HNN parameters

**The HNN itself** can be stateless (pure function: params → predictions), but **learning the HNN** requires state (accumulating errors, updating params). We solve this by keeping prediction in Layer 4 and moving learning to Layer 5.

---

**Last updated**: 2026-02-10
**Status**: Architectural resolution
**Recommendation**: Implement Solution 2 (Learning in Layer 5)
