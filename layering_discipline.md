# Layering Discipline: N → N-1 Only

**Status**: Architectural constraint
**Created**: 2026-02-10
**Rule**: Each Layer N may ONLY interact directly with Layer N-1 (below) and Layer N+1 (above)

---

## The Problem

The FEP implementation plan proposes:
- Layer 4 needs contact forces to predict terrain
- Contact forces come from Layer 2 (in `LowState`)
- **This would violate N → N-1 discipline!** (Layer 4 → Layer 2 skips Layer 3)

---

## The Solution: Observation Chain

**Downward (commands)**: Already correct
```
Layer 5 → Layer 4: GaitParams
Layer 4 → Layer 3: CartesianPositions
Layer 3 → Layer 2: LowCmd (joint angles + gains)
Layer 2 → Layer 1: Torques
```

**Upward (observations)**: Must also respect layering
```
Layer 1 → Layer 2: Raw sensors
Layer 2 → Layer 3: LowState (joints, IMU, contact forces)
Layer 3 → Layer 4: Processed observations (foot states in world frame)
Layer 4 → Layer 5: Prediction errors + terrain estimates
```

---

## Layer 3 Output: Processed Observations

Layer 3 currently outputs joint commands (downward). It must ALSO output processed observations (upward) for Layer 4.

**New file**: `layer_3/observation_publisher.py`

```python
@dataclass
class Layer3Observation:
    """Upward-facing observations from Layer 3 to Layer 4."""

    # Foot state in world frame
    foot_positions_world: np.ndarray  # (4, 3) - FK from joint angles
    foot_velocities_world: np.ndarray # (4, 3) - computed from dq
    foot_contact_states: np.ndarray   # (4,) - binary flags from force sensors
    foot_contact_forces: np.ndarray   # (4, 3) - forces in world frame

    # Base state
    base_orientation: np.ndarray      # (4,) - quaternion from IMU
    base_angular_velocity: np.ndarray # (3,) - rad/s from IMU

    # Timing
    timestamp: float
```

**Key transformations Layer 3 performs**:
1. **Forward kinematics**: Joint angles → foot positions in world frame
2. **Frame transforms**: Contact forces from sensor frame → world frame
3. **Contact detection**: Force threshold → binary contact state
4. **Velocity estimation**: dq + Jacobian → Cartesian velocities

**Why Layer 3?** It already has:
- IK solver (knows kinematics)
- Joint angles and velocities (from LowState)
- IMU data (from LowState)
- Contact forces (from LowState)

It's the natural place to compute world-frame quantities.

---

## Layer 4 Usage: Never Access LowState

Layer 4 prediction now uses ONLY Layer 3's observations.

**Updated**: `layer_4/predictor.py`

```python
class TrajectoryPredictor:
    """Predicts outcomes using ONLY Layer 3 observations."""

    def predict(self,
                positions: CartesianPositions,        # From Layer 4 generator
                actual_obs: Layer3Observation) -> PredictionResult:
        """
        Compare predicted vs actual.

        Args:
            positions: What Layer 4 generated (prediction)
            actual_obs: What Layer 3 observed (reality)

        Returns:
            Prediction errors + terrain estimates
        """
        # Position error
        pos_error = actual_obs.foot_positions_world - positions.foot_positions

        # Velocity error (if HNN is predicting velocities)
        vel_pred = self.hnn.predict_velocities(positions)
        vel_error = actual_obs.foot_velocities_world - vel_pred

        # Contact force error
        force_pred = self.contact_model.predict(positions)
        force_error = actual_obs.foot_contact_forces - force_pred

        # Derive terrain properties from errors
        terrain_stiffness = self.estimate_stiffness(force_error)

        return PredictionResult(pos_error, vel_error, force_error,
                               terrain_stiffness)
```

**No direct DDS subscription to `rt/lowstate`!** Layer 4 only reads from Layer 3.

---

## Layer 4 Output: Prediction Errors

Layer 4 passes processed results to Layer 5.

**New dataclass**: `layer_4/observations.py`

```python
@dataclass
class Layer4Observation:
    """Upward-facing observations from Layer 4 to Layer 5."""

    # What we generated (prediction)
    predicted_trajectory: CartesianPositions

    # What actually happened (from Layer 3)
    actual_positions: np.ndarray      # (4, 3)
    actual_velocities: np.ndarray     # (4, 3)

    # Surprise!
    position_error: np.ndarray        # (4, 3)
    velocity_error: np.ndarray        # (4, 3)

    # Derived terrain model
    terrain_stiffness: float          # Pa or N/m
    terrain_surprise: float           # std(errors) over recent window

    # Timing
    gait_phase: float                 # [0, 1)
    timestamp: float
```

---

## Layer 5 Usage: Never Access Layers 2 or 3

Layer 5 gait selection uses ONLY Layer 4's observations.

**Updated**: `layer_5/fep_selector.py`

```python
class FEPGaitSelector:
    """Select gait using ONLY Layer 4 observations."""

    def select_gait(self,
                    velocity_cmd: float,
                    layer4_obs: Layer4Observation) -> str:
        """
        Choose gait minimizing expected free energy.

        Args:
            velocity_cmd: Desired velocity (from Layer 6 or command interface)
            layer4_obs: Observations from Layer 4 (terrain surprise, errors)

        Returns:
            Gait name ('walk', 'trot', 'bound')
        """
        # Use Layer 4's terrain estimate, not raw sensor data
        terrain_surprise = layer4_obs.terrain_surprise

        efe = {}
        for gait in ['walk', 'trot', 'bound']:
            # Model uses terrain_surprise (abstracted)
            pragmatic = self.predict_success(gait, velocity_cmd, terrain_surprise)
            epistemic = self.predict_info_gain(gait, terrain_surprise)
            efe[gait] = -pragmatic + 0.1 * epistemic

        return min(efe, key=efe.get)
```

**No access to LowState or Layer 3 data!** Only Layer 4's processed observations.

---

## DDS Topic Architecture

### Downward (Commands)
```
Layer 5 publishes: rt/layer5/gait_params     (GaitParams)
Layer 4 publishes: rt/layer4/cartesian_cmd   (CartesianPositions)
Layer 3 publishes: rt/lowcmd                 (LowCmd) - existing
```

### Upward (Observations)
```
Layer 2 publishes: rt/lowstate               (LowState) - existing
Layer 3 publishes: rt/layer3/observations    (Layer3Observation) - NEW
Layer 4 publishes: rt/layer4/observations    (Layer4Observation) - NEW
```

**Critical**: Each layer subscribes ONLY to the layer directly below it.

---

## Implementation Changes

### 1. Add Layer 3 Observation Publisher

**File**: `layer_3/observation_publisher.py`

```python
from dataclasses import dataclass
from cyclonedds import Publisher
import numpy as np

@dataclass
class Layer3Observation:
    # (fields as defined above)
    pass

class ObservationPublisher:
    """Publishes processed observations for Layer 4."""

    def __init__(self):
        self.pub = Publisher("rt/layer3/observations", Layer3Observation)
        self.fk_solver = ForwardKinematics()  # Existing

    def publish(self, lowstate: LowState):
        """Transform LowState → Layer3Observation."""

        # Compute foot positions via FK
        q = np.array([lowstate.motor_state[i].q for i in range(12)])
        foot_pos = self.fk_solver.compute_foot_positions(q)

        # Transform contact forces to world frame
        R = quat_to_rotation(lowstate.imu_state.quaternion)
        forces_world = [R @ f for f in lowstate.foot_forces]

        # Detect contacts (binary)
        contact_threshold = 10.0  # Newtons
        contacts = [np.linalg.norm(f) > contact_threshold for f in forces_world]

        obs = Layer3Observation(
            foot_positions_world=foot_pos,
            foot_velocities_world=self.compute_velocities(q, lowstate.dq),
            foot_contact_states=np.array(contacts),
            foot_contact_forces=np.array(forces_world),
            base_orientation=lowstate.imu_state.quaternion,
            base_angular_velocity=lowstate.imu_state.gyroscope,
            timestamp=lowstate.timestamp
        )

        self.pub.write(obs)
```

**Integration**: Call from existing `controller.py` control loop after processing LowState.

---

### 2. Update Layer 4 to Subscribe to Layer 3

**File**: `layer_4/predictor.py`

```python
from cyclonedds import Subscriber
from layer_3.observation_publisher import Layer3Observation

class TrajectoryPredictor:
    def __init__(self):
        # Subscribe to Layer 3 observations (NOT LowState!)
        self.sub = Subscriber("rt/layer3/observations", Layer3Observation)
        self.sub.init(self._on_observation, 10)

        self.latest_obs = None

    def _on_observation(self, obs: Layer3Observation):
        self.latest_obs = obs

    def predict_and_compare(self, predicted: CartesianPositions):
        """Compare prediction vs Layer 3's actual observations."""
        if self.latest_obs is None:
            return None

        # Compute errors using ONLY Layer 3 data
        pos_error = self.latest_obs.foot_positions_world - predicted.foot_positions
        # ... rest of prediction logic
```

---

### 3. Update Layer 5 to Subscribe to Layer 4

**File**: `layer_5/fep_selector.py`

```python
from cyclonedds import Subscriber
from layer_4.observations import Layer4Observation

class FEPGaitSelector:
    def __init__(self):
        # Subscribe to Layer 4 observations (NOT Layer 3 or LowState!)
        self.sub = Subscriber("rt/layer4/observations", Layer4Observation)
        self.sub.init(self._on_observation, 10)

        self.terrain_history = []

    def _on_observation(self, obs: Layer4Observation):
        self.terrain_history.append(obs.terrain_surprise)
        # Keep last 100 observations
        if len(self.terrain_history) > 100:
            self.terrain_history.pop(0)
```

---

## Verification: Layering Audit

### Commands (Downward)
- ✅ Layer 5 → Layer 4: Publishes `GaitParams` to `rt/layer5/gait_params`
- ✅ Layer 4 → Layer 3: Publishes `CartesianPositions` to `rt/layer4/cartesian_cmd`
- ✅ Layer 3 → Layer 2: Publishes `LowCmd` to `rt/lowcmd`

### Observations (Upward)
- ✅ Layer 2 → Layer 3: Publishes `LowState` to `rt/lowstate`
- ✅ Layer 3 → Layer 4: Publishes `Layer3Observation` to `rt/layer3/observations`
- ✅ Layer 4 → Layer 5: Publishes `Layer4Observation` to `rt/layer4/observations`

### Subscriptions
- ✅ Layer 3 subscribes to: `rt/lowstate` (Layer 2 only)
- ✅ Layer 4 subscribes to: `rt/layer3/observations` (Layer 3 only)
- ✅ Layer 5 subscribes to: `rt/layer4/observations` (Layer 4 only)

**No layer skipping!** Each layer only talks to N-1 and N+1.

---

## Benefits of Strict Layering

1. **Testability**: Can test each layer independently with mock data from layer below
2. **Modularity**: Layer 4 doesn't need to know about DDS LowState message structure
3. **Abstraction**: Layer 5 works with "terrain surprise", not raw force sensors
4. **Evolution**: Can change Layer 2 sensor format without affecting Layer 4+
5. **Clarity**: Information flow is unidirectional and explicit

---

## Cost of Strict Layering

1. **Latency**: Each layer adds ~1ms processing + pub/sub overhead
2. **Bandwidth**: More DDS topics (3 new topics)
3. **Computation**: Layer 3 must compute FK even if not needed for its own logic
4. **Code**: More dataclasses, publishers, subscribers to maintain

**Verdict**: Worth it for architectural clarity and maintainability.

---

## Summary

**Rule**: Each layer N may ONLY:
- Publish commands to Layer N-1 (downward)
- Subscribe to observations from Layer N-1 (upward)
- Publish observations to Layer N+1 (upward)
- Subscribe to commands from Layer N+1 (downward)

**Violation examples** (don't do this):
- ❌ Layer 4 subscribing to `rt/lowstate` (Layer 2)
- ❌ Layer 5 reading joint angles directly
- ❌ Layer 3 publishing to Layer 5

**Correct examples**:
- ✅ Layer 3 publishes `Layer3Observation` with processed foot states
- ✅ Layer 4 subscribes to `rt/layer3/observations`
- ✅ Layer 5 subscribes to `rt/layer4/observations`

---

**Last updated**: 2026-02-10
**Status**: Architectural requirement (must follow)
**Applies to**: All FEP implementation phases
