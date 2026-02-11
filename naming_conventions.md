# Naming Conventions: Semantic Hierarchy

**Status**: Design discussion
**Created**: 2026-02-10
**Purpose**: Establish clear, semantically meaningful names for cross-layer interfaces

---

## The Problem with Generic Names

Current naming: `Layer3Observation`, `Layer4Observation`
- ✅ Clear about layer boundaries
- ❌ No semantic meaning
- ❌ Doesn't communicate *what* the data represents
- ❌ Breaks down when layers get refactored or renamed

**Better approach**: Names should reflect the **abstraction level** and **semantic content**.

---

## Core Naming Principles

### 1. Semantic Content Over Layer Numbers

**Bad**: `Layer3Observation` (what layer? what data?)
**Good**: `KinematicState` (body kinematics in world frame)

### 2. Direction-Specific Suffixes

**Downward (commands)**: What you want to happen
- `Command`, `Target`, `Goal`, `Parameters`

**Upward (observations)**: What actually happened
- `State`, `Observation`, `Measurement`, `Estimate`

### 3. Abstraction Level in Name

Names should indicate level of processing:
- **Raw**: Unprocessed sensor data
- **Processed**: Filtered, transformed, or derived
- **Predicted**: Model-based expectations
- **Estimated**: Inferred hidden states

### 4. Avoid Abbreviations (Except Standard Ones)

**Good**: `KinematicState`, `DynamicsPrediction`, `IMU`
**Bad**: `KinState`, `DynPred`, `GyroAccel`

---

## Proposed Semantic Hierarchy

### Layer 1 → 2: Physical Signals
**Current**: N/A (internal to firmware)
**Semantic**: Raw motor currents, encoder ticks, ADC values
**Proposed**: `PhysicalSignals` (if exposed)

---

### Layer 2 → 3: Actuator State
**Current**: `LowState` (from Unitree SDK)
**Semantic**: Joint encoders, IMU readings, contact forces in sensor frames
**Proposed**: Keep `LowState` (SDK-defined, can't change)

**Why keep it?**
- Part of Unitree SDK contract
- Changing it breaks compatibility
- "Low" refers to "low-level" hardware interface

---

### Layer 3 → 4: Kinematic State
**Current**: `Layer3Observation`
**Semantic**: Body geometry in world frame — foot positions, velocities, contact states

**Option A: Domain-Specific**
```python
@dataclass
class KinematicState:
    """Body kinematics in world frame (Layer 3 → 4)."""
    foot_positions_world: np.ndarray   # (4, 3) meters
    foot_velocities_world: np.ndarray  # (4, 3) m/s
    foot_contact_states: np.ndarray    # (4,) boolean
    foot_contact_forces: np.ndarray    # (4, 3) Newtons, world frame
    base_orientation: np.ndarray       # (4,) quaternion (w, x, y, z)
    base_angular_velocity: np.ndarray  # (3,) rad/s
    timestamp: float
```

**Option B: Frame-Oriented**
```python
@dataclass
class WorldFrameState:
    """Robot state expressed in world coordinates (Layer 3 → 4)."""
    # Same fields as above
```

**Option C: Layer-Aware Semantic**
```python
@dataclass
class IKObservation:
    """Inverse Kinematics layer upward observations (Layer 3 → 4)."""
    # Same fields
```

**Recommendation**: **Option A: `KinematicState`**
- Clearest semantic meaning (geometry/kinematics)
- Matches the physics hierarchy (kinematics → dynamics)
- Not tied to specific implementation (IK could be replaced with learned FK)

---

### Layer 4 → 5: Dynamic State
**Current**: `Layer4Observation`
**Semantic**: Trajectory predictions, dynamics errors, terrain properties

**Option A: Physics-Oriented**
```python
@dataclass
class DynamicState:
    """Dynamics predictions and errors (Layer 4 → 5)."""
    predicted_trajectory: CartesianPositions  # What we expected
    actual_kinematics: KinematicState         # What we observed (from L3)

    # Prediction errors (surprise!)
    position_error: np.ndarray      # (4, 3) meters
    velocity_error: np.ndarray      # (4, 3) m/s
    contact_force_error: np.ndarray # (4, 3) Newtons

    # Inferred terrain properties
    terrain_stiffness: float        # Pa or N/m (derived from errors)
    terrain_friction: float         # Coefficient (if estimated)
    terrain_surprise: float         # std(errors) over recent window

    gait_phase: float               # [0, 1) within current gait cycle
    timestamp: float
```

**Option B: Prediction-Oriented**
```python
@dataclass
class TrajectoryPrediction:
    """Predicted vs actual trajectory state (Layer 4 → 5)."""
    # Same fields
```

**Option C: Error-Oriented (FEP-centric)**
```python
@dataclass
class PredictionError:
    """Hierarchical prediction errors for FEP (Layer 4 → 5)."""
    # Same fields
```

**Recommendation**: **Option A: `DynamicState`**
- Matches physics abstraction (dynamics = forces + motion)
- Parallel to `KinematicState` (natural progression)
- Not tied to specific implementation (could use HNN, MBRL, or analytical models)

---

### Layer 5 → 6: Behavioral State
**Current**: Not yet implemented
**Semantic**: Gait selection, terrain classification, strategic planning

**Proposed**:
```python
@dataclass
class BehavioralState:
    """Locomotion behavior state and terrain model (Layer 5 → 6)."""
    current_gait: str               # 'walk', 'trot', 'bound', 'stand'
    gait_stability: float           # [0, 1] confidence in current gait

    # Terrain classification
    terrain_type: str               # 'flat', 'stairs', 'slope', 'rough'
    terrain_confidence: float       # [0, 1] belief certainty

    # Forward model predictions
    expected_velocity: np.ndarray   # (3,) m/s — predicted forward progress
    velocity_variance: np.ndarray   # (3,) uncertainty in prediction

    # Free energy components
    pragmatic_value: float          # How well are we achieving goals?
    epistemic_value: float          # How much are we learning?

    timestamp: float
```

**Alternative**: `LocomotionState` (more specific than "Behavioral")

---

## Command Naming (Downward Flow)

### Layer 5 → 4: Gait Parameters
**Current**: `GaitParams`
**Semantic**: High-level gait specification

**Keep `GaitParams`** — already semantically clear:
```python
@dataclass
class GaitParams:
    gait_type: str        # 'trot', 'walk', etc.
    vx: float             # m/s forward
    vy: float             # m/s lateral
    wz: float             # rad/s yaw rate
    step_height: float    # meters
    step_length: float    # meters
    gait_freq: float      # Hz
    duty_cycle: float     # fraction in stance
    body_height: float    # meters (optional)
```

---

### Layer 4 → 3: Cartesian Positions
**Current**: `CartesianPositions`
**Semantic**: Desired foot and base positions in space

**Keep `CartesianPositions`** — clear and standard:
```python
@dataclass
class CartesianPositions:
    foot_positions: np.ndarray   # (4, 3) in base frame
    base_corners: np.ndarray     # (4, 3) in base frame
    timestamp: float
```

**Alternative consideration**: `CartesianTarget` or `CartesianGoal`
- Makes direction explicit (downward = target)
- But `Positions` is already clear in context

**Verdict**: Keep `CartesianPositions`

---

### Layer 3 → 2: Joint Commands
**Current**: `LowCmd` (from Unitree SDK)
**Semantic**: Joint angle targets + PD gains

**Keep `LowCmd`** — SDK-defined, can't change

---

## DDS Topic Naming

### Current Scheme
```
rt/lowstate           # Layer 2 → 3 (SDK-defined)
rt/lowcmd             # Layer 3 → 2 (SDK-defined)
```

### Proposed Extension

**Option A: Semantic Names**
```
# Upward (observations)
rt/actuator_state     # Layer 2 → 3 (alias for lowstate)
rt/kinematic_state    # Layer 3 → 4
rt/dynamic_state      # Layer 4 → 5
rt/behavioral_state   # Layer 5 → 6

# Downward (commands)
rt/joint_command      # Layer 3 → 2 (alias for lowcmd)
rt/cartesian_target   # Layer 4 → 3
rt/gait_parameters    # Layer 5 → 4
rt/motion_goal        # Layer 6 → 5
```

**Option B: Layer-Prefixed (Current)**
```
# Upward
rt/lowstate           # Layer 2 → 3 (keep SDK name)
rt/layer3/obs         # Layer 3 → 4
rt/layer4/obs         # Layer 4 → 5

# Downward
rt/lowcmd             # Layer 3 → 2 (keep SDK name)
rt/layer4/cmd         # Layer 4 → 3
rt/layer5/cmd         # Layer 5 → 4
```

**Option C: Hybrid (Semantic + Layer)**
```
# Upward
rt/lowstate                # Layer 2 → 3 (SDK)
rt/l3/kinematic_state      # Layer 3 → 4
rt/l4/dynamic_state        # Layer 4 → 5
rt/l5/behavioral_state     # Layer 5 → 6

# Downward
rt/lowcmd                  # Layer 3 → 2 (SDK)
rt/l4/cartesian_target     # Layer 4 → 3
rt/l5/gait_parameters      # Layer 5 → 4
```

**Recommendation**: **Option C: Hybrid**
- Semantic names communicate content
- Layer prefix enables filtering (`rt/l4/*`)
- Clear namespace separation
- Easy to add diagnostics per layer (`rt/l4/diagnostics`)

---

## Class Hierarchy & Inheritance

### Question: Should states inherit from a base class?

**Option 1: No Inheritance (Flat)**
```python
@dataclass
class KinematicState:
    # fields
    timestamp: float

@dataclass
class DynamicState:
    # different fields
    timestamp: float
```

**Pros**: Simple, explicit, no hidden behavior
**Cons**: Repeated fields (timestamp, etc.)

---

**Option 2: Base Class**
```python
@dataclass
class RobotState:
    """Base class for all upward-flowing state observations."""
    timestamp: float
    layer_id: int  # Which layer produced this

@dataclass
class KinematicState(RobotState):
    foot_positions_world: np.ndarray
    # ...
```

**Pros**: Shared interface, timestamp always present
**Cons**: Inheritance complexity, tight coupling

---

**Option 3: Protocol (Structural Typing)**
```python
from typing import Protocol

class HasTimestamp(Protocol):
    timestamp: float

@dataclass
class KinematicState:
    # fields
    timestamp: float  # Satisfies HasTimestamp

def process_state(state: HasTimestamp):
    print(state.timestamp)  # Works with any state
```

**Pros**: Duck typing, no inheritance, flexible
**Cons**: Less explicit than base class

---

**Recommendation**: **Option 1: Flat (No Inheritance)**
- Matches dataclass philosophy (simple, explicit)
- Each layer's state is independent
- Can always refactor later if needed
- Repeated `timestamp` is minimal overhead

---

## Naming Trade-offs

### Generic vs Semantic

| Approach | Pros | Cons |
|----------|------|------|
| **Generic** (`Layer3Observation`) | Clear layer boundary, easy to generate | No semantic meaning, breaks on refactor |
| **Semantic** (`KinematicState`) | Self-documenting, abstraction-aware | Requires design thought, harder to automate |
| **Hybrid** (`L3_KinematicState`) | Both benefits | Verbose, redundant info |

**Decision**: **Pure Semantic** for state classes, **Hybrid for DDS topics**

---

### Consistency Patterns

**Observation types** (upward):
- `KinematicState` (not `KinematicObservation`)
- `DynamicState` (not `DynamicsObservation`)
- `BehavioralState` (not `BehaviorObservation`)

**Rationale**: "State" is more accurate — these are the robot's estimated state at each abstraction level, not raw observations.

**Command types** (downward):
- `CartesianPositions` (not `CartesianCommand`)
- `GaitParams` (not `GaitCommand`)
- `LowCmd` (SDK-defined, keep as-is)

**Rationale**: Most are already well-named. "Params" and "Positions" clearly indicate intent.

---

## Complete Naming Proposal

### Upward (State Observations)

| Interface | Current Name | Proposed Name | DDS Topic |
|-----------|--------------|---------------|-----------|
| L2 → L3 | `LowState` | `LowState` (keep) | `rt/lowstate` |
| L3 → L4 | `Layer3Observation` | `KinematicState` | `rt/l3/kinematic_state` |
| L4 → L5 | `Layer4Observation` | `DynamicState` | `rt/l4/dynamic_state` |
| L5 → L6 | N/A | `BehavioralState` | `rt/l5/behavioral_state` |

### Downward (Commands/Targets)

| Interface | Current Name | Proposed Name | DDS Topic |
|-----------|--------------|---------------|-----------|
| L6 → L5 | N/A | `MotionGoal` | `rt/l6/motion_goal` |
| L5 → L4 | `GaitParams` | `GaitParams` (keep) | `rt/l5/gait_params` |
| L4 → L3 | `CartesianPositions` | `CartesianPositions` (keep) | `rt/l4/cartesian_target` |
| L3 → L2 | `LowCmd` | `LowCmd` (keep) | `rt/lowcmd` |

---

## Implementation Strategy

### Phase 1: Aliases (Backward Compatible)
```python
# layer_3/observations.py
@dataclass
class KinematicState:
    """Body kinematics in world frame (Layer 3 → 4)."""
    # fields...

# Backward compatibility
Layer3Observation = KinematicState  # Alias during transition
```

### Phase 2: Deprecation Warnings
```python
import warnings

def Layer3Observation(*args, **kwargs):
    warnings.warn(
        "Layer3Observation is deprecated, use KinematicState",
        DeprecationWarning
    )
    return KinematicState(*args, **kwargs)
```

### Phase 3: Full Migration
- Update all code to use new names
- Remove aliases
- Update documentation

---

## Open Questions for Discussion

1. **State vs Observation suffix?**
   - `KinematicState` (current proposal)
   - `KinematicObservation` (more explicit about direction)
   - `Kinematics` (shortest, but maybe too terse)

2. **Should DDS topics include units?**
   - `rt/l3/kinematic_state_100hz` (indicates rate)
   - `rt/l3/kinematic_state` (cleaner, rate is implicit)

3. **Diagnostic topics naming?**
   - `rt/l3/diagnostics` (per-layer)
   - `rt/diagnostics/l3` (centralized namespace)
   - `rt/l3/kinematic_state/diagnostics` (explicit hierarchy)

4. **Prediction vs Actual in DynamicState?**
   - Should `DynamicState` include both predicted and actual?
   - Or separate into `DynamicPrediction` and `DynamicMeasurement`?

5. **Terrain model location?**
   - Part of `DynamicState` (current proposal)
   - Separate `TerrainModel` class
   - Part of `BehavioralState` (Layer 5 owns terrain belief)

---

## Recommendations Summary

1. ✅ **Use semantic names for state classes**: `KinematicState`, `DynamicState`, `BehavioralState`

2. ✅ **Hybrid DDS topic naming**: `rt/l{N}/{semantic_name}` (e.g., `rt/l3/kinematic_state`)

3. ✅ **Keep SDK names unchanged**: `LowState`, `LowCmd`

4. ✅ **No inheritance hierarchy**: Flat dataclasses with repeated fields

5. ✅ **Consistent suffixes**:
   - Upward: `*State` (observations)
   - Downward: `*Params`, `*Positions`, `*Goal` (commands)

6. ✅ **Add type hints and docstrings** to all dataclasses

7. ⚠️ **Reflex constraint**: Fast reflexes (stumble recovery) must live in Layer 3 or Layer 2 — can't wait for Layer 5 FEP inference

---

## Next Steps

1. **Get feedback** on semantic names vs layer numbers
2. **Prototype** `KinematicState` in Layer 3
3. **Measure latency** of full observation chain (L2→L3→L4→L5)
4. **Define reflex boundary**: What stabilization must happen in Layer 3 vs Layer 5?
5. **Document** the FEP "explain away" model per layer (as user described)

---

**Last updated**: 2026-02-10
**Contributors**: User validation on FEP requirements, architectural necessity
**Status**: Proposal — needs consensus before implementation
