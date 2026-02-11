# Architecture Journey: FEP Implementation

**Date**: 2026-02-10
**Status**: Resolution documented
**Purpose**: Capture the architectural discovery process and key insights

---

## The Original Proposal (Issues #10, #17, #1)

**Initial approach**: Implement FEP with observation chain flowing upward through all layers.

Three GitHub issues were created:
- **Layer 3** [#10](https://github.com/Pangeon-Robotics/layer_3/issues/10): Publish `KinematicState` with world-frame kinematics
- **Layer 4** [#17](https://github.com/Pangeon-Robotics/layer_4/issues/17): Add HNN prediction, publish `DynamicState` with errors
- **Layer 5** [#1](https://github.com/Pangeon-Robotics/layer_5/issues/1): Receive errors, do HNN learning, select gaits

**What seemed right**: We designed strict N→N-1 layering with upward observations (see [layering_discipline.md](layering_discipline.md)).

**Fatal flaw**: Passed **low-level sensor arrays** to Layer 5 instead of **high-level abstractions**.

---

## The Rejection (Layer 5 Boundary Violations)

The Layer 5 team correctly **rejected issue #1** with 6 boundary violations:

1. **Forbidden state**: Sensor history (position errors, force errors)
2. **Cross-layer file access**: Direct imports from Layer 4 internals
3. **Non-public API dependency**: Using internal Layer 4 structures
4. **Feature creep**: Neural network training ≠ "motion commands → gait parameters"
5. **Dependency violation**: CycloneDDS in a numpy-only layer
6. **Sensor access violation**: Reading foot positions, forces, velocities

**Why this mattered**: Layer 5's identity is "motion commands → gait parameters". Proposed work was HNN learning and array processing - completely out of scope.

**Philosophy issue filed**: [philosophy#1](https://github.com/Pangeon-Robotics/philosophy/issues/1) - "Where do FEP/learning components live in the 8-layer stack?"

---

## The Key Insight: Abstraction Level

During architecture discussion, we realized:

> **Telemetry DOES flow upward** (as designed in layering_discipline.md), but each layer must **increase abstraction**.

### The Problem

Original `DynamicState` from Layer 4 → Layer 5:
```python
@dataclass
class DynamicState:
    position_error: np.ndarray    # (4, 3) meters  ← TOO LOW-LEVEL
    velocity_error: np.ndarray    # (4, 3) m/s     ← TOO LOW-LEVEL
    force_error: np.ndarray       # (4, 3) Newtons ← TOO LOW-LEVEL
```

**Why wrong**: Layer 5's vocabulary is "step_length, gait_freq, duty_cycle" - NOT "4×3 position error arrays"!

### The Solution

Corrected `TerrainEstimate` from Layer 4 → Layer 5:
```python
@dataclass
class TerrainEstimate:
    roughness: float        # 0-1 (from position error variance)
    compliance: float       # 0-1 (from force error - soft/hard)
    stability: float        # 0-1 (how well predictions match)
    surprise_level: float   # 0-1 (overall error magnitude)
    confidence: float       # 0-1 (data quality)
```

**Why correct**: High-level scalars in Layer 5's vocabulary. Layer 4 does all the array processing internally.

---

## Corrected Architecture

### Abstraction Gradient (Increasing)

```
Layer 3 → Layer 4: Arrays (4×3 foot positions, velocities, forces)
    ↓
Layer 4 → Layer 5: Scalars (roughness, compliance, stability)
    ↓
Layer 5 → Layer 6: Gait decisions (walk, step_height=0.08, duty_cycle=0.75)
```

### Responsibility Split

| Layer | Does | Doesn't Do |
|-------|------|------------|
| **Layer 3** | Forward kinematics, frame transforms, publish `KinematicState` | Prediction, terrain analysis |
| **Layer 4** | HNN prediction, error computation, **terrain abstraction**, publish `TerrainEstimate` | Gait selection |
| **Layer 5** | Terrain-aware gait selection using scalars | HNN learning, array processing, sensor access |

### Key Change: Layer 4 Abstraction Layer

**Layer 4 internal** (never exposed to Layer 5):
```python
class TerrainAnalyzer:
    def analyze(self, predicted_positions: np.ndarray,  # (4, 3)
                predicted_forces: np.ndarray,           # (4, 3)
                actual: KinematicState) -> TerrainEstimate:
        """Convert low-level arrays → high-level scalars."""

        # Compute raw errors (INTERNAL)
        pos_error = actual.foot_positions_world - predicted_positions
        force_error = actual.foot_contact_forces - predicted_forces

        # Abstract into scalar properties (EXPOSED)
        roughness = np.std(pos_error[:, :, 2]) / 0.05  # Normalize
        compliance = np.var(force_error[:, :, 2]) / 1000
        stability = 1.0 - np.mean(np.abs(pos_error)) / 0.1

        return TerrainEstimate(roughness, compliance, stability, ...)
```

**Layer 4 public API** (what Layer 5 calls):
```python
def get_terrain_estimate() -> TerrainEstimate:
    """Returns high-level terrain properties (scalars only)."""
    return current_terrain_estimate
```

---

## New GitHub Issues (Corrected)

All three issues were **closed** and **recreated** with corrected architecture:

- **Layer 3** [#11](https://github.com/Pangeon-Robotics/layer_3/issues/11): Add KinematicState publisher (unchanged - was correct)
- **Layer 4** [#18](https://github.com/Pangeon-Robotics/layer_4/issues/18): Add HNN + terrain analysis with **scalar abstraction**
- **Layer 5** [#2](https://github.com/Pangeon-Robotics/layer_5/issues/2): Terrain-aware gait selection (NO learning, NO arrays)

**Philosophy issue** [#1](https://github.com/Pangeon-Robotics/philosophy/issues/1): Closed as resolved - no architecture changes needed.

---

## Architectural Principles Validated

### ✅ N → N-1 Discipline Preserved
Each layer only accesses Layer N-1's public API:
- Layer 4 subscribes to `rt/l3/kinematic_state` (DDS)
- Layer 5 calls `get_terrain_estimate()` from Layer 4 API

### ✅ Layer Sovereignty Maintained
Layer 5 boundaries.md fully respected:
- "Motion commands → gait parameters" (scope unchanged)
- No sensor access (uses abstracted terrain properties)
- No forbidden state (no sensor history, foot positions, joint angles)
- numpy-only (no CycloneDDS, no JAX/Flax)

### ✅ Increasing Abstraction
Each layer increases abstraction level:
- Layer 2: Raw sensors (joint angles, IMU, forces)
- Layer 3: World-frame kinematics (arrays)
- Layer 4: Terrain properties (scalars)
- Layer 5: Gait decisions (categorical + scalars)

### ✅ API Enrichment Pattern
Layer 4's public API grew without breaking changes:
- Existing: `compute(GaitParams, t) → CartesianPositions`
- Added: `get_terrain_estimate() → TerrainEstimate`

---

## Lessons Learned

### 1. **Vocabulary Matching is Critical**

Each layer has a vocabulary. Interfaces must use the **receiving layer's vocabulary**, not the sending layer's.

**Bad**: Layer 4 sends `position_error: np.ndarray(4,3)` to Layer 5
**Good**: Layer 4 sends `roughness: float` to Layer 5

### 2. **Abstraction Happens in the Sending Layer**

The layer that produces data is responsible for abstracting it to the receiving layer's level.

Layer 4 doesn't just "pass through" arrays - it **processes and abstracts** them into Layer 5's vocabulary.

### 3. **Learning Requires State, But Placement Matters**

The tension: Learning needs state (error history, model params), but sensor data is in instant layers.

**Resolution**: Layer 4 becomes **partially stateful** for learning, but maintains instant API for trajectory generation.

### 4. **Public API ≠ Internal Implementation**

Layer 4 internally works with arrays, error histories, HNN parameters. But its **public API** to Layer 5 returns only high-level scalars.

**Internal freedom**: Use any complexity needed
**External simplicity**: Expose only what's appropriate

### 5. **Boundary Enforcement Works**

Layer 5 team's rejection was **correct and valuable**. Boundary violations would have:
- Broken Layer 5's identity
- Created tight coupling across layers
- Made the system harder to test and evolve

Strong boundaries force good architecture.

---

## Timeline

1. **Initial proposal**: FEP with observation chain (conceptually correct)
2. **Implementation error**: Low-level arrays passed to Layer 5 (wrong abstraction)
3. **Layer 5 rejection**: Identified 6 boundary violations (correct enforcement)
4. **Philosophy debate**: "Where does learning live?" (raised right question)
5. **Key insight**: Abstraction level, not data flow, was the problem
6. **Resolution**: Layer 4 abstracts arrays → scalars before Layer 5 sees them
7. **New issues**: Corrected architecture preserving all boundaries

---

## References

- **Architectural decisions**: [layering_discipline.md](layering_discipline.md), [stateless_vs_learning.md](stateless_vs_learning.md)
- **Naming conventions**: [naming_conventions.md](naming_conventions.md)
- **HNN placement**: [hnn_placement.md](hnn_placement.md)
- **Implementation specs**: [fix-requests/](fix-requests/)

---

**Key Takeaway**: The 8-layer architecture is sound. FEP implementation works within existing boundaries when **abstraction levels are respected**. Each layer N must translate data into Layer N+1's vocabulary before passing it upward.

---

**Last updated**: 2026-02-10
**Status**: Architecture resolved, issues filed, ready for implementation
