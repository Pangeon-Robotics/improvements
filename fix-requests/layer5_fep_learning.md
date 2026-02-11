# Fix Request: Layer 5 - Add Terrain-Aware Gait Selection

**From**: Root architecture (FEP implementation)
**To**: Layer 5 (Locomotion)
**Date**: 2026-02-10 (Revised)
**Issue Type**: Feature Request (new capability)
**Priority**: Medium (enhances gait selection)

---

## Summary

Layer 5 needs to add terrain-aware gait selection by using `TerrainEstimate` from Layer 4's public API. This enables adaptive locomotion where gait type, step height, and other parameters adjust based on terrain properties (roughness, compliance, stability).

**Key architectural principle**: Layer 5 receives only high-level terrain abstractions (scalars) from Layer 4, not raw sensor data. NO HNN learning, NO array processing - just gait selection logic.

---

## Problem

**Current behavior**:
- Layer 5 has basic gait selection (speed-based)
- No terrain awareness
- Fixed step heights and duty cycles per gait type
- Cannot adapt to rough/smooth, hard/soft terrain

**Why this is needed**:
- Rough terrain requires higher steps and slower gaits (walk vs trot)
- Soft terrain requires different duty cycles (more stance time)
- Unstable gaits should trigger fallback to safer options
- Enables robust locomotion across varied environments

**What Layer 5 should NOT do** (per boundaries.md):
- ❌ Read sensors directly (joint angles, IMU, contact forces)
- ❌ Store sensor history or foot positions
- ❌ Do HNN learning or gradient descent
- ❌ Process arrays (4×3 position/force errors)
- ❌ Use CycloneDDS (Layer 5 is an in-process library)

---

## Requested Solution

### 1. Import TerrainEstimate from Layer 4 API

**File**: `layer_5/locomotion.py` (MODIFY)

```python
# Layer 4 public API (allowed import)
from layer_4.api import get_terrain_estimate, TerrainEstimate

# TerrainEstimate contains:
#   roughness: float (0-1)
#   compliance: float (0-1)
#   stability: float (0-1)
#   surprise_level: float (0-1)
#   confidence: float (0-1)
```

---

### 2. Implement Terrain-Aware Gait Selector

**File**: `layer_5/terrain_aware_selector.py` (NEW)

```python
from dataclasses import dataclass
from layer_4.api import get_terrain_estimate, TerrainEstimate

@dataclass
class AdaptiveGaitParams:
    """Gait parameters adjusted for terrain."""
    gait_type: str        # 'walk', 'trot', 'bound'
    step_height: float    # m (adjusted for roughness)
    duty_cycle: float     # (adjusted for compliance)
    confidence: float     # How confident are we in this choice?


class TerrainAwareSelector:
    """
    Select gait based on velocity command AND terrain properties.

    Uses ONLY high-level terrain info from Layer 4 (scalars).
    """

    def select_gait(self,
                    vx: float,  # m/s forward velocity command
                    vy: float,  # m/s lateral velocity
                    wz: float,  # rad/s yaw rate
                    terrain: TerrainEstimate) -> AdaptiveGaitParams:
        """
        Choose gait and parameters based on velocity + terrain.

        Args:
            vx, vy, wz: Motion command from Layer 6
            terrain: High-level terrain properties from Layer 4

        Returns:
            AdaptiveGaitParams with terrain-adjusted parameters
        """
        # Base gait selection (speed-based)
        if abs(vx) < 0.1 and abs(vy) < 0.1 and abs(wz) < 0.1:
            base_gait = 'stand'
            base_step_height = 0.0
        elif abs(vx) < 0.5:
            base_gait = 'walk'
            base_step_height = 0.06
        else:
            base_gait = 'trot'
            base_step_height = 0.05

        # Terrain adjustments
        adjusted_gait, adjusted_height, adjusted_duty = self._adjust_for_terrain(
            base_gait=base_gait,
            base_step_height=base_step_height,
            terrain=terrain,
            velocity=vx
        )

        return AdaptiveGaitParams(
            gait_type=adjusted_gait,
            step_height=adjusted_height,
            duty_cycle=adjusted_duty,
            confidence=terrain.confidence
        )

    def _adjust_for_terrain(self,
                           base_gait: str,
                           base_step_height: float,
                           terrain: TerrainEstimate,
                           velocity: float) -> tuple:
        """
        Adjust gait parameters based on terrain properties.

        High-level rules using ONLY scalar properties:
        - Rough terrain → higher steps, slower gait
        - Soft terrain → longer stance (higher duty cycle)
        - Low stability → fallback to safer gait
        """
        gait = base_gait
        step_height = base_step_height
        duty_cycle = 0.5  # Default for trot

        # Rule 1: Rough terrain needs higher steps
        if terrain.roughness > 0.7:
            step_height += 0.02  # Raise steps by 2cm
            if gait == 'trot' and velocity < 1.5:
                gait = 'walk'  # Slow down on very rough terrain

        # Rule 2: Soft/compliant terrain needs longer stance
        if terrain.compliance > 0.6:
            if gait == 'trot':
                duty_cycle = 0.6  # More stance time (vs 0.5 default)
            elif gait == 'walk':
                duty_cycle = 0.8  # Even more stance (vs 0.75 default)

        # Rule 3: Low stability triggers safe fallback
        if terrain.stability < 0.4:
            if gait == 'trot':
                gait = 'walk'  # Fallback to safer gait
                duty_cycle = 0.75
            step_height = min(step_height + 0.01, 0.08)  # Lift higher (cautious)

        # Rule 4: High surprise → conservative parameters
        if terrain.surprise_level > 0.7:
            step_height = min(step_height + 0.015, 0.08)
            if gait == 'trot':
                duty_cycle = min(duty_cycle + 0.1, 0.7)  # More contact time

        # Get default duty cycle for selected gait
        from gait import get_default_duty_cycle
        if duty_cycle == 0.5:  # Not adjusted above
            duty_cycle = get_default_duty_cycle(gait)

        return gait, step_height, duty_cycle
```

---

### 3. Integrate with Main Locomotion Controller

**File**: `layer_5/locomotion.py` (MODIFY)

```python
from terrain_aware_selector import TerrainAwareSelector
from layer_4.api import get_terrain_estimate, GaitParams

class LocomotionController:
    """Main Layer 5 controller with terrain awareness."""

    def __init__(self):
        self.selector = TerrainAwareSelector()

        # State (Layer 5 vocabulary - no sensors!)
        self.current_gait = 'stand'
        self.ramp_progress = 0.0
        self.transition_phase = 0.0

    def update(self, motion_cmd, dt: float) -> GaitParams:
        """
        Convert motion command → gait parameters.

        Args:
            motion_cmd: MotionCommand from Layer 6 (vx, vy, wz)
            dt: Timestep (typically 0.01s for 100 Hz)

        Returns:
            GaitParams for Layer 4
        """
        # Get terrain info from Layer 4 (public API call)
        terrain = get_terrain_estimate()

        # Select gait using terrain info (scalars only)
        adaptive_params = self.selector.select_gait(
            vx=motion_cmd.vx,
            vy=motion_cmd.vy,
            wz=motion_cmd.wz,
            terrain=terrain
        )

        # Apply transitions/ramping (existing Layer 5 logic)
        smooth_params = self._apply_transitions(
            target_gait=adaptive_params.gait_type,
            target_step_height=adaptive_params.step_height,
            dt=dt
        )

        # Convert to Layer 4 GaitParams
        return GaitParams(
            gait_type=smooth_params.gait_type,
            vx=motion_cmd.vx,
            vy=motion_cmd.vy,
            wz=motion_cmd.wz,
            step_height=smooth_params.step_height,
            step_length=self._velocity_to_step_length(motion_cmd.vx),
            gait_freq=self._velocity_to_frequency(motion_cmd.vx),
            duty_cycle=adaptive_params.duty_cycle
        )

    def _apply_transitions(self, target_gait, target_step_height, dt):
        """Ramp gait parameters smoothly (existing Layer 5 logic)."""
        # ... existing transition code ...
        pass
```

---

## Acceptance Criteria

- [ ] `terrain_aware_selector.py` with `TerrainAwareSelector` class
- [ ] Uses ONLY `TerrainEstimate` scalars (roughness, compliance, stability)
- [ ] NO sensor access (no joint angles, IMU, forces)
- [ ] NO array processing (no 4×3 position/force arrays)
- [ ] NO HNN or learning logic
- [ ] NO CycloneDDS subscriptions (Layer 5 is in-process library)
- [ ] Integrated with `locomotion.py` main controller
- [ ] Terrain-based gait selection rules (rough→walk, soft→high duty, unstable→fallback)
- [ ] Unit tests: Gait selection logic with mock terrain
- [ ] Integration tests: Full pipeline with Layer 4 API
- [ ] Documentation: Update ARCHITECTURE.md with terrain-aware logic

---

## Testing Requirements

### Unit Tests

```python
def test_rough_terrain_increases_step_height():
    """Rough terrain should increase step height."""
    selector = TerrainAwareSelector()

    terrain = TerrainEstimate(
        roughness=0.8,  # Very rough
        compliance=0.5,
        stability=0.6,
        surprise_level=0.4,
        confidence=0.9,
        gait_phase=0.0,
        timestamp=0.0
    )

    params = selector.select_gait(vx=1.0, vy=0.0, wz=0.0, terrain=terrain)

    # Should increase step height for rough terrain
    assert params.step_height > 0.06  # Higher than default


def test_low_stability_triggers_walk():
    """Low stability should fallback to walk gait."""
    selector = TerrainAwareSelector()

    terrain = TerrainEstimate(
        roughness=0.5,
        compliance=0.5,
        stability=0.3,  # Very unstable
        surprise_level=0.6,
        confidence=0.8,
        gait_phase=0.0,
        timestamp=0.0
    )

    # Even at trot speed, should fallback to walk
    params = selector.select_gait(vx=1.5, vy=0.0, wz=0.0, terrain=terrain)
    assert params.gait_type == 'walk'


def test_soft_terrain_increases_duty_cycle():
    """Soft terrain should increase duty cycle (more stance time)."""
    selector = TerrainAwareSelector()

    terrain = TerrainEstimate(
        roughness=0.4,
        compliance=0.7,  # Very soft
        stability=0.7,
        surprise_level=0.3,
        confidence=0.9,
        gait_phase=0.0,
        timestamp=0.0
    )

    params = selector.select_gait(vx=1.0, vy=0.0, wz=0.0, terrain=terrain)

    # Duty cycle should be higher than default 0.5
    assert params.duty_cycle > 0.5
```

### Integration Test

```python
def test_full_pipeline_layer4_to_layer5():
    """Layer 5 should receive terrain info from Layer 4 API."""
    from layer_4.api import get_terrain_estimate
    from layer_5.locomotion import LocomotionController

    controller = LocomotionController()

    # Layer 4 should provide terrain estimate
    terrain = get_terrain_estimate()
    assert hasattr(terrain, 'roughness')
    assert hasattr(terrain, 'compliance')
    assert 0 <= terrain.roughness <= 1

    # Layer 5 should use it for gait selection
    motion_cmd = MotionCommand(vx=1.0, vy=0.0, wz=0.0, body_height=0.465)
    gait_params = controller.update(motion_cmd, dt=0.01)

    assert hasattr(gait_params, 'gait_type')
    assert hasattr(gait_params, 'step_height')
```

---

## Dependencies

### Internal
- Existing `locomotion.py` (main controller)
- Existing `gait_selector.py` (speed-based logic)
- Existing transition/ramping logic

### External
- **Layer 4 API**: `get_terrain_estimate()` must be implemented (BLOCKS this work)
- **No new dependencies**: numpy only (as per Layer 5 constraints)

---

## Effort Estimate

**2-3 weeks**

### Week 1: Terrain-Aware Selector
- Implement `TerrainAwareSelector` class
- Terrain-based gait selection rules
- Parameter adjustment logic
- Unit tests with mock terrain

### Week 2: Integration
- Integrate with `locomotion.py`
- Call Layer 4 API (`get_terrain_estimate()`)
- Smooth parameter transitions
- Integration tests

### Week 3: Validation & Polish
- Test with Layer 4 on varied terrain
- Tune terrain thresholds
- Documentation updates
- Code review

---

## Architecture Notes

**Key change from original proposal**: Layer 5 does NO learning, NO sensor processing, NO array manipulation. It ONLY receives high-level terrain scalars from Layer 4 and uses them for gait selection.

**Layer 5 remains within boundaries**:
- ✅ "Motion commands → gait parameters" (core function unchanged)
- ✅ No sensor access (uses abstracted terrain properties)
- ✅ No forbidden state (no sensor history, foot positions, joint angles)
- ✅ numpy-only (no CycloneDDS, no JAX/Flax)
- ✅ Stateful for transitions only (ramp progress, current gait)

**Upward interface**: Layer 5 calls `get_terrain_estimate()` from Layer 4's public API. This returns scalars, never arrays.

**Layering discipline preserved**: Layer 5 only accesses Layer 4 via public API. No cross-layer imports, no internal Layer 4 modules.

---

**Status**: Open (BLOCKED by Layer 4 API enrichment)
**Assigned to**: Layer 5 team
**Depends on**: Layer 4 `get_terrain_estimate()` implementation
**Version target**: Layer 5 v0.7.0
