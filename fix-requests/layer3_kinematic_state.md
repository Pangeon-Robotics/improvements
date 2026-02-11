# Fix Request: Layer 3 - Add KinematicState Observation Publisher

**From**: Root architecture (FEP implementation)
**To**: Layer 3 (Inverse Kinematics)
**Date**: 2026-02-10
**Issue Type**: Feature Request (new capability)
**Priority**: High (foundational for FEP)

---

## Summary

Layer 3 needs to publish upward observations (`KinematicState`) for Layer 4 to use in dynamics prediction. Currently, Layer 3 only publishes downward commands (`LowCmd`). FEP requires a bidirectional observation chain where each layer passes processed state upward.

---

## Problem

**Current behavior**:
- Layer 3 subscribes to `rt/lowstate` (from Layer 2)
- Layer 3 publishes `rt/lowcmd` (to Layer 2)
- Layer 3 does NOT publish observations upward to Layer 4

**Why this is a problem**:
- Layer 4 cannot access Layer 3's processed state (violates N→N-1 discipline)
- Layer 4 cannot implement dynamics prediction without kinematic observations
- FEP observation chain is broken (no upward flow from L3→L4)

**Contract violation**:
- Architecture requires observation chain: L2→L3→L4→L5
- Layer 3 is not fulfilling its upward interface obligation

---

## Requested Solution

### 1. Define `KinematicState` Dataclass

**File**: `layer_3/observations.py` (NEW)

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class KinematicState:
    """Upward observations to Layer 4 (body kinematics in world frame)."""

    # Foot state (world frame)
    foot_positions_world: np.ndarray   # (4, 3) meters
    foot_velocities_world: np.ndarray  # (4, 3) m/s
    foot_contact_states: np.ndarray    # (4,) boolean
    foot_contact_forces: np.ndarray    # (4, 3) Newtons, world frame

    # Base state
    base_orientation: np.ndarray       # (4,) quaternion (w, x, y, z)
    base_angular_velocity: np.ndarray  # (3,) rad/s

    # Metadata
    timestamp: float
```

### 2. Implement Observation Publisher

**File**: `layer_3/observation_publisher.py` (NEW)

```python
from cyclonedds import Publisher
from observations import KinematicState

class ObservationPublisher:
    """Transforms LowState → KinematicState (upward observations)."""

    def __init__(self):
        self.fk_solver = ForwardKinematics()  # Use existing or implement
        self.pub = Publisher("rt/l3/kinematic_state", KinematicState)

    def publish(self, lowstate: LowState):
        """
        Process raw sensors into world-frame kinematics.

        Steps:
        1. Forward kinematics: joint angles → foot positions (world frame)
        2. Jacobian computation: joint velocities → foot velocities
        3. Frame transformation: sensor forces → world forces (using IMU)
        4. Contact detection: force threshold → binary contact state
        """
        # 1. Forward kinematics
        q = extract_joint_positions(lowstate)  # (12,) radians
        foot_pos_world = self.fk_solver.compute_foot_positions(q)

        # 2. Foot velocities via Jacobian
        dq = extract_joint_velocities(lowstate)  # (12,) rad/s
        foot_vel_world = self.fk_solver.compute_foot_velocities(q, dq)

        # 3. Frame transform: sensor → world using IMU
        R_world_base = quat_to_rotation(lowstate.imu_state.quaternion)
        forces_sensor = extract_foot_forces(lowstate)  # (4, 3) in sensor frame
        forces_world = np.array([R_world_base @ f for f in forces_sensor])

        # 4. Contact detection
        contact_threshold = 50.0  # Newtons
        contacts = np.array([np.linalg.norm(f) > contact_threshold
                            for f in forces_world])

        # Package and publish
        kin_state = KinematicState(
            foot_positions_world=foot_pos_world,
            foot_velocities_world=foot_vel_world,
            foot_contact_states=contacts,
            foot_contact_forces=forces_world,
            base_orientation=lowstate.imu_state.quaternion,
            base_angular_velocity=lowstate.imu_state.gyroscope,
            timestamp=lowstate.timestamp
        )

        self.pub.write(kin_state)
```

### 3. Integrate with Control Loop

**File**: `layer_3/controller.py` (MODIFY)

```python
class Controller:
    def __init__(self):
        # Existing
        self.lowstate_sub = Subscriber("rt/lowstate", LowState)
        self.lowcmd_pub = Publisher("rt/lowcmd", LowCmd)

        # NEW: Observation publisher
        self.obs_publisher = ObservationPublisher()

    def on_lowstate(self, lowstate: LowState):
        # Existing control logic
        obs = self.build_observation(lowstate)
        action = self.policy.infer(obs)
        cmd = action_to_cmd(action)
        self.lowcmd_pub.write(cmd)

        # NEW: Publish upward observations
        self.obs_publisher.publish(lowstate)
```

---

## Acceptance Criteria

- [ ] `observations.py` file created with `KinematicState` dataclass
- [ ] `observation_publisher.py` file created with FK and frame transforms
- [ ] Forward kinematics implemented or integrated (if not already available)
- [ ] Jacobian computation for foot velocities
- [ ] Frame transformation using IMU quaternion
- [ ] Contact detection from force sensors
- [ ] DDS topic `rt/l3/kinematic_state` publishes at 100 Hz
- [ ] Integration with `controller.py` control loop
- [ ] Unit tests: FK correctness, frame transforms, contact detection
- [ ] Integration tests: Verify DDS topic publishes expected data
- [ ] Documentation: Update ARCHITECTURE.md with new upward interface

---

## Testing Requirements

### Unit Tests
```python
def test_forward_kinematics():
    """FK should convert joint angles to foot positions."""
    q = np.array([...])  # Known joint configuration
    fk = ForwardKinematics()
    foot_pos = fk.compute_foot_positions(q)
    assert foot_pos.shape == (4, 3)
    np.testing.assert_allclose(foot_pos[0], expected_FR_position, atol=0.01)

def test_frame_transform():
    """Forces should transform from sensor to world frame."""
    force_sensor = np.array([0, 0, -200])  # Downward in sensor frame
    quat = np.array([1, 0, 0, 0])  # No rotation
    R = quat_to_rotation(quat)
    force_world = R @ force_sensor
    np.testing.assert_allclose(force_world, [0, 0, -200], atol=0.01)

def test_contact_detection():
    """Contact state should be true when force exceeds threshold."""
    forces = np.array([[0, 0, 100], [0, 0, 10], [0, 0, 200], [0, 0, 5]])
    threshold = 50.0
    contacts = detect_contacts(forces, threshold)
    np.testing.assert_array_equal(contacts, [True, False, True, False])
```

### Integration Test
```python
def test_observation_publisher_integration(firmware):
    """KinematicState should be published at 100 Hz."""
    from dds import dds_init, discover_firmware
    from observation_publisher import ObservationPublisher

    session = discover_firmware("b2")
    dds_init(session["domain_id"], session["interface"])

    # Subscribe to observations
    obs_list = []
    sub = Subscriber("rt/l3/kinematic_state", KinematicState)
    sub.init(lambda obs: obs_list.append(obs), 10)

    # Run for 1 second
    time.sleep(1.0)

    # Should receive ~100 observations
    assert 90 <= len(obs_list) <= 110, f"Got {len(obs_list)} observations"

    # Check data validity
    obs = obs_list[0]
    assert obs.foot_positions_world.shape == (4, 3)
    assert obs.foot_velocities_world.shape == (4, 3)
    assert obs.foot_contact_states.shape == (4,)
    assert len(obs.base_orientation) == 4
```

---

## Dependencies

### Internal
- Forward kinematics solver (may need to implement if not available)
- Existing `controller.py` control loop
- DDS infrastructure (`dds.py`)

### External
- Layer 2: `rt/lowstate` topic (already available)
- Layer 4: Will subscribe to `rt/l3/kinematic_state` (not blocking this work)

---

## Effort Estimate

**2-3 weeks**

### Week 1: Core Implementation
- Day 1-2: Implement/verify forward kinematics
- Day 3: Implement Jacobian for velocities
- Day 4: Frame transformations and contact detection
- Day 5: Unit tests

### Week 2: Integration
- Day 1-2: Create `ObservationPublisher` class
- Day 3: Integrate with `controller.py`
- Day 4: DDS topic setup and testing
- Day 5: Integration tests

### Week 3: Validation
- Day 1-2: End-to-end testing with firmware
- Day 3: Performance verification (100 Hz publication)
- Day 4: Documentation updates
- Day 5: Code review and cleanup

---

## References

- **Architecture**: `/Users/graham/code/robotics/improvements/layering_discipline.md`
- **Naming conventions**: `/Users/graham/code/robotics/improvements/naming_conventions.md`
- **Implementation summary**: `/Users/graham/code/robotics/improvements/implementation_summary.md`
- **Test script**: `/Users/graham/code/robotics/test_observation_chain.py` (demonstrates the pattern)

---

## Notes

- This is **foundational** for FEP implementation - Layer 4 and Layer 5 depend on this
- Maintains architectural discipline: N→N-1 only (Layer 3 never skips to Layer 4)
- Observation publishing should have minimal overhead (<1ms per call)
- FK computation may already exist in IK solver - reuse if available
- Contact threshold (50N) is a starting value, may need tuning

---

**Status**: Open
**Assigned to**: Layer 3 team
**Blocks**: Layer 4 implementation
**Version target**: Layer 3 v2.2.0
