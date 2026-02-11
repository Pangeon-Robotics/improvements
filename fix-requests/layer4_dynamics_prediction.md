# Fix Request: Layer 4 - Add Terrain-Aware Dynamics with HNN

**From**: Root architecture (FEP implementation)
**To**: Layer 4 (Cartesian Positions)
**Date**: 2026-02-10 (Revised)
**Issue Type**: Feature Request (new capability)
**Priority**: High (enables adaptive locomotion)

---

## Summary

Layer 4 needs to add terrain-aware dynamics prediction using a Hamiltonian Neural Network (HNN). This involves:
1. Predicting foot trajectories and contact forces
2. Comparing predictions to actual observations from Layer 3
3. Abstracting prediction errors into high-level terrain properties
4. Publishing `TerrainEstimate` to Layer 5 for gait selection

**Key architectural principle**: Layer 4 maintains HNN state and does all learning. Layer 5 receives only high-level terrain abstractions (scalars), never raw sensor arrays.

---

## Problem

**Current behavior**:
- Layer 4 has `compute(GaitParams, t) → CartesianPositions` (trajectory generation)
- Layer 4 is stateless (pure function)
- No dynamics prediction or terrain awareness
- No upward observations published to Layer 5

**Why this is needed**:
- Terrain-adaptive gait selection requires terrain classification
- Layer 5 needs high-level terrain info to choose appropriate gaits
- FEP requires prediction errors to enable learning
- Observation chain is incomplete (L4 doesn't publish to L5)

---

## Requested Solution

### 1. Enrich Layer 4 Public API

**New method**: `get_terrain_estimate() → TerrainEstimate`

**File**: `layer_4/api.py` (MODIFY)

```python
@dataclass
class TerrainEstimate:
    """High-level terrain properties for Layer 5 (abstracted from prediction errors)."""

    # Scalar terrain properties (Layer 5 vocabulary)
    roughness: float        # 0-1 (from position error variance)
    compliance: float       # 0-1 (soft/hard - from force errors)
    stability: float        # 0-1 (gait working well vs poorly)
    surprise_level: float   # 0-1 (overall prediction error magnitude)

    # Confidence
    confidence: float       # 0-1 (data quality, sample size)

    # Metadata
    gait_phase: float       # [0, 1) current phase in gait cycle
    timestamp: float


def get_terrain_estimate() -> TerrainEstimate:
    """
    Get current terrain classification based on prediction errors.

    Returns high-level terrain properties abstracted from internal
    HNN prediction errors. Layer 5 uses these for gait selection.

    Returns:
        TerrainEstimate with scalar terrain properties
    """
    pass
```

**Existing API** (unchanged):
```python
def compute(params: GaitParams, t: float) -> CartesianPositions:
    """Generate foot/base positions for given gait parameters."""
    pass
```

---

### 2. Define Internal Observation Dataclass

**File**: `layer_4/observations.py` (NEW - internal only, not in public API)

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class KinematicState:
    """Upward observations from Layer 3 (internal to Layer 4)."""

    # Foot state (world frame)
    foot_positions_world: np.ndarray   # (4, 3) meters
    foot_velocities_world: np.ndarray  # (4, 3) m/s
    foot_contact_states: np.ndarray    # (4,) boolean
    foot_contact_forces: np.ndarray    # (4, 3) Newtons

    # Base state
    base_orientation: np.ndarray       # (4,) quaternion
    base_angular_velocity: np.ndarray  # (3,) rad/s

    timestamp: float
```

**Note**: This is Layer 4's internal representation of Layer 3 data. It never leaves Layer 4.

---

### 3. Implement Hamiltonian Neural Network

**File**: `layer_4/models/hamiltonian_nn.py` (NEW)

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class HamiltonianNN(nn.Module):
    """Learns H(q, p) for physics-based trajectory prediction."""

    hidden_dim: int = 128

    @nn.compact
    def __call__(self, q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Hamiltonian energy H(q, p).

        Args:
            q: Generalized positions (4 feet × 3D = 12D)
            p: Generalized momenta (4 feet × 3D = 12D)

        Returns:
            H: Scalar energy value
        """
        qp = jnp.concatenate([q, p], axis=-1)  # (24,)

        # MLP
        x = nn.Dense(self.hidden_dim)(qp)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)  # Scalar output

        return x

    def predict_dynamics(self, q: jnp.ndarray, p: jnp.ndarray) -> tuple:
        """
        Compute (q̇, ṗ) via Hamilton's equations using autodiff.

        ∂H/∂p = q̇  (velocity from momentum)
        -∂H/∂q = ṗ (force from position)
        """
        H = lambda q, p: self(q[None, :], p[None, :])[0, 0]

        q_dot = jax.grad(H, argnums=1)(q, p)   # ∂H/∂p
        p_dot = -jax.grad(H, argnums=0)(q, p)  # -∂H/∂q

        return q_dot, p_dot

    def integrate_step(self, q: jnp.ndarray, p: jnp.ndarray,
                      dt: float) -> tuple:
        """
        Symplectic Euler integration (preserves Hamiltonian structure).

        Args:
            q: Current positions (12,)
            p: Current momenta (12,)
            dt: Timestep (0.01s for 100 Hz)

        Returns:
            q_next, p_next: Predicted next state
        """
        q_dot, p_dot = self.predict_dynamics(q, p)

        # Symplectic Euler: update p first, then q
        p_new = p + dt * p_dot
        q_new = q + dt * q_dot

        return q_new, p_new
```

---

### 4. Implement Terrain Analyzer

**File**: `layer_4/terrain_analyzer.py` (NEW)

```python
from collections import deque
import numpy as np
from observations import KinematicState
from api import TerrainEstimate

class TerrainAnalyzer:
    """
    Abstracts prediction errors into high-level terrain properties.

    This is where Layer 4 converts low-level arrays (4×3 position/force errors)
    into high-level scalars (roughness, compliance) for Layer 5.
    """

    def __init__(self, window_size: int = 100):
        self.error_history = deque(maxlen=window_size)
        self.force_history = deque(maxlen=window_size)

    def analyze(self,
                predicted_positions: np.ndarray,  # (4, 3)
                predicted_forces: np.ndarray,     # (4, 3)
                actual: KinematicState,
                gait_phase: float) -> TerrainEstimate:
        """
        Convert prediction errors into terrain properties.

        Args:
            predicted_positions: What HNN predicted
            predicted_forces: What contact model predicted
            actual: What Layer 3 observed (from KinematicState)
            gait_phase: Current phase in gait cycle

        Returns:
            TerrainEstimate with abstracted terrain properties
        """
        # Compute raw errors (INTERNAL - never exposed to Layer 5)
        pos_error = actual.foot_positions_world - predicted_positions
        force_error = actual.foot_contact_forces - predicted_forces

        # Store in history
        self.error_history.append(pos_error)
        self.force_history.append(force_error)

        # Abstract into scalar properties
        roughness = self._compute_roughness()
        compliance = self._compute_compliance()
        stability = self._compute_stability()
        surprise = self._compute_surprise()
        confidence = self._compute_confidence()

        return TerrainEstimate(
            roughness=roughness,
            compliance=compliance,
            stability=stability,
            surprise_level=surprise,
            confidence=confidence,
            gait_phase=gait_phase,
            timestamp=actual.timestamp
        )

    def _compute_roughness(self) -> float:
        """Terrain roughness from position error variance."""
        if len(self.error_history) < 10:
            return 0.5  # Unknown, assume medium

        errors = np.array(self.error_history)
        variance = np.var(errors[:, :, 2])  # Z-axis variance

        # Normalize: 0.001m std = smooth, 0.05m std = very rough
        roughness = np.clip(variance / 0.0025, 0, 1)
        return float(roughness)

    def _compute_compliance(self) -> float:
        """Terrain compliance (soft/hard) from force errors."""
        if len(self.force_history) < 10:
            return 0.5

        errors = np.array(self.force_history)
        force_variance = np.var(errors[:, :, 2])  # Z-axis force

        # High variance = soft/unpredictable, low = hard/predictable
        compliance = np.clip(force_variance / 1000, 0, 1)
        return float(compliance)

    def _compute_stability(self) -> float:
        """Gait stability (how well predictions match reality)."""
        if len(self.error_history) < 10:
            return 0.5

        errors = np.array(self.error_history)
        mean_error = np.mean(np.abs(errors))

        # Low error = high stability
        stability = 1.0 - np.clip(mean_error / 0.1, 0, 1)
        return float(stability)

    def _compute_surprise(self) -> float:
        """Overall prediction error magnitude."""
        if len(self.error_history) < 10:
            return 0.5

        errors = np.array(self.error_history[-10:])  # Recent window
        surprise = np.mean(np.linalg.norm(errors, axis=-1))

        # Normalize: 0.05m = high surprise
        surprise_normalized = np.clip(surprise / 0.05, 0, 1)
        return float(surprise_normalized)

    def _compute_confidence(self) -> float:
        """Confidence in terrain estimate (based on sample size)."""
        samples = len(self.error_history)
        confidence = np.clip(samples / 100, 0, 1)  # Full confidence at 100 samples
        return float(confidence)
```

---

### 5. Implement Dynamics Predictor with Learning

**File**: `layer_4/predictor.py` (NEW)

```python
from cyclonedds import Subscriber
import numpy as np
from observations import KinematicState
from models.hamiltonian_nn import HamiltonianNN
from terrain_analyzer import TerrainAnalyzer
import jax
import optax

class DynamicsPredictor:
    """
    Stateful predictor with online HNN learning.

    Layer 4 becomes partially stateful to support learning:
    - Maintains HNN parameters
    - Stores error history (for terrain analysis)
    - Updates model via gradient descent
    """

    def __init__(self):
        # Subscribe to Layer 3 observations
        self.kin_sub = Subscriber("rt/l3/kinematic_state", KinematicState)
        self.kin_sub.init(self._on_kinematics, 10)

        # HNN model (stateful!)
        self.hnn = HamiltonianNN()
        self.hnn_params = self._load_pretrained_hnn()

        # Optimizer for online learning
        self.optimizer = optax.adam(learning_rate=0.0001)
        self.opt_state = self.optimizer.init(self.hnn_params)

        # Terrain analyzer
        self.terrain_analyzer = TerrainAnalyzer(window_size=100)

        # State
        self.latest_kin = None
        self.current_terrain_estimate = None

    def _on_kinematics(self, kin: KinematicState):
        """Receive kinematic observations from Layer 3."""
        self.latest_kin = kin

    def predict_and_learn(self, gait_params, t: float):
        """
        Predict next state, compare to actual, update HNN.

        This is called internally at 100 Hz by Layer 4's control loop.
        """
        if self.latest_kin is None:
            return

        # Extract phase space state
        q = self.latest_kin.foot_positions_world.flatten()  # (12,)
        v = self.latest_kin.foot_velocities_world.flatten()  # (12,)
        p = v  # Simple: assume unit mass, p = v

        # Predict next state using HNN
        dt = 0.01  # 100 Hz
        q_pred, p_pred = self.hnn.integrate_step(q, p, dt)

        # Reshape predictions
        foot_pos_pred = q_pred.reshape(4, 3)

        # Predict contact forces
        force_pred = self._predict_contact_forces(
            foot_pos_pred,
            self.latest_kin.foot_contact_states
        )

        # Analyze terrain from errors (abstraction happens here!)
        self.current_terrain_estimate = self.terrain_analyzer.analyze(
            predicted_positions=foot_pos_pred,
            predicted_forces=force_pred,
            actual=self.latest_kin,
            gait_phase=(t * gait_params.gait_freq) % 1.0
        )

        # Online learning: update HNN from prediction errors
        self._update_hnn(q, p, q_pred)

    def get_terrain_estimate(self):
        """
        Public API method: Layer 5 calls this.

        Returns high-level terrain abstraction (scalars only).
        """
        if self.current_terrain_estimate is None:
            # Return default if no data yet
            return TerrainEstimate(
                roughness=0.5, compliance=0.5, stability=0.5,
                surprise_level=0.5, confidence=0.0,
                gait_phase=0.0, timestamp=0.0
            )
        return self.current_terrain_estimate

    def _predict_contact_forces(self, positions, contact_states):
        """Simple terrain model: F = k * penetration."""
        forces = np.zeros((4, 3))
        stiffness = 1e6  # Default terrain stiffness

        for i, (pos, contact) in enumerate(zip(positions, contact_states)):
            if contact:
                penetration = max(0, -pos[2])  # How far into ground
                forces[i, 2] = stiffness * penetration
        return forces

    def _update_hnn(self, q, p, q_next_actual):
        """Online learning: gradient descent on HNN parameters."""
        def loss_fn(params):
            q_pred, _ = self.hnn.apply(params, q, p, method='integrate_step', dt=0.01)
            return jax.numpy.mean((q_pred - q_next_actual)**2)

        loss, grads = jax.value_and_grad(loss_fn)(self.hnn_params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.hnn_params = optax.apply_updates(self.hnn_params, updates)

    def _load_pretrained_hnn(self):
        """Load pre-trained HNN weights."""
        # TODO: Load from file
        # For now, initialize randomly
        return self.hnn.init(jax.random.PRNGKey(0),
                            jax.numpy.zeros(12),
                            jax.numpy.zeros(12))
```

---

## Acceptance Criteria

- [ ] `observations.py` with `KinematicState` (internal only)
- [ ] `api.py` updated with `get_terrain_estimate() → TerrainEstimate`
- [ ] `models/hamiltonian_nn.py` with HNN implementation (JAX/Flax)
- [ ] HNN satisfies Hamilton's equations (∂H/∂p = q̇, -∂H/∂q = ṗ)
- [ ] `terrain_analyzer.py` abstracts errors into scalars
- [ ] `predictor.py` with online HNN learning
- [ ] DDS topic: Subscribe to `rt/l3/kinematic_state`
- [ ] Layer 5 can call `get_terrain_estimate()` (public API)
- [ ] Layer 4 internal state: HNN params, error history
- [ ] Unit tests: HNN energy conservation, terrain abstraction accuracy
- [ ] Integration tests: Full prediction pipeline with Layer 3
- [ ] Documentation: Update ARCHITECTURE.md and API.md

---

## Dependencies

### Internal
- Existing `generator.py` (trajectory generation)
- Existing `gait.py` (gait patterns)

### External
- **Layer 3**: Must implement `rt/l3/kinematic_state` publisher (BLOCKS this work)
- **JAX/Flax**: `pip install jax jaxlib flax optax`
- **Training data**: Collect 10k trajectories for pre-training HNN

---

## Effort Estimate

**8-10 weeks**

### Weeks 1-2: HNN Implementation
- Implement `HamiltonianNN` class
- Hamilton's equations via autodiff
- Symplectic integration
- Energy conservation tests

### Weeks 3-4: Data & Training
- Collect trajectories from simulation
- Pre-train HNN offline
- Validate prediction accuracy

### Weeks 5-6: Terrain Analysis
- Implement `TerrainAnalyzer`
- Error abstraction logic
- Scalar property computation
- Unit tests for abstraction

### Weeks 7-8: Integration
- Implement `DynamicsPredictor`
- Online learning loop
- DDS subscription to Layer 3
- Public API: `get_terrain_estimate()`

### Weeks 9-10: Testing & Validation
- Integration tests with Layer 3
- Verify terrain estimates on varied terrain
- Performance profiling (<5ms per cycle)
- Documentation

---

## Architecture Notes

**Key change from original proposal**: Layer 4 now does ALL learning and abstraction internally. Layer 5 receives only high-level terrain properties (scalars), never raw sensor arrays.

**Layer 4 statefulness**: Layer 4 becomes partially stateful to support online learning:
- Maintains HNN parameters (updated via gradient descent)
- Stores error history (100 samples for terrain analysis)
- This is acceptable because learning requires state

**Upward interface**: `get_terrain_estimate()` returns high-level scalars in Layer 5's vocabulary (roughness, compliance, stability), NOT low-level arrays (position_error, force_error).

**Layering discipline preserved**: Layer 4 only accesses Layer 3 via `rt/l3/kinematic_state`. Layer 5 only accesses Layer 4 via `get_terrain_estimate()` API call.

---

**Status**: Open (BLOCKED by Layer 3 KinematicState publisher)
**Assigned to**: Layer 4 team
**Depends on**: Layer 3 `KinematicState` implementation
**Version target**: Layer 4 v0.14.0
