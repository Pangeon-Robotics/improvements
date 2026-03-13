# Momentum-Aware Gait Controller — Design Document

> **Status**: Phases 1-2 implemented. Phase 3 (posture-based corrections) active as of Mar 2026.
>
> Key change in Phase 3: control variables switched from gait knobs
> (step_height, gait_freq, body_height) to body posture corrections
> (body_roll, body_pitch, body_x_offset, body_y_offset). Observation
> expanded from 21D to 29D (added leg_phases + posture_cmd). The same
> chaos control theory applies — speed-gradient and Pyragas now operate
> over posture dimensions instead of gait parameters.

## The Problem

We have a body-state predictor ensemble (K=5 MLPs, 29D→6D) that predicts
delta body state 200ms ahead: **[dvx, dvy, dwz, droll, dpitch, dyaw]**.
This IS a momentum predictor — it tells us where translational momentum
(dvx, dvy), angular momentum (dwz), and rotational energy (droll, dpitch)
will go.

The chaos control paper (Fradkov & Evans 2005) gives the framework:
- The speed-gradient method computes optimal corrections
- Pyragas delayed feedback stabilizes periodic orbits (the gait cycle)
- Key insight: "the more unstable the system, the simpler it is to control"
  — small posture tweaks can have outsized stabilizing effects

## What Changes

**Phase 1**: Passive attenuation — scalar instability → turn_factor → slow down

**Phase 2** (implemented): Active stabilization — 6D momentum prediction → corrective posture adjustments

The ensemble stays. The training pipeline stays. What changes is HOW we use
the predictions at runtime, and WHAT we feed into the model.

## Architecture

### Phase 2a — Close the Sensor Loop

**The single highest-value change.** L5's `_update_telemetry()` already
reads the full robot state every tick (roll, pitch, gyro_xyz, foot_contacts).
We just need to feed it into the model instead of zeros.

```
SimulationManager.send_motion_command()
  ├─ reads robot_state (IMU, foot_contacts)   ← already does this
  ├─ locomotion.update(cmd, dt, body_obs=...)  ← NEW: pass real state
  └─ _update_telemetry()
```

**New in L5:**
- `Locomotion.update()` accepts optional `body_obs: BodyObservation` —
  a lightweight struct with (vx, vy, wz, roll, pitch, yaw, gyro_xyz,
  foot_contacts)
- `_walk_pipeline()` constructs a real 29D observation from body_obs +
  current gait command (includes leg_phases and posture_cmd)
- Ensemble predicts with ACTUAL state → predictions reflect the robot
  tipping, sliding, or losing contact

**Fallback**: When body_obs is None (Phase 1 callers, unit tests), falls
back to synthetic observations exactly as today. Zero breaking changes.

**Runtime cost**: 5 forward passes × (29×128 + 128×128 + 128×6) = 93K
multiply-adds. <0.1ms on any CPU. Negligible at 100Hz.

### Phase 2b — Speed-Gradient Corrections

From Fradkov eq. 35-36: `u = -Ψ[∇_u Q̇(x,u)]`

In our system:
- **State x** = predicted body state at t+200ms (from ensemble)
- **Control u** = posture corrections: [Δbody_roll, Δbody_pitch, Δbody_x_offset, Δbody_y_offset]
- **Goal function** Q(x) = instability we want to minimize:
  `Q = w_roll·droll² + w_pitch·dpitch² + w_vy·dvy²`
  (dvy penalizes lateral momentum drift; dvx and dwz are desired)

**Algorithm** (runs in `_walk_pipeline` after gait parameter computation):

```python
# 1. Base prediction with current gait params
obs_base = build_observation(body_obs, gait_cmd)
pred_base = ensemble_mean(obs_base)
Q_base = w_r * pred_base[3]**2 + w_p * pred_base[4]**2 + w_vy * pred_base[1]**2

# 2. Finite-difference gradient for each control dimension
epsilon = [0.02, 0.02, 0.01, 0.01]  # body_roll(rad), body_pitch(rad), body_x_offset(m), body_y_offset(m)
dQ_du = []
for i, (param_name, eps) in enumerate(zip(controls, epsilon)):
    obs_plus = obs_base.copy()
    obs_plus[param_idx[i]] += eps
    pred_plus = ensemble_mean(obs_plus)
    Q_plus = w_r * pred_plus[3]**2 + w_p * pred_plus[4]**2 + w_vy * pred_plus[1]**2
    dQ_du.append((Q_plus - Q_base) / eps)

# 3. Speed-gradient correction
gamma = 0.5  # learning rate
corrections = [-gamma * g for g in dQ_du]

# 4. Clamp to safe ranges
delta_body_roll     = clamp(corrections[0], -0.10, +0.10)
delta_body_pitch    = clamp(corrections[1], -0.10, +0.10)
delta_body_x_offset = clamp(corrections[2], -0.05, +0.05)
delta_body_y_offset = clamp(corrections[3], -0.05, +0.05)

# 5. Apply to GaitParams (flows through L4 BodyPoseCommand → L3 IK)
params.body_roll     += delta_body_roll
params.body_pitch    += delta_body_pitch
params.body_x_offset += delta_body_x_offset
params.body_y_offset += delta_body_y_offset
```

**Key design decisions:**
- Corrections are ADDITIVE to the base gait params (from velocity_mapper),
  not replacements. The rule-based pipeline sets the operating point;
  the momentum controller nudges it.
- Clamp ranges are deliberately small. Per the chaos paper: "small
  perturbations can stabilize unstable orbits." Large corrections would
  destabilize more than they help.
- `gamma` is conservative. Better to under-correct (the next tick will
  correct more) than over-correct (oscillation).

**Runtime cost**: 4 perturbation directions × 5 ensemble members = 20
additional forward passes. ~0.20ms. Still negligible at 100Hz. Runs at
20Hz (every 5th tick) to reduce further.

### Phase 2c — Pyragas Delayed Feedback

From Fradkov eq. 30: `u(t) = K[x(t) - x(t-τ)]` where τ = orbit period.

A trotting gait IS a periodic orbit. Period τ = 1/gait_freq ≈ 0.33s at
3Hz. The Pyragas controller stabilizes this orbit without knowing its shape —
it just pushes the system toward repeating its own period.

```python
# Ring buffer of body states, indexed by gait phase
tau = 1.0 / gait_freq  # gait period in seconds
n_history = int(tau / dt)  # samples per period

# At each tick:
history_buffer.append(current_body_state)
if len(history_buffer) >= n_history:
    x_now = current_body_state     # [roll, pitch, vy, ...]
    x_prev = history_buffer[-n_history]  # one period ago

    # Correction proportional to deviation from periodic orbit
    K_pyragas = diag([0.1, 0.1, 0.05])  # per-dimension gains
    delta = K_pyragas @ (x_now - x_prev)

    # Map delta to posture corrections:
    # - Roll deviation → body_roll correction
    # - Pitch deviation → body_pitch correction
    # - Lateral velocity deviation → body_y_offset correction
    params.body_roll  += K_pyragas[0] * delta[roll_idx]
    params.body_pitch += K_pyragas[1] * delta[pitch_idx]
    params.body_y_offset -= 0.05 * delta[vy_idx]
```

**Why this works**: During stable trotting, x(t) ≈ x(t-τ). The correction
is zero. When the gait becomes irregular (one stride shorter, asymmetric
contacts, momentum building in roll), x(t) ≠ x(t-τ) and the correction
activates. It naturally adapts to whatever the "good" orbit looks like.

**Runtime cost**: Ring buffer lookup + elementwise multiply. ~0μs.

### Phase 2d — Multi-Step Rollout (Future)

Feed the ensemble's prediction back as input for the next prediction:

```
x₀ → ensemble → dx₁ → x₁ = x₀ + dx₁ → ensemble → dx₂ → x₂ = x₁ + dx₂ → ...
```

Evaluate cumulative instability Q over the trajectory. Optimize the first
control action to minimize the sum. This is model-predictive control (MPC)
with the learned ensemble as the dynamics model.

**Deferred** because:
1. 2a+2b+2c are simpler and may be sufficient
2. Multi-step rollout amplifies model errors (200ms prediction accuracy
   doesn't guarantee 600ms accuracy)
3. Need to validate single-step corrections first

## Implementation Plan

### Step 1: BodyObservation dataclass + plumbing

```python
# In config/defaults.py (or body_model.py)
@dataclass
class BodyObservation:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel: tuple[float, float, float] = (0.0, 0.0, 9.81)
    foot_contacts: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
```

- Add optional `body_obs` param to `Locomotion.update()` and `_walk_pipeline()`
- `simulation.py:send_motion_command()` reads `get_robot_state()`, builds
  `BodyObservation`, passes it down
- No change for callers that don't pass body_obs (backward compat)

### Step 2: Live ensemble prediction in body_model.py

- New class `MomentumPredictor` (extends beyond `LearnedTurnCoupling`)
- Loads same ensemble.npz
- `predict(body_obs, gait_cmd) -> (6,)` — real-time prediction from actual state
- `turn_factor(speed, wz_abs, body_obs) -> float` — state-aware version
- Keep `LearnedTurnCoupling` as fallback for when body_obs is None

### Step 3: Speed-gradient corrections

- `MomentumPredictor.stabilize(body_obs, gait_params) -> GaitParams`
- Computes gradient of Q w.r.t. adjustable params
- Returns corrected GaitParams
- Called from `_walk_pipeline()` after velocity mapping

### Step 4: Pyragas delayed feedback

- Ring buffer in `MomentumPredictor` (stateful, matches gait period)
- `pyragas_correction(body_obs) -> GaitAdjustment`
- Blended with speed-gradient corrections

### Step 5: Training data enrichment (Phase 3)

- `play_collect.py` records the full 29D observation (including leg_phases + posture_cmd)
- Collects data with posture perturbations (body_roll, body_pitch, body_x_offset, body_y_offset)
  in addition to vx/wz
- 6D CEM controller explores posture space via ensemble disagreement
- Retrained ensemble with posture coverage makes finite-difference gradient accurate

## Control Variables and Safe Ranges

| Variable | Base Value | Max Correction | L3 Safety Limit | Rationale |
|----------|-----------|----------------|-----------------|-----------|
| body_roll | 0.0 rad | ±0.10 rad | ±0.26 rad | Lean into/against lateral disturbances |
| body_pitch | 0.0 rad | ±0.10 rad | ±0.26 rad | Lean into/against longitudinal disturbances |
| body_x_offset | 0.0 m | ±0.05 m | ±0.10 m | Shift CoM fore/aft for stability |
| body_y_offset | 0.0 m | ±0.05 m | ±0.10 m | Shift CoM laterally for balance |

Posture corrections flow through: L5 GaitParams → `_l5_to_l4()` → L4 BodyPoseCommand → L3 `transform_feet_to_body_frame()` RPY rotation + CoM offset → IK solver.

**wz** is NOT a control variable here — it's a navigation input from L6.
The turn_factor attenuation stays as a safety bound. The momentum
controller operates on body posture, not trajectory or gait timing.

## Goal Function Design

```python
Q(pred) = 4.0 * pred[3]**2    # droll — lateral tipping (highest weight)
        + 4.0 * pred[4]**2    # dpitch — longitudinal tipping
        + 1.0 * pred[1]**2    # dvy — lateral drift (unwanted)
        + 0.5 * (pred[2] - wz_cmd)**2  # dwz error (tracking)
```

**Not penalized**: dvx (we want forward acceleration), dyaw (controlled by
navigation). The goal function encodes: "keep the robot level and on-track,
don't care how fast it gets there."

## Relationship to Chaos Control Theory

| Paper Method | Our Application |
|-------------|-----------------|
| OGY (linearize near orbit, control in deadzone) | Corrections only activate when |Q| exceeds a threshold — robot running smoothly = no intervention |
| Pyragas (delayed feedback, τ = orbit period) | Ring buffer at gait period, stabilizes trot cycle |
| Speed-gradient (∇_u Q̇) | Finite-difference gradient through ensemble |
| Adaptive control (online parameter estimation) | Ensemble disagreement as uncertainty → conservative when uncertain |
| Neural network identification | The ensemble IS the system identification model |
| "Chaotic = more controllable" | Unstable gait regimes respond most to small corrections |

## The Deadzone Principle (OGY Insight)

From eq. 26: `u_k = Cx̃_k if |x̃_k| ≤ Δ, else 0`

We invert this: corrections activate only when predicted instability exceeds
a threshold. When the gait is stable (Q < Q_threshold), the momentum
controller outputs zero corrections and the base pipeline runs unmodified.
This prevents the controller from "helping" when it's not needed, which
would add noise to an already-good gait.

```python
Q_THRESHOLD = 0.01  # below this, gait is stable — don't intervene
if Q_base < Q_THRESHOLD:
    return base_params  # no correction
# else: compute and apply speed-gradient corrections
```

## Ensemble Disagreement as Caution

The ensemble already gives us epistemic uncertainty for free. When members
disagree (high variance), the model is uncertain about this state — perhaps
it's out of distribution. In that regime:

- Reduce `gamma` (correction strength) proportional to agreement
- Increase correction clamping ranges (be more conservative)
- Log the disagreement for training data prioritization

```python
preds = [predict(ensemble, obs, k) for k in range(K)]
variance = mean(var(preds, axis=0))  # scalar disagreement
confidence = 1.0 / (1.0 + 10.0 * variance)  # [0, 1]
gamma_effective = gamma * confidence  # cautious when uncertain
```

This connects to curiosity-driven play: disagreement = uncertainty =
"collect more data here." The momentum controller naturally identifies
where the model needs more training data.

## What Stays The Same

- **Layer boundaries**: L5 still receives commands from L6, produces
  GaitParams for L4. The momentum controller is internal to L5.
- **Ensemble format**: Same .npz architecture, same inference code.
  `MomentumPredictor` reuses `_predict_ensemble_mean` directly.
- **Chaos control methods**: Speed-gradient (Fradkov), Pyragas, OGY
  deadzone — same algorithms, applied to posture dimensions.

**What changed in Phase 3**: Observation expanded 21D→29D (added
leg_phases + posture_cmd). Control variables switched from gait knobs
to body posture. Training pipeline updated to 6D posture space.

## Validation

1. **A/B test** via `fast_test.py`: Run 12+ seeds with and without
   momentum controller. Compare falls, ATO, target reach rate.
2. **Instability injection**: Add terrain perturbations or periodic
   pushes, measure recovery time with vs without controller.
3. **Correction magnitude monitoring**: Log |corrections| per tick.
   Should be small (<5% of base param) in steady state, spike only
   during perturbations.
4. **Gait regularity metric**: std(stride_period) across steps. Pyragas
   should reduce this.

## File Plan (as implemented)

| File | Changes |
|------|---------|
| `layer_5/body_model.py` | `MomentumPredictor` class (2b, 2c), `LearnedTurnCoupling`, 29D observations, posture corrections |
| `layer_5/locomotion.py` | `set_posture_perturbation()`, posture application in `_walk_pipeline()` |
| `layer_5/simulation.py` | `_build_body_observation()` with leg_phases/posture_cmd, `_l5_to_l4()` posture forwarding |
| `layer_5/config/defaults.py` | `BodyObservation` dataclass with leg_phases + posture_cmd |
| `training/hnn/ensemble.py` | input_dim=29 |
| `training/hnn/play_collect.py` | 6D command callbacks, records leg_phases + posture_cmd |
| `training/hnn/play_controller.py` | 6D CEM over posture space |
| `training/hnn/play_safety.py` | Posture-based clamping |
| `training/hnn/play_train.py` | 29D observation batches |
| `training/configs/b2/play.yaml` | `posture:` section (replaces `perturbation:`) |
