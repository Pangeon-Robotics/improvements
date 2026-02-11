# Free Energy Principle (FEP) for Robotics

**Status**: Conceptual framework
**Created**: 2026-02-10

---

## The Core Idea

**Biological systems minimize "surprise"** (unexpected sensory input) by:
1. **Predicting** what will happen (internal model)
2. **Acting** to make predictions come true (active inference)
3. **Updating** predictions when wrong (learning)

**For robots**: Use prediction errors to adapt behavior and improve internal models.

---

## RIC-7: Robot Inference & Control

A conceptual 7-layer mapping of FEP concepts to control:

| Layer | Function | Signal Type |
|-------|----------|-------------|
| **7** | Mission Logic | Goals, constraints |
| **6** | FEP Inference | Belief trajectories |
| **5** | System ID | Model parameters θ |
| **4** | Hamiltonian Dynamics | Energy gradients (q, p) |
| **3** | PD Stabilization | Target positions (q*, dq*) |
| **2** | Actuation | PWM, currents |
| **1** | Physical Plant | Sensors, motors |

**Signal flow**:
- **Downstream (action)**: Layer 7 → 6 → 3 → 2 → 1
- **Upstream (perception)**: Layer 1 → 4 (predict) → 5 (update) → 6 (minimize error)

---

## Practical Implementation

See [plans/fep_integration.md](plans/fep_integration.md) for concrete implementation plan:
- Layer 4: Trajectory prediction + contact force modeling
- Layer 5: FEP-based adaptive gait selection

**Start simple**: Linear models, lookup tables, no neural networks.

---

## Key References

- **Friston (2010)**: The free-energy principle - a unified brain theory
- **Pio-Lopez et al. (2016)**: Active inference for robot reaching
- **Lanillos & Cheng (2018)**: Robot self-localization via active inference

---

**For implementation details, see**: [plans/fep_integration.md](plans/fep_integration.md)