# Improvements

Research proposals and architectural decisions for the robotics control stack.

---

## Current Documents

| Document | Status | Layers | Summary |
|----------|--------|--------|---------|
| **[ARCHITECTURE_JOURNEY.md](ARCHITECTURE_JOURNEY.md)** | **Historical record** | All | **Story of rejection, insight, and resolution** |
| **[LEARNING_ROADMAP.md](LEARNING_ROADMAP.md)** | **Future work** | Training | **How to train HNN and RL policies (offline)** |
| [free_energy_principle.md](free_energy_principle.md) | Conceptual | 2-5 | FEP theory and active inference overview |
| [layering_discipline.md](layering_discipline.md) | **Architectural requirement** | All | N → N-1 discipline: observation chain, no layer skipping |
| [naming_conventions.md](naming_conventions.md) | **Architectural standard** | All | Semantic naming: KinematicState, DynamicState, TerrainEstimate |
| [hnn_placement.md](hnn_placement.md) | Design specification | 4 | HNN lives in Layer 4, does prediction + learning |
| [stateless_vs_learning.md](stateless_vs_learning.md) | **Architecture resolution** | 4-5 | Layer 4: stateful for learning, Layer 5: gait selection only |

**Start here**: Read [ARCHITECTURE_JOURNEY.md](ARCHITECTURE_JOURNEY.md) to understand how we discovered the correct architecture through Layer 5's boundary rejection.

---

## Implementation Status

**FEP terrain-aware locomotion is COMPLETE** ✅

- **Layer 3** ([#11](https://github.com/Pangeon-Robotics/layer_3/issues/11)): ✅ KinematicState publisher implemented
- **Layer 4** ([#19](https://github.com/Pangeon-Robotics/layer_4/issues/19)): ✅ Simple terrain estimation implemented
- **Layer 5** ([#2](https://github.com/Pangeon-Robotics/layer_5/issues/2)): ✅ Terrain-aware gait selection implemented

**Note**: Original Layer 4 issue [#18](https://github.com/Pangeon-Robotics/layer_4/issues/18) (HNN-based) was closed as out of scope - HNN learning violated Layer 4's stateless architecture. The simpler sensor-heuristic approach (#19) respects boundaries and provides needed functionality.

**Key architectural insight**: Each layer only accesses Layer N-1's API. Telemetry flows upward with increasing abstraction:
- Layer 3 → Layer 4: Arrays (4×3 positions/forces)
- Layer 4 → Layer 5: Scalars (roughness, compliance, stability)

See [ARCHITECTURE_JOURNEY.md](ARCHITECTURE_JOURNEY.md) for the full story.

---

## Next Steps: Adding Learning

**Current**: Rule-based terrain estimation + gait selection (no learning)

**Next**: Add HNN dynamics learning and RL policy training

See [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md) for:
- Where learning happens (offline training, frozen deployment)
- How to train HNN dynamics models
- How to train RL/PPO locomotion policies
- Phased rollout (HNN → RL → Online adaptation)

**Key principle**: Learning happens **outside the layer stack**. Layers receive frozen models and do stateless inference only.

---

## Purpose

Documents here are:
- **Architectural decisions**: Resolved design questions that guide implementation
- **Research proposals**: Forward-looking ideas that may be implemented
- **Cross-cutting concerns**: Topics spanning multiple layers

---

## Adding New Proposals

When adding a proposal:

1. **Create a markdown file** with clear title
2. **Include header**:
   ```markdown
   **Status**: Research proposal | Design specification | Architectural requirement
   **Created**: YYYY-MM-DD
   **Target**: Layer(s) affected
   ```
3. **Structure**: Problem → Solution → Implementation → Success Criteria
4. **Update this README** with a table row

---

## Philosophy Alignment

All proposals must respect:
- **Layered abstraction**: No `if simulation:` branches
- **Hardware abstraction**: Works on real robots
- **Layer sovereignty**: Don't cross boundaries
- **Sim2real transfer**: Generalizes from sim to real

See [../philosophy/](../philosophy/) for full principles.

---

## Relationship to Other Docs

- **improvements/** (here): Research and architectural decisions
- **improvements/fix-requests/**: Detailed implementation specs (GitHub issues)
- **layer_X/plans/**: Near-term implementation tasks within a layer
- **philosophy/**: Established principles and workflows

---

**Last updated**: 2026-02-10
