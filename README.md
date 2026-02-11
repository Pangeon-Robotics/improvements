# Improvements

Research proposals and architectural decisions for the robotics control stack.

---

## Current Documents

| Document | Status | Layers | Summary |
|----------|--------|--------|---------|
| [free_energy_principle.md](free_energy_principle.md) | Conceptual | 2-5 | FEP theory and active inference overview |
| [layering_discipline.md](layering_discipline.md) | **Architectural requirement** | All | N → N-1 discipline: observation chain, no layer skipping |
| [naming_conventions.md](naming_conventions.md) | **Architectural standard** | All | Semantic naming: KinematicState, DynamicState, TerrainEstimate |
| [hnn_placement.md](hnn_placement.md) | Design specification | 4 | HNN lives in Layer 4, does prediction + learning |
| [stateless_vs_learning.md](stateless_vs_learning.md) | **Architecture resolution** | 4-5 | Layer 4: stateful for learning, Layer 5: gait selection only |

---

## Implementation Status

**FEP implementation has been specified and approved** with corrected architecture:

- **Layer 3** ([issue #11](https://github.com/Pangeon-Robotics/layer_3/issues/11)): Add KinematicState publisher
- **Layer 4** ([issue #18](https://github.com/Pangeon-Robotics/layer_4/issues/18)): Add HNN dynamics + terrain analysis
- **Layer 5** ([issue #2](https://github.com/Pangeon-Robotics/layer_5/issues/2)): Add terrain-aware gait selection

**Key architectural insight**: Each layer only accesses Layer N-1's API. Telemetry flows upward with increasing abstraction:
- Layer 3 → Layer 4: Arrays (4×3 positions/forces)
- Layer 4 → Layer 5: Scalars (roughness, compliance, stability)

See [fix-requests/](fix-requests/) for detailed implementation specifications.

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
