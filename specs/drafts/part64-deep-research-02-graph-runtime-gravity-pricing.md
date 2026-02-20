---
source: part64/deep-research-report.md
section: Edge cost, gravity maps, and local pricing
status: done
reviewed_on: 2026-02-20
---

# Part64 Deep Research Spec 02 - Graph Runtime Gravity and Pricing

## Scope
- Implement graph-distance cost, bounded gravity propagation, and local scarcity pricing used by routing.

## Current evidence in code
- Bounded Dijkstra and per-resource gravity maps are implemented in `part64/code/world_web/c_double_buffer_backend.py`.
- Drift and valve terms combine gravity, cost, affinity, saturation, and health in `part64/code/world_web/c_double_buffer_backend.py`.
- Native C runtime computes edge cost and local price (`exp(pressure_hat) * congestion term`) in `part64/code/world_web/native/c_double_buffer_sim.c`.
- Runtime summary exposes `edge_cost`, `gravity`, `node_price`, and diagnostics in `part64/code/world_web/c_double_buffer_backend.py`.

## Verification evidence
- Route behavior and resource-signature gravity routing are tested in `part64/code/tests/test_c_double_buffer_backend.py`.
- Runtime map fields and diagnostics are validated in `part64/code/tests/test_c_double_buffer_backend.py`.

## Notes
- This slice is effectively implemented and covered by targeted tests.
