---
source: part64/deep-research-report.md
section: ViewGraph dynamic projection
status: incoming
reviewed_on: 2026-02-21
---

# Spec: Dynamic ViewGraph Expansion and Focus-driven Materialization

## 1. Purpose
Enhance the ViewGraph projection logic to allow on-demand expansion of bundled/clustered nodes back into their TruthGraph constituent nodes. This implements the "dynamically expanded/compacted under demand" requirement from the Self-Organizing Graph Runtime paper.

## 2. Scope
- **Demand Detection**: Define a "focus region" (mask-based or click-based) that triggers expansion of clusters within a certain radius.
- **Expansion Logic**: Implement the reversible transform in `simulation.py` that replaces a cluster node with its constituent members from the `member_ids` ledger.
- **Materialization**: Ensure constituents are correctly placed in the layout (re-projection of TruthGraph coordinates) and re-attached to the simulation backend.
- **Cooling/Compaction**: Implement a "cold timer" where expanded nodes are re-compacted if demand in their region drops.

## 3. Implementation Plan
- **Phase 1 (Focus Signal)**: Add `focus_radius` and `focus_center` to the simulation state payload.
- **Phase 2 (Growth Guard Refactor)**: Modify `_apply_daimoi_growth_guard_to_file_graph` to exclude nodes from clusters if they are within the focus region.
- **Phase 3 (UI Trigger)**: Update `Canvas.tsx` to emit focus signals based on mouse position or selected nodes.
- **Phase 4 (Temporal Hysteresis)**: Add a stabilization delay to prevent rapid compress/expand cycles ("thrash").

## 4. Acceptance Criteria
- [ ] Clicking or hovering near a cluster node in the UI expands it into individual file/artifact nodes.
- [ ] Moving focus away from expanded nodes eventually re-bundles them if system pressure is high.
- [ ] Expanded nodes preserve their TruthGraph metadata and provenance links.
- [ ] ViewGraph schemas correctly reflect the change in `reconstructable_bundle_count`.
