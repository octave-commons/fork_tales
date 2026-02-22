---
source: part64/deep-research-report.md
section: Auditability and Visualization
status: incoming
reviewed_on: 2026-02-21
---

# Spec: Auditable Path and realization Visualization

## 1. Purpose
Implement UI surfaces to expose the internal audit trails of the system, including shortest paths used for gravity, realized Daimoi sampling outcomes, and local price heatmaps. This makes the "Semantics-as-topology" claim from the Self-Organizing Graph Runtime paper verifiable by users.

## 2. Scope
- **Shortest Path Export**: Modify the C backend (`c_double_buffer_sim.c`) or Python bridge to return the actual node sequences for the top-N gravity paths.
- **Gravity Heatmaps**: Add a canvas overlay in the frontend to visualize $G_k[n]$ (demand potential) and $P[n,k]$ (pressure).
- **Realization Receipts**: Implement a "Receipt Inspector" in the UI that displays the realization hash, logits, and sampled component for a specific absorption event.
- **River Flows**: Visualize consolidated Daimoi movements as "rivers" on ViewGraph edges.

## 3. Implementation Plan
- **Phase 1 (Backend Telemetry)**: Update `cdb_graph_runtime_maps` to optionally store and return path data.
- **Phase 2 (Heatmap Shader)**: Implement a WebGL or Canvas-based heatmap layer in `Canvas.tsx` for demand potentials.
- **Phase 3 (Receipt UI)**: Create a new panel `AuditInspector.tsx` that fetches event metadata from `receipts.log`.
- **Phase 4 (Path Overlay)**: Draw explicit lines for shortest-path routes between Presences and their targets.

## 4. Acceptance Criteria
- [ ] User can toggle a "Demand Field" view showing gravity wells.
- [ ] Clicking a "Realization Receipt" in the log opens a detailed view of the sampling logits and biases.
- [ ] Shortest paths are visually rendered as traces over the ViewGraph.
- [ ] UI provides a "Shadow Price" overlay for each resource type ($k$).
