---
source: part64/deep-research-report.md
section: Diagnostics, visualization, and audit trails
status: inprogress
reviewed_on: 2026-02-20
---

# Part64 Deep Research Spec 08 - Diagnostics and Audit Surfaces

## Scope
- Surface runtime observability for pressure, gravity, routing, compaction, and decision auditability.

## Current evidence in code
- Simulation payload exposes rich route/price/gravity diagnostics in `part64/code/world_web/simulation.py`.
- Projection metadata with grouped membership and digests exists in `part64/code/world_web/simulation.py`.
- Frontend projection and stability panels consume runtime evidence in `part64/frontend/src/App.tsx` and `part64/frontend/src/components/Panels/StabilityObservatoryPanel.tsx`.

## Gaps vs report
- No explicit explanation-cost metric for decisions.
- No built-in generation pipeline for paper-ready figures/heatmaps from runs.
- Mermaid artifacts are documented in report text but not generated from runtime snapshots.

## Definition of done
- Add explanation-cost and reproducibility metrics to runtime outputs.
- Add scripted artifact generation for heatmaps/time-series/audit diagrams.
- Add regression checks for artifact schema and deterministic replay.
