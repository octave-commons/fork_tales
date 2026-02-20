---
source: part64/deep-research-report.md
section: Hybrid simulation and compaction governance
status: inprogress
reviewed_on: 2026-02-20
---

# Part64 Deep Research Spec 06 - Hybrid Simulation and Compaction

## Scope
- Operate a hybrid fluid+particle simulation with pressure-aware routing and governance for compression/expansion.

## Current evidence in code
- Dual backend simulation path (native double-buffer and Python fallback) exists in `part64/code/world_web/simulation.py`.
- Growth pressure detection and consolidation deployment exist in `part64/code/world_web/simulation.py`.
- File graph projection/overflow grouping and deterministic grouping metadata exist in `part64/code/world_web/simulation.py`.

## Verification evidence
- Growth guard deployment and compaction behavior are tested in `part64/code/tests/test_presence_runtime.py`.
- Projection determinism and membership recovery are tested in `part64/code/tests/test_presence_runtime.py`.

## Gaps vs report
- Sentinel and Compactor are not modeled as explicit first-class roles with separate contracts.
- Expansion/uncoarsening governance is limited; no full bundle replay lifecycle.
- Scale-delta Daimoi signaling is partial and not a dedicated contract surface.
