---
source: part64/deep-research-report.md
section: TruthGraph and ViewGraph
status: inprogress
reviewed_on: 2026-02-20
---

# Part64 Deep Research Spec 01 - TruthGraph and ViewGraph

## Scope
- Implement a persistent TruthGraph and a lossless, reconstructable ViewGraph projection (coarsen/expand/replay).

## Current evidence in code
- The simulation graph supports projection with grouped overflow membership metadata in `part64/code/world_web/simulation.py`.
- Projection groups keep `member_edge_ids`, counts, and digest metadata in `part64/code/world_web/simulation.py`.
- Deterministic projection and membership recovery are tested in `part64/code/tests/test_presence_runtime.py`.

## Gaps vs report
- No explicit `TruthGraph` runtime object with immutable provenance operations.
- No persisted surjective map `Pi: V^T -> V^V` contract.
- No explicit `compress/expand/replay lossless` API boundary for bundles.
- Membership data is present, but full reconstruction ledger semantics are not formalized end-to-end.

## Definition of done
- First-class `TruthGraph` and `ViewGraph` schemas exist in runtime payloads.
- Bundle ledger supports exact reconstruct/expand operations under test.
- Projection map and reconstruction are persisted and replayable.
