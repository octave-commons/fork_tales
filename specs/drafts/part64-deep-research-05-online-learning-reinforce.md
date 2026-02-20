---
source: part64/deep-research-report.md
section: Trace credit assignment and REINFORCE updates
status: todo
reviewed_on: 2026-02-20
---

# Part64 Deep Research Spec 05 - Online Learning and Credit Assignment

## Scope
- Implement trace recording, delayed reward credit assignment, and REINFORCE-style online updates for absorption policy parameters.

## Current evidence in code
- No implementation was found for REINFORCE update loops, eligibility traces, or reward baselines in `part64/code`.
- No `w_beta`, `w_T`, `u_pk`, reward-normalization, or gradient update path was located in runtime modules.

## Required work
- Add per-presence trace buffers with scope/time metadata.
- Add reward aggregation and probabilistic credit assignment over traces.
- Add stable online updates with baseline/variance normalization and bounded parameter adapters.
- Add deterministic test harness for learning updates and replay behavior.

## Definition of done
- Runtime exposes learning parameters and trace stats in API payloads.
- Learning update path is unit-tested with deterministic seeds.
- Policy updates measurably affect absorb behavior under controlled scenarios.
