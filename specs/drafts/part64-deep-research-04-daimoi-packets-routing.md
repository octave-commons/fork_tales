---
source: part64/deep-research-report.md
section: Daimoi creation, routing drift, and absorption
status: inprogress
reviewed_on: 2026-02-20
---

# Part64 Deep Research Spec 04 - Daimoi Packets and Routing

## Scope
- Implement probabilistic Daimoi packets, stochastic routing drift, consolidation, and per-presence interpretation.

## Current evidence in code
- Probabilistic Daimoi particle generation is implemented in `part64/code/world_web/daimoi_probabilistic.py`.
- Softmax weighting for Daimoi attention exists in `part64/code/world_web/simulation.py`.
- Route probability, drift terms, and resource-focused routing metadata are emitted by `part64/code/world_web/c_double_buffer_backend.py`.
- Resource Daimoi emission and action-consumption loops are implemented in `part64/code/world_web/simulation.py`.

## Verification evidence
- Probability normalization and collision stress are tested in `part64/code/tests/test_daimoi_probabilistic.py`.
- Resource flow/consumption summaries in simulation payloads are tested in `part64/code/tests/test_daimoi_probabilistic.py` and `part64/code/tests/test_world_web_pm2.py`.

## Gaps vs report
- Component and absorb contracts now exist in runtime payloads, but event-history retention is sampled (bounded) instead of full replay.
- Routing drift diagnostics are still split across particle rows and graph-runtime maps rather than a single consolidated drift ledger.

## Progress notes
- Added packet component contract with `p_i` and `req(c_i,k)` in `part64/code/world_web/daimoi_probabilistic.py` and surfaced it on field-particle rows (`packet_components`, `resource_signature`) plus summary metadata (`packet_contract`).
- Added absorb sampler contract API in `part64/code/world_web/daimoi_probabilistic.py` implementing `beta(x)`, `T(x)`, `q_i`, `cost_i`, and deterministic Gumbel-Max sampling; surfaced per-row sampler summaries and bounded event samples in summary (`absorb_sampler`).
- Updated simulation normalization pass-through in `part64/code/world_web/simulation.py` so new packet/sampler fields remain visible in `/api/simulation` payloads.
- Added regression coverage in `part64/code/tests/test_daimoi_probabilistic.py` for packet-component schema and absorb sampler logits/probabilities.
