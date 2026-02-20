---
source: part64/deep-research-report.md
section: Presence model, need, and mass
status: inprogress
reviewed_on: 2026-02-20
---

# Part64 Deep Research Spec 03 - Presence Need and Mass Model

## Scope
- Implement Presence-local state for mask, influence, need, and mass, then project those into gravity fields.

## Current evidence in code
- Presence runtime state and event stream infrastructure exists in `part64/code/world_web/presence_runtime.py`.
- Presence-derived source mass and need heuristics are computed in `part64/code/world_web/c_double_buffer_backend.py`.
- Resource need signatures influence routing (`resource-signature` mode) in `part64/code/world_web/c_double_buffer_backend.py`.

## Gaps vs report
- No explicit sparse mask `M_p` model with mask-weighted expectations.
- Need equations are now EMA + logistic threshold per resource in runtime diagnostics, but threshold/alpha tuning remains heuristic and not externally configurable.
- Mass is not yet tied to a documented node-mass function over semantic/resource weight.
- Influence pins and formal per-presence source selection policy are implicit.

## Progress notes
- `source_profiles[*].need_model` now exposes `kind`, `priority`, `alpha`, `util_raw`, `util_ema`, and `thresholds` in `part64/code/world_web/c_double_buffer_backend.py`.
- Runtime contract now reports `presence_model.need = ema-logistic-resource-need.v2`.
- EMA + logistic behavior is regression-covered by `test_presence_resource_need_model_uses_ema_and_logistic_thresholds`.

## Definition of done
- Presence schema includes explicit `mask`, `influence`, `need`, and `mass` fields.
- Need update uses stable EMA-driven equations and is test-covered.
- Gravity source selection and mass logic are explicit and documented.
