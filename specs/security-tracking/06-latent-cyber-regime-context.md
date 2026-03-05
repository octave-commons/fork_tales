---
status: in_progress
priority: medium
source_note: docs/notes/security_feature_extractor/2026-03-01-183954-latent-cyber-regime-model.md
depends_on:
  - specs/security-tracking/04-hmm-entity-state-smoothing.md
  - specs/security-tracking/05-security-pipeline-hardening.md
last_reviewed: 2026-03-04
---

# Spec 06: Latent Cyber Regime Context Model

## Purpose

Model macro security posture as a latent state process and feed that posterior into thresholding and crawl-routing policy, without replacing deterministic per-item scoring.

## Current Reality (2026-03-04)

- Regime posterior and policy wiring are active in `cyber_risk_radar` responses.
- Runtime now applies a controlled threshold fallback when strict regime thresholding
  would return an empty result (`count=1` in current baseline runtime snapshot).
- Because global discovery is not progressing, regime observations are likely underfed and
  policy outputs are conservative by default.

## State Model (example)

- `baseline`
- `elevated_chatter`
- `active_exploitation_wave`
- `supply_chain_campaign`
- `geopolitical_targeting_shift`

## Observation Inputs (windowed)

- Aggregated deterministic classifier outputs
- CVE/KEV/PoC and vulnerability-family rates
- Graph-proximity aggregates to bad seeds
- Source-diversity and corroboration rates

## Policy Effects

- Adaptive thresholds by regime posterior
- Adaptive crawl/routing budgets by regime
- Bounded query expansion policy by regime

## Query Additions

- `cyber_regime_state(window_ticks)`
- `cyber_risk_radar(window_ticks, limit)` with regime context fields

## Task Checklist

- [x] Define regime state set and transition constraints in runtime logic.
- [x] Implement deterministic posterior computation and regime-aware query outputs.
- [x] Wire threshold policy to regime state in `cyber_risk_radar`.
- [x] Recalibrate threshold policy to avoid permanently empty cyber surfaces under baseline
      (fallback path now activates when strict thresholding yields no rows).
- [ ] Continue tuning fallback policy to preserve precision while avoiding empty output.
- [ ] Validate regime outputs against sustained non-provisional discovery windows.
- [ ] Expose operator controls/reporting for regime threshold and routing impact.

## Acceptance Criteria

- Regime posterior is stable and replayable.
- Threshold/routing policy shifts are traceable to posterior deltas.
- No direct LLM severity assignment is introduced.
