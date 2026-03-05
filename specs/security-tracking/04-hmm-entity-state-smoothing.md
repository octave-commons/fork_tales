---
status: in_progress
priority: medium
source_note: docs/notes/security_feature_extractor/2026-03-01-182647-hmm-temporal-stabilizer.md
depends_on:
  - specs/security-tracking/03-proximity-feature-engine.md
last_reviewed: 2026-03-04
---

# Spec 04: HMM Entity State Smoothing

## Purpose

Convert twitchy per-window risk signals into stable entity/topic state probabilities.

## Current Reality (2026-03-04)

- State posterior logic is present in proximity/entity risk query flows.
- Smoothed state fields are available in query payloads, but policy coupling remains partial
  and is not yet the primary operator-facing control in local/global radar views.
- Flap-rate and transition-audit instrumentation is not yet tracked as an explicit runtime KPI.

## State Model

- `background`
- `emerging`
- `active`
- `critical`
- optional `decaying`

Use sticky transitions (high self-loop probabilities) to prevent alert flapping.

## Observation Inputs

- Deterministic ranker score (`p_risk` or calibrated score)
- Proximity feature aggregates (semantic/graph/temporal)
- Hard-pattern counts (CVE, IOC patterns)
- Source-diversity/corroboration counts

## Outputs

- `p_active`, `p_critical` per tracked entity/topic
- Last transition tick and current most likely state
- Top evidence references for current state

## Query Additions

- Extend `proximity_radar` with state probabilities, or
- Add `entity_risk_state(window_ticks, limit)`

## Task Checklist

- [x] Define tracked-entity keying/windowing for state bins in proximity workflows.
- [x] Implement deterministic posterior/state-bin estimation logic.
- [x] Integrate feature-to-state mapping and expose posterior outputs.
- [ ] Promote smoothed state to primary alert gating path in local/global radar views.
- [ ] Persist explicit transition markers and audit references for operator inspection.
- [ ] Add flap-rate regression checks against unsmoothed baselines.

## Acceptance Criteria

- Alert flapping decreases versus unsmoothed score-only behavior.
- Transition points are auditable from stored observations.
- Replays over same window history reproduce identical posterior output.
