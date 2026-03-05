---
status: in_progress
priority: medium
source_note: docs/notes/security_feature_extractor/2026-03-01-183148-security-extraction-and-ranking-toolbox.md
depends_on:
  - specs/security-tracking/01-github-security-extraction-foundation.md
  - specs/security-tracking/02-deterministic-security-ranker.md
last_reviewed: 2026-03-04
---

# Spec 05: Security Pipeline Hardening

## Purpose

Add reliability, denoise, and throughput controls around the extractor and ranker so signal quality holds under load.

## Current Reality (2026-03-04)

- Dedupe, source-tier weighting, corroboration scoring, and weak-label logic are wired into
  GitHub threat scoring.
- Operational hardening is still incomplete; max-node headroom mitigation is now in place,
  but live throughput recovery still requires runtime validation.
- Global threat discovery remains seed-only, reducing the practical effect of downstream
  hardening controls.

## In Scope

- Change-aware crawling and dedupe (MinHash/SimHash).
- Frontier prioritization with bounded exploration.
- Weak supervision label-function framework for feature growth.
- Source credibility weighting and corroboration gating.
- Optional anomaly detectors for early unlabeled detection.

## Key Controls

- Bounded queue budgets by lane.
- Stable source-tier weighting in final ranking.
- Redundancy requirement before critical promotion.
- Baseline-vs-model regression checks.

## Task Checklist

- [x] Add near-duplicate collapse in ranked threat rows.
- [x] Implement deterministic weak-label function registry.
- [x] Add source-tier weighting and corroboration boosts in scoring path.
- [ ] Restore healthy live throughput by preventing crawler terminal stop from halting
      discovery indefinitely (headroom/cap trap mitigation landed; validate in live runtime).
- [ ] Add value-aware frontier policy with explicit exploration budget telemetry.
- [ ] Add offline + runtime regression dashboards (precision drift, dedupe ratio,
      corroboration distribution, false-positive trend).
- [ ] Enforce corroboration requirements for high/critical promotion where evidence is thin.

## Acceptance Criteria

- Duplicate/near-duplicate content does not dominate top-K rankings.
- Throughput remains bounded under crawl bursts.
- Critical alerts require corroboration and remain explainable.
- Model regressions are detectable against baseline metrics.
