---
status: in_progress
priority: high
source_note: docs/notes/security_feature_extractor/2026-03-01-175636-deterministic-security-classifier-and-label-sources.md
depends_on:
  - specs/security-tracking/01-github-security-extraction-foundation.md
last_reviewed: 2026-03-04
---

# Spec 02: Deterministic Security Ranker

## Purpose

Replace keyword-only threat scoring with a trained but deterministic classifier used as the primary risk judge.

## Current Reality (2026-03-04)

- Classifier mode is active in runtime scoring (`classifier_version=github_linear_v1`).
- Local radar now applies weak-label min-score filtering to reduce low-security-noise rows,
  but further calibration is still needed for higher precision.
- LLM blend path is requested but not applied in current runtime due `llm_invalid_json`.
- The stable output contract expected by panel consumers still needs explicit alignment
  for model/provenance fields per threat row.

## In Scope

- Build labeled training corpus from structured security sources.
- Train baseline linear classifier (logistic regression / max entropy first).
- Export deterministic inference artifact (weights + feature schema).
- Runtime scoring integration for GitHub threat rows.

## Data Sources (initial)

- CISA KEV (known exploited flag)
- NVD CVE data
- GitHub Advisory Database
- OSV
- EPSS (as prior/feature)

## Feature Groups

- Text: TF-IDF over bounded title/summary/body slices.
- Structured: CVE/CWE/IOC counts, dependency/lockfile touches, label features.
- Contextual: repo/file-path family and source type.

## Model Requirements

- Primary: logistic regression (calibrated probabilities).
- Baseline comparator: Naive Bayes (regression guard).
- Optional secondary: linear SVM/SGD once baseline is stable.

## Runtime Contract

- Inference is deterministic and bounded.
- Model artifact version and feature schema are surfaced in score metadata.
- Rule score remains as fallback and comparison channel.

## Task Checklist

- [x] Integrate deterministic classifier scoring into threat query path with fallback.
- [x] Surface classifier-level scoring metadata at query scope (`classifier_version`, mode).
- [ ] Build reproducible dataset builder + normalization pipeline and artifact manifest.
- [ ] Add precision/recall-at-K evaluation workflow with acceptance thresholds.
- [ ] Expose stable per-row model provenance fields expected by UI consumers.
- [x] Recalibrate score thresholds/weights to reduce non-security false positives in top-K
      (initial weak-label score gating landed for local/cyber surfaces).
- [ ] Continue threshold/weight tuning against live data and regression metrics.
- [ ] Stabilize (or explicitly disable) LLM blend path until parse failures are resolved.

## Acceptance Criteria

- Rank quality improves over rule-only baseline at top-K.
- Inference results are reproducible across repeated runs.
- Missing model artifact cleanly falls back to deterministic rule score.
- Output includes model/version and confidence fields.
