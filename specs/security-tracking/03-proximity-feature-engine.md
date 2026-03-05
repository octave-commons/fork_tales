---
status: in_progress
priority: high
source_note: docs/notes/security_feature_extractor/2026-03-01-182539-proximity-signals-for-new-entities.md
depends_on:
  - specs/security-tracking/01-github-security-extraction-foundation.md
  - specs/security-tracking/02-deterministic-security-ranker.md
last_reviewed: 2026-03-04
---

# Spec 03: Proximity Feature Engine for New Entities

## Purpose

Handle unseen terms and entities by scoring deterministic proximity to known security concepts across semantic, graph, and temporal axes.

## Current Reality (2026-03-04)

- Proximity indexing and state-aware proximity features are wired into GitHub threat scoring.
- Runtime reports show proximity infrastructure active (`proximity_terms_indexed` populated),
  but current surfaced threat quality indicates tuning is still required.
- Global feed usefulness is limited while crawler ingestion remains stopped, reducing
  meaningful unseen-term discovery from fresh external sources.

## In Scope

- Candidate span extraction and provisional entity tracking.
- Semantic proximity features using context embeddings and prototype sets.
- Relational proximity via co-occurrence graph and seed-based propagation scores.
- Temporal proximity via burst and source-diversity signals.

## Feature Contract

- Semantic: `sim_top1`, `sim_top5_mean`, `sim_margin`, `prototype_label`
- Graph: `ppr_score_*`, `seed_neighbor_count`, `weighted_degree`
- Temporal: `burst_score`, `first_seen_seed_overlap`, `source_diversity`

## Atom + Query Additions

- Atom: `proximity_hit(term_id, seed_id, kind, score, ts)`
- Query: `proximity_radar(window_ticks, seed_set, limit)`

## Gating Rule

Promote high-risk unseen items only with two-of-three agreement:
- semantic proximity
- graph proximity
- temporal burst/diversity

## Task Checklist

- [x] Add context span tokenization and proximity term extraction.
- [x] Compute/persist semantic+graph+temporal proximity features in query outputs.
- [x] Emit `proximity_hit` structures and expose `proximity_radar` query output.
- [x] Feed proximity features into deterministic ranker scoring boosts.
- [ ] Tune seed/prototype sets and stopword policy for higher precision.
- [ ] Add operator-facing evidence surfacing so proximity-driven boosts are auditable in UI.
- [ ] Add deterministic replay tests for identical windows with fixed seed sets.

## Acceptance Criteria

- Newly observed terms are ranked without explicit keyword rules.
- Proximity outputs are deterministic for identical input windows.
- Query returns bounded evidence (`embed`, `graph`, `burst`) per term.
- False spikes are reduced by source-diversity and two-of-three gating.
