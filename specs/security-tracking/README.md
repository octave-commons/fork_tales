# Security Tracking Specs

This folder tracks implementation and operations reality for security tracking, ranking,
and threat radar behavior.

## Current Runtime Snapshot (2026-03-04)

- Global radar default path suppresses provisional watchlist seed rows and now excludes
  raw feed/watchlist source evidence by default; live output currently returns
  `count=0` (`provisional_count=0`, `quality.needs_crawl_evidence=true`) until
  model-flagged downstream pages accumulate.
- Local radar currently returns 10 rows (0 high / 1 medium) with weak-label score filtering
  reducing low-security-noise items.
- Cyber regime radar now returns 1 row via controlled threshold fallback when strict regime
  threshold would otherwise produce empty output.
- Web Graph Weaver continuity guard is live-verified under forced-cap pressure
  (`crawl_state.reason=max_nodes_autogrow`, `max_nodes` 5390 -> 7438, frontier remained non-empty).
- Weaver crawl now auto-starts on runtime boot (`config.crawl_autostart=true`) using
  watchlist + graph seed context, so crawl routing no longer depends on manual UI start.
- Weaver runtime now applies large boot floors and PM2 defaults for crawl capacity
  (`max_depth=12`, `max_nodes=2000000`, `concurrency=32`, `max_requests_per_host=64`,
  `entities=128`) so legacy low snapshot/env values no longer pin discovery throughput.
- Threat radar UI ownership is now split by muse lane: witness_thread owns local GitHub
  security radar and chaos owns global geopolitical radar as independent fixed panels.
- Classifier scoring is active (`github_linear_v1`), but LLM blend is not applied
  (`llm_invalid_json`).

## Spec Set

1. `01-github-security-extraction-foundation.md`
2. `02-deterministic-security-ranker.md`
3. `03-proximity-feature-engine.md`
4. `04-hmm-entity-state-smoothing.md`
5. `05-security-pipeline-hardening.md`
6. `06-latent-cyber-regime-context.md`
7. `implementation-order.md`

## Status Board

- `01` in progress (implemented core extraction contracts; global provisional suppression + alerting landed)
- `02` in progress (classifier path active; weak-label score gating landed; calibration/metadata contract still incomplete)
- `03` in progress (proximity features wired; signal quality tuning still needed)
- `04` in progress (state smoothing present; alert policy integration incomplete)
- `05` in progress (pipeline hardening active; live throughput/quality validation still pending)
- `06` in progress (regime model wired; fallback prevents permanently empty cyber surface)

## Source Notes Mapped

- `docs/notes/security_feature_extractor/2026-02-27-191804-github-crawler-security-extraction-spec.md`
- `docs/notes/security_feature_extractor/2026-03-01-175636-deterministic-security-classifier-and-label-sources.md`
- `docs/notes/security_feature_extractor/2026-03-01-182539-proximity-signals-for-new-entities.md`
- `docs/notes/security_feature_extractor/2026-03-01-182647-hmm-temporal-stabilizer.md`
- `docs/notes/security_feature_extractor/2026-03-01-183148-security-extraction-and-ranking-toolbox.md`
- `docs/notes/security_feature_extractor/2026-03-01-183954-latent-cyber-regime-model.md`
