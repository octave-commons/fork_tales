# Security Feature Extractor Notes

This folder groups the notes that directly shape the security feature extractor and threat scoring lane.

## Scope

- Deterministic extraction atoms and scoring features
- Classifier-first ranking strategy
- Proximity features for unseen entities
- Temporal stabilization (HMM) and latent regime context
- Crawler-to-extractor contract for security-relevant signals

## Suggested reading order

1. `2026-02-27-191804-github-crawler-security-extraction-spec.md`
2. `2026-03-01-175636-deterministic-security-classifier-and-label-sources.md`
3. `2026-03-01-182539-proximity-signals-for-new-entities.md`
4. `2026-03-01-182647-hmm-temporal-stabilizer.md`
5. `2026-03-01-183148-security-extraction-and-ranking-toolbox.md`
6. `2026-03-01-183954-latent-cyber-regime-model.md`

## Quick map

- Dataset and labels: `2026-03-01-175636-deterministic-security-classifier-and-label-sources.md`
- Feature design (semantic/graph/temporal): `2026-03-01-182539-proximity-signals-for-new-entities.md`
- Sequence smoothing and state output: `2026-03-01-182647-hmm-temporal-stabilizer.md`
- End-to-end pipeline options: `2026-03-01-183148-security-extraction-and-ranking-toolbox.md`
- Hidden-variable/regime framing: `2026-03-01-183954-latent-cyber-regime-model.md`
- GitHub extraction contract and atoms: `2026-02-27-191804-github-crawler-security-extraction-spec.md`
