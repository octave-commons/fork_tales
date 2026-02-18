# ημ Ingest Text+Image v1

## Priority
- high

## Intent
- Implement `ημ.ingest.text+image.v1` as a deterministic ingest pipeline for `.ημ/`.
- Accept only text and image inputs, reject all other modalities with packetized reasons.
- Produce idempotent embeddings and append-only registry records suitable for replay and audit.

## Requirements
- Scan `.ημ/` with bounded traversal and deterministic lexical ordering.
- Enforce include/exclude filters using MIME-first and extension fallback.
- Canonicalize text before hashing/chunking and hash image raw bytes without mutating source files.
- Route text/image to explicit embedding spaces and compute stable `embed.id` values.
- Upsert embedding vectors to Chroma vecstore collection `ημ_nexus_v1` with provenance metadata.
- Persist append-only registry writes containing status, timestamps, packet refs, and idempotence keys.
- Emit packet stream across ingest stages and write `.Π/` manifest/stats/snapshot artifacts.

## Open Questions
- none (defaults applied)

## Defaults Applied
- Embedding provider target is Ollama, with deterministic local hash-vector fallback when provider calls fail.
- Image preprocessing is represented as a derived artifact spec and never mutates source files.
- Unsupported/oversize files are rejected and moved to `.ημ/_rejected/` with reason metadata.

## Risks
- Live Ollama/Chroma availability can vary at runtime; fallback/defer pathways must preserve determinism.
- Registry format migration must remain backward compatible with existing `ημ.ingest-registry.v1` rows.
- More granular segment-level registry events can increase file size over time.

## Phases
1. Add contract constants, space signatures, filters, and helper primitives.
2. Implement end-to-end ingest pipeline stages with packet emission and idempotent dedupe.
3. Add vecstore upsert integration, snapshot artifact writes, and drift/safe-mode handling.
4. Extend tests for text/image ingest, rejects, and `.Π/` artifact outputs.

## Candidate Files
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/world_web.py`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/tests/test_world_web_pm2.py`
- `specs/drafts/eta-mu-ingest-text-image-v1.md`

## Existing Issues / PRs
- No repository issue/PR references discovered for this contract.

## Complexity Estimate
- high (cross-cutting ingest, registry, embeddings, vecstore, and artifact pipeline)

## Definition of Done
- `.ημ/` ingest handles only text and image files under contract filters.
- Duplicate unchanged segment embeddings are skipped via registry idempotence keys.
- Registry writes are append-only and include packet references and status outcomes.
- `.Π/ημ_ingest_manifest_*`, `.Π/ημ_ingest_stats_*`, and `.Π/ημ_ingest_snapshot_*` artifacts are produced.
- Existing world web tests and new ingest tests pass.

## Session Changelog
- Added contract constants and helper primitives for MIME/ext filtering, canonicalization, segmenting, idempotence keys, packet emission, vecstore routing, and S-expression artifact writes.
- Reworked `sync_eta_mu_inbox` into a staged text+image ingest pipeline with reject/skip/defer/ok statuses, packet refs, and append-only registry behavior.
- Added tests for unsupported-file rejection quarantine and for `.Π` ingest artifact output + registry idempotence fields.
