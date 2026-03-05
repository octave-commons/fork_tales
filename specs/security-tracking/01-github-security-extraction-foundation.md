---
status: in_progress
priority: high
source_note: docs/notes/security_feature_extractor/2026-02-27-191804-github-crawler-security-extraction-spec.md
last_reviewed: 2026-03-04
---

# Spec 01: GitHub Security Extraction Foundation

## Purpose

Create the deterministic ingestion and extraction contract for GitHub security signals so downstream scoring can rely on stable atoms with receipts.

## Current Reality (2026-03-04)

- Core extraction/query surfaces are present and live.
- Runtime ingestion continuity was improved by max-node headroom mitigation, max-node
  auto-grow guard, and weaver restart; a forced-cap live runtime probe confirms
  auto-grow transitions under load.
- Weaver crawl now auto-starts on runtime boot from watchlist + graph seeds, so
  discovery does not require manual UI-triggered start actions.
- Global radar now suppresses raw feed/watchlist source URL evidence by default, so
  only model-flagged downstream pages should surface in threat rows.
- Global threat report path now honors non-provisional filtering controls at runtime,
  and seed-only detection signals are exposed for operations alerting.

## In Scope

- GitHub fetch scheduling under existing governor constraints.
- Canonical `web:url` and `web:resource` graph writes.
- Deterministic extraction atoms for issues, PRs, releases, advisories, diffs, and lockfile/manifests.
- Named-query surfaces for recent changes and security-relevant summaries.

## Out of Scope

- Attribution or actor inference.
- Unbounded mirroring of repositories.
- Final ML ranking logic (handled by Spec 02+).

## Required Output Contract

- `web:resource.kind` in `github:*` namespace.
- Required metadata fields: `canonical_url`, `fetched_ts`, `content_hash`, `repo`, `number`, `labels`, `authors`, `updated_at` when available.
- Bounded atom extraction (max per resource).
- Append-only events for schedule/start/complete/fail/extract phases.

## Required Atoms (v1)

- `mentions(repo, term)`
- `mentions_file(repo, path)`
- `changes_dependency(repo, dep_name, from_ver, to_ver)`
- `references_cve(repo, cve_id)`
- `touches_path(repo, path_prefix)`
- `pr_state(repo, pr_number, state)`
- `pr_merged(repo, pr_number, merged_at)`

## Task Checklist

- [x] Define/finalize seed config for monitored repos/keywords/file patterns.
- [x] Implement deterministic fetch frontier with per-repo and per-url cooldown.
- [x] Implement parser pipeline for metadata + file/diff extraction.
- [x] Emit atoms and graph events with bounded payloads.
- [x] Expose named queries (`github_status`, `github_repo_summary`, `github_find`,
      `github_recent_changes`).
- [x] Restore continuous ingestion operations so crawler discovery resumes after
      `max_nodes_reached` stop states (max-node headroom + auto-grow mitigation landed;
      forced-cap runtime probe verified live `max_nodes_autogrow` transition).
- [x] Enable passive crawl operation on runtime boot so crawler routing starts without
      manual control actions.
- [ ] Confirm global radar surfaces non-provisional live crawl evidence after recovery
      using model-flagged downstream pages (not raw feed/watchlist source URLs).
- [x] Ensure global radar default behavior excludes provisional seed-only rows from the
      surfaced threat list.
- [x] Add a health gate for "seed-only global output" to prevent silent stale operation.

## Acceptance Criteria

- Repeated runs on unchanged inputs produce identical ordering and atoms.
- Rate limits and cooldown prevent spam refetch loops.
- Diff/manifest touches are visible through named queries.
- Every extracted claim is traceable to a fetched resource hash.
