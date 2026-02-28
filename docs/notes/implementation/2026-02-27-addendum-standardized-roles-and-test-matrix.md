---
title: "Addendum: Standardized Roles and Test Matrix"
summary: "Hardening spec for canonical graph roles, event names, allowed files, and test coverage."
category: "planning"
created_at: "2026-02-27T00:37:49"
original_filename: "2026.02.27.00.37.49.md"
original_relpath: "docs/notes/2026.02.27.00.37.49.md"
tags:
  - addendum
  - graph-contract
  - testing
---

──────────────────────────────────────────────────────────────────────────────
(ADDENDUM) Standardized role strings + minimal touched-file list + test matrix
──────────────────────────────────────────────────────────────────────────────

This addendum is a hardening layer for the task prompt. It prevents “semantic
drift” across subsystems by standardizing node/edge roles, event names, IDs,
and a minimal set of files the agent is allowed to touch.

──────────────────────────────────────────────────────────────────────────────
S0) Standard role strings (do not invent new ones unless absolutely necessary)
──────────────────────────────────────────────────────────────────────────────

Constraint: Single True Graph + Single View Graph.
Requirement: Different semantics live in node roles / edge kinds.

Node roles (canonical):
- presence              # existing
- nexus                 # existing
- daimoi                # existing
- web:url               # NEW (external reality anchor)
- web:resource           # NEW (fetched document/text)
- obs:event             # OPTIONAL (append-only receipts; can be file-based instead)
- fact                  # OPTIONAL (if you later choose Fact-as-node; not required now)

Edge kinds (canonical):
# Web contract edges
- web:source_of         # web:resource -> web:url (the URL fetched)
- web:links_to          # web:resource -> web:url (URLs extracted from resource)
- web:canonical_of      # web:url -> web:url (optional canonical redirect/merge)

# Outcome / learning edges (True Graph bookkeeping)
- sim:food              # daimoi -> target (nexus/presence/url)  (win)
- sim:death             # daimoi -> "timeout" or special node id (loss)
- sim:rejected          # daimoi -> target with code (policy/capacity/cooldown)

# Trigger edges (crawler)
- crawl:triggered_fetch # daimoi/presence -> web:url (when encounter schedules fetch)
- crawl:cooldown_block  # daimoi/presence -> web:url (blocked by refractory)

Properties (required on certain node roles):
web:url:
  - canonical_url: string
  - next_allowed_fetch_ts: float epoch seconds
  - fail_count: int
  - last_fetch_ts: float epoch seconds (optional)
  - last_status: string (ok|error|http_*)
web:resource:
  - canonical_url: string (the source URL)
  - fetched_ts: float epoch seconds
  - content_hash: string (sha256)
  - text_excerpt_hash: string (sha256) (optional but recommended)
  - title: string (optional)
  - embedding_ref: string (optional)
  - tags: [string] (optional)

Constraint: Non-physical node roles must not participate in integrator.
- web:url and web:resource should be treated as graph nodes, but do NOT need to
  become moving particles unless you already have collidable proxies. If you do
  create proxies, keep them pinned/static.

──────────────────────────────────────────────────────────────────────────────
S1) Stable ID conventions (keep deterministic; avoid duplicates)
──────────────────────────────────────────────────────────────────────────────

URL node id:
- url_id = "url:" + sha256(canonical_url)[:16]
- canonical_url MUST strip fragments, normalize scheme/host, resolve relative URLs.

Resource node id:
Option A (recommended for dedupe across identical content):
- res_id = "res:" + sha256(content_bytes)[:16]
Option B (recommended if you want per-fetch snapshots):
- res_id = "res:" + sha256(canonical_url + ":" + fetched_ts_iso)[:16]

Constraint: choose ONE option and document it in code.
If you choose Option A, you can still keep fetch history via obs/events.

──────────────────────────────────────────────────────────────────────────────
S2) Event names (append-only JSONL, stable keys, replayable)
──────────────────────────────────────────────────────────────────────────────

Sim/daimoi:
- daimoi_spawn
- daimoi_candidates
- daimoi_collision
- daimoi_timeout
- daimoi_outcome          # (food|death|rejected) + reason + target

Crawler:
- web_fetch_scheduled     # accepted into queue
- web_fetch_started
- web_fetch_completed     # includes resource_id, url_id, content_hash
- web_fetch_failed        # includes status + backoff next_allowed_fetch_ts
- web_extract_links       # includes count + (optional) sample urls
- web_cooldown_blocked

Muse grounding:
- muse_job_enqueued
- muse_job_started
- muse_job_completed      # snapshot_hash + queries_used

Constraint: events must be bounded in size (top-K lists, hashes, counts).

──────────────────────────────────────────────────────────────────────────────
S3) Minimal touched-file list (agent must not roam outside these without need)
──────────────────────────────────────────────────────────────────────────────

The agent is allowed to touch ONLY these files initially:

Core sim + deposition:
- part64/code/world_web/simulation.py               # trail integration + outcome hooks + weaver projection
- part64/code/world_web/nooi.py                     # add explicit reinforcement deposit API if needed
- part64/code/world_web/daimoi_probabilistic.py     # TTL / outcome emit integration if this is the right locus
- part64/code/world_web/daimoi_collision_semantics.py  # if collision win/loss defined here
- part64/code/world_web/c_double_buffer_backend.py  # if needed for trail state bridging from C backend
- part64/code/world_web/native/c_double_buffer_sim.c # ONLY if Python cannot track trail; keep changes minimal

Crawler:
- part64/code/web_graph_weaver.js                   # tighten behavior + ensure endpoints stable
- part64/code/world_web/presence_runtime.py         # add crawler presence loop + job queue (bounded)

Facts + queries:
- part64/code/world_web/projection.py               # if best place to add facts snapshot export
- NEW: part64/code/world_web/facts_snapshot.py      # (agent may create)
- NEW: part64/code/world_web/graph_query.py         # (agent may create)

Muse:
- part64/code/world_web/muse_runtime.py
- part64/code/world_web/server.py

Tests (must update/add; keep deterministic, no external net):
- part64/code/tests/test_nooi_field.py
- part64/code/tests/test_cdb_nooi.py
- part64/code/tests/test_daimoi_probabilistic.py
- part64/code/tests/test_world_web_pm2.py
- part64/code/tests/test_muse_runtime.py
- part64/code/tests/weaver_semantic_references.test.js

Constraint: Avoid touching frontend unless absolutely required for /facts or /graph UX.
The backend endpoints should be enough.

──────────────────────────────────────────────────────────────────────────────
S4) Test matrix (what to add, where, and what “pass” means)
──────────────────────────────────────────────────────────────────────────────

T1 Trail buffer bounds (unit)
- File: test_daimoi_probabilistic.py (or new test_daimoi_trail.py)
- Ensure:
  - trail length clamps at N
  - deterministic order under seed
  - low overhead per tick

T2 Reinforcement deposit polarity (unit)
- File: test_nooi_field.py and/or test_cdb_nooi.py
- Ensure:
  - food deposit increases vector magnitude in same direction
  - death deposit increases vector magnitude in opposite direction
  - field remains bounded (normalize/clamp)

T3 Win/loss event emission (unit)
- File: test_world_web_pm2.py (or new test_outcomes.py)
- Ensure:
  - collision -> food only when allowed
  - TTL -> death exactly at deadline
  - events JSONL created and parseable

T4 Web contract graph shape (integration; no external network)
- Implement a local HTTP fixture server inside the test:
  - page A -> links to B + itself
  - page B -> links to A + C
- Ensure:
  - url nodes exist for A,B,C
  - resource node exists for fetched page
  - edges web:source_of and web:links_to created
  - dedupe works (no duplicate url nodes)
  - cooldown blocks repeated fetch scheduling

T5 Muse grounding (E2E-lite)
- File: test_muse_runtime.py
- Ensure:
  - /facts returns snapshot_hash + summary
  - /graph neighbors returns deterministic list
  - muse reply references only tool outputs (or says unknown)

──────────────────────────────────────────────────────────────────────────────
S5) Crawler trigger semantics (collision vs graph message)
──────────────────────────────────────────────────────────────────────────────

Constraint: support BOTH encounter modes.

Encounter mode A: particle collision
- If URL nodes have collidable proxies, collision event should carry:
  - daimoi_id, url_id, tick
- That should schedule fetch if cooldown allows.

Encounter mode B: graph-edge messaging
- When a crawler daimoi selects a URL node via graph query or edge traversal,
  treat that as an encounter and schedule fetch if cooldown allows.

Stop line: Don’t require particle collision to make crawler work. Graph message mode
must be sufficient for pseudo-autonomy even if physics proxies aren’t ready.

──────────────────────────────────────────────────────────────────────────────
S6) Homeostasis guardrails for new work (do NOT regress tick stability)
──────────────────────────────────────────────────────────────────────────────

Requirement: All new work respects the Tick Governor (slack_ms).
Rules:
- Never schedule more than MAX_FETCHES_PER_TICK (default 2–4).
- Never exceed MAX_CONCURRENT_FETCHES (default 2–8).
- If slack_ms < 0: only process essential sim; defer crawler/muse work.
- Bound all queues (crawler_job_queue_max, muse_job_queue_max).
- On overflow: drop low-priority jobs and emit an event.

──────────────────────────────────────────────────────────────────────────────
S7) Minimal “facts snapshot” shape (first non-toy grounding)
──────────────────────────────────────────────────────────────────────────────

facts_snapshot.json must include:
- snapshot_hash (sha256 of canonical JSON)
- ts, tick
- counts: nodes_by_role, edges_by_kind
- recent_events: last N outcome/crawl events (ids + short fields)
- web:
  - urls: top K by degree or recent touched (url_id, canonical_url, cooldown status)
  - resources: last K fetched (res_id, canonical_url, fetched_ts, link_count)
- invariants_violations: list (if any)

Constraint: stable ordering + top-K. Save full snapshot to disk and return hash.

──────────────────────────────────────────────────────────────────────────────
S8) Demo script expectation (agent must provide a runnable proof)
──────────────────────────────────────────────────────────────────────────────

Must provide a minimal “proof run” command sequence that demonstrates:
1) crawler fetches a seed URL (local fixture is OK)
2) graph contains url/resource nodes + edges
3) daimoi win/loss occurs and deposits reinforcement
4) /facts produces snapshot and /graph can query neighbors
5) muse replies grounded (receipts include snapshot_hash + queries_used)

──────────────────────────────────────────────────────────────────────────────
END ADDENDUM
──────────────────────────────────────────────────────────────────────────────
