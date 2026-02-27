---
title: "Phased Execution Plan with Stop Lines"
summary: "Stepwise implementation plan for daimoi outcomes, web contracts, facts snapshots, and grounded muse replies."
category: "planning"
created_at: "2026-02-27T00:36:56"
original_filename: "2026.02.27.00.36.56.md"
original_relpath: "docs/notes/2026.02.27.00.36.56.md"
tags:
  - execution-plan
  - stop-lines
  - runtime
---

──────────────────────────────────────────────────────────────────────────────
Execution plan (phased, with hard “stop lines”)
──────────────────────────────────────────────────────────────────────────────
Principle (Constraint): Avoid building the city sim now. Prove “external reality”
by making the crawler presence + web graph contract + grounded muse work first.
City sim becomes a later fact-source, not the proof of reality.

Phase 0 — Orientation & invariants (DO NOT CODE YET)
0.1 Confirm the runtime entry points:
    - where daimoi are spawned/updated
    - where collisions/TTL are currently detected (if at all)
    - where NooiField is deposited
    - where the weaver/crawler service is invoked (if present)
    - where chat/muse requests enter
0.2 Write down (in a new doc file) the current identifiers/roles:
    - existing node roles (nexus/presence/etc)
    - existing edge roles in canonical graph merge
    - existing “daimoi record” schema in the code (names, fields)
0.3 Add NO new features until you can point to:
    - the single function where NooiField deposit happens
    - the single place where canonical graph is built/merged

Stop line: if you cannot find these, do not guess. Locate them.

Phase 1 — Daimoi path tracking (bounded trail)
1.1 Add a bounded trail buffer per daimoi (ring buffer).
    Data structure requirement:
      - fixed N length (default 24)
      - stores (tick, x, y, vx, vy) OR (tick, cell_x, cell_y, dir_x, dir_y)
      - O(1) append per tick per daimoi
      - no unbounded history anywhere

1.2 Decide where to store it:
    Option A (preferred): in Python “world_web” layer that already sees
      particle positions/velocities each tick.
    Option B: in native C backend only if Python cannot access consistent state.

1.3 Add unit test:
    - spawn a daimoi, step N+K ticks, assert:
      - trail length == N
      - oldest entries shift out
      - entries are deterministic under fixed seed

Stop line: Do not implement win/loss until trail exists and test passes.

Phase 2 — Win/Loss semantics + events (food/death)
2.1 Implement the win condition (Requirement):
    - For general daimoi: win = “successful interaction with a graph node”
      (choose one deterministic predicate and document it):
        - collision with nexus OR presence proxy
        - or “graph-edge messaging encounter” with a node
    - For crawler daimoi: win = encountering a URL node AND scheduling fetch accepted.

2.2 Implement loss condition (Requirement):
    - death = TTL exceeded without a win
    - TTL must be enforced deterministically in tick loop.

2.3 Emit events (append-only JSONL):
    - daimoi_spawn
    - daimoi_candidates (optional but useful)
    - daimoi_collision (raw)
    - daimoi_timeout
    - daimoi_outcome (derived: food/death + reason + target)

2.4 Add tests:
    - collision triggers food only when allowed/feasible (policy/capacity stub ok)
    - TTL triggers death exactly at deadline
    - events are written in stable order; replayable

Stop line: Do not touch Nooi outcome deposition until win/loss events exist.

Phase 3 — Outcome-conditioned deposition (trail reinforcement)
3.1 Implement “ambient deposit” vs “reinforcement deposit”
    Constraint: Keep physics simple; do not change integrator behavior beyond adding
    the reinforcement deposit into NooiField.

3.2 Implement:
    - on food: deposit forward trail vectors (last N steps)
    - on death: deposit inverse trail vectors (negate direction)
    - magnitude is configurable (default 1.0)
    - deposit must respect per-layer routing if NooiField supports layers.

3.3 Add tests:
    - ensure deposit magnitude changes field in expected direction
    - ensure negative deposit does not explode (clamp/normalize)
    - ensure deposits are bounded per tick (no accidental O(N^2))

Stop line: Do not connect crawler triggers until reinforcement exists.

Phase 4 — Web graph contract (URL + Resource nodes only)
4.1 Implement the “two-node-kind” contract (A4):
    Node roles:
      - web:url
      - web:resource
    Edge roles:
      - web:source_of   (resource -> url it came from)
      - web:links_to    (resource -> urls found within)
      - (optional) web:canonicalizes_to (url -> url) for canonicalization merges

4.2 Canonicalization (Requirement):
    - strip fragments
    - normalize scheme/host
    - resolve relative URLs against base
    - stable URL node ID = hash(canonical_url)

4.3 Resource node identity:
    - resource ID = hash(canonical_url + fetched_ts) OR hash(content_hash)
    - store:
      - canonical_url
      - fetched_ts
      - content_hash
      - text_excerpt_hash
      - title (optional)
      - embedding_ref/tagging (optional)

4.4 Edge creation logic:
    - fetch resource R at URL U
      - ensure url node U exists
      - create resource node R
      - create R --web:source_of--> U
      - extract urls {U1..Un}
      - ensure each Ui exists (dedupe)
      - create R --web:links_to--> Ui for each
    - If Ui existed already: only add new edge; do not duplicate nodes.

4.5 Refractory / cooldown (A2, A3):
    Data required per URL node:
      - next_allowed_fetch_ts
      - fail_count
      - last_status (ok/error/http code)
    Rules:
      - if now < next_allowed_fetch_ts: do not fetch
      - on success: next_allowed_fetch_ts = now + cooldown_ok (e.g., 10m)
      - on failure: exponential backoff (e.g., 1m, 5m, 30m, 2h) with cap
      - enforce global concurrency (e.g., 2-8)
      - enforce per-tick trigger cap (e.g., max 4 fetches scheduled/tick)

4.6 Add integration test (no external network):
    - use a tiny local HTTP server fixture with 2 HTML pages:
      - page A links to B and itself
      - page B links to A and C
    - crawler fetch A
      - assert: nodes created for A,B,C URLs
      - assert: resource node created for A
      - assert: links_to edges created and deduped
    - re-trigger A within cooldown
      - assert: no new fetch scheduled
      - assert: a “cooldown_blocked” event recorded

Stop line: Don’t integrate LLM yet. Get deterministic crawling first.

Phase 5 — Fact extraction (Graph -> Facts snapshot for logic + muse)
5.1 Facts are a projection, not a second graph (Constraint A1).
    Implement a “facts snapshot” that extracts:
      - nodes (id, role, status, key properties)
      - edges (src, role, dst, key properties)
      - recent events (collision/timeout/outcome, fetch events)
      - web:url (canonical_url, cooldown, status)
      - web:resource (canonical_url, content_hash, fetched_ts)

5.2 Determinism requirement:
    - stable ordering (sort keys)
    - stable hashing:
      - snapshot_hash = sha256(canonical_json)
    - write snapshot to disk:
      - world_state/facts/<ts>_<hash>.json

5.3 “logic stub” (light but real):
    Implement a deterministic validator that checks invariants:
      - web:resource must have exactly one web:source_of edge
      - web:links_to edges must point to web:url nodes
      - url nodes must have canonical_url
      - cooldown fields must be sane
    Return violations in snapshot.

Stop line: Do not implement Prolog/Datalog engine yet unless you can keep it tiny.
You can add a minimal SWI-Prolog runner later, but snapshot comes first.

Phase 6 — Tighten webcrawler pseudo-autonomy (Presence behavior)
6.1 Implement a crawler presence loop:
    - has a seed list of URL nodes
    - spawns crawler daimoi that “seek” URL nodes
    - encounter triggers fetch scheduling
    - cooldown prevents spam
    - results create new URL nodes, which become candidates for exploration

6.2 Implement an exploration policy (keep it simple):
    - frontier = URL nodes with:
        - never fetched OR fetched long ago AND cooldown expired
    - prioritize by:
        - low fail_count
        - high degree (many resources link to it)
        - or random with temperature

6.3 Ensure homeostasis:
    - crawler presence must respect tick governor slack:
      - if slack negative: do not schedule work
      - if slack positive: schedule limited work
    - global crawler job queue bounded (drop or defer)

6.4 Add a “status endpoint / report” (or reuse existing):
    - queue length
    - active fetches
    - cooldown blocks count
    - last N fetched URLs
    - error rate

Stop line: Don’t add city sim. This is your first “external reality” proof.

Phase 7 — Muse (chat grounded in facts + deterministic queries)
7.1 Muse tool: /facts
    - returns summary + snapshot_hash + path
7.2 Muse tool: /graph
    - deterministic queries:
        - neighbors(node_id, edge_role?)
        - search(label/url substring)
        - url_status(url or url_id)
        - resource_for_url(url_id)
        - recently_updated(limit)
7.3 Muse response policy (Requirement A5):
    - When /facts or /graph used:
        - Muse must answer using ONLY the tool output.
        - If not present: “unknown” + suggest which query to run.
    - Provide receipts:
        - snapshot_hash
        - node/edge IDs used
        - query names called

7.4 Add two E2E tests:
    - chat: “/facts” returns snapshot info
    - chat: “/graph neighbors <node>” returns deterministic result

──────────────────────────────────────────────────────────────────────────────
Implementation constraints (how to work inside the repo)
──────────────────────────────────────────────────────────────────────────────
I1: Use narrow discovery: glob → grep → read → edit → re-read to confirm.
I2: Keep changes additive; don’t refactor large subsystems.
I3: Prefer full-file replacements for edited files.
I4: All new behavior must be observable via:
    - JSONL events, and/or
    - facts snapshots saved to disk, and/or
    - status endpoint output.
I5: All loops bounded: no unbounded queues, no unbounded trails, no infinite crawls.

──────────────────────────────────────────────────────────────────────────────
Deliverables (files + tests + demo)
──────────────────────────────────────────────────────────────────────────────
D1: Daimoi trail buffer implemented + unit test
D2: Win/loss outcomes implemented + unit tests
D3: Outcome-conditioned Nooi deposit implemented + unit test
D4: Web graph contract implemented + integration test with local HTTP fixture
D5: Facts snapshot exporter implemented + invariants validator
D6: Crawler presence pseudo-autonomous loop + status report + cooldown behavior
D7: Muse tools /facts and /graph + grounded reply policy + E2E test

Demo commands (provide exact commands that work):
- start runtime
- run crawler with seed list (local fixture)
- open chat
- ask /facts and /graph and see grounded reply

At end, report:
- files written
- why safe
- how to verify quickly
