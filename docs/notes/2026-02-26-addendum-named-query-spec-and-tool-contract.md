---
title: "Addendum: Named Query Spec and Tool Contract"
summary: "Defines deterministic named graph queries, facts snapshot contracts, and muse tool behavior."
category: "planning"
created_at: "2026-02-26T21:08:15"
original_filename: "2026.02.26.21.08.15.md"
original_relpath: "docs/notes/2026.02.26.21.08.15.md"
tags:
  - addendum
  - query-spec
  - muse-tools
---

──────────────────────────────────────────────────────────────────────────────
(ADDENDUM 2) Named Query Spec + Tool Contract + Minimal APIs
──────────────────────────────────────────────────────────────────────────────
Purpose: prevent query drift and keep Muse grounded. Muse may ask arbitrary
questions in natural language, but it may only answer using outputs from the
named queries below (or explicitly say “unknown”).

Constraint: Do not implement ad-hoc graph querying endpoints. Only these named
queries exist in v1. Add more only when needed, and document them here.

──────────────────────────────────────────────────────────────────────────────
Q0) Named query transport contract (backend internal)
──────────────────────────────────────────────────────────────────────────────
All named queries are invoked via a single internal function:

  graph_query.run(name: str, args: dict) -> dict

Rules:
- name must match one of the whitelisted names below
- args must be JSON-serializable
- output must be deterministic (stable sort)
- output must be bounded (top-K) and include "truncated": true|false if needed

Muse tools call graph_query.run(...) ONLY. No direct graph access from Muse.

──────────────────────────────────────────────────────────────────────────────
Q1) Query catalog (v1)
──────────────────────────────────────────────────────────────────────────────

(1) summary_state
Args:
  - none
Returns:
  - tick: int
  - counts:
      nodes_by_role: {role: count}
      edges_by_kind: {kind: count}
      daimoi_active: int
  - outcomes_window:
      food_count: int
      death_count: int
      rejected_count: int
      window_ticks: int
  - crawler:
      queue_len: int
      active_fetches: int
      cooldown_blocks: int
      last_fetches: [ {url_id, res_id, ts, status} ] (top 10)

(2) neighbors
Args:
  - node_id: string
  - edge_kind: string|null  (optional filter)
  - direction: "out"|"in"|"both" (default "both")
  - limit: int (default 50)
Returns:
  - node_id
  - neighbors: [ {other_id, edge_kind, direction, other_role, other_label?} ]
  - truncated: bool

(3) search_nodes
Args:
  - query: string (substring match; case-insensitive)
  - role: string|null
  - limit: int (default 50)
Returns:
  - hits: [ {node_id, role, label?, canonical_url?} ]
  - truncated: bool

(4) url_status
Args:
  - url_id: string OR canonical_url: string
Returns:
  - url_id
  - canonical_url
  - next_allowed_fetch_ts
  - last_fetch_ts
  - fail_count
  - last_status
  - inbound_links: int  (degree from web:resource -> web:url)
  - fetched_resources: [res_id] (top 10)

(5) resource_summary
Args:
  - res_id: string
Returns:
  - res_id
  - canonical_url
  - fetched_ts
  - content_hash
  - link_count
  - links_to: [url_id] (top 50)
  - title? / tags? / embedding_ref? if present

(6) recent_outcomes
Args:
  - window_ticks: int (default 600)
  - limit: int (default 50)
Returns:
  - events: [
      {tick, daimoi_id, outcome:"food"|"death"|"rejected",
       target_id?, reason_code?, reinforcement?}
    ]
  - truncated: bool

(7) explain_daimoi
Args:
  - daimoi_id: string
Returns:
  - daimoi_id
  - owner_presence
  - intent
  - born_tick
  - ttl_ticks
  - last_known_tick
  - outcome: "food"|"death"|"unknown"
  - outcome_tick?
  - target_id?
  - reject_reason?  (if rejected)
  - reinforcement: {sign, magnitude, mode, n_steps} | null
  - trail_summary:
      n_steps_recorded
      start_pos, end_pos
      last_dirs: [ {vx,vy} ] top 5
  - receipts:
      snapshot_hash OR event_ids used

──────────────────────────────────────────────────────────────────────────────
Q2) Muse tool contract (HTTP endpoint behavior)
──────────────────────────────────────────────────────────────────────────────

Existing endpoint:
- POST /api/muse/message
Input:
  { "thread_id": "...", "message": "...", "user_ts": "...", "mode": "default" }

Behavior changes required:
- If message begins with "/facts":
    - run facts_snapshot() and return immediate tool output + allow muse to respond
- If message begins with "/graph":
    - parse: "/graph <query_name> <args...>" OR a small DSL
    - run graph_query for that query and return tool output
- Otherwise:
    - enqueue a muse job (async)
    - muse may decide to call named queries if heuristics trigger (optional)
    - muse reply MUST include receipts if any tool was used

Output:
  { "ok": true,
    "accepted": true,
    "job_id": "...",           # if async
    "immediate": {...} | null, # if tool run immediately
    "reply": {...} | null      # if immediate reply
  }

Constraint: Muse must never claim knowledge not present in query outputs.

──────────────────────────────────────────────────────────────────────────────
Q3) Facts snapshot module contract (Graph -> Facts)
──────────────────────────────────────────────────────────────────────────────
Implement:

  facts_snapshot.build(graph, tick, now_ts) -> {
      snapshot_hash,
      path,
      summary,
      violations
  }

Must:
- write canonical JSON to disk:
    world_state/facts/<ts>_<hash>.json
- canonical JSON must contain:
    - tick, ts, snapshot_hash
    - nodes: only top-K by role OR only those relevant to recent events
    - edges: only top-K OR those touching included nodes
    - recent_events: last N (bounded)
    - crawler stats: cooldown/queue summary
    - violations: invariant checker results
- summary returned to Muse must be <= a few KB

Invariant checks (v1 minimal):
- web:resource must have exactly one web:source_of edge
- web:links_to edges must target web:url nodes
- web:url nodes must have canonical_url
- cooldown fields must be sane (next_allowed_fetch_ts >= last_fetch_ts)
- no duplicate node IDs with conflicting roles

──────────────────────────────────────────────────────────────────────────────
Q4) Outcome Judge API (core of Slice A)
──────────────────────────────────────────────────────────────────────────────
Define a single module boundary so sim/crawler/muse all agree:

  outcome_judge.on_tick(tick, events, graph, now_ts) -> {
      outcomes: [...],          # food/death/rejected
      reinforcements: [...],    # deposits to apply
      graph_patches: [...]      # edges/properties to commit
  }

Events input includes:
- collision events
- timeout events
- url encounter events (crawler)
- capacity snapshots (optional)
- policy config (static or fetched)

Output structures (bounded):
Outcome:
  { tick, daimoi_id, outcome, target_id?, reason_code? }
Reinforcement:
  { tick, daimoi_id, sign:"+|-" , magnitude, mode:"trail_forward|trail_inverse", n_steps }
GraphPatch:
  { op:"add_edge|set_prop|add_node", ... }

Constraint: outcome_judge is deterministic. Given same tick+events+graph snapshot,
it returns identical outputs.

──────────────────────────────────────────────────────────────────────────────
Q5) Trail buffer API (single source of truth)
──────────────────────────────────────────────────────────────────────────────
Define minimal class:

  TrailBuffer(N)
    .push(daimoi_id, tick, x, y, vx, vy)
    .get(daimoi_id) -> list[steps] (oldest..newest, length<=N)
    .summary(daimoi_id) -> {n, start,end,last_dirs}

Storage constraint:
- bounded memory: O(num_daimoi * N)
- cleanup on daimoi termination (food/death) to avoid leaks

──────────────────────────────────────────────────────────────────────────────
Q6) Nooi reinforcement deposit API
──────────────────────────────────────────────────────────────────────────────
Do NOT overload the existing “ambient deposit” function. Add explicit API:

  nooi.apply_reinforcement(trail_steps, sign, magnitude, layer, clamp=..., normalize=...)

Rules:
- deposit per step:
    dir = normalize([vx,vy])
    if sign == "-": dir = -dir
    deposit into cell for (x,y) or (cell_x,cell_y)
- magnitude distributed:
    either uniform, or tapered (recent steps stronger); choose one and document
- bounded per tick:
    max total reinforcement steps across all daimoi per tick
    (e.g., MAX_REINFORCEMENT_STEPS_PER_TICK)

──────────────────────────────────────────────────────────────────────────────
Q7) Crawler trigger API (URL encounter -> fetch scheduling)
──────────────────────────────────────────────────────────────────────────────
Define a single scheduling function:

  crawler.schedule_fetch(url_id, canonical_url, source, now_ts, tick) -> {
      accepted: bool,
      reason: "ok|cooldown|queue_full|rate_limited|invalid_url",
      next_allowed_fetch_ts: float,
      job_id?: string
  }

This must enforce refractory/cooldown and global bounds.

On accepted:
- emit event web_fetch_scheduled
- add edge crawl:triggered_fetch (source -> url_id) with tick

On blocked:
- emit event web_cooldown_blocked (or queue_full/rate_limited)
- add edge crawl:cooldown_block (source -> url_id) with reason

──────────────────────────────────────────────────────────────────────────────
Q8) “Stop the toy” verification narrative (agent must include in report)
──────────────────────────────────────────────────────────────────────────────
The agent’s final report must include a short paragraph proving non-toyness:
- show that a URL encounter causes a real fetch
- show that fetch creates web:url and web:resource nodes and edges
- show that outcomes (food/death) cause reinforcement deposits
- show that Muse answers using /facts + /graph only, with snapshot_hash receipts

──────────────────────────────────────────────────────────────────────────────
END ADDENDUM 2
──────────────────────────────────────────────────────────────────────────────
