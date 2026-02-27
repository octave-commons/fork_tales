---
title: "Agent Prompt: Daimoi, Crawler, and Muse Integration"
summary: "Full implementation prompt covering win/loss semantics, web reality contract, fact extraction, and muse behavior."
category: "implementation"
created_at: "2026-02-26T20:21:48"
original_filename: "2026.02.26.20.21.48.md"
original_relpath: "docs/notes/implementation/2026.02.26.20.21.48.md"
tags:
  - implementation
  - agent-prompt
  - runtime
---

## Agent Prompt: Part64 — Daimoi Win/Loss, Web Reality Contract, Crawler Presence, Muse

You are an engineering agent working inside the `fork_tales` codebase, starting from the **part64 runtime** (Python + C/C++). Your goal is to turn the system from a “closed sim” into a **real-world-interacting, auditable coordination substrate** while keeping the physics model simple.

### Operator mode

* Make the smallest set of changes that accomplish the goal.
* Prefer **additive** changes over refactors.
* Prefer **full-file replacements** over diffs when editing files.
* After writing, re-open and confirm the touched regions.
* Provide at the end:

  * files written
  * why the change is safe
  * how to verify quickly (commands + expected output)

---

# 0) Extracted constraints / axioms (do not violate)

### C0 — Single-graph simplicity constraint

Maintain **one True Graph** and **one View Graph** (projection/compaction). Do **not** introduce multiple competing graph models. Represent different kinds of relationships and data as **node roles** and **edge roles**. Keep physics simple by excluding non-physical nodes from the integrator.

### C1 — External reality contract

The system must have an explicit contract with “external reality”:

* external events → graph updates
* graph state → extractable facts → logic queries → validated decisions
* LLM output must be grounded in logic-query results (not raw hallucination).

### C2 — Web nodes constraint

Web knowledge must be represented with exactly two core node kinds:

* **URL nodes**
* **Resource/Text nodes** (a fetched document)

Resources link to URL nodes via edges that represent hyperlinks discovered in the resource.

### C3 — Anti-spam constraint

Crawler must include a **refractory period / cooldown** mechanism per URL to avoid repeated fetch spam. Also implement backoff on errors.

### C4 — Outcome semantics constraint

Daimoi outcomes are defined as:

* **Win (“food”)**: successful interaction/collision with a target graph node (Nexus or Presence, and for crawler: URL node triggers retrieval)
* **Loss (“death”)**: failing to locate/reach a target within allotted time (TTL)

Win/loss must drive deposition into the Nooi Field using the **trail**.

### C5 — LLM “Muse” epistemic constraint

Muse is a Presence connected to a small LLM and must answer via:

* state/fact extraction + logic query results
* receipts (what facts/rules were used)
  Muse should respond asynchronously after some time (scheduled via tick governor slack or a queue).

---

# 1) Deliverables (what to implement)

## 1) Daimoi path tracking

Add per-daimoi path tracking sufficient to deposit trails:

* Maintain a ring buffer of the last **N** steps per daimoi:

  * either grid cell coordinates or normalized position (x,y) + velocity vector (vx,vy) + tick
* Keep memory bounded: ring buffer (fixed N) not unbounded history.
* Ensure this works for daimoi handled in the native C simulation path (if applicable): track at the boundary where you already have per-particle positions/velocities.

### Acceptance tests

* Spawn daimoi, advance simulation, confirm ring buffer populates and maintains fixed size N.
* Confirm trail buffer survives compaction or state sync as appropriate.

## 2) Win/loss deposition logic (Nooi Field)

Implement outcome-conditioned deposition in addition to the existing “ambient motion deposit”:

* Ambient: every tick, movement contributes to NooiField (already present).
* Outcome deposition:

  * **On win**: deposit the **forward trail** vectors for last N steps, weighted by magnitude.
  * **On loss**: deposit **inverse trail** vectors (negate direction), weighted by magnitude.
* Support per-layer deposition (e.g., based on owner/presence id) if NooiField is multi-layer.

### Acceptance tests

* Win produces a measurable increase in local field vector magnitude along trail direction.
* Loss produces a measurable increase in vector magnitude along opposite direction.
* Trail deposits obey cooldown if you apply it to deposition (optional); at minimum, obey TTL semantics.

## 3) Win condition for successful interaction with a graph node

Define and implement “success” consistently:

* A daimoi **wins** when it collides with a qualifying target (initially: Nexus or Presence).
* Emit a structured event into the graph/log indicating:

  * daimoi id, target id, tick, outcome=food, plus optional impact score.

For crawler presence:

* Collision/encounter with a **URL node** triggers a fetch job (see crawler section).

## 4) Loss condition for TTL expiration

Define and implement “death”:

* If TTL expires before a win condition occurs, emit event:

  * daimoi id, tick, outcome=death, reason=timeout.

Ensure TTL is enforced in the simulation loop or scheduler, not “best effort”.

## 5) Fact extraction from the knowledge graph

Build a minimal “Fact Export” system that converts graph state into a stable schema used by logic and muse.

### Requirement: two-layer epistemics

* **Observations**: append-only receipts (events, fetches, collisions, extracts)
* **Facts**: accepted/derived statements suitable for logic queries (can be computed from observations + graph state)

### Output formats (choose one, but implement cleanly)

* CSV tables in a directory (`facts/current/*.csv`) for Datalog engines, OR
* JSONL facts for Prolog ingestion + a small query adapter.

Minimum exported tables:

* `node(node_id, role, status, …)`
* `edge(src, role, dst, …)`
* `daimoi(daimoi_id, owner, intent, born_tick, ttl, …)`
* `event_collision(tick, daimoi_id, target_id, …)`
* `event_timeout(tick, daimoi_id, …)`
* `capacity(tick, target_id, cap, used)`
* `web_url(url_id, canonical_url, next_allowed_fetch_ts, …)`
* `web_resource(res_id, canonical_url, content_hash, fetched_ts, …)`
* `web_link(res_id, url_id)` (resource → url)

Also export derived outcomes:

* `food(tick, daimoi_id, target_id)`
* `death(tick, daimoi_id)`

### Acceptance tests

* Run one sim tick and confirm tables update deterministically.
* Given a fixed event log + seed, fact export should be replayable.

## 6) Tighten web crawler and validate it works pseudo-autonomously

The crawler must be integrated as a Presence/process that is triggered by URL nodes.

### Crawler graph representation (must implement exactly)

* Two node roles:

  * `web:url`
  * `web:resource` (text/document)
* When a web resource is fetched:

  1. Create or update a `web:resource` node.
  2. Extract all URLs from the document.
  3. For each extracted URL:

     * Create a `web:url` node if new; otherwise reuse existing node.
     * Create an edge `web:links_to` (or similar) from the resource → url node.
* Dedupe:

  * Canonicalize URLs (strip fragments, normalize scheme/host, resolve relative URLs).
  * Use a stable node id scheme (hash of canonical URL recommended).

### Trigger mechanism (must support both)

When a daimoi from the crawler presence/process encounters a URL node:

* **Particle collision trigger**: if your sim models URL nodes as collidable proxies, collision causes fetch scheduling.
* **Graph-edge messaging trigger**: if encounter occurs via graph messaging, schedule fetch anyway.

### Refractory / cooldown (must implement)

Add per-URL throttling:

* `next_allowed_fetch_ts`
* exponential backoff on failures
* global concurrency limit and rate limit
* do not refetch within cooldown window even if encountered repeatedly

### Validation requirements

* Provide a seed URL list.
* Run crawler for a bounded time.
* Confirm graph grows:

  * new resource node
  * extracted url nodes
  * edges created, deduped correctly
* Confirm cooldown prevents spam of the same URL.
* Confirm it respects robots.txt (if your crawler claims to) and has a clear user-agent string.

### Optional but strongly preferred

* Embed/tag/analyze document text:

  * If embeddings exist in the system, attach an embedding reference or topic tags to the resource node.
  * If `qwen3-vl:2b` is available, use it only where appropriate (images/screenshots) and keep it bounded by governor slack.

## 7) LLM Muse Presence

Implement a Muse presence that answers chat questions **asynchronously** using only:

* fact export + logic query results + receipts.

### Muse workflow

1. User sends message in chat panel.
2. Message is queued as a “muse job”.
3. Muse waits for available compute budget (use tick governor slack if available).
4. Muse calls a fixed menu of **named logic queries** (not free-form queries).
5. Muse composes a response with:

   * answer
   * receipts: which facts and which rules/queries supported it
   * uncertainties explicitly labeled as uncertainty

### Named query menu (must implement)

Start with these:

* `explain_daimoi(daimoi_id)` → food/death, target, reasons, reinforcement
* `recent_outcomes(window_ticks, limit)` → list of food/death events
* `crawler_status()` → queue length, last fetches, cooldown stats
* `web_resource_summary(res_id|url_id)` → what was fetched, what links extracted
* `graph_summary(scope, n)` → counts + top nodes/edges by activity

### Acceptance tests

* Ask via chat: “Why did daimoi d:42 die?” → muse answers with receipts.
* Ask: “What did the crawler learn from URL X?” → muse references fetched resource node + extracted links, with cooldown status.

---

# 2) Implementation guidance (where to start)

1. Locate part64 runtime entry points:

* Simulation loop / tick governor
* Daimoi definition and updates
* NooiField deposition path
* Graph model and node/edge roles
* Presence runtime / process scheduler
* Existing web crawler code (if present)

2. Add path tracking at the lowest cost boundary:

* Prefer to track in Python if particles are managed there.
* If particle integration is in C, add minimal per-particle trail buffers in C, or export last-N positions from C to Python on demand.

3. Implement win/loss events first:

* Emit collision/timeout events consistently
* Only then add trail deposition

4. Implement crawler v1 as a Presence:

* URL node encounter schedules fetch jobs
* Fetch creates resource node + url nodes + edges
* Cooldown enforced

5. Implement fact export and logic adapter:

* Fact export should cover win/loss + web graph state
* Logic rules should produce deterministic decisions and explainability

6. Muse last:

* It is a client of named queries; it should not force changes to core runtime.

---

# 3) Definition of done checklist

* [ ] Daimoi ring-buffer trail exists (bounded, deterministic)
* [ ] Win condition emits food event; loss emits death event
* [ ] Win/loss produce directional trail deposits into NooiField
* [ ] Web graph uses exactly url/resource nodes; resource→url edges created and deduped
* [ ] Cooldown/backoff prevents URL spam; concurrency bounded
* [ ] Fact export produces stable tables/JSONL used by logic and muse
* [ ] Crawler can run pseudo-autonomously from seed URLs and grow graph
* [ ] Muse answers chat questions using only named logic queries and provides receipts
* [ ] Minimal tests exist:

  * unit: trail buffer + deposition
  * integration: crawl seed + dedupe + cooldown
  * E2E: chat → muse → receipts

---

# 4) Output expectations

At completion, provide:

* A short architecture note (1 page) describing:

  * graph roles used
  * fact schemas
  * named logic queries
  * how to run demo
* A demo script/command:

  * starts runtime
  * runs crawler presence with seeds
  * opens chat panel and shows muse answering:

    * “what got crawled”
    * “why did a daimoi win/lose”
* Files written list + verification steps.
