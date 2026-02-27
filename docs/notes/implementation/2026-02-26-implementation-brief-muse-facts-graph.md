---
title: "Implementation Brief: Muse, Facts, and Graph Grounding"
summary: "Implementation-focused brief for grounded muse tools, facts snapshots, and canonical graph integration."
category: "implementation"
created_at: "2026-02-26T21:00:45"
original_filename: "2026.02.26.21.00.45.md"
original_relpath: "docs/notes/implementation/2026.02.26.21.00.45.md"
tags:
  - implementation
  - muse
  - facts
---

You are an implementation agent working inside: part64/ (ημ — Operation Mindfuck — Part 64).
Goal: implement features (1)–(7) below with additive, test-backed changes.

──────────────────────────────────────────────────────────────────────────────
Captured constraints / axioms (do NOT ask the user to restate these)
──────────────────────────────────────────────────────────────────────────────
A1 (Axiom): Keep a single canonical graph (“one true graph”). Represent different
    semantics via node roles + edge kinds, not separate parallel graphs.
A2 (Requirement): Web integration must not be spammy: add a refractory/cooldown
    mechanism for URL-triggered crawling and cap per-tick triggers.
A3 (Requirement): “Win/Loss” semantics exist for daimoi: success reaching a target
    node deposits positive signal; failure within time deposits negative signal.
A4 (Requirement): External reality contract: web resources create two node types:
    (a) URL nodes, (b) Resource/Text nodes. Resource nodes link to URL nodes.
A5 (Requirement): Build toward “more than a toy”: Muse responses must be grounded
    in simulation state + fact extraction + logic checks (even if minimal at first).
A6 (Constraint): Prefer additive changes; keep existing runtime and tests passing
    (notably: python -m code.tests.test_world_web_pm2).

──────────────────────────────────────────────────────────────────────────────
(7) The LLM Muse — “chat backed by simulation + facts + logic”
──────────────────────────────────────────────────────────────────────────────
Context:
- Frontend already calls POST /api/muse/message.
- Server routes /api/muse/message → MuseRuntimeManager.send_message(...)
  with tool_callback=self._muse_tool_callback and reply_builder=self._muse_reply_builder.

Deliverables:
7.1 Add a new Muse tool: facts_snapshot
    - Trigger conditions:
      - explicit command: user message starts with “/facts”
      - implicit: if the user asks “what’s true / what changed / what do we know /
        cite / sources / why / explain / show graph” (token heuristic is fine)
    - Output:
      - Return a compact payload summary (counts, top nodes, top edges, top urls,
        recently fetched resources, etc.)
      - Store full snapshot on disk in part64/world_state/ (or an existing state dir),
        and return a reference (path + hash) in the tool result.

7.2 Add a new Muse tool: graph_query
    - Trigger:
      - message starts with “/graph …”
      - or heuristic: message contains “neighbors / path / connected / related / cluster”
    - Behavior:
      - Parse minimal intent from text (no heavy NLP required).
      - Execute deterministic queries against the canonical NexusGraph:
        - neighbors(node_id, kind?)
        - search(label substring)
        - url_status(url)
        - resource_for_url(url)
        - recently_touched_nodes(limit)
      - Return results as a small JSON object to be injected into the Muse context.

7.3 “Verified reply” mode (lightweight but real)
    - When facts_snapshot or graph_query ran this turn:
      - Reply builder must inject a short “grounding block” into the system prompt:
        - FACTS: bullet list of relevant facts (IDs + labels + short fields)
        - DERIVATIONS: outputs from graph_query (paths, neighbor sets)
        - UNKNOWN: explicit gaps (if user asks something not supported)
      - Instruct the LLM: “Answer using ONLY FACTS/DERIVATIONS. If not present, say unknown.”
    - Do not overpromise “proof”; just enforce: answer must be traceable to extracted facts.

7.4 Optional (but preferred): integrate logic layer stub without full Prolog/Datalog yet
    - Provide a small deterministic “logic validator” module:
      - validates invariants on facts (e.g., no dangling edges, URL nodes must have url)
      - validates provenance (resource nodes must cite source url)
    - Return violations as part of facts_snapshot.

7.5 Small-model-friendly
    - Keep prompt + tool result small and structured.
    - Prefer short lists, top-K, and stable ordering.

Implementation anchors:
- backend: part64/code/world_web/server.py
  - extend _muse_tool_callback to support:
    - facts_snapshot
    - graph_query
- backend: part64/code/world_web/muse_runtime.py
  - extend _tool_requests() to detect “/facts”, “/graph”, plus simple heuristics
- backend: new module recommended:
  - part64/code/world_web/fact_extraction.py (or facts.py)
  - part64/code/world_web/graph_query.py
- frontend: should already display Muse chat; only adjust if you add new UI affordances.

──────────────────────────────────────────────────────────────────────────────
Repo map / likely insertion points for the whole 1–7 feature set
──────────────────────────────────────────────────────────────────────────────
Daimoi dynamics + collisions:
- part64/code/world_web/daimoi_probabilistic.py
  - Nexus collision path: “Special Nexus handling: collidable routing node.”
  - Add win/loss bookkeeping hooks here (python backend).
- part64/code/world_web/c_double_buffer_backend.py
  - C backend particle rows already include graph_node_id/route_node_id/is_nexus flags.
  - If win/loss must work for C backend too, add outcome detection in simulation layer.

Simulation loop + field deposits:
- part64/code/world_web/simulation.py
  - Where normalized_field_particles are formed and NooiField is deposited.
  - Best place to add:
    - daimoi path tracking ring buffer (backend-agnostic)
    - win/loss deposition logic into NooiField (backend-agnostic)
    - interaction triggers to the crawler weaver when a daimoi hits a URL node

Canonical graph + crawler projection:
- part64/code/world_web/simulation.py
  - _build_weaver_field_graph_uncached(): projects weaver nodes into crawler_nodes
  - _build_canonical_nexus_graph(): merges file_graph + crawler_graph (+ logical_graph)
  - _NEXUS_ROLE_MAP: extend with new node_type mappings as needed

Weaver service (crawler):
- part64/code/web_graph_weaver.js
  - Existing endpoints include:
    - POST /api/weaver/entities/interact  { url, delta, source }
    - POST /api/weaver/control, /seed, etc
  - Prefer NOT to rewrite the crawler internals unless required.
  - Instead: tighten your *projection contract* + your *simulation triggers*.

Frontend chat exists:
- part64/frontend/src/components/Panels/Chat.tsx
- It already routes to /api/muse/message and renders replies.

──────────────────────────────────────────────────────────────────────────────
Acceptance criteria (must be demonstrably true)
──────────────────────────────────────────────────────────────────────────────
AC1 Daimoi path tracking exists:
- Each daimoi (record == “ημ.daimoi-probabilistic.v1”) maintains a bounded trail:
  - last N positions (N configurable; default 24).
- Trail is available to win/loss deposition (even if not rendered in UI).

AC2 Win deposition works:
- Define “win” deterministically (pick ONE, document it in code):
  - For crawler presence: successful interaction == collision/selection of a URL node
    that causes a weaver interaction trigger to succeed (or be accepted).
  - For general daimoi: win == reaching a target node (graph_node_id matches some target)
    OR contacting the correct presence anchor (best_presence == target), depending on mode.
- On win:
  - deposit positive trail into NooiField (and/or another signal field if present)
  - emit a structured event into world_state/ (JSONL is fine)

AC3 Loss deposition works:
- Add a TTL/deadline for daimoi searches (configurable).
- On loss:
  - deposit negative trail into NooiField
  - emit a structured event (JSONL)
  - recycle/respawn daimoi cleanly (no unbounded growth)

AC4 Web nodes satisfy the 2-node-type contract in the canonical graph:
- URL nodes: represent URLs (stable id, url in provenance/extension).
- Resource/Text nodes: represent fetched documents/text excerpts (content_hash/text_excerpt_hash).
- For each fetched resource:
  - resource_node —(refers_to|source_of)—> url_node
  - resource_node —(links_to)—> url_nodes extracted from that resource
- Enforce refractory:
  - repeated triggers for same URL are rate-limited (use weaver cooldown + local cap).

AC5 Fact extraction exists:
- There is a module that extracts a stable facts snapshot from the canonical graph:
  - outputs JSON (and optionally a Datalog facts file later)
  - includes provenance for web facts (source url)
  - stable ordering + hashes for determinism

AC6 Muse can answer grounded questions:
- Add Muse tools:
  - /facts returns facts snapshot summary
  - /graph runs deterministic graph queries
- When those tools run, Muse reply must be constrained to returned facts/derivations.

AC7 Tests / runnable verification:
- Must pass:
  - python -m code.tests.test_world_web_pm2
- Add at least one new unit test each for:
  - daimoi trail buffer bounds
  - weaver projection creating resource/url nodes + edges
  - muse tool routing for /facts and /graph

Manual verification checklist:
- docker compose up --build
- open runtime UI: http://127.0.0.1:8787/
- confirm weaver status: http://127.0.0.1:8787/weaver/api/weaver/status
- in Muse chat:
  - send “/facts” → receive snapshot summary
  - send “/graph neighbors <some node id>” → get deterministic output
- observe crawler triggers:
  - when crawler daimoi hits a url node, server calls weaver/entities/interact
  - repeated hits do not spam (cooldown works)

──────────────────────────────────────────────────────────────────────────────
Definition of Done
──────────────────────────────────────────────────────────────────────────────
- All acceptance criteria AC1–AC7 met.
- Changes are additive; existing behavior preserved unless explicitly improved.
- New behavior is observable:
  - events logged,
  - facts snapshots saved,
  - muse grounded replies.
- No runaway resource usage (bounded trails, bounded per-tick triggers).
