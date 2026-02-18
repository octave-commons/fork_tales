# Web Graph Weaver

Live, ethical crawl instrumentation service for real-time graph growth,
including arXiv paper/PDF citation edges and Wikipedia cross-reference edges.

## Run

- Install code dependencies: `cd code && npm install`
- Start crawler service: `npm run weaver`
- Service root: `http://127.0.0.1:8793/`
- Status API: `http://127.0.0.1:8793/api/weaver/status`
- Graph API: `http://127.0.0.1:8793/api/weaver/graph`
- Event API: `http://127.0.0.1:8793/api/weaver/events`
- WebSocket: `ws://127.0.0.1:8793/ws`

Starter seed set for arXiv + Wikipedia cross-linking:

- `https://arxiv.org/list/cs.AI/recent`
- `https://en.wikipedia.org/wiki/Artificial_intelligence`

The dashboard panel is integrated in the frontend (`WebGraphWeaverPanel`).

## Availability Notes

- When the world runtime starts (`code/world_web.py`), it attempts to bootstrap Web Graph Weaver automatically (configurable with `WEAVER_AUTOSTART`).
- The panel tries multiple local endpoints (`window hostname`, `127.0.0.1`, `localhost`) and shows actionable offline guidance when all are unreachable.
- Override endpoint host/port with env vars:
  - `WEAVER_HOST` (default loopback unless runtime host is non-local)
  - `WEAVER_PORT` (default `8793`)

## Architecture

## Service Layer

- `code/web_graph_weaver.js`
- Async controlled frontier crawler with domain-level guards
- REST control/status endpoints + WebSocket event stream

## Core Loop

`discover -> validate -> fetch -> parse -> extract links -> normalize -> add graph -> repeat`

Every decision emits an event (`node_discovered`, `fetch_started`, `fetch_completed`, `fetch_skipped`, `robots_blocked`, `compliance_update`, `graph_delta`, `crawl_state`).

## Graph Model

Nodes:

- `url` node (`url:<normalized-url>`)
- `domain` node (`domain:<hostname>`)
- `content` node (`content:<mime-type>`)

Edges:

- `hyperlink` (`url -> url`)
- `canonical_redirect` (`url -> url`)
- `domain_membership` (`url -> domain`)
- `content_membership` (`url -> content`)
- `citation` (`arXiv paper -> cited arXiv paper`)
- `paper_pdf` (`arXiv paper -> arXiv PDF`)
- `wiki_reference` (`Wikipedia article -> linked Wikipedia article`)
- `cross_reference` (`arXiv <-> Wikipedia`)

Storage:

- In-memory map for live traversal
- Append-only logs:
  - `world_state/web_graph_weaver.events.jsonl`
  - `world_state/web_graph_weaver.graph_delta.jsonl`
- Snapshot file:
  - `world_state/web_graph_weaver.snapshot.json`

## robots.txt + Rate Limiting

- robots policy fetched per origin and cached (`ROBOTS_CACHE_TTL_MS`)
- `Disallow` and `Allow` are evaluated conservatively
- `crawl-delay` is honored per domain (minimum default delay retained)
- nofollow links are discovered but not traversed
- `429/503` responses trigger exponential backoff per domain
- domain-level `nextAllowedAt` timestamps gate fetch starts
- host-level active request cap is enforced (`max_requests_per_host`)
- node cooldown gate blocks repeat fetches for 10 minutes by default (`WEAVER_NODE_COOLDOWN_MS`)

## Entity-Driven Crawl Flow

- Entity walkers move across known URL nodes and emit `entity_move`, `entity_arrived`, `entity_visit`, and `entity_tick` events.
- Node interactions raise `activation_potential`; when threshold is reached and cooldown is clear, the URL is enqueued.
- Any known URL node can be visited by entities; crawl direction becomes interaction-driven over graph links.
- Interactions are available through API (`POST /api/weaver/entities/interact`) and panel controls.

Defaults:

- node cooldown: `10 minutes`
- activation threshold: `1.0`
- interaction delta: `0.35`
- entity walkers: `4`

## arXiv + Wikipedia Knowledge Mapping

- arXiv pages:
  - detect canonical arXiv IDs from `/abs/...` and `/pdf/...`
  - emit `paper_pdf` edges to canonical PDFs
  - emit `citation` edges for linked/mentioned arXiv papers
  - emit `cross_reference` edges when arXiv pages link to Wikipedia articles
- Wikipedia pages:
  - emit `wiki_reference` edges for in-article links (`/wiki/...`)
  - emit `cross_reference` edges when Wikipedia cites arXiv
- Status metrics now expose semantic edge totals (`citation_edges`, `cross_reference_edges`, etc.)

## Ethical Guardrails

1. robots.txt is evaluated before fetch.
2. crawl-delay is honored and never bypassed.
3. explicit user-agent is sent on every request.
4. opt-out endpoint exists: `POST /api/weaver/opt-out`.
5. no bypass mode is implemented.
6. conservative default concurrency and delay are used.

## Control Endpoints

- `POST /api/weaver/control`
  - `start`, `pause`, `resume`, `stop`
  - supports `max_per_host`, `entity_count`
- `POST /api/weaver/seed`
- `POST /api/weaver/opt-out`
- `DELETE /api/weaver/opt-out`
- `GET /api/weaver/entities`
- `POST /api/weaver/entities/control`
  - `start`, `pause`, `resume`, `stop`, `configure`
- `POST /api/weaver/entities/interact`

## Link Text Analysis (LLM)

- When pages are visited/fetched, text excerpts are analyzed and written to node metadata.
- Events emitted:
  - `link_text_analysis_started`
  - `link_text_analyzed`
  - `link_text_analysis_failed`
- Default LLM target is Ollama-compatible endpoint:
  - `WEAVER_LLM_BASE_URL` (default `http://127.0.0.1:11434`)
  - `WEAVER_LLM_MODEL` (default `qwen2.5:3b-instruct`)
  - `WEAVER_LLM_ENABLED` (default enabled)

## Dashboard Features

- real-time graph canvas growth
- domain/depth/compliance visual differentiation
- traversal/event stream log
- status cards: crawl rate, frontier, compliance, robots blocked, etc.
- domain filter + path highlighting via node selection
