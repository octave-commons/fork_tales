# Web Graph Weaver

Live, ethical crawl instrumentation service for real-time graph growth.

## Run

- Install code dependencies: `cd code && npm install`
- Start crawler service: `npm run weaver`
- Service root: `http://127.0.0.1:8793/`
- Status API: `http://127.0.0.1:8793/api/weaver/status`
- Graph API: `http://127.0.0.1:8793/api/weaver/graph`
- Event API: `http://127.0.0.1:8793/api/weaver/events`
- WebSocket: `ws://127.0.0.1:8793/ws`

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
- `POST /api/weaver/seed`
- `POST /api/weaver/opt-out`
- `DELETE /api/weaver/opt-out`

## Dashboard Features

- real-time graph canvas growth
- domain/depth/compliance visual differentiation
- traversal/event stream log
- status cards: crawl rate, frontier, compliance, robots blocked, etc.
- domain filter + path highlighting via node selection
