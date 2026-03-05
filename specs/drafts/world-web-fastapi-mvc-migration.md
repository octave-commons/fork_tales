# World Web FastAPI MVC Migration

Status: draft
Priority: high
Owner: OpenCode

## Problem

`part64/code/world_web/server.py` contains a single `WorldHandler` class that has grown into a god object with thousands of lines of mixed concerns (HTTP transport, websocket streaming, runtime state/cache, endpoint routing, serialization, and process orchestration).

Measured hotspots in current file:

- `WorldHandler._handle_websocket` (`part64/code/world_web/server.py:4534`) ~1975 lines
- `WorldHandler.do_GET` (`part64/code/world_web/server.py:6516`) ~1984 lines
- `WorldHandler.do_POST` (`part64/code/world_web/server.py:8523`) ~1971 lines
- Route condition checks in handler methods: ~98

## Goals

1. Remove the monolithic request-handler class as the primary implementation surface.
2. Move to FastAPI-first transport and explicit MVC decomposition.
3. Keep endpoint compatibility (`/api/*`, `/ws`, static routes) during migration.
4. Preserve runtime/cache semantics and existing test behavior while slicing.

## Non-Goals

- No endpoint contract redesign in this migration.
- No payload schema changes unless required by bug fixes.
- No broad frontend behavioral changes.

## Requirements

- Preserve existing route paths and status code behavior.
- Preserve websocket protocol compatibility for simulation stream messages.
- Preserve existing runtime cache semantics and fallback headers.
- Keep legacy transport callable during migration window for rollback safety.

## Proposed MVC Decomposition

### Models

- `world_web/mvc/models.py`
  - request profile models (method/path/query/body)
  - response envelope models (status/body/headers/content-type)
  - runtime dependency container (part/vault roots, queue, chamber, trackers)

### Views

- `world_web/mvc/views.py`
  - JSON response builder
  - byte response builder
  - stream/chunk response helpers
  - websocket message framing adapters

### Controllers

- `world_web/mvc/controllers/catalog_controller.py`
- `world_web/mvc/controllers/simulation_controller.py`
- `world_web/mvc/controllers/muse_controller.py`
- `world_web/mvc/controllers/meta_controller.py`
- `world_web/mvc/controllers/presence_controller.py`
- `world_web/mvc/controllers/websocket_controller.py`

Each controller owns route-level decision logic and delegates to existing `*_utils.py` modules for domain operations.

### Transport

- `world_web/fastapi_app.py`
  - FastAPI app/routers as primary runtime transport
  - lifecycle startup/shutdown hooks for runtime dependencies
  - websocket route for `/ws`

- Legacy fallback wrapper remains temporarily for compatibility while route slices migrate.

## Phased Plan

### Phase 1 (now)

- Freeze `server.py` growth policy in `AGENTS.md`.
- Capture migration plan/spec and inventory hotspot spans.
- Set FastAPI transport as default runtime mode (legacy explicit opt-in).

### Phase 2

- Extract `/api/simulation*` GET/POST routes to `simulation_controller.py`.
- Extract runtime simulation bootstrap orchestration to dedicated service module.
- Keep legacy route delegation only for unported paths.

### Phase 3

- Extract `/api/catalog*`, `/api/meta*`, `/api/muse*`, `/api/presence*` route families.
- Move request/response serialization to views module.

### Phase 4

- Extract websocket flow (`/ws`) into websocket controller module.
- Keep wire format and chunking semantics stable.

### Phase 5

- Remove monolithic `WorldHandler` implementation.
- Leave only compatibility shim (or remove entirely if all tests migrate).

## Risks

- Regression risk is highest for websocket stream patch semantics and simulation cache/fallback behavior.
- Legacy tests currently instantiate `make_handler(...)` directly; compatibility shims are required during transition.
- Route ordering/precedence differences between if-chain and router dispatch can alter behavior.

## Verification

- Unit/route tests:
  - `python -m pytest part64/code/tests/test_world_web_pm2.py -q`
  - `python -m pytest part64/code/tests/test_world_web_ws.py -q`
  - `python -m pytest part64/code/tests/test_world_web_bootstrap_cache.py -q`
- Runtime probes:
  - `GET /` -> 200
  - `GET /api/catalog` -> 200
  - `GET /api/simulation?perspective=hybrid&payload=trimmed` -> 200
  - `WS /ws` -> 101

## Open Questions

- Keep `make_handler(...)` for tests only, or replace tests with ASGI client harness immediately?
- Should legacy transport remain available by flag after Phase 5, or be removed entirely?

## Existing Issues / PR

- Existing issue/PR links: none recorded in repository metadata at drafting time.

## Definition of Done

- `WorldHandler` monolith removed as primary implementation path.
- FastAPI app is default and serves all current routes.
- MVC modules own route families with clear responsibility boundaries.
- Regression suites above pass.
- `receipts.log` updated with migration evidence.
