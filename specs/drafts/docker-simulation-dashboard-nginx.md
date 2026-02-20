# Docker Simulation Dashboard + Nginx Reverse Proxy

## Priority
- High

## Requirements
- Add a dashboard surface that shows currently running simulation containers in Docker.
- Keep the dashboard up to date as containers are added/removed during experiments.
- Introduce an nginx reverse proxy service in Docker for runtime access.
- Preserve runtime reliability and avoid adding non-deterministic dependencies.

## Open questions
- None currently; default behavior is label-first discovery with deterministic name fallback.

## Risks
- Docker socket may not be available inside runtime container; dashboard must degrade gracefully.
- Container naming can vary between experiments; discovery must be explicit via labels when possible.
- WebSocket traffic should avoid unnecessary churn under high container counts.

## Existing issues
- None found in repository scan.

## Existing PRs
- None checked in local repository context.

## Complexity estimate
- Medium: backend API + websocket feed + static dashboard + docker compose/nginx wiring.

## Subtasks
1. Add a docker runtime discovery module for simulation containers and cluster awareness.
2. Expose docker simulation payload via API and websocket event stream.
3. Add a dedicated dashboard page that subscribes to websocket updates with polling fallback.
4. Add nginx reverse proxy config and service in compose.
5. Add simulation labels/network conventions in compose for cross-experiment visibility.
6. Add targeted tests for docker discovery/awareness logic.
7. Run focused test/build checks and append receipt.

## Candidate files
- `part64/code/world_web/docker_runtime.py`
- `part64/code/world_web/server.py`
- `part64/code/tests/test_docker_runtime.py`
- `part64/code/static/docker_simulations_dashboard.html`
- `part64/docker-compose.yml`
- `part64/nginx/default.conf`
- `part64/README.md`
- `receipts.log`

## Definition of done
- API endpoint returns running simulation container inventory with awareness metadata.
- Dashboard page renders inventory and live-updates as containers change.
- Nginx proxy runs in compose and forwards websocket/API traffic.
- Compose conventions support discovery across experiment stacks.
- Targeted tests pass and receipt entry records file refs + verification commands.

## Decisions captured during implementation
- Added `part64/code/world_web/docker_runtime.py` with Docker socket discovery, awareness clustering, cache fallback, and stable fingerprint generation.
- Added API route `GET /api/docker/simulations` and websocket event type `docker_simulations` in world runtime server.
- Added standalone runtime-served dashboard at `/dashboard/docker` with websocket subscription and polling fallback.
- Added nginx reverse proxy service in `part64/docker-compose.yml` with websocket support and `/weaver/` forwarding.
- Added explicit simulation labels + shared `eta-mu-sim-net` network in benchmark/song-lab compose files for cross-stack visibility.

## Verification log
- `python -m py_compile code/world_web/docker_runtime.py code/world_web/server.py code/tests/test_docker_runtime.py` (workdir=`part64`)
- `python -m pytest code/tests/test_docker_runtime.py -q` (workdir=`part64`)
- `python -m pytest code/tests/test_world_web_pm2.py::test_world_payload_and_artifact_resolution -q` (workdir=`part64`)
- `docker compose config` (workdir=`part64`)
- `docker compose -f docker-compose.sim-slice-bench.yml config` (workdir=`part64`)
- `docker compose -f docker-compose.muse-song-lab.yml config` (workdir=`part64`)
- `docker run --rm -v /home/err/devel/vaults/fork_tales/part64/nginx/default.conf:/etc/nginx/conf.d/default.conf:ro nginx:1.27-alpine nginx -t`

## Meta operations extension
- Added `part64/code/world_web/meta_ops.py` append-only stores for:
  - failure/observation notes (`.opencode/runtime/meta_notes.v1.jsonl`)
  - training/evaluation run tracking (`.opencode/runtime/meta_runs.v1.jsonl`)
- Extended Docker snapshot lifecycle signals with restart/OOM/health state classification for graceful failure observation.
- Added API surface:
  - `GET /api/meta/overview`
  - `GET|POST /api/meta/notes`
  - `GET|POST /api/meta/runs`
  - `POST /api/meta/objective/enqueue`
- Extended dashboard with Meta Operations panels for failure feed, notes capture, objective queueing, and run tracking.

## Process hardening extension
- Added websocket client capacity guard in runtime server with env-tunable max slots (`RUNTIME_WS_MAX_CLIENTS`) and explicit `503 websocket_capacity_reached` rejection payload when saturated.
- Added runtime guard telemetry synthesis (`eta-mu.runtime-health.v1`) exposed via:
  - `GET /api/runtime/health`
  - websocket `runtime_health` events
- Added pressure-aware websocket loop behavior:
  - degrades broadcast/simulation/docker refresh intervals when guard mode is `degraded` or `critical`
  - optionally skips expensive simulation ticks under critical pressure (`RUNTIME_GUARD_SKIP_SIMULATION_ON_CRITICAL`)
  - emits `simulation_guard` witness events when ticks are skipped
- Added focused tests in `part64/code/tests/test_runtime_hardening.py` for guard-state classification, websocket slot accounting, and runtime-health payload shape.

## Hardening verification log
- `python -m py_compile code/world_web/server.py code/tests/test_runtime_hardening.py` (workdir=`part64`)
- `python -m pytest code/tests/test_runtime_hardening.py code/tests/test_docker_runtime.py code/tests/test_meta_ops.py -q` (workdir=`part64`) -> blocked by pre-existing syntax error in `part64/code/world_web/simulation.py:6722` (`SyntaxError: unmatched '}'`) during test collection.
