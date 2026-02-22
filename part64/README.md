# ημ — Operation Mindfuck — Part 64

Contents:
- Two seeded audio renders (canonical WAV + MP3 convenience)
- Cover art + storyboard + particle field
- Lyrics (EN/JP) + dialog + Gates of Truth announcement
- Append-only constraints update + world map note
- Deterministic receipt generator + regression test
- Docker-first runtime + browser world view server (PM2 fallback)

Run the determinism test:
- `python -m code.tests.test_sonify_determinism`

Run the world runtime integration tests:
- `python -m code.tests.test_world_web_pm2`

Run in Docker (preferred; world + IO + web graph weaver + Chroma):
- `docker compose up --build`
- If ports are already in use, run: `ETA_MU_GATEWAY_PORT=18887 ETA_MU_WEAVER_PORT=18997 docker compose up --build`
- Runtime (nginx gateway): `http://127.0.0.1:8787/`
- Catalog (gateway): `http://127.0.0.1:8787/api/catalog`
- Docker simulation dashboard: `http://127.0.0.1:8787/dashboard/docker`
- Simulation Workbench & Benchmarking: `http://127.0.0.1:8787/dashboard/bench`
- Simulation Profile Portal: `http://127.0.0.1:8787/dashboard/profile?id=<sim_id>`
- Docker simulation API: `http://127.0.0.1:8787/api/docker/simulations`
- WebSocket (gateway): `ws://127.0.0.1:8787/ws`
- Weaver status (direct): `http://127.0.0.1:8793/api/weaver/status`
- Weaver status (via gateway): `http://127.0.0.1:8787/weaver/api/weaver/status`
- Optional TTS proxy target: set `TTS_BASE_URL` (default `http://127.0.0.1:8788` inside container)

Long-running embedding benchmark container (websocket live stream + simple UI):
- Start: `docker compose -f docker-compose.embed-bench.yml up --build`
- UI: `http://127.0.0.1:18890/`
- WebSocket stream: `ws://127.0.0.1:18890/ws`
- API state: `http://127.0.0.1:18890/api/state`
- Tune backend/model with env vars:
  - `EMBED_BENCH_BACKEND` (`openvino` default)
  - `EMBED_BENCH_MODEL` (`nomic-embed-text` default)
  - `OPENVINO_EMBED_ENDPOINT` (default `http://host.docker.internal:18000/v1/embeddings`)
  - `OLLAMA_BASE_URL` (default `http://host.docker.internal:11435`)

Comprehensive model benchmark runner (same compose stack, runner profile):
- Start one-shot run: `docker compose -f docker-compose.embed-bench.yml --profile runner run --rm model-bench-runner`
- Default suite: `scripts/benchmark_suites/whisper_openvino_starter.json`
- Override suite/output:
  - `MODEL_BENCH_SUITE=/workspace/scripts/benchmark_suites/universal_starter.json`
  - `MODEL_BENCH_OUTPUT=/results/model-bench.latest.json`
- Output artifact: `part64/runs/model-bench/*.json`
- Docs: `MODEL_BENCH_RUNNER.md`, `WHISPER_BENCHMARK.md`

**For detailed operational instructions, see [SIMULATION_WORKFLOW.md](SIMULATION_WORKFLOW.md).**

Docker simulation discovery contract for experiment stacks:
- Add labels to each simulation runtime container:
  - `io.fork_tales.simulation=true`
  - `io.fork_tales.simulation.role=experiment`
- Attach simulation containers to network: `eta-mu-sim-net`
- Dashboard auto-discovers running simulations by labels first, then runtime-name fallback.
- Dashboard surfaces per-container CPU, memory, and PID telemetry from Docker stats.
- For tight resource control, define per-service limits in compose:
  - `cpus`
  - `mem_limit` + `memswap_limit`
  - `pids_limit`
- Meta cognition API surface for operating unstable experiments:
  - `GET /api/meta/overview`
  - `GET|POST /api/meta/notes`
  - `GET|POST /api/meta/runs`
  - `POST /api/meta/objective/enqueue`

Muse song lab (parallel tuned simulations for play-song behavior):
- Start 3 tuned runtimes (`baseline`, `chaos`, `stability`) + shared Chroma:
  - `python scripts/muse_song_lab_ctl.py start`
- Start only selected runtimes to reduce load further:
  - `python scripts/muse_song_lab_ctl.py start --runtimes baseline,chaos`
- Inspect runtime + resource status:
  - `python scripts/muse_song_lab_ctl.py status`
- Endpoints:
  - baseline: `http://127.0.0.1:19877/`
  - chaos-tuned: `http://127.0.0.1:19878/`
  - stability-tuned: `http://127.0.0.1:19879/`
- Run cross-runtime task comparison:
  - `python scripts/muse_song_lab_ctl.py bench --regimen world_state/muse_song_training_regime.json`
- Run end-to-end NPU+learning evaluation (NPU checks, environment stimulus, latency benchmark, training, song benchmark):
  - `python scripts/muse_song_lab_ctl.py eval --runtimes baseline,chaos,stability --rounds 4 --output ../.opencode/runtime/sim_learning_eval.latest.json`
- Built-in command probes now include simple modality commands:
  - `Play Music` (expects audio selection)
  - `Open Image` (expects image selection)
- Override/extend task battery by editing:
  - `world_state/muse_song_training_regime.json`
- Stop lab stack:
  - `python scripts/muse_song_lab_ctl.py stop`

Semantic routing training circumstances (concept-seed presences + classifier baseline):
- Circumstances file (images + text seeds + classifier + noise policy):
  - `world_state/muse_semantic_training_circumstances.json`
- Run iterative training/eval cycle against runtime (creates muse if missing, updates workspace pins, emits report):
  - `python scripts/muse_semantic_training_lab.py --runtime http://127.0.0.1:8787`
- Or run across selected song-lab runtimes:
  - `python scripts/muse_song_lab_ctl.py train --runtimes baseline,chaos,stability`
- Output report artifact (default):
  - `../.opencode/runtime/muse_semantic_training.latest.json`
- The script posts aggregate metrics to `POST /api/meta/runs` by default.
- Docker simulation dashboard now visualizes these metrics in **Training Charts** under Meta Operations:
  - `http://127.0.0.1:8787/dashboard/docker`

Run world as local PM2 daemon (legacy fallback) and open in browser:
- `python -m code.world_pm2 start --host 127.0.0.1 --port 8787`

Run complete real-time dashboard across all parts:
- `python -m code.world_web --part-root ./ --vault-root .. --host 127.0.0.1 --port 8791`
- Dashboard: `http://127.0.0.1:8791/`
- Live catalog API: `http://127.0.0.1:8791/api/catalog`
- WebSocket feed: `ws://127.0.0.1:8791/ws`
- Simulation API: `http://127.0.0.1:8791/api/simulation`
- Combined audio stream: `http://127.0.0.1:8791/stream/mix.wav`

Dashboard now includes a WebGL renderer consuming websocket simulation messages from `/ws`.

World Log live signal feeds (optional, append-only runtime logs):
- Read merged event stream: `http://127.0.0.1:8787/api/world/events?limit=180`
- Runtime log files:
  - `.opencode/runtime/wikimedia_stream.v1.jsonl`
  - `.opencode/runtime/nws_alerts.v1.jsonl`
  - `.opencode/runtime/emsc_stream.v1.jsonl`
  - `.opencode/runtime/gibs_layers.v1.jsonl`
- Source toggles (conservative defaults):
  - Wikimedia EventStreams: `WIKIMEDIA_EVENTSTREAMS_ENABLED=1` (default on)
  - NWS alerts: `NWS_ALERTS_ENABLED=1`
  - EMSC earthquakes: `EMSC_STREAM_ENABLED=1`
  - NASA GIBS WMTS layers: `GIBS_LAYERS_ENABLED=1`
- NASA GIBS tuning:
  - `GIBS_LAYERS_CAPABILITIES_ENDPOINT`
  - `GIBS_LAYERS_TARGETS`
  - `GIBS_LAYERS_RATE_LIMIT_PER_POLL`
  - `GIBS_LAYERS_DEDUPE_TTL_SECONDS`
- Example with GIBS enabled:
  - `GIBS_LAYERS_ENABLED=1 GIBS_LAYERS_POLL_INTERVAL_SECONDS=30 python -m code.world_web --part-root ./ --vault-root .. --host 127.0.0.1 --port 8787`

Electron client (sandboxed renderer):
- From `part64/frontend`, install deps: `npm install`
- Dev loop (two terminals):
  - Terminal A: `npm run dev -- --host 127.0.0.1 --port 5173`
  - Terminal B (auto-restarts Electron main/preload on changes): `npm run electron:dev:watch`
- Launch built desktop client: `npm run electron:preview`
- Override runtime endpoints when needed:
  - `WORLD_RUNTIME_URL=http://127.0.0.1:8787 WEAVER_RUNTIME_URL=http://127.0.0.1:8793 npm run electron:start`
- Electron PM2 controls (from `part64/frontend`):
  - `npm run pm2:electron:start`
  - `npm run pm2:electron:status`
  - `npm run pm2:electron:stop`
  - `npm run pm2:electron:delete`
- Linux sandbox prerequisite:
  - If PM2 logs show `No usable sandbox`, configure one of:
    - Chromium SUID helper (`chrome-sandbox` owner root, mode `4755`)
    - or AppArmor/userns policy that allows unprivileged user namespaces.
  - Without one of those, Electron exits instead of running unsandboxed.

Web Graph Weaver crawler service:
- `cd code && npm install`
- `npm run weaver`
- Status: `http://127.0.0.1:8793/api/weaver/status`
- WebSocket: `ws://127.0.0.1:8793/ws`
- Entity runtime status: `http://127.0.0.1:8793/api/weaver/entities`
- Experimental embedding mode defaults to `nomic-embed-text` with forced small context via compose env (`OLLAMA_EMBED_FORCE_NOMIC=1`, `OLLAMA_EMBED_NUM_CTX=512`).
- Example cross-reference crawl seeds:
  - `https://arxiv.org/list/cs.AI/recent`
  - `https://en.wikipedia.org/wiki/Artificial_intelligence`
- Integration guide: `WEB_GRAPH_WEAVER.md`

Embedding provider options (GPU/NPU experimental routing):
- Inspect current provider config:
  - `GET http://127.0.0.1:8787/api/embeddings/provider/options`
- Switch to local GPU (Ollama embeddings):
  - `POST /api/embeddings/provider/options` with `{"preset":"gpu_local","ollama_embed_force_nomic":true}`
- Switch to local NPU (OpenVINO endpoint):
  - `POST /api/embeddings/provider/options` with `{"preset":"npu_local","openvino_endpoint":"http://host.docker.internal:18000/v1/embeddings"}`
- If OpenVINO endpoint requires auth, include one of:
  - `"openvino_bearer_token":"<token>"` (sends `Authorization: Bearer <token>`)
  - `"openvino_api_key":"<key>","openvino_api_key_header":"X-API-Key"`
  - `"openvino_auth_header":"Authorization: Bearer <token>"` for custom header control
- Hybrid auto mode (prefer NPU, then fallback):
  - `POST /api/embeddings/provider/options` with `{"preset":"hybrid_auto"}`

Force/verify NPU embeddings on running runtimes:
- Gateway runtime:
  - `python scripts/ensure_npu_embeddings.py --runtime gateway=http://127.0.0.1:8787`
- Song-lab runtimes:
  - `python scripts/ensure_npu_embeddings.py --runtime song-baseline=http://127.0.0.1:19877 --runtime song-chaos=http://127.0.0.1:19878 --runtime song-stability=http://127.0.0.1:19879`
- If your OpenVINO proxy requires auth, export `OPENVINO_EMBED_API_KEY` or `OPENVINO_EMBED_BEARER_TOKEN` (or keep `PROXY_API_KEY` in `docker-llm-proxy/.env` and the helper scripts auto-detect it).

Useful local PM2 controls (fallback runtime path):
- `python -m code.world_pm2 status`
- `python -m code.world_pm2 restart`
- `python -m code.world_pm2 stop`

Canonical audio:
- `artifacts/audio/*.wav`
