# ημ — Operation Mindfuck — Part 64

Contents:
- Two seeded audio renders (canonical WAV + MP3 convenience)
- Cover art + storyboard + particle field
- Lyrics (EN/JP) + dialog + Gates of Truth announcement
- Append-only constraints update + world map note
- Deterministic receipt generator + regression test
- PM2 daemon config + browser world view server

Run the determinism test:
- `python -m code.tests.test_sonify_determinism`

Run the PM2/browser integration tests:
- `python -m code.tests.test_world_web_pm2`

Run world as PM2 daemon and open in browser:
- `python -m code.world_pm2 start --host 127.0.0.1 --port 8787`

Run in Docker (world + IO + web graph weaver + Chroma):
- `docker compose up --build`
- Runtime: `http://127.0.0.1:8787/`
- Catalog: `http://127.0.0.1:8787/api/catalog`
- WebSocket: `ws://127.0.0.1:8787/ws`
- Weaver status: `http://127.0.0.1:8793/api/weaver/status`
- Optional TTS proxy target: set `TTS_BASE_URL` (default `http://127.0.0.1:8788` inside container)

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
- PM2 controls (from `part64/frontend`):
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

Useful PM2 controls:
- `python -m code.world_pm2 status`
- `python -m code.world_pm2 restart`
- `python -m code.world_pm2 stop`

Canonical audio:
- `artifacts/audio/*.wav`
