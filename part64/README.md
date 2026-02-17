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

Web Graph Weaver crawler service:
- `cd code && npm install`
- `npm run weaver`
- Status: `http://127.0.0.1:8793/api/weaver/status`
- WebSocket: `ws://127.0.0.1:8793/ws`
- Integration guide: `WEB_GRAPH_WEAVER.md`

Useful PM2 controls:
- `python -m code.world_pm2 status`
- `python -m code.world_pm2 restart`
- `python -m code.world_pm2 stop`

Canonical audio:
- `artifacts/audio/*.wav`
