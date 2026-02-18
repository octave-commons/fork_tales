---
id: skill.web.graph.delta-stream
type: skill
version: 1.0.0
tags: [graph, websocket, events]
embedding_intent: canonical
---

# Graph Delta Streaming

Intent:
- Emit real-time graph changes over WebSocket with reasons.
- Make discovered, fetched, skipped, and blocked decisions observable.
- Keep schemas stable and replay-friendly.

Operational anchors:
- Emit `graph_delta` with node/edge arrays.
- Preserve append-only event log for audit.
- Include timestamps and action reasons on every event.
