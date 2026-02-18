---
id: skill.health.telemetry.export
type: skill
version: 1.0.0
tags: [health, telemetry, export]
embedding_intent: canonical
---

# Health Telemetry Export

Intent:
- Export normalized telemetry and alerts to API, stream, and file sinks.
- Keep payloads audit-friendly and replayable.

Operational anchors:
- Append-only event log for telemetry decisions.
- Versioned schemas for downstream consumers.
- Explicit transport failure events.
