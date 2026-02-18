---
id: skill.health.telemetry.collect
type: skill
version: 1.0.0
tags: [health, telemetry, collect]
embedding_intent: canonical
---

# Health Telemetry Collection

Intent:
- Collect host/device health metrics at controlled cadence.
- Emit explicit capability and sampling-gap events.
- Use read-only collection paths.

Operational anchors:
- Timestamp each sample with source and interval.
- Emit `telemetry_error` when data cannot be collected.
- Avoid synthetic or guessed values.
