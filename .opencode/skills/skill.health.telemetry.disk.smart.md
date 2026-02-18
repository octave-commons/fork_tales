---
id: skill.health.telemetry.disk.smart
type: skill
version: 1.0.0
tags: [health, telemetry, disk, smart]
embedding_intent: canonical
---

# Disk SMART Telemetry

Intent:
- Collect SMART health signals when permissions and tooling allow.
- Fold SMART summaries into normalized disk health payloads.

Operational anchors:
- Treat SMART as optional capability.
- Emit capability and permission events when unavailable.
