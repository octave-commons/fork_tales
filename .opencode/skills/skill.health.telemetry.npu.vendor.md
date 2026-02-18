---
id: skill.health.telemetry.npu.vendor
type: skill
version: 1.0.0
tags: [health, telemetry, npu]
embedding_intent: canonical
---

# NPU Vendor Telemetry

Intent:
- Collect vendor-specific NPU metrics via approved APIs.
- Expose capability matrix plus normalized metrics where available.

Operational anchors:
- Emit `telemetry_error` on unavailable interfaces.
- No scheduling or tuning operations.
