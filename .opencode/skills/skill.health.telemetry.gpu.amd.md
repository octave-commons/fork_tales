---
id: skill.health.telemetry.gpu.amd
type: skill
version: 1.0.0
tags: [health, telemetry, gpu, amd]
embedding_intent: canonical
---

# AMD GPU Telemetry

Intent:
- Collect AMD-specific GPU metrics through approved ROCm or vendor APIs.
- Map vendor counters to normalized GPU schema.

Operational anchors:
- Emit capability matrix when stack support is missing.
- Keep collection read-only.
