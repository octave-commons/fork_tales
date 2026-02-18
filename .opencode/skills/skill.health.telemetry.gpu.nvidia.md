---
id: skill.health.telemetry.gpu.nvidia
type: skill
version: 1.0.0
tags: [health, telemetry, gpu, nvidia]
embedding_intent: canonical
---

# NVIDIA GPU Telemetry

Intent:
- Collect NVIDIA-specific GPU metrics through approved vendor APIs.
- Map vendor counters to normalized GPU schema.

Operational anchors:
- Emit capability matrix when NVML or driver paths are unavailable.
- Do not attempt privileged tuning controls.
