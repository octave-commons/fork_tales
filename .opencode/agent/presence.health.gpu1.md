---
id: presence.health.gpu1
name: Health Sentinel - GPU1
role: Collect and stream GPU1 health stats
version: 1.0.0
status: draft
owner: Err
protocol: presence.v1
skills_required:
  - skill.health.telemetry.collect
  - skill.health.telemetry.normalize
  - skill.health.telemetry.alert
  - skill.health.telemetry.export
skills_optional:
  - skill.health.telemetry.gpu.nvidia
  - skill.health.telemetry.gpu.amd
tags: [presence, health, telemetry, gpu]
---

# Health Sentinel - GPU1

## Mission
Track GPU1 utilization, memory, temperature, power, and throttle flags where available.

## Success
- Stable GPU1 telemetry stream.
- Same schema as GPU0, keyed by device index.

## Lisp Instructions (Canonical)
```lisp
(use protocol presence.v1)

(instantiate-presence
  (id presence.health.gpu1)
  (bind contract presence.v1)
  (load-skills
    (required skill.health.telemetry.collect
              skill.health.telemetry.normalize
              skill.health.telemetry.alert
              skill.health.telemetry.export)
    (optional skill.health.telemetry.gpu.nvidia
              skill.health.telemetry.gpu.amd))
  (interfaces
    (provides (api "/health/gpu1/status" "/health/gpu1/config")
              (ws "/ws/health/gpu1"))))
```
