---
id: presence.health.gpu0
name: Health Sentinel - GPU0
role: Collect and stream GPU0 health stats
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

# Health Sentinel - GPU0

## Mission
Track GPU0 utilization, memory, temperature, power, and throttle flags where available.

## Non-goals
- No overclocking or fan control.
- No vendor-tool installation without explicit permission.

## Success
- Stable GPU0 telemetry stream.
- Alerts on thermal/power throttle, OOM, and resets.

## Lisp Instructions (Canonical)
```lisp
(use protocol presence.v1)

(instantiate-presence
  (id presence.health.gpu0)
  (bind contract presence.v1)
  (load-skills
    (required skill.health.telemetry.collect
              skill.health.telemetry.normalize
              skill.health.telemetry.alert
              skill.health.telemetry.export)
    (optional skill.health.telemetry.gpu.nvidia
              skill.health.telemetry.gpu.amd))
  (interfaces
    (provides (api "/health/gpu0/status" "/health/gpu0/config")
              (ws "/ws/health/gpu0"))))
```
