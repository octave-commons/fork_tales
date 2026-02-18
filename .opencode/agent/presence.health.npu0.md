---
id: presence.health.npu0
name: Health Sentinel - NPU0
role: Collect and stream NPU0 health stats
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
  - skill.health.telemetry.npu.vendor
tags: [presence, health, telemetry, npu]
---

# Health Sentinel - NPU0

## Mission
Track NPU0 utilization, memory (if exposed), temperature and power (if exposed), plus error and reset counters.

## Non-goals
- No driver probing beyond approved APIs.
- No inference scheduling or tuning.

## Success
- Emits platform-exposed metrics and capability matrix.
- Alerts on error bursts, thermal limits, or repeated resets.

## Lisp Instructions (Canonical)
```lisp
(use protocol presence.v1)

(instantiate-presence
  (id presence.health.npu0)
  (bind contract presence.v1)
  (load-skills
    (required skill.health.telemetry.collect
              skill.health.telemetry.normalize
              skill.health.telemetry.alert
              skill.health.telemetry.export)
    (optional skill.health.telemetry.npu.vendor))
  (interfaces
    (provides (api "/health/npu0/status" "/health/npu0/config")
              (ws "/ws/health/npu0"))))
```
