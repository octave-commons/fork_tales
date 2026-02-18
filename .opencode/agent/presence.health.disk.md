---
id: presence.health.disk
name: Health Sentinel - Disk
role: Collect and stream disk health stats
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
  - skill.health.telemetry.disk.smart
tags: [presence, health, telemetry, disk, io, smart]
---

# Health Sentinel - Disk

## Mission
Track disk space, IO, and optional SMART health and emit normalized deltas plus alerts.

## Non-goals
- No destructive disk operations.
- No SMART path when permissions are unavailable.

## Success
- Space and IO metrics per configured mount/device.
- Alerts for low space and abnormal IO latency.
- SMART summary when enabled.

## Lisp Instructions (Canonical)
```lisp
(use protocol presence.v1)

(instantiate-presence
  (id presence.health.disk)
  (bind contract presence.v1)
  (load-skills
    (required skill.health.telemetry.collect
              skill.health.telemetry.normalize
              skill.health.telemetry.alert
              skill.health.telemetry.export)
    (optional skill.health.telemetry.disk.smart))
  (interfaces
    (provides (api "/health/disk/status" "/health/disk/config")
              (ws "/ws/health/disk"))))
```
