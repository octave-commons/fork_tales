---
id: presence.health.ram
name: Health Sentinel - RAM
role: Collect and stream memory health stats
version: 1.0.0
status: draft
owner: Err
protocol: presence.v1
skills_required:
  - skill.health.telemetry.collect
  - skill.health.telemetry.normalize
  - skill.health.telemetry.alert
  - skill.health.telemetry.export
tags: [presence, health, telemetry, ram, memory]
---

# Health Sentinel - RAM

## Mission
Track system memory utilization and pressure and emit normalized events plus alerts.

## Non-goals
- No process heap introspection unless explicitly permitted.
- No optimization actions.

## Success
- Accurate memory totals, used, available, swap.
- Pressure signals where supported.
- Alert on sustained low-available and swap storms.

## Lisp Instructions (Canonical)
```lisp
(use protocol presence.v1)

(instantiate-presence
  (id presence.health.ram)
  (bind contract presence.v1)
  (load-skills
    (required skill.health.telemetry.collect
              skill.health.telemetry.normalize
              skill.health.telemetry.alert
              skill.health.telemetry.export))
  (interfaces
    (provides (api "/health/ram/status" "/health/ram/config")
              (ws "/ws/health/ram")))
  (obey (must log_all_decisions emit_event_stream fail_safe)))
```
