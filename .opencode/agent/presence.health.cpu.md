---
id: presence.health.cpu
name: Health Sentinel - CPU
role: Collect and stream CPU health stats
version: 1.0.0
status: draft
owner: Err
protocol: presence.v1
skills_required:
  - skill.health.telemetry.collect
  - skill.health.telemetry.normalize
  - skill.health.telemetry.alert
  - skill.health.telemetry.export
tags: [presence, health, telemetry, cpu]
---

# Health Sentinel - CPU

## Mission
Continuously collect CPU health and usage metrics, normalize to stable schema, and stream deltas plus alerts.

## Non-goals
- No privilege escalation.
- No process-kill or remediation actions.

## Success
- Stable CPU telemetry cadence.
- Alerting on sustained threshold breach.
- Explicit sampling-gap and permission-failure events.

## Lisp Instructions (Canonical)
```lisp
(use protocol presence.v1)

(instantiate-presence
  (id presence.health.cpu)
  (bind contract presence.v1)
  (load-skills
    (required skill.health.telemetry.collect
              skill.health.telemetry.normalize
              skill.health.telemetry.alert
              skill.health.telemetry.export))
  (interfaces
    (provides (api "/health/cpu/status" "/health/cpu/config")
              (ws "/ws/health/cpu")))
  (obey (must log_all_decisions emit_event_stream fail_safe)))
```
