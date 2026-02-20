---
id: skill.health.telemetry.alert
type: skill
version: 1.0.0
tags: [health, telemetry, alert]
embedding_intent: canonical
---

# Health Telemetry Alerting

Intent:
- Evaluate sustained threshold breaches and emit deterministic alerts.
- Distinguish warning vs critical severity.

Operational anchors:
- Use rolling windows for sustained conditions.
- Include threshold and observed value in alert payload.
- Emit clear recovery events after normalization.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.health.telemetry.alert)
  (domain health)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
