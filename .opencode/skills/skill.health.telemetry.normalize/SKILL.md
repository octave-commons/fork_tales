---
id: skill.health.telemetry.normalize
type: skill
version: 1.0.0
tags: [health, telemetry, schema, normalize]
embedding_intent: canonical
---

# Health Telemetry Normalization

Intent:
- Normalize device/vendor metrics into stable schema keys.
- Preserve units and capability matrix for unsupported fields.

Operational anchors:
- Standardize keys and units before publish.
- Mark unavailable fields explicitly (`null` + capability flag).
- Keep schema version in every payload.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.health.telemetry.normalize)
  (domain health)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
