---
id: skill.ux.user-journeys
type: skill
version: 1.0.0
tags: [ux, journeys, flows]
embedding_intent: canonical
---

# User Journeys

Intent:
- Define happy path and failure mode flows from user goals.
- Keep each flow tied to real system states and events.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.ux.user-journeys)
  (domain ux)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
