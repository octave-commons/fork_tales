---
id: skill.ux.component-specs
type: skill
version: 1.0.0
tags: [ux, ui, components]
embedding_intent: canonical
---

# Component Specs

Intent:
- Define component APIs, states, and event bindings.
- Require loading, empty, error, blocked, and paused state behavior.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.ux.component-specs)
  (domain ux)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
