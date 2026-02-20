---
id: skill.dev.graph-storage
type: skill
version: 1.0.0
tags: [dev, graph, storage]
embedding_intent: canonical
---

# Graph Storage

Intent:
- Persist graph nodes/edges with inspectable schema and migration discipline.
- Support in-memory runtime and durable snapshot/delta persistence.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.dev.graph-storage)
  (domain dev)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
