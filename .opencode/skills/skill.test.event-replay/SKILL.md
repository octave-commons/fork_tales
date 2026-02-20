---
id: skill.test.event-replay
type: skill
version: 1.0.0
tags: [test, replay, determinism]
embedding_intent: canonical
---

# Event Replay

Intent:
- Rebuild projected state from logs and assert deterministic equivalence.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.test.event-replay)
  (domain test)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
