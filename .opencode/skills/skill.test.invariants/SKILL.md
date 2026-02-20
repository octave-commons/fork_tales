---
id: skill.test.invariants
type: skill
version: 1.0.0
tags: [test, invariants]
embedding_intent: canonical
---

# Test Invariants

Intent:
- Encode non-negotiable system invariants as automated checks.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.test.invariants)
  (domain test)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
