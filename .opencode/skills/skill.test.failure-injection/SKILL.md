---
id: skill.test.failure-injection
type: skill
version: 1.0.0
tags: [test, failure, resilience]
embedding_intent: canonical
---

# Failure Injection

Intent:
- Inject controlled faults and assert expected fail-safe behavior.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.test.failure-injection)
  (domain test)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
