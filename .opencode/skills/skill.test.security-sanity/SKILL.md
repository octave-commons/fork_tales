---
id: skill.test.security-sanity
type: skill
version: 1.0.0
tags: [test, security, sanity]
embedding_intent: canonical
---

# Security Sanity

Intent:
- Assert baseline security invariants around boundaries and permissions.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.test.security-sanity)
  (domain test)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
