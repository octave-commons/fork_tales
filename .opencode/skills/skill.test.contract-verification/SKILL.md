---
id: skill.test.contract-verification
type: skill
version: 1.0.0
tags: [test, contracts]
embedding_intent: canonical
---

# Contract Verification

Intent:
- Verify conformance against declared protocol and interface contracts.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.test.contract-verification)
  (domain test)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
