---
id: skill.pm.release-checklist
type: skill
version: 1.0.0
tags: [pm, release, checklist]
embedding_intent: canonical
---

# Release Checklist

Intent:
- Gate releases with clear go/no-go checks.
- Ensure verification, documentation, and rollback readiness.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.pm.release-checklist)
  (domain pm)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
