---
id: skill.dev.performance-budget
type: skill
version: 1.0.0
tags: [dev, performance, budget]
embedding_intent: canonical
---

# Performance Budget

Intent:
- Track runtime and UI performance against explicit budgets.
- Prevent regressions with measurable thresholds and alerts.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.dev.performance-budget)
  (domain dev)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
