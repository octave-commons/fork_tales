---
id: skill.pathos.cooldown-model
type: skill
version: 1.0.0
tags: [pathos, cooldown, stability]
embedding_intent: canonical
---

# Cooldown Model

Intent:
- Recommend dampening strategies after high-charge intervals.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.pathos.cooldown-model)
  (domain pathos)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
