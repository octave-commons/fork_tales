---
id: skill.pathos.affect-detection
type: skill
version: 1.0.0
tags: [pathos, affect, detection]
embedding_intent: canonical
---

# Affect Detection

Intent:
- Derive affect vectors from explicit event traces, not anthropomorphic inference.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.pathos.affect-detection)
  (domain pathos)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
