---
id: skill.semantic.cluster.phase2
type: skill
version: 1.0.0
tags: [semantic, clustering, phase2]
embedding_intent: canonical
---

# Semantic Cluster Phase 2

Intent:
- Add optional semantic embedding clustering as a post-baseline extension.
- Keep clustering outputs inspectable and non-authoritative until validated.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.semantic.cluster.phase2)
  (domain semantic)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
