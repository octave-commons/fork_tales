---
id: skill.muse.context-retrieval
type: skill
version: 1.0.0
tags: [muse, context, retrieval]
embedding_intent: canonical
---

# Muse Context Retrieval

Intent:
- Pull relevant historical context into current deliberation.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.muse.context-retrieval)
  (domain muse)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
