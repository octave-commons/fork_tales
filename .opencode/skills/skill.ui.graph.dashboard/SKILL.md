---
id: skill.ui.graph.dashboard
type: skill
version: 1.0.0
tags: [ui, dashboard, graph]
embedding_intent: canonical
---

# Live Graph Dashboard

Intent:
- Visualize graph growth in real time with zoom/pan and filters.
- Show compliance and crawl metrics (rate, frontier, blocks, depth).
- Provide a live event log with reasons for each action.

Operational anchors:
- Canvas renderer updates from websocket deltas.
- Domain/depth/compliance encoding remains consistent.
- Event list remains inspectable and searchable.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.ui.graph.dashboard)
  (domain ui)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
