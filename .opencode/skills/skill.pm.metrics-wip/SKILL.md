---
id: skill.pm.metrics-wip
type: skill
version: 1.0.0
tags: [pm, metrics, wip]
embedding_intent: canonical
---

# Metrics, WIP, and Run Management

**Intent**:
- Track throughput, WIP, and lead-time trends for planning feedback loops.
- Use metrics to improve flow, not punish contributors.
- **Run Tracking**: Management of training and evaluation runs as structured operational artifacts.

**Capabilities**:
- **Run Logging**: Posting run configuration (model ref, dataset ref) and outcomes to `/api/meta/runs`.
- **Performance Delta**: Using `bench_sim_compare.py` to quantify performance gains or regressions.
- **Feedback Loops**: Correlating simulation stability signals with planning backlog priority.

**Execution Guidance**:
- Every significant "experiment" should be tracked as a "Meta Run".
- Use **Training Charts** to visualize the outcome of multiple evaluation cycles.
- Slicing work should prioritize resolving "failing" or "degraded" signals in the Meta Dashboard.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.pm.metrics-wip)
  (domain pm)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
