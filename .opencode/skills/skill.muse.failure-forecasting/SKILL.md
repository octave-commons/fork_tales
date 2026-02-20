---
id: skill.muse.failure-forecasting
type: skill
version: 1.0.0
tags: [muse, failure, forecasting]
embedding_intent: canonical
---

# Muse Failure Forecasting

**Intent**:
- Project likely failure surfaces and preemptive mitigations.
- Analyze simulation instability (OOM, restarts, health timeouts) as precursors to systemic regression.
- Predict second-order effects of configuration changes on runtime behavior.

**Techniques**:
- **Drift Analysis**: Observing performance delta between baseline and experiment simulations.
- **Pressure Projection**: Predicting when resource budget (CPU/Mem/PIDs) will saturate under specific loads.
- **Cascade Mapping**: Identifying how a failure in one container (e.g., Chroma) propagates to others (e.g., Weaver/Meta).

**Operational Context**:
- Monitor the **Failure Signals** feed in the Meta Dashboard (`/dashboard/docker`).
- Use the **Workbench** to test "brittleness hypotheses" with throttled presets.
- Record "Pre-Mortem" notes in the meta notes log before significant architectural shifts.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.muse.failure-forecasting)
  (domain muse)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
