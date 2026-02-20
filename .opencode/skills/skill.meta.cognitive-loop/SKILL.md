---
id: skill.meta.cognitive-loop
type: skill
version: 1.0.0
tags: [meta, cognition, failure, signal, feedback]
embedding_intent: canonical
---

# Meta-Cognitive Loop

**Intent**:
- Close the gap between "failure" and "fix" by treating instability as structured signal.
- Synthesize observations, hypotheses, and objectives into a continuous operational narrative.

**Capabilities**:
- **Signal Analysis**: Interpreting Docker lifecycle signals (OOM, restarts, health) and pressure states.
- **Note Capture**: Documenting observations in `.opencode/runtime/meta_notes.v1.jsonl`.
- **Objective Enqueueing**: Converting failures into high-level tasks in the `meta-objective` queue.
- **Evidence Retrieval**: Analyzing historical runs and notes to identify recurring instability patterns.

**Operational Rituals**:
1. **Signal Triage**: Review the "Failure Signals" feed in the Meta Dashboard.
2. **Annotation**: Save a **Meta Note** for every unexplained simulation degradation.
3. **Task Conversion**: Enqueue an **Objective** when a note identifies a concrete path toward stabilization or training improvement.
4. **Validation**: Mark the loop closed when a subsequent **Meta Run** shows stability in the profile view.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.meta.cognitive-loop)
  (domain meta-cognition)
  (storage-layer "meta_notes.v1.jsonl")
  (signal-source "docker-lifecycle")
  (closure-condition "meta-run-stability"))
```
