# Deep Research Ready (Local Box)

This is a *readiness contract* for doing real deep research later without losing ημ.

## Inputs
- Box corpus (markdown + images + metrics)
- Anchor datasets:
  - `analysis/gates_of_truth_relationships_box/anchors.jsonl`
  - `analysis/gates_of_truth_ideologies_box/anchors.jsonl`
  - `analysis/gates_of_truth_intent_dense_box/intent_anchors.jsonl`
  - `analysis/gates_of_truth_forks_box/anchors.jsonl`

## Output requirements (non-negotiable)
1) Every non-trivial claim must cite at least one anchor (S3).
2) Fork claims must cite an intent event AND include rationale + rank.
3) Entity resolution (S5) is a versioned dataset; changes are auditable.

## “Default voice” suppression
If the model starts narrating compliance (“I matched constraints because…”), that is treated as a failure mode:
- it must output artifacts (anchors / ids / files) instead of explanation.

## Research modes
- **Corpus report**: summarize only anchored claims.
- **Disagreement report**: list conflicting anchored claims; do not reconcile without additional evidence.
- **Speculative branch**: label as speculation; tie to fork candidates only.

## Cephalon hook
Intent packs (`meta/intent_disc/`) can be routed into Cephalon as portable objectives:
- verify anchors (η)
- propose actions within permissions (μ)
- emit trace artifacts back into the box

This turns deep research into a repeatable, auditable build step.
