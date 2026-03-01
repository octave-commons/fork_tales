# Report — Intent as Song (Deep Research Ready)

## Data artifacts
- dense intent (box): `analysis/gates_of_truth_intent_dense_box/intent_events.bin`
- intent anchors: `analysis/gates_of_truth_intent_dense_box/intent_anchors.jsonl`
- intent score: `analysis/gates_of_truth_intent_dense_box/intent_score.csv`
- intent timeline: `analysis/gates_of_truth_intent_dense_box/intent_timeline.csv`

## Rule
Any “feels like a song” output must remain traceable:
- a note (in the score) maps back to an `event_id`
- the `event_id` maps to an anchor snippet hash
- the snippet lives in a source chapter file

## Next deep research move
Use `meta/deep_research/prep_checklist.md` and keep claim budgets small.

This report is a wiring diagram, not a conclusion.
