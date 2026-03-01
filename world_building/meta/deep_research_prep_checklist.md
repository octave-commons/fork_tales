# Deep Research Prep Checklist (Before You Turn It On)

1) **Freeze**: build a snapshot and keep the split zips.
2) **Define claim budget**: max claims in the final report.
3) **Choose audit sample rate**: start at 5% of edges.
4) **Lock entity resolution**: pin an S5 dataset version.
5) **Require anchors**: non-trivial claim/edge/fork must cite S3.
6) **Set fork gate**: minimum intent strength; require rank+rationale.
7) **Ban meta voice**: no “because constraints”; only artifacts/anchors.
8) **Record provenance**: dice runs and tool runs are logs, not vibes.

Output artifacts you should have on disk:
- `search/opmindfuck.sqlite`
- `analysis/gates_of_truth_intent_dense_box/*`
- `analysis/s5_resolution/*`
- `build/dice_runs/*.jsonl`
