---
source: part64/deep-research-report.md
section: Experimental workloads and ablations
status: untested
reviewed_on: 2026-02-20
---

# Part64 Deep Research Spec 07 - Evaluation Workloads and Ablations

## Scope
- Run synthetic workloads and ablation studies across routing, pricing, compaction, and sampling choices.

## Current evidence in code
- Runtime benchmark tooling exists in `part64/scripts/bench_sim_compare.py`.
- Benchmark API wiring exists in `part64/code/world_web/server.py`.
- Workbench and simulation benchmarking workflow docs exist in `part64/SIMULATION_WORKFLOW.md`.

## Why status is untested
- There is no committed automated suite that runs the full workload matrix from the report.
- No ablation runner currently toggles and records all report variants (`no-gravity`, `fixed-price`, `MAP`, `no-compaction`, etc.).
- No canonical artifact bundle for experiment results is produced by CI.

## Definition of done
- Add reproducible workload generators and ablation toggles.
- Add metrics capture + export format for each run.
- Add test or contract checks validating workload harness output schema.
