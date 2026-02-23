# Memory Churn and GC Volatility Remediation (Draft v3)

## Priority
- P1

## Requirements
- Reduce per-edge temporary object allocation in resource-signature route-step fallback.
- Preserve existing route payload schema and selection behavior.
- Keep changes localized to route-step scoring and selection internals.

## Open Questions
- None for this pass.

## Risks
- Score-path divergence from term-path could alter route choice.
- Exploration branch could select wrong edge metadata if edge index tracking is incorrect.

## Complexity Estimate
- Medium

## Existing Issues / PRs
- Follow-up to v2 deferred item.

## Scope

### In Scope
- Add lightweight per-edge score helper that avoids dict construction.
- Refactor fallback loop to compute full route terms once per particle after selection.
- Add regression test for reduced `_route_terms_for_edge` call count.

### Out of Scope
- Native port of resource-aware routing.
- Full adjacency indexing refactor.

## Candidate Files and Hot References
- `part64/code/world_web/c_double_buffer_backend.py:3518` - `compute_graph_route_step_native`
- `part64/code/world_web/c_double_buffer_backend.py:490` - `_route_terms_for_edge`
- `part64/code/tests/test_c_double_buffer_backend.py:544` - resource-signature fallback tests

## Phases

### Phase 1: Route-step scoring path
- Add `_route_score_for_edge` helper for score-only computation.
- Replace per-edge `_route_terms_for_edge` calls in fallback loop with score helper.
- Compute `_route_terms_for_edge` once per particle for selected route.

### Phase 2: Verification
- Extend tests for term-call reduction under resource-signature fallback.
- Run targeted backend tests.

### Phase 3: Receipt
- Append `receipts.log` entry with commands and refs.

## Definition of Done
- No behavior regression in route selection tests.
- Resource-signature fallback no longer allocates term dict per candidate edge.
- Targeted tests pass.
- Receipt appended.

## Session Change Log
- 2026-02-22: Draft created for v3 fallback route-step churn reduction.
