# Memory Churn and GC Volatility Remediation Spec (v3)

## 1. Objective
Reduce Python fallback route-step allocation churn in resource-signature mode by avoiding per-edge term-dict construction.

## 2. Accepted Defaults
- Keep route selection algorithm and payload schema unchanged.
- Use score-only edge evaluation during candidate scan.
- Compute full route-term dict once for selected edge per particle.

## 3. Implementation Plan

### Phase A: Score-only candidate scan
- Add `_route_score_for_edge` helper with same scoring math as term path.
- Track `candidate_edge_indices` instead of candidate term dicts.

### Phase B: Selected-edge term hydration
- After best/explore selection, call `_route_terms_for_edge` once per particle.
- Preserve existing output fields from selected term dict.

### Phase C: Verification
- Add/adjust tests for resource-signature fallback behavior and term-call count.
- Run targeted route-step backend tests.

## 4. Acceptance Criteria
- Route outputs remain schema-compatible.
- Resource-signature route choices remain unchanged for existing tests.
- `_route_terms_for_edge` is not called once per candidate edge in fallback scan.
- Targeted tests pass.

## 5. Affected Files
- `part64/code/world_web/c_double_buffer_backend.py`
- `part64/code/tests/test_c_double_buffer_backend.py`
- `specs/drafts/memory-churn-gc-remediation-v3.md`
- `receipts.log`
