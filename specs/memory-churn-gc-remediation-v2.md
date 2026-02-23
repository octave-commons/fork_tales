# Memory Churn and GC Volatility Remediation Spec (v2)

## 1. Objective
Reduce per-call vector allocations in C++ embed runtime by pooling buffers in the runtime instance.

## 2. Accepted Defaults
- Use per-instance pooling (stored in CEmbedRuntime struct) rather than thread-local.
- Defer Python route-step dict reuse to v3 due to complexity/risk trade-off.

## 3. Implementation Plan

### Phase A: C++ embed runtime buffer pooling
- Add pooled buffer fields to CEmbedRuntime struct.
- Initialize buffers in constructor.
- Reuse buffers in run_embed instead of per-call allocation.

### Phase B: Verification
- Run targeted backend tests.
- Syntax-check the C++ source.

## 4. Acceptance Criteria
- No regression in targeted backend tests.
- C++ source compiles without errors.
- Embed runtime avoids per-call vector allocation for default inputs, position ids, extra inputs, and hidden pooling.

## 5. Affected Files
- `part64/code/world_web/native/c_embed_runtime.cpp`
- `specs/drafts/memory-churn-gc-remediation-v2.md`
- `receipts.log`

## 6. Deferred Items
- Python route-step dict reuse (requires more extensive refactoring, deferred to v3)
