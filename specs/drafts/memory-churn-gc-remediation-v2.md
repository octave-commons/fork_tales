# Memory Churn and GC Volatility Remediation (Draft v2)

## Priority
- P1

## Requirements
- Reduce per-embed vector allocations in C++ embed runtime.
- Reduce per-route-step dict allocations in Python route fallback path.
- Preserve existing runtime payload schemas and behavior.
- Avoid broad refactors; prefer targeted, reviewable patches.

## Open Questions
- ~~Should route terms dict reuse use a pre-allocated list of dicts or a flat struct-like tuple?~~ Deferred to v3.
- ~~Should C++ runtime buffer pooling be thread-local or per-runtime-instance?~~ Per-instance, stored in CEmbedRuntime struct.

## Risks
- Reusing buffers without proper zeroing can leak data between calls. (Mitigated: explicit zero-fill on reuse)
- Thread-local C++ buffers require careful lifecycle management. (N/A: per-instance pooling used)
- Route-term dict reuse can introduce aliasing bugs if caller retains references. (Deferred)

## Complexity Estimate
- Medium

## Existing Issues / PRs
- Follow-up to v1 memory churn remediation.

## Scope

### In Scope
- C++ embed runtime per-call vector allocation pooling.

### Deferred to v3
- Python route-step per-edge dict allocation reduction. (Complexity/risk trade-off)

### Out of Scope
- Frontend render architecture refactors.
- Native port of full resource-signature routing logic.
- Full native route-step implementation.

## Candidate Files and Hot References
- `part64/code/world_web/native/c_embed_runtime.cpp:442` - `run_embed` now uses pooled buffers
- `part64/code/world_web/native/c_embed_runtime.cpp:123` - `CEmbedRuntime` struct with pooled fields

## Phases

### Phase 1: C++ embed runtime buffer pooling ✅
- Added pooled vectors to `CEmbedRuntime` struct.
- Pre-allocated `pooled_default_mask`, `pooled_default_type`, `pooled_position_ids` in constructor.
- Pre-allocated reusable `pooled_extra_i64` and `pooled_extra_f32` buffers.
- Pre-allocated `pooled_hidden` for output pooling.
- Reuse buffers in `run_embed` instead of reallocating.

### Phase 2: Verification and receipt
- Run targeted tests for embed runtime.
- Append `receipts.log` entry.

## Definition of Done
- No behavior regressions in targeted tests.
- C++ embed runtime avoids per-call vector allocation.
- Receipt appended with command evidence.

## Session Change Log
- 2026-02-22: Draft created as follow-up to v1 remediation.
- 2026-02-22: C++ embed runtime pooling implemented.
- 2026-02-22: Route-step dict reuse deferred to v3 (complexity/risk trade-off).
