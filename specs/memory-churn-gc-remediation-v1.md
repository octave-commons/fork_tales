# Memory Churn and GC Volatility Remediation Spec (v1)

## 1. Objective
Reduce avoidable allocation churn and unbounded runtime retention across C/Python/JavaScript surfaces while preserving current simulation and API semantics.

## 2. Accepted Defaults (No Blockers)
- Keep resource-signature routing behavior unchanged in this pass.
- Use active-set + cap pruning for edge-health retention (no strict LRU).
- Prefer bounded retention over full-history caches in Web Graph Weaver runtime maps.

## 3. Implementation Plan

### Phase A: Bridge allocation reduction
- Reuse long-lived ctypes buffers for:
  - `update_nooi`
  - `update_embeddings`
- Replace pad/truncate list reallocation with deterministic in-place copy + zero-fill.

### Phase B: Retention bounds
- Add cap and prune logic for `_EDGE_HEALTH_REGISTRY`.
- Add cap and prune logic for Web Graph Weaver `contentHashIndex` and `domainState`.

### Phase C: Native scratch hygiene
- Export native scratch-release function for thread-local scratch arrays.
- Invoke release hook from backend shutdown.

### Phase D: Verification
- Backend targeted tests for CDB graph runtime and world web endpoints/tests.
- Frontend build verification for TypeScript/React surface.

## 4. Acceptance Criteria
- No regression in targeted backend tests.
- Frontend build passes.
- No per-call ctypes array creation in CDB bridge update methods.
- Edge health registry growth is bounded by explicit policy.
- Web graph retention maps have explicit bounded policy.
- Native scratch release function is callable and wired.

## 5. Affected Files
- `part64/code/world_web/c_double_buffer_backend.py`
- `part64/code/world_web/native/c_double_buffer_sim.c`
- `part64/code/web_graph_weaver.js`
- `receipts.log`
