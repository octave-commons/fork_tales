# Memory Churn and GC Volatility Remediation (Draft v1)

## Priority
- P0

## Requirements
- Reduce per-tick temporary allocations on C/Python bridge hot paths.
- Bound retained runtime maps that can grow without a cap.
- Preserve existing runtime payload schemas and route behavior.
- Avoid broad refactors; prefer targeted, reviewable patches.
- Keep C/Python/JavaScript surfaces aligned with existing conventions.

## Open Questions
- Should route-step resource-signature mode stay Python-first for now, or be partially migrated to native C in this pass?
- Do we want strict LRU behavior for edge-health retention, or a simpler active-set + cap policy?

## Risks
- Over-pruning caches may alter route continuity behavior.
- Reusing ctypes buffers can expose stale-data bugs if fill/zero rules are incorrect.
- Native scratch-release wiring can miss thread-local allocations if called from a different thread context.
- Web graph retention caps can reduce dedupe effectiveness at very large crawl sizes.

## Complexity Estimate
- Medium

## Existing Issues / PRs
- No local issue/PR references discovered for this exact remediation pass.

## Scope

### In Scope
- Python bridge buffer reuse for `update_nooi` and `update_embeddings`.
- Edge-health registry bounding to avoid unbounded dict growth.
- Native scratch-buffer release API and shutdown hook.
- Web Graph Weaver retention map caps/pruning.

### Out of Scope
- Full native port of resource-signature routing logic.
- Frontend render architecture refactors.
- Runtime protocol/schema redesign.

## Candidate Files and Hot References
- `part64/code/world_web/c_double_buffer_backend.py:2494`
- `part64/code/world_web/c_double_buffer_backend.py:2515`
- `part64/code/world_web/c_double_buffer_backend.py:31`
- `part64/code/world_web/c_double_buffer_backend.py:3254`
- `part64/code/world_web/native/c_double_buffer_sim.c:186`
- `part64/code/world_web/native/c_double_buffer_sim.c:3335`
- `part64/code/web_graph_weaver.js:1108`
- `part64/code/web_graph_weaver.js:1158`

## Phases

### Phase 1: Spec and guardrails
- Save this draft.
- Define low-risk acceptance constraints for memory behavior.

### Phase 2: Python/C bridge churn reduction
- Reuse persistent ctypes buffers in `_CDBEngine.update_nooi`.
- Reuse persistent ctypes buffers in `_CDBEngine.update_embeddings`.
- Ensure deterministic truncation/zero-fill semantics.

### Phase 3: Retention bounding
- Add capped pruning for `_EDGE_HEALTH_REGISTRY`.
- Add bounded retention for Web Graph Weaver maps (`contentHashIndex`, `domainState`).

### Phase 4: Native scratch release hygiene
- Add exported native function to free thread-local scratch arrays.
- Wire call from backend shutdown path.

### Phase 5: Verification and receipt
- Run focused backend tests and frontend build/check.
- Append `receipts.log` entry with changed file refs and verification refs.

## Definition of Done
- No behavior regressions in targeted backend tests.
- Frontend build/check remains green.
- Buffer update hot paths avoid per-call ctypes array allocation.
- Edge-health and weaver retention maps are explicitly bounded.
- Native scratch release API is reachable and called on shutdown.
- Receipt appended with command evidence.

## Session Change Log
- 2026-02-22: Draft created from audit pass of C/Python/JS memory churn paths.
