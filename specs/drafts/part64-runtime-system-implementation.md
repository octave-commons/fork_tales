# Part64 Runtime System Packet Implementation

## Priority
- High

## Complexity
- Moderate (backend runtime + tests + manifest schema)

## Requirements
- Implement deterministic task queue semantics with persistence, dedupe, and replay behavior.
- Emit receipts for queue enqueue/dequeue operations.
- Add PromptDB refresh polling/debounce behavior and expose refresh stats via `/api/catalog`.
- Make push-truth proof schema explicit in `manifest.lith` (required refs + required hashes + host handle).
- Wire KeeperOfContracts presence output when gate-relevant open questions remain unresolved.

## Open Questions
- None blocking for this implementation pass. Defaults:
  - Queue storage: file-backed event log with in-memory replay.
  - Drift sampling: polling-based refresh from catalog access.

## Risks
- Broad changes in `world_web.py` can regress existing API contracts if compatibility keys are removed.
- Adding stricter gate checks can unexpectedly block push-truth in environments with incomplete receipts.
- Queue receipt writes can fail on read-only filesystems.

## Files Planned
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/world_web.py`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/tests/test_world_web_pm2.py`
- `manifest.lith`

## Phases

### Phase 1: Runtime primitives
- Add promptdb polling/debounce refresh state and catalog stats.
- Add file-backed task queue with dedupe/replay and receipt emission hooks.

### Phase 2: Gate + presence wiring
- Extend drift scan with unresolved open-question detection.
- Emit KeeperOfContracts `say_intent` payload when unresolved questions block a gate.
- Extend push-truth dry-run with explicit proof schema parsing and requirement checks.

### Phase 3: Verification
- Add/adjust tests for queue behavior, drift/presence behavior, and manifest proof schema behavior.
- Run targeted test suite and report outcomes.

## Existing Issues / PRs
- No local issue/PR metadata discovered for this exact task.

## Definition of Done
- `/api/catalog` includes promptdb refresh strategy/stats.
- Queue operations are deterministic and persisted; enqueue/dequeue emit receipts.
- Push-truth dry-run returns explicit proof schema and requirement evaluation.
- KeeperOfContracts signal appears when unresolved gate questions exist.
- Updated tests pass.

## Session Change Log
- Added polling+debounce PromptDB refresh cache and refresh stats exposure in catalog payload.
- Added file-backed task queue with dedupe/replay and receipt emission on enqueue/dequeue.
- Extended drift scan with open-question resolution status and KeeperOfContracts signal payload.
- Extended push-truth dry-run with `manifest.lith` proof schema checks (refs/hashes/host handle).
- Added/updated backend tests and validated with `python -m pytest code/tests`.
