# Part64 Simulation Smoothing

## Priority
- High

## Complexity
- Moderate (frontend render loop + websocket cadence)

## Requirements
- Reduce visible jitter in particle simulation updates.
- Keep websocket simulation stream responsive without overloading the UI thread.
- Preserve existing API and UI behavior for catalog and simulation consumers.

## Open Questions
- None blocking. Defaulting to:
  - backend sends simulation frames at higher cadence,
  - frontend avoids unnecessary canvas resets/re-initialization.

## Risks
- Higher simulation cadence can increase CPU/network load if catalog payloads are sent too frequently.
- Frontend loop refactors can introduce stale data bugs if refs are not synchronized.

## Files Planned
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/world_web.py`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/components/Simulation/Canvas.tsx`

## Phases

### Phase 1: Backend cadence tuning
- Decouple simulation tick cadence from catalog broadcast cadence.
- Keep cached catalog between broadcasts and stream simulation more frequently.

### Phase 2: Frontend render stabilization
- Remove per-update overlay loop reinitialization.
- Prevent canvas width/height resets when dimensions have not changed.
- Read latest simulation/catalog through refs during RAF loops.

### Phase 3: Verification
- Run backend tests and frontend build.
- Validate runtime endpoints/ws connectivity still pass.

## Definition of Done
- Simulation movement appears smoother under the same workload.
- Backend tests pass.
- Frontend build passes.

## Session Change Log
- Increased websocket simulation cadence and decoupled catalog refresh cadence in `world_web.py`.
- Added catalog signature gating to avoid rebroadcasting unchanged catalog payloads on every sim tick.
- Updated simulation canvas loops to avoid unnecessary canvas resize resets and overlay effect reinitialization.
- Switched overlay loop to live refs for `catalog`/`simulation` so RAF animation stays continuous during incoming WS updates.
- Verified with backend tests, frontend production build, runtime HTTP checks, and websocket handshake check.
