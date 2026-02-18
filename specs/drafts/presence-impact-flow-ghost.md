# Presence Impact Flow + Ghost Role

## Priority
- High

## Complexity
- Moderate (runtime telemetry + simulation overlay + diagnostics)

## Requirements
- Expose how presences are affected by file deltas and witness touches (mouse clicks).
- Make Receipt River / 領収書の川 flow visibly data-driven in the simulation overlay.
- Surface auto-committing ghost role as File Sentinel / ファイルの哨戒者 with clear runtime status.
- Keep fork-tax law visible in runtime diagnostics and preserve additive constraints.

## Open Questions
- None blocking. Defaults:
  - File influence comes from catalog delta snapshots.
  - Click influence comes from `/api/witness` and world interactions.

## Risks
- Additional per-frame overlay rendering can reduce FPS on low-end GPUs.
- Runtime snapshots can drift if tracker and websocket loops diverge in cadence.

## Files Planned
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/world_web.py`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/tests/test_world_web_pm2.py`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/components/Simulation/Canvas.tsx`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/components/Panels/Vitals.tsx`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/components/Panels/PresenceMusicCommandCenter.tsx`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/types/index.ts`

## Definition of Done
- Simulation payload includes additive `presence_dynamics` telemetry.
- Overlay visibly renders river flow and File Sentinel ghost pulse/status.
- Diagnostics and vitals expose file/click influence and fork-tax state.
- Backend tests and frontend build pass.

## Session Change Log
- Added append-only runtime influence tracker for witness clicks and catalog file deltas.
- Extended websocket and `/api/simulation` payload generation with `presence_dynamics`.
- Added Receipt River / 領収書の川 flow ribbons and File Sentinel / ファイルの哨戒者 ghost overlay in simulation canvas.
- Extended vitals + diagnostics with file/click influence, ghost pulse, and fork-tax balance.
- Appended `C-68-presence-impact-telemetry` to constraints ledger without removing prior constraints.
