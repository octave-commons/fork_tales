# 家_映 v1 Implementation Draft

## Priority
- High (runtime ontology + projection UI coupling)

## Requirements
- Add protocol artifact `.opencode/protocol/家_映.v1.lisp` with data-only contract form.
- Implement projection runtime where UI state is derived from runtime facts (catalog/simulation/task queue/influence) and never used as source-of-truth.
- Support default boot perspective `hybrid` and expose perspective selection (`hybrid`, `causal-time`, `swimlanes`) for projection snapshots.
- Emit explainable mass/priority/area decisions per UI element.
- Model chat as field-bound lens sessions, including presence and memory scope metadata.
- Keep canonical runtime endpoints healthy: `/`, `/api/catalog`, `/ws`.

## Existing Context (Files + Lines)
- Backend catalog + simulation sources:
  - `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/world_web.py:670` (`collect_catalog`)
  - `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/world_web.py:1827` (`build_named_field_overlays`)
  - `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/world_web.py:2027` (`build_simulation_state`)
  - `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/world_web.py:3528` (`/api/catalog`)
  - `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/world_web.py:3560` (`/api/simulation`)
- Frontend consumers:
  - `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/hooks/useWorldState.ts:11`
  - `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/types/index.ts:221`
  - `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/App.tsx:68`
  - `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/components/Simulation/Canvas.tsx:41`
- Tests to extend:
  - `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/tests/test_world_web_pm2.py:180`

## Risks
- Layout instability/jitter if projection mass changes too aggressively tick-to-tick.
- Type/API drift between backend projection payload and frontend TS interfaces.
- Excess payload size if projection snapshots are over-expanded in websocket events.

## Mitigations
- Clamp and smooth area/priority/pulse values in projection builder.
- Keep projection records compact and deterministic.
- Add tests for perspective defaults and explainability fields.

## Open Questions
- None. User selected default perspective as `hybrid`.

## Existing Issues / PRs
- `gh issue list --limit 20`: no git remotes found.
- `gh pr list --limit 20`: no git remotes found.

## Phases

### Phase 1 — Protocol + Backend Projection Model
- Add `.opencode/protocol/家_映.v1.lisp`.
- Add backend projection builders for:
  - `場/snapshot`, `心/state`, `映/element`, `映/state`, `映/layout`, `映/chat-session`, `映/vector-view`, `映/tick-view`.
- Add perspective normalization/defaulting (`hybrid`).
- Add explainability map for each derived `映/state`.

### Phase 2 — API + Stream Integration
- Include projection metadata in `/api/catalog` payload.
- Add dedicated projection endpoint for perspective queries (`/api/ui/projection`).
- Include projection in websocket catalog broadcast payloads.

### Phase 3 — Frontend Projection Wiring
- Extend TS types for projection records.
- Consume projection payload in `useWorldState`.
- Make major dashboard sections projection-aware (size/emphasis from derived state).
- Surface perspective and explainability in UI.

### Phase 4 — Verification + Receipts
- Extend/adjust Python tests for projection behavior.
- Run backend tests and frontend build.
- Append receipt line referencing intent + protocol contract.

## Complexity Estimate
- Medium-high (cross-cutting backend + frontend + protocol + tests).

## Definition of Done
- Protocol file exists at `.opencode/protocol/家_映.v1.lisp` and validates as data-only form.
- Runtime exposes projection data with default `hybrid` perspective and selectable alternatives.
- Frontend reflects projection-derived emphasis (area/priority/pulse) and explainability.
- `/`, `/api/catalog`, `/ws` remain available.
- Tests/build pass and receipt appended.
