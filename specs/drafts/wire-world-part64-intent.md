# Wire World Part64 Intent Implementation

## Priority
- High

## Complexity
- Moderate-to-complex (multi-file backend + frontend + contracts + tests)

## Requirements
- Add PromptDB contract files under `.opencode/promptdb/contracts`.
- Wire PromptDB packet indexing into catalog payload.
- Add backend endpoints:
  - `POST /api/presence/say`
  - `POST /api/drift/scan`
  - `POST /api/push-truth/dry-run`
- Keep existing `/api/catalog` and `/ws` behavior compatible.
- Add frontend slash commands:
  - `/say <PresenceId>`
  - `/drift`
  - `/push-truth --dry-run`
- Add tests in `code/tests/test_world_web_pm2.py` for promptdb catalog indexing and presence-say response shape.

## Open Questions
- None blocking. Implementation assumes the canonical runtime root is `.fork_Π_ημ_frags/ημ_op_mf_part_64`.

## Risks
- PromptDB packet parsing is data-only and must avoid eval.
- Existing catalog schema consumers may assume only artifact-driven entries.
- Frontend command routing currently only handles `/ledger`; extending command parsing must preserve existing chat flow.

## Existing Issues/PRs
- No local evidence found of existing issue/PR specific to this wiring task.

## Files Planned
- `.opencode/promptdb/contracts/ui.contract.lisp`
- `.opencode/promptdb/contracts/presence-say.contract.lisp`
- `.opencode/promptdb/contracts/receipts.v2.contract.lisp`
- `.opencode/promptdb/00_wire_world.intent.lisp`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/world_web.py`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/tests/test_world_web_pm2.py`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/App.tsx`

## Subtasks
1. Create contracts and intent packet files.
2. Add PromptDB packet scan + catalog integration.
3. Add presence-say/drift-scan/push-truth dry-run backend endpoints.
4. Add frontend slash command handlers.
5. Add/update tests.
6. Verify diagnostics, tests, and frontend build.

## Definition of Done
- Endpoints respond with deterministic JSON payloads and expected keys.
- Catalog includes PromptDB packet metadata.
- Slash commands invoke corresponding endpoints and surface results.
- Updated tests pass.
- Modified files show zero LSP errors.
