# Truth Binding v1

## Priority
- high

## Intent
- Bind Truth as a world-scoped judged claim instead of a raw blob.
- Encode explicit name/operator/invariant surfaces for `名`, `真`, `理`, and `伪`.
- Keep the path eta -> mu -> receipts auditable for push-truth style gates.

## Requirements
- Add protocol artifact for truth records and operators.
- Add PromptDB contract defining invariants and gate implications.
- Add intent packet that references and routes truth binding as executable intent data.
- Append receipt evidence tying this work to protocol + contract + intent refs.

## Risks
- Unicode glyph use in protocol/contract may require UTF-8-safe tools.
- Runtime does not yet execute truth operators directly; this pass defines canonical surfaces only.

## Definition of Done
- `.opencode/protocol/truth.v1.lisp` exists and is data-only.
- `.opencode/promptdb/contracts/truth-layer.contract.lisp` exists with proof refs.
- `.opencode/promptdb/03_bind_truth.intent.lisp` exists and cites the above artifacts.
- `receipts.log` includes decision entry for truth binding artifacts.

## Files
- `.opencode/protocol/truth.v1.lisp`
- `.opencode/promptdb/contracts/truth-layer.contract.lisp`
- `.opencode/promptdb/03_bind_truth.intent.lisp`
- `.opencode/promptdb/contracts/receipts.v2.contract.lisp`
- `receipts.log`

## Implementation Extension (Simulation Binding)
- Runtime now computes and serves `truth_state` through catalog and simulation payloads.
- Catalog signature now tracks truth claim/gate deltas to trigger deterministic refresh.
- Simulation now emits truth-linked particles around `gates_of_truth` for claim visibility.
- Frontend types and simulation overlay now render truth gate status (`status`, `kappa`, `theta`, blocked reason).
- Backend tests include truth-state presence assertions for both catalog and simulation.

## Change Log
- 2026-02-16: Bound truth state into runtime payload/UI surfaces and validated with pytest + frontend build + runtime endpoint checks.
