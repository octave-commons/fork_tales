# Eta Observational Layer (Draft)

## Priority
- High (preserve raw signal before interpretation)

## Requirements
- Introduce a distinct `.η/` substrate with append-only semantics.
- Add protocol surface for raw observations and pre-packaging field impact.
- Wire `η -> μ -> Π` as an explicit contract boundary.
- Record adoption in append-only receipts.

## Scope
- Create `.η/{stream,raw,live}` with tracked placeholders.
- Add `.opencode/protocol/eta.v1.lisp`.
- Add `.opencode/promptdb/contracts/eta-layer.contract.lisp`.
- Add `.opencode/promptdb/02_eta_layer.intent.lisp`.
- Extend `promethean.receipts/v2` kinds for eta-layer evidence.

## Risks
- Ambiguity between `.η/` and `.ημ/` responsibilities in day-to-day intake.
- Inconsistent receipts if teams emit eta events without contract refs.

## Mitigations
- Keep `.η/README.md` short and explicit about invariants.
- Require eta-layer receipts to cite protocol + intent + contract refs.

## Files
- `.η/README.md`
- `.η/stream/.gitkeep`
- `.η/raw/.gitkeep`
- `.η/live/.gitkeep`
- `.opencode/protocol/eta.v1.lisp`
- `.opencode/promptdb/contracts/eta-layer.contract.lisp`
- `.opencode/promptdb/02_eta_layer.intent.lisp`
- `.opencode/promptdb/contracts/receipts.v2.contract.lisp`
- `receipts.log`

## Definition Of Done
- `.η/` exists with documented immutability rules.
- `eta.v1` protocol is present and parseable as data-only Lisp.
- eta-layer contract and intent packet are present in PromptDB.
- `receipts.log` has an append-only decision entry for eta-layer adoption.
