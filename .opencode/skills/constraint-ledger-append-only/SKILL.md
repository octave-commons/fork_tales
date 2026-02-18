---
name: constraint-ledger-append-only
description: Maintain additive constraint evolution with disable-by-append protocol and explicit changelog discipline.
metadata:
  owner: project
  version: 1
---

# Constraint Ledger Append-Only

Use this skill whenever constraints are introduced, adjusted, or retired.

## Rules

- Add new constraints by append only.
- Never remove a prior constraint line.
- To retire behavior, add a disable line referencing the original constraint ID.

## Procedure

1. Inspect current active constraints ledger.
2. Add new or adjusted line with unique ID and short rationale.
3. If disabling, append explicit disable record.
4. Keep references to affected modules and tests.

## Guardrails

- No silent semantic drift in existing IDs.
- No retroactive edits that erase historical behavior.
