# S5 Resolution Dataset (Versioned)

This folder makes **entity resolution first-class**.

- `entities_base.jsonl` is the starting canonical list (from Character Profiles).
- All changes go in `changes/*.jsonl` as *reviewed deltas*.
- Every merge/split/alias is a **claim** → should be anchored (S3) in the story or notes.

## Change record shape
- change_id
- ts
- kind: alias_add | alias_remove | merge | split | retire
- entity_id(s)
- rationale
- witness
- anchor_claim_id (optional but preferred)

## Audit rule
No change is “real” until reviewed + merged into a new dataset version.
