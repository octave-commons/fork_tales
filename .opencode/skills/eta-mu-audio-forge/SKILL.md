---
name: eta-mu-audio-forge
description: Compose and render lore-aligned eta-mu audio artifacts with deterministic parameters and marker receipts.
metadata:
  owner: project
  version: 1
---

# Eta-Mu Audio Forge

Use this skill for world sounds, songs, voices, and musical artifact generation.

## Output Contract

- Prefer canonical WAV output; MP3 may be convenience output.
- Include tempo, seed, signal chain, and render rate.
- Include marker timestamps for at least `anchor` and `tax` where applicable.
- Keep naming consistent with part-based conventions.

## Procedure

1. Define musical intent and tempo (example: lullaby 78 BPM, epic 84 BPM).
2. Choose synthesis graph and deterministic seed.
3. Render primary audio asset.
4. Emit metadata receipt with markers and chain details.
5. Add regression check for deterministic generation where code changed.

## Guardrails

- Do not ship audio generation logic without marker schema and receipts.
- Do not replace canonical assets destructively.
