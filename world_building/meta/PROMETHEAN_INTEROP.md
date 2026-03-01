# Promethean Interop Notes

This box is meant to be ingestible by a Promethean-style system (Cephalon/Eidolon/Nexus).

## Suggested Nexus mapping

Treat each file as a Nexus node:

- `nexus.id` = `sha256` (stable content identity)
- `nexus.path` = `relpath`
- `nexus.kind` = `mime` or extension bucket
- `nexus.tags` = top-2 circuit aliases + any path-derived tags (e.g., `gates_of_truth`, `manuscript`, `images`)
- `nexus.vector64` = bootstrap coordinates (this repo's 64-D scaffold)
- `nexus.text` = extracted text (if any)

## Eidolon / field notes

The 8 circuits provide **layered indexing**:
- quick retrieval via circuit filtering
- approximate clustering via vector64
- later replacement with embeddings per circuit (8 independent embedding spaces) if desired

## Cephalon ingestion loop (high level)

P: add artifacts  
R: run `tools/reindex.py`  
N: query via `tools/search.py` / `tools/nearest.py`  
Π: update `meta/circuit_schema.json`  
A: apply changes into prompts, story canon, or evaluation sets  
feedback: repeat

See also: `meta/PRNPIA_PLAYBOOK.md`.
