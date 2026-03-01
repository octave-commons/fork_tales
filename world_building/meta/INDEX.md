# Operation Mindfuck — The Box

This archive is a **single, hash-verified corpus** of Operation Mindfuck artifacts currently available as files.

## Structure

- `content/` — extracted primary artifacts
- `source_archives/` — original zips (provenance)
- `meta/` — index + manifest + circuit schema
- `search/` — SQLite FTS index (`opmindfuck.sqlite`)
- `tools/` — Python tooling (search, nearest, reindex)

## Search

### Full-text search
```bash
python tools/search.py "gates of truth"
python tools/search.py "ημ" --limit 25
python tools/search.py "protocol" --circuit "Protocol/Law"
```

### Nearest neighbors in the 64-D scaffold
```bash
python tools/nearest.py --query "glitch loop daemon" --top 15
python tools/nearest.py --query "pandora box myth" --top 15
```

## The 8 circuits (Prometheus Rising → Promethean mapping)

1. Bio-Survival → Watcher/Alive  
2. Emotional-Territorial → Permissions/Pack  
3. Symbolic → Language/Model  
4. Domestic-Moral → Protocol/Law  
5. Neuro-Somatic → Body/Sound  
6. Neuro-Electric → Loop/Glitch  
7. Neuro-Genetic → Myth/Lineage  
8. Neuro-Atomic → Cosmic/Meta  

Each artifact has:
- `circuit_scores`: 8 floats
- `vector64`: 64 floats = (8 circuits × 8 dims)

**Important:** this is a *bootstrap map* based on keywords + structure. Use it to navigate; swap in embeddings later if desired.

## P→R→N→Π→A→(feedback)→P

The loop is implemented as:
- P: add/curate artifacts under `content/`
- R: run `tools/reindex.py`
- N: inspect `meta/MANIFEST.json` / `search` results
- Π: refine schema/keywords in `meta/circuit_schema.json`
- A: rerun indexing and query the corpus
- feedback: repeat until the map matches the territory

See `meta/PRNPIA_PLAYBOOK.md`.

## What's missing
See `meta/MISSING.md`.
