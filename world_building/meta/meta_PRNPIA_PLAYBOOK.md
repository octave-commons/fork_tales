# P→R→N→Π→A→(feedback)→P — Playbook

## P — Put artifacts in the box
- Add files under `content/` (markdown, images, pdfs, exports, anything).
- Keep folder names meaningful; paths are indexed.

## R — Re-index
```bash
python tools/reindex.py
```

## N — Navigate
- Full text:
```bash
python tools/search.py "your query"
```
- Circuit-filtered:
```bash
python tools/search.py "your query" --circuit "Loop/Glitch"
```
- Nearest in 64-D map:
```bash
python tools/nearest.py --query "your query" --top 20
```

## Π — Update the model (schema)
Edit:
- `meta/circuit_schema.json` (keywords, descriptions)

Then rerun `reindex.py`.

## A — Apply
Use the indexed corpus to:
- Generate reports
- Draft new chapters
- Build training/evaluation sets for Cephalon
- Create “canon” indexes for Gates of Truth and related mythos

## feedback
Repeat until:
- search hits are stable,
- circuit clustering matches intuition,
- “ημ gap” feels like a useful boundary rather than a blind spot.
