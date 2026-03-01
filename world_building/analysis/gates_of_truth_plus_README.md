# Gates of Truth — Relationship + Ideology Pack (Plus)

Generated from `Gates_of_Truth_Production_Bundle_v1.zip`.

## Files
- `relationship.graphml` — import into Gephi/Cytoscape
- `nodes.csv` — per-node metrics (centralities, community id, ideology vector)
- `edges.csv` — weighted co-occurrence edges + polarity (supportive/neutral/antagonistic)
- `character_chapter_mentions.csv` — mention counts per chapter (matrix)
- `chapters.csv` — chapter-level stats + 8-circuit scores + keywords
- `scenes.csv` — scene segmentation (blank-line split) with entities per scene
- `ideology.json` — per-node 8-circuit scores + top keywords
- `relationship_network.png` — quick visual
- `relationship_network.dot` — Graphviz DOT

## Edge polarity
Polarity is a **bootstrap heuristic**:
- collect scene text where both characters appear
- compute simple sentiment = (#positive words) - (#negative words)
- average per edge → label supportive/neutral/antagonistic

This is navigation, not truth. Tune lexicons in the regeneration script.

## Regenerate
See `operation-mindfuck/tools/analyze_gates_of_truth_plus.py`.
