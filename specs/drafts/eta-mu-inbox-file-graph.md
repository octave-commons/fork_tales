# ημ Inbox File Graph Integration

## Priority
- high

## Intent
- Treat `.ημ/` as an inbox for new artifacts.
- Integrate inbox files into runtime knowledge and simulation graph.
- Keep `.ημ/` empty after processing so it remains an active inbox.

## Requirements
- Backend must ingest files from `/home/err/devel/vaults/fork_tales/.ημ/` into a durable knowledge index.
- Ingestion must preserve source artifacts by moving them to an archive path outside `.ημ/`.
- `.ημ/` should be empty after successful processing (except possible empty directories before cleanup).
- Simulation payload must expose a file graph with:
  - file nodes
  - field nodes
  - edges linking file nodes to categorized fields
  - node metadata sufficient for click/open interactions
- Field categorization must include text-aware classification (path/name/content heuristics).
- Frontend simulation overlay must render graph nodes and edges and allow clicking file nodes.

## Open Questions
- none (defaults applied)

## Defaults Applied
- Processing mode is auto-apply during catalog/simulation refresh.
- Inbox files are moved to `.opencode/knowledge/archive/` and indexed in `.opencode/runtime/eta_mu_knowledge.v1.jsonl`.
- Clicking a file node records a witness touch and opens the file URL.

## Risks
- Large inbox bursts can increase refresh latency while initial ingestion runs.
- Text extraction from mixed encodings can be noisy.
- Browser popup policies may block file-open behavior in some contexts.

## Phases
1. Add backend inbox ingestion + archive + knowledge index.
2. Add backend file graph model in simulation payload and catalog metadata.
3. Extend frontend types and simulation overlay rendering/interactions.
4. Validate with tests/build/runtime endpoint checks and append receipt.

## Candidate Files
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/world_web.py`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/tests/test_world_web_pm2.py`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/types/index.ts`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/components/Simulation/Canvas.tsx`
- `receipts.log`

## Existing Issues / PRs
- No git remote configured; no GH issue/PR references available.

## Complexity Estimate
- medium-high (cross-cutting backend + frontend + data migration semantics)

## Definition of Done
- `.ημ/` files are ingested and moved out through runtime processing.
- Simulation renders categorized file graph with clickable file nodes.
- `/api/catalog`, `/api/simulation`, `/api/ui/projection`, and `/ws` remain healthy.
- Backend tests and frontend build pass.
- Receipt entry appended with refs and validation evidence.
