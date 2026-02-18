# Weaver Graph Field Integration

## Priority
- high

## Intent
- Integrate Web Graph Weaver crawl graph into runtime knowledge surfaces.
- Render crawler nodes in simulation field space with clickable interactions.
- Keep categorization explainable via field scores and edge weights.

## Requirements
- Backend fetches/derives crawler graph from Weaver service and attaches it to catalog/simulation payloads.
- Crawler graph contains nodes, edges, and field-category assignments compatible with simulation overlay.
- Frontend simulation overlay draws crawler graph with clickable URL nodes.
- Click on crawler URL node should witness-touch and open target URL.

## Defaults Applied
- Crawler graph fetched from `http://127.0.0.1:8793/api/weaver/graph` with capped limits.
- Unavailable Weaver service falls back to empty graph without breaking runtime endpoints.

## Risks
- Large crawler graph snapshots can increase response latency.
- External URLs may be blocked by browser popup policies.

## Definition of Done
- `/api/catalog` and `/api/simulation` expose `crawler_graph` with category stats.
- Simulation overlay visibly renders crawler graph nodes/edges and node click behavior.
- Tests and frontend build pass.
- Receipt entry appended with evidence.

## Change Log
- 2026-02-16: Completed crawler graph payload typing and simulation overlay parity for crawler nodes/edges.
- 2026-02-16: Added deterministic backend tests for crawler graph catalog/simulation integration.
- 2026-02-16: Updated catalog signature to include crawler graph status/stat deltas for websocket refresh visibility.
