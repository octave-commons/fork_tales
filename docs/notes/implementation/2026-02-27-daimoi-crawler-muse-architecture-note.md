<!-- SPDX-License-Identifier: GPL-3.0-or-later -->

# Daimoi, Crawler, and Muse Architecture Note

Date: 2026-02-27

## 1) Graph roles

- One canonical graph is used for runtime truth (`nexus_graph`), with view-focused compaction/projection layered on top for UI.
- Web graph contract uses two primary node roles:
  - `web:url`
  - `web:resource`
- Web relationship edges include:
  - `web:source_of` (resource -> source URL)
  - `web:links_to` (resource -> discovered URL links)
- Runtime collision/route data stays attached to daimoi rows (`graph_node_id`, `route_node_id`) instead of creating a separate graph model.

## 2) Daimoi outcomes and Nooi deposition

- Ambient motion deposition is still applied every tick for non-nexus particles.
- Outcome-conditioned deposition now uses bounded per-daimoi trail history:
  - A fixed trail ring stores recent `(x, y, vx, vy, tick)` samples keyed by daimoi id.
  - `food` outcomes deposit along the forward trail direction.
  - `death` outcomes deposit along the inverse trail direction.
- Outcome trail receipts include `daimoi_id`, `tick`, and `trail_steps` for auditability.

## 3) Fact snapshot and table export

- `build_facts_snapshot` still emits a canonical snapshot JSON, and now also exports JSONL tables in `world_state/facts/current/`.
- Exported tables:
  - `node`
  - `edge`
  - `daimoi`
  - `event_collision`
  - `event_timeout`
  - `capacity`
  - `web_url`
  - `web_resource`
  - `web_link`
  - `food`
  - `death`

## 4) Named logic query menu

- The named query surface now supports:
  - `explain_daimoi`
  - `recent_outcomes`
  - `crawler_status`
  - `web_resource_summary`
  - `graph_summary`
- Existing query names remain supported (`overview`, `neighbors`, `search`, `url_status`, `resource_for_url`, `recently_updated`, `role_slice`).

## 5) Muse grounding behavior

- Muse tool request parsing keeps explicit commands (`/facts`, `/graph ...`) and now also maps common intent phrases to named queries.
- Grounded replies continue to include receipts (`snapshot_hash`, `queries_used`).

## 6) Quick demo

Run from repository root:

```bash
bash part64/scripts/demo_daimoi_crawler_muse.sh
```

This script:

- starts the runtime stack,
- checks health endpoints,
- requests grounded Muse answers for:
  - crawler learning status,
  - daimoi win/loss explanation.
