# Web Graph Weaver (Draft)

## Goal

Implement a live, ethical web graph crawler system with:

- a crawler service that emits transparent traversal events,
- a graph model with inspectable node/edge semantics,
- a real-time dashboard showing graph growth and compliance posture.

## Priority

High

## Complexity Estimate

High (service + protocol + live graph rendering)

## Requirements

1. Ethical crawl guardrails are first-class: robots.txt, crawl-delay, nofollow, explicit user-agent, no bypass behavior.
2. Service exposes real-time event stream and status/control REST endpoints.
3. Graph visibly grows in UI from a seed URL.
4. Compliance and skip/blocked actions are explicit and auditable in event stream.
5. System supports pause/resume and conservative defaults.

## Open Questions

- Which persistence backend for baseline?
  - Default chosen: append-only JSONL + periodic JSON snapshot (inspectable, no new DB infra).
- Which service port should be used?
  - Default chosen: `8793` for the crawler service to avoid collision with existing world daemon (`8787`).

## Risks

1. Public web variability can induce unstable crawl behavior; mitigate with strict limits and fail-safe skips.
2. Real-time rendering can degrade with large node counts; mitigate with canvas simplification and filtered views.
3. Robots parsing edge cases; mitigate with conservative interpretation and explicit blocked events.

## Existing Issues / PRs

- `gh issue list` unavailable in this local checkout (no git remotes configured).
- `gh pr list` unavailable in this local checkout (no git remotes configured).

## Phased Plan

### Phase 1 - Crawler Core

- Build Node service runtime with frontier queue, URL normalization, robots cache, and domain rate limiting.
- Implement event bus and WebSocket broadcast for crawl lifecycle events.
- Add REST endpoints for status, start/pause/resume, opt-out, and graph snapshots.

### Phase 2 - Graph Data Model

- Represent URL, domain, and content-type nodes.
- Represent hyperlink, redirect, and domain-membership edges.
- Emit graph deltas in real time and persist append-only deltas for auditability.

### Phase 3 - Dashboard UX

- Build Web Graph Weaver panel with:
  - status + compliance cards,
  - live graph canvas,
  - real-time event stream,
  - filters and path-highlighting affordances.
- Integrate panel into current app while preserving existing visual language.

### Phase 4 - Verification + Documentation

- Run frontend build and relevant backend checks.
- Validate crawler lifecycle from one seed URL.
- Document architecture, rate-limiting, robots handling, and graph schema.

## Code Targets

- `.../ημ_op_mf_part_64/code/web_graph_weaver.js` (new crawler service)
- `.../ημ_op_mf_part_64/code/package.json` (scripts/deps)
- `.../ημ_op_mf_part_64/ecosystem.config.cjs` (optional process wiring)
- `.../ημ_op_mf_part_64/frontend/src/components/Panels/WebGraphWeaverPanel.tsx` (new panel)
- `.../ημ_op_mf_part_64/frontend/src/App.tsx` (panel integration)
- `.../ημ_op_mf_part_64/world_state/constraints.md` (append-only constraints)
- `receipts.log` (append-only receipt)

## Definition of Done

1. Starting from 1 seed URL, nodes/edges appear live in graph panel.
2. robots-blocked and skipped actions are visibly logged.
3. Pause/resume works via UI controls.
4. Duplicate URLs are not recrawled.
5. Compliance metrics are visible and update in real time.
