---
source: user-session-2026-02-20
section: Single nexus graph unification
status: inprogress
reviewed_on: 2026-02-20
last_progress: 2026-02-20
---

# Part64 Deep Research Spec 09 - Single Nexus Graph Unification

## Priority
- high

## Complexity
- high (runtime schema unification + migration + frontend overlays + tests)

## Intent
- Replace parallel graph surfaces (`file_graph`, `logical_graph`, `crawler_graph`, truth/view contracts) with one canonical `nexus_graph`.
- Keep logical graph semantics by modeling them as a Logos-sourced subgraph inside `nexus_graph`.
- Preserve backward compatibility during migration with deterministic adapters.

## Requirements
- One node schema (`NexusNode`) and one edge schema (`NexusEdge`) for runtime and frontend payloads.
- Node role labels stay data-driven (`presence`, `resource`, `concept`, `anchor`, `logical`, `daimon`) instead of bespoke graph objects.
- `logical_graph` is emitted as a projection view from `nexus_graph` (not separately authored).
- `truth_graph` and `view_graph` contracts become ledger/projection metadata attached to `nexus_graph`.
- Ownership and purpose are explicit on daimoi packets and never inferred from rendering.
- Add schema-versioned adapters for existing consumers until full cutover.

## Current evidence in code
- Separate graph contracts are built in `part64/code/world_web/simulation.py:437` and `part64/code/world_web/simulation.py:478`.
- Logical graph is built independently in `part64/code/world_web/simulation.py:2719` and attached in `part64/code/world_web/simulation.py:8393`.
- Frontend has distinct `FileGraph` and `LogicalGraph` interfaces in `part64/frontend/src/types/index.ts:224` and `part64/frontend/src/types/index.ts:458`.
- Overlay still treats logic as a separate layer in `part64/frontend/src/app/coreSimulationConfig.ts:18`.

## Progress (2026-02-20)
- [x] Added canonical `NexusNode`, `NexusEdge`, `NexusGraph` TypeScript interfaces in `types/index.ts`.
- [x] Added canonical `Presence`, `Daimoi`, `Field`, `FieldRegistry` TypeScript interfaces.
- [x] Added record constants in `constants.py`: `NEXUS_GRAPH_RECORD`, `DAIMON_RECORD`, `PRESENCE_RECORD`, `SHARED_FIELD_RECORD`, `FIELD_REGISTRY_RECORD`.
- [x] Added `_build_canonical_nexus_node`, `_build_canonical_nexus_edge`, `_build_canonical_nexus_graph` in `simulation.py`.
- [x] Added `_build_field_registry` in `simulation.py`.
- [x] Added `_project_legacy_file_graph_from_nexus`, `_project_legacy_logical_graph_from_nexus` for backward compatibility.
- [x] Simulation state now emits `nexus_graph` and `field_registry` fields.
- [x] Exported canonical builders from `world_web/__init__.py`.
- [x] Added tests in `test_world_web_pm2.py`:
  - `test_simulation_state_includes_canonical_nexus_graph_and_field_registry`
  - `test_canonical_nexus_node_builder_maps_legacy_types`
  - `test_canonical_field_registry_is_bounded`

## Gaps vs target model
- Parallel graph payloads duplicate node identity and edge semantics.
- Logical nodes are not first-class nexus nodes with shared contracts.
- Ownership/purpose metadata is inconsistent across surfaces.
- Test coverage validates per-graph payloads but not a unified graph identity contract.

## Files planned
- `part64/code/world_web/simulation.py` (contract emission and compatibility adapters)
- `part64/code/world_web/catalog.py` (catalog payload shape)
- `part64/code/world_web/projection.py` (view generation from canonical graph)
- `part64/frontend/src/types/index.ts` (single graph types + deprecated aliases)
- `part64/frontend/src/components/Simulation/Canvas.tsx` (single graph overlay resolver)
- `part64/frontend/src/app/coreSimulationConfig.ts` (layer naming and toggles)
- `part64/code/tests/test_world_web_pm2.py` (payload contract tests)
- `part64/code/tests/test_presence_runtime.py` (projection and reconstruction invariants)

## Subtasks
1. Define canonical `nexus_graph.v1` schema and migration adapter matrix.
2. Refactor backend graph assembly to produce only canonical nodes/edges plus compatibility projections.
3. Refactor frontend to consume canonical graph while preserving current toggles.
4. Rebuild logical graph as Logos projection over canonical graph.
5. Add deterministic tests for identity stability and projection equivalence.
6. Remove deprecated payload branches after one release cycle.

## Risks
- Schema churn can break existing panel/overlay assumptions.
- Projection parity bugs can desync logical overlays during migration.
- Payload size can grow if compatibility adapters duplicate data.

## Existing issues
- No open GitHub issues found (`gh issue list --state all --limit 10 --json number,title,state` returned `[]`).

## Existing PRs
- No open GitHub PRs found (`gh pr list --state all --limit 10 --json number,title,state` returned `[]`).

## Definition of done
- `nexus_graph` is the single source graph contract in catalog and simulation payloads.
- `logical_graph` is generated from `nexus_graph` as a projection and matches current semantic coverage.
- Frontend overlays render from one graph source with no regression in layer controls.
- Targeted backend and frontend tests pass with deterministic seeds.
