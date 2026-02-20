---
source: user-session-2026-02-20
section: Model audit for alien concepts
status: inprogress
reviewed_on: 2026-02-20
---

# Part64 Deep Research Spec 11 - Model Audit: Alien Concepts

## Priority
- high

## Complexity
- analysis-only (documentation + mapping, no code changes)

## Intent
- Audit the codebase for concepts that do not fit the unified model: **Daimoi**, **Nexus**, **Presences**, **Fields**.
- Identify concepts that need to be migrated, deprecated, or reinterpreted as projections/views of canonical types.
- Provide a migration mapping for each alien concept.

## The Unified Model

The target model has exactly four primitive types:

| Type | Description |
|------|-------------|
| **Presence** | AI physics-based agent with spec embedding, need vector, priority, mass |
| **Nexus** | Graph node/particle representing a resource with embedding, capacity, demand, role |
| **Daimoi** | Free particle with carrier embedding, seed embedding, type distribution, owner, location |
| **Field** | Shared global scalar field (demand, flow, entropy, graph) that all presences contribute to |

Everything else must be a projection, view, or instance of one of these four.

---

## Alien Concepts Found

### 1. Separate Graph Payloads (should be unified Nexus graph)

| Concept | Location | Record Constant | Migration Target |
|---------|----------|-----------------|------------------|
| `FileGraph` | `types/index.ts:273` | `ημ.file-graph.v1` | Unified `nexus_graph` with `role=file` |
| `CrawlerGraph` | `types/index.ts:341` | `ημ.crawler-graph.v1` | Unified `nexus_graph` with `role=crawler` |
| `LogicalGraph` | `types/index.ts:458` | `ημ.logical-graph.v1` | Unified `nexus_graph` with `role=logical` (Logos projection) |
| `TruthGraph` | `simulation.py:103` | `eta-mu.truth-graph.v1` | Ledger metadata on `nexus_graph` |
| `ViewGraph` | `simulation.py:105` | `eta-mu.view-graph.v1` | Projection metadata on `nexus_graph` |

**Evidence:**
- `part64/code/world_web/simulation.py:437` - `_build_truth_graph_contract`
- `part64/code/world_web/simulation.py:478` - `_build_view_graph_contract`
- `part64/code/world_web/simulation.py:2719` - `_build_logical_graph`
- `part64/frontend/src/types/index.ts:273` - `interface FileGraph`
- `part64/frontend/src/types/index.ts:341` - `interface CrawlerGraph`
- `part64/frontend/src/types/index.ts:458` - `interface LogicalGraph`

**Migration:** All graph payloads become projections of a single `nexus_graph`.

---

### 2. Node Type Fragmentation (should be Nexus with role labels)

| Node Kind | Current Home | Migration |
|-----------|--------------|-----------|
| `file_node` | `FileGraph.file_nodes` | `NexusNode` with `role="file"` |
| `field_node` | `FileGraph.field_nodes` | `NexusNode` with `role="field"` |
| `tag_node` | `FileGraph.tag_nodes` | `NexusNode` with `role="tag"` |
| `crawler_node` | `CrawlerGraph.crawler_nodes` | `NexusNode` with `role="crawler"` |
| `logical_node` (file/fact/rule/derivation/contradiction/gate/event/tag) | `LogicalGraph.nodes` | `NexusNode` with `role` matching kind |

**Evidence:**
- `part64/code/world_web/simulation.py:419-432` - node_type branching
- `part64/frontend/src/types/index.ts:224-262` - `FileGraphNode` with `node_type: "field" | "file" | "presence" | "tag" | "crawler"`

**Migration:** Single `NexusNode` schema with `role` field.

---

### 3. Graph-Adjacent Concepts (should be Nexus or Daimoi)

| Concept | Description | Migration |
|---------|-------------|-----------|
| `SimPoint` | Simulation particle | `Daimoi` with location in continuous space |
| `BackendFieldParticle` | Field visualization particle | `Daimoi` or derived from field sampling |
| `FileGraphEmbeddingParticle` | Embedding-space particle | `Daimoi` with semantic embedding |
| `Echo` | Transient visual echo | `Daimoi` with short life |

**Evidence:**
- `part64/frontend/src/types/index.ts:35` - `interface SimPoint`
- `part64/frontend/src/types/index.ts:45` - `interface BackendFieldParticle`
- `part64/frontend/src/types/index.ts:133` - `interface FileGraphEmbeddingParticle`
- `part64/frontend/src/types/index.ts:531` - `interface Echo`

---

### 4. Presence-Related Concepts (should be Presence or projections)

| Concept | Description | Migration |
|---------|-------------|-----------|
| `WorldPresence` | Game-world presence description | `Presence` with world-specific metadata |
| `FileGraphConceptPresence` | Concept presence in file graph | `Presence` or `NexusNode` with `role="concept"` |
| `FileGraphOrganizerPresence` | Organizer presence in file graph | `Presence` or `NexusNode` with `role="organizer"` |
| `MuseRuntimeMuseRow` | Muse workspace instance | `Presence` (Muse is a presence type) |
| `PresenceMusicPresence` | Music collaboration presence | `Presence` with music-specific lens |
| `PresenceImpact` | Presence influence telemetry | Derived field/diagnostic, not a primitive |

**Evidence:**
- `part64/frontend/src/types/index.ts:551` - `interface WorldPresence`
- `part64/frontend/src/types/index.ts:195` - `interface FileGraphConceptPresence`
- `part64/frontend/src/types/index.ts:210` - `interface FileGraphOrganizerPresence`
- `part64/frontend/src/types/index.ts:898` - `interface MuseRuntimeMuseRow`
- `part64/frontend/src/types/index.ts:1685` - `interface PresenceMusicPresence`
- `part64/frontend/src/types/index.ts:787` - `interface PresenceImpact`

---

### 5. World/Lore Concepts (should be Presence/Nexus projections)

| Concept | Description | Migration |
|---------|-------------|-----------|
| `WorldPerson` | Person in game world | `NexusNode` with `role="person"` |
| `WorldSong` | Song in game world | `NexusNode` with `role="song"` |
| `WorldBook` | Book in game world | `NexusNode` with `role="book"` |
| `WorldSummary` | World state summary | Projection over world nexus nodes |
| `MythSummary` | Myth ledger summary | Diagnostic/field derived from ledger events |
| `EntityState` / `EntityVitals` | Entity state in simulation | `NexusNode` or `Daimoi` depending on mobility |
| `EntityManifestItem` | Entity manifest entry | `NexusNode` manifest |

**Evidence:**
- `part64/frontend/src/types/index.ts:557` - `interface WorldPerson`
- `part64/frontend/src/types/index.ts:569` - `interface WorldSong`
- `part64/frontend/src/types/index.ts:577` - `interface WorldBook`
- `part64/frontend/src/types/index.ts:585` - `interface WorldSummary`
- `part64/frontend/src/types/index.ts:540` - `interface MythSummary`
- `part64/frontend/src/types/index.ts:27` - `interface EntityState`
- `part64/frontend/src/types/index.ts:1468` - `interface EntityManifestItem`

---

### 6. Truth/Proof Concepts (should be Nexus with provenance)

| Concept | Description | Migration |
|---------|-------------|-----------|
| `TruthState` | Truth binding state | Ledger metadata + nexus projection |
| `TruthStateClaim` | Individual claim | `NexusNode` with `role="claim"` |
| `TruthStateProofEntry` | Proof entry | `NexusNode` with `role="proof"` or edge |
| `PainField` | Test failure heat map | Diagnostic derived from test-failure nexus nodes |
| `PainFieldTestFailure` | Test failure node | `NexusNode` with `role="test_failure"` |
| `PainFieldNodeHeat` | Node heat in pain field | Field-derived diagnostic |
| `PainFieldDebugTarget` | Debug target in pain field | Field-derived diagnostic |

**Evidence:**
- `part64/frontend/src/types/index.ts:382` - `interface TruthState`
- `part64/frontend/src/types/index.ts:365` - `interface TruthStateClaim`
- `part64/frontend/src/types/index.ts:375` - `interface TruthStateProofEntry`
- `part64/frontend/src/types/index.ts:517` - `interface PainField`
- `part64/frontend/src/types/index.ts:481` - `interface PainFieldTestFailure`

---

### 7. Governance/Routing Concepts (should be Field-derived or Ledger)

| Concept | Description | Migration |
|---------|-------------|-----------|
| `CouncilChamber` | Decision council class | Ledger events + presence votes |
| `CouncilVote` | Individual vote | Ledger event |
| `CouncilTally` | Vote tally | Diagnostic derived from vote events |
| `CouncilDecision` | Decision record | Ledger event |
| `CouncilSnapshot` | Council state | Diagnostic derived from events |
| `TaskQueue` | Task queue class | Ledger event stream |
| `TaskQueueTask` | Task in queue | `Daimoi` with task metadata or ledger event |
| `TaskQueueSnapshot` | Queue state | Diagnostic derived from events |
| `SimulationGrowthGuard` | Growth pressure guard | Field-derived diagnostic |
| `SimulationGrowthEvent` | Growth event | Ledger event |
| `DriftGateBlock` | Gate block reason | Diagnostic derived from gate state |
| `DriftScanPayload` | Drift scan results | Diagnostic derived from field state |
| `StudyStability` | System stability score | Field-derived diagnostic |
| `StudySignals` | System signals | Field-derived diagnostics |
| `StudySnapshotPayload` | Study snapshot | Aggregated field diagnostics |

**Evidence:**
- `part64/code/world_web/chamber.py:752` - `class CouncilChamber`
- `part64/code/world_web/chamber.py:1314` - `class TaskQueue`
- `part64/frontend/src/types/index.ts:1304-1367` - Council interfaces
- `part64/frontend/src/types/index.ts:1287-1302` - Task queue interfaces
- `part64/frontend/src/types/index.ts:1040-1079` - `SimulationGrowthGuard`
- `part64/frontend/src/types/index.ts:1415-1466` - Study interfaces

---

### 8. Runtime/Telemetry Concepts (should be Field-derived diagnostics)

| Concept | Description | Migration |
|---------|-------------|-----------|
| `PresenceRuntimeSnapshot` | Presence runtime state | Diagnostic derived from presence states |
| `PresenceRuntimeCounts` | Runtime event counts | Field-derived braid metric |
| `PresenceDynamics` | Presence dynamics telemetry | Field-derived diagnostics |
| `MuseRuntimeSnapshot` | Muse runtime state | Diagnostic derived from Muse presence |
| `MuseEvent` | Muse event | Ledger event |
| `ResourceHeartbeatSnapshot` | Resource heartbeat | Diagnostic derived from resource nexus |
| `ResourceDeviceSnapshot` | Device snapshot | Nexus node state |
| `ResourceDaimoiSummary` | Resource daimoi stats | Field-derived diagnostic |
| `ResourceConsumptionSummary` | Consumption stats | Field-derived diagnostic |
| `DaimoiProbabilisticSummary` | Daimoi stats | Field-derived diagnostic |
| `ForkTaxState` | Fork tax state | Ledger-derived diagnostic |
| `GhostRoleState` | Ghost role state | Presence state (auto-commit presence) |
| `WitnessThreadState` | Witness thread state | Diagnostic derived from ledger continuity |
| `WitnessLineagePayload` | Git lineage state | Diagnostic derived from repo state |

**Evidence:**
- `part64/frontend/src/types/index.ts:882-896` - Presence runtime interfaces
- `part64/frontend/src/types/index.ts:922-944` - Muse runtime interfaces
- `part64/frontend/src/types/index.ts:839-864` - Resource heartbeat interfaces
- `part64/frontend/src/types/index.ts:1154-1184` - Resource summary interfaces
- `part64/frontend/src/types/index.ts:946-1028` - Daimoi summary interface
- `part64/frontend/src/types/index.ts:804-822` - Fork tax / ghost role interfaces
- `part64/frontend/src/types/index.ts:1218-1278` - Witness interfaces

---

### 9. UI Projection Concepts (should be view/projection layer only)

| Concept | Description | Migration |
|---------|-------------|-----------|
| `UIProjectionBundle` | Full UI projection | View layer over nexus graph (no changes needed - already a view) |
| `UIProjectionElement` | Projection element | View layer |
| `UIProjectionElementState` | Element state | Derived from nexus/field state |
| `UIProjectionLayout` | Layout state | View layer |
| `UIProjectionCoherence` | Coherence state | Field-derived diagnostic |
| `UIProjectionFieldSnapshot` | Field snapshot | Field sample |
| `UIProjectionChatSession` | Chat session | View layer (chat with presence) |

**Evidence:**
- `part64/frontend/src/types/index.ts:750-785` - UIProjectionBundle
- `part64/frontend/src/types/index.ts:683-718` - UIProjection element interfaces
- `part64/frontend/src/types/index.ts:667-681` - UIProjectionCoherence

**Note:** UI projection concepts are **view layer** and don't need migration - they're already projections over the underlying model.

---

### 10. External Event Concepts (should be Nexus nodes or Ledger events)

| Concept | Record | Description | Migration |
|---------|--------|-------------|-----------|
| `WikimediaEvent` | `eta-mu.wikimedia-event.v1` | Wikipedia edit event | `NexusNode` with `role="wikimedia_event"` or ledger event |
| `NWSAlert` | `eta-mu.nws-alert.v1` | Weather alert | `NexusNode` with `role="nws_alert"` |
| `SWPCAlert` | `eta-mu.swpc-alert.v1` | Space weather alert | `NexusNode` with `role="swpc_alert"` |
| `GIBSLayer` | `eta-mu.gibs-layer.v1` | Satellite layer | `NexusNode` with `role="gibs_layer"` |
| `EONETEvent` | `eta-mu.eonet-event.v1` | Earth event | `NexusNode` with `role="eonet_event"` |
| `EMSCEvent` | `eta-mu.emsc-event.v1` | Earthquake event | `NexusNode` with `role="emsc_event"` |
| `WorldLogEvent` | - | World log event | `NexusNode` with `role="log_event"` |

**Evidence:**
- `part64/code/world_web/chamber.py:167-178` - External event record constants
- `part64/frontend/src/types/index.ts:1498-1530` - WorldLog interfaces

---

### 11. Artifact/Ingest Concepts (should be Nexus nodes)

| Concept | Record | Description | Migration |
|---------|--------|-------------|-----------|
| `ArchiveManifest` | `ημ.archive-manifest.v1` | Archive manifest | `NexusNode` with `role="archive"` |
| `IngestRegistry` | `ημ.ingest-registry.v1` | Ingest registry | Nexus metadata |
| `EmbeddingDB` | `ημ.embedding-db.v1` | Embedding database | Nexus embeddings |
| `DocMeta` | `ημ.docmeta.v1` | Document metadata | `NexusNode` metadata |
| `Packet` | `ημ.packet.v1` | Ingest packet | `Daimoi` (packet = message) |
| `ImageComment` | `ημ.image-comment.v1` | Image comment | `NexusNode` with `role="comment"` |
| `ChatTraining` | `ημ.chat-training.v1` | Chat training data | `NexusNode` with `role="training"` |

**Evidence:**
- `part64/code/world_web/constants.py:159-171,509,519,626` - Record constants

---

### 12. Music/Audio Concepts (should be Nexus nodes or Presence lens)

| Concept | Description | Migration |
|---------|-------------|-----------|
| `InstrumentState` | Instrument state | Presence lens or nexus state |
| `InstrumentPad` | Instrument pad | `NexusNode` with `role="pad"` |
| `VoiceLine` | Voice line | `NexusNode` with `role="voice_line"` |
| `VoicePack` | Voice pack | `NexusNode` with `role="voice_pack"` |
| `PresenceMusicSession` | Music session | Nexus graph (session = subgraph) |
| `PresenceMusicUtterance` | Utterance | `Daimoi` or `NexusNode` |
| `PresenceMusicDecision` | Decision | Ledger event |
| `PresenceMusicPlan` | Plan | `NexusNode` with `role="plan"` |
| `PresenceMusicTrack` | Track | `NexusNode` with `role="track"` |
| `PresenceMusicClip` | Clip | `NexusNode` with `role="clip"` |
| `PresenceMusicLyric` | Lyric | `NexusNode` with `role="lyric"` |
| `PresenceMusicLoop` | Loop | `NexusNode` with `role="loop"` |
| `EtaMuLedgerRow` | Ledger row | Ledger event |
| `MixMeta` | Mix metadata | Nexus metadata |

**Evidence:**
- `part64/frontend/src/types/index.ts:1617-1798` - Music/audio interfaces

---

### 13. Miscellaneous Concepts

| Concept | Description | Migration |
|---------|-------------|-----------|
| `CatalogItem` | Catalog item | Nexus node reference |
| `NamedFieldItem` | Named field | `NexusNode` with `role="field"` (already partially there) |
| `ChatMessage` | Chat message | `Daimoi` with chat metadata |
| `MuseWorkspaceContext` | Workspace context | Presence state |
| `NooiFieldCell` / `NooiFieldGrid` | Nooi field grid | Field sample grid (already field-derived) |

**Evidence:**
- `part64/frontend/src/types/index.ts:10-21` - `CatalogItem`
- `part64/frontend/src/types/index.ts:1480-1489` - `NamedFieldItem`
- `part64/frontend/src/types/index.ts:1658-1669` - `ChatMessage`
- `part64/frontend/src/types/index.ts:1652-1656` - `MuseWorkspaceContext`
- `part64/frontend/src/types/index.ts:1186-1216` - Nooi field interfaces

---

## Summary Table

| Category | Concepts Found | Migration Strategy |
|----------|----------------|-------------------|
| **Graph payloads** | 5 | Unify to single `nexus_graph` |
| **Node types** | 10+ | Single `NexusNode` with `role` field |
| **Particles** | 4 | Convert to `Daimoi` |
| **Presence variants** | 6 | Single `Presence` with type/lens |
| **World/Lore** | 8 | Nexus nodes with role labels |
| **Truth/Proof** | 7 | Nexus nodes + ledger + field diagnostics |
| **Governance** | 15 | Ledger events + field diagnostics |
| **Telemetry** | 14 | Field-derived braid metrics |
| **UI Projection** | 7 | Keep as view layer (no changes) |
| **External events** | 7 | Nexus nodes or ledger events |
| **Artifacts** | 7 | Nexus nodes with role labels |
| **Music/Audio** | 14 | Nexus nodes + Daimoi + ledger |
| **Miscellaneous** | 5 | Various mappings |

---

## Migration Priorities

### High Priority (blocks unification)
1. **Graph payload unification** - `file_graph`, `crawler_graph`, `logical_graph` → `nexus_graph`
2. **Node type unification** - All node variants → `NexusNode` with `role`
3. **Particle unification** - `SimPoint`, `Echo`, etc. → `Daimoi`

### Medium Priority (semantic consistency)
4. **Presence consolidation** - All presence variants → single `Presence` type
5. **Truth/Proof migration** - Claims, proofs → Nexus nodes with provenance
6. **Governance migration** - Council, TaskQueue → ledger events

### Low Priority (cleanup)
7. **World/Lore migration** - Game entities → Nexus nodes
8. **Artifact migration** - Archive, ingest → Nexus nodes
9. **Music/Audio migration** - Music entities → Nexus nodes

---

## Definition of Done

- All concepts in this audit are mapped to one of: Presence, Nexus, Daimoi, Field, or "View Layer".
- No alien concepts remain without a migration path.
- Spec 09 (unified nexus graph) and Spec 10 (shared fields) updated to reference this audit.
- Migration checklist added to implementation tracking.
