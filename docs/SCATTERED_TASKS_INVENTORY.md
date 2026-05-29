# Comprehensive Scattered Tasks & Specs Inventory

**Generated:** 2026-03-12  
**Scope:** `/home/err/devel/vaults/fork_tales/` and parent workspace references

---

## Table of Contents
1. [Code TODOs & FIXMEs](#1-code-todos--fixmes)
2. [Active Specs (specs/)](#2-active-specs)
3. [Draft Specs (specs/drafts/) — Unimplemented](#3-draft-specs--unimplemented)
4. [Incoming Specs — Unimplemented Research Ideas](#4-incoming-specs--unimplemented-research-ideas)
5. [Security Tracking Pipeline — In Progress](#5-security-tracking-pipeline--in-progress)
6. [MCP Lith Nexus — Status](#6-mcp-lith-nexus--status)
7. [CLI Tools — Status](#7-cli-tools--status)
8. [Contracts — Unimplemented Handlers](#8-contracts--unimplemented-handlers)
9. [Docker LLM Proxy — Missing Features](#9-docker-llm-proxy--missing-features)
10. [OpenCode Agent/Skill/Command System — Designed but Unverified](#10-opencode-agentskillcommand-system)
11. [World Building — Narrative & Creative Backlog](#11-world-building--narrative--creative-backlog)
12. [Notes & Research — Scattered Ideas](#12-notes--research--scattered-ideas)
13. [.ημ State & Known Issues](#13-ημ-state--known-issues)
14. [.η Observational Layer — Skeleton](#14-η-observational-layer--skeleton)
15. [.sisyphus — Empty](#15-sisyphus--empty)
16. [.fork_Π_ημ_frags — Visual Archive](#16-fork_π_ημ_frags--visual-archive)
17. [Recent Receipts Summary](#17-recent-receipts-summary)
18. [Prioritized Dreams List](#18-prioritized-dreams-list)

---

## 1. Code TODOs & FIXMEs

| File | Line | Type | Description |
|------|------|------|-------------|
| `contracts/契_ημ_ingest_v1.mjs` | 411 | TODO | Implement local vision + embedding stack |
| `contracts/契_ημ_ingest_v1.mjs` | 413 | TODO | `vision.describe` — local vision model |
| `contracts/契_ημ_ingest_v1.mjs` | 416 | TODO | `embed.write` — vectorstore write |
| `contracts/契_ημ_ingest_v1.mjs` | 511 | TODO | Decrement concurrency on handler completion |
| `part64/code/world_web/simulation_nexus.py` | 830 | TODO | Add attribution to top_contributors |
| `hacks/shared/data-structures/maps/ordered.sibilant` | 74 | TODO | Write node-based list operations for ordered maps |
| `lib/analyze.mjs` | 62 | TODO | Placeholder TODO in analyze output |
| `docker-llm-proxy/src/rotator_library/client.py` | 129 | TODO | Remove litellm workaround |
| `docker-llm-proxy/src/rotator_library/client.py` | 3707 | HACK | Fix global requests if present |
| `docker-llm-proxy/src/rotator_library/client.py` | 3709 | TODO | Properly track archived requests per quota group |
| `docker-llm-proxy/src/rotator_library/utils/suppress_litellm_warnings.py` | 13 | TODO | Remove litellm warning suppression workaround |
| `docker-llm-proxy/README.md` | 34,268,352 | TODO | Add TUI screenshot placeholders (3x) |

---

## 2. Active Specs

Located in `specs/`:

| Spec | Status | Summary |
|------|--------|---------|
| `c-sim-perf-optimization-v1.md` | **Designed, not started** | C simulation hot-path optimization (CSR edges, force caching, chaos decimation). Detailed 5-phase plan with benchmarks. |
| `npu-benchmark-spec.md` | **Designed, not started** | NPU embedding benchmark protocol — allocation-free harness, device verification, MRL dimension ladder. |
| `memory-churn-gc-remediation-v1/v2/v3.md` | **Designed, partially addressed** | Bridge allocation reduction, retention bounds, native scratch hygiene. 3 evolving versions. |
| `2026-02-22-barnes-hut-c-runtime-request.md` | **Request captured** | Barnes-Hut tree for O(N log N) force approximation in C sim. |
| `incoming-viewgraph-expansion.md` | **Incoming** | Dynamic ViewGraph expansion/compaction under demand. |
| `incoming-reinforce-learning.md` | **Incoming** | REINFORCE policy gradient for Presence absorption. |
| `incoming-audit-visualization.md` | **Incoming** | Auditable path visualization, gravity heatmaps, receipt inspector UI. |

### Security Tracking Sub-specs (`specs/security-tracking/`):

| Spec | Status |
|------|--------|
| `01-github-security-extraction-foundation.md` | In progress |
| `02-deterministic-security-ranker.md` | In progress |
| `03-proximity-feature-engine.md` | In progress, needs tuning |
| `04-hmm-entity-state-smoothing.md` | In progress, not fully policy-driving |
| `05-security-pipeline-hardening.md` | In progress, throughput unverified |
| `06-latent-cyber-regime-context.md` | In progress, empty-result fallback wired |

---

## 3. Draft Specs — Unimplemented

Located in `specs/drafts/` — **40 files**, all representing designed-but-unbuilt features:

### Core Runtime & Graph
- **`part64-runtime-system-implementation.md`** — Full runtime system implementation plan
- **`part64-simulation-smoothing.md`** — Simulation smoothing/interpolation
- **`nexus-daimoi-simulation-mathematical-alignment.md`** — Math alignment for Nexus/Daimoi
- **`nexus-daimoi-semantic-fields.md`** — Semantic field dynamics
- **`weaver-graph-field-integration.md`** — Weaver↔graph field integration

### Deep Research Series (12 specs)
- `part64-deep-research-01` through `part64-deep-research-12` — A complete research program covering:
  - TruthGraph↔ViewGraph lossless projection
  - Graph runtime gravity and pricing
  - Presence needs, mass, and gravity
  - Daimoi packets and routing
  - Online learning (REINFORCE)
  - Hybrid simulation compaction
  - Evaluation workloads and ablations
  - Diagnostics and visualization audit
  - Unified Nexus graph
  - Shared fields and Daimoi dynamics
  - Model audit and alien concepts
  - Smart card field priority

### Infrastructure & Tooling
- **`mcp-lith-nexus.md`** — MCP Lith Nexus server (partially implemented — see section 6)
- **`promptdb-lisp-interpreter-v0.1.md`** — Deterministic Lisp interpreter for PromptDB
- **`tech-debt-hardening-p0.md`** — Cache integrity and lint hygiene (session log suggests partial fix)
- **`frontend-testing-coverage-ci.md`** — Frontend test coverage/CI
- **`frontend-mock-patterns.md`** — Frontend mock patterns
- **`docker-simulation-dashboard-nginx.md`** — Docker simulation dashboard via nginx
- **`fork-tax-git-cadence-protocol.md`** — Git cadence protocol for Π fork tax
- **`world-web-fastapi-mvc-migration.md`** — FastAPI MVC migration for world_web

### Protocol & UX
- **`wire-world-part64-intent.md`** — Wire PromptDB into world runtime (slash commands, presence API)
- **`eta-mu-inbox-file-graph.md`** — Treat `.ημ/` as an inbox for graph integration
- **`eta-mu-ingest-text-image-v1.md`** — Text/image ingestion pipeline
- **`eta-observational-layer.md`** — `.η/` observational layer protocol
- **`truth-binding-v1.md`** — Truth as judged claims with operators
- **`presence-impact-flow-ghost.md`** — Presence impact and ghost flow
- **`presence-webrtc-communication-reset.md`** — WebRTC audio-first Presence communication
- **`inspiration-atlas-field-ui.md`** — Inspiration atlas with field-weighted UI
- **`house-ui-projection-v1.md`** — House UI projection

### Memory/GC (draft versions)
- `memory-churn-gc-remediation-v1/v2/v3.md` — Draft evolution of memory remediation

---

## 4. Incoming Specs — Unimplemented Research Ideas

From the deep research report, three major features remain as "incoming" (designed but no implementation started):

1. **Dynamic ViewGraph Expansion** — On-demand cluster expansion/compaction in the UI
2. **REINFORCE Learning for Presence Absorption** — Stochastic policy gradient with trace buffers and eligibility traces
3. **Auditable Path Visualization** — Shortest path overlays, gravity heatmaps, receipt inspector, river flow visualization

---

## 5. Security Tracking Pipeline — In Progress

**Open tasks from `specs/security-tracking/implementation-order.md`:**

- [ ] **P0**: Confirm non-provisional global evidence returns under live crawl
- [ ] **P1**: Continue tuning weak-label/threshold calibration
- [ ] **P1**: Stabilize or disable LLM blend path (eliminate `llm_invalid_json`)
- [ ] **P1**: Add status dashboard for dedupe/corroboration/regime threshold impacts

---

## 6. MCP Lith Nexus — Status

**Location:** `mcp-lith-nexus/`  
**Status:** Phase 1 partially implemented (package skeleton, server, service, Lith parser, query, write paths exist as TypeScript)

**What exists:**
- TypeScript package with `@modelcontextprotocol/sdk` dependency
- Source files: `index.ts`, `http.ts`, `server.ts`, `service.ts`, `backend.ts`, `config.ts`, `lith.ts`, `query.ts`, `write.ts`, `types.ts`, `utils.ts`, `format.ts`, `runtime.ts`
- Tests: `server.test.ts`, `service.test.ts`
- Config file: `mcp.lith-nexus.config.lith`

**What remains (from spec):**
- [ ] Phase 2: Lith parser integration with Python-side `part64` canonical graph
- [ ] Phase 3: Full Nexus graph query with `(query ...)` evaluator
- [ ] Phase 4: Deterministic write paths (`promptdb.create_fact`, `nexus.create_resource`)
- [ ] Phase 5: Integration tests, HTTP surface mount, nginx proxy, `opencode.jsonc` remote MCP entry

---

## 7. CLI Tools — Status

Located in `cli/`:

| Tool | Status | Description |
|------|--------|-------------|
| `live-choir.mjs` | **Working (mock mode)** | Multi-entity chat with frame firewall. Currently uses mock provider. |
| `frame-firewall.mjs` | **Working** | Analyzes utterances for manipulation frames (guilt, authority, urgency, etc.) |
| `ralph-loop.mjs` | **Working** | Iterative agent loop with completion promises |
| `ulw-loop.mjs` | **Working** | Wrapper around ralph-loop with `--ulw` flag |

**Unimplemented CLI dreams:**
- Real LLM provider integration for live-choir (currently mock only)
- Vision + embedding stack for ingest contract

---

## 8. Contracts — Unimplemented Handlers

In `contracts/契_ημ_ingest_v1.mjs`:
- **`vision.describe`** — Needs local vision model implementation
- **`embed.write`** — Needs vectorstore write implementation
- **Concurrency decrement** — Handler completion tracking incomplete

---

## 9. Docker LLM Proxy — Missing Features

From `docker-llm-proxy/src/proxy_app/quota_viewer.py` TODO list:

**Display Improvements:**
- [ ] Color legend/help screen
- [ ] Show credential email/project ID
- [ ] Keyboard shortcut hints
- [ ] Terminal resize / responsive layout

**Global Stats Fix:**
- [ ] Track archived requests per quota group (avoid double-counting)

**Data & Refresh:**
- [ ] Auto-refresh option
- [ ] Last refresh timestamp prominence
- [ ] Cache invalidation for view switching
- [ ] Non-OAuth provider support (API keys)

**Remote Management:**
- [ ] Connection testing before save
- [ ] Import/export remote configs
- [ ] SSH tunnel support

**Quota Groups:**
- [ ] Show models per quota group (expandable)
- [ ] Historical quota usage graphs
- [ ] Low-quota alerts/notifications

---

## 10. OpenCode Agent/Skill/Command System

### Agents (`.opencode/agent/`) — ~35 agent definitions
Many are presence/muse-themed agents that may not all have active runtime counterparts. Key unverified agents:
- `presence.web-graph-weaver.md`
- `presence.muse.*` (futures, alignment, aesthetic, emergence, memory, sophia, stability, trickster, archon, compression, chaos, symmetry, efficiency)
- `presence.health.*` (cpu, gpu0, gpu1, npu0, ram, disk)
- `presence.ethos.guardian.md`, `presence.pathos.field.md`
- `presence.dev.integrator.md`, `presence.ux.orchestrator.md`
- `presence.pm.operator.md`, `presence.test.verifier.md`

### Skills (`.opencode/skills/`) — ~80 skill definitions
Massive skill library including: semantic-linking, graph-storage, event-streaming, observability, telemetry (GPU/NPU/disk/SMART), trust-gradient, usability-review, cognitive-loop, risk-register, testing-verification, performance-budget, harm-analysis, boundary-enforcement, and many more.

### Commands (`.opencode/command/` + `.opencode/commands/`) — ~25 commands
Including: `sing.md`, `roll.md`, `route.md`, `scene.mage.md`, `lyrics.new.md`, `song.new.md`, `art.cover.md`, `fork-tax-commit.md`, `process-improvements.md`, `promptdb-compile.md`, `simulation.portal.md`, `constraint.add.md`, and more.

---

## 11. World Building — Narrative & Creative Backlog

### Narrative Chapters (`narrative/`)
49 chapters written (Chapter 01–49). Status: complete as written content.

### Songs (`world_building/songs/`)
~30+ song lyrics and Suno prompts. Mix of implemented audio and lyrics-only.

### Myths (`world_building/myth/`)
Myth engine specs, prototype plans, pantheon and gods document, trace mythology system.

### Novel Chapters (`world_building/color_of_consequence/`)
3 chapters of "The Color of Consequence" novel.

### World Bible & Characters
- `world_building/bible/World_Bible.md`
- `world_building/characters/Character_Profiles.md`

### Meta/Gaps Identified
- `world_building/meta/MISSING.md` — Chat transcripts/canvases never exported
- `world_building/meta/ETA_MU_GAP.md` — The perception↔action boundary problem
- `world_building/meta/PROMETHEAN_INTEROP.md` — Promethean system integration
- `world_building/meta/CEPHALON_INGESTION.md` — Cephalon ingestion pipeline
- `world_building/meta/PRNPIA_PLAYBOOK.md` — PRNPIA operational playbook

### World Building Analysis
~250+ analysis markdown files tracking audit steps, constraints, memory seeds, skill drills, plot beats, s5 resolution passes, deep research protocol versions, and more.

---

## 12. Notes & Research — Scattered Ideas

### Implementation Notes (`docs/notes/implementation/`)
- Phased execution plan with stop lines
- Agent prompt for Daimoi crawler muse
- Implementation brief: muse facts graph
- Named query spec and tool contract addendum
- Standardized roles and test matrix
- Daimoi crawler muse architecture note

### System Design Notes (`docs/notes/system_design/`)
- Hole responses field and collisions
- Hybrid field-graph formalism
- Sigil and Daimoi visual language

### Research Notes (`docs/notes/research/`)
- Nexus-Daimoi hybrid math spec
- Distributed ECS analogues survey
- Local embeddings benchmarking and MRL selection

### Creative Notes (`docs/notes/creative/`)
- Manifest oath lyrics draft
- Entropy choir protocol Suno lyrics
- Glitch choir style and dialog seed
- Manifest oath play script
- Suno prompts for manifest oath motifs
- Sing contract crystallize bloom upgrade

### Claims
- `docs/notes/claims/2026-02-26-falsifiable-system-claims.md` — 4+ falsifiable claims for the Nexus system (homeostasis, routing, constraint enforcement)

### Security Feature Extractor Notes (`docs/notes/security_feature_extractor/`)
- GitHub crawler security extraction spec
- Proximity signals for new entities
- HMM temporal stabilizer
- Latent cyber regime model
- Security extraction and ranking toolbox
- Deterministic security classifier and label sources

### Unsorted Root Notes
- `2026.02.27.11.52.57.md`, `2026.02.27.21.29.38.md`
- `2026.03.01.*` (multiple timestamped notes)
- `2026.03.05.13.10.01.md`

---

## 13. .ημ State & Known Issues

**Last Π snapshot:** 2026-03-06T06:43:45Z on branch `feature/eta-mu-tts-fix`  
**Head:** `921a946` (dirty)  
**Known issues from state:**
- HTTP `/api/simulation` remains slower/more fragile than websocket delta path
- NPU-bias tweak and frontend delta contract fix are included but may need validation

**Sub-directories:**
- `03_ARTIFACTS/` — Contains narrative audio archive
- `docs/` — Additional documentation
- `operation-mindfuck/` — Core operation-mindfuck artifacts
- `_rejected/` — Rejected artifacts

---

## 14. .η Observational Layer — Skeleton

Structure exists with `stream/`, `raw/`, `live/` directories but all are empty (gitkeep only). The protocol is defined but no observations have been recorded through it.

---

## 15. .sisyphus — Empty

Task tracking directory exists but is completely empty. No tasks tracked here.

---

## 16. .fork_Π_ημ_frags — Visual Archive

Contains ~300+ screenshots, storyboards, cover art, and ChatGPT-generated images spanning Aug 2025 – Feb 2026. Also contains multiple Π snapshot archives (zip files with SHA256 checksums) and a `.μη_ports/` directory. This is a visual history archive, not active task tracking.

---

## 17. Recent Receipts Summary

Last ~20 receipt entries (2026-03-05 to 2026-03-11) show active work on:
- Runtime module extraction/refactoring (simulation.py, server.py decomposition)
- PM2 transport config fixes for frontend WebSocket stability
- Frontend animation restoration
- TTS narrator output fix
- NPU device access and cosine sidecar configuration
- Study API stampede fix
- NPU visibility improvements
- Frontend simulation delta contract alignment
- OpenPlanner memory bridge integration
- Open-hax signal foundation package extraction
- Docker CPU budget clamping
- Workspace migration of eta-mu substrate
- Sidecar retune (GPU+NPU adaptive split)

---

## 18. Prioritized Dreams List

Based on all discovered scattered notes, specs, and ideas, here is a prioritized list of "dreams" — features the user has expressed wanting but hasn't fully built:

### Tier 1 — High Priority, Partially Started
1. **MCP Lith Nexus completion** — TypeScript skeleton exists, needs graph integration, write paths, and HTTP mount
2. **Security Tracking Pipeline completion** — 6 specs all in-progress, several open tasks remain
3. **C Simulation Performance Optimization** — Detailed spec with phases, no code changes yet
4. **NPU Benchmark Harness** — Complete spec, no harness binary yet
5. **Memory Churn GC Remediation** — 3 versions of spec, some addressed in receipts but formal completion unclear

### Tier 2 — High Priority, Not Started
6. **PromptDB Lisp Interpreter** — Replace regex-based compiler with proper parser/evaluator
7. **Dynamic ViewGraph Expansion** — On-demand cluster expansion/compaction in UI
8. **REINFORCE Learning for Presences** — Policy gradient with trace buffers
9. **Auditable Path Visualization** — Gravity heatmaps, receipt inspector, path overlays
10. **Wire World Part64 Intent** — PromptDB contracts into runtime, slash commands, presence API
11. **Presence WebRTC Communication** — Audio-first WebRTC presence calls

### Tier 3 — Medium Priority, Designed
12. **Eta-Mu Inbox File Graph** — `.ημ/` as active inbox with graph integration
13. **Eta Observational Layer** — `.η/` protocol with append-only semantics
14. **Truth Binding v1** — Truth as judged claims with named operators
15. **Inspiration Atlas + Field UI** — Dashboard panel reacting to live field dynamics
16. **FastAPI MVC Migration** — Decompose `world_web` into FastAPI patterns
17. **Frontend Testing Coverage CI** — Proper frontend test coverage gating
18. **Barnes-Hut Tree** — O(N log N) force approximation for particle sim

### Tier 4 — Research & Exploration
19. **Deep Research Series (12 specs)** — Complete research program from TruthGraph to smart cards
20. **Nexus-Daimoi Semantic Fields** — Mathematical alignment of semantic field dynamics
21. **House UI Projection** — Speculative UI concept
22. **Weaver-Graph Field Integration** — Weaver↔graph bidirectional integration
23. **Presence Impact Flow Ghost** — Ghost/impact flow modeling
24. **Simulation Smoothing** — Interpolation and smoothing techniques

### Tier 5 — Creative & Narrative Dreams
25. **Live Choir with real LLM providers** — Currently mock-only
26. **Vision + Embedding stack** — For ingest contract (local vision model + vectorstore)
27. **Chat transcript export** — Missing from world building archive
28. **Myth Engine implementation** — Clojure backend + web view prototype designed but not built
29. **Docker LLM Proxy TUI improvements** — ~15 UI/UX items listed
30. **Falsifiable claims test harness** — Test the 4+ system claims in simulation

---

*This inventory was compiled by scanning all source files, spec directories, notes, state files, receipts, and documentation across the fork_tales project.*
