# Constraints Ledger (append-only)

> Never delete constraints. If a constraint must stop applying, append a DISABLE record.

## Active Constraints

- C-61-π: Π artifacts are DERIVED and must name canonical parents + sha256
- C-61-audio: audio recipes must declare tempo/seed/signal-chain/stutter/render rate
- C-61-tests: every generator ships with a deterministic regression test
- C-61-ports: declare drift tolerance when cross-platform determinism is hard

- C-62-ledger-lines: human + JSON manifest required
- C-62-seed-derivation: no hidden RNG
- C-62-art-naming: bpm/part in filenames
- C-62-disable-protocol: disable-by-append only

- C-63-manifest-required: every artifact listed in manifest.json w/ sha256 + role
- C-63-audio-markers: named marker timestamps: anchor, tax
- C-63-soundchain: explicit synthesis graph receipt

- C-64-world-snapshot: every part adds exactly one "world map" note (what changed, where)
- C-64-audio-canonical: WAV is canonical; MP3 is convenience (encoder drift allowed)
- C-64-marker-schema: markers must include name, t_seconds, and purpose
- C-64-creative-commons: all artifacts in this bundle are CC BY-SA (share-alike)
- C-64-web-daemon: world server must expose / and /healthz for browser and process checks
- C-64-pm2-runtime: pm2 ecosystem config is the canonical daemon launch recipe
- C-64-web-tests: world daemon behavior must have deterministic regression tests
- C-64-websocket-feed: dashboard must provide realtime catalog updates over websocket
- C-64-mix-stream: dashboard must expose one combined WAV stream for all discoverable WAV assets
- C-64-webgl-view: dashboard must render websocket simulation frames via WebGL canvas

- C-66-glitch-overlay: Named forms must have distinct visual behaviors in the overlay
- C-66-bilingual-resonance: All generated audio/text must support EN/JA duality
- C-68-named-field-gradients: canonical named fields must expose gradient metadata for overlay rendering
- C-68-overlay-catalog: /api/catalog must include named_fields for deterministic UI hydration
- C-68-presence-impact-telemetry: simulation must expose append-only presence impact telemetry from file drift + witness touch, including fork-tax state and bilingual canonical labels
- C-68-fork-tax-payment-path: runtime must expose a reversible payment path for fork-tax balance and preserve payment events in decision_ledger.jsonl
- C-68-witness-thread-lineage: simulation must expose witness thread continuity lineage (what changed, where, why) for UI traceability
- C-68-file-influence-beacons: simulation overlay must render explicit file-influence beacons and linkage paths so operators can point to concrete file pressure in-field
- C-69-web-graph-weaver: crawler instrumentation must expose transparent discover/skip/fetch/compliance events over websocket and REST status surfaces
- C-69-ethical-crawl-guardrails: crawler traversal must respect robots.txt, crawl-delay, and nofollow while preserving conservative rate limits and explicit opt-out
- C-69-presence-protocol-v1: presence agents must conform to .opencode/protocol/presence.lisp identity/mission/obligation/interface/doctor shape
- C-69-health-presence-coverage: health telemetry presences must cover cpu, ram, disk, gpu0, gpu1, and npu0 with required telemetry skill bindings
- C-69-role-presence-triad: UX orchestrator, dev integrator, and PM operator presences must exist with explicit skill bindings and lisp instantiation blocks
- C-69-permission-field-append-only: permission law must be event-sourced append-only in .opencode/perm/log.lisp and projected by fold rules
- C-69-permission-check-gate: external IO actions must pass permission/check with scoped reasoned decisions, defaulting to deny when uncertain
- C-69-tester-presence-verification: tester verifier presence must enforce invariants, contract conformance, replay determinism, and failure injection coverage
- C-69-ethos-advisory-boundary: ethos guardian remains advisory/event-emitting and cannot execute IO or override permissions
- C-69-pathos-observational-only: pathos field can emit salience/tension/trust signals but cannot execute actions
- C-69-muse-hypothesis-tagging: muse outputs must be hypothesis-level with low confidence and no execution authority
- C-69-persona-switch-lens: persona switching may change deliberation weighting and tone but must not mutate permissions/logs or truth state
- C-69-weaver-availability-bootstrap: runtime should bootstrap local Web Graph Weaver when available and UI must provide endpoint fallback plus actionable offline guidance

## Disable Records

(none)
