# Security Tracking Implementation Order

## Recommended Sequence

1. **Spec 01**: `01-github-security-extraction-foundation.md`
   - Establish deterministic atoms, receipts, and named query contracts.
2. **Spec 02**: `02-deterministic-security-ranker.md`
   - Replace weak keyword-only scoring with deterministic classifier scoring.
3. **Spec 03**: `03-proximity-feature-engine.md`
   - Improve unseen-term/entity handling across semantic/graph/time features.
4. **Spec 04**: `04-hmm-entity-state-smoothing.md`
   - Stabilize alert state transitions and reduce flapping.
5. **Spec 05**: `05-security-pipeline-hardening.md`
   - Harden dedupe/corroboration/throughput under sustained runtime load.
6. **Spec 06**: `06-latent-cyber-regime-context.md`
   - Add macro regime context after core quality is stable.

## Current Progress (2026-03-04)

1. **Spec 01**: in progress (core implemented; provisional filtering + seed-only alerts added)
2. **Spec 02**: in progress (classifier active; weak-label score gating for local/cyber added)
3. **Spec 03**: in progress (proximity wired, needs tuning)
4. **Spec 04**: in progress (state logic present, not fully policy-driving)
5. **Spec 05**: in progress (max-node cap trap mitigated; runtime throughput still needs verification)
6. **Spec 06**: in progress (regime path active; empty-result fallback now wired)

## Reality-Aligned Task List

- [x] P0: Recover a key discovery blocker by preventing max-node cap traps on restore/start
      (added dynamic headroom above restored URL count + raised hard node cap).
- [x] P0: Ensure global radar default path can exclude provisional watchlist seed rows in
      runtime API responses (fixed boolean handling for `include_provisional=false`).
- [x] P0: Add an operations check for repeated seed-only global output
      (runtime streak + alert state and report quality fields).
- [x] P0: Verify end-to-end live discovery recovery in runtime after weaver restart
      (forced-cap runtime probe observed `crawl_state.reason=max_nodes_autogrow`
      with cap growth from 5390 to 7438 while frontier remained non-empty).
- [x] P0: Ensure crawler runtime can operate passively without manual UI start
      by auto-starting crawl on boot from watchlist + graph seed context.
- [x] P0: Raise crawler runtime floor/default limits to large values and verify they
      persist across runtime restarts (depth/nodes/concurrency/host/entity caps).
- [x] P0: Split threat radar ownership into independent muse lanes so global and local
      radars are anchored/operated as separate presences (chaos=global,
      witness_thread=local).
- [ ] P0: Confirm non-provisional global evidence returns in global radar output
      under live runtime crawl conditions using model-flagged downstream pages only
      (raw feed/watchlist source URL evidence is now suppressed by default).
- [x] P1: Recalibrate local/cyber thresholds so non-noise security rows can surface
      without promoting generic development PR noise (added weak-label min-score filter
      and cyber threshold fallback when strict regime threshold returns empty).
- [ ] P1: Continue tuning weak-label/threshold calibration against live data to improve
      precision while keeping cyber surface non-empty.
- [ ] P1: Stabilize or disable LLM blend path until `llm_invalid_json` failure mode is
      eliminated.
- [ ] P1: Add a thin status dashboard/report for dedupe, corroboration, and regime
      threshold impacts so tuning decisions are evidence-backed.

## Why This Order Still Holds

- Early phases maximize immediate quality lift with low architectural risk.
- Later phases depend on stable atoms/features and sufficient non-provisional signal.
- Regime modeling is most useful after extraction/ranking quality and runtime continuity
  are both stable.
