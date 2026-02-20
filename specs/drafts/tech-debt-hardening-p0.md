# Tech Debt Hardening P0: Cache Integrity and Lint Hygiene

## Priority
- Critical

## Requirements
- Preserve structured runtime cache defaults in `world_web/constants.py`.
- Remove duplicate lock/cache reinitialization that overwrites populated defaults.
- Eliminate `KeyError` risk in `build_mix_stream` when cache shape drifts.
- Restore deterministic backend behavior for catalog + mix generation paths.
- Reduce false-negative frontend strict lint failures from generated coverage artifacts.

## Open Questions
- None blocking for P0.

## Risks
- `constants.py` is heavily imported; cache shape changes can surface latent assumptions.
- Existing dirty working tree increases merge conflict risk for large-file edits.
- Runtime-affecting change can regress websocket bootstrap paths if cache semantics shift.

## Complexity Estimate
- Medium

## Existing Issues / PRs
- No linked issue/PR metadata found in local repo for this exact fix pass.

## Phases

### Phase 1: Baseline and spec capture
- Save this remediation spec.
- Reproduce failure mode (`KeyError: "fingerprint"`) from `build_mix_stream`.

### Phase 2: Cache integrity hardening
- Remove duplicate tail reinitialization block in `constants.py` that resets caches to `{}`.
- Keep required cache symbols with typed/structured defaults.
- Harden `build_mix_stream` cache reads to tolerate future shape drift.

### Phase 3: Verification
- Run targeted backend test that exercises mix stream path.
- Run at least one additional focused runtime test for nearby behavior.
- Run frontend lint in strict mode after ignore-path adjustment (if touched this pass).

## Definition of Done
- `build_mix_stream` no longer throws `KeyError` in a fresh process.
- `test_catalog_library_and_dashboard_render` passes.
- Cache defaults in `constants.py` are not silently overwritten at import time.
- Any lint/config adjustments are verified with command output.

## Candidate Files
- `specs/drafts/tech-debt-hardening-p0.md`
- `part64/code/world_web/constants.py`
- `part64/code/world_web/simulation.py`
- `part64/code/tests/test_world_web_pm2.py`
- `part64/frontend/eslint.config.js`
- `receipts.log`

## Session Change Log
- Saved P0 draft spec and reproduced `build_mix_stream` `KeyError("fingerprint")` from an empty cache-shape path.
- Removed duplicate trailing cache/lock reinitialization block in `constants.py` and kept typed defaults.
- Hardened `build_mix_stream` cache access to use safe lookups and defensive meta copying.
- Added `test_cache_integrity.py` regression tests for cache-shape safety.
- Updated frontend eslint ignores to exclude generated `coverage/` artifacts from strict lint input.
