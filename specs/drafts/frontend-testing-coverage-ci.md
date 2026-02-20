# Frontend Testing, Coverage, and CI Gating

## Priority
- High

## Requirements
- Add a frontend coverage report command and produce a current baseline report.
- Add component tests for `ChatPanel` workspace sync behavior and message routing.
- Add panel integration tests for muse workspace bindings.
- Add CI gating so `npm test` runs on every pull request.

## Open Questions
- None.

## Risks
- `ChatPanel` has runtime fetch side effects and intervals that can introduce nondeterminism in tests.
- Large prop/type surfaces may make tests brittle without focused fixtures.
- Coverage run across all frontend files will likely show low baseline percentages due to broad existing surface.

## Complexity Estimate
- Medium-High

## Phases
1. Test infrastructure updates
   - Add coverage tooling/scripts and verify command behavior.
2. Component test implementation
   - Add deterministic `ChatPanel` tests for workspace sync and routing.
3. Integration test implementation
   - Add `MusePresencePanel` integration tests for binding propagation.
4. CI gating
   - Add GitHub Actions workflow to run frontend tests on PRs.
5. Verification and reporting
   - Run tests, coverage, and build; report baseline coverage and changed files.

## Definition of Done
- Frontend unit/integration tests exist for requested scenarios and pass locally.
- Coverage report command runs successfully and baseline numbers are captured.
- CI workflow is present and runs `npm test` on pull requests.
- Build still passes after all test/CI changes.

## Candidate Files
- `part64/frontend/package.json`
- `part64/frontend/package-lock.json`
- `part64/frontend/src/components/Panels/Chat.test.tsx`
- `part64/frontend/src/components/Panels/MusePresencePanel.test.tsx`
- `.github/workflows/frontend-tests.yml`
- `receipts.log`
