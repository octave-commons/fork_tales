# Fork Tax Git Cadence Protocol

## Priority
- High

## Complexity
- Moderate (contract + executable audit/cycle workflow)

## Requirements
- Create a standard, presence-aligned protocol that keeps commit and push cadence explicit.
- Provide a deterministic git audit that reports drift in terms of facts, asks, and repairs.
- Provide a safe checkpoint cycle that can stage tracked changes, commit, and push when upstream is configured.
- Emit append-only receipt evidence for fork-tax checkpoint actions.
- Keep protocol additive and avoid destructive git operations.

## Open Questions
- None blocking. Defaults:
  - Commit cadence threshold: 120 minutes since last commit.
  - Push cadence threshold: 5 commits ahead of upstream.
  - Untracked warning threshold: 40 items.
  - Cycle stages tracked changes only unless `--include-untracked` is provided.

## Risks
- Auto-cycling can stage unintended changes if users do not review status first.
- Push operations fail when branch has no upstream or no remote configured.
- Receipt rows can drift from canonical formatting if fields include unescaped separators.

## Existing Issues / PRs
- Issues: none discovered from current local git context.
- PRs: none discovered from current local git context.

## Files Planned
- `contracts/contract_fork_tax_git_v1.mjs`
- `.opencode/command/promethean.fork-tax-git.v1.md`
- `.opencode/commands/fork-tax-commit.md`
- `package.json`

## Definition of Done
- Git audit command returns machine-readable status and presence-shaped repair guidance.
- Cycle command can commit checkpoints and optionally push with explicit gate failures.
- Receipt append helper writes one canonical decision line for successful checkpoint runs.
- Fork-tax protocol contract exists under `.opencode/command` and maps responsibilities to presences.
- Basic script self-tests pass.

## Session Change Log
- Initialized protocol draft and implementation plan for commit/push fork-tax cadence.
