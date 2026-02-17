---
description: Run fork-tax git audit/cycle protocol with presence guidance
agent: plan
---
Run audit first:
- `node contracts/contract_fork_tax_git_v1.mjs --audit`

When a checkpoint is required, run cycle:
- `node contracts/contract_fork_tax_git_v1.mjs --cycle --message "$ARGUMENTS" --owner Err --dod "checkpoint committed and push-ready"`

To include push:
- `node contracts/contract_fork_tax_git_v1.mjs --cycle --push --message "$ARGUMENTS" --owner Err --dod "checkpoint committed and push-ready"`
