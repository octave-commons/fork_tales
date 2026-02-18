---
name: intent-protocol
description: Make intent explicit + witnessable
compatibility: opencode
---

## Intent Protocol (ημ)

### Label speech acts (required)
- **want**: desired outcome
- **know**: verified facts
- **guess**: assumptions / uncertainty
- **do**: intended action
- **verify**: how you will check
- **witness**: evidence artifacts (hashes, traces, bboxes, receipts)

### Bans
- No mind-reading: avoid 'you want/you meant' without evidence.

### Output discipline
- If a tool was not run, never claim it was.
- Separate facts vs assumptions vs unknowns.

### Receipt pattern
Emit immutable receipts (observation/witness) and link claims to evidence refs.
