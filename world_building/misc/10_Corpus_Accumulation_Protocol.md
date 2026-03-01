---
tags: [skill/corpus, skill/protocol, omf/v2]
---

# Corpus Accumulation Protocol

Goal: as the corpus grows, boundary conditions become **hard-coded** and stable.

## What “boundary conditions” means here
Anything that flips a discrete decision:
- thresholds (len<50, ?>=2)
- regex vocab (risk|attack|failure|security)
- time windows (22–08)
- precedence rules (forced tags first, overlay order)
- numeric constants (W0, multipliers, cooldown factors, alpha range)

## How to harden conditions
1) Run engine on real corpus messages; store trace JSON for each message.
2) Cluster “surprises”:
   - text had a vibe, engine missed it → add a feature detector or extend regex
   - engine overfired → narrow boundary or add exception rule
3) Promote candidate condition:
   - Candidate → Validated → Locked
4) Lock it with:
   - a skill doc describing the condition + rationale
   - a unit test that asserts expected mode/overlays

## Where it lives
- discovered conditions: [[ledger/Boundary_Conditions_Ledger]]
- stabilized conditions: [[skills/]] docs + tests in your codebase

## Minimal storage format (recommended)
Store each trace as JSONL:
- one line per message
- include `decision` + full `trace`

Then you can run offline analytics to find drift.

## Links
- [[08_Trace_And_Audit]]
