---
name: gates-runtime-check
description: Run canonical runtime checks for Gates of Truth surfaces and report evidence compactly.
metadata:
  owner: project
  version: 1
---

# Gates Runtime Check

Use this skill when validating canonical panel runtime status.

## Required Checks

Run and record outcomes for:

- `http://127.0.0.1:8787/`
- `http://127.0.0.1:8787/api/catalog`
- `ws://127.0.0.1:8787/ws`

## Procedure

1. Confirm process is reachable and serving root panel.
2. Confirm catalog endpoint returns valid payload.
3. Confirm websocket endpoint accepts connection and pushes updates.
4. Report facts, then interpretation, then risks.

## Guardrails

- Never claim pass/fail without command-backed evidence.
- If one check fails, include likely blast radius and containment step.
