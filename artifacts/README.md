# PromptDB Router (OpenCode plugin)

This bundle turns your Lisp-style prompt fragments into an **append-only callable database** under `.opencode/promptdb/`,
and exposes a deterministic routing tool (`router.decide`) as an OpenCode plugin.

## What you get

- `.opencode/plugins/promptdb-router.ts`
  - Custom tools:
    - `router.decide` — deterministic skill/tool/command recommendations
    - `promptdb.reload` — reload fragments into the in-memory index
    - `promptdb.list` — show loaded fragments + provides
  - Policy hook:
    - `tool.execute.before` — blocks obvious destructive shell usage (starter example)
  - Compaction hook:
    - `experimental.session.compacting` — injects PromptDB kernel notes

- `.opencode/promptdb/fragments/*.oprompt`
  - Example fragments (commands, skills, rules, and the `/sing` prompt packet)

- `scripts/`
- `promptdb_compile.mjs` — compiles fragments into `.opencode/commands/*.md` + `.opencode/skills/*/SKILL.md`
- `opencode_register.mjs` — content-address register helper (optional)
- `eta_mu_ledger.mjs` — tiny η/μ ledger CLI; emits one JSONL record per utterance

- `python/eta_mu_lisp_datalog.py`
  - A tiny s-expr + stratified datalog engine + `(observation.v2 ...)` compiler (offline audit / CI)

## Quick start

1) Drop the contents of this zip into your repo root.

2) Run OpenCode once (it will install `.opencode/package.json` deps via Bun automatically).

3) Compile fragments into standard OpenCode commands/skills:

```bash
node scripts/promptdb_compile.mjs
```

4) In OpenCode, try:

- `/route vision observe the omni panel overlay`
- `/promptdb-compile`

Or call the tool directly from an agent:

- `router.decide({ prompt: "..." })`

## Notes

- The plugin intentionally keeps parsing minimal and deterministic.
- Files are treated as the ledger; disable-only is the policy (never delete history).
- For stronger safety: remove "default evidence" in the python rules and require explicit evidence-refs.
