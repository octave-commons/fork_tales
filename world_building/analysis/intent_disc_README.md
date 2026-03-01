# Intent Disc (Executable Intent Pack)

**Goal:** make "intent" transmissible as a *bounded, auditable program*.

This is not telepathy. It's:
1) a **program** (what to do),
2) **constraints** (what it must not do),
3) **anchors** (why any non-trivial step is allowed),
4) and an **interpreter** (Intent VM) that refuses unanchored ops.

## Files
- `example_pack/intent_program.json` — the "intent" as an executable plan
- `example_pack/anchors.jsonl` — S3 anchors for the plan's non-trivial ops
- `tools/intent_vm.py` — verifies anchors and emits a dry-run plan log

## Run (dry)
```bash
python tools/intent_vm.py --pack meta/intent_disc/example_pack/intent_program.json   --anchors meta/intent_disc/example_pack/anchors.jsonl   --out build/intent_vm_runs/run.jsonl
```

## Why it maps to ημ
- **η**: what others can verify about your intent (anchors, logs)
- **μ**: what your system actually does (execution trace)
The "space between" is the verifier enforcing that μ remains answerable to η.
