# AGENTS.md

Practical guidance for coding agents working in this repository.

## 1) Purpose and priorities

- Keep the runtime healthy and verifiable.
- Prefer deterministic changes over stylistic churn.
- Preserve existing project conventions.
- Make small, reviewable edits unless user asks for broad refactors.

Priority order when making decisions:

1. Correctness and safety
2. Passing tests and build
3. Compatibility with existing architecture
4. Clarity and maintainability
5. Speed and convenience

## 2) Repository map

- `part64/frontend`: React + TypeScript + Vite UI.
- `part64/code`: Python runtime and APIs; Node web graph weaver script.
- `receipts.log`: append-only execution/change receipts.
- `contracts/`: repository contract scripts.
- Root `package.json`: top-level utility scripts.

## 3) Environment assumptions

- Platform: Linux/macOS shell compatible commands.
- Python module execution is used heavily (`python -m ...`).
- Node is required for frontend and some tooling.

## 4) Core commands

Run from repository root unless noted.

### Root scripts

- `npm run live`
- `npm run fork-tax:audit`
- `npm run fork-tax:cycle`
- `npm run fork-tax:test`

### Frontend (`part64/frontend`)

- Install: `npm install`
- Dev server: `npm run dev`
- Build: `npm run build`
- Lint: `npm run lint`
- Preview built app: `npm run preview`

### Python runtime (`part64`)

- Start world server: `python -m code.world_pm2 start --host 127.0.0.1 --port 8787`
- PM2 status: `python -m code.world_pm2 status`
- PM2 restart: `python -m code.world_pm2 restart`
- PM2 stop: `python -m code.world_pm2 stop`
- Dashboard server: `python -m code.world_web --part-root ./ --vault-root .. --host 127.0.0.1 --port 8791`

### Docker runtime (optional)

- `docker compose up --build`

## 5) Test commands

### Full or suite-level

- PM2/browser integration suite: `python -m code.tests.test_world_web_pm2`
- World life suite: `python -m code.tests.test_world_life`
- Myth bridge suite: `python -m code.tests.test_myth_bridge`
- Sonify determinism suite: `python -m code.tests.test_sonify_determinism`

### Single test file

- `python -m pytest part64/code/tests/test_world_web_pm2.py -q`
- `python -m pytest part64/code/tests/test_world_life.py -q`

### Single test function

- `python -m pytest part64/code/tests/test_world_web_pm2.py::test_world_payload_and_artifact_resolution -q`
- `python -m pytest part64/code/tests/test_world_web_pm2.py::test_pm2_parse_args_defaults -q`

If `pytest` is unavailable in the environment, use the module-level test commands above.

## 6) Runtime verification (required for runtime-affecting changes)

Verify these endpoints after relevant changes:

- `http://127.0.0.1:8787/` returns HTTP 200
- `http://127.0.0.1:8787/api/catalog` returns HTTP 200
- `ws://127.0.0.1:8787/ws` accepts websocket connection

If you changed dashboard runtime (`world_web`), also verify:

- `http://127.0.0.1:8791/`
- `http://127.0.0.1:8791/api/catalog`
- `ws://127.0.0.1:8791/ws`

## 7) Frontend code style

Source of truth: `part64/frontend/eslint.config.js` and TypeScript configs.

- Language: TypeScript (`strict` mode enabled).
- Keep `noUnusedLocals` and `noUnusedParameters` clean.
- Use ES module imports; avoid CommonJS in frontend.
- Follow existing naming conventions:
  - Components/types: `PascalCase`
  - Variables/functions: `camelCase`
  - Constants: `UPPER_SNAKE_CASE` for true constants only
- Prefer small pure helpers over monolithic components.
- Co-locate types where practical; share via `src/types` when reused.
- Avoid adding dependencies unless justified by repeated complexity.

## 8) Python code style

Observed project conventions in `part64/code`:

- Use type hints pervasively (`dict[str, Any]`, etc.).
- Keep module-level constants uppercase.
- Prefer deterministic helper functions for transforms.
- Keep error handling explicit; return structured payloads for API responses.
- Avoid hidden side effects in utility functions.
- Maintain compatibility with current import patterns (`code.<module>` with local fallback where used).

## 9) Editing and change discipline

- Make focused changes directly related to the request.
- Do not rewrite unrelated files for style alone.
- Keep files ASCII unless file already relies on Unicode content.
- Add comments only for non-obvious logic.
- Preserve append-only artifacts and logs where expected.

## 10) Receipts and traceability

- If you modify runtime behavior, tests, or contract-sensitive paths, append a receipt to `receipts.log`.
- Receipt entries should include:
  - timestamp
  - change kind
  - owner/origin
  - definition of done reference
  - file refs and verification refs
- Do not delete or rewrite historical receipt entries.

## 11) Git workflow expectations

- Inspect current tree state before edits.
- Never revert user changes unless explicitly instructed.
- Avoid destructive git commands (`reset --hard`, force pushes) unless explicitly requested.
- Commit only when user asks for a commit.
- Prefer concise commit messages that explain why the change exists.

## 12) Agent behavior expectations

- Start with quick repository investigation before coding.
- Propose a short phased plan for non-trivial work.
- Run relevant tests/builds after each significant phase.
- Report exactly what changed, what was verified, and what remains.
- If blocked, ask one targeted question with a recommended default.

## 13) Cursor/Copilot rule integration

Current status in this repository:

- No `.cursorrules` file found.
- No `.cursor/rules/` directory found.
- No `.github/copilot-instructions.md` found.

If these files are added later:

- Treat them as additive constraints.
- Merge their instructions with this file.
- In conflicts, prefer the most specific rule for the touched path/tool.

## 14) Definition of done checklist

Before declaring completion:

1. Relevant code paths updated and consistent.
2. Build/lint/tests run for impacted surfaces.
3. Runtime endpoints verified when runtime was changed.
4. `receipts.log` updated when required.
5. Final report includes changed files and verification commands.

Keep this document practical. Update it whenever commands, test entrypoints,
or repository conventions change.
