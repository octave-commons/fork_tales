# Prompt Lisp Interpreter & PromptDB Compiler (v0.1)

## Priority
- High

## Complexity
- Complex (multi-file parser/evaluator/compiler/test changes)

## Requirements
- Deterministic s-expr parser with line/column errors.
- Two-tier model: data-preserving by default, template evaluation only under `(template ...)` and `(render ...)`.
- Template built-ins: `join`, `nl`, `if`, `=`, `present?`, `var`, `set`, `escape`, `include`.
- Safety constraints: include allowlist roots, include cycle/depth/file caps, output cap, forbidden forms explicit errors.
- Compiler emits OpenCode command/skill surfaces from enabled fragments in stable path order.
- Receipt-to-facts compiler for `(observation.v2 ...)` for downstream audit/routing.
- Append-only source policy: never delete/modify fragments or receipts during compile.

## Existing Context
- Current compiler is regex-based: `artifacts/scripts/promptdb_compile.mjs`.
- Existing minimal datalog + receipt compiler is Python: `artifacts/python/eta_mu_lisp_datalog.py`.
- Prompt fragments/rules live in `.opencode/promptdb/fragments` and `.opencode/promptdb/rules`.

## Risks
- Existing command/skill outputs use slightly inconsistent frontmatter conventions.
- Include safety can be bypassed if path normalization is incomplete.
- Rendering recursion and large payloads can break determinism/performance without caps.

## Open Questions
- None blocking. Defaulting to implementing core in Node scripts and preserving Python audit tooling.

## Phases

### Phase 1: Core Prompt Lisp runtime
- Add a dedicated runtime module (tokenizer, parser, printer, template evaluator, include guardrails, structured errors).
- Expose deterministic `render(expr, env)` with witness metadata.

### Phase 2: Compiler integration
- Refactor `artifacts/scripts/promptdb_compile.mjs` to use parsed fragments rather than regex extraction.
- Implement stable scan + enabled filtering + command/skill target emission.

### Phase 3: Receipt facts interface
- Add receipt compiler helpers to convert `observation.v2` into fact tuples for datalog rules.

### Phase 4: Tests and verification
- Golden tests: parse -> print -> parse equivalence.
- Render snapshots + determinism tests.
- Audit interface tests for receipt fact extraction.
- Run diagnostics/tests/build commands and fix failures.

## Candidate Files
- `artifacts/scripts/prompt_lisp.mjs` (new)
- `artifacts/scripts/promptdb_compile.mjs` (refactor)
- `artifacts/scripts/tests/prompt_lisp.test.mjs` (new)

## Existing Issues / PRs
- None discovered locally.

## Definition of Done
- Deterministic parser/evaluator/compiler implemented per spec v0.1.
- Command + skill outputs generated from parsed fragments.
- Error model includes file/line/column and explicit codes for forbidden/cycle/limits.
- Tests cover golden/determinism/audit conversion and pass.
