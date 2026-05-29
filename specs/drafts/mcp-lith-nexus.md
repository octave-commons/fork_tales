# MCP Lith Nexus

## Priority
- High

## Complexity
- Complex (new TypeScript package, parser/indexer/runtime, deterministic write path, MCP surface, tests)

## Requirements
- Build a Lith-first MCP server that serves repo files and graph resources with Lith payloads.
- Index configurable Lith/S-expr roots plus optional Markdown specs with embedded Lith code blocks.
- Parse top-level Lith forms into AST nodes with source spans, original text slices, canonical printed forms, and stable hashes.
- Build a Nexus graph with minimum node kinds `file`, `form`, `packet`, `contract`, `fact`, `spec`, `tag` and minimum edge kinds `contains`, `declares`, `references`, `tagged`, `depends_on`, `derived_from`.
- Expose MCP resources for raw repo files, graph index, graph nodes, graph edge neighborhoods, and PromptDB convenience URIs.
- Expose MCP tools `lith.find`, `lith.read`, `nexus.query`, `nexus.create_resource`, and `promptdb.create_fact`.
- Keep write paths deterministic, reject obvious secret material, and no-op on duplicate canonical content.
- Refresh the in-memory index after writes so new resources are immediately queryable.

## Existing Context
- Existing deterministic S-expression tooling lives in `artifacts/scripts/prompt_lisp.mjs`.
- PromptDB packets/contracts already live under `.opencode/promptdb/`.
- `part64/code/world_web/simulation_nexus.py` already defines the canonical `nexus_graph` contract.
- `part64/code/world_web/chamber.py` already exposes legacy PromptDB indexing that should stay backward compatible.
- Repo guidance prefers small, reviewable, deterministic changes and in-repo continuity.

## Risks
- MCP SDK package choice must avoid unstable integration drift while still matching current protocol semantics.
- Lith query/config examples use vectors (`[...]`), so parser support must go beyond plain parens.
- Resource URIs and idempotent write paths need stable normalization across mixed file kinds.
- Large repos can make naive full reindexing expensive if refresh logic is not scoped.

## Open Questions
- None blocking. Use the existing `part64` canonical graph contract as the source graph, extend it with Lith/PromptDB first-class nodes, and expose it through a stdio MCP server package under `mcp-lith-nexus/`.

## Phases

### Phase 1: Package + MCP skeleton
- Add a dedicated TypeScript package under `mcp-lith-nexus/`.
- Wire stdio server startup, config loading, `resources/list`, `resources/read`, and `lith.find`.

### Phase 2: Lith parser + indexed store
- Implement a parser/printer with source spans, canonical forms, hashes, and Markdown fenced-block extraction in Python so `part64` and MCP share the same extracted facts.
- Build a repo Lith index plus PromptDB-compatible legacy snapshot adapters.

### Phase 3: Nexus graph + query
- Merge Lith packets/contracts/facts/forms into `part64` logical graph assembly and let canonical `nexus_graph` carry them as first-class nodes.
- Add graph node/edge resources and a limited `(query ...)` evaluator on top of that canonical graph.

### Phase 4: Deterministic writes
- Implement `promptdb.create_fact` and `nexus.create_resource` with no-secret checks and no-op receipts.
- Incrementally refresh the graph after writes.

### Phase 5: Verification
- Add MCP integration tests, parser/index tests, and write-path idempotence tests.
- Mount the MCP HTTP surface into the `part64` runtime stack, proxy it through `part64/nginx/default.conf`, and expose a matching remote MCP entry in `opencode.jsonc`.
- Run package typecheck/tests and append a receipt entry.

## Candidate Files
- `part64/code/world_web/lith_nexus_index.py`
- `part64/code/world_web/lith_nexus_snapshot.py`
- `part64/code/world_web/lith_nexus_cli.py`
- `part64/code/tests/test_lith_nexus_index.py`
- `part64/Dockerfile.system`
- `part64/docker-compose.yml`
- `part64/docker-compose.muse-song-lab.yml`
- `part64/docker-compose.sim-slice-bench.yml`
- `part64/ecosystem.config.cjs`
- `part64/ecosystem.bench.config.cjs`
- `part64/nginx/default.conf`
- `mcp-lith-nexus/package.json`
- `mcp-lith-nexus/tsconfig.json`
- `mcp-lith-nexus/src/index.ts`
- `mcp-lith-nexus/src/http.ts`
- `mcp-lith-nexus/src/runtime.ts`
- `mcp-lith-nexus/src/server.ts`
- `mcp-lith-nexus/src/service.ts`
- `mcp-lith-nexus/src/backend.ts`
- `mcp-lith-nexus/src/config.ts`
- `mcp-lith-nexus/src/lith.ts`
- `mcp-lith-nexus/src/query.ts`
- `mcp-lith-nexus/src/write.ts`
- `mcp-lith-nexus/src/service.test.ts`
- `mcp-lith-nexus/src/server.test.ts`
- `mcp.lith-nexus.config.lith`
- `opencode.jsonc`

## Existing Issues / PRs
- None discovered locally for this package.

## Definition of Done
- MCP clients can list/read Lith-first resources and discover all required tools/prompts.
- `lith.find` and `nexus.query` return deterministic Lith payloads over indexed repo data.
- `promptdb.create_fact` and `nexus.create_resource` write deterministic in-repo files, reject obvious secrets, and no-op on duplicates.
- New writes are queryable without restarting the process.
- Targeted typecheck and tests pass.
