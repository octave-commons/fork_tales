# Spec — GitHub Crawler for `fork_tales/part64`

Created with the assistance of an AI.

## Signal

Add a **GitHub crawler Presence** to the Part64 runtime that ingests **public GitHub change signals** (PRs, issues, releases, advisories, diffs/lockfiles) into the **one true graph** using the existing **external reality contract** and **anti-spam homeostasis** constraints.

This crawler is **not** for attribution or “who is behind it.” It is an **observability organ**: surface reachability, dependency changes, and security-relevant deltas with receipts.

---

## Goals

1. **External reality feed (GitHub):** continuously but safely ingest GitHub repo activity into the canonical graph.
2. **Receipts-first:** every ingested claim must be traceable to a fetched URL/API response via hashes.
3. **Anti-spam + governor integration:** respect tick governor slack, per-repo rate limits, and per-URL cooldown.
4. **Deterministic queryability:** expose results through named queries and `/facts` snapshots so Muse can remain grounded.
5. **Security posture support:** make it easy to answer:

   * “Did a dependency/lockfile change?”
   * “Is a CVE/keyword being discussed?”
   * “Which PRs are ‘important’ by objective heuristics?”
   * “What code paths mention <library> / <parser> / <token>?” (bounded)

## Non-goals

* No attribution. No “state actor” inference.
* No full repo mirroring.
* No unbounded crawling.
* No secret scraping beyond what the configured GitHub token permits.

---

## Architecture Overview

### Components

* **`github_presence` (new):** a Presence loop that schedules GitHub fetch jobs under the tick governor.
* **`crawler.schedule_fetch` (existing/target):** single scheduling gate; extended to support GitHub sources.
* **`web_graph_weaver` (existing optional):** may remain for general web crawling; GitHub crawler can be separate or layered.
* **`facts_snapshot` + `graph_query` (existing/target):** expose deterministic state.

### Data flow (Reality Contract)

1. **Encounter / schedule**

* GitHub presence chooses a target URL/API endpoint (seeded list + frontier policy).
* Calls `crawler.schedule_fetch(...)` with source=`github`.

2. **Observation**

* Fetch result becomes an **Observation** event (append-only):

  * `github_fetch_completed` / `github_fetch_failed`
  * plus optional derived observation `github_extract_atoms`.

3. **Facts (optional v1):**

* Minimal v1 can store only observations + structured extraction.
* If logic gate exists, it can promote atoms into facts later.

4. **Graph commit**

* Create/merge **`web:url`** nodes and **`web:resource`** nodes.
* Add edges:

  * `web:source_of` (resource -> url)
  * `web:links_to` (resource -> url) for discovered URLs
  * `crawl:triggered_fetch` and `crawl:cooldown_block`

> **Constraint:** keep **exactly** the existing web node roles (`web:url`, `web:resource`) unless there is a strong need for GitHub-specific node roles. Prefer to encode GitHub semantics as **resource “kind” metadata**.

---

## Canonical Graph Representation

### Node roles (no new roles in v1)

* `web:url`
* `web:resource`

### Required properties

#### `web:url`

* `canonical_url: string`
* `next_allowed_fetch_ts: float`
* `last_fetch_ts: float` (optional)
* `fail_count: int`
* `last_status: string` (ok|error|http_*)
* `source_hint: "github"|"weaver"|...` (optional)

#### `web:resource`

* `canonical_url: string`
* `fetched_ts: float`
* `content_hash: string` (sha256)
* `text_excerpt_hash: string` (sha256; optional)
* `title: string` (optional)
* `kind: string` (recommended; see below)
* `repo: string` (optional; `owner/name`)
* `number: int` (optional; PR/issue number)
* `labels: [string]` (optional)
* `authors: [string]` (optional)
* `updated_at: string` (optional)

### Resource kinds

A `web:resource.kind` tag provides typed meaning without new node roles:

* `github:repo`
* `github:pr`
* `github:issue`
* `github:release`
* `github:advisory`
* `github:compare`
* `github:diff`
* `github:file` (raw file content)

---

## Fetch Strategy

### Modes

#### Mode A — GitHub REST API (preferred)

* Pros: structured, consistent, easy extraction.
* Cons: requires token for higher rate limits.

#### Mode B — HTML/Raw URL scraping (fallback)

* Pros: works without token.
* Cons: brittle; more parsing effort.

Spec requirement:

* Support Mode A; optionally support Mode B for resilience.

### Canonical URL forms

Normalize to stable URLs:

* Repo: `https://github.com/{owner}/{repo}`
* PR list: `https://github.com/{owner}/{repo}/pulls`
* Issue list: `https://github.com/{owner}/{repo}/issues`
* PR: `https://github.com/{owner}/{repo}/pull/{n}`
* Issue: `https://github.com/{owner}/{repo}/issues/{n}`
* Release: `https://github.com/{owner}/{repo}/releases/tag/{tag}`
* Compare: `https://github.com/{owner}/{repo}/compare/{base}...{head}`
* Raw file: `https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}`

If using API:

* Resource still references canonical HTML URL in `canonical_url`, but store `api_endpoint` in resource metadata if desired.

---

## Frontier Policy (bounded, deterministic)

### Seed configuration

A small config file defines monitored repos and keywords.

**`part64/world_state/config/github_seeds.json`** (recommended):

* `repos: ["owner/repo", ...]`
* `keywords: ["xml", "xxe", "cve-", "oauth", "token", ...]`
* `file_patterns: ["package.json", "pnpm-lock.yaml", "requirements.txt", ...]`
* `max_repos: int`
* `max_items_per_repo: int` (per sweep)

### Priority scoring (importance)

Compute `importance_score` per candidate item (PR/issue/release):

* +3: touches lockfiles/manifests
* +3: contains security keywords (CVE, XXE, token)
* +2: changes auth/credentials paths (heuristic by file path)
* +2: high comment velocity / reactions (if available)
* +1: merged/closed recently
* +1: labeled `security`, `bug`, `hotfix` (if available)

**Stop rule:** only take top-K per repo per sweep.

### Scheduling cadence

* Use tick governor slack:

  * if `slack_ms < 0`: do nothing (defer)
  * else: schedule up to `MAX_GITHUB_FETCHES_PER_TICK` (default 1–2)
* Global concurrency `MAX_CONCURRENT_GITHUB_FETCHES` (default 2–4)
* Per-repo cooldown `GITHUB_REPO_COOLDOWN_S` (default 300s)
* Per-URL cooldown via existing `web:url.next_allowed_fetch_ts`.

---

## Extraction (Observation → Atoms)

When a GitHub page/API response is fetched, extract **candidate atoms** (observations) as a bounded list.

### Atom schema (v1)

Store atoms as part of the `web:resource` metadata and/or emit `github_extract_atoms` event.

Atoms (examples):

* `mentions(repo, term)`
* `mentions_file(repo, path)`
* `changes_dependency(repo, dep_name, from_ver?, to_ver?)`
* `references_cve(repo, cve_id)`
* `touches_path(repo, path_prefix)`
* `pr_state(repo, pr_number, state)`
* `pr_merged(repo, pr_number, merged_at)`

**Bounded:** max 50 atoms per resource.

### Minimal parsing requirements

* For PR/issue: title, body excerpt, labels, author, updated_at.
* For PR diff/compare: filenames touched (top-K 200), and grep keywords in diff (top-K matches).
* For raw file content: compute content_hash + optional lightweight parsing:

  * package manifests (name/version)
  * lockfile diff (dependency version change detection if feasible)

---

## Events (append-only receipts)

Emit JSONL events (stable order, bounded payloads):

* `github_fetch_scheduled`
* `github_fetch_started`
* `github_fetch_completed` (include `url_id`, `res_id`, `content_hash`)
* `github_fetch_failed` (status, backoff, next_allowed_fetch_ts)
* `github_extract_atoms` (counts + top atoms)

> Use the existing event logging conventions from the web/crawler contracts.

---

## Security & Safety Requirements

1. **No credential leakage:**

   * Never log tokens.
   * Redact `Authorization` headers from any debug output.

2. **Token scope minimization:**

   * If GitHub token is used, require read-only scopes.

3. **Robots and ToS constraints:**

   * Prefer GitHub API usage.
   * Respect rate limits.

4. **Outbound network containment:**

   * Domain allowlist default: `github.com`, `api.github.com`, `raw.githubusercontent.com`.

5. **Deterministic bounded behavior:**

   * Stable sorting.
   * Hard caps everywhere.
   * Backoff on failures.

---

## Muse Integration

### Named queries (add to the v1 catalog)

These integrate with the existing “named query” design (single dispatcher).

* `github_status()`

  * queue length, active fetches, cooldown blocks, last fetches

* `github_repo_summary(repo)`

  * most recent resources, top PRs/issues by importance_score, last updated

* `github_find(term, repo?, limit?)`

  * search across extracted atoms (mentions/cves/dep changes)

* `github_recent_changes(window_ticks, limit)`

  * recent PR merges, dependency changes, advisory mentions

> Implementation note: these should be realized via the same `graph_query.run(name,args)` gate and served by `/graph`.

### `/facts` snapshot additions

Include a `github` section:

* monitored repos
* last N fetched PR/issue/release resources
* top N atoms by frequency
* invariants violations (if any)

Muse must answer GitHub questions only using `/facts` and `/graph` outputs.

---

## Invariants / Validators (v1 minimal)

* Every `web:resource` must have exactly one `web:source_of` edge.
* All `web:links_to` targets must be `web:url`.
* Every `web:url` must have `canonical_url`.
* For GitHub resources, `kind` must be one of `github:*`.

---

## Tests

### Unit tests

1. URL canonicalization:

   * strips fragments
   * stable hashing

2. Cooldown enforcement:

   * scheduling blocked when `now < next_allowed_fetch_ts`
   * backoff increases on failures

3. Atom extraction boundedness:

   * max atoms per resource
   * stable order

### Integration tests (no external network)

Mock GitHub API server (local fixture):

* endpoints for repo, PR list, PR item, compare/diff
* ensure:

  * `web:url` nodes created for fetched endpoints
  * `web:resource` nodes created for responses
  * edges `web:source_of` and `web:links_to` created
  * `github_extract_atoms` event emitted
  * cooldown prevents refetch

### E2E-lite

* Run Part64 with fixture server seeds.
* `/facts` returns snapshot with github section.
* `/graph github_repo_summary owner/repo` returns deterministic list.

---

## Acceptance Criteria

1. From a seed repo list, the system fetches PR/issue/release pages (or API) and creates `web:url` + `web:resource` nodes.
2. For PRs, it extracts at least: title, number, updated_at, labels (if available), and touched filenames (if diff fetched).
3. It assigns an `importance_score` deterministically and only schedules top-K per repo.
4. Cooldown/backoff prevents spam.
5. `/facts` and `/graph` can report recent GitHub activity with receipts (hashes + ids).

---

## Implementation Notes (minimal touched files)

Prefer creating new modules rather than refactoring:

* `part64/code/world_web/github_presence.py` (new)
* `part64/code/world_web/github_fetcher.py` (new)
* `part64/code/world_web/github_extract.py` (new)
* `part64/code/world_web/graph_query.py` (extend named queries)
* `part64/code/world_web/facts_snapshot.py` (extend summary)

Existing integration points (likely):

* `presence_runtime.py` (register new presence loop)
* `server.py` / `muse_runtime.py` (ensure `/graph` routes to named queries)

---
