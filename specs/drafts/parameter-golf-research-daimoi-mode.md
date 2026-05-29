# Parameter Golf Research Daimoi Mode (Draft) — 2026-03-20

## Goal
Upgrade the Fork Tales Web Graph Weaver from a mostly link-following crawler with light activation heuristics into a **research-daimoi field router** specialized for Parameter Golf and adjacent compact-model research.

The crawler should prefer:
- primary raw artifacts,
- semantically relevant sources,
- corroborating evidence that changes the outcomespace,
- cluster representatives over navigation exhaust.

## Problem statement
Current `web_graph_weaver.js` has useful infrastructure:
- ethical crawling
- graph delta emission
- event logging
- entity movement
- activation potential
- optional text summarization

But the routing logic is still too close to:
- outgoing links
- activation thresholds
- random jitter

This creates well-known failure modes for research crawling:
- GitHub navigation dominates patch/raw source extraction
- secondary sources can sprawl into tertiary noise
- semantically valuable but weakly linked nodes are underweighted
- the crawl does not explicitly reason about source quality, corroboration value, or motif coverage

## Desired shift
Move from:
- URL traversal with entity flavor

to:
- **evidence-field traversal with daimoi semantics**

## Core Fork Tales translation

### Presences
Introduce explicit research presences, each with:
- a spec embedding / lens
- need weights over motifs / source kinds
- priority
- optional budget/cost preference

Initial presences:
1. **Quantizer**
   - seeks QAT, low-bit export, outlier protection, codebooks
2. **Recursor**
   - seeks recurrence, layer sharing, phase-conditioned reuse, depth compression
3. **Evaluator**
   - seeks sliding-window eval, TTT, iterative refinement, eval-time compute
4. **Tokenizer**
   - seeks vocab/tokenizer/head tradeoffs
5. **Witness**
   - seeks corroborating edges between raw and secondary evidence
6. **Archivist**
   - prefers raw artifacts and canonical sources over summaries

### Daimoi
Each daimon packet should carry more than a URL. It should carry a typed semantic packet:
- `source_kind_probs`
  - patch
  - submission_json
  - train_log
  - raw_markdown
  - arxiv_abs
  - arxiv_pdf
  - repo_html
  - blog
  - dashboard_json
- `motif_probs`
  - quantization
  - recurrence
  - tokenizer
  - optimizer
  - eval_time_compute
  - artifact_interface
  - search_procedure
- `source_quality`
  - primary
  - secondary
  - tertiary
- `novelty_score`
- `corroboration_potential`
- `cluster_id?`

## Source hierarchy
The crawler should explicitly value sources in this order:

### Tier 1 — primary raw signals
- patch diffs
- raw `submission.json`
- raw `train.log`
- raw repo source files
- raw markdown notes/specs
- arXiv abstract / PDF
- dataset/benchmark manifests

### Tier 2 — semi-primary synthesized artifacts
- unofficial dashboard JSON
- research-garden note pages
- project docs that summarize experiments but link back to raw artifacts

### Tier 3 — secondary corroboration
- blogs
- commentary pages
- social discussions
- discussion threads

Rule:
- Tier 3 may corroborate or contextualize,
- but Tier 3 should not dominate expansion when Tier 1 is available.

## Proposed routing score
For a candidate node `n`, a presence-specific score should look like:

```text
score(n, presence) =
  + w1 * presence_alignment(n)
  + w2 * raw_signal_bonus(n)
  + w3 * corroboration_gain(n)
  + w4 * novelty_gain(n)
  + w5 * cluster_bridge_value(n)
  - w6 * navigation_noise_penalty(n)
  - w7 * domain_cost_penalty(n)
  - w8 * duplicate_cluster_penalty(n)
```

### Interpretation
- `presence_alignment`: does this node match what the presence wants?
- `raw_signal_bonus`: raw artifact > summary page
- `corroboration_gain`: does this node support or clarify a promising claim?
- `novelty_gain`: are we still blind in this motif cluster?
- `cluster_bridge_value`: does this connect two important evidence islands?
- `navigation_noise_penalty`: login pages, generic trending pages, marketing pages, etc.
- `domain_cost_penalty`: expensive/noisy domains or pages with low information density
- `duplicate_cluster_penalty`: too many nearly identical nodes in the same cluster

## Cluster-first expansion
The crawler should not expand every discovered URL equally.

Instead:
1. embed/featurize nodes from summaries, titles, and known source metadata
2. assign them to motif clusters
3. prefer expanding:
   - cluster representatives
   - nodes with high raw-signal density
   - nodes that resolve uncertainty between sources
   - nodes that bridge important clusters
4. suppress expansion within low-yield or already-saturated clusters

## Required graph edge kinds
Add or elevate these edge semantics:
- `implements`
- `corroborates`
- `contradicts`
- `summarizes`
- `same_motif`
- `same_family`
- `raw_support_for`
- `secondary_analysis_of`
- `derives_from`
- `outcomespace_relevant_to`

This changes the graph from “pages linked to pages” into “evidence supports hypotheses and strategy families.”

## Immediate heuristic improvements
Before a full embedding/cluster implementation, add rule-based improvements:

### Positive bonuses
- `patch-diff.githubusercontent.com` URLs
- `raw.githubusercontent.com` URLs inside relevant repos
- `arxiv.org/abs/*`
- `parameter-golf.github.io/data/*.json`
- exact PR URLs for tracked submissions

### Negative penalties
- GitHub login URLs
- generic GitHub trending/search/marketplace pages
- generic social/profile pages
- marketing landing pages without parameter-golf motifs
- repeated query pages that mostly recurse into UI chrome

### Allowlist bias
Add a mode where the crawler prefers these hosts/path classes:
- `parameter-golf.github.io/data/*`
- `patch-diff.githubusercontent.com/raw/openai/parameter-golf/pull/*`
- `raw.githubusercontent.com/openai/parameter-golf/*`
- `raw.githubusercontent.com/agustif/parameter-golf-research-garden/*`
- `arxiv.org/abs/*`
- optionally `huggingface.co/*` when directly referenced by tracked sources

## Remote deployment target
Current remote host:
- `error@ussy3.promethean.rest`

Current service:
- container `parameter-golf-weaver`
- port `8793`

Current state:
- service works
- curated seed file transfer works
- focused reseeding works better than broad HTML seeds
- but the current scorer still expands too much on generic HTML descendants

## Phases

### Phase 0 — policy/spec
- freeze source hierarchy
- freeze seed cluster taxonomy
- freeze scoring terms and penalty classes

### Phase 1 — source typing
- infer `source_kind` and `source_quality`
- emit these fields into node metadata and events

### Phase 2 — allowlist / denylist routing
- bias candidate scoring by host/path class
- penalize navigation-noise patterns

#### Phase 2a — concrete Parameter Golf routing policy
- hard-block known dead descendants such as `patch-diff.githubusercontent.com/raw/openai/parameter-golf/pull/*.patch` once identified as structurally robots-blocked in this mode
- redirect blocked patch-diff discoveries toward canonical bridge URLs like `https://github.com/openai/parameter-golf/pull/<n>` instead of letting frontier work churn on dead raw artifacts
- strongly favor raw follow-ons when available:
  - raw `submission.json`
  - raw `train.log`
  - raw `train_gpt.py`
- allowlist high-value evidence roots:
  - `parameter-golf.github.io/data/*.json`
  - `raw.githubusercontent.com/openai/parameter-golf/*`
  - `raw.githubusercontent.com/agustif/parameter-golf-research-garden/*`
  - `arxiv.org/abs/*`
- hard-block or strongly penalize noisy descendants:
  - GitHub login / marketplace / trending / search / feature pages
  - GitHub `blob/tree/commits` HTML loops unless no better route exists
  - Slack invite paths
  - DOI / ADS index hops when they are not adding fresh evidence

#### Phase 2b — bridge preference
- when a candidate frontier contains both navigational GitHub pages and raw record artifacts, route into the raw record artifact first
- reward transitions like:
  - `parameter_golf_pr -> submission_json`
  - `parameter_golf_pr -> train_log`
  - `parameter_golf_pr -> train_gpt.py`
  - `research_garden_note -> raw parameter-golf artifact`
  - `leaderboard_json -> parameter_golf_pr`
- explicitly punish same-kind loops such as `patch_diff -> patch_diff` and `repo_navigation -> repo_navigation`

### Phase 3 — motif clustering
- assign motif vectors / clusters to nodes
- prefer cluster representatives and bridge nodes

### Phase 4 — presence-aware daimoi routing
- create explicit research presences
- score candidates by presence demand alignment
- let entities route by presence-specific fields instead of flat heuristics

### Phase 5 — evidence graph outputs
- emit strategy-family views
- emit corroboration chains
- emit outcomespace summaries for the Parameter Golf search lab

## Definition of done
The crawler should be considered upgraded when:
1. raw artifacts dominate top-ranked discoveries
2. generic GitHub/navigation pages no longer dominate frontier growth
3. nodes are typed by source kind and motif cluster
4. the graph can answer “what evidence supports this strategy family?”
5. different presences route differently over the same frontier graph

## Why this matters
Parameter Golf is already producing a public evidence field:
- patches
- manifests
- train logs
- papers
- summary layers

A real Fork Tales daimoi system should be able to move through that field like a research organism,
not like a polite spider.
