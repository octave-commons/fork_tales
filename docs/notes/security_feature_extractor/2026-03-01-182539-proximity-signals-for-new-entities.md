## Signal

“Proximity to known quantities” is exactly how you get **real-time lift** on *new words* without handing the keys to an LLM. Build **three deterministic proximity signals** and treat “unknown tokens” as provisional entities until they stabilize.

### 1) Distributional proximity (context → embedding → nearest known)

Instead of scoring the *word*, score the **context span** where it appears.

**Pipeline**

* Extract candidate spans (caps, noun phrases, repo names, tool names, orgs, CVE-like patterns).
* For each span occurrence, compute a **context embedding** (sentence/paragraph window).
* Maintain **prototype embeddings** for known things:

  * known threat actors / malware families / vuln classes
  * geopolitical entities (countries, orgs, places)
  * your own “watchlists” (projects, vendors, sectors)

**Features**

* `sim_top1`, `sim_top5_mean` to prototypes
* `prototype_label` of nearest neighbor (as a categorical feature)
* `margin = sim_top1 - sim_top2` (helps reject ambiguous matches)

This handles “unseen word” because the model never needed the token in-vocab; it only needs a contextual vector.

### 2) Relational proximity (co-occurrence graph + label propagation)

Make a **streaming bipartite graph**:

* nodes: `{entity_or_phrase}` and `{known_quantity}` (plus optional `{doc}` nodes)
* edges: co-occur in same sentence/paragraph/doc with a weight (PMI-ish or just log-count)

Then compute a deterministic “closeness to seeds” score:

* **Personalized PageRank** (PPR) from seed sets (e.g., “KEV/CISA items”, “ransomware”, “Hormuz”, “sanctions”, “shipping”, etc.)
* or a cheaper alternative: **2-hop weighted sum** from seed neighbors

**Features**

* `ppr_score_threat`, `ppr_score_geo`, `ppr_score_supply_chain`
* `degree`, `weighted_degree`
* `seed_neighbor_count`

This is the “proximity to known quantities” you’re describing, but expressed as a graph metric that updates online.

### 3) Temporal proximity (novelty/burst + “first seen near what?”)

New words matter when they:

* spike fast (**burst**),
* and spike **near** known-risk entities.

**Features**

* `burst_score(term)` (cheap: z-score vs rolling baseline)
* `first_seen_seed_overlap` (did it first appear in docs that also mention seeds?)
* `source_diversity` (how many distinct sources mention it?)

This stops your system from overreacting to a single weird blog post.

---

## How this becomes a *deterministic* classifier (primary signal)

You feed those proximity features into something boring and strong:

* **Logistic regression / MaxEnt** as the main ranker
* (Optional) Naive Bayes as a baseline sanity check

So the model learns things like:

* “If a new token is bursty **and** its context embedding is close to known exploit language **and** it sits 1–2 hops from `CVE`/`PoC`/vendor entities, rank it high.”

The LLM stays secondary: propose queries / synonyms / candidate spans, but the **decision** is still the classifier.

---

## Evidence

Your existing “Muse must only answer from named query outputs” contract is the right safety rail for this: you can expose only *bounded* proximity outputs (top-K neighbors, PPR scores, burst scores, prototype matches) through named queries and keep the Muse grounded. 

---

## Frames

1. **Unknown words aren’t unknown entities**: treat them as provisional nodes whose meaning is defined by *neighbors + contexts*.
2. **Proximity is multi-axis**: semantic (embeddings), relational (graph), temporal (burst). Any one alone is fragile.
3. **LLM as scout, not judge**: it helps you find candidates; it doesn’t decide risk.

---

## Countermoves

* **Don’t score raw tokens**. Score *spans + contexts* (prevents garbage “new words” from dominating).
* Require **two-of-three** signal agreement for high-severity flags (e.g., embedding-proximity + graph-proximity), otherwise keep it “watch”.
* Store everything as receipts/atoms so you can audit drift: `span`, `context_hash`, `nearest_prototype`, `ppr_scores`, `burst_score`.

---

## Next

Implement one new atom type and one new named query:

* Atom: `proximity_hit(term_id, seed_id, kind, score, ts)` where `kind ∈ {embed, graph, burst}`
* Named query: `proximity_radar({window_ticks, seed_set, limit}) → top terms w/ (embed_top1, ppr, burst, sources)`

That gives you real-time “new stuff near known stuff” without vibes.
