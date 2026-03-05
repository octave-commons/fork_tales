## Signal

Here’s a “toolbox menu” that plays well with crawling + extraction + threat detection + denoising, **with LLMs in a supporting role** (querying, labeling, summarizing) and **classic/statistical models as primary signals**.

### Acquisition and crawling

**Frontier + scheduling**

* **Politeness + backpressure as first-class signals**: per-domain token buckets; global tick governor; queues by lane (security/news/code) with caps.
* **Priority queue with exploration**: rank URLs by estimated value, but reserve a small % for exploration (ε-greedy).
* **Contextual bandits** (UCB/Thompson) for “which sources/queries/lanes pay off” using reward = “did it yield new high-signal atoms?”

**Content change detection**

* **ETag/Last-Modified** + diff-aware fetch (don’t re-embed unchanged pages).
* **Shingle similarity** (MinHash/SimHash) to detect “same story, different URL” and “minor edits.”

### Extraction (turn blobs into atoms)

**Deterministic parsing first**

* Regex/finite patterns for “hard entities” (CVE/CWE, IOC-ish patterns, version strings, package coords, commit SHAs, file paths).
* Aho–Corasick / trie matching for high-throughput keyword sets (seed lists, vendor lists, malware lexicons).

**Entity resolution**

* Canonicalization rules + **fuzzy matching** (edit distance / keyboard distance) for typosquats and aliasing.
* “Same-entity” clustering using **embedding + string** features (e.g., “LockBit 3.0” vs “LockBit Black”).

**Weak supervision (huge win)**

* Use **label functions** (rule snippets) to assign noisy labels (e.g., “exploit in the wild”, “PoC”, “credential theft”) and then train a model that learns how to combine them.
* LLM helps here by *generating candidate label functions*—but the output is still deterministic functions you own.

### Feature extraction for “new stuff” (your proximity idea)

**Three proximity axes (deterministic)**

* **Semantic proximity**: context embeddings → nearest prototype + margin.
* **Graph proximity**: co-occurrence graph + PPR from seed sets.
* **Temporal proximity**: burst/change-point + source diversity.

This yields robust features even for unseen tokens.

### Scoring and classification

**Boring models that punch above their weight**

* **Logistic regression (max entropy)** / **linear SVM** on TF-IDF + your structured features (counts of CVE patterns, prototype label, PPR scores, burst score, source type).
* **Naive Bayes** as a baseline (and regression test against “getting worse”).
* **Calibration** (Platt / isotonic) so scores mean something (and thresholds don’t rot immediately).

**Ranking > classification**

* Most of what you want is “what should I look at next?” So optimize **Precision@K / Recall@K**, not accuracy.

### Temporal stabilizers (denoise “flappy” signals)

* **HMM/HSMM**: convert twitchy per-item scores into stable states (background → emerging → active → critical).
* **EWMA / CUSUM / change-point detection**: “something shifted” alarms without LLM judgement.
* **Kalman-ish smoothing** for continuous signals (price, volume, mention rates).

### Anomaly detection

Great when you don’t have labels yet.

* **Isolation Forest / One-class SVM** on feature vectors (new repo behaviors, unusual dependency scripts, sudden maintainer changes).
* **Robust z-scores / MAD** on time series (burst detection, new source turning noisy).

### Graph-native threat detection (super relevant for supply chain)

* **Bipartite graphs**: {package/repo} ↔ {advisory/CVE/maintainer/domain}
* **Belief propagation / label propagation** from known-bad seeds.
* **Community detection** to spot clusters of mutually reinforcing “weirdness” (same domains, same authors, same patterns).

### Source credibility and consensus (world lane especially)

* **Source weighting** by historical precision: official advisories > wires > blogs > social.
* **Redundancy requirement**: promote to “active” only after N independent sources (HMM observation feature).
* **Stance + contradiction tracking**: not “truth,” just “how many sources agree vs conflict.”

### Where the LLM fits (without becoming the judge)

Use it for:

* **Query expansion** (synonyms, aliases, local jargon)
* **Candidate span proposal** (what to treat as an entity)
* **Weak labeling suggestions** (generate label-function drafts)
* **Summaries/explanations** strictly from named-query outputs/atoms

Not for:

* assigning final severity,
* deciding truth,
* “one-shot” risk ratings.

---

## Evidence

No web sources needed here; these are stable, well-known techniques and patterns for streaming IR/NLP + security triage pipelines.

---

## Frames

1. **Pipeline frame:** crawl → normalize → extract atoms → score → smooth over time → alert. Each stage is small and testable.
2. **Ranking frame:** you’re building an attention allocator, not an oracle.
3. **Defense-in-depth frame:** multiple weak, deterministic signals beat one “smart” but un-auditable signal.

---

## Countermoves

* Don’t upgrade anything to “critical” from a single model. Require **two-of-three**: (semantic proximity, graph proximity, temporal burst) + source tier.
* Keep a “dumb baseline” alive so you can detect when your fancy system silently regresses.
* Build **model cards as atoms**: `(model_version, features_used, calibration, thresholds)` so drift is observable.

---

## Next

Pick **one** lane (GitHub security or world news) and implement exactly these three additions first:

1. **MinHash/SimHash dedupe**,
2. **logreg ranker** (TF-IDF + structured features),
3. **HMM state smoothing** (emerging/active/critical).
