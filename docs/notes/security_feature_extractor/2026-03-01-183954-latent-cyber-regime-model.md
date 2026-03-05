## Signal

Your intuition is right: in cyber, the “hidden variables” are often **not in the code**. They’re things like *actor intent, campaign phase, geopolitical incentives, target selection, tradeoffs, and timing*. A code-only radar will miss those because it’s watching the wrong sensors.

The clean way to add “hidden variable exploration” without letting an LLM freestyle is:

**(A) Keep your primary per-item risk scoring deterministic** (classic ML + hard extractors)
**(B) Add a *latent context model* that runs over time** and outputs a small state/posterior you can treat as *just another feature* and routing policy knob.

### 1) HMM in the “right place”: regime/context over time

Use an HMM where the hidden state is **macro cyber posture / campaign regime**, not “what this text means”.

Example states (pick 4–6):

* `S0 baseline`
* `S1 elevated chatter`
* `S2 active exploitation wave`
* `S3 supply-chain / dependency campaign`
* `S4 geopolitically-driven targeting shift`

**Observations** are what your system can already produce deterministically:

* `p_risk` from your classifier (logreg/SVM)
* counts/rates of hard entities in a window: CVE mentions, KEV matches, “PoC” mentions, vendor/product clusters, auth-bypass patterns, etc.
* `source_diversity` / corroboration count
* graph proximity to known bad seeds (PPR / 2-hop)

Then the HMM gives you:
`P(state | last N windows)` — a stable posterior like “we’re 0.78 in active exploitation wave”.

You don’t need the HMM to be “right” in a semantic sense; you need it to be **stable + predictive of which alarms should be treated as urgent**.

### 2) How this explores “hidden variables”

Once you have a regime posterior, you can make the system do smart things that a code-only scorer can’t:

* **Adaptive thresholds:** same static score can mean different urgency under `S2 active exploitation`.
* **Adaptive crawling:** in `S3 supply-chain`, spend more budget on new repos, dependency graphs, CI scripts.
* **Query expansion bounded by state:** the LLM can propose search terms, but only inside “allowed expansions” for the current state.

This keeps “context” as a *controlled influence*, not vibes.

### 3) “Unseen new words” in real time: treat them as provisional entities + learn by proximity

This is the other half: you won’t know new malware names, repo names, or campaign terms at first.

Use **proximity features** (deterministic):

* semantic proximity: context embedding → nearest prototype + margin
* relational proximity: co-occurrence graph → PPR from seeds
* temporal proximity: burst + source diversity

Then the HMM uses those observations to decide whether a new term is merely noisy novelty or part of an emerging regime shift.

### 4) Other “hidden-variable-ish” techniques worth adding (cheap → powerful)

**Topic discovery (find the hidden connectors)**

* **NMF on TF-IDF** (fast, deterministic, often better than LDA for interpretability). Produces “latent themes” you can name later.
* **LDA** if you want a generative story; NMF if you want “top words per topic” that behave.

Use topic IDs as features:

* `topic_7_weight`, `topic_12_weight`…
  Then you’ll literally see new “campaign topics” appear as vectors, even before you can name them.

**Change-point detection (when the world shifts)**

* CUSUM / EWMA control charts on:

  * KEV-like hits
  * exploit/PoC mentions
  * repo-discovery rate
  * “new vendor appears” rate
    This gives you “something changed” alarms without needing semantics.

**Graph community + propagation**

* community detection (clusters of repos/domains/authors)
* label propagation from known-bad seeds
  This is another way to “discover hidden variables”: the hidden variable is often “same operator/infrastructure”.

**Weak supervision**

* encode domain knowledge as label functions (regex + heuristics + provenance constraints)
* train a simple model to combine them
  LLM can *suggest* label functions, but the result is still deterministic rules + a trained combiner.

### 5) How to keep Muse grounded while doing all this

Expose the HMM + topic model only via **named queries** (top-K, bounded), same as your existing tool contract: `graph_query.run(name,args)->dict` and “Muse may only answer using named query outputs.” 

Add queries like:

* `cyber_risk_radar(window_ticks, limit)`
* `cyber_regime_state(window_ticks)` → `{state_posteriors, last_transition_tick}`
* `emerging_terms(window_ticks, limit)` → `{term, burst, ppr, prototype_match}`

Now the LLM can explain *what the system thinks* without becoming the decider.

---

## Evidence

Your existing “named query only” grounding contract already matches this architecture (single `graph_query.run` entrypoint, deterministic + bounded outputs, no ad-hoc querying). 

---

## Frames

1. **Hidden variables as regimes, not secrets:** you’re modeling *latent context* (campaign phase / targeting shift), not conjuring a cabal.
2. **Classic ML judges, LLM narrates:** the LLM helps propose expansions and explanations; it doesn’t assign severity.
3. **Discovery is topology + time:** most “unseen connectors” show up as co-occurrence structure and bursts before they show up as “known names.”

---

## Countermoves

* Don’t make the HMM state space huge. 4–6 states, sticky transitions, or it becomes interpretability soup.
* Don’t let “geopolitical context” directly set alarm level. Let it bias *priors + routing + thresholds* only.
* Require at least two independent signals for “critical” (e.g., high classifier score + graph proximity, then HMM just stabilizes).

---

## Next

Write down **5 cyber regimes** you care about (names above are fine) and **8 observation signals** you can compute deterministically per 15–60 minute window; that’s enough to stand up an HMM whose posterior becomes a first-class feature and policy knob.
