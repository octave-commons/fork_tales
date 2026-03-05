## Signal

Yes — **HMMs are a good fit**, but *not* as “the text classifier.” They’re best as a **temporal stabilizer / regime detector** that sits *on top of* your per-document scoring and proximity features.

Think of it like:
**(instant score) → (HMM smooths over time) → (stateful risk)**

### Where an HMM helps your “new words in real time” problem

When a new token / phrase / repo name appears, you don’t know what it is yet. An HMM gives you a disciplined way to say:

* “This looks like background noise”
* “This is **emerging**”
* “This is **confirmed / sustained**”
* “This cooled off”

…without the score flapping every time a single source posts something spicy.

---

## A practical HMM design for your lanes

### Hidden state (what you *really* want)

Use 3–5 discrete states per tracked entity (term/topic/repo/story-thread):

* `S0: background`
* `S1: emerging`
* `S2: active/high`
* `S3: critical/confirmed`
* (optional) `S4: decaying`

Make it **sticky**: strong self-loops so you only transition when evidence persists.

### Observations (what you can measure deterministically per tick/window)

Your earlier proximity features plug in cleanly:

* `burst_score` (temporal novelty)
* `graph_proximity` (PPR / 2-hop to seed sets)
* `embed_proximity` (nearest prototype + margin)
* `source_diversity` (how many independent sources)
* “hard patterns” counts (CVE/IOC/etc.)

You have two clean ways to feed these into HMM emissions:

**Option A (simple, robust): emissions are based on an upstream classifier score**

* Train a cheap model (logreg/SVM) that outputs `p_risk` per item/window.
* Emission model per state is something like **Gaussian over logit(p_risk)** or discretized bins.

**Option B (classic burst detector): Poisson emissions on mention counts**

* For each term/topic, track count per window.
* HMM states correspond to different Poisson rates (`λ0` low, `λ1` high, etc.).
* This is extremely good at “suddenly everybody is talking about X”.

You can run **both**, and treat them as separate observation channels.

### Output (what you use as the primary signal)

The main deliverable from the HMM is:

* `P(state = critical | history)`
* plus a **Viterbi path** if you want segment boundaries (“when did it flip to active?”)

That becomes your deterministic “risk state” that daimoi can react to.

---

## Why HMM beats “LLM as judge” here

* It’s **auditable** (transition matrix + emissions are explicit).
* It’s **hard to game** with one-off keyword spam (stickiness + diversity features).
* It naturally encodes “we need persistence before we panic.”

---

## Evidence

No external sources used; this is standard sequence-model design applied to your “proximity + novelty + multi-source” feature set.

---

## Frames

1. **HMM as inertia**: it turns a twitchy score into a stable state machine.
2. **Unknown words become entities**: you track them as provisional nodes; the HMM tells you when they graduate from “weird” to “real.”
3. **Classifier ≠ temporal model**: the classifier is “what is this?”; the HMM is “is it sticking around and growing?”

---

## Countermoves

* Don’t make the HMM read raw text. Feed it **scores + proximity + burst + source-diversity**.
* Don’t overcomplicate emissions early. Start with “score-binned emissions” and a sticky transition matrix.
* If you notice “events have duration,” consider **HSMM** (hidden *semi*-Markov) later; it handles dwell time better than vanilla HMM.

---

## Next

Implement a **4-state sticky HMM per tracked term/topic**, with emissions driven by your current deterministic features (or a logreg `p_risk`), and expose only the bounded outputs via a named query like:

`proximity_radar → {term, p_critical, p_active, last_transition_tick, top_evidence}`
