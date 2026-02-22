---
source: part64/deep-research-report.md
section: Online Learning and Credit Assignment
status: incoming
reviewed_on: 2026-02-21
---

# Spec: Online REINFORCE Learning for Presence Absorption

## 1. Purpose
Implement the stochastic policy gradient (REINFORCE) update loop for Presence absorption parameters ($w_\beta, w_T, u_{p,k}$) as defined in the Self-Organizing Graph Runtime paper. This enables Presences to adapt their "lenses" and "needs" based on realized outcomes and rewards.

## 2. Scope
- **Trace Buffer**: Implement per-Presence ring buffers to store absorption decisions (sampled component, logits, feature vector $x$, time).
- **Reward Signal**: Define a reward function $r(t)$ based on resource need satisfaction, successful task completion, and "Value-from-Content" events.
- **Credit Assignment**: Implement eligibility traces $elig(\tau, t) = \exp(-(t-\tau_{time})/\tau_e)$ to distribute rewards back to recent decisions.
- **Update Loop**: Implement the REINFORCE update: $\theta \leftarrow \theta + \eta \cdot (r - \bar{r}) \cdot \nabla_\theta \log \pi(a|x)$.

## 3. Implementation Plan
- **Phase 1 (Data Structures)**: Add `trace_buffer` and `r_bar` (reward baseline) to Presence state in `presence_runtime.py`.
- **Phase 2 (Logging)**: Update `daimoi_probabilistic.py` to push realization traces into the buffer upon absorption.
- **Phase 3 (Rewards)**: Implement a `PresenceRewardManager` that monitors `receipts.log` and resource levels to emit reward signals.
- **Phase 4 (Updates)**: Implement the gradient update math in Python, using the saved feature vectors and realization hashes.

## 4. Acceptance Criteria
- [ ] Presence parameters $w_\beta$ and $w_T$ change over time in response to controlled reward signals.
- [ ] Eligibility traces correctly decay reward influence over time.
- [ ] Reward baseline $\bar{r}$ stabilizes learning and prevents divergence.
- [ ] Receipt schema includes the realization hash used for the update.
