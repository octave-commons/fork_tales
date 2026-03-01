# World State Update — 20260213_254115_seed3721599640

Lantern introduces Staleness Siren: if freshness exceeds threshold, the choir emits hiss and blocks promotion.

Live rules:
- **ForkBudget**: 1 forks/cycle.
- **SpendMapping**: every fork must name the question it buys; unnamed forks are void.
- **ReceiptLint + InputSeal**: inputs/outputs required; step cap=2; include input_hash + output_hash.
- **Quorum**: μ requires 3 independent receipts (doc + measurement) **and** ReceiptEcho (2 consecutive PASS).
- **FreshnessLock**: high-volatility claims need sources within 56 days (else η).
- **DateWindowGate**: absolute windows (America/Phoenix); echo failures trigger window-narrowing.
- **QuietAnchors**: untested anchors decay after 3 loops; whisper reefs slow traversal.

Cheapest test cue:
“Name the question, seal the input, run twice, then cross-check doc vs measurement hashes.”
