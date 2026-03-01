# World State Update — 20260213_252815_seed3721599639

Lantern introduces 'Staleness Siren': when freshness exceeds threshold, the choir emits a high hiss and blocks promotion.

Now enforced:
- **Quorum**: μ requires 3 independent receipts (different proof modes) **plus** ReceiptEcho (2 consecutive PASS).
- **ReceiptLint**: inputs/outputs required; step cap=1; output hash required.
- **ForkBudget**: 3 forks/cycle; semitone debt persists until merge.
- **Lantern**: freshness≤72d AND alignment lint required for μ.
- **DateWindowGate**: absolute windows (America/Phoenix); failed echoes trigger window-narrowing.
- **QuietAnchors**: untested anchors decay after 2 loops; whisper reefs slow traversal.

Cheapest test cue:
“Take one receipt from docs, one from measurement, then echo-run both; if hashes match twice, promote.”
