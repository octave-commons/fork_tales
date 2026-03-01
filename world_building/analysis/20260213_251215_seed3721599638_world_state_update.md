# World State Update — 20260213_251215_seed3721599638

ReceiptLint v2.2 adds OutputHash: outputs must include a checksum line.

Live rules:
- **ReceiptEcho**: a receipt must PASS twice consecutively to count as μ (fluke guard).
- **ReceiptLint**: inputs/outputs required; step cap=2; output must include a checksum line.
- **ForkBudget**: 4 forks/cycle; semitone debt persists until merge.
- **Lantern**: freshness≤70d + alignment lint required=True for μ.
- **DateWindowGate**: absolute windows (America/Phoenix); auto-filled windows remain η until confirmed.
- **QuietAnchors**: untested anchors decay after 1 loops; whisper quarantine limits outward links.

Cheapest test cue:
“Run the same receipt twice; if the output hash matches, promote; if not, mint boundary and stay η.”
