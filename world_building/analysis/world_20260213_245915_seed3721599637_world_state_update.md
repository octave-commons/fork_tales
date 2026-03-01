# World State Update — 20260213_245915_seed3721599637

Lantern splits brightness into two channels: freshness and alignment; both must be bright for μ.

Enforced:
- **ReceiptLint v2.1**: receipts must include explicit inputs/outputs; preferred single-step; **step cap = 2**.
- **ForkBudget**: **2 forks/cycle**. Spend lowers the Choir’s sub-bass one semitone until you merge.
- **Lantern (dual channel)**: freshness threshold **42 days** AND quote-alignment must both be “bright” for μ.
- **DateWindowGate**: absolute event window + timezone (**America/Phoenix**). Also records event-date vs publish-date separately.
- **Hysteresis (diverse)**: demoted claims require **2 receipts** from different proof modes to regain μ.
- **QuietAnchors**: untested anchors decay after **1 loops** and can spread whisper-state to neighbors.

Cheapest test cue (from chorus):
“Try a one-step receipt that outputs a checksum; if it doesn’t match twice, keep it η.”
