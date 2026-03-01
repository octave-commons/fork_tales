# Gates of Truth — Dense Intent Compression (Local Box Corpus)

This dataset is built from the *box itself*, including any new chapters added over time.

- `intent_events.bin` — 16 bytes/event (fixed-width)
- `intent_anchors.jsonl` — S3-style anchors for every intent event
- `intent_events.jsonl` — readable mirror
- `intent_timeline.csv` — per-chapter intent energy

Claim IDs are prefixed with `intent_box:` to distinguish from production-bundle runs.
