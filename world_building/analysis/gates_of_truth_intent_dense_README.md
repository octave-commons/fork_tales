# Gates of Truth — Dense Intent Compression

This folder stores *typed intent events* in both human-readable and binary-packed forms.

## Files
- `entities_index.json` — entity IDs + intent type codes + binary record schema
- `intent_events.jsonl` — one JSON per event (debuggable)
- `intent_events.bin` — packed events (**16 bytes per event**) for dense storage
- `intent_anchors.jsonl` — S3 anchors for every intent event (evidence snippets)
- `intent_timeline.csv` — per-chapter intent metrics (count/sum/mean/max)

## Why binary?
You said: **compress intent as densely as possible.**
This format is:
- fixed-width
- stream-friendly
- trivially mem-map-able
- stable across rebuilds (schema versioned in `entities_index.json`)

## Decode quick
Use:
- `operation-mindfuck/tools/intent_codec.py` (decode + pretty-print)
