# Live Choir

`live-choir` turns each chat line into a multi-entity exchange:

- Each entity has its own model config and tool list.
- Presence profiles shape motive per detected frame.
- The frame firewall (`eta/mu`) is run first, then entities respond.
- `sing_line` is a tool call, so entities "sing" their output in-chat.

## Run

```bash
npm run live
```

Type messages and watch entity turns stream as JSONL.

Plain text mode:

```bash
node cli/live-choir.mjs --plain
```

Custom entity config:

```bash
node cli/live-choir.mjs --config entities.sample.json --plain
```

## Model + Tool Separation

- Entity model is declared in `model` (`provider`, `name`).
- Entity tools are declared in `tools` and enforced per entity.
- Presence profile maps frame pressure to motive:
  - `guilt`
  - `authority`
  - `urgency`
  - `vagueness`
  - `agency_theft`

Current provider support is `mock` so it runs without API keys.
