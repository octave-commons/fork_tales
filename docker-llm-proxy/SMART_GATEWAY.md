# Promethean Smart Gateway

This stack adds a field-aware model router and a React control panel.

## What it does

- Routes OpenAI-compatible chat requests by `gateway.mode`, `gateway.field`, and `gateway.hardware`.
- Keeps provider/model format (`provider/model`) end-to-end.
- Supports public provider tag aliasing (`octave-commons`) to internal provider key (`octave_commons`).
- Adds route preview API before execution.

## New API endpoints

- `GET /v1/gateway/config`
- `POST /v1/gateway/route`

Both require the same proxy API key used for `/v1/chat/completions`.

## Request contract

```json
{
  "model": "octave-commons/promethean",
  "messages": [{ "role": "user", "content": "Hello" }],
  "stream": false,
  "gateway": {
    "mode": "smart",
    "field": "code",
    "hardware": "gpu"
  }
}
```

Modes:

- `smart`: field/hardware-based routing
- `direct`: route directly to `gateway.direct_model`
- `hardway`: route to hardway path (or `gateway.hardway_model`)

## UI

The React UI is served by the `gateway-ui` service.

- URL: `http://localhost:${UI_PORT:-5173}`
- Proxy URL default in UI: `http://localhost:${PORT:-18000}`

## Compose startup

```bash
docker compose up -d --build
docker compose logs -f llm-proxy gateway-ui
```

## TensorFlow training profile

A TensorFlow trainer service is available behind the `training` profile:

```bash
mkdir -p training-data/qwen3_vl/images training-output
docker compose --profile training up -d --build tf-trainer
docker compose logs -f tf-trainer
```

Harness files live in `training/tf_qwen3_vl/`.

The trainer auto-builds dataset shards from system corpus + simulation labels before training:

- corpus root: `/vault` (mounted repo)
- label priors: `.opencode/runtime/eta_mu_knowledge.v1.jsonl`
- interaction priors: `part64/world_state/decision_ledger.jsonl`

For image-focused simulation optimization, use:

- `docker compose --profile training up -d --build tf-image-trainer`
- model export target: `training-output/export/qwen3-vl-2b-image/`

To serve the exported image router as an OpenAI-compatible local provider:

- `docker compose --profile serving up -d --build tf-image-bridge`
- bridge endpoint: `http://localhost:${TF_IMAGE_ROUTER_PORT:-8501}/v1`
- model id: `tensorflow/qwen3-vl-2b-image`

The proxy compose now prewires `TENSORFLOW_API_BASE` to `tf-image-bridge` so smart routes can hit the local TensorFlow provider when the bridge is running.

## Provider setup for routing targets

Define providers in `.env` using this pattern:

- `<PROVIDER>_API_BASE`
- `<PROVIDER>_API_KEY_1`
- `<PROVIDER>_MODELS`

For your custom org tag:

- Public model entry: `octave-commons/promethean`
- Internal env prefix: `OCTAVE_COMMONS_*`
