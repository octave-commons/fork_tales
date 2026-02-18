# TensorFlow Qwen3-VL Fine-Tune Harness

This is a TensorFlow training harness for your custom Qwen3-VL:2B implementation.

Important:

- `qwen3-vl:2b` from Ollama is an inference artifact, not a directly trainable checkpoint.
- You need your own TensorFlow backbone implementation and compatible tokenizer/weights source.
- `model_impl.py` currently contains a small reference VLM to validate the loop and plumbing.

## Corpus + simulation labeling pipeline

`prepare_corpus_dataset.py` builds training JSONL by combining:

- system file corpus under `/vault` (repo files)
- simulation labels from:
  - `.opencode/runtime/eta_mu_knowledge.v1.jsonl`
  - `part64/world_state/decision_ledger.jsonl`

It also emits `training_examples.v1.jsonl` entries aligned to:

- `contracts/ημ_ui_training_constitution.v1.lith`
- `training.example.v1` record shape (event + L0/L1/L2/L3/L4/L6 snapshot pointers + label)

The script uses live field ontology (`f1..f8`) and touch priors to auto-label examples.

Run dataset build only:

```bash
python prepare_corpus_dataset.py --vault-root /vault --out-dir /data/qwen3_vl
```

## Data format

Create JSONL files:

```json
{"image":"images/0001.jpg","prompt":"Describe this PCB fault.","response":"The capacitor near U3 appears burnt."}
```

You can also use `messages` format; the loader derives prompt/response automatically.

## Configure

Edit `config.yaml`:

- `model.tokenizer_id` -> set your Qwen tokenizer id
- dataset paths under `/data/...`
- output paths under `/output/...`

## Run with Docker Compose profile

From repo root (`docker-llm-proxy`):

```bash
mkdir -p training-data/qwen3_vl/images training-output
docker compose --profile training up -d --build tf-trainer
docker compose logs -f tf-trainer
```

The trainer service now auto-runs `prepare_corpus_dataset.py` before `train.py`.

## Image training loop (simulation optimization)

Runs image-only supervision over simulation-labeled corpus entries.

```bash
docker compose --profile training up -d --build tf-image-trainer
docker compose logs -f tf-image-trainer
```

This uses:

- `train_image_loop.py`
- `config.image.yaml`
- labels generated from simulation field metadata (`f1..f8`)

Stop:

```bash
docker compose --profile training stop tf-trainer
```

## Run directly in container one-shot

```bash
docker compose run --rm tf-trainer python train.py --config /workspace/config.yaml
```

Smoke test config (quick trainability check):

```bash
docker compose run --rm tf-trainer sh -lc "python prepare_corpus_dataset.py --vault-root /vault --out-dir /data/qwen3_vl --max-file-samples 96 --max-knowledge-samples 96 && python train.py --config /workspace/config.smoke.yaml"
```

Image-loop smoke test:

```bash
docker compose run --rm tf-trainer sh -lc "python prepare_corpus_dataset.py --vault-root /vault --out-dir /data/qwen3_vl --max-file-samples 48 --max-knowledge-samples 64 && python train_image_loop.py --config /workspace/config.image.smoke.yaml"
```

## Next step for real Qwen3-VL:2B fine-tuning

1. Replace `build_model()` in `model_impl.py` with your TensorFlow Qwen3-VL graph.
2. Load your pretrained weights into that graph.
3. Keep output logits shape `[batch, seq, vocab]` so the existing loss/loop works.
4. Export saved model and wire your TensorFlow inference endpoint into the gateway via:
   - `TENSORFLOW_API_BASE`
   - `TENSORFLOW_API_KEY_1`
   - `TENSORFLOW_MODELS`

## Local serving bridge for exported image router

This folder now includes `serve_image_router.py`, an OpenAI-compatible bridge that serves:

- `GET /v1/models`
- `POST /v1/chat/completions`

It loads the latest `qwen3-vl-2b-image*` SavedModel under `/output/export` (or `TF_IMAGE_ROUTER_EXPORT_DIR`).

Start it with compose:

```bash
docker compose --profile serving up -d --build tf-image-bridge
docker compose logs -f tf-image-bridge
```

Then test the bridge directly:

```bash
curl -s http://localhost:8501/v1/models \
  -H "Authorization: Bearer local-tensorflow-token"
```
