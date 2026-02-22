# Model Benchmark Runner

`scripts/benchmark_runner.py` is a suite-driven benchmark orchestrator for
cross-model evaluation.

It is designed as a starter foundation for:
- any model family (via runner plugins)
- any dataset (HF and local JSON/JSONL adapters)
- any scoring criteria (metric keys or expressions)

Current built-in runners:
- `whisper_asr` (Whisper ASR with OpenVINO + Optimum)
- `embedding_text` (text embeddings through `code.world_web.ai._embed_text`)

## Run from Compose

From `part64/`:

```bash
docker compose -f docker-compose.embed-bench.yml --profile runner run --rm model-bench-runner
```

By default this runs:
- `MODEL_BENCH_SUITE=/workspace/scripts/benchmark_suites/whisper_openvino_starter.json`
- `MODEL_BENCH_OUTPUT=/results/model-bench.latest.json`

## Run Directly (Local Python)

```bash
python3 scripts/benchmark_runner.py \
  --suite scripts/benchmark_suites/universal_starter.json \
  --output runs/model-bench/universal.latest.json
```

## Suite Format (JSON)

```json
{
  "id": "suite-id",
  "description": "what this suite is testing",
  "dataset": {
    "source": "hf",
    "id": "hf-internal-testing/librispeech_asr_dummy",
    "config": "clean",
    "split": "validation",
    "max_samples": 24,
    "input_column": "audio",
    "reference_column": "text",
    "sampling_rate": 16000
  },
  "criteria": [
    {
      "id": "latency",
      "metric": "p95_latency_ms",
      "goal": "min",
      "weight": 0.4,
      "runners": ["whisper_asr"]
    },
    {
      "id": "custom_speed",
      "expression": "x_realtime",
      "goal": "max",
      "weight": 0.6,
      "runners": ["whisper_asr"]
    }
  ],
  "benchmarks": [
    {
      "id": "case_1",
      "runner": "whisper_asr",
      "model": "openai/whisper-base.en",
      "device": "NPU",
      "params": {
        "language": "en"
      }
    }
  ]
}
```

## Criteria Options

Each criterion supports:
- `metric`: metric key from benchmark output
- `expression`: arithmetic expression over metric keys (`+ - * / **`, `abs`, `min`, `max`, `sqrt`, `log`)
- `goal`: `min`, `max`, or `target`
- `weight`: positive weight in final score
- `target`: required when `goal` is `target`
- `runners`: optional list to scope criterion to runner types

## Dataset Sources

`dataset.source` values:
- `hf`: Hugging Face dataset (`id`, optional `config`, `split`)
- `json` / `jsonl` / `local_json` / `local_jsonl`: local file (`path`)

Row extraction uses:
- `input_column`
- `reference_column` (optional)

Case-level dataset overrides are supported under each benchmark item.

## Starter Suites

- `scripts/benchmark_suites/whisper_openvino_starter.json`
- `scripts/benchmark_suites/embedding_openvino_starter.json`
- `scripts/benchmark_suites/universal_starter.json`
