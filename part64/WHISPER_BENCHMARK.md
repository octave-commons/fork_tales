# Whisper Benchmark Container

This benchmark stack evaluates multiple Whisper model sizes across OpenVINO
devices (`NPU`, `GPU`, `CPU`) using a Hugging Face dataset.

Default dataset:
- id: `hf-internal-testing/librispeech_asr_dummy`
- config: `clean`
- split: `validation`

Default model sizes:
- `tiny` (`openai/whisper-tiny.en`)
- `base` (`openai/whisper-base.en`)
- `small` (`openai/whisper-small.en`)

## Run

From `part64/`:

```bash
docker compose -f docker-compose.whisper-bench.yml up --build
```

The container is configured as `privileged` so device plugins can access local
accelerators for NPU/GPU runs.

## Output

- JSON report: `runs/whisper-bench/whisper-benchmark.latest.json`

Each row includes:
- `model_id`, `device`, `status`
- model load time, inference time, average/p95 latency
- real-time factors (`real_time_factor`, `x_realtime`)
- WER against the dataset transcript column (when available)

## Common Overrides

```bash
WHISPER_BENCH_MODELS=tiny,base,small,medium \
WHISPER_BENCH_DEVICES=NPU,GPU,CPU \
WHISPER_BENCH_MAX_SAMPLES=64 \
WHISPER_BENCH_DATASET=hf-internal-testing/librispeech_asr_dummy \
WHISPER_BENCH_DATASET_CONFIG=clean \
WHISPER_BENCH_SPLIT=validation \
docker compose -f docker-compose.whisper-bench.yml up --build
```

Optional strict modes:
- `WHISPER_BENCH_FAIL_ON_MISSING_DEVICE=1`
- `WHISPER_BENCH_FAIL_ON_ERROR=1`
