# Whisper Benchmark (Unified Bench System)

Whisper benchmarking now runs through the same comprehensive benchmark runner
used by the embedding benchmark stack.

Default whisper suite:
- `scripts/benchmark_suites/whisper_openvino_starter.json`
- Dataset: `hf-internal-testing/librispeech_asr_dummy` (`clean` / `validation`)
- Devices: `NPU`, `GPU`, `CPU`
- Sizes: `tiny`, `base`, `small`

## Run via Unified Compose Stack

From `part64/`:

```bash
docker compose -f docker-compose.embed-bench.yml --profile runner run --rm model-bench-runner
```

Alternative compatibility command:

```bash
docker compose -f docker-compose.whisper-bench.yml up --build
```

## Output

- JSON report: `runs/model-bench/model-bench.latest.json`
  (or `runs/whisper-bench/whisper-benchmark.latest.json` in compatibility mode)

Each row includes:
- case id, runner id, model id, device, status
- model load time, inference timing, average/p95 latency
- realtime metrics (`real_time_factor`, `x_realtime`)
- WER when transcript references are available
- weighted score breakdown from criteria

## Override Suite or Output

```bash
MODEL_BENCH_SUITE=/workspace/scripts/benchmark_suites/whisper_openvino_starter.json \
MODEL_BENCH_OUTPUT=/results/whisper-benchmark.latest.json \
docker compose -f docker-compose.embed-bench.yml --profile runner run --rm model-bench-runner
```

For full suite schema and multi-runner benchmarking, see `MODEL_BENCH_RUNNER.md`.
