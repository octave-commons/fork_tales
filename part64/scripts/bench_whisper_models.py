#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from datasets import Audio, load_dataset
from jiwer import wer
from openvino.runtime import Core
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline


MODEL_ALIASES: dict[str, str] = {
    "tiny": "openai/whisper-tiny.en",
    "base": "openai/whisper-base.en",
    "small": "openai/whisper-small.en",
    "medium": "openai/whisper-medium.en",
    "tiny.en": "openai/whisper-tiny.en",
    "base.en": "openai/whisper-base.en",
    "small.en": "openai/whisper-small.en",
    "medium.en": "openai/whisper-medium.en",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
}

REFERENCE_KEYS: tuple[str, ...] = (
    "text",
    "sentence",
    "transcript",
    "transcription",
    "normalized_text",
)


@dataclass
class EvalSample:
    audio: dict[str, Any]
    reference: str
    duration_seconds: float


@dataclass
class BenchmarkResult:
    model_id: str
    device: str
    status: str
    detail: str
    sample_count: int
    model_load_seconds: float
    inference_seconds: float
    audio_seconds: float
    mean_latency_ms: float
    p95_latency_ms: float
    samples_per_second: float
    real_time_factor: float
    x_realtime: float
    wer: float | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _split_csv(raw: str) -> list[str]:
    return _dedupe(part.strip() for part in str(raw or "").split(","))


def _env_flag(name: str, default: bool) -> bool:
    text = str(os.getenv(name, "")).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _safe_int(raw: Any, default: int, minimum: int = 1) -> int:
    try:
        value = int(str(raw))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * max(0.0, min(1.0, p))
    lower = int(rank)
    upper = min(len(values) - 1, lower + 1)
    weight = rank - lower
    return float(values[lower] + (values[upper] - values[lower]) * weight)


def _resolve_model_ids(raw_models: list[str]) -> list[str]:
    resolved: list[str] = []
    for raw_model in raw_models:
        key = str(raw_model).strip().lower()
        if not key:
            continue
        model_id = MODEL_ALIASES.get(key, str(raw_model).strip())
        if model_id:
            resolved.append(model_id)
    return _dedupe(resolved)


def _probe_openvino_devices() -> list[str]:
    try:
        devices = [str(name).upper() for name in Core().available_devices]
    except Exception:
        return []
    return sorted(_dedupe(devices))


def _is_device_available(requested_device: str, openvino_devices: list[str]) -> bool:
    wanted = str(requested_device).strip().upper()
    if wanted in {"CPU", "GPU", "NPU"}:
        return any(name.startswith(wanted) for name in openvino_devices)
    return wanted in openvino_devices


def _resolve_reference_key(columns: list[str]) -> str | None:
    for key in REFERENCE_KEYS:
        if key in columns:
            return key
    return None


def _load_eval_samples(
    *,
    dataset_id: str,
    dataset_config: str | None,
    split: str,
    max_samples: int,
    sampling_rate: int,
) -> tuple[list[EvalSample], str | None]:
    load_kwargs: dict[str, Any] = {"split": split}
    if dataset_config:
        load_kwargs["name"] = dataset_config

    dataset = load_dataset(dataset_id, **load_kwargs)
    if not hasattr(dataset, "column_names"):
        raise RuntimeError("dataset split did not resolve to a tabular dataset")

    columns = list(getattr(dataset, "column_names"))
    if "audio" not in columns:
        raise RuntimeError(
            f"dataset '{dataset_id}' split '{split}' has no 'audio' column"
        )

    row_count = len(dataset)
    if row_count <= 0:
        raise RuntimeError(f"dataset '{dataset_id}' split '{split}' is empty")

    limited_count = min(max_samples, row_count)
    if limited_count < row_count:
        dataset = dataset.select(range(limited_count))

    dataset = dataset.cast_column("audio", Audio(sampling_rate=int(sampling_rate)))
    reference_key = _resolve_reference_key(columns)

    samples: list[EvalSample] = []
    for row in dataset:
        audio = row.get("audio") if isinstance(row, dict) else None
        if not isinstance(audio, dict):
            continue
        array = audio.get("array")
        sample_rate = int(audio.get("sampling_rate", sampling_rate) or sampling_rate)
        if array is None or sample_rate <= 0:
            continue
        duration_seconds = float(len(array)) / float(sample_rate)
        if duration_seconds <= 0.0:
            continue

        reference = ""
        if reference_key:
            reference = _normalize_text(row.get(reference_key, ""))

        samples.append(
            EvalSample(
                audio={"array": array, "sampling_rate": sample_rate},
                reference=reference,
                duration_seconds=duration_seconds,
            )
        )

    if not samples:
        raise RuntimeError("no valid audio rows were loaded from the dataset")
    return samples, reference_key


def _build_asr_pipeline(model_id: str, device: str):
    processor = AutoProcessor.from_pretrained(model_id)
    ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        export=True,
        compile=False,
        device=device,
    )
    ov_model.compile()
    return pipeline(
        task="automatic-speech-recognition",
        model=ov_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )


def _generation_kwargs(language: str | None) -> dict[str, str]:
    clean = str(language or "").strip()
    if not clean:
        return {"task": "transcribe"}
    return {"task": "transcribe", "language": clean}


def _benchmark_model_device(
    *,
    model_id: str,
    device: str,
    samples: list[EvalSample],
    language: str | None,
) -> BenchmarkResult:
    load_started = time.perf_counter()
    try:
        recognizer = _build_asr_pipeline(model_id=model_id, device=device)
    except Exception as exc:
        return BenchmarkResult(
            model_id=model_id,
            device=device,
            status="error",
            detail=f"load_failed:{exc.__class__.__name__}:{exc}",
            sample_count=0,
            model_load_seconds=round(time.perf_counter() - load_started, 6),
            inference_seconds=0.0,
            audio_seconds=0.0,
            mean_latency_ms=0.0,
            p95_latency_ms=0.0,
            samples_per_second=0.0,
            real_time_factor=0.0,
            x_realtime=0.0,
            wer=None,
        )

    model_load_seconds = time.perf_counter() - load_started
    kwargs = _generation_kwargs(language)
    latencies_ms: list[float] = []
    references: list[str] = []
    predictions: list[str] = []

    try:
        for sample in samples:
            started = time.perf_counter()
            result = recognizer(sample.audio, generate_kwargs=kwargs)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)

            if isinstance(result, dict):
                text = result.get("text", "")
            else:
                text = str(result)
            predictions.append(_normalize_text(text))
            references.append(sample.reference)
    except Exception as exc:
        return BenchmarkResult(
            model_id=model_id,
            device=device,
            status="error",
            detail=f"inference_failed:{exc.__class__.__name__}:{exc}",
            sample_count=len(latencies_ms),
            model_load_seconds=round(model_load_seconds, 6),
            inference_seconds=round(sum(latencies_ms) / 1000.0, 6),
            audio_seconds=round(
                sum(sample.duration_seconds for sample in samples[: len(latencies_ms)]),
                6,
            ),
            mean_latency_ms=round(statistics.fmean(latencies_ms), 6)
            if latencies_ms
            else 0.0,
            p95_latency_ms=round(_percentile(sorted(latencies_ms), 0.95), 6)
            if latencies_ms
            else 0.0,
            samples_per_second=0.0,
            real_time_factor=0.0,
            x_realtime=0.0,
            wer=None,
        )

    ordered_latencies = sorted(latencies_ms)
    sample_count = len(latencies_ms)
    inference_seconds = sum(latencies_ms) / 1000.0
    audio_seconds = sum(sample.duration_seconds for sample in samples)
    mean_latency_ms = statistics.fmean(latencies_ms) if latencies_ms else 0.0
    p95_latency_ms = _percentile(ordered_latencies, 0.95)
    samples_per_second = (
        float(sample_count) / inference_seconds if inference_seconds > 0.0 else 0.0
    )
    real_time_factor = inference_seconds / audio_seconds if audio_seconds > 0.0 else 0.0
    x_realtime = audio_seconds / inference_seconds if inference_seconds > 0.0 else 0.0

    has_references = any(reference for reference in references)
    wer_value = wer(references, predictions) if has_references else None

    return BenchmarkResult(
        model_id=model_id,
        device=device,
        status="ok",
        detail="",
        sample_count=sample_count,
        model_load_seconds=round(model_load_seconds, 6),
        inference_seconds=round(inference_seconds, 6),
        audio_seconds=round(audio_seconds, 6),
        mean_latency_ms=round(mean_latency_ms, 6),
        p95_latency_ms=round(p95_latency_ms, 6),
        samples_per_second=round(samples_per_second, 6),
        real_time_factor=round(real_time_factor, 6),
        x_realtime=round(x_realtime, 6),
        wer=round(float(wer_value), 6) if wer_value is not None else None,
    )


def _print_results(results: list[BenchmarkResult]) -> None:
    print(
        f"{'model':34} {'device':6} {'status':8} {'n':>4} {'load_s':>8} "
        f"{'infer_s':>8} {'x_rt':>8} {'wer':>8}"
    )
    print("-" * 92)
    for row in results:
        short_model = row.model_id[:34]
        wer_text = f"{row.wer:.4f}" if row.wer is not None else "-"
        print(
            f"{short_model:34} {row.device:6} {row.status:8} {row.sample_count:4d} "
            f"{row.model_load_seconds:8.3f} {row.inference_seconds:8.3f} "
            f"{row.x_realtime:8.3f} {wer_text:>8}"
        )
        if row.detail:
            print(f"  detail: {row.detail[:220]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Whisper model sizes across OpenVINO devices "
            "(NPU/GPU/CPU) using a Hugging Face dataset"
        )
    )
    parser.add_argument(
        "--models",
        default=os.getenv("WHISPER_BENCH_MODELS", "tiny,base,small"),
        help=(
            "Comma-separated model aliases or HF model ids. Default: tiny,base,small"
        ),
    )
    parser.add_argument(
        "--devices",
        default=os.getenv("WHISPER_BENCH_DEVICES", "NPU,GPU,CPU"),
        help="Comma-separated OpenVINO devices to benchmark. Default: NPU,GPU,CPU",
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv(
            "WHISPER_BENCH_DATASET", "hf-internal-testing/librispeech_asr_dummy"
        ),
        help="Hugging Face dataset id with an audio column",
    )
    parser.add_argument(
        "--dataset-config",
        default=os.getenv("WHISPER_BENCH_DATASET_CONFIG", "clean"),
        help="Optional Hugging Face dataset config name (empty string disables)",
    )
    parser.add_argument(
        "--split",
        default=os.getenv("WHISPER_BENCH_SPLIT", "validation"),
        help="Dataset split. Default: validation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=_safe_int(os.getenv("WHISPER_BENCH_MAX_SAMPLES", "32"), 32, 1),
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=_safe_int(os.getenv("WHISPER_BENCH_SAMPLING_RATE", "16000"), 16000, 1),
        help="Audio sampling rate for dataset casting. Default: 16000",
    )
    parser.add_argument(
        "--language",
        default=os.getenv("WHISPER_BENCH_LANGUAGE", "en"),
        help="Language token passed to Whisper generation (empty string disables)",
    )
    parser.add_argument(
        "--output",
        default=os.getenv(
            "WHISPER_BENCH_OUTPUT", "/results/whisper-benchmark.latest.json"
        ),
        help="Output JSON path",
    )
    parser.add_argument(
        "--fail-on-missing-device",
        action="store_true",
        default=_env_flag("WHISPER_BENCH_FAIL_ON_MISSING_DEVICE", False),
        help="Return non-zero if any requested device is unavailable",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        default=_env_flag("WHISPER_BENCH_FAIL_ON_ERROR", False),
        help="Return non-zero if any benchmark row errors",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    model_ids = _resolve_model_ids(_split_csv(args.models))
    if not model_ids:
        raise RuntimeError("no models resolved from --models")

    requested_devices = [device.upper() for device in _split_csv(args.devices)]
    if not requested_devices:
        raise RuntimeError("no devices resolved from --devices")

    dataset_config = str(args.dataset_config or "").strip() or None
    language = str(args.language or "").strip() or None
    sampling_rate = _safe_int(args.sampling_rate, 16000, 1)
    max_samples = _safe_int(args.max_samples, 32, 1)

    openvino_devices = _probe_openvino_devices()
    print(
        "openvino devices: "
        + (", ".join(openvino_devices) if openvino_devices else "(none detected)")
    )
    print(
        f"dataset: {args.dataset}"
        + (f"/{dataset_config}" if dataset_config else "")
        + f" split={args.split} max_samples={max_samples}"
    )

    samples, reference_key = _load_eval_samples(
        dataset_id=str(args.dataset).strip(),
        dataset_config=dataset_config,
        split=str(args.split).strip(),
        max_samples=max_samples,
        sampling_rate=sampling_rate,
    )
    print(
        f"loaded samples: {len(samples)}"
        + (f" reference_column={reference_key}" if reference_key else "")
    )

    results: list[BenchmarkResult] = []
    for device in requested_devices:
        available = _is_device_available(device, openvino_devices)
        if not available:
            reason = f"device_unavailable:{device};available={','.join(openvino_devices) or 'none'}"
            for model_id in model_ids:
                results.append(
                    BenchmarkResult(
                        model_id=model_id,
                        device=device,
                        status="skipped",
                        detail=reason,
                        sample_count=0,
                        model_load_seconds=0.0,
                        inference_seconds=0.0,
                        audio_seconds=0.0,
                        mean_latency_ms=0.0,
                        p95_latency_ms=0.0,
                        samples_per_second=0.0,
                        real_time_factor=0.0,
                        x_realtime=0.0,
                        wer=None,
                    )
                )
            continue

        for model_id in model_ids:
            print(f"benchmarking model={model_id} device={device} ...")
            result = _benchmark_model_device(
                model_id=model_id,
                device=device,
                samples=samples,
                language=language,
            )
            results.append(result)

    _print_results(results)

    output_path = Path(str(args.output)).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "dataset": {
            "id": str(args.dataset).strip(),
            "config": dataset_config,
            "split": str(args.split).strip(),
            "max_samples": max_samples,
            "sampling_rate": sampling_rate,
            "reference_column": reference_key,
        },
        "request": {
            "models": model_ids,
            "devices": requested_devices,
            "language": language,
            "fail_on_missing_device": bool(args.fail_on_missing_device),
            "fail_on_error": bool(args.fail_on_error),
        },
        "runtime": {
            "openvino_devices": openvino_devices,
        },
        "results": [asdict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote benchmark report: {output_path}")

    ok_count = sum(1 for row in results if row.status == "ok")
    error_count = sum(1 for row in results if row.status == "error")
    skipped_count = sum(1 for row in results if row.status == "skipped")

    if ok_count == 0:
        return 1
    if bool(args.fail_on_error) and error_count > 0:
        return 2
    if bool(args.fail_on_missing_device) and skipped_count > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
