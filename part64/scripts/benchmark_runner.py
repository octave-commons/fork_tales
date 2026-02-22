#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DEFAULT_SUITE_PATH = "scripts/benchmark_suites/universal_starter.json"
DEFAULT_OUTPUT_PATH = "runs/model-bench/model-bench.latest.json"

REFERENCE_COLUMN_ALIASES: tuple[str, ...] = (
    "text",
    "sentence",
    "transcript",
    "transcription",
    "normalized_text",
)


@dataclass(frozen=True)
class DatasetSpec:
    source: str
    dataset_id: str
    dataset_config: str | None
    split: str
    path: str | None
    input_column: str
    reference_column: str | None
    max_samples: int
    sampling_rate: int


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    runner: str
    model: str
    device: str
    params: dict[str, Any]
    dataset_override: dict[str, Any]


@dataclass(frozen=True)
class CriterionSpec:
    criterion_id: str
    goal: str
    weight: float
    metric: str | None
    expression: str | None
    target: float | None
    runners: tuple[str, ...]


@dataclass
class CaseResult:
    case_id: str
    runner: str
    model: str
    device: str
    status: str
    detail: str
    dataset: dict[str, Any]
    metrics: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        parsed = int(default)
    if minimum is not None:
        parsed = max(int(minimum), parsed)
    return parsed


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * max(0.0, min(1.0, float(p)))
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(values[lower])
    weight = rank - lower
    return float(values[lower] + (values[upper] - values[lower]) * weight)


def _resolve_reference_column(
    *, requested: str | None, available_columns: list[str]
) -> str | None:
    if requested and requested in available_columns:
        return requested
    for key in REFERENCE_COLUMN_ALIASES:
        if key in available_columns:
            return key
    return None


def _normalize_dataset_spec(raw: dict[str, Any] | None) -> DatasetSpec:
    row = raw if isinstance(raw, dict) else {}
    source = str(row.get("source", "hf")).strip().lower() or "hf"
    dataset_id = str(row.get("id", row.get("dataset", ""))).strip()
    dataset_config_text = str(row.get("config", row.get("name", ""))).strip()
    dataset_config = dataset_config_text or None
    split = str(row.get("split", "validation")).strip() or "validation"
    path_text = str(row.get("path", "")).strip()
    path = path_text or None
    input_column = str(row.get("input_column", "text")).strip() or "text"
    reference_column_text = str(row.get("reference_column", "")).strip()
    reference_column = reference_column_text or None
    max_samples = _safe_int(row.get("max_samples", 32), 32, 1)
    sampling_rate = _safe_int(row.get("sampling_rate", 16000), 16000, 1)
    return DatasetSpec(
        source=source,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        split=split,
        path=path,
        input_column=input_column,
        reference_column=reference_column,
        max_samples=max_samples,
        sampling_rate=sampling_rate,
    )


def _merge_dataset_spec(base: DatasetSpec, override: dict[str, Any]) -> DatasetSpec:
    payload = {
        "source": base.source,
        "id": base.dataset_id,
        "config": base.dataset_config,
        "split": base.split,
        "path": base.path,
        "input_column": base.input_column,
        "reference_column": base.reference_column,
        "max_samples": base.max_samples,
        "sampling_rate": base.sampling_rate,
    }
    for key, value in (override or {}).items():
        payload[key] = value
    return _normalize_dataset_spec(payload)


def _load_local_rows(
    spec: DatasetSpec,
    *,
    suite_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not spec.path:
        raise RuntimeError("local dataset source requires dataset.path")
    path = Path(spec.path).expanduser()
    if not path.is_absolute():
        path = (suite_root / path).resolve()
    if not path.exists():
        raise RuntimeError(f"local dataset path not found: {path}")

    rows_raw: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            item = json.loads(text)
            if isinstance(item, dict):
                rows_raw.append(item)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows_raw = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict):
            bucket = payload.get("rows")
            if isinstance(bucket, list):
                rows_raw = [row for row in bucket if isinstance(row, dict)]

    if not rows_raw:
        raise RuntimeError(f"local dataset has no rows: {path}")

    rows: list[dict[str, Any]] = []
    for row in rows_raw[: spec.max_samples]:
        if spec.input_column not in row:
            continue
        rows.append(
            {
                "input": row.get(spec.input_column),
                "reference": (
                    row.get(spec.reference_column, "") if spec.reference_column else ""
                ),
                "raw": row,
            }
        )

    if not rows:
        raise RuntimeError(
            f"local dataset has no usable rows for input column '{spec.input_column}'"
        )

    metadata = {
        "source": spec.source,
        "path": str(path),
        "split": "local",
        "input_column": spec.input_column,
        "reference_column": spec.reference_column,
        "loaded_rows": len(rows),
    }
    return rows, metadata


def _load_hf_rows(spec: DatasetSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not spec.dataset_id:
        raise RuntimeError("hf dataset source requires dataset.id")

    from datasets import Audio, load_dataset

    kwargs: dict[str, Any] = {"split": spec.split}
    if spec.dataset_config:
        kwargs["name"] = spec.dataset_config

    dataset = load_dataset(spec.dataset_id, **kwargs)
    if not hasattr(dataset, "column_names"):
        raise RuntimeError("dataset split did not resolve to a tabular dataset")

    columns = list(getattr(dataset, "column_names"))
    if spec.input_column not in columns:
        raise RuntimeError(
            f"dataset '{spec.dataset_id}' missing input column '{spec.input_column}'"
        )

    reference_column = _resolve_reference_column(
        requested=spec.reference_column,
        available_columns=columns,
    )

    row_count = len(dataset)
    if row_count <= 0:
        raise RuntimeError(
            f"dataset '{spec.dataset_id}' split '{spec.split}' returned no rows"
        )

    limit = min(spec.max_samples, row_count)
    if limit < row_count:
        dataset = dataset.select(range(limit))

    if spec.input_column == "audio":
        dataset = dataset.cast_column("audio", Audio(sampling_rate=spec.sampling_rate))

    rows: list[dict[str, Any]] = []
    for row in dataset:
        if not isinstance(row, dict):
            continue
        if spec.input_column not in row:
            continue
        rows.append(
            {
                "input": row.get(spec.input_column),
                "reference": row.get(reference_column, "") if reference_column else "",
                "raw": row,
            }
        )

    if not rows:
        raise RuntimeError(
            f"dataset '{spec.dataset_id}' had no usable rows for column '{spec.input_column}'"
        )

    metadata = {
        "source": "hf",
        "id": spec.dataset_id,
        "config": spec.dataset_config,
        "split": spec.split,
        "input_column": spec.input_column,
        "reference_column": reference_column,
        "loaded_rows": len(rows),
        "sampling_rate": spec.sampling_rate,
    }
    return rows, metadata


def _dataset_cache_key(spec: DatasetSpec) -> str:
    return json.dumps(asdict(spec), sort_keys=True)


def _load_dataset_rows(
    spec: DatasetSpec,
    *,
    suite_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if spec.source in {"json", "jsonl", "local", "local_json", "local_jsonl"}:
        return _load_local_rows(spec, suite_root=suite_root)
    if spec.source == "hf":
        return _load_hf_rows(spec)
    raise RuntimeError(f"unsupported dataset source '{spec.source}'")


def _embedding_benchmark(
    *, case: BenchmarkCase, rows: list[dict[str, Any]], dataset_meta: dict[str, Any]
) -> CaseResult:
    try:
        from code.world_web.ai import (  # type: ignore
            _embed_text,
            _eta_mu_normalize_vector,
            _eta_mu_resize_vector,
        )
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id,
            runner=case.runner,
            model=case.model,
            device=case.device,
            status="error",
            detail=f"import_failed:{exc.__class__.__name__}:{exc}",
            dataset=dataset_meta,
            metrics={},
        )

    backend = str(case.params.get("backend", "openvino")).strip().lower() or "openvino"
    target_dim = _safe_int(case.params.get("target_dim", 0), 0, 0)
    max_chars = _safe_int(case.params.get("max_chars", 0), 0, 0)

    env_overrides: dict[str, str] = {
        "EMBEDDINGS_BACKEND": backend,
    }
    if case.device:
        env_overrides["OPENVINO_EMBED_DEVICE"] = str(case.device).upper()

    old_values: dict[str, str | None] = {}
    for key, value in env_overrides.items():
        old_values[key] = os.getenv(key)
        os.environ[key] = value

    latencies_ms: list[float] = []
    vector_dims: list[int] = []
    success_count = 0
    failure_count = 0
    errors: dict[str, int] = {}

    for row in rows:
        sample = str(row.get("input", ""))
        if not sample:
            continue
        text = sample[:max_chars] if max_chars > 0 else sample
        started = time.perf_counter()
        try:
            vector = _embed_text(text, model=case.model or None)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            if not isinstance(vector, list) or not vector:
                failure_count += 1
                errors["embedding_none"] = errors.get("embedding_none", 0) + 1
                continue

            resized = [float(value) for value in vector]
            if target_dim > 0:
                resized = _eta_mu_resize_vector(resized, target_dim)
                resized = _eta_mu_normalize_vector(resized)
            vector_dims.append(len(resized))
            success_count += 1
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            failure_count += 1
            key = f"{exc.__class__.__name__}"
            errors[key] = errors.get(key, 0) + 1

    for key, value in old_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    if success_count <= 0:
        detail = (
            ",".join(f"{key}:{count}" for key, count in sorted(errors.items()))
            or "no_successful_embeddings"
        )
        return CaseResult(
            case_id=case.case_id,
            runner=case.runner,
            model=case.model,
            device=case.device,
            status="error",
            detail=detail,
            dataset=dataset_meta,
            metrics={
                "sample_count": len(rows),
                "success_count": success_count,
                "failure_count": failure_count,
            },
        )

    ordered = sorted(latencies_ms)
    total_seconds = sum(latencies_ms) / 1000.0
    metrics = {
        "sample_count": len(rows),
        "success_count": success_count,
        "failure_count": failure_count,
        "mean_latency_ms": statistics.fmean(latencies_ms),
        "p95_latency_ms": _percentile(ordered, 0.95),
        "max_latency_ms": max(latencies_ms),
        "throughput_items_per_second": (
            float(success_count) / total_seconds if total_seconds > 0.0 else 0.0
        ),
        "vector_dim_mean": statistics.fmean(vector_dims) if vector_dims else 0.0,
        "vector_dim_min": min(vector_dims) if vector_dims else 0,
        "vector_dim_max": max(vector_dims) if vector_dims else 0,
    }
    return CaseResult(
        case_id=case.case_id,
        runner=case.runner,
        model=case.model,
        device=case.device,
        status="ok",
        detail="",
        dataset=dataset_meta,
        metrics=metrics,
    )


def _whisper_benchmark(
    *, case: BenchmarkCase, rows: list[dict[str, Any]], dataset_meta: dict[str, Any]
) -> CaseResult:
    try:
        from jiwer import wer
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq
        from transformers import AutoProcessor, pipeline
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id,
            runner=case.runner,
            model=case.model,
            device=case.device,
            status="error",
            detail=f"import_failed:{exc.__class__.__name__}:{exc}",
            dataset=dataset_meta,
            metrics={},
        )

    model_id = str(case.model).strip()
    if not model_id:
        return CaseResult(
            case_id=case.case_id,
            runner=case.runner,
            model=case.model,
            device=case.device,
            status="error",
            detail="model_id_missing",
            dataset=dataset_meta,
            metrics={},
        )

    language = str(case.params.get("language", "")).strip()
    task = str(case.params.get("task", "transcribe")).strip() or "transcribe"
    chunk_length_s = _safe_float(case.params.get("chunk_length_s", 0.0), 0.0)
    batch_size = _safe_int(case.params.get("batch_size", 1), 1, 1)

    load_started = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            export=True,
            compile=False,
            device=str(case.device or "CPU"),
        )
        ov_model.compile()
        recognizer = pipeline(
            task="automatic-speech-recognition",
            model=ov_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            batch_size=batch_size,
        )
    except Exception as exc:
        return CaseResult(
            case_id=case.case_id,
            runner=case.runner,
            model=case.model,
            device=case.device,
            status="error",
            detail=f"load_failed:{exc.__class__.__name__}:{exc}",
            dataset=dataset_meta,
            metrics={"sample_count": 0},
        )
    model_load_seconds = time.perf_counter() - load_started

    generate_kwargs: dict[str, Any] = {"task": task}
    if language:
        generate_kwargs["language"] = language

    latencies_ms: list[float] = []
    references: list[str] = []
    predictions: list[str] = []
    failure_count = 0
    audio_seconds = 0.0
    processed_rows = 0
    errors: dict[str, int] = {}

    pipeline_kwargs: dict[str, Any] = {"generate_kwargs": generate_kwargs}
    if chunk_length_s > 0.0:
        pipeline_kwargs["chunk_length_s"] = chunk_length_s

    for row in rows:
        audio = row.get("input")
        started = time.perf_counter()
        try:
            output = recognizer(audio, **pipeline_kwargs)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            processed_rows += 1

            if isinstance(audio, dict):
                array = audio.get("array")
                rate = _safe_int(audio.get("sampling_rate", 0), 0, 0)
                if array is not None and rate > 0:
                    audio_seconds += float(len(array)) / float(rate)

            text = output.get("text", "") if isinstance(output, dict) else str(output)
            predictions.append(_normalize_text(text))
            references.append(_normalize_text(row.get("reference", "")))
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            failure_count += 1
            key = exc.__class__.__name__
            errors[key] = errors.get(key, 0) + 1

    success_count = max(0, processed_rows)
    if success_count <= 0:
        detail = (
            ",".join(f"{key}:{count}" for key, count in sorted(errors.items()))
            or "no_successful_transcripts"
        )
        return CaseResult(
            case_id=case.case_id,
            runner=case.runner,
            model=case.model,
            device=case.device,
            status="error",
            detail=detail,
            dataset=dataset_meta,
            metrics={
                "sample_count": len(rows),
                "success_count": 0,
                "failure_count": failure_count,
                "model_load_seconds": model_load_seconds,
            },
        )

    ordered = sorted(latencies_ms)
    inference_seconds = sum(latencies_ms) / 1000.0
    has_reference = any(text for text in references)
    wer_value = wer(references, predictions) if has_reference else None

    metrics: dict[str, Any] = {
        "sample_count": len(rows),
        "success_count": success_count,
        "failure_count": failure_count,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "audio_seconds": audio_seconds,
        "mean_latency_ms": statistics.fmean(latencies_ms),
        "p95_latency_ms": _percentile(ordered, 0.95),
        "max_latency_ms": max(latencies_ms),
        "throughput_items_per_second": (
            float(success_count) / inference_seconds if inference_seconds > 0.0 else 0.0
        ),
        "real_time_factor": (
            inference_seconds / audio_seconds if audio_seconds > 0.0 else 0.0
        ),
        "x_realtime": (
            audio_seconds / inference_seconds if inference_seconds > 0.0 else 0.0
        ),
    }
    if wer_value is not None:
        metrics["wer"] = float(wer_value)

    return CaseResult(
        case_id=case.case_id,
        runner=case.runner,
        model=case.model,
        device=case.device,
        status="ok",
        detail="",
        dataset=dataset_meta,
        metrics=metrics,
    )


RUNNER_MAP: dict[str, Callable[..., CaseResult]] = {
    "embedding_text": _embedding_benchmark,
    "whisper_asr": _whisper_benchmark,
}


def _parse_cases(payload: dict[str, Any]) -> list[BenchmarkCase]:
    rows = payload.get("benchmarks")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("suite requires a non-empty 'benchmarks' list")

    cases: list[BenchmarkCase] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("id", f"case_{index + 1:03d}")).strip()
        runner = str(row.get("runner", "")).strip().lower()
        model = str(row.get("model", "")).strip()
        device = str(row.get("device", "")).strip().upper()
        params = row.get("params") if isinstance(row.get("params"), dict) else {}
        dataset_override = (
            row.get("dataset") if isinstance(row.get("dataset"), dict) else {}
        )
        if not case_id or not runner:
            continue
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                runner=runner,
                model=model,
                device=device,
                params=dict(params),
                dataset_override=dict(dataset_override),
            )
        )

    if not cases:
        raise RuntimeError("suite has no valid benchmark cases")
    return cases


def _parse_criteria(payload: dict[str, Any]) -> list[CriterionSpec]:
    rows = payload.get("criteria")
    if not isinstance(rows, list):
        return []

    criteria: list[CriterionSpec] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        criterion_id = str(row.get("id", f"criterion_{index + 1:03d}")).strip()
        goal = str(row.get("goal", "min")).strip().lower() or "min"
        if goal not in {"min", "max", "target"}:
            goal = "min"
        weight = _safe_float(row.get("weight", 1.0), 1.0)
        metric = str(row.get("metric", "")).strip() or None
        expression = str(row.get("expression", "")).strip() or None
        target = (
            _safe_float(row.get("target"), 0.0)
            if goal == "target" and row.get("target") is not None
            else None
        )

        runners_raw = row.get("runners")
        runners: list[str] = []
        if isinstance(runners_raw, list):
            runners = [
                str(item).strip().lower() for item in runners_raw if str(item).strip()
            ]
        elif isinstance(row.get("runner"), str):
            runners = [str(row.get("runner")).strip().lower()]

        if not criterion_id or weight <= 0.0:
            continue
        if metric is None and expression is None:
            continue

        criteria.append(
            CriterionSpec(
                criterion_id=criterion_id,
                goal=goal,
                weight=weight,
                metric=metric,
                expression=expression,
                target=target,
                runners=tuple(runners),
            )
        )
    return criteria


def _criterion_applies(criterion: CriterionSpec, runner: str) -> bool:
    if not criterion.runners:
        return True
    return runner.lower() in criterion.runners


def _safe_eval_expression(expression: str, variables: dict[str, Any]) -> float | None:
    allowed_functions: dict[str, Callable[..., Any]] = {
        "abs": abs,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "log": math.log,
    }

    allowed_binops: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
        ast.Mod: lambda a, b: a % b,
    }
    allowed_unary: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("non-numeric constant")
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"unknown variable {node.id}")
            value = _safe_float(variables[node.id], float("nan"))
            if not math.isfinite(value):
                raise ValueError(f"non-finite variable {node.id}")
            return value
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_binops:
                raise ValueError("operator not allowed")
            left = _eval(node.left)
            right = _eval(node.right)
            return float(allowed_binops[op_type](left, right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_unary:
                raise ValueError("unary operator not allowed")
            return float(allowed_unary[op_type](_eval(node.operand)))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("call target not allowed")
            fn = allowed_functions.get(node.func.id)
            if fn is None:
                raise ValueError("function not allowed")
            args = [_eval(arg) for arg in node.args]
            return float(fn(*args))
        raise ValueError(f"unsupported syntax node {type(node).__name__}")

    try:
        result = float(_eval(tree))
    except Exception:
        return None
    if not math.isfinite(result):
        return None
    return result


def _criterion_value(criterion: CriterionSpec, result: CaseResult) -> float | None:
    if not _criterion_applies(criterion, result.runner):
        return None
    if result.status != "ok":
        return None

    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    if criterion.metric is not None:
        raw = metrics.get(criterion.metric)
        if raw is None:
            return None
        value = _safe_float(raw, float("nan"))
        if not math.isfinite(value):
            return None
        return value
    if criterion.expression is not None:
        return _safe_eval_expression(criterion.expression, metrics)
    return None


def _normalize_criterion_score(
    *, criterion: CriterionSpec, value: float, values: list[float]
) -> float:
    if not values:
        return 0.0
    low = min(values)
    high = max(values)
    span = high - low

    if criterion.goal == "max":
        if span <= 1e-12:
            return 1.0
        return max(0.0, min(1.0, (value - low) / span))
    if criterion.goal == "target":
        if criterion.target is None:
            return 0.0
        distances = [abs(v - criterion.target) for v in values]
        max_distance = max(distances) if distances else 0.0
        if max_distance <= 1e-12:
            return 1.0
        return max(0.0, min(1.0, 1.0 - (abs(value - criterion.target) / max_distance)))

    if span <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, (high - value) / span))


def _score_results(
    criteria: list[CriterionSpec],
    results: list[CaseResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    values_by_criterion: dict[str, list[float]] = {
        criterion.criterion_id: [] for criterion in criteria
    }
    value_lookup: dict[tuple[int, str], float] = {}

    for index, result in enumerate(results):
        for criterion in criteria:
            value = _criterion_value(criterion, result)
            if value is None:
                continue
            value_lookup[(index, criterion.criterion_id)] = value
            values_by_criterion[criterion.criterion_id].append(value)

    output_rows: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        criterion_rows: list[dict[str, Any]] = []
        weighted_sum = 0.0
        weight_total = 0.0

        for criterion in criteria:
            key = (index, criterion.criterion_id)
            value = value_lookup.get(key)
            if value is None:
                continue
            normalized = _normalize_criterion_score(
                criterion=criterion,
                value=value,
                values=values_by_criterion.get(criterion.criterion_id, []),
            )
            weighted_sum += normalized * criterion.weight
            weight_total += criterion.weight
            criterion_rows.append(
                {
                    "id": criterion.criterion_id,
                    "goal": criterion.goal,
                    "weight": criterion.weight,
                    "value": value,
                    "normalized": normalized,
                }
            )

        total_score = (weighted_sum / weight_total) if weight_total > 0.0 else None
        output_rows.append(
            {
                "case_id": result.case_id,
                "runner": result.runner,
                "model": result.model,
                "device": result.device,
                "status": result.status,
                "detail": result.detail,
                "dataset": result.dataset,
                "metrics": result.metrics,
                "score": {
                    "total": total_score,
                    "weight_total": weight_total,
                    "criteria": criterion_rows,
                },
            }
        )

    ranked_ok = [
        row
        for row in output_rows
        if row.get("status") == "ok"
        and isinstance((row.get("score") or {}).get("total"), (int, float))
    ]
    ranked_ok.sort(
        key=lambda row: (
            -float((row.get("score") or {}).get("total", 0.0)),
            float((row.get("metrics") or {}).get("mean_latency_ms", 1e18)),
        )
    )
    for rank, row in enumerate(ranked_ok, start=1):
        row["rank"] = rank

    leaderboard = [
        {
            "rank": row.get("rank"),
            "case_id": row.get("case_id"),
            "runner": row.get("runner"),
            "model": row.get("model"),
            "device": row.get("device"),
            "score": (row.get("score") or {}).get("total"),
            "mean_latency_ms": (row.get("metrics") or {}).get("mean_latency_ms"),
            "p95_latency_ms": (row.get("metrics") or {}).get("p95_latency_ms"),
            "throughput_items_per_second": (
                (row.get("metrics") or {}).get("throughput_items_per_second")
            ),
        }
        for row in ranked_ok
    ]
    return output_rows, leaderboard


def _parse_suite(
    path: Path,
) -> tuple[dict[str, Any], DatasetSpec, list[BenchmarkCase], list[CriterionSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("suite file must be a JSON object")

    default_dataset = _normalize_dataset_spec(
        payload.get("dataset") if isinstance(payload.get("dataset"), dict) else None
    )
    cases = _parse_cases(payload)
    criteria = _parse_criteria(payload)
    return payload, default_dataset, cases, criteria


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(
        f"{'case_id':24} {'runner':14} {'status':8} {'score':>8} "
        f"{'mean_ms':>10} {'p95_ms':>10} {'throughput':>11}"
    )
    print("-" * 96)
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        score_block = row.get("score") if isinstance(row.get("score"), dict) else {}
        score_value = score_block.get("total")
        score_text = (
            f"{float(score_value):.4f}"
            if isinstance(score_value, (int, float))
            else "-"
        )
        print(
            f"{str(row.get('case_id', ''))[:24]:24} "
            f"{str(row.get('runner', ''))[:14]:14} "
            f"{str(row.get('status', ''))[:8]:8} "
            f"{score_text:>8} "
            f"{_safe_float(metrics.get('mean_latency_ms', 0.0), 0.0):10.3f} "
            f"{_safe_float(metrics.get('p95_latency_ms', 0.0), 0.0):10.3f} "
            f"{_safe_float(metrics.get('throughput_items_per_second', 0.0), 0.0):11.3f}"
        )
        detail = str(row.get("detail", "")).strip()
        if detail:
            print(f"  detail: {detail[:240]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Comprehensive model benchmark runner: "
            "run pluggable model benchmarks on arbitrary datasets with weighted criteria"
        )
    )
    parser.add_argument(
        "--suite",
        default=os.getenv("MODEL_BENCH_SUITE", DEFAULT_SUITE_PATH),
        help="Path to suite JSON file",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("MODEL_BENCH_OUTPUT", DEFAULT_OUTPUT_PATH),
        help="Path to JSON output artifact",
    )
    parser.add_argument(
        "--continue-on-error",
        default=os.getenv("MODEL_BENCH_CONTINUE_ON_ERROR", "1"),
        help="Continue after per-case errors (1/0). Default: 1",
    )
    parser.add_argument(
        "--list-runners",
        action="store_true",
        help="Print available runner ids and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if bool(args.list_runners):
        for key in sorted(RUNNER_MAP):
            print(key)
        return 0

    suite_path = Path(str(args.suite)).expanduser()
    if not suite_path.is_absolute():
        suite_path = Path.cwd() / suite_path
    if not suite_path.exists():
        raise RuntimeError(f"suite file not found: {suite_path}")

    continue_on_error = _as_bool(args.continue_on_error, True)

    suite_payload, default_dataset, cases, criteria = _parse_suite(suite_path)

    dataset_cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    case_results: list[CaseResult] = []

    for case in cases:
        case_dataset = _merge_dataset_spec(default_dataset, case.dataset_override)
        cache_key = _dataset_cache_key(case_dataset)
        if cache_key not in dataset_cache:
            rows, dataset_meta = _load_dataset_rows(case_dataset)
            dataset_cache[cache_key] = (rows, dataset_meta)
        rows, dataset_meta = dataset_cache[cache_key]

        runner = RUNNER_MAP.get(case.runner)
        if runner is None:
            result = CaseResult(
                case_id=case.case_id,
                runner=case.runner,
                model=case.model,
                device=case.device,
                status="error",
                detail=f"runner_not_supported:{case.runner}",
                dataset=dataset_meta,
                metrics={},
            )
            case_results.append(result)
            if not continue_on_error:
                break
            continue

        print(
            f"running case={case.case_id} runner={case.runner} "
            f"model={case.model or '(none)'} device={case.device or '(default)'}"
        )

        started = time.perf_counter()
        try:
            result = runner(case=case, rows=rows, dataset_meta=dataset_meta)
        except Exception as exc:
            result = CaseResult(
                case_id=case.case_id,
                runner=case.runner,
                model=case.model,
                device=case.device,
                status="error",
                detail=f"runner_exception:{exc.__class__.__name__}:{exc}",
                dataset=dataset_meta,
                metrics={},
            )
        elapsed = time.perf_counter() - started

        if not isinstance(result.metrics, dict):
            result.metrics = {}
        result.metrics.setdefault("case_wall_seconds", elapsed)
        case_results.append(result)

        if result.status != "ok" and not continue_on_error:
            break

    scored_rows, leaderboard = _score_results(criteria, case_results)
    _print_summary(scored_rows)

    output_path = Path(str(args.output)).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": _utc_now_iso(),
        "suite_path": str(suite_path),
        "suite": {
            "id": suite_payload.get("id", ""),
            "description": suite_payload.get("description", ""),
        },
        "dataset_default": asdict(default_dataset),
        "criteria": [asdict(item) for item in criteria],
        "results": scored_rows,
        "leaderboard": leaderboard,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote benchmark report: {output_path}")

    ok_count = sum(1 for row in scored_rows if row.get("status") == "ok")
    if ok_count == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
