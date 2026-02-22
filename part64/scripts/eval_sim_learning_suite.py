#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PART_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RuntimeRef:
    label: str
    url: str


def _request_json(url: str, timeout_seconds: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout_seconds) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise RuntimeError(f"expected object payload from {url}")
    return data


def _post_json(
    url: str, body: dict[str, Any], timeout_seconds: float
) -> dict[str, Any]:
    payload_bytes = json.dumps(body).encode("utf-8")
    request = Request(
        url=url,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise RuntimeError(f"expected object payload from {url}")
        return decoded
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            decoded = json.loads(text)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
        return {"ok": False, "error": f"http_{exc.code}", "detail": text[:240]}
    except TimeoutError:
        return {"ok": False, "error": "timeout"}
    except URLError as exc:
        return {"ok": False, "error": f"url_error:{exc.reason}"}
    except Exception as exc:  # pragma: no cover - network dependent
        return {"ok": False, "error": f"request_failed:{exc.__class__.__name__}"}


def _simulation_generated_at(payload: dict[str, Any]) -> str:
    return str(payload.get("generated_at", "") or "").strip()


def _probe_simulation_payload(
    *, runtime_url: str, timeout_seconds: float, baseline_generated_at: str
) -> dict[str, Any]:
    url = f"{runtime_url}/api/simulation?perspective=hybrid&compact=1"
    wait_window = max(120.0, float(timeout_seconds) * 2.0)
    deadline = time.monotonic() + wait_window
    last_payload: dict[str, Any] | None = None
    last_error: Exception | None = None

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break

        attempt_timeout = min(max(remaining, 15.0), max(timeout_seconds, 45.0))
        try:
            payload = _request_json(url, attempt_timeout)
            last_payload = payload
            generated_at = _simulation_generated_at(payload)
            if not baseline_generated_at:
                return payload
            if generated_at and generated_at != baseline_generated_at:
                return payload
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc

        time.sleep(1.0)

    if last_payload is not None:
        return last_payload
    if last_error is not None:
        raise last_error
    raise TimeoutError("simulation_probe_timeout")


def _parse_runtime_arg(raw: str) -> RuntimeRef:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "runtime must be in label=http://host:port format"
        )
    label, url = raw.split("=", 1)
    clean_label = label.strip()
    clean_url = url.strip().rstrip("/")
    if not clean_label or not clean_url.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError(
            "runtime must be in label=http://host:port format"
        )
    return RuntimeRef(clean_label, clean_url)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] + ((values[upper] - values[lower]) * weight)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")


def _load_key_from_env_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        rows = path.read_text("utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    for row in rows:
        text = row.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("PROXY_API_KEY="):
            return text.split("=", 1)[1].strip().strip('"').strip("'")
        if text.startswith("OPENVINO_EMBED_API_KEY="):
            return text.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _resolve_openvino_api_key(proxy_env_path: str | None = None) -> str:
    direct = str(os.getenv("OPENVINO_EMBED_API_KEY", "") or "").strip()
    if direct:
        return direct
    proxy_direct = str(os.getenv("PROXY_API_KEY", "") or "").strip()
    if proxy_direct:
        return proxy_direct
    if proxy_env_path:
        key = _load_key_from_env_file(Path(proxy_env_path).resolve())
        if key:
            return key
    return _load_key_from_env_file(PART_ROOT.parent / "docker-llm-proxy" / ".env")


def _has_cpu_fallback_signal(text: str) -> bool:
    probe = str(text or "").strip().lower()
    if not probe:
        return False
    if "cpu" in probe and (
        "fallback" in probe or "fall back" in probe or "falling back" in probe
    ):
        return True
    if "ov cpu" in probe and ("fallback" in probe or "unsupported" in probe):
        return True
    if "ze_result_error_unsupported_feature" in probe and "cpu" in probe:
        return True
    if "unsupported" in probe and "npu" in probe and "cpu" in probe:
        return True
    return False


def _ensure_npu(
    runtime: RuntimeRef,
    timeout_seconds: float,
    *,
    openvino_api_key: str,
) -> dict[str, Any]:
    apply_payload: dict[str, Any] = {
        "preset": "npu_local",
        "backend": "openvino",
        "openvino_device": "NPU",
        "openvino_endpoint": "http://host.docker.internal:18000/v1/embeddings",
        "openvino_model": "nomic-embed-text",
        "openvino_timeout_sec": 12.0,
    }
    if openvino_api_key:
        apply_payload["openvino_api_key"] = openvino_api_key
        apply_payload["openvino_api_key_header"] = "X-API-Key"
        apply_payload["openvino_bearer_token"] = openvino_api_key

    apply = _post_json(
        f"{runtime.url}/api/embeddings/provider/options",
        apply_payload,
        timeout_seconds,
    )

    baseline_generated_at = ""
    try:
        baseline_payload = _request_json(
            f"{runtime.url}/api/simulation?perspective=hybrid&compact=1",
            max(8.0, min(timeout_seconds, 20.0)),
        )
        baseline_generated_at = _simulation_generated_at(baseline_payload)
    except Exception:
        baseline_generated_at = ""

    _post_json(
        f"{runtime.url}/api/presence/user/input",
        {
            "events": [
                {
                    "kind": "input",
                    "target": "simulation",
                    "message": "npu c runtime probe",
                    "embed_daimoi": True,
                }
            ]
        },
        timeout_seconds,
    )
    options = _request_json(
        f"{runtime.url}/api/embeddings/provider/options", timeout_seconds
    )
    config = (
        options.get("config", {}) if isinstance(options.get("config"), dict) else {}
    )
    c_runtime_source = ""
    c_runtime_error = ""
    c_runtime_cpu_fallback = False
    c_runtime_cpu_fallback_detail = ""
    daimoi_backend = ""
    daimoi_backend_error = ""
    try:
        simulation = _probe_simulation_payload(
            runtime_url=runtime.url,
            timeout_seconds=timeout_seconds,
            baseline_generated_at=baseline_generated_at,
        )
        presence = (
            simulation.get("presence_dynamics", {})
            if isinstance(simulation.get("presence_dynamics", {}), dict)
            else {}
        )
        probabilistic = (
            presence.get("daimoi_probabilistic", {})
            if isinstance(presence.get("daimoi_probabilistic", {}), dict)
            else {}
        )
        c_runtime_source = str(
            probabilistic.get("embedding_runtime_source", "")
        ).strip()
        c_runtime_error = str(probabilistic.get("embedding_runtime_error", "")).strip()
        c_runtime_cpu_fallback = bool(
            probabilistic.get("embedding_runtime_cpu_fallback", False)
        )
        c_runtime_cpu_fallback_detail = str(
            probabilistic.get("embedding_runtime_cpu_fallback_detail", "")
        ).strip()
        daimoi_backend = str(probabilistic.get("backend", "")).strip().lower()
        daimoi_backend_error = str(probabilistic.get("backend_error", "")).strip()
    except Exception as exc:
        c_runtime_error = f"simulation_probe_failed:{exc.__class__.__name__}"

    backend = str(config.get("backend", "")).strip().lower()
    device = str(config.get("openvino_device", "")).strip().upper()
    fallback_signaled = (
        bool(c_runtime_cpu_fallback)
        or _has_cpu_fallback_signal(c_runtime_cpu_fallback_detail)
        or _has_cpu_fallback_signal(c_runtime_error)
    )
    probe_ok = (
        c_runtime_source.startswith("c-onnxruntime:NPU")
        and not fallback_signaled
        and not c_runtime_error
        and (not daimoi_backend or daimoi_backend == "c-double-buffer")
    )
    probe_error = c_runtime_error
    if fallback_signaled and not probe_error:
        probe_error = c_runtime_cpu_fallback_detail or "cpu_fallback_detected"
    if not probe_error and daimoi_backend and daimoi_backend != "c-double-buffer":
        probe_error = daimoi_backend_error or f"daimoi_backend={daimoi_backend}"
    return {
        "label": runtime.label,
        "runtime": runtime.url,
        "ok": backend == "openvino" and device == "NPU" and probe_ok,
        "backend": backend,
        "openvino_device": device,
        "openvino_endpoint": str(config.get("openvino_endpoint", "")).strip(),
        "probe_ok": probe_ok,
        "probe_error": probe_error,
        "c_runtime_source": c_runtime_source,
        "c_runtime_error": c_runtime_error,
        "c_runtime_cpu_fallback": c_runtime_cpu_fallback,
        "c_runtime_cpu_fallback_detail": c_runtime_cpu_fallback_detail,
        "daimoi_backend": daimoi_backend,
        "daimoi_backend_error": daimoi_backend_error,
        "apply_ok": bool(apply.get("ok", False)),
    }


def _inject_environment_events(
    runtime: RuntimeRef,
    *,
    event_count: int,
    timeout_seconds: float,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    prompts = [
        "play music near the witness thread",
        "open image for stability sigil",
        "route to fork tax canticle",
        "focus the nearest archive card",
        "choose strongest semantic match",
    ]
    kinds = ["hover", "click", "input", "keydown"]
    targets = ["simulation", "chat", "catalog", "workspace"]

    events: list[dict[str, Any]] = []
    for index in range(max(1, event_count)):
        events.append(
            {
                "kind": kinds[index % len(kinds)],
                "target": targets[index % len(targets)],
                "x_ratio": round(rng.random(), 6),
                "y_ratio": round(rng.random(), 6),
                "embed_daimoi": True,
                "message": prompts[index % len(prompts)],
            }
        )

    response = _post_json(
        f"{runtime.url}/api/presence/user/input",
        {"events": events, "meta": {"origin": "eval_sim_learning_suite"}},
        timeout_seconds,
    )
    return {
        "label": runtime.label,
        "runtime": runtime.url,
        "ok": bool(response.get("ok", False)),
        "processed": int(response.get("processed", 0) or 0),
        "event_count": int(response.get("event_count", 0) or 0),
        "error": str(response.get("error", "")).strip(),
    }


def _bench_simulation_latency(
    runtime: RuntimeRef,
    *,
    request_count: int,
    warmup: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    endpoint = f"{runtime.url}/api/simulation?perspective=hybrid&compact=1"
    for _ in range(max(0, warmup)):
        try:
            _request_json(endpoint, timeout_seconds)
        except Exception:
            continue

    latencies: list[float] = []
    failures = 0
    for _ in range(max(1, request_count)):
        started = time.perf_counter()
        try:
            payload = _request_json(endpoint, timeout_seconds)
            if payload.get("ok") is False:
                failures += 1
                continue
            latencies.append((time.perf_counter() - started) * 1000.0)
        except Exception:
            failures += 1

    if not latencies:
        return {
            "label": runtime.label,
            "runtime": runtime.url,
            "ok": False,
            "samples": 0,
            "failures": failures,
            "error": "no_successful_samples",
        }

    ordered = sorted(latencies)
    return {
        "label": runtime.label,
        "runtime": runtime.url,
        "ok": True,
        "samples": len(latencies),
        "failures": failures,
        "mean_ms": statistics.fmean(latencies),
        "p50_ms": _percentile(ordered, 0.50),
        "p95_ms": _percentile(ordered, 0.95),
        "p99_ms": _percentile(ordered, 0.99),
        "max_ms": max(latencies),
    }


def _run_training(
    runtime: RuntimeRef,
    *,
    circumstances: Path,
    output_path: Path,
    rounds: int,
    timeout_seconds: float,
    seed: int,
) -> dict[str, Any]:
    command = [
        "python",
        "scripts/muse_semantic_training_lab.py",
        "--runtime",
        runtime.url,
        "--circumstances",
        str(circumstances),
        "--output",
        str(output_path),
        "--timeout",
        str(timeout_seconds),
        "--seed",
        str(seed),
    ]
    if rounds > 0:
        command.extend(["--rounds", str(rounds)])

    completed = subprocess.run(
        command,
        cwd=str(PART_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    payload: dict[str, Any] = {}
    if output_path.exists():
        try:
            parsed = json.loads(output_path.read_text("utf-8"))
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = {}

    summary = (
        payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    )
    return {
        "label": runtime.label,
        "runtime": runtime.url,
        "ok": completed.returncode == 0 and bool(payload.get("ok", False)),
        "exit_code": int(completed.returncode),
        "report": str(output_path),
        "summary": {
            "samples": int(summary.get("samples", 0) or 0),
            "rounds": int(summary.get("rounds", 0) or 0),
            "requested_rate": float(summary.get("requested_rate", 0.0) or 0.0),
            "modality_accuracy": float(summary.get("modality_accuracy", 0.0) or 0.0),
            "target_accuracy": float(summary.get("target_accuracy", 0.0) or 0.0),
            "routed_accuracy": float(summary.get("routed_accuracy", 0.0) or 0.0),
            "blocked_rate": float(summary.get("blocked_rate", 0.0) or 0.0),
            "latency_mean_ms": float(summary.get("latency_mean_ms", 0.0) or 0.0),
            "latency_p95_ms": float(summary.get("latency_p95_ms", 0.0) or 0.0),
        },
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-6:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-6:]),
    }


def _run_song_benchmark(
    runtimes: list[RuntimeRef],
    *,
    regimen: Path,
    rounds: int,
    timeout_seconds: float,
    output_path: Path,
) -> dict[str, Any]:
    command = [
        "python",
        "scripts/bench_muse_song_lab.py",
        "--regimen",
        str(regimen),
        "--timeout",
        str(timeout_seconds),
        "--json-out",
        str(output_path),
    ]
    if rounds > 0:
        command.extend(["--rounds", str(rounds)])
    for runtime in runtimes:
        command.extend(["--runtime", f"{runtime.label}={runtime.url}"])

    completed = subprocess.run(
        command,
        cwd=str(PART_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    payload: dict[str, Any] = {}
    if output_path.exists():
        try:
            parsed = json.loads(output_path.read_text("utf-8"))
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = {}

    ranking = (
        payload.get("ranking", []) if isinstance(payload.get("ranking"), list) else []
    )
    return {
        "ok": completed.returncode == 0 and bool(payload.get("ok", False)),
        "exit_code": int(completed.returncode),
        "report": str(output_path),
        "ranking": ranking,
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-8:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-8:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate NPU embedding config, simulation latency, and presence-training outcomes"
    )
    parser.add_argument(
        "--runtime",
        action="append",
        default=[],
        help="Runtime endpoint in label=http://host:port format (repeatable)",
    )
    parser.add_argument(
        "--circumstances",
        default="world_state/muse_semantic_training_circumstances.json",
        help="Semantic training circumstances JSON path",
    )
    parser.add_argument(
        "--regimen",
        default="world_state/muse_song_training_regime.json",
        help="Song benchmark regimen JSON path",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=35.0,
        help="HTTP timeout seconds",
    )
    parser.add_argument(
        "--latency-requests",
        type=int,
        default=24,
        help="Timed /api/simulation requests per runtime",
    )
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=6,
        help="Warmup /api/simulation requests per runtime",
    )
    parser.add_argument(
        "--training-rounds",
        type=int,
        default=4,
        help="Rounds override for semantic training (0 keeps circumstances default)",
    )
    parser.add_argument(
        "--song-rounds",
        type=int,
        default=2,
        help="Rounds override for song benchmark (0 keeps regimen default)",
    )
    parser.add_argument(
        "--environment-events",
        type=int,
        default=48,
        help="Synthetic environment events injected per runtime before training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4207,
        help="Deterministic base seed",
    )
    parser.add_argument(
        "--output",
        default="../.opencode/runtime/sim_learning_eval.latest.json",
        help="Aggregate evaluation report output path",
    )
    parser.add_argument(
        "--skip-song-benchmark",
        action="store_true",
        help="Skip bench_muse_song_lab execution",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip muse_semantic_training_lab execution",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return success even if one or more checks fail",
    )
    parser.add_argument(
        "--proxy-env-path",
        default="",
        help="Optional path to proxy .env file for API key discovery",
    )
    args = parser.parse_args()

    runtimes: list[RuntimeRef] = []
    if args.runtime:
        for raw in args.runtime:
            runtimes.append(_parse_runtime_arg(str(raw)))
    else:
        runtimes = [
            RuntimeRef("song-baseline", "http://127.0.0.1:19877"),
            RuntimeRef("song-chaos", "http://127.0.0.1:19878"),
            RuntimeRef("song-stability", "http://127.0.0.1:19879"),
        ]

    timeout_seconds = max(1.0, float(args.timeout))
    openvino_api_key = _resolve_openvino_api_key(
        str(args.proxy_env_path).strip() or None
    )
    output_path = Path(str(args.output)).resolve()
    circumstances = Path(str(args.circumstances)).resolve()
    regimen = Path(str(args.regimen)).resolve()

    npu_checks: list[dict[str, Any]] = []
    env_injections: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    song_benchmark: dict[str, Any] = {
        "ok": False,
        "skipped": bool(args.skip_song_benchmark),
    }

    failures = 0
    for runtime in runtimes:
        check = _ensure_npu(
            runtime,
            timeout_seconds,
            openvino_api_key=openvino_api_key,
        )
        npu_checks.append(check)
        print(
            f"npu-check {runtime.label}: backend={check.get('backend')} device={check.get('openvino_device')} "
            f"c_runtime={check.get('c_runtime_source') or '(none)'} "
            f"cpu_fallback={check.get('c_runtime_cpu_fallback')} "
            f"daimoi_backend={check.get('daimoi_backend') or '(none)'} ok={check.get('ok')}"
        )
        if not bool(check.get("ok", False)):
            failures += 1

        env_result = _inject_environment_events(
            runtime,
            event_count=max(1, int(args.environment_events)),
            timeout_seconds=timeout_seconds,
            seed=int(args.seed) + (len(env_injections) * 97),
        )
        env_injections.append(env_result)
        if not bool(env_result.get("ok", False)):
            failures += 1

        latency = _bench_simulation_latency(
            runtime,
            request_count=max(1, int(args.latency_requests)),
            warmup=max(0, int(args.latency_warmup)),
            timeout_seconds=timeout_seconds,
        )
        latency_rows.append(latency)
        print(
            f"latency {runtime.label}: ok={latency.get('ok')} "
            f"mean={float(latency.get('mean_ms', 0.0)):.2f}ms p95={float(latency.get('p95_ms', 0.0)):.2f}ms"
        )
        if not bool(latency.get("ok", False)):
            failures += 1

        if not args.skip_training:
            training_report = (
                output_path.parent / f"sim_learning_training.{runtime.label}.json"
            )
            training = _run_training(
                runtime,
                circumstances=circumstances,
                output_path=training_report,
                rounds=max(0, int(args.training_rounds)),
                timeout_seconds=timeout_seconds,
                seed=int(args.seed) + (len(training_rows) * 313),
            )
            training_rows.append(training)
            summary = training.get("summary", {})
            print(
                f"training {runtime.label}: ok={training.get('ok')} "
                f"target={float(summary.get('target_accuracy', 0.0)) * 100:.1f}% "
                f"routed={float(summary.get('routed_accuracy', 0.0)) * 100:.1f}%"
            )
            if not bool(training.get("ok", False)):
                failures += 1

    if not args.skip_song_benchmark:
        bench_report = output_path.parent / "sim_learning_song_benchmark.json"
        song_benchmark = _run_song_benchmark(
            runtimes,
            regimen=regimen,
            rounds=max(0, int(args.song_rounds)),
            timeout_seconds=timeout_seconds,
            output_path=bench_report,
        )
        print(
            f"song-benchmark ok={song_benchmark.get('ok')} report={song_benchmark.get('report')}"
        )
        if not bool(song_benchmark.get("ok", False)):
            failures += 1

    training_valid = [
        row for row in training_rows if isinstance(row.get("summary"), dict)
    ]
    mean_target_accuracy = (
        statistics.fmean(
            float((row.get("summary", {}) or {}).get("target_accuracy", 0.0) or 0.0)
            for row in training_valid
        )
        if training_valid
        else 0.0
    )
    mean_routed_accuracy = (
        statistics.fmean(
            float((row.get("summary", {}) or {}).get("routed_accuracy", 0.0) or 0.0)
            for row in training_valid
        )
        if training_valid
        else 0.0
    )

    best_latency = sorted(
        [row for row in latency_rows if bool(row.get("ok", False))],
        key=lambda row: float(row.get("mean_ms", 10_000_000.0)),
    )
    best_runtime = str(best_latency[0].get("label", "")) if best_latency else ""

    report = {
        "ok": failures == 0,
        "record": "eta-mu.sim-learning-eval.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "runtimes": [{"label": row.label, "url": row.url} for row in runtimes],
            "timeout": timeout_seconds,
            "latency_requests": int(args.latency_requests),
            "latency_warmup": int(args.latency_warmup),
            "training_rounds": int(args.training_rounds),
            "song_rounds": int(args.song_rounds),
            "environment_events": int(args.environment_events),
            "seed": int(args.seed),
            "circumstances": str(circumstances),
            "regimen": str(regimen),
        },
        "npu_checks": npu_checks,
        "environment_injections": env_injections,
        "simulation_latency": latency_rows,
        "training": training_rows,
        "song_benchmark": song_benchmark,
        "summary": {
            "failure_count": failures,
            "best_latency_runtime": best_runtime,
            "mean_target_accuracy": mean_target_accuracy,
            "mean_routed_accuracy": mean_routed_accuracy,
        },
    }
    _save_json(output_path, report)
    print(f"report={output_path}")

    if failures > 0 and not bool(args.allow_failures):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
