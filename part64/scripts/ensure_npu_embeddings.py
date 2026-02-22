#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class RuntimeCheck:
    label: str
    runtime: str
    ok: bool
    backend: str
    openvino_device: str
    openvino_endpoint: str
    probe_ok: bool
    probe_error: str
    c_runtime_source: str
    c_runtime_error: str
    c_runtime_cpu_fallback: bool
    c_runtime_cpu_fallback_detail: str
    daimoi_backend: str
    daimoi_backend_error: str


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
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise RuntimeError(f"expected object payload from {url}")
        return data
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
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
    *, runtime: str, timeout_seconds: float, baseline_generated_at: str
) -> dict[str, Any]:
    url = f"{runtime}/api/simulation?perspective=hybrid&compact=1"
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


def _parse_runtime_arg(raw: str) -> tuple[str, str]:
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
    return clean_label, clean_url


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

    default_proxy_env = (
        Path(__file__).resolve().parents[2] / "docker-llm-proxy" / ".env"
    )
    return _load_key_from_env_file(default_proxy_env)


def _check_runtime(
    *,
    label: str,
    runtime: str,
    timeout_seconds: float,
    openvino_endpoint: str,
    openvino_model: str,
    openvino_timeout_sec: float,
    openvino_api_key: str,
    probe_text: str,
) -> RuntimeCheck:
    options_url = f"{runtime}/api/embeddings/provider/options"
    apply_payload = {
        "preset": "npu_local",
        "backend": "openvino",
        "openvino_device": "NPU",
        "openvino_endpoint": openvino_endpoint,
        "openvino_model": openvino_model,
        "openvino_timeout_sec": float(openvino_timeout_sec),
    }
    if openvino_api_key:
        apply_payload["openvino_api_key"] = openvino_api_key
        apply_payload["openvino_api_key_header"] = "X-API-Key"
        apply_payload["openvino_bearer_token"] = openvino_api_key
    _post_json(options_url, apply_payload, timeout_seconds)

    baseline_generated_at = ""
    try:
        baseline_payload = _request_json(
            f"{runtime}/api/simulation?perspective=hybrid&compact=1",
            max(8.0, min(timeout_seconds, 20.0)),
        )
        baseline_generated_at = _simulation_generated_at(baseline_payload)
    except Exception:
        baseline_generated_at = ""

    _post_json(
        f"{runtime}/api/presence/user/input",
        {
            "events": [
                {
                    "kind": "input",
                    "target": "simulation",
                    "message": str(probe_text).strip() or "npu c runtime probe",
                    "embed_daimoi": True,
                }
            ]
        },
        timeout_seconds,
    )

    options = _request_json(options_url, timeout_seconds)
    config = (
        options.get("config", {}) if isinstance(options.get("config"), dict) else {}
    )
    backend = str(config.get("backend", "")).strip().lower()
    device = str(config.get("openvino_device", "")).strip().upper()
    endpoint = str(config.get("openvino_endpoint", "")).strip()

    c_runtime_source = ""
    c_runtime_error = ""
    c_runtime_cpu_fallback = False
    c_runtime_cpu_fallback_detail = ""
    daimoi_backend = ""
    daimoi_backend_error = ""
    try:
        simulation = _probe_simulation_payload(
            runtime=runtime,
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

    ok = backend == "openvino" and device == "NPU" and probe_ok
    return RuntimeCheck(
        label=label,
        runtime=runtime,
        ok=ok,
        backend=backend,
        openvino_device=device,
        openvino_endpoint=endpoint,
        probe_ok=probe_ok,
        probe_error=probe_error,
        c_runtime_source=c_runtime_source,
        c_runtime_error=c_runtime_error,
        c_runtime_cpu_fallback=c_runtime_cpu_fallback,
        c_runtime_cpu_fallback_detail=c_runtime_cpu_fallback_detail,
        daimoi_backend=daimoi_backend,
        daimoi_backend_error=daimoi_backend_error,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Force and verify NPU embeddings config per runtime"
    )
    parser.add_argument(
        "--runtime",
        action="append",
        default=[],
        help="Runtime endpoint in label=http://host:port format (repeatable)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=35.0,
        help="HTTP timeout seconds",
    )
    parser.add_argument(
        "--openvino-endpoint",
        default="http://host.docker.internal:18000/v1/embeddings",
        help="OpenVINO embeddings endpoint to apply",
    )
    parser.add_argument(
        "--openvino-model",
        default="nomic-embed-text",
        help="OpenVINO embeddings model to apply",
    )
    parser.add_argument(
        "--openvino-timeout-sec",
        type=float,
        default=12.0,
        help="OpenVINO request timeout to apply",
    )
    parser.add_argument(
        "--proxy-env-path",
        default="",
        help="Optional path to proxy .env file for API key discovery",
    )
    parser.add_argument(
        "--probe-text",
        default="NPU embedding probe through openvino backend",
        help="Probe text for /api/embeddings/db/query",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional JSON report output path",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return success even if one or more runtimes fail",
    )
    args = parser.parse_args()

    runtimes: list[tuple[str, str]] = []
    if args.runtime:
        for raw in args.runtime:
            runtimes.append(_parse_runtime_arg(str(raw)))
    else:
        runtimes = [("gateway", "http://127.0.0.1:8787")]

    timeout_seconds = max(1.0, float(args.timeout))
    openvino_api_key = _resolve_openvino_api_key(
        str(args.proxy_env_path).strip() or None
    )
    checks: list[RuntimeCheck] = []
    failures = 0
    for label, runtime in runtimes:
        try:
            result = _check_runtime(
                label=label,
                runtime=runtime,
                timeout_seconds=timeout_seconds,
                openvino_endpoint=str(args.openvino_endpoint).strip(),
                openvino_model=str(args.openvino_model).strip(),
                openvino_timeout_sec=float(args.openvino_timeout_sec),
                openvino_api_key=openvino_api_key,
                probe_text=str(args.probe_text),
            )
        except Exception as exc:
            result = RuntimeCheck(
                label=label,
                runtime=runtime,
                ok=False,
                backend="",
                openvino_device="",
                openvino_endpoint="",
                probe_ok=False,
                probe_error=f"{exc.__class__.__name__}:{exc}",
                c_runtime_source="",
                c_runtime_error="",
                c_runtime_cpu_fallback=False,
                c_runtime_cpu_fallback_detail="",
                daimoi_backend="",
                daimoi_backend_error="",
            )
        checks.append(result)
        if not result.ok:
            failures += 1

        print(
            f"{label}: backend={result.backend or '(none)'} device={result.openvino_device or '(none)'} "
            f"c_runtime={result.c_runtime_source or '(none)'} "
            f"cpu_fallback={result.c_runtime_cpu_fallback} "
            f"daimoi_backend={result.daimoi_backend or '(none)'} ok={result.ok}"
        )
        if result.probe_error:
            print(f"  probe_error={result.probe_error}")

    report = {
        "ok": failures == 0,
        "record": "eta-mu.npu-embedding-check.v1",
        "runtimes": [
            {
                "label": row.label,
                "runtime": row.runtime,
                "ok": row.ok,
                "backend": row.backend,
                "openvino_device": row.openvino_device,
                "openvino_endpoint": row.openvino_endpoint,
                "probe_ok": row.probe_ok,
                "probe_error": row.probe_error,
                "c_runtime_source": row.c_runtime_source,
                "c_runtime_error": row.c_runtime_error,
                "c_runtime_cpu_fallback": row.c_runtime_cpu_fallback,
                "c_runtime_cpu_fallback_detail": row.c_runtime_cpu_fallback_detail,
                "daimoi_backend": row.daimoi_backend,
                "daimoi_backend_error": row.daimoi_backend_error,
            }
            for row in checks
        ],
    }

    if str(args.json_out).strip():
        out_path = Path(str(args.json_out)).resolve()
        _save_json(out_path, report)
        print(f"json_report={out_path}")

    if failures > 0 and not bool(args.allow_failures):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
