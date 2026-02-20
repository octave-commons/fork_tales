from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import uuid
from typing import Any


SIM_SLICE_POINT_BUDGET_ID = "sim_point_budget.v1"
SIM_SLICE_REDIS_MODE = "redis"
SIM_SLICE_UDS_MODE = "uds"
SIM_SLICE_OFFLOAD_MODE_ENV = "SIM_SLICE_OFFLOAD_MODE"
SIM_SLICE_ASYNC_ENV = "SIM_SLICE_ASYNC"
SIM_SLICE_ASYNC_STALE_MS_ENV = "SIM_SLICE_ASYNC_STALE_MS"
SIM_SLICE_REDIS_CLI_ENV = "SIM_SLICE_REDIS_CLI"
SIM_SLICE_REDIS_HOST_ENV = "SIM_SLICE_REDIS_HOST"
SIM_SLICE_REDIS_PORT_ENV = "SIM_SLICE_REDIS_PORT"
SIM_SLICE_REDIS_PASSWORD_ENV = "SIM_SLICE_REDIS_PASSWORD"
SIM_SLICE_REDIS_JOBS_STREAM_ENV = "SIM_SLICE_REDIS_JOBS_STREAM"
SIM_SLICE_REDIS_REPLY_PREFIX_ENV = "SIM_SLICE_REDIS_REPLY_PREFIX"
SIM_SLICE_REDIS_NAMESPACE_ENV = "SIM_SLICE_REDIS_NAMESPACE"
SIM_SLICE_REDIS_TIMEOUT_MS_ENV = "SIM_SLICE_REDIS_TIMEOUT_MS"
SIM_SLICE_REDIS_POLL_MS_ENV = "SIM_SLICE_REDIS_POLL_MS"
SIM_SLICE_UDS_PATH_ENV = "SIM_SLICE_UDS_PATH"
SIM_SLICE_UDS_TIMEOUT_MS_ENV = "SIM_SLICE_UDS_TIMEOUT_MS"


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return fallback


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _env_flag(name: str, *, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    if raw in {"1", "true", "yes", "on", "enabled"}:
        return True
    if raw in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


class _DoubleBufferedSnapshot:
    def __init__(self, initial: dict[str, Any]) -> None:
        self._slots: list[dict[str, Any]] = [dict(initial), dict(initial)]
        self._active_index = 0
        self._swap_lock = threading.Lock()

    def read(self) -> dict[str, Any]:
        return self._slots[self._active_index]

    def publish(self, snapshot: dict[str, Any]) -> None:
        with self._swap_lock:
            back_index = 1 - self._active_index
            self._slots[back_index] = snapshot
            self._active_index = back_index


class _SimPointBudgetAsyncWorker:
    def __init__(self) -> None:
        self._request_lock = threading.Lock()
        self._pending_request: dict[str, Any] | None = None
        self._request_event = threading.Event()
        self._result = _DoubleBufferedSnapshot(
            {
                "ready": False,
                "mode": "",
                "budget": 0,
                "source": "",
                "fallback": True,
                "reason": "",
                "job_id": "",
                "transport_latency_ms": 0.0,
                "produced_monotonic": 0.0,
            }
        )
        self._thread_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _ensure_thread(self) -> None:
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._run,
                name="sim-slice-async-worker",
                daemon=True,
            )
            self._thread.start()

    def submit(
        self,
        *,
        mode: str,
        cpu_utilization: float,
        max_sim_points: int,
        local_budget: int,
    ) -> None:
        self._ensure_thread()
        with self._request_lock:
            self._pending_request = {
                "mode": mode,
                "cpu_utilization": cpu_utilization,
                "max_sim_points": max_sim_points,
                "local_budget": local_budget,
            }
        self._request_event.set()

    def latest(self) -> dict[str, Any]:
        return self._result.read()

    def _run(self) -> None:
        while True:
            self._request_event.wait()
            with self._request_lock:
                request = self._pending_request
                self._pending_request = None
                self._request_event.clear()

            if not request:
                continue

            mode = str(request.get("mode", "")).strip().lower()
            cpu_utilization = _safe_float(request.get("cpu_utilization", 0.0), 0.0)
            max_sim_points = max(64, _safe_int(request.get("max_sim_points", 800), 800))
            local_budget = max(64, _safe_int(request.get("local_budget", 800), 800))

            started = time.monotonic()
            remote_budget: int | None = None
            remote_meta: dict[str, Any] = {
                "source": "python-fallback",
                "reason": "unsupported-mode",
                "job_id": "",
            }
            if mode == SIM_SLICE_REDIS_MODE:
                remote_budget, remote_meta = _request_sim_point_budget_via_redis(
                    cpu_utilization=cpu_utilization,
                    max_sim_points=max_sim_points,
                )
            elif mode == SIM_SLICE_UDS_MODE:
                remote_budget, remote_meta = _request_sim_point_budget_via_uds(
                    cpu_utilization=cpu_utilization,
                    max_sim_points=max_sim_points,
                )
            transport_latency_ms = round((time.monotonic() - started) * 1000.0, 3)

            if remote_budget is None:
                self._result.publish(
                    {
                        "ready": True,
                        "mode": mode,
                        "budget": local_budget,
                        "source": str(
                            remote_meta.get("source", "python-fallback")
                            or "python-fallback"
                        ),
                        "fallback": True,
                        "reason": str(
                            remote_meta.get("reason", "unknown") or "unknown"
                        ),
                        "job_id": str(remote_meta.get("job_id", "") or ""),
                        "transport_latency_ms": transport_latency_ms,
                        "produced_monotonic": time.monotonic(),
                    }
                )
                continue

            self._result.publish(
                {
                    "ready": True,
                    "mode": mode,
                    "budget": int(remote_budget),
                    "source": str(remote_meta.get("source", "c-worker") or "c-worker"),
                    "fallback": False,
                    "reason": "",
                    "job_id": str(remote_meta.get("job_id", "") or ""),
                    "transport_latency_ms": transport_latency_ms,
                    "produced_monotonic": time.monotonic(),
                }
            )


_SIM_POINT_BUDGET_ASYNC_WORKER = _SimPointBudgetAsyncWorker()


def _local_sim_point_budget(cpu_utilization: float, max_sim_points: int) -> int:
    if cpu_utilization >= 90.0:
        return max(256, int(max_sim_points * 0.55))
    if cpu_utilization >= 78.0:
        return max(320, int(max_sim_points * 0.74))
    return max_sim_points


def _redis_cli_base_command() -> list[str]:
    cli = str(os.getenv(SIM_SLICE_REDIS_CLI_ENV, "redis-cli") or "redis-cli").strip()
    host = str(os.getenv(SIM_SLICE_REDIS_HOST_ENV, "127.0.0.1") or "127.0.0.1").strip()
    port = max(1, _safe_int(os.getenv(SIM_SLICE_REDIS_PORT_ENV, "6379"), 6379))
    command = [cli, "--raw", "-h", host, "-p", str(port)]
    password = str(os.getenv(SIM_SLICE_REDIS_PASSWORD_ENV, "") or "").strip()
    if password:
        command.extend(["-a", password])
    return command


def _run_redis_cli(
    args: list[str],
    *,
    timeout_seconds: float,
) -> tuple[bool, str, str]:
    command = [*_redis_cli_base_command(), *args]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(0.05, timeout_seconds),
            check=False,
        )
    except FileNotFoundError:
        return False, "", "redis-cli-missing"
    except subprocess.TimeoutExpired:
        return False, "", "redis-cli-timeout"
    except Exception as exc:
        return False, "", f"redis-cli-error:{exc.__class__.__name__}"

    if proc.returncode != 0:
        detail = (
            str(proc.stderr or "").strip()
            or str(proc.stdout or "").strip()
            or f"redis-cli-exit:{proc.returncode}"
        )
        return False, str(proc.stdout or "").strip(), detail

    return True, str(proc.stdout or "").strip(), ""


def _request_sim_point_budget_via_redis(
    *,
    cpu_utilization: float,
    max_sim_points: int,
) -> tuple[int | None, dict[str, Any]]:
    namespace = str(
        os.getenv(SIM_SLICE_REDIS_NAMESPACE_ENV, "eta_mu") or "eta_mu"
    ).strip()
    jobs_stream_default = f"{namespace}:sim_slice_jobs"
    reply_prefix_default = f"{namespace}:sim_slice_reply"

    jobs_stream = str(
        os.getenv(SIM_SLICE_REDIS_JOBS_STREAM_ENV, jobs_stream_default)
        or jobs_stream_default
    ).strip()
    reply_prefix = str(
        os.getenv(SIM_SLICE_REDIS_REPLY_PREFIX_ENV, reply_prefix_default)
        or reply_prefix_default
    ).strip()
    timeout_ms = max(5, _safe_int(os.getenv(SIM_SLICE_REDIS_TIMEOUT_MS_ENV, "35"), 35))
    poll_ms = max(2, _safe_int(os.getenv(SIM_SLICE_REDIS_POLL_MS_ENV, "6"), 6))

    job_id = uuid.uuid4().hex
    reply_key = f"{reply_prefix}:{job_id}"

    ok, _, error = _run_redis_cli(
        [
            "XADD",
            jobs_stream,
            "*",
            "slice",
            SIM_SLICE_POINT_BUDGET_ID,
            "job_id",
            job_id,
            "cpu_utilization",
            f"{cpu_utilization:.6f}",
            "max_sim_points",
            str(max_sim_points),
            "reply_key",
            reply_key,
            "submitted_ms",
            str(int(time.time() * 1000.0)),
        ],
        timeout_seconds=0.3,
    )
    if not ok:
        return None, {
            "source": "python-fallback",
            "reason": f"enqueue-failed:{error}",
            "job_id": job_id,
        }

    deadline = time.monotonic() + (timeout_ms / 1000.0)
    while time.monotonic() < deadline:
        ok, stdout, error = _run_redis_cli(["GET", reply_key], timeout_seconds=0.2)
        if not ok:
            return None, {
                "source": "python-fallback",
                "reason": f"read-failed:{error}",
                "job_id": job_id,
            }
        if not stdout:
            time.sleep(poll_ms / 1000.0)
            continue

        response_job_id = ""
        response_source = "c-worker"
        response_budget = 0
        payload = stdout
        try:
            data = json.loads(payload)
        except Exception:
            data = {}

        if isinstance(data, dict):
            response_job_id = str(data.get("job_id", "")).strip()
            response_source = str(data.get("source", "c-worker")).strip() or "c-worker"
            response_budget = _safe_int(data.get("sim_point_budget", 0), 0)
        else:
            response_budget = _safe_int(payload, 0)

        if response_job_id and response_job_id != job_id:
            time.sleep(poll_ms / 1000.0)
            continue

        _run_redis_cli(["DEL", reply_key], timeout_seconds=0.2)
        if response_budget > 0:
            return response_budget, {
                "source": response_source,
                "job_id": job_id,
            }
        return None, {
            "source": "python-fallback",
            "reason": "invalid-worker-payload",
            "job_id": job_id,
        }

    _run_redis_cli(["DEL", reply_key], timeout_seconds=0.2)
    return None, {
        "source": "python-fallback",
        "reason": "redis-worker-timeout",
        "job_id": job_id,
    }


def _request_sim_point_budget_via_uds(
    *,
    cpu_utilization: float,
    max_sim_points: int,
) -> tuple[int | None, dict[str, Any]]:
    socket_path = str(
        os.getenv(SIM_SLICE_UDS_PATH_ENV, "/tmp/eta_mu_sim_slice.sock")
        or "/tmp/eta_mu_sim_slice.sock"
    ).strip()
    timeout_ms = max(4, _safe_int(os.getenv(SIM_SLICE_UDS_TIMEOUT_MS_ENV, "35"), 35))
    timeout_seconds = timeout_ms / 1000.0
    job_id = uuid.uuid4().hex

    request_payload = {
        "slice": SIM_SLICE_POINT_BUDGET_ID,
        "job_id": job_id,
        "cpu_utilization": round(cpu_utilization, 6),
        "max_sim_points": int(max_sim_points),
        "submitted_ms": int(time.time() * 1000.0),
    }

    raw_response = ""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout_seconds)
            client.connect(socket_path)
            encoded_request = (
                json.dumps(request_payload, ensure_ascii=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            client.sendall(encoded_request)

            chunks: list[bytes] = []
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
            raw_response = b"".join(chunks).decode("utf-8", errors="replace").strip()
    except FileNotFoundError:
        return None, {
            "source": "python-fallback",
            "reason": "uds-socket-missing",
            "job_id": job_id,
        }
    except socket.timeout:
        return None, {
            "source": "python-fallback",
            "reason": "uds-timeout",
            "job_id": job_id,
        }
    except OSError as exc:
        return None, {
            "source": "python-fallback",
            "reason": f"uds-error:{exc.__class__.__name__}",
            "job_id": job_id,
        }

    if not raw_response:
        return None, {
            "source": "python-fallback",
            "reason": "uds-empty-response",
            "job_id": job_id,
        }

    try:
        response_payload = json.loads(raw_response)
    except Exception:
        return None, {
            "source": "python-fallback",
            "reason": "uds-invalid-json",
            "job_id": job_id,
        }
    if not isinstance(response_payload, dict):
        return None, {
            "source": "python-fallback",
            "reason": "uds-nonobject-response",
            "job_id": job_id,
        }

    response_job_id = str(response_payload.get("job_id", "")).strip()
    if response_job_id and response_job_id != job_id:
        return None, {
            "source": "python-fallback",
            "reason": "uds-job-mismatch",
            "job_id": job_id,
        }

    budget_value = _safe_int(response_payload.get("sim_point_budget", 0), 0)
    if budget_value <= 0:
        return None, {
            "source": "python-fallback",
            "reason": "uds-invalid-budget",
            "job_id": job_id,
        }

    return budget_value, {
        "source": str(response_payload.get("source", "c-worker")).strip() or "c-worker",
        "job_id": job_id,
    }


def _request_sim_point_budget_remote_sync(
    *,
    mode: str,
    cpu_utilization: float,
    max_sim_points: int,
) -> tuple[int | None, dict[str, Any]]:
    if mode == SIM_SLICE_REDIS_MODE:
        return _request_sim_point_budget_via_redis(
            cpu_utilization=cpu_utilization,
            max_sim_points=max_sim_points,
        )
    if mode == SIM_SLICE_UDS_MODE:
        return _request_sim_point_budget_via_uds(
            cpu_utilization=cpu_utilization,
            max_sim_points=max_sim_points,
        )
    return None, {
        "source": "python-fallback",
        "reason": "unsupported-mode",
        "job_id": "",
    }


def resolve_sim_point_budget_slice(
    *,
    cpu_utilization: float,
    max_sim_points: int,
) -> tuple[int, dict[str, Any]]:
    cpu_value = max(0.0, _safe_float(cpu_utilization, 0.0))
    max_points_value = max(64, _safe_int(max_sim_points, 800))
    local_budget = _local_sim_point_budget(cpu_value, max_points_value)

    mode = str(os.getenv(SIM_SLICE_OFFLOAD_MODE_ENV, "local") or "local").strip()
    normalized_mode = mode.lower()
    started = time.monotonic()

    if normalized_mode not in {SIM_SLICE_REDIS_MODE, SIM_SLICE_UDS_MODE}:
        return local_budget, {
            "mode": normalized_mode or "local",
            "source": "python-local",
            "fallback": False,
            "latency_ms": 0.0,
        }

    async_enabled = _env_flag(SIM_SLICE_ASYNC_ENV, default=False)
    if async_enabled:
        stale_limit_ms = max(
            20,
            _safe_int(os.getenv(SIM_SLICE_ASYNC_STALE_MS_ENV, "5000"), 5000),
        )
        _SIM_POINT_BUDGET_ASYNC_WORKER.submit(
            mode=normalized_mode,
            cpu_utilization=cpu_value,
            max_sim_points=max_points_value,
            local_budget=local_budget,
        )

        latest = _SIM_POINT_BUDGET_ASYNC_WORKER.latest()
        latency_ms = round((time.monotonic() - started) * 1000.0, 3)
        latest_mode = str(latest.get("mode", "")).strip().lower()
        latest_ready = bool(latest.get("ready", False))
        produced_monotonic = _safe_float(latest.get("produced_monotonic", 0.0), 0.0)
        age_ms = (
            max(0.0, (time.monotonic() - produced_monotonic) * 1000.0)
            if produced_monotonic > 0.0
            else 0.0
        )
        if latest_ready and latest_mode == normalized_mode and age_ms <= stale_limit_ms:
            return max(
                64, _safe_int(latest.get("budget", local_budget), local_budget)
            ), {
                "mode": normalized_mode,
                "source": str(latest.get("source", "c-worker") or "c-worker"),
                "fallback": bool(latest.get("fallback", False)),
                "reason": str(latest.get("reason", "") or ""),
                "job_id": str(latest.get("job_id", "") or ""),
                "latency_ms": latency_ms,
                "transport_latency_ms": _safe_float(
                    latest.get("transport_latency_ms", 0.0),
                    0.0,
                ),
                "age_ms": round(age_ms, 3),
                "async": True,
            }

        return local_budget, {
            "mode": normalized_mode,
            "source": "python-local",
            "fallback": True,
            "reason": (
                "async-stale"
                if latest_ready
                and latest_mode == normalized_mode
                and age_ms > stale_limit_ms
                else "async-warmup"
            ),
            "job_id": str(latest.get("job_id", "") or ""),
            "latency_ms": latency_ms,
            "transport_latency_ms": _safe_float(
                latest.get("transport_latency_ms", 0.0),
                0.0,
            ),
            "age_ms": round(age_ms, 3),
            "async": True,
        }

    remote_budget, remote_meta = _request_sim_point_budget_remote_sync(
        mode=normalized_mode,
        cpu_utilization=cpu_value,
        max_sim_points=max_points_value,
    )
    latency_ms = round((time.monotonic() - started) * 1000.0, 3)
    if remote_budget is None:
        return local_budget, {
            "mode": normalized_mode,
            "source": str(
                remote_meta.get("source", "python-fallback") or "python-fallback"
            ),
            "fallback": True,
            "reason": str(remote_meta.get("reason", "unknown") or "unknown"),
            "job_id": str(remote_meta.get("job_id", "") or ""),
            "latency_ms": latency_ms,
        }

    return remote_budget, {
        "mode": normalized_mode,
        "source": str(remote_meta.get("source", "c-worker") or "c-worker"),
        "fallback": False,
        "job_id": str(remote_meta.get("job_id", "") or ""),
        "latency_ms": latency_ms,
    }
