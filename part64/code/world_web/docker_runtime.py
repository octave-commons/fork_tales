from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from typing import Any
from urllib.parse import quote


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = str(os.getenv(name, str(default)) or str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = str(os.getenv(name, str(default)) or str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, value)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, *, default: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _csv_list(raw: str) -> list[str]:
    values: list[str] = []
    for item in str(raw or "").split(","):
        cleaned = item.strip().lower()
        if cleaned and cleaned not in values:
            values.append(cleaned)
    return values


DOCKER_SIMULATION_SOCKET_PATH = (
    str(
        os.getenv("DOCKER_SOCKET_PATH", "/var/run/docker.sock")
        or "/var/run/docker.sock"
    ).strip()
    or "/var/run/docker.sock"
)
DOCKER_SIMULATION_API_TIMEOUT_SECONDS = _env_float(
    "DOCKER_SIMULATION_API_TIMEOUT_SECONDS",
    1.8,
    0.2,
)
DOCKER_SIMULATION_LABEL_KEY = (
    str(
        os.getenv("DOCKER_SIMULATION_LABEL_KEY", "io.fork_tales.simulation")
        or "io.fork_tales.simulation"
    ).strip()
    or "io.fork_tales.simulation"
)
DOCKER_SIMULATION_ROLE_LABEL_KEY = (
    str(
        os.getenv("DOCKER_SIMULATION_ROLE_LABEL_KEY", "io.fork_tales.simulation.role")
        or "io.fork_tales.simulation.role"
    ).strip()
    or "io.fork_tales.simulation.role"
)
DOCKER_SIMULATION_NETWORK_HINT = (
    str(
        os.getenv("DOCKER_SIMULATION_NETWORK_HINT", "eta-mu-sim-net")
        or "eta-mu-sim-net"
    ).strip()
    or "eta-mu-sim-net"
)
DOCKER_SIMULATION_NAME_HINTS = _csv_list(
    str(
        os.getenv(
            "DOCKER_SIMULATION_NAME_HINTS",
            "eta-mu,muse-song,sim-slice,simulation,experiment",
        )
        or "eta-mu,muse-song,sim-slice,simulation,experiment"
    )
)
DOCKER_SIMULATION_PRIVATE_PORT_HINT = _env_int(
    "DOCKER_SIMULATION_PRIVATE_PORT_HINT",
    8787,
    1,
)
DOCKER_SIMULATION_CACHE_SECONDS = _env_float(
    "DOCKER_SIMULATION_CACHE_SECONDS",
    2.0,
    0.2,
)
DOCKER_SIMULATION_CACHE_WAIT_SECONDS = _env_float(
    "DOCKER_SIMULATION_CACHE_WAIT_SECONDS",
    2.4,
    0.1,
)
DOCKER_SIMULATION_WS_REFRESH_SECONDS = _env_float(
    "DOCKER_SIMULATION_WS_REFRESH_SECONDS",
    2.5,
    1.0,
)
DOCKER_SIMULATION_WS_HEARTBEAT_SECONDS = max(
    DOCKER_SIMULATION_WS_REFRESH_SECONDS,
    _env_float(
        "DOCKER_SIMULATION_WS_HEARTBEAT_SECONDS",
        18.0,
        1.0,
    ),
)
DOCKER_SIMULATION_RESOURCE_WORKERS = min(
    24,
    _env_int(
        "DOCKER_SIMULATION_RESOURCE_WORKERS",
        6,
        1,
    ),
)
DOCKER_SIMULATION_RESOURCE_SAMPLE_LIMIT = _env_int(
    "DOCKER_SIMULATION_RESOURCE_SAMPLE_LIMIT",
    160,
    1,
)
DOCKER_SIMULATION_RESOURCE_WARNING_RATIO = _env_float(
    "DOCKER_SIMULATION_RESOURCE_WARNING_RATIO",
    0.75,
    0.05,
)
DOCKER_SIMULATION_RESOURCE_CRITICAL_RATIO = max(
    DOCKER_SIMULATION_RESOURCE_WARNING_RATIO,
    _env_float(
        "DOCKER_SIMULATION_RESOURCE_CRITICAL_RATIO",
        0.9,
        0.05,
    ),
)

_SIMULATION_ROLE_HINTS = {
    "simulation",
    "experiment",
    "runtime",
    "world-runtime",
}

_DOCKER_SIMULATION_CACHE_LOCK = threading.Lock()
_DOCKER_SIMULATION_CACHE_CONDITION = threading.Condition(_DOCKER_SIMULATION_CACHE_LOCK)
_DOCKER_SIMULATION_CACHE_REFRESH_IN_FLIGHT = False
_DOCKER_SIMULATION_CACHE: dict[str, Any] = {
    "monotonic": 0.0,
    "payload": None,
}


class _UnixSocketHTTPConnection(HTTPConnection):
    def __init__(self, socket_path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        timeout = (
            float(self.timeout)
            if isinstance(self.timeout, (int, float))
            else DOCKER_SIMULATION_API_TIMEOUT_SECONDS
        )
        sock.settimeout(timeout)
        sock.connect(self._socket_path)
        self.sock = sock


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _docker_api_json(
    path: str,
    *,
    timeout_seconds: float | None = None,
) -> tuple[Any | None, str]:
    socket_path = Path(DOCKER_SIMULATION_SOCKET_PATH)
    if not socket_path.exists():
        return None, "docker_socket_missing"

    timeout = DOCKER_SIMULATION_API_TIMEOUT_SECONDS
    if timeout_seconds is not None:
        timeout = max(
            0.1, _safe_float(timeout_seconds, DOCKER_SIMULATION_API_TIMEOUT_SECONDS)
        )

    connection = _UnixSocketHTTPConnection(
        str(socket_path),
        timeout=timeout,
    )
    try:
        connection.request(
            "GET",
            path,
            headers={
                "Accept": "application/json",
                "Host": "docker",
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        payload_bytes = response.read()
        if response.status >= 400:
            return None, f"docker_api_http_{response.status}"
        text = payload_bytes.decode("utf-8", errors="replace").strip()
        if not text:
            return None, "docker_api_empty_response"
        try:
            return json.loads(text), ""
        except json.JSONDecodeError:
            return None, "docker_api_invalid_json"
    except PermissionError:
        return None, "docker_socket_permission_denied"
    except FileNotFoundError:
        return None, "docker_socket_missing"
    except (OSError, HTTPException, socket.timeout) as exc:
        return None, f"docker_api_error:{exc.__class__.__name__}"
    finally:
        try:
            connection.close()
        except Exception:
            pass


def _docker_api_post_json(
    path: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float | None = None,
) -> tuple[dict[str, Any] | None, str]:
    socket_path = Path(DOCKER_SIMULATION_SOCKET_PATH)
    if not socket_path.exists():
        return None, "docker_socket_missing"

    timeout = DOCKER_SIMULATION_API_TIMEOUT_SECONDS
    if timeout_seconds is not None:
        timeout = max(
            0.1, _safe_float(timeout_seconds, DOCKER_SIMULATION_API_TIMEOUT_SECONDS)
        )

    connection = _UnixSocketHTTPConnection(
        str(socket_path),
        timeout=timeout,
    )
    try:
        body = json.dumps(payload).encode("utf-8")
        connection.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "Accept": "application/json",
                "Host": "docker",
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        payload_bytes = response.read()
        if response.status >= 400:
            return None, f"docker_api_http_{response.status}"
        text = payload_bytes.decode("utf-8", errors="replace").strip()
        try:
            return (json.loads(text) if text else {}), ""
        except json.JSONDecodeError:
            return {}, ""  # Some updates return empty body
    except PermissionError:
        return None, "docker_socket_permission_denied"
    except FileNotFoundError:
        return None, "docker_socket_missing"
    except (OSError, HTTPException, socket.timeout) as exc:
        return None, f"docker_api_error:{exc.__class__.__name__}"
    finally:
        try:
            connection.close()
        except Exception:
            pass


def _docker_api_post(
    path: str,
    *,
    timeout_seconds: float | None = None,
) -> str:
    socket_path = Path(DOCKER_SIMULATION_SOCKET_PATH)
    if not socket_path.exists():
        return "docker_socket_missing"

    timeout = DOCKER_SIMULATION_API_TIMEOUT_SECONDS
    if timeout_seconds is not None:
        timeout = max(
            0.1, _safe_float(timeout_seconds, DOCKER_SIMULATION_API_TIMEOUT_SECONDS)
        )

    connection = _UnixSocketHTTPConnection(
        str(socket_path),
        timeout=timeout,
    )
    try:
        connection.request(
            "POST",
            path,
            headers={
                "Accept": "application/json",
                "Host": "docker",
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        response.read()
        if response.status >= 400:
            return f"docker_api_http_{response.status}"
        return ""
    except PermissionError:
        return "docker_socket_permission_denied"
    except FileNotFoundError:
        return "docker_socket_missing"
    except (OSError, HTTPException, socket.timeout) as exc:
        return f"docker_api_error:{exc.__class__.__name__}"
    finally:
        try:
            connection.close()
        except Exception:
            pass


def update_container_resources(
    container_id: str,
    *,
    nano_cpus: int | None = None,
    memory_bytes: int | None = None,
    pids_limit: int | None = None,
) -> tuple[bool, str]:
    if not container_id:
        return False, "invalid_container_id"

    updates: dict[str, Any] = {}
    if nano_cpus is not None:
        updates["NanoCpus"] = max(1000000, int(nano_cpus))  # Min 0.001 CPU
    if memory_bytes is not None:
        updates["Memory"] = max(4194304, int(memory_bytes))  # Min 4MB
        # Ideally set swap too, usually memory + swap. Docker requires MemorySwap >= Memory.
        # If MemorySwap is -1 it's unlimited. If 0 it's unset (or disabled?).
        # For safety, let's set MemorySwap = Memory (no swap) or just leave it if it was set higher?
        # Docker API update behavior for swap is tricky if not specified.
        # Let's try setting MemorySwap = Memory to strictly limit total footprint.
        updates["MemorySwap"] = updates["Memory"]
    if pids_limit is not None:
        updates["PidsLimit"] = max(20, int(pids_limit))

    if not updates:
        return True, "no_changes"

    quoted_id = quote(container_id, safe="")
    result, error = _docker_api_post_json(
        f"/containers/{quoted_id}/update",
        updates,
        timeout_seconds=2.5,
    )
    if error:
        return False, error

    # Check for warnings in result
    result_payload = result if isinstance(result, dict) else {}
    warnings = result_payload.get("Warnings")
    if warnings:
        return True, f"updated_with_warnings:{'; '.join(str(w) for w in warnings)}"

    return True, ""


def control_simulation_container(
    container_id: str,
    *,
    action: str,
    stop_timeout_seconds: float = 12.0,
) -> tuple[bool, str]:
    clean_id = str(container_id or "").strip()
    if not clean_id:
        return False, "invalid_container_id"

    verb = str(action or "").strip().lower()
    if verb not in {"start", "stop"}:
        return False, "unsupported_action"

    quoted_id = quote(clean_id, safe="")
    if verb == "stop":
        timeout_value = max(1, min(45, int(_safe_float(stop_timeout_seconds, 12.0))))
        endpoint = f"/containers/{quoted_id}/stop?t={timeout_value}"
    else:
        endpoint = f"/containers/{quoted_id}/start"

    error = _docker_api_post(endpoint, timeout_seconds=max(2.0, stop_timeout_seconds))
    if not error:
        return True, ""

    if error == "docker_api_http_304":
        return True, "already_in_requested_state"

    return False, error


def _normalized_labels(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    labels: dict[str, str] = {}
    for key, value in raw.items():
        clean_key = str(key or "").strip()
        if not clean_key:
            continue
        labels[clean_key] = str(value or "").strip()
    return labels


def _container_name(container: dict[str, Any]) -> str:
    names_raw = container.get("Names", [])
    if isinstance(names_raw, list):
        for item in names_raw:
            clean = str(item or "").strip().lstrip("/")
            if clean:
                return clean
    clean_name = str(container.get("Name", "") or "").strip().lstrip("/")
    if clean_name:
        return clean_name
    return str(container.get("Id", "") or "")[:12]


def _container_networks(container: dict[str, Any]) -> list[str]:
    network_settings = (
        container.get("NetworkSettings", {})
        if isinstance(container.get("NetworkSettings"), dict)
        else {}
    )
    networks_payload = (
        network_settings.get("Networks", {})
        if isinstance(network_settings.get("Networks"), dict)
        else {}
    )
    networks = sorted(
        [
            str(name or "").strip()
            for name in networks_payload.keys()
            if str(name or "").strip()
        ]
    )
    return networks


def _container_ports(container: dict[str, Any]) -> list[dict[str, Any]]:
    rows = (
        container.get("Ports", []) if isinstance(container.get("Ports"), list) else []
    )
    ports: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        private_port = _safe_int(row.get("PrivatePort"), 0)
        public_port = _safe_int(row.get("PublicPort"), 0)
        protocol = str(row.get("Type", "tcp") or "tcp").strip().lower() or "tcp"
        host_ip = str(row.get("IP", "") or "").strip()
        if private_port <= 0 and public_port <= 0:
            continue
        entry: dict[str, Any] = {
            "private_port": private_port if private_port > 0 else None,
            "public_port": public_port if public_port > 0 else None,
            "protocol": protocol,
            "host_ip": host_ip,
        }
        if public_port > 0 and protocol == "tcp":
            entry["url"] = f"http://127.0.0.1:{public_port}"
        ports.append(entry)
    ports.sort(
        key=lambda row: (
            int(row.get("private_port") or 0),
            int(row.get("public_port") or 0),
            str(row.get("protocol") or ""),
        )
    )
    return ports


def _container_created_at(container: dict[str, Any]) -> str:
    created = _safe_int(container.get("Created"), 0)
    if created <= 0:
        return ""
    try:
        return datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def _is_simulation_container(container: dict[str, Any]) -> bool:
    labels = _normalized_labels(container.get("Labels"))
    if DOCKER_SIMULATION_LABEL_KEY in labels:
        return _safe_bool(labels.get(DOCKER_SIMULATION_LABEL_KEY), default=False)

    role_hint = (
        str(labels.get(DOCKER_SIMULATION_ROLE_LABEL_KEY, "") or "").strip().lower()
    )
    if role_hint in _SIMULATION_ROLE_HINTS:
        return True

    ports = _container_ports(container)
    has_runtime_private_port = any(
        int(row.get("private_port") or 0) == DOCKER_SIMULATION_PRIVATE_PORT_HINT
        for row in ports
    )
    if not has_runtime_private_port:
        return False

    name = _container_name(container).lower()
    image = str(container.get("Image", "") or "").strip().lower()
    service = str(labels.get("com.docker.compose.service", "") or "").strip().lower()
    project = str(labels.get("com.docker.compose.project", "") or "").strip().lower()
    identity = " ".join([name, image, service, project])
    if "eta-mu" in identity:
        return True
    return any(hint in identity for hint in DOCKER_SIMULATION_NAME_HINTS)


def _simulation_endpoints(ports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    endpoints: list[dict[str, Any]] = []
    for row in ports:
        private_port = _safe_int(row.get("private_port"), 0)
        url = str(row.get("url", "") or "").strip()
        if not url:
            continue
        kind = "tcp"
        if private_port == 8787:
            kind = "world"
        elif private_port == 8793:
            kind = "weaver"
        endpoints.append(
            {
                "kind": kind,
                "private_port": private_port,
                "url": url,
            }
        )
    return endpoints


def _selected_labels(labels: dict[str, str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for key, value in labels.items():
        if key.startswith("io.fork_tales.") or key.startswith("com.docker.compose."):
            selected[key] = value
    if DOCKER_SIMULATION_LABEL_KEY in labels:
        selected[DOCKER_SIMULATION_LABEL_KEY] = labels[DOCKER_SIMULATION_LABEL_KEY]
    if DOCKER_SIMULATION_ROLE_LABEL_KEY in labels:
        selected[DOCKER_SIMULATION_ROLE_LABEL_KEY] = labels[
            DOCKER_SIMULATION_ROLE_LABEL_KEY
        ]
    return selected


def _sanitize_simulation_route_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    safe = "".join(
        ch if (ch.isalnum() or ch in {"-", "_", "."}) else "-" for ch in token
    )
    safe = safe.strip("-._")
    return safe


def _simulation_route_payload(*, name: str, service: str) -> dict[str, str]:
    route_id = _sanitize_simulation_route_token(service)
    if not route_id:
        route_id = _sanitize_simulation_route_token(name)

    if not route_id:
        return {
            "id": "",
            "host": "",
            "world_path": "",
            "weaver_path": "",
        }

    base_path = f"/sim/{route_id}"
    return {
        "id": route_id,
        "host": route_id,
        "world_path": f"{base_path}/",
        "weaver_path": f"{base_path}/weaver/",
    }


def _simulation_control_policy(*, service: str, role: str) -> dict[str, Any]:
    clean_service = str(service or "").strip().lower()
    clean_role = str(role or "").strip().lower()

    if clean_role == "world-runtime" or clean_service == "eta-mu-system":
        return {
            "can_start_stop": False,
            "reason": "core_portal_runtime_protected",
        }

    return {
        "can_start_stop": True,
        "reason": "",
    }


def _count_cpuset_cores(cpuset_cpus: str) -> int:
    total = 0
    for token in str(cpuset_cpus or "").split(","):
        chunk = token.strip()
        if not chunk:
            continue
        if "-" in chunk:
            left, _, right = chunk.partition("-")
            start = _safe_int(left, -1)
            end = _safe_int(right, -1)
            if start < 0 or end < 0:
                continue
            if end < start:
                start, end = end, start
            total += max(0, end - start + 1)
            continue
        if _safe_int(chunk, -1) >= 0:
            total += 1
    return total


def _extract_container_limits(inspect_payload: Any) -> dict[str, Any]:
    host_config = (
        inspect_payload.get("HostConfig", {})
        if isinstance(inspect_payload, dict)
        and isinstance(inspect_payload.get("HostConfig"), dict)
        else {}
    )
    nano_cpus = _safe_int(host_config.get("NanoCpus"), 0)
    cpu_quota = _safe_int(host_config.get("CpuQuota"), 0)
    cpu_period = _safe_int(host_config.get("CpuPeriod"), 0)
    cpuset_cpus = str(host_config.get("CpusetCpus", "") or "").strip()
    cpuset_cores = _count_cpuset_cores(cpuset_cpus)

    cpu_limit_cores = 0.0
    if nano_cpus > 0:
        cpu_limit_cores = float(nano_cpus) / 1_000_000_000.0
    elif cpu_quota > 0 and cpu_period > 0:
        cpu_limit_cores = float(cpu_quota) / float(cpu_period)
    elif cpuset_cores > 0:
        cpu_limit_cores = float(cpuset_cores)

    memory_limit_bytes = _safe_int(host_config.get("Memory"), 0)
    memory_reservation_bytes = _safe_int(host_config.get("MemoryReservation"), 0)
    memory_swap_bytes = _safe_int(host_config.get("MemorySwap"), 0)
    pids_limit = _safe_int(host_config.get("PidsLimit"), 0)

    constrained = bool(
        cpu_limit_cores > 0.0
        or memory_limit_bytes > 0
        or pids_limit > 0
        or cpuset_cores > 0
    )
    strict = bool(cpu_limit_cores > 0.0 and memory_limit_bytes > 0 and pids_limit > 0)

    return {
        "cpu_limit_cores": round(cpu_limit_cores, 3) if cpu_limit_cores > 0.0 else None,
        "cpu_period": cpu_period if cpu_period > 0 else None,
        "cpu_quota": cpu_quota if cpu_quota > 0 else None,
        "nano_cpus": nano_cpus if nano_cpus > 0 else None,
        "cpuset_cpus": cpuset_cpus,
        "cpuset_cores": cpuset_cores if cpuset_cores > 0 else None,
        "memory_limit_bytes": memory_limit_bytes if memory_limit_bytes > 0 else None,
        "memory_reservation_bytes": (
            memory_reservation_bytes if memory_reservation_bytes > 0 else None
        ),
        "memory_swap_bytes": memory_swap_bytes if memory_swap_bytes > 0 else None,
        "pids_limit": pids_limit if pids_limit > 0 else None,
        "constrained": constrained,
        "strict": strict,
    }


def _maybe_int(value: Any) -> int | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _compact_text(value: Any, *, limit: int = 220) -> str:
    compact = " ".join(str(value or "").strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: max(1, limit - 1)].rstrip() + "…"


def _lifecycle_snapshot_for_container(
    inspect_payload: Any,
    *,
    pressure_state: str,
) -> dict[str, Any]:
    inspect_row = inspect_payload if isinstance(inspect_payload, dict) else {}
    state = (
        inspect_row.get("State", {})
        if isinstance(inspect_row.get("State"), dict)
        else {}
    )
    health = state.get("Health", {}) if isinstance(state.get("Health"), dict) else {}
    health_logs = health.get("Log", []) if isinstance(health.get("Log"), list) else []
    last_health_log = (
        health_logs[-1] if health_logs and isinstance(health_logs[-1], dict) else {}
    )

    running = bool(state.get("Running", False))
    status = str(state.get("Status", "") or "").strip().lower() or "unknown"
    restart_count = max(
        0,
        _safe_int(
            inspect_row.get("RestartCount", state.get("RestartCount", 0)),
            0,
        ),
    )
    oom_killed = bool(state.get("OOMKilled", False))
    runtime_error = _compact_text(state.get("Error", ""), limit=180)
    health_status = (
        str(health.get("Status", "none") or "none").strip().lower() or "none"
    )
    health_failing_streak = max(0, _safe_int(health.get("FailingStreak", 0), 0))
    last_probe_exit_code = _maybe_int(last_health_log.get("ExitCode"))
    last_probe_output = _compact_text(last_health_log.get("Output", ""), limit=220)

    signals: list[str] = []
    if oom_killed:
        if (
            running
            and status == "running"
            and health_status in {"healthy", "starting", "none"}
        ):
            signals.append("oom_killed_recent")
        else:
            signals.append("oom_killed")
    if health_status == "unhealthy":
        signals.append("health_unhealthy")
    if health_status == "starting" and health_failing_streak > 0:
        signals.append("health_probe_flaky")
    if restart_count > 0:
        signals.append("restarted")
    if runtime_error:
        signals.append("runtime_error")
    if not running or status in {"dead", "exited"}:
        signals.append("not_running")
    if pressure_state == "critical":
        signals.append("resource_pressure_critical")
    elif pressure_state == "warning":
        signals.append("resource_pressure_warning")

    stability = "healthy"
    if any(
        signal
        for signal in signals
        if signal
        in {
            "oom_killed",
            "health_unhealthy",
            "runtime_error",
            "not_running",
            "resource_pressure_critical",
        }
    ):
        stability = "failing"
    elif signals:
        stability = "degraded"

    return {
        "running": running,
        "status": status,
        "restart_count": restart_count,
        "oom_killed": oom_killed,
        "error": runtime_error,
        "health_status": health_status,
        "health_failing_streak": health_failing_streak,
        "last_probe_exit_code": last_probe_exit_code,
        "last_probe_output": last_probe_output,
        "started_at": _compact_text(state.get("StartedAt", ""), limit=64),
        "finished_at": _compact_text(state.get("FinishedAt", ""), limit=64),
        "signals": signals,
        "stability": stability,
    }


def _collect_io_bytes(blkio_payload: Any) -> tuple[int, int]:
    rows = (
        blkio_payload.get("io_service_bytes_recursive", [])
        if isinstance(blkio_payload, dict)
        and isinstance(blkio_payload.get("io_service_bytes_recursive"), list)
        else []
    )
    read_bytes = 0
    write_bytes = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        op = str(row.get("op", "") or "").strip().lower()
        value = _safe_int(row.get("value"), 0)
        if value <= 0:
            continue
        if op == "read":
            read_bytes += value
        elif op == "write":
            write_bytes += value
    return read_bytes, write_bytes


def _collect_network_bytes(networks_payload: Any) -> tuple[int, int]:
    networks = networks_payload if isinstance(networks_payload, dict) else {}
    rx_bytes = 0
    tx_bytes = 0
    for stats in networks.values():
        if not isinstance(stats, dict):
            continue
        rx_bytes += max(0, _safe_int(stats.get("rx_bytes"), 0))
        tx_bytes += max(0, _safe_int(stats.get("tx_bytes"), 0))
    return rx_bytes, tx_bytes


def _ratio(used: float, limit: float) -> float | None:
    if limit <= 0:
        return None
    if used <= 0:
        return 0.0
    return used / limit


def _pressure_state(max_ratio: float | None, *, constrained: bool) -> str:
    if max_ratio is None:
        return "unbounded" if not constrained else "ok"
    if max_ratio >= DOCKER_SIMULATION_RESOURCE_CRITICAL_RATIO:
        return "critical"
    if max_ratio >= DOCKER_SIMULATION_RESOURCE_WARNING_RATIO:
        return "warning"
    return "ok"


def _resource_snapshot_for_container(
    container_id: str,
    *,
    stats_payload: Any,
    inspect_payload: Any,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    del container_id
    limits = _extract_container_limits(inspect_payload)

    cpu_stats = (
        stats_payload.get("cpu_stats", {})
        if isinstance(stats_payload, dict)
        and isinstance(stats_payload.get("cpu_stats"), dict)
        else {}
    )
    precpu_stats = (
        stats_payload.get("precpu_stats", {})
        if isinstance(stats_payload, dict)
        and isinstance(stats_payload.get("precpu_stats"), dict)
        else {}
    )
    cpu_usage = (
        cpu_stats.get("cpu_usage", {})
        if isinstance(cpu_stats.get("cpu_usage"), dict)
        else {}
    )
    precpu_usage = (
        precpu_stats.get("cpu_usage", {})
        if isinstance(precpu_stats.get("cpu_usage"), dict)
        else {}
    )

    cpu_total = _safe_int(cpu_usage.get("total_usage"), 0)
    cpu_total_prev = _safe_int(precpu_usage.get("total_usage"), 0)
    system_total = _safe_int(cpu_stats.get("system_cpu_usage"), 0)
    system_total_prev = _safe_int(precpu_stats.get("system_cpu_usage"), 0)
    cpu_delta = max(0, cpu_total - cpu_total_prev)
    system_delta = max(0, system_total - system_total_prev)

    online_cpus = _safe_int(cpu_stats.get("online_cpus"), 0)
    if online_cpus <= 0:
        percpu_usage = cpu_usage.get("percpu_usage", [])
        if isinstance(percpu_usage, list) and percpu_usage:
            online_cpus = len(percpu_usage)
    if online_cpus <= 0:
        online_cpus = 1

    cpu_percent = 0.0
    if cpu_delta > 0 and system_delta > 0:
        cpu_percent = (
            (float(cpu_delta) / float(system_delta)) * float(online_cpus) * 100.0
        )

    memory_stats = (
        stats_payload.get("memory_stats", {})
        if isinstance(stats_payload, dict)
        and isinstance(stats_payload.get("memory_stats"), dict)
        else {}
    )
    memory_usage_bytes = max(0, _safe_int(memory_stats.get("usage"), 0))
    memory_stats_limit = max(0, _safe_int(memory_stats.get("limit"), 0))
    memory_limit_bytes = _safe_int(limits.get("memory_limit_bytes"), 0)
    effective_memory_limit = (
        memory_limit_bytes if memory_limit_bytes > 0 else memory_stats_limit
    )
    memory_ratio = _ratio(float(memory_usage_bytes), float(effective_memory_limit))

    pids_stats = (
        stats_payload.get("pids_stats", {})
        if isinstance(stats_payload, dict)
        and isinstance(stats_payload.get("pids_stats"), dict)
        else {}
    )
    pids_current = max(0, _safe_int(pids_stats.get("current"), 0))
    pids_limit = _safe_int(limits.get("pids_limit"), 0)
    pids_ratio = _ratio(float(pids_current), float(pids_limit))

    cpu_limit_cores = _safe_float(limits.get("cpu_limit_cores"), 0.0)
    cpu_ratio = None
    if cpu_limit_cores > 0.0:
        cpu_ratio = _ratio(float(cpu_percent), float(cpu_limit_cores) * 100.0)

    rx_bytes, tx_bytes = _collect_network_bytes(
        stats_payload.get("networks", {}) if isinstance(stats_payload, dict) else {}
    )
    blkio_read_bytes, blkio_write_bytes = _collect_io_bytes(
        stats_payload.get("blkio_stats", {}) if isinstance(stats_payload, dict) else {}
    )

    ratios = [
        ratio
        for ratio in [cpu_ratio, memory_ratio, pids_ratio]
        if isinstance(ratio, float)
    ]
    max_ratio = max(ratios) if ratios else None
    constrained = bool(limits.get("constrained", False))

    resource_errors = [str(item) for item in (errors or []) if str(item)]
    if not isinstance(stats_payload, dict):
        resource_errors.append("missing_stats_payload")
    if not isinstance(inspect_payload, dict):
        resource_errors.append("missing_inspect_payload")

    return {
        "observed_at": _now_iso(),
        "limits": limits,
        "usage": {
            "cpu_percent": round(cpu_percent, 2),
            "cpu_cores_estimate": round(cpu_percent / 100.0, 3),
            "memory_usage_bytes": memory_usage_bytes,
            "memory_limit_bytes": (
                effective_memory_limit if effective_memory_limit > 0 else None
            ),
            "memory_percent": (
                round(memory_ratio * 100.0, 2)
                if isinstance(memory_ratio, float)
                else None
            ),
            "pids_current": pids_current,
            "pids_limit": pids_limit if pids_limit > 0 else None,
            "network_rx_bytes": rx_bytes,
            "network_tx_bytes": tx_bytes,
            "blkio_read_bytes": blkio_read_bytes,
            "blkio_write_bytes": blkio_write_bytes,
        },
        "pressure": {
            "cpu_ratio": round(cpu_ratio, 4) if isinstance(cpu_ratio, float) else None,
            "memory_ratio": (
                round(memory_ratio, 4) if isinstance(memory_ratio, float) else None
            ),
            "pids_ratio": round(pids_ratio, 4)
            if isinstance(pids_ratio, float)
            else None,
            "max_ratio": round(max_ratio, 4) if isinstance(max_ratio, float) else None,
            "state": _pressure_state(max_ratio, constrained=constrained),
            "warning_ratio": DOCKER_SIMULATION_RESOURCE_WARNING_RATIO,
            "critical_ratio": DOCKER_SIMULATION_RESOURCE_CRITICAL_RATIO,
        },
        "errors": resource_errors,
    }


def _collect_container_resource_payloads(
    container_ids: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[str]]]:
    unique_ids: list[str] = []
    seen: set[str] = set()
    for container_id in container_ids:
        clean_id = str(container_id or "").strip()
        if not clean_id or clean_id in seen:
            continue
        seen.add(clean_id)
        unique_ids.append(clean_id)

    if len(unique_ids) > DOCKER_SIMULATION_RESOURCE_SAMPLE_LIMIT:
        unique_ids = unique_ids[:DOCKER_SIMULATION_RESOURCE_SAMPLE_LIMIT]

    stats_by_id: dict[str, dict[str, Any]] = {}
    inspect_by_id: dict[str, dict[str, Any]] = {}
    errors_by_id: dict[str, list[str]] = {}

    if not unique_ids:
        return stats_by_id, inspect_by_id, errors_by_id

    worker_count = min(len(unique_ids), DOCKER_SIMULATION_RESOURCE_WORKERS)

    def _load_container_resources(
        container_id: str,
    ) -> tuple[str, Any, Any, str, str]:
        quoted_id = quote(container_id, safe="")
        inspect_payload, inspect_error = _docker_api_json(
            f"/containers/{quoted_id}/json"
        )
        stats_payload, stats_error = _docker_api_json(
            f"/containers/{quoted_id}/stats?stream=0"
        )
        return container_id, inspect_payload, stats_payload, inspect_error, stats_error

    if worker_count <= 1:
        for container_id in unique_ids:
            try:
                (
                    _,
                    inspect_payload,
                    stats_payload,
                    inspect_error,
                    stats_error,
                ) = _load_container_resources(container_id)
            except Exception as exc:
                errors_by_id.setdefault(container_id, []).append(
                    f"resource_collection_error:{exc.__class__.__name__}"
                )
                continue

            if isinstance(inspect_payload, dict):
                inspect_by_id[container_id] = inspect_payload
            elif inspect_error:
                errors_by_id.setdefault(container_id, []).append(inspect_error)

            if isinstance(stats_payload, dict):
                stats_by_id[container_id] = stats_payload
            elif stats_error:
                errors_by_id.setdefault(container_id, []).append(stats_error)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_load_container_resources, container_id): container_id
                for container_id in unique_ids
            }
            for future in as_completed(futures):
                container_id = futures[future]
                try:
                    (
                        _,
                        inspect_payload,
                        stats_payload,
                        inspect_error,
                        stats_error,
                    ) = future.result()
                except Exception as exc:
                    errors_by_id.setdefault(container_id, []).append(
                        f"resource_collection_error:{exc.__class__.__name__}"
                    )
                    continue

                if isinstance(inspect_payload, dict):
                    inspect_by_id[container_id] = inspect_payload
                elif inspect_error:
                    errors_by_id.setdefault(container_id, []).append(inspect_error)

                if isinstance(stats_payload, dict):
                    stats_by_id[container_id] = stats_payload
                elif stats_error:
                    errors_by_id.setdefault(container_id, []).append(stats_error)

    return stats_by_id, inspect_by_id, errors_by_id


def _simulation_entry(
    container: dict[str, Any],
    *,
    stats_by_id: dict[str, dict[str, Any]] | None = None,
    inspect_by_id: dict[str, dict[str, Any]] | None = None,
    resource_errors_by_id: dict[str, list[str]] | None = None,
) -> dict[str, Any] | None:
    if not _is_simulation_container(container):
        return None

    labels = _normalized_labels(container.get("Labels"))
    ports = _container_ports(container)
    container_id = str(container.get("Id", "") or "").strip()
    if not container_id:
        return None

    stats_payload = (
        stats_by_id.get(container_id) if isinstance(stats_by_id, dict) else None
    )
    inspect_payload = (
        inspect_by_id.get(container_id) if isinstance(inspect_by_id, dict) else None
    )
    resource_errors = (
        resource_errors_by_id.get(container_id, [])
        if isinstance(resource_errors_by_id, dict)
        else []
    )
    resource_snapshot = _resource_snapshot_for_container(
        container_id,
        stats_payload=stats_payload,
        inspect_payload=inspect_payload,
        errors=resource_errors,
    )
    pressure_state = (
        str(((resource_snapshot.get("pressure") or {}).get("state") or ""))
        .strip()
        .lower()
    )
    lifecycle_snapshot = _lifecycle_snapshot_for_container(
        inspect_payload,
        pressure_state=pressure_state,
    )

    role = str(labels.get(DOCKER_SIMULATION_ROLE_LABEL_KEY, "") or "").strip().lower()
    if not role:
        role = "simulation"

    name = _container_name(container)
    service = str(labels.get("com.docker.compose.service", "") or "").strip()
    route = _simulation_route_payload(name=name, service=service)
    control = _simulation_control_policy(service=service, role=role)

    return {
        "id": container_id,
        "short_id": container_id[:12],
        "name": name,
        "service": service,
        "project": str(labels.get("com.docker.compose.project", "") or "").strip(),
        "image": str(container.get("Image", "") or "").strip(),
        "state": str(container.get("State", "") or "").strip() or "unknown",
        "status": str(container.get("Status", "") or "").strip(),
        "role": role,
        "created_at": _container_created_at(container),
        "ports": ports,
        "endpoints": _simulation_endpoints(ports),
        "networks": _container_networks(container),
        "labels": _selected_labels(labels),
        "resources": resource_snapshot,
        "lifecycle": lifecycle_snapshot,
        "route": route,
        "control": control,
        "awareness": {
            "network_count": 0,
            "peer_count": 0,
            "peer_ids": [],
            "peer_names": [],
        },
    }


def _attach_awareness(simulations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {
        str(row.get("id", "")): row for row in simulations if str(row.get("id", ""))
    }
    by_network: dict[str, list[str]] = {}
    for row in simulations:
        for network_name in row.get("networks", []):
            clean_network = str(network_name or "").strip()
            if not clean_network:
                continue
            by_network.setdefault(clean_network, []).append(str(row.get("id", "")))

    for row in simulations:
        own_id = str(row.get("id", ""))
        peers: set[str] = set()
        networks = [
            str(item or "").strip()
            for item in row.get("networks", [])
            if str(item or "").strip()
        ]
        for network_name in networks:
            for peer_id in by_network.get(network_name, []):
                if peer_id and peer_id != own_id:
                    peers.add(peer_id)
        peer_ids = sorted(peers)
        peer_names = [
            str(by_id.get(peer_id, {}).get("name", peer_id) or peer_id)
            for peer_id in peer_ids
        ]
        row["awareness"] = {
            "network_count": len(networks),
            "peer_count": len(peer_ids),
            "peer_ids": peer_ids,
            "peer_names": peer_names,
        }
    return simulations


def _network_clusters(simulations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_network: dict[str, list[dict[str, Any]]] = {}
    for row in simulations:
        for network_name in row.get("networks", []):
            clean_network = str(network_name or "").strip()
            if not clean_network:
                continue
            by_network.setdefault(clean_network, []).append(row)

    clusters: list[dict[str, Any]] = []
    for network_name, members in by_network.items():
        ordered = sorted(
            members,
            key=lambda row: (
                str(row.get("project", "") or ""),
                str(row.get("service", "") or ""),
                str(row.get("name", "") or ""),
            ),
        )
        clusters.append(
            {
                "network": network_name,
                "simulation_count": len(ordered),
                "simulations": [
                    {
                        "id": str(item.get("id", "") or ""),
                        "name": str(item.get("name", "") or ""),
                        "service": str(item.get("service", "") or ""),
                        "project": str(item.get("project", "") or ""),
                    }
                    for item in ordered
                ],
            }
        )

    clusters.sort(
        key=lambda row: (
            -_safe_int(row.get("simulation_count"), 0),
            str(row.get("network", "") or ""),
        )
    )
    return clusters


def _snapshot_fingerprint(
    simulations: list[dict[str, Any]],
    *,
    source_error: str = "",
) -> str:
    seed_rows = [
        {
            "id": str(row.get("id", "") or ""),
            "state": str(row.get("state", "") or ""),
            "status": str(row.get("status", "") or ""),
            "networks": list(row.get("networks", [])),
            "ports": list(row.get("ports", [])),
            "peers": list((row.get("awareness") or {}).get("peer_ids", [])),
            "resource": {
                "cpu_percent": _safe_float(
                    ((row.get("resources") or {}).get("usage") or {}).get(
                        "cpu_percent"
                    ),
                    0.0,
                ),
                "memory_usage_bytes": _safe_int(
                    ((row.get("resources") or {}).get("usage") or {}).get(
                        "memory_usage_bytes"
                    ),
                    0,
                ),
                "pids_current": _safe_int(
                    ((row.get("resources") or {}).get("usage") or {}).get(
                        "pids_current"
                    ),
                    0,
                ),
                "pressure_state": str(
                    ((row.get("resources") or {}).get("pressure") or {}).get(
                        "state", ""
                    )
                    or ""
                ),
            },
            "lifecycle": {
                "stability": str(((row.get("lifecycle") or {}).get("stability") or "")),
                "health_status": str(
                    ((row.get("lifecycle") or {}).get("health_status") or "")
                ),
                "restart_count": _safe_int(
                    ((row.get("lifecycle") or {}).get("restart_count") or 0),
                    0,
                ),
                "oom_killed": bool(
                    (row.get("lifecycle") or {}).get("oom_killed", False)
                ),
            },
        }
        for row in simulations
    ]
    seed_rows.sort(key=lambda row: row.get("id", ""))
    seed_blob = json.dumps(
        {
            "error": str(source_error or ""),
            "rows": seed_rows,
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(seed_blob.encode("ascii", errors="ignore")).hexdigest()


def build_docker_simulation_snapshot(
    containers: list[dict[str, Any]],
    *,
    generated_at: str | None = None,
    source_error: str = "",
    stats_by_id: dict[str, dict[str, Any]] | None = None,
    inspect_by_id: dict[str, dict[str, Any]] | None = None,
    resource_errors_by_id: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    simulations = [
        entry
        for entry in (
            _simulation_entry(
                row,
                stats_by_id=stats_by_id,
                inspect_by_id=inspect_by_id,
                resource_errors_by_id=resource_errors_by_id,
            )
            for row in containers
        )
        if isinstance(entry, dict)
    ]
    simulations.sort(
        key=lambda row: (
            str(row.get("project", "") or ""),
            str(row.get("service", "") or ""),
            str(row.get("name", "") or ""),
        )
    )
    _attach_awareness(simulations)
    clusters = _network_clusters(simulations)

    running_simulations = [
        row
        for row in simulations
        if str(row.get("state", "") or "").strip().lower() == "running"
    ]

    total_cpu_percent = 0.0
    total_memory_usage_bytes = 0
    total_memory_limit_bytes = 0
    constrained_simulations = 0
    strict_simulations = 0
    warning_simulations = 0
    critical_simulations = 0
    failing_simulations = 0
    degraded_simulations = 0
    healthy_simulations = 0
    oom_killed_simulations = 0
    unhealthy_health_simulations = 0
    restarted_simulations = 0

    for row in simulations:
        resources = (
            row.get("resources", {}) if isinstance(row.get("resources"), dict) else {}
        )
        usage = (
            resources.get("usage", {})
            if isinstance(resources.get("usage"), dict)
            else {}
        )
        limits = (
            resources.get("limits", {})
            if isinstance(resources.get("limits"), dict)
            else {}
        )
        pressure = (
            resources.get("pressure", {})
            if isinstance(resources.get("pressure"), dict)
            else {}
        )

        total_cpu_percent += max(0.0, _safe_float(usage.get("cpu_percent"), 0.0))
        total_memory_usage_bytes += max(
            0, _safe_int(usage.get("memory_usage_bytes"), 0)
        )
        memory_limit = max(0, _safe_int(usage.get("memory_limit_bytes"), 0))
        if memory_limit > 0:
            total_memory_limit_bytes += memory_limit

        if bool(limits.get("constrained", False)):
            constrained_simulations += 1
        if bool(limits.get("strict", False)):
            strict_simulations += 1

        pressure_state = str(pressure.get("state", "") or "").strip().lower()
        if pressure_state == "warning":
            warning_simulations += 1
        elif pressure_state == "critical":
            critical_simulations += 1

        lifecycle = (
            row.get("lifecycle", {}) if isinstance(row.get("lifecycle"), dict) else {}
        )
        stability = str(lifecycle.get("stability", "healthy") or "healthy").lower()
        if stability == "failing":
            failing_simulations += 1
        elif stability == "degraded":
            degraded_simulations += 1
        else:
            healthy_simulations += 1

        if bool(lifecycle.get("oom_killed", False)):
            oom_killed_simulations += 1
        if str(lifecycle.get("health_status", "") or "").strip().lower() == "unhealthy":
            unhealthy_health_simulations += 1
        if _safe_int(lifecycle.get("restart_count"), 0) > 0:
            restarted_simulations += 1

    unconstrained_simulations = max(0, len(simulations) - constrained_simulations)
    generated = generated_at or _now_iso()
    fingerprint = _snapshot_fingerprint(simulations, source_error=source_error)

    return {
        "ok": not bool(source_error),
        "record": "eta-mu.docker-simulations.v1",
        "generated_at": generated,
        "stale": False,
        "fingerprint": fingerprint,
        "summary": {
            "running_simulations": len(running_simulations),
            "total_simulations": len(simulations),
            "network_clusters": len(clusters),
            "constrained_simulations": constrained_simulations,
            "strict_simulations": strict_simulations,
            "unconstrained_simulations": unconstrained_simulations,
            "warning_simulations": warning_simulations,
            "critical_simulations": critical_simulations,
            "healthy_simulations": healthy_simulations,
            "degraded_simulations": degraded_simulations,
            "failing_simulations": failing_simulations,
            "oom_killed_simulations": oom_killed_simulations,
            "health_unhealthy_simulations": unhealthy_health_simulations,
            "restarted_simulations": restarted_simulations,
            "total_cpu_percent": round(total_cpu_percent, 2),
            "total_memory_usage_bytes": total_memory_usage_bytes,
            "total_memory_limit_bytes": (
                total_memory_limit_bytes if total_memory_limit_bytes > 0 else None
            ),
        },
        "source": {
            "mode": "docker-socket",
            "socket_path": DOCKER_SIMULATION_SOCKET_PATH,
            "label_key": DOCKER_SIMULATION_LABEL_KEY,
            "role_label_key": DOCKER_SIMULATION_ROLE_LABEL_KEY,
            "name_hints": list(DOCKER_SIMULATION_NAME_HINTS),
            "private_port_hint": DOCKER_SIMULATION_PRIVATE_PORT_HINT,
            "resource_workers": DOCKER_SIMULATION_RESOURCE_WORKERS,
            "resource_sample_limit": DOCKER_SIMULATION_RESOURCE_SAMPLE_LIMIT,
            "error": str(source_error or ""),
        },
        "resource_policy": {
            "warning_ratio": DOCKER_SIMULATION_RESOURCE_WARNING_RATIO,
            "critical_ratio": DOCKER_SIMULATION_RESOURCE_CRITICAL_RATIO,
            "strict_limit_contract": {
                "cpu_limit_required": True,
                "memory_limit_required": True,
                "pids_limit_required": True,
            },
        },
        "discovery_contract": {
            "label_key": DOCKER_SIMULATION_LABEL_KEY,
            "role_label_key": DOCKER_SIMULATION_ROLE_LABEL_KEY,
            "network_hint": DOCKER_SIMULATION_NETWORK_HINT,
            "example_labels": {
                DOCKER_SIMULATION_LABEL_KEY: "true",
                DOCKER_SIMULATION_ROLE_LABEL_KEY: "experiment",
                "io.fork_tales.simulation.name": "eta-mu-exp-a",
            },
        },
        "simulations": simulations,
        "clusters": clusters,
    }


def collect_docker_simulation_snapshot(
    *, force_refresh: bool = False
) -> dict[str, Any]:
    global _DOCKER_SIMULATION_CACHE_REFRESH_IN_FLIGHT

    now_monotonic = time.monotonic()
    with _DOCKER_SIMULATION_CACHE_CONDITION:
        cached_payload = _DOCKER_SIMULATION_CACHE.get("payload")
        cached_monotonic = float(_DOCKER_SIMULATION_CACHE.get("monotonic", 0.0) or 0.0)
        if (
            not force_refresh
            and isinstance(cached_payload, dict)
            and now_monotonic - cached_monotonic <= DOCKER_SIMULATION_CACHE_SECONDS
        ):
            return copy.deepcopy(cached_payload)

        if _DOCKER_SIMULATION_CACHE_REFRESH_IN_FLIGHT:
            wait_deadline = now_monotonic + DOCKER_SIMULATION_CACHE_WAIT_SECONDS
            while _DOCKER_SIMULATION_CACHE_REFRESH_IN_FLIGHT:
                remaining = wait_deadline - time.monotonic()
                if remaining <= 0:
                    break
                _DOCKER_SIMULATION_CACHE_CONDITION.wait(timeout=remaining)

            cached_payload = _DOCKER_SIMULATION_CACHE.get("payload")
            cached_monotonic = float(
                _DOCKER_SIMULATION_CACHE.get("monotonic", 0.0) or 0.0
            )
            if isinstance(cached_payload, dict):
                cache_age = time.monotonic() - cached_monotonic
                if cache_age <= max(
                    DOCKER_SIMULATION_CACHE_SECONDS,
                    DOCKER_SIMULATION_CACHE_WAIT_SECONDS,
                ):
                    return copy.deepcopy(cached_payload)
                if not force_refresh:
                    return copy.deepcopy(cached_payload)

            if _DOCKER_SIMULATION_CACHE_REFRESH_IN_FLIGHT:
                return build_docker_simulation_snapshot(
                    [],
                    generated_at=_now_iso(),
                    source_error="docker_snapshot_refresh_busy",
                )

        _DOCKER_SIMULATION_CACHE_REFRESH_IN_FLIGHT = True

    try:
        raw_containers, error = _docker_api_json("/containers/json?all=1")
        if error:
            failed_snapshot = build_docker_simulation_snapshot(
                [],
                generated_at=_now_iso(),
                source_error=error,
            )
            with _DOCKER_SIMULATION_CACHE_CONDITION:
                cached_payload = _DOCKER_SIMULATION_CACHE.get("payload")
                if isinstance(cached_payload, dict):
                    stale = copy.deepcopy(cached_payload)
                    stale["ok"] = False
                    stale["stale"] = True
                    stale["error"] = error
                    stale["generated_at"] = failed_snapshot.get(
                        "generated_at", _now_iso()
                    )
                    stale["fingerprint"] = failed_snapshot.get(
                        "fingerprint", stale.get("fingerprint", "")
                    )
                    source_block = stale.get("source")
                    if not isinstance(source_block, dict):
                        source_block = {}
                        stale["source"] = source_block
                    source_block["error"] = error
                    source_block["stale_from_cache"] = True
                    return stale
            return failed_snapshot

        if not isinstance(raw_containers, list):
            return build_docker_simulation_snapshot(
                [],
                generated_at=_now_iso(),
                source_error="docker_api_invalid_payload",
            )

        containers = [row for row in raw_containers if isinstance(row, dict)]

        simulation_ids = [
            str(row.get("Id", "") or "").strip()
            for row in containers
            if _is_simulation_container(row)
        ]
        stats_by_id, inspect_by_id, resource_errors_by_id = (
            _collect_container_resource_payloads(simulation_ids)
        )

        snapshot = build_docker_simulation_snapshot(
            containers,
            generated_at=_now_iso(),
            source_error="",
            stats_by_id=stats_by_id,
            inspect_by_id=inspect_by_id,
            resource_errors_by_id=resource_errors_by_id,
        )
        with _DOCKER_SIMULATION_CACHE_CONDITION:
            _DOCKER_SIMULATION_CACHE["payload"] = copy.deepcopy(snapshot)
            _DOCKER_SIMULATION_CACHE["monotonic"] = now_monotonic
        return snapshot
    finally:
        with _DOCKER_SIMULATION_CACHE_CONDITION:
            _DOCKER_SIMULATION_CACHE_REFRESH_IN_FLIGHT = False
            _DOCKER_SIMULATION_CACHE_CONDITION.notify_all()


def reset_docker_simulation_cache_for_tests() -> None:
    global _DOCKER_SIMULATION_CACHE_REFRESH_IN_FLIGHT
    with _DOCKER_SIMULATION_CACHE_CONDITION:
        _DOCKER_SIMULATION_CACHE["payload"] = None
        _DOCKER_SIMULATION_CACHE["monotonic"] = 0.0
        _DOCKER_SIMULATION_CACHE_REFRESH_IN_FLIGHT = False
        _DOCKER_SIMULATION_CACHE_CONDITION.notify_all()
