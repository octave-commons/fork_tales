from __future__ import annotations
import os
import time
import shutil
import subprocess
import threading
import hashlib
import re
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

from .constants import (
    RESOURCE_LOG_TAIL_MAX_BYTES,
    RESOURCE_LOG_TAIL_MAX_LINES,
    RESOURCE_SNAPSHOT_CACHE_SECONDS,
    FILE_SENTINEL_PROFILE,
    _RESOURCE_MONITOR_LOCK,
    _RESOURCE_MONITOR_CACHE,
    _RESOURCE_HEARTBEAT_INGEST_LOCK,
    _RESOURCE_HEARTBEAT_INGEST_CACHE,
    _OPENVINO_EMBED_LOCK,
    _OPENVINO_EMBED_RUNTIME,
)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _status_from_utilization(
    utilization: float,
    *,
    watch_threshold: float = 72.0,
    hot_threshold: float = 90.0,
) -> str:
    bounded = max(0.0, min(100.0, float(utilization)))
    if bounded >= hot_threshold:
        return "hot"
    if bounded >= watch_threshold:
        return "watch"
    return "ok"


def _safe_env_metric(name: str, default: float = 0.0) -> float:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _parse_proc_meminfo_mb() -> tuple[float, float]:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists() or not meminfo_path.is_file():
        return 0.0, 0.0
    total_kb = 0.0
    available_kb = 0.0
    try:
        for raw_line in meminfo_path.read_text("utf-8").splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, remainder = line.split(":", 1)
            token = remainder.strip().split(" ", 1)[0]
            try:
                value = float(token)
            except (TypeError, ValueError):
                continue
            if key == "MemTotal":
                total_kb = value
            elif key == "MemAvailable":
                available_kb = value
    except OSError:
        return 0.0, 0.0
    return (total_kb / 1024.0), (available_kb / 1024.0)


def _cpu_percent_per_core_snapshot(
    *,
    cpu_count: int,
    fallback_utilization: float,
) -> list[float]:
    if psutil is not None:
        try:
            return [
                max(0.0, min(100.0, _safe_float(value, 0.0)))
                for value in psutil.cpu_percent(percpu=True)
            ]
        except Exception:
            pass

    stat_path = Path("/proc/stat")
    if stat_path.exists() and stat_path.is_file():
        rows: list[float] = []
        try:
            for raw_line in stat_path.read_text("utf-8").splitlines():
                line = raw_line.strip()
                if not line.startswith("cpu"):
                    continue
                prefix, _, remainder = line.partition(" ")
                if prefix == "cpu":
                    continue
                if not prefix[3:].isdigit():
                    continue
                tokens = [token for token in remainder.split(" ") if token]
                if len(tokens) < 4:
                    continue
                values = [max(0.0, _safe_float(token, 0.0)) for token in tokens[:8]]
                total = sum(values)
                idle = values[3] + (values[4] if len(values) > 4 else 0.0)
                if total <= 0.0:
                    rows.append(0.0)
                    continue
                rows.append(max(0.0, min(100.0, ((total - idle) / total) * 100.0)))
            if rows:
                return rows
        except Exception:
            pass

    approx = max(0.0, min(100.0, _safe_float(fallback_utilization, 0.0)))
    return [approx for _ in range(max(1, int(cpu_count)))]


def _tail_text_lines(
    path: Path,
    *,
    max_bytes: int,
    max_lines: int,
) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            data = handle.read(max_bytes)
    except OSError:
        return []

    text = data.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    return lines[-max_lines:]


def _collect_nvidia_metrics() -> list[dict[str, float]]:
    nvidia_smi_bin = shutil.which("nvidia-smi")
    if not nvidia_smi_bin:
        return []
    cmd = [
        nvidia_smi_bin,
        "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=0.45,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []

    rows: list[dict[str, float]] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            utilization = max(0.0, min(100.0, float(parts[0])))
        except (TypeError, ValueError):
            utilization = 0.0
        try:
            memory = max(0.0, min(100.0, float(parts[1])))
        except (TypeError, ValueError):
            memory = 0.0
        try:
            temperature = max(0.0, float(parts[2]))
        except (TypeError, ValueError):
            temperature = 0.0
        rows.append(
            {
                "utilization": round(utilization, 2),
                "memory": round(memory, 2),
                "temperature": round(temperature, 2),
            }
        )
    return rows


def _resource_log_watch(part_root: Path | None = None) -> dict[str, Any]:
    if part_root is None:
        return {
            "path": "",
            "line_count": 0,
            "error_count": 0,
            "warn_count": 0,
            "error_ratio": 0.0,
            "warn_ratio": 0.0,
            "latest": "",
        }

    candidates = [
        part_root / "world_state" / "weaver-error.log",
        part_root / "world_state" / "weaver-out.log",
        part_root / "world_state" / "world-web.log",
    ]
    log_path = next(
        (path for path in candidates if path.exists() and path.is_file()), None
    )
    if log_path is None:
        return {
            "path": "",
            "line_count": 0,
            "error_count": 0,
            "warn_count": 0,
            "error_ratio": 0.0,
            "warn_ratio": 0.0,
            "latest": "",
        }

    tail = _tail_text_lines(
        log_path,
        max_bytes=RESOURCE_LOG_TAIL_MAX_BYTES,
        max_lines=RESOURCE_LOG_TAIL_MAX_LINES,
    )
    lowered = [line.lower() for line in tail]
    error_tokens = ("error", "traceback", "exception", "fatal", "panic")
    warn_tokens = ("warn", "warning", "timeout", "retry", "blocked")
    error_count = sum(
        1 for line in lowered if any(token in line for token in error_tokens)
    )
    warn_count = sum(
        1 for line in lowered if any(token in line for token in warn_tokens)
    )
    line_count = len(tail)
    latest = tail[-1] if tail else ""

    return {
        "path": str(log_path),
        "line_count": line_count,
        "error_count": error_count,
        "warn_count": warn_count,
        "error_ratio": round(error_count / max(1, line_count), 4),
        "warn_ratio": round(warn_count / max(1, line_count), 4),
        "latest": latest[:240],
    }


def _split_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_-]+", text.lower()) if token]


def _eta_mu_percentile(values: list[float], p: float) -> float:
    import math

    if not values:
        return 0.0
    bounded = sorted(float(item) for item in values)
    ratio = _clamp01(p)
    if len(bounded) == 1:
        return bounded[0]
    index = ratio * (len(bounded) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return bounded[lower]
    weight = index - lower
    return bounded[lower] * (1.0 - weight) + bounded[upper] * weight


def _infer_eta_mu_field_scores(
    *,
    rel_path: str,
    kind: str,
    text_excerpt: str,
) -> dict[str, float]:
    from .constants import FIELD_TO_PRESENCE, ETA_MU_FIELD_KEYWORDS

    scores = {field_id: 0.0 for field_id in FIELD_TO_PRESENCE}

    kind_key = kind.strip().lower()
    if kind_key in {"audio", "image", "video"}:
        scores["f1"] += 0.42
        scores["f6"] += 0.08
    elif kind_key == "text":
        # Rebalanced: spread text file bias across multiple fields
        # instead of overwhelming f6 (mage_of_receipts)
        scores["f3"] += 0.18  # coherence/focus - generic docs
        scores["f6"] += 0.14  # creative synthesis - reduced from 0.46
        scores["f7"] += 0.12  # truth/contract - spec/docs
        scores["f8"] += 0.08  # runtime/ops - config/scripts
    else:
        scores["f4"] += 0.24
        scores["f8"] += 0.18

    rel_lower = rel_path.lower()
    if rel_lower.endswith(".zip"):
        scores["f5"] += 0.24
        scores["f4"] += 0.12
    if rel_lower.endswith(".lisp"):
        # Lisp files: contracts/truth (f7) + creative (f6)
        scores["f7"] += 0.22
        scores["f6"] += 0.14
    if rel_lower.endswith(".md") or rel_lower.endswith(".txt"):
        # Markdown: spread between coherence (f3) and truth (f7)
        scores["f3"] += 0.14
        scores["f7"] += 0.12

    # Path-based routing hints for better distribution
    if "/test" in rel_lower or "/tests" in rel_lower or rel_lower.startswith("test_"):
        scores["f7"] += 0.16  # validation/truth
    if (
        "/ops" in rel_lower
        or "/deploy" in rel_lower
        or ".yml" in rel_lower
        or ".yaml" in rel_lower
    ):
        scores["f8"] += 0.18  # runtime/ops
    if "/docs" in rel_lower or "readme" in rel_lower or "guide" in rel_lower:
        scores["f3"] += 0.12  # coherence/reference
    if "/.git" in rel_lower or ".gitmodules" in rel_lower or ".gitignore" in rel_lower:
        scores["f8"] += 0.20  # ops/process
    if "receipt" in rel_lower or "ledger" in rel_lower or "contract" in rel_lower:
        scores["f7"] += 0.18  # truth/contracts
    if "audit" in rel_lower or "review" in rel_lower:
        scores["f2"] += 0.14  # witness/thread
    if "wip" in rel_lower or "draft" in rel_lower or "tmp" in rel_lower:
        scores["f4"] += 0.16  # drift/delta
    if "spec" in rel_lower or "specs" in rel_lower or "design" in rel_lower:
        scores["f3"] += 0.14  # coherence/focus
    if (
        "story" in rel_lower
        or "lyric" in rel_lower
        or "song" in rel_lower
        or "creative" in rel_lower
    ):
        scores["f6"] += 0.16  # creative synthesis

    # Philosophical concept routing hints
    if "ethic" in rel_lower or "moral" in rel_lower or "virtue" in rel_lower:
        scores["f9"] += 0.20  # good/virtue
    if "corrupt" in rel_lower or "sin" in rel_lower or "wicked" in rel_lower:
        scores["f10"] += 0.20  # evil/corruption
    if "justice" in rel_lower or "law" in rel_lower or "right" in rel_lower:
        scores["f11"] += 0.18  # right/justice
    if "wrong" in rel_lower or "error" in rel_lower or "false" in rel_lower:
        scores["f12"] += 0.18  # wrong/error
    if "death" in rel_lower or "mortal" in rel_lower or "legacy" in rel_lower:
        scores["f13"] += 0.20  # dead/finality
    if "life" in rel_lower or "bio" in rel_lower or "vital" in rel_lower:
        scores["f14"] += 0.20  # living/vitality
    if "chaos" in rel_lower or "random" in rel_lower or "noise" in rel_lower:
        scores["f15"] += 0.24  # chaos/unpredictability

    combined = f"{rel_path} {text_excerpt}"
    tokens = _clean_tokens(combined)
    for token in tokens:
        for field_id, keywords in ETA_MU_FIELD_KEYWORDS.items():
            if token in keywords:
                scores[field_id] += 0.07

    for field_id, keywords in ETA_MU_FIELD_KEYWORDS.items():
        if any(keyword in rel_lower for keyword in keywords):
            scores[field_id] += 0.12

    total = sum(max(0.0, value) for value in scores.values())
    if total <= 0:
        fallback = "f6" if kind_key == "text" else "f1"
        scores[fallback] = 1.0
        return scores

    normalized: dict[str, float] = {}
    for field_id, value in scores.items():
        normalized[field_id] = round(max(0.0, value) / total, 4)

    if all(value <= 0.0 for value in normalized.values()):
        fallback = "f6" if kind_key == "text" else "f1"
        normalized[fallback] = 1.0
    return normalized


def _dominant_eta_mu_field(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "f6", 1.0
    dominant_field = max(
        scores.keys(), key=lambda key: _safe_float(scores.get(key, 0.0), 0.0)
    )
    return dominant_field, _safe_float(scores.get(dominant_field, 0.0), 0.0)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_deep_clone(payload: Any) -> Any:
    import json

    return json.loads(json.dumps(payload))


def _stable_ratio(seed: str, offset: int = 0) -> float:
    digest = hashlib.sha256(f"{seed}|{offset}".encode("utf-8")).digest()
    return int.from_bytes(digest[:2], "big") / 65535.0


def _normalize_field_scores(
    scores: dict[str, float], *, fallback_field: str = "f3"
) -> dict[str, float]:
    from .constants import FIELD_TO_PRESENCE

    normalized = {
        field_id: max(0.0, _safe_float(scores.get(field_id, 0.0), 0.0))
        for field_id in FIELD_TO_PRESENCE
    }
    total = sum(normalized.values())
    if total <= 0.0:
        fallback = fallback_field if fallback_field in FIELD_TO_PRESENCE else "f3"
        normalized[fallback] = 1.0
        total = 1.0
    return {field_id: round(value / total, 4) for field_id, value in normalized.items()}


def _resource_auto_embedding_order(
    snapshot: dict[str, Any] | None = None,
) -> list[str]:
    payload = snapshot if isinstance(snapshot, dict) else _resource_monitor_snapshot()
    devices = payload.get("devices", {}) if isinstance(payload, dict) else {}
    npu = devices.get("npu0", {}) if isinstance(devices, dict) else {}
    cpu = devices.get("cpu", {}) if isinstance(devices, dict) else {}
    gpu1 = devices.get("gpu1", {}) if isinstance(devices, dict) else {}

    npu_status = str(npu.get("status", "ok")).strip().lower()
    cpu_utilization = _safe_float(cpu.get("utilization", 0.0), 0.0)
    gpu_utilization = _safe_float(gpu1.get("utilization", 0.0), 0.0)
    openvino_ready = bool(str(os.getenv("OPENVINO_EMBED_ENDPOINT", "") or "").strip())
    if not openvino_ready:
        openvino_device = str(
            os.getenv("OPENVINO_EMBED_DEVICE", "NPU") or "NPU"
        ).upper()
        openvino_ready = "NPU" in openvino_device

    order: list[str] = []
    if openvino_ready and npu_status != "hot":
        order.append("openvino")
    if cpu_utilization < 95.0:
        order.append("tensorflow")
    if gpu_utilization < 92.0:
        order.append("ollama")
    order.extend(["tensorflow", "ollama", "openvino"])

    deduped: list[str] = []
    seen: set[str] = set()
    for item in order:
        key = str(item).strip().lower()
        if key not in {"openvino", "tensorflow", "ollama"}:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _resource_auto_text_order(
    snapshot: dict[str, Any] | None = None,
) -> list[str]:
    payload = snapshot if isinstance(snapshot, dict) else _resource_monitor_snapshot()
    devices = payload.get("devices", {}) if isinstance(payload, dict) else {}
    cpu = devices.get("cpu", {}) if isinstance(devices, dict) else {}
    gpu1 = devices.get("gpu1", {}) if isinstance(devices, dict) else {}
    log_watch = payload.get("log_watch", {}) if isinstance(payload, dict) else {}

    cpu_utilization = _safe_float(cpu.get("utilization", 0.0), 0.0)
    gpu_utilization = _safe_float(gpu1.get("utilization", 0.0), 0.0)
    error_ratio = _safe_float(log_watch.get("error_ratio", 0.0), 0.0)

    if cpu_utilization >= 88.0 or error_ratio >= 0.5:
        preferred = ["ollama", "tensorflow"]
    elif gpu_utilization < 85.0:
        preferred = ["ollama", "tensorflow"]
    else:
        preferred = ["tensorflow", "ollama"]

    deduped: list[str] = []
    seen: set[str] = set()
    for item in [*preferred, "tensorflow", "ollama"]:
        key = str(item).strip().lower()
        if key not in {"tensorflow", "ollama"}:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


_NPU_LAST_BUSY_US = 0
_NPU_LAST_CHECK_TIME = 0.0
_NPU_LOCK = threading.Lock()


def _collect_npu_metrics() -> dict[str, Any]:
    global _NPU_LAST_BUSY_US, _NPU_LAST_CHECK_TIME
    npu_base = Path("/sys/devices/pci0000:00/0000:00:0b.0")
    if not npu_base.exists():
        return {}

    try:
        busy_raw = npu_base.joinpath("npu_busy_time_us").read_text().strip()
        freq_cur = npu_base.joinpath("npu_current_frequency_mhz").read_text().strip()
        freq_max = npu_base.joinpath("npu_max_frequency_mhz").read_text().strip()
        mem_raw = npu_base.joinpath("npu_memory_utilization").read_text().strip()

        busy_us = _safe_int(busy_raw, 0)
        now = time.time()

        utilization = 0.0
        with _NPU_LOCK:
            if _NPU_LAST_CHECK_TIME > 0:
                dt = now - _NPU_LAST_CHECK_TIME
                dbusy = busy_us - _NPU_LAST_BUSY_US
                if dt > 0:
                    # utilization = (busy_us delta / elapsed_us) * 100
                    utilization = (dbusy / (dt * 1_000_000.0)) * 100.0

            _NPU_LAST_BUSY_US = busy_us
            _NPU_LAST_CHECK_TIME = now

        return {
            "utilization": round(_clamp01(utilization / 100.0) * 100.0, 2),
            "busy_time_us": busy_us,
            "current_freq_mhz": _safe_int(freq_cur, 0),
            "max_freq_mhz": _safe_int(freq_max, 0),
            "memory_bytes": _safe_int(mem_raw, 0),
        }
    except Exception:
        return {}


def _collect_intel_gpu_metrics() -> dict[str, Any]:
    intel_base = Path("/sys/class/drm/card1")  # Based on discovery
    if not intel_base.exists():
        return {}

    try:
        freq_cur = intel_base.joinpath("gt_cur_freq_mhz").read_text().strip()
        freq_max = intel_base.joinpath("gt_max_freq_mhz").read_text().strip()
        return {
            "current_freq_mhz": _safe_int(freq_cur, 0),
            "max_freq_mhz": _safe_int(freq_max, 0),
        }
    except Exception:
        return {}


def _resource_monitor_snapshot(part_root: Path | None = None) -> dict[str, Any]:
    now_monotonic = time.monotonic()
    part_key = str(part_root.resolve()) if isinstance(part_root, Path) else ""

    with _RESOURCE_MONITOR_LOCK:
        cached_checked = _safe_float(
            _RESOURCE_MONITOR_CACHE.get("checked_monotonic", 0.0), 0.0
        )
        cached_part = str(_RESOURCE_MONITOR_CACHE.get("part_root", ""))
        cached_snapshot = _RESOURCE_MONITOR_CACHE.get("snapshot")
        if (
            isinstance(cached_snapshot, dict)
            and (now_monotonic - cached_checked) <= RESOURCE_SNAPSHOT_CACHE_SECONDS
            and cached_part == part_key
        ):
            return _json_deep_clone(cached_snapshot)

    cpu_count = max(1, int(os.cpu_count() or 1))
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except OSError:
        load_1m, load_5m, load_15m = (0.0, 0.0, 0.0)
    cpu_utilization = _clamp01(load_1m / max(1, cpu_count)) * 100.0
    cpu_per_core = _cpu_percent_per_core_snapshot(
        cpu_count=cpu_count,
        fallback_utilization=cpu_utilization,
    )

    memory_total_mb, memory_available_mb = _parse_proc_meminfo_mb()
    memory_pressure = 0.0
    if memory_total_mb > 0.0:
        memory_pressure = _clamp01(
            (memory_total_mb - max(0.0, memory_available_mb)) / memory_total_mb
        )

    nvidia_rows = _collect_nvidia_metrics()
    gpu1_default = nvidia_rows[0] if len(nvidia_rows) >= 1 else {}
    gpu2_default = nvidia_rows[1] if len(nvidia_rows) >= 2 else {}

    openvino_device = (
        str(os.getenv("OPENVINO_EMBED_DEVICE", "NPU") or "NPU").strip() or "NPU"
    )
    with _OPENVINO_EMBED_LOCK:
        openvino_loaded = _OPENVINO_EMBED_RUNTIME.get("model") is not None

    gpu1_utilization = _safe_env_metric(
        "ETA_MU_GPU1_UTILIZATION",
        _safe_float(gpu1_default.get("utilization", 0.0), 0.0),
    )
    gpu1_memory = _safe_env_metric(
        "ETA_MU_GPU1_MEMORY",
        _safe_float(gpu1_default.get("memory", 0.0), 0.0),
    )
    gpu1_temperature = _safe_env_metric(
        "ETA_MU_GPU1_TEMP",
        _safe_float(gpu1_default.get("temperature", 0.0), 0.0),
    )

    gpu2_utilization = _safe_env_metric(
        "ETA_MU_GPU2_UTILIZATION",
        _safe_float(gpu2_default.get("utilization", 0.0), 0.0),
    )
    gpu2_memory = _safe_env_metric(
        "ETA_MU_GPU2_MEMORY",
        _safe_float(gpu2_default.get("memory", 0.0), 0.0),
    )
    gpu2_temperature = _safe_env_metric(
        "ETA_MU_GPU2_TEMP",
        _safe_float(gpu2_default.get("temperature", 0.0), 0.0),
    )

    npu_util_default = (
        22.0 if ("NPU" in openvino_device.upper() and openvino_loaded) else 0.0
    )
    npu_utilization = _safe_env_metric("ETA_MU_NPU0_UTILIZATION", npu_util_default)
    npu_queue_depth = max(0.0, _safe_env_metric("ETA_MU_NPU0_QUEUE_DEPTH", 0.0))
    npu_temperature = _safe_env_metric("ETA_MU_NPU0_TEMP", 0.0)

    log_watch = _resource_log_watch(part_root=part_root)
    intel_gpu = _collect_intel_gpu_metrics()
    npu_metrics = _collect_npu_metrics()

    effective_npu_utilization = max(
        0.0,
        min(100.0, _safe_float(npu_metrics.get("utilization"), npu_utilization)),
    )

    devices = {
        "cpu": {
            "utilization": round(cpu_utilization, 2),
            "per_core": [round(c, 2) for c in cpu_per_core],
            "load_avg": {
                "m1": round(load_1m, 3),
                "m5": round(load_5m, 3),
                "m15": round(load_15m, 3),
            },
            "memory_pressure": round(memory_pressure, 4),
            "status": _status_from_utilization(
                cpu_utilization, watch_threshold=70.0, hot_threshold=88.0
            ),
        },
        "gpu1": {
            "name": "NVIDIA",
            "utilization": round(max(0.0, min(100.0, gpu1_utilization)), 2),
            "memory": round(max(0.0, min(100.0, gpu1_memory)), 2),
            "temperature": round(max(0.0, gpu1_temperature), 2),
            "status": _status_from_utilization(
                gpu1_utilization, watch_threshold=76.0, hot_threshold=93.0
            ),
        },
        "gpu_intel": {
            "name": "Intel",
            "utilization": round(max(0.0, min(100.0, gpu2_utilization)), 2),
            "memory": round(max(0.0, min(100.0, gpu2_memory)), 2),
            "temperature": round(max(0.0, gpu2_temperature), 2),
            "current_freq_mhz": intel_gpu.get("current_freq_mhz"),
            "max_freq_mhz": intel_gpu.get("max_freq_mhz"),
            "status": _status_from_utilization(
                gpu2_utilization, watch_threshold=76.0, hot_threshold=93.0
            ),
        },
        "gpu2": {
            "name": "Intel (Legacy Alias)",
            "utilization": round(max(0.0, min(100.0, gpu2_utilization)), 2),
            "memory": round(max(0.0, min(100.0, gpu2_memory)), 2),
            "temperature": round(max(0.0, gpu2_temperature), 2),
            "current_freq_mhz": intel_gpu.get("current_freq_mhz"),
            "max_freq_mhz": intel_gpu.get("max_freq_mhz"),
            "status": _status_from_utilization(
                gpu2_utilization, watch_threshold=76.0, hot_threshold=93.0
            ),
        },
        "npu0": {
            "name": "Intel NPU",
            "utilization": round(effective_npu_utilization, 2),
            "busy_time_us": npu_metrics.get("busy_time_us"),
            "current_freq_mhz": npu_metrics.get("current_freq_mhz"),
            "memory_bytes": npu_metrics.get("memory_bytes"),
            "queue_depth": int(max(0.0, npu_queue_depth)),
            "temperature": round(max(0.0, npu_temperature), 2),
            "device": openvino_device,
            "status": _status_from_utilization(
                effective_npu_utilization, watch_threshold=78.0, hot_threshold=95.0
            ),
        },
    }
    hot_devices = [
        device_id
        for device_id, row in devices.items()
        if isinstance(row, dict) and str(row.get("status", "")) == "hot"
    ]

    snapshot = {
        "record": "ημ.resource-heartbeat.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_seconds": RESOURCE_SNAPSHOT_CACHE_SECONDS,
        "host": {
            "cpu_count": cpu_count,
            "memory_total_mb": round(memory_total_mb, 2),
            "memory_available_mb": round(memory_available_mb, 2),
        },
        "devices": devices,
        "log_watch": log_watch,
        "hot_devices": hot_devices,
    }
    snapshot["auto_backend"] = {
        "embeddings_order": _resource_auto_embedding_order(snapshot=snapshot),
        "text_order": _resource_auto_text_order(snapshot=snapshot),
    }

    with _RESOURCE_MONITOR_LOCK:
        _RESOURCE_MONITOR_CACHE["checked_monotonic"] = now_monotonic
        _RESOURCE_MONITOR_CACHE["part_root"] = part_key
        _RESOURCE_MONITOR_CACHE["snapshot"] = _json_deep_clone(snapshot)

    return snapshot


class RuntimeInfluenceTracker:
    CLICK_WINDOW_SECONDS = 45.0
    FILE_WINDOW_SECONDS = 120.0
    LOG_WINDOW_SECONDS = 180.0
    RESOURCE_WINDOW_SECONDS = 180.0
    COMPUTE_WINDOW_SECONDS = 180.0
    USER_INPUT_WINDOW_SECONDS = 120.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._click_events: list[dict[str, Any]] = []
        self._file_events: list[dict[str, Any]] = []
        self._log_events: list[dict[str, Any]] = []
        self._resource_events: list[dict[str, Any]] = []
        self._compute_events: list[dict[str, Any]] = []
        self._user_input_events: list[dict[str, Any]] = []
        self._fork_tax_debt = 0.0
        self._fork_tax_paid = 0.0

    def _prune(self, now: float) -> None:
        self._click_events = [
            row
            for row in self._click_events
            if (now - float(row.get("ts", 0.0))) <= self.CLICK_WINDOW_SECONDS
        ]
        self._file_events = [
            row
            for row in self._file_events
            if (now - float(row.get("ts", 0.0))) <= self.FILE_WINDOW_SECONDS
        ]
        self._log_events = [
            row
            for row in self._log_events
            if (now - float(row.get("ts", 0.0))) <= self.LOG_WINDOW_SECONDS
        ]
        self._resource_events = [
            row
            for row in self._resource_events
            if (now - float(row.get("ts", 0.0))) <= self.RESOURCE_WINDOW_SECONDS
        ]
        self._compute_events = [
            row
            for row in self._compute_events
            if (now - float(row.get("ts", 0.0))) <= self.COMPUTE_WINDOW_SECONDS
        ]
        self._user_input_events = [
            row
            for row in self._user_input_events
            if (now - float(row.get("ts", 0.0))) <= self.USER_INPUT_WINDOW_SECONDS
        ]

    def record_witness(self, *, event_type: str, target: str) -> None:
        now = time.time()
        with self._lock:
            self._click_events.append(
                {
                    "ts": now,
                    "event_type": event_type,
                    "target": target,
                }
            )
            payment = 1.0
            if "fork" in target.lower() or "tax" in target.lower():
                payment += 1.0
            if event_type == "world_interact":
                payment += 0.25
            self._fork_tax_paid += payment
            self._prune(now)

    def record_file_delta(self, delta: dict[str, Any]) -> None:
        added_count = int(delta.get("added_count", 0))
        updated_count = int(delta.get("updated_count", 0))
        removed_count = int(delta.get("removed_count", 0))
        total = max(0, added_count + updated_count + removed_count)
        if total <= 0:
            return

        score = (added_count * 1.6) + (updated_count * 1.0) + (removed_count * 1.3)
        now = time.time()
        with self._lock:
            self._file_events.append(
                {
                    "ts": now,
                    "added": added_count,
                    "updated": updated_count,
                    "removed": removed_count,
                    "changes": total,
                    "score": round(score, 3),
                    "sample_paths": list(delta.get("sample_paths", []))[:6],
                }
            )
            self._fork_tax_debt += score
            self._prune(now)

    def record_resource_heartbeat(
        self,
        heartbeat: dict[str, Any],
        *,
        source: str = "runtime",
    ) -> None:
        if not isinstance(heartbeat, dict):
            return
        now = time.time()
        with self._lock:
            self._resource_events.append(
                {
                    "ts": now,
                    "source": str(source).strip() or "runtime",
                    "heartbeat": _json_deep_clone(heartbeat),
                }
            )
            self._prune(now)

    def record_runtime_log(
        self,
        *,
        level: str,
        message: str,
        source: str = "runtime",
    ) -> None:
        clean_message = str(message).strip()
        if not clean_message:
            return
        now = time.time()
        with self._lock:
            self._log_events.append(
                {
                    "ts": now,
                    "source": str(source).strip() or "runtime",
                    "level": str(level).strip().lower() or "info",
                    "message": clean_message[:240],
                }
            )
            self._prune(now)

    def record_user_input(
        self,
        *,
        kind: str,
        target: str,
        message: str,
        x_ratio: float | None = None,
        y_ratio: float | None = None,
        embed_daimoi: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> None:
        clean_kind = str(kind or "input").strip().lower() or "input"
        clean_target = str(target or "simulation").strip() or "simulation"
        clean_message = str(message or "").strip()
        payload: dict[str, Any] = {
            "kind": clean_kind,
            "target": clean_target[:240],
            "message": clean_message[:320],
            "embed_daimoi": bool(embed_daimoi),
        }
        if x_ratio is not None:
            payload["x_ratio"] = round(_clamp01(_safe_float(x_ratio, 0.5)), 6)
        if y_ratio is not None:
            payload["y_ratio"] = round(_clamp01(_safe_float(y_ratio, 0.5)), 6)
        if isinstance(meta, dict) and meta:
            payload["meta"] = {
                str(key): value for key, value in list(meta.items())[:12]
            }

        now = time.time()
        payload["ts"] = now
        with self._lock:
            self._user_input_events.append(payload)
            self._prune(now)

    def record_compute_job(
        self,
        *,
        kind: str,
        op: str,
        backend: str,
        resource: str,
        emitter_presence_id: str,
        target_presence_id: str = "",
        model: str = "",
        status: str = "ok",
        latency_ms: float | None = None,
        error: str = "",
    ) -> None:
        clean_kind = str(kind).strip().lower() or "unknown"
        clean_op = str(op).strip().lower() or "job"
        clean_backend = str(backend).strip().lower() or "unknown"
        clean_resource = str(resource).strip().lower() or "unknown"
        clean_emitter = str(emitter_presence_id).strip() or "health_sentinel_cpu"
        clean_target = str(target_presence_id).strip()
        clean_model = str(model).strip()
        clean_status = str(status).strip().lower() or "ok"
        clean_error = str(error).strip()[:180]

        numeric_latency: float | None = None
        if latency_ms is not None:
            try:
                numeric_latency = max(0.0, min(float(latency_ms), 240000.0))
            except (TypeError, ValueError):
                numeric_latency = None

        now = time.time()
        event_id_material = (
            f"{now:.6f}|{clean_kind}|{clean_op}|{clean_backend}|{clean_resource}|"
            f"{clean_emitter}|{clean_target}|{clean_status}|{clean_model}|{clean_error}"
        )
        event_id = hashlib.sha1(event_id_material.encode("utf-8")).hexdigest()[:14]
        event: dict[str, Any] = {
            "id": f"compute:{event_id}",
            "ts": now,
            "kind": clean_kind,
            "op": clean_op,
            "backend": clean_backend,
            "resource": clean_resource,
            "emitter_presence_id": clean_emitter,
            "target_presence_id": clean_target,
            "model": clean_model,
            "status": clean_status,
        }
        if numeric_latency is not None:
            event["latency_ms"] = round(numeric_latency, 3)
        if clean_error:
            event["error"] = clean_error

        with self._lock:
            self._compute_events.append(event)
            if len(self._compute_events) > 512:
                self._compute_events = self._compute_events[-512:]
            self._prune(now)

    def pay_fork_tax(
        self,
        *,
        amount: float,
        source: str,
        target: str,
    ) -> dict[str, Any]:
        applied = max(0.25, min(float(amount), 144.0))
        now = time.time()
        with self._lock:
            self._fork_tax_paid += applied
            self._click_events.append(
                {
                    "ts": now,
                    "event_type": "fork_tax_payment",
                    "target": target,
                    "source": source,
                    "amount": round(applied, 3),
                }
            )
            self._prune(now)
        return {
            "applied": round(applied, 3),
            "source": source,
            "target": target,
        }

    def snapshot(
        self,
        queue_snapshot: dict[str, Any] | None = None,
        *,
        part_root: Path | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        queue_snapshot = queue_snapshot or {}
        with self._lock:
            self._prune(now)
            click_rows = list(self._click_events)
            file_rows = list(self._file_events)
            log_rows = list(self._log_events)
            resource_rows = list(self._resource_events)
            compute_rows = list(self._compute_events)
            user_input_rows = list(self._user_input_events)
            fork_tax_debt = float(self._fork_tax_debt)
            fork_tax_paid = float(self._fork_tax_paid)

        clicks_recent = len(click_rows)
        file_changes_recent = sum(int(row.get("changes", 0)) for row in file_rows)
        queue_event_count = int(queue_snapshot.get("event_count", 0))
        queue_pending_count = int(queue_snapshot.get("pending_count", 0))

        paid_effective = fork_tax_paid + (queue_event_count * 0.25)
        balance = max(0.0, fork_tax_debt - paid_effective)
        paid_ratio = (
            1.0 if fork_tax_debt <= 0 else _clamp01(paid_effective / fork_tax_debt)
        )

        auto_commit_pulse = _clamp01(
            (file_changes_recent * 0.08)
            + (queue_event_count * 0.05)
            + (queue_pending_count * 0.06)
        )
        if queue_pending_count > 0:
            status_en = "staging receipts"
            status_ja = "領収書を段取り中"
        elif file_changes_recent > 0:
            status_en = "watching drift"
            status_ja = "ドリフトを監視中"
        else:
            status_en = "gate idle"
            status_ja = "門前で待機中"

        recent_targets = [
            str(row.get("target", ""))
            for row in sorted(
                click_rows, key=lambda item: float(item.get("ts", 0.0)), reverse=True
            )
            if row.get("target")
        ][:6]
        recent_file_paths: list[str] = []
        for row in sorted(
            file_rows, key=lambda item: float(item.get("ts", 0.0)), reverse=True
        ):
            for path in row.get("sample_paths", []):
                value = str(path).strip()
                if value and value not in recent_file_paths:
                    recent_file_paths.append(value)
                if len(recent_file_paths) >= 8:
                    break
            if len(recent_file_paths) >= 8:
                break

        recent_logs = sorted(
            log_rows,
            key=lambda item: float(item.get("ts", 0.0)),
            reverse=True,
        )
        recent_user_inputs = sorted(
            user_input_rows,
            key=lambda item: float(item.get("ts", 0.0)),
            reverse=True,
        )
        latest_log = recent_logs[0] if recent_logs else {}
        log_error_count = sum(
            1
            for row in log_rows
            if str(row.get("level", "")).lower() in {"error", "fatal"}
        )
        log_warn_count = sum(
            1
            for row in log_rows
            if str(row.get("level", "")).lower() in {"warn", "warning"}
        )

        latest_resource_event = None
        if resource_rows:
            latest_resource_event = max(
                resource_rows,
                key=lambda item: float(item.get("ts", 0.0)),
            )
        resource_heartbeat = (
            _json_deep_clone(latest_resource_event.get("heartbeat", {}))
            if isinstance(latest_resource_event, dict)
            else None
        )
        if not isinstance(resource_heartbeat, dict) or not resource_heartbeat:
            resource_heartbeat = _resource_monitor_snapshot(part_root=part_root)

        recent_compute_rows = sorted(
            compute_rows,
            key=lambda item: float(item.get("ts", 0.0)),
            reverse=True,
        )
        compute_llm_count = sum(
            1 for row in compute_rows if str(row.get("kind", "")).lower() == "llm"
        )
        compute_embedding_count = sum(
            1
            for row in compute_rows
            if str(row.get("kind", "")).lower() in {"embedding", "embeddings"}
        )
        compute_ok_count = sum(
            1
            for row in compute_rows
            if str(row.get("status", "")).lower() in {"ok", "success", "cached"}
        )
        compute_error_count = sum(
            1
            for row in compute_rows
            if str(row.get("status", "")).lower() in {"error", "failed", "timeout"}
        )
        compute_resource_counts: dict[str, int] = {}
        for row in compute_rows:
            resource_key = (
                str(row.get("resource", "unknown")).strip().lower() or "unknown"
            )
            compute_resource_counts[resource_key] = (
                int(compute_resource_counts.get(resource_key, 0)) + 1
            )

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "clicks_45s": clicks_recent,
            "file_changes_120s": file_changes_recent,
            "log_events_180s": len(log_rows),
            "resource_events_180s": len(resource_rows),
            "user_inputs_120s": len(user_input_rows),
            "recent_click_targets": recent_targets,
            "recent_file_paths": recent_file_paths,
            "recent_user_inputs": [
                {
                    "kind": str(row.get("kind", "input")),
                    "target": str(row.get("target", "")),
                    "message": str(row.get("message", "")),
                    "x_ratio": row.get("x_ratio"),
                    "y_ratio": row.get("y_ratio"),
                    "embed_daimoi": bool(row.get("embed_daimoi", False)),
                }
                for row in recent_user_inputs[:24]
            ],
            "recent_logs": [
                {
                    "level": str(row.get("level", "info")),
                    "source": str(row.get("source", "runtime")),
                    "message": str(row.get("message", "")),
                }
                for row in recent_logs[:4]
            ],
            "last_log": {
                "level": str(latest_log.get("level", "")),
                "source": str(latest_log.get("source", "")),
                "message": str(latest_log.get("message", "")),
            },
            "log_summary": {
                "event_count": len(log_rows),
                "error_count": log_error_count,
                "warn_count": log_warn_count,
            },
            "compute_jobs_180s": len(compute_rows),
            "compute_summary": {
                "llm_jobs": compute_llm_count,
                "embedding_jobs": compute_embedding_count,
                "ok_count": compute_ok_count,
                "error_count": compute_error_count,
                "resource_counts": compute_resource_counts,
            },
            "compute_jobs": [
                {
                    "id": str(row.get("id", "")),
                    "at": datetime.fromtimestamp(
                        float(row.get("ts", now)), timezone.utc
                    ).isoformat(),
                    "ts": float(row.get("ts", now)),
                    "kind": str(row.get("kind", "")),
                    "op": str(row.get("op", "")),
                    "backend": str(row.get("backend", "")),
                    "resource": str(row.get("resource", "")),
                    "emitter_presence_id": str(row.get("emitter_presence_id", "")),
                    "target_presence_id": str(row.get("target_presence_id", "")),
                    "model": str(row.get("model", "")),
                    "status": str(row.get("status", "")),
                    "latency_ms": (
                        round(float(row.get("latency_ms", 0.0)), 3)
                        if row.get("latency_ms") is not None
                        else None
                    ),
                    "error": str(row.get("error", "")),
                }
                for row in recent_compute_rows[:32]
            ],
            "resource_heartbeat": resource_heartbeat,
            "fork_tax": {
                "law_en": "Pay the fork tax; annotate every drift with proof.",
                "law_ja": "フォーク税は法。ドリフトごとに証明を注釈せよ。",
                "debt": round(fork_tax_debt, 3),
                "paid": round(paid_effective, 3),
                "balance": round(balance, 3),
                "paid_ratio": round(paid_ratio, 4),
            },
            "ghost": {
                "id": FILE_SENTINEL_PROFILE["id"],
                "en": FILE_SENTINEL_PROFILE["en"],
                "ja": FILE_SENTINEL_PROFILE["ja"],
                "auto_commit_pulse": round(auto_commit_pulse, 4),
                "queue_pending": queue_pending_count,
                "status_en": status_en,
                "status_ja": status_ja,
            },
        }


_INFLUENCE_TRACKER = RuntimeInfluenceTracker()
