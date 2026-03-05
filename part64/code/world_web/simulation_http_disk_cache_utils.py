"""Helpers for simulation HTTP disk cache paths and payload IO."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable


def simulation_http_disk_cache_path(
    part_root: Path,
    perspective: str,
    *,
    default_perspective: str,
) -> Path:
    key = str(perspective or default_perspective).strip().lower()
    safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in key)
    safe = safe.strip("_") or default_perspective
    return (part_root / "world_state" / f"simulation_http_cache_{safe}.json").resolve()


def simulation_http_runtime_reference_mtime(candidate_paths: list[Path]) -> float:
    newest = 0.0
    for path in candidate_paths:
        try:
            if path.exists() and path.is_file():
                newest = max(newest, float(path.stat().st_mtime))
        except Exception:
            continue
    return newest


def simulation_http_disk_cache_load(
    part_root: Path,
    *,
    perspective: str,
    max_age_seconds: float,
    disk_cache_enabled: bool,
    default_perspective: str,
    runtime_reference_mtime: float,
    safe_float: Callable[[Any, float], float],
    time_seconds_fn: Callable[[], float] = time.time,
) -> bytes | None:
    if not disk_cache_enabled:
        return None
    cache_age_limit = max(0.0, safe_float(max_age_seconds, 0.0))
    if cache_age_limit <= 0.0:
        return None

    cache_path = simulation_http_disk_cache_path(
        part_root,
        perspective,
        default_perspective=default_perspective,
    )
    try:
        if not cache_path.exists() or not cache_path.is_file():
            return None
        stat = cache_path.stat()
        if (
            runtime_reference_mtime > 0.0
            and float(stat.st_mtime) < runtime_reference_mtime
        ):
            return None
        age = time_seconds_fn() - float(stat.st_mtime)
        if age < 0.0 or age > cache_age_limit:
            return None
        payload = cache_path.read_bytes()
        if not payload:
            return None
        return payload
    except Exception:
        return None


def simulation_http_disk_cache_has_payload(
    part_root: Path,
    *,
    perspective: str,
    max_age_seconds: float,
    disk_cache_enabled: bool,
    default_perspective: str,
    runtime_reference_mtime: float,
    safe_float: Callable[[Any, float], float],
    time_seconds_fn: Callable[[], float] = time.time,
) -> bool:
    if not disk_cache_enabled:
        return False
    cache_age_limit = max(0.0, safe_float(max_age_seconds, 0.0))
    if cache_age_limit <= 0.0:
        return False

    cache_path = simulation_http_disk_cache_path(
        part_root,
        perspective,
        default_perspective=default_perspective,
    )
    try:
        if not cache_path.exists() or not cache_path.is_file():
            return False
        stat = cache_path.stat()
        if (
            runtime_reference_mtime > 0.0
            and float(stat.st_mtime) < runtime_reference_mtime
        ):
            return False
        age = time_seconds_fn() - float(stat.st_mtime)
        if age < 0.0 or age > cache_age_limit:
            return False
        return int(getattr(stat, "st_size", 0) or 0) > 0
    except Exception:
        return False


def simulation_http_disk_cache_store(
    part_root: Path,
    *,
    perspective: str,
    body: bytes,
    disk_cache_enabled: bool,
    default_perspective: str,
) -> None:
    if not disk_cache_enabled:
        return
    if not isinstance(body, (bytes, bytearray)):
        return
    payload = bytes(body)
    if not payload:
        return

    cache_path = simulation_http_disk_cache_path(
        part_root,
        perspective,
        default_perspective=default_perspective,
    )
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(payload)
        tmp_path.replace(cache_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def simulation_http_disk_cache_invalidate(
    *,
    part_root: Path | None,
    disk_cache_enabled: bool,
    default_perspective: str,
    perspective_options: list[Any],
) -> None:
    if part_root is None or not disk_cache_enabled:
        return

    perspectives: set[str] = set()
    default_value = str(default_perspective or "").strip().lower()
    if default_value:
        perspectives.add(default_value)
    for option in perspective_options:
        if isinstance(option, dict):
            option_id = str(option.get("id", "") or "").strip().lower()
            if option_id:
                perspectives.add(option_id)

    for perspective in perspectives:
        cache_path = simulation_http_disk_cache_path(
            part_root,
            perspective,
            default_perspective=default_perspective,
        )
        try:
            cache_path.unlink(missing_ok=True)
        except Exception:
            continue
