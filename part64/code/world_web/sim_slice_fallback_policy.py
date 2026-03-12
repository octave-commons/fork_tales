from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def build_sim_slice_worker_fallback_snapshot(
    *,
    mode: str,
    local_budget: int,
    remote_meta: dict[str, Any] | None,
    transport_latency_ms: float,
    produced_monotonic: float,
) -> dict[str, Any]:
    meta = remote_meta if isinstance(remote_meta, dict) else {}
    return {
        "ready": True,
        "mode": str(mode or ""),
        "budget": max(0, int(local_budget)),
        "source": str(meta.get("source", "python-fallback") or "python-fallback"),
        "fallback": True,
        "reason": str(meta.get("reason", "unknown") or "unknown"),
        "job_id": str(meta.get("job_id", "") or ""),
        "transport_latency_ms": round(max(0.0, float(transport_latency_ms)), 3),
        "produced_monotonic": max(0.0, float(produced_monotonic)),
    }


def build_sim_slice_worker_success_snapshot(
    *,
    mode: str,
    remote_budget: int,
    remote_meta: dict[str, Any] | None,
    transport_latency_ms: float,
    produced_monotonic: float,
) -> dict[str, Any]:
    meta = remote_meta if isinstance(remote_meta, dict) else {}
    return {
        "ready": True,
        "mode": str(mode or ""),
        "budget": max(0, int(remote_budget)),
        "source": str(meta.get("source", "c-worker") or "c-worker"),
        "fallback": False,
        "reason": "",
        "job_id": str(meta.get("job_id", "") or ""),
        "transport_latency_ms": round(max(0.0, float(transport_latency_ms)), 3),
        "produced_monotonic": max(0.0, float(produced_monotonic)),
    }


def build_sim_slice_local_meta(*, mode: str) -> dict[str, Any]:
    return {
        "mode": str(mode or "local") or "local",
        "source": "python-local",
        "fallback": False,
        "latency_ms": 0.0,
    }


def build_sim_slice_async_cached_meta(
    *,
    mode: str,
    latest: dict[str, Any] | None,
    latency_ms: float,
    age_ms: float,
) -> dict[str, Any]:
    snapshot = latest if isinstance(latest, dict) else {}
    return {
        "mode": str(mode or "local"),
        "source": str(snapshot.get("source", "c-worker") or "c-worker"),
        "fallback": bool(snapshot.get("fallback", False)),
        "reason": str(snapshot.get("reason", "") or ""),
        "job_id": str(snapshot.get("job_id", "") or ""),
        "latency_ms": round(max(0.0, float(latency_ms)), 3),
        "transport_latency_ms": _safe_float(
            snapshot.get("transport_latency_ms", 0.0), 0.0
        ),
        "age_ms": round(max(0.0, float(age_ms)), 3),
        "async": True,
    }


def build_sim_slice_async_fallback_meta(
    *,
    mode: str,
    latest: dict[str, Any] | None,
    latency_ms: float,
    age_ms: float,
    stale_limit_ms: int,
) -> dict[str, Any]:
    snapshot = latest if isinstance(latest, dict) else {}
    latest_ready = bool(snapshot.get("ready", False))
    latest_mode = str(snapshot.get("mode", "") or "").strip().lower()
    normalized_mode = str(mode or "").strip().lower()
    reason = "async-warmup"
    if (
        latest_ready
        and latest_mode == normalized_mode
        and age_ms > float(stale_limit_ms)
    ):
        reason = "async-stale"
    return {
        "mode": str(mode or "local"),
        "source": "python-local",
        "fallback": True,
        "reason": reason,
        "job_id": str(snapshot.get("job_id", "") or ""),
        "latency_ms": round(max(0.0, float(latency_ms)), 3),
        "transport_latency_ms": _safe_float(
            snapshot.get("transport_latency_ms", 0.0), 0.0
        ),
        "age_ms": round(max(0.0, float(age_ms)), 3),
        "async": True,
    }


def build_sim_slice_remote_fallback_meta(
    *,
    mode: str,
    remote_meta: dict[str, Any] | None,
    latency_ms: float,
) -> dict[str, Any]:
    meta = remote_meta if isinstance(remote_meta, dict) else {}
    return {
        "mode": str(mode or "local"),
        "source": str(meta.get("source", "python-fallback") or "python-fallback"),
        "fallback": True,
        "reason": str(meta.get("reason", "unknown") or "unknown"),
        "job_id": str(meta.get("job_id", "") or ""),
        "latency_ms": round(max(0.0, float(latency_ms)), 3),
    }


def build_sim_slice_remote_success_meta(
    *,
    mode: str,
    remote_meta: dict[str, Any] | None,
    latency_ms: float,
) -> dict[str, Any]:
    meta = remote_meta if isinstance(remote_meta, dict) else {}
    return {
        "mode": str(mode or "local"),
        "source": str(meta.get("source", "c-worker") or "c-worker"),
        "fallback": False,
        "job_id": str(meta.get("job_id", "") or ""),
        "latency_ms": round(max(0.0, float(latency_ms)), 3),
    }


def resolve_sim_slice_cached_budget(
    latest: dict[str, Any] | None, local_budget: int
) -> int:
    snapshot = latest if isinstance(latest, dict) else {}
    return max(64, _safe_int(snapshot.get("budget", local_budget), local_budget))
