from __future__ import annotations

from typing import Any, Callable


SIM_SLICE_REDIS_MODE = "redis"
SIM_SLICE_UDS_MODE = "uds"
SIM_SLICE_LOCAL_MODE = "local"

_SUPPORTED_REMOTE_MODES: set[str] = {
    SIM_SLICE_REDIS_MODE,
    SIM_SLICE_UDS_MODE,
}


def normalize_sim_slice_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return SIM_SLICE_LOCAL_MODE
    return text


def is_supported_remote_mode(mode: str) -> bool:
    return str(mode or "") in _SUPPORTED_REMOTE_MODES


def request_sim_point_budget_by_mode(
    *,
    mode: str,
    cpu_utilization: float,
    max_sim_points: int,
    request_via_redis: Callable[..., tuple[int | None, dict[str, Any]]],
    request_via_uds: Callable[..., tuple[int | None, dict[str, Any]]],
) -> tuple[int | None, dict[str, Any]]:
    mode_key = str(mode or "")
    if mode_key == SIM_SLICE_REDIS_MODE:
        return request_via_redis(
            cpu_utilization=cpu_utilization,
            max_sim_points=max_sim_points,
        )
    if mode_key == SIM_SLICE_UDS_MODE:
        return request_via_uds(
            cpu_utilization=cpu_utilization,
            max_sim_points=max_sim_points,
        )
    return None, {
        "source": "python-fallback",
        "reason": "unsupported-mode",
        "job_id": "",
    }
