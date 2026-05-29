from __future__ import annotations

from typing import Any

from code.world_web.sim_slice_offload_strategy import (
    is_supported_remote_mode,
    normalize_sim_slice_mode,
    request_sim_point_budget_by_mode,
)


def test_normalize_sim_slice_mode_defaults_to_local() -> None:
    assert normalize_sim_slice_mode("") == "local"
    assert normalize_sim_slice_mode(None) == "local"
    assert normalize_sim_slice_mode(" UDS ") == "uds"


def test_is_supported_remote_mode_only_accepts_remote_modes() -> None:
    assert is_supported_remote_mode("redis") is True
    assert is_supported_remote_mode("uds") is True
    assert is_supported_remote_mode("local") is False


def test_request_sim_point_budget_by_mode_dispatches_handlers() -> None:
    calls: list[tuple[str, float, int]] = []

    def _redis(
        *, cpu_utilization: float, max_sim_points: int
    ) -> tuple[int, dict[str, Any]]:
        calls.append(("redis", cpu_utilization, max_sim_points))
        return 640, {"source": "redis-worker", "job_id": "r1"}

    def _uds(
        *, cpu_utilization: float, max_sim_points: int
    ) -> tuple[int, dict[str, Any]]:
        calls.append(("uds", cpu_utilization, max_sim_points))
        return 480, {"source": "uds-worker", "job_id": "u1"}

    redis_budget, redis_meta = request_sim_point_budget_by_mode(
        mode="redis",
        cpu_utilization=87.5,
        max_sim_points=800,
        request_via_redis=_redis,
        request_via_uds=_uds,
    )
    uds_budget, uds_meta = request_sim_point_budget_by_mode(
        mode="uds",
        cpu_utilization=73.0,
        max_sim_points=900,
        request_via_redis=_redis,
        request_via_uds=_uds,
    )

    assert redis_budget == 640
    assert redis_meta.get("source") == "redis-worker"
    assert uds_budget == 480
    assert uds_meta.get("source") == "uds-worker"
    assert calls == [
        ("redis", 87.5, 800),
        ("uds", 73.0, 900),
    ]


def test_request_sim_point_budget_by_mode_unsupported_returns_fallback() -> None:
    def _noop(
        *, cpu_utilization: float, max_sim_points: int
    ) -> tuple[int, dict[str, Any]]:
        return 0, {"source": "noop", "job_id": "noop"}

    budget, meta = request_sim_point_budget_by_mode(
        mode="local",
        cpu_utilization=61.0,
        max_sim_points=700,
        request_via_redis=_noop,
        request_via_uds=_noop,
    )

    assert budget is None
    assert meta.get("source") == "python-fallback"
    assert meta.get("reason") == "unsupported-mode"
