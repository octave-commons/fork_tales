from __future__ import annotations

from code.world_web.sim_slice_fallback_policy import (
    build_sim_slice_async_cached_meta,
    build_sim_slice_async_fallback_meta,
    build_sim_slice_local_meta,
    build_sim_slice_remote_fallback_meta,
    build_sim_slice_remote_success_meta,
    build_sim_slice_worker_fallback_snapshot,
    build_sim_slice_worker_success_snapshot,
    resolve_sim_slice_cached_budget,
)


def test_build_sim_slice_worker_snapshots_capture_success_and_fallback() -> None:
    fallback = build_sim_slice_worker_fallback_snapshot(
        mode="uds",
        local_budget=320,
        remote_meta={
            "source": "python-fallback",
            "reason": "uds-timeout",
            "job_id": "j1",
        },
        transport_latency_ms=5.2,
        produced_monotonic=12.0,
    )
    success = build_sim_slice_worker_success_snapshot(
        mode="redis",
        remote_budget=640,
        remote_meta={"source": "c-worker", "job_id": "j2"},
        transport_latency_ms=3.6,
        produced_monotonic=14.0,
    )
    assert fallback["fallback"] is True
    assert fallback["reason"] == "uds-timeout"
    assert success["fallback"] is False
    assert success["budget"] == 640


def test_build_sim_slice_async_meta_distinguishes_cached_and_stale() -> None:
    latest = {
        "ready": True,
        "mode": "uds",
        "source": "c-worker",
        "fallback": False,
        "reason": "",
        "job_id": "j3",
        "transport_latency_ms": 2.5,
        "budget": 480,
    }
    cached = build_sim_slice_async_cached_meta(
        mode="uds",
        latest=latest,
        latency_ms=0.4,
        age_ms=10.0,
    )
    stale = build_sim_slice_async_fallback_meta(
        mode="uds",
        latest=latest,
        latency_ms=0.6,
        age_ms=9000.0,
        stale_limit_ms=5000,
    )
    assert cached["fallback"] is False
    assert cached["async"] is True
    assert stale["fallback"] is True
    assert stale["reason"] == "async-stale"
    assert resolve_sim_slice_cached_budget(latest, 320) == 480


def test_build_sim_slice_sync_meta_covers_local_success_and_fallback() -> None:
    local = build_sim_slice_local_meta(mode="local")
    fallback = build_sim_slice_remote_fallback_meta(
        mode="redis",
        remote_meta={
            "source": "python-fallback",
            "reason": "enqueue-failed",
            "job_id": "j4",
        },
        latency_ms=7.8,
    )
    success = build_sim_slice_remote_success_meta(
        mode="redis",
        remote_meta={"source": "c-worker", "job_id": "j5"},
        latency_ms=3.2,
    )
    assert local == {
        "mode": "local",
        "source": "python-local",
        "fallback": False,
        "latency_ms": 0.0,
    }
    assert fallback["reason"] == "enqueue-failed"
    assert success["fallback"] is False
