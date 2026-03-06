from __future__ import annotations

import time
from typing import Any

from code.world_web import simulation_ws_shared_controller as shared_controller_module


def test_shared_stream_registry_reuses_single_stream_per_key() -> None:
    shared_controller_module.reset_shared_simulation_stream_registry_for_tests()
    calls = 0

    def _collect() -> tuple[dict[str, Any], str] | None:
        nonlocal calls
        calls += 1
        return ({"type": "simulation", "simulation": {"timestamp": "t0"}}, "t0")

    lease_one = shared_controller_module._SHARED_STREAM_REGISTRY.subscribe(
        stream_key="hybrid:trimmed",
        collect_frame=_collect,
        refresh_seconds=0.02,
        heartbeat_seconds=1.0,
    )
    lease_two = shared_controller_module._SHARED_STREAM_REGISTRY.subscribe(
        stream_key="hybrid:trimmed",
        collect_frame=_collect,
        refresh_seconds=0.02,
        heartbeat_seconds=1.0,
    )
    try:
        frame_one = lease_one.wait_for_frame(0, timeout_seconds=0.6)
        frame_two = lease_two.wait_for_frame(0, timeout_seconds=0.6)
        assert frame_one is not None
        assert frame_two is not None
        assert frame_one.seq == frame_two.seq
        assert frame_one.payload == frame_two.payload
        assert calls >= 1
    finally:
        lease_one.close()
        lease_two.close()


def test_shared_stream_publishes_new_seq_when_fingerprint_changes() -> None:
    shared_controller_module.reset_shared_simulation_stream_registry_for_tests()
    calls = 0

    def _collect() -> tuple[dict[str, Any], str] | None:
        nonlocal calls
        calls += 1
        if calls < 2:
            return (
                {"type": "simulation", "simulation": {"timestamp": "t0"}},
                "t0",
            )
        return ({"type": "simulation", "simulation": {"timestamp": "t1"}}, "t1")

    lease = shared_controller_module._SHARED_STREAM_REGISTRY.subscribe(
        stream_key="hybrid:trimmed",
        collect_frame=_collect,
        refresh_seconds=0.02,
        heartbeat_seconds=1.0,
    )
    try:
        frame_one = lease.wait_for_frame(0, timeout_seconds=0.6)
        assert frame_one is not None
        deadline = time.monotonic() + 0.8
        frame_two = None
        while time.monotonic() < deadline:
            candidate = lease.wait_for_frame(frame_one.seq, timeout_seconds=0.1)
            if candidate is None:
                continue
            frame_two = candidate
            break
        assert frame_two is not None
        assert frame_two.seq > frame_one.seq
        assert frame_two.fingerprint == "t1"
    finally:
        lease.close()
