from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

from code.world_web import server as world_server


def test_runtime_guard_state_marks_critical_from_resource_pressure() -> None:
    guard = world_server._runtime_guard_state(
        {
            "devices": {
                "cpu": {
                    "utilization": 96.0,
                    "memory_pressure": 0.94,
                }
            },
            "log_watch": {
                "error_ratio": 0.7,
            },
            "hot_devices": ["cpu"],
        }
    )

    assert guard["mode"] == "critical"
    assert "cpu_hot" in guard["reasons"]
    assert "memory_pressure_high" in guard["reasons"]
    assert "runtime_log_error_ratio_high" in guard["reasons"]


def test_runtime_guard_state_marks_degraded_for_hot_devices() -> None:
    guard = world_server._runtime_guard_state(
        {
            "devices": {
                "cpu": {
                    "utilization": 35.0,
                    "memory_pressure": 0.3,
                }
            },
            "log_watch": {
                "error_ratio": 0.2,
            },
            "hot_devices": ["gpu0"],
        }
    )

    assert guard["mode"] == "degraded"
    assert "hot_devices" in guard["reasons"]


def test_runtime_ws_client_capacity_is_enforced() -> None:
    previous_max = world_server._RUNTIME_WS_MAX_CLIENTS
    previous_count = world_server._RUNTIME_WS_CLIENT_COUNT
    try:
        world_server._RUNTIME_WS_MAX_CLIENTS = 2
        world_server._RUNTIME_WS_CLIENT_COUNT = 0

        assert world_server._runtime_ws_try_acquire_client_slot() is True
        assert world_server._runtime_ws_try_acquire_client_slot() is True
        assert world_server._runtime_ws_try_acquire_client_slot() is False
        assert world_server._runtime_ws_client_snapshot()["active_clients"] == 2

        world_server._runtime_ws_release_client_slot()
        assert world_server._runtime_ws_client_snapshot()["active_clients"] == 1
    finally:
        world_server._RUNTIME_WS_MAX_CLIENTS = previous_max
        world_server._RUNTIME_WS_CLIENT_COUNT = previous_count


def test_runtime_health_payload_includes_guard_and_websocket_snapshot() -> None:
    with tempfile.TemporaryDirectory() as td:
        payload = world_server._runtime_health_payload(Path(td))
    assert payload["ok"] is True
    assert payload["record"] == "eta-mu.runtime-health.v1"
    assert isinstance(payload.get("guard"), dict)
    assert isinstance(payload.get("websocket"), dict)


def test_simulation_http_cached_body_requires_exact_key_when_requested() -> None:
    with world_server._SIMULATION_HTTP_CACHE_LOCK:
        previous = dict(world_server._SIMULATION_HTTP_CACHE)
        world_server._SIMULATION_HTTP_CACHE["key"] = "hybrid|a|b|1|2|0.1|3"
        world_server._SIMULATION_HTTP_CACHE["prepared_monotonic"] = time.monotonic()
        world_server._SIMULATION_HTTP_CACHE["body"] = b'{"ok":true}'
    try:
        hit = world_server._simulation_http_cached_body(
            cache_key="hybrid|a|b|1|2|0.1|3",
            perspective="hybrid",
            max_age_seconds=1.0,
            require_exact_key=True,
        )
        miss = world_server._simulation_http_cached_body(
            cache_key="hybrid|x|y|1|2|0.1|3",
            perspective="hybrid",
            max_age_seconds=1.0,
            require_exact_key=True,
        )
        assert hit == b'{"ok":true}'
        assert miss is None
    finally:
        with world_server._SIMULATION_HTTP_CACHE_LOCK:
            world_server._SIMULATION_HTTP_CACHE.update(previous)


def test_simulation_http_cached_body_can_fallback_by_perspective() -> None:
    with world_server._SIMULATION_HTTP_CACHE_LOCK:
        previous = dict(world_server._SIMULATION_HTTP_CACHE)
        world_server._SIMULATION_HTTP_CACHE["key"] = "hybrid|f|g|4|5|0.2|9"
        world_server._SIMULATION_HTTP_CACHE["prepared_monotonic"] = (
            time.monotonic() - 0.4
        )
        world_server._SIMULATION_HTTP_CACHE["body"] = b'{"ok":true,"cached":1}'
    try:
        fallback_hit = world_server._simulation_http_cached_body(
            perspective="hybrid",
            max_age_seconds=2.0,
        )
        fallback_miss = world_server._simulation_http_cached_body(
            perspective="lineage",
            max_age_seconds=2.0,
        )
        assert fallback_hit == b'{"ok":true,"cached":1}'
        assert fallback_miss is None
    finally:
        with world_server._SIMULATION_HTTP_CACHE_LOCK:
            world_server._SIMULATION_HTTP_CACHE.update(previous)


def test_simulation_http_cached_body_respects_age_limit() -> None:
    with world_server._SIMULATION_HTTP_CACHE_LOCK:
        previous = dict(world_server._SIMULATION_HTTP_CACHE)
        world_server._SIMULATION_HTTP_CACHE["key"] = "hybrid|old|entry|1|1|0.0|0"
        world_server._SIMULATION_HTTP_CACHE["prepared_monotonic"] = (
            time.monotonic() - 5.0
        )
        world_server._SIMULATION_HTTP_CACHE["body"] = b'{"ok":true,"cached":"old"}'
    try:
        expired = world_server._simulation_http_cached_body(
            perspective="hybrid",
            max_age_seconds=0.5,
        )
        assert expired is None
    finally:
        with world_server._SIMULATION_HTTP_CACHE_LOCK:
            world_server._SIMULATION_HTTP_CACHE.update(previous)


def test_simulation_http_cache_key_tracks_queue_pressure_but_ignores_noise() -> None:
    catalog = {
        "file_graph": {"generated_at": "fg"},
        "crawler_graph": {"generated_at": "cg"},
    }
    key_a = world_server._simulation_http_cache_key(
        perspective="hybrid",
        catalog=catalog,
        queue_snapshot={"pending_count": 7, "event_count": 33},
        influence_snapshot={"queue_ratio": 0.104, "compute_jobs_180s": 9},
    )
    key_b = world_server._simulation_http_cache_key(
        perspective="hybrid",
        catalog=catalog,
        queue_snapshot={"pending_count": 7, "event_count": 39},
        influence_snapshot={"queue_ratio": 0.103, "compute_jobs_180s": 11},
    )
    assert key_a != key_b

    key_c = world_server._simulation_http_cache_key(
        perspective="hybrid",
        catalog=catalog,
        queue_snapshot={"pending_count": 7, "event_count": 33},
        influence_snapshot={"queue_ratio": 0.193, "compute_jobs_180s": 44},
    )
    assert key_a == key_c


def test_simulation_http_wait_for_exact_cache_returns_inflight_hit() -> None:
    with world_server._SIMULATION_HTTP_CACHE_LOCK:
        previous = dict(world_server._SIMULATION_HTTP_CACHE)
        world_server._SIMULATION_HTTP_CACHE["key"] = ""
        world_server._SIMULATION_HTTP_CACHE["prepared_monotonic"] = 0.0
        world_server._SIMULATION_HTTP_CACHE["body"] = b""

    def _writer() -> None:
        time.sleep(0.05)
        with world_server._SIMULATION_HTTP_CACHE_LOCK:
            world_server._SIMULATION_HTTP_CACHE["key"] = "hybrid|fg|cg|7|4|0.10|2"
            world_server._SIMULATION_HTTP_CACHE["prepared_monotonic"] = time.monotonic()
            world_server._SIMULATION_HTTP_CACHE["body"] = b'{"ok":true,"inflight":1}'

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()
    try:
        hit = world_server._simulation_http_wait_for_exact_cache(
            cache_key="hybrid|fg|cg|7|4|0.10|2",
            perspective="hybrid",
            max_wait_seconds=0.5,
            poll_seconds=0.01,
        )
        assert hit == b'{"ok":true,"inflight":1}'
    finally:
        writer.join(timeout=1.0)
        with world_server._SIMULATION_HTTP_CACHE_LOCK:
            world_server._SIMULATION_HTTP_CACHE.update(previous)


def test_simulation_http_trim_catalog_caps_heavy_lists() -> None:
    previous = {
        "enabled": world_server._SIMULATION_HTTP_TRIM_ENABLED,
        "max_items": world_server._SIMULATION_HTTP_MAX_ITEMS,
        "max_file_nodes": world_server._SIMULATION_HTTP_MAX_FILE_NODES,
        "max_file_edges": world_server._SIMULATION_HTTP_MAX_FILE_EDGES,
        "max_field_nodes": world_server._SIMULATION_HTTP_MAX_FIELD_NODES,
        "max_tag_nodes": world_server._SIMULATION_HTTP_MAX_TAG_NODES,
        "max_render_nodes": world_server._SIMULATION_HTTP_MAX_RENDER_NODES,
        "max_crawler_nodes": world_server._SIMULATION_HTTP_MAX_CRAWLER_NODES,
        "max_crawler_edges": world_server._SIMULATION_HTTP_MAX_CRAWLER_EDGES,
        "max_crawler_field_nodes": world_server._SIMULATION_HTTP_MAX_CRAWLER_FIELD_NODES,
    }
    try:
        world_server._SIMULATION_HTTP_TRIM_ENABLED = True
        world_server._SIMULATION_HTTP_MAX_ITEMS = 2
        world_server._SIMULATION_HTTP_MAX_FILE_NODES = 3
        world_server._SIMULATION_HTTP_MAX_FILE_EDGES = 4
        world_server._SIMULATION_HTTP_MAX_FIELD_NODES = 1
        world_server._SIMULATION_HTTP_MAX_TAG_NODES = 1
        world_server._SIMULATION_HTTP_MAX_RENDER_NODES = 2
        world_server._SIMULATION_HTTP_MAX_CRAWLER_NODES = 2
        world_server._SIMULATION_HTTP_MAX_CRAWLER_EDGES = 3
        world_server._SIMULATION_HTTP_MAX_CRAWLER_FIELD_NODES = 1

        catalog = {
            "items": [1, 2, 3, 4],
            "file_graph": {
                "file_nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
                "field_nodes": [{"id": "f1"}, {"id": "f2"}],
                "tag_nodes": [{"id": "t1"}, {"id": "t2"}],
                "nodes": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
                "edges": [
                    {"id": "e1"},
                    {"id": "e2"},
                    {"id": "e3"},
                    {"id": "e4"},
                    {"id": "e5"},
                ],
                "stats": {"file_count": 4, "edge_count": 5},
            },
            "crawler_graph": {
                "crawler_nodes": [{"id": "c1"}, {"id": "c2"}, {"id": "c3"}],
                "field_nodes": [{"id": "cf1"}, {"id": "cf2"}],
                "nodes": [{"id": "cn1"}, {"id": "cn2"}, {"id": "cn3"}],
                "edges": [
                    {"id": "ce1"},
                    {"id": "ce2"},
                    {"id": "ce3"},
                    {"id": "ce4"},
                ],
                "stats": {"crawler_count": 3, "edge_count": 4},
            },
        }
        trimmed = world_server._simulation_http_trim_catalog(catalog)

        assert len(trimmed["items"]) == 2
        assert len(trimmed["file_graph"]["file_nodes"]) == 3
        assert len(trimmed["file_graph"]["edges"]) == 4
        assert len(trimmed["crawler_graph"]["crawler_nodes"]) == 2
        assert len(trimmed["crawler_graph"]["edges"]) == 3

        assert len(catalog["items"]) == 4
        assert len(catalog["file_graph"]["file_nodes"]) == 4
        assert len(catalog["crawler_graph"]["crawler_nodes"]) == 3
    finally:
        world_server._SIMULATION_HTTP_TRIM_ENABLED = bool(previous["enabled"])
        world_server._SIMULATION_HTTP_MAX_ITEMS = int(previous["max_items"])
        world_server._SIMULATION_HTTP_MAX_FILE_NODES = int(previous["max_file_nodes"])
        world_server._SIMULATION_HTTP_MAX_FILE_EDGES = int(previous["max_file_edges"])
        world_server._SIMULATION_HTTP_MAX_FIELD_NODES = int(previous["max_field_nodes"])
        world_server._SIMULATION_HTTP_MAX_TAG_NODES = int(previous["max_tag_nodes"])
        world_server._SIMULATION_HTTP_MAX_RENDER_NODES = int(
            previous["max_render_nodes"]
        )
        world_server._SIMULATION_HTTP_MAX_CRAWLER_NODES = int(
            previous["max_crawler_nodes"]
        )
        world_server._SIMULATION_HTTP_MAX_CRAWLER_EDGES = int(
            previous["max_crawler_edges"]
        )
        world_server._SIMULATION_HTTP_MAX_CRAWLER_FIELD_NODES = int(
            previous["max_crawler_field_nodes"]
        )


def test_simulation_http_disk_cache_store_and_load_roundtrip() -> None:
    previous_enabled = world_server._SIMULATION_HTTP_DISK_CACHE_ENABLED
    previous_seconds = world_server._SIMULATION_HTTP_DISK_CACHE_SECONDS
    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        try:
            world_server._SIMULATION_HTTP_DISK_CACHE_ENABLED = True
            world_server._SIMULATION_HTTP_DISK_CACHE_SECONDS = 120.0

            payload = b'{"ok":true,"disk":1}'
            world_server._simulation_http_disk_cache_store(
                part_root,
                perspective="hybrid",
                body=payload,
            )

            loaded = world_server._simulation_http_disk_cache_load(
                part_root,
                perspective="hybrid",
                max_age_seconds=120.0,
            )
            assert loaded == payload

            cache_path = world_server._simulation_http_disk_cache_path(
                part_root, "hybrid"
            )
            stale_ts = time.time() - 360.0
            os.utime(cache_path, (stale_ts, stale_ts))
            expired = world_server._simulation_http_disk_cache_load(
                part_root,
                perspective="hybrid",
                max_age_seconds=30.0,
            )
            assert expired is None
        finally:
            world_server._SIMULATION_HTTP_DISK_CACHE_ENABLED = previous_enabled
            world_server._SIMULATION_HTTP_DISK_CACHE_SECONDS = previous_seconds


def test_simulation_http_failure_backoff_records_and_clears() -> None:
    previous_cooldown = world_server._SIMULATION_HTTP_FAILURE_COOLDOWN_SECONDS
    previous_reset = world_server._SIMULATION_HTTP_FAILURE_STREAK_RESET_SECONDS
    with world_server._SIMULATION_HTTP_FAILURE_LOCK:
        previous_state = dict(world_server._SIMULATION_HTTP_FAILURE_STATE)
    try:
        world_server._SIMULATION_HTTP_FAILURE_COOLDOWN_SECONDS = 2.0
        world_server._SIMULATION_HTTP_FAILURE_STREAK_RESET_SECONDS = 10.0
        world_server._simulation_http_failure_clear()

        remaining, error_name, streak = (
            world_server._simulation_http_failure_backoff_snapshot()
        )
        assert remaining == 0.0
        assert error_name == ""
        assert streak == 0

        world_server._simulation_http_failure_record("MemoryError")
        remaining, error_name, streak = (
            world_server._simulation_http_failure_backoff_snapshot()
        )
        assert 0.0 < remaining <= 2.0
        assert error_name == "MemoryError"
        assert streak == 1

        world_server._simulation_http_failure_record("TimeoutError")
        _, error_name, streak = world_server._simulation_http_failure_backoff_snapshot()
        assert error_name == "TimeoutError"
        assert streak == 2

        world_server._simulation_http_failure_clear()
        remaining, error_name, streak = (
            world_server._simulation_http_failure_backoff_snapshot()
        )
        assert remaining == 0.0
        assert error_name == ""
        assert streak == 0
    finally:
        world_server._SIMULATION_HTTP_FAILURE_COOLDOWN_SECONDS = previous_cooldown
        world_server._SIMULATION_HTTP_FAILURE_STREAK_RESET_SECONDS = previous_reset
        with world_server._SIMULATION_HTTP_FAILURE_LOCK:
            world_server._SIMULATION_HTTP_FAILURE_STATE.update(previous_state)


def test_simulation_http_failure_streak_resets_after_window() -> None:
    previous_reset = world_server._SIMULATION_HTTP_FAILURE_STREAK_RESET_SECONDS
    with world_server._SIMULATION_HTTP_FAILURE_LOCK:
        previous_state = dict(world_server._SIMULATION_HTTP_FAILURE_STATE)
    try:
        world_server._SIMULATION_HTTP_FAILURE_STREAK_RESET_SECONDS = 1.0
        with world_server._SIMULATION_HTTP_FAILURE_LOCK:
            world_server._SIMULATION_HTTP_FAILURE_STATE["last_failure_monotonic"] = (
                time.monotonic() - 4.0
            )
            world_server._SIMULATION_HTTP_FAILURE_STATE["last_error"] = "Old"
            world_server._SIMULATION_HTTP_FAILURE_STATE["streak"] = 7

        world_server._simulation_http_failure_record("Fresh")
        _, error_name, streak = world_server._simulation_http_failure_backoff_snapshot()
        assert error_name == "Fresh"
        assert streak == 1
    finally:
        world_server._SIMULATION_HTTP_FAILURE_STREAK_RESET_SECONDS = previous_reset
        with world_server._SIMULATION_HTTP_FAILURE_LOCK:
            world_server._SIMULATION_HTTP_FAILURE_STATE.update(previous_state)
