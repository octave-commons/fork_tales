from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
from typing import Any

from code.world_web import sim_slice_bridge


def test_sim_point_budget_slice_local_thresholds(monkeypatch: Any) -> None:
    monkeypatch.setenv("SIM_SLICE_OFFLOAD_MODE", "local")

    high_budget, high_meta = sim_slice_bridge.resolve_sim_point_budget_slice(
        cpu_utilization=94.0,
        max_sim_points=1000,
    )
    medium_budget, medium_meta = sim_slice_bridge.resolve_sim_point_budget_slice(
        cpu_utilization=82.0,
        max_sim_points=1000,
    )
    low_budget, low_meta = sim_slice_bridge.resolve_sim_point_budget_slice(
        cpu_utilization=41.0,
        max_sim_points=1000,
    )

    assert high_budget == 550
    assert medium_budget == 740
    assert low_budget == 1000
    assert high_meta.get("source") == "python-local"
    assert medium_meta.get("fallback") is False
    assert low_meta.get("mode") == "local"


def test_sim_point_budget_slice_redis_fallback_when_cli_missing(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("SIM_SLICE_OFFLOAD_MODE", "redis")
    monkeypatch.setenv("SIM_SLICE_REDIS_CLI", "./nonexistent-redis-cli")

    budget, meta = sim_slice_bridge.resolve_sim_point_budget_slice(
        cpu_utilization=91.0,
        max_sim_points=800,
    )

    assert budget == max(256, int(800 * 0.55))
    assert meta.get("fallback") is True
    assert str(meta.get("source", "")).startswith("python-fallback")
    assert "redis-cli" in str(meta.get("reason", ""))


def test_sim_point_budget_slice_redis_reads_worker_payload(monkeypatch: Any) -> None:
    monkeypatch.setenv("SIM_SLICE_OFFLOAD_MODE", "redis")
    monkeypatch.setenv("SIM_SLICE_REDIS_CLI", "redis-cli")

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert check is False
        assert timeout > 0.0

        if "XADD" in command:
            return subprocess.CompletedProcess(command, 0, "1700000000000-0\n", "")
        if "GET" in command:
            payload = {
                "sim_point_budget": 612,
                "source": "c-budget-worker.v1",
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        if "DEL" in command:
            return subprocess.CompletedProcess(command, 0, "1\n", "")
        raise AssertionError(f"unexpected redis-cli command: {command}")

    monkeypatch.setattr(sim_slice_bridge.subprocess, "run", fake_run)

    budget, meta = sim_slice_bridge.resolve_sim_point_budget_slice(
        cpu_utilization=85.0,
        max_sim_points=1000,
    )

    assert budget == 612
    assert meta.get("mode") == "redis"
    assert meta.get("fallback") is False
    assert meta.get("source") == "c-budget-worker.v1"


def test_sim_point_budget_slice_uds_fallback_when_socket_missing(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("SIM_SLICE_OFFLOAD_MODE", "uds")
    monkeypatch.setenv("SIM_SLICE_UDS_PATH", "/tmp/nonexistent-sim-slice.sock")

    budget, meta = sim_slice_bridge.resolve_sim_point_budget_slice(
        cpu_utilization=91.0,
        max_sim_points=800,
    )

    assert budget == max(256, int(800 * 0.55))
    assert meta.get("mode") == "uds"
    assert meta.get("fallback") is True
    assert str(meta.get("reason", "")).startswith("uds-")


def test_sim_point_budget_slice_uds_reads_worker_payload(monkeypatch: Any) -> None:
    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    socket_path = f"/tmp/sim-slice-test-{os.getpid()}-{threading.get_ident()}.sock"
    try:
        server_socket.bind(socket_path)
        server_socket.listen(1)

        def serve_once() -> None:
            conn, _ = server_socket.accept()
            with conn:
                payload_raw = b""
                while not payload_raw.endswith(b"\n"):
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    payload_raw += chunk
                payload = json.loads(payload_raw.decode("utf-8"))
                response = {
                    "job_id": payload.get("job_id", ""),
                    "sim_point_budget": 677,
                    "source": "c-uds-worker.v1",
                }
                conn.sendall((json.dumps(response) + "\n").encode("utf-8"))

        thread = threading.Thread(target=serve_once, daemon=True)
        thread.start()

        monkeypatch.setenv("SIM_SLICE_OFFLOAD_MODE", "uds")
        monkeypatch.setenv("SIM_SLICE_UDS_PATH", socket_path)
        monkeypatch.setenv("SIM_SLICE_UDS_TIMEOUT_MS", "120")

        budget, meta = sim_slice_bridge.resolve_sim_point_budget_slice(
            cpu_utilization=85.0,
            max_sim_points=1000,
        )
        thread.join(timeout=2.0)

        assert budget == 677
        assert meta.get("mode") == "uds"
        assert meta.get("fallback") is False
        assert meta.get("source") == "c-uds-worker.v1"
    finally:
        server_socket.close()
        try:
            os.unlink(socket_path)
        except OSError:
            pass


def test_sim_point_budget_slice_async_uses_double_buffer_snapshot(
    monkeypatch: Any,
) -> None:
    class _FakeAsyncWorker:
        def __init__(self) -> None:
            self.submissions: list[dict[str, Any]] = []

        def submit(
            self,
            *,
            mode: str,
            cpu_utilization: float,
            max_sim_points: int,
            local_budget: int,
        ) -> None:
            self.submissions.append(
                {
                    "mode": mode,
                    "cpu_utilization": cpu_utilization,
                    "max_sim_points": max_sim_points,
                    "local_budget": local_budget,
                }
            )

        def latest(self) -> dict[str, Any]:
            return {
                "ready": True,
                "mode": "uds",
                "budget": 701,
                "source": "c-uds-worker.v1",
                "fallback": False,
                "reason": "",
                "job_id": "abc123",
                "transport_latency_ms": 4.2,
                "produced_monotonic": (sim_slice_bridge.time.monotonic() - 0.01),
            }

    fake_worker = _FakeAsyncWorker()
    monkeypatch.setattr(sim_slice_bridge, "_SIM_POINT_BUDGET_ASYNC_WORKER", fake_worker)
    monkeypatch.setenv("SIM_SLICE_OFFLOAD_MODE", "uds")
    monkeypatch.setenv("SIM_SLICE_ASYNC", "1")

    budget, meta = sim_slice_bridge.resolve_sim_point_budget_slice(
        cpu_utilization=82.0,
        max_sim_points=1000,
    )

    assert budget == 701
    assert meta.get("mode") == "uds"
    assert meta.get("source") == "c-uds-worker.v1"
    assert meta.get("fallback") is False
    assert meta.get("async") is True
    assert fake_worker.submissions


def test_sim_point_budget_slice_async_warmup_falls_back_local(
    monkeypatch: Any,
) -> None:
    class _FakeAsyncWorker:
        def submit(
            self,
            *,
            mode: str,
            cpu_utilization: float,
            max_sim_points: int,
            local_budget: int,
        ) -> None:
            return None

        def latest(self) -> dict[str, Any]:
            return {
                "ready": False,
                "mode": "uds",
                "budget": 0,
                "source": "",
                "fallback": True,
                "reason": "",
                "job_id": "",
                "transport_latency_ms": 0.0,
                "produced_monotonic": 0.0,
            }

    monkeypatch.setattr(
        sim_slice_bridge,
        "_SIM_POINT_BUDGET_ASYNC_WORKER",
        _FakeAsyncWorker(),
    )
    monkeypatch.setenv("SIM_SLICE_OFFLOAD_MODE", "uds")
    monkeypatch.setenv("SIM_SLICE_ASYNC", "1")

    budget, meta = sim_slice_bridge.resolve_sim_point_budget_slice(
        cpu_utilization=91.0,
        max_sim_points=800,
    )

    assert budget == max(256, int(800 * 0.55))
    assert meta.get("mode") == "uds"
    assert meta.get("source") == "python-local"
    assert meta.get("fallback") is True
    assert meta.get("reason") == "async-warmup"
    assert meta.get("async") is True
