from __future__ import annotations

from pathlib import Path
from typing import Any

from code.world_web import docker_runtime


def _container_row(
    *,
    container_id: str,
    name: str,
    service: str,
    project: str,
    labels: dict[str, str] | None = None,
    private_port: int = 8787,
    public_port: int = 19877,
    networks: list[str] | None = None,
    state: str = "running",
    status: str = "Up 2 minutes",
) -> dict[str, Any]:
    merged_labels = {
        "com.docker.compose.service": service,
        "com.docker.compose.project": project,
    }
    if isinstance(labels, dict):
        merged_labels.update(labels)

    return {
        "Id": container_id,
        "Names": [f"/{name}"],
        "Image": "fork-tales/eta-mu:latest",
        "State": state,
        "Status": status,
        "Created": 1700000000,
        "Labels": merged_labels,
        "Ports": [
            {
                "PrivatePort": private_port,
                "PublicPort": public_port,
                "Type": "tcp",
                "IP": "0.0.0.0",
            }
        ],
        "NetworkSettings": {
            "Networks": {
                network: {}
                for network in (
                    networks if isinstance(networks, list) else ["eta-mu-sim-net"]
                )
            }
        },
    }


def test_build_docker_simulation_snapshot_tracks_network_awareness() -> None:
    snapshot = docker_runtime.build_docker_simulation_snapshot(
        [
            _container_row(
                container_id="a" * 64,
                name="eta-mu-song-chaos",
                service="eta-mu-song-chaos",
                project="song-lab",
                labels={"io.fork_tales.simulation": "true"},
                public_port=19878,
            ),
            _container_row(
                container_id="b" * 64,
                name="eta-mu-song-stability",
                service="eta-mu-song-stability",
                project="song-lab",
                labels={"io.fork_tales.simulation": "true"},
                public_port=19879,
            ),
            _container_row(
                container_id="c" * 64,
                name="sim-slice-worker-c",
                service="sim-slice-worker-c",
                project="bench",
                labels={"io.fork_tales.simulation": "false"},
                private_port=7711,
                public_port=0,
            ),
        ],
        generated_at="2026-02-19T09:00:00+00:00",
    )

    assert snapshot["ok"] is True
    assert snapshot["summary"]["running_simulations"] == 2
    assert snapshot["summary"]["total_simulations"] == 2
    assert snapshot["summary"]["network_clusters"] == 1
    assert len(snapshot["simulations"]) == 2

    first = snapshot["simulations"][0]
    second = snapshot["simulations"][1]
    assert first["awareness"]["peer_count"] == 1
    assert second["awareness"]["peer_count"] == 1
    assert first["awareness"]["peer_names"] == [second["name"]]
    assert first["endpoints"][0]["kind"] == "world"
    assert first["endpoints"][0]["url"].startswith("http://127.0.0.1:")
    assert isinstance(snapshot["fingerprint"], str)
    assert len(snapshot["fingerprint"]) == 40


def test_build_snapshot_uses_name_fallback_only_with_runtime_port_hint() -> None:
    snapshot = docker_runtime.build_docker_simulation_snapshot(
        [
            _container_row(
                container_id="d" * 64,
                name="eta-mu-local",
                service="eta-mu-local",
                project="bench",
                labels={},
                private_port=8787,
                public_port=18877,
            ),
            _container_row(
                container_id="e" * 64,
                name="sim-slice-worker-uds",
                service="sim-slice-worker-uds",
                project="bench",
                labels={},
                private_port=7712,
                public_port=0,
            ),
        ],
        generated_at="2026-02-19T09:05:00+00:00",
    )

    assert snapshot["summary"]["running_simulations"] == 1
    assert len(snapshot["simulations"]) == 1
    assert snapshot["simulations"][0]["name"] == "eta-mu-local"


def test_build_snapshot_includes_gateway_route_and_control_policy() -> None:
    snapshot = docker_runtime.build_docker_simulation_snapshot(
        [
            _container_row(
                container_id="g" * 64,
                name="part64-eta-mu-system-1",
                service="eta-mu-system",
                project="part64",
                labels={
                    "io.fork_tales.simulation": "true",
                    "io.fork_tales.simulation.role": "world-runtime",
                },
                public_port=8787,
            ),
            _container_row(
                container_id="h" * 64,
                name="part64-eta-mu-cdb-1",
                service="eta-mu-cdb",
                project="part64",
                labels={"io.fork_tales.simulation": "true"},
                public_port=18880,
            ),
        ],
        generated_at="2026-02-19T22:40:00+00:00",
    )

    sims = snapshot["simulations"]
    system = next(row for row in sims if row["service"] == "eta-mu-system")
    cdb = next(row for row in sims if row["service"] == "eta-mu-cdb")

    assert system["route"]["id"] == "eta-mu-system"
    assert system["route"]["world_path"] == "/sim/eta-mu-system/"
    assert system["control"]["can_start_stop"] is False
    assert system["control"]["reason"] == "core_portal_runtime_protected"

    assert cdb["route"]["id"] == "eta-mu-cdb"
    assert cdb["route"]["world_path"] == "/sim/eta-mu-cdb/"
    assert cdb["route"]["weaver_path"] == "/sim/eta-mu-cdb/weaver/"
    assert cdb["control"]["can_start_stop"] is True


def test_control_simulation_container_handles_docker_status_codes(
    monkeypatch: Any,
) -> None:
    calls: list[str] = []

    def _fake_post(path: str, *, timeout_seconds: float | None = None) -> str:
        del timeout_seconds
        calls.append(path)
        if path.endswith("/start"):
            return ""
        if "/stop" in path:
            return "docker_api_http_304"
        return "docker_api_http_500"

    monkeypatch.setattr(docker_runtime, "_docker_api_post", _fake_post)

    ok_start, err_start = docker_runtime.control_simulation_container(
        "abc123",
        action="start",
    )
    assert ok_start is True
    assert err_start == ""

    ok_stop, err_stop = docker_runtime.control_simulation_container(
        "abc123",
        action="stop",
        stop_timeout_seconds=9.0,
    )
    assert ok_stop is True
    assert err_stop == "already_in_requested_state"
    assert calls[0].endswith("/containers/abc123/start")
    assert "/containers/abc123/stop?t=9" in calls[1]


def test_build_snapshot_includes_resource_usage_and_limit_pressure() -> None:
    container_id = "r" * 64
    snapshot = docker_runtime.build_docker_simulation_snapshot(
        [
            _container_row(
                container_id=container_id,
                name="eta-mu-resource",
                service="eta-mu-resource",
                project="bench",
                labels={"io.fork_tales.simulation": "true"},
                private_port=8787,
                public_port=18881,
            )
        ],
        generated_at="2026-02-19T10:00:00+00:00",
        inspect_by_id={
            container_id: {
                "HostConfig": {
                    "NanoCpus": 1_000_000_000,
                    "Memory": 536_870_912,
                    "PidsLimit": 120,
                }
            }
        },
        stats_by_id={
            container_id: {
                "cpu_stats": {
                    "cpu_usage": {"total_usage": 300_000_000},
                    "system_cpu_usage": 700_000_000,
                    "online_cpus": 2,
                },
                "precpu_stats": {
                    "cpu_usage": {"total_usage": 100_000_000},
                    "system_cpu_usage": 300_000_000,
                },
                "memory_stats": {
                    "usage": 314_572_800,
                    "limit": 4_294_967_296,
                },
                "pids_stats": {"current": 60},
                "networks": {
                    "eta-mu-sim-net": {
                        "rx_bytes": 10_240,
                        "tx_bytes": 2_048,
                    }
                },
                "blkio_stats": {
                    "io_service_bytes_recursive": [
                        {"op": "Read", "value": 4_096},
                        {"op": "Write", "value": 2_048},
                    ]
                },
            }
        },
    )

    simulation = snapshot["simulations"][0]
    resources = simulation["resources"]
    assert resources["limits"]["strict"] is True
    assert resources["limits"]["cpu_limit_cores"] == 1.0
    assert resources["usage"]["cpu_percent"] == 100.0
    assert resources["usage"]["memory_limit_bytes"] == 536_870_912
    assert resources["usage"]["memory_percent"] == 58.59
    assert resources["usage"]["pids_current"] == 60
    assert resources["usage"]["network_rx_bytes"] == 10_240
    assert resources["usage"]["blkio_write_bytes"] == 2_048
    assert resources["pressure"]["state"] == "critical"

    assert snapshot["summary"]["strict_simulations"] == 1
    assert snapshot["summary"]["constrained_simulations"] == 1
    assert snapshot["summary"]["unconstrained_simulations"] == 0
    assert snapshot["summary"]["critical_simulations"] == 1


def test_build_snapshot_includes_lifecycle_failure_signals() -> None:
    container_id = "l" * 64
    snapshot = docker_runtime.build_docker_simulation_snapshot(
        [
            _container_row(
                container_id=container_id,
                name="eta-mu-chaos",
                service="eta-mu-chaos",
                project="song",
                labels={"io.fork_tales.simulation": "true"},
                private_port=8787,
                public_port=19878,
            )
        ],
        generated_at="2026-02-19T10:05:00+00:00",
        inspect_by_id={
            container_id: {
                "HostConfig": {
                    "NanoCpus": 1_100_000_000,
                    "Memory": 6_501_171_200,
                    "PidsLimit": 112,
                },
                "RestartCount": 3,
                "State": {
                    "Running": True,
                    "Status": "running",
                    "OOMKilled": True,
                    "Error": "",
                    "Health": {
                        "Status": "unhealthy",
                        "FailingStreak": 5,
                        "Log": [
                            {
                                "ExitCode": 1,
                                "Output": "Health check exceeded timeout",
                            }
                        ],
                    },
                },
            }
        },
        stats_by_id={
            container_id: {
                "cpu_stats": {
                    "cpu_usage": {"total_usage": 900_000_000},
                    "system_cpu_usage": 1_700_000_000,
                    "online_cpus": 2,
                },
                "precpu_stats": {
                    "cpu_usage": {"total_usage": 100_000_000},
                    "system_cpu_usage": 900_000_000,
                },
                "memory_stats": {
                    "usage": 2_500_000_000,
                    "limit": 6_600_000_000,
                },
                "pids_stats": {"current": 28},
            }
        },
    )

    simulation = snapshot["simulations"][0]
    lifecycle = simulation["lifecycle"]
    assert lifecycle["stability"] == "failing"
    assert lifecycle["health_status"] == "unhealthy"
    assert lifecycle["oom_killed"] is True
    assert lifecycle["restart_count"] == 3
    assert "oom_killed" in lifecycle["signals"]
    assert "health_unhealthy" in lifecycle["signals"]
    assert "restarted" in lifecycle["signals"]

    assert snapshot["summary"]["failing_simulations"] == 1
    assert snapshot["summary"]["healthy_simulations"] == 0
    assert snapshot["summary"]["oom_killed_simulations"] == 1
    assert snapshot["summary"]["health_unhealthy_simulations"] == 1
    assert snapshot["summary"]["restarted_simulations"] == 1


def test_collect_snapshot_uses_cached_payload_when_refresh_fails(
    monkeypatch: Any,
) -> None:
    docker_runtime.reset_docker_simulation_cache_for_tests()

    good_rows = [
        _container_row(
            container_id="f" * 64,
            name="eta-mu-system",
            service="eta-mu-system",
            project="core",
            labels={"io.fork_tales.simulation": "true"},
            private_port=8787,
            public_port=8788,
        )
    ]

    monkeypatch.setattr(
        docker_runtime,
        "_docker_api_json",
        lambda _path: (good_rows, ""),
    )
    warm = docker_runtime.collect_docker_simulation_snapshot(force_refresh=True)
    assert warm["ok"] is True
    assert warm["summary"]["running_simulations"] == 1

    monkeypatch.setattr(
        docker_runtime,
        "_docker_api_json",
        lambda _path: (None, "docker_socket_missing"),
    )
    stale = docker_runtime.collect_docker_simulation_snapshot(force_refresh=True)
    assert stale["ok"] is False
    assert stale["stale"] is True
    assert stale["summary"]["running_simulations"] == 1
    assert stale["source"]["error"] == "docker_socket_missing"


def test_collect_resources_avoids_threadpool_when_single_worker(
    monkeypatch: Any,
) -> None:
    def _unexpected_threadpool(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("threadpool should not be used for single-worker mode")

    def _fake_docker_api_json(
        path: str, *, timeout_seconds: float | None = None
    ) -> tuple[Any, str]:
        del timeout_seconds
        if path.endswith("/json"):
            return ({"HostConfig": {}}, "")
        if path.endswith("/stats?stream=0"):
            return ({"memory_stats": {"usage": 0, "limit": 0}}, "")
        return (None, "unexpected_path")

    monkeypatch.setattr(docker_runtime, "DOCKER_SIMULATION_RESOURCE_WORKERS", 1)
    monkeypatch.setattr(docker_runtime, "ThreadPoolExecutor", _unexpected_threadpool)
    monkeypatch.setattr(docker_runtime, "_docker_api_json", _fake_docker_api_json)

    stats_by_id, inspect_by_id, errors_by_id = (
        docker_runtime._collect_container_resource_payloads(["abc123"])
    )

    assert "abc123" in inspect_by_id
    assert "abc123" in stats_by_id
    assert errors_by_id == {}


def test_docker_dashboard_static_includes_preview_contract() -> None:
    part_root = Path(__file__).resolve().parents[2]
    source = (
        part_root / "code" / "static" / "docker_simulations_dashboard.html"
    ).read_text("utf-8")

    assert "simulation-preview-frame" in source
    assert "preview unavailable for this container" in source
    assert "preview paused for stability" in source
    assert "previewUrlForSimulation" in source
    assert 'sandbox="allow-same-origin allow-scripts"' in source
    assert "Resource Budget" in source
    assert "resourceStateFromRatio" in source
    assert "/api/meta/overview" in source
    assert "/ws?stream=docker" in source
    assert "/api/docker/simulations/control" in source
    assert "open via gate" in source
    assert "/api/meta/notes" in source
    assert "/api/meta/objective/enqueue" in source
    assert "Training Charts" in source
    assert "renderTrainingCharts" in source
    assert "target_accuracy" in source
    assert "wsConnected" in source
    assert "pollSnapshotFallback" in source
