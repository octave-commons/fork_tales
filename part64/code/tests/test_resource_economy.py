from __future__ import annotations

import unittest.mock
from typing import Any
from code.world_web import resource_economy


def test_sync_sub_sim_presences_adds_new_impacts() -> None:
    impacts: list[dict[str, Any]] = [{"id": "existing"}]
    snapshot = {
        "simulations": [
            {
                "id": "container1",
                "name": "sim-alpha",
                "resources": {
                    "usage": {"cpu_percent": 50.0, "memory_usage_bytes": 1024},
                    "limits": {
                        "nano_cpus": 900_000_000,
                        "memory_limit_bytes": 2 * 1024 * 1024 * 1024,
                    },
                },
            }
        ]
    }
    resource_economy.sync_sub_sim_presences(impacts, snapshot)
    assert len(impacts) == 2
    new_impact = impacts[1]
    assert new_impact["id"] == "presence.sim.sim-alpha"
    assert new_impact["presence_type"] == "sub-sim"
    assert new_impact["container_id"] == "container1"
    assert new_impact["economy"]["cpu_usage"] == 50.0
    assert new_impact["economy"]["base_cpu_nano"] == 900_000_000
    assert new_impact["economy"]["base_mem_bytes"] == 2 * 1024 * 1024 * 1024


def test_process_resource_cycle_decays_wallet_and_updates_docker() -> None:
    impacts: list[dict[str, Any]] = [
        {
            "id": "presence.sim.sim-alpha",
            "presence_type": "sub-sim",
            "container_id": "container1",
            "resource_wallet": {"cpu": 10.0, "ram": 5.0},
            "economy": {
                "base_cpu_nano": 1_100_000_000,
                "base_mem_bytes": 6 * 1024 * 1024 * 1024,
            },
        }
    ]

    with unittest.mock.patch(
        "code.world_web.resource_economy.update_container_resources"
    ) as mock_update:
        mock_update.return_value = (True, "ok")

        # First tick - should update because no last update time
        resource_economy.process_resource_cycle(impacts, now=1000.0)

        # Verify wallet decayed
        wallet = impacts[0]["resource_wallet"]
        assert wallet["cpu"] < 10.0
        assert wallet["ram"] < 5.0

        # Verify docker update called
        assert mock_update.called
        args, kwargs = mock_update.call_args
        assert args[0] == "container1"
        assert kwargs["nano_cpus"] >= 1_100_000_000
        assert kwargs["memory_bytes"] >= 6 * 1024 * 1024 * 1024

        # Second tick immediate - should SKIP update due to cooldown
        mock_update.reset_mock()
        resource_economy.process_resource_cycle(impacts, now=1001.0)
        assert not mock_update.called

        # Third tick after cooldown - should update
        resource_economy.process_resource_cycle(impacts, now=1020.0)
        assert mock_update.called


def test_sync_sub_sim_presences_updates_existing_floor_without_losing_wallet() -> None:
    impacts: list[dict[str, Any]] = [
        {
            "id": "presence.sim.sim-alpha",
            "presence_type": "sub-sim",
            "container_id": "container1",
            "resource_wallet": {"cpu": 3.0, "ram": 2.0},
            "economy": {
                "base_cpu_nano": 250_000_000,
                "base_mem_bytes": 512 * 1024 * 1024,
            },
        }
    ]
    snapshot = {
        "simulations": [
            {
                "id": "container1",
                "name": "sim-alpha",
                "resources": {
                    "usage": {"cpu_percent": 12.0, "memory_usage_bytes": 2048},
                    "limits": {
                        "nano_cpus": 950_000_000,
                        "memory_limit_bytes": 3 * 1024 * 1024 * 1024,
                    },
                },
            }
        ]
    }

    resource_economy.sync_sub_sim_presences(impacts, snapshot)
    assert len(impacts) == 1
    impact = impacts[0]
    assert impact["resource_wallet"]["cpu"] == 3.0
    assert impact["resource_wallet"]["ram"] == 2.0
    assert impact["economy"]["base_cpu_nano"] == 950_000_000
    assert impact["economy"]["base_mem_bytes"] == 3 * 1024 * 1024 * 1024
