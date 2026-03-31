from __future__ import annotations

from pathlib import Path
from typing import Any

from code.world_web import simulation_ws_shared_payload_utils as shared_payload_utils


def test_collect_shared_stream_frame_uses_cached_timestamp_fingerprint() -> None:
    simulation_payload = {
        "timestamp": "2026-03-05T17:00:00Z",
        "presence_dynamics": {"field_particles": []},
    }
    projection_payload = {"record": "家_映.v1", "perspective": "hybrid"}

    def _load_cached_payload(**_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return simulation_payload, projection_payload

    payload, fingerprint = shared_payload_utils.collect_shared_stream_frame(
        part_root=Path("."),
        perspective="hybrid",
        payload_mode="trimmed",
        load_cached_payload=_load_cached_payload,
        json_compact=lambda value: str(value),
    )

    assert payload.get("type") == "simulation"
    assert payload.get("simulation") == simulation_payload
    assert payload.get("projection") == projection_payload
    assert fingerprint == "2026-03-05T17:00:00Z"


def test_collect_shared_stream_frame_builds_bootstrap_payload_on_cache_miss() -> None:
    def _load_cached_payload(**_kwargs: Any) -> None:
        return None

    payload, fingerprint = shared_payload_utils.collect_shared_stream_frame(
        part_root=Path("."),
        perspective="hybrid",
        payload_mode="trimmed",
        load_cached_payload=_load_cached_payload,
        json_compact=lambda value: str(value),
    )

    assert fingerprint == "shared-ws-cache-miss"
    assert payload.get("type") == "simulation"
    simulation_payload = payload.get("simulation", {})
    assert isinstance(simulation_payload, dict)
    assert simulation_payload.get("perspective") == "hybrid"
    dynamics = simulation_payload.get("presence_dynamics", {})
    assert isinstance(dynamics, dict)
    daimoi = dynamics.get("daimoi_probabilistic", {})
    assert isinstance(daimoi, dict)
    assert daimoi.get("disabled_reason") == "shared_ws_cache_miss"
