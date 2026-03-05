"""Websocket cached payload helpers for simulation stream paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def simulation_ws_normalize_delta_stream_mode(
    mode: str,
    *,
    default_delta_stream_mode: str,
) -> str:
    clean = str(mode or "").strip().lower()
    if clean in {"worker", "workers", "thread", "threads", "subsystem", "subsystems"}:
        return "workers"
    if clean in {"world", "single", "legacy", "combined"}:
        return "world"
    if default_delta_stream_mode in {
        "worker",
        "workers",
        "thread",
        "threads",
        "subsystem",
        "subsystems",
    }:
        return "workers"
    return "world"


def simulation_ws_normalize_payload_mode(mode: str) -> str:
    clean = str(mode or "").strip().lower()
    if clean in {"full", "complete", "debug", "debug-full"}:
        return "full"
    return "trimmed"


def simulation_ws_normalize_particle_payload_mode(mode: str) -> str:
    clean = str(mode or "").strip().lower()
    if clean in {"full", "rich", "complete", "debug"}:
        return "full"
    return "lite"


def simulation_ws_lite_field_particles(
    rows: Any,
    *,
    max_rows: int | None = None,
    particle_lite_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    row_limit = len(rows) if max_rows is None else max(0, int(max_rows))
    if row_limit <= 0:
        return []
    compact_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index >= row_limit:
            break
        if not isinstance(row, dict):
            continue
        compact: dict[str, Any] = {}
        for key in particle_lite_keys:
            if key in row:
                compact[key] = row.get(key)
        if not str(compact.get("id", "") or "").strip():
            compact["id"] = str(row.get("id", "") or f"ws:{index}")
        compact_rows.append(compact)
    return compact_rows


def simulation_ws_load_cached_payload(
    *,
    part_root: Path,
    perspective: str,
    payload_mode: str = "trimmed",
    allow_disabled_particle_dynamics: bool = False,
    normalize_payload_mode: Callable[[str], str],
    http_compact_cached_body_reader: Callable[..., bytes | None],
    http_cached_body_reader: Callable[..., bytes | None],
    http_disk_cache_load: Callable[..., bytes | None],
    http_disk_cache_path: Callable[[Path, str], Path],
    ws_decode_cached_payload: Callable[[Any], dict[str, Any] | None],
    ws_cache_max_age_seconds: float,
    ws_payload_is_sparse: Callable[[dict[str, Any]], bool],
    ws_payload_has_disabled_particle_dynamics: Callable[[dict[str, Any]], bool],
    ws_payload_is_bootstrap_only: Callable[[dict[str, Any]], bool],
    ws_collect_node_positions: Callable[
        [dict[str, Any]], tuple[dict[str, tuple[float, float]], dict[str, float]]
    ],
    ws_trim_simulation_payload: Callable[[dict[str, Any]], dict[str, Any]],
    ws_compact_graph_payload: Callable[..., dict[str, Any]],
    ws_extract_stream_particles: Callable[..., Any],
    safe_float: Callable[[Any, float], float],
    max_graph_node_positions: int = 2200,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    payload_mode_key = normalize_payload_mode(payload_mode)
    cache_profile = "full" if payload_mode_key == "full" else "compact"
    cache_perspective = f"{perspective}|profile:{cache_profile}"
    loaded_from_compact_cache = False

    def _load_stale_disk_payload() -> dict[str, Any] | None:
        for perspective_key in (cache_perspective, perspective):
            stale_cache_path = http_disk_cache_path(part_root, perspective_key)
            try:
                if stale_cache_path.exists() and stale_cache_path.is_file():
                    stale_body = stale_cache_path.read_bytes()
                    decoded = ws_decode_cached_payload(stale_body)
                    if isinstance(decoded, dict):
                        return decoded
            except Exception:
                continue
        return None

    payload: dict[str, Any] | None = None
    if payload_mode_key != "full":
        compact_cached_body = http_compact_cached_body_reader(
            perspective=cache_perspective,
            max_age_seconds=ws_cache_max_age_seconds,
        )
        if compact_cached_body is None:
            compact_cached_body = http_compact_cached_body_reader(
                perspective=perspective,
                max_age_seconds=ws_cache_max_age_seconds,
            )
        payload = ws_decode_cached_payload(compact_cached_body)
        loaded_from_compact_cache = isinstance(payload, dict)

    if payload is None:
        cached_body = http_cached_body_reader(
            perspective=cache_perspective,
            max_age_seconds=ws_cache_max_age_seconds,
        )
        if cached_body is None:
            cached_body = http_cached_body_reader(
                perspective=perspective,
                max_age_seconds=ws_cache_max_age_seconds,
            )
        if cached_body is None:
            cached_body = http_disk_cache_load(
                part_root,
                perspective=cache_perspective,
                max_age_seconds=ws_cache_max_age_seconds,
            )
        if cached_body is None:
            cached_body = http_disk_cache_load(
                part_root,
                perspective=perspective,
                max_age_seconds=ws_cache_max_age_seconds,
            )

        payload = ws_decode_cached_payload(cached_body)
    if payload_mode_key == "full":
        stale_payload = _load_stale_disk_payload()
        if payload is None:
            payload = stale_payload
        elif (
            ws_payload_is_sparse(payload)
            and isinstance(stale_payload, dict)
            and not ws_payload_is_sparse(stale_payload)
        ):
            payload = stale_payload

    if payload is None:
        return None

    if (
        ws_payload_has_disabled_particle_dynamics(payload)
        and not allow_disabled_particle_dynamics
    ):
        return None

    if ws_payload_is_bootstrap_only(payload):
        return None

    projection = payload.get("projection", {})
    if not isinstance(projection, dict):
        projection = {}
    if payload_mode_key == "full":
        simulation_payload = dict(payload)
        simulation_payload.pop("projection", None)
        return simulation_payload, projection

    node_positions, node_text_chars = ws_collect_node_positions(payload)
    simulation_payload = ws_trim_simulation_payload(payload)
    simulation_payload.pop("projection", None)
    simulation_payload.update(
        ws_compact_graph_payload(
            payload,
            assume_trimmed=loaded_from_compact_cache,
        )
    )
    ws_extract_stream_particles(
        simulation_payload,
        node_positions=node_positions,
        node_text_chars=node_text_chars,
    )
    dynamics = simulation_payload.get("presence_dynamics", {})
    if isinstance(dynamics, dict) and node_positions:
        graph_node_positions: dict[str, dict[str, float]] = {}
        for node_id, coords in node_positions.items():
            if len(graph_node_positions) >= max_graph_node_positions:
                break
            if not (isinstance(coords, tuple) and len(coords) >= 2):
                continue
            graph_node_positions[node_id] = {
                "x": round(max(0.0, min(1.0, safe_float(coords[0], 0.5))), 5),
                "y": round(max(0.0, min(1.0, safe_float(coords[1], 0.5))), 5),
            }
        dynamics["graph_node_positions"] = graph_node_positions
        simulation_payload["presence_dynamics"] = dynamics
    return simulation_payload, projection


def simulation_ws_should_bridge_to_http_cache(
    simulation_payload: dict[str, Any],
    *,
    min_field_particles: int,
    payload_has_disabled_particle_dynamics: Callable[[dict[str, Any]], bool],
) -> bool:
    if not isinstance(simulation_payload, dict):
        return False
    if payload_has_disabled_particle_dynamics(simulation_payload):
        return False

    dynamics = simulation_payload.get("presence_dynamics", {})
    if not isinstance(dynamics, dict):
        return False
    rows = dynamics.get("field_particles", [])
    if not isinstance(rows, list):
        return False

    min_rows = max(1, int(min_field_particles))
    return len(rows) >= min_rows
