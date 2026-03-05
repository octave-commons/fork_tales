from __future__ import annotations

import json
import subprocess
import sys
from http import HTTPStatus
from pathlib import Path
from typing import Any


def _sim_manager_command(part_root: Path, *args: str) -> str:
    return subprocess.check_output(
        [sys.executable, "scripts/sim_manager.py", *args],
        cwd=part_root,
        text=True,
    )


def _parse_json_payload(raw_payload: str) -> Any:
    text = str(raw_payload or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {
            "ok": False,
            "error": "simulation_manager_invalid_json",
            "detail": text,
        }


def simulation_presets_get_response(*, part_root: Path) -> tuple[Any, int]:
    presets_path = part_root / "world_state" / "sim_presets.json"
    if not presets_path.exists():
        return {"presets": []}, int(HTTPStatus.OK)

    try:
        payload = _parse_json_payload(presets_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ok": False,
            "error": "simulation_presets_read_failed",
            "detail": exc.__class__.__name__,
        }, int(HTTPStatus.INTERNAL_SERVER_ERROR)
    return payload, int(HTTPStatus.OK)


def simulation_instances_list_response(*, part_root: Path) -> tuple[Any, int]:
    try:
        payload = _parse_json_payload(_sim_manager_command(part_root, "list-active"))
        return payload, int(HTTPStatus.OK)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, int(HTTPStatus.OK)


def simulation_instance_delete_response(
    *,
    part_root: Path,
    instance_id: str,
) -> tuple[Any, int]:
    target_id = str(instance_id or "").strip()
    try:
        payload = _parse_json_payload(
            _sim_manager_command(part_root, "stop", target_id)
        )
        return payload, int(HTTPStatus.OK)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, int(HTTPStatus.OK)


def simulation_instances_spawn_response(
    *,
    part_root: Path,
    req_payload: dict[str, Any],
) -> tuple[Any, int]:
    preset_id = str(req_payload.get("preset_id", "") or "").strip()
    port = 18900
    try:
        active_payload = _parse_json_payload(
            _sim_manager_command(part_root, "list-active")
        )
        used_ports: set[int] = set()
        if isinstance(active_payload, list):
            for row in active_payload:
                if not isinstance(row, dict):
                    continue
                try:
                    row_port = int(row.get("port", 0) or 0)
                except Exception:
                    row_port = 0
                if row_port > 0:
                    used_ports.add(row_port)
        while port in used_ports and port < 18950:
            port += 1

        payload = _parse_json_payload(
            _sim_manager_command(
                part_root,
                "spawn",
                "--preset",
                preset_id,
                "--port",
                str(port),
            )
        )
        return payload, int(HTTPStatus.OK)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, int(HTTPStatus.OK)
