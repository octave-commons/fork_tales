"""Small stateless server helper utilities."""

from __future__ import annotations

import hashlib
import mimetypes
import time
from pathlib import Path
from typing import Any


def safe_bool_query(value: str, default: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def docker_simulation_identifier_set(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    direct_keys = ("id", "short_id", "name", "service")
    for key in direct_keys:
        clean = str(row.get(key, "") or "").strip().lower()
        if clean:
            values.add(clean)

    route = row.get("route", {}) if isinstance(row.get("route"), dict) else {}
    route_id = str(route.get("id", "") or "").strip().lower()
    if route_id:
        values.add(route_id)

    return values


def find_docker_simulation_row(
    snapshot: dict[str, Any],
    identifier: str,
) -> dict[str, Any] | None:
    lookup = str(identifier or "").strip().lower()
    if not lookup:
        return None
    rows = snapshot.get("simulations", []) if isinstance(snapshot, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if lookup in docker_simulation_identifier_set(row):
            return row
    return None


def project_vector(embedding: list[float] | None) -> list[float]:
    if not isinstance(embedding, list) or not embedding:
        return [0.0, 0.0, 0.0]
    head = [float(value) for value in embedding[:3]]
    while len(head) < 3:
        head.append(0.0)
    max_mag = max(abs(head[0]), abs(head[1]), abs(head[2]), 1.0)
    return [
        round(head[0] / max_mag, 6),
        round(head[1] / max_mag, 6),
        round(head[2] / max_mag, 6),
    ]


def normalize_audio_upload_name(file_name: str, mime: str) -> str:
    source_name = Path(str(file_name or "upload")).name
    source_name = source_name.replace("\x00", "").strip() or "upload"
    base = "".join(
        ch if (ch.isalnum() or ch in {"-", "_"}) else "-"
        for ch in Path(source_name).stem
    ).strip("-")
    if not base:
        base = "upload"

    ext = Path(source_name).suffix.lower()
    if not ext or len(ext) > 12:
        guessed_ext = mimetypes.guess_extension(mime or "") or ""
        ext = guessed_ext.lower() if guessed_ext else ".mp3"

    digest = hashlib.sha1(
        f"{source_name}|{time.time_ns()}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{base[:48]}-{digest}{ext}"
