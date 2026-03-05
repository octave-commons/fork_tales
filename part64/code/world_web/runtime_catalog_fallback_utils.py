"""Runtime catalog fallback and isolated collection helpers."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote


def collect_runtime_catalog_isolated(
    part_root: Path,
    vault_root: Path,
    *,
    runtime_catalog_subprocess_enabled: bool,
    runtime_catalog_subprocess_script: str,
    runtime_catalog_subprocess_timeout_seconds: float,
) -> tuple[dict[str, Any] | None, str]:
    if not runtime_catalog_subprocess_enabled:
        return None, "catalog_subprocess_disabled"

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                runtime_catalog_subprocess_script,
                str(part_root),
                str(vault_root),
            ],
            capture_output=True,
            text=True,
            timeout=runtime_catalog_subprocess_timeout_seconds,
            check=False,
        )
    except Exception as exc:
        return None, f"catalog_subprocess_failed:{exc.__class__.__name__}"

    if proc.returncode != 0:
        return None, f"catalog_subprocess_exit:{proc.returncode}"

    stdout = str(proc.stdout or "").strip()
    if not stdout:
        return None, "catalog_subprocess_empty_output"

    candidates = [stdout, *reversed(stdout.splitlines())]
    for candidate in candidates:
        line = str(candidate).strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload, ""

    return None, "catalog_subprocess_invalid_json"


def fallback_kind_for_path(
    path: Path,
    *,
    audio_suffixes: set[str],
    image_suffixes: set[str],
    video_suffixes: set[str],
) -> str:
    suffix = path.suffix.lower()
    if suffix in audio_suffixes:
        return "audio"
    if suffix in image_suffixes:
        return "image"
    if suffix in video_suffixes:
        return "video"
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type and mime_type.startswith("text/"):
        return "text"
    return "file"


def fallback_rel_path(path: Path, vault_root: Path, part_root: Path) -> str:
    try:
        rel_path = path.resolve().relative_to(vault_root.resolve())
        return str(rel_path).replace("\\", "/")
    except ValueError:
        try:
            rel_path = path.resolve().relative_to(part_root.resolve())
            return str(rel_path).replace("\\", "/")
        except ValueError:
            return path.name


def runtime_catalog_fallback_items(
    part_root: Path,
    vault_root: Path,
    *,
    load_manifest: Callable[[Path], dict[str, Any]],
    fallback_rel_path_fn: Callable[[Path, Path, Path], str],
    fallback_kind_for_path_fn: Callable[[Path], str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    manifest = load_manifest(part_root)
    manifest_entries: list[dict[str, Any]] = []
    for key in ("files", "artifacts"):
        value = manifest.get(key, [])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    manifest_entries.append(item)

    part_label = str(manifest.get("part") or manifest.get("name") or part_root.name)
    items: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    seen_paths: set[str] = set()

    for entry in manifest_entries:
        rel_source = str(entry.get("path", "")).strip()
        if not rel_source:
            continue
        candidate = (part_root / rel_source).resolve()
        if not candidate.exists() or not candidate.is_file():
            continue

        rel_path = fallback_rel_path_fn(candidate, vault_root, part_root)
        if rel_path in seen_paths:
            continue
        seen_paths.add(rel_path)

        kind = fallback_kind_for_path_fn(candidate)
        counts[kind] = int(counts.get(kind, 0)) + 1
        role = str(entry.get("role", "unknown")).strip() or "unknown"
        stat = candidate.stat()
        items.append(
            {
                "part": part_label,
                "name": candidate.name,
                "role": role,
                "display_name": {"en": candidate.name, "ja": candidate.name},
                "display_role": {"en": role, "ja": role},
                "kind": kind,
                "bytes": int(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
                "rel_path": rel_path,
                "url": "/library/" + quote(rel_path),
            }
        )

    items.sort(
        key=lambda row: (
            str(row.get("part", "")),
            str(row.get("kind", "")),
            str(row.get("name", "")),
        ),
        reverse=True,
    )
    return items, counts


def runtime_catalog_fallback(
    part_root: Path,
    vault_root: Path,
    *,
    runtime_catalog_fallback_items_fn: Callable[
        [Path, Path], tuple[list[dict[str, Any]], dict[str, int]]
    ],
    entity_manifest: Any,
    build_named_field_overlays: Callable[[Any], Any],
    projection_default_perspective: str,
    projection_perspective_options: Callable[[], list[dict[str, Any]]],
) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    fallback_items, fallback_counts = runtime_catalog_fallback_items_fn(
        part_root,
        vault_root,
    )
    cover_fields = [
        {
            "id": str(item.get("rel_path", "")),
            "part": str(item.get("part", "")),
            "display_name": item.get("display_name", {}),
            "display_role": item.get("display_role", {}),
            "url": str(item.get("url", "")),
            "seed": hashlib.sha1(
                str(item.get("rel_path", "")).encode("utf-8")
            ).hexdigest(),
        }
        for item in fallback_items
        if str(item.get("role", "")) == "cover_art"
    ]
    inbox_stub = {
        "record": "ημ.inbox.v1",
        "path": str((part_root / ".ημ").resolve()),
        "pending_count": 0,
        "processed_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "rejected_count": 0,
        "deferred_count": 0,
        "is_empty": True,
        "knowledge_entries": 0,
        "registry_entries": 0,
        "last_ingested_at": "",
        "errors": [],
        "sync_status": "deferred",
    }
    file_graph_stub = {
        "record": "ημ.file-graph.v1",
        "generated_at": now_iso,
        "inbox": inbox_stub,
        "nodes": [],
        "field_nodes": [],
        "tag_nodes": [],
        "file_nodes": [],
        "edges": [],
        "stats": {
            "field_count": 0,
            "file_count": 0,
            "edge_count": 0,
            "kind_counts": {},
            "field_counts": {},
            "knowledge_entries": 0,
        },
    }
    crawler_graph_stub = {
        "record": "ημ.crawler-graph.v1",
        "generated_at": now_iso,
        "source": {"endpoint": "", "service": "weaver"},
        "status": {},
        "nodes": [],
        "field_nodes": [],
        "crawler_nodes": [],
        "edges": [],
        "stats": {
            "field_count": 0,
            "crawler_count": 0,
            "edge_count": 0,
            "kind_counts": {},
            "field_counts": {},
            "nodes_total": 0,
            "edges_total": 0,
            "url_nodes_total": 0,
        },
    }
    return {
        "generated_at": now_iso,
        "part_roots": [str(part_root.resolve())],
        "counts": fallback_counts,
        "canonical_terms": [],
        "entity_manifest": entity_manifest,
        "named_fields": build_named_field_overlays(entity_manifest),
        "ui_default_perspective": projection_default_perspective,
        "ui_perspectives": projection_perspective_options(),
        "cover_fields": cover_fields,
        "eta_mu_inbox": inbox_stub,
        "file_graph": file_graph_stub,
        "crawler_graph": crawler_graph_stub,
        "truth_state": {},
        "test_failures": [],
        "test_coverage": {},
        "promptdb": {},
        "world_log": {
            "ok": True,
            "record": "ημ.world-log.v1",
            "generated_at": now_iso,
            "count": 0,
            "limit": 0,
            "pending_inbox": 0,
            "sources": {},
            "kinds": {},
            "relation_count": 0,
            "events": [],
        },
        "items": fallback_items,
        "pi_archive": {
            "record": "ημ.pi-archive.v1",
            "generated_at": "",
            "hash": {},
            "signature": {},
            "portable": {},
            "ledger_count": 0,
            "status": "deferred",
        },
        "runtime_state": "fallback",
    }
