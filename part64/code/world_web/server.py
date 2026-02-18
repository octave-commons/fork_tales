from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen

from .ai import (
    _apply_embedding_provider_options,
    _embedding_provider_options,
    _ollama_embed,
    build_chat_reply,
    build_image_commentary,
    build_presence_say_payload,
    build_voice_lines,
    transcribe_audio_bytes,
    utterances_to_ledger_rows,
)
from .catalog import (
    _read_library_archive_member,
    build_world_payload,
    collect_catalog,
    collect_zip_catalog,
    load_manifest,
    resolve_library_member,
    resolve_library_path,
    sync_eta_mu_inbox,
)
from .chamber import (
    CouncilChamber,
    TaskQueue,
    _load_study_snapshot_events,
    _study_snapshot_log_path,
    build_world_log_payload,
    build_pi_archive_payload,
    build_drift_scan_payload,
    build_push_truth_dry_run_payload,
    build_study_snapshot,
    export_study_snapshot,
    validate_pi_archive_portable,
)
from .constants import (
    AUDIO_SUFFIXES,
    CATALOG_BROADCAST_HEARTBEAT_SECONDS,
    CATALOG_REFRESH_SECONDS,
    COUNCIL_DECISION_LOG_REL,
    ENTITY_MANIFEST,
    IMAGE_SUFFIXES,
    PROJECTION_DEFAULT_PERSPECTIVE,
    SIM_TICK_SECONDS,
    TTS_BASE_URL,
    VIDEO_SUFFIXES,
    WEAVER_AUTOSTART,
    WEAVER_HOST_ENV,
    WEAVER_PORT,
    WS_MAGIC,
)
from .db import (
    _embedding_db_delete,
    _embedding_db_list,
    _embedding_db_query,
    _embedding_db_status,
    _embedding_db_upsert,
    _create_image_comment,
    _get_chroma_collection,
    _list_image_comments,
    _list_presence_accounts,
    _load_life_interaction_builder,
    _load_life_tracker_class,
    _load_mycelial_echo_documents,
    _load_myth_tracker_class,
    _normalize_embedding_vector,
    _upsert_presence_account,
)
from .metrics import _INFLUENCE_TRACKER, _safe_float, _resource_monitor_snapshot
from .paths import _ensure_receipts_log_path
from .projection import (
    attach_ui_projection,
    build_ui_projection,
    normalize_projection_perspective,
    projection_perspective_options,
)
from .simulation import (
    build_mix_stream,
    build_named_field_overlays,
    build_simulation_state,
)


_RUNTIME_CATALOG_CACHE_LOCK = threading.Lock()
_RUNTIME_CATALOG_REFRESH_LOCK = threading.Lock()
_RUNTIME_CATALOG_CACHE: dict[str, Any] = {
    "catalog": None,
    "refreshed_monotonic": 0.0,
    "last_error": "",
    "inbox_sync_monotonic": 0.0,
    "inbox_sync_snapshot": None,
    "inbox_sync_error": "",
}
_RUNTIME_CATALOG_CACHE_SECONDS = max(
    CATALOG_REFRESH_SECONDS,
    float(os.getenv("RUNTIME_CATALOG_CACHE_SECONDS", "10.0") or "10.0"),
)
_RUNTIME_ETA_MU_SYNC_SECONDS = max(
    0.5,
    float(os.getenv("RUNTIME_ETA_MU_SYNC_SECONDS", "6.0") or "6.0"),
)
_RUNTIME_INBOX_SYNC_LOCK = threading.Lock()
_RUNTIME_CATALOG_SUBPROCESS_TIMEOUT_SECONDS = max(
    8.0,
    float(os.getenv("RUNTIME_CATALOG_SUBPROCESS_TIMEOUT_SECONDS", "75.0") or "75.0"),
)
_RUNTIME_CATALOG_SUBPROCESS_ENABLED = str(
    os.getenv("RUNTIME_CATALOG_SUBPROCESS_ENABLED", "1") or "1"
).strip().lower() not in {"0", "false", "no", "off"}
_RUNTIME_CATALOG_SUBPROCESS_SCRIPT = (
    "import json,sys;"
    "from pathlib import Path;"
    "import code.world_web as ww;"
    "payload=ww.collect_catalog("
    "Path(sys.argv[1]),"
    "Path(sys.argv[2]),"
    "sync_inbox=False,"
    "include_pi_archive=False,"
    "include_world_log=False"
    ");"
    "sys.stdout.write(json.dumps(payload,ensure_ascii=False))"
)


def _effective_request_embed_model(model: str | None) -> str | None:
    force_nomic = str(
        os.getenv("OLLAMA_EMBED_FORCE_NOMIC", "0") or "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if force_nomic:
        return "nomic-embed-text"
    normalized = str(model or "").strip()
    return normalized or None


def _resolve_runtime_library_path(
    vault_root: Path,
    part_root: Path,
    request_path: str,
) -> Path | None:
    lib_path = resolve_library_path(vault_root, request_path)
    if lib_path is not None:
        return lib_path

    parsed = urlparse(request_path)
    raw_path = unquote(parsed.path)
    if not raw_path.startswith("/library/"):
        return None
    relative = raw_path.removeprefix("/library/")
    if not relative:
        return None

    candidate = (part_root.resolve() / relative).resolve()
    part_resolved = part_root.resolve()
    if candidate == part_resolved or part_resolved in candidate.parents:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _collect_runtime_catalog_isolated(
    part_root: Path,
    vault_root: Path,
) -> tuple[dict[str, Any] | None, str]:
    if not _RUNTIME_CATALOG_SUBPROCESS_ENABLED:
        return None, "catalog_subprocess_disabled"

    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _RUNTIME_CATALOG_SUBPROCESS_SCRIPT,
                str(part_root),
                str(vault_root),
            ],
            capture_output=True,
            text=True,
            timeout=_RUNTIME_CATALOG_SUBPROCESS_TIMEOUT_SECONDS,
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


def _fallback_kind_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type and mime_type.startswith("text/"):
        return "text"
    return "file"


def _fallback_rel_path(path: Path, vault_root: Path, part_root: Path) -> str:
    try:
        rel_path = path.resolve().relative_to(vault_root.resolve())
        return str(rel_path).replace("\\", "/")
    except ValueError:
        try:
            rel_path = path.resolve().relative_to(part_root.resolve())
            return str(rel_path).replace("\\", "/")
        except ValueError:
            return path.name


def _runtime_catalog_fallback_items(
    part_root: Path,
    vault_root: Path,
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

        rel_path = _fallback_rel_path(candidate, vault_root, part_root)
        if rel_path in seen_paths:
            continue
        seen_paths.add(rel_path)

        kind = _fallback_kind_for_path(candidate)
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


def _runtime_catalog_fallback(part_root: Path, vault_root: Path) -> dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    fallback_items, fallback_counts = _runtime_catalog_fallback_items(
        part_root, vault_root
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
        "entity_manifest": ENTITY_MANIFEST,
        "named_fields": build_named_field_overlays(ENTITY_MANIFEST),
        "ui_default_perspective": PROJECTION_DEFAULT_PERSPECTIVE,
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


def _weaver_probe_host(bind_host: str) -> str:
    host = bind_host.strip().lower()
    if not host or host in {"0.0.0.0", "::", "localhost"}:
        return "127.0.0.1"
    return bind_host


def _weaver_health_check(host: str, port: int, timeout_s: float = 0.8) -> bool:
    target = f"http://{host}:{port}/healthz"
    req = Request(target, method="GET")
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return int(getattr(resp, "status", 0)) == 200
    except Exception:
        return False


def _ensure_weaver_service(part_root: Path, world_host: str) -> None:
    del part_root
    if not WEAVER_AUTOSTART:
        return
    probe_host = _weaver_probe_host(WEAVER_HOST_ENV or world_host)
    if _weaver_health_check(probe_host, WEAVER_PORT):
        return

    script_path = (
        Path(__file__).resolve().parent.parent / "web_graph_weaver.js"
    ).resolve()
    if not script_path.exists() or not script_path.is_file():
        return
    node_binary = shutil.which("node")
    if not node_binary:
        return

    env = os.environ.copy()
    env.setdefault("WEAVER_HOST", WEAVER_HOST_ENV)
    env.setdefault("WEAVER_PORT", str(WEAVER_PORT))
    try:
        subprocess.Popen(
            [node_binary, str(script_path)],
            cwd=str(script_path.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        return


def _parse_multipart_form(raw_body: bytes, content_type: str) -> dict[str, Any] | None:
    import re

    match = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type, re.I)
    if match is None:
        return None
    boundary_token = (match.group(1) or match.group(2) or "").strip()
    if not boundary_token:
        return None

    delimiter = b"--" + boundary_token.encode("utf-8", errors="ignore")
    data: dict[str, Any] = {}
    for part in raw_body.split(delimiter):
        chunk = part.strip()
        if not chunk or chunk == b"--":
            continue
        if chunk.endswith(b"--"):
            chunk = chunk[:-2].strip()
        head, sep, body = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        if body.endswith(b"\r\n"):
            body = body[:-2]

        disposition = ""
        part_content_type = ""
        for line in head.decode("utf-8", errors="ignore").split("\r\n"):
            low = line.lower()
            if low.startswith("content-disposition:"):
                disposition = line.split(":", 1)[1].strip()
            elif low.startswith("content-type:"):
                part_content_type = line.split(":", 1)[1].strip()

        name_match = re.search(r'name="([^"]+)"', disposition)
        if name_match is None:
            continue
        field_name = name_match.group(1)
        file_match = re.search(r'filename="([^"]*)"', disposition)
        if file_match is not None:
            data[field_name] = {
                "filename": file_match.group(1),
                "content_type": part_content_type,
                "value": body,
            }
        else:
            data[field_name] = body.decode("utf-8", errors="ignore")
    return data


def resolve_artifact_path(part_root: Path, request_path: str) -> Path | None:
    parsed = urlparse(request_path)
    raw_path = unquote(parsed.path)
    if not raw_path.startswith("/artifacts/"):
        return None

    relative = raw_path.removeprefix("/")
    if not relative:
        return None

    candidate = (part_root / relative).resolve()
    artifacts_root = (part_root / "artifacts").resolve()
    if artifacts_root == candidate or artifacts_root in candidate.parents:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def websocket_accept_value(client_key: str) -> str:
    accept_seed = client_key + WS_MAGIC
    digest = hashlib.sha1(accept_seed.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("utf-8")


def websocket_frame_text(message: str) -> bytes:
    payload = message.encode("utf-8")
    length = len(payload)
    header = bytearray([0x81])
    if length <= 125:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + payload


def render_index(payload: dict[str, Any], catalog: dict[str, Any]) -> str:
    del payload, catalog
    return ""


def _safe_bool_query(value: str, default: bool = False) -> bool:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _project_vector(embedding: list[float] | None) -> list[float]:
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


def _normalize_audio_upload_name(file_name: str, mime: str) -> str:
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


class WorldHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    part_root: Path = Path(".")
    vault_root: Path = Path("..")
    host_label: str = "127.0.0.1:8787"
    task_queue: TaskQueue
    council_chamber: CouncilChamber
    myth_tracker: Any
    life_tracker: Any
    life_interaction_builder: Any

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: int = HTTPStatus.OK,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self._set_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if isinstance(extra_headers, dict):
            for key, value in extra_headers.items():
                self.send_header(str(key), str(value))
        self.end_headers()
        if body:
            try:
                self.wfile.write(body)
            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
                OSError,
            ):
                pass

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _send_ws_event(self, payload: dict[str, Any]) -> None:
        frame = websocket_frame_text(json.dumps(payload, ensure_ascii=False))
        self.connection.sendall(frame)

    def _read_json_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return None
        raw = self.rfile.read(length)
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            return None
        return decoded if isinstance(decoded, dict) else None

    def _read_raw_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _collect_catalog_fast(self) -> dict[str, Any]:
        return collect_catalog(
            self.part_root,
            self.vault_root,
            sync_inbox=False,
            include_pi_archive=False,
            include_world_log=False,
        )

    def _schedule_runtime_inbox_sync(self) -> None:
        if not _RUNTIME_INBOX_SYNC_LOCK.acquire(blocking=False):
            return

        def _sync() -> None:
            try:
                snapshot = sync_eta_mu_inbox(self.vault_root)
                now_monotonic = time.monotonic()
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["inbox_sync_monotonic"] = now_monotonic
                    _RUNTIME_CATALOG_CACHE["inbox_sync_snapshot"] = dict(snapshot)
                    _RUNTIME_CATALOG_CACHE["inbox_sync_error"] = ""
                    cached_catalog = _RUNTIME_CATALOG_CACHE.get("catalog")
                    if isinstance(cached_catalog, dict):
                        next_catalog = dict(cached_catalog)
                        next_catalog["eta_mu_inbox"] = dict(snapshot)
                        _RUNTIME_CATALOG_CACHE["catalog"] = next_catalog
            except Exception as sync_exc:
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["inbox_sync_error"] = (
                        f"inbox_sync_failed:{sync_exc.__class__.__name__}"
                    )
            finally:
                _RUNTIME_INBOX_SYNC_LOCK.release()

        threading.Thread(target=_sync, daemon=True).start()

    def _schedule_runtime_catalog_refresh(self) -> None:
        if not _RUNTIME_CATALOG_REFRESH_LOCK.acquire(blocking=False):
            return

        def _refresh() -> None:
            try:
                now_monotonic = time.monotonic()
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    last_sync = float(
                        _RUNTIME_CATALOG_CACHE.get("inbox_sync_monotonic", 0.0)
                    )
                    previous_sync_snapshot = _RUNTIME_CATALOG_CACHE.get(
                        "inbox_sync_snapshot"
                    )
                should_sync = (
                    now_monotonic - last_sync >= _RUNTIME_ETA_MU_SYNC_SECONDS
                    or previous_sync_snapshot is None
                )

                fresh_catalog = self._collect_catalog_fast()
                if isinstance(previous_sync_snapshot, dict):
                    fresh_catalog["eta_mu_inbox"] = dict(previous_sync_snapshot)
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["catalog"] = fresh_catalog
                    _RUNTIME_CATALOG_CACHE["refreshed_monotonic"] = time.monotonic()
                    _RUNTIME_CATALOG_CACHE["last_error"] = ""

                if should_sync:
                    self._schedule_runtime_inbox_sync()
            except Exception as exc:
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["last_error"] = (
                        f"catalog_refresh_failed:{exc.__class__.__name__}"
                    )
            finally:
                _RUNTIME_CATALOG_REFRESH_LOCK.release()

        threading.Thread(target=_refresh, daemon=True).start()

    def _runtime_catalog_base(self) -> dict[str, Any]:
        now_monotonic = time.monotonic()
        with _RUNTIME_CATALOG_CACHE_LOCK:
            cached_catalog = _RUNTIME_CATALOG_CACHE.get("catalog")
            refreshed_monotonic = float(
                _RUNTIME_CATALOG_CACHE.get("refreshed_monotonic", 0.0)
            )
            inbox_sync_snapshot = _RUNTIME_CATALOG_CACHE.get("inbox_sync_snapshot")

        if not isinstance(cached_catalog, dict):
            fresh_catalog, isolated_error = _collect_runtime_catalog_isolated(
                self.part_root,
                self.vault_root,
            )
            try:
                if fresh_catalog is None:
                    fresh_catalog = self._collect_catalog_fast()
                cache_error = isolated_error
                if cache_error == "catalog_subprocess_disabled":
                    cache_error = ""
                if isinstance(inbox_sync_snapshot, dict):
                    fresh_catalog["eta_mu_inbox"] = dict(inbox_sync_snapshot)
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["catalog"] = fresh_catalog
                    _RUNTIME_CATALOG_CACHE["refreshed_monotonic"] = time.monotonic()
                    _RUNTIME_CATALOG_CACHE["last_error"] = cache_error
                self._schedule_runtime_inbox_sync()
                return dict(fresh_catalog)
            except Exception as exc:
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["last_error"] = (
                        f"catalog_inline_failed:{exc.__class__.__name__}"
                    )

        cache_age = now_monotonic - refreshed_monotonic
        if cache_age >= _RUNTIME_CATALOG_CACHE_SECONDS:
            self._schedule_runtime_catalog_refresh()

        if isinstance(cached_catalog, dict):
            return dict(cached_catalog)
        fallback_catalog = _runtime_catalog_fallback(self.part_root, self.vault_root)
        if isinstance(inbox_sync_snapshot, dict):
            fallback_catalog["eta_mu_inbox"] = dict(inbox_sync_snapshot)
        return fallback_catalog

    def _runtime_catalog(
        self,
        *,
        perspective: str = PROJECTION_DEFAULT_PERSPECTIVE,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        catalog = self._runtime_catalog_base()
        queue_snapshot = self.task_queue.snapshot(include_pending=False)
        council_snapshot = self.council_chamber.snapshot(include_decisions=False)
        catalog["task_queue"] = queue_snapshot
        catalog["council"] = council_snapshot

        resource_snapshot = _resource_monitor_snapshot(part_root=self.part_root)
        _INFLUENCE_TRACKER.record_resource_heartbeat(
            resource_snapshot,
            source="runtime.catalog",
        )
        influence_snapshot = _INFLUENCE_TRACKER.snapshot(
            queue_snapshot=queue_snapshot,
            part_root=self.part_root,
        )
        catalog["presence_runtime"] = influence_snapshot

        attach_ui_projection(
            catalog,
            perspective=perspective,
            queue_snapshot=queue_snapshot,
            influence_snapshot=influence_snapshot,
        )
        return (
            catalog,
            queue_snapshot,
            council_snapshot,
            influence_snapshot,
            resource_snapshot,
        )

    def _runtime_simulation(
        self,
        catalog: dict[str, Any],
        queue_snapshot: dict[str, Any],
        influence_snapshot: dict[str, Any],
        *,
        perspective: str = PROJECTION_DEFAULT_PERSPECTIVE,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            myth_summary = self.myth_tracker.snapshot(catalog)
        except Exception:
            myth_summary = {}
        try:
            world_summary = self.life_tracker.snapshot(
                catalog,
                myth_summary,
                ENTITY_MANIFEST,
            )
        except Exception:
            world_summary = {}

        simulation = build_simulation_state(
            catalog,
            myth_summary,
            world_summary,
            influence_snapshot=influence_snapshot,
            queue_snapshot=queue_snapshot,
        )
        projection = build_ui_projection(
            catalog,
            simulation,
            perspective=perspective,
            queue_snapshot=queue_snapshot,
            influence_snapshot=influence_snapshot,
        )
        simulation["projection"] = projection
        simulation["perspective"] = perspective
        return simulation, projection

    def _handle_websocket(self, *, perspective: str) -> None:
        ws_key = str(self.headers.get("Sec-WebSocket-Key", "")).strip()
        if not ws_key:
            self._send_bytes(
                b"missing websocket key",
                "text/plain; charset=utf-8",
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", websocket_accept_value(ws_key))
        self.end_headers()

        perspective_key = normalize_projection_perspective(perspective)
        self.connection.settimeout(1.0)
        catalog: dict[str, Any] = {}
        queue_snapshot: dict[str, Any] = {}
        influence_snapshot: dict[str, Any] = {}

        try:
            catalog, queue_snapshot, _, influence_snapshot, _ = self._runtime_catalog(
                perspective=perspective_key,
            )
            _, mix_meta = build_mix_stream(catalog, self.vault_root)
            self._send_ws_event(
                {
                    "type": "catalog",
                    "catalog": catalog,
                    "mix": mix_meta,
                }
            )

            simulation, projection = self._runtime_simulation(
                catalog,
                queue_snapshot,
                influence_snapshot,
                perspective=perspective_key,
            )
            self._send_ws_event(
                {
                    "type": "simulation",
                    "simulation": simulation,
                    "projection": projection,
                }
            )
        except Exception:
            try:
                self._send_ws_event(
                    {
                        "type": "error",
                        "error": "initial websocket payload failed",
                    }
                )
            except Exception:
                return
            return

        last_catalog_refresh = time.monotonic()
        last_catalog_broadcast = last_catalog_refresh
        last_sim_tick = last_catalog_refresh

        while True:
            now_monotonic = time.monotonic()
            try:
                if now_monotonic - last_catalog_refresh >= CATALOG_REFRESH_SECONDS:
                    catalog, queue_snapshot, _, influence_snapshot, _ = (
                        self._runtime_catalog(
                            perspective=perspective_key,
                        )
                    )
                    last_catalog_refresh = now_monotonic

                if (
                    now_monotonic - last_catalog_broadcast
                    >= CATALOG_BROADCAST_HEARTBEAT_SECONDS
                ):
                    _, mix_meta = build_mix_stream(catalog, self.vault_root)
                    self._send_ws_event(
                        {
                            "type": "catalog",
                            "catalog": catalog,
                            "mix": mix_meta,
                        }
                    )
                    last_catalog_broadcast = now_monotonic

                if now_monotonic - last_sim_tick >= SIM_TICK_SECONDS:
                    simulation, projection = self._runtime_simulation(
                        catalog,
                        queue_snapshot,
                        influence_snapshot,
                        perspective=perspective_key,
                    )
                    self._send_ws_event(
                        {
                            "type": "simulation",
                            "simulation": simulation,
                            "projection": projection,
                        }
                    )
                    last_sim_tick = now_monotonic

                try:
                    chunk = self.connection.recv(2)
                    if not chunk:
                        break
                except socket.timeout:
                    continue
            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
                OSError,
            ):
                break

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._set_cors_headers()
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/ws":
            perspective = normalize_projection_perspective(
                str(
                    params.get(
                        "perspective",
                        [PROJECTION_DEFAULT_PERSPECTIVE],
                    )[0]
                    or PROJECTION_DEFAULT_PERSPECTIVE
                )
            )
            self._handle_websocket(perspective=perspective)
            return

        if parsed.path == "/api/voice-lines":
            mode = str(params.get("mode", ["canonical"])[0] or "canonical")
            payload_voice = build_voice_lines(
                "ollama" if mode.strip().lower() == "ollama" else "canonical"
            )
            self._send_json(payload_voice)
            return

        if parsed.path == "/healthz":
            payload = build_world_payload(self.part_root)
            catalog = self._runtime_catalog_base()
            file_count = len(catalog.get("items", []))
            entropy = int(time.time() * 1000) % 100
            self._send_json(
                {
                    "ok": True,
                    "status": "alive",
                    "organism": {
                        "spore_count": file_count,
                        "mycelial_density": 1.0,
                        "pulse_rate": "78bpm",
                        "substrate_entropy": f"{entropy}%",
                        "growth_phase": "fruiting",
                    },
                    "part": payload.get("part"),
                    "items": file_count,
                }
            )
            return

        if parsed.path == "/api/mix":
            catalog, _, _, _, _ = self._runtime_catalog(
                perspective=PROJECTION_DEFAULT_PERSPECTIVE,
            )
            _, mix_meta = build_mix_stream(catalog, self.vault_root)
            self._send_json(mix_meta)
            return

        if parsed.path == "/api/tts":
            text = str(params.get("text", [""])[0] or "").strip()
            speed = str(params.get("speed", ["1.0"])[0] or "1.0").strip()

            if not text:
                self._send_json(
                    {"ok": False, "error": "empty text"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            tts_url = f"{TTS_BASE_URL}/tts?text={quote(text)}&speed={speed}"
            try:
                with urlopen(Request(tts_url, method="GET"), timeout=30) as resp:
                    self._send_bytes(resp.read(), "audio/wav")
                return
            except Exception as sidecar_error:
                fallback_error = ""
                speed_ratio = max(0.6, min(1.6, _safe_float(speed, 1.0)))
                words_per_minute = int(round(max(90, min(320, 170 * speed_ratio))))
                safe_text = text[:600]

                with tempfile.NamedTemporaryFile(
                    prefix="eta_mu_tts_",
                    suffix=".wav",
                    delete=False,
                ) as tmp_file:
                    fallback_path = Path(tmp_file.name)

                try:
                    command_candidates = (
                        [
                            "espeak-ng",
                            "-s",
                            str(words_per_minute),
                            "-w",
                            str(fallback_path),
                            safe_text,
                        ],
                        [
                            "espeak",
                            "-s",
                            str(words_per_minute),
                            "-w",
                            str(fallback_path),
                            safe_text,
                        ],
                    )
                    rendered = False
                    for command in command_candidates:
                        try:
                            result = subprocess.run(
                                command,
                                check=False,
                                capture_output=True,
                                text=True,
                                timeout=18,
                            )
                        except FileNotFoundError:
                            continue

                        if result.returncode != 0:
                            fallback_error = (
                                result.stderr or result.stdout or ""
                            ).strip()
                            continue

                        if (
                            not fallback_path.exists()
                            or fallback_path.stat().st_size <= 44
                        ):
                            fallback_error = "fallback wav missing or empty"
                            continue

                        self._send_bytes(fallback_path.read_bytes(), "audio/wav")
                        rendered = True
                        break

                    if rendered:
                        return

                    if not fallback_error:
                        fallback_error = "espeak fallback unavailable"
                except Exception as fallback_exc:
                    fallback_error = str(fallback_exc)
                finally:
                    try:
                        fallback_path.unlink(missing_ok=True)
                    except OSError:
                        pass

                self._send_json(
                    {
                        "ok": False,
                        "error": str(sidecar_error),
                        "fallback_error": fallback_error,
                    },
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if parsed.path == "/api/embeddings/db/status":
            self._send_json(_embedding_db_status(self.vault_root))
            return

        if parsed.path == "/api/embeddings/provider/options":
            self._send_json(_embedding_provider_options())
            return

        if parsed.path == "/api/embeddings/db/list":
            limit = max(
                1,
                min(
                    500,
                    int(_safe_float(str(params.get("limit", ["50"])[0] or "50"), 50.0)),
                ),
            )
            include_vectors = _safe_bool_query(
                str(params.get("include_vectors", ["false"])[0] or "false"),
                default=False,
            )
            self._send_json(
                _embedding_db_list(
                    self.vault_root,
                    limit=limit,
                    include_vectors=include_vectors,
                )
            )
            return

        if parsed.path == "/api/presence/accounts":
            limit = max(
                1,
                min(
                    500,
                    int(_safe_float(str(params.get("limit", ["64"])[0] or "64"), 64.0)),
                ),
            )
            self._send_json(_list_presence_accounts(self.vault_root, limit=limit))
            return

        if parsed.path == "/api/image/comments":
            image_ref = str(params.get("image_ref", [""])[0] or "").strip()
            limit = max(
                1,
                min(
                    1000,
                    int(
                        _safe_float(
                            str(params.get("limit", ["120"])[0] or "120"),
                            120.0,
                        )
                    ),
                ),
            )
            self._send_json(
                _list_image_comments(
                    self.vault_root,
                    image_ref=image_ref,
                    limit=limit,
                )
            )
            return

        if parsed.path == "/api/world/events":
            limit = max(
                12,
                min(
                    800,
                    int(
                        _safe_float(
                            str(params.get("limit", ["180"])[0] or "180"),
                            180.0,
                        )
                    ),
                ),
            )
            self._send_json(
                build_world_log_payload(
                    self.part_root,
                    self.vault_root,
                    limit=limit,
                )
            )
            return

        if parsed.path == "/api/catalog":
            perspective = normalize_projection_perspective(
                str(
                    params.get(
                        "perspective",
                        [PROJECTION_DEFAULT_PERSPECTIVE],
                    )[0]
                    or PROJECTION_DEFAULT_PERSPECTIVE
                )
            )
            catalog, _, _, _, _ = self._runtime_catalog(perspective=perspective)
            self._send_json(catalog)
            return

        if parsed.path == "/api/zips":
            member_limit = int(
                _safe_float(
                    str(params.get("member_limit", ["220"])[0] or "220"),
                    220.0,
                )
            )
            self._send_json(
                collect_zip_catalog(
                    self.part_root,
                    self.vault_root,
                    member_limit=member_limit,
                )
            )
            return

        if parsed.path == "/api/pi/archive":
            catalog = self._collect_catalog_fast()
            queue_snapshot = self.task_queue.snapshot(include_pending=False)
            archive = build_pi_archive_payload(
                self.part_root,
                self.vault_root,
                catalog=catalog,
                queue_snapshot=queue_snapshot,
            )
            self._send_json({"ok": True, "archive": archive})
            return

        if parsed.path == "/api/ui/projection":
            perspective = normalize_projection_perspective(
                str(
                    params.get(
                        "perspective",
                        [PROJECTION_DEFAULT_PERSPECTIVE],
                    )[0]
                    or PROJECTION_DEFAULT_PERSPECTIVE
                )
            )
            catalog, queue_snapshot, _, influence_snapshot, _ = self._runtime_catalog(
                perspective=perspective,
            )
            simulation, projection = self._runtime_simulation(
                catalog,
                queue_snapshot,
                influence_snapshot,
                perspective=perspective,
            )
            self._send_json(
                {
                    "ok": True,
                    "projection": projection,
                    "simulation": simulation,
                    "default_perspective": PROJECTION_DEFAULT_PERSPECTIVE,
                    "perspectives": projection_perspective_options(),
                }
            )
            return

        if parsed.path == "/api/task/queue":
            self._send_json(
                {
                    "ok": True,
                    "queue": self.task_queue.snapshot(include_pending=True),
                }
            )
            return

        if parsed.path == "/api/council":
            limit = max(
                1,
                min(
                    128,
                    int(_safe_float(str(params.get("limit", ["16"])[0] or "16"), 16.0)),
                ),
            )
            self._send_json(
                {
                    "ok": True,
                    "council": self.council_chamber.snapshot(
                        include_decisions=True,
                        limit=limit,
                    ),
                }
            )
            return

        if parsed.path == "/api/study":
            limit = max(
                1,
                min(
                    128,
                    int(_safe_float(str(params.get("limit", ["16"])[0] or "16"), 16.0)),
                ),
            )
            include_truth_state = _safe_bool_query(
                str(params.get("include_truth", ["false"])[0] or "false"),
                default=False,
            )
            queue_snapshot = self.task_queue.snapshot(include_pending=True)
            council_snapshot = self.council_chamber.snapshot(
                include_decisions=True,
                limit=limit,
            )
            drift_payload = build_drift_scan_payload(self.part_root, self.vault_root)
            resource_snapshot = _resource_monitor_snapshot(part_root=self.part_root)
            _INFLUENCE_TRACKER.record_resource_heartbeat(
                resource_snapshot,
                source="api.study",
            )

            truth_gate_blocked: bool | None = None
            if include_truth_state:
                try:
                    truth_state = self._collect_catalog_fast().get("truth_state", {})
                    gate = (
                        truth_state.get("gate", {})
                        if isinstance(truth_state, dict)
                        else {}
                    )
                    if isinstance(gate, dict):
                        truth_gate_blocked = bool(gate.get("blocked", False))
                except Exception:
                    truth_gate_blocked = None

            self._send_json(
                build_study_snapshot(
                    self.part_root,
                    self.vault_root,
                    queue_snapshot=queue_snapshot,
                    council_snapshot=council_snapshot,
                    drift_payload=drift_payload,
                    truth_gate_blocked=truth_gate_blocked,
                    resource_snapshot=resource_snapshot,
                )
            )
            return

        if parsed.path == "/api/study/history":
            limit = max(
                1,
                min(
                    256,
                    int(_safe_float(str(params.get("limit", ["16"])[0] or "16"), 16.0)),
                ),
            )
            events = _load_study_snapshot_events(self.vault_root, limit=limit)
            self._send_json(
                {
                    "ok": True,
                    "record": "ημ.study-history.v1",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "path": str(_study_snapshot_log_path(self.vault_root)),
                    "count": len(events),
                    "events": events,
                }
            )
            return

        if parsed.path == "/api/resource/heartbeat":
            heartbeat = _resource_monitor_snapshot(part_root=self.part_root)
            _INFLUENCE_TRACKER.record_resource_heartbeat(
                heartbeat,
                source="api.resource.heartbeat",
            )
            queue_snapshot = self.task_queue.snapshot(include_pending=False)
            runtime_snapshot = _INFLUENCE_TRACKER.snapshot(
                queue_snapshot=queue_snapshot,
                part_root=self.part_root,
            )
            self._send_json(
                {
                    "ok": True,
                    "record": "ημ.resource-heartbeat.response.v1",
                    "heartbeat": heartbeat,
                    "runtime": runtime_snapshot,
                }
            )
            return

        if parsed.path == "/api/named-fields":
            self._send_json(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "mode": "gradients",
                    "named_fields": build_named_field_overlays(ENTITY_MANIFEST),
                }
            )
            return

        if parsed.path == "/api/memories":
            docs = _load_mycelial_echo_documents(limit=10)
            now_iso = datetime.now(timezone.utc).isoformat()
            memories = [
                {
                    "id": f"mem:{idx}",
                    "text": str(doc),
                    "metadata": {"timestamp": now_iso},
                }
                for idx, doc in enumerate(docs)
                if str(doc).strip()
            ]
            self._send_json({"ok": True, "memories": memories})
            return

        if parsed.path == "/api/myth":
            catalog = self._collect_catalog_fast()
            try:
                myth_summary = self.myth_tracker.snapshot(catalog)
            except Exception:
                myth_summary = {}
            self._send_json(myth_summary)
            return

        if parsed.path == "/api/world":
            catalog = self._collect_catalog_fast()
            try:
                myth_summary = self.myth_tracker.snapshot(catalog)
            except Exception:
                myth_summary = {}
            try:
                world_summary = self.life_tracker.snapshot(
                    catalog,
                    myth_summary,
                    ENTITY_MANIFEST,
                )
            except Exception:
                world_summary = {"generated_at": datetime.now(timezone.utc).isoformat()}
            self._send_json(world_summary)
            return

        if parsed.path == "/api/simulation":
            perspective = normalize_projection_perspective(
                str(
                    params.get(
                        "perspective",
                        [PROJECTION_DEFAULT_PERSPECTIVE],
                    )[0]
                    or PROJECTION_DEFAULT_PERSPECTIVE
                )
            )
            catalog, queue_snapshot, _, influence_snapshot, _ = self._runtime_catalog(
                perspective=perspective,
            )
            simulation, projection = self._runtime_simulation(
                catalog,
                queue_snapshot,
                influence_snapshot,
                perspective=perspective,
            )
            simulation["projection"] = projection
            self._send_json(simulation)
            return

        if parsed.path == "/stream/mix.wav":
            catalog, _, _, _, _ = self._runtime_catalog(
                perspective=PROJECTION_DEFAULT_PERSPECTIVE,
            )
            wav_bytes, _ = build_mix_stream(catalog, self.vault_root)
            if wav_bytes:
                self._send_bytes(wav_bytes, "audio/wav")
            else:
                self._send_bytes(
                    b"no wav sources available for mix",
                    "text/plain; charset=utf-8",
                    status=HTTPStatus.NOT_FOUND,
                )
            return

        if parsed.path.startswith("/library/"):
            member = resolve_library_member(self.path)
            lib_path = _resolve_runtime_library_path(
                self.vault_root,
                self.part_root,
                self.path,
            )
            if lib_path is None:
                self._send_json(
                    {"ok": False, "error": "library_not_found"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return

            if member:
                payload = _read_library_archive_member(lib_path, member)
                if payload is None:
                    self._send_json(
                        {"ok": False, "error": "library_member_not_found"},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                payload_bytes, payload_type = payload
                self._send_bytes(payload_bytes, payload_type)
                return

            mime_type = (
                mimetypes.guess_type(lib_path.name)[0] or "application/octet-stream"
            )
            self._send_bytes(lib_path.read_bytes(), mime_type)
            return

        if parsed.path.startswith("/artifacts/"):
            artifact = resolve_artifact_path(self.part_root, self.path)
            if artifact is None:
                self._send_json(
                    {"ok": False, "error": "artifact_not_found"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            mime_type = (
                mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
            )
            self._send_bytes(artifact.read_bytes(), mime_type)
            return

        if parsed.path == "/":
            index_path = (self.part_root / "frontend" / "dist" / "index.html").resolve()
            if index_path.exists() and index_path.is_file():
                self._send_bytes(index_path.read_bytes(), "text/html; charset=utf-8")
                return

            world_payload = build_world_payload(self.part_root)
            catalog, _, _, _, _ = self._runtime_catalog(
                perspective=PROJECTION_DEFAULT_PERSPECTIVE,
            )
            html_doc = render_index(world_payload, catalog)
            body = html_doc.encode("utf-8") if html_doc else b""
            self._send_bytes(body, "text/html; charset=utf-8")
            return

        # Optional static frontend assets when served from part64 runtime directly.
        dist_root = (self.part_root / "frontend" / "dist").resolve()
        if parsed.path != "/":
            requested = parsed.path.lstrip("/")
            candidate = (dist_root / requested).resolve()
            if (
                dist_root in candidate.parents
                and candidate.exists()
                and candidate.is_file()
            ):
                mime_type = (
                    mimetypes.guess_type(candidate.name)[0]
                    or "application/octet-stream"
                )
                self._send_bytes(candidate.read_bytes(), mime_type)
                return

        self._send_json(
            {"ok": False, "error": "not found"},
            status=HTTPStatus.NOT_FOUND,
        )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/api/eta-mu-ledger":
            req = self._read_json_body() or {}
            rows_input: list[str] = []
            utterances_raw = req.get("utterances")
            if isinstance(utterances_raw, list):
                rows_input = [str(item) for item in utterances_raw]
            else:
                text_raw = req.get("text")
                if isinstance(text_raw, str):
                    rows_input = text_raw.splitlines()

            rows = utterances_to_ledger_rows(rows_input)
            jsonl = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
            self._send_json(
                {
                    "ok": True,
                    "rows": rows,
                    "jsonl": f"{jsonl}\n" if jsonl else "",
                }
            )
            return

        if parsed.path == "/api/eta-mu/sync":
            req = self._read_json_body() or {}
            wait = _safe_bool_query(
                str(req.get("wait", "false") or "false"), default=False
            )
            force = _safe_bool_query(
                str(req.get("force", "true") or "true"), default=True
            )

            if force:
                with _RUNTIME_CATALOG_CACHE_LOCK:
                    _RUNTIME_CATALOG_CACHE["inbox_sync_monotonic"] = 0.0

            if wait:
                try:
                    snapshot = sync_eta_mu_inbox(self.vault_root)
                    with _RUNTIME_CATALOG_CACHE_LOCK:
                        _RUNTIME_CATALOG_CACHE["inbox_sync_monotonic"] = (
                            time.monotonic()
                        )
                        _RUNTIME_CATALOG_CACHE["inbox_sync_snapshot"] = dict(snapshot)
                        _RUNTIME_CATALOG_CACHE["inbox_sync_error"] = ""
                        _RUNTIME_CATALOG_CACHE["refreshed_monotonic"] = 0.0
                    self._schedule_runtime_catalog_refresh()
                    self._send_json(
                        {
                            "ok": True,
                            "record": "ημ.inbox.sync.v1",
                            "status": "completed",
                            "snapshot": snapshot,
                        }
                    )
                except Exception as exc:
                    with _RUNTIME_CATALOG_CACHE_LOCK:
                        _RUNTIME_CATALOG_CACHE["inbox_sync_error"] = (
                            f"inbox_sync_failed:{exc.__class__.__name__}"
                        )
                    self._send_json(
                        {
                            "ok": False,
                            "record": "ημ.inbox.sync.v1",
                            "status": "failed",
                            "error": f"{exc.__class__.__name__}: {exc}",
                        },
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return

            self._schedule_runtime_catalog_refresh()
            with _RUNTIME_CATALOG_CACHE_LOCK:
                snapshot = _RUNTIME_CATALOG_CACHE.get("inbox_sync_snapshot")
                sync_error = str(_RUNTIME_CATALOG_CACHE.get("inbox_sync_error", ""))
            self._send_json(
                {
                    "ok": True,
                    "record": "ημ.inbox.sync.v1",
                    "status": "scheduled",
                    "snapshot": dict(snapshot) if isinstance(snapshot, dict) else None,
                    "error": sync_error,
                }
            )
            return

        if parsed.path == "/api/presence/accounts/upsert":
            req = self._read_json_body() or {}
            presence_id = str(req.get("presence_id", "") or "").strip()
            display_name = str(req.get("display_name", "") or "").strip()
            handle = str(req.get("handle", "") or "").strip()
            avatar = str(req.get("avatar", "") or "").strip()
            bio = str(req.get("bio", "") or "").strip()
            tags_raw = req.get("tags", [])
            tags = (
                [str(item).strip() for item in tags_raw if str(item).strip()]
                if isinstance(tags_raw, list)
                else []
            )
            result = _upsert_presence_account(
                self.vault_root,
                presence_id=presence_id,
                display_name=display_name,
                handle=handle,
                avatar=avatar,
                bio=bio,
                tags=tags,
            )
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/image/commentary":
            req = self._read_json_body() or {}
            image_b64 = str(req.get("image_base64", "") or "").strip()
            image_ref = str(req.get("image_ref", "") or "").strip()
            mime = str(req.get("mime", "image/png") or "image/png").strip()
            presence_id = str(
                req.get("presence_id", "witness_thread") or "witness_thread"
            ).strip()
            prompt = str(req.get("prompt", "") or "").strip()
            persist = _safe_bool_query(
                str(req.get("persist", "true") or "true"),
                default=True,
            )

            if not image_b64:
                self._send_json(
                    {"ok": False, "error": "missing image_base64"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            try:
                image_bytes = base64.b64decode(image_b64, validate=False)
            except (ValueError, OSError):
                self._send_json(
                    {"ok": False, "error": "invalid image_base64 payload"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            commentary_payload = build_image_commentary(
                image_bytes=image_bytes,
                mime=mime,
                image_ref=image_ref,
                presence_id=presence_id,
                prompt=prompt,
                model=str(req.get("model", "") or "").strip() or None,
            )
            if not bool(commentary_payload.get("ok", False)):
                self._send_json(commentary_payload, status=HTTPStatus.BAD_REQUEST)
                return

            _upsert_presence_account(
                self.vault_root,
                presence_id=presence_id,
                display_name=str(req.get("display_name", "") or presence_id),
                handle=str(req.get("handle", "") or presence_id),
                avatar=str(req.get("avatar", "") or ""),
                bio=str(req.get("bio", "") or ""),
                tags=["image-commentary"],
            )

            entry_payload: dict[str, Any] | None = None
            if persist:
                created = _create_image_comment(
                    self.vault_root,
                    image_ref=image_ref
                    or str(
                        commentary_payload.get("analysis", {}).get("image_sha256", "")
                    )[:16],
                    presence_id=presence_id,
                    comment=str(commentary_payload.get("commentary", "")),
                    metadata={
                        "mime": mime,
                        "model": commentary_payload.get("model", ""),
                        "backend": commentary_payload.get("backend", ""),
                        "analysis": commentary_payload.get("analysis", {}),
                    },
                )
                if bool(created.get("ok", False)):
                    entry_payload = dict(created.get("entry", {}))

            self._send_json(
                {
                    "ok": True,
                    "record": "ημ.image-commentary.v1",
                    "presence_id": presence_id,
                    "image_ref": image_ref,
                    "commentary": commentary_payload.get("commentary", ""),
                    "model": commentary_payload.get("model", ""),
                    "backend": commentary_payload.get("backend", ""),
                    "analysis": commentary_payload.get("analysis", {}),
                    "persisted": bool(entry_payload),
                    "entry": entry_payload,
                }
            )
            return

        if parsed.path == "/api/image/comments":
            req = self._read_json_body() or {}
            image_ref = str(req.get("image_ref", "") or "").strip()
            presence_id = str(req.get("presence_id", "") or "").strip()
            comment = str(req.get("comment", "") or "").strip()
            metadata_raw = req.get("metadata", {})
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            result = _create_image_comment(
                self.vault_root,
                image_ref=image_ref,
                presence_id=presence_id,
                comment=comment,
                metadata=metadata,
            )
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/embeddings/db/upsert":
            req = self._read_json_body() or {}
            entry_id = str(req.get("id", "") or "").strip()
            text = str(req.get("text", "") or "")
            metadata_raw = req.get("metadata", {})
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            model = _effective_request_embed_model(req.get("model"))

            embedding = _normalize_embedding_vector(req.get("embedding"))
            if embedding is None and text.strip():
                embedding = _normalize_embedding_vector(
                    _ollama_embed(text, model=model)
                )

            if embedding is None:
                self._send_json(
                    {
                        "ok": False,
                        "error": "missing or invalid embedding; provide embedding or text with reachable embeddings backend",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            result = _embedding_db_upsert(
                self.vault_root,
                entry_id=entry_id,
                text=text,
                embedding=embedding,
                metadata=metadata,
                model=model,
            )
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/embeddings/provider/options":
            req = self._read_json_body() or {}
            result = _apply_embedding_provider_options(req)
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/embeddings/db/query":
            req = self._read_json_body() or {}
            query_text = str(req.get("query", req.get("text", "")) or "").strip()
            model = _effective_request_embed_model(req.get("model"))
            query_embedding = _normalize_embedding_vector(req.get("embedding"))
            if query_embedding is None and query_text:
                query_embedding = _normalize_embedding_vector(
                    _ollama_embed(query_text, model=model)
                )

            if query_embedding is None:
                self._send_json(
                    {
                        "ok": False,
                        "error": "missing or invalid query embedding; provide embedding or query text with reachable embeddings backend",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            top_k = max(
                1,
                min(100, int(_safe_float(str(req.get("top_k", 5) or 5), 5.0))),
            )
            min_score = _safe_float(str(req.get("min_score", -1.0) or -1.0), -1.0)
            include_vectors = _safe_bool_query(
                str(req.get("include_vectors", "false") or "false"),
                default=False,
            )

            result = _embedding_db_query(
                self.vault_root,
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
                include_vectors=include_vectors,
            )
            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/embeddings/db/delete":
            req = self._read_json_body() or {}
            result = _embedding_db_delete(
                self.vault_root,
                entry_id=str(req.get("id", "") or "").strip(),
            )
            status = (
                HTTPStatus.OK if bool(result.get("ok", False)) else HTTPStatus.NOT_FOUND
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/input-stream":
            req = self._read_json_body() or {}
            stream_type = str(req.get("type", "unknown") or "unknown")
            data = req.get("data", {})

            if isinstance(data, dict) and stream_type in {
                "runtime_log",
                "log",
                "stderr",
                "stdout",
            }:
                _INFLUENCE_TRACKER.record_runtime_log(
                    level=str(data.get("level", "info") or "info"),
                    message=str(data.get("message", "") or ""),
                    source=str(data.get("source", "stream") or "stream"),
                )

            if isinstance(data, dict) and stream_type in {
                "resource_heartbeat",
                "heartbeat",
                "health",
            }:
                _INFLUENCE_TRACKER.record_resource_heartbeat(
                    data,
                    source="input-stream",
                )

            input_str = (
                f"{stream_type}: {json.dumps(data, ensure_ascii=False, default=str)}"
            )
            embedding = _normalize_embedding_vector(_ollama_embed(input_str))
            force_vector = _project_vector(embedding)

            collection = _get_chroma_collection()
            if collection and embedding:
                try:
                    memory_id = f"mem_{int(time.time() * 1000)}"
                    collection.add(
                        ids=[memory_id],
                        embeddings=[embedding],
                        metadatas=[
                            {
                                "type": stream_type,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        ],
                        documents=[input_str],
                    )
                except Exception:
                    pass

            queue_snapshot = self.task_queue.snapshot(include_pending=False)
            influence_snapshot = _INFLUENCE_TRACKER.snapshot(
                queue_snapshot=queue_snapshot,
                part_root=self.part_root,
            )

            council_catalog: dict[str, Any] = {}
            if stream_type in {"file_changed", "file_added", "file_removed"}:
                try:
                    council_catalog = self._collect_catalog_fast()
                except Exception:
                    council_catalog = {}

            council_result = self.council_chamber.consider_event(
                event_type=stream_type,
                data=data if isinstance(data, dict) else {"value": data},
                catalog=council_catalog,
                influence_snapshot=influence_snapshot,
            )

            self._send_json(
                {
                    "ok": True,
                    "force": force_vector,
                    "embedding_dim": len(embedding) if embedding else 0,
                    "resource": influence_snapshot.get("resource_heartbeat", {}),
                    "council": council_result,
                }
            )
            return

        if parsed.path == "/api/upload":
            content_type = str(self.headers.get("Content-Type", "") or "")
            file_name = "upload.mp3"
            mime = "audio/mpeg"
            language = "ja"
            file_bytes = b""

            if content_type.lower().startswith("multipart/form-data"):
                form_data = (
                    _parse_multipart_form(self._read_raw_body(), content_type) or {}
                )
                file_field = form_data.get("file")
                if not isinstance(file_field, dict):
                    self._send_json(
                        {"ok": False, "error": "missing multipart file field 'file'"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                file_name = str(
                    file_field.get("filename", "upload.mp3") or "upload.mp3"
                )
                mime = str(
                    file_field.get("content_type")
                    or form_data.get("mime")
                    or mimetypes.guess_type(file_name)[0]
                    or "audio/mpeg"
                )
                language = str(form_data.get("language", "ja") or "ja")
                raw_value = file_field.get("value", b"")
                if isinstance(raw_value, (bytes, bytearray)):
                    file_bytes = bytes(raw_value)
            else:
                req = self._read_json_body() or {}
                file_name = str(req.get("name", "upload.mp3") or "upload.mp3")
                mime = str(req.get("mime", "audio/mpeg") or "audio/mpeg")
                language = str(req.get("language", "ja") or "ja")
                file_b64 = str(req.get("base64", "") or "").strip()
                if file_b64:
                    try:
                        file_bytes = base64.b64decode(file_b64, validate=False)
                    except (ValueError, OSError):
                        file_bytes = b""

            if not file_bytes:
                self._send_json(
                    {"ok": False, "error": "missing or invalid audio payload"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            try:
                safe_name = _normalize_audio_upload_name(file_name, mime)
                save_path = self.part_root / "artifacts" / "audio" / safe_name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(file_bytes)

                result = transcribe_audio_bytes(
                    file_bytes,
                    mime=mime,
                    language=language,
                )
                transcribed_text = str(result.get("text", "") or "").strip()

                collection = _get_chroma_collection()
                if collection and transcribed_text:
                    memory_text = (
                        f"The Weaver learned a new frequency: {transcribed_text}"
                    )
                    payload: dict[str, Any] = {
                        "ids": [f"learn_{int(time.time() * 1000)}"],
                        "metadatas": [
                            {
                                "type": "learned_echo",
                                "source": safe_name,
                                "mime": mime,
                                "language": language,
                                "engine": result.get("engine", "none"),
                            }
                        ],
                        "documents": [memory_text],
                    }
                    embed = _normalize_embedding_vector(_ollama_embed(memory_text))
                    if embed:
                        payload["embeddings"] = [embed]
                    try:
                        collection.add(**payload)
                    except Exception:
                        pass

                self._send_json(
                    {
                        "ok": True,
                        "status": "learned" if transcribed_text else "stored",
                        "engine": result.get("engine", "none"),
                        "text": transcribed_text,
                        "transcription_error": result.get("error"),
                        "url": f"/artifacts/audio/{safe_name}",
                    }
                )
            except Exception as exc:
                self._send_json(
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if parsed.path == "/api/handoff":
            handoff_report: list[str] = []
            collection = _get_chroma_collection()
            if collection:
                try:
                    results = collection.get(limit=50)
                    docs = (
                        results.get("documents", [])
                        if isinstance(results, dict)
                        else []
                    )
                    handoff_report.append("# MISSION HANDOFF / 引き継ぎ")
                    handoff_report.append(
                        f"Generated at: {datetime.now(timezone.utc).isoformat()}"
                    )
                    handoff_report.append("\n## RECENT MEMORY ECHOES")
                    for item in list(docs)[-10:]:
                        handoff_report.append(f"- {str(item)}")
                except Exception:
                    pass

            try:
                constraints_path = self.part_root / "world_state" / "constraints.md"
                handoff_report.append("\n## ACTIVE CONSTRAINTS")
                handoff_report.append(constraints_path.read_text("utf-8"))
            except Exception:
                pass

            self._send_bytes(
                "\n".join(handoff_report).encode("utf-8"),
                "text/markdown; charset=utf-8",
            )
            return

        if parsed.path == "/api/fork-tax/pay":
            req = self._read_json_body() or {}
            amount_raw = req.get("amount", 1.0)
            try:
                amount = float(amount_raw)
            except (TypeError, ValueError):
                self._send_json(
                    {"ok": False, "error": "amount must be numeric"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            source = str(req.get("source", "command-center") or "command-center")
            target = str(req.get("target", "fork_tax_canticle") or "fork_tax_canticle")
            payment = _INFLUENCE_TRACKER.pay_fork_tax(
                amount=amount,
                source=source,
                target=target,
            )

            timestamp = datetime.now(timezone.utc).isoformat()
            entry = {
                "timestamp": timestamp,
                "event": "fork_tax_payment",
                "target": target,
                "source": source,
                "amount": payment["applied"],
                "witness_id": hashlib.sha1(
                    str(time.time_ns()).encode("utf-8")
                ).hexdigest()[:8],
            }
            ledger_path = (
                self.part_root / "world_state" / "decision_ledger.jsonl"
            ).resolve()
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

            queue_snapshot = self.task_queue.snapshot(include_pending=False)
            runtime_snapshot = _INFLUENCE_TRACKER.snapshot(
                queue_snapshot=queue_snapshot,
                part_root=self.part_root,
            )
            self._send_json(
                {
                    "ok": True,
                    "status": "recorded",
                    "payment": payment,
                    "runtime": runtime_snapshot,
                }
            )
            return

        if parsed.path == "/api/chat":
            req = self._read_json_body() or {}
            messages_raw = req.get("messages", [])
            mode = str(req.get("mode", "ollama") or "ollama")
            multi_entity = bool(req.get("multi_entity", False))
            raw_presence_ids = req.get("presence_ids", [])

            presence_ids: list[str] = []
            if isinstance(raw_presence_ids, list):
                for item in raw_presence_ids:
                    value = str(item).strip()
                    if value:
                        presence_ids.append(value)

            if isinstance(messages_raw, list):
                messages = [item for item in messages_raw if isinstance(item, dict)]
            else:
                messages = []

            context = build_world_payload(self.part_root)
            resource_heartbeat = _resource_monitor_snapshot(part_root=self.part_root)
            _INFLUENCE_TRACKER.record_resource_heartbeat(
                resource_heartbeat,
                source="api.chat",
            )
            context["resource_heartbeat"] = resource_heartbeat
            self._send_json(
                build_chat_reply(
                    messages,
                    mode=mode,
                    context=context,
                    multi_entity=multi_entity,
                    presence_ids=presence_ids,
                )
            )
            return

        if parsed.path == "/api/world/interact":
            req = self._read_json_body() or {}
            person_id = str(req.get("person_id", "") or "").strip()
            action = str(req.get("action", "speak") or "speak").strip() or "speak"

            catalog = self._collect_catalog_fast()
            try:
                myth_summary = self.myth_tracker.snapshot(catalog)
            except Exception:
                myth_summary = {}
            try:
                world_summary = self.life_tracker.snapshot(
                    catalog,
                    myth_summary,
                    ENTITY_MANIFEST,
                )
            except Exception:
                world_summary = {}

            try:
                result = self.life_interaction_builder(world_summary, person_id, action)
            except Exception as exc:
                result = {
                    "ok": False,
                    "error": f"interaction_runtime_error:{exc.__class__.__name__}",
                    "line_en": "Interaction failed.",
                    "line_ja": "対話に失敗しました。",
                }
            if not isinstance(result, dict):
                result = {
                    "ok": False,
                    "error": "invalid_interaction_payload",
                    "line_en": "Interaction failed.",
                    "line_ja": "対話に失敗しました。",
                }

            if bool(result.get("ok", False)):
                timestamp = datetime.now(timezone.utc).isoformat()
                entry = {
                    "timestamp": timestamp,
                    "event": "world_interact",
                    "target": str((result.get("presence") or {}).get("id", "unknown")),
                    "person_id": person_id,
                    "action": action,
                    "witness_id": hashlib.sha1(
                        str(time.time_ns()).encode("utf-8")
                    ).hexdigest()[:8],
                }
                ledger_path = (
                    self.part_root / "world_state" / "decision_ledger.jsonl"
                ).resolve()
                ledger_path.parent.mkdir(parents=True, exist_ok=True)
                with ledger_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                _INFLUENCE_TRACKER.record_witness(
                    event_type="world_interact",
                    target=str(entry.get("target", "unknown")),
                )

            status = (
                HTTPStatus.OK
                if bool(result.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/presence/say":
            req = self._read_json_body() or {}
            text = str(req.get("text", "") or "").strip()
            presence_id = str(req.get("presence_id", "") or "").strip()
            catalog, queue_snapshot, _, _, _ = self._runtime_catalog(
                perspective=PROJECTION_DEFAULT_PERSPECTIVE,
            )
            catalog["presence_runtime"] = _INFLUENCE_TRACKER.snapshot(
                queue_snapshot=queue_snapshot,
                part_root=self.part_root,
            )
            self._send_json(build_presence_say_payload(catalog, text, presence_id))
            return

        if parsed.path == "/api/drift/scan":
            self._send_json(build_drift_scan_payload(self.part_root, self.vault_root))
            return

        if parsed.path == "/api/push-truth/dry-run":
            self._send_json(
                build_push_truth_dry_run_payload(self.part_root, self.vault_root)
            )
            return

        if parsed.path == "/api/pi/archive/portable":
            req = self._read_json_body() or {}
            archive_raw = req.get("archive")
            archive = archive_raw if isinstance(archive_raw, dict) else {}
            payload = validate_pi_archive_portable(archive)
            status = (
                HTTPStatus.OK
                if bool(payload.get("ok", False))
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json(payload, status=status)
            return

        if parsed.path == "/api/study/export":
            req = self._read_json_body() or {}
            label = str(req.get("label", "") or "").strip()
            owner = str(req.get("owner", "Err") or "Err").strip() or "Err"
            include_truth_state = bool(req.get("include_truth", False))
            refs_raw = req.get("refs", [])
            refs = (
                [str(item).strip() for item in refs_raw if str(item).strip()]
                if isinstance(refs_raw, list)
                else []
            )

            queue_snapshot = self.task_queue.snapshot(include_pending=True)
            council_snapshot = self.council_chamber.snapshot(
                include_decisions=True,
                limit=128,
            )
            drift_payload = build_drift_scan_payload(self.part_root, self.vault_root)
            resource_snapshot = _resource_monitor_snapshot(part_root=self.part_root)
            _INFLUENCE_TRACKER.record_resource_heartbeat(
                resource_snapshot,
                source="api.study.export",
            )

            truth_gate_blocked: bool | None = None
            if include_truth_state:
                try:
                    truth_state = self._collect_catalog_fast().get("truth_state", {})
                    gate = (
                        truth_state.get("gate", {})
                        if isinstance(truth_state, dict)
                        else {}
                    )
                    if isinstance(gate, dict):
                        truth_gate_blocked = bool(gate.get("blocked", False))
                except Exception:
                    truth_gate_blocked = None

            self._send_json(
                export_study_snapshot(
                    self.part_root,
                    self.vault_root,
                    queue_snapshot=queue_snapshot,
                    council_snapshot=council_snapshot,
                    drift_payload=drift_payload,
                    truth_gate_blocked=truth_gate_blocked,
                    resource_snapshot=resource_snapshot,
                    label=label,
                    owner=owner,
                    refs=refs,
                    host=self.host_label,
                    manifest="manifest.lith",
                )
            )
            return

        if parsed.path == "/api/transcribe":
            req = self._read_json_body() or {}
            audio_b64 = str(req.get("audio_base64", "") or "").strip()
            mime = str(req.get("mime", "audio/webm") or "audio/webm")
            language = str(req.get("language", "ja") or "ja")
            if not audio_b64:
                self._send_json(
                    {
                        "ok": False,
                        "engine": "none",
                        "text": "",
                        "error": "missing audio_base64",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            try:
                audio_bytes = base64.b64decode(
                    audio_b64.encode("utf-8"), validate=False
                )
            except (ValueError, OSError):
                self._send_json(
                    {
                        "ok": False,
                        "engine": "none",
                        "text": "",
                        "error": "invalid base64 audio",
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            self._send_json(
                transcribe_audio_bytes(audio_bytes, mime=mime, language=language)
            )
            return

        if parsed.path == "/api/ux/critique":
            req = self._read_json_body() or {}
            projection_raw = req.get("projection")
            projection: dict[str, Any]
            if isinstance(projection_raw, dict):
                projection = dict(projection_raw)
            else:
                projection = {}
            try:
                try:
                    from ux_critic import critique_ux
                except ImportError:
                    from code.ux_critic import critique_ux

                critique = critique_ux(projection)
                self._send_json({"ok": True, "critique": critique})
            except ImportError as exc:
                self._send_json(
                    {"ok": False, "error": f"ux_critic module missing: {exc}"},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except Exception as exc:
                self._send_json(
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if parsed.path == "/api/witness":
            req = self._read_json_body() or {}
            event_type = str(req.get("type", "touch") or "touch")
            target = str(req.get("target", "unknown") or "unknown")

            timestamp = datetime.now(timezone.utc).isoformat()
            entry = {
                "timestamp": timestamp,
                "event": event_type,
                "target": target,
                "witness_id": hashlib.sha1(
                    str(time.time_ns()).encode("utf-8")
                ).hexdigest()[:8],
            }
            ledger_path = (
                self.part_root / "world_state" / "decision_ledger.jsonl"
            ).resolve()
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            _INFLUENCE_TRACKER.record_witness(event_type=event_type, target=target)
            self._send_json(
                {
                    "ok": True,
                    "status": "recorded",
                    "collapse_id": entry["witness_id"],
                }
            )
            return

        if parsed.path == "/api/task/enqueue":
            req = self._read_json_body() or {}
            kind = str(req.get("kind", "runtime-task") or "runtime-task").strip()
            payload_raw = req.get("payload", {})
            payload = (
                payload_raw if isinstance(payload_raw, dict) else {"value": payload_raw}
            )
            dedupe_key = str(req.get("dedupe_key", "") or "").strip()
            owner = str(req.get("owner", "Err") or "Err").strip()
            refs_raw = req.get("refs", [])
            refs = (
                [str(item).strip() for item in refs_raw if str(item).strip()]
                if isinstance(refs_raw, list)
                else []
            )
            self._send_json(
                self.task_queue.enqueue(
                    kind=kind,
                    payload=payload,
                    dedupe_key=dedupe_key,
                    owner=owner,
                    refs=refs,
                )
            )
            return

        if parsed.path == "/api/task/dequeue":
            req = self._read_json_body() or {}
            owner = str(req.get("owner", "Err") or "Err").strip()
            refs_raw = req.get("refs", [])
            refs = (
                [str(item).strip() for item in refs_raw if str(item).strip()]
                if isinstance(refs_raw, list)
                else []
            )
            result = self.task_queue.dequeue(owner=owner, refs=refs)
            status = (
                HTTPStatus.OK if bool(result.get("ok", False)) else HTTPStatus.CONFLICT
            )
            self._send_json(result, status=status)
            return

        if parsed.path == "/api/council/vote":
            req = self._read_json_body() or {}
            decision_id = str(req.get("decision_id", "") or "").strip()
            member_id = str(req.get("member_id", "") or "").strip()
            vote_value = str(req.get("vote", "") or "").strip().lower()
            reason = str(req.get("reason", "") or "").strip()
            actor = str(req.get("actor", "Err") or "Err").strip() or "Err"
            if (
                not decision_id
                or not member_id
                or vote_value not in {"yes", "no", "abstain"}
            ):
                self._send_json(
                    {
                        "ok": False,
                        "error": "invalid_request",
                        "required": ["decision_id", "member_id", "vote"],
                        "allowed_votes": ["yes", "no", "abstain"],
                    },
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            result = self.council_chamber.vote(
                decision_id=decision_id,
                member_id=member_id,
                vote=vote_value,
                reason=reason,
                actor=actor,
            )
            status = HTTPStatus.OK
            if not bool(result.get("ok", False)):
                error = str(result.get("error", "")).strip()
                if error == "decision_not_found":
                    status = HTTPStatus.NOT_FOUND
                elif error in {"member_not_in_council", "invalid_vote"}:
                    status = HTTPStatus.BAD_REQUEST
                else:
                    status = HTTPStatus.CONFLICT
            self._send_json(result, status=status)
            return

        self._send_json(
            {"ok": False, "error": "not found"},
            status=HTTPStatus.NOT_FOUND,
        )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print(f"[world-web] {self.address_string()} - {format % args}")


def make_handler(
    part_root: Path,
    vault_root: Path,
    host: str = "127.0.0.1",
    port: int = 8787,
):
    receipts_path = _ensure_receipts_log_path(vault_root, part_root)
    queue_log_path = (
        vault_root / ".opencode" / "runtime" / "task_queue.v1.jsonl"
    ).resolve()
    council_log_path = (vault_root / COUNCIL_DECISION_LOG_REL).resolve()

    task_queue = TaskQueue(
        queue_log_path,
        receipts_path,
        owner="Err",
        host=f"{host}:{port}",
    )
    council_chamber = CouncilChamber(
        council_log_path,
        receipts_path,
        owner="Err",
        host=f"{host}:{port}",
        part_root=part_root,
        vault_root=vault_root,
    )

    myth_tracker_class = _load_myth_tracker_class()
    life_tracker_class = _load_life_tracker_class()
    life_interaction_builder = _load_life_interaction_builder()

    try:
        myth_tracker = myth_tracker_class()
    except Exception:

        class _NullMythTracker:
            def snapshot(self, _catalog: dict[str, Any]) -> dict[str, Any]:
                return {}

        myth_tracker = _NullMythTracker()

    try:
        life_tracker = life_tracker_class()
    except Exception:

        class _NullLifeTracker:
            def snapshot(
                self,
                _catalog: dict[str, Any],
                _myth_summary: dict[str, Any],
                _entity_manifest: list[dict[str, Any]],
            ) -> dict[str, Any]:
                return {}

        life_tracker = _NullLifeTracker()

    class BoundWorldHandler(WorldHandler):
        pass

    BoundWorldHandler.part_root = part_root.resolve()
    BoundWorldHandler.vault_root = vault_root.resolve()
    BoundWorldHandler.host_label = f"{host}:{port}"
    BoundWorldHandler.task_queue = task_queue
    BoundWorldHandler.council_chamber = council_chamber
    BoundWorldHandler.myth_tracker = myth_tracker
    BoundWorldHandler.life_tracker = life_tracker
    BoundWorldHandler.life_interaction_builder = life_interaction_builder
    return BoundWorldHandler


def serve(
    part_root: Path,
    vault_root: Path,
    host: str = "127.0.0.1",
    port: int = 8787,
):
    _ensure_weaver_service(part_root, host)
    handler_class = make_handler(part_root, vault_root, host, port)
    server = ThreadingHTTPServer((host, port), handler_class)
    print(f"Starting server on {host}:{port}")
    server.serve_forever()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--part-root", type=Path, default=Path("."))
    parser.add_argument("--vault-root", type=Path, default=Path(".."))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    serve(args.part_root, args.vault_root, args.host, args.port)
    return 0
