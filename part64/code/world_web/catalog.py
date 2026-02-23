from __future__ import annotations
import os
import time
import json
import hashlib
import fnmatch
import math
import mimetypes
import unicodedata
import random
import re
import shutil
import sys
import zipfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs, quote, unquote
from collections import defaultdict
from hashlib import sha1

from .constants import (
    AUDIO_SUFFIXES,
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    ETA_MU_TEXT_SUFFIXES,
    ETA_MU_INBOX_DIRNAME,
    ETA_MU_KNOWLEDGE_INDEX_REL,
    ETA_MU_KNOWLEDGE_ARCHIVE_REL,
    ETA_MU_REGISTRY_REL,
    ETA_MU_INGEST_VECSTORE_COLLECTION,
    ETA_MU_INGEST_SPACE_TEXT_ID,
    ETA_MU_INGEST_TEXT_MODEL,
    ETA_MU_INGEST_TEXT_MODEL_DIGEST,
    ETA_MU_INGEST_TEXT_DIMS,
    ETA_MU_INGEST_TEXT_CHUNK_TARGET,
    ETA_MU_INGEST_TEXT_CHUNK_OVERLAP,
    ETA_MU_INGEST_TEXT_CHUNK_MIN,
    ETA_MU_INGEST_TEXT_CHUNK_MAX,
    ETA_MU_INGEST_SPACE_IMAGE_ID,
    ETA_MU_INGEST_IMAGE_MODEL,
    ETA_MU_INGEST_IMAGE_MODEL_DIGEST,
    ETA_MU_INGEST_IMAGE_DIMS,
    ETA_MU_INGEST_SPACE_SET_ID,
    ETA_MU_INGEST_VECSTORE_ID,
    ETA_MU_INGEST_INCLUDE_TEXT_MIME,
    ETA_MU_INGEST_INCLUDE_IMAGE_MIME,
    ETA_MU_INGEST_INCLUDE_AUDIO_MIME,
    ETA_MU_INGEST_INCLUDE_TEXT_EXT,
    ETA_MU_INGEST_INCLUDE_IMAGE_EXT,
    ETA_MU_INGEST_INCLUDE_AUDIO_EXT,
    ETA_MU_INGEST_EXCLUDE_REL_PATHS,
    ETA_MU_INGEST_EXCLUDE_GLOBS,
    ETA_MU_INGEST_MAX_TEXT_BYTES,
    ETA_MU_INGEST_MAX_IMAGE_BYTES,
    ETA_MU_INGEST_MAX_AUDIO_BYTES,
    ETA_MU_INGEST_MAX_SCAN_FILES,
    ETA_MU_INGEST_MAX_SCAN_DEPTH,
    ETA_MU_INGEST_SAFE_MODE,
    ETA_MU_INGEST_CONTRACT_ID,
    ETA_MU_INGEST_PACKET_RECORD,
    ETA_MU_INGEST_REGISTRY_RECORD,
    ETA_MU_DOCMETA_RECORD,
    ETA_MU_DOCMETA_CYCLE_SECONDS,
    ETA_MU_DOCMETA_MAX_PER_CYCLE,
    ETA_MU_DOCMETA_TEXT_CHAR_LIMIT,
    ETA_MU_DOCMETA_SUMMARY_CHAR_LIMIT,
    ETA_MU_DOCMETA_TAG_LIMIT,
    ETA_MU_DOCMETA_LLM_TIMEOUT_SECONDS,
    ETA_MU_DOCMETA_LLM_MODEL,
    ETA_MU_FILE_GRAPH_LAYER_BLEND,
    ETA_MU_FILE_GRAPH_LAYER_LIMIT,
    ETA_MU_FILE_GRAPH_LAYER_POINT_LIMIT,
    ETA_MU_FILE_GRAPH_TAG_LIMIT,
    ETA_MU_FILE_GRAPH_TAG_EDGE_LIMIT,
    ETA_MU_FILE_GRAPH_TAG_PAIR_EDGE_LIMIT,
    ETA_MU_FILE_GRAPH_ORGANIZER_CLUSTER_THRESHOLD,
    PROJECTION_FIELD_SCHEMAS,
    FIELD_TO_PRESENCE,
    ETA_MU_FIELD_KEYWORDS,
    NAME_HINTS,
    ROLE_HINTS,
    _ETA_MU_INBOX_LOCK,
    _ETA_MU_INBOX_CACHE,
    ETA_MU_INBOX_DEBOUNCE_SECONDS,
    FILE_ORGANIZER_STOPWORDS,
    CANONICAL_NAMED_FIELD_IDS,
    ETA_MU_FILE_GRAPH_ACTIVE_EMBED_LAYERS,
    ETA_MU_FILE_GRAPH_LAYER_ORDER,
    ETA_MU_MAX_GRAPH_FILES,
    FILE_ORGANIZER_PROFILE,
    ETA_MU_FILE_GRAPH_ORGANIZER_MIN_GROUP_SIZE,
    ETA_MU_FILE_GRAPH_ORGANIZER_MAX_CONCEPTS,
    ETA_MU_FILE_GRAPH_ORGANIZER_TERMS_PER_CONCEPT,
    ETA_MU_FILE_GRAPH_RECORD,
    CANONICAL_TERMS,
    ENTITY_MANIFEST,
    PROJECTION_DEFAULT_PERSPECTIVE,
    WORLD_LOG_EVENT_LIMIT,
    ETA_MU_ARCHIVE_MANIFEST_RECORD,
    ETA_MU_KNOWLEDGE_RECORD,
    ETA_MU_INGEST_MANIFEST_PREFIX,
    ETA_MU_INGEST_STATS_PREFIX,
    ETA_MU_INGEST_SNAPSHOT_PREFIX,
    ETA_MU_INGEST_OUTPUT_EXT,
    ETA_MU_INGEST_HEALTH,
    ETA_MU_INGEST_STABILITY,
    ETA_MU_INGEST_MAX_CONCURRENCY,
    ETA_MU_INGEST_BATCH_LIMIT,
)
from .metrics import (
    _safe_float,
    _safe_int,
    _clamp01,
    _json_deep_clone,
    _stable_ratio,
    _infer_eta_mu_field_scores,
    _dominant_eta_mu_field,
    _eta_mu_percentile,
    _split_csv_items,
    _INFLUENCE_TRACKER,
)
from .ai import (
    _embed_text,
    _ollama_embed,
    _ollama_generate_text,
    _eta_mu_detect_modality,
    _eta_mu_guess_mime,
    _eta_mu_canonicalize_text,
    _eta_mu_text_segments,
    _eta_mu_text_language_hint,
    _eta_mu_embed_id,
    _eta_mu_registry_idempotence_key,
    _eta_mu_emit_packet,
    _eta_mu_image_derive_segment,
    _eta_mu_registry_reference_key,
    _eta_mu_json_sha256,
    _eta_mu_model_key,
    _eta_mu_space_forms,
    _eta_mu_embed_vector_for_segment,
    _stable_file_digest,
)
from .db import (
    _load_eta_mu_knowledge_entries,
    _load_eta_mu_docmeta_entries,
    _load_eta_mu_registry_entries,
    _normalize_embedding_vector,
    _embedding_db_upsert,
    _append_eta_mu_registry_record,
    _append_eta_mu_knowledge_record,
    _append_eta_mu_docmeta_record,
    _eta_mu_registry_known_entries,
    _eta_mu_collect_registry_idempotence,
    _eta_mu_vecstore_upsert_batch,
    _load_embeddings_db_state,
    _average_embedding_vectors,
    _assign_meaning_clusters,
)
from .paths import (
    _eta_mu_substrate_root,
    _eta_mu_inbox_root,
    _eta_mu_knowledge_archive_root,
    discover_part_roots,
    _safe_rel_path,
    _eta_mu_inbox_rel_path,
    _eta_mu_scan_candidates,
    _eta_mu_rejected_target_path,
    _archive_member_name,
    _archive_container_id,
    _eta_mu_output_root,
    _eta_mu_iso_compact,
    _eta_mu_write_sexp_artifact,
    _cleanup_empty_inbox_dirs,
    _part_label,
    _sanitize_archive_name,
)

from .projection import (
    _eta_mu_embed_layer_identity,
    _eta_mu_embed_layer_is_active,
    _eta_mu_embed_layer_order_index,
    _eta_mu_embed_layer_point,
    _semantic_xy_from_embedding,
    projection_perspective_options,
)


def _world_web_symbol(name: str, default: Any) -> Any:
    module = sys.modules.get("code.world_web")
    if module is None:
        return default
    return getattr(module, name, default)


def load_manifest(part_root: Path) -> dict[str, Any]:
    manifest_path = part_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text("utf-8"))


def load_constraints(part_root: Path) -> list[str]:
    constraints_path = part_root / "world_state" / "constraints.md"
    if not constraints_path.exists():
        return []
    rows: list[str] = []
    for line in constraints_path.read_text("utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- C-"):
            rows.append(stripped[2:])
    return rows


def build_world_payload(part_root: Path) -> dict[str, Any]:
    manifest = load_manifest(part_root)
    files = manifest.get("files", [])
    roles: dict[str, int] = {}
    for item in files:
        role = str(item.get("role", "unknown"))
        roles[role] = roles.get(role, 0) + 1
    return {
        "part": manifest.get("part"),
        "seed_label": manifest.get("seed_label"),
        "file_count": len(files),
        "roles": roles,
        "constraints": load_constraints(part_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def classify_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if _looks_like_text_file(path):
        return "text"
    return "file"


def _looks_like_text_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in ETA_MU_TEXT_SUFFIXES:
        return True
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("text/"):
        return True
    try:
        with path.open("rb") as handle:
            sample = handle.read(8192)
    except OSError:
        return False
    return _is_probably_text_bytes(sample)


def _is_probably_text_bytes(sample: bytes) -> bool:
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    control_count = sum(1 for byte in sample if byte < 9 or (13 < byte < 32))
    control_ratio = control_count / max(1, len(sample))
    if control_ratio > 0.03:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return control_ratio < 0.01


def build_display_name(item: dict[str, Any], file_path: Path) -> dict[str, str]:
    name = str(item.get("name", file_path.name))
    en, ja = infer_bilingual_name(name)
    return {"en": en, "ja": ja}


def infer_bilingual_name(text: str) -> tuple[str, str]:
    clean = text.strip()
    if not clean:
        return "", ""
    for key, hints in NAME_HINTS:
        if key in clean.lower():
            return hints[0], hints[1]
    return clean, clean


def build_display_role(role: str) -> dict[str, str]:
    hints = ROLE_HINTS.get(role, (role, role))
    return {"en": hints[0], "ja": hints[1]}


def _manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key in ("files", "artifacts"):
        value = manifest.get(key, [])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    entries.append(item)
    return entries


def _library_url_for_archive_member(archive_rel_path: str, member_path: str) -> str:
    return "/library/" + quote(archive_rel_path) + "?member=" + quote(member_path)


def _normalize_archive_member_path(member_path: str) -> str | None:
    raw = str(member_path or "").strip().replace("\\", "/")
    if not raw:
        return None
    parts: list[str] = []
    for part in raw.split("/"):
        token = part.strip()
        if not token or token == ".":
            continue
        if token == "..":
            return None
        parts.append(token)
    return "/".join(parts) if parts else None


def _zip_member_kind(member_path: str, *, is_dir: bool) -> str:
    if is_dir:
        return "directory"
    return classify_kind(Path(member_path))


def _zip_member_extension(member_path: str, *, is_dir: bool) -> str:
    if is_dir:
        return "<dir>"
    suffix = Path(member_path).suffix.lower().strip()
    return suffix if suffix else "<none>"


def _field_schema_name_map() -> dict[str, str]:
    return {
        str(row.get("field", "")).strip(): str(row.get("name", ""))
        for row in PROJECTION_FIELD_SCHEMAS
        if str(row.get("field", "")).strip()
    }


def _organizer_terms_from_entry(entry: dict[str, Any]) -> list[str]:
    if not isinstance(entry, dict):
        return []
    candidates = [
        str(entry.get(k, ""))
        for k in ("name", "source_rel_path", "archive_rel_path", "text_excerpt")
    ]
    terms = []
    for cand in candidates:
        for token in re.findall(r"[A-Za-z0-9_-]+", cand.lower()):
            norm = token.strip("_-")
            if (
                len(norm) >= 3
                and not norm.isdigit()
                and norm not in FILE_ORGANIZER_STOPWORDS
            ):
                terms.append(norm)
    deduped, seen = [], set()
    for t in terms:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped[:48]


def _concept_presence_id(cluster_id: str) -> str:
    seed = str(cluster_id or "").strip() or "concept"
    return f"presence:concept:{sha1(seed.encode('utf-8')).hexdigest()[:12]}"


def _concept_presence_label(terms: list[str], index: int) -> str:
    compact = [t for t in terms if t]
    return (
        f"Concept: {' / '.join(compact[:2])}"
        if compact
        else f"Concept Group {index + 1}"
    )


_ETA_MU_DOCMETA_CYCLE_LOCK = threading.Lock()
_ETA_MU_DOCMETA_CYCLE_BY_ROOT: dict[str, float] = {}


def _normalize_docmeta_path(path_like: str) -> str:
    raw = str(path_like or "").strip().replace("\\", "/")
    if not raw:
        return ""
    parts: list[str] = []
    for token in raw.split("/"):
        piece = token.strip()
        if not piece or piece == ".":
            continue
        if piece == "..":
            if parts:
                parts.pop()
            continue
        parts.append(piece)
    return "/".join(parts)


def _docmeta_key_candidates(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    entry_id = str(row.get("id") or row.get("knowledge_id") or "").strip()
    if entry_id:
        keys.append(f"id:{entry_id}")
    source_hash = str(row.get("source_hash", "")).strip()
    if source_hash:
        keys.append(f"hash:{source_hash}")
    source_path = _normalize_docmeta_path(str(row.get("source_rel_path", "")))
    if source_path:
        keys.append(f"source:{source_path}")
    archive_path = _normalize_docmeta_path(
        str(row.get("archive_rel_path") or row.get("archived_rel_path") or "")
    )
    if archive_path:
        keys.append(f"archive:{archive_path}")
    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _normalize_docmeta_tag(raw: Any) -> str:
    token = unicodedata.normalize("NFKC", str(raw or "")).strip().lower()
    if not token:
        return ""
    token = token.replace("-", "_")
    token = re.sub(r"[^a-z0-9_\s]", " ", token)
    token = re.sub(r"\s+", "_", token)
    token = token.strip("_")
    if len(token) < 2 or token.isdigit():
        return ""
    return token[:36].rstrip("_")


def _docmeta_label_from_tag(tag: str) -> str:
    normalized = _normalize_docmeta_tag(tag)
    if not normalized:
        return ""
    return " ".join(part.capitalize() for part in normalized.split("_") if part)


def _dedupe_docmeta_tags(
    raw_tags: Any, *, limit: int = ETA_MU_DOCMETA_TAG_LIMIT
) -> list[str]:
    max_tags = max(1, int(limit or ETA_MU_DOCMETA_TAG_LIMIT))
    tags: list[str] = []
    if isinstance(raw_tags, list):
        tags = [str(item) for item in raw_tags]
    elif isinstance(raw_tags, str):
        tags = re.split(r"[,\n;|]", raw_tags)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in tags:
        normalized = _normalize_docmeta_tag(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) >= max_tags:
            break
    return deduped


def _heuristic_docmeta_tags(entry: dict[str, Any], source_text: str) -> list[str]:
    seeds = _organizer_terms_from_entry(entry)
    dominant_field = str(entry.get("dominant_field", "")).strip()
    if dominant_field:
        seeds.append(dominant_field)
    kind_value = str(entry.get("kind", "")).strip()
    if kind_value:
        seeds.append(kind_value)

    lower_source = str(source_text or "").lower()
    if lower_source:
        for field_id, keywords in ETA_MU_FIELD_KEYWORDS.items():
            if any(
                str(keyword).strip().lower() in lower_source for keyword in keywords
            ):
                seeds.append(field_id)

    return _dedupe_docmeta_tags(seeds, limit=ETA_MU_DOCMETA_TAG_LIMIT)


def _heuristic_docmeta_summary(entry: dict[str, Any], source_text: str) -> str:
    max_len = max(80, int(ETA_MU_DOCMETA_SUMMARY_CHAR_LIMIT))
    excerpt = str(entry.get("text_excerpt", "")).strip()
    if excerpt:
        compact = " ".join(excerpt.split())
        return compact[:max_len]

    compact_source = " ".join(str(source_text or "").split())
    if compact_source:
        return compact_source[:max_len]

    name = str(entry.get("name", "")).strip() or "document"
    kind_value = str(entry.get("kind", "file")).strip() or "file"
    fallback = f"{name} ({kind_value})"
    return fallback[:max_len]


def _parse_docmeta_json(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        return {}

    candidates: list[str] = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.append(str(fenced.group(1)))
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _build_docmeta_source_text(entry: dict[str, Any], vault_root: Path) -> str:
    max_len = max(200, int(ETA_MU_DOCMETA_TEXT_CHAR_LIMIT))
    parts: list[str] = []

    for key in ("name", "source_rel_path", "archived_rel_path", "archive_rel_path"):
        value = str(entry.get(key, "")).strip()
        if value:
            parts.append(value)

    excerpt = str(entry.get("text_excerpt", "")).strip()
    if excerpt:
        parts.append(excerpt)

    kind_value = str(entry.get("kind", "")).strip().lower()
    archive_kind = str(entry.get("archive_kind", "")).strip().lower()
    archive_rel_path = str(
        entry.get("archive_rel_path", entry.get("archived_rel_path", ""))
    ).strip()
    archive_member_path = str(entry.get("archive_member_path", "")).strip()
    if (
        kind_value == "text"
        and archive_kind == "zip"
        and archive_rel_path
        and archive_member_path
    ):
        archive_path = (_eta_mu_substrate_root(vault_root) / archive_rel_path).resolve()
        payload_pair = _read_library_archive_member(archive_path, archive_member_path)
        if payload_pair is not None:
            payload_bytes, content_type = payload_pair
            if "text" in content_type or "json" in content_type:
                decoded = payload_bytes.decode("utf-8", errors="replace")
                compact = "\n".join(decoded.splitlines())
                if compact:
                    parts.append(compact[:max_len])

    source_text = "\n".join(part for part in parts if part)
    return source_text[:max_len]


def _build_docmeta_record(entry: dict[str, Any], vault_root: Path) -> dict[str, Any]:
    source_text = _build_docmeta_source_text(entry, vault_root)
    summary = ""
    tags: list[str] = []
    model_name = ""
    strategy = "heuristic"

    prompt = (
        "You are creating metadata for a simulation knowledge graph. "
        "Return JSON only with keys summary and tags. "
        "summary must be a single sentence under 280 chars. "
        "tags must be 3-8 concise lowercase tokens using underscores.\n"
        f"name: {str(entry.get('name', '')).strip()}\n"
        f"kind: {str(entry.get('kind', '')).strip()}\n"
        f"dominant_field: {str(entry.get('dominant_field', '')).strip()}\n"
        f"source_rel_path: {str(entry.get('source_rel_path', '')).strip()}\n"
        f"text:\n{source_text}"
    )
    text, model_name = _ollama_generate_text(
        prompt,
        model=ETA_MU_DOCMETA_LLM_MODEL or None,
        timeout_s=ETA_MU_DOCMETA_LLM_TIMEOUT_SECONDS,
    )
    if text:
        payload = _parse_docmeta_json(text)
        summary = str(payload.get("summary", "")).strip()
        tags = _dedupe_docmeta_tags(
            payload.get("tags", []), limit=ETA_MU_DOCMETA_TAG_LIMIT
        )
        if summary or tags:
            strategy = "llm"

    if not summary:
        summary = _heuristic_docmeta_summary(entry, source_text)
    if not tags:
        tags = _heuristic_docmeta_tags(entry, source_text)

    labels = [
        label for label in [_docmeta_label_from_tag(tag) for tag in tags] if label
    ]
    record = {
        "record": ETA_MU_DOCMETA_RECORD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_id": str(entry.get("id", "")).strip(),
        "source_hash": str(entry.get("source_hash", "")).strip(),
        "source_rel_path": str(entry.get("source_rel_path", "")).strip(),
        "archive_rel_path": str(
            entry.get("archive_rel_path", entry.get("archived_rel_path", ""))
        ).strip(),
        "kind": str(entry.get("kind", "")).strip(),
        "summary": summary[: max(80, int(ETA_MU_DOCMETA_SUMMARY_CHAR_LIMIT))],
        "tags": tags[: max(1, int(ETA_MU_DOCMETA_TAG_LIMIT))],
        "labels": labels[: max(1, int(ETA_MU_DOCMETA_TAG_LIMIT))],
        "model": str(model_name or ""),
        "strategy": strategy,
    }
    return record


def _index_docmeta_records(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in _docmeta_key_candidates(row):
            if key and key not in index:
                index[key] = row
    return index


def _docmeta_for_entry(
    entry: dict[str, Any], index: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    for key in _docmeta_key_candidates(entry):
        row = index.get(key)
        if isinstance(row, dict):
            return row
    return {}


def _refresh_eta_mu_docmeta_index(
    vault_root: Path,
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows = _load_eta_mu_docmeta_entries(vault_root)
    index = _index_docmeta_records(rows)
    if not entries:
        return index

    root_key = str(_eta_mu_substrate_root(vault_root))
    now_monotonic = time.monotonic()
    cycle_seconds = max(
        0.0,
        _safe_float(
            _world_web_symbol(
                "ETA_MU_DOCMETA_CYCLE_SECONDS", ETA_MU_DOCMETA_CYCLE_SECONDS
            ),
            ETA_MU_DOCMETA_CYCLE_SECONDS,
        ),
    )
    max_per_cycle = max(
        1,
        _safe_int(
            _world_web_symbol(
                "ETA_MU_DOCMETA_MAX_PER_CYCLE", ETA_MU_DOCMETA_MAX_PER_CYCLE
            ),
            ETA_MU_DOCMETA_MAX_PER_CYCLE,
        ),
    )
    with _ETA_MU_DOCMETA_CYCLE_LOCK:
        last = _safe_float(_ETA_MU_DOCMETA_CYCLE_BY_ROOT.get(root_key, 0.0), 0.0)
        if now_monotonic - last < cycle_seconds:
            return index
        _ETA_MU_DOCMETA_CYCLE_BY_ROOT[root_key] = now_monotonic

    candidates: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        existing = _docmeta_for_entry(entry, index)
        source_hash = str(entry.get("source_hash", "")).strip()
        if existing:
            existing_hash = str(existing.get("source_hash", "")).strip()
            summary_ok = bool(str(existing.get("summary", "")).strip())
            tags_ok = len(
                _dedupe_docmeta_tags(
                    existing.get("tags", []), limit=ETA_MU_DOCMETA_TAG_LIMIT
                )
            )
            if (
                (not source_hash or not existing_hash or source_hash == existing_hash)
                and summary_ok
                and tags_ok > 0
            ):
                continue
        candidates.append(entry)

    generated = 0
    for entry in candidates:
        if generated >= max_per_cycle:
            break
        record = _build_docmeta_record(entry, vault_root)
        _append_eta_mu_docmeta_record(vault_root, record)
        generated += 1

    if generated > 0:
        rows = _load_eta_mu_docmeta_entries(vault_root)
        index = _index_docmeta_records(rows)
    return index


def sync_eta_mu_inbox(vault_root: Path) -> dict[str, Any]:
    base_root, inbox_root = (
        _eta_mu_substrate_root(vault_root),
        _eta_mu_inbox_root(vault_root),
    )
    now_monotonic, spaces = time.monotonic(), _eta_mu_space_forms()

    debounce_seconds = max(
        0.0,
        _safe_float(
            _world_web_symbol(
                "ETA_MU_INBOX_DEBOUNCE_SECONDS", ETA_MU_INBOX_DEBOUNCE_SECONDS
            ),
            ETA_MU_INBOX_DEBOUNCE_SECONDS,
        ),
    )

    with _ETA_MU_INBOX_LOCK:
        if (
            str(_ETA_MU_INBOX_CACHE.get("root", "")) == str(inbox_root)
            and _ETA_MU_INBOX_CACHE.get("snapshot")
            and (
                now_monotonic
                - float(_ETA_MU_INBOX_CACHE.get("last_checked_monotonic", 0.0))
            )
            < debounce_seconds
        ):
            return dict(_ETA_MU_INBOX_CACHE["snapshot"])

    if not inbox_root.exists() or not inbox_root.is_dir():
        snapshot = {
            "record": "ημ.inbox.v1",
            "path": str(inbox_root),
            "pending_count": 0,
            "processed_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "rejected_count": 0,
            "deferred_count": 0,
            "is_empty": True,
            "knowledge_entries": len(_load_eta_mu_knowledge_entries(vault_root)),
            "registry_entries": len(_load_eta_mu_registry_entries(vault_root)),
            "last_ingested_at": "",
            "contract": ETA_MU_INGEST_CONTRACT_ID,
            "spaces": {
                "text": {
                    "id": spaces["text"]["id"],
                    "signature": spaces["text"]["signature"],
                    "collection": spaces["text"].get("collection", ""),
                },
                "image": {
                    "id": spaces["image"]["id"],
                    "signature": spaces["image"]["signature"],
                    "collection": spaces["image"].get("collection", ""),
                },
                "space_set": {
                    "id": spaces["space_set"]["id"],
                    "signature": spaces["space_set"]["signature"],
                },
                "vecstore": {
                    "id": spaces["vecstore"]["id"],
                    "collection": spaces["vecstore"].get("collection", ""),
                    "layer_mode": spaces["vecstore"].get("layer_mode", "single"),
                },
            },
            "errors": [],
        }
        with _ETA_MU_INBOX_LOCK:
            _ETA_MU_INBOX_CACHE.update(
                {
                    "root": str(inbox_root),
                    "snapshot": dict(snapshot),
                    "last_checked_monotonic": now_monotonic,
                }
            )
        return snapshot

    archive_root = _eta_mu_knowledge_archive_root(vault_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    field_name_map = _field_schema_name_map()
    pending_files = _eta_mu_scan_candidates(inbox_root)
    processed_count = skipped_count = failed_count = rejected_count = deferred_count = 0
    (
        errors,
        last_ingested_at,
        modality_counts,
        embed_call_ms,
        vecstore_call_ms,
        packets,
        manifest_sources,
    ) = [], "", {"text": 0, "image": 0, "audio": 0}, [], [], [], []
    registry_entries = _load_eta_mu_registry_entries(vault_root)
    registry_known, registry_known_idempotence = (
        _eta_mu_registry_known_entries(registry_entries),
        _eta_mu_collect_registry_idempotence(registry_entries),
    )

    _eta_mu_emit_packet(
        packets,
        kind="ingest-scan",
        body={
            "scope": "scope.ημ.ingest",
            "root": str(inbox_root),
            "candidate_count": len(pending_files),
            "ordering": "path-lexical",
            "max_files": ETA_MU_INGEST_MAX_SCAN_FILES,
            "max_depth": ETA_MU_INGEST_MAX_SCAN_DEPTH,
        },
    )

    for file_path in pending_files:
        rel_source, inbox_rel = (
            _safe_rel_path(file_path, base_root),
            _eta_mu_inbox_rel_path(file_path, inbox_root),
        )
        try:
            stat = file_path.stat()
            source_bytes, source_mtime_utc, raw = (
                int(stat.st_size),
                datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                file_path.read_bytes(),
            )
        except OSError as exc:
            failed_count += 1
            errors.append({"path": rel_source, "error": str(exc)})
            continue

        source_hash, mime = (
            hashlib.sha256(raw).hexdigest(),
            _eta_mu_guess_mime(file_path, raw),
        )
        modality, decision = _eta_mu_detect_modality(path=file_path, mime=mime)

        if modality is None:
            rejected_count += 1
            rejected_at = datetime.now(timezone.utc).isoformat()
            reject_packet = _eta_mu_emit_packet(
                packets,
                kind="ingest-reject",
                body={
                    "source_rel_path": rel_source,
                    "inbox_rel_path": inbox_rel,
                    "reason": decision,
                    "mime": mime,
                    "bytes": source_bytes,
                },
            )
            try:
                target = _eta_mu_rejected_target_path(
                    inbox_root=inbox_root,
                    source_path=file_path,
                    source_hash=source_hash,
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, target)
                file_path.unlink()
            except OSError as exc:
                failed_count += 1
                errors.append({"path": rel_source, "error": str(exc)})
                continue
            _append_eta_mu_registry_record(
                vault_root,
                {
                    "record": ETA_MU_INGEST_REGISTRY_RECORD,
                    "event": "rejected",
                    "status": "reject",
                    "reason": decision,
                    "registry_key": _eta_mu_registry_reference_key(
                        source_hash=source_hash,
                        rel_source=rel_source,
                        source_bytes=source_bytes,
                        modality="reject",
                    ),
                    "content_sha256": source_hash,
                    "source_hash": source_hash,
                    "source_rel_path": rel_source,
                    "source_name": file_path.name,
                    "source_kind": "reject",
                    "source_modality": "reject",
                    "source_mime": mime,
                    "source_bytes": source_bytes,
                    "source_mtime_utc": source_mtime_utc,
                    "packet_refs": [reject_packet["id"]],
                    "time": rejected_at,
                },
            )
            continue

        if (
            (modality == "text" and source_bytes > ETA_MU_INGEST_MAX_TEXT_BYTES)
            or (modality == "image" and source_bytes > ETA_MU_INGEST_MAX_IMAGE_BYTES)
            or (modality == "audio" and source_bytes > ETA_MU_INGEST_MAX_AUDIO_BYTES)
        ):
            rejected_count += 1
            rejected_at = datetime.now(timezone.utc).isoformat()
            _eta_mu_emit_packet(
                packets,
                kind="ingest-reject",
                body={
                    "source_rel_path": rel_source,
                    "inbox_rel_path": inbox_rel,
                    "reason": f"max-bytes:{modality}",
                    "bytes": source_bytes,
                },
            )
            file_path.unlink()
            continue  # Simplified rejection

        modality_counts[modality] += 1
        found_packet = _eta_mu_emit_packet(
            packets,
            kind="ingest-found",
            body={
                "source_rel_path": rel_source,
                "inbox_rel_path": inbox_rel,
                "source_hash": source_hash,
                "mime": mime,
                "modality": modality,
                "bytes": source_bytes,
                "decision": decision,
            },
        )

        if modality == "text":
            can_text = _eta_mu_canonicalize_text(raw)
            segments = _eta_mu_text_segments(
                can_text, language_hint=_eta_mu_text_language_hint(file_path)
            )
            excerpt = (" ".join(can_text.split()))[:1599] + "…"
            space = spaces["text"]
        elif modality == "image":
            segments = [
                _eta_mu_image_derive_segment(
                    source_hash=source_hash,
                    source_bytes=source_bytes,
                    source_rel_path=rel_source,
                    mime=mime,
                    image_bytes=raw,
                )
            ]
            excerpt = ""
            space = spaces["image"]
        elif modality == "audio":
            segments = [
                {
                    "text": f"audio artifact: {file_path.name} ({mime})",
                    "source_hash": source_hash,
                }
            ]
            excerpt = f"Audio file: {file_path.name}"
            space = spaces["audio"]
        else:
            # Fallback for unexpected modality
            segments = []
            excerpt = ""
            space = spaces["text"]

        segment_plans = []
        for seg in segments:
            eid = _eta_mu_embed_id(
                space_signature=str(space.get("signature", "")),
                source_hash=source_hash,
                segment=seg,
            )
            ik = _eta_mu_registry_idempotence_key(
                embed_id=eid,
                source_hash=source_hash,
                space_signature=str(space.get("signature", "")),
                segment=seg,
            )
            segment_plans.append(
                {"segment": dict(seg), "embed_id": eid, "idempotence_key": ik}
            )

        chunk_packet = _eta_mu_emit_packet(
            packets,
            kind="chunk-plan",
            body={
                "source_rel_path": rel_source,
                "modality": modality,
                "segment_count": len(segment_plans),
                "space_id": str(space.get("id", "")),
                "space_signature": str(space.get("signature", "")),
            },
        )
        registry_key = _eta_mu_registry_reference_key(
            source_hash=source_hash,
            rel_source=rel_source,
            source_bytes=source_bytes,
            modality=modality,
        )

        if all(
            str(p["idempotence_key"]) in registry_known_idempotence
            for p in segment_plans
        ):
            skipped_count += 1
            _append_eta_mu_registry_record(
                vault_root,
                {
                    "record": ETA_MU_INGEST_REGISTRY_RECORD,
                    "event": "skipped",
                    "status": "skip",
                    "reason": "duplicate_unchanged",
                    "registry_key": registry_key,
                    "source_name": file_path.name,
                    "source_rel_path": rel_source,
                    "source_hash": source_hash,
                    "idempotence_key": str(segment_plans[0].get("idempotence_key", "")),
                    "time": datetime.now(timezone.utc).isoformat(),
                },
            )
            file_path.unlink()
            continue  # Simplified skip

        vec_collection = (
            str(space.get("collection", ETA_MU_INGEST_VECSTORE_COLLECTION)).strip()
            or ETA_MU_INGEST_VECSTORE_COLLECTION
        )
        vec_rows, embed_refs = [], []

        for plan in segment_plans:
            seg = plan["segment"]
            _eta_mu_emit_packet(
                packets,
                kind="call-plan",
                body={
                    "resource": "rc.ollama.embed",
                    "op": "embed",
                    "source_rel_path": rel_source,
                    "embed_id": plan["embed_id"],
                },
            )
            started = time.perf_counter()
            vec, m_name, fallback = _eta_mu_embed_vector_for_segment(
                modality=modality,
                segment_text=str(seg.get("text", "")),
                space=space,
                embed_id=plan["embed_id"],
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            embed_call_ms.append(elapsed)

            model_key = _eta_mu_model_key(
                model_name=m_name,
                model_digest=str((space.get("model") or {}).get("digest", "none")),
                settings_hash=_eta_mu_json_sha256(
                    {"space_signature": space.get("signature"), "modality": modality}
                ),
            )
            layer_idnt = _eta_mu_embed_layer_identity(
                collection=vec_collection,
                space_id=str(space.get("id", "")),
                space_signature=str(space.get("signature", "")),
                model_name=m_name,
            )

            ref_packet = _eta_mu_emit_packet(
                packets,
                kind="embed-ref",
                body={
                    "embed_id": plan["embed_id"],
                    "source_rel_path": rel_source,
                    "source_hash": source_hash,
                    "model_key": model_key,
                    "dims": len(vec),
                },
            )
            vec_rows.append(
                {
                    "id": plan["embed_id"],
                    "embedding": vec,
                    "metadata": {
                        "source.loc": rel_source,
                        "source.hash": source_hash,
                        "time": datetime.now(timezone.utc).isoformat(),
                    },
                    "document": str(seg.get("text", "")),
                    "model": m_name,
                }
            )
            embed_refs.append(
                {
                    **plan,
                    "model_key": model_key,
                    "model_name": m_name,
                    "layer_id": layer_idnt["id"],
                    "packet_ref": ref_packet["id"],
                }
            )

        v_res = _eta_mu_vecstore_upsert_batch(
            vault_root,
            vec_rows,
            collection_name=vec_collection,
            space_set_signature=str(spaces["space_set"].get("signature", "")),
        )
        if v_res.get("drift"):
            deferred_count += 1

        scores = _infer_eta_mu_field_scores(
            rel_path=rel_source, kind=modality, text_excerpt=excerpt
        )
        dom_f, dom_w = _dominant_eta_mu_field(scores)
        digest = _stable_file_digest(file_path)
        ingested_at = datetime.now(timezone.utc).isoformat()

        from .paths import _sanitize_archive_name

        archive_rel = f"{ETA_MU_KNOWLEDGE_ARCHIVE_REL}/{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/{digest[:12]}_{_sanitize_archive_name(file_path.name)}.zip"
        archive_abs = (base_root / archive_rel).resolve()
        archive_abs.parent.mkdir(parents=True, exist_ok=True)

        record_id = f"knowledge.{sha1(f'{digest}|{rel_source}|{ingested_at}'.encode()).hexdigest()[:14]}"
        manifest_payload = {
            "record": "ημ.archive-manifest.v1",
            "record_id": record_id,
            "contract": ETA_MU_INGEST_CONTRACT_ID,
            "fingerprint": digest,
            "ingested_at": ingested_at,
            "source_rel_path": rel_source,
            "source_name": file_path.name,
            "source_kind": modality,
            "archive_rel_path": archive_rel,
            "archive_member_path": _archive_member_name(file_path.name),
        }

        with zipfile.ZipFile(archive_abs, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(file_path, arcname=_archive_member_name(file_path.name))
            zf.writestr("manifest.json", json.dumps(manifest_payload))

        file_path.unlink()
        archive_member_path = _archive_member_name(file_path.name)
        archive_container_id = _archive_container_id(archive_rel)
        archive_url = "/library/" + quote(archive_rel)
        record = {
            "record": ETA_MU_KNOWLEDGE_RECORD,
            "id": record_id,
            "name": file_path.name,
            "source_rel_path": rel_source,
            "archived_rel_path": archive_rel,
            "archive_rel_path": archive_rel,
            "archive_kind": "zip",
            "archive_url": archive_url,
            "archive_member_path": archive_member_path,
            "archive_manifest_path": "manifest.json",
            "archive_container_id": archive_container_id,
            "archive_manifest_record": ETA_MU_ARCHIVE_MANIFEST_RECORD,
            "archive_manifest_url": _library_url_for_archive_member(
                archive_rel, "manifest.json"
            ),
            "url": _library_url_for_archive_member(archive_rel, archive_member_path),
            "kind": modality,
            "mime": mime,
            "source_hash": source_hash,
            "contract": ETA_MU_INGEST_CONTRACT_ID,
            "bytes": source_bytes,
            "mtime_utc": source_mtime_utc,
            "text_excerpt": excerpt,
            "ingested_at": ingested_at,
            "dominant_field": dom_f,
            "dominant_presence": FIELD_TO_PRESENCE.get(dom_f, "anchor_registry"),
            "dominant_weight": round(_clamp01(dom_w), 4),
            "field_scores": scores,
            "embedding_links": [
                {
                    "kind": "stored_in_archive",
                    "target": archive_container_id,
                    "member_path": archive_member_path,
                    "weight": 1.0,
                },
                *[
                    {
                        "kind": "embed_ref",
                        "embed_id": str(ref.get("embed_id", "")),
                        "idempotence_key": str(ref.get("idempotence_key", "")),
                        "vecstore_collection": vec_collection,
                        "space_id": str(space.get("id", "")),
                        "space_signature": str(space.get("signature", "")),
                        "model_name": str(ref.get("model_name", "")),
                        "model_key": str(ref.get("model_key", "")),
                        "layer_id": str(ref.get("layer_id", "")),
                        "packet_ref": str(ref.get("packet_ref", "")),
                        "weight": 1.0,
                    }
                    for ref in embed_refs
                    if str(ref.get("embed_id", "")).strip()
                ],
            ],
        }
        _append_eta_mu_knowledge_record(vault_root, record)
        for ref in embed_refs:
            _append_eta_mu_registry_record(
                vault_root,
                {
                    "record": ETA_MU_INGEST_REGISTRY_RECORD,
                    "event": "ingested",
                    "status": "ok",
                    "registry_key": registry_key,
                    "idempotence_key": ref["idempotence_key"],
                    "embed_id": str(ref.get("embed_id", "")),
                    "knowledge_id": record_id,
                    "source_name": file_path.name,
                    "source_rel_path": rel_source,
                    "source_hash": source_hash,
                    "source_kind": modality,
                    "source_mime": mime,
                    "source_bytes": source_bytes,
                    "time": ingested_at,
                },
            )
        manifest_sources.append(
            {
                "source_rel_path": rel_source,
                "source_hash": source_hash,
                "modality": modality,
            }
        )
        processed_count += 1
        last_ingested_at = ingested_at

    _cleanup_empty_inbox_dirs(inbox_root)
    remaining_candidates = _eta_mu_scan_candidates(inbox_root)
    entries = _load_eta_mu_knowledge_entries(vault_root)
    generated_at = datetime.now(timezone.utc).isoformat()
    output_root = _eta_mu_output_root(vault_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_stamp = _eta_mu_iso_compact()
    manifest_rel = (
        output_root
        / (ETA_MU_INGEST_MANIFEST_PREFIX + output_stamp + ETA_MU_INGEST_OUTPUT_EXT)
    ).resolve()
    stats_rel = (
        output_root
        / (ETA_MU_INGEST_STATS_PREFIX + output_stamp + ETA_MU_INGEST_OUTPUT_EXT)
    ).resolve()
    snapshot_rel = (
        output_root
        / (ETA_MU_INGEST_SNAPSHOT_PREFIX + output_stamp + ETA_MU_INGEST_OUTPUT_EXT)
    ).resolve()

    manifest_payload = {
        "record": "ημ.ingest-manifest.v1",
        "contract": ETA_MU_INGEST_CONTRACT_ID,
        "generated_at": generated_at,
        "spaces": {
            "text": spaces["text"],
            "image": spaces["image"],
            "space_set": spaces["space_set"],
            "vecstore": spaces["vecstore"],
        },
        "sources": manifest_sources,
        "packet_refs": [str(packet.get("id", "")) for packet in packets],
    }
    stats_payload = {
        "record": "ημ.ingest-stats.v1",
        "contract": ETA_MU_INGEST_CONTRACT_ID,
        "generated_at": generated_at,
        "counts": {
            "text": int(modality_counts.get("text", 0)),
            "image": int(modality_counts.get("image", 0)),
            "processed": processed_count,
            "skipped": skipped_count,
            "rejected": rejected_count,
            "deferred": deferred_count,
            "failed": failed_count,
        },
        "costs": {
            "embed_ms_total": round(sum(embed_call_ms), 3),
            "vecstore_ms_total": round(sum(vecstore_call_ms), 3),
            "embed_calls": len(embed_call_ms),
            "vecstore_calls": len(vecstore_call_ms),
        },
        "latency": {
            "embed_p95_ms": round(_eta_mu_percentile(embed_call_ms, 0.95), 3),
            "vecstore_p95_ms": round(_eta_mu_percentile(vecstore_call_ms, 0.95), 3),
        },
        "capacity": {
            "embed_max_concurrency": ETA_MU_INGEST_MAX_CONCURRENCY,
            "embed_batch_limit": ETA_MU_INGEST_BATCH_LIMIT,
        },
    }
    snapshot_payload = {
        "record": "ημ.ingest-snapshot.v1",
        "contract": ETA_MU_INGEST_CONTRACT_ID,
        "generated_at": generated_at,
        "scope": "scope.ημ.ingest",
        "gates": {
            "health": ETA_MU_INGEST_HEALTH,
            "stability": ETA_MU_INGEST_STABILITY,
            "coherence": "和",
            "sync": "同",
            "safe_mode": ETA_MU_INGEST_SAFE_MODE,
        },
        "packet_count": len(packets),
        "packet_last": str(packets[-1].get("id", "")) if packets else "",
        "hashes": {
            "manifest_sha256": _eta_mu_json_sha256(manifest_payload),
            "stats_sha256": _eta_mu_json_sha256(stats_payload),
        },
    }

    artifact_errors: list[str] = []
    try:
        _eta_mu_write_sexp_artifact(manifest_rel, manifest_payload)
    except OSError as exc:
        artifact_errors.append(f"manifest:{exc}")
    try:
        _eta_mu_write_sexp_artifact(stats_rel, stats_payload)
    except OSError as exc:
        artifact_errors.append(f"stats:{exc}")
    try:
        _eta_mu_write_sexp_artifact(snapshot_rel, snapshot_payload)
    except OSError as exc:
        artifact_errors.append(f"snapshot:{exc}")

    if artifact_errors:
        failed_count += len(artifact_errors)
        for message in artifact_errors:
            errors.append({"path": ".Π", "error": message})

    manifest_rel_path = _safe_rel_path(manifest_rel, base_root)
    stats_rel_path = _safe_rel_path(stats_rel, base_root)
    snapshot_rel_path = _safe_rel_path(snapshot_rel, base_root)

    _eta_mu_emit_packet(
        packets,
        kind="snapshot-seal",
        body={
            "manifest": manifest_rel_path,
            "stats": stats_rel_path,
            "snapshot": snapshot_rel_path,
            "packet_count": len(packets),
            "coherence": "和",
            "sync": "同",
        },
    )

    snapshot = {
        "record": "ημ.inbox.v1",
        "path": str(inbox_root),
        "pending_count": len(remaining_candidates),
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "rejected_count": rejected_count,
        "deferred_count": deferred_count,
        "is_empty": len(remaining_candidates) == 0,
        "knowledge_entries": len(entries),
        "registry_entries": len(_load_eta_mu_registry_entries(vault_root)),
        "last_ingested_at": last_ingested_at,
        "contract": ETA_MU_INGEST_CONTRACT_ID,
        "spaces": {
            "text": {
                "id": spaces["text"]["id"],
                "signature": spaces["text"]["signature"],
                "collection": spaces["text"].get("collection", ""),
            },
            "image": {
                "id": spaces["image"]["id"],
                "signature": spaces["image"]["signature"],
                "collection": spaces["image"].get("collection", ""),
            },
            "space_set": {
                "id": spaces["space_set"]["id"],
                "signature": spaces["space_set"]["signature"],
            },
            "vecstore": {
                "id": spaces["vecstore"]["id"],
                "collection": spaces["vecstore"]["collection"],
                "collections": spaces["vecstore"].get("collections", {}),
                "layer_mode": spaces["vecstore"].get("layer_mode", "single"),
                "signature": spaces["vecstore"]["signature"],
            },
        },
        "stats": stats_payload["counts"],
        "artifacts": {
            "manifest": manifest_rel_path,
            "stats": stats_rel_path,
            "snapshot": snapshot_rel_path,
        },
        "packets": {
            "count": len(packets),
            "last": str(packets[-1].get("id", "")) if packets else "",
        },
        "errors": errors,
    }

    with _ETA_MU_INBOX_LOCK:
        _ETA_MU_INBOX_CACHE.update(
            {
                "root": str(inbox_root),
                "snapshot": dict(snapshot),
                "last_checked_monotonic": now_monotonic,
            }
        )
    return snapshot


def _build_eta_mu_file_graph_legacy(
    vault_root: Path, *, inbox_snapshot: dict[str, Any] | None = None
) -> dict[str, Any]:
    inbox_state = (
        dict(inbox_snapshot)
        if isinstance(inbox_snapshot, dict)
        else sync_eta_mu_inbox(vault_root)
    )
    entries = _load_eta_mu_knowledge_entries(vault_root)
    entity_lookup = {
        str(e.get("id")): e for e in ENTITY_MANIFEST if str(e.get("id")).strip()
    }
    field_nodes = []
    for fid in CANONICAL_NAMED_FIELD_IDS:
        e = entity_lookup.get(fid)
        if e:
            field_nodes.append(
                {
                    "id": f"field:{fid}",
                    "node_id": fid,
                    "node_type": "field",
                    "label": str(e.get("en", fid)),
                    "x": round(_safe_float(e.get("x", 0.5)), 4),
                    "y": round(_safe_float(e.get("y", 0.5)), 4),
                    "hue": int(_safe_float(e.get("hue", 200))),
                }
            )

    file_nodes, edges, organizer_cluster_rows, terms_by_node = [], [], [], {}
    active_layers = _split_csv_items(ETA_MU_FILE_GRAPH_ACTIVE_EMBED_LAYERS) or ["*"]
    order_layers = _split_csv_items(ETA_MU_FILE_GRAPH_LAYER_ORDER)
    embedding_state = _load_embeddings_db_state(vault_root)

    for idx, entry in enumerate(entries[:ETA_MU_MAX_GRAPH_FILES]):
        eid = str(entry.get("id")).strip()
        if not eid:
            continue
        dom_f = entry.get("dominant_field", "f3")
        dom_p = entry.get(
            "dominant_presence", FIELD_TO_PRESENCE.get(dom_f, "anchor_registry")
        )
        anchor = entity_lookup.get(dom_p, {"x": 0.5, "y": 0.5, "hue": 200})

        seed = sha1(f"{eid}|{idx}".encode()).digest()
        angle = (int.from_bytes(seed[0:2], "big") / 65535.0) * math.tau
        radius = 0.06 + (int.from_bytes(seed[2:4], "big") / 65535.0) * 0.19
        x, y = (
            _clamp01(anchor["x"] + math.cos(angle) * radius),
            _clamp01(anchor["y"] + math.sin(angle) * radius),
        )

        nid = f"file:{sha1(eid.encode()).hexdigest()[:14]}"
        terms_by_node[nid] = _organizer_terms_from_entry(entry)

        file_nodes.append(
            {
                "id": nid,
                "node_id": eid,
                "node_type": "file",
                "name": str(entry.get("name")),
                "label": str(entry.get("name")),
                "kind": str(entry.get("kind", "file")),
                "x": round(x, 4),
                "y": round(y, 4),
                "hue": anchor["hue"],
                "url": entry.get("url"),
                "dominant_field": dom_f,
                "archive_kind": str(entry.get("archive_kind", "")),
                "archive_member_path": str(entry.get("archive_member_path", "")),
                "archive_container_id": str(entry.get("archive_container_id", "")),
            }
        )
        edges.append(
            {
                "id": f"edge:{nid}:{dom_f}",
                "source": nid,
                "target": f"field:{dom_p}",
                "field": dom_f,
                "weight": 1.0,
                "kind": "categorizes",
            }
        )

    concept_presences: list[dict[str, Any]] = []
    concept_by_field: dict[str, str] = {}
    for field_id in sorted(
        {str(node.get("dominant_field", "f3")) for node in file_nodes}
    ):
        concept_id = _concept_presence_id(field_id)
        concept_by_field[field_id] = concept_id
        concept_presences.append(
            {
                "id": concept_id,
                "label": _concept_presence_label([field_id], len(concept_presences)),
                "organized_by": "file_organizer",
            }
        )

    embed_layers: list[dict[str, Any]] = []
    layer_id = "embed-layer:default"
    layer_points = []
    for node in file_nodes:
        node_points = [
            {
                "id": f"point:{node['id']}",
                "x": node.get("x", 0.5),
                "y": node.get("y", 0.5),
                "weight": 1.0,
            }
        ]
        node["embed_layer_points"] = node_points
        node["embed_layer_count"] = len(node_points)
        node["organized_by"] = "file_organizer"
        concept_id = concept_by_field.get(str(node.get("dominant_field", "f3")), "")
        node["concept_presence_id"] = concept_id
        layer_points.extend(node_points)
        if concept_id:
            edges.append(
                {
                    "id": f"edge:{node['id']}:{concept_id}",
                    "source": node["id"],
                    "target": concept_id,
                    "kind": "organized_into",
                    "weight": 1.0,
                }
            )
    if layer_points:
        embed_layers.append(
            {
                "id": layer_id,
                "label": "default",
                "points": layer_points,
            }
        )
    for concept in concept_presences:
        edges.append(
            {
                "id": f"edge:organizer:{concept['id']}",
                "source": f"presence:{FILE_ORGANIZER_PROFILE['id']}",
                "target": concept["id"],
                "kind": "spawns_presence",
                "weight": 1.0,
            }
        )

    archive_count = len(
        [
            node
            for node in file_nodes
            if str(node.get("archive_kind", "")).strip() == "zip"
        ]
    )

    return {
        "record": ETA_MU_FILE_GRAPH_RECORD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inbox": inbox_state,
        "nodes": field_nodes + file_nodes,
        "field_nodes": field_nodes,
        "file_nodes": file_nodes,
        "edges": edges,
        "organizer_presence": {
            "id": f"presence:{FILE_ORGANIZER_PROFILE['id']}",
            "label": FILE_ORGANIZER_PROFILE["en"],
        },
        "concept_presences": concept_presences,
        "embed_layers": embed_layers,
        "stats": {
            "field_count": len(field_nodes),
            "file_count": len(file_nodes),
            "edge_count": len(edges),
            "archive_count": archive_count,
            "embed_layer_count": len(embed_layers),
        },
    }


def build_eta_mu_file_graph(
    vault_root: Path,
    *,
    inbox_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inbox_state = (
        dict(inbox_snapshot)
        if isinstance(inbox_snapshot, dict)
        else sync_eta_mu_inbox(vault_root)
    )
    entries = _load_eta_mu_knowledge_entries(vault_root)
    docmeta_index = _refresh_eta_mu_docmeta_index(vault_root, entries)
    entity_lookup = {
        str(entity.get("id", "")): entity
        for entity in ENTITY_MANIFEST
        if str(entity.get("id", "")).strip()
    }

    field_nodes: list[dict[str, Any]] = []
    for field_id in CANONICAL_NAMED_FIELD_IDS:
        entity = entity_lookup.get(field_id)
        if entity is None:
            continue
        field_nodes.append(
            {
                "id": f"field:{field_id}",
                "node_id": field_id,
                "node_type": "field",
                "field": next(
                    (
                        key
                        for key, presence_id in FIELD_TO_PRESENCE.items()
                        if presence_id == field_id
                    ),
                    "f3",
                ),
                "label": str(entity.get("en", field_id)),
                "label_ja": str(entity.get("ja", "")),
                "x": round(_safe_float(entity.get("x", 0.5), 0.5), 4),
                "y": round(_safe_float(entity.get("y", 0.5), 0.5), 4),
                "hue": int(_safe_float(entity.get("hue", 200), 200.0)),
            }
        )

    file_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    kind_counts: dict[str, int] = defaultdict(int)
    field_counts: dict[str, int] = defaultdict(int)
    archive_count = 0
    compressed_bytes_total = 0
    embedding_state = _load_embeddings_db_state(vault_root)
    active_layer_patterns = _split_csv_items(ETA_MU_FILE_GRAPH_ACTIVE_EMBED_LAYERS)
    if not active_layer_patterns:
        active_layer_patterns = ["*"]
    order_layer_patterns = _split_csv_items(ETA_MU_FILE_GRAPH_LAYER_ORDER)
    layer_summary: dict[str, dict[str, Any]] = {}
    organizer_cluster_rows: list[dict[str, Any]] = []
    organizer_terms_by_node: dict[str, list[str]] = {}
    tag_members: dict[str, set[str]] = defaultdict(set)
    tag_labels: dict[str, str] = {}
    tag_pair_counts: dict[tuple[str, str], int] = defaultdict(int)

    for index, entry in enumerate(entries[:ETA_MU_MAX_GRAPH_FILES]):
        entry_id = str(entry.get("id", "")).strip()
        if not entry_id:
            continue
        dominant_field = str(entry.get("dominant_field", "f3")).strip() or "f3"
        dominant_presence = str(
            entry.get("dominant_presence", "anchor_registry")
        ).strip()
        if dominant_presence not in entity_lookup:
            dominant_presence = FIELD_TO_PRESENCE.get(dominant_field, "anchor_registry")
        anchor = entity_lookup.get(dominant_presence, {"x": 0.5, "y": 0.5, "hue": 200})

        seed = sha1(f"{entry_id}|{index}".encode("utf-8")).digest()
        angle = (int.from_bytes(seed[0:2], "big") / 65535.0) * math.tau
        radius = 0.06 + (int.from_bytes(seed[2:4], "big") / 65535.0) * 0.19
        jitter_x = ((seed[4] / 255.0) - 0.5) * 0.05
        jitter_y = ((seed[5] / 255.0) - 0.5) * 0.05
        x = _clamp01(
            _safe_float(anchor.get("x", 0.5), 0.5) + math.cos(angle) * radius + jitter_x
        )
        y = _clamp01(
            _safe_float(anchor.get("y", 0.5), 0.5) + math.sin(angle) * radius + jitter_y
        )
        hue = int(_safe_float(anchor.get("hue", 200), 200.0))
        file_bytes = int(_safe_float(entry.get("bytes", 0), 0.0))
        importance = _clamp01(min(1.0, math.log2(max(2, file_bytes + 1)) / 20.0))
        archive_kind = str(entry.get("archive_kind", "")).strip().lower()
        archive_rel_path = str(
            entry.get("archive_rel_path", entry.get("archived_rel_path", ""))
        )
        archive_member_path = str(entry.get("archive_member_path", ""))
        archive_manifest_path = str(entry.get("archive_manifest_path", ""))
        archive_container_id = str(entry.get("archive_container_id", ""))
        archive_bytes = int(_safe_float(entry.get("archive_bytes", 0), 0.0))
        embedding_links_raw = entry.get("embedding_links", [])
        embedding_links = (
            [dict(item) for item in embedding_links_raw if isinstance(item, dict)]
            if isinstance(embedding_links_raw, list)
            else []
        )

        node_id = f"file:{sha1(entry_id.encode('utf-8')).hexdigest()[:14]}"
        entry_collection = (
            str(
                entry.get(
                    "vecstore_collection",
                    ETA_MU_INGEST_VECSTORE_COLLECTION,
                )
            ).strip()
            or ETA_MU_INGEST_VECSTORE_COLLECTION
        )
        node_layer_groups: dict[str, dict[str, Any]] = {}
        for link in embedding_links:
            if str(link.get("kind", "")).strip().lower() != "embed_ref":
                continue
            layer_identity = _eta_mu_embed_layer_identity(
                collection=str(link.get("vecstore_collection", entry_collection)),
                space_id=str(link.get("space_id", entry.get("space_id", ""))),
                space_signature=str(
                    link.get("space_signature", entry.get("space_signature", ""))
                ),
                model_name=str(link.get("model_name", "")),
            )
            layer_id = str(layer_identity.get("id", "")).strip()
            if not layer_id:
                continue
            group = node_layer_groups.setdefault(
                layer_id,
                {
                    **layer_identity,
                    "embed_ids": [],
                    "reference_count": 0,
                },
            )
            embed_id = str(link.get("embed_id", "")).strip()
            if embed_id:
                group["embed_ids"].append(embed_id)
            group["reference_count"] = int(group.get("reference_count", 0)) + 1

        for layer in node_layer_groups.values():
            layer["active"] = _eta_mu_embed_layer_is_active(
                layer, active_layer_patterns
            )
            layer_id = str(layer.get("id", "")).strip()
            if not layer_id:
                continue
            summary = layer_summary.setdefault(
                layer_id,
                {
                    "id": layer_id,
                    "key": str(layer.get("key", "")),
                    "label": str(layer.get("label", "")),
                    "collection": str(layer.get("collection", "")),
                    "space_id": str(layer.get("space_id", "")),
                    "space_signature": str(layer.get("space_signature", "")),
                    "model_name": str(layer.get("model_name", "")),
                    "reference_count": 0,
                    "_node_ids": set(),
                },
            )
            summary["reference_count"] = int(summary.get("reference_count", 0)) + int(
                layer.get("reference_count", 0)
            )
            node_ids = summary.get("_node_ids")
            if not isinstance(node_ids, set):
                node_ids = set()
                summary["_node_ids"] = node_ids
            node_ids.add(node_id)

        ordered_layers = sorted(
            node_layer_groups.values(),
            key=lambda row: (
                0 if bool(row.get("active", False)) else 1,
                _eta_mu_embed_layer_order_index(row, order_layer_patterns),
                -int(row.get("reference_count", 0)),
                str(row.get("label", "")),
            ),
        )

        node_embed_vectors: list[list[float]] = []
        embed_layer_points: list[dict[str, Any]] = []
        for layer_index, layer in enumerate(
            ordered_layers[:ETA_MU_FILE_GRAPH_LAYER_POINT_LIMIT]
        ):
            embed_vectors: list[list[float]] = []
            for embed_id in layer.get("embed_ids", []):
                record = embedding_state.get(str(embed_id), {})
                if not isinstance(record, dict):
                    continue
                vector = _normalize_embedding_vector(record.get("embedding", []))
                if vector is not None:
                    embed_vectors.append(vector)
                    node_embed_vectors.append(vector)

            semantic_vector = _average_embedding_vectors(embed_vectors)
            semantic_xy = (
                _semantic_xy_from_embedding(semantic_vector)
                if semantic_vector is not None
                else None
            )
            layer_seed = str(layer.get("key", "")).strip() or (
                f"{node_id}|{layer_index}"
            )
            point_x, point_y, source = _eta_mu_embed_layer_point(
                node_x=x,
                node_y=y,
                layer_seed=layer_seed,
                index=layer_index,
                semantic_xy=semantic_xy,
            )
            hue_seed = int(_safe_float(hue, 200.0))
            layer_hue = int((hue_seed + int(_stable_ratio(layer_seed, 17) * 180)) % 360)
            embed_layer_points.append(
                {
                    "id": str(layer.get("id", "")),
                    "key": str(layer.get("key", "")),
                    "label": str(layer.get("label", "")),
                    "collection": str(layer.get("collection", "")),
                    "space_id": str(layer.get("space_id", "")),
                    "space_signature": str(layer.get("space_signature", "")),
                    "model_name": str(layer.get("model_name", "")),
                    "x": point_x,
                    "y": point_y,
                    "hue": layer_hue,
                    "active": bool(layer.get("active", False)),
                    "source": source,
                    "reference_count": int(layer.get("reference_count", 0)),
                    "embed_ids": [str(item) for item in layer.get("embed_ids", [])],
                }
            )

        if not embed_layer_points and archive_kind == "zip":
            fallback_layer_id = "embed-layer:default"
            fallback_seed = f"{node_id}|fallback"
            fallback_x, fallback_y, fallback_source = _eta_mu_embed_layer_point(
                node_x=x,
                node_y=y,
                layer_seed=fallback_seed,
                index=0,
                semantic_xy=None,
            )
            embed_layer_points.append(
                {
                    "id": fallback_layer_id,
                    "key": "default",
                    "label": "default",
                    "collection": entry_collection,
                    "space_id": str(entry.get("space_id", "")),
                    "space_signature": str(entry.get("space_signature", "")),
                    "model_name": str(entry.get("model_name", "")),
                    "x": fallback_x,
                    "y": fallback_y,
                    "hue": hue,
                    "active": True,
                    "source": fallback_source,
                    "reference_count": 1,
                    "embed_ids": [],
                }
            )

        organizer_terms_by_node[node_id] = _organizer_terms_from_entry(entry)
        node_mean_vector = _average_embedding_vectors(node_embed_vectors)
        if node_mean_vector is not None:
            organizer_cluster_rows.append(
                {
                    "node_id": node_id,
                    "vector": node_mean_vector,
                }
            )

        docmeta_row = _docmeta_for_entry(entry, docmeta_index)
        docmeta_summary = str(docmeta_row.get("summary", "")).strip()
        if not docmeta_summary:
            docmeta_summary = _heuristic_docmeta_summary(
                entry,
                str(entry.get("text_excerpt", "")),
            )
        docmeta_tags = _dedupe_docmeta_tags(
            docmeta_row.get("tags", []),
            limit=ETA_MU_DOCMETA_TAG_LIMIT,
        )
        if not docmeta_tags:
            docmeta_tags = _heuristic_docmeta_tags(entry, docmeta_summary)
        docmeta_labels = [
            label
            for label in [_docmeta_label_from_tag(tag) for tag in docmeta_tags]
            if label
        ]

        file_nodes.append(
            {
                "id": node_id,
                "node_id": entry_id,
                "node_type": "file",
                "name": str(entry.get("name", "")),
                "label": str(entry.get("name", "")),
                "kind": str(entry.get("kind", "file")),
                "x": round(x, 4),
                "y": round(y, 4),
                "hue": hue,
                "importance": round(importance, 4),
                "source_rel_path": str(entry.get("source_rel_path", "")),
                "archived_rel_path": str(entry.get("archived_rel_path", "")),
                "archive_rel_path": archive_rel_path,
                "archive_kind": archive_kind,
                "archive_url": str(entry.get("archive_url", "")),
                "archive_member_path": archive_member_path,
                "archive_manifest_path": archive_manifest_path,
                "archive_manifest_url": str(entry.get("archive_manifest_url", "")),
                "archive_bytes": archive_bytes,
                "archive_container_id": archive_container_id,
                "embedding_links": embedding_links,
                "embed_layer_points": embed_layer_points,
                "embed_layer_count": (
                    max(1, len(node_layer_groups))
                    if archive_kind == "zip"
                    else len(node_layer_groups)
                ),
                "vecstore_collection": entry_collection,
                "url": str(entry.get("url", "")),
                "dominant_field": dominant_field,
                "dominant_presence": dominant_presence,
                "field_scores": dict(entry.get("field_scores", {})),
                "text_excerpt": str(entry.get("text_excerpt", "")),
                "summary": docmeta_summary,
                "tags": docmeta_tags,
                "labels": docmeta_labels,
            }
        )

        bounded_tags = docmeta_tags[: max(1, int(ETA_MU_DOCMETA_TAG_LIMIT))]
        if bounded_tags:
            unique_tags = sorted(dict.fromkeys(bounded_tags))
            for tag_index, tag_token in enumerate(unique_tags):
                if (
                    len(tag_members) >= ETA_MU_FILE_GRAPH_TAG_LIMIT
                    and tag_token not in tag_members
                ):
                    continue
                tag_members[tag_token].add(node_id)
                if tag_token not in tag_labels:
                    fallback_label = (
                        docmeta_labels[tag_index]
                        if tag_index < len(docmeta_labels)
                        else _docmeta_label_from_tag(tag_token)
                    )
                    tag_labels[tag_token] = fallback_label
            for left_index in range(len(unique_tags)):
                for right_index in range(left_index + 1, len(unique_tags)):
                    left_tag = unique_tags[left_index]
                    right_tag = unique_tags[right_index]
                    pair: tuple[str, str] = (
                        (left_tag, right_tag)
                        if left_tag <= right_tag
                        else (right_tag, left_tag)
                    )
                    tag_pair_counts[pair] += 1

        kind_counts[str(entry.get("kind", "file"))] += 1
        field_counts[dominant_field] += 1
        if archive_kind == "zip":
            archive_count += 1
            compressed_bytes_total += archive_bytes

        field_scores = entry.get("field_scores", {})
        if not isinstance(field_scores, dict):
            field_scores = {}
        ranked = sorted(
            [
                (str(field), _safe_float(weight, 0.0))
                for field, weight in field_scores.items()
            ],
            key=lambda row: row[1],
            reverse=True,
        )
        if not ranked:
            ranked = [(dominant_field, 1.0)]

        for edge_index, (field_id, weight) in enumerate(ranked[:2]):
            if weight <= 0.0:
                continue
            target_presence = FIELD_TO_PRESENCE.get(field_id, dominant_presence)
            if target_presence not in entity_lookup:
                continue
            edges.append(
                {
                    "id": f"edge:{node_id}:{field_id}:{edge_index}",
                    "source": node_id,
                    "target": f"field:{target_presence}",
                    "field": field_id,
                    "weight": round(_clamp01(weight), 4),
                    "kind": "categorizes",
                }
            )

    organizer_node_id = f"presence:{FILE_ORGANIZER_PROFILE['id']}"
    organizer_presence = {
        "id": organizer_node_id,
        "node_id": FILE_ORGANIZER_PROFILE["id"],
        "node_type": "presence",
        "presence_kind": "organizer",
        "field": "f3",
        "label": FILE_ORGANIZER_PROFILE["en"],
        "label_ja": FILE_ORGANIZER_PROFILE["ja"],
        "x": 0.5,
        "y": 0.12,
        "hue": 46,
        "created_count": 0,
    }
    field_nodes.append(dict(organizer_presence))

    concept_presences: list[dict[str, Any]] = []
    concept_assignments: dict[str, str] = {}
    file_node_lookup = {
        str(node.get("id", "")): node for node in file_nodes if isinstance(node, dict)
    }
    if organizer_cluster_rows:
        _cluster_assignments, cluster_rows = _assign_meaning_clusters(
            organizer_cluster_rows,
            threshold=ETA_MU_FILE_GRAPH_ORGANIZER_CLUSTER_THRESHOLD,
        )
        for concept_index, cluster in enumerate(cluster_rows):
            members = [
                str(item)
                for item in cluster.get("members", [])
                if str(item).strip() and str(item) in file_node_lookup
            ]
            members = sorted(dict.fromkeys(members))
            if len(members) < ETA_MU_FILE_GRAPH_ORGANIZER_MIN_GROUP_SIZE:
                continue
            if len(concept_presences) >= ETA_MU_FILE_GRAPH_ORGANIZER_MAX_CONCEPTS:
                break

            cluster_id = (
                str(cluster.get("id", "")).strip() or f"concept-{concept_index + 1}"
            )
            presence_id = _concept_presence_id(cluster_id)
            term_counts: dict[str, int] = defaultdict(int)
            for member in members:
                for term in organizer_terms_by_node.get(member, []):
                    term_counts[term] += 1
            sorted_terms = [
                term
                for term, _ in sorted(
                    term_counts.items(),
                    key=lambda row: (-int(row[1]), str(row[0])),
                )
            ]
            concept_terms = sorted_terms[:ETA_MU_FILE_GRAPH_ORGANIZER_TERMS_PER_CONCEPT]
            label_en = _concept_presence_label(concept_terms, concept_index)
            label_ja = (
                f"概念: {' / '.join(concept_terms[:2])}"
                if concept_terms
                else f"概念グループ {concept_index + 1}"
            )

            centroid_x = _clamp01(
                sum(
                    _safe_float(file_node_lookup[member].get("x", 0.5), 0.5)
                    for member in members
                )
                / max(1, len(members))
            )
            centroid_y = _clamp01(
                sum(
                    _safe_float(file_node_lookup[member].get("y", 0.5), 0.5)
                    for member in members
                )
                / max(1, len(members))
            )
            orbit = 0.03 + (_stable_ratio(presence_id, concept_index) * 0.08)
            angle = _stable_ratio(presence_id, concept_index + 13) * math.tau
            concept_x = _clamp01(centroid_x + math.cos(angle) * orbit)
            concept_y = _clamp01(centroid_y + math.sin(angle) * orbit)
            concept_hue = int(24 + (_stable_ratio(presence_id, 7) * 300)) % 360

            concept_presence = {
                "id": presence_id,
                "cluster_id": cluster_id,
                "label": label_en,
                "label_ja": label_ja,
                "terms": concept_terms,
                "cohesion": round(
                    _clamp01(_safe_float(cluster.get("cohesion", 0.0), 0.0)), 4
                ),
                "file_count": len(members),
                "members": members,
                "x": round(concept_x, 4),
                "y": round(concept_y, 4),
                "hue": concept_hue,
                "created_by": FILE_ORGANIZER_PROFILE["id"],
            }
            concept_presences.append(concept_presence)
            field_nodes.append(
                {
                    "id": presence_id,
                    "node_id": presence_id,
                    "node_type": "presence",
                    "presence_kind": "concept",
                    "field": "f6",
                    "label": label_en,
                    "label_ja": label_ja,
                    "x": round(concept_x, 4),
                    "y": round(concept_y, 4),
                    "hue": concept_hue,
                    "cohesion": concept_presence["cohesion"],
                    "member_count": len(members),
                }
            )

            spawn_edge_seed = f"{organizer_node_id}|{presence_id}|spawn"
            edges.append(
                {
                    "id": "edge:"
                    + hashlib.sha1(spawn_edge_seed.encode("utf-8")).hexdigest()[:16],
                    "source": organizer_node_id,
                    "target": presence_id,
                    "field": "f3",
                    "weight": round(
                        _clamp01(_safe_float(cluster.get("cohesion", 0.0), 0.0)),
                        4,
                    ),
                    "kind": "spawns_presence",
                }
            )

            for member in members:
                concept_assignments[member] = presence_id
                member_node = file_node_lookup.get(member)
                if isinstance(member_node, dict):
                    member_node["concept_presence_id"] = presence_id
                    member_node["concept_presence_label"] = label_en
                    member_node["organized_by"] = FILE_ORGANIZER_PROFILE["id"]
                organize_edge_seed = f"{member}|{presence_id}|organize"
                edges.append(
                    {
                        "id": "edge:"
                        + hashlib.sha1(organize_edge_seed.encode("utf-8")).hexdigest()[
                            :16
                        ],
                        "source": member,
                        "target": presence_id,
                        "field": "f6",
                        "weight": round(
                            _clamp01(_safe_float(cluster.get("cohesion", 0.0), 0.0)),
                            4,
                        ),
                        "kind": "organized_by_presence",
                    }
                )

        organizer_presence["created_count"] = len(concept_presences)
        for node in field_nodes:
            if isinstance(node, dict) and str(node.get("id", "")) == organizer_node_id:
                node["created_count"] = len(concept_presences)
                break

    if not concept_presences:
        fallback_concepts: dict[str, dict[str, Any]] = {}
        for file_node in file_nodes:
            if not isinstance(file_node, dict):
                continue
            member_id = str(file_node.get("id", "")).strip()
            if not member_id:
                continue
            field_id = str(file_node.get("dominant_field", "f3")).strip() or "f3"
            concept_id = _concept_presence_id(field_id)
            concept = fallback_concepts.get(concept_id)
            if concept is None:
                label_en = _concept_presence_label([field_id], len(fallback_concepts))
                concept = {
                    "id": concept_id,
                    "cluster_id": field_id,
                    "label": label_en,
                    "label_ja": f"概念: {field_id}",
                    "terms": [field_id],
                    "cohesion": 1.0,
                    "file_count": 0,
                    "members": [],
                    "x": round(_safe_float(file_node.get("x", 0.5), 0.5), 4),
                    "y": round(_safe_float(file_node.get("y", 0.5), 0.5), 4),
                    "hue": int(_safe_float(file_node.get("hue", 200), 200.0)),
                    "created_by": FILE_ORGANIZER_PROFILE["id"],
                }
                fallback_concepts[concept_id] = concept
                concept_presences.append(concept)
                field_nodes.append(
                    {
                        "id": concept_id,
                        "node_id": concept_id,
                        "node_type": "presence",
                        "presence_kind": "concept",
                        "field": "f6",
                        "label": concept["label"],
                        "label_ja": concept["label_ja"],
                        "x": concept["x"],
                        "y": concept["y"],
                        "hue": concept["hue"],
                        "cohesion": concept["cohesion"],
                        "member_count": 0,
                    }
                )
                spawn_edge_seed = f"{organizer_node_id}|{concept_id}|spawn-fallback"
                edges.append(
                    {
                        "id": "edge:"
                        + hashlib.sha1(spawn_edge_seed.encode("utf-8")).hexdigest()[
                            :16
                        ],
                        "source": organizer_node_id,
                        "target": concept_id,
                        "field": "f3",
                        "weight": 1.0,
                        "kind": "spawns_presence",
                    }
                )

            concept["members"].append(member_id)
            concept["file_count"] = int(concept.get("file_count", 0)) + 1
            concept_assignments[member_id] = concept_id
            file_node["concept_presence_id"] = concept_id
            file_node["concept_presence_label"] = str(concept.get("label", ""))
            file_node["organized_by"] = FILE_ORGANIZER_PROFILE["id"]
            organize_edge_seed = f"{member_id}|{concept_id}|organize-fallback"
            edges.append(
                {
                    "id": "edge:"
                    + hashlib.sha1(organize_edge_seed.encode("utf-8")).hexdigest()[:16],
                    "source": member_id,
                    "target": concept_id,
                    "field": "f6",
                    "weight": 1.0,
                    "kind": "organized_by_presence",
                }
            )

        member_counts = {
            str(concept.get("id", "")): int(concept.get("file_count", 0))
            for concept in concept_presences
            if isinstance(concept, dict)
        }
        for node in field_nodes:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", ""))
            if node_id == organizer_node_id:
                node["created_count"] = len(concept_presences)
            if node_id in member_counts:
                node["member_count"] = member_counts[node_id]
        organizer_presence["created_count"] = len(concept_presences)

    tag_nodes: list[dict[str, Any]] = []
    tag_node_ids: dict[str, str] = {}
    tag_edge_count = 0
    tag_pair_edge_count = 0
    ranked_tags = sorted(tag_members.items(), key=lambda row: (-len(row[1]), row[0]))
    for tag_token, members in ranked_tags[:ETA_MU_FILE_GRAPH_TAG_LIMIT]:
        member_ids = sorted(
            {
                str(member_id)
                for member_id in members
                if str(member_id).strip() and str(member_id) in file_node_lookup
            }
        )
        if not member_ids:
            continue
        node_id = f"tag:{sha1(tag_token.encode('utf-8')).hexdigest()[:14]}"
        centroid_x = _clamp01(
            sum(
                _safe_float(file_node_lookup[member_id].get("x", 0.5), 0.5)
                for member_id in member_ids
            )
            / max(1, len(member_ids))
        )
        centroid_y = _clamp01(
            sum(
                _safe_float(file_node_lookup[member_id].get("y", 0.5), 0.5)
                for member_id in member_ids
            )
            / max(1, len(member_ids))
        )
        orbit = 0.024 + (_stable_ratio(tag_token, len(member_ids)) * 0.06)
        angle = _stable_ratio(tag_token, 91) * math.tau
        tag_x = _clamp01(centroid_x + math.cos(angle) * orbit)
        tag_y = _clamp01(centroid_y + math.sin(angle) * orbit)
        tag_hue = int(18 + (_stable_ratio(tag_token, 53) * 300)) % 360
        label = str(tag_labels.get(tag_token, "")).strip() or _docmeta_label_from_tag(
            tag_token
        )
        tag_nodes.append(
            {
                "id": node_id,
                "node_id": tag_token,
                "node_type": "tag",
                "tag": tag_token,
                "label": label,
                "x": round(tag_x, 4),
                "y": round(tag_y, 4),
                "hue": tag_hue,
                "member_count": len(member_ids),
            }
        )
        tag_node_ids[tag_token] = node_id

        for member_index, member_id in enumerate(member_ids):
            if tag_edge_count >= ETA_MU_FILE_GRAPH_TAG_EDGE_LIMIT:
                break
            relation_weight = round(
                _clamp01(0.48 + min(0.46, len(member_ids) / 18.0)),
                4,
            )
            edge_seed = f"{member_id}|{tag_token}|tag:{member_index}"
            edges.append(
                {
                    "id": "edge:"
                    + hashlib.sha1(edge_seed.encode("utf-8")).hexdigest()[:16],
                    "source": member_id,
                    "target": node_id,
                    "field": "f6",
                    "weight": relation_weight,
                    "kind": "labeled_as",
                }
            )
            tag_edge_count += 1

    sorted_tag_pairs = sorted(
        tag_pair_counts.items(),
        key=lambda row: (-int(row[1]), row[0][0], row[0][1]),
    )
    for (left_tag, right_tag), pair_count in sorted_tag_pairs:
        if tag_pair_edge_count >= ETA_MU_FILE_GRAPH_TAG_PAIR_EDGE_LIMIT:
            break
        source_id = tag_node_ids.get(left_tag)
        target_id = tag_node_ids.get(right_tag)
        if not source_id or not target_id:
            continue
        edge_seed = f"{left_tag}|{right_tag}|pair"
        edges.append(
            {
                "id": "edge:"
                + hashlib.sha1(edge_seed.encode("utf-8")).hexdigest()[:16],
                "source": source_id,
                "target": target_id,
                "field": "f6",
                "weight": round(_clamp01(0.16 + min(0.84, pair_count / 8.0)), 4),
                "kind": "relates_tag",
            }
        )
        tag_pair_edge_count += 1

    if not layer_summary and file_nodes:
        fallback_node_ids = {
            str(node.get("id", ""))
            for node in file_nodes
            if isinstance(node, dict) and str(node.get("id", "")).strip()
        }
        layer_summary["embed-layer:default"] = {
            "id": "embed-layer:default",
            "key": "default",
            "label": "default",
            "collection": ETA_MU_INGEST_VECSTORE_COLLECTION,
            "space_id": "",
            "space_signature": "",
            "model_name": "",
            "reference_count": len(fallback_node_ids),
            "_node_ids": fallback_node_ids,
        }

    embed_layers: list[dict[str, Any]] = []
    for summary in layer_summary.values():
        node_ids = summary.get("_node_ids")
        if not isinstance(node_ids, set):
            node_ids = set()
        row = {
            "id": str(summary.get("id", "")),
            "key": str(summary.get("key", "")),
            "label": str(summary.get("label", "")),
            "collection": str(summary.get("collection", "")),
            "space_id": str(summary.get("space_id", "")),
            "space_signature": str(summary.get("space_signature", "")),
            "model_name": str(summary.get("model_name", "")),
            "file_count": len(node_ids),
            "reference_count": int(summary.get("reference_count", 0)),
        }
        row["active"] = _eta_mu_embed_layer_is_active(row, active_layer_patterns)
        embed_layers.append(row)

    embed_layers.sort(
        key=lambda row: (
            0 if bool(row.get("active", False)) else 1,
            _eta_mu_embed_layer_order_index(row, order_layer_patterns),
            -int(row.get("file_count", 0)),
            -int(row.get("reference_count", 0)),
            str(row.get("label", "")),
        )
    )
    embed_layers = embed_layers[:ETA_MU_FILE_GRAPH_LAYER_LIMIT]
    embed_layer_active_count = sum(
        1 for row in embed_layers if bool(row.get("active", False))
    )

    nodes = [*field_nodes, *tag_nodes, *file_nodes]
    return {
        "record": ETA_MU_FILE_GRAPH_RECORD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inbox": inbox_state,
        "nodes": nodes,
        "field_nodes": field_nodes,
        "tag_nodes": tag_nodes,
        "file_nodes": file_nodes,
        "embed_layers": embed_layers,
        "organizer_presence": organizer_presence,
        "concept_presences": concept_presences,
        "edges": edges,
        "stats": {
            "field_count": len(field_nodes),
            "file_count": len(file_nodes),
            "edge_count": len(edges),
            "kind_counts": dict(kind_counts),
            "field_counts": dict(field_counts),
            "embed_layer_count": len(embed_layers),
            "embed_layer_active_count": embed_layer_active_count,
            "organizer_presence_count": 1,
            "concept_presence_count": len(concept_presences),
            "organized_file_count": len(concept_assignments),
            "tag_count": len(tag_nodes),
            "tag_edge_count": tag_edge_count,
            "tag_pair_edge_count": tag_pair_edge_count,
            "docmeta_enriched_count": len(
                [
                    node
                    for node in file_nodes
                    if str(node.get("summary", "")).strip()
                    and len(node.get("tags", [])) > 0
                ]
            ),
            "knowledge_entries": len(entries),
            "archive_count": archive_count,
            "compressed_bytes_total": compressed_bytes_total,
        },
    }


def _eta_mu_inbox_snapshot_without_sync(vault_root: Path) -> dict[str, Any]:
    inbox_root = _eta_mu_inbox_root(vault_root)
    with _ETA_MU_INBOX_LOCK:
        if str(_ETA_MU_INBOX_CACHE.get("root", "")) == str(inbox_root):
            cached_snapshot = _ETA_MU_INBOX_CACHE.get("snapshot")
            if isinstance(cached_snapshot, dict):
                return dict(cached_snapshot)

    spaces = _eta_mu_space_forms()
    return {
        "record": "ημ.inbox.v1",
        "path": str(inbox_root),
        "pending_count": 0,
        "processed_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "rejected_count": 0,
        "deferred_count": 0,
        "is_empty": True,
        "knowledge_entries": len(_load_eta_mu_knowledge_entries(vault_root)),
        "registry_entries": len(_load_eta_mu_registry_entries(vault_root)),
        "last_ingested_at": "",
        "contract": ETA_MU_INGEST_CONTRACT_ID,
        "spaces": {
            "text": {
                "id": spaces["text"]["id"],
                "signature": spaces["text"]["signature"],
                "collection": spaces["text"].get("collection", ""),
            },
            "image": {
                "id": spaces["image"]["id"],
                "signature": spaces["image"]["signature"],
                "collection": spaces["image"].get("collection", ""),
            },
            "space_set": {
                "id": spaces["space_set"]["id"],
                "signature": spaces["space_set"]["signature"],
            },
            "vecstore": {
                "id": spaces["vecstore"]["id"],
                "collection": spaces["vecstore"].get("collection", ""),
                "layer_mode": spaces["vecstore"].get("layer_mode", "single"),
            },
        },
        "errors": [],
        "sync_status": "skipped",
    }


def _catalog_rel_path(path: Path, *, vault_root: Path, part_root: Path) -> str:
    resolved_path = path.resolve()
    try:
        rel_path = resolved_path.relative_to(vault_root.resolve())
        return str(rel_path).replace("\\", "/")
    except ValueError:
        pass

    try:
        rel_path = resolved_path.relative_to(part_root.resolve())
        return str(rel_path).replace("\\", "/")
    except ValueError:
        return resolved_path.name


def _world_log_fallback_payload(
    *,
    limit: int,
    pending_inbox: int = 0,
    error: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "record": "ημ.world-log.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": 0,
        "limit": int(limit),
        "pending_inbox": max(0, int(pending_inbox)),
        "sources": {},
        "kinds": {},
        "relation_count": 0,
        "events": [],
    }
    if error:
        payload["error"] = error
    return payload


def collect_catalog(
    part_root: Path,
    vault_root: Path,
    *,
    sync_inbox: bool = True,
    include_pi_archive: bool = True,
    include_world_log: bool = True,
) -> dict[str, Any]:
    from .simulation import (
        build_simulation_state,
        build_weaver_field_graph,
        _load_test_signal_artifacts,
        _build_logical_graph,
        _build_pain_field,
        _materialize_heat_values,
        build_named_field_overlays,
    )
    from .chamber import (
        collect_promptdb_packets,
        build_truth_binding_state,
        build_pi_archive_payload,
        build_world_log_payload,
    )

    if sync_inbox:
        try:
            inbox_snapshot = sync_eta_mu_inbox(vault_root)
        except Exception as exc:
            inbox_snapshot = _eta_mu_inbox_snapshot_without_sync(vault_root)
            inbox_errors = list(inbox_snapshot.get("errors", []))
            inbox_errors.append(
                {
                    "path": str(_eta_mu_inbox_root(vault_root)),
                    "error": f"sync_eta_mu_inbox_failed:{exc.__class__.__name__}",
                }
            )
            inbox_snapshot["errors"] = inbox_errors
            inbox_snapshot["sync_status"] = "failed"
    else:
        inbox_snapshot = _eta_mu_inbox_snapshot_without_sync(vault_root)
    file_graph = build_eta_mu_file_graph(vault_root, inbox_snapshot=inbox_snapshot)

    items, counts, seen = [], defaultdict(int), set()
    for root in discover_part_roots(vault_root, part_root):
        manifest = load_manifest(root)
        for item in _manifest_entries(manifest):
            path = (root / str(item.get("path", ""))).resolve()
            if not path.exists() or not path.is_file():
                continue
            rel = _catalog_rel_path(path, vault_root=vault_root, part_root=root)
            if rel in seen:
                continue
            seen.add(rel)
            kind = classify_kind(path)
            counts[kind] += 1
            items.append(
                {
                    "part": _part_label(root, manifest),
                    "name": path.name,
                    "role": str(item.get("role", "unknown")),
                    "display_name": build_display_name(item, path),
                    "display_role": build_display_role(
                        str(item.get("role", "unknown"))
                    ),
                    "kind": kind,
                    "bytes": path.stat().st_size,
                    "mtime_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "rel_path": rel,
                    "url": "/library/" + quote(rel),
                }
            )

    items.sort(key=lambda row: (row["part"], row["kind"], row["name"]), reverse=True)
    cover_fields = [
        {
            "id": item["rel_path"],
            "part": item["part"],
            "display_name": item["display_name"],
            "display_role": item["display_role"],
            "url": item["url"],
            "seed": sha1(item["rel_path"].encode("utf-8")).hexdigest(),
        }
        for item in items
        if item.get("role") == "cover_art"
    ]
    promptdb_index = collect_promptdb_packets(vault_root)
    test_failures, test_coverage = _load_test_signal_artifacts(part_root, vault_root)
    if include_world_log:
        try:
            world_log_payload = build_world_log_payload(
                part_root,
                vault_root,
                limit=WORLD_LOG_EVENT_LIMIT,
            )
        except Exception as exc:
            world_log_payload = _world_log_fallback_payload(
                limit=WORLD_LOG_EVENT_LIMIT,
                pending_inbox=int(inbox_snapshot.get("pending_count", 0) or 0),
                error=f"world_log_unavailable:{exc.__class__.__name__}",
            )
    else:
        world_log_payload = _world_log_fallback_payload(
            limit=WORLD_LOG_EVENT_LIMIT,
            pending_inbox=int(inbox_snapshot.get("pending_count", 0) or 0),
            error="world_log_deferred:runtime_fast_path",
        )
    catalog = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "part_roots": [str(p) for p in discover_part_roots(vault_root, part_root)],
        "counts": dict(counts),
        "canonical_terms": [{"en": en, "ja": ja} for en, ja in CANONICAL_TERMS],
        "entity_manifest": ENTITY_MANIFEST,
        "named_fields": build_named_field_overlays(ENTITY_MANIFEST),
        "ui_default_perspective": PROJECTION_DEFAULT_PERSPECTIVE,
        "ui_perspectives": projection_perspective_options(),
        "cover_fields": cover_fields,
        "eta_mu_inbox": inbox_snapshot,
        "file_graph": file_graph,
        "crawler_graph": build_weaver_field_graph(part_root, vault_root),
        "truth_state": build_truth_binding_state(
            part_root, vault_root, promptdb_index=promptdb_index
        ),
        "test_failures": test_failures,
        "test_coverage": test_coverage,
        "promptdb": promptdb_index,
        "world_log": world_log_payload,
        "items": items,
    }
    catalog["logical_graph"] = _build_logical_graph(catalog)
    catalog["pain_field"] = _build_pain_field(catalog, catalog["logical_graph"])
    catalog["heat_values"] = _materialize_heat_values(catalog, catalog["pain_field"])
    if include_pi_archive:
        pi_archive = build_pi_archive_payload(part_root, vault_root, catalog=catalog)
        catalog["pi_archive"] = {
            "record": "ημ.pi-archive.v1",
            "generated_at": pi_archive.get("generated_at", ""),
            "hash": pi_archive.get("hash", {}),
            "signature": pi_archive.get("signature", {}),
            "portable": pi_archive.get("portable", {}),
            "ledger_count": int((pi_archive.get("ledger") or {}).get("count", 0) or 0),
        }
    else:
        catalog["pi_archive"] = {
            "record": "ημ.pi-archive.v1",
            "generated_at": "",
            "hash": {},
            "signature": {},
            "portable": {},
            "ledger_count": 0,
            "status": "deferred",
        }
    return catalog


def collect_zip_catalog(
    part_root: Path, vault_root: Path, *, member_limit: int = 220
) -> dict[str, Any]:
    safe_member_limit = max(40, min(1200, int(member_limit or 220)))
    zip_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for root in discover_part_roots(vault_root, part_root):
        for zip_path in root.rglob("*.zip"):
            if not zip_path.is_file():
                continue
            resolved = zip_path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            zip_paths.append(resolved)

    zip_paths.sort(
        key=lambda path: (path.stat().st_mtime if path.exists() else 0.0, str(path)),
        reverse=True,
    )
    zips: list[dict[str, Any]] = []
    for zip_path in zip_paths:
        rel_path = _safe_rel_path(zip_path, vault_root)
        archive_url = "/library/" + quote(rel_path)
        archive_id = sha1(rel_path.encode("utf-8")).hexdigest()[:16]
        try:
            stat = zip_path.stat()
            mtime_utc = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()
            archive_bytes = int(stat.st_size)
        except OSError:
            mtime_utc = ""
            archive_bytes = 0

        try:
            with zipfile.ZipFile(zip_path, "r") as archive_zip:
                infos = archive_zip.infolist()
        except (OSError, ValueError, zipfile.BadZipFile) as err:
            zips.append(
                {
                    "id": f"zip:{archive_id}",
                    "name": zip_path.name,
                    "rel_path": rel_path,
                    "url": archive_url,
                    "bytes": archive_bytes,
                    "mtime_utc": mtime_utc,
                    "error": str(err),
                    "members_total": 0,
                    "files_total": 0,
                    "dirs_total": 0,
                    "uncompressed_bytes_total": 0,
                    "compressed_bytes_total": 0,
                    "compression_ratio": 0.0,
                    "members_truncated": False,
                    "type_counts": {},
                    "extension_counts": [],
                    "top_level_entries": [],
                    "members": [],
                }
            )
            continue

        members: list[dict[str, Any]] = []
        type_counts: dict[str, int] = defaultdict(int)
        extension_counts: dict[str, int] = defaultdict(int)
        top_level_counts: dict[str, int] = defaultdict(int)
        files_total = 0
        dirs_total = 0
        uncompressed_total = 0
        compressed_total = 0

        for info in infos:
            raw_name = str(info.filename or "")
            normalized = _normalize_archive_member_path(raw_name)
            if normalized is None:
                continue
            is_dir = bool(info.is_dir() or raw_name.endswith("/"))
            kind = _zip_member_kind(normalized, is_dir=is_dir)
            ext = _zip_member_extension(normalized, is_dir=is_dir)
            member_bytes = 0 if is_dir else int(info.file_size)
            member_compressed_bytes = 0 if is_dir else int(info.compress_size)
            member_depth = normalized.count("/")
            top_level = normalized.split("/", 1)[0]

            type_counts[kind] += 1
            extension_counts[ext] += 1
            top_level_counts[top_level] += 1
            if is_dir:
                dirs_total += 1
            else:
                files_total += 1
                uncompressed_total += member_bytes
                compressed_total += member_compressed_bytes

            if len(members) < safe_member_limit:
                members.append(
                    {
                        "path": normalized,
                        "kind": kind,
                        "ext": ext,
                        "depth": member_depth,
                        "is_dir": is_dir,
                        "bytes": member_bytes,
                        "compressed_bytes": member_compressed_bytes,
                        "url": (
                            _library_url_for_archive_member(rel_path, normalized)
                            if not is_dir
                            else ""
                        ),
                    }
                )

        members_total = files_total + dirs_total
        ratio = (
            1.0 - (compressed_total / max(1, uncompressed_total))
            if uncompressed_total > 0
            else 0.0
        )
        zips.append(
            {
                "id": f"zip:{archive_id}",
                "name": zip_path.name,
                "rel_path": rel_path,
                "url": archive_url,
                "bytes": archive_bytes,
                "mtime_utc": mtime_utc,
                "members_total": members_total,
                "files_total": files_total,
                "dirs_total": dirs_total,
                "uncompressed_bytes_total": uncompressed_total,
                "compressed_bytes_total": compressed_total,
                "compression_ratio": round(_clamp01(ratio), 4),
                "members_truncated": members_total > len(members),
                "type_counts": dict(sorted(type_counts.items())),
                "extension_counts": [
                    {"ext": ext, "count": count}
                    for ext, count in sorted(
                        extension_counts.items(),
                        key=lambda row: (-row[1], row[0]),
                    )
                ],
                "top_level_entries": [
                    {"name": name, "count": count}
                    for name, count in sorted(
                        top_level_counts.items(),
                        key=lambda row: (-row[1], row[0]),
                    )
                ],
                "members": members,
            }
        )

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "member_limit": safe_member_limit,
        "zip_count": len(zips),
        "zips": zips,
    }


def _read_library_archive_member(
    archive_path: Path, member_path: str
) -> tuple[bytes, str] | None:
    if archive_path.suffix.lower() != ".zip":
        return None
    norm_member = _normalize_archive_member_path(member_path)
    if not norm_member:
        return None
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            with zf.open(norm_member, "r") as h:
                payload = h.read()
    except:
        return None
    ctype = mimetypes.guess_type(norm_member)[0] or "application/octet-stream"
    if ctype.startswith("text/"):
        ctype += "; charset=utf-8"
    elif ctype == "application/json":
        ctype = "application/json; charset=utf-8"
    return payload, ctype


def resolve_library_path(vault_root: Path, request_path: str) -> Path | None:
    parsed = urlparse(request_path)
    raw_path = unquote(parsed.path)
    if not raw_path.startswith("/library/"):
        return None

    relative = raw_path.removeprefix("/library/")
    if not relative:
        return None

    roots: list[Path] = []
    primary_root = vault_root.resolve()
    roots.append(primary_root)
    substrate_root = _eta_mu_substrate_root(vault_root)
    if substrate_root not in roots:
        roots.append(substrate_root)

    for root in roots:
        candidate = (root / relative).resolve()
        if candidate == root or root in candidate.parents:
            if candidate.exists() and candidate.is_file():
                return candidate

    normalized_relative = str(relative).strip().lstrip("/")
    if normalized_relative:
        entries = _load_eta_mu_knowledge_entries(vault_root)
        archive_relative = ""
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            source_rel_path = str(entry.get("source_rel_path", "")).strip().lstrip("/")
            if source_rel_path != normalized_relative:
                continue
            archive_relative = (
                str(entry.get("archive_rel_path", entry.get("archived_rel_path", "")))
                .strip()
                .lstrip("/")
            )
            if archive_relative:
                break

        if archive_relative:
            for root in roots:
                archive_candidate = (root / archive_relative).resolve()
                if archive_candidate == root or root in archive_candidate.parents:
                    if archive_candidate.exists() and archive_candidate.is_file():
                        return archive_candidate
    return None


def resolve_library_member(request_path: str) -> str | None:
    parsed = urlparse(request_path)
    params = parse_qs(parsed.query)
    return _normalize_archive_member_path(str((params.get("member") or [""])[0] or ""))
