from __future__ import annotations
import os
import time
import json
import hashlib
import threading
import math
import importlib
from hashlib import sha1
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import (
    ETA_MU_EMBEDDINGS_DB_REL,
    ETA_MU_EMBEDDINGS_DB_RECORD,
    ETA_MU_GRAPH_MOVES_REL,
    ETA_MU_FILE_GRAPH_MOVES_RECORD,
    MYCELIAL_ECHO_CACHE_SECONDS,
    _EMBEDDINGS_DB_LOCK,
    _EMBEDDINGS_DB_CACHE,
    _FILE_GRAPH_MOVES_LOCK,
    _FILE_GRAPH_MOVES_CACHE,
    _MYCELIAL_ECHO_CACHE_LOCK,
    _MYCELIAL_ECHO_CACHE,
    _ETA_MU_KNOWLEDGE_LOCK,
    _ETA_MU_KNOWLEDGE_CACHE,
    _ETA_MU_DOCMETA_LOCK,
    _ETA_MU_DOCMETA_CACHE,
    _ETA_MU_REGISTRY_LOCK,
    _ETA_MU_REGISTRY_CACHE,
    _STUDY_SNAPSHOT_LOCK,
    _STUDY_SNAPSHOT_CACHE,
    PRESENCE_ACCOUNT_RECORD,
    SIMULATION_METADATA_RECORD,
    IMAGE_COMMENT_RECORD,
    _PRESENCE_ACCOUNTS_LOCK,
    _PRESENCE_ACCOUNTS_CACHE,
    _SIMULATION_METADATA_LOCK,
    _SIMULATION_METADATA_CACHE,
    _IMAGE_COMMENTS_LOCK,
    _IMAGE_COMMENTS_CACHE,
)
from .metrics import _safe_float, _clamp01
from .paths import (
    _embeddings_db_path,
    _file_graph_moves_path,
    _eta_mu_knowledge_index_path,
    _eta_mu_docmeta_path,
    _eta_mu_registry_path,
    _study_snapshot_log_path,
    _presence_accounts_log_path,
    _simulation_metadata_log_path,
    _image_comments_log_path,
)

try:
    import chromadb
except ImportError:
    chromadb = None

_CHROMA_CLIENT: Any = None
_CHROMA_CLIENT_LOCK = threading.Lock()
_CHROMA_WORLD_COLLECTION: Any = None


def _load_myth_tracker_class() -> type[Any]:
    for module_name in ("code.myth_bridge", "myth_bridge"):
        try:
            module = importlib.import_module(module_name)
            tracker_class = getattr(module, "MythStateTracker", None)
            if tracker_class is not None:
                return tracker_class
        except Exception:
            continue

    class NullMythTracker:
        def snapshot(self, _catalog: dict[str, Any]) -> dict[str, Any]:
            return {}

    return NullMythTracker


def _load_life_tracker_class() -> type[Any]:
    for module_name in ("code.world_life", "world_life"):
        try:
            module = importlib.import_module(module_name)
            tracker_class = getattr(module, "LifeStateTracker", None)
            if tracker_class is not None:
                return tracker_class
        except Exception:
            continue

    class NullLifeTracker:
        def snapshot(
            self,
            _catalog: dict[str, Any],
            _myth_summary: dict[str, Any],
            _entity_manifest: list[dict[str, Any]],
        ) -> dict[str, Any]:
            return {}

    return NullLifeTracker


def _load_life_interaction_builder() -> Any:
    for module_name in ("code.world_life", "world_life"):
        try:
            module = importlib.import_module(module_name)
            builder = getattr(module, "build_interaction_response", None)
            if builder is not None:
                return builder
        except Exception:
            continue

    def _null_builder(
        _world_summary: dict[str, Any], _person_id: str, _action: str = "speak"
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "error": "interaction_unavailable",
            "line_en": "The myth interface is not ready yet.",
            "line_ja": "神話インターフェースはまだ準備中です。",
        }

    return _null_builder


def _get_chroma_collection():
    global _CHROMA_CLIENT, _CHROMA_WORLD_COLLECTION
    if chromadb is None:
        return None

    with _CHROMA_CLIENT_LOCK:
        if _CHROMA_WORLD_COLLECTION is not None:
            return _CHROMA_WORLD_COLLECTION

        if _CHROMA_CLIENT is None:
            try:
                host = os.getenv("CHROMA_HOST", "127.0.0.1")
                port = int(os.getenv("CHROMA_PORT", "8000"))
                _CHROMA_CLIENT = chromadb.HttpClient(host=host, port=port)
            except Exception:
                return None

        try:
            collection = _CHROMA_CLIENT.get_or_create_collection(name="world_memories")
        except Exception:
            return None
        _CHROMA_WORLD_COLLECTION = collection
        return collection


def _load_mycelial_echo_documents(limit: int = 12) -> list[str]:
    try:
        safe_limit = int(limit)
    except (TypeError, ValueError):
        safe_limit = 12
    safe_limit = max(1, min(64, safe_limit))

    now_monotonic = time.monotonic()
    with _MYCELIAL_ECHO_CACHE_LOCK:
        cached_limit = int(_MYCELIAL_ECHO_CACHE.get("limit", 0) or 0)
        cached_checked = _safe_float(
            _MYCELIAL_ECHO_CACHE.get("checked_monotonic", 0.0),
            0.0,
        )
        cached_docs = _MYCELIAL_ECHO_CACHE.get("docs", [])
        if (
            cached_limit == safe_limit
            and isinstance(cached_docs, list)
            and (now_monotonic - cached_checked) <= MYCELIAL_ECHO_CACHE_SECONDS
        ):
            return [str(row) for row in cached_docs if isinstance(row, str)]

    docs: list[str] = []
    collection = _get_chroma_collection()
    if collection is not None:
        try:
            results = collection.get(limit=safe_limit)
            raw_docs = results.get("documents", []) if isinstance(results, dict) else []
            if isinstance(raw_docs, list):
                docs = [str(row) for row in raw_docs if isinstance(row, str)]
        except Exception:
            docs = []

    with _MYCELIAL_ECHO_CACHE_LOCK:
        _MYCELIAL_ECHO_CACHE["limit"] = safe_limit
        _MYCELIAL_ECHO_CACHE["checked_monotonic"] = now_monotonic
        _MYCELIAL_ECHO_CACHE["docs"] = list(docs)

    return docs


def _normalize_embedding_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    out: list[float] = []
    for item in value:
        try:
            numeric = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        out.append(numeric)
    return out


def _deterministic_embedding_vector(seed: str, dims: int = 256) -> list[float]:
    width = max(8, min(2048, int(dims or 256)))
    values: list[float] = []
    for index in range(width):
        token = hashlib.sha256(f"{seed}|{index}".encode("utf-8")).digest()
        raw = int.from_bytes(token[:4], "big") / 4294967295.0
        values.append((raw * 2.0) - 1.0)
    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude > 0.0:
        values = [value / magnitude for value in values]
    return values


def _embedding_vector_dims(vault_root: Path, fallback: int = 256) -> int:
    state = _load_embeddings_db_state(vault_root)
    for row in state.values():
        vector = _normalize_embedding_vector(row.get("embedding", []))
        if vector is not None and len(vector) > 0:
            return len(vector)
    return max(8, int(fallback or 256))


def _upsert_entity_embedding(
    vault_root: Path,
    *,
    entity_id: str,
    text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    clean_entity_id = str(entity_id or "").strip()
    if not clean_entity_id:
        return {"ok": False, "error": "missing entity id"}
    dims = _embedding_vector_dims(vault_root, fallback=256)
    vector = _deterministic_embedding_vector(clean_entity_id + "|" + text, dims=dims)
    return _embedding_db_upsert(
        vault_root,
        entry_id=clean_entity_id,
        text=str(text or ""),
        embedding=vector,
        metadata=metadata,
        model="world-log:deterministic-v1",
    )


def _clone_embedding_record(record: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(record)
    embedding = record.get("embedding", [])
    metadata = record.get("metadata", {})
    cloned["embedding"] = list(embedding) if isinstance(embedding, list) else []
    cloned["metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
    return cloned


def _average_embedding_vectors(vectors: list[list[float]]) -> list[float] | None:
    normalized_vectors: list[list[float]] = []
    dims = 0
    for vector in vectors:
        normalized = _normalize_embedding_vector(vector)
        if normalized is None:
            continue
        if dims == 0:
            dims = len(normalized)
        if len(normalized) != dims:
            continue
        normalized_vectors.append(normalized)
    if not normalized_vectors:
        return None

    accum = [0.0] * dims
    for vector in normalized_vectors:
        for idx, value in enumerate(vector):
            accum[idx] += value
    mean = [value / len(normalized_vectors) for value in accum]
    magnitude = math.sqrt(sum(value * value for value in mean))
    if magnitude > 0.0:
        mean = [value / magnitude for value in mean]
    return mean


def _load_embeddings_db_state(vault_root: Path) -> dict[str, dict[str, Any]]:
    db_path = _embeddings_db_path(vault_root)
    if not db_path.exists() or not db_path.is_file():
        return {}

    try:
        stat = db_path.stat()
    except OSError:
        return {}

    with _EMBEDDINGS_DB_LOCK:
        cached_path = str(_EMBEDDINGS_DB_CACHE.get("path", ""))
        cached_mtime = int(_EMBEDDINGS_DB_CACHE.get("mtime_ns", 0))
        cached_state_raw = _EMBEDDINGS_DB_CACHE.get("state", {})
        cached_state = cached_state_raw if isinstance(cached_state_raw, dict) else {}
        if cached_path == str(db_path) and cached_mtime == int(stat.st_mtime_ns):
            return {
                entry_id: _clone_embedding_record(record)
                for entry_id, record in cached_state.items()
            }

        state: dict[str, dict[str, Any]] = {}
        for raw in db_path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if not isinstance(row, dict):
                continue
            if str(row.get("record", "")).strip() != ETA_MU_EMBEDDINGS_DB_RECORD:
                continue

            event = str(row.get("event", "upsert")).strip().lower()
            entry_id = str(row.get("id", "")).strip()
            if not entry_id:
                continue

            if event == "delete":
                state.pop(entry_id, None)
                continue

            embedding = _normalize_embedding_vector(row.get("embedding"))
            if embedding is None:
                continue

            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            state[entry_id] = {
                "id": entry_id,
                "text": str(row.get("text", "")),
                "embedding": embedding,
                "dim": len(embedding),
                "metadata": metadata,
                "model": str(row.get("model", "")),
                "content_hash": str(row.get("content_hash", "")),
                "created_at": str(row.get("created_at", row.get("time", ""))),
                "updated_at": str(row.get("updated_at", row.get("time", ""))),
                "time": str(row.get("time", "")),
            }

        _EMBEDDINGS_DB_CACHE["path"] = str(db_path)
        _EMBEDDINGS_DB_CACHE["mtime_ns"] = int(stat.st_mtime_ns)
        _EMBEDDINGS_DB_CACHE["state"] = {
            entry_id: _clone_embedding_record(record)
            for entry_id, record in state.items()
        }
        return {
            entry_id: _clone_embedding_record(record)
            for entry_id, record in state.items()
        }


def _embedding_db_upsert(
    vault_root: Path,
    *,
    entry_id: str,
    text: str,
    embedding: list[float],
    metadata: dict[str, Any] | None,
    model: str | None,
) -> dict[str, Any]:
    from .constants import ETA_MU_EMBEDDINGS_DB_RECORD

    normalized = _normalize_embedding_vector(embedding)
    if normalized is None:
        return {"ok": False, "error": "invalid embedding vector"}

    now = datetime.now(timezone.utc).isoformat()
    clean_text = str(text)
    model_name = str(model or "")
    clean_metadata = metadata if isinstance(metadata, dict) else {}

    chosen_id = str(entry_id or "").strip()
    if not chosen_id:
        seed = f"{clean_text}|{now}|{len(normalized)}"
        chosen_id = f"emb_{sha1(seed.encode('utf-8')).hexdigest()[:16]}"

    state = _load_embeddings_db_state(vault_root)
    existing = state.get(chosen_id)
    content_hash = _embedding_payload_hash(
        text=clean_text,
        embedding=normalized,
        metadata=clean_metadata,
        model=model_name,
    )

    if existing and str(existing.get("content_hash", "")) == content_hash:
        return {
            "ok": True,
            "status": "unchanged",
            "entry": _embedding_db_entry_public(existing),
        }

    created_at = str(existing.get("created_at", now)) if existing else now
    record = {
        "record": ETA_MU_EMBEDDINGS_DB_RECORD,
        "event": "upsert",
        "id": chosen_id,
        "text": clean_text,
        "embedding": normalized,
        "dim": len(normalized),
        "metadata": clean_metadata,
        "model": model_name,
        "content_hash": content_hash,
        "created_at": created_at,
        "updated_at": now,
        "time": now,
    }
    _append_embeddings_db_event(vault_root, record)
    return {
        "ok": True,
        "status": "upserted",
        "entry": _embedding_db_entry_public(record),
    }


def _embedding_db_upsert_append_only(
    vault_root: Path,
    *,
    entry_id: str,
    text: str,
    embedding: list[float],
    metadata: dict[str, Any] | None,
    model: str | None,
) -> dict[str, Any]:
    from .constants import ETA_MU_EMBEDDINGS_DB_RECORD

    normalized = _normalize_embedding_vector(embedding)
    if normalized is None:
        return {"ok": False, "error": "invalid embedding vector"}

    now = datetime.now(timezone.utc).isoformat()
    clean_text = str(text)
    model_name = str(model or "")
    clean_metadata = metadata if isinstance(metadata, dict) else {}

    chosen_id = str(entry_id or "").strip()
    if not chosen_id:
        seed = f"{clean_text}|{now}|{len(normalized)}"
        chosen_id = f"emb_{sha1(seed.encode('utf-8')).hexdigest()[:16]}"

    content_hash = _embedding_payload_hash(
        text=clean_text,
        embedding=normalized,
        metadata=clean_metadata,
        model=model_name,
    )
    record = {
        "record": ETA_MU_EMBEDDINGS_DB_RECORD,
        "event": "upsert",
        "id": chosen_id,
        "text": clean_text,
        "embedding": normalized,
        "dim": len(normalized),
        "metadata": clean_metadata,
        "model": model_name,
        "content_hash": content_hash,
        "created_at": now,
        "updated_at": now,
        "time": now,
    }
    _append_embeddings_db_event(vault_root, record)
    return {
        "ok": True,
        "status": "upserted",
        "entry": _embedding_db_entry_public(record),
    }


def _embedding_db_delete(vault_root: Path, *, entry_id: str) -> dict[str, Any]:
    from .constants import ETA_MU_EMBEDDINGS_DB_RECORD

    chosen_id = str(entry_id or "").strip()
    if not chosen_id:
        return {"ok": False, "error": "missing id"}

    state = _load_embeddings_db_state(vault_root)
    if chosen_id not in state:
        return {"ok": False, "error": "not found", "id": chosen_id}

    now = datetime.now(timezone.utc).isoformat()
    record = {
        "record": ETA_MU_EMBEDDINGS_DB_RECORD,
        "event": "delete",
        "id": chosen_id,
        "time": now,
    }
    _append_embeddings_db_event(vault_root, record)
    return {"ok": True, "status": "deleted", "id": chosen_id, "time": now}


def _embedding_db_list(
    vault_root: Path,
    *,
    limit: int = 50,
    include_vectors: bool = False,
) -> dict[str, Any]:
    bounded_limit = max(1, min(500, int(limit)))
    state = _load_embeddings_db_state(vault_root)
    rows = sorted(
        state.values(),
        key=lambda row: str(row.get("updated_at", "")),
        reverse=True,
    )
    selected = rows[:bounded_limit]
    return {
        "ok": True,
        "record": "ημ.embedding-db.list.v1",
        "path": str(_embeddings_db_path(vault_root)),
        "total": len(state),
        "count": len(selected),
        "entries": [
            _embedding_db_entry_public(row, include_vector=include_vectors)
            for row in selected
        ],
    }


def _embedding_db_query(
    vault_root: Path,
    *,
    query_embedding: list[float],
    top_k: int = 5,
    min_score: float = -1.0,
    include_vectors: bool = False,
) -> dict[str, Any]:
    normalized_query = _normalize_embedding_vector(query_embedding)
    if normalized_query is None:
        return {"ok": False, "error": "invalid query embedding"}

    bounded_top_k = max(1, min(100, int(top_k)))
    score_floor = max(-1.0, min(1.0, float(min_score)))
    state = _load_embeddings_db_state(vault_root)

    matches: list[dict[str, Any]] = []
    for row in state.values():
        candidate = _normalize_embedding_vector(row.get("embedding", []))
        if candidate is None:
            continue
        score = _cosine_similarity(normalized_query, candidate)
        if score is None or score < score_floor:
            continue
        entry = _embedding_db_entry_public(row, include_vector=include_vectors)
        entry["score"] = round(score, 6)
        matches.append(entry)

    matches.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    selected = matches[:bounded_top_k]
    return {
        "ok": True,
        "record": "ημ.embedding-db.query.v1",
        "path": str(_embeddings_db_path(vault_root)),
        "query_dim": len(normalized_query),
        "top_k": bounded_top_k,
        "match_count": len(selected),
        "results": selected,
    }


def _embedding_db_status(vault_root: Path) -> dict[str, Any]:
    from .ai import _embedding_provider_status

    state = _load_embeddings_db_state(vault_root)
    dims = sorted(
        {
            int(row.get("dim", 0) or 0)
            for row in state.values()
            if int(row.get("dim", 0) or 0) > 0
        }
    )
    return {
        "ok": True,
        "record": "ημ.embedding-db.status.v1",
        "path": str(_embeddings_db_path(vault_root)),
        "entry_count": len(state),
        "dimensions": dims,
        "provider": _embedding_provider_status(),
    }


def _embedding_payload_hash(
    *,
    text: str,
    embedding: list[float],
    metadata: dict[str, Any],
    model: str,
) -> str:
    payload = {
        "text": str(text),
        "embedding": embedding,
        "metadata": metadata,
        "model": str(model),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _append_embeddings_db_event(vault_root: Path, record: dict[str, Any]) -> None:
    db_path = _embeddings_db_path(vault_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with db_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    try:
        stat = db_path.stat()
    except OSError:
        stat = None
    with _EMBEDDINGS_DB_LOCK:
        cache_path = str(_EMBEDDINGS_DB_CACHE.get("path", ""))
        cache_state_raw = _EMBEDDINGS_DB_CACHE.get("state", {})
        cache_state = cache_state_raw if isinstance(cache_state_raw, dict) else {}
        if cache_path != str(db_path) or not cache_state:
            _EMBEDDINGS_DB_CACHE["path"] = ""
            _EMBEDDINGS_DB_CACHE["mtime_ns"] = 0
            _EMBEDDINGS_DB_CACHE["state"] = {}
            return

        entry_id = str(record.get("id", "")).strip()
        event_kind = str(record.get("event", "upsert")).strip().lower()
        next_state = {
            key: _clone_embedding_record(value) for key, value in cache_state.items()
        }

        if entry_id:
            if event_kind == "delete":
                next_state.pop(entry_id, None)
            else:
                embedding = _normalize_embedding_vector(record.get("embedding", []))
                if embedding is not None:
                    metadata = record.get("metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    next_state[entry_id] = {
                        "id": entry_id,
                        "text": str(record.get("text", "")),
                        "embedding": embedding,
                        "dim": len(embedding),
                        "metadata": metadata,
                        "model": str(record.get("model", "")),
                        "content_hash": str(record.get("content_hash", "")),
                        "created_at": str(
                            record.get("created_at", record.get("time", ""))
                        ),
                        "updated_at": str(
                            record.get("updated_at", record.get("time", ""))
                        ),
                        "time": str(record.get("time", "")),
                    }

        _EMBEDDINGS_DB_CACHE["path"] = str(db_path)
        _EMBEDDINGS_DB_CACHE["mtime_ns"] = int(stat.st_mtime_ns) if stat else 0
        _EMBEDDINGS_DB_CACHE["state"] = {
            key: _clone_embedding_record(value) for key, value in next_state.items()
        }


def _embedding_db_entry_public(
    record: dict[str, Any],
    *,
    include_vector: bool = False,
) -> dict[str, Any]:
    payload = {
        "id": str(record.get("id", "")),
        "text": str(record.get("text", "")),
        "dim": int(record.get("dim", 0) or 0),
        "metadata": dict(record.get("metadata", {})),
        "model": str(record.get("model", "")),
        "created_at": str(record.get("created_at", "")),
        "updated_at": str(record.get("updated_at", "")),
        "content_hash": str(record.get("content_hash", "")),
    }
    if include_vector:
        payload["embedding"] = list(record.get("embedding", []))
    return payload


def _cosine_similarity(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    if len(left) != len(right):
        return None

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for lhs, rhs in zip(left, right):
        dot += lhs * rhs
        left_norm += lhs * lhs
        right_norm += rhs * rhs

    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return dot / math.sqrt(left_norm * right_norm)


def _eta_mu_registry_known_entries(
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    known: dict[str, dict[str, Any]] = {}
    for row in entries:
        key = str(row.get("registry_key", "")).strip()
        if not key:
            continue
        event = str(row.get("event", "")).strip().lower()
        if event in {"ingested", "skipped"}:
            known[key] = dict(row)
    return known


def _eta_mu_collect_registry_idempotence(
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    known: dict[str, dict[str, Any]] = {}
    for row in entries:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "")).strip().lower()
        event = str(row.get("event", "")).strip().lower()
        if status not in {"ok", "skip", "defer"} and event not in {
            "ingested",
            "skipped",
            "deferred",
        }:
            continue
        idempotence_key = str(row.get("idempotence_key", "")).strip()
        if not idempotence_key:
            continue
        known[idempotence_key] = dict(row)
    return known


def _eta_mu_vecstore_upsert_batch(
    vault_root: Path,
    rows: list[dict[str, Any]],
    *,
    collection_name: str,
    space_set_signature: str,
) -> dict[str, Any]:
    from .constants import ETA_MU_INGEST_VECSTORE_COLLECTION

    chosen_collection = (
        str(collection_name or ETA_MU_INGEST_VECSTORE_COLLECTION).strip()
        or ETA_MU_INGEST_VECSTORE_COLLECTION
    )
    if not rows:
        return {
            "ok": True,
            "backend": "none",
            "collection": chosen_collection,
            "upserted": 0,
            "ids": [],
            "drift": False,
        }

    baseline = str(os.getenv("ETA_MU_VECSTORE_SIGNATURE_BASELINE", "") or "").strip()
    drift = bool(baseline) and baseline != space_set_signature
    if drift:
        return {
            "ok": False,
            "backend": "deferred",
            "collection": chosen_collection,
            "upserted": 0,
            "ids": [],
            "drift": True,
            "error": "vecstore_signature_drift",
        }

    ids = [str(row.get("id", "")) for row in rows]
    embeddings = [list(row.get("embedding", [])) for row in rows]
    metadatas = [dict(row.get("metadata", {})) for row in rows]
    documents = [str(row.get("document", "")) for row in rows]

    def _mirror_rows_to_embedding_db() -> list[str]:
        mirrored_ids: list[str] = []
        for row in rows:
            result = _embedding_db_upsert(
                vault_root,
                entry_id=str(row.get("id", "")),
                text=str(row.get("document", "")),
                embedding=list(row.get("embedding", [])),
                metadata=dict(row.get("metadata", {})),
                model=str(row.get("model", "")),
            )
            if bool(result.get("ok")):
                mirrored_ids.append(str(row.get("id", "")))
        return mirrored_ids

    collection = _get_eta_mu_chroma_collection(chosen_collection)
    if collection is not None:
        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            mirrored_ids = _mirror_rows_to_embedding_db()
            return {
                "ok": True,
                "backend": "chroma",
                "collection": chosen_collection,
                "upserted": len(ids),
                "ids": ids,
                "mirrored": len(mirrored_ids),
                "drift": False,
            }
        except Exception as exc:
            error_text = str(exc)
    else:
        error_text = "chroma_unavailable"

    fallback_ids = _mirror_rows_to_embedding_db()

    return {
        "ok": len(fallback_ids) == len(rows),
        "backend": "eta_mu_embeddings_db",
        "collection": chosen_collection,
        "upserted": len(fallback_ids),
        "ids": fallback_ids,
        "drift": False,
        "error": error_text,
    }


def _get_eta_mu_chroma_collection(collection_name: str | None = None) -> Any:
    global _CHROMA_CLIENT
    from .constants import ETA_MU_INGEST_VECSTORE_COLLECTION

    if chromadb is None:
        return None

    if _CHROMA_CLIENT is None:
        try:
            host = os.getenv("CHROMA_HOST", "127.0.0.1")
            port = int(os.getenv("CHROMA_PORT", "8000"))
            _CHROMA_CLIENT = chromadb.HttpClient(host=host, port=port)
        except Exception:
            return None

    try:
        chosen_collection = (
            str(collection_name or ETA_MU_INGEST_VECSTORE_COLLECTION).strip()
            or ETA_MU_INGEST_VECSTORE_COLLECTION
        )
        return _CHROMA_CLIENT.get_or_create_collection(name=chosen_collection)
    except Exception:
        return None


def _load_eta_mu_knowledge_entries(vault_root: Path) -> list[dict[str, Any]]:
    index_path = _eta_mu_knowledge_index_path(vault_root)
    if not index_path.exists() or not index_path.is_file():
        return []
    try:
        stat = index_path.stat()
    except OSError:
        return []

    with _ETA_MU_KNOWLEDGE_LOCK:
        cached_path = str(_ETA_MU_KNOWLEDGE_CACHE.get("path", ""))
        cached_mtime = int(_ETA_MU_KNOWLEDGE_CACHE.get("mtime_ns", 0))
        if cached_path == str(index_path) and cached_mtime == int(stat.st_mtime_ns):
            return [dict(item) for item in _ETA_MU_KNOWLEDGE_CACHE.get("entries", [])]

        entries: list[dict[str, Any]] = []
        for raw in index_path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                entries.append(row)

        entries.sort(key=lambda row: str(row.get("ingested_at", "")), reverse=True)
        _ETA_MU_KNOWLEDGE_CACHE["path"] = str(index_path)
        _ETA_MU_KNOWLEDGE_CACHE["mtime_ns"] = int(stat.st_mtime_ns)
        _ETA_MU_KNOWLEDGE_CACHE["entries"] = [dict(item) for item in entries]
        return entries


def _load_eta_mu_docmeta_entries(vault_root: Path) -> list[dict[str, Any]]:
    docmeta_path = _eta_mu_docmeta_path(vault_root)
    if not docmeta_path.exists() or not docmeta_path.is_file():
        return []
    try:
        stat = docmeta_path.stat()
    except OSError:
        return []

    with _ETA_MU_DOCMETA_LOCK:
        cached_path = str(_ETA_MU_DOCMETA_CACHE.get("path", ""))
        cached_mtime = int(_ETA_MU_DOCMETA_CACHE.get("mtime_ns", 0))
        if cached_path == str(docmeta_path) and cached_mtime == int(stat.st_mtime_ns):
            return [dict(item) for item in _ETA_MU_DOCMETA_CACHE.get("entries", [])]

        entries: list[dict[str, Any]] = []
        for raw in docmeta_path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                entries.append(row)

        entries.sort(key=lambda row: str(row.get("generated_at", "")), reverse=True)
        _ETA_MU_DOCMETA_CACHE["path"] = str(docmeta_path)
        _ETA_MU_DOCMETA_CACHE["mtime_ns"] = int(stat.st_mtime_ns)
        _ETA_MU_DOCMETA_CACHE["entries"] = [dict(item) for item in entries]
        return entries


def _load_eta_mu_registry_entries(vault_root: Path) -> list[dict[str, Any]]:
    registry_path = _eta_mu_registry_path(vault_root)
    if not registry_path.exists() or not registry_path.is_file():
        return []
    try:
        stat = registry_path.stat()
    except OSError:
        return []

    with _ETA_MU_REGISTRY_LOCK:
        cached_path = str(_ETA_MU_REGISTRY_CACHE.get("path", ""))
        cached_mtime = int(_ETA_MU_REGISTRY_CACHE.get("mtime_ns", 0))
        if cached_path == str(registry_path) and cached_mtime == int(stat.st_mtime_ns):
            return [dict(item) for item in _ETA_MU_REGISTRY_CACHE.get("entries", [])]

        entries: list[dict[str, Any]] = []
        for raw in registry_path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                entries.append(row)

        _ETA_MU_REGISTRY_CACHE["path"] = str(registry_path)
        _ETA_MU_REGISTRY_CACHE["mtime_ns"] = int(stat.st_mtime_ns)
        _ETA_MU_REGISTRY_CACHE["entries"] = [dict(item) for item in entries]
        return entries


def _load_study_snapshot_events(
    vault_root: Path, *, limit: int = 256
) -> list[dict[str, Any]]:
    path = _study_snapshot_log_path(vault_root)
    if not path.exists() or not path.is_file():
        return []
    try:
        stat = path.stat()
    except OSError:
        return []

    with _STUDY_SNAPSHOT_LOCK:
        events: list[dict[str, Any]] = []
        for raw in path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                events.append(row)

        events.sort(
            key=lambda row: str(row.get("ts", row.get("time", ""))), reverse=True
        )
        bounded = events[: max(1, int(limit))]
        _STUDY_SNAPSHOT_CACHE["path"] = str(path)
        _STUDY_SNAPSHOT_CACHE["mtime_ns"] = int(stat.st_mtime_ns)
        _STUDY_SNAPSHOT_CACHE["events"] = [dict(item) for item in bounded]
        return [dict(item) for item in bounded]


def _append_study_snapshot_event(vault_root: Path, event: dict[str, Any]) -> Path:
    path = _study_snapshot_log_path(vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with _STUDY_SNAPSHOT_LOCK:
        _STUDY_SNAPSHOT_CACHE["path"] = ""
        _STUDY_SNAPSHOT_CACHE["mtime_ns"] = 0
        _STUDY_SNAPSHOT_CACHE["events"] = []
    return path


def _append_eta_mu_registry_record(vault_root: Path, record: dict[str, Any]) -> None:
    from .paths import _eta_mu_registry_path

    registry_path = _eta_mu_registry_path(vault_root)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with _ETA_MU_REGISTRY_LOCK:
        _ETA_MU_REGISTRY_CACHE["path"] = ""
        _ETA_MU_REGISTRY_CACHE["mtime_ns"] = 0
        _ETA_MU_REGISTRY_CACHE["entries"] = []


def _append_eta_mu_knowledge_record(vault_root: Path, record: dict[str, Any]) -> None:
    from .paths import _eta_mu_knowledge_index_path

    index_path = _eta_mu_knowledge_index_path(vault_root)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with _ETA_MU_KNOWLEDGE_LOCK:
        _ETA_MU_KNOWLEDGE_CACHE["path"] = ""
        _ETA_MU_KNOWLEDGE_CACHE["mtime_ns"] = 0
        _ETA_MU_KNOWLEDGE_CACHE["entries"] = []


def _append_eta_mu_docmeta_record(vault_root: Path, record: dict[str, Any]) -> None:
    docmeta_path = _eta_mu_docmeta_path(vault_root)
    docmeta_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with docmeta_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with _ETA_MU_DOCMETA_LOCK:
        _ETA_MU_DOCMETA_CACHE["path"] = ""
        _ETA_MU_DOCMETA_CACHE["mtime_ns"] = 0
        _ETA_MU_DOCMETA_CACHE["entries"] = []


def _load_simulation_metadata_entries(vault_root: Path) -> list[dict[str, Any]]:
    path = _simulation_metadata_log_path(vault_root)
    if not path.exists() or not path.is_file():
        return []
    try:
        stat = path.stat()
    except OSError:
        return []

    with _SIMULATION_METADATA_LOCK:
        cached_path = str(_SIMULATION_METADATA_CACHE.get("path", ""))
        cached_mtime = int(_SIMULATION_METADATA_CACHE.get("mtime_ns", 0))
        if cached_path == str(path) and cached_mtime == int(stat.st_mtime_ns):
            return [
                dict(item) for item in _SIMULATION_METADATA_CACHE.get("entries", [])
            ]

        rows: list[dict[str, Any]] = []
        for raw in path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                rows.append(row)

        rows.sort(key=lambda row: str(row.get("updated_at", row.get("time", ""))))
        _SIMULATION_METADATA_CACHE["path"] = str(path)
        _SIMULATION_METADATA_CACHE["mtime_ns"] = int(stat.st_mtime_ns)
        _SIMULATION_METADATA_CACHE["entries"] = [dict(item) for item in rows]
        return [dict(item) for item in rows]


def _append_simulation_metadata_record(
    vault_root: Path, record: dict[str, Any]
) -> None:
    path = _simulation_metadata_log_path(vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with _SIMULATION_METADATA_LOCK:
        _SIMULATION_METADATA_CACHE["path"] = ""
        _SIMULATION_METADATA_CACHE["mtime_ns"] = 0
        _SIMULATION_METADATA_CACHE["entries"] = []


def _simulation_metadata_state(vault_root: Path) -> dict[str, dict[str, Any]]:
    from .constants import SIMULATION_METADATA_RECORD

    state: dict[str, dict[str, Any]] = {}
    for row in _load_simulation_metadata_entries(vault_root):
        if str(row.get("record", "")).strip() != SIMULATION_METADATA_RECORD:
            continue
        presence_id = str(row.get("presence_id", "")).strip()
        if not presence_id:
            continue
        state[presence_id] = dict(row)
    return state


def _upsert_simulation_metadata(
    vault_root: Path,
    *,
    presence_id: str,
    label: str,
    description: str,
    tags: list[str] | None = None,
    process_info: dict[str, Any] | None = None,
    benchmark_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .constants import SIMULATION_METADATA_RECORD, PRESENCE_ACCOUNT_RECORD

    chosen_id = str(presence_id or "").strip()
    if not chosen_id:
        return {"ok": False, "error": "missing presence_id"}

    now = datetime.now(timezone.utc).isoformat()
    state = _simulation_metadata_state(vault_root)
    existing = state.get(chosen_id, {})
    clean_tags = [str(item).strip() for item in (tags or []) if str(item).strip()]

    # Merge process info and benchmark results with existing if not provided
    final_process = (
        process_info if process_info is not None else existing.get("process_info", {})
    )
    final_benchmark = (
        benchmark_results
        if benchmark_results is not None
        else existing.get("benchmark_results", {})
    )

    record = {
        "record": SIMULATION_METADATA_RECORD,
        "presence_id": chosen_id,
        "label": str(label or existing.get("label") or chosen_id),
        "description": str(description or existing.get("description") or ""),
        "tags": clean_tags or list(existing.get("tags", [])),
        "process_info": final_process,
        "benchmark_results": final_benchmark,
        "created_at": str(existing.get("created_at") or now),
        "updated_at": now,
        "time": now,
    }
    _append_simulation_metadata_record(vault_root, record)

    # Sync to Presence Account for field embedding
    # We map 'description' to 'bio' and 'label' to 'display_name'
    account_result = _upsert_presence_account(
        vault_root,
        presence_id=chosen_id,
        display_name=record["label"],
        handle=chosen_id,
        avatar=str(existing.get("avatar", "")),  # Preserve existing avatar
        bio=record["description"],
        tags=record["tags"],
    )

    return {
        "ok": True,
        "record": "ημ.simulation-metadata.upsert.v1",
        "entry": record,
        "account_sync": account_result.get("ok", False),
    }


def _list_simulation_metadata(vault_root: Path, *, limit: int = 64) -> dict[str, Any]:
    state = _simulation_metadata_state(vault_root)
    rows = sorted(
        state.values(),
        key=lambda row: str(row.get("updated_at", "")),
        reverse=True,
    )
    bounded = max(1, min(500, int(limit)))
    selected = rows[:bounded]
    return {
        "ok": True,
        "record": "ημ.simulation-metadata.list.v1",
        "path": str(_simulation_metadata_log_path(vault_root)),
        "total": len(rows),
        "count": len(selected),
        "entries": selected,
    }


def _load_presence_account_entries(vault_root: Path) -> list[dict[str, Any]]:
    path = _presence_accounts_log_path(vault_root)
    if not path.exists() or not path.is_file():
        return []
    try:
        stat = path.stat()
    except OSError:
        return []

    with _PRESENCE_ACCOUNTS_LOCK:
        cached_path = str(_PRESENCE_ACCOUNTS_CACHE.get("path", ""))
        cached_mtime = int(_PRESENCE_ACCOUNTS_CACHE.get("mtime_ns", 0))
        if cached_path == str(path) and cached_mtime == int(stat.st_mtime_ns):
            return [dict(item) for item in _PRESENCE_ACCOUNTS_CACHE.get("entries", [])]

        rows: list[dict[str, Any]] = []
        for raw in path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                rows.append(row)

        rows.sort(key=lambda row: str(row.get("updated_at", row.get("time", ""))))
        _PRESENCE_ACCOUNTS_CACHE["path"] = str(path)
        _PRESENCE_ACCOUNTS_CACHE["mtime_ns"] = int(stat.st_mtime_ns)
        _PRESENCE_ACCOUNTS_CACHE["entries"] = [dict(item) for item in rows]
        return [dict(item) for item in rows]


def _append_presence_account_record(vault_root: Path, record: dict[str, Any]) -> None:
    path = _presence_accounts_log_path(vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with _PRESENCE_ACCOUNTS_LOCK:
        _PRESENCE_ACCOUNTS_CACHE["path"] = ""
        _PRESENCE_ACCOUNTS_CACHE["mtime_ns"] = 0
        _PRESENCE_ACCOUNTS_CACHE["entries"] = []


def _presence_accounts_state(vault_root: Path) -> dict[str, dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    for row in _load_presence_account_entries(vault_root):
        if str(row.get("record", "")).strip() != PRESENCE_ACCOUNT_RECORD:
            continue
        presence_id = str(row.get("presence_id", "")).strip()
        if not presence_id:
            continue
        state[presence_id] = {
            "presence_id": presence_id,
            "display_name": str(row.get("display_name", "") or presence_id),
            "handle": str(row.get("handle", "") or presence_id),
            "avatar": str(row.get("avatar", "") or ""),
            "bio": str(row.get("bio", "") or ""),
            "tags": [
                str(item).strip() for item in row.get("tags", []) if str(item).strip()
            ],
            "created_at": str(row.get("created_at", row.get("time", ""))),
            "updated_at": str(row.get("updated_at", row.get("time", ""))),
            "time": str(row.get("time", "")),
        }
    return state


def _upsert_presence_account(
    vault_root: Path,
    *,
    presence_id: str,
    display_name: str,
    handle: str,
    avatar: str,
    bio: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    chosen_id = str(presence_id or "").strip()
    if not chosen_id:
        return {"ok": False, "error": "missing presence_id"}

    now = datetime.now(timezone.utc).isoformat()
    state = _presence_accounts_state(vault_root)
    existing = state.get(chosen_id)
    clean_tags = [str(item).strip() for item in (tags or []) if str(item).strip()]

    record = {
        "record": PRESENCE_ACCOUNT_RECORD,
        "presence_id": chosen_id,
        "display_name": str(
            display_name or (existing or {}).get("display_name") or chosen_id
        ),
        "handle": str(handle or (existing or {}).get("handle") or chosen_id),
        "avatar": str(avatar or (existing or {}).get("avatar") or ""),
        "bio": str(bio or (existing or {}).get("bio") or ""),
        "tags": clean_tags or list((existing or {}).get("tags", [])),
        "created_at": str((existing or {}).get("created_at") or now),
        "updated_at": now,
        "time": now,
    }
    _append_presence_account_record(vault_root, record)
    profile_text = " | ".join(
        [
            str(record.get("display_name", "")),
            str(record.get("handle", "")),
            str(record.get("bio", "")),
            " ".join(str(tag) for tag in record.get("tags", [])),
        ]
    ).strip()
    _upsert_entity_embedding(
        vault_root,
        entity_id=f"presence-account:{record['presence_id']}",
        text=profile_text,
        metadata={
            "entity_kind": "presence_account",
            "presence_id": record["presence_id"],
            "handle": record["handle"],
            "updated_at": record["updated_at"],
        },
    )
    return {
        "ok": True,
        "record": "ημ.presence-account.upsert.v1",
        "entry": {
            "presence_id": record["presence_id"],
            "display_name": record["display_name"],
            "handle": record["handle"],
            "avatar": record["avatar"],
            "bio": record["bio"],
            "tags": list(record["tags"]),
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
        },
    }


def _list_presence_accounts(vault_root: Path, *, limit: int = 64) -> dict[str, Any]:
    bounded = max(1, min(500, int(limit)))
    state = _presence_accounts_state(vault_root)
    rows = sorted(
        state.values(),
        key=lambda row: str(row.get("updated_at", "")),
        reverse=True,
    )
    selected = rows[:bounded]
    return {
        "ok": True,
        "record": "ημ.presence-account.list.v1",
        "path": str(_presence_accounts_log_path(vault_root)),
        "total": len(rows),
        "count": len(selected),
        "entries": selected,
    }


def _load_image_comment_entries(vault_root: Path) -> list[dict[str, Any]]:
    path = _image_comments_log_path(vault_root)
    if not path.exists() or not path.is_file():
        return []
    try:
        stat = path.stat()
    except OSError:
        return []

    with _IMAGE_COMMENTS_LOCK:
        cached_path = str(_IMAGE_COMMENTS_CACHE.get("path", ""))
        cached_mtime = int(_IMAGE_COMMENTS_CACHE.get("mtime_ns", 0))
        if cached_path == str(path) and cached_mtime == int(stat.st_mtime_ns):
            return [dict(item) for item in _IMAGE_COMMENTS_CACHE.get("entries", [])]

        rows: list[dict[str, Any]] = []
        for raw in path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                rows.append(row)

        rows.sort(key=lambda row: str(row.get("created_at", row.get("time", ""))))
        _IMAGE_COMMENTS_CACHE["path"] = str(path)
        _IMAGE_COMMENTS_CACHE["mtime_ns"] = int(stat.st_mtime_ns)
        _IMAGE_COMMENTS_CACHE["entries"] = [dict(item) for item in rows]
        return [dict(item) for item in rows]


def _append_image_comment_record(vault_root: Path, record: dict[str, Any]) -> None:
    path = _image_comments_log_path(vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with _IMAGE_COMMENTS_LOCK:
        _IMAGE_COMMENTS_CACHE["path"] = ""
        _IMAGE_COMMENTS_CACHE["mtime_ns"] = 0
        _IMAGE_COMMENTS_CACHE["entries"] = []


def _create_image_comment(
    vault_root: Path,
    *,
    image_ref: str,
    presence_id: str,
    comment: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_image_ref = str(image_ref or "").strip()
    clean_presence_id = str(presence_id or "").strip()
    clean_comment = str(comment or "").strip()
    if not clean_image_ref:
        return {"ok": False, "error": "missing image_ref"}
    if not clean_presence_id:
        return {"ok": False, "error": "missing presence_id"}
    if not clean_comment:
        return {"ok": False, "error": "missing comment"}

    now = datetime.now(timezone.utc).isoformat()
    comment_seed = (
        f"{clean_image_ref}|{clean_presence_id}|{clean_comment}|{now}|{time.time_ns()}"
    )
    comment_id = "imgc_" + sha1(comment_seed.encode("utf-8")).hexdigest()[:16]
    record = {
        "record": IMAGE_COMMENT_RECORD,
        "id": comment_id,
        "image_ref": clean_image_ref,
        "presence_id": clean_presence_id,
        "comment": clean_comment,
        "metadata": metadata if isinstance(metadata, dict) else {},
        "created_at": now,
        "time": now,
    }
    _append_image_comment_record(vault_root, record)
    _upsert_entity_embedding(
        vault_root,
        entity_id=f"image-comment:{comment_id}",
        text=f"{clean_presence_id}: {clean_comment}",
        metadata={
            "entity_kind": "image_comment",
            "comment_id": comment_id,
            "image_ref": clean_image_ref,
            "presence_id": clean_presence_id,
            "created_at": now,
        },
    )
    return {
        "ok": True,
        "record": "ημ.image-comment.create.v1",
        "entry": dict(record),
    }


def _list_image_comments(
    vault_root: Path,
    *,
    image_ref: str = "",
    limit: int = 120,
) -> dict[str, Any]:
    bounded = max(1, min(1000, int(limit)))
    selected_ref = str(image_ref or "").strip()
    entries = []
    for row in _load_image_comment_entries(vault_root):
        if str(row.get("record", "")).strip() != IMAGE_COMMENT_RECORD:
            continue
        if selected_ref and str(row.get("image_ref", "")).strip() != selected_ref:
            continue
        entries.append(row)
    entries.sort(
        key=lambda row: str(row.get("created_at", row.get("time", ""))), reverse=True
    )
    sliced = entries[:bounded]
    return {
        "ok": True,
        "record": "ημ.image-comment.list.v1",
        "path": str(_image_comments_log_path(vault_root)),
        "image_ref": selected_ref,
        "total": len(entries),
        "count": len(sliced),
        "entries": [dict(item) for item in sliced],
    }


def _assign_meaning_clusters(
    rows: list[dict[str, Any]],
    threshold: float = 0.84,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    cutoff = max(-1.0, min(1.0, _safe_float(threshold, 0.84)))
    clusters: list[dict[str, Any]] = []
    assignments: dict[str, str] = {}

    for row in rows:
        node_id = str(row.get("node_id", "")).strip()
        vector = _normalize_embedding_vector(row.get("vector", []))
        if not node_id or vector is None:
            continue

        best_index = -1
        best_score = -2.0
        for index, cluster in enumerate(clusters):
            centroid = _normalize_embedding_vector(cluster.get("centroid", []))
            if centroid is None:
                continue
            score = _cosine_similarity(vector, centroid)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_index = index

        if best_index >= 0 and best_score >= cutoff:
            cluster = clusters[best_index]
            count = int(cluster.get("count", 1))
            centroid = _normalize_embedding_vector(cluster.get("centroid", [])) or list(
                vector
            )
            if len(centroid) != len(vector):
                centroid = list(vector)
            merged = [
                ((centroid[idx] * count) + vector[idx]) / max(1, count + 1)
                for idx in range(len(vector))
            ]
            cluster["centroid"] = _normalize_embedding_vector(merged) or merged
            cluster["count"] = count + 1
            cluster["members"].append(node_id)
            cluster["similarity_sum"] = _safe_float(
                cluster.get("similarity_sum", 0.0), 0.0
            ) + max(-1.0, min(1.0, best_score))
            cluster_id = str(cluster.get("id", ""))
        else:
            cluster_id = (
                "meaning:" + hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:12]
            )
            clusters.append(
                {
                    "id": cluster_id,
                    "centroid": list(vector),
                    "count": 1,
                    "members": [node_id],
                    "similarity_sum": 1.0,
                }
            )
        assignments[node_id] = cluster_id

    cluster_rows: list[dict[str, Any]] = []
    for cluster in clusters:
        count = max(1, int(cluster.get("count", 1)))
        similarity_mean = _safe_float(cluster.get("similarity_sum", 0.0), 0.0) / count
        cohesion = _clamp01((similarity_mean + 1.0) / 2.0)
        members = [
            str(item) for item in cluster.get("members", []) if str(item).strip()
        ]
        members = sorted(dict.fromkeys(members))
        cluster_rows.append(
            {
                "id": str(cluster.get("id", "")),
                "size": len(members),
                "cohesion": round(cohesion, 4),
                "members": members,
            }
        )

    cluster_rows.sort(
        key=lambda row: (-int(row.get("size", 0)), str(row.get("id", "")))
    )
    return assignments, cluster_rows
