from __future__ import annotations
import os
import time
import json
import base64
import hashlib
import threading
import subprocess
import tempfile
import wave
import socket
import re
import math
import random
import mimetypes
import unicodedata
import importlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from .constants import (
    TTS_BASE_URL,
    VOICE_LINE_BANK,
    ENTITY_MANIFEST,
    KEEPER_OF_CONTRACTS_PROFILE,
    FILE_SENTINEL_PROFILE,
    THE_COUNCIL_PROFILE,
    FILE_ORGANIZER_PROFILE,
    HEALTH_SENTINEL_CPU_PROFILE,
    HEALTH_SENTINEL_GPU1_PROFILE,
    HEALTH_SENTINEL_GPU2_PROFILE,
    HEALTH_SENTINEL_NPU0_PROFILE,
    _PRESENCE_ALIASES,
    _WHISPER_MODEL_LOCK,
    _WHISPER_MODEL,
    _TENSORFLOW_RUNTIME_LOCK,
    _TENSORFLOW_RUNTIME,
    _OPENVINO_EMBED_LOCK,
    _OPENVINO_EMBED_RUNTIME,
    SYSTEM_PROMPT_TEMPLATE,
    PART_67_PROLOGUE,
    IMAGE_COMMENTARY_MODEL,
    IMAGE_COMMENTARY_TIMEOUT_SECONDS,
    IMAGE_COMMENTARY_MAX_BYTES,
)
from .metrics import (
    _safe_float,
    _resource_monitor_snapshot,
    _clamp01,
    _resource_auto_embedding_order,
    _resource_auto_text_order,
)

CHAT_TOOLS_BY_TYPE = {
    "flow": ["sing_line", "pulse_tag"],
    "network": ["echo_proof", "sing_line"],
    "glitch": ["glitch_tag", "sing_line"],
    "geo": ["anchor_register", "pulse_tag"],
    "portal": ["truth_gate", "sing_line"],
}

CLAIM_CUE_RE = re.compile(
    r"\b(should|will|obvious|clearly|must|need to|going to|plan to|we should|we will|it's obvious|it is obvious)\b",
    re.IGNORECASE,
)
COMMIT_RE = re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)\]}>\"']+", re.IGNORECASE)
FILE_RE = re.compile(r"\b[./~]?[-\w]+(?:/[-\w.]+)+\.[a-z0-9]{1,8}\b", re.IGNORECASE)
PR_RE = re.compile(r"\b(?:PR|pr)\s*#\d+\b")
OVERLAY_TAGS = ("[[PULSE]]", "[[GLITCH]]", "[[SING]]")


def _world_web_symbol(name: str, default: Any) -> Any:
    module = sys.modules.get("code.world_web")
    if module is None:
        return default
    return getattr(module, name, default)


def _compute_resource_from_backend(backend: str, *, device: str = "") -> str:
    backend_key = str(backend).strip().lower()
    device_key = str(device).strip().lower()
    if "npu" in backend_key or "npu" in device_key:
        return "npu"
    if "gpu" in backend_key or "gpu" in device_key:
        return "gpu"
    if backend_key in {"ollama", "vllm", "openai"}:
        return "gpu"
    if backend_key in {"tensorflow", "cpu"}:
        return "cpu"
    return "cpu"


def _compute_emitter_presence_id(resource: str) -> str:
    key = str(resource).strip().lower()
    if key == "gpu":
        return "health_sentinel_gpu1"
    if key == "npu":
        return "health_sentinel_npu0"
    return "health_sentinel_cpu"


def _record_compute_job(
    *,
    kind: str,
    op: str,
    backend: str,
    model: str,
    status: str,
    latency_ms: float,
    target_presence_id: str = "",
    error: str = "",
    device: str = "",
) -> None:
    tracker = _world_web_symbol("_INFLUENCE_TRACKER", None)
    if tracker is None:
        return
    recorder = getattr(tracker, "record_compute_job", None)
    if not callable(recorder):
        return

    resource = _compute_resource_from_backend(backend, device=device)
    try:
        recorder(
            kind=str(kind).strip().lower() or "unknown",
            op=str(op).strip().lower() or "job",
            backend=str(backend).strip().lower() or "unknown",
            resource=resource,
            emitter_presence_id=_compute_emitter_presence_id(resource),
            target_presence_id=str(target_presence_id).strip(),
            model=str(model).strip(),
            status=str(status).strip().lower() or "ok",
            latency_ms=max(0.0, float(latency_ms)),
            error=str(error).strip(),
        )
    except Exception:
        return


def _normalize_embedding_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    out = []
    for item in value:
        try:
            numeric = float(item)
            import math

            if not math.isfinite(numeric):
                return None
            out.append(numeric)
        except:
            return None
    return out


def _ollama_base_url() -> str:
    raw = str(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "").strip()
    raw = raw.rstrip("/")
    for suffix in ("/api/generate", "/api/embeddings", "/api/embed"):
        if raw.endswith(suffix):
            trimmed = raw[: -len(suffix)].rstrip("/")
            if trimmed:
                return trimmed
    return raw


def _ollama_endpoint() -> tuple[str, str, str, float]:
    base_url = _ollama_base_url()
    endpoint = f"{base_url}/api/generate"
    embeddings_endpoint = f"{base_url}/api/embeddings"
    model = str(
        os.getenv("OLLAMA_MODEL", "qwen3-vl:2b-instruct") or "qwen3-vl:2b-instruct"
    )
    timeout_s = float(os.getenv("OLLAMA_TIMEOUT_SEC", "30") or "30")
    return endpoint, embeddings_endpoint, model, timeout_s


def _embedding_payload_vector(raw: Any) -> list[float] | None:
    if isinstance(raw, dict):
        direct = _normalize_embedding_vector(raw.get("embedding"))
        if direct is not None:
            return direct
        embeddings = raw.get("embeddings")
        if isinstance(embeddings, list):
            for item in embeddings:
                vector = _normalize_embedding_vector(item)
                if vector is not None:
                    return vector
        data = raw.get("data")
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                vector = _normalize_embedding_vector(item.get("embedding"))
                if vector is not None:
                    return vector
    return None


def _ollama_embed_remote(text: str, model: str | None = None) -> list[float] | None:
    _, endpoint, _, default_timeout = _ollama_endpoint()
    embed_default_model = str(
        os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text") or "nomic-embed-text"
    ).strip()
    force_nomic = str(
        os.getenv("OLLAMA_EMBED_FORCE_NOMIC", "0") or "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    chosen_model = (
        "nomic-embed-text"
        if force_nomic
        else (model or embed_default_model or "nomic-embed-text").strip()
    )

    raw_max_chars = str(os.getenv("OLLAMA_EMBED_MAX_CHARS", "0") or "0").strip()
    try:
        max_chars = int(float(raw_max_chars))
    except (TypeError, ValueError):
        max_chars = 0
    if max_chars < 0:
        max_chars = 0
    if max_chars > 64000:
        max_chars = 64000
    sample_text = text[:max_chars] if max_chars > 0 else text

    raw_num_ctx = str(os.getenv("OLLAMA_EMBED_NUM_CTX", "") or "").strip()
    embed_options: dict[str, Any] = {}
    if raw_num_ctx:
        try:
            num_ctx = int(float(raw_num_ctx))
            if num_ctx > 0:
                embed_options["num_ctx"] = max(128, min(8192, num_ctx))
        except (TypeError, ValueError):
            pass

    candidates: list[str] = [endpoint]
    if endpoint.endswith("/api/embeddings"):
        candidates.append(f"{endpoint[: -len('/api/embeddings')]}/api/embed")

    opener = _world_web_symbol("urlopen", urlopen)
    seen: set[str] = set()
    for candidate in candidates:
        target = candidate.strip()
        if not target or target in seen:
            continue
        seen.add(target)

        payload: dict[str, Any]
        if target.endswith("/api/embed"):
            payload = {
                "model": chosen_model,
                "input": sample_text,
            }
        else:
            payload = {
                "model": chosen_model,
                "prompt": sample_text,
            }
        if embed_options:
            payload["options"] = embed_options

        req = Request(
            target,
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )

        try:
            with opener(req, timeout=default_timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
                embedding = _embedding_payload_vector(raw)
                if embedding is not None:
                    return embedding
        except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            continue

    return None


def _embedding_backend() -> str:
    backend = str(os.getenv("EMBEDDINGS_BACKEND", "ollama") or "ollama").strip().lower()
    if backend in {"openvino", "ollama", "tensorflow", "auto", "torch"}:
        return backend
    return "ollama"


def _text_generation_backend() -> str:
    backend = (
        str(os.getenv("TEXT_GENERATION_BACKEND", "ollama") or "ollama").strip().lower()
    )
    if backend in {"ollama", "tensorflow", "auto"}:
        return backend
    return "ollama"


def _tensorflow_embed_hash_bins(model: str | None = None) -> int:
    if model:
        raw_model = str(model).strip().lower()
        if raw_model.startswith("tf-hash:"):
            maybe = raw_model.split(":", 1)[-1]
            try:
                value = int(maybe)
                return max(64, min(8192, value))
            except (TypeError, ValueError):
                pass
    raw = str(os.getenv("TF_EMBED_HASH_BINS", "512") or "512").strip()
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = 512
    return max(64, min(8192, value))


def _load_tensorflow_module() -> Any | None:
    with _TENSORFLOW_RUNTIME_LOCK:
        module = _TENSORFLOW_RUNTIME.get("module")
        if module is not None:
            return module
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        with _TENSORFLOW_RUNTIME_LOCK:
            _TENSORFLOW_RUNTIME["module"] = None
            _TENSORFLOW_RUNTIME["error"] = str(exc)
            _TENSORFLOW_RUNTIME["loaded_at"] = datetime.now(timezone.utc).isoformat()
            _TENSORFLOW_RUNTIME["version"] = ""
        return None

    with _TENSORFLOW_RUNTIME_LOCK:
        _TENSORFLOW_RUNTIME["module"] = tf
        _TENSORFLOW_RUNTIME["error"] = ""
        _TENSORFLOW_RUNTIME["loaded_at"] = datetime.now(timezone.utc).isoformat()
        _TENSORFLOW_RUNTIME["version"] = str(getattr(tf, "__version__", ""))
    return tf


def _tensorflow_runtime_status() -> dict[str, Any]:
    with _TENSORFLOW_RUNTIME_LOCK:
        return {
            "loaded": _TENSORFLOW_RUNTIME.get("module") is not None,
            "error": str(_TENSORFLOW_RUNTIME.get("error", "")),
            "loaded_at": str(_TENSORFLOW_RUNTIME.get("loaded_at", "")),
            "version": str(_TENSORFLOW_RUNTIME.get("version", "")),
        }


def _tensorflow_embed(text: str, model: str | None = None) -> list[float] | None:
    prompt = str(text or "").strip()
    if not prompt:
        return None
    loader = _world_web_symbol("_load_tensorflow_module", _load_tensorflow_module)
    tf = loader()
    if tf is None:
        return None
    bins = _tensorflow_embed_hash_bins(model=model)
    try:
        source = tf.constant([prompt], dtype=tf.string)
        lowered = tf.strings.lower(source)
        cleaned = tf.strings.regex_replace(lowered, r"[^a-z0-9_\-\s]+", " ")
        cleaned = tf.strings.regex_replace(cleaned, r"\s+", " ")
        tokens = tf.strings.split(cleaned)
        hashed = tf.strings.to_hash_bucket_fast(tokens, bins)
        flat = getattr(hashed, "flat_values", hashed)
        indices = tf.cast(flat, tf.int32)
        counts = tf.math.bincount(
            indices,
            minlength=bins,
            maxlength=bins,
            dtype=tf.float32,
        )
        magnitude = tf.linalg.norm(counts)
        normalized = tf.cond(
            tf.greater(magnitude, tf.constant(0.0, dtype=tf.float32)),
            lambda: counts / magnitude,
            lambda: counts,
        )
        return _normalize_embedding_vector(normalized.numpy().tolist())
    except Exception:
        return None


def _vector_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    score = 0.0
    for i in range(n):
        score += a[i] * b[i]
    return float(score)


def _clean_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_-]+", text.lower()) if token]


def _tensorflow_generation_candidates() -> list[str]:
    candidates = [
        "Gate remains blocked until open questions are resolved with receipt-backed proof refs.",
        "Council approves only after overlap quorum and gate checks both pass.",
        "Small lawful move: resolve one open question and append a receipt line with intent refs.",
        "File Sentinel says: capture concrete file deltas and map them to affected presences.",
        "Witness Thread says: move eta claims toward mu by attaching artifact paths and deterministic checks.",
        "Keeper of Contracts says: drift is reduced by closing unresolved questions with explicit decisions.",
        "File Organizer says: cluster related files and keep concept memberships coherent.",
        "Receipt River says: no change is true until it flows through append-only receipts.",
    ]
    for row in VOICE_LINE_BANK:
        line = str(row.get("line_en", "")).strip()
        if line:
            candidates.append(line)
    deduped: list[str] = []
    seen: set[str] = set()
    for line in candidates:
        key = line.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _tensorflow_guidance_line(prompt: str) -> str:
    tokens = set(_clean_tokens(prompt))
    guidance: list[str] = []
    if {"gate", "blocked", "truth", "push", "approval", "approve"} & tokens:
        guidance.append(
            "Gate repair path: close unresolved open-question items, then record receipts with intent refs."
        )
    if {"council", "vote", "quorum", "overlap"} & tokens:
        guidance.append(
            "Council path: verify overlap-boundary members and required yes votes before action."
        )
    if {"drift", "repair", "fix", "stability"} & tokens:
        guidance.append(
            "Stability path: run drift scan, address blocked reasons, and re-check study snapshot."
        )
    if {"witness", "proof", "receipt", "artifact"} & tokens:
        guidance.append(
            "Proof path: link artifacts, receipts, and manifest refs so claims become verifiable."
        )
    return " ".join(guidance[:2]).strip()


def _tensorflow_generate_text(
    prompt: str,
    model: str | None = None,
    timeout_s: float | None = None,
) -> tuple[str | None, str]:
    del timeout_s
    query = str(prompt or "").strip()
    if not query:
        return None, "tensorflow-hash-v1"

    embed_fn = _world_web_symbol("_tensorflow_embed", _tensorflow_embed)
    query_vec = embed_fn(query, model=model)
    if query_vec is None:
        return None, "tensorflow-hash-v1"

    best_line = ""
    best_score = -1.0
    for candidate in _tensorflow_generation_candidates():
        candidate_vec = embed_fn(candidate, model=model)
        if candidate_vec is None:
            continue
        score = _vector_similarity(query_vec, candidate_vec)
        if score > best_score:
            best_score = score
            best_line = candidate

    if not best_line:
        return None, "tensorflow-hash-v1"

    guidance = _tensorflow_guidance_line(query)
    output = best_line
    if guidance:
        output = f"{best_line} {guidance}"
    return output.strip(), "tensorflow-hash-v1"


def _embedding_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "true" if default else "false") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _safe_embedding_int_env(
    name: str, default: int, *, min_value: int, max_value: int
) -> int:
    raw = str(os.getenv(name, str(default)) or str(default)).strip()
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _safe_embedding_timeout_env(name: str, default: float) -> float:
    raw = str(os.getenv(name, str(default)) or str(default)).strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(default)
    return max(0.2, min(120.0, value))


def _normalize_openvino_device(raw: str | None) -> str:
    value = str(raw or "").strip().upper()
    if not value:
        return "NPU"
    if re.fullmatch(r"[A-Z0-9:_,-]{2,48}", value):
        return value
    return "NPU"


def _openvino_auth_header_name() -> str:
    raw_header = str(os.getenv("OPENVINO_EMBED_AUTH_HEADER", "") or "").strip()
    if raw_header:
        if ":" in raw_header:
            return raw_header.split(":", 1)[0].strip()
        return "Authorization"

    bearer = str(os.getenv("OPENVINO_EMBED_BEARER_TOKEN", "") or "").strip()
    if bearer:
        return "Authorization"

    api_key = str(os.getenv("OPENVINO_EMBED_API_KEY", "") or "").strip()
    if api_key:
        name = str(
            os.getenv("OPENVINO_EMBED_API_KEY_HEADER", "X-API-Key") or "X-API-Key"
        ).strip()
        return name or "X-API-Key"

    return ""


def _openvino_auth_mode() -> str:
    raw_header = str(os.getenv("OPENVINO_EMBED_AUTH_HEADER", "") or "").strip()
    if raw_header:
        return "custom"
    if str(os.getenv("OPENVINO_EMBED_BEARER_TOKEN", "") or "").strip():
        return "bearer"
    if str(os.getenv("OPENVINO_EMBED_API_KEY", "") or "").strip():
        return "api_key"
    return "none"


def _openvino_embed_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}

    raw_header = str(os.getenv("OPENVINO_EMBED_AUTH_HEADER", "") or "").strip()
    if raw_header:
        if ":" in raw_header:
            key, value = raw_header.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                headers[key] = value
                return headers
        else:
            value = raw_header.strip()
            if value:
                headers["Authorization"] = value
                return headers

    bearer = str(os.getenv("OPENVINO_EMBED_BEARER_TOKEN", "") or "").strip()
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
        return headers

    api_key = str(os.getenv("OPENVINO_EMBED_API_KEY", "") or "").strip()
    if api_key:
        header_name = str(
            os.getenv("OPENVINO_EMBED_API_KEY_HEADER", "X-API-Key") or "X-API-Key"
        ).strip()
        if not header_name:
            header_name = "X-API-Key"
        headers[header_name] = api_key

    return headers


def _normalize_embedding_runtime_vector(
    values: list[float], normalize: bool
) -> list[float]:
    if not normalize:
        return values
    magnitude = math.sqrt(sum(item * item for item in values))
    if magnitude <= 0.0:
        return values
    return [item / magnitude for item in values]


def _record_openvino_runtime(
    *,
    key: str,
    model: str | None,
    error: str,
) -> None:
    with _OPENVINO_EMBED_LOCK:
        _OPENVINO_EMBED_RUNTIME["key"] = key
        _OPENVINO_EMBED_RUNTIME["tokenizer"] = None
        _OPENVINO_EMBED_RUNTIME["model"] = model
        _OPENVINO_EMBED_RUNTIME["error"] = error
        _OPENVINO_EMBED_RUNTIME["loaded_at"] = datetime.now(timezone.utc).isoformat()


def _openvino_embed_candidates(endpoint: str) -> list[str]:
    base = endpoint.rstrip("/")
    if not base:
        return []
    candidates = [base]
    if base.endswith("/api/embeddings"):
        candidates.append(f"{base[: -len('/api/embeddings')]}/api/embed")
        candidates.append(f"{base[: -len('/api/embeddings')]}/v1/embeddings")
    elif base.endswith("/api/embed"):
        candidates.append(f"{base[: -len('/api/embed')]}/api/embeddings")
        candidates.append(f"{base[: -len('/api/embed')]}/v1/embeddings")
    elif base.endswith("/v1/embeddings"):
        candidates.append(f"{base[: -len('/v1/embeddings')]}/api/embeddings")
        candidates.append(f"{base[: -len('/v1/embeddings')]}/api/embed")
    else:
        candidates.extend(
            [
                f"{base}/api/embeddings",
                f"{base}/api/embed",
                f"{base}/v1/embeddings",
            ]
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _embedding_provider_options() -> dict[str, Any]:
    return {
        "ok": True,
        "record": "ημ.embedding-provider.options.v1",
        "config": {
            "backend": _embedding_backend(),
            "text_generation_backend": _text_generation_backend(),
            "ollama_base_url": _ollama_base_url(),
            "ollama_embed_model": str(
                os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
                or "nomic-embed-text"
            ).strip(),
            "ollama_embed_force_nomic": _embedding_flag(
                "OLLAMA_EMBED_FORCE_NOMIC", False
            ),
            "ollama_embed_num_ctx": _safe_embedding_int_env(
                "OLLAMA_EMBED_NUM_CTX",
                512,
                min_value=128,
                max_value=8192,
            ),
            "ollama_embed_max_chars": _safe_embedding_int_env(
                "OLLAMA_EMBED_MAX_CHARS",
                2400,
                min_value=0,
                max_value=64000,
            ),
            "openvino_endpoint": str(
                os.getenv("OPENVINO_EMBED_ENDPOINT", "") or ""
            ).strip(),
            "openvino_model": str(os.getenv("OPENVINO_EMBED_MODEL", "") or "").strip(),
            "openvino_device": _normalize_openvino_device(
                os.getenv("OPENVINO_EMBED_DEVICE", "NPU")
            ),
            "openvino_timeout_sec": _safe_embedding_timeout_env(
                "OPENVINO_EMBED_TIMEOUT_SEC", 12.0
            ),
            "openvino_auth_mode": _openvino_auth_mode(),
            "openvino_auth_header_name": _openvino_auth_header_name(),
            "openvino_api_key_header": str(
                os.getenv("OPENVINO_EMBED_API_KEY_HEADER", "X-API-Key") or "X-API-Key"
            ).strip()
            or "X-API-Key",
        },
        "presets": {
            "gpu_local": {
                "backend": "ollama",
                "description": "Use local Ollama embeddings (GPU when available).",
            },
            "npu_local": {
                "backend": "openvino",
                "openvino_device": "NPU",
                "description": "Use OpenVINO embedding endpoint on NPU.",
            },
            "hybrid_auto": {
                "backend": "auto",
                "openvino_device": "NPU",
                "description": "Prefer NPU OpenVINO, then local fallbacks.",
            },
        },
    }


def _apply_embedding_provider_options(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"ok": False, "error": "payload must be object"}

    req = dict(payload)
    preset = str(req.get("preset", "") or "").strip().lower()
    if preset in {"gpu", "gpu_local", "local_gpu"}:
        req.setdefault("backend", "ollama")
        req.setdefault("ollama_embed_force_nomic", True)
    elif preset in {"npu", "npu_local", "local_npu"}:
        req.setdefault("backend", "openvino")
        req.setdefault("openvino_device", "NPU")
        req.setdefault("ollama_embed_force_nomic", True)
    elif preset in {"hybrid", "auto", "hybrid_auto", "local_auto"}:
        req.setdefault("backend", "auto")
        req.setdefault("openvino_device", "NPU")

    backend_raw = req.get("backend")
    if backend_raw is not None:
        backend = str(backend_raw).strip().lower()
        if backend not in {"openvino", "ollama", "tensorflow", "auto"}:
            return {
                "ok": False,
                "error": "invalid backend (expected openvino|ollama|tensorflow|auto)",
            }
        os.environ["EMBEDDINGS_BACKEND"] = backend

    if req.get("ollama_base_url") is not None:
        base_url = str(req.get("ollama_base_url") or "").strip().rstrip("/")
        if base_url:
            os.environ["OLLAMA_BASE_URL"] = base_url

    if req.get("ollama_embed_model") is not None:
        model = str(req.get("ollama_embed_model") or "").strip()
        if model:
            os.environ["OLLAMA_EMBED_MODEL"] = model

    if req.get("ollama_embed_force_nomic") is not None:
        force = bool(req.get("ollama_embed_force_nomic"))
        os.environ["OLLAMA_EMBED_FORCE_NOMIC"] = "1" if force else "0"

    if req.get("ollama_embed_num_ctx") is not None:
        try:
            ctx = int(float(str(req.get("ollama_embed_num_ctx"))))
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid ollama_embed_num_ctx"}
        os.environ["OLLAMA_EMBED_NUM_CTX"] = str(max(128, min(8192, ctx)))

    if req.get("ollama_embed_max_chars") is not None:
        try:
            max_chars = int(float(str(req.get("ollama_embed_max_chars"))))
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid ollama_embed_max_chars"}
        os.environ["OLLAMA_EMBED_MAX_CHARS"] = str(max(0, min(64000, max_chars)))

    if req.get("openvino_endpoint") is not None:
        endpoint = str(req.get("openvino_endpoint") or "").strip().rstrip("/")
        os.environ["OPENVINO_EMBED_ENDPOINT"] = endpoint

    if req.get("openvino_model") is not None:
        openvino_model = str(req.get("openvino_model") or "").strip()
        os.environ["OPENVINO_EMBED_MODEL"] = openvino_model

    if req.get("openvino_device") is not None:
        os.environ["OPENVINO_EMBED_DEVICE"] = _normalize_openvino_device(
            str(req.get("openvino_device") or "")
        )

    if req.get("openvino_timeout_sec") is not None:
        try:
            timeout_s = float(str(req.get("openvino_timeout_sec")))
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid openvino_timeout_sec"}
        os.environ["OPENVINO_EMBED_TIMEOUT_SEC"] = str(max(0.2, min(120.0, timeout_s)))

    if req.get("openvino_auth_header") is not None:
        os.environ["OPENVINO_EMBED_AUTH_HEADER"] = str(
            req.get("openvino_auth_header") or ""
        ).strip()

    if req.get("openvino_bearer_token") is not None:
        os.environ["OPENVINO_EMBED_BEARER_TOKEN"] = str(
            req.get("openvino_bearer_token") or ""
        ).strip()

    if req.get("openvino_api_key") is not None:
        os.environ["OPENVINO_EMBED_API_KEY"] = str(
            req.get("openvino_api_key") or ""
        ).strip()

    if req.get("openvino_api_key_header") is not None:
        header_name = str(req.get("openvino_api_key_header") or "").strip()
        os.environ["OPENVINO_EMBED_API_KEY_HEADER"] = header_name or "X-API-Key"

    _record_openvino_runtime(key="", model=None, error="")

    return {
        "ok": True,
        "record": "ημ.embedding-provider.options.v1",
        "applied": {
            "preset": preset,
            "backend": _embedding_backend(),
        },
        "provider": _embedding_provider_status(),
        "options": _embedding_provider_options(),
    }


def _ollama_generate_text_remote(
    prompt: str,
    model: str | None = None,
    timeout_s: float | None = None,
) -> tuple[str | None, str]:
    endpoint, _, default_model, default_timeout = _ollama_endpoint()
    chosen_model = model or default_model
    chosen_timeout = timeout_s or default_timeout
    payload = {
        "model": chosen_model,
        "prompt": prompt,
        "stream": False,
    }
    req = Request(
        endpoint,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    opener = _world_web_symbol("urlopen", urlopen)
    try:
        with opener(req, timeout=chosen_timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None, chosen_model

    text = str(raw.get("response", "")).strip()
    return (text or None), chosen_model


def _ollama_generate_text(
    prompt: str,
    model: str | None = None,
    timeout_s: float | None = None,
) -> tuple[str | None, str]:
    started = time.perf_counter()
    backend_fn = _world_web_symbol("_text_generation_backend", _text_generation_backend)
    backend = str(backend_fn()).strip().lower()
    tf_generate = _world_web_symbol(
        "_tensorflow_generate_text", _tensorflow_generate_text
    )
    remote_generate = _world_web_symbol(
        "_ollama_generate_text_remote", _ollama_generate_text_remote
    )

    if backend == "tensorflow":
        text, chosen_model = tf_generate(prompt, model=model, timeout_s=timeout_s)
        _record_compute_job(
            kind="llm",
            op="text_generate.tensorflow",
            backend="tensorflow",
            model=chosen_model,
            status="ok" if text else "error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            target_presence_id="witness_thread",
            error="" if text else "tensorflow_no_text",
        )
        return text, chosen_model
    if backend == "auto":
        chosen_model = model or "auto"
        last_backend = "auto"
        order_fn = _world_web_symbol(
            "_resource_auto_text_order", _resource_auto_text_order
        )
        for candidate in order_fn():
            last_backend = str(candidate).strip().lower() or "auto"
            if candidate == "tensorflow":
                text, chosen_model = tf_generate(
                    prompt, model=model, timeout_s=timeout_s
                )
            else:
                text, chosen_model = remote_generate(
                    prompt,
                    model=model,
                    timeout_s=timeout_s,
                )
            if text:
                _record_compute_job(
                    kind="llm",
                    op=f"text_generate.{last_backend}",
                    backend=last_backend,
                    model=chosen_model,
                    status="ok",
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    target_presence_id="witness_thread",
                )
                return text, chosen_model
        _record_compute_job(
            kind="llm",
            op=f"text_generate.{last_backend}",
            backend=last_backend,
            model=chosen_model,
            status="error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            target_presence_id="witness_thread",
            error="auto_route_no_text",
        )
        return None, chosen_model
    text, chosen_model = remote_generate(prompt, model=model, timeout_s=timeout_s)
    _record_compute_job(
        kind="llm",
        op="text_generate.ollama",
        backend="ollama",
        model=chosen_model,
        status="ok" if text else "error",
        latency_ms=(time.perf_counter() - started) * 1000.0,
        target_presence_id="witness_thread",
        error="" if text else "ollama_no_text",
    )
    return text, chosen_model


def _ollama_embed(text: str, model: str | None = None) -> list[float] | None:
    started = time.perf_counter()
    backend_fn = _world_web_symbol("_embedding_backend", _embedding_backend)
    backend = str(backend_fn()).strip().lower()
    tf_embed = _world_web_symbol("_tensorflow_embed", _tensorflow_embed)
    openvino_embed = _world_web_symbol("_openvino_embed", _openvino_embed)
    remote_embed = _world_web_symbol("_ollama_embed_remote", _ollama_embed_remote)

    if backend == "tensorflow":
        vector = tf_embed(text, model=model)
        _record_compute_job(
            kind="embedding",
            op="embed.tensorflow",
            backend="tensorflow",
            model=str(model or ""),
            status="ok" if vector is not None else "error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            target_presence_id="file_organizer",
            error="" if vector is not None else "tensorflow_no_vector",
        )
        return vector
    if backend == "openvino":
        vector = openvino_embed(text, model=model)
        _record_compute_job(
            kind="embedding",
            op="embed.openvino",
            backend="openvino",
            model=str(model or ""),
            status="ok" if vector is not None else "error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            target_presence_id="file_organizer",
            error="" if vector is not None else "openvino_no_vector",
            device=str(os.getenv("OPENVINO_EMBED_DEVICE", "NPU") or "NPU"),
        )
        return vector
    if backend == "auto":
        order_fn = _world_web_symbol(
            "_resource_auto_embedding_order", _resource_auto_embedding_order
        )
        last_backend = "auto"
        chosen_model = str(model or "")
        for candidate in order_fn():
            last_backend = str(candidate).strip().lower() or "auto"
            if candidate == "openvino":
                vector = openvino_embed(text, model=model)
            elif candidate == "tensorflow":
                vector = tf_embed(text, model=model)
            else:
                vector = remote_embed(text, model=model)
            if vector is not None:
                if not chosen_model:
                    if last_backend == "openvino":
                        chosen_model = str(
                            os.getenv("OPENVINO_EMBED_MODEL", "")
                            or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
                            or "nomic-embed-text"
                        ).strip()
                    else:
                        chosen_model = str(
                            model
                            or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
                            or "nomic-embed-text"
                        ).strip()
                _record_compute_job(
                    kind="embedding",
                    op=f"embed.{last_backend}",
                    backend=last_backend,
                    model=chosen_model,
                    status="ok",
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    target_presence_id="file_organizer",
                    device=(
                        str(os.getenv("OPENVINO_EMBED_DEVICE", "NPU") or "NPU")
                        if last_backend == "openvino"
                        else ""
                    ),
                )
                return vector
        _record_compute_job(
            kind="embedding",
            op=f"embed.{last_backend}",
            backend=last_backend,
            model=chosen_model,
            status="error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            target_presence_id="file_organizer",
            error="auto_route_no_vector",
            device=(
                str(os.getenv("OPENVINO_EMBED_DEVICE", "NPU") or "NPU")
                if last_backend == "openvino"
                else ""
            ),
        )
        return None
    vector = remote_embed(text, model=model)
    chosen_model = str(
        model
        or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        or "nomic-embed-text"
    ).strip()
    _record_compute_job(
        kind="embedding",
        op="embed.ollama",
        backend="ollama",
        model=chosen_model,
        status="ok" if vector is not None else "error",
        latency_ms=(time.perf_counter() - started) * 1000.0,
        target_presence_id="file_organizer",
        error="" if vector is not None else "ollama_no_vector",
    )
    return vector


def _embedding_provider_status() -> dict[str, Any]:
    with _OPENVINO_EMBED_LOCK:
        runtime_snapshot = {
            "loaded": _OPENVINO_EMBED_RUNTIME.get("model") is not None,
            "error": str(_OPENVINO_EMBED_RUNTIME.get("error", "")),
            "loaded_at": str(_OPENVINO_EMBED_RUNTIME.get("loaded_at", "")),
            "key": str(_OPENVINO_EMBED_RUNTIME.get("key", "")),
        }
    resource_snapshot = _resource_monitor_snapshot()

    return {
        "backend": _embedding_backend(),
        "text_generation_backend": _text_generation_backend(),
        "openvino": {
            "endpoint": str(os.getenv("OPENVINO_EMBED_ENDPOINT", "") or "").strip(),
            "model": str(os.getenv("OPENVINO_EMBED_MODEL", "") or "").strip(),
            "device": str(os.getenv("OPENVINO_EMBED_DEVICE", "NPU") or "NPU").strip()
            or "NPU",
            "normalize": _embedding_flag("OPENVINO_EMBED_NORMALIZE", True),
            "auth_mode": _openvino_auth_mode(),
            "auth_header_name": _openvino_auth_header_name(),
            "runtime": runtime_snapshot,
        },
        "tensorflow": {
            "hash_bins": _tensorflow_embed_hash_bins(),
            "runtime": _tensorflow_runtime_status(),
        },
        "resource": {
            "record": str(resource_snapshot.get("record", "")),
            "generated_at": str(resource_snapshot.get("generated_at", "")),
            "devices": resource_snapshot.get("devices", {}),
            "hot_devices": resource_snapshot.get("hot_devices", []),
            "auto_backend": resource_snapshot.get("auto_backend", {}),
            "log_watch": resource_snapshot.get("log_watch", {}),
        },
    }


def build_voice_lines(mode: str = "canonical") -> dict[str, Any]:
    return {
        "mode": "canonical",
        "model": None,
        "lines": VOICE_LINE_BANK,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _chat_fallback_reply(user_text: str) -> str:
    normalized = user_text.lower()
    if "receipt" in normalized or "river" in normalized:
        return "Receipt River answers: persistence is a song that keeps receipts alive."
    if "fork" in normalized or "tax" in normalized:
        return "Fork Tax Canticle answers: we do not punish choice, we prove we chose."
    if "anchor" in normalized:
        return "Anchor Registry answers: hold drift with coordinates, not chains."
    if "witness" in normalized or "proof" in normalized:
        return "Witness Thread answers: proof is a thread you keep pulling until night admits it."
    return "Gates of Truth answers: speak your line, and we will annotate the flame without erasing it."


def _entity_lookup() -> dict[str, dict[str, Any]]:
    lookup = {
        str(item.get("id", "")): item
        for item in ENTITY_MANIFEST
        if item.get("id") and item.get("id") != "core_pulse"
    }
    for profile in (
        KEEPER_OF_CONTRACTS_PROFILE,
        FILE_SENTINEL_PROFILE,
        FILE_ORGANIZER_PROFILE,
        THE_COUNCIL_PROFILE,
        HEALTH_SENTINEL_CPU_PROFILE,
        HEALTH_SENTINEL_GPU1_PROFILE,
        HEALTH_SENTINEL_GPU2_PROFILE,
        HEALTH_SENTINEL_NPU0_PROFILE,
    ):
        profile_id = str(profile.get("id", "")).strip()
        if profile_id and profile_id not in lookup:
            lookup[profile_id] = profile
    return lookup


def _extract_overlay_tags(text: str) -> list[str]:
    found = []
    for tag in OVERLAY_TAGS:
        if tag in text:
            found.append(tag)
    return found


def _apply_presence_tool(tool_name: str, line: str, user_text: str) -> str:
    if tool_name == "sing_line":
        return f"♪ {line} ♪"
    if tool_name == "pulse_tag":
        return f"[[PULSE]] {line}"
    if tool_name == "glitch_tag":
        return f"[[GLITCH]] {line}"
    if tool_name == "echo_proof":
        return f"{line} / prove it with artifacts."
    if tool_name == "anchor_register":
        return f"{line} / Anchor Registry keeps drift bounded."
    if tool_name == "truth_gate":
        return f"{line} / Gates of Truth remain append-only."
    if tool_name == "quote_user":
        return f"{line} / heard: {user_text.strip()[:72]}"
    return line


def _canonical_presence_line(
    entity: dict[str, Any], user_text: str, turn_index: int
) -> str:
    en = str(entity.get("en", "Unknown Presence"))
    ja = str(entity.get("ja", "場"))
    lower = user_text.lower()
    if "fork" in lower or "tax" in lower:
        return f"{en} / {ja}: pay the fork tax, then witness the path."
    if "anchor" in lower:
        return f"{en} / {ja}: anchor first, then sing the decision."
    if turn_index == 0:
        return f"{en} / {ja}: I receive your line and hold it in the ledger."
    return f"{en} / {ja}: I answer the previous voice with witness-light."


def _normalize_presence_id(raw_presence_id: str) -> str:
    raw = raw_presence_id.strip()
    if not raw:
        return "witness_thread"
    normalized = raw.lower().replace(" ", "_")
    alias = _PRESENCE_ALIASES.get(raw) or _PRESENCE_ALIASES.get(normalized)
    if alias:
        return alias
    return raw


def _resolve_presence_entity(requested_presence_id: str) -> tuple[str, dict[str, Any]]:
    presence_id = _normalize_presence_id(requested_presence_id)
    entity = next(
        (item for item in ENTITY_MANIFEST if str(item.get("id")) == presence_id),
        None,
    )
    if entity is not None:
        return presence_id, entity
    if presence_id == "keeper_of_contracts":
        return presence_id, KEEPER_OF_CONTRACTS_PROFILE
    if presence_id == "file_sentinel":
        return presence_id, FILE_SENTINEL_PROFILE
    if presence_id == "the_council":
        return presence_id, THE_COUNCIL_PROFILE
    if presence_id == "file_organizer":
        return presence_id, FILE_ORGANIZER_PROFILE
    if presence_id == "health_sentinel_cpu":
        return presence_id, HEALTH_SENTINEL_CPU_PROFILE
    if presence_id == "health_sentinel_gpu1":
        return presence_id, HEALTH_SENTINEL_GPU1_PROFILE
    if presence_id == "health_sentinel_gpu2":
        return presence_id, HEALTH_SENTINEL_GPU2_PROFILE
    if presence_id == "health_sentinel_npu0":
        return presence_id, HEALTH_SENTINEL_NPU0_PROFILE
    fallback = ENTITY_MANIFEST[0]
    return str(fallback.get("id", "witness_thread")), fallback


def _select_chat_entities(
    user_text: str,
    presence_ids: list[str] | None,
    multi_entity: bool,
) -> list[dict[str, Any]]:
    lookup = _entity_lookup()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    if presence_ids:
        for item in presence_ids:
            key = _normalize_presence_id(str(item))
            if key in lookup:
                entity = lookup[key]
                entity_id = str(entity.get("id", "")).strip()
                if entity_id and entity_id in selected_ids:
                    continue
                selected.append(entity)
                if entity_id:
                    selected_ids.add(entity_id)
        if selected:
            return selected[:3]

    normalized = user_text.lower()
    hints = [
        ("anchor", "anchor_registry"),
        ("witness", "witness_thread"),
        ("receipt", "receipt_river"),
        ("river", "receipt_river"),
        ("fork", "fork_tax_canticle"),
        ("tax", "fork_tax_canticle"),
        ("truth", "gates_of_truth"),
        ("cpu", "health_sentinel_cpu"),
        ("gpu1", "health_sentinel_gpu1"),
        ("gpu2", "health_sentinel_gpu2"),
        ("npu", "health_sentinel_npu0"),
        ("resource", "health_sentinel_cpu"),
        ("heartbeat", "health_sentinel_cpu"),
    ]
    for token, key in hints:
        if token in normalized and key in lookup:
            entity = lookup[key]
            entity_id = str(entity.get("id", "")).strip()
            if entity_id and entity_id in selected_ids:
                continue
            selected.append(entity)
            if entity_id:
                selected_ids.add(entity_id)

    if not selected:
        fallback_order = ["receipt_river", "witness_thread", "gates_of_truth"]
        for key in fallback_order:
            if key in lookup:
                entity = lookup[key]
                entity_id = str(entity.get("id", "")).strip()
                if entity_id and entity_id in selected_ids:
                    continue
                selected.append(entity)
                if entity_id:
                    selected_ids.add(entity_id)

    max_entities = 3 if multi_entity else 1
    return selected[:max_entities]


def _presence_prompt(
    entity: dict[str, Any],
    allowed_tools: list[str],
    history_text: str,
    prior_turns: list[dict[str, Any]],
    context_block: str,
) -> str:
    prior_lines = "\n".join(
        f"- {item.get('presence_id', 'unknown')}: {item.get('output', '')}"
        for item in prior_turns
    )
    prior_lines = prior_lines or "(none)"

    return (
        "You are one presence in the eta-mu world daemon.\n"
        f"Presence: {entity.get('en')} / {entity.get('ja')}\n"
        f"Allowed tools: {', '.join(allowed_tools) if allowed_tools else 'none'}\n"
        "Write 1-2 short lines, bilingual EN/JA style.\n"
        "Use trigger tags only if needed: [[PULSE]] [[GLITCH]] [[SING]].\n"
        f"Context:\n{context_block}\n"
        f"Conversation:\n{history_text}\n"
        f"Prior presence turns:\n{prior_lines}\n"
        "Response:"
    )


def _presence_generate_utterance(
    *,
    mode: str,
    entity: dict[str, Any],
    presence_id: str,
    allowed_tools: list[str],
    history_text: str,
    prior_turns: list[dict[str, Any]],
    context_block: str,
    base_text: str,
    generate_text_fn: Any,
):
    if mode != "ollama":
        return base_text, "canonical", "ok", None, None

    prompt = _presence_prompt(
        entity,
        allowed_tools,
        history_text,
        prior_turns,
        context_block,
    )
    generated, model_name = generate_text_fn(prompt)
    if generated:
        return generated, "ollama", "ok", model_name, None

    return (
        base_text,
        "canonical",
        "fallback",
        None,
        {
            "presence_id": presence_id,
            "error_code": "ollama_unavailable",
            "fallback_used": True,
        },
    )


def _presence_apply_tools(
    *,
    composed: str,
    user_text: str,
    allowed_tools: list[str],
) -> tuple[str, list[dict[str, str]]]:
    tool_outputs: list[dict[str, str]] = []
    next_text = composed

    for tool_name in allowed_tools[:2]:
        updated = _apply_presence_tool(tool_name, next_text, user_text)
        if updated == next_text:
            continue
        tool_outputs.append(
            {
                "name": tool_name,
                "output": updated,
            }
        )
        next_text = updated

    return next_text, tool_outputs


def _compose_presence_response(
    trimmed: list[dict[str, str]],
    mode: str,
    context: dict[str, Any] | None,
    presence_ids: list[str] | None,
    multi_entity: bool,
    generate_text_fn: Any,
) -> dict[str, Any]:
    user_text = trimmed[-1]["text"]
    entities = _select_chat_entities(user_text, presence_ids, multi_entity)

    context_lines = []
    if context:
        context_lines.append(f"Files: {context.get('file_count', 0)}")
        roles = context.get("roles", {})
        if isinstance(roles, dict) and roles:
            context_lines.append(
                "Roles: " + ", ".join(f"{k}:{v}" for k, v in roles.items())
            )
        constraints = context.get("constraints", [])
        if isinstance(constraints, list) and constraints:
            context_lines.append(
                "Constraints: " + " | ".join(str(c) for c in constraints[-3:])
            )
    context_block = "\n".join(context_lines) if context_lines else "(no context)"

    history_text = "\n".join(f"{row['role']}: {row['text']}" for row in trimmed)
    turns: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    models: list[str] = []
    final_lines: list[str] = []

    for idx, entity in enumerate(entities):
        presence_id = str(entity.get("id", "unknown"))
        allowed_tools = list(
            CHAT_TOOLS_BY_TYPE.get(str(entity.get("type", "")), ["quote_user"])
        )
        base_text = _canonical_presence_line(entity, user_text, idx)
        composed_seed, chosen_mode, status, model_name, failure = (
            _presence_generate_utterance(
                mode=mode,
                entity=entity,
                presence_id=presence_id,
                allowed_tools=allowed_tools,
                history_text=history_text,
                prior_turns=turns,
                context_block=context_block,
                base_text=base_text,
                generate_text_fn=generate_text_fn,
            )
        )
        if model_name is not None:
            models.append(model_name)
        if failure is not None:
            failures.append(failure)

        composed, tool_outputs = _presence_apply_tools(
            composed=composed_seed.strip(),
            user_text=user_text,
            allowed_tools=allowed_tools,
        )

        turn = {
            "presence_id": presence_id,
            "presence_en": entity.get("en", "Unknown"),
            "presence_ja": entity.get("ja", "未知"),
            "mode": chosen_mode,
            "model": model_name,
            "status": status,
            "tools": [item["name"] for item in tool_outputs],
            "tool_outputs": tool_outputs,
            "output": composed,
        }
        turns.append(turn)
        final_lines.append(composed)

    final_lines = final_lines or [_chat_fallback_reply(user_text)]

    combined_reply = "\n".join(final_lines)
    overlay_tags = _extract_overlay_tags(combined_reply)

    return {
        "mode": "ollama" if models else "canonical",
        "model": models[0] if models else None,
        "reply": combined_reply,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trace": {
            "version": "v1",
            "multi_entity": multi_entity,
            "entities": turns,
            "overlay_tags": overlay_tags,
            "failures": failures,
        },
    }


def detect_artifact_refs(utterance: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for regex in (URL_RE, FILE_RE, PR_RE, COMMIT_RE):
        for match in regex.finditer(utterance):
            token = match.group(0)
            if token not in seen:
                refs.append(token)
                seen.add(token)
    return refs


def analyze_utterance(
    utterance: str,
    idx: int,
    now: datetime | None = None,
) -> dict[str, Any]:
    text = utterance.strip()
    artifact_refs = detect_artifact_refs(text)
    eta_claim = CLAIM_CUE_RE.search(text) is not None
    mu_proof = len(artifact_refs) > 0
    if mu_proof:
        classification = "mu-proof"
    elif eta_claim:
        classification = "eta-claim"
    else:
        classification = "neutral"

    ts = (now or datetime.now(timezone.utc)).isoformat()
    return {
        "ts": ts,
        "idx": idx,
        "utterance": text,
        "classification": classification,
        "eta_claim": eta_claim,
        "mu_proof": mu_proof,
        "artifact_refs": artifact_refs,
    }


def utterances_to_ledger_rows(
    utterances: list[str],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in utterances:
        text = str(item).strip()
        if not text:
            continue
        rows.append(analyze_utterance(text, len(rows) + 1, now))
    return rows


def build_chat_reply(
    messages: list[dict[str, Any]],
    mode: str = "ollama",
    context: dict[str, Any] | None = None,
    multi_entity: bool = False,
    presence_ids: list[str] | None = None,
    generate_text_fn: Any = _ollama_generate_text,
) -> dict[str, Any]:
    trimmed = [
        {
            "role": str(item.get("role", "user")),
            "text": str(item.get("text", "")).strip(),
        }
        for item in messages[-10:]
        if str(item.get("text", "")).strip()
    ]
    if not trimmed:
        trimmed = [{"role": "user", "text": "Sing the field names."}]

    if multi_entity:
        return _compose_presence_response(
            trimmed,
            mode,
            context,
            presence_ids,
            multi_entity,
            generate_text_fn,
        )

    if mode == "ollama":
        user_query = trimmed[-1]["text"]
        history = "\n".join(f"{row['role']}: {row['text']}" for row in trimmed)

        ctx_lines = []

        collection_getter = _world_web_symbol("_get_chroma_collection", None)
        collection = collection_getter() if callable(collection_getter) else None
        if collection:
            query_embedding = _ollama_embed(user_query)
            if query_embedding:
                try:
                    collection_any: Any = collection
                    results = collection_any.query(
                        query_embeddings=[query_embedding], n_results=3
                    )
                    mems = results.get("documents", [[]])[0]
                    if mems:
                        ctx_lines.append("- Recovered Memory Fragments (Echoes):")
                        for m in mems:
                            ctx_lines.append(f"  * {m}")
                except Exception:
                    pass

        if context:
            files = context.get("file_count", 0)
            roles = context.get("roles", {})
            constraints = context.get("constraints", [])
            ctx_lines.append(f"- Files: {files}")
            ctx_lines.append(
                f"- Active Roles: {', '.join(f'{k}:{v}' for k, v in roles.items())}"
            )
            if constraints:
                ctx_lines.append("- Active Constraints:")
                for c in constraints[-5:]:
                    ctx_lines.append(f"  * {c}")

        context_block = (
            "\n".join(ctx_lines) if ctx_lines else "(No additional context available)"
        )

        link_density = 0
        if context:
            items = context.get("items", 0)
            if items > 0:
                link_density = context.get("file_count", 0) / items

        consolidation_block = ""
        if link_density > 0.8:
            consolidation_block = f"CONSOLIDATION STATUS: Active.\n{PART_67_PROLOGUE}"

        prompt = (
            f"{SYSTEM_PROMPT_TEMPLATE.format(context_block=context_block, consolidation_block=consolidation_block)}"
            + f"\n\nConversation:\n{history}\nAssistant:"
        )

        text, model = generate_text_fn(prompt)
        if text:
            return {
                "mode": "ollama",
                "model": model,
                "reply": text,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

    return {
        "mode": "canonical",
        "model": None,
        "reply": _chat_fallback_reply(trimmed[-1]["text"]),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_presence_say_payload(
    catalog: dict[str, Any],
    text: str,
    requested_presence_id: str,
) -> dict[str, Any]:
    clean_text = text.strip()
    tokens = _clean_tokens(clean_text)
    presence_id, entity = _resolve_presence_entity(requested_presence_id)
    concept_presence: dict[str, Any] | None = None
    file_graph = catalog.get("file_graph", {}) if isinstance(catalog, dict) else {}
    concept_rows = (
        file_graph.get("concept_presences", []) if isinstance(file_graph, dict) else []
    )
    concept_lookup: dict[str, dict[str, Any]] = {}
    if isinstance(concept_rows, list):
        for row in concept_rows:
            if not isinstance(row, dict):
                continue
            concept_id = str(row.get("id", "")).strip()
            if not concept_id:
                continue
            concept_lookup[concept_id] = row
            concept_lookup[_normalize_presence_id(concept_id)] = row
    requested_key = _normalize_presence_id(requested_presence_id)
    if requested_key in concept_lookup:
        concept_presence = concept_lookup[requested_key]
        presence_id = str(concept_presence.get("id", requested_key))
        entity = {
            "id": presence_id,
            "en": str(concept_presence.get("label", "Concept Presence")),
            "ja": str(concept_presence.get("label_ja", "概念プレゼンス")),
            "type": "concept",
        }

    presence_runtime = (
        catalog.get("presence_runtime", {}) if isinstance(catalog, dict) else {}
    )
    if not isinstance(presence_runtime, dict):
        presence_runtime = {}
    resource_heartbeat = (
        presence_runtime.get("resource_heartbeat", {})
        if isinstance(presence_runtime, dict)
        else {}
    )
    if not isinstance(resource_heartbeat, dict) or not resource_heartbeat:
        resource_heartbeat = (
            catalog.get("resource_heartbeat", {}) if isinstance(catalog, dict) else {}
        )
    if not isinstance(resource_heartbeat, dict) or not resource_heartbeat:
        resource_heartbeat = _resource_monitor_snapshot()
    resource_devices = (
        resource_heartbeat.get("devices", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    resource_log_watch = (
        resource_heartbeat.get("log_watch", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )

    asks: list[str] = []
    if "?" in clean_text:
        asks.append(clean_text)
    if "why" in tokens:
        asks.append("Why is this state true in the current catalog?")
    if presence_id == "the_council" and "why" in tokens:
        asks.append("Which gate reason currently blocks council approval?")
    if presence_id.startswith("health_sentinel_") and (
        {"heartbeat", "health", "status", "resource"} & set(tokens)
    ):
        asks.append(
            "Which device is currently hottest and which backend should take the next load?"
        )

    repairs: list[str] = []
    if "fix" in tokens or "repair" in tokens:
        repairs.append(
            "Trace mismatch against canonical constraints and propose patch."
        )
    if "drift" in tokens:
        repairs.append("Run drift scan and report blocked gates.")
    if presence_id == "keeper_of_contracts":
        repairs.append("Resolve open gate questions and append decision receipts.")
    if presence_id == "file_sentinel":
        repairs.append("Watch file deltas and emit append-only receipt traces.")
    if presence_id == "file_organizer":
        repairs.append(
            "Sort files into concept groups by embedding similarity and mint concept presences."
        )
    if presence_id == "the_council":
        repairs.append(
            "Evaluate overlap-boundary quorum and gate reasons before approving restart actions."
        )
    if presence_id.startswith("presence:concept:"):
        repairs.append(
            "Refine this concept presence membership and keep grouped files coherent."
        )
    if presence_id.startswith("health_sentinel_"):
        repairs.append(
            "Read heartbeat telemetry, classify hot devices, and route embedding/text backends to cooler lanes."
        )
        repairs.append(
            "Ingest runtime log tail into embedding memory so future answers cite concrete runtime traces."
        )

    facts = [
        f"presence_id={presence_id}",
        f"catalog_items={len(catalog.get('items', []))}",
        f"promptdb_packets={int(catalog.get('promptdb', {}).get('packet_count', 0))}",
    ]
    if isinstance(file_graph, dict):
        stats = file_graph.get("stats", {})
        if isinstance(stats, dict):
            facts.append(
                f"concept_presence_count={int(_safe_float(stats.get('concept_presence_count', 0), 0.0))}"
            )
            facts.append(
                f"organized_file_count={int(_safe_float(stats.get('organized_file_count', 0), 0.0))}"
            )
    council = catalog.get("council", {}) if isinstance(catalog, dict) else {}
    if isinstance(council, dict):
        facts.append(
            f"council_pending={int(_safe_float(council.get('pending_count', 0), 0.0))}"
        )
        facts.append(
            f"council_decisions={int(_safe_float(council.get('decision_count', 0), 0.0))}"
        )
        facts.append(
            f"council_approved={int(_safe_float(council.get('approved_count', 0), 0.0))}"
        )
    if concept_presence is not None:
        facts.append(
            f"concept_terms={','.join(str(item) for item in concept_presence.get('terms', [])[:3])}"
        )
        facts.append(
            f"concept_file_count={int(_safe_float(concept_presence.get('file_count', 0), 0.0))}"
        )

    cpu_row = (
        resource_devices.get("cpu", {}) if isinstance(resource_devices, dict) else {}
    )
    gpu1_row = (
        resource_devices.get("gpu1", {}) if isinstance(resource_devices, dict) else {}
    )
    gpu2_row = (
        resource_devices.get("gpu2", {}) if isinstance(resource_devices, dict) else {}
    )
    npu0_row = (
        resource_devices.get("npu0", {}) if isinstance(resource_devices, dict) else {}
    )
    facts.append(
        f"cpu_utilization={round(_safe_float(cpu_row.get('utilization', 0.0), 0.0), 2)}"
    )
    facts.append(
        f"gpu1_utilization={round(_safe_float(gpu1_row.get('utilization', 0.0), 0.0), 2)}"
    )
    facts.append(
        f"gpu2_utilization={round(_safe_float(gpu2_row.get('utilization', 0.0), 0.0), 2)}"
    )
    facts.append(
        f"npu0_utilization={round(_safe_float(npu0_row.get('utilization', 0.0), 0.0), 2)}"
    )
    facts.append(
        "resource_hot_devices="
        + ",".join(
            [
                str(item)
                for item in resource_heartbeat.get("hot_devices", [])
                if str(item).strip()
            ]
        )
    )
    if isinstance(resource_log_watch, dict):
        facts.append(
            f"runtime_log_error_count={int(_safe_float(resource_log_watch.get('error_count', 0), 0.0))}"
        )
        latest_log = str(resource_log_watch.get("latest", "")).strip()
        if latest_log:
            facts.append(f"runtime_log_latest={latest_log[:120]}")

    if clean_text:
        facts.append(f"user_text={clean_text[:160]}")

    say_intent = {
        "facts": facts,
        "asks": asks,
        "repairs": repairs,
        "constraints": {
            "no_new_facts": True,
            "cite_refs": True,
            "max_lines": 8,
        },
    }

    rendered = f"[{entity.get('en', 'Presence')}] says: facts={len(facts)} asks={len(asks)} repairs={len(repairs)}"

    return {
        "ok": True,
        "presence_id": presence_id,
        "presence_name": {
            "en": str(entity.get("en", "Presence")),
            "ja": str(entity.get("ja", "プレゼンス")),
        },
        "say_intent": say_intent,
        "rendered_text": rendered,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _audio_suffix_for_mime(mime: str) -> str:
    m = mime.lower().split(";", 1)[0].strip()
    return {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/ogg": ".ogg",
        "audio/webm": ".webm",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
    }.get(m, ".webm")


def _maybe_convert_to_wav(path: Path) -> Path:
    if path.suffix.lower() == ".wav":
        return path
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return path

    wav_path = path.with_suffix(".wav")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return path
    return wav_path if wav_path.exists() else path


def _load_faster_whisper_model() -> Any:
    global _WHISPER_MODEL
    with _WHISPER_MODEL_LOCK:
        if _WHISPER_MODEL is False:
            return None
        if _WHISPER_MODEL is not None:
            return _WHISPER_MODEL
        try:
            mod = importlib.import_module("faster_whisper")
            whisper_model = getattr(mod, "WhisperModel")
        except Exception:
            _WHISPER_MODEL = False
            return None

        model_name = os.getenv("FASTER_WHISPER_MODEL", "small")
        device = os.getenv("FASTER_WHISPER_DEVICE", "auto")
        compute_type = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8")
        try:
            _WHISPER_MODEL = whisper_model(
                model_name,
                device=device,
                compute_type=compute_type,
            )
        except Exception:
            _WHISPER_MODEL = False
            return None
        return _WHISPER_MODEL


def _transcribe_with_faster_whisper(
    path: Path,
    language: str,
) -> tuple[str | None, str | None]:
    model = _load_faster_whisper_model()
    if model is None:
        return None, "unavailable"
    try:
        segments, _info = model.transcribe(
            str(path),
            language=language,
            vad_filter=True,
            beam_size=1,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    except Exception as exc:
        return None, f"error:{exc.__class__.__name__}"
    if not text.strip():
        return None, "no-speech"
    return text.strip(), None


def _transcribe_with_whisper_cpp(
    path: Path,
    language: str,
) -> tuple[str | None, str | None]:
    whisper_bin = os.getenv("WHISPER_CPP_BIN", "whisper-cli")
    model_path = os.getenv("WHISPER_CPP_MODEL", "").strip()
    if not model_path:
        return None, "unconfigured"

    executable = (
        whisper_bin if Path(whisper_bin).exists() else shutil.which(whisper_bin)
    )
    if not executable:
        return None, "missing-binary"

    wav_path = _maybe_convert_to_wav(path)
    out_base = wav_path.with_suffix("")
    cmd = [
        str(executable),
        "-m",
        model_path,
        "-f",
        str(wav_path),
        "-l",
        language,
        "-otxt",
        "-of",
        str(out_base),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None, "exec-failed"

    txt_path = out_base.with_suffix(".txt")
    if not txt_path.exists():
        return None, "no-output"
    text = txt_path.read_text("utf-8", errors="ignore").strip()
    if not text:
        return None, "no-speech"
    return text, None


def transcribe_audio_bytes(
    audio_bytes: bytes,
    mime: str = "audio/webm",
    language: str = "ja",
) -> dict[str, Any]:
    if not audio_bytes:
        return {
            "ok": False,
            "engine": "none",
            "text": "",
            "error": "empty audio payload",
        }

    suffix = _audio_suffix_for_mime(mime)
    with tempfile.TemporaryDirectory(prefix="eta_mu_stt_") as td:
        src = Path(td) / f"input{suffix}"
        src.write_bytes(audio_bytes)
        prepared = _maybe_convert_to_wav(src)

        text_fast, fast_reason = _transcribe_with_faster_whisper(prepared, language)
        if text_fast:
            return {
                "ok": True,
                "engine": "faster-whisper",
                "text": text_fast,
                "error": None,
            }

        if fast_reason != "unavailable":
            return {
                "ok": False,
                "engine": "faster-whisper",
                "text": "",
                "error": "No speech detected or model could not decode audio.",
            }

        text_cpp, cpp_reason = _transcribe_with_whisper_cpp(prepared, language)
        if text_cpp:
            return {
                "ok": True,
                "engine": "whisper.cpp",
                "text": text_cpp,
                "error": None,
            }

        if cpp_reason not in ("unconfigured", "missing-binary"):
            return {
                "ok": False,
                "engine": "whisper.cpp",
                "text": "",
                "error": "whisper.cpp ran but no speech was detected.",
            }

    return {
        "ok": False,
        "engine": "none",
        "text": "",
        "error": "No STT backend active. Install faster-whisper or set WHISPER_CPP_MODEL.",
    }


def _tensorflow_image_fingerprint(image_bytes: bytes) -> dict[str, Any]:
    loader = _world_web_symbol("_load_tensorflow_module", _load_tensorflow_module)
    tf = loader()
    if tf is None:
        return {
            "ok": False,
            "error": "tensorflow_unavailable",
            "width": 0,
            "height": 0,
            "channels": 0,
            "mean_luma": 0.0,
        }
    try:
        decoded = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
        shape = decoded.shape
        height = int(shape[0] or 0)
        width = int(shape[1] or 0)
        channels = int(shape[2] or 0)
        as_float = tf.cast(decoded, tf.float32) / 255.0
        luma = (
            as_float[:, :, 0] * 0.2126
            + as_float[:, :, 1] * 0.7152
            + as_float[:, :, 2] * 0.0722
        )
        mean_luma = float(tf.reduce_mean(luma).numpy())
        return {
            "ok": True,
            "error": "",
            "width": max(0, width),
            "height": max(0, height),
            "channels": max(0, channels),
            "mean_luma": round(mean_luma, 6),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"tensorflow_decode_failed:{exc.__class__.__name__}",
            "width": 0,
            "height": 0,
            "channels": 0,
            "mean_luma": 0.0,
        }


def build_image_commentary(
    *,
    image_bytes: bytes,
    mime: str,
    image_ref: str,
    presence_id: str,
    prompt: str,
    model: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not image_bytes:
        return {
            "ok": False,
            "error": "empty image payload",
            "commentary": "",
            "model": "",
            "backend": "none",
        }
    if len(image_bytes) > IMAGE_COMMENTARY_MAX_BYTES:
        return {
            "ok": False,
            "error": "image payload too large",
            "commentary": "",
            "model": "",
            "backend": "none",
        }

    clean_mime = str(mime or "application/octet-stream").strip().lower()
    clean_prompt = str(prompt or "").strip()
    clean_presence_id = str(presence_id or "unknown").strip() or "unknown"
    clean_image_ref = str(image_ref or "").strip()
    chosen_model = str(
        model or IMAGE_COMMENTARY_MODEL or "qwen3-vl:2b-instruct"
    ).strip()
    chosen_timeout = float(timeout_s or IMAGE_COMMENTARY_TIMEOUT_SECONDS)

    fingerprint = _tensorflow_image_fingerprint(image_bytes)
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    size_label = f"{fingerprint.get('width', 0)}x{fingerprint.get('height', 0)}"
    luma_label = f"{float(fingerprint.get('mean_luma', 0.0)):.3f}"
    prompt_bits = [
        "You are an image analyst for the eta-mu world runtime.",
        f"Presence account: {clean_presence_id}",
        f"Image ref: {clean_image_ref or '[inline-upload]'}",
        f"MIME: {clean_mime}",
        f"TensorFlow fingerprint: size={size_label} channels={int(fingerprint.get('channels', 0))} mean_luma={luma_label}",
        "Write one concise, concrete observation and one actionable follow-up sentence.",
    ]
    if clean_prompt:
        prompt_bits.append(f"User request: {clean_prompt}")
    prompt_text = "\n".join(prompt_bits)

    endpoint = f"{_ollama_base_url()}/api/generate"
    payload = {
        "model": chosen_model,
        "prompt": prompt_text,
        "stream": False,
        "images": [base64.b64encode(image_bytes).decode("ascii")],
    }
    req = Request(
        endpoint,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    opener = _world_web_symbol("urlopen", urlopen)
    request_error = ""

    try:
        with opener(req, timeout=chosen_timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
            commentary = str(raw.get("response", "")).strip()
            if commentary:
                _record_compute_job(
                    kind="llm",
                    op="image_commentary.ollama",
                    backend="ollama",
                    model=chosen_model,
                    status="ok",
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    target_presence_id=clean_presence_id,
                )
                return {
                    "ok": True,
                    "error": "",
                    "commentary": commentary,
                    "model": chosen_model,
                    "backend": "ollama+tensorflow-fingerprint",
                    "analysis": {
                        "image_sha256": image_sha,
                        "mime": clean_mime,
                        "tensorflow": fingerprint,
                    },
                }
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        request_error = exc.__class__.__name__

    fallback_line = (
        f"[{clean_presence_id}] sees image {clean_image_ref or image_sha[:12]} "
        f"({size_label}, luma={luma_label}) and requests a manual verification pass."
    )
    if clean_prompt:
        fallback_line = f"{fallback_line} Request context: {clean_prompt[:180]}"
    _record_compute_job(
        kind="llm",
        op="image_commentary.ollama",
        backend="ollama",
        model=chosen_model,
        status="fallback",
        latency_ms=(time.perf_counter() - started) * 1000.0,
        target_presence_id=clean_presence_id,
        error=request_error or "empty_or_unavailable_response",
    )
    return {
        "ok": True,
        "error": "",
        "commentary": fallback_line,
        "model": chosen_model,
        "backend": "tensorflow-fallback",
        "analysis": {
            "image_sha256": image_sha,
            "mime": clean_mime,
            "tensorflow": fingerprint,
        },
    }


def _openvino_embed(
    text: str,
    model: str | None = None,
    timeout_s: float | None = None,
    device: str | None = None,
    **_: Any,
) -> list[float] | None:
    prompt = str(text or "").strip()
    if not prompt:
        return None

    endpoint = str(os.getenv("OPENVINO_EMBED_ENDPOINT", "") or "").strip()
    if not endpoint:
        _record_openvino_runtime(
            key="",
            model=None,
            error="openvino_endpoint_unset",
        )
        return None

    chosen_model = str(
        model
        or os.getenv("OPENVINO_EMBED_MODEL", "")
        or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        or "nomic-embed-text"
    ).strip()
    chosen_device = _normalize_openvino_device(
        device or os.getenv("OPENVINO_EMBED_DEVICE", "NPU")
    )
    chosen_timeout = (
        float(timeout_s)
        if timeout_s is not None
        else _safe_embedding_timeout_env("OPENVINO_EMBED_TIMEOUT_SEC", 12.0)
    )
    normalize = _embedding_flag("OPENVINO_EMBED_NORMALIZE", True)
    max_chars = _safe_embedding_int_env(
        "OPENVINO_EMBED_MAX_CHARS",
        _safe_embedding_int_env(
            "OLLAMA_EMBED_MAX_CHARS", 2400, min_value=0, max_value=64000
        ),
        min_value=0,
        max_value=64000,
    )
    sample_text = prompt[:max_chars] if max_chars > 0 else prompt

    opener = _world_web_symbol("urlopen", urlopen)
    last_error = "openvino_request_failed"
    for target in _openvino_embed_candidates(endpoint):
        payload_variants: list[dict[str, Any]] = []
        if target.endswith("/v1/embeddings"):
            payload_variants.append(
                {
                    "model": chosen_model,
                    "input": [sample_text],
                    "options": {
                        "device": chosen_device,
                    },
                }
            )

        payload_variants.extend(
            [
                {
                    "model": chosen_model,
                    "input": sample_text,
                    "device": chosen_device,
                    "normalize": normalize,
                },
                {
                    "model": chosen_model,
                    "text": sample_text,
                    "device": chosen_device,
                    "normalize": normalize,
                },
                {
                    "model": chosen_model,
                    "prompt": sample_text,
                    "device": chosen_device,
                    "normalize": normalize,
                },
            ]
        )

        for payload in payload_variants:
            req = Request(
                target,
                method="POST",
                headers=_openvino_embed_headers(),
                data=json.dumps(payload).encode("utf-8"),
            )
            try:
                with opener(req, timeout=chosen_timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
                embedding = _embedding_payload_vector(raw)
                if embedding is None:
                    continue
                normalized = _normalize_embedding_runtime_vector(embedding, normalize)
                if not normalized:
                    continue
                _record_openvino_runtime(
                    key=f"{target}|{chosen_model}|{chosen_device}",
                    model=chosen_model,
                    error="",
                )
                return normalized
            except (
                URLError,
                TimeoutError,
                OSError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                last_error = f"{exc.__class__.__name__}:{exc}"
                continue

    _record_openvino_runtime(
        key=f"{endpoint}|{chosen_model}|{chosen_device}",
        model=None,
        error=last_error[:220],
    )
    return None


def _embed_text(text: str, **kwargs) -> list[float] | None:
    b = _embedding_backend()
    model = kwargs.get("model")
    if b == "torch":
        return _torch_embed(text, model=model)
    if b == "ollama":
        return _ollama_embed(text, model=model)
    if b == "openvino":
        started = time.perf_counter()
        vector = _openvino_embed(text, model=model)
        chosen_model = str(
            model
            or os.getenv("OPENVINO_EMBED_MODEL", "")
            or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            or "nomic-embed-text"
        ).strip()
        _record_compute_job(
            kind="embedding",
            op="embed_text.openvino",
            backend="openvino",
            model=chosen_model,
            status="ok" if vector is not None else "error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            target_presence_id="file_organizer",
            error="" if vector is not None else "openvino_no_vector",
            device=str(os.getenv("OPENVINO_EMBED_DEVICE", "NPU") or "NPU"),
        )
        return vector
    if b == "tensorflow":
        started = time.perf_counter()
        vector = _tensorflow_embed(text, model=model)
        _record_compute_job(
            kind="embedding",
            op="embed_text.tensorflow",
            backend="tensorflow",
            model=str(model or "tf-hash"),
            status="ok" if vector is not None else "error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            target_presence_id="file_organizer",
            error="" if vector is not None else "tensorflow_no_vector",
        )
        return vector
    return None


def _eta_mu_detect_modality(
    *,
    path: Path,
    mime: str,
) -> tuple[str | None, str]:
    from .constants import (
        ETA_MU_INGEST_INCLUDE_TEXT_MIME,
        ETA_MU_INGEST_INCLUDE_IMAGE_MIME,
        ETA_MU_INGEST_INCLUDE_TEXT_EXT,
        ETA_MU_INGEST_INCLUDE_IMAGE_EXT,
    )

    normalized_mime = str(mime or "").strip().lower()
    suffix = path.suffix.lower()

    if normalized_mime.startswith("text/"):
        return "text", "mime-text"
    if normalized_mime in ETA_MU_INGEST_INCLUDE_TEXT_MIME:
        return "text", "mime-text"
    if normalized_mime in ETA_MU_INGEST_INCLUDE_IMAGE_MIME:
        return "image", "mime-image"

    if suffix in ETA_MU_INGEST_INCLUDE_TEXT_EXT:
        return "text", "ext-text"
    if suffix in ETA_MU_INGEST_INCLUDE_IMAGE_EXT:
        return "image", "ext-image"

    return None, "unsupported-modality"


def _eta_mu_guess_mime(path: Path, raw: bytes) -> str:
    mime = str(mimetypes.guess_type(str(path))[0] or "").strip().lower()
    if mime:
        return mime

    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw[:6] in {b"GIF87a", b"GIF89a"}:
        return "image/gif"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if _is_probably_text_bytes(raw[:8192]):
        return "text/plain"
    return "application/octet-stream"


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


_TORCH_MODEL_CACHE: dict[str, Any] = {}
_TORCH_LOCK = threading.Lock()


def _torch_embed(text: str, model: str | None = None) -> list[float] | None:
    """
    Run embedding directly in-process on CPU/GPU via sentence-transformers/PyTorch.
    This avoids HTTP overhead and leverages local hardware directly.
    """
    prompt = str(text or "").strip()
    if not prompt:
        return None

    model_name = str(
        model or os.getenv("TORCH_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    ).strip()

    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    cache_key = f"{model_name}|{device}"

    st_model = _TORCH_MODEL_CACHE.get(cache_key)
    if st_model is None:
        with _TORCH_LOCK:
            st_model = _TORCH_MODEL_CACHE.get(cache_key)
            if st_model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    st_model = SentenceTransformer(
                        model_name, trust_remote_code=True, device=device
                    )
                    _TORCH_MODEL_CACHE[cache_key] = st_model
                except Exception as e:
                    # print(f"[Torch Backend Error] Load failed: {e}")
                    _record_compute_job(
                        kind="embedding",
                        op="embed.torch.load_error",
                        backend="torch",
                        model=model_name,
                        status="error",
                        latency_ms=0.0,
                        error=str(e)[:120],
                    )
                    return None

    started = time.perf_counter()
    try:
        # Run inference
        # If model supports MRL (Matryoshka), we can just take the full vector here
        # and let the caller slice it if needed.
        # Nomic v1.5 returns 768 dim by default.
        embedding = st_model.encode(prompt, convert_to_numpy=True)
        vector = _normalize_embedding_vector(embedding.tolist())

        _record_compute_job(
            kind="embedding",
            op="embed.torch",
            backend="torch",
            model=model_name,
            status="ok",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            target_presence_id="file_organizer",  # Default
            device=device,
        )
        return vector
    except Exception as e:
        # print(f"[Torch Backend Error] Inference failed: {e}")
        _record_compute_job(
            kind="embedding",
            op="embed.torch.infer_error",
            backend="torch",
            model=model_name,
            status="error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=str(e)[:120],
        )
        return None


def _eta_mu_canonicalize_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines = [
        unicodedata.normalize("NFC", line.rstrip(" \t\v\f"))
        for line in text.split("\n")
    ]
    return "\n".join(normalized_lines)


def _eta_mu_text_segments(
    canonical_text: str, *, language_hint: str
) -> list[dict[str, Any]]:
    policy = _eta_mu_chunk_policy()
    target = int(policy["target"])
    overlap = int(policy["overlap"])
    minimum = int(policy["min"])
    maximum = int(policy["max"])

    text = str(canonical_text)
    if not text:
        return [
            {
                "id": "seg-0000",
                "start": 0,
                "end": 0,
                "unit": "char",
                "text": "",
                "language_hint": language_hint,
            }
        ]

    chunk_size = max(minimum, min(maximum, target))
    stride = max(1, chunk_size - overlap)

    segments: list[dict[str, Any]] = []
    start = 0
    idx = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + chunk_size)
        if segments and (text_len - start) < minimum:
            tail = segments[-1]
            tail_start = int(tail.get("start", 0))
            tail["end"] = text_len
            tail["text"] = text[tail_start:text_len]
            break

        segment_text = text[start:end]
        segments.append(
            {
                "id": f"seg-{idx:04d}",
                "start": start,
                "end": end,
                "unit": "char",
                "text": segment_text,
                "language_hint": language_hint,
            }
        )
        if end >= text_len:
            break
        start += stride
        idx += 1

    return segments


def _eta_mu_text_language_hint(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix:
        return suffix
    return "text"


def _eta_mu_embed_id(
    *,
    space_signature: str,
    source_hash: str,
    segment: dict[str, Any],
) -> str:
    seed = "|".join(
        [
            str(space_signature),
            str(source_hash),
            str(segment.get("id", "")),
            str(int(_safe_float(segment.get("start", 0), 0.0))),
            str(int(_safe_float(segment.get("end", 0), 0.0))),
            str(segment.get("unit", "")),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _eta_mu_registry_idempotence_key(
    *,
    embed_id: str,
    source_hash: str,
    space_signature: str,
    segment: dict[str, Any],
) -> str:
    payload = {
        "embed_id": str(embed_id),
        "source_hash": str(source_hash),
        "space_signature": str(space_signature),
        "segment": {
            "id": str(segment.get("id", "")),
            "start": int(_safe_float(segment.get("start", 0), 0.0)),
            "end": int(_safe_float(segment.get("end", 0), 0.0)),
            "unit": str(segment.get("unit", "")),
        },
    }
    return _eta_mu_json_sha256(payload)


def _eta_mu_emit_packet(
    packets: list[dict[str, Any]],
    *,
    kind: str,
    body: dict[str, Any],
    links: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .constants import ETA_MU_INGEST_PACKET_RECORD

    now = datetime.now(timezone.utc).isoformat()
    packet_seed = _eta_mu_json_sha256(
        {
            "kind": kind,
            "body": body,
            "links": links or {},
            "index": len(packets),
            "ts": now,
        }
    )
    packet = {
        "record": ETA_MU_INGEST_PACKET_RECORD,
        "v": "opencode.packet/v1",
        "id": f"packet:{kind}:{packet_seed[:16]}",
        "kind": kind,
        "title": kind,
        "tags": ["eta-mu", "ingest", kind],
        "body": dict(body),
        "links": dict(links) if isinstance(links, dict) else {},
        "time": now,
    }
    packets.append(packet)
    return packet


def _extract_openai_chat_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        text_value = item.get("text")
        if isinstance(text_value, str) and text_value.strip():
            parts.append(text_value.strip())
            continue
        if isinstance(item.get("content"), str) and str(item.get("content")).strip():
            parts.append(str(item.get("content")).strip())
    return " ".join(parts).strip()


def _extract_openai_chat_response_text(raw: Any) -> str:
    if not isinstance(raw, dict):
        return ""
    choices = raw.get("choices", [])
    if not isinstance(choices, list):
        return ""

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            parsed = _extract_openai_chat_content_text(message.get("content"))
            if parsed:
                return parsed
        text_value = choice.get("text")
        if isinstance(text_value, str) and text_value.strip():
            return text_value.strip()
    return ""


def _eta_mu_image_vllm_caption_for_embedding(
    *,
    image_bytes: bytes,
    mime: str,
    source_rel_path: str,
) -> dict[str, str]:
    enabled = str(os.getenv("ETA_MU_IMAGE_VISION_ENABLED", "0") or "0").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return {
            "caption": "",
            "model": "",
            "backend": "vllm-disabled",
            "error": "image_vision_disabled",
        }

    base_url = (
        str(os.getenv("ETA_MU_IMAGE_VISION_BASE_URL", "") or "").strip().rstrip("/")
    )
    if not base_url:
        return {
            "caption": "",
            "model": "",
            "backend": "vllm",
            "error": "vllm_base_url_unset",
        }

    endpoint = base_url
    if endpoint.endswith("/v1/chat/completions"):
        pass
    elif endpoint.endswith("/v1"):
        endpoint = f"{endpoint}/chat/completions"
    else:
        endpoint = f"{endpoint}/v1/chat/completions"

    try:
        max_bytes = int(
            float(
                str(
                    os.getenv("ETA_MU_IMAGE_VISION_MAX_BYTES", "16000000") or "16000000"
                )
            )
        )
    except (TypeError, ValueError):
        max_bytes = 16_000_000
    max_bytes = max(1024, min(40_000_000, max_bytes))
    if len(image_bytes) > max_bytes:
        return {
            "caption": "",
            "model": "",
            "backend": "vllm",
            "error": "image_payload_too_large",
        }

    try:
        timeout_s = float(
            str(os.getenv("ETA_MU_IMAGE_VISION_TIMEOUT_SECONDS", "12") or "12")
        )
    except (TypeError, ValueError):
        timeout_s = 12.0
    timeout_s = max(0.2, min(120.0, timeout_s))

    try:
        max_tokens = int(
            float(str(os.getenv("ETA_MU_IMAGE_VISION_MAX_TOKENS", "180") or "180"))
        )
    except (TypeError, ValueError):
        max_tokens = 180
    max_tokens = max(32, min(1200, max_tokens))

    try:
        caption_limit = int(
            float(
                str(os.getenv("ETA_MU_IMAGE_VISION_CAPTION_MAX_CHARS", "640") or "640")
            )
        )
    except (TypeError, ValueError):
        caption_limit = 640
    caption_limit = max(120, min(4000, caption_limit))

    chosen_model = str(
        os.getenv("ETA_MU_IMAGE_VISION_MODEL", "")
        or IMAGE_COMMENTARY_MODEL
        or "qwen3-vl:2b-instruct"
    ).strip()
    prompt = str(
        os.getenv(
            "ETA_MU_IMAGE_VISION_PROMPT",
            "Describe this image for retrieval embeddings in two concise factual sentences. "
            "Mention objects, setting, any readable text, and salient actions.",
        )
        or ""
    ).strip()
    if not prompt:
        prompt = "Describe this image for retrieval embeddings in two concise factual sentences."

    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime or 'application/octet-stream'};base64,{image_base64}"
    payload = {
        "model": chosen_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}\nImage source: {source_rel_path}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    api_key = str(os.getenv("ETA_MU_IMAGE_VISION_API_KEY", "") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = Request(
        endpoint,
        method="POST",
        headers=headers,
        data=json.dumps(payload).encode("utf-8"),
    )
    opener = _world_web_symbol("urlopen", urlopen)
    started = time.perf_counter()

    try:
        with opener(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        _record_compute_job(
            kind="llm",
            op="image_caption.vllm",
            backend="vllm",
            model=chosen_model,
            status="error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            target_presence_id="file_organizer",
            error=f"{exc.__class__.__name__}",
        )
        return {
            "caption": "",
            "model": chosen_model,
            "backend": "vllm",
            "error": f"{exc.__class__.__name__}",
        }

    caption = _extract_openai_chat_response_text(raw)
    caption = re.sub(r"\s+", " ", caption).strip()
    if not caption:
        _record_compute_job(
            kind="llm",
            op="image_caption.vllm",
            backend="vllm",
            model=chosen_model,
            status="error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            target_presence_id="file_organizer",
            error="vllm_empty_caption",
        )
        return {
            "caption": "",
            "model": chosen_model,
            "backend": "vllm",
            "error": "vllm_empty_caption",
        }
    if len(caption) > caption_limit:
        caption = caption[:caption_limit].rstrip() + "..."

    _record_compute_job(
        kind="llm",
        op="image_caption.vllm",
        backend="vllm",
        model=chosen_model,
        status="ok",
        latency_ms=(time.perf_counter() - started) * 1000.0,
        target_presence_id="file_organizer",
    )

    return {
        "caption": caption,
        "model": chosen_model,
        "backend": "vllm",
        "error": "",
    }


def _eta_mu_image_derive_segment(
    *,
    source_hash: str,
    source_bytes: int,
    source_rel_path: str,
    mime: str,
    image_bytes: bytes | None = None,
) -> dict[str, Any]:
    derive_spec = {
        "decode": "rgb",
        "resize": {
            "long_edge": 768,
            "mode": "fit",
            "interpolation": "bicubic",
        },
        "strip_metadata": True,
    }
    derive_hash = _eta_mu_json_sha256(
        {
            "spec": derive_spec,
            "source_hash": source_hash,
            "source_bytes": source_bytes,
            "source_rel_path": source_rel_path,
        }
    )
    descriptor = (
        "image-preproc "
        + f"mime={mime} source={source_rel_path} source_hash={source_hash} "
        + f"source_bytes={source_bytes} derive_hash={derive_hash}"
    )

    vision_caption = ""
    vision_model = ""
    vision_backend = ""
    vision_error = ""
    if image_bytes:
        vision_payload = _eta_mu_image_vllm_caption_for_embedding(
            image_bytes=image_bytes,
            mime=mime,
            source_rel_path=source_rel_path,
        )
        vision_caption = str(vision_payload.get("caption", "")).strip()
        vision_model = str(vision_payload.get("model", "")).strip()
        vision_backend = str(vision_payload.get("backend", "")).strip()
        vision_error = str(vision_payload.get("error", "")).strip()
        if vision_caption:
            descriptor = f"{descriptor} vision-caption={vision_caption}"

    segment = {
        "id": "img-0000",
        "start": 0,
        "end": 0,
        "unit": "image",
        "text": descriptor,
        "derive_spec": derive_spec,
        "derive_hash": derive_hash,
    }
    if vision_backend:
        segment["vision_backend"] = vision_backend
    if vision_model:
        segment["vision_model"] = vision_model
    if vision_caption:
        segment["vision_caption"] = vision_caption
    if vision_error:
        segment["vision_error"] = vision_error
    return segment


def _eta_mu_registry_reference_key(
    *,
    source_hash: str,
    rel_source: str,
    source_bytes: int,
    modality: str,
) -> str:
    return _eta_mu_registry_key(
        content_sha256=source_hash,
        rel_source=rel_source,
        size=max(0, int(source_bytes)),
        kind=modality,
    )


def _eta_mu_registry_key(
    *,
    content_sha256: str,
    rel_source: str,
    size: int,
    kind: str,
) -> str:
    material = {
        "content_sha256": str(content_sha256),
        "rel_source": str(rel_source).replace("\\", "/"),
        "size": int(size),
        "kind": str(kind).strip().lower() or "file",
    }
    payload = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _eta_mu_json_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _stable_file_digest(path: Path) -> str:
    h = hashlib.sha1()
    try:
        stat = path.stat()
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(stat.st_mtime_ns).encode("utf-8"))
        with path.open("rb") as handle:
            h.update(handle.read(65536))
    except OSError:
        h.update(str(path).encode("utf-8"))
        h.update(str(time.time_ns()).encode("utf-8"))
    return h.hexdigest()


def _eta_mu_model_key(
    *,
    model_name: str,
    model_digest: str,
    settings_hash: str,
) -> dict[str, str]:
    return {
        "provider": "ollama",
        "name": model_name,
        "digest": model_digest,
        "settings_hash": settings_hash,
    }


def _eta_mu_normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude <= 0.0:
        return vector
    return [value / magnitude for value in vector]


def _eta_mu_resize_vector(vector: list[float], dims: int) -> list[float]:
    if dims <= 0:
        return list(vector)
    if len(vector) == dims:
        return list(vector)
    if not vector:
        return [0.0] * dims

    out = [0.0] * dims
    for idx, value in enumerate(vector):
        out[idx % dims] += float(value)
    return out


def _eta_mu_deterministic_vector(seed: str, dims: int) -> list[float]:
    rng_seed = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(rng_seed)
    vector = [(rng.random() * 2.0) - 1.0 for _ in range(max(1, dims))]
    return _eta_mu_normalize_vector(vector)


def _eta_mu_embed_vector_for_segment(
    *,
    modality: str,
    segment_text: str,
    space: dict[str, Any],
    embed_id: str,
) -> tuple[list[float], str, bool]:
    model = str((space.get("model") or {}).get("name", "")).strip()
    dims = max(1, int(_safe_float(space.get("dims", 1), 1.0)))

    remote_vector: list[float] | None = None
    if modality == "text":
        remote_vector = _embed_text(segment_text, model=model)
    elif modality == "image":
        remote_vector = _embed_text(segment_text, model=model)

    fallback_used = False
    if remote_vector is None:
        fallback_used = True
        _record_compute_job(
            kind="embedding",
            op="eta_mu_embed.deterministic_fallback",
            backend="deterministic",
            model=model,
            status="fallback",
            latency_ms=0.0,
            target_presence_id="file_organizer",
            error="upstream_embedding_unavailable",
        )
        remote_vector = _eta_mu_deterministic_vector(
            embed_id + "|" + segment_text, dims
        )

    vector = _normalize_embedding_vector(remote_vector)
    if vector is None:
        fallback_used = True
        _record_compute_job(
            kind="embedding",
            op="eta_mu_embed.deterministic_fallback",
            backend="deterministic",
            model=model,
            status="fallback",
            latency_ms=0.0,
            target_presence_id="file_organizer",
            error="upstream_embedding_invalid",
        )
        vector = _eta_mu_deterministic_vector(embed_id + "|" + segment_text, dims)
    vector = _eta_mu_resize_vector(vector, dims)
    vector = _eta_mu_normalize_vector(vector)
    return vector, model or "", fallback_used


def _eta_mu_chunk_policy() -> dict[str, int]:
    from .constants import (
        ETA_MU_INGEST_TEXT_CHUNK_TARGET,
        ETA_MU_INGEST_TEXT_CHUNK_OVERLAP,
        ETA_MU_INGEST_TEXT_CHUNK_MIN,
        ETA_MU_INGEST_TEXT_CHUNK_MAX,
        ETA_MU_INGEST_SAFE_MODE,
    )

    target = ETA_MU_INGEST_TEXT_CHUNK_TARGET
    overlap = ETA_MU_INGEST_TEXT_CHUNK_OVERLAP
    minimum = ETA_MU_INGEST_TEXT_CHUNK_MIN
    maximum = ETA_MU_INGEST_TEXT_CHUNK_MAX

    if ETA_MU_INGEST_SAFE_MODE:
        target = min(maximum, max(minimum, int(target * 1.25)))
        overlap = max(0, int(overlap * 0.5))

    target = max(minimum, min(maximum, target))
    overlap = max(0, min(target - 1, overlap)) if target > 1 else 0
    return {
        "target": target,
        "overlap": overlap,
        "min": minimum,
        "max": maximum,
    }


def _eta_mu_space_forms() -> dict[str, Any]:
    from .constants import (
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
        ETA_MU_INGEST_VECSTORE_COLLECTION,
        ETA_MU_INGEST_VECSTORE_LAYER_MODE,
    )

    vecstore_collection = str(
        _world_web_symbol(
            "ETA_MU_INGEST_VECSTORE_COLLECTION",
            ETA_MU_INGEST_VECSTORE_COLLECTION,
        )
    ).strip()
    layer_mode = str(
        _world_web_symbol(
            "ETA_MU_INGEST_VECSTORE_LAYER_MODE",
            ETA_MU_INGEST_VECSTORE_LAYER_MODE,
        )
    ).strip()

    text_space: dict[str, Any] = {
        "id": ETA_MU_INGEST_SPACE_TEXT_ID,
        "modality": "text",
        "model": {
            "provider": "ollama",
            "name": ETA_MU_INGEST_TEXT_MODEL,
            "digest": ETA_MU_INGEST_TEXT_MODEL_DIGEST or "none",
        },
        "dims": ETA_MU_INGEST_TEXT_DIMS,
        "metric": "cosine",
        "normalize": True,
        "dtype": "f16",
        "preproc": {
            "chunk": {
                "policy": "token-ish",
                "target": ETA_MU_INGEST_TEXT_CHUNK_TARGET,
                "overlap": ETA_MU_INGEST_TEXT_CHUNK_OVERLAP,
                "min": ETA_MU_INGEST_TEXT_CHUNK_MIN,
                "max": ETA_MU_INGEST_TEXT_CHUNK_MAX,
            },
            "strip": {
                "frontmatter": True,
                "codeblocks": "keep",
                "html": "keep",
            },
            "language_hint": "extension",
        },
        "time": "none",
    }
    text_space["signature"] = _eta_mu_json_sha256(text_space)
    text_space["collection"] = _eta_mu_vecstore_collection_for_space(text_space)

    image_space: dict[str, Any] = {
        "id": ETA_MU_INGEST_SPACE_IMAGE_ID,
        "modality": "image",
        "model": {
            "provider": "ollama",
            "name": ETA_MU_INGEST_IMAGE_MODEL,
            "digest": ETA_MU_INGEST_IMAGE_MODEL_DIGEST or "none",
        },
        "dims": ETA_MU_INGEST_IMAGE_DIMS,
        "metric": "cosine",
        "normalize": True,
        "dtype": "f16",
        "preproc": {
            "decode": "rgb",
            "resize": {
                "long_edge": 768,
                "mode": "fit",
                "interpolation": "bicubic",
            },
            "strip_metadata": True,
        },
        "time": "none",
    }
    image_space["signature"] = _eta_mu_json_sha256(image_space)
    image_space["collection"] = _eta_mu_vecstore_collection_for_space(image_space)

    space_set = {
        "id": ETA_MU_INGEST_SPACE_SET_ID,
        "members": [ETA_MU_INGEST_SPACE_TEXT_ID, ETA_MU_INGEST_SPACE_IMAGE_ID],
        "routing": {
            "text": ETA_MU_INGEST_SPACE_TEXT_ID,
            "image": ETA_MU_INGEST_SPACE_IMAGE_ID,
        },
        "collections": {
            "text": str(text_space.get("collection", vecstore_collection)),
            "image": str(image_space.get("collection", vecstore_collection)),
        },
        "time": "none",
    }
    space_set["signature"] = _eta_mu_json_sha256(space_set)

    vecstore = {
        "id": ETA_MU_INGEST_VECSTORE_ID,
        "backend": "chroma",
        "collection": vecstore_collection,
        "layer_mode": layer_mode,
        "collections": {
            "text": str(text_space.get("collection", vecstore_collection)),
            "image": str(image_space.get("collection", vecstore_collection)),
        },
        "space_set": ETA_MU_INGEST_SPACE_SET_ID,
        "index": {
            "kind": "hnsw",
            "params": {
                "m": 16,
                "ef_construction": 200,
                "ef_search": 64,
            },
        },
    }
    vecstore["signature"] = _eta_mu_json_sha256(
        {
            "collection": vecstore["collection"],
            "layer_mode": vecstore["layer_mode"],
            "collections": vecstore["collections"],
            "index": vecstore["index"],
            "space_set_signature": space_set["signature"],
        }
    )

    return {
        "text": text_space,
        "image": image_space,
        "space_set": space_set,
        "vecstore": vecstore,
    }


def _eta_mu_vecstore_collection_for_space(space: dict[str, Any]) -> str:
    from .constants import (
        ETA_MU_INGEST_VECSTORE_LAYER_MODE,
        ETA_MU_INGEST_VECSTORE_COLLECTION,
    )

    mode = str(
        _world_web_symbol(
            "ETA_MU_INGEST_VECSTORE_LAYER_MODE",
            ETA_MU_INGEST_VECSTORE_LAYER_MODE,
        )
    ).strip()
    base = str(
        _world_web_symbol(
            "ETA_MU_INGEST_VECSTORE_COLLECTION",
            ETA_MU_INGEST_VECSTORE_COLLECTION,
        )
    ).strip()
    if mode in {"", "single", "none", "off"}:
        return base

    space_id = str(space.get("id", "")).strip()
    space_signature = str(space.get("signature", "")).strip()
    model_name = str((space.get("model") or {}).get("name", "")).strip()

    if mode == "space":
        token = _sanitize_collection_name_token(space_id)
    elif mode == "signature":
        token = _sanitize_collection_name_token(space_signature[:12])
    elif mode == "model":
        token = _sanitize_collection_name_token(model_name)
    elif mode in {"space-signature", "space_signature"}:
        token = _sanitize_collection_name_token(f"{space_id}_{space_signature[:10]}")
    elif mode in {"space-model", "space_model"}:
        token = _sanitize_collection_name_token(f"{space_id}_{model_name}")
    else:
        token = _sanitize_collection_name_token(mode)

    if not token:
        return base
    return f"{base}__{token}"


def _sanitize_collection_name_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip().lower())
    cleaned = cleaned.strip("._-")
    return cleaned or "layer"


def _eta_mu_embed_layer_identity(
    *,
    collection: str,
    space_id: str,
    space_signature: str,
    model_name: str,
) -> dict[str, str]:
    from .constants import ETA_MU_INGEST_VECSTORE_COLLECTION

    clean_collection = str(collection or ETA_MU_INGEST_VECSTORE_COLLECTION).strip()
    clean_space_id = str(space_id or "").strip()
    clean_signature = str(space_signature or "").strip()
    clean_model = str(model_name or "").strip()
    layer_key = "|".join(
        [clean_collection, clean_space_id, clean_signature, clean_model]
    )
    layer_id = "layer:" + hashlib.sha1(layer_key.encode("utf-8")).hexdigest()[:12]

    label_parts = [clean_collection]
    if clean_space_id:
        label_parts.append(clean_space_id)
    if clean_model:
        label_parts.append(clean_model)
    elif clean_signature:
        label_parts.append(clean_signature[:10])

    return {
        "id": layer_id,
        "key": layer_key,
        "label": " · ".join(part for part in label_parts if part),
        "collection": clean_collection,
        "space_id": clean_space_id,
        "space_signature": clean_signature,
        "model_name": clean_model,
    }
