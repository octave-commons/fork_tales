from __future__ import annotations

import argparse
import base64
import colorsys
import fnmatch
import hmac
import hashlib
import html
import importlib
import io
import json
import math
import mimetypes
import os
import random
import re
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unicodedata
import wave
import webbrowser
import zipfile
from array import array
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha1
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import (
    parse_qs,
    parse_qsl,
    quote,
    urlencode,
    urlparse,
    urlunparse,
    unquote,
)
from urllib.request import Request, urlopen


try:
    from code.lore import (
        ENTITY_MANIFEST,
        CANONICAL_TERMS,
        VOICE_LINE_BANK,
        NAME_HINTS,
        ROLE_HINTS,
        SYSTEM_PROMPT_TEMPLATE,
        PART_67_PROLOGUE,
    )
except ImportError:
    from lore import (
        ENTITY_MANIFEST,
        CANONICAL_TERMS,
        VOICE_LINE_BANK,
        NAME_HINTS,
        ROLE_HINTS,
        SYSTEM_PROMPT_TEMPLATE,
        PART_67_PROLOGUE,
    )


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


AUDIO_SUFFIXES = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}
WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_SIM_POINTS = max(256, int(os.getenv("MAX_SIM_POINTS", "2048") or "2048"))
SIM_TICK_SECONDS = float(os.getenv("SIM_TICK_SECONDS", "0.2") or "0.2")
CATALOG_REFRESH_SECONDS = float(os.getenv("CATALOG_REFRESH_SECONDS", "1.5") or "1.5")
CATALOG_BROADCAST_HEARTBEAT_SECONDS = float(
    os.getenv("CATALOG_BROADCAST_HEARTBEAT_SECONDS", "6.0") or "6.0"
)
DAIMO_FORCE_KAPPA = max(
    0.02,
    float(os.getenv("DAIMO_FORCE_KAPPA", "0.22") or "0.22"),
)
DAIMO_DAMPING = max(
    0.0,
    min(0.99, float(os.getenv("DAIMO_DAMPING", "0.88") or "0.88")),
)
DAIMO_DT_SECONDS = max(
    0.02,
    min(0.4, float(os.getenv("DAIMO_DT_SECONDS", "0.2") or "0.2")),
)
DAIMO_MAX_TRACKED_ENTITIES = max(
    24,
    int(os.getenv("DAIMO_MAX_TRACKED_ENTITIES", "280") or "280"),
)
DAIMO_PROFILE_DEFS: tuple[dict[str, Any], ...] = (
    {
        "id": "daimo:core",
        "name": "Core Daimoi",
        "ctx": "主",
        "base_budget": 9.0,
        "w": 1.0,
        "temperature": 0.34,
    },
    {
        "id": "daimo:self",
        "name": "Self Daimoi",
        "ctx": "己",
        "base_budget": 7.0,
        "w": 0.9,
        "temperature": 0.42,
    },
    {
        "id": "daimo:you",
        "name": "You Daimoi",
        "ctx": "汝",
        "base_budget": 7.0,
        "w": 0.88,
        "temperature": 0.48,
    },
    {
        "id": "daimo:they",
        "name": "They Daimoi",
        "ctx": "彼",
        "base_budget": 6.0,
        "w": 0.84,
        "temperature": 0.56,
    },
    {
        "id": "daimo:world",
        "name": "World Daimoi",
        "ctx": "世",
        "base_budget": 8.0,
        "w": 0.94,
        "temperature": 0.4,
    },
)
CANONICAL_NAMED_FIELD_IDS = (
    "receipt_river",
    "witness_thread",
    "fork_tax_canticle",
    "mage_of_receipts",
    "keeper_of_receipts",
    "anchor_registry",
    "gates_of_truth",
)
ETA_MU_INBOX_DIRNAME = ".ημ"
ETA_MU_KNOWLEDGE_INDEX_REL = ".opencode/runtime/eta_mu_knowledge.v1.jsonl"
ETA_MU_KNOWLEDGE_ARCHIVE_REL = ".opencode/knowledge/archive"
ETA_MU_REGISTRY_REL = ".Π/ημ_registry.jsonl"
ETA_MU_EMBEDDINGS_DB_REL = ".opencode/runtime/eta_mu_embeddings.v1.jsonl"
ETA_MU_GRAPH_MOVES_REL = ".opencode/runtime/eta_mu_graph_moves.v1.json"
ETA_MU_KNOWLEDGE_RECORD = "ημ.knowledge.v1"
ETA_MU_ARCHIVE_MANIFEST_RECORD = "ημ.archive-manifest.v1"
ETA_MU_INGEST_REGISTRY_RECORD = "ημ.ingest-registry.v1"
ETA_MU_EMBEDDINGS_DB_RECORD = "ημ.embedding-db.v1"
ETA_MU_FILE_GRAPH_RECORD = "ημ.file-graph.v1"
ETA_MU_FILE_GRAPH_MOVES_RECORD = "ημ.file-graph.moves.v1"
ETA_MU_FILE_GRAPH_HEALTH_RECORD = "ημ.graph-health.v1"
ETA_MU_CRAWLER_GRAPH_RECORD = "ημ.crawler-graph.v1"
ETA_MU_TRUTH_STATE_RECORD = "ημ.truth-state.v1"
ETA_MU_PI_ARCHIVE_RECORD = "ημ.pi-archive.v1"
ETA_MU_INBOX_DEBOUNCE_SECONDS = max(
    0.0,
    float(os.getenv("ETA_MU_INBOX_DEBOUNCE_SECONDS", "1.0") or "1.0"),
)
ETA_MU_MAX_GRAPH_FILES = max(
    64,
    int(os.getenv("ETA_MU_MAX_GRAPH_FILES", "1200") or "1200"),
)
WEAVER_GRAPH_CACHE_SECONDS = max(
    0.2,
    float(os.getenv("WEAVER_GRAPH_CACHE_SECONDS", "2.0") or "2.0"),
)
WEAVER_GRAPH_NODE_LIMIT = max(
    200,
    int(os.getenv("WEAVER_GRAPH_NODE_LIMIT", "900") or "900"),
)
WEAVER_GRAPH_EDGE_LIMIT = max(
    300,
    int(os.getenv("WEAVER_GRAPH_EDGE_LIMIT", "2200") or "2200"),
)
WEAVER_GRAPH_HEALTH_TIMEOUT_SECONDS = max(
    0.05,
    float(os.getenv("WEAVER_GRAPH_HEALTH_TIMEOUT_SECONDS", "0.12") or "0.12"),
)
WEAVER_GRAPH_FETCH_TIMEOUT_SECONDS = max(
    0.08,
    float(os.getenv("WEAVER_GRAPH_FETCH_TIMEOUT_SECONDS", "0.35") or "0.35"),
)
WEAVER_SEMANTIC_ANALYSIS_ENABLED = str(
    os.getenv("WEAVER_SEMANTIC_ANALYSIS_ENABLED", "1") or "1"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WEAVER_SEMANTIC_NODE_LIMIT = max(
    1,
    int(os.getenv("WEAVER_SEMANTIC_NODE_LIMIT", "48") or "48"),
)
WEAVER_SEMANTIC_LABEL_LIMIT = max(
    1,
    int(os.getenv("WEAVER_SEMANTIC_LABEL_LIMIT", "6") or "6"),
)
WEAVER_SEMANTIC_TIMEOUT_SECONDS = max(
    0.1,
    float(os.getenv("WEAVER_SEMANTIC_TIMEOUT_SECONDS", "1.4") or "1.4"),
)
WEAVER_SEMANTIC_RETRY_COOLDOWN_SECONDS = max(
    0.2,
    float(os.getenv("WEAVER_SEMANTIC_RETRY_COOLDOWN_SECONDS", "12.0") or "12.0"),
)
WEAVER_SEMANTIC_EMBED_DIMS = max(
    8,
    int(os.getenv("WEAVER_SEMANTIC_EMBED_DIMS", "192") or "192"),
)
WEAVER_SEMANTIC_EMBED_PREVIEW_DIMS = max(
    3,
    int(os.getenv("WEAVER_SEMANTIC_EMBED_PREVIEW_DIMS", "24") or "24"),
)
WEAVER_ACTIVE_SEARCH_ENABLED = str(
    os.getenv("WEAVER_ACTIVE_SEARCH_ENABLED", "1") or "1"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WEAVER_ACTIVE_SEARCH_QUERIES = str(
    os.getenv("WEAVER_ACTIVE_SEARCH_QUERIES", "") or ""
).strip()
WEAVER_ACTIVE_SEARCH_INTERVAL_SECONDS = max(
    3.0,
    float(os.getenv("WEAVER_ACTIVE_SEARCH_INTERVAL_SECONDS", "20.0") or "20.0"),
)
WEAVER_ACTIVE_SEARCH_RESULT_LIMIT = max(
    1,
    int(os.getenv("WEAVER_ACTIVE_SEARCH_RESULT_LIMIT", "8") or "8"),
)
WEAVER_ACTIVE_SEARCH_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("WEAVER_ACTIVE_SEARCH_TIMEOUT_SECONDS", "1.4") or "1.4"),
)
WEAVER_ACTIVE_SEARCH_ENGINE = str(
    os.getenv("WEAVER_ACTIVE_SEARCH_ENGINE", "duckduckgo") or "duckduckgo"
).strip()
MULTIMODAL_PRESENCE_RECORD = "ημ.multimodal-presence.v1"
MULTIMODAL_PRESENCE_MAX_ITEMS = max(
    1,
    int(os.getenv("MULTIMODAL_PRESENCE_MAX_ITEMS", "72") or "72"),
)
MULTIMODAL_PRESENCE_CACHE_SECONDS = max(
    0.2,
    float(os.getenv("MULTIMODAL_PRESENCE_CACHE_SECONDS", "2.0") or "2.0"),
)
MULTIMODAL_PRESENCE_AUDIO_WAVE_POINTS = max(
    16,
    int(os.getenv("MULTIMODAL_PRESENCE_AUDIO_WAVE_POINTS", "96") or "96"),
)
MULTIMODAL_PRESENCE_AUDIO_SPECTRO_FRAMES = max(
    4,
    int(os.getenv("MULTIMODAL_PRESENCE_AUDIO_SPECTRO_FRAMES", "20") or "20"),
)
MULTIMODAL_PRESENCE_AUDIO_SPECTRO_BANDS = max(
    4,
    int(os.getenv("MULTIMODAL_PRESENCE_AUDIO_SPECTRO_BANDS", "16") or "16"),
)
TRUTH_BINDING_CACHE_SECONDS = max(
    0.2,
    float(os.getenv("TRUTH_BINDING_CACHE_SECONDS", "1.2") or "1.2"),
)
TRUTH_BINDING_GUARD_THETA = max(
    0.0,
    min(
        1.0,
        float(os.getenv("TRUTH_BINDING_GUARD_THETA", "0.72") or "0.72"),
    ),
)
TRUTH_ALLOWED_PROOF_KINDS = (
    ":logic/bridge",
    ":evidence/record",
    ":score/run",
    ":gov/adjudication",
    ":trace/record",
)
PI_ARCHIVE_REQUIRED_RECEIPT_FIELDS = (
    "ts",
    "kind",
    "origin",
    "owner",
    "dod",
    "refs",
)
ETA_MU_TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".ndjson",
    ".lisp",
    ".sexp",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".tsv",
    ".xml",
    ".html",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".py",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".java",
    ".go",
    ".rs",
    ".sh",
    ".bat",
    ".ps1",
    ".ini",
    ".cfg",
    ".conf",
    ".rtf",
}
ETA_MU_INGEST_INCLUDE_TEXT_MIME = {
    "application/javascript",
    "application/json",
    "application/ld+json",
    "application/toml",
    "application/typescript",
    "application/x-clojure",
    "application/x-lisp",
    "application/x-markdown",
    "application/x-python",
    "application/x-ruby",
    "application/x-scheme",
    "application/x-sh",
    "application/x-yaml",
    "application/xml",
    "text/markdown",
}
ETA_MU_INGEST_INCLUDE_IMAGE_MIME = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}
ETA_MU_INGEST_INCLUDE_TEXT_EXT = {
    ".bash",
    ".c",
    ".clj",
    ".cljs",
    ".cpp",
    ".cs",
    ".edn",
    ".go",
    ".h",
    ".hpp",
    ".hy",
    ".java",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".kt",
    ".lisp",
    ".markdown",
    ".md",
    ".org",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scm",
    ".sh",
    ".sibilant",
    ".ss",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}
ETA_MU_INGEST_INCLUDE_IMAGE_EXT = {
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}
ETA_MU_INGEST_EXCLUDE_REL_PATHS = {
    "_notes",
    "_rejected",
    ".DS_Store",
    ".git",
    "node_modules",
}
ETA_MU_INGEST_EXCLUDE_GLOBS = (
    "**/*.lock",
    "**/*.map",
    "**/*.min.*",
)
ETA_MU_INGEST_MAX_TEXT_BYTES = max(
    1024,
    int(os.getenv("ETA_MU_INGEST_MAX_TEXT_BYTES", "2000000") or "2000000"),
)
ETA_MU_INGEST_MAX_IMAGE_BYTES = max(
    4096,
    int(os.getenv("ETA_MU_INGEST_MAX_IMAGE_BYTES", "20000000") or "20000000"),
)
ETA_MU_INGEST_MAX_SCAN_FILES = max(
    1,
    int(os.getenv("ETA_MU_INGEST_MAX_SCAN_FILES", "5000") or "5000"),
)
ETA_MU_INGEST_MAX_SCAN_DEPTH = max(
    1,
    int(os.getenv("ETA_MU_INGEST_MAX_SCAN_DEPTH", "12") or "12"),
)
ETA_MU_INGEST_TEXT_CHUNK_TARGET = max(
    64,
    int(os.getenv("ETA_MU_INGEST_TEXT_CHUNK_TARGET", "800") or "800"),
)
ETA_MU_INGEST_TEXT_CHUNK_OVERLAP = max(
    0,
    int(os.getenv("ETA_MU_INGEST_TEXT_CHUNK_OVERLAP", "120") or "120"),
)
ETA_MU_INGEST_TEXT_CHUNK_MIN = max(
    32,
    int(os.getenv("ETA_MU_INGEST_TEXT_CHUNK_MIN", "120") or "120"),
)
ETA_MU_INGEST_TEXT_CHUNK_MAX = max(
    ETA_MU_INGEST_TEXT_CHUNK_TARGET,
    int(os.getenv("ETA_MU_INGEST_TEXT_CHUNK_MAX", "1200") or "1200"),
)
ETA_MU_INGEST_SPACE_TEXT_ID = "ημ.text.v1"
ETA_MU_INGEST_SPACE_IMAGE_ID = "ημ.image.v1"
ETA_MU_INGEST_SPACE_SET_ID = "ημ.nexus.v1"
ETA_MU_INGEST_VECSTORE_ID = "vecstore.main"
ETA_MU_INGEST_VECSTORE_COLLECTION = "ημ_nexus_v1"
ETA_MU_INGEST_TEXT_MODEL = str(
    os.getenv("ETA_MU_TEXT_EMBED_MODEL", "nomic-embed-text") or "nomic-embed-text"
).strip()
ETA_MU_INGEST_TEXT_MODEL_DIGEST = str(
    os.getenv("ETA_MU_TEXT_EMBED_DIGEST", "none") or "none"
).strip()
ETA_MU_INGEST_TEXT_DIMS = max(
    64,
    int(os.getenv("ETA_MU_TEXT_EMBED_DIMS", "768") or "768"),
)
ETA_MU_INGEST_IMAGE_MODEL = str(
    os.getenv("ETA_MU_IMAGE_EMBED_MODEL", "nomic-embed-text") or "nomic-embed-text"
).strip()
ETA_MU_INGEST_IMAGE_MODEL_DIGEST = str(
    os.getenv("ETA_MU_IMAGE_EMBED_DIGEST", "none") or "none"
).strip()
ETA_MU_INGEST_IMAGE_DIMS = max(
    64,
    int(os.getenv("ETA_MU_IMAGE_EMBED_DIMS", "768") or "768"),
)
ETA_MU_INGEST_EMBED_TIMEOUT_SECONDS = max(
    0.1,
    float(os.getenv("ETA_MU_INGEST_EMBED_TIMEOUT_SECONDS", "30") or "30"),
)
ETA_MU_INGEST_BATCH_LIMIT = max(
    1,
    int(os.getenv("ETA_MU_INGEST_BATCH_LIMIT", "64") or "64"),
)
ETA_MU_INGEST_MAX_CONCURRENCY = max(
    1,
    int(os.getenv("ETA_MU_INGEST_MAX_CONCURRENCY", "2") or "2"),
)
ETA_MU_INGEST_HEALTH = str(os.getenv("ETA_MU_INGEST_HEALTH", "活") or "活").strip()
ETA_MU_INGEST_STABILITY = str(
    os.getenv("ETA_MU_INGEST_STABILITY", "安") or "安"
).strip()
ETA_MU_INGEST_SAFE_MODE = (
    ETA_MU_INGEST_HEALTH != "活" or ETA_MU_INGEST_STABILITY != "安"
)
ETA_MU_INGEST_CONTRACT_ID = "ημ.ingest.text+image.v1"
ETA_MU_INGEST_PACKET_RECORD = "ημ.packet.v1"
ETA_MU_INGEST_MANIFEST_PREFIX = "ημ_ingest_manifest_"
ETA_MU_INGEST_STATS_PREFIX = "ημ_ingest_stats_"
ETA_MU_INGEST_SNAPSHOT_PREFIX = "ημ_ingest_snapshot_"
ETA_MU_INGEST_OUTPUT_EXT = ".sexp"
ETA_MU_INGEST_VECSTORE_LAYER_MODE = (
    str(os.getenv("ETA_MU_INGEST_VECSTORE_LAYER_MODE", "single") or "single")
    .strip()
    .lower()
)
ETA_MU_FILE_GRAPH_LAYER_LIMIT = max(
    1,
    int(os.getenv("ETA_MU_FILE_GRAPH_LAYER_LIMIT", "8") or "8"),
)
ETA_MU_FILE_GRAPH_LAYER_POINT_LIMIT = max(
    1,
    int(os.getenv("ETA_MU_FILE_GRAPH_LAYER_POINT_LIMIT", "4") or "4"),
)
ETA_MU_FILE_GRAPH_LAYER_BLEND = max(
    0.0,
    min(1.0, float(os.getenv("ETA_MU_FILE_GRAPH_LAYER_BLEND", "0.38") or "0.38")),
)
ETA_MU_FILE_GRAPH_ACTIVE_EMBED_LAYERS = str(
    os.getenv("ETA_MU_FILE_GRAPH_ACTIVE_EMBED_LAYERS", "*") or "*"
).strip()
ETA_MU_FILE_GRAPH_LAYER_ORDER = str(
    os.getenv("ETA_MU_FILE_GRAPH_LAYER_ORDER", "") or ""
).strip()
ETA_MU_FILE_GRAPH_ORGANIZER_CLUSTER_THRESHOLD = max(
    -1.0,
    min(
        1.0,
        float(
            os.getenv("ETA_MU_FILE_GRAPH_ORGANIZER_CLUSTER_THRESHOLD", "0.84") or "0.84"
        ),
    ),
)
ETA_MU_FILE_GRAPH_ORGANIZER_MAX_CONCEPTS = max(
    1,
    int(os.getenv("ETA_MU_FILE_GRAPH_ORGANIZER_MAX_CONCEPTS", "18") or "18"),
)
ETA_MU_FILE_GRAPH_ORGANIZER_MIN_GROUP_SIZE = max(
    1,
    int(os.getenv("ETA_MU_FILE_GRAPH_ORGANIZER_MIN_GROUP_SIZE", "1") or "1"),
)
ETA_MU_FILE_GRAPH_ORGANIZER_TERMS_PER_CONCEPT = max(
    1,
    int(os.getenv("ETA_MU_FILE_GRAPH_ORGANIZER_TERMS_PER_CONCEPT", "3") or "3"),
)
COUNCIL_EVENT_VERSION = "eta-mu.council/v1"
COUNCIL_DECISION_LOG_REL = ".opencode/runtime/council_decisions.v1.jsonl"
COUNCIL_DECISION_HISTORY_LIMIT = max(
    1,
    int(os.getenv("COUNCIL_DECISION_HISTORY_LIMIT", "64") or "64"),
)
COUNCIL_MIN_OVERLAP_MEMBERS = max(
    2,
    int(os.getenv("COUNCIL_MIN_OVERLAP_MEMBERS", "2") or "2"),
)
COUNCIL_MIN_MEMBER_WORLD_INFLUENCE = max(
    0.0,
    min(
        1.0,
        float(os.getenv("COUNCIL_MIN_MEMBER_WORLD_INFLUENCE", "0.35") or "0.35"),
    ),
)
STUDY_EVENT_VERSION = "eta-mu.study/v1"
STUDY_SNAPSHOT_LOG_REL = ".opencode/runtime/study_snapshots.v1.jsonl"
STUDY_SNAPSHOT_HISTORY_LIMIT = max(
    1,
    int(os.getenv("STUDY_SNAPSHOT_HISTORY_LIMIT", "256") or "256"),
)
RESOURCE_SNAPSHOT_CACHE_SECONDS = max(
    0.2,
    float(os.getenv("RESOURCE_SNAPSHOT_CACHE_SECONDS", "1.2") or "1.2"),
)
RESOURCE_LOG_TAIL_MAX_LINES = max(
    4,
    int(os.getenv("RESOURCE_LOG_TAIL_MAX_LINES", "40") or "40"),
)
RESOURCE_LOG_TAIL_MAX_BYTES = max(
    2048,
    int(os.getenv("RESOURCE_LOG_TAIL_MAX_BYTES", "65536") or "65536"),
)
DOCKER_AUTORESTART_ENABLED = str(
    os.getenv("DOCKER_AUTORESTART_ENABLED", "1") or "1"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DOCKER_AUTORESTART_REQUIRE_COUNCIL = str(
    os.getenv("DOCKER_AUTORESTART_REQUIRE_COUNCIL", "1") or "1"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DOCKER_AUTORESTART_COOLDOWN_SECONDS = max(
    0.0,
    float(os.getenv("DOCKER_AUTORESTART_COOLDOWN_SECONDS", "25") or "25"),
)
DOCKER_AUTORESTART_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("DOCKER_AUTORESTART_TIMEOUT_SECONDS", "90") or "90"),
)
DOCKER_AUTORESTART_SERVICES = str(
    os.getenv("DOCKER_AUTORESTART_SERVICES", "eta-mu-system") or "eta-mu-system"
).strip()
DOCKER_AUTORESTART_INCLUDE_GLOBS = str(
    os.getenv(
        "DOCKER_AUTORESTART_INCLUDE_GLOBS",
        "**/*.py,**/*.js,**/*.ts,**/*.tsx,**/*.json,**/*.yml,**/*.yaml,Dockerfile*,docker-compose*.yml",
    )
    or "**/*.py,**/*.js,**/*.ts,**/*.tsx,**/*.json,**/*.yml,**/*.yaml,Dockerfile*,docker-compose*.yml"
).strip()
DOCKER_AUTORESTART_EXCLUDE_GLOBS = str(
    os.getenv(
        "DOCKER_AUTORESTART_EXCLUDE_GLOBS",
        "**/.git/**,**/node_modules/**,**/world_state/**,**/.opencode/runtime/**,**/__pycache__/**",
    )
    or "**/.git/**,**/node_modules/**,**/world_state/**,**/.opencode/runtime/**,**/__pycache__/**"
).strip()
FIELD_TO_PRESENCE = {
    "f1": "receipt_river",
    "f2": "witness_thread",
    "f3": "anchor_registry",
    "f4": "keeper_of_receipts",
    "f5": "fork_tax_canticle",
    "f6": "mage_of_receipts",
    "f7": "gates_of_truth",
    "f8": "anchor_registry",
}
ETA_MU_FIELD_KEYWORDS = {
    "f1": {
        "audio",
        "wav",
        "mp3",
        "m4a",
        "ogg",
        "flac",
        "image",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "cover",
        "art",
        "video",
        "render",
        "mix",
        "stream",
    },
    "f2": {
        "witness",
        "thread",
        "touch",
        "lineage",
        "trace",
        "collapse",
        "observer",
        "entangle",
    },
    "f3": {
        "coherence",
        "focus",
        "atlas",
        "center",
        "balance",
        "catalog",
        "index",
        "reference",
    },
    "f4": {
        "drift",
        "delta",
        "change",
        "patch",
        "churn",
        "error",
        "fail",
        "stale",
        "todo",
        "tmp",
    },
    "f5": {
        "fork",
        "tax",
        "debt",
        "canticle",
        "payment",
        "paid",
        "balance",
        "settle",
    },
    "f6": {
        "prompt",
        "intent",
        "packet",
        "lisp",
        "story",
        "lyrics",
        "song",
        "note",
        "readme",
        "chatgpt",
        "text",
        "memory",
        "research",
    },
    "f7": {
        "gate",
        "truth",
        "receipt",
        "proof",
        "contract",
        "policy",
        "validate",
        "manifest",
        "ledger",
    },
    "f8": {
        "queue",
        "decision",
        "council",
        "runtime",
        "task",
        "ops",
        "status",
        "process",
        "pm2",
    },
}
PROMPTDB_PACKET_GLOBS = ("*.intent.lisp", "*.packet.lisp")
PROMPTDB_CONTRACT_GLOBS = ("*.contract.lisp",)
PROMPTDB_REFRESH_DEBOUNCE_SECONDS = max(
    0.0,
    float(os.getenv("PROMPTDB_REFRESH_DEBOUNCE_SECONDS", "1.2") or "1.2"),
)
PROMPTDB_OPEN_QUESTIONS_PACKET = "diagrams/part64_runtime_system.packet.lisp"
TASK_QUEUE_EVENT_VERSION = "eta-mu.task-queue/v1"
WEAVER_AUTOSTART = str(
    os.getenv("WEAVER_AUTOSTART", "1") or "1"
).strip().lower() not in {
    "0",
    "false",
    "off",
    "no",
}
WEAVER_PORT = int(os.getenv("WEAVER_PORT", "8793") or "8793")
WEAVER_HOST_ENV = str(os.getenv("WEAVER_HOST", "") or "").strip()
TTS_BASE_URL = str(os.getenv("TTS_BASE_URL", "http://127.0.0.1:8788") or "").rstrip("/")
KEEPER_OF_CONTRACTS_PROFILE = {
    "id": "keeper_of_contracts",
    "en": "Keeper of Contracts",
    "ja": "契約の番人",
    "type": "portal",
}
FILE_SENTINEL_PROFILE = {
    "id": "file_sentinel",
    "en": "File Sentinel",
    "ja": "ファイルの哨戒者",
    "type": "network",
}
THE_COUNCIL_PROFILE = {
    "id": "the_council",
    "en": "The Council",
    "ja": "評議会",
    "type": "council",
}
FILE_ORGANIZER_PROFILE = {
    "id": "file_organizer",
    "en": "File Organizer",
    "ja": "ファイル分類師",
    "type": "library",
}
HEALTH_SENTINEL_CPU_PROFILE = {
    "id": "health_sentinel_cpu",
    "en": "Health Sentinel - CPU",
    "ja": "健全監視 - CPU",
    "type": "network",
}
HEALTH_SENTINEL_GPU1_PROFILE = {
    "id": "health_sentinel_gpu1",
    "en": "Health Sentinel - GPU1",
    "ja": "健全監視 - GPU1",
    "type": "network",
}
HEALTH_SENTINEL_GPU2_PROFILE = {
    "id": "health_sentinel_gpu2",
    "en": "Health Sentinel - GPU2",
    "ja": "健全監視 - GPU2",
    "type": "network",
}
HEALTH_SENTINEL_NPU0_PROFILE = {
    "id": "health_sentinel_npu0",
    "en": "Health Sentinel - NPU0",
    "ja": "健全監視 - NPU0",
    "type": "network",
}
FILE_ORGANIZER_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "into",
    "your",
    "you",
    "our",
    "are",
    "was",
    "were",
    "will",
    "have",
    "has",
    "had",
    "about",
    "part",
    "file",
    "files",
    "note",
    "notes",
    "new",
    "old",
    "main",
    "index",
    "readme",
}
_PRESENCE_ALIASES = {
    "keeperofcontracts": "keeper_of_contracts",
    "keeper_of_contracts": "keeper_of_contracts",
    "keeper-of-contracts": "keeper_of_contracts",
    "KeeperOfContracts": "keeper_of_contracts",
    "file_sentinel": "file_sentinel",
    "file-sentinel": "file_sentinel",
    "FileSentinel": "file_sentinel",
    "auto_commit_ghost": "file_sentinel",
    "auto-commit-ghost": "file_sentinel",
    "autocommitghost": "file_sentinel",
    "the_council": "the_council",
    "the-council": "the_council",
    "thecouncil": "the_council",
    "council": "the_council",
    "Council": "the_council",
    "file_organizer": "file_organizer",
    "file-organizer": "file_organizer",
    "fileorganizer": "file_organizer",
    "FileOrganizer": "file_organizer",
    "health_sentinel_cpu": "health_sentinel_cpu",
    "health-sentinel-cpu": "health_sentinel_cpu",
    "healthsentinelcpu": "health_sentinel_cpu",
    "cpu": "health_sentinel_cpu",
    "health_sentinel_gpu1": "health_sentinel_gpu1",
    "health-sentinel-gpu1": "health_sentinel_gpu1",
    "healthsentinelgpu1": "health_sentinel_gpu1",
    "gpu1": "health_sentinel_gpu1",
    "health_sentinel_gpu2": "health_sentinel_gpu2",
    "health-sentinel-gpu2": "health_sentinel_gpu2",
    "healthsentinelgpu2": "health_sentinel_gpu2",
    "gpu2": "health_sentinel_gpu2",
    "health_sentinel_npu0": "health_sentinel_npu0",
    "health-sentinel-npu0": "health_sentinel_npu0",
    "healthsentinelnpu0": "health_sentinel_npu0",
    "npu0": "health_sentinel_npu0",
}

PROJECTION_DEFAULT_PERSPECTIVE = "hybrid"
PROJECTION_PERSPECTIVES: dict[str, dict[str, str]] = {
    "hybrid": {
        "id": "perspective.hybrid",
        "name": "Hybrid",
        "merge": "hybrid",
        "description": "Wallclock ordering with causal overlays.",
    },
    "causal-time": {
        "id": "perspective.causal-time",
        "name": "Causal Time",
        "merge": "causal-time",
        "description": "Prioritize causal links over wallclock sequence.",
    },
    "swimlanes": {
        "id": "perspective.swimlanes",
        "name": "Swimlanes",
        "merge": "swimlanes",
        "description": "Parallel lanes with threaded causality.",
    },
}
PROJECTION_FIELD_SCHEMAS: list[dict[str, Any]] = [
    {
        "field": "f1",
        "name": "artifact_flux",
        "delta_keys": ["audio_ratio", "file_ratio", "mix_sources"],
        "interpretation": {
            "en": "Flow of package/media artifacts entering the field.",
            "ja": "場へ流入する成果物フラックス。",
        },
    },
    {
        "field": "f2",
        "name": "witness_tension",
        "delta_keys": ["click_ratio", "continuity_index", "lineage_links"],
        "interpretation": {
            "en": "Witness pressure and continuity tension across presences.",
            "ja": "証人圧と連続性テンション。",
        },
    },
    {
        "field": "f3",
        "name": "coherence_focus",
        "delta_keys": ["balance_ratio", "promptdb_ratio", "catalog_entropy"],
        "interpretation": {
            "en": "How coherent and centered current world attention is.",
            "ja": "現在の焦点と整合性。",
        },
    },
    {
        "field": "f4",
        "name": "drift_pressure",
        "delta_keys": ["file_ratio", "queue_pending_ratio", "drift_index"],
        "interpretation": {
            "en": "Pressure from unresolved drift and file churn.",
            "ja": "未解決ドリフト圧。",
        },
    },
    {
        "field": "f5",
        "name": "fork_tax_balance",
        "delta_keys": ["paid_ratio", "balance_ratio", "debt"],
        "interpretation": {
            "en": "Current fork-tax settlement state.",
            "ja": "フォーク税の支払い均衡。",
        },
    },
    {
        "field": "f6",
        "name": "curiosity_drive",
        "delta_keys": ["text_ratio", "promptdb_ratio", "interaction_ratio"],
        "interpretation": {
            "en": "Learning and exploration demand in the world.",
            "ja": "探索・学習ドライブ。",
        },
    },
    {
        "field": "f7",
        "name": "gate_pressure",
        "delta_keys": ["queue_ratio", "drift_index", "proof_gap"],
        "interpretation": {
            "en": "Push-truth and gate readiness pressure.",
            "ja": "門圧と真理押出圧。",
        },
    },
    {
        "field": "f8",
        "name": "council_heat",
        "delta_keys": ["queue_events_ratio", "pending_ratio", "decision_load"],
        "interpretation": {
            "en": "Operational contention and governance heat.",
            "ja": "運用・統治ヒート。",
        },
    },
]
PROJECTION_ELEMENTS: list[dict[str, Any]] = [
    {
        "id": "nexus.ui.command_center",
        "kind": "panel",
        "title": "Presence Call Deck",
        "binds_to": ["/api/catalog", "/api/presence/say", "/stream/mix.wav"],
        "field_bindings": {"f2": 0.28, "f3": 0.16, "f6": 0.3, "f7": 0.12, "f8": 0.14},
        "presence": "anchor_registry",
        "tags": ["communication", "webrtc", "presence", "voice"],
        "lane": "voice",
    },
    {
        "id": "nexus.ui.web_graph_weaver",
        "kind": "panel",
        "title": "Web Graph Weaver",
        "binds_to": ["/api/weaver/status", "/ws"],
        "field_bindings": {"f2": 0.18, "f4": 0.34, "f6": 0.2, "f8": 0.28},
        "presence": "witness_thread",
        "tags": ["graph", "drift"],
        "lane": "senses",
    },
    {
        "id": "nexus.ui.inspiration_atlas",
        "kind": "panel",
        "title": "Inspiration Atlas",
        "binds_to": ["/api/catalog"],
        "field_bindings": {"f1": 0.18, "f3": 0.24, "f6": 0.4, "f8": 0.18},
        "presence": "mage_of_receipts",
        "tags": ["atlas", "memory"],
        "lane": "memory",
    },
    {
        "id": "nexus.ui.simulation_map",
        "kind": "overlay",
        "title": "Everything Dashboard",
        "binds_to": ["/api/simulation", "/ws", "/stream/mix.wav"],
        "field_bindings": {"f1": 0.22, "f2": 0.2, "f4": 0.24, "f7": 0.18, "f8": 0.16},
        "presence": "receipt_river",
        "tags": ["simulation", "overlay", "field"],
        "lane": "senses",
    },
    {
        "id": "nexus.ui.chat.witness_thread",
        "kind": "chat-lens",
        "title": "Chat Lens: Witness Thread",
        "binds_to": ["/api/chat", "/api/presence/say"],
        "field_bindings": {"f2": 0.38, "f3": 0.14, "f6": 0.22, "f7": 0.14, "f8": 0.12},
        "presence": "witness_thread",
        "tags": ["chat", "lens", "witness"],
        "lane": "voice",
        "memory_scope": "shared",
    },
    {
        "id": "nexus.ui.entity_vitals",
        "kind": "widget",
        "title": "Entity Vitals",
        "binds_to": ["/api/simulation"],
        "field_bindings": {"f1": 0.18, "f2": 0.22, "f3": 0.2, "f5": 0.2, "f8": 0.2},
        "presence": "keeper_of_receipts",
        "tags": ["vitals", "telemetry"],
        "lane": "voice",
    },
    {
        "id": "nexus.ui.omni_archive",
        "kind": "list",
        "title": "Omni Panel",
        "binds_to": ["/api/catalog", "/api/memories"],
        "field_bindings": {"f1": 0.24, "f3": 0.25, "f6": 0.28, "f8": 0.23},
        "presence": "keeper_of_receipts",
        "tags": ["archive", "covers", "memory"],
        "lane": "memory",
    },
    {
        "id": "nexus.ui.myth_commons",
        "kind": "panel",
        "title": "Myth Commons",
        "binds_to": ["/api/world", "/api/world/interact"],
        "field_bindings": {"f2": 0.22, "f3": 0.22, "f6": 0.2, "f7": 0.2, "f8": 0.16},
        "presence": "gates_of_truth",
        "tags": ["world", "people", "interaction"],
        "lane": "voice",
    },
    {
        "id": "nexus.ui.projection_ledger",
        "kind": "chart",
        "title": "Projection Ledger",
        "binds_to": ["/api/ui/projection"],
        "field_bindings": {"f3": 0.2, "f4": 0.2, "f5": 0.2, "f7": 0.2, "f8": 0.2},
        "presence": "keeper_of_contracts",
        "tags": ["projection", "explainability", "governance"],
        "lane": "council",
    },
    {
        "id": "nexus.ui.autopilot_ledger",
        "kind": "ledger",
        "title": "Autopilot Ledger",
        "binds_to": [
            "/api/study",
            "/api/drift/scan",
            "/api/push-truth/dry-run",
        ],
        "field_bindings": {"f2": 0.14, "f4": 0.26, "f7": 0.3, "f8": 0.3},
        "presence": "gates_of_truth",
        "tags": ["autopilot", "governance", "ledger", "operations"],
        "lane": "council",
    },
    {
        "id": "nexus.ui.stability_observatory",
        "kind": "panel",
        "title": "Stability Observatory",
        "binds_to": [
            "/api/study/snapshot",
            "/api/task/queue",
            "/api/file-graph/summary",
        ],
        "field_bindings": {"f2": 0.2, "f4": 0.3, "f7": 0.22, "f8": 0.28},
        "presence": "gates_of_truth",
        "tags": ["stability", "drift", "study", "governance"],
        "lane": "council",
    },
]

_MIX_CACHE_LOCK = threading.Lock()
_MIX_CACHE: dict[str, Any] = {"fingerprint": "", "wav": b"", "meta": {}}
_WHISPER_MODEL_LOCK = threading.Lock()
_WHISPER_MODEL: Any = None
_PROMPTDB_CACHE_LOCK = threading.Lock()
_PROMPTDB_CACHE: dict[str, Any] = {
    "root": "",
    "signature": "",
    "snapshot": None,
    "checks": 0,
    "refreshes": 0,
    "cache_hits": 0,
    "last_checked_at": "",
    "last_refreshed_at": "",
    "last_decision": "cold-start",
    "last_check_monotonic": 0.0,
}
_ETA_MU_INBOX_LOCK = threading.Lock()
_ETA_MU_INBOX_CACHE: dict[str, Any] = {
    "root": "",
    "last_checked_monotonic": 0.0,
    "snapshot": None,
}
_ETA_MU_KNOWLEDGE_LOCK = threading.Lock()
_ETA_MU_KNOWLEDGE_CACHE: dict[str, Any] = {
    "path": "",
    "mtime_ns": 0,
    "entries": [],
}
_ETA_MU_REGISTRY_LOCK = threading.Lock()
_ETA_MU_REGISTRY_CACHE: dict[str, Any] = {
    "path": "",
    "mtime_ns": 0,
    "entries": [],
}
_EMBEDDINGS_DB_LOCK = threading.Lock()
_EMBEDDINGS_DB_CACHE: dict[str, Any] = {
    "path": "",
    "mtime_ns": 0,
    "state": {},
}
_FILE_GRAPH_MOVES_LOCK = threading.Lock()
_FILE_GRAPH_MOVES_CACHE: dict[str, Any] = {
    "path": "",
    "mtime_ns": 0,
    "moves": {},
}
_OPENVINO_EMBED_LOCK = threading.Lock()
_OPENVINO_EMBED_RUNTIME: dict[str, Any] = {
    "key": "",
    "tokenizer": None,
    "model": None,
    "error": "",
    "loaded_at": "",
}
_WEAVER_GRAPH_CACHE_LOCK = threading.Lock()
_WEAVER_GRAPH_CACHE: dict[str, Any] = {
    "key": "",
    "checked_monotonic": 0.0,
    "snapshot": None,
}
_TRUTH_BINDING_CACHE_LOCK = threading.Lock()
_TRUTH_BINDING_CACHE: dict[str, Any] = {
    "key": "",
    "checked_monotonic": 0.0,
    "snapshot": None,
}
_TENSORFLOW_RUNTIME_LOCK = threading.Lock()
_TENSORFLOW_RUNTIME: dict[str, Any] = {
    "module": None,
    "error": "",
    "loaded_at": "",
    "version": "",
}
_STUDY_SNAPSHOT_LOCK = threading.Lock()
_STUDY_SNAPSHOT_CACHE: dict[str, Any] = {
    "path": "",
    "mtime_ns": 0,
    "events": [],
}
_DAIMO_DYNAMICS_LOCK = threading.Lock()
_DAIMO_DYNAMICS_CACHE: dict[str, Any] = {
    "entities": {},
    "last_gc_monotonic": 0.0,
}
_RESOURCE_MONITOR_LOCK = threading.Lock()
_RESOURCE_MONITOR_CACHE: dict[str, Any] = {
    "checked_monotonic": 0.0,
    "part_root": "",
    "snapshot": None,
}
_RESOURCE_HEARTBEAT_INGEST_LOCK = threading.Lock()
_RESOURCE_HEARTBEAT_INGEST_CACHE: dict[str, Any] = {
    "signature": "",
    "ts": 0.0,
}


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _status_from_utilization(
    utilization: float,
    *,
    watch_threshold: float = 72.0,
    hot_threshold: float = 90.0,
) -> str:
    bounded = max(0.0, min(100.0, float(utilization)))
    if bounded >= hot_threshold:
        return "hot"
    if bounded >= watch_threshold:
        return "watch"
    return "ok"


def _safe_env_metric(name: str, default: float = 0.0) -> float:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _parse_proc_meminfo_mb() -> tuple[float, float]:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists() or not meminfo_path.is_file():
        return 0.0, 0.0
    total_kb = 0.0
    available_kb = 0.0
    try:
        for raw_line in meminfo_path.read_text("utf-8").splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, remainder = line.split(":", 1)
            token = remainder.strip().split(" ", 1)[0]
            try:
                value = float(token)
            except (TypeError, ValueError):
                continue
            if key == "MemTotal":
                total_kb = value
            elif key == "MemAvailable":
                available_kb = value
    except OSError:
        return 0.0, 0.0
    return (total_kb / 1024.0), (available_kb / 1024.0)


def _tail_text_lines(
    path: Path,
    *,
    max_bytes: int,
    max_lines: int,
) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes), os.SEEK_SET)
            data = handle.read(max_bytes)
    except OSError:
        return []

    text = data.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    return lines[-max_lines:]


def _collect_nvidia_metrics() -> list[dict[str, float]]:
    nvidia_smi_bin = shutil.which("nvidia-smi")
    if not nvidia_smi_bin:
        return []
    cmd = [
        nvidia_smi_bin,
        "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=0.45,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []

    rows: list[dict[str, float]] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            utilization = max(0.0, min(100.0, float(parts[0])))
        except (TypeError, ValueError):
            utilization = 0.0
        try:
            memory = max(0.0, min(100.0, float(parts[1])))
        except (TypeError, ValueError):
            memory = 0.0
        try:
            temperature = max(0.0, float(parts[2]))
        except (TypeError, ValueError):
            temperature = 0.0
        rows.append(
            {
                "utilization": round(utilization, 2),
                "memory": round(memory, 2),
                "temperature": round(temperature, 2),
            }
        )
    return rows


def _resource_log_watch(part_root: Path | None = None) -> dict[str, Any]:
    if part_root is None:
        return {
            "path": "",
            "line_count": 0,
            "error_count": 0,
            "warn_count": 0,
            "error_ratio": 0.0,
            "warn_ratio": 0.0,
            "latest": "",
        }

    candidates = [
        part_root / "world_state" / "weaver-error.log",
        part_root / "world_state" / "weaver-out.log",
        part_root / "world_state" / "world-web.log",
    ]
    log_path = next(
        (path for path in candidates if path.exists() and path.is_file()), None
    )
    if log_path is None:
        return {
            "path": "",
            "line_count": 0,
            "error_count": 0,
            "warn_count": 0,
            "error_ratio": 0.0,
            "warn_ratio": 0.0,
            "latest": "",
        }

    tail = _tail_text_lines(
        log_path,
        max_bytes=RESOURCE_LOG_TAIL_MAX_BYTES,
        max_lines=RESOURCE_LOG_TAIL_MAX_LINES,
    )
    lowered = [line.lower() for line in tail]
    error_tokens = ("error", "traceback", "exception", "fatal", "panic")
    warn_tokens = ("warn", "warning", "timeout", "retry", "blocked")
    error_count = sum(
        1 for line in lowered if any(token in line for token in error_tokens)
    )
    warn_count = sum(
        1 for line in lowered if any(token in line for token in warn_tokens)
    )
    line_count = len(tail)
    latest = tail[-1] if tail else ""

    return {
        "path": str(log_path),
        "line_count": line_count,
        "error_count": error_count,
        "warn_count": warn_count,
        "error_ratio": round(error_count / max(1, line_count), 4),
        "warn_ratio": round(warn_count / max(1, line_count), 4),
        "latest": latest[:240],
    }


def _resource_auto_embedding_order(
    snapshot: dict[str, Any] | None = None,
) -> list[str]:
    payload = snapshot if isinstance(snapshot, dict) else _resource_monitor_snapshot()
    devices = payload.get("devices", {}) if isinstance(payload, dict) else {}
    npu = devices.get("npu0", {}) if isinstance(devices, dict) else {}
    cpu = devices.get("cpu", {}) if isinstance(devices, dict) else {}
    gpu1 = devices.get("gpu1", {}) if isinstance(devices, dict) else {}

    npu_status = str(npu.get("status", "ok")).strip().lower()
    cpu_utilization = _safe_float(cpu.get("utilization", 0.0), 0.0)
    gpu_utilization = _safe_float(gpu1.get("utilization", 0.0), 0.0)
    openvino_ready = bool(str(os.getenv("OPENVINO_EMBED_ENDPOINT", "") or "").strip())
    if not openvino_ready:
        openvino_device = str(
            os.getenv("OPENVINO_EMBED_DEVICE", "NPU") or "NPU"
        ).upper()
        openvino_ready = "NPU" in openvino_device

    order: list[str] = []
    if openvino_ready and npu_status != "hot":
        order.append("openvino")
    if cpu_utilization < 95.0:
        order.append("tensorflow")
    if gpu_utilization < 92.0:
        order.append("ollama")
    order.extend(["tensorflow", "ollama", "openvino"])

    deduped: list[str] = []
    seen: set[str] = set()
    for item in order:
        key = str(item).strip().lower()
        if key not in {"openvino", "tensorflow", "ollama"}:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _resource_auto_text_order(
    snapshot: dict[str, Any] | None = None,
) -> list[str]:
    payload = snapshot if isinstance(snapshot, dict) else _resource_monitor_snapshot()
    devices = payload.get("devices", {}) if isinstance(payload, dict) else {}
    cpu = devices.get("cpu", {}) if isinstance(devices, dict) else {}
    gpu1 = devices.get("gpu1", {}) if isinstance(devices, dict) else {}
    log_watch = payload.get("log_watch", {}) if isinstance(payload, dict) else {}

    cpu_utilization = _safe_float(cpu.get("utilization", 0.0), 0.0)
    gpu_utilization = _safe_float(gpu1.get("utilization", 0.0), 0.0)
    error_ratio = _safe_float(log_watch.get("error_ratio", 0.0), 0.0)

    if cpu_utilization >= 88.0 or error_ratio >= 0.5:
        preferred = ["ollama", "tensorflow"]
    elif gpu_utilization < 85.0:
        preferred = ["ollama", "tensorflow"]
    else:
        preferred = ["tensorflow", "ollama"]

    deduped: list[str] = []
    seen: set[str] = set()
    for item in [*preferred, "tensorflow", "ollama"]:
        key = str(item).strip().lower()
        if key not in {"tensorflow", "ollama"}:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _resource_monitor_snapshot(part_root: Path | None = None) -> dict[str, Any]:
    now_monotonic = time.monotonic()
    part_key = str(part_root.resolve()) if isinstance(part_root, Path) else ""

    with _RESOURCE_MONITOR_LOCK:
        cached_checked = _safe_float(
            _RESOURCE_MONITOR_CACHE.get("checked_monotonic", 0.0), 0.0
        )
        cached_part = str(_RESOURCE_MONITOR_CACHE.get("part_root", ""))
        cached_snapshot = _RESOURCE_MONITOR_CACHE.get("snapshot")
        if (
            isinstance(cached_snapshot, dict)
            and (now_monotonic - cached_checked) <= RESOURCE_SNAPSHOT_CACHE_SECONDS
            and cached_part == part_key
        ):
            return _json_deep_clone(cached_snapshot)

    cpu_count = max(1, int(os.cpu_count() or 1))
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except OSError:
        load_1m, load_5m, load_15m = (0.0, 0.0, 0.0)
    cpu_utilization = _clamp01(load_1m / max(1, cpu_count)) * 100.0

    memory_total_mb, memory_available_mb = _parse_proc_meminfo_mb()
    memory_pressure = 0.0
    if memory_total_mb > 0.0:
        memory_pressure = _clamp01(
            (memory_total_mb - max(0.0, memory_available_mb)) / memory_total_mb
        )

    nvidia_rows = _collect_nvidia_metrics()
    gpu1_default = nvidia_rows[0] if len(nvidia_rows) >= 1 else {}
    gpu2_default = nvidia_rows[1] if len(nvidia_rows) >= 2 else {}

    openvino_device = (
        str(os.getenv("OPENVINO_EMBED_DEVICE", "NPU") or "NPU").strip() or "NPU"
    )
    with _OPENVINO_EMBED_LOCK:
        openvino_loaded = _OPENVINO_EMBED_RUNTIME.get("model") is not None

    gpu1_utilization = _safe_env_metric(
        "ETA_MU_GPU1_UTILIZATION",
        _safe_float(gpu1_default.get("utilization", 0.0), 0.0),
    )
    gpu1_memory = _safe_env_metric(
        "ETA_MU_GPU1_MEMORY",
        _safe_float(gpu1_default.get("memory", 0.0), 0.0),
    )
    gpu1_temperature = _safe_env_metric(
        "ETA_MU_GPU1_TEMP",
        _safe_float(gpu1_default.get("temperature", 0.0), 0.0),
    )

    gpu2_utilization = _safe_env_metric(
        "ETA_MU_GPU2_UTILIZATION",
        _safe_float(gpu2_default.get("utilization", 0.0), 0.0),
    )
    gpu2_memory = _safe_env_metric(
        "ETA_MU_GPU2_MEMORY",
        _safe_float(gpu2_default.get("memory", 0.0), 0.0),
    )
    gpu2_temperature = _safe_env_metric(
        "ETA_MU_GPU2_TEMP",
        _safe_float(gpu2_default.get("temperature", 0.0), 0.0),
    )

    npu_util_default = (
        22.0 if ("NPU" in openvino_device.upper() and openvino_loaded) else 0.0
    )
    npu_utilization = _safe_env_metric("ETA_MU_NPU0_UTILIZATION", npu_util_default)
    npu_queue_depth = max(0.0, _safe_env_metric("ETA_MU_NPU0_QUEUE_DEPTH", 0.0))
    npu_temperature = _safe_env_metric("ETA_MU_NPU0_TEMP", 0.0)

    log_watch = _resource_log_watch(part_root=part_root)
    devices = {
        "cpu": {
            "utilization": round(cpu_utilization, 2),
            "load_avg": {
                "m1": round(load_1m, 3),
                "m5": round(load_5m, 3),
                "m15": round(load_15m, 3),
            },
            "memory_pressure": round(memory_pressure, 4),
            "status": _status_from_utilization(
                cpu_utilization, watch_threshold=70.0, hot_threshold=88.0
            ),
        },
        "gpu1": {
            "utilization": round(max(0.0, min(100.0, gpu1_utilization)), 2),
            "memory": round(max(0.0, min(100.0, gpu1_memory)), 2),
            "temperature": round(max(0.0, gpu1_temperature), 2),
            "status": _status_from_utilization(
                gpu1_utilization, watch_threshold=76.0, hot_threshold=93.0
            ),
        },
        "gpu2": {
            "utilization": round(max(0.0, min(100.0, gpu2_utilization)), 2),
            "memory": round(max(0.0, min(100.0, gpu2_memory)), 2),
            "temperature": round(max(0.0, gpu2_temperature), 2),
            "status": _status_from_utilization(
                gpu2_utilization, watch_threshold=76.0, hot_threshold=93.0
            ),
        },
        "npu0": {
            "utilization": round(max(0.0, min(100.0, npu_utilization)), 2),
            "queue_depth": int(max(0.0, npu_queue_depth)),
            "temperature": round(max(0.0, npu_temperature), 2),
            "device": openvino_device,
            "status": _status_from_utilization(
                npu_utilization, watch_threshold=78.0, hot_threshold=95.0
            ),
        },
    }
    hot_devices = [
        device_id
        for device_id, row in devices.items()
        if isinstance(row, dict) and str(row.get("status", "")) == "hot"
    ]

    snapshot = {
        "record": "ημ.resource-heartbeat.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_seconds": RESOURCE_SNAPSHOT_CACHE_SECONDS,
        "host": {
            "cpu_count": cpu_count,
            "memory_total_mb": round(memory_total_mb, 2),
            "memory_available_mb": round(memory_available_mb, 2),
        },
        "devices": devices,
        "log_watch": log_watch,
        "hot_devices": hot_devices,
    }
    snapshot["auto_backend"] = {
        "embeddings_order": _resource_auto_embedding_order(snapshot=snapshot),
        "text_order": _resource_auto_text_order(snapshot=snapshot),
    }

    with _RESOURCE_MONITOR_LOCK:
        _RESOURCE_MONITOR_CACHE["checked_monotonic"] = now_monotonic
        _RESOURCE_MONITOR_CACHE["part_root"] = part_key
        _RESOURCE_MONITOR_CACHE["snapshot"] = _json_deep_clone(snapshot)

    return snapshot


def _ingest_resource_heartbeat_memory(heartbeat: dict[str, Any]) -> None:
    if not isinstance(heartbeat, dict):
        return

    devices = heartbeat.get("devices", {}) if isinstance(heartbeat, dict) else {}
    cpu = devices.get("cpu", {}) if isinstance(devices, dict) else {}
    gpu1 = devices.get("gpu1", {}) if isinstance(devices, dict) else {}
    npu = devices.get("npu0", {}) if isinstance(devices, dict) else {}
    log_watch = heartbeat.get("log_watch", {}) if isinstance(heartbeat, dict) else {}
    latest_log = str(log_watch.get("latest", "")).strip()

    summary_line = (
        "resource-heartbeat "
        f"cpu={_safe_float(cpu.get('utilization', 0.0), 0.0):.1f}% "
        f"gpu1={_safe_float(gpu1.get('utilization', 0.0), 0.0):.1f}% "
        f"npu0={_safe_float(npu.get('utilization', 0.0), 0.0):.1f}% "
        f"errors={int(_safe_float(log_watch.get('error_count', 0), 0.0))}"
    )
    if latest_log:
        summary_line += " log=" + latest_log[:120]

    signature = hashlib.sha1(summary_line.encode("utf-8")).hexdigest()
    now_monotonic = time.monotonic()
    with _RESOURCE_HEARTBEAT_INGEST_LOCK:
        last_signature = str(_RESOURCE_HEARTBEAT_INGEST_CACHE.get("signature", ""))
        last_ts = _safe_float(_RESOURCE_HEARTBEAT_INGEST_CACHE.get("ts", 0.0), 0.0)
        if signature == last_signature and (now_monotonic - last_ts) <= 18.0:
            return
        _RESOURCE_HEARTBEAT_INGEST_CACHE["signature"] = signature
        _RESOURCE_HEARTBEAT_INGEST_CACHE["ts"] = now_monotonic

    collection = _get_chroma_collection()
    if collection is None:
        return
    embedding = _ollama_embed(summary_line)
    if embedding is None:
        return

    try:
        collection.add(
            ids=[f"resource_{int(time.time() * 1000)}"],
            embeddings=[embedding],
            metadatas=[
                {
                    "type": "resource_heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "hot_device_count": len(heartbeat.get("hot_devices", [])),
                    "error_count": int(
                        _safe_float(
                            (heartbeat.get("log_watch", {}) or {}).get(
                                "error_count", 0
                            ),
                            0.0,
                        )
                    ),
                }
            ],
            documents=[summary_line],
        )
    except Exception:
        pass


class RuntimeInfluenceTracker:
    CLICK_WINDOW_SECONDS = 45.0
    FILE_WINDOW_SECONDS = 120.0
    LOG_WINDOW_SECONDS = 180.0
    RESOURCE_WINDOW_SECONDS = 180.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._click_events: list[dict[str, Any]] = []
        self._file_events: list[dict[str, Any]] = []
        self._log_events: list[dict[str, Any]] = []
        self._resource_events: list[dict[str, Any]] = []
        self._fork_tax_debt = 0.0
        self._fork_tax_paid = 0.0

    def _prune(self, now: float) -> None:
        self._click_events = [
            row
            for row in self._click_events
            if (now - float(row.get("ts", 0.0))) <= self.CLICK_WINDOW_SECONDS
        ]
        self._file_events = [
            row
            for row in self._file_events
            if (now - float(row.get("ts", 0.0))) <= self.FILE_WINDOW_SECONDS
        ]
        self._log_events = [
            row
            for row in self._log_events
            if (now - float(row.get("ts", 0.0))) <= self.LOG_WINDOW_SECONDS
        ]
        self._resource_events = [
            row
            for row in self._resource_events
            if (now - float(row.get("ts", 0.0))) <= self.RESOURCE_WINDOW_SECONDS
        ]

    def record_witness(self, *, event_type: str, target: str) -> None:
        now = time.time()
        with self._lock:
            self._click_events.append(
                {
                    "ts": now,
                    "event_type": event_type,
                    "target": target,
                }
            )
            payment = 1.0
            if "fork" in target.lower() or "tax" in target.lower():
                payment += 1.0
            if event_type == "world_interact":
                payment += 0.25
            self._fork_tax_paid += payment
            self._prune(now)

    def record_file_delta(self, delta: dict[str, Any]) -> None:
        added_count = int(delta.get("added_count", 0))
        updated_count = int(delta.get("updated_count", 0))
        removed_count = int(delta.get("removed_count", 0))
        total = max(0, added_count + updated_count + removed_count)
        if total <= 0:
            return

        score = (added_count * 1.6) + (updated_count * 1.0) + (removed_count * 1.3)
        now = time.time()
        with self._lock:
            self._file_events.append(
                {
                    "ts": now,
                    "added": added_count,
                    "updated": updated_count,
                    "removed": removed_count,
                    "changes": total,
                    "score": round(score, 3),
                    "sample_paths": list(delta.get("sample_paths", []))[:6],
                }
            )
            self._fork_tax_debt += score
            self._prune(now)

    def record_resource_heartbeat(
        self,
        heartbeat: dict[str, Any],
        *,
        source: str = "runtime",
    ) -> None:
        if not isinstance(heartbeat, dict):
            return
        now = time.time()
        with self._lock:
            self._resource_events.append(
                {
                    "ts": now,
                    "source": str(source).strip() or "runtime",
                    "heartbeat": _json_deep_clone(heartbeat),
                }
            )
            self._prune(now)

    def record_runtime_log(
        self,
        *,
        level: str,
        message: str,
        source: str = "runtime",
    ) -> None:
        clean_message = str(message).strip()
        if not clean_message:
            return
        now = time.time()
        with self._lock:
            self._log_events.append(
                {
                    "ts": now,
                    "source": str(source).strip() or "runtime",
                    "level": str(level).strip().lower() or "info",
                    "message": clean_message[:240],
                }
            )
            self._prune(now)

    def pay_fork_tax(
        self,
        *,
        amount: float,
        source: str,
        target: str,
    ) -> dict[str, Any]:
        applied = max(0.25, min(float(amount), 144.0))
        now = time.time()
        with self._lock:
            self._fork_tax_paid += applied
            self._click_events.append(
                {
                    "ts": now,
                    "event_type": "fork_tax_payment",
                    "target": target,
                    "source": source,
                    "amount": round(applied, 3),
                }
            )
            self._prune(now)
        return {
            "applied": round(applied, 3),
            "source": source,
            "target": target,
        }

    def snapshot(
        self,
        queue_snapshot: dict[str, Any] | None = None,
        *,
        part_root: Path | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        queue_snapshot = queue_snapshot or {}
        with self._lock:
            self._prune(now)
            click_rows = list(self._click_events)
            file_rows = list(self._file_events)
            log_rows = list(self._log_events)
            resource_rows = list(self._resource_events)
            fork_tax_debt = float(self._fork_tax_debt)
            fork_tax_paid = float(self._fork_tax_paid)

        clicks_recent = len(click_rows)
        file_changes_recent = sum(int(row.get("changes", 0)) for row in file_rows)
        queue_event_count = int(queue_snapshot.get("event_count", 0))
        queue_pending_count = int(queue_snapshot.get("pending_count", 0))

        paid_effective = fork_tax_paid + (queue_event_count * 0.25)
        balance = max(0.0, fork_tax_debt - paid_effective)
        paid_ratio = (
            1.0 if fork_tax_debt <= 0 else _clamp01(paid_effective / fork_tax_debt)
        )

        auto_commit_pulse = _clamp01(
            (file_changes_recent * 0.08)
            + (queue_event_count * 0.05)
            + (queue_pending_count * 0.06)
        )
        if queue_pending_count > 0:
            status_en = "staging receipts"
            status_ja = "領収書を段取り中"
        elif file_changes_recent > 0:
            status_en = "watching drift"
            status_ja = "ドリフトを監視中"
        else:
            status_en = "gate idle"
            status_ja = "門前で待機中"

        recent_targets = [
            str(row.get("target", ""))
            for row in sorted(
                click_rows, key=lambda item: float(item.get("ts", 0.0)), reverse=True
            )
            if row.get("target")
        ][:6]
        recent_file_paths: list[str] = []
        for row in sorted(
            file_rows, key=lambda item: float(item.get("ts", 0.0)), reverse=True
        ):
            for path in row.get("sample_paths", []):
                value = str(path).strip()
                if value and value not in recent_file_paths:
                    recent_file_paths.append(value)
                if len(recent_file_paths) >= 8:
                    break
            if len(recent_file_paths) >= 8:
                break

        recent_logs = sorted(
            log_rows,
            key=lambda item: float(item.get("ts", 0.0)),
            reverse=True,
        )
        latest_log = recent_logs[0] if recent_logs else {}
        log_error_count = sum(
            1
            for row in log_rows
            if str(row.get("level", "")).lower() in {"error", "fatal"}
        )
        log_warn_count = sum(
            1
            for row in log_rows
            if str(row.get("level", "")).lower() in {"warn", "warning"}
        )

        latest_resource_event = None
        if resource_rows:
            latest_resource_event = max(
                resource_rows,
                key=lambda item: float(item.get("ts", 0.0)),
            )
        resource_heartbeat = (
            _json_deep_clone(latest_resource_event.get("heartbeat", {}))
            if isinstance(latest_resource_event, dict)
            else None
        )
        if not isinstance(resource_heartbeat, dict) or not resource_heartbeat:
            resource_heartbeat = _resource_monitor_snapshot(part_root=part_root)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "clicks_45s": clicks_recent,
            "file_changes_120s": file_changes_recent,
            "log_events_180s": len(log_rows),
            "resource_events_180s": len(resource_rows),
            "recent_click_targets": recent_targets,
            "recent_file_paths": recent_file_paths,
            "recent_logs": [
                {
                    "level": str(row.get("level", "info")),
                    "source": str(row.get("source", "runtime")),
                    "message": str(row.get("message", "")),
                }
                for row in recent_logs[:4]
            ],
            "last_log": {
                "level": str(latest_log.get("level", "")),
                "source": str(latest_log.get("source", "")),
                "message": str(latest_log.get("message", "")),
            },
            "log_summary": {
                "event_count": len(log_rows),
                "error_count": log_error_count,
                "warn_count": log_warn_count,
            },
            "resource_heartbeat": resource_heartbeat,
            "fork_tax": {
                "law_en": "Pay the fork tax; annotate every drift with proof.",
                "law_ja": "フォーク税は法。ドリフトごとに証明を注釈せよ。",
                "debt": round(fork_tax_debt, 3),
                "paid": round(paid_effective, 3),
                "balance": round(balance, 3),
                "paid_ratio": round(paid_ratio, 4),
            },
            "ghost": {
                "id": FILE_SENTINEL_PROFILE["id"],
                "en": FILE_SENTINEL_PROFILE["en"],
                "ja": FILE_SENTINEL_PROFILE["ja"],
                "auto_commit_pulse": round(auto_commit_pulse, 4),
                "queue_pending": queue_pending_count,
                "status_en": status_en,
                "status_ja": status_ja,
            },
        }


_MYTH_TRACKER = _load_myth_tracker_class()()
_LIFE_TRACKER = _load_life_tracker_class()()
_LIFE_INTERACTION_BUILDER = _load_life_interaction_builder()
_INFLUENCE_TRACKER = RuntimeInfluenceTracker()
_WEAVER_PROCESS: subprocess.Popen[Any] | None = None
_WEAVER_BOOT_LOCK = threading.Lock()

CLAIM_CUE_RE = re.compile(
    r"\b(should|will|obvious|clearly|must|need to|going to|plan to|we should|we will|it's obvious|it is obvious)\b",
    re.IGNORECASE,
)
COMMIT_RE = re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)\]}>\"']+", re.IGNORECASE)
FILE_RE = re.compile(r"\b[./~]?[-\w]+(?:/[-\w.]+)+\.[a-z0-9]{1,8}\b", re.IGNORECASE)
PR_RE = re.compile(r"\b(?:PR|pr)\s*#\d+\b")

OVERLAY_TAGS = ("[[PULSE]]", "[[GLITCH]]", "[[SING]]")
CHAT_TOOLS_BY_TYPE = {
    "flow": ["sing_line", "pulse_tag"],
    "network": ["echo_proof", "sing_line"],
    "glitch": ["glitch_tag", "sing_line"],
    "geo": ["anchor_register", "pulse_tag"],
    "portal": ["truth_gate", "sing_line"],
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
    global _WEAVER_PROCESS

    if not WEAVER_AUTOSTART:
        return

    default_bind_host = "127.0.0.1"
    if world_host not in {"127.0.0.1", "localhost"}:
        default_bind_host = "0.0.0.0"
    bind_host = WEAVER_HOST_ENV or default_bind_host
    probe_host = _weaver_probe_host(bind_host)

    if _weaver_health_check(probe_host, WEAVER_PORT):
        return

    with _WEAVER_BOOT_LOCK:
        if _weaver_health_check(probe_host, WEAVER_PORT):
            return

        if _WEAVER_PROCESS is not None and _WEAVER_PROCESS.poll() is None:
            return

        node_bin = shutil.which("node")
        if node_bin is None:
            print("[world-web] weaver autostart skipped: node is not installed")
            return

        script_path = part_root / "code" / "web_graph_weaver.js"
        if not script_path.exists():
            print(f"[world-web] weaver autostart skipped: missing {script_path}")
            return

        world_state_dir = part_root / "world_state"
        world_state_dir.mkdir(parents=True, exist_ok=True)
        out_log = world_state_dir / "weaver-out.log"
        err_log = world_state_dir / "weaver-error.log"

        env = os.environ.copy()
        env.setdefault("WEAVER_HOST", bind_host)
        env.setdefault("WEAVER_PORT", str(WEAVER_PORT))

        with open(out_log, "ab") as stdout_handle, open(err_log, "ab") as stderr_handle:
            _WEAVER_PROCESS = subprocess.Popen(
                [node_bin, str(script_path)],
                cwd=str(script_path.parent),
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )

        deadline = time.time() + 3.5
        while time.time() < deadline:
            if _weaver_health_check(probe_host, WEAVER_PORT, timeout_s=0.45):
                print(
                    f"[world-web] web graph weaver ready: http://{probe_host}:{WEAVER_PORT}/"
                )
                return
            if _WEAVER_PROCESS is not None and _WEAVER_PROCESS.poll() is not None:
                break
            time.sleep(0.2)

        return_code = (
            _WEAVER_PROCESS.poll() if _WEAVER_PROCESS is not None else "unknown"
        )
        print(f"[world-web] weaver autostart failed (rc={return_code}). See {err_log}")


def load_manifest(part_root: Path) -> dict[str, Any]:
    manifest_path = part_root / "manifest.json"
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


try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

_CHROMA_CLIENT: Any = None


def _get_chroma_collection():
    global _CHROMA_CLIENT
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
        return _CHROMA_CLIENT.get_or_create_collection(name="world_memories")
    except Exception:
        return None


def _embeddings_db_path(vault_root: Path) -> Path:
    base = vault_root.resolve()
    return (base / ETA_MU_EMBEDDINGS_DB_REL).resolve()


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


def _clone_embedding_record(record: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(record)
    embedding = record.get("embedding", [])
    metadata = record.get("metadata", {})
    cloned["embedding"] = list(embedding) if isinstance(embedding, list) else []
    cloned["metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
    return cloned


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


def _append_embeddings_db_event(vault_root: Path, record: dict[str, Any]) -> None:
    db_path = _embeddings_db_path(vault_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with db_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with _EMBEDDINGS_DB_LOCK:
        _EMBEDDINGS_DB_CACHE["path"] = ""
        _EMBEDDINGS_DB_CACHE["mtime_ns"] = 0
        _EMBEDDINGS_DB_CACHE["state"] = {}


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


def _embedding_db_upsert(
    vault_root: Path,
    *,
    entry_id: str,
    text: str,
    embedding: list[float],
    metadata: dict[str, Any] | None,
    model: str | None,
) -> dict[str, Any]:
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


def _embedding_db_delete(vault_root: Path, *, entry_id: str) -> dict[str, Any]:
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


def _file_graph_moves_path(vault_root: Path) -> Path:
    base = vault_root.resolve()
    return (base / ETA_MU_GRAPH_MOVES_REL).resolve()


def _clone_file_graph_moves(
    moves: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(file_id): {
            "file_id": str(row.get("file_id", file_id)),
            "x": round(_clamp01(_safe_float(row.get("x", 0.5), 0.5)), 4),
            "y": round(_clamp01(_safe_float(row.get("y", 0.5), 0.5)), 4),
            "moved_at": str(row.get("moved_at", "")),
            "moved_by": str(row.get("moved_by", "Err") or "Err"),
            "note": str(row.get("note", "")),
        }
        for file_id, row in moves.items()
        if str(file_id).strip() and isinstance(row, dict)
    }


def _load_file_graph_moves_state(vault_root: Path) -> dict[str, Any]:
    path = _file_graph_moves_path(vault_root)
    mtime_ns = 0
    if path.exists() and path.is_file():
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0

    with _FILE_GRAPH_MOVES_LOCK:
        if str(_FILE_GRAPH_MOVES_CACHE.get("path", "")) == str(path) and int(
            _FILE_GRAPH_MOVES_CACHE.get("mtime_ns", 0) or 0
        ) == int(mtime_ns):
            cached_moves = _FILE_GRAPH_MOVES_CACHE.get("moves", {})
            if isinstance(cached_moves, dict):
                return {
                    "record": ETA_MU_FILE_GRAPH_MOVES_RECORD,
                    "path": str(path),
                    "updated_at": str(_FILE_GRAPH_MOVES_CACHE.get("updated_at", "")),
                    "moves": _clone_file_graph_moves(cached_moves),
                }

    payload: dict[str, Any] = {}
    if path.exists() and path.is_file():
        try:
            raw = json.loads(path.read_text("utf-8"))
            if isinstance(raw, dict):
                payload = raw
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}

    updated_at = str(payload.get("updated_at", "")).strip()
    raw_moves = payload.get("moves", {})
    if not isinstance(raw_moves, dict):
        raw_moves = {}

    normalized_moves: dict[str, dict[str, Any]] = {}
    for file_id_raw, row in raw_moves.items():
        file_id = str(file_id_raw).strip()
        if not file_id or not isinstance(row, dict):
            continue
        moved_at = str(row.get("moved_at", "")).strip() or updated_at
        normalized_moves[file_id] = {
            "file_id": file_id,
            "x": round(_clamp01(_safe_float(row.get("x", 0.5), 0.5)), 4),
            "y": round(_clamp01(_safe_float(row.get("y", 0.5), 0.5)), 4),
            "moved_at": moved_at,
            "moved_by": str(row.get("moved_by", "Err") or "Err").strip() or "Err",
            "note": str(row.get("note", "") or "").strip(),
        }

    with _FILE_GRAPH_MOVES_LOCK:
        _FILE_GRAPH_MOVES_CACHE["path"] = str(path)
        _FILE_GRAPH_MOVES_CACHE["mtime_ns"] = int(mtime_ns)
        _FILE_GRAPH_MOVES_CACHE["updated_at"] = updated_at
        _FILE_GRAPH_MOVES_CACHE["moves"] = _clone_file_graph_moves(normalized_moves)

    return {
        "record": ETA_MU_FILE_GRAPH_MOVES_RECORD,
        "path": str(path),
        "updated_at": updated_at,
        "moves": normalized_moves,
    }


def _write_file_graph_moves_state(
    vault_root: Path,
    moves: dict[str, dict[str, Any]],
    *,
    updated_at: str | None = None,
) -> dict[str, Any]:
    path = _file_graph_moves_path(vault_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = str(updated_at or datetime.now(timezone.utc).isoformat())
    normalized_moves = _clone_file_graph_moves(moves)
    payload = {
        "record": ETA_MU_FILE_GRAPH_MOVES_RECORD,
        "updated_at": timestamp,
        "moves": {
            key: normalized_moves[key] for key in sorted(normalized_moves.keys())
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    mtime_ns = 0
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0

    with _FILE_GRAPH_MOVES_LOCK:
        _FILE_GRAPH_MOVES_CACHE["path"] = str(path)
        _FILE_GRAPH_MOVES_CACHE["mtime_ns"] = int(mtime_ns)
        _FILE_GRAPH_MOVES_CACHE["updated_at"] = timestamp
        _FILE_GRAPH_MOVES_CACHE["moves"] = _clone_file_graph_moves(normalized_moves)

    return {
        "record": ETA_MU_FILE_GRAPH_MOVES_RECORD,
        "path": str(path),
        "updated_at": timestamp,
        "moves": normalized_moves,
    }


def _file_graph_moves_snapshot(vault_root: Path) -> dict[str, Any]:
    state = _load_file_graph_moves_state(vault_root)
    moves = state.get("moves", {}) if isinstance(state, dict) else {}
    if not isinstance(moves, dict):
        moves = {}
    rows = [moves[key] for key in sorted(moves.keys())]
    return {
        "ok": True,
        "record": ETA_MU_FILE_GRAPH_MOVES_RECORD,
        "path": str(state.get("path", "")),
        "updated_at": str(state.get("updated_at", "")),
        "count": len(rows),
        "moves": rows,
    }


def _file_graph_set_move(
    vault_root: Path,
    *,
    file_id: str,
    x: float,
    y: float,
    moved_by: str,
    note: str = "",
) -> dict[str, Any]:
    key = str(file_id).strip()
    if not key:
        return {"ok": False, "error": "missing file_id"}

    state = _load_file_graph_moves_state(vault_root)
    moves = state.get("moves", {}) if isinstance(state, dict) else {}
    if not isinstance(moves, dict):
        moves = {}
    mutable_moves = _clone_file_graph_moves(moves)

    row = {
        "file_id": key,
        "x": round(_clamp01(_safe_float(x, 0.5)), 4),
        "y": round(_clamp01(_safe_float(y, 0.5)), 4),
        "moved_at": datetime.now(timezone.utc).isoformat(),
        "moved_by": str(moved_by or "Err").strip() or "Err",
        "note": str(note or "").strip(),
    }
    mutable_moves[key] = row
    snapshot = _write_file_graph_moves_state(vault_root, mutable_moves)
    return {
        "ok": True,
        "status": "upserted",
        "move": row,
        "count": len(snapshot.get("moves", {})),
        "updated_at": str(snapshot.get("updated_at", "")),
        "path": str(snapshot.get("path", "")),
    }


def _file_graph_clear_move(vault_root: Path, *, file_id: str) -> dict[str, Any]:
    key = str(file_id).strip()
    if not key:
        return {"ok": False, "error": "missing file_id"}

    state = _load_file_graph_moves_state(vault_root)
    moves = state.get("moves", {}) if isinstance(state, dict) else {}
    if not isinstance(moves, dict):
        moves = {}
    mutable_moves = _clone_file_graph_moves(moves)
    removed = mutable_moves.pop(key, None)
    if removed is None:
        return {"ok": False, "error": "move not found", "file_id": key}

    snapshot = _write_file_graph_moves_state(vault_root, mutable_moves)
    return {
        "ok": True,
        "status": "deleted",
        "file_id": key,
        "count": len(snapshot.get("moves", {})),
        "updated_at": str(snapshot.get("updated_at", "")),
        "path": str(snapshot.get("path", "")),
    }


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


def _semantic_xy_from_embedding(vector: list[float]) -> tuple[float, float] | None:
    normalized = _normalize_embedding_vector(vector)
    if normalized is None:
        return None

    x = 0.0
    y = 0.0
    for idx, value in enumerate(normalized):
        angle = (((idx * 2654435761) % 360) / 360.0) * math.tau
        x += value * math.cos(angle)
        y += value * math.sin(angle)

    magnitude = math.sqrt((x * x) + (y * y))
    if magnitude > 0.0:
        x /= magnitude
        y /= magnitude

    return (
        round(_clamp01(0.5 + (x * 0.28)), 4),
        round(_clamp01(0.5 + (y * 0.28)), 4),
    )


def _normalize_field_scores(
    scores: dict[str, float], *, fallback_field: str = "f3"
) -> dict[str, float]:
    normalized = {
        field_id: max(0.0, _safe_float(scores.get(field_id, 0.0), 0.0))
        for field_id in FIELD_TO_PRESENCE
    }
    total = sum(normalized.values())
    if total <= 0.0:
        fallback = fallback_field if fallback_field in FIELD_TO_PRESENCE else "f3"
        normalized[fallback] = 1.0
        total = 1.0
    return {field_id: round(value / total, 4) for field_id, value in normalized.items()}


def _field_scores_from_position(
    x: float,
    y: float,
    field_anchors: dict[str, tuple[float, float]],
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for field_id in FIELD_TO_PRESENCE:
        anchor_x, anchor_y = field_anchors.get(field_id, (0.5, 0.5))
        dx = x - anchor_x
        dy = y - anchor_y
        distance = math.sqrt((dx * dx) + (dy * dy))
        raw[field_id] = 1.0 / (0.04 + (distance * distance * 6.0))
    return _normalize_field_scores(raw)


def _blend_field_scores(
    base_scores: dict[str, float],
    positional_scores: dict[str, float],
    *,
    position_weight: float,
    fallback_field: str,
) -> dict[str, float]:
    blend = _clamp01(_safe_float(position_weight, 0.0))
    if blend <= 0.0:
        return _normalize_field_scores(base_scores, fallback_field=fallback_field)
    if blend >= 1.0:
        return _normalize_field_scores(positional_scores, fallback_field=fallback_field)

    mixed: dict[str, float] = {}
    for field_id in FIELD_TO_PRESENCE:
        left = _safe_float(base_scores.get(field_id, 0.0), 0.0)
        right = _safe_float(positional_scores.get(field_id, 0.0), 0.0)
        mixed[field_id] = (left * (1.0 - blend)) + (right * blend)
    return _normalize_field_scores(mixed, fallback_field=fallback_field)


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


def _organizer_terms_from_entry(entry: dict[str, Any]) -> list[str]:
    if not isinstance(entry, dict):
        return []

    candidates = [
        str(entry.get("name", "")),
        str(entry.get("source_rel_path", "")),
        str(entry.get("archive_rel_path", "")),
        str(entry.get("text_excerpt", "")),
    ]
    terms: list[str] = []
    for candidate in candidates:
        for token in re.findall(r"[A-Za-z0-9_-]+", candidate.lower()):
            normalized = token.strip("_-")
            if len(normalized) < 3:
                continue
            if normalized.isdigit():
                continue
            if normalized in FILE_ORGANIZER_STOPWORDS:
                continue
            terms.append(normalized)

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped[:48]


def _concept_presence_id(cluster_id: str) -> str:
    seed = str(cluster_id or "").strip() or "concept"
    return f"presence:concept:{sha1(seed.encode('utf-8')).hexdigest()[:12]}"


def _concept_presence_label(terms: list[str], index: int) -> str:
    compact = [term for term in terms if term]
    if compact:
        head = " / ".join(compact[:2])
        return f"Concept: {head}"
    return f"Concept Group {index + 1}"


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


def _part_label(part_root: Path, manifest: dict[str, Any]) -> str:
    if "part" in manifest:
        return str(manifest["part"])
    if "name" in manifest:
        return str(manifest["name"])
    return part_root.name


def discover_part_roots(vault_root: Path, part_root: Path) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    candidates = [part_root, *sorted(vault_root.glob("ημ_op_mf_part_*"))]
    for candidate in candidates:
        if not candidate.is_dir():
            continue

        direct_manifest = candidate / "manifest.json"
        nested_manifest = candidate / candidate.name / "manifest.json"
        resolved: Path | None = None
        if direct_manifest.exists():
            resolved = candidate.resolve()
        elif nested_manifest.exists():
            resolved = (candidate / candidate.name).resolve()

        if resolved is not None and resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)

    return roots


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


def _eta_mu_substrate_root(vault_root: Path) -> Path:
    primary = vault_root.resolve()
    primary_roots: list[Path] = [primary, *primary.parents]
    cwd = Path.cwd().resolve()
    fallback_roots: list[Path] = [cwd, *cwd.parents]

    seen: set[Path] = set()

    def _dedupe(roots: list[Path]) -> list[Path]:
        ordered: list[Path] = []
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            ordered.append(root)
        return ordered

    primary_lineage = _dedupe(primary_roots)
    fallback_lineage = _dedupe(fallback_roots)

    for root in primary_lineage:
        inbox = root / ETA_MU_INBOX_DIRNAME
        if inbox.exists() and inbox.is_dir():
            return root
    for root in primary_lineage:
        opencode_dir = root / ".opencode"
        if opencode_dir.exists() and opencode_dir.is_dir():
            return root

    for root in fallback_lineage:
        inbox = root / ETA_MU_INBOX_DIRNAME
        if inbox.exists() and inbox.is_dir():
            return root
    for root in fallback_lineage:
        opencode_dir = root / ".opencode"
        if opencode_dir.exists() and opencode_dir.is_dir():
            return root

    return primary


def _eta_mu_inbox_root(vault_root: Path) -> Path:
    base = _eta_mu_substrate_root(vault_root)
    return (base / ETA_MU_INBOX_DIRNAME).resolve()


def _eta_mu_knowledge_archive_root(vault_root: Path) -> Path:
    base = _eta_mu_substrate_root(vault_root)
    return (base / ETA_MU_KNOWLEDGE_ARCHIVE_REL).resolve()


def _eta_mu_knowledge_index_path(vault_root: Path) -> Path:
    base = _eta_mu_substrate_root(vault_root)
    return (base / ETA_MU_KNOWLEDGE_INDEX_REL).resolve()


def _eta_mu_registry_path(vault_root: Path) -> Path:
    base = _eta_mu_substrate_root(vault_root)
    return (base / ETA_MU_REGISTRY_REL).resolve()


def _safe_rel_path(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.name
    return str(rel).replace("\\", "/")


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


def _read_text_excerpt(path: Path, max_chars: int = 1600) -> str:
    if not _looks_like_text_file(path):
        return ""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    sample = raw[:65536]
    text = sample.decode("utf-8", errors="replace")
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


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


def _file_content_sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(131072), b""):
                h.update(chunk)
    except OSError:
        h.update(str(path).encode("utf-8"))
        h.update(str(time.time_ns()).encode("utf-8"))
    return h.hexdigest()


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


def _eta_mu_output_root(vault_root: Path) -> Path:
    base = _eta_mu_substrate_root(vault_root)
    return (base / ".Π").resolve()


def _eta_mu_iso_compact(ts: datetime | None = None) -> str:
    stamp = ts or datetime.now(timezone.utc)
    return stamp.strftime("%Y%m%dT%H%M%SZ")


def _eta_mu_json_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _split_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _sanitize_collection_name_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip().lower())
    cleaned = cleaned.strip("._-")
    return cleaned or "layer"


def _eta_mu_vecstore_collection_for_space(space: dict[str, Any]) -> str:
    mode = ETA_MU_INGEST_VECSTORE_LAYER_MODE
    base = ETA_MU_INGEST_VECSTORE_COLLECTION
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


def _eta_mu_embed_layer_identity(
    *,
    collection: str,
    space_id: str,
    space_signature: str,
    model_name: str,
) -> dict[str, str]:
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


def _eta_mu_embed_layer_matches(pattern: str, layer: dict[str, Any]) -> bool:
    candidate_values = [
        str(layer.get("id", "")),
        str(layer.get("key", "")),
        str(layer.get("collection", "")),
        str(layer.get("space_id", "")),
        str(layer.get("space_signature", "")),
        str(layer.get("model_name", "")),
        str(layer.get("label", "")),
    ]
    clean_pattern = str(pattern or "").strip()
    if not clean_pattern:
        return False
    if clean_pattern == "*":
        return True
    for value in candidate_values:
        if not value:
            continue
        if fnmatch.fnmatch(value, clean_pattern):
            return True
    return False


def _eta_mu_embed_layer_is_active(
    layer: dict[str, Any],
    active_patterns: list[str],
) -> bool:
    if not active_patterns or active_patterns == ["*"]:
        return True
    return any(
        _eta_mu_embed_layer_matches(pattern, layer) for pattern in active_patterns
    )


def _eta_mu_embed_layer_order_index(
    layer: dict[str, Any],
    order_patterns: list[str],
) -> int:
    if not order_patterns:
        return 0
    for index, pattern in enumerate(order_patterns):
        if _eta_mu_embed_layer_matches(pattern, layer):
            return index
    return len(order_patterns) + 1


def _eta_mu_embed_layer_point(
    *,
    node_x: float,
    node_y: float,
    layer_seed: str,
    index: int,
    semantic_xy: tuple[float, float] | None,
) -> tuple[float, float, str]:
    if semantic_xy is None:
        target_x = _stable_ratio(layer_seed, index * 13 + 3)
        target_y = _stable_ratio(layer_seed, index * 13 + 7)
        source = "deterministic"
    else:
        target_x, target_y = semantic_xy
        source = "semantic"

    orbit = 0.008 + (index % 4) * 0.004
    angle = _stable_ratio(layer_seed, index * 17 + 11) * math.tau
    target_x = _clamp01(target_x + math.cos(angle) * orbit)
    target_y = _clamp01(target_y + math.sin(angle) * orbit)

    blend = _clamp01(ETA_MU_FILE_GRAPH_LAYER_BLEND)
    layered_x = _clamp01((node_x * (1.0 - blend)) + (target_x * blend))
    layered_y = _clamp01((node_y * (1.0 - blend)) + (target_y * blend))
    return round(layered_x, 4), round(layered_y, 4), source


def _eta_mu_space_forms() -> dict[str, Any]:
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
            "text": str(
                text_space.get("collection", ETA_MU_INGEST_VECSTORE_COLLECTION)
            ),
            "image": str(
                image_space.get("collection", ETA_MU_INGEST_VECSTORE_COLLECTION)
            ),
        },
        "time": "none",
    }
    space_set["signature"] = _eta_mu_json_sha256(space_set)

    vecstore = {
        "id": ETA_MU_INGEST_VECSTORE_ID,
        "backend": "chroma",
        "collection": ETA_MU_INGEST_VECSTORE_COLLECTION,
        "layer_mode": ETA_MU_INGEST_VECSTORE_LAYER_MODE,
        "collections": {
            "text": str(
                text_space.get("collection", ETA_MU_INGEST_VECSTORE_COLLECTION)
            ),
            "image": str(
                image_space.get("collection", ETA_MU_INGEST_VECSTORE_COLLECTION)
            ),
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


def _eta_mu_inbox_rel_path(path: Path, inbox_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(inbox_root.resolve())
    except ValueError:
        return path.name
    return str(rel).replace("\\", "/")


def _eta_mu_is_excluded_inbox_rel(rel_path: str) -> bool:
    normalized = str(rel_path).strip().replace("\\", "/")
    if not normalized:
        return True

    parts = [token for token in normalized.split("/") if token]
    if any(token in ETA_MU_INGEST_EXCLUDE_REL_PATHS for token in parts):
        return True

    if any(
        fnmatch.fnmatch(normalized, pattern) for pattern in ETA_MU_INGEST_EXCLUDE_GLOBS
    ):
        return True
    return False


def _eta_mu_scan_candidates(inbox_root: Path) -> list[Path]:
    if not inbox_root.exists() or not inbox_root.is_dir():
        return []

    candidates: list[Path] = []
    for path in sorted(inbox_root.rglob("*")):
        if not path.is_file():
            continue
        rel = _eta_mu_inbox_rel_path(path, inbox_root)
        if _eta_mu_is_excluded_inbox_rel(rel):
            continue
        depth = max(0, len([token for token in rel.split("/") if token]) - 1)
        if depth > ETA_MU_INGEST_MAX_SCAN_DEPTH:
            continue
        candidates.append(path)
        if len(candidates) >= ETA_MU_INGEST_MAX_SCAN_FILES:
            break
    return candidates


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


def _eta_mu_detect_modality(
    *,
    path: Path,
    mime: str,
) -> tuple[str | None, str]:
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


def _eta_mu_canonicalize_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines = [
        unicodedata.normalize("NFC", line.rstrip(" \t\v\f"))
        for line in text.split("\n")
    ]
    return "\n".join(normalized_lines)


def _eta_mu_text_language_hint(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix:
        return suffix
    return "text"


def _eta_mu_chunk_policy() -> dict[str, int]:
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


def _eta_mu_image_derive_segment(
    *,
    source_hash: str,
    source_bytes: int,
    source_rel_path: str,
    mime: str,
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
    return {
        "id": "img-0000",
        "start": 0,
        "end": 0,
        "unit": "image",
        "text": descriptor,
        "derive_spec": derive_spec,
        "derive_hash": derive_hash,
    }


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
        remote_vector = _ollama_embed(segment_text, model=model)

    fallback_used = False
    if remote_vector is None:
        fallback_used = True
        remote_vector = _eta_mu_deterministic_vector(
            embed_id + "|" + segment_text, dims
        )

    vector = _normalize_embedding_vector(remote_vector)
    if vector is None:
        fallback_used = True
        vector = _eta_mu_deterministic_vector(embed_id + "|" + segment_text, dims)
    vector = _eta_mu_resize_vector(vector, dims)
    vector = _eta_mu_normalize_vector(vector)
    return vector, model or "", fallback_used


def _eta_mu_emit_packet(
    packets: list[dict[str, Any]],
    *,
    kind: str,
    body: dict[str, Any],
    links: dict[str, Any] | None = None,
) -> dict[str, Any]:
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


def _eta_mu_sexp_atom(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "nil"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return "0.0"
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        inner = " ".join(_eta_mu_sexp_atom(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, dict):
        parts: list[str] = []
        for key in sorted(value.keys()):
            raw_key = str(key)
            if re.fullmatch(r"[A-Za-z0-9_.+\-/]+", raw_key):
                key_form = f":{raw_key}"
            else:
                key_form = json.dumps(raw_key, ensure_ascii=False)
            parts.append(f"({key_form} {_eta_mu_sexp_atom(value[key])})")
        return f"(map {' '.join(parts)})"
    return json.dumps(str(value), ensure_ascii=False)


def _eta_mu_write_sexp_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "(artifact " + _eta_mu_sexp_atom(payload) + ")\n"
    path.write_text(content, encoding="utf-8")


def _eta_mu_percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    bounded = sorted(float(item) for item in values)
    ratio = _clamp01(p)
    if len(bounded) == 1:
        return bounded[0]
    index = ratio * (len(bounded) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return bounded[lower]
    weight = index - lower
    return bounded[lower] * (1.0 - weight) + bounded[upper] * weight


def _eta_mu_rejected_target_path(
    *,
    inbox_root: Path,
    source_path: Path,
    source_hash: str,
) -> Path:
    rel = _eta_mu_inbox_rel_path(source_path, inbox_root)
    parts = [token for token in rel.split("/") if token]
    if not parts:
        parts = [source_path.name]
    leaf = _sanitize_archive_name(parts[-1])
    prefix = source_hash[:12] if source_hash else "unknown"
    stamped = f"{prefix}_{leaf}" if leaf else prefix
    return (inbox_root / "_rejected" / stamped).resolve()


def _get_eta_mu_chroma_collection(collection_name: str | None = None) -> Any:
    global _CHROMA_CLIENT
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


def _eta_mu_vecstore_upsert_batch(
    vault_root: Path,
    rows: list[dict[str, Any]],
    *,
    collection_name: str,
    space_set_signature: str,
) -> dict[str, Any]:
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

    collection = _get_eta_mu_chroma_collection(chosen_collection)
    if collection is not None:
        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            return {
                "ok": True,
                "backend": "chroma",
                "collection": chosen_collection,
                "upserted": len(ids),
                "ids": ids,
                "drift": False,
            }
        except Exception as exc:
            error_text = str(exc)
    else:
        error_text = "chroma_unavailable"

    fallback_ids: list[str] = []
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
            fallback_ids.append(str(row.get("id", "")))

    return {
        "ok": len(fallback_ids) == len(rows),
        "backend": "eta_mu_embeddings_db",
        "collection": chosen_collection,
        "upserted": len(fallback_ids),
        "ids": fallback_ids,
        "drift": False,
        "error": error_text,
    }


def _field_schema_name_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in PROJECTION_FIELD_SCHEMAS:
        key = str(row.get("field", "")).strip()
        if key:
            mapping[key] = str(row.get("name", key))
    return mapping


def _infer_eta_mu_field_scores(
    *,
    rel_path: str,
    kind: str,
    text_excerpt: str,
) -> dict[str, float]:
    scores = {field_id: 0.0 for field_id in FIELD_TO_PRESENCE}

    kind_key = kind.strip().lower()
    if kind_key in {"audio", "image", "video"}:
        scores["f1"] += 0.42
        scores["f6"] += 0.08
    elif kind_key == "text":
        scores["f6"] += 0.46
        scores["f3"] += 0.2
        scores["f7"] += 0.12
    else:
        scores["f4"] += 0.24
        scores["f8"] += 0.18

    rel_lower = rel_path.lower()
    if rel_lower.endswith(".zip"):
        scores["f5"] += 0.24
        scores["f4"] += 0.12
    if rel_lower.endswith(".lisp"):
        scores["f6"] += 0.24
        scores["f7"] += 0.1
    if rel_lower.endswith(".md") or rel_lower.endswith(".txt"):
        scores["f6"] += 0.18
        scores["f3"] += 0.1

    combined = f"{rel_path} {text_excerpt}"
    tokens = _clean_tokens(combined)
    for token in tokens:
        for field_id, keywords in ETA_MU_FIELD_KEYWORDS.items():
            if token in keywords:
                scores[field_id] += 0.07

    for field_id, keywords in ETA_MU_FIELD_KEYWORDS.items():
        if any(keyword in rel_lower for keyword in keywords):
            scores[field_id] += 0.12

    total = sum(max(0.0, value) for value in scores.values())
    if total <= 0:
        fallback = "f6" if kind_key == "text" else "f1"
        scores[fallback] = 1.0
        return scores

    normalized: dict[str, float] = {}
    for field_id, value in scores.items():
        normalized[field_id] = round(max(0.0, value) / total, 4)

    if all(value <= 0.0 for value in normalized.values()):
        fallback = "f6" if kind_key == "text" else "f1"
        normalized[fallback] = 1.0
    return normalized


def _dominant_eta_mu_field(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "f6", 1.0
    dominant_field = max(
        scores.keys(), key=lambda key: _safe_float(scores.get(key, 0.0), 0.0)
    )
    return dominant_field, _safe_float(scores.get(dominant_field, 0.0), 0.0)


def _sanitize_archive_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "artifact"


def _archive_member_name(name: str) -> str:
    return f"payload/{_sanitize_archive_name(name)}"


def _archive_container_id(archive_rel_path: str) -> str:
    digest = sha1(archive_rel_path.encode("utf-8")).hexdigest()[:14]
    return f"archive:{digest}"


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
    if not parts:
        return None
    return "/".join(parts)


def resolve_library_member(request_path: str) -> str | None:
    parsed = urlparse(request_path)
    params = parse_qs(parsed.query)
    raw = str((params.get("member") or [""])[0] or "")
    return _normalize_archive_member_path(raw)


def _read_library_archive_member(
    archive_path: Path, member_path: str
) -> tuple[bytes, str] | None:
    if archive_path.suffix.lower() != ".zip":
        return None
    normalized_member = _normalize_archive_member_path(member_path)
    if normalized_member is None:
        return None
    try:
        with zipfile.ZipFile(archive_path, "r") as archive_zip:
            with archive_zip.open(normalized_member, "r") as handle:
                payload = handle.read()
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None

    content_type = (
        mimetypes.guess_type(normalized_member)[0] or "application/octet-stream"
    )
    if content_type.startswith("text/"):
        content_type = f"{content_type}; charset=utf-8"
    elif content_type == "application/json":
        content_type = "application/json; charset=utf-8"
    return payload, content_type


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


def _append_eta_mu_knowledge_record(vault_root: Path, record: dict[str, Any]) -> None:
    index_path = _eta_mu_knowledge_index_path(vault_root)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with _ETA_MU_KNOWLEDGE_LOCK:
        _ETA_MU_KNOWLEDGE_CACHE["path"] = ""
        _ETA_MU_KNOWLEDGE_CACHE["mtime_ns"] = 0
        _ETA_MU_KNOWLEDGE_CACHE["entries"] = []


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


def _append_eta_mu_registry_record(vault_root: Path, record: dict[str, Any]) -> None:
    registry_path = _eta_mu_registry_path(vault_root)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    with _ETA_MU_REGISTRY_LOCK:
        _ETA_MU_REGISTRY_CACHE["path"] = ""
        _ETA_MU_REGISTRY_CACHE["mtime_ns"] = 0
        _ETA_MU_REGISTRY_CACHE["entries"] = []


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


def _cleanup_empty_inbox_dirs(inbox_root: Path) -> None:
    for path in sorted(inbox_root.rglob("*"), reverse=True):
        if not path.is_dir():
            continue
        try:
            path.rmdir()
        except OSError:
            continue


def sync_eta_mu_inbox(vault_root: Path) -> dict[str, Any]:
    base_root = _eta_mu_substrate_root(vault_root)
    inbox_root = _eta_mu_inbox_root(vault_root)
    now_monotonic = time.monotonic()
    spaces = _eta_mu_space_forms()

    with _ETA_MU_INBOX_LOCK:
        cached_root = str(_ETA_MU_INBOX_CACHE.get("root", ""))
        cached_snapshot = _ETA_MU_INBOX_CACHE.get("snapshot")
        elapsed = now_monotonic - float(
            _ETA_MU_INBOX_CACHE.get("last_checked_monotonic", 0.0)
        )
        if (
            cached_snapshot is not None
            and cached_root == str(inbox_root)
            and elapsed < ETA_MU_INBOX_DEBOUNCE_SECONDS
        ):
            return dict(cached_snapshot)

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
                    "collection": spaces["vecstore"]["collection"],
                    "layer_mode": spaces["vecstore"].get("layer_mode", "single"),
                },
            },
            "errors": [],
        }
        with _ETA_MU_INBOX_LOCK:
            _ETA_MU_INBOX_CACHE["root"] = str(inbox_root)
            _ETA_MU_INBOX_CACHE["snapshot"] = dict(snapshot)
            _ETA_MU_INBOX_CACHE["last_checked_monotonic"] = now_monotonic
        return snapshot

    archive_root = _eta_mu_knowledge_archive_root(vault_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    field_name_map = _field_schema_name_map()

    pending_files = _eta_mu_scan_candidates(inbox_root)
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    rejected_count = 0
    deferred_count = 0
    errors: list[dict[str, str]] = []
    last_ingested_at = ""
    modality_counts: dict[str, int] = {"text": 0, "image": 0}
    embed_call_ms: list[float] = []
    vecstore_call_ms: list[float] = []
    packets: list[dict[str, Any]] = []
    manifest_sources: list[dict[str, Any]] = []

    registry_entries = _load_eta_mu_registry_entries(vault_root)
    registry_known = _eta_mu_registry_known_entries(registry_entries)
    registry_known_idempotence = _eta_mu_collect_registry_idempotence(registry_entries)

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
        rel_source = _safe_rel_path(file_path, base_root)
        inbox_rel = _eta_mu_inbox_rel_path(file_path, inbox_root)

        try:
            stat = file_path.stat()
            source_bytes = int(stat.st_size)
            source_mtime_utc = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()
            raw = file_path.read_bytes()
        except OSError as exc:
            failed_count += 1
            errors.append({"path": rel_source, "error": str(exc)})
            continue

        source_hash = hashlib.sha256(raw).hexdigest()
        mime = _eta_mu_guess_mime(file_path, raw)
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
                reject_target = _eta_mu_rejected_target_path(
                    inbox_root=inbox_root,
                    source_path=file_path,
                    source_hash=source_hash,
                )
                reject_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, reject_target)
                file_path.unlink()
            except OSError as exc:
                failed_count += 1
                errors.append({"path": rel_source, "error": str(exc)})
                continue

            reject_registry = {
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
                "idempotence_key": "",
                "embed_id": "",
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
            }
            try:
                _append_eta_mu_registry_record(vault_root, reject_registry)
            except OSError as exc:
                failed_count += 1
                errors.append({"path": rel_source, "error": str(exc)})
            continue

        if modality == "text" and source_bytes > ETA_MU_INGEST_MAX_TEXT_BYTES:
            rejected_count += 1
            rejected_at = datetime.now(timezone.utc).isoformat()
            reject_packet = _eta_mu_emit_packet(
                packets,
                kind="ingest-reject",
                body={
                    "source_rel_path": rel_source,
                    "inbox_rel_path": inbox_rel,
                    "reason": "max-bytes:text",
                    "bytes": source_bytes,
                    "limit": ETA_MU_INGEST_MAX_TEXT_BYTES,
                },
            )
            try:
                reject_target = _eta_mu_rejected_target_path(
                    inbox_root=inbox_root,
                    source_path=file_path,
                    source_hash=source_hash,
                )
                reject_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, reject_target)
                file_path.unlink()
            except OSError as exc:
                failed_count += 1
                errors.append({"path": rel_source, "error": str(exc)})
                continue

            reject_registry = {
                "record": ETA_MU_INGEST_REGISTRY_RECORD,
                "event": "rejected",
                "status": "reject",
                "reason": "max-bytes:text",
                "registry_key": _eta_mu_registry_reference_key(
                    source_hash=source_hash,
                    rel_source=rel_source,
                    source_bytes=source_bytes,
                    modality=modality,
                ),
                "idempotence_key": "",
                "embed_id": "",
                "content_sha256": source_hash,
                "source_hash": source_hash,
                "source_rel_path": rel_source,
                "source_name": file_path.name,
                "source_kind": modality,
                "source_modality": modality,
                "source_mime": mime,
                "source_bytes": source_bytes,
                "source_mtime_utc": source_mtime_utc,
                "packet_refs": [reject_packet["id"]],
                "time": rejected_at,
            }
            try:
                _append_eta_mu_registry_record(vault_root, reject_registry)
            except OSError as exc:
                failed_count += 1
                errors.append({"path": rel_source, "error": str(exc)})
            continue

        if modality == "image" and source_bytes > ETA_MU_INGEST_MAX_IMAGE_BYTES:
            rejected_count += 1
            rejected_at = datetime.now(timezone.utc).isoformat()
            reject_packet = _eta_mu_emit_packet(
                packets,
                kind="ingest-reject",
                body={
                    "source_rel_path": rel_source,
                    "inbox_rel_path": inbox_rel,
                    "reason": "max-bytes:image",
                    "bytes": source_bytes,
                    "limit": ETA_MU_INGEST_MAX_IMAGE_BYTES,
                },
            )
            try:
                reject_target = _eta_mu_rejected_target_path(
                    inbox_root=inbox_root,
                    source_path=file_path,
                    source_hash=source_hash,
                )
                reject_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, reject_target)
                file_path.unlink()
            except OSError as exc:
                failed_count += 1
                errors.append({"path": rel_source, "error": str(exc)})
                continue

            reject_registry = {
                "record": ETA_MU_INGEST_REGISTRY_RECORD,
                "event": "rejected",
                "status": "reject",
                "reason": "max-bytes:image",
                "registry_key": _eta_mu_registry_reference_key(
                    source_hash=source_hash,
                    rel_source=rel_source,
                    source_bytes=source_bytes,
                    modality=modality,
                ),
                "idempotence_key": "",
                "embed_id": "",
                "content_sha256": source_hash,
                "source_hash": source_hash,
                "source_rel_path": rel_source,
                "source_name": file_path.name,
                "source_kind": modality,
                "source_modality": modality,
                "source_mime": mime,
                "source_bytes": source_bytes,
                "source_mtime_utc": source_mtime_utc,
                "packet_refs": [reject_packet["id"]],
                "time": rejected_at,
            }
            try:
                _append_eta_mu_registry_record(vault_root, reject_registry)
            except OSError as exc:
                failed_count += 1
                errors.append({"path": rel_source, "error": str(exc)})
            continue

        modality_counts[modality] = modality_counts.get(modality, 0) + 1
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
            canonical_text = _eta_mu_canonicalize_text(raw)
            segments = _eta_mu_text_segments(
                canonical_text,
                language_hint=_eta_mu_text_language_hint(file_path),
            )
            excerpt_dense = " ".join(canonical_text.split())
            excerpt = (
                excerpt_dense[:1599] + "…"
                if len(excerpt_dense) > 1600
                else excerpt_dense
            )
            space = spaces["text"]
        else:
            canonical_text = ""
            segments = [
                _eta_mu_image_derive_segment(
                    source_hash=source_hash,
                    source_bytes=source_bytes,
                    source_rel_path=rel_source,
                    mime=mime,
                )
            ]
            excerpt = ""
            space = spaces["image"]

        segment_plans: list[dict[str, Any]] = []
        for segment in segments:
            embed_id = _eta_mu_embed_id(
                space_signature=str(space.get("signature", "")),
                source_hash=source_hash,
                segment=segment,
            )
            idempotence_key = _eta_mu_registry_idempotence_key(
                embed_id=embed_id,
                source_hash=source_hash,
                space_signature=str(space.get("signature", "")),
                segment=segment,
            )
            segment_plans.append(
                {
                    "segment": dict(segment),
                    "embed_id": embed_id,
                    "idempotence_key": idempotence_key,
                }
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
                "segments": [
                    {
                        "id": str(plan["segment"].get("id", "")),
                        "start": int(_safe_float(plan["segment"].get("start", 0), 0.0)),
                        "end": int(_safe_float(plan["segment"].get("end", 0), 0.0)),
                        "unit": str(plan["segment"].get("unit", "")),
                        "embed_id": str(plan.get("embed_id", "")),
                    }
                    for plan in segment_plans
                ],
            },
        )

        registry_key = _eta_mu_registry_reference_key(
            source_hash=source_hash,
            rel_source=rel_source,
            source_bytes=source_bytes,
            modality=modality,
        )

        if segment_plans and all(
            str(plan.get("idempotence_key", "")) in registry_known_idempotence
            for plan in segment_plans
        ):
            skipped_count += 1
            skipped_at = datetime.now(timezone.utc).isoformat()
            duplicate_of = registry_known.get(registry_key, {})
            skip_packet = _eta_mu_emit_packet(
                packets,
                kind="note",
                body={
                    "op": "skip",
                    "reason": "duplicate_unchanged",
                    "source_rel_path": rel_source,
                    "segment_count": len(segment_plans),
                },
            )
            try:
                file_path.unlink()
            except OSError as exc:
                failed_count += 1
                errors.append({"path": rel_source, "error": str(exc)})
                continue

            for plan in segment_plans:
                skip_record = {
                    "record": ETA_MU_INGEST_REGISTRY_RECORD,
                    "event": "skipped",
                    "status": "skip",
                    "reason": "duplicate_unchanged",
                    "registry_key": registry_key,
                    "idempotence_key": str(plan.get("idempotence_key", "")),
                    "embed_id": str(plan.get("embed_id", "")),
                    "space_id": str(space.get("id", "")),
                    "space_signature": str(space.get("signature", "")),
                    "segment": {
                        "id": str(plan["segment"].get("id", "")),
                        "start": int(_safe_float(plan["segment"].get("start", 0), 0.0)),
                        "end": int(_safe_float(plan["segment"].get("end", 0), 0.0)),
                        "unit": str(plan["segment"].get("unit", "")),
                    },
                    "content_sha256": source_hash,
                    "source_hash": source_hash,
                    "source_rel_path": rel_source,
                    "source_name": file_path.name,
                    "source_kind": modality,
                    "source_modality": modality,
                    "source_mime": mime,
                    "source_bytes": source_bytes,
                    "source_mtime_utc": source_mtime_utc,
                    "duplicate_of": str(duplicate_of.get("knowledge_id", "")),
                    "duplicate_archive_rel_path": str(
                        duplicate_of.get(
                            "archive_rel_path",
                            duplicate_of.get("archived_rel_path", ""),
                        )
                    ),
                    "packet_refs": [
                        found_packet["id"],
                        chunk_packet["id"],
                        skip_packet["id"],
                    ],
                    "time": skipped_at,
                }
                try:
                    _append_eta_mu_registry_record(vault_root, skip_record)
                except OSError as exc:
                    failed_count += 1
                    errors.append({"path": rel_source, "error": str(exc)})
                    continue
                registry_known[registry_key] = dict(skip_record)
                registry_known_idempotence[str(plan.get("idempotence_key", ""))] = dict(
                    skip_record
                )
            continue

        missing_plans = [
            plan
            for plan in segment_plans
            if str(plan.get("idempotence_key", "")) not in registry_known_idempotence
        ]

        vec_collection = (
            str(
                space.get(
                    "collection",
                    spaces["vecstore"].get(
                        "collection", ETA_MU_INGEST_VECSTORE_COLLECTION
                    ),
                )
            ).strip()
            or ETA_MU_INGEST_VECSTORE_COLLECTION
        )

        packet_refs_for_file = [found_packet["id"], chunk_packet["id"]]
        vec_rows: list[dict[str, Any]] = []
        embed_refs: list[dict[str, Any]] = []

        for plan in missing_plans:
            segment = dict(plan.get("segment", {}))
            segment_text = str(segment.get("text", ""))
            segment_start = int(_safe_float(segment.get("start", 0), 0.0))
            segment_end = int(_safe_float(segment.get("end", 0), 0.0))
            segment_unit = str(segment.get("unit", ""))

            call_plan_packet = _eta_mu_emit_packet(
                packets,
                kind="call-plan",
                body={
                    "resource": "rc.ollama.embed",
                    "op": "embed",
                    "source_rel_path": rel_source,
                    "embed_id": str(plan.get("embed_id", "")),
                    "estimated_ms": round(
                        10.0 + (0.8 * max(1.0, len(segment_text) / 4.0)), 3
                    ),
                    "permit": "ok",
                    "safe_mode": ETA_MU_INGEST_SAFE_MODE,
                },
            )
            packet_refs_for_file.append(call_plan_packet["id"])

            embed_started = time.perf_counter()
            vector, model_name, fallback_used = _eta_mu_embed_vector_for_segment(
                modality=modality,
                segment_text=segment_text,
                space=space,
                embed_id=str(plan.get("embed_id", "")),
            )
            embed_elapsed_ms = (time.perf_counter() - embed_started) * 1000.0
            embed_call_ms.append(embed_elapsed_ms)

            settings_hash = _eta_mu_json_sha256(
                {
                    "space_signature": str(space.get("signature", "")),
                    "safe_mode": ETA_MU_INGEST_SAFE_MODE,
                    "modality": modality,
                    "chunk_policy": _eta_mu_chunk_policy()
                    if modality == "text"
                    else {"kind": "image-preproc"},
                }
            )
            model_key = _eta_mu_model_key(
                model_name=model_name,
                model_digest=str(
                    (space.get("model") or {}).get("digest", "none") or "none"
                ),
                settings_hash=settings_hash,
            )
            layer_identity = _eta_mu_embed_layer_identity(
                collection=vec_collection,
                space_id=str(space.get("id", "")),
                space_signature=str(space.get("signature", "")),
                model_name=str(model_name or ""),
            )

            call_act_packet = _eta_mu_emit_packet(
                packets,
                kind="call-act",
                body={
                    "resource": "rc.ollama.embed",
                    "op": "embed",
                    "source_rel_path": rel_source,
                    "embed_id": str(plan.get("embed_id", "")),
                    "status": "ok",
                    "ms": round(embed_elapsed_ms, 3),
                    "dims": len(vector),
                    "fallback_used": fallback_used,
                    "model_key": model_key,
                },
            )
            packet_refs_for_file.append(call_act_packet["id"])

            embed_ref_packet = _eta_mu_emit_packet(
                packets,
                kind="embed-ref",
                body={
                    "embed_id": str(plan.get("embed_id", "")),
                    "source_rel_path": rel_source,
                    "source_hash": source_hash,
                    "space_id": str(space.get("id", "")),
                    "space_signature": str(space.get("signature", "")),
                    "vecstore_collection": vec_collection,
                    "layer_id": str(layer_identity.get("id", "")),
                    "layer_key": str(layer_identity.get("key", "")),
                    "segment": {
                        "id": str(segment.get("id", "")),
                        "start": segment_start,
                        "end": segment_end,
                        "unit": segment_unit,
                    },
                    "model_key": model_key,
                    "dims": len(vector),
                },
            )
            packet_refs_for_file.append(embed_ref_packet["id"])

            vec_rows.append(
                {
                    "id": str(plan.get("embed_id", "")),
                    "embedding": vector,
                    "metadata": {
                        "source.loc": rel_source,
                        "source.hash": source_hash,
                        "mime": mime,
                        "segment.id": str(segment.get("id", "")),
                        "segment.start": segment_start,
                        "segment.end": segment_end,
                        "segment.unit": segment_unit,
                        "space.id": str(space.get("id", "")),
                        "space.signature": str(space.get("signature", "")),
                        "vecstore.collection": vec_collection,
                        "model_key": model_key,
                        "time": datetime.now(timezone.utc).isoformat(),
                    },
                    "document": segment_text,
                    "model": str(model_name or ""),
                }
            )
            embed_refs.append(
                {
                    "embed_id": str(plan.get("embed_id", "")),
                    "idempotence_key": str(plan.get("idempotence_key", "")),
                    "space_id": str(space.get("id", "")),
                    "space_signature": str(space.get("signature", "")),
                    "vecstore_collection": vec_collection,
                    "segment": {
                        "id": str(segment.get("id", "")),
                        "start": segment_start,
                        "end": segment_end,
                        "unit": segment_unit,
                    },
                    "model_key": model_key,
                    "model_name": str(model_name or ""),
                    "layer_id": str(layer_identity.get("id", "")),
                    "layer_key": str(layer_identity.get("key", "")),
                    "layer_label": str(layer_identity.get("label", "")),
                    "packet_ref": embed_ref_packet["id"],
                }
            )

        vec_plan_packet = _eta_mu_emit_packet(
            packets,
            kind="call-plan",
            body={
                "resource": "rc.vecstore.main",
                "op": "upsert",
                "source_rel_path": rel_source,
                "item_count": len(vec_rows),
                "collection": vec_collection,
                "permit": "ok",
            },
        )
        packet_refs_for_file.append(vec_plan_packet["id"])

        vecstore_started = time.perf_counter()
        vec_result = _eta_mu_vecstore_upsert_batch(
            vault_root,
            vec_rows,
            collection_name=vec_collection,
            space_set_signature=str(spaces["space_set"].get("signature", "")),
        )
        vecstore_elapsed_ms = (time.perf_counter() - vecstore_started) * 1000.0
        vecstore_call_ms.append(vecstore_elapsed_ms)

        vec_act_packet = _eta_mu_emit_packet(
            packets,
            kind="call-act",
            body={
                "resource": "rc.vecstore.main",
                "op": "upsert",
                "source_rel_path": rel_source,
                "status": "ok" if bool(vec_result.get("ok")) else "error",
                "backend": str(vec_result.get("backend", "")),
                "collection": str(vec_result.get("collection", vec_collection)),
                "upserted": int(_safe_float(vec_result.get("upserted", 0), 0.0)),
                "drift": bool(vec_result.get("drift")),
                "ms": round(vecstore_elapsed_ms, 3),
            },
        )
        packet_refs_for_file.append(vec_act_packet["id"])

        upsert_packet = _eta_mu_emit_packet(
            packets,
            kind="vecstore-upsert",
            body={
                "source_rel_path": rel_source,
                "backend": str(vec_result.get("backend", "")),
                "collection": str(vec_result.get("collection", vec_collection)),
                "ids": [str(item) for item in vec_result.get("ids", [])],
                "upserted": int(_safe_float(vec_result.get("upserted", 0), 0.0)),
                "drift": bool(vec_result.get("drift")),
            },
        )
        packet_refs_for_file.append(upsert_packet["id"])

        deferred_this_file = (
            bool(vec_result.get("drift"))
            and str(vec_result.get("backend", "")).strip() == "deferred"
        )
        if deferred_this_file:
            deferred_count += 1

        scores = _infer_eta_mu_field_scores(
            rel_path=rel_source,
            kind=modality,
            text_excerpt=excerpt,
        )
        dominant_field, dominant_weight = _dominant_eta_mu_field(scores)
        presence_id = FIELD_TO_PRESENCE.get(dominant_field, "anchor_registry")
        digest = _stable_file_digest(file_path)
        ingested_at = datetime.now(timezone.utc).isoformat()
        top_scores = sorted(scores.items(), key=lambda row: row[1], reverse=True)[:3]

        sanitized = _sanitize_archive_name(file_path.name)
        archive_rel = (
            f"{ETA_MU_KNOWLEDGE_ARCHIVE_REL}/"
            f"{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/"
            f"{digest[:12]}_{sanitized}.zip"
        )
        archive_abs = (base_root / archive_rel).resolve()
        archive_abs.parent.mkdir(parents=True, exist_ok=True)
        if archive_abs.exists():
            archive_rel = (
                f"{ETA_MU_KNOWLEDGE_ARCHIVE_REL}/"
                f"{datetime.now(timezone.utc).strftime('%Y/%m/%d')}/"
                f"{digest[:12]}_{int(time.time() * 1000)}_{sanitized}.zip"
            )
            archive_abs = (base_root / archive_rel).resolve()

        try:
            record_id_seed = f"{digest}|{rel_source}|{ingested_at}"
            record_id = (
                f"knowledge.{sha1(record_id_seed.encode('utf-8')).hexdigest()[:14]}"
            )
            archive_member_path = _archive_member_name(file_path.name)
            archive_manifest_path = "manifest.json"
            archive_container_id = _archive_container_id(archive_rel)
            archive_url = "/library/" + quote(archive_rel)
            archive_manifest = {
                "record": ETA_MU_ARCHIVE_MANIFEST_RECORD,
                "record_id": record_id,
                "contract": ETA_MU_INGEST_CONTRACT_ID,
                "fingerprint": digest,
                "ingested_at": ingested_at,
                "source_rel_path": rel_source,
                "source_name": file_path.name,
                "source_kind": modality,
                "source_mime": mime,
                "source_hash": source_hash,
                "source_bytes": int(stat.st_size),
                "source_mtime_utc": source_mtime_utc,
                "archive_rel_path": archive_rel,
                "archive_member_path": archive_member_path,
                "archive_container_id": archive_container_id,
                "space": {
                    "id": str(space.get("id", "")),
                    "signature": str(space.get("signature", "")),
                    "collection": vec_collection,
                },
                "space_set": {
                    "id": str(spaces["space_set"].get("id", "")),
                    "signature": str(spaces["space_set"].get("signature", "")),
                },
                "segments": [
                    {
                        "id": str(plan["segment"].get("id", "")),
                        "start": int(_safe_float(plan["segment"].get("start", 0), 0.0)),
                        "end": int(_safe_float(plan["segment"].get("end", 0), 0.0)),
                        "unit": str(plan["segment"].get("unit", "")),
                        "embed_id": str(plan.get("embed_id", "")),
                        "idempotence_key": str(plan.get("idempotence_key", "")),
                    }
                    for plan in segment_plans
                ],
                "packet_refs": packet_refs_for_file,
                "dominant_field": dominant_field,
                "dominant_presence": presence_id,
                "dominant_weight": round(_clamp01(dominant_weight), 4),
                "field_scores": {
                    key: round(_safe_float(value, 0.0), 4)
                    for key, value in scores.items()
                },
                "top_fields": [
                    {
                        "field": field_id,
                        "name": field_name_map.get(field_id, field_id),
                        "weight": round(_safe_float(weight, 0.0), 4),
                    }
                    for field_id, weight in top_scores
                    if _safe_float(weight, 0.0) > 0.0
                ],
                "text_excerpt": excerpt,
            }

            with zipfile.ZipFile(
                archive_abs,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive_zip:
                archive_zip.write(file_path, arcname=archive_member_path)
                archive_zip.writestr(
                    archive_manifest_path,
                    json.dumps(
                        archive_manifest,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    ),
                )

            file_path.unlink()
            archive_stat = archive_abs.stat()
            record = {
                "record": ETA_MU_KNOWLEDGE_RECORD,
                "id": record_id,
                "fingerprint": digest,
                "ingested_at": ingested_at,
                "name": file_path.name,
                "source_rel_path": rel_source,
                "archived_rel_path": archive_rel,
                "archive_rel_path": archive_rel,
                "archive_kind": "zip",
                "archive_url": archive_url,
                "archive_member_path": archive_member_path,
                "archive_manifest_path": archive_manifest_path,
                "archive_manifest_record": ETA_MU_ARCHIVE_MANIFEST_RECORD,
                "archive_manifest_url": _library_url_for_archive_member(
                    archive_rel,
                    archive_manifest_path,
                ),
                "archive_bytes": int(archive_stat.st_size),
                "archive_container_id": archive_container_id,
                "embedding_links": [
                    {
                        "kind": "stored_in_archive",
                        "target": archive_container_id,
                        "member_path": archive_member_path,
                        "weight": 1.0,
                    }
                ]
                + [
                    {
                        "kind": "embed_ref",
                        "embed_id": str(embed_ref.get("embed_id", "")),
                        "space_id": str(embed_ref.get("space_id", "")),
                        "space_signature": str(embed_ref.get("space_signature", "")),
                        "vecstore_collection": str(
                            embed_ref.get("vecstore_collection", "")
                        ),
                        "model_name": str(embed_ref.get("model_name", "")),
                        "layer_id": str(embed_ref.get("layer_id", "")),
                        "layer_key": str(embed_ref.get("layer_key", "")),
                        "layer_label": str(embed_ref.get("layer_label", "")),
                        "vecstore_key": str(embed_ref.get("embed_id", "")),
                        "packet_ref": str(embed_ref.get("packet_ref", "")),
                        "weight": round(1.0 / max(1, len(embed_refs)), 6),
                    }
                    for embed_ref in embed_refs
                ],
                "url": _library_url_for_archive_member(
                    archive_rel,
                    archive_member_path,
                ),
                "kind": modality,
                "mime": mime,
                "source_hash": source_hash,
                "contract": ETA_MU_INGEST_CONTRACT_ID,
                "bytes": int(stat.st_size),
                "mtime_utc": source_mtime_utc,
                "text_excerpt": excerpt,
                "space_id": str(space.get("id", "")),
                "space_signature": str(space.get("signature", "")),
                "space_set_id": str(spaces["space_set"].get("id", "")),
                "space_set_signature": str(spaces["space_set"].get("signature", "")),
                "vecstore_id": str(spaces["vecstore"].get("id", "")),
                "vecstore_collection": vec_collection,
                "vecstore_signature": str(spaces["vecstore"].get("signature", "")),
                "vecstore_backend": str(vec_result.get("backend", "")),
                "vecstore_drift": bool(vec_result.get("drift")),
                "dominant_field": dominant_field,
                "dominant_presence": presence_id,
                "dominant_weight": round(_clamp01(dominant_weight), 4),
                "field_scores": {
                    key: round(_safe_float(value, 0.0), 4)
                    for key, value in scores.items()
                },
                "top_fields": [
                    {
                        "field": field_id,
                        "name": field_name_map.get(field_id, field_id),
                        "weight": round(_safe_float(weight, 0.0), 4),
                    }
                    for field_id, weight in top_scores
                    if _safe_float(weight, 0.0) > 0.0
                ],
                "packet_refs": packet_refs_for_file,
            }
            _append_eta_mu_knowledge_record(vault_root, record)

            for embed_ref in embed_refs:
                registry_record = {
                    "record": ETA_MU_INGEST_REGISTRY_RECORD,
                    "event": "deferred" if deferred_this_file else "ingested",
                    "status": "defer" if deferred_this_file else "ok",
                    "reason": "vecstore_signature_drift"
                    if deferred_this_file
                    else "embedded",
                    "registry_key": registry_key,
                    "idempotence_key": str(embed_ref.get("idempotence_key", "")),
                    "embed_id": str(embed_ref.get("embed_id", "")),
                    "space_id": str(space.get("id", "")),
                    "space_signature": str(space.get("signature", "")),
                    "space_set_id": str(spaces["space_set"].get("id", "")),
                    "space_set_signature": str(
                        spaces["space_set"].get("signature", "")
                    ),
                    "vecstore_id": str(spaces["vecstore"].get("id", "")),
                    "vecstore_collection": vec_collection,
                    "vecstore_signature": str(spaces["vecstore"].get("signature", "")),
                    "vecstore_backend": str(vec_result.get("backend", "")),
                    "vecstore_drift": bool(vec_result.get("drift")),
                    "segment": dict(embed_ref.get("segment", {})),
                    "content_sha256": source_hash,
                    "source_hash": source_hash,
                    "fingerprint": digest,
                    "source_rel_path": rel_source,
                    "source_name": file_path.name,
                    "source_kind": modality,
                    "source_modality": modality,
                    "source_mime": mime,
                    "source_bytes": source_bytes,
                    "source_mtime_utc": source_mtime_utc,
                    "knowledge_id": record_id,
                    "archive_rel_path": archive_rel,
                    "archive_member_path": archive_member_path,
                    "packet_refs": packet_refs_for_file,
                    "time": ingested_at,
                }
                _append_eta_mu_registry_record(vault_root, registry_record)
                registry_known[registry_key] = dict(registry_record)
                registry_known_idempotence[
                    str(embed_ref.get("idempotence_key", ""))
                ] = dict(registry_record)

            manifest_sources.append(
                {
                    "source_rel_path": rel_source,
                    "source_hash": source_hash,
                    "mime": mime,
                    "modality": modality,
                    "space_id": str(space.get("id", "")),
                    "space_signature": str(space.get("signature", "")),
                    "embed_ids": [
                        str(embed_ref.get("embed_id", "")) for embed_ref in embed_refs
                    ],
                    "vecstore_collection": vec_collection,
                    "vecstore_backend": str(vec_result.get("backend", "")),
                    "vecstore_ids": [str(item) for item in vec_result.get("ids", [])],
                    "packet_refs": packet_refs_for_file,
                    "drift": bool(vec_result.get("drift")),
                }
            )

            processed_count += 1
            last_ingested_at = ingested_at
        except Exception as exc:
            failed_count += 1
            errors.append({"path": rel_source, "error": str(exc)})

    _cleanup_empty_inbox_dirs(inbox_root)
    remaining = _eta_mu_scan_candidates(inbox_root)
    entries = _load_eta_mu_knowledge_entries(vault_root)

    generated_at = datetime.now(timezone.utc).isoformat()
    output_root = _eta_mu_output_root(vault_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_stamp = _eta_mu_iso_compact()
    manifest_path = output_root / (
        ETA_MU_INGEST_MANIFEST_PREFIX + output_stamp + ETA_MU_INGEST_OUTPUT_EXT
    )
    stats_path = output_root / (
        ETA_MU_INGEST_STATS_PREFIX + output_stamp + ETA_MU_INGEST_OUTPUT_EXT
    )
    snapshot_path = output_root / (
        ETA_MU_INGEST_SNAPSHOT_PREFIX + output_stamp + ETA_MU_INGEST_OUTPUT_EXT
    )

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
        _eta_mu_write_sexp_artifact(manifest_path, manifest_payload)
    except OSError as exc:
        artifact_errors.append(f"manifest:{exc}")
    try:
        _eta_mu_write_sexp_artifact(stats_path, stats_payload)
    except OSError as exc:
        artifact_errors.append(f"stats:{exc}")
    try:
        _eta_mu_write_sexp_artifact(snapshot_path, snapshot_payload)
    except OSError as exc:
        artifact_errors.append(f"snapshot:{exc}")

    if artifact_errors:
        failed_count += len(artifact_errors)
        for message in artifact_errors:
            errors.append({"path": ".Π", "error": message})

    manifest_rel = _safe_rel_path(manifest_path, base_root)
    stats_rel = _safe_rel_path(stats_path, base_root)
    snapshot_rel = _safe_rel_path(snapshot_path, base_root)

    _eta_mu_emit_packet(
        packets,
        kind="snapshot-seal",
        body={
            "manifest": manifest_rel,
            "stats": stats_rel,
            "snapshot": snapshot_rel,
            "packet_count": len(packets),
            "coherence": "和",
            "sync": "同",
        },
    )

    snapshot = {
        "record": "ημ.inbox.v1",
        "path": str(inbox_root),
        "pending_count": len(remaining),
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "rejected_count": rejected_count,
        "deferred_count": deferred_count,
        "is_empty": len(remaining) == 0,
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
            "manifest": manifest_rel,
            "stats": stats_rel,
            "snapshot": snapshot_rel,
        },
        "packets": {
            "count": len(packets),
            "last": str(packets[-1].get("id", "")) if packets else "",
        },
        "errors": errors,
    }

    with _ETA_MU_INBOX_LOCK:
        _ETA_MU_INBOX_CACHE["root"] = str(inbox_root)
        _ETA_MU_INBOX_CACHE["snapshot"] = dict(snapshot)
        _ETA_MU_INBOX_CACHE["last_checked_monotonic"] = now_monotonic

    return snapshot


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

        organizer_terms_by_node[node_id] = _organizer_terms_from_entry(entry)
        node_mean_vector = _average_embedding_vectors(node_embed_vectors)
        if node_mean_vector is not None:
            organizer_cluster_rows.append(
                {
                    "node_id": node_id,
                    "vector": node_mean_vector,
                }
            )

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
                "embed_layer_count": len(node_layer_groups),
                "vecstore_collection": entry_collection,
                "url": str(entry.get("url", "")),
                "dominant_field": dominant_field,
                "dominant_presence": dominant_presence,
                "field_scores": dict(entry.get("field_scores", {})),
                "text_excerpt": str(entry.get("text_excerpt", "")),
            }
        )

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

    nodes = [*field_nodes, *file_nodes]
    return {
        "record": ETA_MU_FILE_GRAPH_RECORD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inbox": inbox_state,
        "nodes": nodes,
        "field_nodes": field_nodes,
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
            "knowledge_entries": len(entries),
            "archive_count": archive_count,
            "compressed_bytes_total": compressed_bytes_total,
        },
    }


def _json_deep_clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _weaver_service_base_url() -> str:
    bind_host = WEAVER_HOST_ENV or "127.0.0.1"
    probe_host = _weaver_probe_host(bind_host)
    return f"http://{probe_host}:{WEAVER_PORT}"


def _read_weaver_snapshot_file(part_root: Path) -> dict[str, Any] | None:
    snapshot_path = (
        part_root / "world_state" / "web_graph_weaver.snapshot.json"
    ).resolve()
    if not snapshot_path.exists() or not snapshot_path.is_file():
        return None
    try:
        payload = json.loads(snapshot_path.read_text("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    graph = payload.get("graph", {})
    if not isinstance(graph, dict):
        return None
    status = payload.get("status", {})
    if not isinstance(status, dict):
        status = {}
    return {
        "ok": True,
        "graph": graph,
        "status": status,
        "source": str(snapshot_path),
    }


def _fetch_weaver_graph_payload(part_root: Path) -> dict[str, Any]:
    def _graph_node_count(graph_payload: dict[str, Any]) -> int:
        counts = graph_payload.get("counts", {})
        if isinstance(counts, dict):
            from_counts = int(_safe_float(counts.get("nodes_total", 0), 0.0))
            if from_counts > 0:
                return from_counts
        nodes = graph_payload.get("nodes", [])
        if isinstance(nodes, list):
            return len(nodes)
        return 0

    base_url = _weaver_service_base_url()
    parsed = urlparse(base_url)
    probe_host = parsed.hostname or "127.0.0.1"
    if _weaver_health_check(
        probe_host,
        WEAVER_PORT,
        timeout_s=WEAVER_GRAPH_HEALTH_TIMEOUT_SECONDS,
    ):
        graph_req = Request(
            (
                f"{base_url}/api/weaver/graph"
                f"?node_limit={WEAVER_GRAPH_NODE_LIMIT}"
                f"&edge_limit={WEAVER_GRAPH_EDGE_LIMIT}"
            ),
            method="GET",
        )
        status_req = Request(f"{base_url}/api/weaver/status", method="GET")
        try:
            with urlopen(graph_req, timeout=WEAVER_GRAPH_FETCH_TIMEOUT_SECONDS) as resp:
                graph_payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            with urlopen(
                status_req, timeout=WEAVER_GRAPH_FETCH_TIMEOUT_SECONDS
            ) as resp:
                status_payload = json.loads(
                    resp.read().decode("utf-8", errors="ignore")
                )
            graph = (
                graph_payload.get("graph", {})
                if isinstance(graph_payload, dict)
                else {}
            )
            status = status_payload if isinstance(status_payload, dict) else {}
            if isinstance(graph, dict):
                live_nodes = _graph_node_count(graph)
                if live_nodes <= 0:
                    fallback = _read_weaver_snapshot_file(part_root)
                    if fallback is not None:
                        fallback_graph = fallback.get("graph", {})
                        if isinstance(fallback_graph, dict):
                            fallback_nodes = _graph_node_count(fallback_graph)
                            if fallback_nodes > 0:
                                fallback_status = fallback.get("status", {})
                                merged_status: dict[str, Any] = {}
                                if isinstance(fallback_status, dict):
                                    merged_status.update(fallback_status)
                                if isinstance(status, dict):
                                    merged_status.update(status)
                                return {
                                    "ok": True,
                                    "graph": fallback_graph,
                                    "status": merged_status,
                                    "source": str(fallback.get("source", "")),
                                }
                return {
                    "ok": True,
                    "graph": graph,
                    "status": status,
                    "source": f"{base_url}/api/weaver/graph",
                }
        except Exception:
            pass

    fallback = _read_weaver_snapshot_file(part_root)
    if fallback is not None:
        return fallback

    return {
        "ok": False,
        "graph": {"nodes": [], "edges": [], "counts": {}},
        "status": {},
        "source": "",
    }


def _infer_weaver_field_scores(node: dict[str, Any]) -> dict[str, float]:
    scores = {field_id: 0.0 for field_id in FIELD_TO_PRESENCE}
    kind = str(node.get("kind", "")).strip().lower()
    url = str(node.get("url", "") or node.get("label", "")).strip().lower()
    domain = str(node.get("domain", "")).strip().lower()
    title = str(node.get("title", "")).strip().lower()
    content_type = str(node.get("content_type", "")).strip().lower()

    if kind == "url":
        scores["f2"] += 0.24
        scores["f6"] += 0.24
        scores["f3"] += 0.12
    elif kind == "domain":
        scores["f2"] += 0.32
        scores["f8"] += 0.22
        scores["f3"] += 0.1
    elif kind == "content":
        if (
            content_type.startswith("image/")
            or content_type.startswith("audio/")
            or content_type.startswith("video/")
        ):
            scores["f1"] += 0.5
        else:
            scores["f6"] += 0.42
            scores["f3"] += 0.18

    combined = " ".join(filter(None, [url, domain, title, content_type]))
    tokens = _clean_tokens(combined)
    for token in tokens:
        for field_id, keywords in ETA_MU_FIELD_KEYWORDS.items():
            if token in keywords:
                scores[field_id] += 0.06

    for needle in ("policy", "privacy", "terms", "robots", "compliance", "license"):
        if needle in combined:
            scores["f7"] += 0.15
    for needle in ("blog", "news", "article", "docs", "wiki", "readme"):
        if needle in combined:
            scores["f6"] += 0.12
            scores["f3"] += 0.08
    for needle in ("status", "dashboard", "metrics", "api", "admin"):
        if needle in combined:
            scores["f8"] += 0.11

    total = sum(max(0.0, value) for value in scores.values())
    if total <= 0.0:
        fallback = "f2" if kind in {"domain", "url"} else "f6"
        scores[fallback] = 1.0
        return scores

    normalized: dict[str, float] = {}
    for field_id, value in scores.items():
        normalized[field_id] = round(max(0.0, value) / total, 4)
    return normalized


def _crawler_node_importance(node: dict[str, Any], dominant_weight: float) -> float:
    kind = str(node.get("kind", "")).strip().lower()
    if kind == "domain":
        return _clamp01(0.35 + (dominant_weight * 0.55))
    if kind == "content":
        return _clamp01(0.28 + (dominant_weight * 0.5))

    depth = _safe_float(node.get("depth", 0), 0.0)
    status = str(node.get("status", "")).strip().lower()
    compliance = str(node.get("compliance", "")).strip().lower()
    score = 0.22 + (dominant_weight * 0.5)
    score += _clamp01(1.0 - (depth / 8.0)) * 0.18
    if status in {"fetched", "duplicate"}:
        score += 0.08
    if compliance in {"allowed", "pending"}:
        score += 0.05
    return _clamp01(score)


def _build_weaver_field_graph_uncached(
    part_root: Path, vault_root: Path
) -> dict[str, Any]:
    source_payload = _fetch_weaver_graph_payload(part_root)
    graph_payload = (
        source_payload.get("graph", {}) if isinstance(source_payload, dict) else {}
    )
    status_payload = (
        source_payload.get("status", {}) if isinstance(source_payload, dict) else {}
    )
    if not isinstance(graph_payload, dict):
        graph_payload = {}
    if not isinstance(status_payload, dict):
        status_payload = {}

    raw_nodes = graph_payload.get("nodes", [])
    raw_edges = graph_payload.get("edges", [])
    if not isinstance(raw_nodes, list):
        raw_nodes = []
    if not isinstance(raw_edges, list):
        raw_edges = []

    entity_lookup = {
        str(entity.get("id", "")): entity
        for entity in ENTITY_MANIFEST
        if str(entity.get("id", "")).strip()
    }
    field_name_map = _field_schema_name_map()

    field_nodes: list[dict[str, Any]] = []
    for field_id in CANONICAL_NAMED_FIELD_IDS:
        entity = entity_lookup.get(field_id)
        if entity is None:
            continue
        mapped_field = next(
            (
                key
                for key, presence_id in FIELD_TO_PRESENCE.items()
                if presence_id == field_id
            ),
            "f3",
        )
        field_nodes.append(
            {
                "id": f"crawler-field:{field_id}",
                "node_id": field_id,
                "node_type": "field",
                "field": mapped_field,
                "label": str(entity.get("en", field_id)),
                "label_ja": str(entity.get("ja", "")),
                "x": round(_safe_float(entity.get("x", 0.5), 0.5), 4),
                "y": round(_safe_float(entity.get("y", 0.5), 0.5), 4),
                "hue": int(_safe_float(entity.get("hue", 200), 200.0)),
            }
        )

    crawler_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_id_map: dict[str, str] = {}
    kind_counts: dict[str, int] = defaultdict(int)
    field_counts: dict[str, int] = defaultdict(int)

    for index, node in enumerate(raw_nodes[:WEAVER_GRAPH_NODE_LIMIT]):
        if not isinstance(node, dict):
            continue
        original_id = str(node.get("id", "")).strip()
        if not original_id:
            continue
        scores = _infer_weaver_field_scores(node)
        dominant_field, dominant_weight = _dominant_eta_mu_field(scores)
        dominant_presence = FIELD_TO_PRESENCE.get(dominant_field, "anchor_registry")
        anchor = entity_lookup.get(dominant_presence, {"x": 0.5, "y": 0.5, "hue": 200})
        seed = sha1(f"crawler|{original_id}|{index}".encode("utf-8")).digest()
        angle = (int.from_bytes(seed[0:2], "big") / 65535.0) * math.tau
        radius = 0.05 + (int.from_bytes(seed[2:4], "big") / 65535.0) * 0.2
        jitter_x = ((seed[4] / 255.0) - 0.5) * 0.042
        jitter_y = ((seed[5] / 255.0) - 0.5) * 0.042
        x = _clamp01(
            _safe_float(anchor.get("x", 0.5), 0.5) + math.cos(angle) * radius + jitter_x
        )
        y = _clamp01(
            _safe_float(anchor.get("y", 0.5), 0.5) + math.sin(angle) * radius + jitter_y
        )
        kind = str(node.get("kind", "url")).strip().lower() or "url"
        if kind == "domain":
            hue = 176
        elif kind == "content":
            hue = 22
        else:
            hue = int(_safe_float(anchor.get("hue", 200), 200.0))

        graph_node_id = f"crawler:{sha1(original_id.encode('utf-8')).hexdigest()[:16]}"
        node_id_map[original_id] = graph_node_id
        kind_counts[kind] += 1
        field_counts[dominant_field] += 1
        importance = _crawler_node_importance(node, dominant_weight)
        label = str(
            node.get("title", "")
            or node.get("domain", "")
            or node.get("label", "")
            or original_id
        )
        crawler_nodes.append(
            {
                "id": graph_node_id,
                "node_id": original_id,
                "node_type": "crawler",
                "crawler_kind": kind,
                "label": label,
                "x": round(x, 4),
                "y": round(y, 4),
                "hue": int(hue),
                "importance": round(importance, 4),
                "url": str(node.get("url", "") or ""),
                "domain": str(node.get("domain", "") or ""),
                "title": str(node.get("title", "") or ""),
                "status": str(node.get("status", "") or ""),
                "content_type": str(node.get("content_type", "") or ""),
                "compliance": str(node.get("compliance", "") or ""),
                "dominant_field": dominant_field,
                "dominant_presence": dominant_presence,
                "field_scores": {
                    key: round(_safe_float(value, 0.0), 4)
                    for key, value in scores.items()
                },
            }
        )

        ranked = sorted(
            [
                (str(field), _safe_float(weight, 0.0))
                for field, weight in scores.items()
            ],
            key=lambda row: row[1],
            reverse=True,
        )
        for edge_index, (field_id, weight) in enumerate(ranked[:2]):
            if weight <= 0:
                continue
            target_presence = FIELD_TO_PRESENCE.get(field_id, dominant_presence)
            if target_presence not in entity_lookup:
                continue
            edges.append(
                {
                    "id": f"crawler-edge:{graph_node_id}:{field_id}:{edge_index}",
                    "source": graph_node_id,
                    "target": f"crawler-field:{target_presence}",
                    "field": field_id,
                    "weight": round(_clamp01(weight), 4),
                    "kind": "categorizes",
                }
            )

    for edge in raw_edges[:WEAVER_GRAPH_EDGE_LIMIT]:
        if not isinstance(edge, dict):
            continue
        source_id = node_id_map.get(str(edge.get("source", "")).strip())
        target_id = node_id_map.get(str(edge.get("target", "")).strip())
        if not source_id or not target_id:
            continue
        kind = str(edge.get("kind", "hyperlink") or "hyperlink")
        if kind == "domain_membership":
            weight = 0.25
        elif kind == "content_membership":
            weight = 0.22
        elif kind == "canonical_redirect":
            weight = 0.34
        else:
            weight = 0.28
        edges.append(
            {
                "id": f"crawl-link:{str(edge.get('id', '')) or sha1((source_id + target_id + kind).encode('utf-8')).hexdigest()[:14]}",
                "source": source_id,
                "target": target_id,
                "field": "",
                "weight": round(weight, 4),
                "kind": kind,
            }
        )

    graph_counts = graph_payload.get("counts", {})
    if not isinstance(graph_counts, dict):
        graph_counts = {}
    nodes = [*field_nodes, *crawler_nodes]
    return {
        "record": ETA_MU_CRAWLER_GRAPH_RECORD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "endpoint": str(source_payload.get("source", "")),
            "service": "web-graph-weaver",
        },
        "status": status_payload,
        "nodes": nodes,
        "field_nodes": field_nodes,
        "crawler_nodes": crawler_nodes,
        "edges": edges,
        "stats": {
            "field_count": len(field_nodes),
            "crawler_count": len(crawler_nodes),
            "edge_count": len(edges),
            "kind_counts": dict(kind_counts),
            "field_counts": dict(field_counts),
            "nodes_total": int(
                _safe_float(
                    graph_counts.get("nodes_total", len(crawler_nodes)),
                    float(len(crawler_nodes)),
                )
            ),
            "edges_total": int(
                _safe_float(
                    graph_counts.get("edges_total", len(edges)), float(len(edges))
                )
            ),
            "url_nodes_total": int(
                _safe_float(
                    graph_counts.get("url_nodes_total", kind_counts.get("url", 0)),
                    float(kind_counts.get("url", 0)),
                )
            ),
        },
    }


def build_weaver_field_graph(part_root: Path, vault_root: Path) -> dict[str, Any]:
    substrate_root = _eta_mu_substrate_root(vault_root)
    cache_key = f"{part_root.resolve()}|{substrate_root}|{_weaver_service_base_url()}"
    now_monotonic = time.monotonic()
    with _WEAVER_GRAPH_CACHE_LOCK:
        cached_key = str(_WEAVER_GRAPH_CACHE.get("key", ""))
        cached_snapshot = _WEAVER_GRAPH_CACHE.get("snapshot")
        elapsed = now_monotonic - float(
            _WEAVER_GRAPH_CACHE.get("checked_monotonic", 0.0)
        )
        if (
            cached_snapshot is not None
            and cached_key == cache_key
            and elapsed < WEAVER_GRAPH_CACHE_SECONDS
        ):
            return _json_deep_clone(cached_snapshot)

    snapshot = _build_weaver_field_graph_uncached(part_root, vault_root)
    with _WEAVER_GRAPH_CACHE_LOCK:
        _WEAVER_GRAPH_CACHE["key"] = cache_key
        _WEAVER_GRAPH_CACHE["snapshot"] = _json_deep_clone(snapshot)
        _WEAVER_GRAPH_CACHE["checked_monotonic"] = now_monotonic
    return snapshot


def _truth_world_id() -> str:
    text = str(os.getenv("TRUTH_BINDING_WORLD_ID", "") or "").strip()
    if text:
        return text
    return "127.0.0.1:8787"


def _resolve_truth_ref(ref: str, vault_root: Path, part_root: Path) -> str:
    token = str(ref or "").strip()
    if not token:
        return ""
    if "://" in token or token.startswith(("runtime:", "artifact:", "gate:")):
        return token

    path_candidate = Path(token)
    if path_candidate.is_absolute():
        return str(path_candidate)

    vault_resolved = vault_root.resolve()
    for base in (vault_resolved, part_root.resolve(), Path.cwd().resolve()):
        resolved = (base / token).resolve()
        if not resolved.exists():
            continue
        try:
            return str(resolved.relative_to(vault_resolved)).replace("\\", "/")
        except ValueError:
            return str(resolved)
    return token


def _receipt_kind_counts(receipts_path: Path | None) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if (
        receipts_path is None
        or not receipts_path.exists()
        or not receipts_path.is_file()
    ):
        return {}
    for raw_line in receipts_path.read_text("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = _parse_receipt_line(line)
        kind = str(row.get("kind", "")).strip()
        if kind:
            counts[kind] += 1
    return dict(counts)


def _default_truth_state() -> dict[str, Any]:
    world_id = _truth_world_id()
    claim = {
        "id": "claim.push_truth_gate_ready",
        "text": "push-truth gate is ready for apply",
        "status": "undecided",
        "kappa": 0.0,
        "world": world_id,
        "proof_refs": [],
        "theta": TRUTH_BINDING_GUARD_THETA,
    }
    return {
        "record": ETA_MU_TRUTH_STATE_RECORD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "name_binding": {
            "id": "gates_of_truth",
            "symbol": "Gates_of_Truth",
            "glyph": "真",
            "ascii": "TRUTH",
            "law": "Truth requires world scope (ω) + proof refs + receipts.",
        },
        "world": {
            "id": world_id,
            "ctx/ω-world": world_id,
            "ctx_omega_world": world_id,
        },
        "claim": claim,
        "claims": [claim],
        "guard": {
            "theta": TRUTH_BINDING_GUARD_THETA,
            "passes": False,
        },
        "gate": {
            "target": "push-truth",
            "blocked": True,
            "reasons": ["truth-state-unavailable"],
        },
        "invariants": {
            "world_scoped": bool(world_id),
            "proof_required": False,
            "proof_kind_subset": True,
            "receipts_parse_ok": False,
            "sim_bead_mint_blocked": True,
            "truth_binding_registered": False,
        },
        "proof": {
            "required_kinds": list(TRUTH_ALLOWED_PROOF_KINDS),
            "entries": [],
            "counts": {
                "total": 0,
                "present": 0,
                "by_kind": {kind: 0 for kind in TRUTH_ALLOWED_PROOF_KINDS},
            },
        },
        "artifacts": {
            "pi_zip_count": 0,
            "host_handle": "",
            "host_has_github_gist": False,
            "truth_receipt_count": 0,
        },
        "schema": {
            "source": "",
            "required_refs": [],
            "required_hashes": [],
            "host_handle": "",
            "missing_refs": [],
            "missing_hashes": [],
        },
        "needs": ["truth-state-unavailable"],
    }


def _build_truth_binding_state_uncached(
    part_root: Path,
    vault_root: Path,
    *,
    promptdb_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        dry_run = build_push_truth_dry_run_payload(part_root, vault_root)
    except Exception as exc:
        fallback = _default_truth_state()
        fallback["gate"] = {
            "target": "push-truth",
            "blocked": True,
            "reasons": [f"dry-run-error:{exc.__class__.__name__}"],
        }
        fallback["needs"] = ["push-truth dry-run payload unavailable"]
        return fallback

    promptdb = promptdb_index if isinstance(promptdb_index, dict) else {}
    if not promptdb:
        promptdb = collect_promptdb_packets(vault_root)

    proof_schema = dry_run.get("proof_schema", {})
    if not isinstance(proof_schema, dict):
        proof_schema = {}
    artifacts = dry_run.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    gate = dry_run.get("gate", {})
    if not isinstance(gate, dict):
        gate = {}

    world_id = _truth_world_id()
    required_refs = [
        str(item).strip()
        for item in proof_schema.get("required_refs", [])
        if str(item).strip()
    ]
    required_hashes = [
        str(item).strip()
        for item in proof_schema.get("required_hashes", [])
        if str(item).strip()
    ]
    missing_refs = [
        str(item).strip()
        for item in proof_schema.get("missing_refs", [])
        if str(item).strip()
    ]
    missing_hashes = [
        str(item).strip()
        for item in proof_schema.get("missing_hashes", [])
        if str(item).strip()
    ]
    pi_zip_paths = [
        str(item).strip() for item in artifacts.get("pi_zip", []) if str(item).strip()
    ]
    host_handle = str(
        artifacts.get("host_handle") or proof_schema.get("host_handle", "")
    ).strip()
    host_has_github_gist = bool(artifacts.get("host_has_github_gist", False))
    gate_blocked = bool(gate.get("blocked", True))
    gate_reasons = [
        str(item).strip() for item in gate.get("reasons", []) if str(item).strip()
    ]

    truth_refs = [
        ".opencode/protocol/truth.v1.lisp",
        ".opencode/promptdb/contracts/truth-layer.contract.lisp",
        ".opencode/promptdb/03_bind_truth.intent.lisp",
    ]
    truth_ref_presence = {
        ref: _proof_ref_exists(ref, vault_root, part_root) for ref in truth_refs
    }
    truth_binding_registered = all(truth_ref_presence.values())

    packet_paths = [
        str(item.get("path", ""))
        for item in promptdb.get("packets", [])
        if isinstance(item, dict)
    ]
    contract_paths = [
        str(item.get("path", ""))
        for item in promptdb.get("contracts", [])
        if isinstance(item, dict)
    ]
    truth_packet_indexed = any(
        path.endswith("03_bind_truth.intent.lisp") for path in packet_paths
    )
    truth_contract_indexed = any(
        path.endswith("truth-layer.contract.lisp") for path in contract_paths
    )

    receipts_path = _locate_receipts_log(vault_root, part_root)
    receipt_kinds = _receipt_kind_counts(receipts_path)

    proof_entries: list[dict[str, Any]] = []
    schema_source = str(proof_schema.get("source", "")).strip()
    if schema_source:
        proof_entries.append(
            {
                "kind": ":logic/bridge",
                "ref": _resolve_truth_ref(schema_source, vault_root, part_root),
                "present": True,
                "detail": "manifest proof-schema source",
            }
        )

    for ref in required_refs:
        present = _proof_ref_exists(ref, vault_root, part_root)
        proof_entries.append(
            {
                "kind": ":trace/record",
                "ref": _resolve_truth_ref(ref, vault_root, part_root),
                "present": bool(present),
                "detail": "required-proof-ref",
            }
        )

    for path in pi_zip_paths[:8]:
        proof_entries.append(
            {
                "kind": ":evidence/record",
                "ref": _resolve_truth_ref(path, vault_root, part_root),
                "present": True,
                "detail": "pi-zip-artifact",
            }
        )

    if host_handle:
        proof_entries.append(
            {
                "kind": ":trace/record",
                "ref": host_handle,
                "present": host_has_github_gist,
                "detail": "host-handle",
            }
        )

    proof_entries.append(
        {
            "kind": ":score/run",
            "ref": "runtime:/api/push-truth/dry-run",
            "present": True,
            "detail": "gate dry-run",
        }
    )
    proof_entries.append(
        {
            "kind": ":gov/adjudication",
            "ref": "gate:push-truth",
            "present": not gate_blocked,
            "detail": "gate-pass"
            if not gate_blocked
            else ",".join(gate_reasons) or "blocked",
        }
    )

    for ref in truth_refs:
        proof_entries.append(
            {
                "kind": ":trace/record",
                "ref": ref,
                "present": bool(truth_ref_presence.get(ref)),
                "detail": "truth-binding-artifact",
            }
        )

    if receipts_path is not None:
        proof_entries.append(
            {
                "kind": ":trace/record",
                "ref": _resolve_truth_ref(str(receipts_path), vault_root, part_root),
                "present": True,
                "detail": "receipts-log",
            }
        )

    proof_kind_counts: dict[str, int] = defaultdict(int)
    proof_present_count = 0
    for entry in proof_entries:
        kind = str(entry.get("kind", "")).strip()
        if kind:
            proof_kind_counts[kind] += 1
        if bool(entry.get("present")):
            proof_present_count += 1
    for kind in TRUTH_ALLOWED_PROOF_KINDS:
        proof_kind_counts.setdefault(kind, 0)

    proof_kind_subset = all(
        kind in TRUTH_ALLOWED_PROOF_KINDS for kind in proof_kind_counts.keys()
    )
    receipts_parse_ok = "receipts-parse-failed" not in gate_reasons

    score = 0.0
    score += 0.3 if not gate_blocked else 0.0
    score += 0.16 if receipts_parse_ok else 0.0
    score += 0.11 if schema_source else 0.0
    score += 0.1 if required_refs and not missing_refs else 0.0
    score += 0.08 if required_hashes and not missing_hashes else 0.0
    score += 0.08 if pi_zip_paths else 0.0
    score += 0.07 if host_has_github_gist else 0.0
    score += 0.1 if truth_binding_registered else 0.0
    score += 0.06 if truth_packet_indexed else 0.0
    score += 0.04 if truth_contract_indexed else 0.0
    if gate_blocked:
        score = min(score, 0.49)
    elif score < 0.55:
        score = min(1.0, score + 0.12)
    kappa = round(_clamp01(score), 4)

    if gate_blocked:
        claim_status = "refuted"
    elif proof_present_count <= 0 or not proof_kind_subset or not world_id:
        claim_status = "undecided"
    else:
        claim_status = "proved"

    artifact_claim_status = "proved" if truth_binding_registered else "refuted"
    artifact_claim_kappa = 0.91 if truth_binding_registered else 0.33

    primary_proof_refs = [
        str(entry.get("ref", ""))
        for entry in proof_entries
        if bool(entry.get("present")) and str(entry.get("ref", "")).strip()
    ][:8]
    claims = [
        {
            "id": "claim.push_truth_gate_ready",
            "text": "push-truth gate is ready for apply",
            "status": claim_status,
            "kappa": kappa,
            "world": world_id,
            "proof_refs": primary_proof_refs,
            "theta": TRUTH_BINDING_GUARD_THETA,
        },
        {
            "id": "claim.truth_binding_registered",
            "text": "truth protocol, contract, and intent are registered",
            "status": artifact_claim_status,
            "kappa": artifact_claim_kappa,
            "world": world_id,
            "proof_refs": truth_refs,
            "theta": TRUTH_BINDING_GUARD_THETA,
        },
    ]

    return {
        "record": ETA_MU_TRUTH_STATE_RECORD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "name_binding": {
            "id": "gates_of_truth",
            "symbol": "Gates_of_Truth",
            "glyph": "真",
            "ascii": "TRUTH",
            "law": "Truth requires world scope (ω) + proof refs + receipts.",
        },
        "world": {
            "id": world_id,
            "ctx/ω-world": world_id,
            "ctx_omega_world": world_id,
        },
        "claim": claims[0],
        "claims": claims,
        "guard": {
            "theta": TRUTH_BINDING_GUARD_THETA,
            "passes": claims[0]["status"] == "proved"
            and _safe_float(claims[0].get("kappa", 0.0), 0.0)
            >= TRUTH_BINDING_GUARD_THETA,
        },
        "gate": {
            "target": str(gate.get("target", "push-truth") or "push-truth"),
            "blocked": gate_blocked,
            "reasons": gate_reasons,
        },
        "invariants": {
            "world_scoped": bool(world_id),
            "proof_required": proof_present_count > 0,
            "proof_kind_subset": proof_kind_subset,
            "receipts_parse_ok": receipts_parse_ok,
            "sim_bead_mint_blocked": True,
            "truth_binding_registered": truth_binding_registered,
            "promptdb_truth_packet_indexed": truth_packet_indexed,
            "promptdb_truth_contract_indexed": truth_contract_indexed,
        },
        "proof": {
            "required_kinds": list(TRUTH_ALLOWED_PROOF_KINDS),
            "entries": proof_entries,
            "counts": {
                "total": len(proof_entries),
                "present": proof_present_count,
                "by_kind": dict(proof_kind_counts),
            },
        },
        "artifacts": {
            "pi_zip_count": len(pi_zip_paths),
            "host_handle": host_handle,
            "host_has_github_gist": host_has_github_gist,
            "truth_receipt_count": int(
                _safe_float(receipt_kinds.get(":truth", 0), 0.0)
                + _safe_float(receipt_kinds.get(":refutation", 0), 0.0)
                + _safe_float(receipt_kinds.get(":adjudication", 0), 0.0)
            ),
            "decision_receipt_count": int(
                _safe_float(receipt_kinds.get(":decision", 0), 0.0)
            ),
        },
        "schema": {
            "source": schema_source,
            "required_refs": required_refs,
            "required_hashes": required_hashes,
            "host_handle": host_handle,
            "missing_refs": missing_refs,
            "missing_hashes": missing_hashes,
        },
        "needs": [str(item) for item in dry_run.get("needs", []) if str(item).strip()],
    }


def build_truth_binding_state(
    part_root: Path,
    vault_root: Path,
    *,
    promptdb_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    promptdb_packet_count = int(
        _safe_float(
            (promptdb_index or {}).get("packet_count", 0)
            if isinstance(promptdb_index, dict)
            else 0,
            0.0,
        )
    )
    promptdb_contract_count = int(
        _safe_float(
            (promptdb_index or {}).get("contract_count", 0)
            if isinstance(promptdb_index, dict)
            else 0,
            0.0,
        )
    )
    cache_key = (
        f"{part_root.resolve()}|{vault_root.resolve()}"
        f"|p{promptdb_packet_count}|c{promptdb_contract_count}"
    )
    now_monotonic = time.monotonic()

    with _TRUTH_BINDING_CACHE_LOCK:
        cached_key = str(_TRUTH_BINDING_CACHE.get("key", ""))
        cached_snapshot = _TRUTH_BINDING_CACHE.get("snapshot")
        elapsed = now_monotonic - float(
            _TRUTH_BINDING_CACHE.get("checked_monotonic", 0.0)
        )
        if (
            cached_snapshot is not None
            and cached_key == cache_key
            and elapsed < TRUTH_BINDING_CACHE_SECONDS
        ):
            return _json_deep_clone(cached_snapshot)

    snapshot = _build_truth_binding_state_uncached(
        part_root,
        vault_root,
        promptdb_index=promptdb_index,
    )
    with _TRUTH_BINDING_CACHE_LOCK:
        _TRUTH_BINDING_CACHE["key"] = cache_key
        _TRUTH_BINDING_CACHE["snapshot"] = _json_deep_clone(snapshot)
        _TRUTH_BINDING_CACHE["checked_monotonic"] = now_monotonic
    return snapshot


def _normalize_hint_text(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")


def infer_bilingual_name(text: str) -> tuple[str, str] | None:
    normalized = _normalize_hint_text(text)
    for hint, pair in NAME_HINTS:
        if hint in normalized:
            return pair
    return None


def build_display_name(item: dict[str, Any], file_path: Path) -> dict[str, str]:
    raw_meta = item.get("meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    title_en = str(meta.get("title_en", "") or meta.get("title", "")).strip()
    title_ja = str(meta.get("title_jp", "") or meta.get("title_ja", "")).strip()

    inferred = infer_bilingual_name(title_en) or infer_bilingual_name(file_path.name)
    if inferred is not None:
        en_default, ja_default = inferred
    else:
        en_default = file_path.stem.replace("_", " ").strip().title()
        ja_default = ""

    return {
        "en": title_en or en_default,
        "ja": title_ja or ja_default,
    }


def build_display_role(role: str) -> dict[str, str]:
    en, ja = ROLE_HINTS.get(role, (role or "unknown", "分類未定"))
    return {"en": en, "ja": ja}


def _manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key in ("files", "artifacts"):
        value = manifest.get(key, [])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    entries.append(item)
    return entries


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
    return None


def _zip_member_kind(member_path: str, *, is_dir: bool) -> str:
    if is_dir:
        return "dir"
    return classify_kind(Path(member_path))


def _zip_member_extension(member_path: str, *, is_dir: bool) -> str:
    if is_dir:
        return "dir"
    suffix = Path(member_path).suffix.lower().lstrip(".")
    return suffix or "(none)"


def collect_zip_catalog(
    part_root: Path,
    vault_root: Path,
    *,
    member_limit: int = 220,
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
        key=lambda path: (
            path.stat().st_mtime if path.exists() else 0.0,
            str(path),
        ),
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


def collect_catalog(part_root: Path, vault_root: Path) -> dict[str, Any]:
    inbox_snapshot = sync_eta_mu_inbox(vault_root)
    eta_mu_file_graph = build_eta_mu_file_graph(
        vault_root,
        inbox_snapshot=inbox_snapshot,
    )
    crawler_graph = build_weaver_field_graph(part_root, vault_root)
    items: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    seen_paths: set[str] = set()

    for root in discover_part_roots(vault_root, part_root):
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text("utf-8"))
        part = _part_label(root, manifest)
        for item in _manifest_entries(manifest):
            rel = str(item.get("path", "")).strip()
            if not rel:
                continue

            file_path = (root / rel).resolve()
            if not file_path.exists() or not file_path.is_file():
                continue

            try:
                relative_to_vault = file_path.relative_to(vault_root.resolve())
            except ValueError:
                continue

            rel_norm = str(relative_to_vault).replace("\\", "/")
            if rel_norm in seen_paths:
                continue
            seen_paths.add(rel_norm)

            stat = file_path.stat()
            kind = classify_kind(file_path)
            role = str(item.get("role", "unknown"))
            counts[kind] += 1
            items.append(
                {
                    "part": part,
                    "name": file_path.name,
                    "role": role,
                    "display_name": build_display_name(item, file_path),
                    "display_role": build_display_role(role),
                    "kind": kind,
                    "bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "rel_path": rel_norm,
                    "url": "/library/" + quote(rel_norm),
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
    truth_state = build_truth_binding_state(
        part_root,
        vault_root,
        promptdb_index=promptdb_index,
    )
    test_failures, test_coverage = _load_test_signal_artifacts(part_root, vault_root)

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
        "file_graph": eta_mu_file_graph,
        "crawler_graph": crawler_graph,
        "truth_state": truth_state,
        "test_failures": test_failures,
        "test_coverage": test_coverage,
        "promptdb": promptdb_index,
        "items": items,
    }
    catalog["logical_graph"] = _build_logical_graph(catalog)
    catalog["pain_field"] = _build_pain_field(catalog, catalog["logical_graph"])
    catalog["heat_values"] = _materialize_heat_values(catalog, catalog["pain_field"])
    pi_archive = build_pi_archive_payload(
        part_root,
        vault_root,
        catalog=catalog,
    )
    catalog["pi_archive"] = {
        "record": ETA_MU_PI_ARCHIVE_RECORD,
        "generated_at": pi_archive.get("generated_at", ""),
        "hash": pi_archive.get("hash", {}),
        "signature": pi_archive.get("signature", {}),
        "portable": pi_archive.get("portable", {}),
        "ledger_count": int((pi_archive.get("ledger") or {}).get("count", 0) or 0),
    }
    return catalog


def _catalog_signature(catalog: dict[str, Any]) -> str:
    rows: list[str] = []
    for item in catalog.get("items", []):
        rel_path = str(item.get("rel_path", ""))
        rows.append(
            "|".join(
                [
                    rel_path,
                    str(item.get("bytes", 0)),
                    str(item.get("mtime_utc", "")),
                    str(item.get("kind", "")),
                ]
            )
        )
    promptdb = catalog.get("promptdb", {})
    rows.append(f"promptdb_packets={int(promptdb.get('packet_count', 0))}")
    rows.append(f"promptdb_contracts={int(promptdb.get('contract_count', 0))}")
    crawler_graph = catalog.get("crawler_graph", {})
    if isinstance(crawler_graph, dict):
        crawler_stats = crawler_graph.get("stats", {})
        crawler_status = crawler_graph.get("status", {})
        if not isinstance(crawler_stats, dict):
            crawler_stats = {}
        if not isinstance(crawler_status, dict):
            crawler_status = {}
        rows.append(
            f"crawler_count={int(_safe_float(crawler_stats.get('crawler_count', 0), 0.0))}"
        )
        rows.append(
            f"crawler_edge_count={int(_safe_float(crawler_stats.get('edge_count', 0), 0.0))}"
        )
        rows.append(
            f"crawler_url_total={int(_safe_float(crawler_stats.get('url_nodes_total', 0), 0.0))}"
        )
        rows.append(
            f"crawler_alive={int(_safe_float(crawler_status.get('alive', 0), 0.0))}"
        )
        rows.append(
            f"crawler_queue={int(_safe_float(crawler_status.get('queue_size', 0), 0.0))}"
        )
    truth_state = catalog.get("truth_state", {})
    if isinstance(truth_state, dict):
        claim = truth_state.get("claim", {})
        guard = truth_state.get("guard", {})
        gate = truth_state.get("gate", {})
        if not isinstance(claim, dict):
            claim = {}
        if not isinstance(guard, dict):
            guard = {}
        if not isinstance(gate, dict):
            gate = {}
        rows.append(f"truth_claim_status={str(claim.get('status', 'undecided'))}")
        rows.append(
            f"truth_claim_kappa={round(_safe_float(claim.get('kappa', 0.0), 0.0), 4)}"
        )
        rows.append(f"truth_guard_pass={int(bool(guard.get('passes', False)))}")
        rows.append(f"truth_gate_blocked={int(bool(gate.get('blocked', True)))}")
        reasons = [
            str(item).strip() for item in gate.get("reasons", []) if str(item).strip()
        ]
        rows.append("truth_gate_reasons=" + ",".join(sorted(reasons)))
    rows.sort()
    return sha1("\n".join(rows).encode("utf-8")).hexdigest()


def _catalog_item_signature_map(catalog: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in catalog.get("items", []):
        rel_path = str(item.get("rel_path", "")).strip()
        if not rel_path:
            continue
        mapping[rel_path] = "|".join(
            [
                str(item.get("mtime_utc", "")),
                str(item.get("bytes", 0)),
                str(item.get("kind", "")),
            ]
        )
    return mapping


def _catalog_delta_stats(
    previous_catalog: dict[str, Any],
    current_catalog: dict[str, Any],
) -> dict[str, Any]:
    previous = _catalog_item_signature_map(previous_catalog)
    current = _catalog_item_signature_map(current_catalog)

    previous_keys = set(previous)
    current_keys = set(current)
    added = sorted(current_keys - previous_keys)
    removed = sorted(previous_keys - current_keys)
    updated = sorted(
        key
        for key in (current_keys & previous_keys)
        if current.get(key) != previous.get(key)
    )
    sample_paths = [*added[:3], *updated[:3], *removed[:2]]
    return {
        "added_count": len(added),
        "updated_count": len(updated),
        "removed_count": len(removed),
        "sample_paths": sample_paths,
        "total_changes": len(added) + len(updated) + len(removed),
    }


def _extract_lisp_string(source: str, key: str) -> str | None:
    match = re.search(rf"\({re.escape(key)}\s+\"([^\"]+)\"\)", source)
    if match:
        return match.group(1)
    return None


def _extract_lisp_keyword(source: str, key: str) -> str | None:
    match = re.search(rf"\({re.escape(key)}\s+(:[-\w]+)\)", source)
    if match:
        return match.group(1)
    return None


def _extract_contract_name(source: str) -> str | None:
    match = re.search(r"\(contract\s+\"([^\"]+)\"", source)
    if match:
        return match.group(1)
    match = re.search(r"\(契\b[\s\S]*?\(id\s+\"([^\"]+)\"", source)
    if match:
        return match.group(1)
    return None


def parse_promptdb_packet(packet_path: Path, promptdb_root: Path) -> dict[str, Any]:
    source = packet_path.read_text("utf-8")
    packet_id = _extract_lisp_string(source, "id")
    title = _extract_lisp_string(source, "title")
    version = _extract_lisp_string(source, "v")
    kind = _extract_lisp_keyword(source, "kind")

    tags_match = re.search(r"\(tags\s+\[([^\]]*)\]\)", source, re.DOTALL)
    tags = re.findall(r":[-\w]+", tags_match.group(1)) if tags_match else []

    routing = {
        "target": _extract_lisp_keyword(source, "target"),
        "handler": _extract_lisp_keyword(source, "handler"),
        "mode": _extract_lisp_keyword(source, "mode"),
    }

    rel_path = str(packet_path.relative_to(promptdb_root.parent)).replace("\\", "/")
    node_key = packet_id or rel_path

    return {
        "node_key": node_key,
        "path": rel_path,
        "id": packet_id,
        "v": version,
        "kind": kind,
        "title": title,
        "tags": tags,
        "routing": routing,
    }


def parse_promptdb_contract(contract_path: Path, promptdb_root: Path) -> dict[str, Any]:
    source = contract_path.read_text("utf-8")
    contract_name = _extract_contract_name(source)
    rel_path = str(contract_path.relative_to(promptdb_root.parent)).replace("\\", "/")
    node_key = contract_name or rel_path
    return {
        "node_key": node_key,
        "path": rel_path,
        "id": contract_name,
        "v": "",
        "kind": ":contract",
        "title": contract_name or contract_path.stem,
        "tags": [":contract"],
        "routing": {"target": None, "handler": None, "mode": None},
    }


def _iter_promptdb_files(promptdb_root: Path) -> list[Path]:
    rows: list[Path] = []
    for pattern in (*PROMPTDB_PACKET_GLOBS, *PROMPTDB_CONTRACT_GLOBS):
        rows.extend(path for path in promptdb_root.rglob(pattern) if path.is_file())
    deduped = {path.resolve(): path for path in rows}
    return [deduped[key] for key in sorted(deduped)]


def _promptdb_signature(paths: list[Path], promptdb_root: Path) -> str:
    rows: list[str] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = str(path.relative_to(promptdb_root)).replace("\\", "/")
        rows.append(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}")
    rows.sort()
    return sha1("\n".join(rows).encode("utf-8")).hexdigest()


def _clone_promptdb_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "root": str(snapshot.get("root", "")),
        "packet_count": int(snapshot.get("packet_count", 0)),
        "contract_count": int(snapshot.get("contract_count", 0)),
        "file_count": int(snapshot.get("file_count", 0)),
        "packets": [dict(item) for item in snapshot.get("packets", [])],
        "contracts": [dict(item) for item in snapshot.get("contracts", [])],
        "errors": [dict(item) for item in snapshot.get("errors", [])],
    }


def _build_promptdb_snapshot(promptdb_root: Path, paths: list[Path]) -> dict[str, Any]:
    packets: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_keys: set[str] = set()

    for packet_path in paths:
        is_contract = packet_path.name.endswith(".contract.lisp")
        try:
            parsed = (
                parse_promptdb_contract(packet_path, promptdb_root)
                if is_contract
                else parse_promptdb_packet(packet_path, promptdb_root)
            )
            node_key = str(parsed.get("node_key") or "")
            if not node_key or node_key in seen_keys:
                continue
            seen_keys.add(node_key)
            if is_contract:
                contracts.append(parsed)
            else:
                packets.append(parsed)
        except Exception as exc:
            rel = str(packet_path).replace("\\", "/")
            errors.append({"path": rel, "error": str(exc)})

    return {
        "root": str(promptdb_root),
        "packet_count": len(packets),
        "contract_count": len(contracts),
        "file_count": len(paths),
        "packets": packets,
        "contracts": contracts,
        "errors": errors,
    }


def collect_promptdb_packets(vault_root: Path) -> dict[str, Any]:
    promptdb_root = locate_promptdb_root(vault_root)
    checked_at = datetime.now(timezone.utc).isoformat()
    if promptdb_root is None:
        return {
            "root": "",
            "packet_count": 0,
            "contract_count": 0,
            "file_count": 0,
            "packets": [],
            "contracts": [],
            "errors": [],
            "refresh": {
                "strategy": "polling+debounce",
                "debounce_ms": int(PROMPTDB_REFRESH_DEBOUNCE_SECONDS * 1000),
                "checks": 0,
                "refreshes": 0,
                "cache_hits": 0,
                "last_checked_at": checked_at,
                "last_refreshed_at": "",
                "last_decision": "promptdb-root-missing",
            },
        }

    with _PROMPTDB_CACHE_LOCK:
        root_str = str(promptdb_root)
        now_monotonic = time.monotonic()
        _PROMPTDB_CACHE["checks"] += 1
        _PROMPTDB_CACHE["last_checked_at"] = checked_at

        if _PROMPTDB_CACHE.get("root") != root_str:
            _PROMPTDB_CACHE["root"] = root_str
            _PROMPTDB_CACHE["signature"] = ""
            _PROMPTDB_CACHE["snapshot"] = None
            _PROMPTDB_CACHE["last_decision"] = "root-changed"
            _PROMPTDB_CACHE["last_check_monotonic"] = 0.0

        elapsed = now_monotonic - float(
            _PROMPTDB_CACHE.get("last_check_monotonic", 0.0)
        )
        snapshot = _PROMPTDB_CACHE.get("snapshot")
        if snapshot is not None and elapsed < PROMPTDB_REFRESH_DEBOUNCE_SECONDS:
            _PROMPTDB_CACHE["cache_hits"] += 1
            _PROMPTDB_CACHE["last_decision"] = "debounced-cache"
        else:
            paths = _iter_promptdb_files(promptdb_root)
            signature = _promptdb_signature(paths, promptdb_root)
            if snapshot is None or signature != _PROMPTDB_CACHE.get("signature"):
                snapshot = _build_promptdb_snapshot(promptdb_root, paths)
                _PROMPTDB_CACHE["snapshot"] = snapshot
                _PROMPTDB_CACHE["signature"] = signature
                _PROMPTDB_CACHE["refreshes"] += 1
                _PROMPTDB_CACHE["last_refreshed_at"] = checked_at
                _PROMPTDB_CACHE["last_decision"] = "refreshed"
            else:
                _PROMPTDB_CACHE["cache_hits"] += 1
                _PROMPTDB_CACHE["last_decision"] = "cache-hit"
        _PROMPTDB_CACHE["last_check_monotonic"] = now_monotonic

        stable_snapshot = _clone_promptdb_snapshot(
            _PROMPTDB_CACHE.get("snapshot")
            or {
                "root": root_str,
                "packet_count": 0,
                "contract_count": 0,
                "file_count": 0,
                "packets": [],
                "contracts": [],
                "errors": [],
            }
        )

        stable_snapshot["refresh"] = {
            "strategy": "polling+debounce",
            "debounce_ms": int(PROMPTDB_REFRESH_DEBOUNCE_SECONDS * 1000),
            "checks": int(_PROMPTDB_CACHE.get("checks", 0)),
            "refreshes": int(_PROMPTDB_CACHE.get("refreshes", 0)),
            "cache_hits": int(_PROMPTDB_CACHE.get("cache_hits", 0)),
            "last_checked_at": str(_PROMPTDB_CACHE.get("last_checked_at", "")),
            "last_refreshed_at": str(_PROMPTDB_CACHE.get("last_refreshed_at", "")),
            "last_decision": str(_PROMPTDB_CACHE.get("last_decision", "unknown")),
        }
        return stable_snapshot


def locate_promptdb_root(vault_root: Path) -> Path | None:
    candidates: list[Path] = []
    candidates.append(vault_root.resolve())
    candidates.extend(vault_root.resolve().parents)
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents)

    seen: set[Path] = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        candidate = base / ".opencode" / "promptdb"
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _clean_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_-]+", text.lower()) if token]


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

    rendered = (
        f"[{entity.get('en', 'Presence')}] says: "
        f"facts={len(facts)} asks={len(asks)} repairs={len(repairs)}"
    )

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


def _locate_receipts_log(vault_root: Path, part_root: Path) -> Path | None:
    candidates: list[Path] = []
    for base in [vault_root.resolve(), part_root.resolve(), Path.cwd().resolve()]:
        candidates.append(base / "receipts.log")
        for parent in base.parents:
            candidates.append(parent / "receipts.log")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _parse_receipt_line(line: str) -> dict[str, str]:
    row: dict[str, str] = {}
    parts = [part.strip() for part in line.split(" | ") if part.strip()]
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            continue
        row[key.strip()] = value.strip()
    return row


def _split_receipt_refs(refs_value: str) -> list[str]:
    return [item.strip() for item in refs_value.split(",") if item.strip()]


def _ensure_receipts_log_path(vault_root: Path, part_root: Path) -> Path:
    located = _locate_receipts_log(vault_root, part_root)
    if located is not None:
        return located
    fallback = (vault_root / "receipts.log").resolve()
    fallback.parent.mkdir(parents=True, exist_ok=True)
    if not fallback.exists():
        fallback.touch()
    return fallback


def _append_receipt_line(
    receipts_path: Path,
    *,
    kind: str,
    origin: str,
    owner: str,
    dod: str,
    refs: list[str],
    pi: str,
    host: str,
    manifest: str,
    note: str | None = None,
    tests: str | None = None,
    drift: str | None = None,
) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    kind_value = kind if kind.startswith(":") else f":{kind}"
    refs_value = ",".join(sorted({ref for ref in refs if ref}))
    fields = [
        f"ts={ts}",
        f"kind={kind_value}",
        f"origin={origin}",
        f"owner={owner}",
        f"dod={dod}",
        f"pi={pi}",
        f"host={host}",
        f"manifest={manifest}",
        f"refs={refs_value}",
    ]
    if note:
        fields.append(f"note={note}")
    if tests:
        fields.append(f"tests={tests}")
    if drift:
        fields.append(f"drift={drift}")
    line = " | ".join(fields)
    with receipts_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return line


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _strip_transient_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_transient_fields(item)
            for key, item in value.items()
            if key not in {"generated_at", "checked_at"}
        }
    if isinstance(value, list):
        return [_strip_transient_fields(item) for item in value]
    return value


def _build_receipt_rows(receipts_path: Path | None) -> list[dict[str, str]]:
    if (
        receipts_path is None
        or not receipts_path.exists()
        or not receipts_path.is_file()
    ):
        return []
    rows: list[dict[str, str]] = []
    for raw in receipts_path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = _parse_receipt_line(line)
        if row:
            rows.append(row)
    return rows


def _build_pi_archive_signature(
    archive_hash: str,
    *,
    world_id: str,
    host: str,
) -> dict[str, str]:
    key = str(os.getenv("PI_ARCHIVE_SIGNING_KEY", "") or "")
    if key:
        digest = hmac.new(
            key.encode("utf-8"),
            archive_hash.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "algo": "hmac-sha256",
            "key_id": str(
                os.getenv("PI_ARCHIVE_SIGNING_KEY_ID", "env:PI_ARCHIVE_SIGNING_KEY")
                or "env:PI_ARCHIVE_SIGNING_KEY"
            ),
            "value": digest,
        }

    digest = hashlib.sha256(
        f"{archive_hash}|{world_id}|{host}".encode("utf-8")
    ).hexdigest()
    return {
        "algo": "sha256",
        "key_id": "derived:world-host",
        "value": digest,
    }


def validate_pi_archive_portable(archive: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    required_sections = ["snapshot", "ledger", "catalog", "hash", "signature"]
    for key in required_sections:
        if not isinstance(archive.get(key), dict):
            errors.append(f"missing:{key}")

    ledger = archive.get("ledger", {})
    if not isinstance(ledger, dict):
        ledger = {}
    entries = ledger.get("entries", [])
    if not isinstance(entries, list):
        entries = []

    missing_required_fields = set()
    for row in entries:
        if not isinstance(row, dict):
            continue
        for field in PI_ARCHIVE_REQUIRED_RECEIPT_FIELDS:
            if not str(row.get(field, "")).strip():
                missing_required_fields.add(field)
    if entries and missing_required_fields:
        errors.append(
            "ledger_missing_fields:" + ",".join(sorted(missing_required_fields))
        )

    digest = str(((archive.get("hash") or {}).get("canonical_sha256", ""))).strip()
    if not digest or len(digest) != 64:
        errors.append("invalid:hash.canonical_sha256")
    signature = archive.get("signature", {})
    if not isinstance(signature, dict) or not str(signature.get("value", "")).strip():
        errors.append("invalid:signature.value")

    portable = len(errors) == 0
    return {
        "ok": portable,
        "portable": portable,
        "errors": errors,
        "required_sections": required_sections,
    }


def build_pi_archive_payload(
    part_root: Path,
    vault_root: Path,
    *,
    catalog: dict[str, Any] | None = None,
    queue_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    catalog_payload = (
        _json_deep_clone(catalog)
        if isinstance(catalog, dict)
        else collect_catalog(part_root, vault_root)
    )
    queue_payload = (
        _json_deep_clone(queue_snapshot)
        if isinstance(queue_snapshot, dict)
        else {"pending_count": 0, "event_count": 0, "pending": []}
    )

    world_id = _truth_world_id()
    host = f"{world_id}/ws"
    receipts_path = _locate_receipts_log(vault_root, part_root)
    receipt_rows = _build_receipt_rows(receipts_path)
    ledger_entries = [_strip_transient_fields(row) for row in receipt_rows]

    truth_state = catalog_payload.get("truth_state", {})
    if not isinstance(truth_state, dict):
        truth_state = {}

    snapshot = {
        "record": "ημ.snapshot.v1",
        "world": {
            "id": world_id,
            "runtime": {
                "base": "http://127.0.0.1:8787/",
                "must_have": ["/api/catalog", "/ws"],
            },
        },
        "truth_state": _strip_transient_fields(truth_state),
        "task_queue": {
            "pending_count": int(queue_payload.get("pending_count", 0) or 0),
            "event_count": int(queue_payload.get("event_count", 0) or 0),
        },
    }

    ledger = {
        "record": "ημ.ledger.v1",
        "path": str(receipts_path) if receipts_path else "",
        "count": len(ledger_entries),
        "entries": ledger_entries,
    }

    promptdb = catalog_payload.get("promptdb", {})
    if not isinstance(promptdb, dict):
        promptdb = {}
    catalog_summary = {
        "record": "ημ.catalog.v1",
        "signature": _catalog_signature(catalog_payload),
        "counts": _strip_transient_fields(catalog_payload.get("counts", {})),
        "item_count": len(catalog_payload.get("items", [])),
        "ui_default_perspective": str(
            catalog_payload.get(
                "ui_default_perspective", PROJECTION_DEFAULT_PERSPECTIVE
            )
        ),
        "promptdb": {
            "packet_count": int(promptdb.get("packet_count", 0) or 0),
            "contract_count": int(promptdb.get("contract_count", 0) or 0),
            "signature": str(promptdb.get("signature", "") or ""),
        },
    }

    hash_input = {
        "snapshot": snapshot,
        "ledger": ledger,
        "catalog": catalog_summary,
    }
    canonical_sha256 = hashlib.sha256(_canonical_json_bytes(hash_input)).hexdigest()
    hash_block = {
        "algo": "sha256",
        "canonical_sha256": canonical_sha256,
    }
    signature = _build_pi_archive_signature(
        canonical_sha256, world_id=world_id, host=host
    )

    archive = {
        "record": ETA_MU_PI_ARCHIVE_RECORD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "world_id": world_id,
        "snapshot": snapshot,
        "ledger": ledger,
        "catalog": catalog_summary,
        "hash": hash_block,
        "signature": signature,
        "invariants": {
            "deterministic_hash": True,
            "append_only_ledger": True,
            "truth_unit": "pi-package+host",
            "components": ["snapshot", "ledger", "catalog", "hash", "signature"],
        },
    }
    portable = validate_pi_archive_portable(archive)
    archive["portable"] = {
        "ok": portable["ok"],
        "errors": portable["errors"],
    }
    return archive


def _path_glob_allowed(
    path_value: str, includes: list[str], excludes: list[str]
) -> bool:
    normalized = _normalize_path_for_file_id(path_value)
    if not normalized:
        return False
    include_match = True
    if includes:
        include_match = any(
            fnmatch.fnmatch(normalized, pattern) for pattern in includes
        )
    if not include_match:
        return False
    if excludes and any(fnmatch.fnmatch(normalized, pattern) for pattern in excludes):
        return False
    return True


def _event_source_rel_path(path_value: str, vault_root: Path, part_root: Path) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    normalized = raw.replace("\\", "/")
    if normalized.startswith("/library/"):
        normalized = normalized[len("/library/") :]
    candidate = Path(normalized)
    if candidate.is_absolute():
        for base in [vault_root.resolve(), part_root.resolve()]:
            try:
                rel = candidate.resolve().relative_to(base)
                return _normalize_path_for_file_id(str(rel))
            except (OSError, ValueError):
                continue
    return _normalize_path_for_file_id(normalized)


def _pairwise_overlap_boundaries(members: list[str]) -> list[dict[str, str]]:
    clean_members = [item for item in members if str(item).strip()]
    boundaries: list[dict[str, str]] = []
    for left_idx in range(len(clean_members)):
        for right_idx in range(left_idx + 1, len(clean_members)):
            left = clean_members[left_idx]
            right = clean_members[right_idx]
            boundary_id = hashlib.sha1(f"{left}|{right}".encode("utf-8")).hexdigest()[
                :12
            ]
            boundaries.append(
                {"id": f"boundary:{boundary_id}", "left": left, "right": right}
            )
    return boundaries


def _influence_world_map(influence_snapshot: dict[str, Any]) -> dict[str, float]:
    rows = _projection_presence_impacts(None, influence_snapshot)
    world_map: dict[str, float] = {}
    for row in rows:
        presence_id = str(row.get("id", "")).strip()
        if not presence_id:
            continue
        affects = row.get("affects", {}) if isinstance(row.get("affects"), dict) else {}
        world_map[presence_id] = _clamp01(_safe_float(affects.get("world", 0.0), 0.0))

    file_ratio = _clamp01(
        _safe_float(influence_snapshot.get("file_changes_120s", 0.0), 0.0) / 24.0
    )
    click_ratio = _clamp01(
        _safe_float(influence_snapshot.get("clicks_45s", 0.0), 0.0) / 18.0
    )
    queue_pending_ratio = _clamp01(
        _safe_float(
            (influence_snapshot.get("task_queue") or {}).get("pending_count", 0.0), 0.0
        )
        / 8.0
    )
    world_map.setdefault(
        FILE_SENTINEL_PROFILE["id"], _clamp01((file_ratio * 0.72) + 0.22)
    )
    world_map.setdefault(
        FILE_ORGANIZER_PROFILE["id"], _clamp01((file_ratio * 0.58) + 0.28)
    )
    world_map.setdefault(
        THE_COUNCIL_PROFILE["id"], _clamp01((queue_pending_ratio * 0.6) + 0.34)
    )
    return world_map


def _file_graph_overlap_context(
    file_graph: dict[str, Any],
    source_rel_path: str,
) -> dict[str, Any]:
    normalized_source = _normalize_path_for_file_id(source_rel_path)
    file_nodes = (
        file_graph.get("file_nodes", []) if isinstance(file_graph, dict) else []
    )
    if not isinstance(file_nodes, list):
        file_nodes = []

    best_node: dict[str, Any] | None = None
    best_score = -1
    for node in file_nodes:
        if not isinstance(node, dict):
            continue
        candidates = [
            _normalize_path_for_file_id(str(node.get("source_rel_path", ""))),
            _normalize_path_for_file_id(str(node.get("archived_rel_path", ""))),
            _normalize_path_for_file_id(str(node.get("archive_rel_path", ""))),
            _normalize_path_for_file_id(str(node.get("name", ""))),
        ]
        candidates = [value for value in candidates if value]
        for candidate in candidates:
            score = 0
            if candidate == normalized_source:
                score = 100 + len(candidate)
            elif normalized_source and (
                candidate.endswith(normalized_source)
                or normalized_source.endswith(candidate)
            ):
                score = min(len(candidate), len(normalized_source))
            if score > best_score:
                best_score = score
                best_node = node

    members: list[str] = []
    field = "f3"
    node_id = ""
    if isinstance(best_node, dict):
        node_id = str(best_node.get("id", "")).strip()
        field = str(best_node.get("dominant_field", "f3")).strip() or "f3"
        dominant_presence = str(best_node.get("dominant_presence", "")).strip()
        if dominant_presence:
            members.append(dominant_presence)
        field_scores = (
            best_node.get("field_scores", {})
            if isinstance(best_node.get("field_scores"), dict)
            else {}
        )
        ranked_fields = sorted(
            [
                (str(key), _safe_float(value, 0.0))
                for key, value in field_scores.items()
                if _safe_float(value, 0.0) > 0.0
            ],
            key=lambda row: row[1],
            reverse=True,
        )
        for field_id, weight in ranked_fields[:3]:
            if weight <= 0.12:
                continue
            member = FIELD_TO_PRESENCE.get(field_id, "")
            if member:
                members.append(member)
        concept_presence_id = str(best_node.get("concept_presence_id", "")).strip()
        if concept_presence_id:
            members.append(concept_presence_id)
    else:
        inferred_kind = classify_kind(Path(normalized_source or "unknown.txt"))
        scores = _infer_eta_mu_field_scores(
            rel_path=normalized_source,
            kind=inferred_kind,
            text_excerpt="",
        )
        ranked = sorted(scores.items(), key=lambda row: row[1], reverse=True)
        if ranked:
            field = str(ranked[0][0])
        for field_id, weight in ranked[:3]:
            if _safe_float(weight, 0.0) <= 0.1:
                continue
            mapped = FIELD_TO_PRESENCE.get(str(field_id), "")
            if mapped:
                members.append(mapped)

    members.append(FILE_SENTINEL_PROFILE["id"])
    members = [str(item).strip() for item in members if str(item).strip()]
    members = list(dict.fromkeys(members))
    boundaries = _pairwise_overlap_boundaries(members)
    return {
        "source_rel_path": normalized_source,
        "node_id": node_id,
        "field": field,
        "members": members,
        "boundary_pairs": boundaries,
    }


def _docker_compose_service_names(compose_path: Path) -> list[str]:
    if not compose_path.exists() or not compose_path.is_file():
        return []
    try:
        lines = compose_path.read_text("utf-8").splitlines()
    except OSError:
        return []

    services: list[str] = []
    in_services = False
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if not in_services:
            if stripped == "services:":
                in_services = True
            continue
        if indent == 0:
            break
        if indent == 2:
            match = re.match(r"([A-Za-z0-9_.-]+):\s*$", stripped)
            if match:
                services.append(match.group(1))
    return services


def _council_auto_vote(
    member_id: str,
    *,
    event_type: str,
    gate_blocked: bool,
    influence_world: dict[str, float],
    overlap_count: int,
) -> tuple[str, str]:
    member = str(member_id).strip()
    if member == "gates_of_truth":
        if gate_blocked:
            return "no", "gate blocked by unresolved drift signals"
        return "yes", "gate is clear"
    if member == FILE_SENTINEL_PROFILE["id"]:
        if event_type in {"file_changed", "file_added", "file_removed"}:
            return "yes", "file sentinel witnessed actionable file delta"
        return "abstain", "no file delta observed"
    if member == THE_COUNCIL_PROFILE["id"]:
        if overlap_count >= COUNCIL_MIN_OVERLAP_MEMBERS and not gate_blocked:
            return "yes", "overlap boundary quorum satisfied"
        return "no", "overlap quorum or gate condition failed"
    if member.startswith("presence:concept:"):
        if overlap_count >= COUNCIL_MIN_OVERLAP_MEMBERS:
            return "yes", "concept boundary overlaps impacted resource"
        return "abstain", "concept not strongly coupled to current overlap"

    influence = _clamp01(_safe_float(influence_world.get(member, 0.5), 0.5))
    if influence >= COUNCIL_MIN_MEMBER_WORLD_INFLUENCE:
        return "yes", f"world influence {influence:.2f} >= threshold"
    return "abstain", f"world influence {influence:.2f} below threshold"


class CouncilChamber:
    def __init__(
        self,
        decision_log_path: Path,
        receipts_path: Path,
        *,
        owner: str,
        host: str,
        part_root: Path,
        vault_root: Path,
        manifest: str = "manifest.lith",
    ) -> None:
        self._decision_log_path = decision_log_path.resolve()
        self._receipts_path = receipts_path.resolve()
        self._owner = owner
        self._host = host
        self._part_root = part_root.resolve()
        self._vault_root = vault_root.resolve()
        self._manifest = manifest
        self._lock = threading.Lock()
        self._event_count = 0
        self._decisions: dict[str, dict[str, Any]] = {}
        self._last_restart_ts = 0.0
        self._load_from_log()

    def _load_from_log(self) -> None:
        if not self._decision_log_path.exists():
            return
        for raw in self._decision_log_path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            decision = event.get("decision")
            if isinstance(decision, dict):
                decision_id = str(decision.get("id", "")).strip()
                if decision_id:
                    self._decisions[decision_id] = decision
                    action = decision.get("action", {})
                    if isinstance(action, dict) and bool(action.get("ok", False)):
                        self._last_restart_ts = max(
                            self._last_restart_ts,
                            _safe_float(action.get("unix_ts", 0.0), 0.0),
                        )
            self._event_count += 1

    def _append_event(self, *, op: str, decision: dict[str, Any]) -> None:
        self._decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "v": COUNCIL_EVENT_VERSION,
            "ts": datetime.now(timezone.utc).isoformat(),
            "op": op,
            "decision": decision,
        }
        with self._decision_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._event_count += 1

    def _required_yes(self, member_count: int) -> int:
        return max(2, int(math.ceil(max(1, member_count) * 0.6)))

    def _tally_votes(
        self, votes: list[dict[str, Any]], member_count: int
    ) -> dict[str, Any]:
        yes = 0
        no = 0
        abstain = 0
        for row in votes:
            vote = str(row.get("vote", "abstain")).strip().lower()
            if vote == "yes":
                yes += 1
            elif vote == "no":
                no += 1
            else:
                abstain += 1
        required_yes = self._required_yes(member_count)
        approved = yes >= required_yes and no == 0
        return {
            "yes": yes,
            "no": no,
            "abstain": abstain,
            "required_yes": required_yes,
            "approved": approved,
        }

    def _restart_action(self, decision: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        if now - self._last_restart_ts < DOCKER_AUTORESTART_COOLDOWN_SECONDS:
            return {
                "attempted": False,
                "ok": False,
                "result": "cooldown",
                "reason": "restart cooldown active",
                "unix_ts": now,
            }

        compose_rel = (
            str(
                os.getenv("DOCKER_AUTORESTART_COMPOSE_FILE", "docker-compose.yml")
                or "docker-compose.yml"
            ).strip()
            or "docker-compose.yml"
        )
        compose_path = (self._part_root / compose_rel).resolve()
        services_available = _docker_compose_service_names(compose_path)
        if not services_available:
            return {
                "attempted": False,
                "ok": False,
                "result": "compose-missing",
                "reason": "docker compose file/services not found",
                "compose_file": str(compose_path),
                "unix_ts": now,
            }

        service_targets = _split_csv_items(DOCKER_AUTORESTART_SERVICES)
        if not service_targets:
            service_targets = list(services_available)
        unknown = [item for item in service_targets if item not in services_available]
        if unknown:
            return {
                "attempted": False,
                "ok": False,
                "result": "invalid-services",
                "reason": "unknown compose services",
                "unknown": unknown,
                "available": services_available,
                "compose_file": str(compose_path),
                "unix_ts": now,
            }

        command = [
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "restart",
            *service_targets,
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                timeout=DOCKER_AUTORESTART_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return {
                "attempted": True,
                "ok": False,
                "result": "docker-missing",
                "reason": "docker binary not found",
                "command": command,
                "unix_ts": now,
            }
        except subprocess.TimeoutExpired:
            return {
                "attempted": True,
                "ok": False,
                "result": "timeout",
                "reason": "docker restart timed out",
                "command": command,
                "unix_ts": now,
            }
        except subprocess.CalledProcessError as exc:
            return {
                "attempted": True,
                "ok": False,
                "result": "failed",
                "reason": "docker compose restart failed",
                "command": command,
                "stdout": str(exc.stdout or "")[:1200],
                "stderr": str(exc.stderr or "")[:1200],
                "unix_ts": now,
            }

        self._last_restart_ts = now
        return {
            "attempted": True,
            "ok": True,
            "result": "executed",
            "command": command,
            "stdout": str(completed.stdout or "")[:1200],
            "stderr": str(completed.stderr or "")[:1200],
            "services": service_targets,
            "compose_file": str(compose_path),
            "unix_ts": now,
        }

    def _sorted_decisions(
        self, *, limit: int = COUNCIL_DECISION_HISTORY_LIMIT
    ) -> list[dict[str, Any]]:
        decisions = [
            dict(item) for item in self._decisions.values() if isinstance(item, dict)
        ]
        decisions.sort(
            key=lambda row: _safe_float(row.get("created_unix", 0.0), 0.0),
            reverse=True,
        )
        return decisions[: max(1, limit)]

    def snapshot(
        self, *, include_decisions: bool = False, limit: int = 16
    ) -> dict[str, Any]:
        decisions = self._sorted_decisions(limit=COUNCIL_DECISION_HISTORY_LIMIT)
        pending = [
            row
            for row in decisions
            if str(row.get("status", "")).strip() in {"pending", "awaiting-votes"}
        ]
        approved = [
            row
            for row in decisions
            if str(row.get("status", "")).strip() in {"approved", "executed"}
        ]
        payload = {
            "decision_log": str(self._decision_log_path),
            "event_count": self._event_count,
            "decision_count": len(decisions),
            "pending_count": len(pending),
            "approved_count": len(approved),
            "auto_restart_enabled": DOCKER_AUTORESTART_ENABLED,
            "require_council": DOCKER_AUTORESTART_REQUIRE_COUNCIL,
            "cooldown_seconds": DOCKER_AUTORESTART_COOLDOWN_SECONDS,
        }
        if include_decisions:
            payload["decisions"] = decisions[: max(1, limit)]
        return payload

    def consider_event(
        self,
        *,
        event_type: str,
        data: dict[str, Any],
        catalog: dict[str, Any],
        influence_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        event_kind = str(event_type or "").strip()
        if event_kind not in {"file_changed", "file_added", "file_removed"}:
            return {
                "ok": False,
                "status": "ignored",
                "reason": "event-not-actionable",
            }
        if not DOCKER_AUTORESTART_ENABLED:
            return {
                "ok": False,
                "status": "disabled",
                "reason": "docker auto-restart disabled",
            }

        source_rel_path = _event_source_rel_path(
            str(data.get("path", "")),
            self._vault_root,
            self._part_root,
        )
        includes = _split_csv_items(DOCKER_AUTORESTART_INCLUDE_GLOBS)
        excludes = _split_csv_items(DOCKER_AUTORESTART_EXCLUDE_GLOBS)
        if not _path_glob_allowed(source_rel_path, includes, excludes):
            return {
                "ok": False,
                "status": "filtered",
                "source_rel_path": source_rel_path,
                "reason": "path filtered by include/exclude policy",
            }

        file_graph = catalog.get("file_graph", {}) if isinstance(catalog, dict) else {}
        overlap = _file_graph_overlap_context(
            file_graph if isinstance(file_graph, dict) else {},
            source_rel_path,
        )
        members = [
            str(item) for item in overlap.get("members", []) if str(item).strip()
        ]
        if THE_COUNCIL_PROFILE["id"] not in members:
            members.append(THE_COUNCIL_PROFILE["id"])
        members = list(dict.fromkeys(members))

        if len(members) < COUNCIL_MIN_OVERLAP_MEMBERS:
            return {
                "ok": False,
                "status": "no-council",
                "source_rel_path": source_rel_path,
                "overlap": overlap,
                "reason": "insufficient presence overlap",
            }

        drift = build_drift_scan_payload(self._part_root, self._vault_root)
        blocked_gates = (
            drift.get("blocked_gates", [])
            if isinstance(drift.get("blocked_gates"), list)
            else []
        )
        gate_reasons = [
            str(item.get("reason", "unknown"))
            for item in blocked_gates
            if isinstance(item, dict)
        ]
        gate_blocked = bool(gate_reasons)

        created_at = datetime.now(timezone.utc).isoformat()
        created_unix = time.time()
        decision_seed = (
            f"{event_kind}|{source_rel_path}|{','.join(sorted(members))}|{created_at}"
        )
        decision_id = (
            "decision:council:"
            + hashlib.sha1(decision_seed.encode("utf-8")).hexdigest()[:14]
        )
        council_id = (
            "council:"
            + hashlib.sha1(
                f"{source_rel_path}|{','.join(sorted(members))}".encode("utf-8")
            ).hexdigest()[:12]
        )

        influence_world = _influence_world_map(influence_snapshot)
        votes: list[dict[str, Any]] = []
        for member_id in members:
            vote, reason = _council_auto_vote(
                member_id,
                event_type=event_kind,
                gate_blocked=gate_blocked,
                influence_world=influence_world,
                overlap_count=len(members),
            )
            votes.append(
                {
                    "member_id": member_id,
                    "vote": vote,
                    "reason": reason,
                    "mode": "auto",
                    "ts": created_at,
                }
            )

        tally = self._tally_votes(votes, len(members))
        approved = bool(tally.get("approved", False)) and not gate_blocked
        status = "approved" if approved else "awaiting-votes"
        if gate_blocked:
            status = "blocked"

        decision = {
            "id": decision_id,
            "kind": "docker.restart.on-change",
            "status": status,
            "created_at": created_at,
            "created_unix": created_unix,
            "source_event": {
                "type": event_kind,
                "data": dict(data),
            },
            "resource": {
                "source_rel_path": source_rel_path,
                "field": str(overlap.get("field", "f3")),
                "node_id": str(overlap.get("node_id", "")),
            },
            "space": {
                "field": str(overlap.get("field", "f3")),
                "members": members,
                "boundary_pairs": overlap.get("boundary_pairs", []),
                "overlap_count": len(members),
            },
            "council": {
                "id": council_id,
                "members": members,
                "required_yes": int(tally.get("required_yes", 2)),
                "votes": votes,
                "tally": tally,
            },
            "gate": {
                "blocked": gate_blocked,
                "reasons": gate_reasons,
            },
            "action": {
                "attempted": False,
                "ok": False,
                "result": "pending-approval",
                "unix_ts": created_unix,
            },
        }

        if approved and (not DOCKER_AUTORESTART_REQUIRE_COUNCIL or approved):
            decision["action"] = self._restart_action(decision)
            action = decision.get("action", {})
            if isinstance(action, dict):
                if bool(action.get("ok", False)):
                    decision["status"] = "executed"
                elif str(action.get("result", "")) in {
                    "cooldown",
                    "compose-missing",
                    "invalid-services",
                }:
                    decision["status"] = "blocked"
                else:
                    decision["status"] = "error"

        with self._lock:
            self._decisions[decision_id] = decision
            self._append_event(op="decision", decision=decision)

        _append_receipt_line(
            self._receipts_path,
            kind=":decision",
            origin="council",
            owner=self._owner,
            dod="Council evaluated docker restart decision and applied boundary vote policy",
            pi="part64-runtime-system",
            host=self._host,
            manifest=self._manifest,
            refs=[
                f"decision:{decision_id}",
                f"council:{council_id}",
                source_rel_path,
                "council:docker-autorestart",
                ".opencode/promptdb/contracts/receipts.v2.contract.lisp",
            ],
            note=f"status={decision.get('status', '')}; members={len(members)}; gate_blocked={gate_blocked}",
        )
        return {
            "ok": True,
            "decision": decision,
            "council": self.snapshot(include_decisions=False),
        }

    def vote(
        self,
        *,
        decision_id: str,
        member_id: str,
        vote: str,
        reason: str,
        actor: str,
    ) -> dict[str, Any]:
        with self._lock:
            decision = self._decisions.get(decision_id)
            if not isinstance(decision, dict):
                return {"ok": False, "error": "decision_not_found"}

            council = decision.get("council", {})
            members = [
                str(item).strip()
                for item in council.get("members", [])
                if str(item).strip()
            ]
            member = str(member_id).strip()
            if member not in members:
                return {
                    "ok": False,
                    "error": "member_not_in_council",
                    "members": members,
                }

            vote_value = str(vote).strip().lower()
            if vote_value not in {"yes", "no", "abstain"}:
                return {
                    "ok": False,
                    "error": "invalid_vote",
                    "allowed": ["yes", "no", "abstain"],
                }

            votes = council.get("votes", [])
            if not isinstance(votes, list):
                votes = []
            ts = datetime.now(timezone.utc).isoformat()
            replaced = False
            for row in votes:
                if str(row.get("member_id", "")) == member:
                    row["vote"] = vote_value
                    row["reason"] = str(reason).strip()
                    row["mode"] = "manual"
                    row["actor"] = actor
                    row["ts"] = ts
                    replaced = True
                    break
            if not replaced:
                votes.append(
                    {
                        "member_id": member,
                        "vote": vote_value,
                        "reason": str(reason).strip(),
                        "mode": "manual",
                        "actor": actor,
                        "ts": ts,
                    }
                )

            tally = self._tally_votes(votes, len(members))
            gate = (
                decision.get("gate", {})
                if isinstance(decision.get("gate"), dict)
                else {}
            )
            gate_blocked = bool(gate.get("blocked", False))
            approved = bool(tally.get("approved", False)) and not gate_blocked
            decision_status = "approved" if approved else "awaiting-votes"
            if gate_blocked:
                decision_status = "blocked"

            council["votes"] = votes
            council["tally"] = tally
            decision["council"] = council
            decision["status"] = decision_status

            if approved and not bool(
                (decision.get("action") or {}).get("attempted", False)
            ):
                decision["action"] = self._restart_action(decision)
                action = (
                    decision.get("action", {})
                    if isinstance(decision.get("action"), dict)
                    else {}
                )
                if bool(action.get("ok", False)):
                    decision["status"] = "executed"
                elif str(action.get("result", "")) in {
                    "cooldown",
                    "compose-missing",
                    "invalid-services",
                }:
                    decision["status"] = "blocked"
                else:
                    decision["status"] = "error"

            self._decisions[decision_id] = decision
            self._append_event(op="vote", decision=decision)

        return {
            "ok": True,
            "decision": decision,
            "council": self.snapshot(include_decisions=False),
        }


class TaskQueue:
    def __init__(
        self,
        queue_log_path: Path,
        receipts_path: Path,
        *,
        owner: str,
        host: str,
        manifest: str = "manifest.lith",
    ) -> None:
        self._queue_log_path = queue_log_path.resolve()
        self._receipts_path = receipts_path.resolve()
        self._owner = owner
        self._host = host
        self._manifest = manifest
        self._lock = threading.Lock()
        self._pending: list[dict[str, Any]] = []
        self._dedupe_index: dict[str, str] = {}
        self._event_count = 0
        self._load_from_log()

    def _load_from_log(self) -> None:
        if not self._queue_log_path.exists():
            return

        for raw in self._queue_log_path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            op = str(event.get("op", "")).strip()
            if op == "enqueue":
                task = event.get("task")
                if not isinstance(task, dict):
                    continue
                task_id = str(task.get("id", "")).strip()
                if not task_id or any(
                    str(item.get("id", "")) == task_id for item in self._pending
                ):
                    continue
                self._pending.append(task)
                dedupe_key = str(task.get("dedupe_key", "")).strip()
                if dedupe_key:
                    self._dedupe_index[dedupe_key] = task_id
            elif op == "dequeue":
                task_id = str(event.get("task_id", "")).strip()
                self._remove_pending_task(task_id)
            self._event_count += 1

    def _remove_pending_task(self, task_id: str) -> dict[str, Any] | None:
        if not task_id:
            return None
        for idx, task in enumerate(self._pending):
            if str(task.get("id", "")).strip() != task_id:
                continue
            removed = self._pending.pop(idx)
            dedupe_key = str(removed.get("dedupe_key", "")).strip()
            if dedupe_key:
                self._dedupe_index.pop(dedupe_key, None)
            return removed
        return None

    def _append_event(self, event: dict[str, Any]) -> None:
        self._queue_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._queue_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._event_count += 1

    def enqueue(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        dedupe_key: str,
        owner: str | None = None,
        dod: str = "task queued with persisted log and receipt",
        refs: list[str] | None = None,
    ) -> dict[str, Any]:
        refs = refs or []
        normalized_dedupe = dedupe_key.strip() or (
            f"{kind}:{json.dumps(payload, sort_keys=True, ensure_ascii=False)}"
        )
        owner_value = (owner or self._owner).strip() or self._owner
        with self._lock:
            existing_id = self._dedupe_index.get(normalized_dedupe)
            if existing_id:
                existing_task = next(
                    (
                        item
                        for item in self._pending
                        if str(item.get("id", "")) == existing_id
                    ),
                    None,
                )
                if existing_task is not None:
                    return {
                        "ok": True,
                        "deduped": True,
                        "task": existing_task,
                        "queue": self.snapshot(include_pending=False),
                    }

            created_at = datetime.now(timezone.utc).isoformat()
            task_id_seed = f"{normalized_dedupe}|{created_at}|{time.time_ns()}"
            task_id = f"task-{sha1(task_id_seed.encode('utf-8')).hexdigest()[:12]}"
            task = {
                "id": task_id,
                "kind": kind,
                "payload": payload,
                "dedupe_key": normalized_dedupe,
                "owner": owner_value,
                "status": "pending",
                "created_at": created_at,
            }
            event = {
                "v": TASK_QUEUE_EVENT_VERSION,
                "ts": created_at,
                "op": "enqueue",
                "task": task,
            }
            self._pending.append(task)
            self._dedupe_index[normalized_dedupe] = task_id
            self._append_event(event)
            _append_receipt_line(
                self._receipts_path,
                kind=":decision",
                origin="task-queue",
                owner=owner_value,
                dod=dod,
                pi="part64-runtime-system",
                host=self._host,
                manifest=self._manifest,
                refs=[
                    "task-queue:enqueue",
                    f"task:{task_id}",
                    ".opencode/promptdb/diagrams/part64_runtime_system.packet.lisp",
                    ".opencode/promptdb/contracts/receipts.v2.contract.lisp",
                    *refs,
                ],
            )
            return {
                "ok": True,
                "deduped": False,
                "task": task,
                "queue": self.snapshot(include_pending=False),
            }

    def dequeue(
        self,
        *,
        owner: str | None = None,
        dod: str = "task dequeued with persisted log and receipt",
        refs: list[str] | None = None,
    ) -> dict[str, Any]:
        refs = refs or []
        owner_value = (owner or self._owner).strip() or self._owner
        with self._lock:
            if not self._pending:
                return {
                    "ok": False,
                    "error": "empty_queue",
                    "queue": self.snapshot(include_pending=False),
                }

            task = self._pending.pop(0)
            dedupe_key = str(task.get("dedupe_key", "")).strip()
            if dedupe_key:
                self._dedupe_index.pop(dedupe_key, None)

            ts = datetime.now(timezone.utc).isoformat()
            event = {
                "v": TASK_QUEUE_EVENT_VERSION,
                "ts": ts,
                "op": "dequeue",
                "task_id": task.get("id"),
            }
            self._append_event(event)
            _append_receipt_line(
                self._receipts_path,
                kind=":decision",
                origin="task-queue",
                owner=owner_value,
                dod=dod,
                pi="part64-runtime-system",
                host=self._host,
                manifest=self._manifest,
                refs=[
                    "task-queue:dequeue",
                    f"task:{task.get('id', '')}",
                    ".opencode/promptdb/diagrams/part64_runtime_system.packet.lisp",
                    ".opencode/promptdb/contracts/receipts.v2.contract.lisp",
                    *refs,
                ],
            )
            return {
                "ok": True,
                "task": task,
                "queue": self.snapshot(include_pending=False),
            }

    def snapshot(self, *, include_pending: bool = False) -> dict[str, Any]:
        data = {
            "queue_log": str(self._queue_log_path),
            "pending_count": len(self._pending),
            "dedupe_keys": len(self._dedupe_index),
            "event_count": self._event_count,
        }
        if include_pending:
            data["pending"] = [dict(item) for item in self._pending]
        return data


def _extract_lisp_vector_strings(source: str, key: str) -> list[str]:
    match = re.search(rf"\({re.escape(key)}\s+\[([^\]]*)\]\)", source, re.DOTALL)
    if not match:
        return []
    return [item.strip() for item in re.findall(r'"([^\"]+)"', match.group(1))]


def _collect_open_questions(vault_root: Path) -> list[dict[str, str]]:
    promptdb_root = locate_promptdb_root(vault_root)
    if promptdb_root is None:
        return []
    packet_path = promptdb_root / PROMPTDB_OPEN_QUESTIONS_PACKET
    if not packet_path.exists() or not packet_path.is_file():
        return []

    source = packet_path.read_text("utf-8")
    matches = re.findall(
        r"\(q\s+\(id\s+([^)\s]+)\)\s+\(text\s+\"([^\"]+)\"\)\)",
        source,
    )
    return [
        {"id": question_id.strip(), "text": question_text.strip()}
        for question_id, question_text in matches
    ]


def _partition_open_questions(
    open_questions: list[dict[str, str]],
    receipt_refs: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    resolved: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []
    refs_text = "\n".join(receipt_refs)
    for item in open_questions:
        question_id = str(item.get("id", "")).strip()
        if question_id and question_id in refs_text:
            resolved.append(item)
        else:
            unresolved.append(item)
    return resolved, unresolved


def _build_keeper_of_contracts_signal(
    unresolved_questions: list[dict[str, str]],
    blocked_gates: list[dict[str, Any]],
    promptdb_packet_count: int,
) -> dict[str, Any] | None:
    if not unresolved_questions or not blocked_gates:
        return None

    top_questions = unresolved_questions[:3]
    text = " ".join(f"{q['id']}: {q['text']}" for q in top_questions)
    payload = build_presence_say_payload(
        {
            "items": [],
            "promptdb": {"packet_count": promptdb_packet_count},
        },
        text=f"gate unresolved {text}",
        requested_presence_id="keeper_of_contracts",
    )

    asks = [f"Resolve {item['id']}" for item in top_questions]
    payload["say_intent"]["asks"] = asks
    payload["say_intent"]["facts"].append(f"blocked_gates={len(blocked_gates)}")
    payload["say_intent"]["facts"].append(
        f"unresolved_questions={len(unresolved_questions)}"
    )
    payload["say_intent"]["repairs"].append(
        "Append receipt refs for resolved question ids (q.*) and gate decisions."
    )
    return payload


def _extract_manifest_proof_schema(manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None or not manifest_path.exists():
        return {
            "source": "",
            "required_refs": [],
            "required_hashes": [],
            "host_handle": "",
        }

    source = manifest_path.read_text("utf-8")
    return {
        "source": str(manifest_path),
        "required_refs": _extract_lisp_vector_strings(source, "required-refs"),
        "required_hashes": _extract_lisp_vector_strings(source, "required-hashes"),
        "host_handle": _extract_lisp_string(source, "host-handle") or "",
    }


def _proof_ref_exists(ref: str, vault_root: Path, part_root: Path) -> bool:
    candidate_ref = ref.strip()
    if not candidate_ref:
        return False
    if "://" in candidate_ref:
        return True
    if candidate_ref.startswith("runtime:"):
        return True
    if candidate_ref.startswith("artifact:"):
        return True

    path_candidate = Path(candidate_ref)
    if path_candidate.is_absolute():
        return path_candidate.exists()

    for base in (vault_root.resolve(), part_root.resolve(), Path.cwd().resolve()):
        resolved = (base / candidate_ref).resolve()
        if resolved.exists():
            return True
    return False


def _sha256_for_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pi_zip_name_hash_token(path: Path) -> str:
    stem = path.stem.strip()
    for prefix in ("Π.", "Pi.", "pi."):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return stem.strip().lower()


def _pi_zip_name_check(path: Path, sha256_hex: str) -> dict[str, Any]:
    expected_sha12 = sha256_hex[:12].lower()
    name_sha12 = _pi_zip_name_hash_token(path)
    return {
        "path": str(path),
        "name": path.name,
        "expected_sha12": expected_sha12,
        "name_sha12": name_sha12,
        "matches_sha12": name_sha12 == expected_sha12,
    }


def build_drift_scan_payload(part_root: Path, vault_root: Path) -> dict[str, Any]:
    required_keys = [
        "ts",
        "kind",
        "origin",
        "owner",
        "dod",
        "pi",
        "host",
        "manifest",
        "refs",
    ]
    receipts_path = _locate_receipts_log(vault_root, part_root)
    active_drifts: list[dict[str, Any]] = []
    blocked_gates: list[dict[str, Any]] = []
    parse_ok = True
    row_count = 0
    has_intent_ref = False
    parsed_refs: list[str] = []

    promptdb_index = collect_promptdb_packets(vault_root)
    promptdb_packet_count = int(promptdb_index.get("packet_count", 0))
    open_questions = _collect_open_questions(vault_root)

    if receipts_path is None:
        parse_ok = False
        active_drifts.append(
            {
                "id": "missing_receipts_log",
                "severity": "high",
                "detail": "receipts.log not found",
            }
        )
    else:
        for raw_line in receipts_path.read_text("utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            row_count += 1
            row = _parse_receipt_line(line)
            if not all(key in row for key in required_keys):
                parse_ok = False
                active_drifts.append(
                    {
                        "id": "receipt_line_missing_keys",
                        "severity": "medium",
                        "detail": f"line {row_count} missing required keys",
                    }
                )
                continue
            refs = _split_receipt_refs(row.get("refs", ""))
            parsed_refs.extend(refs)
            if any("00_wire_world.intent.lisp" in ref for ref in refs):
                has_intent_ref = True

    if not parse_ok:
        blocked_gates.append(
            {
                "target": "push-truth",
                "reason": "receipts-parse-failed",
            }
        )
    if not has_intent_ref:
        active_drifts.append(
            {
                "id": "missing_intent_receipt_ref",
                "severity": "high",
                "detail": "push-truth receipt ref to 00_wire_world.intent.lisp not found",
            }
        )
        blocked_gates.append(
            {
                "target": "push-truth",
                "reason": "missing-intent-ref",
            }
        )

    resolved_questions, unresolved_questions = _partition_open_questions(
        open_questions, parsed_refs
    )
    if unresolved_questions:
        active_drifts.append(
            {
                "id": "open_questions_unresolved",
                "severity": "medium",
                "detail": f"{len(unresolved_questions)} open gate questions unresolved",
                "question_ids": [item["id"] for item in unresolved_questions],
            }
        )
        blocked_gates.append(
            {
                "target": "push-truth",
                "reason": "open-questions-unresolved",
                "question_ids": [item["id"] for item in unresolved_questions],
            }
        )

    keeper_signal = _build_keeper_of_contracts_signal(
        unresolved_questions=unresolved_questions,
        blocked_gates=blocked_gates,
        promptdb_packet_count=promptdb_packet_count,
    )

    receipts_parse = {
        "path": str(receipts_path) if receipts_path else "",
        "ok": parse_ok,
        "rows": row_count,
        "has_intent_ref": has_intent_ref,
    }

    return {
        "ok": True,
        "receipts": {
            "path": receipts_parse["path"],
            "parse_ok": receipts_parse["ok"],
            "rows": receipts_parse["rows"],
            "has_intent_ref": receipts_parse["has_intent_ref"],
        },
        "receipts_parse": receipts_parse,
        "drifts": active_drifts,
        "active_drifts": active_drifts,
        "blocked_gates": blocked_gates,
        "open_questions": {
            "total": len(open_questions),
            "resolved_count": len(resolved_questions),
            "unresolved_count": len(unresolved_questions),
            "resolved": resolved_questions,
            "unresolved": unresolved_questions,
        },
        "keeper_of_contracts": keeper_signal,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _find_truth_artifacts(vault_root: Path) -> list[str]:
    artifacts: list[str] = []
    for path in vault_root.rglob("Π*.zip"):
        if path.is_file():
            artifacts.append(str(path))
    artifacts.sort()
    return artifacts


def build_push_truth_dry_run_payload(
    part_root: Path, vault_root: Path
) -> dict[str, Any]:
    drift = build_drift_scan_payload(part_root, vault_root)
    artifact_paths = [Path(path) for path in _find_truth_artifacts(vault_root)]
    artifact_hashes: list[dict[str, str]] = []
    pi_zip_name_checks: list[dict[str, Any]] = []
    pi_zip_name_mismatches: list[dict[str, Any]] = []
    for path in artifact_paths:
        if not path.exists() or not path.is_file():
            continue
        sha256_hex = _sha256_for_path(path)
        artifact_hashes.append({"path": str(path), "sha256": sha256_hex})
        name_check = _pi_zip_name_check(path, sha256_hex)
        pi_zip_name_checks.append(name_check)
        if not bool(name_check.get("matches_sha12", False)):
            pi_zip_name_mismatches.append(name_check)

    manifest_candidates = [
        vault_root / "manifest.lith",
        part_root / "manifest.lith",
        Path.cwd() / "manifest.lith",
    ]
    manifest_path = next((p for p in manifest_candidates if p.exists()), None)
    manifest_sha = _sha256_for_path(manifest_path) if manifest_path else ""

    proof_schema = _extract_manifest_proof_schema(manifest_path)
    required_refs = [str(item) for item in proof_schema.get("required_refs", [])]
    required_hashes = [str(item) for item in proof_schema.get("required_hashes", [])]
    host_handle = str(proof_schema.get("host_handle", "")).strip()

    missing_required_refs = [
        ref
        for ref in required_refs
        if not _proof_ref_exists(ref, vault_root, part_root)
    ]
    available_hashes = {
        "sha256:pi_zip": [
            item.get("sha256") for item in artifact_hashes if item.get("sha256")
        ],
        "sha256:manifest": [manifest_sha] if manifest_sha else [],
    }
    missing_required_hashes = [
        key for key in required_hashes if not available_hashes.get(key)
    ]
    host_has_github_gist = bool(host_handle) and (
        "github:" in host_handle
        or "gist:" in host_handle
        or "gist.github.com" in host_handle
    )

    needs: list[str] = []
    advisories: list[str] = []
    if not artifact_paths:
        needs.append("truth artifact zip (artifacts/truth/Π*.zip)")
    if manifest_path is None:
        needs.append("manifest.lith")
    if drift.get("blocked_gates"):
        needs.append("resolve blocked push-truth gates")
    if not required_refs:
        needs.append("manifest proof-schema required_refs")
    if missing_required_refs:
        needs.append(
            "proof refs missing: " + ", ".join(sorted(set(missing_required_refs)))
        )
    if not required_hashes:
        needs.append("manifest proof-schema required_hashes")
    if missing_required_hashes:
        needs.append(
            "proof hashes missing: " + ", ".join(sorted(set(missing_required_hashes)))
        )
    if pi_zip_name_mismatches:
        mismatch_preview = ", ".join(
            f"{Path(str(item.get('path', ''))).name}->{str(item.get('expected_sha12', ''))}"
            for item in pi_zip_name_mismatches[:4]
        )
        advisories.append("Pi zip filename/hash mismatch: " + mismatch_preview)
    if not host_handle:
        needs.append("manifest proof-schema host-handle")

    blocked = bool(needs)
    reasons = [item.get("reason", "unknown") for item in drift.get("blocked_gates", [])]
    if missing_required_refs:
        reasons.append("missing-proof-refs")
    if missing_required_hashes:
        reasons.append("missing-proof-hashes")
    if not host_handle:
        reasons.append("missing-host-handle")

    return {
        "ok": True,
        "gate": {
            "target": "push-truth",
            "blocked": blocked,
            "reasons": reasons,
        },
        "needs": needs,
        "advisories": advisories,
        "predicted_drifts": drift.get("active_drifts", []),
        "proof_schema": {
            "source": str(proof_schema.get("source", "")),
            "required_refs": required_refs,
            "required_hashes": required_hashes,
            "host_handle": host_handle,
            "missing_refs": missing_required_refs,
            "missing_hashes": missing_required_hashes,
        },
        "artifacts": {
            "pi_zip": [str(path) for path in artifact_paths],
            "pi_zip_hashes": artifact_hashes,
            "pi_zip_name_checks": pi_zip_name_checks,
            "pi_zip_name_mismatch_count": len(pi_zip_name_mismatches),
            "host_has_github_gist": host_has_github_gist,
            "host_handle": host_handle,
            "manifest": {
                "path": str(manifest_path) if manifest_path else "",
                "sha256": manifest_sha,
            },
        },
        "plan": [
            "Run /api/drift/scan and resolve blocked gates",
            "Generate or locate Π zip artifact",
            "Satisfy manifest proof schema refs/hashes/host-handle",
            "Bind manifest and append push-truth receipt",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _study_stability_label(score: float) -> str:
    if score >= 0.8:
        return "stable"
    if score >= 0.56:
        return "watch"
    return "unstable"


def build_study_snapshot(
    part_root: Path,
    vault_root: Path,
    *,
    queue_snapshot: dict[str, Any] | None = None,
    council_snapshot: dict[str, Any] | None = None,
    drift_payload: dict[str, Any] | None = None,
    truth_gate_blocked: bool | None = None,
    resource_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    queue = queue_snapshot if isinstance(queue_snapshot, dict) else {}
    council = council_snapshot if isinstance(council_snapshot, dict) else {}
    drift = drift_payload if isinstance(drift_payload, dict) else {}
    resource = (
        resource_snapshot
        if isinstance(resource_snapshot, dict)
        else _resource_monitor_snapshot(part_root=part_root)
    )
    if not isinstance(resource, dict):
        resource = {}

    blocked_gates = [
        item for item in drift.get("blocked_gates", []) if isinstance(item, dict)
    ]
    active_drifts = [
        item for item in drift.get("active_drifts", []) if isinstance(item, dict)
    ]
    open_questions = (
        drift.get("open_questions", {})
        if isinstance(drift.get("open_questions"), dict)
        else {}
    )
    receipts_parse = (
        drift.get("receipts_parse", {})
        if isinstance(drift.get("receipts_parse"), dict)
        else {}
    )

    blocked_gate_count = len(blocked_gates)
    active_drift_count = len(active_drifts)
    queue_pending = int(_safe_float(queue.get("pending_count", 0), 0.0))
    queue_events = int(_safe_float(queue.get("event_count", 0), 0.0))
    council_pending = int(_safe_float(council.get("pending_count", 0), 0.0))
    council_approved = int(_safe_float(council.get("approved_count", 0), 0.0))
    unresolved_questions = int(
        _safe_float(open_questions.get("unresolved_count", 0), 0.0)
    )
    resource_hot_count = len(
        [item for item in resource.get("hot_devices", []) if str(item).strip()]
    )
    resource_log_watch = (
        resource.get("log_watch", {})
        if isinstance(resource.get("log_watch"), dict)
        else {}
    )
    resource_log_error_ratio = _safe_float(
        resource_log_watch.get("error_ratio", 0.0), 0.0
    )

    truth_blocked = bool(truth_gate_blocked)
    if truth_gate_blocked is None:
        truth_blocked = blocked_gate_count > 0

    blocked_penalty = _clamp01(blocked_gate_count / 4.0) * 0.34
    drift_penalty = _clamp01(active_drift_count / 8.0) * 0.18
    queue_penalty = _clamp01(queue_pending / 8.0) * 0.2
    council_penalty = _clamp01(council_pending / 5.0) * 0.16
    truth_penalty = 0.12 if truth_blocked else 0.0
    resource_penalty = _clamp01(resource_hot_count / 3.0) * 0.08
    resource_log_penalty = _clamp01(resource_log_error_ratio) * 0.06
    score = _clamp01(
        1.0
        - blocked_penalty
        - drift_penalty
        - queue_penalty
        - council_penalty
        - truth_penalty
        - resource_penalty
        - resource_log_penalty
    )

    warnings: list[dict[str, Any]] = []
    for gate in blocked_gates[:8]:
        warnings.append(
            {
                "code": "gate.blocked",
                "severity": "high",
                "message": f"{gate.get('target', 'gate')}: {gate.get('reason', 'blocked')}",
            }
        )
    if unresolved_questions > 0:
        warnings.append(
            {
                "code": "drift.open_questions",
                "severity": "medium",
                "message": f"{unresolved_questions} open questions unresolved",
            }
        )
    if queue_pending > 0:
        warnings.append(
            {
                "code": "queue.pending",
                "severity": "medium",
                "message": f"{queue_pending} tasks pending in queue",
            }
        )
    if council_pending > 0:
        warnings.append(
            {
                "code": "council.pending",
                "severity": "medium",
                "message": f"{council_pending} council decisions awaiting closure",
            }
        )
    if resource_hot_count > 0:
        warnings.append(
            {
                "code": "runtime.resource_hot",
                "severity": "medium",
                "message": "hot resources: "
                + ", ".join(
                    [
                        str(item)
                        for item in resource.get("hot_devices", [])
                        if str(item).strip()
                    ]
                ),
            }
        )
    if resource_log_error_ratio >= 0.45:
        warnings.append(
            {
                "code": "runtime.log_error_ratio",
                "severity": "medium",
                "message": (
                    "runtime log error ratio elevated: "
                    + str(round(resource_log_error_ratio, 3))
                ),
            }
        )

    receipts_path = str(receipts_parse.get("path", "")).strip()
    receipts_within_vault = False
    if receipts_path:
        try:
            resolved = Path(receipts_path).resolve()
            vault_resolved = vault_root.resolve()
            receipts_within_vault = (
                resolved == vault_resolved or vault_resolved in resolved.parents
            )
        except OSError:
            receipts_within_vault = False

    if receipts_path and not receipts_within_vault:
        warnings.append(
            {
                "code": "runtime.receipts_path_outside_vault",
                "severity": "high",
                "message": f"receipts path not under vault root: {receipts_path}",
            }
        )

    decisions = [
        item for item in council.get("decisions", []) if isinstance(item, dict)
    ]
    decision_status_counts: dict[str, int] = {}
    for row in decisions:
        status = str(row.get("status", "unknown")).strip() or "unknown"
        decision_status_counts[status] = decision_status_counts.get(status, 0) + 1

    return {
        "ok": True,
        "record": "ημ.study-snapshot.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stability": {
            "score": round(score, 4),
            "label": _study_stability_label(score),
            "components": {
                "blocked_gate_penalty": round(blocked_penalty, 4),
                "drift_penalty": round(drift_penalty, 4),
                "queue_penalty": round(queue_penalty, 4),
                "council_penalty": round(council_penalty, 4),
                "truth_penalty": round(truth_penalty, 4),
                "resource_penalty": round(resource_penalty, 4),
                "resource_log_penalty": round(resource_log_penalty, 4),
            },
        },
        "signals": {
            "blocked_gate_count": blocked_gate_count,
            "active_drift_count": active_drift_count,
            "queue_pending_count": queue_pending,
            "queue_event_count": queue_events,
            "council_pending_count": council_pending,
            "council_approved_count": council_approved,
            "council_decision_count": int(
                _safe_float(council.get("decision_count", 0), 0.0)
            ),
            "decision_status_counts": decision_status_counts,
            "truth_gate_blocked": truth_blocked,
            "open_questions_unresolved": unresolved_questions,
            "resource_hot_count": resource_hot_count,
            "resource_log_error_ratio": round(resource_log_error_ratio, 4),
        },
        "runtime": {
            "part_root": str(part_root.resolve()),
            "vault_root": str(vault_root.resolve()),
            "receipts_path": receipts_path,
            "receipts_parse_ok": bool(receipts_parse.get("ok", False)),
            "receipts_rows": int(_safe_float(receipts_parse.get("rows", 0), 0.0)),
            "receipts_has_intent_ref": bool(
                receipts_parse.get("has_intent_ref", False)
            ),
            "receipts_path_within_vault": receipts_within_vault,
            "resource": resource,
        },
        "warnings": warnings,
        "drift": drift,
        "queue": queue,
        "council": council,
    }


def _study_snapshot_log_path(vault_root: Path) -> Path:
    return (vault_root / STUDY_SNAPSHOT_LOG_REL).resolve()


def _load_study_snapshot_events(
    vault_root: Path,
    *,
    limit: int = 16,
) -> list[dict[str, Any]]:
    safe_limit = max(
        1,
        min(STUDY_SNAPSHOT_HISTORY_LIMIT, int(limit or 16)),
    )
    log_path = _study_snapshot_log_path(vault_root)
    if not log_path.exists() or not log_path.is_file():
        return []

    try:
        stat = log_path.stat()
    except OSError:
        return []

    with _STUDY_SNAPSHOT_LOCK:
        cached_path = str(_STUDY_SNAPSHOT_CACHE.get("path", ""))
        cached_mtime = int(_STUDY_SNAPSHOT_CACHE.get("mtime_ns", 0))
        if cached_path == str(log_path) and cached_mtime == int(stat.st_mtime_ns):
            cached = _STUDY_SNAPSHOT_CACHE.get("events", [])
            if isinstance(cached, list):
                return [
                    dict(item) for item in cached[:safe_limit] if isinstance(item, dict)
                ]

        events: list[dict[str, Any]] = []
        for raw in log_path.read_text("utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(row, dict):
                events.append(row)

        events.sort(key=lambda row: str(row.get("ts", "")), reverse=True)
        _STUDY_SNAPSHOT_CACHE["path"] = str(log_path)
        _STUDY_SNAPSHOT_CACHE["mtime_ns"] = int(stat.st_mtime_ns)
        _STUDY_SNAPSHOT_CACHE["events"] = [dict(item) for item in events]
        return [dict(item) for item in events[:safe_limit]]


def _append_study_snapshot_event(vault_root: Path, event: dict[str, Any]) -> Path:
    log_path = _study_snapshot_log_path(vault_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False)
    with _STUDY_SNAPSHOT_LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        _STUDY_SNAPSHOT_CACHE["path"] = ""
        _STUDY_SNAPSHOT_CACHE["mtime_ns"] = 0
        _STUDY_SNAPSHOT_CACHE["events"] = []
    return log_path


def export_study_snapshot(
    part_root: Path,
    vault_root: Path,
    *,
    queue_snapshot: dict[str, Any],
    council_snapshot: dict[str, Any],
    drift_payload: dict[str, Any],
    truth_gate_blocked: bool | None = None,
    resource_snapshot: dict[str, Any] | None = None,
    label: str = "",
    owner: str = "Err",
    refs: list[str] | None = None,
    host: str = "127.0.0.1:8787",
    manifest: str = "manifest.lith",
) -> dict[str, Any]:
    snapshot = build_study_snapshot(
        part_root,
        vault_root,
        queue_snapshot=queue_snapshot,
        council_snapshot=council_snapshot,
        drift_payload=drift_payload,
        truth_gate_blocked=truth_gate_blocked,
        resource_snapshot=resource_snapshot,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    digest_seed = "|".join(
        [
            now_iso,
            str(label).strip(),
            str((snapshot.get("stability", {}) or {}).get("score", "0")),
            str((snapshot.get("signals", {}) or {}).get("blocked_gate_count", "0")),
            str((snapshot.get("signals", {}) or {}).get("active_drift_count", "0")),
            str((snapshot.get("signals", {}) or {}).get("queue_pending_count", "0")),
            str((snapshot.get("signals", {}) or {}).get("council_pending_count", "0")),
        ]
    )
    event_id = f"study:{sha1(digest_seed.encode('utf-8')).hexdigest()[:16]}"
    event = {
        "v": STUDY_EVENT_VERSION,
        "ts": now_iso,
        "op": "export",
        "id": event_id,
        "label": str(label).strip(),
        "owner": str(owner).strip() or "Err",
        "snapshot": snapshot,
    }
    log_path = _append_study_snapshot_event(vault_root, event)
    refs_rows = [
        str(log_path),
        "study:export",
        f"study:{event_id}",
        ".opencode/promptdb/contracts/receipts.v2.contract.lisp",
        *([str(item).strip() for item in refs or [] if str(item).strip()]),
    ]
    _append_receipt_line(
        _ensure_receipts_log_path(vault_root, part_root),
        kind=":decision",
        origin="study",
        owner=str(owner).strip() or "Err",
        dod="exported study snapshot evidence",
        pi="part64-runtime-system",
        host=host,
        manifest=manifest,
        refs=refs_rows,
        note=("" if not str(label).strip() else f"label={str(label).strip()}"),
    )
    latest = _load_study_snapshot_events(vault_root, limit=8)
    history_count = len(
        _load_study_snapshot_events(
            vault_root,
            limit=STUDY_SNAPSHOT_HISTORY_LIMIT,
        )
    )
    return {
        "ok": True,
        "event": event,
        "history": {
            "record": "ημ.study-history.v1",
            "path": str(log_path),
            "count": history_count,
            "latest": latest,
        },
    }


def build_named_field_overlays(
    entity_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for entity in entity_manifest:
        key = str(entity.get("id", "")).strip()
        if key:
            by_id[key] = entity

    overlays: list[dict[str, Any]] = []
    for idx, field_id in enumerate(CANONICAL_NAMED_FIELD_IDS):
        item = by_id.get(field_id)
        if item is None:
            continue

        hue = int(item.get("hue", 200))
        overlays.append(
            {
                "id": field_id,
                "en": str(item.get("en", field_id.replace("_", " ").title())),
                "ja": str(item.get("ja", "")),
                "type": str(item.get("type", "flow")),
                "x": float(item.get("x", 0.5)),
                "y": float(item.get("y", 0.5)),
                "freq": float(item.get("freq", 220.0)),
                "hue": hue,
                "gradient": {
                    "mode": "radial",
                    "radius": round(0.2 + (idx % 3) * 0.035, 3),
                    "stops": [
                        {
                            "offset": 0.0,
                            "color": f"hsla({hue}, 88%, 74%, 0.36)",
                        },
                        {
                            "offset": 0.52,
                            "color": f"hsla({hue}, 76%, 58%, 0.2)",
                        },
                        {
                            "offset": 1.0,
                            "color": f"hsla({(hue + 28) % 360}, 72%, 44%, 0.0)",
                        },
                    ],
                },
                "motion": {
                    "drift_hz": round(0.07 + idx * 0.013, 3),
                    "wobble_px": 5 + (idx % 4) * 3,
                },
            }
        )

    return overlays


def projection_perspective_options() -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for key in ("hybrid", "causal-time", "swimlanes"):
        meta = PROJECTION_PERSPECTIVES[key]
        options.append(
            {
                "id": key,
                "symbol": str(meta.get("id", f"perspective.{key}")),
                "name": str(meta.get("name", key.title())),
                "merge": str(meta.get("merge", key)),
                "description": str(meta.get("description", "")),
                "default": key == PROJECTION_DEFAULT_PERSPECTIVE,
            }
        )
    return options


def normalize_projection_perspective(raw: str | None) -> str:
    text = str(raw or "").strip().lower().replace("_", "-")
    aliases = {
        "": PROJECTION_DEFAULT_PERSPECTIVE,
        "hybrid": "hybrid",
        "causal": "causal-time",
        "causal-time": "causal-time",
        "causaltime": "causal-time",
        "swimlane": "swimlanes",
        "swimlanes": "swimlanes",
        "lanes": "swimlanes",
    }
    resolved = aliases.get(text, text)
    if resolved not in PROJECTION_PERSPECTIVES:
        return PROJECTION_DEFAULT_PERSPECTIVE
    return resolved


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _projection_source_influence(
    simulation: dict[str, Any] | None,
    catalog: dict[str, Any],
    influence_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(influence_snapshot, dict) and influence_snapshot:
        return influence_snapshot
    if isinstance(simulation, dict):
        dynamics = simulation.get("presence_dynamics")
        if isinstance(dynamics, dict) and dynamics:
            return dynamics
    runtime = catalog.get("presence_runtime")
    if isinstance(runtime, dict):
        return runtime
    return {}


def _projection_presence_impacts(
    simulation: dict[str, Any] | None,
    influence: dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(simulation, dict):
        dynamics = simulation.get("presence_dynamics")
        if isinstance(dynamics, dict):
            rows = dynamics.get("presence_impacts")
            if isinstance(rows, list):
                return [item for item in rows if isinstance(item, dict)]

    clicks = int(
        _safe_float(
            influence.get("click_events", influence.get("clicks_45s", 0)),
            0.0,
        )
    )
    files = int(
        _safe_float(
            influence.get("file_events", influence.get("file_changes_120s", 0)),
            0.0,
        )
    )
    click_ratio = _clamp01(clicks / 18.0)
    file_ratio = _clamp01(files / 24.0)
    impacts: list[dict[str, Any]] = []
    by_id = {str(item.get("id", "")): item for item in ENTITY_MANIFEST}
    for idx, presence_id in enumerate(CANONICAL_NAMED_FIELD_IDS):
        meta = by_id.get(presence_id, {})
        file_signal = _clamp01(file_ratio * (0.55 + (idx % 3) * 0.12))
        click_signal = _clamp01(click_ratio * (0.62 + (idx % 4) * 0.08))
        world_signal = _clamp01((file_signal * 0.58) + (click_signal * 0.42))
        impacts.append(
            {
                "id": presence_id,
                "en": str(meta.get("en", presence_id.replace("_", " ").title())),
                "ja": str(meta.get("ja", "")),
                "affected_by": {
                    "files": round(file_signal, 4),
                    "clicks": round(click_signal, 4),
                },
                "affects": {
                    "world": round(world_signal, 4),
                    "ledger": round(_clamp01(world_signal * 0.86), 4),
                },
                "notes_en": "Fallback projection impact synthesized from runtime pressure.",
                "notes_ja": "ランタイム圧から合成したフォールバック影響。",
            }
        )
    return impacts


def _build_projection_field_snapshot(
    catalog: dict[str, Any],
    simulation: dict[str, Any] | None,
    *,
    perspective: str,
    queue_snapshot: dict[str, Any] | None,
    influence_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    ts_ms = int(time.time() * 1000)
    perspective_key = normalize_projection_perspective(perspective)
    queue = queue_snapshot or catalog.get("task_queue", {})
    influence = _projection_source_influence(simulation, catalog, influence_snapshot)

    counts = catalog.get("counts", {})
    item_count = len(catalog.get("items", []))
    promptdb = catalog.get("promptdb", {})

    audio_ratio = _clamp01(_safe_float(counts.get("audio", 0), 0.0) / 24.0)
    image_ratio = _clamp01(_safe_float(counts.get("image", 0), 0.0) / 220.0)
    text_ratio = _clamp01(_safe_float(counts.get("text", 0), 0.0) / 90.0)
    promptdb_ratio = _clamp01(_safe_float(promptdb.get("packet_count", 0), 0.0) / 120.0)

    clicks = int(
        _safe_float(
            influence.get("click_events", influence.get("clicks_45s", 0)),
            0.0,
        )
    )
    files = int(
        _safe_float(
            influence.get("file_events", influence.get("file_changes_120s", 0)),
            0.0,
        )
    )
    click_ratio = _clamp01(clicks / 18.0)
    file_ratio = _clamp01(files / 24.0)

    pending_count = int(_safe_float(queue.get("pending_count", 0), 0.0))
    queue_events = int(_safe_float(queue.get("event_count", 0), 0.0))
    queue_pending_ratio = _clamp01(pending_count / 8.0)
    queue_events_ratio = _clamp01(queue_events / 48.0)

    fork_tax = influence.get("fork_tax", {}) if isinstance(influence, dict) else {}
    paid_ratio = _clamp01(_safe_float(fork_tax.get("paid_ratio", 0.5), 0.5))
    balance_raw = max(0.0, _safe_float(fork_tax.get("balance", 0.0), 0.0))
    debt_raw = max(0.0, _safe_float(fork_tax.get("debt", balance_raw), balance_raw))
    fork_balance_ratio = _clamp01(
        1.0
        - (
            balance_raw
            / max(1.0, debt_raw + _safe_float(fork_tax.get("paid", 0.0), 0.0))
        )
    )

    witness_continuity = _safe_float(
        (
            ((simulation or {}).get("presence_dynamics") or {}).get("witness_thread")
            or {}
        ).get("continuity_index", click_ratio),
        click_ratio,
    )
    witness_ratio = _clamp01(max(click_ratio, witness_continuity))

    drift_index = _clamp01((file_ratio * 0.58) + (queue_pending_ratio * 0.42))
    proof_gap = _clamp01((1.0 - paid_ratio) * 0.64 + drift_index * 0.36)
    artifact_flux = _clamp01(
        (audio_ratio * 0.49)
        + (image_ratio * 0.11)
        + (file_ratio * 0.24)
        + (promptdb_ratio * 0.16)
    )
    coherence_focus = _clamp01(
        1.0
        - (abs(audio_ratio - text_ratio) * 0.74)
        - (drift_index * 0.24)
        + (witness_ratio * 0.14)
    )
    curiosity_drive = _clamp01(
        (text_ratio * 0.26)
        + (promptdb_ratio * 0.42)
        + (click_ratio * 0.2)
        + (artifact_flux * 0.12)
    )
    gate_pressure = _clamp01(
        (drift_index * 0.48) + (proof_gap * 0.3) + (queue_pending_ratio * 0.22)
    )
    council_heat = _clamp01(
        (queue_events_ratio * 0.56)
        + (queue_pending_ratio * 0.29)
        + (gate_pressure * 0.15)
    )

    if perspective_key == "causal-time":
        witness_ratio = _clamp01((witness_ratio * 0.76) + (gate_pressure * 0.24))
        drift_index = _clamp01((drift_index * 0.74) + (gate_pressure * 0.26))
    elif perspective_key == "swimlanes":
        council_heat = _clamp01((council_heat * 0.76) + (queue_pending_ratio * 0.24))
        coherence_focus = _clamp01((coherence_focus * 0.82) + (curiosity_drive * 0.18))

    impacts = _projection_presence_impacts(simulation, influence)
    applied_reiso = [
        f"rei.{str(item.get('id', 'presence')).strip()}.impulse"
        for item in sorted(
            impacts,
            key=lambda row: _safe_float(
                (row.get("affects", {}) or {}).get("world", 0.0), 0.0
            ),
            reverse=True,
        )[:4]
        if str(item.get("id", "")).strip()
    ]

    world_tick = int(
        _safe_float(((simulation or {}).get("world") or {}).get("tick", 0), 0.0)
    )
    witness_lineage = (
        ((simulation or {}).get("presence_dynamics") or {}).get("witness_thread") or {}
    ).get("lineage", []) or []
    vectors = {
        "f1": {
            "energy": round(artifact_flux, 4),
            "audio_ratio": round(audio_ratio, 4),
            "file_ratio": round(file_ratio, 4),
            "mix_sources": int(
                _safe_float((catalog.get("mix") or {}).get("sources", 0), 0.0)
            ),
        },
        "f2": {
            "energy": round(witness_ratio, 4),
            "click_ratio": round(click_ratio, 4),
            "continuity_index": round(witness_ratio, 4),
            "lineage_links": len(witness_lineage),
        },
        "f3": {
            "energy": round(coherence_focus, 4),
            "balance_ratio": round(_clamp01(1.0 - abs(audio_ratio - text_ratio)), 4),
            "promptdb_ratio": round(promptdb_ratio, 4),
            "catalog_entropy": round(_clamp01(item_count / 400.0), 4),
        },
        "f4": {
            "energy": round(drift_index, 4),
            "file_ratio": round(file_ratio, 4),
            "queue_pending_ratio": round(queue_pending_ratio, 4),
            "drift_index": round(drift_index, 4),
        },
        "f5": {
            "energy": round(_clamp01(1.0 - paid_ratio), 4),
            "paid_ratio": round(paid_ratio, 4),
            "balance_ratio": round(fork_balance_ratio, 4),
            "debt": round(debt_raw, 4),
        },
        "f6": {
            "energy": round(curiosity_drive, 4),
            "text_ratio": round(text_ratio, 4),
            "promptdb_ratio": round(promptdb_ratio, 4),
            "interaction_ratio": round(click_ratio, 4),
        },
        "f7": {
            "energy": round(gate_pressure, 4),
            "queue_ratio": round(queue_pending_ratio, 4),
            "drift_index": round(drift_index, 4),
            "proof_gap": round(proof_gap, 4),
        },
        "f8": {
            "energy": round(council_heat, 4),
            "queue_events_ratio": round(queue_events_ratio, 4),
            "pending_ratio": round(queue_pending_ratio, 4),
            "decision_load": round(_clamp01((queue_events + pending_count) / 52.0), 4),
        },
    }

    merge_mode = {
        "hybrid": "hybrid",
        "causal-time": "causal",
        "swimlanes": "wallclock",
    }.get(perspective_key, "hybrid")

    snapshot_seed = (
        f"{perspective_key}|{ts_ms}|{item_count}|{queue_events}|{pending_count}"
    )
    snapshot_id = (
        f"field.snapshot.{sha1(snapshot_seed.encode('utf-8')).hexdigest()[:12]}"
    )
    return {
        "record": "場/snapshot.v1",
        "id": snapshot_id,
        "ts": ts_ms,
        "vectors": vectors,
        "applied_reiso": applied_reiso,
        "merge_mode": merge_mode,
        "ticks": [
            f"tick.world.{world_tick}",
            f"tick.queue.{queue_events}",
            f"tick.catalog.{item_count}",
        ],
    }


def _projection_field_levels(field_snapshot: dict[str, Any]) -> dict[str, float]:
    vectors = field_snapshot.get("vectors", {})
    levels: dict[str, float] = {}
    for field_name in ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]:
        field_row = vectors.get(field_name, {})
        levels[field_name] = _clamp01(_safe_float(field_row.get("energy", 0.0), 0.0))
    return levels


def _projection_dominant_field(
    field_levels: dict[str, float],
    field_bindings: dict[str, Any],
) -> tuple[str, float]:
    if not field_bindings:
        return "f3", field_levels.get("f3", 0.0)
    best_field = "f3"
    best_score = -1.0
    for key, weight in field_bindings.items():
        score = _clamp01(_safe_float(weight, 0.0)) * field_levels.get(str(key), 0.0)
        if score > best_score:
            best_score = score
            best_field = str(key)
    return best_field, field_levels.get(best_field, 0.0)


def _build_projection_coherence_state(
    field_snapshot: dict[str, Any],
    *,
    perspective: str,
) -> dict[str, Any]:
    levels = _projection_field_levels(field_snapshot)
    ts_ms = int(time.time() * 1000)
    centroid = {
        "x": round((levels["f3"] * 0.6) + (levels["f6"] * 0.4), 4),
        "y": round((levels["f2"] * 0.52) + (levels["f4"] * 0.48), 4),
        "z": round((levels["f1"] * 0.45) + (levels["f7"] * 0.55), 4),
    }
    tension = _clamp01(
        (levels["f2"] * 0.4) + (levels["f4"] * 0.25) + (levels["f7"] * 0.35)
    )
    drift = _clamp01(
        (levels["f4"] * 0.45) + (levels["f5"] * 0.25) + (levels["f8"] * 0.3)
    )
    entropy = _clamp01((1.0 - levels["f3"]) * 0.58 + levels["f8"] * 0.42)

    perspective_scores = {
        "hybrid": _clamp01(
            (levels["f1"] * 0.28) + (levels["f3"] * 0.34) + (levels["f6"] * 0.38)
        ),
        "causal-time": _clamp01(
            (levels["f2"] * 0.37) + (levels["f4"] * 0.28) + (levels["f7"] * 0.35)
        ),
        "swimlanes": _clamp01(
            (levels["f8"] * 0.42) + (levels["f4"] * 0.24) + (levels["f5"] * 0.34)
        ),
    }
    dominant_perspective = max(
        perspective_scores.keys(),
        key=lambda key: perspective_scores.get(key, 0.0),
    )
    coherence_seed = f"{field_snapshot.get('id', '')}|{perspective}|{ts_ms}"
    return {
        "record": "心/state.v1",
        "id": f"coherence.{sha1(coherence_seed.encode('utf-8')).hexdigest()[:12]}",
        "ts": ts_ms,
        "centroid": centroid,
        "tension": round(tension, 4),
        "drift": round(drift, 4),
        "entropy": round(entropy, 4),
        "dominant_perspective": dominant_perspective,
        "perspective_score": round(
            perspective_scores.get(normalize_projection_perspective(perspective), 0.0),
            4,
        ),
    }


def _build_projection_element_states(
    field_snapshot: dict[str, Any],
    coherence_state: dict[str, Any],
    *,
    perspective: str,
    simulation: dict[str, Any] | None,
    influence_snapshot: dict[str, Any] | None,
    catalog: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    perspective_key = normalize_projection_perspective(perspective)
    levels = _projection_field_levels(field_snapshot)
    influence = influence_snapshot or {}

    impact_rows = _projection_presence_impacts(simulation, influence)
    impact_map = {str(item.get("id", "")): item for item in impact_rows}
    ts_ms = int(time.time() * 1000)
    clamps = {
        "record": "映/clamp.v1",
        "min_area": 0.1,
        "max_area": 0.36,
        "max_pulse": 0.92,
        "decay_half_life": {
            "mass_ms": 2400,
            "priority_ms": 1800,
            "pulse_ms": 1200,
        },
    }

    elements: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []

    logical_graph: dict[str, Any] = {}
    if isinstance(simulation, dict):
        logical_graph = simulation.get("logical_graph", {})
    if not logical_graph and isinstance(catalog, dict):
        logical_graph = catalog.get("logical_graph", {})
    if not isinstance(logical_graph, dict):
        logical_graph = {}

    graph_nodes = (
        logical_graph.get("nodes", []) if isinstance(logical_graph, dict) else []
    )

    test_nodes = [
        node
        for node in graph_nodes
        if isinstance(node, dict) and str(node.get("kind", "")) == "test"
    ]

    for template in PROJECTION_ELEMENTS:
        element_id = str(template.get("id", "")).strip()
        if not element_id:
            continue
        field_bindings = dict(template.get("field_bindings", {}))
        binding_sum = sum(
            max(0.0, _safe_float(weight, 0.0)) for weight in field_bindings.values()
        )
        if binding_sum <= 0:
            field_signal = levels.get("f3", 0.0)
        else:
            weighted = 0.0
            for field_key, weight in field_bindings.items():
                weighted += levels.get(str(field_key), 0.0) * max(
                    0.0, _safe_float(weight, 0.0)
                )
            field_signal = _clamp01(weighted / binding_sum)

        presence_id = str(template.get("presence", "")).strip()
        impact = impact_map.get(presence_id, {}) if presence_id else {}
        affected = impact.get("affected_by", {}) if isinstance(impact, dict) else {}
        affects = impact.get("affects", {}) if isinstance(impact, dict) else {}
        presence_signal = _clamp01(
            (_safe_float(affects.get("world", 0.0), 0.0) * 0.6)
            + (_safe_float(affected.get("files", 0.0), 0.0) * 0.23)
            + (_safe_float(affected.get("clicks", 0.0), 0.0) * 0.17)
        )
        queue_signal = levels.get("f8", 0.0)
        causal_signal = levels.get("f7", 0.0)

        if perspective_key == "causal-time":
            field_signal = _clamp01(
                (field_signal * 0.68)
                + (levels.get("f2", 0.0) * 0.19)
                + (causal_signal * 0.13)
            )
        elif perspective_key == "swimlanes":
            field_signal = _clamp01(
                (field_signal * 0.72)
                + (queue_signal * 0.2)
                + (levels.get("f4", 0.0) * 0.08)
            )

        kind = str(template.get("kind", "panel"))
        mass = _clamp01(
            (field_signal * 0.56) + (presence_signal * 0.24) + (queue_signal * 0.2)
        )
        priority = _clamp01(
            (mass * 0.67) + (causal_signal * 0.23) + (levels.get("f2", 0.0) * 0.1)
        )
        area = 0.1 + (mass * 0.24) + (priority * 0.04)
        if kind == "chat-lens":
            area += 0.05
            priority = _clamp01(priority + 0.08)
        if "governance" in (template.get("tags") or []):
            priority = _clamp01(priority + (levels.get("f7", 0.0) * 0.1))
        if kind == "test":
            priority = _clamp01(priority + 0.12)
            mass = _clamp01(mass + 0.15)
            if str(template.get("status", "")) == "failed":
                pulse = 0.95
                mass = _clamp01(mass + 0.2)
        if perspective_key == "swimlanes":
            area *= 0.88
        area = max(clamps["min_area"], min(clamps["max_area"], area))

        opacity = _clamp01(0.42 + (mass * 0.46) + (presence_signal * 0.12))
        pulse = min(
            clamps["max_pulse"],
            _clamp01(
                (field_signal * 0.42) + (presence_signal * 0.31) + (queue_signal * 0.27)
            ),
        )

        dominant_field, dominant_level = _projection_dominant_field(
            levels, field_bindings
        )
        explain = {
            "field_signal": round(field_signal, 4),
            "presence_signal": round(presence_signal, 4),
            "queue_signal": round(queue_signal, 4),
            "causal_signal": round(causal_signal, 4),
            "dominant_field": dominant_field,
            "dominant_level": round(dominant_level, 4),
            "field_bindings": {
                str(key): round(_safe_float(value, 0.0), 4)
                for key, value in field_bindings.items()
            },
            "reason_en": (
                f"{template.get('title', 'Element')} expands on {dominant_field} "
                f"under {perspective_key} perspective coupling."
            ),
            "reason_ja": "投影は場の優勢軸に従って拡縮される。",
            "coherence_tension": round(
                _safe_float(coherence_state.get("tension", 0.0), 0.0), 4
            ),
        }
        sources = [
            str(field_snapshot.get("id", "")),
            str(coherence_state.get("id", "")),
            f"perspective:{perspective_key}",
        ]
        if presence_id:
            sources.append(f"presence:{presence_id}")

        elements.append(
            {
                "record": "映/element.v1",
                "id": element_id,
                "kind": kind,
                "title": str(template.get("title", element_id)),
                "binds_to": list(template.get("binds_to", [])),
                "field_bindings": {
                    str(key): round(_safe_float(value, 0.0), 4)
                    for key, value in field_bindings.items()
                },
                "presence": presence_id,
                "tags": list(template.get("tags", [])),
                "lane": str(template.get("lane", "voice")),
                "memory_scope": str(template.get("memory_scope", "shared")),
            }
        )
        states.append(
            {
                "record": "映/state.v1",
                "element_id": element_id,
                "ts": ts_ms,
                "mass": round(mass, 4),
                "priority": round(priority, 4),
                "area": round(area, 4),
                "opacity": round(opacity, 4),
                "pulse": round(pulse, 4),
                "sources": sources,
                "explain": explain,
            }
        )

    for test_node in test_nodes:
        test_id = str(test_node.get("id", ""))
        label = str(test_node.get("label", "Test"))
        status = str(test_node.get("status", "failed"))
        if not test_id:
            continue

        element_id = f"ui:test:{test_id}"
        kind = "test"
        lane = "council" if perspective_key == "causal-time" else "senses"

        # High impact for failing tests
        field_signal = levels.get("f7", 0.0)  # Gates of Truth
        presence_signal = 0.8
        queue_signal = 0.5

        mass = _clamp01(0.7 + (field_signal * 0.2))
        priority = _clamp01(0.85 + (field_signal * 0.1))
        area = 0.18
        opacity = 0.95
        pulse = 0.95

        sources = [str(field_snapshot.get("id", "")), test_id]

        explain = {
            "reason_en": f"Active test entity: {label} ({status})",
            "reason_ja": f"アクティブなテスト実体: {label} ({status})",
            "status": status,
        }

        elements.append(
            {
                "record": "映/element.v1",
                "id": element_id,
                "kind": kind,
                "title": label,
                "glyph": "試",
                "status": status,
                "lane": lane,
                "tags": ["test", "quality", "signal"],
                "binds_to": [],
                "field_bindings": {"f7": 0.8, "f4": 0.2},
            }
        )

        states.append(
            {
                "record": "映/state.v1",
                "element_id": element_id,
                "ts": ts_ms,
                "mass": round(mass, 4),
                "priority": round(priority, 4),
                "area": round(area, 4),
                "opacity": round(opacity, 4),
                "pulse": round(pulse, 4),
                "sources": sources,
                "explain": explain,
            }
        )

    return elements, states, clamps


def _build_projection_rects_swimlanes(
    states: list[dict[str, Any]],
    elements: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    state_map = {str(state.get("element_id", "")): state for state in states}
    lane_order = ["senses", "voice", "memory", "council"]
    lane_to_elements: dict[str, list[dict[str, Any]]] = {
        lane: [] for lane in lane_order
    }
    for element in elements:
        lane = str(element.get("lane", "voice"))
        if lane not in lane_to_elements:
            lane_to_elements[lane] = []
            lane_order.append(lane)
        lane_to_elements[lane].append(element)

    lane_count = max(1, len(lane_order))
    lane_width = 1.0 / lane_count
    rects: dict[str, dict[str, float]] = {}
    for lane_index, lane in enumerate(lane_order):
        lane_elements = lane_to_elements.get(lane, [])
        lane_elements.sort(
            key=lambda element: _safe_float(
                (state_map.get(str(element.get("id", "")), {}) or {}).get(
                    "priority", 0.0
                ),
                0.0,
            ),
            reverse=True,
        )
        y_cursor = 0.0
        for element in lane_elements:
            element_id = str(element.get("id", ""))
            state = state_map.get(element_id, {})
            height = max(
                0.12, min(0.38, _safe_float(state.get("area", 0.12), 0.12) * 1.85)
            )
            if y_cursor + height > 1.0:
                height = max(0.08, 1.0 - y_cursor)
            if height <= 0.0:
                break
            rects[element_id] = {
                "x": round((lane_index * lane_width) + 0.008, 4),
                "y": round(min(0.98, y_cursor + 0.01), 4),
                "w": round(max(0.05, lane_width - 0.016), 4),
                "h": round(max(0.08, height - 0.018), 4),
            }
            y_cursor += height
    return rects


def _build_projection_rects_grid(
    states: list[dict[str, Any]],
    *,
    perspective: str,
) -> dict[str, dict[str, float]]:
    if perspective == "causal-time":
        ordered = sorted(
            states,
            key=lambda state: (
                _safe_float(
                    (state.get("explain", {}) or {}).get("causal_signal", 0.0), 0.0
                ),
                _safe_float(state.get("priority", 0.0), 0.0),
            ),
            reverse=True,
        )
    else:
        ordered = sorted(
            states,
            key=lambda state: _safe_float(state.get("priority", 0.0), 0.0),
            reverse=True,
        )

    placements: list[dict[str, Any]] = []
    cursor_x = 0
    cursor_y = 0
    row_height = 0

    for state in ordered:
        area = _safe_float(state.get("area", 0.1), 0.1)
        mass = _safe_float(state.get("mass", 0.0), 0.0)
        pulse = _safe_float(state.get("pulse", 0.0), 0.0)
        width_units = max(3, min(12, int(round(area * 22))))
        height_units = max(2, min(5, int(round(1 + (mass * 2.4) + (pulse * 1.3)))))
        if cursor_x + width_units > 12:
            cursor_y += row_height
            cursor_x = 0
            row_height = 0
        placements.append(
            {
                "id": str(state.get("element_id", "")),
                "x": cursor_x,
                "y": cursor_y,
                "w": width_units,
                "h": height_units,
            }
        )
        cursor_x += width_units
        row_height = max(row_height, height_units)

    total_rows = max(1, cursor_y + row_height)
    rects: dict[str, dict[str, float]] = {}
    for placement in placements:
        element_id = placement["id"]
        if not element_id:
            continue
        rects[element_id] = {
            "x": round(placement["x"] / 12.0, 4),
            "y": round(placement["y"] / total_rows, 4),
            "w": round(placement["w"] / 12.0, 4),
            "h": round(placement["h"] / total_rows, 4),
        }
    return rects


def _build_projection_layout(
    elements: list[dict[str, Any]],
    states: list[dict[str, Any]],
    *,
    perspective: str,
    clamps: dict[str, Any],
) -> dict[str, Any]:
    ts_ms = int(time.time() * 1000)
    perspective_key = normalize_projection_perspective(perspective)
    if perspective_key == "swimlanes":
        rects = _build_projection_rects_swimlanes(states, elements)
    else:
        rects = _build_projection_rects_grid(states, perspective=perspective_key)

    for element in elements:
        element_id = str(element.get("id", ""))
        if not element_id or element_id in rects:
            continue
        rects[element_id] = {"x": 0.0, "y": 0.0, "w": 0.25, "h": 0.2}

    layout_seed = f"layout|{perspective_key}|{ts_ms}|{len(elements)}"
    return {
        "record": "映/layout.v1",
        "id": f"layout.{sha1(layout_seed.encode('utf-8')).hexdigest()[:12]}",
        "ts": ts_ms,
        "perspective": perspective_key,
        "elements": [str(element.get("id", "")) for element in elements],
        "rects": rects,
        "states": states,
        "clamps": clamps,
        "notes": "Derived projection from field vectors + presence impacts + queue pressure.",
    }


def _build_projection_chat_sessions(
    elements: list[dict[str, Any]],
    states: list[dict[str, Any]],
    *,
    perspective: str,
) -> list[dict[str, Any]]:
    ts_ms = int(time.time() * 1000)
    perspective_key = normalize_projection_perspective(perspective)
    state_map = {str(state.get("element_id", "")): state for state in states}
    sessions: list[dict[str, Any]] = []
    for element in elements:
        if str(element.get("kind", "")) != "chat-lens":
            continue
        element_id = str(element.get("id", ""))
        if not element_id:
            continue
        presence = str(element.get("presence", "witness_thread"))
        state = state_map.get(element_id, {})
        mass = _safe_float(state.get("mass", 0.0), 0.0)
        if perspective_key == "causal-time":
            memory_scope = "council"
        elif perspective_key == "swimlanes":
            memory_scope = "local"
        else:
            memory_scope = str(element.get("memory_scope", "shared"))
        sessions.append(
            {
                "record": "映/chat-session.v1",
                "id": f"chat-session:{presence}",
                "ts": ts_ms,
                "presence": presence,
                "lens_element": element_id,
                "field_bindings": dict(element.get("field_bindings", {})),
                "memory_scope": memory_scope,
                "tags": ["chat", "field-bound", perspective_key],
                "status": "active" if mass >= 0.42 else "listening",
            }
        )
    return sessions


def _build_projection_vector_view(
    field_snapshot: dict[str, Any],
    elements: list[dict[str, Any]],
    states: list[dict[str, Any]],
    *,
    perspective: str,
) -> dict[str, Any]:
    ts_ms = int(time.time() * 1000)
    perspective_key = normalize_projection_perspective(perspective)
    mode = {
        "hybrid": "axes",
        "causal-time": "barycentric-slice",
        "swimlanes": "cluster",
    }.get(perspective_key, "axes")
    state_by_id = {str(item.get("element_id", "")): item for item in states}
    ranked_elements = sorted(
        elements,
        key=lambda element: _safe_float(
            (state_by_id.get(str(element.get("id", "")), {}) or {}).get(
                "priority", 0.0
            ),
            0.0,
        ),
        reverse=True,
    )
    overlay_presences = [
        str(item.get("presence", ""))
        for item in ranked_elements
        if str(item.get("presence", "")).strip()
    ][:4]
    return {
        "record": "映/vector-view.v1",
        "id": f"vector-view:{perspective_key}",
        "ts": ts_ms,
        "field_snapshot": str(field_snapshot.get("id", "")),
        "mode": mode,
        "axes": {
            "x": "f3.coherence_focus",
            "y": "f2.witness_tension",
            "z": "f7.gate_pressure",
        },
        "overlay_reiso": list(field_snapshot.get("applied_reiso", []))[:6],
        "overlay_presences": overlay_presences,
        "show_causality": perspective_key != "swimlanes",
    }


def _build_projection_tick_view(
    field_snapshot: dict[str, Any],
    queue_snapshot: dict[str, Any],
    *,
    perspective: str,
) -> dict[str, Any]:
    ts_ms = int(time.time() * 1000)
    perspective_key = normalize_projection_perspective(perspective)
    merge = PROJECTION_PERSPECTIVES[perspective_key]["merge"]
    return {
        "record": "映/tick-view.v1",
        "id": f"tick-view:{perspective_key}",
        "ts": ts_ms,
        "sources": [
            *list(field_snapshot.get("ticks", [])),
            f"queue.pending.{int(_safe_float(queue_snapshot.get('pending_count', 0), 0.0))}",
            f"queue.events.{int(_safe_float(queue_snapshot.get('event_count', 0), 0.0))}",
        ],
        "window": {
            "lookback_seconds": 120,
            "sample_ms": int(max(50, SIM_TICK_SECONDS * 1000)),
        },
        "show_causal": perspective_key != "hybrid",
        "merge": merge,
    }


def build_ui_projection(
    catalog: dict[str, Any],
    simulation: dict[str, Any] | None = None,
    *,
    perspective: str = PROJECTION_DEFAULT_PERSPECTIVE,
    queue_snapshot: dict[str, Any] | None = None,
    influence_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    perspective_key = normalize_projection_perspective(perspective)
    queue = queue_snapshot or catalog.get("task_queue", {})
    influence = _projection_source_influence(simulation, catalog, influence_snapshot)
    field_snapshot = _build_projection_field_snapshot(
        catalog,
        simulation,
        perspective=perspective_key,
        queue_snapshot=queue,
        influence_snapshot=influence,
    )
    coherence_state = _build_projection_coherence_state(
        field_snapshot,
        perspective=perspective_key,
    )
    elements, states, clamps = _build_projection_element_states(
        field_snapshot,
        coherence_state,
        perspective=perspective_key,
        simulation=simulation,
        influence_snapshot=influence,
        catalog=catalog,
    )
    layout = _build_projection_layout(
        elements,
        states,
        perspective=perspective_key,
        clamps=clamps,
    )
    chat_sessions = _build_projection_chat_sessions(
        elements,
        states,
        perspective=perspective_key,
    )
    vector_view = _build_projection_vector_view(
        field_snapshot,
        elements,
        states,
        perspective=perspective_key,
    )
    tick_view = _build_projection_tick_view(
        field_snapshot,
        queue,
        perspective=perspective_key,
    )

    return {
        "record": "家_映.v1",
        "contract": "家_映.v1",
        "ts": int(time.time() * 1000),
        "perspective": perspective_key,
        "default_perspective": PROJECTION_DEFAULT_PERSPECTIVE,
        "perspectives": projection_perspective_options(),
        "field_schemas": PROJECTION_FIELD_SCHEMAS,
        "field_snapshot": field_snapshot,
        "coherence": coherence_state,
        "elements": elements,
        "states": states,
        "layout": layout,
        "chat_sessions": chat_sessions,
        "vector_view": vector_view,
        "tick_view": tick_view,
        "queue": {
            "pending_count": int(_safe_float(queue.get("pending_count", 0), 0.0)),
            "event_count": int(_safe_float(queue.get("event_count", 0), 0.0)),
        },
    }


def attach_ui_projection(
    catalog: dict[str, Any],
    *,
    perspective: str,
    simulation: dict[str, Any] | None = None,
    queue_snapshot: dict[str, Any] | None = None,
    influence_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    projection = build_ui_projection(
        catalog,
        simulation,
        perspective=perspective,
        queue_snapshot=queue_snapshot,
        influence_snapshot=influence_snapshot,
    )
    catalog["ui_default_perspective"] = PROJECTION_DEFAULT_PERSPECTIVE
    catalog["ui_perspectives"] = projection_perspective_options()
    catalog["ui_projection"] = projection
    return projection


def _mix_fingerprint(catalog: dict[str, Any]) -> str:
    rows: list[str] = []
    for item in catalog.get("items", []):
        rel_path = str(item.get("rel_path", ""))
        if rel_path.lower().endswith(".wav"):
            rows.append(
                "|".join(
                    [
                        rel_path,
                        str(item.get("bytes", 0)),
                        str(item.get("mtime_utc", "")),
                    ]
                )
            )
    rows.sort()
    return sha1("\n".join(rows).encode("utf-8")).hexdigest()


def _collect_mix_sources(catalog: dict[str, Any], vault_root: Path) -> list[Path]:
    paths: list[Path] = []
    for item in catalog.get("items", []):
        rel_path = str(item.get("rel_path", ""))
        if not rel_path.lower().endswith(".wav"):
            continue
        candidate = (vault_root / rel_path).resolve()
        if candidate.exists() and candidate.is_file():
            paths.append(candidate)
    return paths


def _mix_wav_sources(sources: list[Path]) -> tuple[bytes, dict[str, Any]]:
    if not sources:
        return b"", {"sources": 0, "sample_rate": 0, "duration_seconds": 0.0}

    sample_rate = 44100
    clips: list[tuple[array, int]] = []
    max_frames = 0

    for src in sources:
        with wave.open(str(src), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            if sampwidth != 2:
                continue
            if channels not in (1, 2):
                continue

            frames_raw = wf.readframes(wf.getnframes())
            pcm = array("h")
            pcm.frombytes(frames_raw)
            frames = len(pcm) // channels
            if frames == 0:
                continue

            sample_rate = framerate
            clips.append((pcm, channels))
            if frames > max_frames:
                max_frames = frames

    if not clips or max_frames == 0:
        return b"", {"sources": 0, "sample_rate": 0, "duration_seconds": 0.0}

    gain = 1.0 / max(1, len(clips))
    mix = [0] * (max_frames * 2)

    for pcm, channels in clips:
        if channels == 1:
            frame_count = len(pcm)
            for i in range(frame_count):
                value = int(pcm[i] * gain)
                idx = i * 2
                mix[idx] += value
                mix[idx + 1] += value
            continue

        frame_count = len(pcm) // 2
        for i in range(frame_count):
            src_idx = i * 2
            dst_idx = i * 2
            mix[dst_idx] += int(pcm[src_idx] * gain)
            mix[dst_idx + 1] += int(pcm[src_idx + 1] * gain)

    out = array("h")
    for value in mix:
        if value > 32767:
            out.append(32767)
        elif value < -32768:
            out.append(-32768)
        else:
            out.append(value)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf_out:
        wf_out.setnchannels(2)
        wf_out.setsampwidth(2)
        wf_out.setframerate(sample_rate)
        wf_out.writeframes(out.tobytes())

    meta = {
        "sources": len(clips),
        "sample_rate": sample_rate,
        "duration_seconds": round(max_frames / sample_rate, 3),
    }
    return buffer.getvalue(), meta


def build_mix_stream(
    catalog: dict[str, Any], vault_root: Path
) -> tuple[bytes, dict[str, Any]]:
    fingerprint = _mix_fingerprint(catalog)
    with _MIX_CACHE_LOCK:
        if _MIX_CACHE["fingerprint"] == fingerprint and _MIX_CACHE["wav"]:
            return _MIX_CACHE["wav"], _MIX_CACHE["meta"]

    sources = _collect_mix_sources(catalog, vault_root)
    wav, meta = _mix_wav_sources(sources)
    meta["fingerprint"] = fingerprint

    with _MIX_CACHE_LOCK:
        _MIX_CACHE["fingerprint"] = fingerprint
        _MIX_CACHE["wav"] = wav
        _MIX_CACHE["meta"] = meta
    return wav, meta


def _normalize_path_for_file_id(path_like: str) -> str:
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


def _file_id_for_path(path_like: str) -> str:
    normalized = _normalize_path_for_file_id(path_like)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _stable_entity_id(prefix: str, seed: str, width: int = 20) -> str:
    token = hashlib.sha256(seed.encode("utf-8")).hexdigest()[: max(8, width)]
    return f"{prefix}:{token}"


def _extract_coverage_spans(raw: Any) -> list[dict[str, Any]]:
    path_keys = (
        "file",
        "path",
        "source",
        "source_path",
        "source_file",
        "filename",
        "rel_path",
        "covered_file",
    )
    start_keys = ("start_line", "startLine", "line_start", "lineStart", "line", "start")
    end_keys = ("end_line", "endLine", "line_end", "lineEnd", "end")
    symbol_keys = ("symbol", "function", "name")
    weight_keys = ("w", "weight", "ratio", "hit_ratio", "hits", "coverage")

    spans: list[dict[str, Any]] = []

    def _extract_path(payload: dict[str, Any], fallback: str) -> str:
        for key in path_keys:
            value = payload.get(key)
            if isinstance(value, str):
                normalized = _normalize_path_for_file_id(value)
                if normalized:
                    return normalized
        return fallback

    def _extract_weight(payload: dict[str, Any], fallback: float) -> float:
        for key in weight_keys:
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return max(0.0, float(value))
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    continue
                try:
                    return max(0.0, float(text))
                except ValueError:
                    continue
        return max(0.0, fallback)

    def _extract_line(payload: dict[str, Any], keys: tuple[str, ...]) -> int:
        for key in keys:
            if key not in payload:
                continue
            parsed = _safe_int(payload.get(key), 0)
            if parsed > 0:
                return parsed
        return 0

    def _extract_symbol(payload: dict[str, Any]) -> str:
        for key in symbol_keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _emit_span(path_value: str, payload: dict[str, Any], weight: float) -> None:
        normalized = _normalize_path_for_file_id(path_value)
        if not normalized:
            return
        start_line = _extract_line(payload, start_keys)
        end_line = _extract_line(payload, end_keys)
        if start_line <= 0 and end_line > 0:
            start_line = end_line
        if start_line <= 0:
            start_line = 1
        if end_line <= 0:
            end_line = start_line
        if end_line < start_line:
            end_line = start_line
        span_row: dict[str, Any] = {
            "path": normalized,
            "start_line": start_line,
            "end_line": end_line,
            "symbol": _extract_symbol(payload),
            "weight": max(0.0, weight),
        }
        spans.append(span_row)

    def _walk(item: Any, fallback_path: str, fallback_weight: float) -> None:
        if isinstance(item, str):
            normalized = _normalize_path_for_file_id(item)
            if normalized:
                _emit_span(normalized, {}, fallback_weight)
            return

        if isinstance(item, list):
            for sub in item:
                _walk(sub, fallback_path, fallback_weight)
            return

        if not isinstance(item, dict):
            return

        local_path = _extract_path(item, fallback_path)
        local_weight = _extract_weight(item, fallback_weight)

        nested_markers = (
            "spans",
            "covered_spans",
            "ranges",
            "files",
            "covered_files",
            "coverage",
        )
        before = len(spans)
        for key in nested_markers:
            nested = item.get(key)
            if isinstance(nested, list):
                for sub in nested:
                    _walk(sub, local_path, local_weight)
            elif key == "coverage" and isinstance(nested, dict):
                _walk(nested, local_path, local_weight)

        has_nested = any(
            isinstance(item.get(key), (list, dict)) for key in nested_markers
        )
        if has_nested and len(spans) > before:
            return

        if local_path:
            _emit_span(local_path, item, local_weight)

    _walk(raw, "", 1.0)
    return spans


def _stable_ratio(seed: str, offset: int = 0) -> float:
    digest = hashlib.sha256(f"{seed}|{offset}".encode("utf-8")).digest()
    return int.from_bytes(digest[:2], "big") / 65535.0


def _coerce_test_failure_rows(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _append_row(item: Any) -> None:
        if isinstance(item, dict):
            rows.append(dict(item))
            return
        text = str(item).strip()
        if text:
            rows.append({"name": text, "status": "failed"})

    if isinstance(payload, list):
        for item in payload:
            _append_row(item)
        return rows

    if not isinstance(payload, dict):
        return rows

    for key in ("failures", "failed_tests", "failing_tests"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                _append_row(item)
            if rows:
                return rows

    tests = payload.get("tests")
    if isinstance(tests, list):
        for item in tests:
            if isinstance(item, dict):
                status = (
                    str(item.get("status") or item.get("outcome") or "").strip().lower()
                )
                if status and status not in {"failed", "error", "failing", "xfailed"}:
                    continue
                row = dict(item)
                if "status" not in row:
                    row["status"] = status or "failed"
                if not str(row.get("name", "")).strip():
                    row["name"] = str(
                        item.get("nodeid") or item.get("test") or item.get("id") or ""
                    ).strip()
                if str(row.get("name", "")).strip():
                    rows.append(row)
            else:
                _append_row(item)
        if rows:
            return rows

    name = str(
        payload.get("name")
        or payload.get("test")
        or payload.get("nodeid")
        or payload.get("id")
        or ""
    ).strip()
    if name:
        row = dict(payload)
        if "status" not in row:
            row["status"] = "failed"
        rows.append(row)
    return rows


def _parse_test_failures_text(raw_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        test_name, sep, covered_block = line.partition("|")
        normalized_name = test_name.strip()
        if not normalized_name:
            continue
        row: dict[str, Any] = {
            "name": normalized_name,
            "status": "failed",
        }
        if sep:
            covered_files = [
                token.strip()
                for token in re.split(r"[,\s]+", covered_block.strip())
                if token.strip()
            ]
            if covered_files:
                row["covered_files"] = covered_files
        rows.append(row)
    return rows


def _load_test_failures_from_path(candidate: Path) -> list[dict[str, Any]]:
    suffix = candidate.suffix.lower()
    try:
        raw_text = candidate.read_text("utf-8")
    except OSError:
        return []

    if suffix in {".json", ".jsonl", ".ndjson"}:
        if suffix in {".jsonl", ".ndjson"}:
            rows: list[dict[str, Any]] = []
            for raw_line in raw_text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except (ValueError, json.JSONDecodeError):
                    continue
                rows.extend(_coerce_test_failure_rows(payload))
            return rows
        try:
            payload = json.loads(raw_text)
        except (ValueError, json.JSONDecodeError):
            return []
        return _coerce_test_failure_rows(payload)

    return _parse_test_failures_text(raw_text)


def _normalize_coverage_source_path(
    raw_path: str,
    part_root: Path,
    vault_root: Path,
) -> str:
    source = str(raw_path or "").strip()
    if not source:
        return ""
    if source.startswith("file://"):
        source = unquote(urlparse(source).path)
    source = source.strip()
    if not source:
        return ""

    path_obj = Path(source)
    if path_obj.is_absolute():
        try:
            resolved = path_obj.resolve(strict=False)
        except OSError:
            resolved = path_obj
        for root in (part_root, vault_root):
            try:
                rel_path = resolved.relative_to(root.resolve())
            except (OSError, ValueError):
                continue
            normalized = _normalize_path_for_file_id(str(rel_path))
            if normalized:
                return normalized
        normalized_abs = _normalize_path_for_file_id(str(resolved))
        if normalized_abs:
            return normalized_abs

    return _normalize_path_for_file_id(source)


def _line_hits_to_spans(line_hits: list[tuple[int, int]]) -> list[dict[str, Any]]:
    sorted_hits = sorted(
        [
            (max(1, int(line)), max(0, int(hits)))
            for line, hits in line_hits
            if int(hits) > 0
        ],
        key=lambda row: row[0],
    )
    if not sorted_hits:
        return []

    spans: list[dict[str, Any]] = []
    start_line, prev_line, total_hits = (
        sorted_hits[0][0],
        sorted_hits[0][0],
        sorted_hits[0][1],
    )
    for line_no, hits in sorted_hits[1:]:
        if line_no <= prev_line + 1:
            prev_line = line_no
            total_hits += hits
            continue
        spans.append(
            {
                "start_line": start_line,
                "end_line": prev_line,
                "hits": total_hits,
            }
        )
        start_line = line_no
        prev_line = line_no
        total_hits = hits

    spans.append(
        {
            "start_line": start_line,
            "end_line": prev_line,
            "hits": total_hits,
        }
    )
    return spans


def _parse_lcov_payload(
    raw_text: str,
    part_root: Path,
    vault_root: Path,
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    by_test_sets: dict[str, set[str]] = defaultdict(set)
    by_test_span_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    current_test = ""
    current_source = ""
    current_line_hits: list[tuple[int, int]] = []
    da_found = 0
    da_hit = 0
    lf_value: int | None = None
    lh_value: int | None = None
    brda_found = 0
    brda_hit = 0
    brf_value: int | None = None
    brh_value: int | None = None
    fnda_found = 0
    fnda_hit = 0
    fnf_value: int | None = None
    fnh_value: int | None = None

    def _reset_record() -> None:
        nonlocal da_found
        nonlocal da_hit
        nonlocal lf_value
        nonlocal lh_value
        nonlocal brda_found
        nonlocal brda_hit
        nonlocal brf_value
        nonlocal brh_value
        nonlocal fnda_found
        nonlocal fnda_hit
        nonlocal fnf_value
        nonlocal fnh_value
        nonlocal current_line_hits
        da_found = 0
        da_hit = 0
        lf_value = None
        lh_value = None
        brda_found = 0
        brda_hit = 0
        brf_value = None
        brh_value = None
        fnda_found = 0
        fnda_hit = 0
        fnf_value = None
        fnh_value = None
        current_line_hits = []

    def _flush_record() -> None:
        nonlocal current_source
        if not current_source:
            return
        normalized_path = _normalize_coverage_source_path(
            current_source,
            part_root,
            vault_root,
        )
        if not normalized_path:
            current_source = ""
            _reset_record()
            return

        lines_found = max(0, int(lf_value if lf_value is not None else da_found))
        lines_hit = max(0, int(lh_value if lh_value is not None else da_hit))
        lines_hit = min(lines_hit, lines_found)

        branches_found = max(0, int(brf_value if brf_value is not None else brda_found))
        branches_hit = max(0, int(brh_value if brh_value is not None else brda_hit))
        branches_hit = min(branches_hit, branches_found)

        functions_found = max(
            0, int(fnf_value if fnf_value is not None else fnda_found)
        )
        functions_hit = max(0, int(fnh_value if fnh_value is not None else fnda_hit))
        functions_hit = min(functions_hit, functions_found)

        entry = files.get(normalized_path)
        if entry is None:
            entry = {
                "file_id": _file_id_for_path(normalized_path),
                "lines_found": 0,
                "lines_hit": 0,
                "branches_found": 0,
                "branches_hit": 0,
                "functions_found": 0,
                "functions_hit": 0,
                "tests": [],
            }
            files[normalized_path] = entry

        entry["lines_found"] = int(entry.get("lines_found", 0)) + lines_found
        entry["lines_hit"] = int(entry.get("lines_hit", 0)) + lines_hit
        entry["branches_found"] = int(entry.get("branches_found", 0)) + branches_found
        entry["branches_hit"] = int(entry.get("branches_hit", 0)) + branches_hit
        entry["functions_found"] = (
            int(entry.get("functions_found", 0)) + functions_found
        )
        entry["functions_hit"] = int(entry.get("functions_hit", 0)) + functions_hit

        normalized_test = current_test.strip()
        if normalized_test:
            by_test_sets[normalized_test].add(normalized_path)
            tests = entry.get("tests", [])
            if isinstance(tests, list) and normalized_test not in tests:
                tests.append(normalized_test)
                entry["tests"] = tests

            spans = _line_hits_to_spans(current_line_hits)
            for span in spans:
                by_test_span_rows[normalized_test].append(
                    {
                        "file": normalized_path,
                        "start_line": int(span.get("start_line", 1)),
                        "end_line": int(span.get("end_line", 1)),
                        "hits": int(span.get("hits", 0)),
                    }
                )

        current_source = ""
        _reset_record()

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("TN:"):
            current_test = line[3:].strip()
            continue
        if line.startswith("SF:"):
            _flush_record()
            current_source = line[3:].strip()
            _reset_record()
            continue
        if line == "end_of_record":
            _flush_record()
            continue
        if not current_source:
            continue

        if line.startswith("DA:"):
            da_payload = line[3:].split(",", 2)
            if len(da_payload) >= 2:
                line_no = max(1, int(_safe_float(da_payload[0], 0.0)))
                hits = max(0, int(_safe_float(da_payload[1], 0.0)))
                current_line_hits.append((line_no, hits))
                da_found += 1
                if hits > 0:
                    da_hit += 1
            continue
        if line.startswith("LF:"):
            lf_value = max(0, int(_safe_float(line[3:], 0.0)))
            continue
        if line.startswith("LH:"):
            lh_value = max(0, int(_safe_float(line[3:], 0.0)))
            continue
        if line.startswith("BRDA:"):
            br_payload = line[5:].split(",", 3)
            if len(br_payload) >= 4:
                taken = br_payload[3].strip()
                if taken != "-":
                    taken_count = max(0, int(_safe_float(taken, 0.0)))
                    brda_found += 1
                    if taken_count > 0:
                        brda_hit += 1
            continue
        if line.startswith("BRF:"):
            brf_value = max(0, int(_safe_float(line[4:], 0.0)))
            continue
        if line.startswith("BRH:"):
            brh_value = max(0, int(_safe_float(line[4:], 0.0)))
            continue
        if line.startswith("FNDA:"):
            fn_payload = line[5:].split(",", 1)
            if fn_payload:
                taken_count = max(0, int(_safe_float(fn_payload[0], 0.0)))
                fnda_found += 1
                if taken_count > 0:
                    fnda_hit += 1
            continue
        if line.startswith("FNF:"):
            fnf_value = max(0, int(_safe_float(line[4:], 0.0)))
            continue
        if line.startswith("FNH:"):
            fnh_value = max(0, int(_safe_float(line[4:], 0.0)))
            continue

    _flush_record()

    files_payload: dict[str, dict[str, Any]] = {}
    for path_key, entry in files.items():
        lines_found = max(0, int(_safe_float(entry.get("lines_found", 0), 0.0)))
        lines_hit = max(0, int(_safe_float(entry.get("lines_hit", 0), 0.0)))
        lines_hit = min(lines_hit, lines_found)
        branches_found = max(0, int(_safe_float(entry.get("branches_found", 0), 0.0)))
        branches_hit = max(0, int(_safe_float(entry.get("branches_hit", 0), 0.0)))
        branches_hit = min(branches_hit, branches_found)
        functions_found = max(0, int(_safe_float(entry.get("functions_found", 0), 0.0)))
        functions_hit = max(0, int(_safe_float(entry.get("functions_hit", 0), 0.0)))
        functions_hit = min(functions_hit, functions_found)
        line_rate = (lines_hit / lines_found) if lines_found > 0 else 0.0
        uncovered_ratio = 1.0 - line_rate if lines_found > 0 else 0.0

        tests = entry.get("tests", [])
        files_payload[path_key] = {
            "file_id": str(entry.get("file_id", "")),
            "lines_found": lines_found,
            "lines_hit": lines_hit,
            "line_rate": round(_clamp01(line_rate), 6),
            "uncovered_ratio": round(_clamp01(uncovered_ratio), 6),
            "branches_found": branches_found,
            "branches_hit": branches_hit,
            "functions_found": functions_found,
            "functions_hit": functions_hit,
            "tests": sorted(str(item) for item in tests if str(item).strip())
            if isinstance(tests, list)
            else [],
        }

    hottest_files = [
        path_key
        for path_key, _ in sorted(
            files_payload.items(),
            key=lambda item: (
                -_safe_float(item[1].get("uncovered_ratio", 0.0), 0.0),
                -_safe_float(item[1].get("lines_found", 0.0), 0.0),
                item[0],
            ),
        )
    ]

    by_test_spans: dict[str, list[dict[str, Any]]] = {}
    for test_name, rows in sorted(by_test_span_rows.items()):
        if not rows:
            continue
        merged: dict[tuple[str, int, int], dict[str, Any]] = {}
        for row in rows:
            path_key = _normalize_path_for_file_id(str(row.get("file", "")))
            start_line = max(1, int(_safe_float(row.get("start_line", 1), 1.0)))
            end_line = max(
                start_line,
                int(_safe_float(row.get("end_line", start_line), float(start_line))),
            )
            hits = max(0, int(_safe_float(row.get("hits", 0), 0.0)))
            if not path_key:
                continue
            key = (path_key, start_line, end_line)
            entry = merged.get(key)
            if entry is None:
                entry = {
                    "file": path_key,
                    "start_line": start_line,
                    "end_line": end_line,
                    "hits": 0,
                }
                merged[key] = entry
            entry["hits"] = int(entry.get("hits", 0)) + hits

        if not merged:
            continue
        merged_rows = sorted(
            merged.values(),
            key=lambda row: (
                str(row.get("file", "")),
                int(row.get("start_line", 0)),
                int(row.get("end_line", 0)),
            ),
        )
        total_hits = sum(
            max(0, int(_safe_float(row.get("hits", 0), 0.0))) for row in merged_rows
        )
        fallback_weight = 1.0 / len(merged_rows)

        output_rows: list[dict[str, Any]] = []
        for row in merged_rows:
            row_hits = max(0, int(_safe_float(row.get("hits", 0), 0.0)))
            if total_hits > 0:
                weight = row_hits / total_hits
            else:
                weight = fallback_weight
            output_rows.append(
                {
                    "file": str(row.get("file", "")),
                    "start_line": int(_safe_float(row.get("start_line", 1), 1.0)),
                    "end_line": int(_safe_float(row.get("end_line", 1), 1.0)),
                    "hits": row_hits,
                    "weight": round(max(0.0, weight), 6),
                }
            )

        if output_rows:
            by_test_spans[test_name] = output_rows

    totals: dict[str, Any] = {
        "lines_found": sum(
            max(0, int(_safe_float(row.get("lines_found", 0), 0.0)))
            for row in files_payload.values()
        ),
        "lines_hit": sum(
            max(0, int(_safe_float(row.get("lines_hit", 0), 0.0)))
            for row in files_payload.values()
        ),
        "files": len(files_payload),
    }
    totals["line_rate"] = round(
        (totals["lines_hit"] / totals["lines_found"])
        if totals["lines_found"] > 0
        else 0.0,
        6,
    )

    return {
        "record": "ημ.test-coverage.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "lcov",
        "files": files_payload,
        "by_test": {
            test_name: sorted(paths)
            for test_name, paths in sorted(by_test_sets.items())
            if paths
        },
        "by_test_spans": by_test_spans,
        "hottest_files": hottest_files,
        "totals": totals,
    }


def _load_test_coverage_from_path(
    candidate: Path,
    part_root: Path,
    vault_root: Path,
) -> dict[str, Any]:
    suffix = candidate.suffix.lower()
    try:
        raw_text = candidate.read_text("utf-8")
    except OSError:
        return {}

    if suffix == ".json":
        try:
            payload = json.loads(raw_text)
        except (ValueError, json.JSONDecodeError):
            return {}
        if isinstance(payload, dict):
            return payload
        return {}

    is_lcov = (
        candidate.name.lower().endswith(".info") and "lcov" in candidate.name.lower()
    )
    if is_lcov:
        return _parse_lcov_payload(raw_text, part_root, vault_root)

    return {}


def _load_test_signal_artifacts(
    part_root: Path,
    vault_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failure_candidates = [
        part_root / "world_state" / "failing_tests.json",
        part_root / "world_state" / "failing_tests.jsonl",
        part_root / "world_state" / "failing_tests.ndjson",
        part_root / "world_state" / "failing_tests.txt",
        part_root / "world_state" / "test_failures.json",
        part_root / ".opencode" / "runtime" / "failing_tests.json",
        part_root / ".opencode" / "runtime" / "failing_tests.jsonl",
        part_root / ".opencode" / "runtime" / "failing_tests.ndjson",
        part_root / ".opencode" / "runtime" / "failing_tests.txt",
        part_root / ".opencode" / "runtime" / "test_failures.json",
        vault_root / ".opencode" / "runtime" / "failing_tests.json",
        vault_root / ".opencode" / "runtime" / "failing_tests.jsonl",
        vault_root / ".opencode" / "runtime" / "failing_tests.ndjson",
        vault_root / ".opencode" / "runtime" / "failing_tests.txt",
        vault_root / ".opencode" / "runtime" / "test_failures.json",
    ]
    coverage_candidates = [
        part_root / "coverage" / "lcov.info",
        part_root / "world_state" / "lcov.info",
        part_root / ".opencode" / "runtime" / "lcov.info",
        vault_root / ".opencode" / "runtime" / "lcov.info",
        part_root / "world_state" / "test_coverage.json",
        part_root / ".opencode" / "runtime" / "test_coverage.json",
        vault_root / ".opencode" / "runtime" / "test_coverage.json",
    ]

    failures: list[dict[str, Any]] = []
    for candidate in failure_candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.exists() or not resolved.is_file():
            continue
        rows = _load_test_failures_from_path(resolved)
        if rows:
            failures = rows
            break

    coverage_payload: dict[str, Any] = {}
    for candidate in coverage_candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.exists() or not resolved.is_file():
            continue
        payload = _load_test_coverage_from_path(resolved, part_root, vault_root)
        if payload:
            coverage_payload = payload
            break

    return failures, coverage_payload


def _build_logical_graph(catalog: dict[str, Any]) -> dict[str, Any]:
    file_graph = catalog.get("file_graph") if isinstance(catalog, dict) else {}
    truth_state = catalog.get("truth_state") if isinstance(catalog, dict) else {}
    if not isinstance(file_graph, dict):
        file_graph = {}
    if not isinstance(truth_state, dict):
        truth_state = {}

    file_nodes_raw = file_graph.get("file_nodes", [])
    if not isinstance(file_nodes_raw, list):
        file_nodes_raw = []

    claims_raw = truth_state.get("claims", [])
    if not isinstance(claims_raw, list) or not claims_raw:
        claim_single = truth_state.get("claim", {})
        if isinstance(claim_single, dict) and claim_single:
            claims_raw = [claim_single]
        else:
            claims_raw = []

    proof = truth_state.get("proof", {})
    if not isinstance(proof, dict):
        proof = {}
    proof_entries = proof.get("entries", [])
    if not isinstance(proof_entries, list):
        proof_entries = []
    required_kinds = proof.get("required_kinds", [])
    if not isinstance(required_kinds, list):
        required_kinds = []

    gate = truth_state.get("gate", {})
    if not isinstance(gate, dict):
        gate = {}

    graph_nodes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    joins_source_to_file: dict[str, str] = {}
    file_path_to_node: dict[str, str] = {}
    file_id_to_node: dict[str, str] = {}

    test_failures = (
        catalog.get("test_failures", []) if isinstance(catalog, dict) else []
    )
    if not isinstance(test_failures, list):
        test_failures = []

    for file_node in file_nodes_raw:
        if not isinstance(file_node, dict):
            continue
        source_rel_path = str(
            file_node.get("source_rel_path")
            or file_node.get("archived_rel_path")
            or file_node.get("archive_rel_path")
            or file_node.get("name")
            or ""
        )
        normalized_path = _normalize_path_for_file_id(source_rel_path)
        if not normalized_path:
            continue
        file_id = _file_id_for_path(normalized_path)
        if not file_id:
            continue
        node_id = f"logical:file:{file_id[:24]}"
        source_uri = f"library:/{normalized_path}"
        file_path_to_node[normalized_path] = node_id
        file_id_to_node[file_id] = node_id
        joins_source_to_file[source_uri] = file_id
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "file",
                "label": str(
                    file_node.get("label") or file_node.get("name") or normalized_path
                ),
                "file_id": file_id,
                "source_uri": source_uri,
                "path": normalized_path,
                "x": round(_clamp01(_safe_float(file_node.get("x", 0.5), 0.5)), 4),
                "y": round(_clamp01(_safe_float(file_node.get("y", 0.5), 0.5)), 4),
                "confidence": 1.0,
                "provenance": {
                    "source_uri": source_uri,
                    "file_id": file_id,
                },
            }
        )

    for idx, row in enumerate(test_failures):
        if not isinstance(row, dict):
            continue
        test_name = str(row.get("name") or row.get("test") or "").strip()
        if not test_name:
            continue
        test_id_seed = f"test:{test_name}|{idx}"
        test_node_id = f"logical:test:{hashlib.sha256(test_id_seed.encode('utf-8')).hexdigest()[:24]}"
        test_node = {
            "id": test_node_id,
            "kind": "test",
            "label": test_name,
            "glyph": "試",
            "status": str(row.get("status", "failed")),
            "x": 0.5,
            "y": 0.5,
            "confidence": 1.0,
        }
        graph_nodes.append(test_node)

        covered_files = row.get("covered_files", [])
        if isinstance(covered_files, list):
            for path_item in covered_files:
                normalized_path = _normalize_path_for_file_id(str(path_item))
                target_node_id = file_path_to_node.get(normalized_path)
                if not target_node_id:
                    file_id = _file_id_for_path(normalized_path)
                    target_node_id = file_id_to_node.get(file_id)
                if target_node_id:
                    graph_edges.append(
                        {
                            "source": test_node_id,
                            "target": target_node_id,
                            "kind": "covers",
                            "weight": 0.8,
                        }
                    )

    rule_nodes: list[str] = []
    rule_nodes_by_kind: dict[str, str] = {}
    for idx, kind in enumerate(required_kinds):
        kind_text = str(kind).strip()
        if not kind_text:
            continue
        node_id = (
            f"logical:rule:{hashlib.sha256(kind_text.encode('utf-8')).hexdigest()[:20]}"
        )
        x = 0.18 + (_stable_ratio(kind_text, idx) * 0.2)
        y = 0.2 + (_stable_ratio(kind_text, idx + 19) * 0.35)
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "rule",
                "label": kind_text,
                "x": round(_clamp01(x), 4),
                "y": round(_clamp01(y), 4),
                "confidence": 1.0,
                "provenance": {"required_kind": kind_text},
            }
        )
        rule_nodes.append(node_id)
        rule_nodes_by_kind[kind_text] = node_id

    fact_nodes: list[str] = []
    for idx, claim in enumerate(claims_raw):
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("id") or f"claim:{idx}")
        claim_text = str(claim.get("text") or claim_id)
        status = str(claim.get("status", "undecided")).strip() or "undecided"
        kappa = round(_clamp01(_safe_float(claim.get("kappa", 0.0), 0.0)), 4)
        node_id = (
            f"logical:fact:{hashlib.sha256(claim_id.encode('utf-8')).hexdigest()[:22]}"
        )
        orbit = 0.14 + (_stable_ratio(claim_id, idx) * 0.09)
        angle = _stable_ratio(claim_id, idx + 7) * math.tau
        x = 0.72 + math.cos(angle) * orbit
        y = 0.5 + math.sin(angle) * orbit
        proof_refs = [
            str(item).strip()
            for item in claim.get("proof_refs", [])
            if str(item).strip()
        ]
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "fact",
                "label": claim_text,
                "status": status,
                "confidence": kappa,
                "x": round(_clamp01(x), 4),
                "y": round(_clamp01(y), 4),
                "provenance": {
                    "claim_id": claim_id,
                    "proof_refs": proof_refs,
                },
            }
        )
        fact_nodes.append(node_id)

        for ref in proof_refs:
            normalized_ref = _normalize_path_for_file_id(ref)
            file_node_id = file_path_to_node.get(normalized_ref)
            if not file_node_id:
                continue
            graph_edges.append(
                {
                    "id": f"logical:edge:prove:{hashlib.sha256((file_node_id + node_id + ref).encode('utf-8')).hexdigest()[:20]}",
                    "source": file_node_id,
                    "target": node_id,
                    "kind": "proves",
                    "weight": 1.0,
                }
            )

    derivation_nodes: list[str] = []
    for idx, entry in enumerate(proof_entries):
        if not isinstance(entry, dict):
            continue
        ref = str(entry.get("ref", "")).strip()
        kind = str(entry.get("kind", "")).strip()
        present = bool(entry.get("present", False))
        detail = str(entry.get("detail", "")).strip()
        base = f"{kind}|{ref}|{idx}"
        node_id = f"logical:derivation:{hashlib.sha256(base.encode('utf-8')).hexdigest()[:20]}"
        x = 0.42 + (_stable_ratio(base, idx) * 0.22)
        y = 0.42 + (_stable_ratio(base, idx + 27) * 0.3)
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "derivation",
                "label": detail or ref or kind or f"derivation-{idx + 1}",
                "status": "present" if present else "missing",
                "confidence": 1.0 if present else 0.0,
                "x": round(_clamp01(x), 4),
                "y": round(_clamp01(y), 4),
                "provenance": {
                    "kind": kind,
                    "ref": ref,
                    "present": present,
                },
            }
        )
        derivation_nodes.append(node_id)

        rule_node = rule_nodes_by_kind.get(kind)
        if rule_node:
            graph_edges.append(
                {
                    "id": f"logical:edge:rule:{hashlib.sha256((rule_node + node_id).encode('utf-8')).hexdigest()[:20]}",
                    "source": rule_node,
                    "target": node_id,
                    "kind": "requires",
                    "weight": 0.9,
                }
            )

        if fact_nodes:
            target_fact = fact_nodes[idx % len(fact_nodes)]
            graph_edges.append(
                {
                    "id": f"logical:edge:derive:{hashlib.sha256((node_id + target_fact).encode('utf-8')).hexdigest()[:20]}",
                    "source": node_id,
                    "target": target_fact,
                    "kind": "derives",
                    "weight": 0.82 if present else 0.36,
                }
            )

        normalized_ref = _normalize_path_for_file_id(ref)
        file_node_id = file_path_to_node.get(normalized_ref)
        if file_node_id:
            graph_edges.append(
                {
                    "id": f"logical:edge:source:{hashlib.sha256((file_node_id + node_id + normalized_ref).encode('utf-8')).hexdigest()[:20]}",
                    "source": file_node_id,
                    "target": node_id,
                    "kind": "source",
                    "weight": 0.92,
                }
            )

    gate_target = str(gate.get("target") or "push-truth")
    gate_node_id = (
        f"logical:gate:{hashlib.sha256(gate_target.encode('utf-8')).hexdigest()[:20]}"
    )
    graph_nodes.append(
        {
            "id": gate_node_id,
            "kind": "gate",
            "label": gate_target,
            "status": "blocked" if bool(gate.get("blocked", True)) else "ready",
            "confidence": 1.0,
            "x": 0.76,
            "y": 0.54,
            "provenance": {"target": gate_target},
        }
    )

    for fact_id in fact_nodes:
        graph_edges.append(
            {
                "id": f"logical:edge:gate:{hashlib.sha256((fact_id + gate_node_id).encode('utf-8')).hexdigest()[:20]}",
                "source": fact_id,
                "target": gate_node_id,
                "kind": "feeds",
                "weight": 0.74,
            }
        )

    contradiction_nodes = 0
    gate_reasons = [
        str(item).strip() for item in gate.get("reasons", []) if str(item).strip()
    ]
    for idx, reason in enumerate(gate_reasons[:6]):
        node_id = f"logical:contradiction:{hashlib.sha256(reason.encode('utf-8')).hexdigest()[:20]}"
        x = 0.86 + (_stable_ratio(reason, idx) * 0.1)
        y = 0.42 + (_stable_ratio(reason, idx + 33) * 0.24)
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "contradiction",
                "label": reason,
                "status": "active",
                "confidence": 1.0,
                "x": round(_clamp01(x), 4),
                "y": round(_clamp01(y), 4),
                "provenance": {"reason": reason},
            }
        )
        graph_edges.append(
            {
                "id": f"logical:edge:block:{hashlib.sha256((node_id + gate_node_id).encode('utf-8')).hexdigest()[:20]}",
                "source": node_id,
                "target": gate_node_id,
                "kind": "blocks",
                "weight": 1.0,
            }
        )
        contradiction_nodes += 1

    for node in graph_nodes:
        if node.get("kind") != "fact" or str(node.get("status")) != "refuted":
            continue
        reason = str(node.get("label", "refuted-fact"))
        node_id = f"logical:contradiction:{hashlib.sha256((reason + ':fact').encode('utf-8')).hexdigest()[:20]}"
        x = _clamp01(_safe_float(node.get("x", 0.5), 0.5) + 0.08)
        y = _clamp01(_safe_float(node.get("y", 0.5), 0.5) + 0.04)
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "contradiction",
                "label": reason,
                "status": "refuted",
                "confidence": 1.0,
                "x": round(x, 4),
                "y": round(y, 4),
                "provenance": {"from_fact": str(node.get("id", ""))},
            }
        )
        graph_edges.append(
            {
                "id": f"logical:edge:contradict:{hashlib.sha256((str(node.get('id')) + node_id).encode('utf-8')).hexdigest()[:20]}",
                "source": str(node.get("id", "")),
                "target": node_id,
                "kind": "contradicts",
                "weight": 1.0,
            }
        )
        contradiction_nodes += 1

    return {
        "record": "ημ.logical-graph.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": graph_nodes,
        "edges": graph_edges,
        "joins": {
            "file_ids": sorted(file_id_to_node.keys()),
            "file_index": {
                path: _file_id_for_path(path)
                for path in sorted(file_path_to_node.keys())
            },
            "source_to_file": dict(sorted(joins_source_to_file.items())),
        },
        "stats": {
            "file_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "file"]
            ),
            "fact_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "fact"]
            ),
            "rule_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "rule"]
            ),
            "derivation_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "derivation"]
            ),
            "contradiction_nodes": contradiction_nodes,
            "gate_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "gate"]
            ),
            "edge_count": len(graph_edges),
        },
    }


def _build_pain_field(
    catalog: dict[str, Any], logical_graph: dict[str, Any]
) -> dict[str, Any]:
    failures_raw = catalog.get("test_failures", []) if isinstance(catalog, dict) else []
    coverage_raw = catalog.get("test_coverage", {}) if isinstance(catalog, dict) else {}
    if not isinstance(failures_raw, list):
        failures_raw = []
    if not isinstance(coverage_raw, dict):
        coverage_raw = {}

    nodes = logical_graph.get("nodes", []) if isinstance(logical_graph, dict) else []
    edges = logical_graph.get("edges", []) if isinstance(logical_graph, dict) else []
    joins = logical_graph.get("joins", {}) if isinstance(logical_graph, dict) else {}
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    if not isinstance(joins, dict):
        joins = {}

    node_by_id = {
        str(node.get("id", "")): node
        for node in nodes
        if isinstance(node, dict) and str(node.get("id", "")).strip()
    }
    file_index = joins.get("file_index", {})
    if not isinstance(file_index, dict):
        file_index = {}
    file_id_to_path: dict[str, str] = {}
    for path_key, file_id_value in file_index.items():
        normalized_path = _normalize_path_for_file_id(str(path_key))
        file_id_key = str(file_id_value).strip()
        if normalized_path and file_id_key:
            file_id_to_path[file_id_key] = normalized_path
    file_id_to_node = {
        str(node.get("file_id", "")): str(node.get("id", ""))
        for node in nodes
        if isinstance(node, dict)
        and str(node.get("kind", "")) == "file"
        and str(node.get("file_id", "")).strip()
    }

    region_rows: list[dict[str, Any]] = []
    region_by_id: dict[str, dict[str, Any]] = {}
    region_by_file_id: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("kind", "")).strip() != "file":
            continue
        file_id = str(node.get("file_id", "")).strip()
        node_id = str(node.get("id", "")).strip()
        if not file_id or not node_id:
            continue
        region_key = str(node.get("path") or node.get("label") or file_id)
        region_seed = f"world-web|node|{node_id}|{region_key}"
        region_id = _stable_entity_id("region", region_seed)
        region_row: dict[str, Any] = {
            "region_id": region_id,
            "region_kind": "node",
            "region_key": region_key,
            "node_id": node_id,
            "file_id": file_id,
            "x": round(_clamp01(_safe_float(node.get("x", 0.5), 0.5)), 4),
            "y": round(_clamp01(_safe_float(node.get("y", 0.5), 0.5)), 4),
            "label": str(node.get("label", "")),
        }
        region_rows.append(region_row)
        region_by_id[region_id] = region_row
        region_by_file_id[file_id] = region_id

    region_rows.sort(key=lambda row: str(row.get("region_id", "")))

    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target:
            continue
        weight = _clamp01(_safe_float(edge.get("weight", 0.4), 0.4))
        adjacency[source].append((target, weight))
        adjacency[target].append((source, weight * 0.92))

    coverage_by_test = coverage_raw.get("by_test", {})
    if not isinstance(coverage_by_test, dict):
        coverage_by_test = {}

    coverage_by_test_spans = coverage_raw.get("by_test_spans", {})
    if not isinstance(coverage_by_test_spans, dict):
        coverage_by_test_spans = {}

    coverage_by_test_lower: dict[str, Any] = {}
    for key, value in coverage_by_test.items():
        normalized_key = str(key).strip().lower()
        if not normalized_key:
            continue
        coverage_by_test_lower[normalized_key] = value

    coverage_by_test_spans_lower: dict[str, Any] = {}
    for key, value in coverage_by_test_spans.items():
        normalized_key = str(key).strip().lower()
        if not normalized_key:
            continue
        coverage_by_test_spans_lower[normalized_key] = value

    hottest_files_raw = coverage_raw.get("hottest_files", [])
    hottest_files: list[str] = []
    if isinstance(hottest_files_raw, list):
        hottest_files = [str(path) for path in hottest_files_raw if str(path).strip()]
    if not hottest_files:
        files_metrics = coverage_raw.get("files", {})
        if isinstance(files_metrics, dict):
            scored_paths: list[tuple[str, float, float]] = []
            for path_key, metrics in files_metrics.items():
                path_text = str(path_key).strip()
                if not path_text:
                    continue
                line_rate = _clamp01(
                    _safe_float(
                        metrics.get("line_rate", 0.0)
                        if isinstance(metrics, dict)
                        else 0.0,
                        0.0,
                    )
                )
                lines_found = _safe_float(
                    metrics.get("lines_found", 0.0)
                    if isinstance(metrics, dict)
                    else 0.0,
                    0.0,
                )
                uncovered = max(0.0, 1.0 - line_rate)
                scored_paths.append((path_text, uncovered, lines_found))
            hottest_files = [
                path
                for path, _, _ in sorted(
                    scored_paths,
                    key=lambda row: (-row[1], -row[2], row[0]),
                )
            ]

    hottest_file_rank: dict[str, int] = {}
    for index, path_key in enumerate(hottest_files):
        normalized = _normalize_path_for_file_id(path_key)
        if normalized and normalized not in hottest_file_rank:
            hottest_file_rank[normalized] = index

    failing_tests: list[dict[str, Any]] = []
    test_span_weights: dict[tuple[str, str], float] = {}
    span_region_weights: dict[str, dict[str, float]] = defaultdict(dict)
    span_rows_by_id: dict[str, dict[str, Any]] = {}
    region_heat_raw: dict[str, float] = defaultdict(float)
    seeded_node_heat: dict[str, float] = defaultdict(float)

    for idx, row in enumerate(failures_raw):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "failed")).strip().lower()
        if status not in {"failed", "error", "xfailed", "failing"}:
            continue
        test_name = str(
            row.get("name") or row.get("test") or row.get("nodeid") or f"test-{idx + 1}"
        ).strip()
        if not test_name:
            continue
        message = str(row.get("message") or row.get("error") or "")

        coverage_sources: list[Any] = []
        for key in (
            "covered_spans",
            "spans",
            "coverage_spans",
            "covered_files",
            "files",
            "coverage",
        ):
            value = row.get(key)
            if value is not None:
                coverage_sources.append(value)

        from_coverage_spans = coverage_by_test_spans.get(test_name)
        if from_coverage_spans is None:
            from_coverage_spans = coverage_by_test_spans_lower.get(test_name.lower())
        if from_coverage_spans is not None:
            coverage_sources.append(from_coverage_spans)

        from_coverage = coverage_by_test.get(test_name)
        if from_coverage is None:
            from_coverage = coverage_by_test_lower.get(test_name.lower())
        if from_coverage is not None:
            coverage_sources.append(from_coverage)

        if not coverage_sources and hottest_files:
            coverage_sources.append(hottest_files[:3])

        normalized_spans: list[dict[str, Any]] = []
        for source in coverage_sources:
            normalized_spans.extend(_extract_coverage_spans(source))
        if not normalized_spans:
            continue

        severity = max(0.0, _safe_float(row.get("severity", 1.0), 1.0))
        signal_w = _clamp01(
            _safe_float(
                row.get(
                    "signal_w",
                    row.get("signal_weight", row.get("signal/w", 1.0)),
                ),
                1.0,
            )
        )
        suite_name = str(
            row.get("suite") or row.get("module") or row.get("file") or ""
        ).strip()
        runner_name = str(
            row.get("runner") or row.get("framework") or row.get("tool") or ""
        ).strip()
        test_id = _stable_entity_id(
            "test",
            f"{test_name}|{suite_name}|{runner_name}",
        )

        span_weights_for_test: dict[str, float] = defaultdict(float)
        covered_paths: set[str] = set()
        covered_file_ids: set[str] = set()

        for span in normalized_spans:
            path_value = _normalize_path_for_file_id(str(span.get("path") or ""))
            if not path_value:
                continue
            file_id = _file_id_for_path(path_value)
            if not file_id:
                continue

            start_line = max(1, _safe_int(span.get("start_line", 1), 1))
            end_line = max(
                start_line, _safe_int(span.get("end_line", start_line), start_line)
            )
            symbol = str(span.get("symbol", "")).strip()
            weight_raw = max(0.0, _safe_float(span.get("weight", 1.0), 1.0))
            if weight_raw <= 0.0:
                weight_raw = 1.0

            span_id = _stable_entity_id(
                "span",
                f"{file_id}|{start_line}|{end_line}|{symbol}",
            )
            span_weights_for_test[span_id] += weight_raw
            covered_paths.add(path_value)
            covered_file_ids.add(file_id)

            span_row = span_rows_by_id.get(span_id)
            if span_row is None:
                span_row = {
                    "id": span_id,
                    "file_id": file_id,
                    "path": path_value,
                    "start_line": start_line,
                    "end_line": end_line,
                    "symbol": symbol,
                }
                span_rows_by_id[span_id] = span_row

            region_id = region_by_file_id.get(file_id, "")
            if region_id:
                span_region_weights.setdefault(span_id, {})[region_id] = max(
                    span_region_weights.get(span_id, {}).get(region_id, 0.0),
                    1.0,
                )

        if not span_weights_for_test:
            continue

        total_span_weight = sum(span_weights_for_test.values())
        if total_span_weight <= 0.0:
            total_span_weight = float(len(span_weights_for_test))

        region_ids_for_test: set[str] = set()
        for span_id, raw_weight in sorted(span_weights_for_test.items()):
            edge_weight = (
                raw_weight / total_span_weight if total_span_weight > 0 else 0.0
            )
            if edge_weight <= 0.0:
                continue
            test_span_key = (test_id, span_id)
            test_span_weights[test_span_key] = max(
                test_span_weights.get(test_span_key, 0.0),
                edge_weight,
            )

            for region_id, region_weight in sorted(
                span_region_weights.get(span_id, {}).items()
            ):
                region_ids_for_test.add(region_id)
                contrib = severity * signal_w * edge_weight * max(0.0, region_weight)
                if contrib <= 0.0:
                    continue
                region_heat_raw[region_id] += contrib
                region_info = region_by_id.get(region_id, {})
                node_id = str(region_info.get("node_id", "")).strip()
                if node_id:
                    seeded_node_heat[node_id] += contrib

        span_ids_sorted = sorted(span_weights_for_test.keys())
        normalized_files = sorted(covered_paths)
        file_ids = sorted(covered_file_ids)
        if not normalized_files and hottest_files:
            normalized_files = [
                _normalize_path_for_file_id(path)
                for path in hottest_files[:3]
                if _normalize_path_for_file_id(path)
            ]

        failing_tests.append(
            {
                "id": test_id,
                "name": test_name,
                "status": status,
                "message": message,
                "severity": round(severity, 4),
                "signal_w": round(signal_w, 4),
                "failure_glyph": "破",
                "covered_files": normalized_files,
                "file_ids": file_ids,
                "span_ids": span_ids_sorted,
                "region_ids": sorted(region_ids_for_test),
            }
        )

    node_heat: dict[str, float] = {
        node_id: _clamp01(_safe_float(heat, 0.0))
        for node_id, heat in seeded_node_heat.items()
        if _safe_float(heat, 0.0) > 0.0
    }

    hop_decay = 0.58
    max_hops = 4
    current_frontier = sorted(node_heat.items(), key=lambda row: row[0])
    for _hop in range(max_hops):
        next_frontier: list[tuple[str, float]] = []
        for node_id, heat in current_frontier:
            if heat <= 0.02:
                continue
            for neighbor_id, edge_weight in adjacency.get(node_id, []):
                next_heat = _clamp01(heat * hop_decay * max(0.1, edge_weight))
                if next_heat <= 0.01:
                    continue
                if next_heat <= node_heat.get(neighbor_id, 0.0) + 0.004:
                    continue
                node_heat[neighbor_id] = next_heat
                next_frontier.append((neighbor_id, next_heat))
        current_frontier = next_frontier
        if not current_frontier:
            break

    def _heat_sort_key(item: tuple[str, float]) -> tuple[float, int, str]:
        node_id, heat_value = item
        node = node_by_id.get(node_id, {})
        if not isinstance(node, dict):
            node = {}
        file_id = str(node.get("file_id", "")).strip()
        path_value = file_id_to_path.get(file_id, "")
        if not path_value:
            path_value = _normalize_path_for_file_id(str(node.get("path", "")))
        rank = hottest_file_rank.get(path_value, 1_000_000)
        return (-_clamp01(_safe_float(heat_value, 0.0)), rank, node_id)

    heat_nodes: list[dict[str, Any]] = []
    for node_id, heat in sorted(node_heat.items(), key=_heat_sort_key):
        node = node_by_id.get(node_id, {})
        if not isinstance(node, dict):
            node = {}
        node_file_id = str(node.get("file_id", "")).strip()
        node_path = _normalize_path_for_file_id(str(node.get("path", "")))
        if node_file_id and node_path and node_file_id not in file_id_to_path:
            file_id_to_path[node_file_id] = node_path
        heat_nodes.append(
            {
                "node_id": node_id,
                "kind": str(node.get("kind", "unknown")),
                "heat": round(_clamp01(heat), 4),
                "x": round(_clamp01(_safe_float(node.get("x", 0.5), 0.5)), 4),
                "y": round(_clamp01(_safe_float(node.get("y", 0.5), 0.5)), 4),
                "file_id": str(node.get("file_id", "")),
                "label": str(node.get("label", "")),
            }
        )

    debug_target: dict[str, Any] = {
        "meaning": "DEBUG",
        "glyph": "診",
        "grounded": False,
        "source": "none",
        "node_id": "",
        "file_id": "",
        "region_id": "",
        "path": "",
        "label": "",
        "heat": 0.0,
        "x": 0.5,
        "y": 0.5,
        "reason": "no-active-failure-signal",
    }

    hottest_node = next(
        (
            row
            for row in heat_nodes
            if str(row.get("node_id", "")).strip()
            or str(row.get("file_id", "")).strip()
        ),
        None,
    )
    if isinstance(hottest_node, dict):
        node_id = str(hottest_node.get("node_id", "")).strip()
        file_id = str(hottest_node.get("file_id", "")).strip()
        node = node_by_id.get(node_id, {}) if node_id else {}
        if not isinstance(node, dict):
            node = {}

        path_value = file_id_to_path.get(file_id, "")
        if not path_value:
            path_value = _normalize_path_for_file_id(str(node.get("path", "")))
        if file_id and path_value and file_id not in file_id_to_path:
            file_id_to_path[file_id] = path_value

        label_value = str(hottest_node.get("label", "")).strip()
        if not label_value:
            label_value = str(node.get("label", "")).strip()
        if not label_value and path_value:
            label_value = Path(path_value).name

        debug_target = {
            "meaning": "DEBUG",
            "glyph": "診",
            "grounded": True,
            "source": "pain_field.max_heat",
            "node_id": node_id,
            "file_id": file_id,
            "region_id": region_by_file_id.get(file_id, ""),
            "path": path_value,
            "label": label_value,
            "heat": round(_clamp01(_safe_float(hottest_node.get("heat", 0.0), 0.0)), 4),
            "x": round(_clamp01(_safe_float(hottest_node.get("x", 0.5), 0.5)), 4),
            "y": round(_clamp01(_safe_float(hottest_node.get("y", 0.5), 0.5)), 4),
            "reason": "points-to-hottest-file",
        }
    elif hottest_files:
        fallback_path = _normalize_path_for_file_id(str(hottest_files[0]))
        fallback_file_id = _file_id_for_path(fallback_path) if fallback_path else ""
        fallback_node_id = file_id_to_node.get(fallback_file_id, "")
        fallback_node = node_by_id.get(fallback_node_id, {}) if fallback_node_id else {}
        if not isinstance(fallback_node, dict):
            fallback_node = {}
        if (
            fallback_file_id
            and fallback_path
            and fallback_file_id not in file_id_to_path
        ):
            file_id_to_path[fallback_file_id] = fallback_path

        label_value = str(fallback_node.get("label", "")).strip()
        if not label_value and fallback_path:
            label_value = Path(fallback_path).name

        debug_target = {
            "meaning": "DEBUG",
            "glyph": "診",
            "grounded": bool(fallback_path),
            "source": "coverage.hottest_files",
            "node_id": fallback_node_id,
            "file_id": fallback_file_id,
            "region_id": region_by_file_id.get(fallback_file_id, ""),
            "path": fallback_path,
            "label": label_value,
            "heat": 0.0,
            "x": round(_clamp01(_safe_float(fallback_node.get("x", 0.5), 0.5)), 4),
            "y": round(_clamp01(_safe_float(fallback_node.get("y", 0.5), 0.5)), 4),
            "reason": "fallback-to-coverage-hottest-file",
        }

    heat_regions: list[dict[str, Any]] = []
    for region_id, raw_heat in sorted(
        region_heat_raw.items(), key=lambda row: (-row[1], row[0])
    ):
        if raw_heat <= 0.0:
            continue
        region = region_by_id.get(region_id, {})
        heat_regions.append(
            {
                "region_id": region_id,
                "node_id": str(region.get("node_id", "")),
                "file_id": str(region.get("file_id", "")),
                "heat": round(_clamp01(_safe_float(raw_heat, 0.0)), 4),
                "heat_raw": round(max(0.0, _safe_float(raw_heat, 0.0)), 6),
                "glyph": "熱",
            }
        )

    span_rows = sorted(
        span_rows_by_id.values(),
        key=lambda row: (
            str(row.get("path", "")),
            int(row.get("start_line", 0)),
            int(row.get("end_line", 0)),
            str(row.get("id", "")),
        ),
    )

    test_covers_span_rows: list[dict[str, Any]] = []
    for (test_id, span_id), weight in sorted(
        test_span_weights.items(), key=lambda row: (row[0][0], row[0][1])
    ):
        test_covers_span_rows.append(
            {
                "id": _stable_entity_id(
                    "edge", f"{test_id}|{span_id}|覆/test-covers-span"
                ),
                "rel": "覆/test-covers-span",
                "test_id": test_id,
                "span_id": span_id,
                "w": round(_clamp01(_safe_float(weight, 0.0)), 6),
            }
        )

    span_maps_region_rows: list[dict[str, Any]] = []
    for span_id, region_weights in sorted(span_region_weights.items()):
        for region_id, weight in sorted(region_weights.items()):
            span_maps_region_rows.append(
                {
                    "id": _stable_entity_id(
                        "edge", f"{span_id}|{region_id}|覆/span-maps-to-region"
                    ),
                    "rel": "覆/span-maps-to-region",
                    "span_id": span_id,
                    "region_id": region_id,
                    "w": round(_clamp01(_safe_float(weight, 0.0)), 6),
                }
            )

    max_heat = max((row.get("heat", 0.0) for row in heat_nodes), default=0.0)
    return {
        "record": "ημ.pain-field.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active": bool(failing_tests),
        "decay": hop_decay,
        "hops": max_hops,
        "failing_tests": failing_tests,
        "spans": span_rows,
        "regions": region_rows,
        "relations": {
            "覆/test-covers-span": test_covers_span_rows,
            "覆/span-maps-to-region": span_maps_region_rows,
        },
        "heat_regions": heat_regions,
        "glyphs": {
            "locus": "址",
            "heat": "熱",
            "coverage": "覆",
            "failure": "破",
            "debug": "診",
        },
        "debug": debug_target,
        "grounded_meanings": {"DEBUG": debug_target},
        "node_heat": heat_nodes,
        "max_heat": round(_clamp01(_safe_float(max_heat, 0.0)), 4),
        "join_key": "file_id=sha256(normalized_path)",
        "region_join_key": "region_id=sha256(world|region_kind|region_key)",
    }


def _materialize_heat_values(
    catalog: dict[str, Any],
    pain_field: dict[str, Any],
) -> dict[str, Any]:
    named_fields = catalog.get("named_fields", []) if isinstance(catalog, dict) else []
    if not isinstance(named_fields, list):
        named_fields = []

    by_presence: dict[str, dict[str, Any]] = {}
    for row in named_fields:
        if not isinstance(row, dict):
            continue
        presence_id = str(row.get("id", "")).strip()
        if presence_id:
            by_presence[presence_id] = row

    for entity in ENTITY_MANIFEST:
        if not isinstance(entity, dict):
            continue
        presence_id = str(entity.get("id", "")).strip()
        if presence_id and presence_id not in by_presence:
            by_presence[presence_id] = entity

    field_anchors: dict[str, tuple[float, float]] = {}
    region_meta: dict[str, dict[str, Any]] = {}
    for field_id, presence_id in FIELD_TO_PRESENCE.items():
        item = by_presence.get(presence_id, {})
        x = _clamp01(_safe_float(item.get("x", 0.5), 0.5))
        y = _clamp01(_safe_float(item.get("y", 0.5), 0.5))
        field_anchors[field_id] = (x, y)
        region_meta[field_id] = {
            "region_id": field_id,
            "presence_id": presence_id,
            "en": str(item.get("en", presence_id)),
            "ja": str(item.get("ja", "")),
            "x": round(x, 4),
            "y": round(y, 4),
        }

    node_heat_rows = (
        pain_field.get("node_heat", []) if isinstance(pain_field, dict) else []
    )
    if not isinstance(node_heat_rows, list):
        node_heat_rows = []

    region_heat_raw: dict[str, float] = {field_id: 0.0 for field_id in field_anchors}
    locate_rows: list[dict[str, Any]] = []
    for row in node_heat_rows[:240]:
        if not isinstance(row, dict):
            continue
        entity_id = str(row.get("node_id", "")).strip()
        if not entity_id:
            continue
        heat = _clamp01(_safe_float(row.get("heat", 0.0), 0.0))
        if heat <= 0.0:
            continue

        x = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        y = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
        locate_scores = _field_scores_from_position(x, y, field_anchors)
        ranked_scores = sorted(
            locate_scores.items(),
            key=lambda item: (-_safe_float(item[1], 0.0), item[0]),
        )

        for field_id, locate_weight in ranked_scores:
            region_heat_raw[field_id] += heat * _clamp01(
                _safe_float(locate_weight, 0.0)
            )

        for field_id, locate_weight in ranked_scores[:4]:
            locate_rows.append(
                {
                    "kind": "址",
                    "entity_id": entity_id,
                    "region_id": field_id,
                    "weight": round(_clamp01(_safe_float(locate_weight, 0.0)), 4),
                }
            )

    max_raw_heat = max(region_heat_raw.values(), default=0.0)
    regions: list[dict[str, Any]] = []
    for rank, (field_id, raw_heat) in enumerate(
        sorted(region_heat_raw.items(), key=lambda item: (-item[1], item[0])), start=1
    ):
        value = 0.0
        if max_raw_heat > 0.0:
            value = _clamp01(raw_heat / max_raw_heat)
        meta = region_meta.get(
            field_id,
            {
                "region_id": field_id,
                "presence_id": FIELD_TO_PRESENCE.get(field_id, ""),
                "en": field_id,
                "ja": "",
                "x": 0.5,
                "y": 0.5,
            },
        )
        regions.append(
            {
                **meta,
                "rank": rank,
                "raw": round(max(0.0, raw_heat), 6),
                "value": round(value, 4),
            }
        )

    facts = [
        {
            "kind": "熱/value",
            "region_id": row.get("region_id", ""),
            "value": row.get("value", 0.0),
            "raw": row.get("raw", 0.0),
        }
        for row in regions
    ]
    return {
        "record": "ημ.heat-values.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active": max_raw_heat > 0.0,
        "source": "pain_field.node_heat",
        "regions": regions,
        "facts": facts,
        "locate": locate_rows,
        "max_raw": round(max(0.0, max_raw_heat), 6),
    }


def _daimoi_softmax_weights(
    rows: list[tuple[str, float]],
    *,
    temperature: float,
) -> dict[str, float]:
    if not rows:
        return {}
    temp = max(0.05, _safe_float(temperature, 0.42))
    max_score = max(_safe_float(score, 0.0) for _, score in rows)
    expo: list[tuple[str, float]] = []
    for entity_id, score in rows:
        value = math.exp((_safe_float(score, 0.0) - max_score) / temp)
        expo.append((entity_id, value))
    total = sum(value for _, value in expo)
    if total <= 0.0:
        uniform = 1.0 / max(1, len(expo))
        return {entity_id: uniform for entity_id, _ in expo}
    return {entity_id: value / total for entity_id, value in expo}


def _build_daimoi_state(
    heat_values: dict[str, Any],
    pain_field: dict[str, Any],
    *,
    queue_ratio: float = 0.0,
    resource_ratio: float = 0.0,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    if not isinstance(heat_values, dict):
        heat_values = {}
    if not isinstance(pain_field, dict):
        pain_field = {}

    node_heat_rows = pain_field.get("node_heat", [])
    if not isinstance(node_heat_rows, list):
        node_heat_rows = []

    relations: dict[str, list[dict[str, Any]]] = {
        "霊/attend": [],
        "霊/push": [],
        "霊/link": [],
        "霊/bind": [],
    }

    if not node_heat_rows:
        return {
            "record": "ημ.daimoi.v1",
            "generated_at": generated_at,
            "glyph": "霊",
            "active": False,
            "pressure": {
                "queue_ratio": round(_clamp01(_safe_float(queue_ratio, 0.0)), 4),
                "resource_ratio": round(_clamp01(_safe_float(resource_ratio, 0.0)), 4),
            },
            "daimoi": [],
            "relations": relations,
            "entities": [],
            "physics": {
                "kappa": round(DAIMO_FORCE_KAPPA, 6),
                "damping": round(DAIMO_DAMPING, 6),
                "dt": round(DAIMO_DT_SECONDS, 6),
            },
        }

    region_heat: dict[str, float] = {}
    region_centers: dict[str, tuple[float, float]] = {}
    for row in heat_values.get("regions", []):
        if not isinstance(row, dict):
            continue
        region_id = str(row.get("region_id", "")).strip()
        if not region_id:
            continue
        region_heat[region_id] = max(
            region_heat.get(region_id, 0.0),
            _clamp01(_safe_float(row.get("value", row.get("heat", 0.0)), 0.0)),
        )
        region_centers[region_id] = (
            _clamp01(_safe_float(row.get("x", 0.5), 0.5)),
            _clamp01(_safe_float(row.get("y", 0.5), 0.5)),
        )
    for row in heat_values.get("facts", []):
        if not isinstance(row, dict):
            continue
        region_id = str(row.get("region_id", "")).strip()
        if not region_id:
            continue
        region_heat[region_id] = max(
            region_heat.get(region_id, 0.0),
            _clamp01(_safe_float(row.get("value", row.get("heat", 0.0)), 0.0)),
        )
        region_centers.setdefault(region_id, (0.5, 0.5))

    if not region_heat:
        region_heat = {field_id: 0.0 for field_id in FIELD_TO_PRESENCE}
    entity_manifest_by_id = {
        str(row.get("id", "")): row
        for row in ENTITY_MANIFEST
        if isinstance(row, dict) and str(row.get("id", "")).strip()
    }
    field_anchors: dict[str, tuple[float, float]] = {}
    for field_id, presence_id in FIELD_TO_PRESENCE.items():
        if field_id in region_centers:
            field_anchors[field_id] = region_centers[field_id]
            continue
        entity = entity_manifest_by_id.get(presence_id, {})
        field_anchors[field_id] = (
            _clamp01(_safe_float(entity.get("x", 0.5), 0.5)),
            _clamp01(_safe_float(entity.get("y", 0.5), 0.5)),
        )
        region_centers[field_id] = field_anchors[field_id]
        region_heat.setdefault(field_id, 0.0)

    locate_rows = heat_values.get("locate", [])
    if not isinstance(locate_rows, list):
        locate_rows = []
    locate_by_entity: dict[str, dict[str, float]] = defaultdict(dict)
    for row in locate_rows:
        if not isinstance(row, dict):
            continue
        entity_id = str(row.get("entity_id", "")).strip()
        region_id = str(row.get("region_id", "")).strip()
        if not entity_id or not region_id:
            continue
        weight = _clamp01(_safe_float(row.get("weight", row.get("w", 0.0)), 0.0))
        if weight <= 0.0:
            continue
        locate_by_entity.setdefault(entity_id, {})[region_id] = max(
            locate_by_entity.get(entity_id, {}).get(region_id, 0.0),
            weight,
        )

    entities: list[dict[str, Any]] = []
    for row in node_heat_rows[:DAIMO_MAX_TRACKED_ENTITIES]:
        if not isinstance(row, dict):
            continue
        entity_id = str(row.get("node_id", "")).strip()
        if not entity_id:
            continue
        x = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        y = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
        heat = _clamp01(_safe_float(row.get("heat", 0.0), 0.0))
        locate = dict(locate_by_entity.get(entity_id, {}))
        if not locate:
            locate = _field_scores_from_position(x, y, field_anchors)

        score = 0.0
        for region_id, locate_weight in locate.items():
            score += _clamp01(_safe_float(locate_weight, 0.0)) * _clamp01(
                _safe_float(region_heat.get(region_id, 0.0), 0.0)
            )
        if score <= 0.0:
            score = heat * 0.1

        entities.append(
            {
                "id": entity_id,
                "x": x,
                "y": y,
                "heat": heat,
                "score": score,
                "mass": max(0.35, 0.8 + ((1.0 - heat) * 2.2)),
                "locate": locate,
            }
        )

    entities.sort(
        key=lambda row: (
            -_safe_float(row.get("score", 0.0), 0.0),
            str(row.get("id", "")),
        )
    )
    entity_by_id = {
        str(row.get("id", "")): row
        for row in entities
        if str(row.get("id", "")).strip()
    }
    failing_tests = pain_field.get("failing_tests", [])
    if not isinstance(failing_tests, list):
        failing_tests = []

    pressure_queue = _clamp01(_safe_float(queue_ratio, 0.0))
    pressure_resource = _clamp01(_safe_float(resource_ratio, 0.0))
    pressure = _clamp01((pressure_queue * 0.58) + (pressure_resource * 0.42))
    budget_scale = max(0.4, 1.0 - (pressure * 0.5))

    daimo_rows: list[dict[str, Any]] = []
    force_by_entity: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])

    for profile_index, profile in enumerate(DAIMO_PROFILE_DEFS):
        daimo_id = str(profile.get("id", f"daimo:{profile_index}"))
        ctx = str(profile.get("ctx", "世"))
        base_budget = max(
            1,
            int(
                round(_safe_float(profile.get("base_budget", 6.0), 6.0) * budget_scale)
            ),
        )
        daimo_w = _clamp01(_safe_float(profile.get("w", 0.88), 0.88))
        temperature = max(0.05, _safe_float(profile.get("temperature", 0.42), 0.42))
        top_k = max(1, min(6, base_budget // 2))

        scored_entities: list[tuple[str, float]] = []
        for entity_index, entity in enumerate(entities[:64]):
            entity_id = str(entity.get("id", ""))
            if not entity_id:
                continue
            base_score = max(0.0, _safe_float(entity.get("score", 0.0), 0.0))
            if base_score <= 0.0:
                continue
            entity_heat = _clamp01(_safe_float(entity.get("heat", 0.0), 0.0))
            if ctx == "主":
                ctx_gain = 1.08 + (entity_heat * 0.08)
            elif ctx == "己":
                ctx_gain = 0.95 + ((1.0 - entity_heat) * 0.08)
            elif ctx == "汝":
                ctx_gain = 0.98 + (
                    abs(_safe_float(entity.get("x", 0.5), 0.5) - 0.5) * 0.06
                )
            elif ctx == "彼":
                ctx_gain = 0.98 + (
                    abs(_safe_float(entity.get("y", 0.5), 0.5) - 0.5) * 0.06
                )
            else:
                ctx_gain = 1.02 + (pressure * 0.08)
            jitter = (
                _stable_ratio(f"{daimo_id}|{entity_id}", entity_index) - 0.5
            ) * 0.12
            scored = max(0.0, (base_score * ctx_gain) + jitter)
            if scored <= 0.0:
                continue
            scored_entities.append((entity_id, scored))

        scored_entities.sort(key=lambda row: (-row[1], row[0]))
        top_scored = scored_entities[:top_k]
        attention = _daimoi_softmax_weights(top_scored, temperature=temperature)

        attend_count = 0
        push_count = 0
        bind_count = 0
        link_count = 0

        for entity_id, entity_score in top_scored:
            attend_w = _clamp01(_safe_float(attention.get(entity_id, 0.0), 0.0))
            if attend_w <= 0.0:
                continue
            attend_count += 1
            relations["霊/attend"].append(
                {
                    "id": _stable_entity_id(
                        "edge", f"{daimo_id}|{entity_id}|霊/attend"
                    ),
                    "rel": "霊/attend",
                    "daimo_id": daimo_id,
                    "entity_id": entity_id,
                    "w": round(attend_w, 6),
                    "score": round(max(0.0, entity_score), 6),
                }
            )

            entity = entity_by_id.get(entity_id, {})
            if not isinstance(entity, dict):
                continue
            locate = entity.get("locate", {})
            if not isinstance(locate, dict):
                locate = {}
            px = _clamp01(_safe_float(entity.get("x", 0.5), 0.5))
            py = _clamp01(_safe_float(entity.get("y", 0.5), 0.5))

            vec_x = 0.0
            vec_y = 0.0
            best_region = ""
            best_signal = 0.0
            for region_id, locate_weight in locate.items():
                signal = _clamp01(_safe_float(locate_weight, 0.0)) * _clamp01(
                    _safe_float(region_heat.get(str(region_id), 0.0), 0.0)
                )
                if signal <= 0.0:
                    continue
                bx, by = region_centers.get(str(region_id), (0.5, 0.5))
                dx = bx - px
                dy = by - py
                magnitude = math.sqrt((dx * dx) + (dy * dy))
                if magnitude > 1e-8:
                    vec_x += signal * (dx / magnitude)
                    vec_y += signal * (dy / magnitude)
                if signal > best_signal:
                    best_signal = signal
                    best_region = str(region_id)

            vector_magnitude = math.sqrt((vec_x * vec_x) + (vec_y * vec_y))
            if vector_magnitude > 1e-8:
                dir_x = vec_x / vector_magnitude
                dir_y = vec_y / vector_magnitude
            else:
                dir_x = 0.0
                dir_y = 0.0

            fx = DAIMO_FORCE_KAPPA * daimo_w * attend_w * dir_x
            fy = DAIMO_FORCE_KAPPA * daimo_w * attend_w * dir_y
            if abs(fx) + abs(fy) > 1e-10:
                force_pair = force_by_entity[entity_id]
                force_pair[0] += fx
                force_pair[1] += fy

            push_count += 1
            relations["霊/push"].append(
                {
                    "id": _stable_entity_id("edge", f"{daimo_id}|{entity_id}|霊/push"),
                    "rel": "霊/push",
                    "daimo_id": daimo_id,
                    "entity_id": entity_id,
                    "region_id": best_region,
                    "fx": round(fx, 8),
                    "fy": round(fy, 8),
                    "w": round(attend_w, 6),
                }
            )

        if len(top_scored) >= 2:
            a_id = top_scored[0][0]
            b_id = top_scored[1][0]
            a_w = _clamp01(_safe_float(attention.get(a_id, 0.0), 0.0))
            b_w = _clamp01(_safe_float(attention.get(b_id, 0.0), 0.0))
            link_w = math.sqrt(max(0.0, a_w * b_w))
            link_count += 1
            relations["霊/link"].append(
                {
                    "id": _stable_entity_id(
                        "edge", f"{daimo_id}|{a_id}|{b_id}|霊/link"
                    ),
                    "rel": "霊/link",
                    "daimo_id": daimo_id,
                    "entity_a": a_id,
                    "entity_b": b_id,
                    "w": round(_clamp01(link_w), 6),
                }
            )

        fact_id = str((failing_tests[0] if failing_tests else {}).get("id", "")).strip()
        if fact_id and top_scored:
            anchor_entity = top_scored[0][0]
            anchor_w = _clamp01(_safe_float(attention.get(anchor_entity, 0.0), 0.0))
            bind_count += 1
            relations["霊/bind"].append(
                {
                    "id": _stable_entity_id("edge", f"{daimo_id}|{fact_id}|霊/bind"),
                    "rel": "霊/bind",
                    "daimo_id": daimo_id,
                    "fact_id": fact_id,
                    "w": round(anchor_w, 6),
                }
            )

        if not top_scored:
            state = "idle"
        elif push_count > 0:
            state = "move"
        elif bind_count > 0:
            state = "bind"
        else:
            state = "seek"

        emitted_total = attend_count + push_count + bind_count + link_count
        daimo_rows.append(
            {
                "id": daimo_id,
                "name": str(profile.get("name", daimo_id)),
                "ctx": ctx,
                "state": state,
                "budget": float(base_budget),
                "w": round(daimo_w, 4),
                "at_iso": generated_at,
                "emitted": {
                    "attend": attend_count,
                    "push": push_count,
                    "bind": bind_count,
                    "link": link_count,
                    "total": emitted_total,
                },
            }
        )

    entity_rows: list[dict[str, Any]] = []
    for entity in entities:
        entity_id = str(entity.get("id", "")).strip()
        if not entity_id:
            continue
        force = force_by_entity.get(entity_id, [0.0, 0.0])
        fx = _safe_float(force[0], 0.0)
        fy = _safe_float(force[1], 0.0)
        entity_rows.append(
            {
                "id": entity_id,
                "x": round(_clamp01(_safe_float(entity.get("x", 0.5), 0.5)), 4),
                "y": round(_clamp01(_safe_float(entity.get("y", 0.5), 0.5)), 4),
                "heat": round(_clamp01(_safe_float(entity.get("heat", 0.0), 0.0)), 4),
                "score": round(max(0.0, _safe_float(entity.get("score", 0.0), 0.0)), 6),
                "mass": round(max(0.35, _safe_float(entity.get("mass", 1.0), 1.0)), 6),
                "force": {
                    "fx": round(fx, 8),
                    "fy": round(fy, 8),
                    "magnitude": round(math.sqrt((fx * fx) + (fy * fy)), 8),
                },
            }
        )

    return {
        "record": "ημ.daimoi.v1",
        "generated_at": generated_at,
        "glyph": "霊",
        "active": bool(relations["霊/attend"]),
        "pressure": {
            "queue_ratio": round(pressure_queue, 4),
            "resource_ratio": round(pressure_resource, 4),
            "blend": round(pressure, 4),
        },
        "daimoi": daimo_rows,
        "relations": relations,
        "entities": entity_rows,
        "physics": {
            "kappa": round(DAIMO_FORCE_KAPPA, 6),
            "damping": round(DAIMO_DAMPING, 6),
            "dt": round(DAIMO_DT_SECONDS, 6),
        },
    }


def _apply_daimoi_dynamics_to_pain_field(
    pain_field: dict[str, Any],
    daimoi_state: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(pain_field, dict):
        return {}
    node_heat_rows = pain_field.get("node_heat", [])
    if not isinstance(node_heat_rows, list):
        node_heat_rows = []

    relations = (
        daimoi_state.get("relations", {}) if isinstance(daimoi_state, dict) else {}
    )
    if not isinstance(relations, dict):
        relations = {}
    push_rows = relations.get("霊/push", [])
    if not isinstance(push_rows, list):
        push_rows = []

    force_by_entity: dict[str, tuple[float, float]] = {}
    for row in push_rows:
        if not isinstance(row, dict):
            continue
        entity_id = str(row.get("entity_id", "")).strip()
        if not entity_id:
            continue
        fx = _safe_float(row.get("fx", 0.0), 0.0)
        fy = _safe_float(row.get("fy", 0.0), 0.0)
        prev_fx, prev_fy = force_by_entity.get(entity_id, (0.0, 0.0))
        force_by_entity[entity_id] = (prev_fx + fx, prev_fy + fy)

    physics = daimoi_state.get("physics", {}) if isinstance(daimoi_state, dict) else {}
    if not isinstance(physics, dict):
        physics = {}
    dt = max(
        0.02,
        min(0.4, _safe_float(physics.get("dt", DAIMO_DT_SECONDS), DAIMO_DT_SECONDS)),
    )
    damping = max(
        0.0,
        min(
            0.99,
            _safe_float(physics.get("damping", DAIMO_DAMPING), DAIMO_DAMPING),
        ),
    )

    updated_rows: list[dict[str, Any]] = []
    active_ids: set[str] = set()
    now_monotonic = time.monotonic()

    with _DAIMO_DYNAMICS_LOCK:
        cache_entities = _DAIMO_DYNAMICS_CACHE.get("entities", {})
        if not isinstance(cache_entities, dict):
            cache_entities = {}

        for row in node_heat_rows:
            if not isinstance(row, dict):
                continue
            entity_id = str(row.get("node_id", "")).strip()
            if not entity_id:
                updated_rows.append(dict(row))
                continue
            active_ids.add(entity_id)

            base_x = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
            base_y = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
            heat = _clamp01(_safe_float(row.get("heat", 0.0), 0.0))
            mass = max(0.35, 0.7 + ((1.0 - heat) * 2.0))

            cached = cache_entities.get(entity_id, {})
            if not isinstance(cached, dict):
                cached = {}
            prev_x = _clamp01(_safe_float(cached.get("x", base_x), base_x))
            prev_y = _clamp01(_safe_float(cached.get("y", base_y), base_y))
            prev_vx = _safe_float(cached.get("vx", 0.0), 0.0)
            prev_vy = _safe_float(cached.get("vy", 0.0), 0.0)

            fx, fy = force_by_entity.get(entity_id, (0.0, 0.0))
            tether = 0.18
            fx += (base_x - prev_x) * tether
            fy += (base_y - prev_y) * tether

            next_vx = (prev_vx * damping) + ((dt / mass) * fx)
            next_vy = (prev_vy * damping) + ((dt / mass) * fy)
            next_x = _clamp01(prev_x + (dt * next_vx))
            next_y = _clamp01(prev_y + (dt * next_vy))
            speed = math.sqrt((next_vx * next_vx) + (next_vy * next_vy))

            cache_entities[entity_id] = {
                "x": next_x,
                "y": next_y,
                "vx": next_vx,
                "vy": next_vy,
                "ts": now_monotonic,
            }

            updated = dict(row)
            updated["x"] = round(next_x, 4)
            updated["y"] = round(next_y, 4)
            updated["vx"] = round(next_vx, 6)
            updated["vy"] = round(next_vy, 6)
            updated["speed"] = round(speed, 6)
            updated_rows.append(updated)

        stale_before = now_monotonic - 120.0
        for entity_id in list(cache_entities.keys()):
            if entity_id in active_ids:
                continue
            state = cache_entities.get(entity_id, {})
            state_ts = _safe_float(
                state.get("ts", 0.0) if isinstance(state, dict) else 0.0,
                0.0,
            )
            if state_ts < stale_before:
                cache_entities.pop(entity_id, None)

        _DAIMO_DYNAMICS_CACHE["entities"] = cache_entities
        _DAIMO_DYNAMICS_CACHE["last_gc_monotonic"] = now_monotonic

    output = dict(pain_field)
    output["node_heat"] = updated_rows
    output["motion"] = {
        "record": "ημ.daimoi-motion.v1",
        "glyph": "霊",
        "active": bool(force_by_entity),
        "dt": round(dt, 6),
        "damping": round(damping, 6),
        "entity_count": len(updated_rows),
        "forced_entities": len(force_by_entity),
    }
    return output


def websocket_accept_value(client_key: str) -> str:
    digest = sha1((client_key + WS_MAGIC).encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


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


def build_simulation_state(
    catalog: dict[str, Any],
    myth_summary: dict[str, Any] | None = None,
    world_summary: dict[str, Any] | None = None,
    *,
    influence_snapshot: dict[str, Any] | None = None,
    queue_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = time.time()
    resource_budget_snapshot = _resource_monitor_snapshot()
    budget_devices = (
        resource_budget_snapshot.get("devices", {})
        if isinstance(resource_budget_snapshot, dict)
        else {}
    )
    budget_cpu = _safe_float(
        (budget_devices.get("cpu", {}) if isinstance(budget_devices, dict) else {}).get(
            "utilization", 0.0
        ),
        0.0,
    )
    if budget_cpu >= 90.0:
        sim_point_budget = max(256, int(MAX_SIM_POINTS * 0.55))
    elif budget_cpu >= 78.0:
        sim_point_budget = max(320, int(MAX_SIM_POINTS * 0.74))
    else:
        sim_point_budget = MAX_SIM_POINTS

    points: list[dict[str, float]] = []
    items = catalog.get("items", [])
    file_graph = catalog.get("file_graph") if isinstance(catalog, dict) else None
    crawler_graph = catalog.get("crawler_graph") if isinstance(catalog, dict) else None
    truth_state = catalog.get("truth_state") if isinstance(catalog, dict) else None
    logical_graph = catalog.get("logical_graph") if isinstance(catalog, dict) else None
    if not isinstance(logical_graph, dict):
        logical_graph = _build_logical_graph(
            catalog if isinstance(catalog, dict) else {}
        )
    pain_field = catalog.get("pain_field") if isinstance(catalog, dict) else None
    if not isinstance(pain_field, dict):
        pain_field = _build_pain_field(
            catalog if isinstance(catalog, dict) else {},
            logical_graph,
        )
    heat_values = _materialize_heat_values(
        catalog if isinstance(catalog, dict) else {},
        pain_field,
    )
    graph_file_nodes = (
        file_graph.get("file_nodes", []) if isinstance(file_graph, dict) else []
    )
    graph_crawler_nodes = (
        crawler_graph.get("crawler_nodes", [])
        if isinstance(crawler_graph, dict)
        else []
    )

    for idx, item in enumerate(items[:sim_point_budget]):
        key = (
            f"{item.get('rel_path', '')}|{item.get('part', '')}|{item.get('kind', '')}|{idx}"
        ).encode("utf-8")
        digest = sha1(key).digest()

        x = (int.from_bytes(digest[0:2], "big") / 65535.0) * 2.0 - 1.0
        base_y = (int.from_bytes(digest[2:4], "big") / 65535.0) * 2.0 - 1.0
        phase = (digest[4] / 255.0) * math.tau
        speed = 0.4 + (digest[5] / 255.0) * 0.9
        wobble = math.sin(now * speed + phase) * 0.11
        y = max(-1.0, min(1.0, base_y + wobble))

        size = 2.8 + (digest[6] / 255.0) * 9.0
        r = 0.2 + (digest[7] / 255.0) * 0.75
        g = 0.2 + (digest[8] / 255.0) * 0.75
        b = 0.2 + (digest[9] / 255.0) * 0.75

        kind = str(item.get("kind", ""))
        if kind == "audio":
            size += 2.2
            r = min(1.0, r + 0.18)
            g = min(1.0, g + 0.16)
        elif kind == "video":
            b = min(1.0, b + 0.2)
        elif kind == "image":
            g = min(1.0, g + 0.1)

        points.append(
            {
                "x": round(x, 5),
                "y": round(y, 5),
                "size": round(size, 5),
                "r": round(r, 5),
                "g": round(g, 5),
                "b": round(b, 5),
            }
        )

    remaining_capacity = max(0, sim_point_budget - len(points))
    for node in list(graph_file_nodes)[:remaining_capacity]:
        if not isinstance(node, dict):
            continue
        x_norm = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
        y_norm = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
        hue = _safe_float(node.get("hue", 200), 200.0)
        importance = _clamp01(_safe_float(node.get("importance", 0.4), 0.4))
        r_raw, g_raw, b_raw = colorsys.hsv_to_rgb((hue % 360.0) / 360.0, 0.58, 0.95)
        points.append(
            {
                "x": round((x_norm * 2.0) - 1.0, 5),
                "y": round(1.0 - (y_norm * 2.0), 5),
                "size": round(2.6 + (importance * 6.2), 5),
                "r": round(r_raw, 5),
                "g": round(g_raw, 5),
                "b": round(b_raw, 5),
            }
        )

    remaining_capacity = max(0, sim_point_budget - len(points))
    for node in list(graph_crawler_nodes)[:remaining_capacity]:
        if not isinstance(node, dict):
            continue
        x_norm = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
        y_norm = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
        hue = _safe_float(node.get("hue", 180), 180.0)
        importance = _clamp01(_safe_float(node.get("importance", 0.3), 0.3))
        kind = str(node.get("crawler_kind", "url")).strip().lower()
        saturation = 0.66 if kind == "url" else 0.52
        value = 0.96 if kind == "url" else 0.9
        r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
            (hue % 360.0) / 360.0, saturation, value
        )
        points.append(
            {
                "x": round((x_norm * 2.0) - 1.0, 5),
                "y": round(1.0 - (y_norm * 2.0), 5),
                "size": round(2.2 + (importance * 5.0), 5),
                "r": round(r_raw, 5),
                "g": round(g_raw, 5),
                "b": round(b_raw, 5),
            }
        )

    truth_claims = (
        truth_state.get("claims", []) if isinstance(truth_state, dict) else []
    )
    if not isinstance(truth_claims, list):
        truth_claims = []
    truth_guard = truth_state.get("guard", {}) if isinstance(truth_state, dict) else {}
    if not isinstance(truth_guard, dict):
        truth_guard = {}
    truth_gate = truth_state.get("gate", {}) if isinstance(truth_state, dict) else {}
    if not isinstance(truth_gate, dict):
        truth_gate = {}
    truth_gate_blocked = bool(truth_gate.get("blocked", True))
    truth_guard_pass = bool(truth_guard.get("passes", False))

    remaining_capacity = max(0, sim_point_budget - len(points))
    if remaining_capacity > 0 and truth_claims:
        claim_x = 0.76
        claim_y = 0.54
        for claim_index, claim in enumerate(truth_claims[: min(3, remaining_capacity)]):
            if not isinstance(claim, dict):
                continue
            kappa = _clamp01(_safe_float(claim.get("kappa", 0.0), 0.0))
            status = str(claim.get("status", "undecided")).strip().lower()
            if status == "proved":
                hue = 136.0
            elif status == "refuted":
                hue = 12.0
            else:
                hue = 52.0
            if truth_guard_pass:
                hue = 150.0
            elif truth_gate_blocked:
                hue = max(0.0, hue - 12.0)

            orbit = 0.012 + (claim_index * 0.014)
            phase = now * (0.45 + claim_index * 0.11)
            x_norm = _clamp01(claim_x + (math.cos(phase) * orbit))
            y_norm = _clamp01(claim_y + (math.sin(phase) * orbit))
            saturation = 0.72 if status == "proved" else 0.78
            value = 0.96 if status == "proved" else 0.88
            r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
                (hue % 360.0) / 360.0,
                saturation,
                value,
            )
            points.append(
                {
                    "x": round((x_norm * 2.0) - 1.0, 5),
                    "y": round(1.0 - (y_norm * 2.0), 5),
                    "size": round(3.2 + (kappa * 5.8), 5),
                    "r": round(r_raw, 5),
                    "g": round(g_raw, 5),
                    "b": round(b_raw, 5),
                }
            )

    counts = catalog.get("counts", {})

    entity_states = []
    for e in ENTITY_MANIFEST:
        base_seed = int(sha1(e["id"].encode("utf-8")).hexdigest()[:8], 16)
        t = now + (base_seed % 1000)

        bpm = 60 + (math.sin(t * 0.1) * 20) + ((base_seed % 20) - 10)

        vitals = {}
        for k, unit in e.get("flavor_vitals", {}).items():
            val_seed = (base_seed + hash(k)) % 1000
            val = abs(
                math.sin(t * (0.05 + (val_seed % 10) / 100)) * (100 + (val_seed % 50))
            )
            if unit == "%":
                val = val % 100
            vitals[k] = f"{val:.1f}{unit}"

        entity_states.append(
            {
                "id": e["id"],
                "bpm": round(bpm, 1),
                "stability": round(90 + math.sin(t * 0.02) * 9, 1),
                "resonance": round(e["freq"] + math.sin(t) * 2, 1),
                "vitals": vitals,
            }
        )

    # Mycelial Echoes (Memory Particles)
    echo_particles = []
    collection = _get_chroma_collection()
    if collection:
        try:
            results = collection.get(limit=12)
            docs = results.get("documents", [])
            for i, doc in enumerate(docs):
                # Map memory doc to an ephemeral particle
                seed = int(sha1(doc.encode("utf-8")).hexdigest()[:8], 16)
                t_off = now + (seed % 500)
                # Particles orbit the center or drift
                echo_particles.append(
                    {
                        "id": f"echo_{i}",
                        "text": doc[:24] + "...",
                        "x": 0.5 + math.sin(t_off * 0.15) * 0.35,
                        "y": 0.5 + math.cos(t_off * 0.12) * 0.35,
                        "hue": (200 + (seed % 100)) % 360,
                        "life": 0.5 + math.sin(t_off * 0.5) * 0.5,
                    }
                )
        except Exception:
            pass

    queue_snapshot = queue_snapshot or {}
    influence = influence_snapshot or _INFLUENCE_TRACKER.snapshot(
        queue_snapshot=queue_snapshot
    )

    clicks_recent = int(influence.get("clicks_45s", 0))
    file_changes_recent = int(influence.get("file_changes_120s", 0))
    queue_pending_count = int(queue_snapshot.get("pending_count", 0))
    queue_event_count = int(queue_snapshot.get("event_count", 0))

    audio_count = int(counts.get("audio", 0))
    audio_ratio = _clamp01(audio_count / 12.0)
    click_ratio = _clamp01(clicks_recent / 18.0)
    file_ratio = _clamp01(file_changes_recent / 24.0)
    queue_ratio = _clamp01((queue_pending_count + queue_event_count * 0.25) / 16.0)
    resource_heartbeat = (
        influence.get("resource_heartbeat", {}) if isinstance(influence, dict) else {}
    )
    if not isinstance(resource_heartbeat, dict) or not resource_heartbeat:
        resource_heartbeat = resource_budget_snapshot
    resource_devices = (
        resource_heartbeat.get("devices", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    resource_cpu_util = _safe_float(
        (
            resource_devices.get("cpu", {})
            if isinstance(resource_devices, dict)
            else {}
        ).get("utilization", 0.0),
        0.0,
    )
    resource_gpu_util = _safe_float(
        (
            resource_devices.get("gpu1", {})
            if isinstance(resource_devices, dict)
            else {}
        ).get("utilization", 0.0),
        0.0,
    )
    resource_npu_util = _safe_float(
        (
            resource_devices.get("npu0", {})
            if isinstance(resource_devices, dict)
            else {}
        ).get("utilization", 0.0),
        0.0,
    )
    resource_ratio = _clamp01(
        max(resource_cpu_util, resource_gpu_util, resource_npu_util) / 100.0
    )

    river_flow_rate = round(
        1.2 + (audio_ratio * 4.4) + (file_ratio * 7.2) + (click_ratio * 2.6), 3
    )
    river_turbulence = round(_clamp01((file_ratio * 0.72) + (click_ratio * 0.4)), 4)

    manifest_lookup = {
        str(item.get("id", "")): item for item in ENTITY_MANIFEST if item.get("id")
    }
    impact_order = [
        *CANONICAL_NAMED_FIELD_IDS,
        FILE_SENTINEL_PROFILE["id"],
        FILE_ORGANIZER_PROFILE["id"],
        HEALTH_SENTINEL_CPU_PROFILE["id"],
        HEALTH_SENTINEL_GPU1_PROFILE["id"],
        HEALTH_SENTINEL_GPU2_PROFILE["id"],
        HEALTH_SENTINEL_NPU0_PROFILE["id"],
    ]
    base_file = {
        "receipt_river": 0.94,
        "witness_thread": 0.38,
        "fork_tax_canticle": 0.84,
        "mage_of_receipts": 0.88,
        "keeper_of_receipts": 0.9,
        "anchor_registry": 0.64,
        "gates_of_truth": 0.73,
        "file_sentinel": 1.0,
        "file_organizer": 0.86,
        "health_sentinel_cpu": 0.58,
        "health_sentinel_gpu1": 0.54,
        "health_sentinel_gpu2": 0.5,
        "health_sentinel_npu0": 0.52,
    }
    base_click = {
        "receipt_river": 0.52,
        "witness_thread": 0.94,
        "fork_tax_canticle": 0.66,
        "mage_of_receipts": 0.57,
        "keeper_of_receipts": 0.61,
        "anchor_registry": 0.83,
        "gates_of_truth": 0.8,
        "file_sentinel": 0.55,
        "file_organizer": 0.62,
        "health_sentinel_cpu": 0.44,
        "health_sentinel_gpu1": 0.36,
        "health_sentinel_gpu2": 0.34,
        "health_sentinel_npu0": 0.32,
    }
    base_emit = {
        "receipt_river": 0.95,
        "witness_thread": 0.71,
        "fork_tax_canticle": 0.79,
        "mage_of_receipts": 0.73,
        "keeper_of_receipts": 0.81,
        "anchor_registry": 0.68,
        "gates_of_truth": 0.75,
        "file_sentinel": 0.82,
        "file_organizer": 0.78,
        "health_sentinel_cpu": 0.68,
        "health_sentinel_gpu1": 0.74,
        "health_sentinel_gpu2": 0.7,
        "health_sentinel_npu0": 0.77,
    }
    base_resource = {
        "receipt_river": 0.2,
        "witness_thread": 0.15,
        "fork_tax_canticle": 0.31,
        "mage_of_receipts": 0.28,
        "keeper_of_receipts": 0.26,
        "anchor_registry": 0.24,
        "gates_of_truth": 0.33,
        "file_sentinel": 0.44,
        "file_organizer": 0.36,
        "health_sentinel_cpu": 0.92,
        "health_sentinel_gpu1": 0.96,
        "health_sentinel_gpu2": 0.9,
        "health_sentinel_npu0": 0.95,
    }

    presence_impacts: list[dict[str, Any]] = []
    for presence_id in impact_order:
        if presence_id == FILE_SENTINEL_PROFILE["id"]:
            meta = FILE_SENTINEL_PROFILE
        elif presence_id == FILE_ORGANIZER_PROFILE["id"]:
            meta = FILE_ORGANIZER_PROFILE
        else:
            meta = manifest_lookup.get(
                presence_id,
                {
                    "id": presence_id,
                    "en": presence_id.replace("_", " ").title(),
                    "ja": "",
                },
            )

        file_influence = _clamp01(
            (file_ratio * float(base_file.get(presence_id, 0.5))) + (queue_ratio * 0.22)
        )
        click_influence = _clamp01(
            click_ratio * float(base_click.get(presence_id, 0.5))
        )
        resource_influence = _clamp01(
            resource_ratio * float(base_resource.get(presence_id, 0.22))
        )
        total_influence = _clamp01(
            (file_influence * 0.52)
            + (click_influence * 0.28)
            + (resource_influence * 0.2)
        )
        emits_flow = _clamp01(
            (total_influence * 0.72)
            + (audio_ratio * float(base_emit.get(presence_id, 0.5)) * 0.35)
        )

        if presence_id == "receipt_river":
            notes_en = (
                "River flow accelerates when files move and witnesses touch the field."
            )
            notes_ja = "ファイル変化と触れた証人で、川の流れは加速する。"
        elif presence_id == "file_sentinel":
            notes_en = "Auto-committing ghost stages proof paths before the gate asks."
            notes_ja = "自動コミットの幽霊は、門に問われる前に証明経路を段取る。"
        elif presence_id == "file_organizer":
            notes_en = "Organizer presence groups files into concept clusters from embedding space."
            notes_ja = "分類師プレゼンスは埋め込み空間から概念クラスタを編成する。"
        elif presence_id == "fork_tax_canticle":
            notes_en = "Fork tax pressure rises with unresolved file drift."
            notes_ja = "未解決のファイルドリフトでフォーク税圧は上がる。"
        elif presence_id == "witness_thread":
            notes_en = "Mouse touches tighten witness linkage across presences."
            notes_ja = "マウスの接触はプレゼンス間の証人連結を強める。"
        elif presence_id == "health_sentinel_cpu":
            notes_en = (
                "CPU sentinel throttles particle budgets when host pressure rises."
            )
            notes_ja = "CPU哨戒はホスト圧上昇時に粒子予算を絞る。"
        elif presence_id == "health_sentinel_gpu1":
            notes_en = (
                "GPU1 sentinel maps throughput and thermals into backend selection."
            )
            notes_ja = "GPU1哨戒は処理量と熱をバックエンド選択へ写像する。"
        elif presence_id == "health_sentinel_gpu2":
            notes_en = "GPU2 sentinel absorbs burst load to keep field vectors stable."
            notes_ja = "GPU2哨戒は突発負荷を吸収し、場のベクトルを安定化する。"
        elif presence_id == "health_sentinel_npu0":
            notes_en = (
                "NPU sentinel tracks efficient inferencing for embedding pathways."
            )
            notes_ja = "NPU哨戒は埋め込み経路の効率推論を監視する。"
        else:
            notes_en = "Presence responds to blended file and witness pressure."
            notes_ja = "このプレゼンスはファイル圧と証人圧の混合に応答する。"

        presence_impacts.append(
            {
                "id": presence_id,
                "en": str(meta.get("en", "Presence")),
                "ja": str(meta.get("ja", "プレゼンス")),
                "affected_by": {
                    "files": round(file_influence, 4),
                    "clicks": round(click_influence, 4),
                    "resource": round(resource_influence, 4),
                },
                "affects": {
                    "world": round(emits_flow, 4),
                    "ledger": round(_clamp01(total_influence * 0.86), 4),
                },
                "notes_en": notes_en,
                "notes_ja": notes_ja,
            }
        )

    witness_meta = manifest_lookup.get(
        "witness_thread",
        {
            "id": "witness_thread",
            "en": "Witness Thread",
            "ja": "証人の糸",
        },
    )
    witness_impact = next(
        (item for item in presence_impacts if item.get("id") == "witness_thread"),
        None,
    )
    lineage: list[dict[str, str]] = []
    seen_lineage_refs: set[str] = set()

    for target in list(influence.get("recent_click_targets", []))[:6]:
        ref = str(target).strip()
        if not ref or ref in seen_lineage_refs:
            continue
        seen_lineage_refs.add(ref)
        lineage.append(
            {
                "kind": "touch",
                "ref": ref,
                "why_en": "Witness touch linked this target into continuity.",
                "why_ja": "証人の接触がこの対象を連続線へ接続した。",
            }
        )

    for path in list(influence.get("recent_file_paths", []))[:8]:
        ref = str(path).strip()
        if not ref or ref in seen_lineage_refs:
            continue
        seen_lineage_refs.add(ref)
        lineage.append(
            {
                "kind": "file",
                "ref": ref,
                "why_en": "File drift supplied provenance for witness continuity.",
                "why_ja": "ファイルドリフトが証人連続性の来歴を供給した。",
            }
        )

    if not lineage:
        lineage.append(
            {
                "kind": "idle",
                "ref": "awaiting-touch",
                "why_en": "No recent witness touch; continuity waits for the next trace.",
                "why_ja": "直近の証人接触なし。次の痕跡を待機中。",
            }
        )

    linked_presence_ids = [
        str(item.get("id", ""))
        for item in sorted(
            [row for row in presence_impacts if row.get("id") != "witness_thread"],
            key=lambda row: float(row.get("affected_by", {}).get("clicks", 0.0)),
            reverse=True,
        )
        if str(item.get("id", "")).strip()
    ][:4]

    witness_thread_state = {
        "id": str(witness_meta.get("id", "witness_thread")),
        "en": str(witness_meta.get("en", "Witness Thread")),
        "ja": str(witness_meta.get("ja", "証人の糸")),
        "continuity_index": round(
            _clamp01((click_ratio * 0.54) + (file_ratio * 0.3) + (queue_ratio * 0.16)),
            4,
        ),
        "click_pressure": round(click_ratio, 4),
        "file_pressure": round(file_ratio, 4),
        "linked_presences": linked_presence_ids,
        "lineage": lineage[:6],
        "notes_en": str(
            (witness_impact or {}).get(
                "notes_en",
                "Witness Thread binds touch and file drift into explicit continuity.",
            )
        ),
        "notes_ja": str(
            (witness_impact or {}).get(
                "notes_ja",
                "証人の糸は接触とファイルドリフトを明示的な連続性へ束ねる。",
            )
        ),
    }

    fork_tax = dict(influence.get("fork_tax", {}))
    if not fork_tax:
        fork_tax = {
            "law_en": "Pay the fork tax; annotate every drift with proof.",
            "law_ja": "フォーク税は法。",
            "debt": 0.0,
            "paid": 0.0,
            "balance": 0.0,
            "paid_ratio": 1.0,
        }
    if not str(fork_tax.get("law_ja", "")).strip():
        fork_tax["law_ja"] = "フォーク税は法。"

    ghost = dict(influence.get("ghost", {}))
    ghost.setdefault("id", FILE_SENTINEL_PROFILE["id"])
    ghost.setdefault("en", FILE_SENTINEL_PROFILE["en"])
    ghost.setdefault("ja", FILE_SENTINEL_PROFILE["ja"])
    ghost["auto_commit_pulse"] = round(
        _clamp01(
            float(ghost.get("auto_commit_pulse", 0.0))
            + (file_ratio * 0.12)
            + (queue_ratio * 0.08)
        ),
        4,
    )
    ghost["actions_60s"] = int((file_changes_recent * 0.5) + (queue_event_count * 0.8))
    ghost["status_en"] = str(ghost.get("status_en", "gate idle"))
    ghost["status_ja"] = str(ghost.get("status_ja", "門前で待機中"))

    presence_dynamics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "simulation_budget": {
            "point_limit": int(sim_point_budget),
            "point_limit_max": int(MAX_SIM_POINTS),
            "cpu_utilization": round(resource_cpu_util, 2),
        },
        "click_events": clicks_recent,
        "file_events": file_changes_recent,
        "recent_click_targets": list(influence.get("recent_click_targets", []))[:6],
        "recent_file_paths": list(influence.get("recent_file_paths", []))[:8],
        "resource_heartbeat": resource_heartbeat,
        "river_flow": {
            "unit": "m3/s",
            "rate": river_flow_rate,
            "turbulence": river_turbulence,
        },
        "ghost": ghost,
        "fork_tax": fork_tax,
        "witness_thread": witness_thread_state,
        "presence_impacts": presence_impacts,
    }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(points),
        "audio": int(counts.get("audio", 0)),
        "image": int(counts.get("image", 0)),
        "video": int(counts.get("video", 0)),
        "points": points,
        "file_graph": file_graph
        if isinstance(file_graph, dict)
        else {
            "record": ETA_MU_FILE_GRAPH_RECORD,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "inbox": {
                "record": "ημ.inbox.v1",
                "path": "",
                "pending_count": 0,
                "processed_count": 0,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 0,
                "last_ingested_at": "",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
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
        },
        "crawler_graph": crawler_graph
        if isinstance(crawler_graph, dict)
        else {
            "record": ETA_MU_CRAWLER_GRAPH_RECORD,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": {"endpoint": "", "service": "web-graph-weaver"},
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
        },
        "truth_state": truth_state
        if isinstance(truth_state, dict)
        else _default_truth_state(),
        "logical_graph": logical_graph,
        "pain_field": pain_field,
        "heat_values": heat_values,
        "entities": entity_states,
        "echoes": echo_particles,
        "presence_dynamics": presence_dynamics,
        "myth": myth_summary or {},
        "world": world_summary or {},
    }


def _ollama_base_url() -> str:
    raw = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
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
    model = os.getenv("OLLAMA_MODEL", "qwen3-vl:4b-instruct")
    timeout_s = float(os.getenv("OLLAMA_TIMEOUT_SEC", "30"))
    return endpoint, embeddings_endpoint, model, timeout_s


def _ollama_embed_remote(text: str, model: str | None = None) -> list[float] | None:
    _, endpoint, _, default_timeout = _ollama_endpoint()
    embed_default_model = str(
        os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text") or "nomic-embed-text"
    ).strip()
    chosen_model = (model or embed_default_model or "nomic-embed-text").strip()

    candidates: list[str] = [endpoint]
    if endpoint.endswith("/api/embeddings"):
        candidates.append(f"{endpoint[: -len('/api/embeddings')]}/api/embed")

    seen: set[str] = set()
    for candidate in candidates:
        target = candidate.strip()
        if not target or target in seen:
            continue
        seen.add(target)

        if target.endswith("/api/embed"):
            payload = {
                "model": chosen_model,
                "input": text,
            }
        else:
            payload = {
                "model": chosen_model,
                "prompt": text,
            }

        req = Request(
            target,
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )

        try:
            with urlopen(req, timeout=default_timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
                embedding = _embedding_payload_vector(raw)
                if embedding is not None:
                    return embedding
        except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            continue

    return None


def _ollama_embed(text: str, model: str | None = None) -> list[float] | None:
    backend = _embedding_backend()
    if backend == "tensorflow":
        return _tensorflow_embed(text, model=model)
    if backend == "openvino":
        return _openvino_embed(text, model=model)
    if backend == "auto":
        for candidate in _resource_auto_embedding_order():
            if candidate == "openvino":
                vector = _openvino_embed(text, model=model)
            elif candidate == "tensorflow":
                vector = _tensorflow_embed(text, model=model)
            else:
                vector = _ollama_embed_remote(text, model=model)
            if vector is not None:
                return vector
        return None
    return _ollama_embed_remote(text, model=model)


def _embedding_backend() -> str:
    backend = str(os.getenv("EMBEDDINGS_BACKEND", "ollama") or "ollama").strip().lower()
    if backend in {"openvino", "ollama", "tensorflow", "auto"}:
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
    tf = _load_tensorflow_module()
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
    prompt: str, model: str | None = None, timeout_s: float | None = None
) -> tuple[str | None, str]:
    del timeout_s
    query = str(prompt or "").strip()
    if not query:
        return None, "tensorflow-hash-v1"

    query_vec = _tensorflow_embed(query, model=model)
    if query_vec is None:
        return None, "tensorflow-hash-v1"

    best_line = ""
    best_score = -1.0
    for candidate in _tensorflow_generation_candidates():
        candidate_vec = _tensorflow_embed(candidate, model=model)
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


def _tensor_like_to_list(value: Any) -> Any:
    current = value
    try:
        detach = getattr(current, "detach", None)
        if callable(detach):
            current = detach()
    except Exception:
        pass
    try:
        cpu = getattr(current, "cpu", None)
        if callable(cpu):
            current = cpu()
    except Exception:
        pass
    try:
        numpy_fn = getattr(current, "numpy", None)
        if callable(numpy_fn):
            current = numpy_fn()
    except Exception:
        pass
    to_list_fn = getattr(current, "tolist", None)
    if callable(to_list_fn):
        try:
            return to_list_fn()
        except Exception:
            pass
    if isinstance(current, tuple):
        return [_tensor_like_to_list(item) for item in current]
    return current


def _mean_pool_embedding(
    hidden_state: Any,
    attention_mask: Any,
    *,
    normalize: bool,
) -> list[float] | None:
    hidden_raw = _tensor_like_to_list(hidden_state)
    if not isinstance(hidden_raw, list) or not hidden_raw:
        return None

    token_rows_raw: list[Any]
    first = hidden_raw[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        token_rows_raw = first
    elif isinstance(first, list):
        token_rows_raw = hidden_raw
    else:
        return _normalize_embedding_vector(hidden_raw)

    if not token_rows_raw:
        return None

    first_row = _normalize_embedding_vector(token_rows_raw[0])
    if first_row is None:
        return None
    dim = len(first_row)
    accum = [0.0] * dim
    weight_total = 0.0

    mask_raw = _tensor_like_to_list(attention_mask)
    mask_values: list[float] | None = None
    if isinstance(mask_raw, list) and mask_raw:
        if isinstance(mask_raw[0], list):
            mask_values = [float(_safe_float(str(item), 0.0)) for item in mask_raw[0]]
        else:
            mask_values = [float(_safe_float(str(item), 0.0)) for item in mask_raw]

    for index, row_raw in enumerate(token_rows_raw):
        row = _normalize_embedding_vector(row_raw)
        if row is None or len(row) != dim:
            continue
        weight = 1.0
        if mask_values is not None and index < len(mask_values):
            weight = float(mask_values[index])
        if weight <= 0.0:
            continue
        for dim_idx, value in enumerate(row):
            accum[dim_idx] += value * weight
        weight_total += weight

    if weight_total <= 0.0:
        return None

    vector = [value / weight_total for value in accum]
    if normalize:
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude > 0.0:
            vector = [value / magnitude for value in vector]
    return vector


def _load_openvino_embedder(
    model_id: str,
    device: str,
) -> tuple[Any | None, Any | None]:
    key = f"{model_id}|{device}"
    with _OPENVINO_EMBED_LOCK:
        if (
            str(_OPENVINO_EMBED_RUNTIME.get("key", "")) == key
            and _OPENVINO_EMBED_RUNTIME.get("tokenizer") is not None
            and _OPENVINO_EMBED_RUNTIME.get("model") is not None
        ):
            return (
                _OPENVINO_EMBED_RUNTIME.get("tokenizer"),
                _OPENVINO_EMBED_RUNTIME.get("model"),
            )

    try:
        from transformers import AutoTokenizer
        from optimum.intel.openvino import OVModelForFeatureExtraction
    except Exception as exc:
        with _OPENVINO_EMBED_LOCK:
            _OPENVINO_EMBED_RUNTIME["key"] = key
            _OPENVINO_EMBED_RUNTIME["tokenizer"] = None
            _OPENVINO_EMBED_RUNTIME["model"] = None
            _OPENVINO_EMBED_RUNTIME["error"] = str(exc)
            _OPENVINO_EMBED_RUNTIME["loaded_at"] = datetime.now(
                timezone.utc
            ).isoformat()
        return None, None

    tokenizer = None
    model = None
    err = ""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            model = OVModelForFeatureExtraction.from_pretrained(model_id, export=False)
        except Exception:
            model = OVModelForFeatureExtraction.from_pretrained(model_id, export=True)

        to_fn = getattr(model, "to", None)
        if callable(to_fn):
            try:
                to_fn(device)
            except Exception:
                pass

        compile_fn = getattr(model, "compile", None)
        if callable(compile_fn):
            try:
                compile_fn()
            except Exception:
                pass
    except Exception as exc:
        err = str(exc)
        tokenizer = None
        model = None

    with _OPENVINO_EMBED_LOCK:
        _OPENVINO_EMBED_RUNTIME["key"] = key
        _OPENVINO_EMBED_RUNTIME["tokenizer"] = tokenizer
        _OPENVINO_EMBED_RUNTIME["model"] = model
        _OPENVINO_EMBED_RUNTIME["error"] = err
        _OPENVINO_EMBED_RUNTIME["loaded_at"] = datetime.now(timezone.utc).isoformat()

    return tokenizer, model


def _openvino_embed_http(text: str, model: str | None = None) -> list[float] | None:
    endpoint = str(os.getenv("OPENVINO_EMBED_ENDPOINT", "") or "").strip()
    if not endpoint:
        return None

    timeout_s = max(
        0.1,
        _safe_float(
            str(os.getenv("OPENVINO_EMBED_TIMEOUT_SEC", "45") or "45"),
            45.0,
        ),
    )
    chosen_model = str(model or os.getenv("OPENVINO_EMBED_MODEL", "") or "").strip()

    payload: dict[str, Any] = {"input": text}
    if chosen_model:
        payload["model"] = chosen_model

    req = Request(
        endpoint,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
            return _embedding_payload_vector(raw)
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None


def _openvino_embed_local(text: str, model: str | None = None) -> list[float] | None:
    model_id = str(model or os.getenv("OPENVINO_EMBED_MODEL", "") or "").strip()
    if not model_id:
        return None
    device = str(os.getenv("OPENVINO_EMBED_DEVICE", "NPU") or "NPU").strip() or "NPU"
    max_length = max(
        8,
        min(
            4096,
            int(
                _safe_float(
                    str(os.getenv("OPENVINO_EMBED_MAX_LENGTH", "512") or "512"),
                    512.0,
                )
            ),
        ),
    )
    normalize = _embedding_flag("OPENVINO_EMBED_NORMALIZE", True)

    tokenizer, model_obj = _load_openvino_embedder(model_id, device)
    if tokenizer is None or model_obj is None:
        return None

    try:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        outputs = model_obj(**encoded)
    except Exception:
        return None

    hidden_state = getattr(outputs, "last_hidden_state", None)
    if hidden_state is None:
        if isinstance(outputs, (list, tuple)) and outputs:
            hidden_state = outputs[0]
        elif isinstance(outputs, dict):
            hidden_state = outputs.get("last_hidden_state")
    if hidden_state is None:
        return None

    attention_mask = None
    if isinstance(encoded, dict):
        attention_mask = encoded.get("attention_mask")

    return _mean_pool_embedding(
        hidden_state,
        attention_mask,
        normalize=normalize,
    )


def _openvino_embed(text: str, model: str | None = None) -> list[float] | None:
    from_http = _openvino_embed_http(text, model=model)
    if from_http is not None:
        return from_http
    return _openvino_embed_local(text, model=model)


def _embed_text(text: str, model: str | None = None) -> list[float] | None:
    prompt = str(text or "").strip()
    if not prompt:
        return None

    backend = _embedding_backend()
    if backend == "openvino":
        return _openvino_embed(prompt, model=model)
    if backend == "tensorflow":
        return _tensorflow_embed(prompt, model=model)
    if backend == "auto":
        for candidate in _resource_auto_embedding_order():
            if candidate == "openvino":
                vector = _openvino_embed(prompt, model=model)
            elif candidate == "tensorflow":
                vector = _tensorflow_embed(prompt, model=model)
            else:
                vector = _ollama_embed_remote(prompt, model=model)
            if vector is not None:
                return vector
        return None
    return _ollama_embed_remote(prompt, model=model)


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


def project_vector(embedding: list[float], dims: int = 3) -> list[float]:
    if not embedding:
        return [0.0] * dims

    out = [0.0] * dims
    for i, val in enumerate(embedding):
        out[i % dims] += val

    mag = math.sqrt(sum(x * x for x in out))
    if mag > 0:
        out = [x / mag for x in out]

    return out


def _ollama_generate_text_remote(
    prompt: str, model: str | None = None, timeout_s: float | None = None
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
    try:
        with urlopen(req, timeout=chosen_timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None, chosen_model

    text = str(raw.get("response", "")).strip()
    return (text or None), chosen_model


def _ollama_generate_text(
    prompt: str, model: str | None = None, timeout_s: float | None = None
) -> tuple[str | None, str]:
    backend = _text_generation_backend()
    if backend == "tensorflow":
        return _tensorflow_generate_text(prompt, model=model, timeout_s=timeout_s)
    if backend == "auto":
        chosen_model = model or "auto"
        for candidate in _resource_auto_text_order():
            if candidate == "tensorflow":
                text, chosen_model = _tensorflow_generate_text(
                    prompt,
                    model=model,
                    timeout_s=timeout_s,
                )
            else:
                text, chosen_model = _ollama_generate_text_remote(
                    prompt,
                    model=model,
                    timeout_s=timeout_s,
                )
            if text:
                return text, chosen_model
        return None, chosen_model
    return _ollama_generate_text_remote(prompt, model=model, timeout_s=timeout_s)


def _extract_json_array(text: str) -> list[Any] | None:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


def _ollama_generate_lines() -> tuple[list[dict[str, str]] | None, str | None]:
    fields_block = "\n".join(
        f"- {entry['en']} / {entry['ja']}: {entry['line_en']}"
        for entry in VOICE_LINE_BANK
    )
    prompt = (
        "Rewrite each field into one short spoken-sung line. "
        "Keep tone luminous and mythic. Keep bilingual EN/JA. "
        "Return JSON array only, each item has keys id,line_en,line_ja.\n"
        f"{fields_block}"
    )
    text, model = _ollama_generate_text(prompt)
    if not text:
        return None, model

    parsed = _extract_json_array(text)
    if parsed is None:
        return None, model

    by_id = {entry["id"]: entry for entry in VOICE_LINE_BANK}
    output: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        key = str(item.get("id", "")).strip()
        if key not in by_id:
            continue
        source = by_id[key]
        line_en = str(item.get("line_en", "")).strip() or source["line_en"]
        line_ja = str(item.get("line_ja", "")).strip() or source["line_ja"]
        output.append(
            {
                "id": source["id"],
                "en": source["en"],
                "ja": source["ja"],
                "line_en": line_en,
                "line_ja": line_ja,
            }
        )

    if len(output) < len(VOICE_LINE_BANK):
        return None, model
    output.sort(key=lambda row: row["id"])
    return output, model


def build_voice_lines(mode: str = "canonical") -> dict[str, Any]:
    if mode == "ollama":
        generated, model = _ollama_generate_lines()
        if generated:
            return {
                "mode": "ollama",
                "model": model,
                "lines": generated,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

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
        return f"\u266a {line} \u266a"
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
    ja = str(entity.get("ja", "\u5834"))
    lower = user_text.lower()
    if "fork" in lower or "tax" in lower:
        return f"{en} / {ja}: pay the fork tax, then witness the path."
    if "anchor" in lower:
        return f"{en} / {ja}: anchor first, then sing the decision."
    if turn_index == 0:
        return f"{en} / {ja}: I receive your line and hold it in the ledger."
    return f"{en} / {ja}: I answer the previous voice with witness-light."


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
    if not prior_lines:
        prior_lines = "(none)"

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
        status = "ok"
        chosen_mode = "canonical"
        model_name: str | None = None

        if mode == "ollama":
            prompt = _presence_prompt(
                entity, allowed_tools, history_text, turns, context_block
            )
            generated, model_name = generate_text_fn(prompt)
            if generated:
                base_text = generated
                chosen_mode = "ollama"
                if model_name:
                    models.append(model_name)
            else:
                status = "fallback"
                failures.append(
                    {
                        "presence_id": presence_id,
                        "error_code": "ollama_unavailable",
                        "fallback_used": True,
                    }
                )

        tool_outputs = []
        composed = base_text.strip()
        for tool_name in allowed_tools[:2]:
            updated = _apply_presence_tool(tool_name, composed, user_text)
            if updated != composed:
                tool_outputs.append(
                    {
                        "name": tool_name,
                        "output": updated,
                    }
                )
                composed = updated

        turn = {
            "presence_id": presence_id,
            "presence_en": entity.get("en", "Unknown"),
            "presence_ja": entity.get("ja", "\u672a\u77e5"),
            "mode": chosen_mode,
            "model": model_name,
            "status": status,
            "tools": [item["name"] for item in tool_outputs],
            "tool_outputs": tool_outputs,
            "output": composed,
        }
        turns.append(turn)
        final_lines.append(composed)

    if not final_lines:
        fallback = _chat_fallback_reply(user_text)
        final_lines = [fallback]

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
    utterance: str, idx: int, now: datetime | None = None
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
    utterances: list[str], now: datetime | None = None
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

        # Retrieve memories
        collection = _get_chroma_collection()
        if collection:
            query_embedding = _ollama_embed(user_query)
            if query_embedding:
                try:
                    results = collection.query(
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

        # Consolidation check
        link_density = 0
        if context:
            items = context.get("items", 0)
            if items > 0:
                link_density = context.get("file_count", 0) / items  # approximate

        consolidation_block = ""
        if link_density > 0.8:  # Threshold for Part 67
            consolidation_block = f"CONSOLIDATION STATUS: Active.\n{PART_67_PROLOGUE}"

        prompt = f"{SYSTEM_PROMPT_TEMPLATE.format(context_block=context_block, consolidation_block=consolidation_block)}\n\nConversation:\n{history}\nAssistant:"

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
            WhisperModel = getattr(mod, "WhisperModel")
        except Exception:
            _WHISPER_MODEL = False
            return None

        model_name = os.getenv("FASTER_WHISPER_MODEL", "small")
        device = os.getenv("FASTER_WHISPER_DEVICE", "auto")
        compute_type = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8")
        try:
            _WHISPER_MODEL = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
        except Exception:
            _WHISPER_MODEL = False
            return None
        return _WHISPER_MODEL


def _transcribe_with_faster_whisper(
    path: Path, language: str
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
    path: Path, language: str
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
    audio_bytes: bytes, mime: str = "audio/webm", language: str = "ja"
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


def _normalize_audio_upload_name(file_name: str, mime: str) -> str:
    fallback_suffix = _audio_suffix_for_mime(mime)
    token = Path(file_name or "").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(token).stem).strip("._")
    if not stem:
        stem = f"upload_{int(time.time() * 1000)}"
    suffix = Path(token).suffix.lower() or fallback_suffix
    if suffix not in AUDIO_SUFFIXES and suffix != ".webm":
        suffix = fallback_suffix
    return f"{stem}{suffix}"


def _parse_multipart_form(raw_body: bytes, content_type: str) -> dict[str, Any] | None:
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


def render_index(payload: dict[str, Any], catalog: dict[str, Any]) -> str:
    # Deprecated in favor of templates/index.html
    return ""


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

    class WorldHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_bytes(
            self, body: bytes, content_type: str, status: int = HTTPStatus.OK
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_ws_event(self, payload: dict[str, Any]) -> None:
            frame = websocket_frame_text(json.dumps(payload))
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
                return json.loads(raw.decode("utf-8"))
            except (ValueError, json.JSONDecodeError):
                return None

        def _read_raw_body(self) -> bytes:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return b""
            if length <= 0:
                return b""
            return self.rfile.read(length)

        def _handle_websocket(
            self,
            perspective: str = PROJECTION_DEFAULT_PERSPECTIVE,
        ) -> None:
            perspective_key = normalize_projection_perspective(perspective)
            ws_key = self.headers.get("Sec-WebSocket-Key", "")
            if not ws_key:
                self._send_bytes(
                    b"missing websocket key", "text/plain", HTTPStatus.BAD_REQUEST
                )
                return

            accept = websocket_accept_value(ws_key)
            self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()

            self.connection.settimeout(1.0)
            cached_catalog = collect_catalog(part_root, vault_root)
            cached_signature = _catalog_signature(cached_catalog)
            _, cached_mix = build_mix_stream(cached_catalog, vault_root)
            initial_queue_snapshot = task_queue.snapshot(include_pending=False)
            cached_catalog["task_queue"] = initial_queue_snapshot
            cached_catalog["council"] = council_chamber.snapshot(
                include_decisions=False
            )
            cached_catalog["presence_runtime"] = _INFLUENCE_TRACKER.snapshot(
                queue_snapshot=initial_queue_snapshot,
                part_root=part_root,
            )
            attach_ui_projection(
                cached_catalog,
                perspective=perspective_key,
                queue_snapshot=initial_queue_snapshot,
                influence_snapshot=cached_catalog.get("presence_runtime", {}),
            )
            last_catalog_refresh = time.monotonic()
            last_catalog_broadcast = last_catalog_refresh
            self._send_ws_event(
                {
                    "type": "catalog",
                    "catalog": cached_catalog,
                    "mix": cached_mix,
                }
            )

            while True:
                try:
                    now_monotonic = time.monotonic()
                    if now_monotonic - last_catalog_refresh >= CATALOG_REFRESH_SECONDS:
                        previous_catalog = cached_catalog
                        refreshed_catalog = collect_catalog(part_root, vault_root)
                        refreshed_signature = _catalog_signature(refreshed_catalog)
                        catalog_changed = refreshed_signature != cached_signature
                        delta = _catalog_delta_stats(
                            previous_catalog, refreshed_catalog
                        )
                        if int(delta.get("total_changes", 0)) > 0:
                            _INFLUENCE_TRACKER.record_file_delta(delta)

                        refreshed_queue_snapshot = task_queue.snapshot(
                            include_pending=False
                        )
                        refreshed_catalog["task_queue"] = refreshed_queue_snapshot
                        refreshed_catalog["council"] = council_chamber.snapshot(
                            include_decisions=False
                        )
                        refreshed_catalog["presence_runtime"] = (
                            _INFLUENCE_TRACKER.snapshot(
                                queue_snapshot=refreshed_queue_snapshot,
                                part_root=part_root,
                            )
                        )
                        attach_ui_projection(
                            refreshed_catalog,
                            perspective=perspective_key,
                            queue_snapshot=refreshed_queue_snapshot,
                            influence_snapshot=refreshed_catalog.get(
                                "presence_runtime", {}
                            ),
                        )
                        cached_catalog = refreshed_catalog
                        if catalog_changed:
                            cached_signature = refreshed_signature
                            _, cached_mix = build_mix_stream(cached_catalog, vault_root)

                        if catalog_changed or (
                            now_monotonic - last_catalog_broadcast
                            >= CATALOG_BROADCAST_HEARTBEAT_SECONDS
                        ):
                            self._send_ws_event(
                                {
                                    "type": "catalog",
                                    "catalog": cached_catalog,
                                    "mix": cached_mix,
                                }
                            )
                            last_catalog_broadcast = now_monotonic
                        last_catalog_refresh = now_monotonic

                    current_myth = _MYTH_TRACKER.snapshot(cached_catalog)
                    current_world = _LIFE_TRACKER.snapshot(
                        cached_catalog, current_myth, ENTITY_MANIFEST
                    )
                    current_queue_snapshot = task_queue.snapshot(include_pending=False)
                    current_influence = _INFLUENCE_TRACKER.snapshot(
                        queue_snapshot=current_queue_snapshot,
                        part_root=part_root,
                    )
                    current_simulation = build_simulation_state(
                        cached_catalog,
                        current_myth,
                        current_world,
                        influence_snapshot=current_influence,
                        queue_snapshot=current_queue_snapshot,
                    )
                    current_projection = build_ui_projection(
                        cached_catalog,
                        current_simulation,
                        perspective=perspective_key,
                        queue_snapshot=current_queue_snapshot,
                        influence_snapshot=current_influence,
                    )
                    self._send_ws_event(
                        {
                            "type": "simulation",
                            "simulation": current_simulation,
                            "projection": current_projection,
                        }
                    )
                    time.sleep(max(0.05, SIM_TICK_SECONDS))
                except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                    return
                except Exception:
                    time.sleep(max(0.05, SIM_TICK_SECONDS))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)

            if parsed.path == "/ws":
                params = parse_qs(parsed.query)
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
                params = parse_qs(parsed.query)
                mode = str(params.get("mode", ["canonical"])[0] or "canonical")
                payload_voice = build_voice_lines(
                    "ollama" if mode == "ollama" else "canonical"
                )
                body = json.dumps(payload_voice).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            artifact = resolve_artifact_path(part_root, self.path)
            if artifact is not None:
                mime_type = (
                    mimetypes.guess_type(str(artifact))[0] or "application/octet-stream"
                )
                self._send_bytes(artifact.read_bytes(), mime_type)
                return

            library_item = resolve_library_path(vault_root, self.path)
            if library_item is not None:
                library_member = resolve_library_member(self.path)
                if library_member:
                    member_payload = _read_library_archive_member(
                        library_item,
                        library_member,
                    )
                    if member_payload is None:
                        self._send_bytes(
                            b"not found",
                            "text/plain; charset=utf-8",
                            status=HTTPStatus.NOT_FOUND,
                        )
                        return
                    payload, payload_type = member_payload
                    self._send_bytes(payload, payload_type)
                    return
                mime_type = (
                    mimetypes.guess_type(str(library_item))[0]
                    or "application/octet-stream"
                )
                self._send_bytes(library_item.read_bytes(), mime_type)
                return

            if parsed.path == "/healthz":
                payload = build_world_payload(part_root)
                catalog = collect_catalog(part_root, vault_root)

                file_count = len(catalog["items"])
                link_density = sum(
                    len(x.get("files", []))
                    for x in [
                        load_manifest(p)
                        for p in discover_part_roots(vault_root, part_root)
                    ]
                ) / max(1, file_count)
                entropy = int(time.time() * 1000) % 100

                body = json.dumps(
                    {
                        "ok": True,
                        "status": "alive",
                        "organism": {
                            "spore_count": file_count,
                            "mycelial_density": round(link_density, 2),
                            "pulse_rate": "78bpm",
                            "substrate_entropy": f"{entropy}%",
                            "growth_phase": "fruiting",
                        },
                        "part": payload.get("part"),
                        "items": file_count,
                    }
                ).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/stream/mix.wav":
                catalog = collect_catalog(part_root, vault_root)
                mix_wav, _mix_meta = build_mix_stream(catalog, vault_root)
                if mix_wav:
                    self._send_bytes(mix_wav, "audio/wav")
                    return
                self._send_bytes(
                    b"no wav sources available for mix",
                    "text/plain; charset=utf-8",
                    status=HTTPStatus.NOT_FOUND,
                )
                return

            if parsed.path == "/api/mix":
                catalog = collect_catalog(part_root, vault_root)
                _mix_wav, mix_meta = build_mix_stream(catalog, vault_root)
                body = json.dumps(mix_meta).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/tts":
                params = parse_qs(parsed.query)
                text = str(params.get("text", [""])[0]).strip()
                speed = str(params.get("speed", ["1.0"])[0]).strip()

                if not text:
                    self._send_bytes(
                        json.dumps({"ok": False, "error": "empty text"}).encode(
                            "utf-8"
                        ),
                        "application/json",
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                # Proxy to TTS Sidecar (Tier 0/1)
                tts_url = f"{TTS_BASE_URL}/tts?text={quote(text)}&speed={speed}"
                try:
                    with urlopen(tts_url, timeout=30) as resp:
                        self._send_bytes(resp.read(), "audio/wav")
                except Exception as sidecar_error:
                    fallback_error = ""
                    speed_ratio = max(0.6, min(1.6, _safe_float(speed, 1.0)))
                    words_per_minute = int(round(max(90, min(320, 170 * speed_ratio))))
                    safe_text = text[:600]

                    with tempfile.NamedTemporaryFile(
                        prefix="eta_mu_tts_", suffix=".wav", delete=False
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
                        for cmd in command_candidates:
                            try:
                                result = subprocess.run(
                                    cmd,
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

                    self._send_bytes(
                        json.dumps(
                            {
                                "ok": False,
                                "error": str(sidecar_error),
                                "fallback_error": fallback_error,
                            }
                        ).encode("utf-8"),
                        "application/json",
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return

            if parsed.path == "/api/embeddings/db/status":
                body = json.dumps(_embedding_db_status(vault_root)).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/embeddings/db/list":
                params = parse_qs(parsed.query)
                limit = max(
                    1,
                    min(
                        500,
                        int(
                            _safe_float(
                                str(params.get("limit", ["50"])[0] or "50"),
                                50.0,
                            )
                        ),
                    ),
                )
                include_vectors = str(
                    params.get("include_vectors", ["false"])[0] or "false"
                ).strip().lower() in {"1", "true", "yes", "on"}
                body = json.dumps(
                    _embedding_db_list(
                        vault_root,
                        limit=limit,
                        include_vectors=include_vectors,
                    )
                ).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/memories":
                collection = _get_chroma_collection()
                mems = []
                if collection:
                    try:
                        results = collection.get(limit=10)  # last 10
                        ids = results.get("ids", [])
                        docs = results.get("documents", [])
                        metas = results.get("metadatas", [])
                        for i in range(len(ids)):
                            mems.append(
                                {"id": ids[i], "text": docs[i], "metadata": metas[i]}
                            )
                    except Exception:
                        pass
                body = json.dumps({"ok": True, "memories": mems}).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/memories":
                collection = _get_chroma_collection()
                mems = []
                if collection:
                    try:
                        results = collection.get(limit=10)  # last 10
                        ids = results.get("ids", [])
                        docs = results.get("documents", [])
                        metas = results.get("metadatas", [])
                        for i in range(len(ids)):
                            mems.append(
                                {"id": ids[i], "text": docs[i], "metadata": metas[i]}
                            )
                    except Exception:
                        pass
                body = json.dumps({"ok": True, "memories": mems}).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/catalog":
                params = parse_qs(parsed.query)
                perspective = normalize_projection_perspective(
                    str(
                        params.get(
                            "perspective",
                            [PROJECTION_DEFAULT_PERSPECTIVE],
                        )[0]
                        or PROJECTION_DEFAULT_PERSPECTIVE
                    )
                )
                catalog = collect_catalog(part_root, vault_root)
                queue_snapshot = task_queue.snapshot(include_pending=False)
                catalog["task_queue"] = queue_snapshot
                catalog["council"] = council_chamber.snapshot(include_decisions=False)
                catalog_resource = _resource_monitor_snapshot(part_root=part_root)
                _INFLUENCE_TRACKER.record_resource_heartbeat(
                    catalog_resource,
                    source="api.catalog",
                )
                _ingest_resource_heartbeat_memory(catalog_resource)
                influence_snapshot = _INFLUENCE_TRACKER.snapshot(
                    queue_snapshot=queue_snapshot,
                    part_root=part_root,
                )
                catalog["presence_runtime"] = influence_snapshot
                attach_ui_projection(
                    catalog,
                    perspective=perspective,
                    queue_snapshot=queue_snapshot,
                    influence_snapshot=influence_snapshot,
                )
                body = json.dumps(catalog).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/zips":
                params = parse_qs(parsed.query)
                member_limit = int(
                    _safe_float(
                        str(params.get("member_limit", ["220"])[0] or "220"), 220.0
                    )
                )
                body = json.dumps(
                    collect_zip_catalog(
                        part_root,
                        vault_root,
                        member_limit=member_limit,
                    )
                ).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/pi/archive":
                catalog = collect_catalog(part_root, vault_root)
                queue_snapshot = task_queue.snapshot(include_pending=False)
                archive = build_pi_archive_payload(
                    part_root,
                    vault_root,
                    catalog=catalog,
                    queue_snapshot=queue_snapshot,
                )
                body = json.dumps({"ok": True, "archive": archive}).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/ui/projection":
                params = parse_qs(parsed.query)
                perspective = normalize_projection_perspective(
                    str(
                        params.get(
                            "perspective",
                            [PROJECTION_DEFAULT_PERSPECTIVE],
                        )[0]
                        or PROJECTION_DEFAULT_PERSPECTIVE
                    )
                )
                catalog = collect_catalog(part_root, vault_root)
                queue_snapshot = task_queue.snapshot(include_pending=False)
                catalog["task_queue"] = queue_snapshot
                catalog["council"] = council_chamber.snapshot(include_decisions=False)
                influence_snapshot = _INFLUENCE_TRACKER.snapshot(
                    queue_snapshot=queue_snapshot,
                    part_root=part_root,
                )
                catalog["presence_runtime"] = influence_snapshot
                myth_summary = _MYTH_TRACKER.snapshot(catalog)
                world_summary = _LIFE_TRACKER.snapshot(
                    catalog, myth_summary, ENTITY_MANIFEST
                )
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
                body = json.dumps(
                    {
                        "ok": True,
                        "projection": projection,
                        "default_perspective": PROJECTION_DEFAULT_PERSPECTIVE,
                        "perspectives": projection_perspective_options(),
                    }
                ).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/task/queue":
                body = json.dumps(
                    {
                        "ok": True,
                        "queue": task_queue.snapshot(include_pending=True),
                    }
                ).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/council":
                params = parse_qs(parsed.query)
                limit = max(
                    1,
                    min(
                        128,
                        int(
                            _safe_float(
                                str(params.get("limit", ["16"])[0] or "16"), 16.0
                            )
                        ),
                    ),
                )
                body = json.dumps(
                    {
                        "ok": True,
                        "council": council_chamber.snapshot(
                            include_decisions=True,
                            limit=limit,
                        ),
                    }
                ).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/study":
                params = parse_qs(parsed.query)
                limit = max(
                    1,
                    min(
                        128,
                        int(
                            _safe_float(
                                str(params.get("limit", ["16"])[0] or "16"), 16.0
                            )
                        ),
                    ),
                )
                include_truth_state = str(
                    params.get("include_truth", ["false"])[0] or "false"
                ).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                queue_snapshot = task_queue.snapshot(include_pending=True)
                council_snapshot = council_chamber.snapshot(
                    include_decisions=True,
                    limit=limit,
                )
                drift_payload = build_drift_scan_payload(part_root, vault_root)
                resource_snapshot = _resource_monitor_snapshot(part_root=part_root)
                _INFLUENCE_TRACKER.record_resource_heartbeat(
                    resource_snapshot,
                    source="api.study",
                )
                _ingest_resource_heartbeat_memory(resource_snapshot)
                truth_gate_blocked: bool | None = None
                if include_truth_state:
                    try:
                        catalog = collect_catalog(part_root, vault_root)
                        truth = (
                            catalog.get("truth_state", {})
                            if isinstance(catalog.get("truth_state"), dict)
                            else {}
                        )
                        gate = truth.get("gate", {}) if isinstance(truth, dict) else {}
                        if isinstance(gate, dict):
                            truth_gate_blocked = bool(gate.get("blocked", False))
                    except Exception:
                        truth_gate_blocked = None

                payload = build_study_snapshot(
                    part_root,
                    vault_root,
                    queue_snapshot=queue_snapshot,
                    council_snapshot=council_snapshot,
                    drift_payload=drift_payload,
                    truth_gate_blocked=truth_gate_blocked,
                    resource_snapshot=resource_snapshot,
                )
                body = json.dumps(payload).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/study/history":
                params = parse_qs(parsed.query)
                limit = max(
                    1,
                    min(
                        STUDY_SNAPSHOT_HISTORY_LIMIT,
                        int(
                            _safe_float(
                                str(params.get("limit", ["16"])[0] or "16"),
                                16.0,
                            )
                        ),
                    ),
                )
                events = _load_study_snapshot_events(vault_root, limit=limit)
                payload = {
                    "ok": True,
                    "record": "ημ.study-history.v1",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "path": str(_study_snapshot_log_path(vault_root)),
                    "count": len(events),
                    "events": events,
                }
                body = json.dumps(payload).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/resource/heartbeat":
                heartbeat = _resource_monitor_snapshot(part_root=part_root)
                _INFLUENCE_TRACKER.record_resource_heartbeat(
                    heartbeat,
                    source="api.resource.heartbeat",
                )
                _ingest_resource_heartbeat_memory(heartbeat)
                queue_snapshot = task_queue.snapshot(include_pending=False)
                runtime_snapshot = _INFLUENCE_TRACKER.snapshot(
                    queue_snapshot=queue_snapshot,
                    part_root=part_root,
                )
                body = json.dumps(
                    {
                        "ok": True,
                        "record": "ημ.resource-heartbeat.response.v1",
                        "heartbeat": heartbeat,
                        "runtime": runtime_snapshot,
                    }
                ).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/named-fields":
                body = json.dumps(
                    {
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "mode": "gradients",
                        "named_fields": build_named_field_overlays(ENTITY_MANIFEST),
                    }
                ).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/simulation":
                params = parse_qs(parsed.query)
                perspective = normalize_projection_perspective(
                    str(
                        params.get(
                            "perspective",
                            [PROJECTION_DEFAULT_PERSPECTIVE],
                        )[0]
                        or PROJECTION_DEFAULT_PERSPECTIVE
                    )
                )
                catalog = collect_catalog(part_root, vault_root)
                queue_snapshot = task_queue.snapshot(include_pending=False)
                catalog["task_queue"] = queue_snapshot
                catalog["council"] = council_chamber.snapshot(include_decisions=False)
                myth_summary = _MYTH_TRACKER.snapshot(catalog)
                world_summary = _LIFE_TRACKER.snapshot(
                    catalog, myth_summary, ENTITY_MANIFEST
                )
                influence_snapshot = _INFLUENCE_TRACKER.snapshot(
                    queue_snapshot=queue_snapshot,
                    part_root=part_root,
                )
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
                body = json.dumps(simulation).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/myth":
                catalog = collect_catalog(part_root, vault_root)
                myth_summary = _MYTH_TRACKER.snapshot(catalog)
                body = json.dumps(myth_summary).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/world":
                catalog = collect_catalog(part_root, vault_root)
                myth_summary = _MYTH_TRACKER.snapshot(catalog)
                world_summary = _LIFE_TRACKER.snapshot(
                    catalog, myth_summary, ENTITY_MANIFEST
                )
                body = json.dumps(world_summary).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path != "/":
                # Check if file exists in frontend/dist/assets
                dist_root = part_root / "frontend" / "dist"
                requested_path = parsed.path.lstrip("/")
                candidate = (dist_root / requested_path).resolve()
                if (
                    dist_root in candidate.parents
                    and candidate.exists()
                    and candidate.is_file()
                ):
                    mime_type = (
                        mimetypes.guess_type(candidate)[0] or "application/octet-stream"
                    )
                    self._send_bytes(candidate.read_bytes(), mime_type)
                    return

                body = b"not found"
                self._send_bytes(
                    body, "text/plain; charset=utf-8", status=HTTPStatus.NOT_FOUND
                )
                return

            # Serve index.html from frontend/dist
            index_path = part_root / "frontend" / "dist" / "index.html"
            if index_path.exists():
                self._send_bytes(index_path.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send_bytes(
                    b"Frontend not built",
                    "text/plain; charset=utf-8",
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
            return

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
                jsonl = "\n".join(json.dumps(row) for row in rows)
                body = json.dumps(
                    {
                        "ok": True,
                        "rows": rows,
                        "jsonl": f"{jsonl}\n" if jsonl else "",
                    }
                ).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/embeddings/db/upsert":
                req = self._read_json_body() or {}
                entry_id = str(req.get("id", "") or "").strip()
                text = str(req.get("text", "") or "")
                metadata_raw = req.get("metadata", {})
                metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
                model = str(req.get("model", "") or "").strip() or None

                embedding = _normalize_embedding_vector(req.get("embedding"))
                if embedding is None and text.strip():
                    embedding = _ollama_embed(text, model=model)
                    embedding = _normalize_embedding_vector(embedding)

                if embedding is None:
                    self._send_bytes(
                        json.dumps(
                            {
                                "ok": False,
                                "error": "missing or invalid embedding; provide embedding or text with reachable ollama embeddings",
                            }
                        ).encode("utf-8"),
                        "application/json; charset=utf-8",
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                result = _embedding_db_upsert(
                    vault_root,
                    entry_id=entry_id,
                    text=text,
                    embedding=embedding,
                    metadata=metadata,
                    model=model,
                )
                status = HTTPStatus.OK
                if not bool(result.get("ok", False)):
                    status = HTTPStatus.BAD_REQUEST
                self._send_bytes(
                    json.dumps(result).encode("utf-8"),
                    "application/json; charset=utf-8",
                    status=status,
                )
                return

            if parsed.path == "/api/embeddings/db/query":
                req = self._read_json_body() or {}
                query_text = str(req.get("query", req.get("text", "")) or "").strip()
                model = str(req.get("model", "") or "").strip() or None
                query_embedding = _normalize_embedding_vector(req.get("embedding"))
                if query_embedding is None and query_text:
                    query_embedding = _ollama_embed(query_text, model=model)
                    query_embedding = _normalize_embedding_vector(query_embedding)

                if query_embedding is None:
                    self._send_bytes(
                        json.dumps(
                            {
                                "ok": False,
                                "error": "missing or invalid query embedding; provide embedding or query text with reachable ollama embeddings",
                            }
                        ).encode("utf-8"),
                        "application/json; charset=utf-8",
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                top_k = max(
                    1,
                    min(
                        100,
                        int(_safe_float(str(req.get("top_k", 5) or 5), 5.0)),
                    ),
                )
                min_score = _safe_float(str(req.get("min_score", -1.0) or -1.0), -1.0)
                include_vectors = str(
                    req.get("include_vectors", "false") or "false"
                ).strip().lower() in {"1", "true", "yes", "on"}

                result = _embedding_db_query(
                    vault_root,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    min_score=min_score,
                    include_vectors=include_vectors,
                )
                status = HTTPStatus.OK
                if not bool(result.get("ok", False)):
                    status = HTTPStatus.BAD_REQUEST
                self._send_bytes(
                    json.dumps(result).encode("utf-8"),
                    "application/json; charset=utf-8",
                    status=status,
                )
                return

            if parsed.path == "/api/embeddings/db/delete":
                req = self._read_json_body() or {}
                result = _embedding_db_delete(
                    vault_root,
                    entry_id=str(req.get("id", "") or "").strip(),
                )
                status = (
                    HTTPStatus.OK
                    if bool(result.get("ok", False))
                    else HTTPStatus.NOT_FOUND
                )
                self._send_bytes(
                    json.dumps(result).encode("utf-8"),
                    "application/json; charset=utf-8",
                    status=status,
                )
                return

            if parsed.path == "/api/input-stream":
                req = self._read_json_body() or {}
                stream_type = str(req.get("type", "unknown"))
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
                    _ingest_resource_heartbeat_memory(data)

                input_str = f"{stream_type}: {json.dumps(data)}"
                embedding = _ollama_embed(input_str)
                force_vector = (
                    project_vector(embedding) if embedding else [0.0, 0.0, 0.0]
                )

                collection = _get_chroma_collection()
                if collection and embedding:
                    try:
                        mem_id = f"mem_{int(time.time() * 1000)}"
                        collection.add(
                            ids=[mem_id],
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

                queue_snapshot = task_queue.snapshot(include_pending=False)
                influence_snapshot = _INFLUENCE_TRACKER.snapshot(
                    queue_snapshot=queue_snapshot,
                    part_root=part_root,
                )
                council_catalog: dict[str, Any] = {}
                if stream_type in {"file_changed", "file_added", "file_removed"}:
                    try:
                        council_catalog = collect_catalog(part_root, vault_root)
                    except Exception:
                        council_catalog = {}
                council_result = council_chamber.consider_event(
                    event_type=stream_type,
                    data=data if isinstance(data, dict) else {"value": data},
                    catalog=council_catalog,
                    influence_snapshot=influence_snapshot,
                )

                body = json.dumps(
                    {
                        "ok": True,
                        "force": force_vector,
                        "embedding_dim": len(embedding) if embedding else 0,
                        "resource": influence_snapshot.get("resource_heartbeat", {}),
                        "council": council_result,
                    }
                ).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/upload":
                content_type = str(self.headers.get("Content-Type", ""))
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
                        self._send_bytes(
                            json.dumps(
                                {
                                    "ok": False,
                                    "error": "missing multipart file field 'file'",
                                }
                            ).encode("utf-8"),
                            "application/json; charset=utf-8",
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return

                    file_name = str(file_field.get("filename", "upload.mp3"))
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
                    file_name = str(req.get("name", "upload.mp3"))
                    mime = str(req.get("mime", "audio/mpeg") or "audio/mpeg")
                    language = str(req.get("language", "ja") or "ja")
                    file_b64 = str(req.get("base64", "")).strip()
                    if file_b64:
                        try:
                            file_bytes = base64.b64decode(file_b64, validate=False)
                        except (ValueError, OSError):
                            file_bytes = b""

                if not file_bytes:
                    self._send_bytes(
                        json.dumps(
                            {"ok": False, "error": "missing or invalid audio payload"}
                        ).encode("utf-8"),
                        "application/json; charset=utf-8",
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                try:
                    safe_name = _normalize_audio_upload_name(file_name, mime)
                    save_path = part_root / "artifacts" / "audio" / safe_name
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    save_path.write_bytes(file_bytes)

                    result = transcribe_audio_bytes(
                        file_bytes,
                        mime=mime,
                        language=language,
                    )
                    transcribed_text = str(result.get("text", "")).strip()

                    collection = _get_chroma_collection()
                    if collection and transcribed_text:
                        memory_text = (
                            f"The Weaver learned a new frequency: {transcribed_text}"
                        )
                        memory_payload: dict[str, Any] = {
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
                        embed = _ollama_embed(memory_text)
                        if embed:
                            memory_payload["embeddings"] = [embed]
                        try:
                            collection.add(**memory_payload)
                        except Exception:
                            pass

                    body = json.dumps(
                        {
                            "ok": True,
                            "status": "learned" if transcribed_text else "stored",
                            "engine": result.get("engine", "none"),
                            "text": transcribed_text,
                            "transcription_error": result.get("error"),
                            "url": f"/artifacts/audio/{safe_name}",
                        }
                    ).encode("utf-8")
                    self._send_bytes(body, "application/json; charset=utf-8")
                except Exception as exc:
                    self._send_bytes(
                        json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"),
                        "application/json; charset=utf-8",
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
                return

            if parsed.path == "/api/handoff":
                collection = _get_chroma_collection()
                handoff_report = []
                if collection:
                    try:
                        results = collection.get(limit=50)
                        docs = results.get("documents", [])
                        handoff_report.append("# MISSION HANDOFF / 引き継ぎ")
                        handoff_report.append(
                            f"Generated at: {datetime.now(timezone.utc).isoformat()}"
                        )
                        handoff_report.append("\n## RECENT MEMORY ECHOES")
                        for d in docs[-10:]:
                            handoff_report.append(f"- {d}")
                    except Exception:
                        pass

                try:
                    cons_path = part_root / "world_state" / "constraints.md"
                    handoff_report.append("\n## ACTIVE CONSTRAINTS")
                    handoff_report.append(cons_path.read_text())
                except Exception:
                    pass

                body = "\n".join(handoff_report).encode("utf-8")
                self._send_bytes(body, "text/markdown; charset=utf-8")
                return

            if parsed.path == "/api/witness":
                req = self._read_json_body() or {}
                event_type = str(req.get("type", "touch"))
                target = str(req.get("target", "unknown"))

                timestamp = datetime.now(timezone.utc).isoformat()
                entry = {
                    "timestamp": timestamp,
                    "event": event_type,
                    "target": target,
                    "witness_id": hashlib.sha1(str(time.time()).encode()).hexdigest()[
                        :8
                    ],
                }

                ledger_path = part_root / "world_state" / "decision_ledger.jsonl"
                with open(ledger_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                _INFLUENCE_TRACKER.record_witness(event_type=event_type, target=target)

                collection = _get_chroma_collection()
                if collection:
                    try:
                        collection.add(
                            ids=[f"wit_{int(time.time() * 1000)}"],
                            embeddings=[
                                _ollama_embed(f"Witnessed {target} via {event_type}")
                            ],
                            metadatas=[{"type": "witness", "target": target}],
                            documents=[f"The Witness touched {target} at {timestamp}"],
                        )
                    except Exception:
                        pass

                body = json.dumps(
                    {
                        "ok": True,
                        "status": "recorded",
                        "collapse_id": entry["witness_id"],
                    }
                ).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/fork-tax/pay":
                req = self._read_json_body() or {}
                amount_raw = req.get("amount", 1.0)
                try:
                    amount = float(amount_raw)
                except (TypeError, ValueError):
                    self._send_bytes(
                        json.dumps(
                            {
                                "ok": False,
                                "error": "amount must be numeric",
                            }
                        ).encode("utf-8"),
                        "application/json; charset=utf-8",
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                source = str(req.get("source", "command-center") or "command-center")
                target = str(
                    req.get("target", "fork_tax_canticle") or "fork_tax_canticle"
                )
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
                    "witness_id": hashlib.sha1(str(time.time()).encode()).hexdigest()[
                        :8
                    ],
                }
                ledger_path = part_root / "world_state" / "decision_ledger.jsonl"
                with open(ledger_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

                queue_snapshot = task_queue.snapshot(include_pending=False)
                runtime_snapshot = _INFLUENCE_TRACKER.snapshot(
                    queue_snapshot=queue_snapshot,
                    part_root=part_root,
                )
                body = json.dumps(
                    {
                        "ok": True,
                        "status": "recorded",
                        "payment": payment,
                        "runtime": runtime_snapshot,
                    }
                ).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
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
                        val = str(item).strip()
                        if val:
                            presence_ids.append(val)
                messages: list[dict[str, Any]]
                if isinstance(messages_raw, list):
                    messages = [item for item in messages_raw if isinstance(item, dict)]
                else:
                    messages = []

                context = build_world_payload(part_root)
                heartbeat = _resource_monitor_snapshot(part_root=part_root)
                _INFLUENCE_TRACKER.record_resource_heartbeat(
                    heartbeat,
                    source="api.chat",
                )
                _ingest_resource_heartbeat_memory(heartbeat)
                context["resource_heartbeat"] = heartbeat
                response = build_chat_reply(
                    messages,
                    mode=mode,
                    context=context,
                    multi_entity=multi_entity,
                    presence_ids=presence_ids,
                )
                body = json.dumps(response).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/world/interact":
                req = self._read_json_body() or {}
                person_id = str(req.get("person_id", "")).strip()
                action = str(req.get("action", "speak") or "speak")

                catalog = collect_catalog(part_root, vault_root)
                myth_summary = _MYTH_TRACKER.snapshot(catalog)
                world_summary = _LIFE_TRACKER.snapshot(
                    catalog, myth_summary, ENTITY_MANIFEST
                )
                result = _LIFE_INTERACTION_BUILDER(world_summary, person_id, action)

                if result.get("ok"):
                    timestamp = datetime.now(timezone.utc).isoformat()
                    entry = {
                        "timestamp": timestamp,
                        "event": "world_interact",
                        "target": result.get("presence", {}).get("id", "unknown"),
                        "person_id": person_id,
                        "action": action,
                        "witness_id": hashlib.sha1(
                            str(time.time()).encode()
                        ).hexdigest()[:8],
                    }
                    ledger_path = part_root / "world_state" / "decision_ledger.jsonl"
                    with open(ledger_path, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                    _INFLUENCE_TRACKER.record_witness(
                        event_type="world_interact",
                        target=str(entry.get("target", "unknown")),
                    )

                body = json.dumps(result).encode("utf-8")
                status = HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST
                self._send_bytes(body, "application/json; charset=utf-8", status=status)
                return

            if parsed.path == "/api/presence/say":
                req = self._read_json_body() or {}
                text = str(req.get("text", "")).strip()
                presence_id = str(req.get("presence_id", "")).strip()
                catalog = collect_catalog(part_root, vault_root)
                queue_snapshot = task_queue.snapshot(include_pending=False)
                heartbeat = _resource_monitor_snapshot(part_root=part_root)
                _INFLUENCE_TRACKER.record_resource_heartbeat(
                    heartbeat,
                    source="api.presence.say",
                )
                _ingest_resource_heartbeat_memory(heartbeat)
                catalog["presence_runtime"] = _INFLUENCE_TRACKER.snapshot(
                    queue_snapshot=queue_snapshot,
                    part_root=part_root,
                )
                catalog["resource_heartbeat"] = heartbeat
                payload = build_presence_say_payload(catalog, text, presence_id)
                body = json.dumps(payload).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/drift/scan":
                payload = build_drift_scan_payload(part_root, vault_root)
                body = json.dumps(payload).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/push-truth/dry-run":
                payload = build_push_truth_dry_run_payload(part_root, vault_root)
                body = json.dumps(payload).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            if parsed.path == "/api/pi/archive/portable":
                req = self._read_json_body() or {}
                archive_raw = req.get("archive")
                archive = archive_raw if isinstance(archive_raw, dict) else {}
                payload = validate_pi_archive_portable(archive)
                body = json.dumps(payload).encode("utf-8")
                status = HTTPStatus.OK if payload.get("ok") else HTTPStatus.BAD_REQUEST
                self._send_bytes(body, "application/json; charset=utf-8", status=status)
                return

            if parsed.path == "/api/task/enqueue":
                req = self._read_json_body() or {}
                kind = str(req.get("kind", "runtime-task") or "runtime-task").strip()
                payload_raw = req.get("payload", {})
                payload = (
                    payload_raw
                    if isinstance(payload_raw, dict)
                    else {"value": payload_raw}
                )
                dedupe_key = str(req.get("dedupe_key", "")).strip()
                owner = str(req.get("owner", "Err") or "Err").strip()
                refs_raw = req.get("refs", [])
                refs = (
                    [str(item).strip() for item in refs_raw if str(item).strip()]
                    if isinstance(refs_raw, list)
                    else []
                )
                result = task_queue.enqueue(
                    kind=kind,
                    payload=payload,
                    dedupe_key=dedupe_key,
                    owner=owner,
                    refs=refs,
                )
                body = json.dumps(result).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
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
                    or vote_value
                    not in {
                        "yes",
                        "no",
                        "abstain",
                    }
                ):
                    body = json.dumps(
                        {
                            "ok": False,
                            "error": "invalid_request",
                            "required": ["decision_id", "member_id", "vote"],
                            "allowed_votes": ["yes", "no", "abstain"],
                        }
                    ).encode("utf-8")
                    self._send_bytes(
                        body,
                        "application/json; charset=utf-8",
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                result = council_chamber.vote(
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
                body = json.dumps(result).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8", status=status)
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

                queue_snapshot = task_queue.snapshot(include_pending=True)
                council_snapshot = council_chamber.snapshot(
                    include_decisions=True,
                    limit=COUNCIL_DECISION_HISTORY_LIMIT,
                )
                drift_payload = build_drift_scan_payload(part_root, vault_root)
                resource_snapshot = _resource_monitor_snapshot(part_root=part_root)
                _INFLUENCE_TRACKER.record_resource_heartbeat(
                    resource_snapshot,
                    source="api.study.export",
                )
                _ingest_resource_heartbeat_memory(resource_snapshot)
                truth_gate_blocked: bool | None = None
                if include_truth_state:
                    try:
                        catalog = collect_catalog(part_root, vault_root)
                        truth = (
                            catalog.get("truth_state", {})
                            if isinstance(catalog.get("truth_state"), dict)
                            else {}
                        )
                        gate = truth.get("gate", {}) if isinstance(truth, dict) else {}
                        if isinstance(gate, dict):
                            truth_gate_blocked = bool(gate.get("blocked", False))
                    except Exception:
                        truth_gate_blocked = None

                result = export_study_snapshot(
                    part_root,
                    vault_root,
                    queue_snapshot=queue_snapshot,
                    council_snapshot=council_snapshot,
                    drift_payload=drift_payload,
                    truth_gate_blocked=truth_gate_blocked,
                    resource_snapshot=resource_snapshot,
                    label=label,
                    owner=owner,
                    refs=refs,
                    host=f"{host}:{port}",
                    manifest="manifest.lith",
                )
                body = json.dumps(result).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
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
                result = task_queue.dequeue(owner=owner, refs=refs)
                status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
                body = json.dumps(result).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8", status=status)
                return

            if parsed.path == "/api/transcribe":
                req = self._read_json_body() or {}
                audio_b64 = str(req.get("audio_base64", "")).strip()
                mime = str(req.get("mime", "audio/webm") or "audio/webm")
                language = str(req.get("language", "ja") or "ja")
                if not audio_b64:
                    body = json.dumps(
                        {
                            "ok": False,
                            "engine": "none",
                            "text": "",
                            "error": "missing audio_base64",
                        }
                    ).encode("utf-8")
                    self._send_bytes(
                        body,
                        "application/json; charset=utf-8",
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                try:
                    audio_bytes = base64.b64decode(
                        audio_b64.encode("utf-8"), validate=False
                    )
                except (ValueError, OSError):
                    body = json.dumps(
                        {
                            "ok": False,
                            "engine": "none",
                            "text": "",
                            "error": "invalid base64 audio",
                        }
                    ).encode("utf-8")
                    self._send_bytes(
                        body,
                        "application/json; charset=utf-8",
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                result = transcribe_audio_bytes(
                    audio_bytes, mime=mime, language=language
                )
                body = json.dumps(result).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return

            body = json.dumps({"ok": False, "error": "not found"}).encode("utf-8")
            self._send_bytes(
                body,
                "application/json; charset=utf-8",
                status=HTTPStatus.NOT_FOUND,
            )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[world-web] {self.address_string()} - {format % args}")

    return WorldHandler


def serve(
    part_root: Path, vault_root: Path, host: str, port: int, open_browser_flag: bool
) -> None:
    _ensure_weaver_service(part_root, host)
    handler_cls = make_handler(part_root, vault_root, host=host, port=port)
    server = ThreadingHTTPServer((host, port), handler_cls)
    url = f"http://{host}:{port}/"
    print(f"world daemon ready: {url}")
    if open_browser_flag:
        webbrowser.open(url)
    server.serve_forever()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve eta-mu world state in a browser"
    )
    parser.add_argument(
        "--part-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--vault-root", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    part_root = args.part_root.resolve()
    vault_root = args.vault_root.resolve() if args.vault_root else part_root.parent
    serve(part_root, vault_root, args.host, args.port, args.open_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
