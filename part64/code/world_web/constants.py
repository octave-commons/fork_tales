from __future__ import annotations
import os
import threading
from typing import Any
from pathlib import Path

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

# Media and Format Suffixes
AUDIO_SUFFIXES = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}

# WebSocket
WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Simulation and Catalog Parameters
MAX_SIM_POINTS = max(256, int(os.getenv("MAX_SIM_POINTS", "2048") or "2048"))
SIM_TICK_SECONDS = max(0.001, float(os.getenv("SIM_TICK_SECONDS", "0.2") or "0.2"))
CATALOG_REFRESH_SECONDS = float(os.getenv("CATALOG_REFRESH_SECONDS", "1.5") or "1.5")
CATALOG_BROADCAST_HEARTBEAT_SECONDS = float(
    os.getenv("CATALOG_BROADCAST_HEARTBEAT_SECONDS", "6.0") or "6.0"
)
MYCELIAL_ECHO_CACHE_SECONDS = max(
    0.2,
    float(os.getenv("MYCELIAL_ECHO_CACHE_SECONDS", "2.5") or "2.5"),
)


def simulation_tick_seconds() -> float:
    return max(0.001, float(SIM_TICK_SECONDS))


# Daimoi Dynamics
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
        "id": "daimo:resource",
        "name": "Resource Daimoi",
        "ctx": "資",
        "base_budget": 12.0,
        "w": 1.05,
        "temperature": 0.28,
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

# Named Fields
CANONICAL_NAMED_FIELD_IDS = (
    "receipt_river",
    "witness_thread",
    "fork_tax_canticle",
    "mage_of_receipts",
    "keeper_of_receipts",
    "anchor_registry",
    "gates_of_truth",
    "file_sentinel",
    "change_fog",
    "path_ward",
    "manifest_lith",
    "resolution_weaver",
    "core_pulse",
)

# Eta Mu Storage
ETA_MU_INBOX_DIRNAME = ".ημ"
ETA_MU_KNOWLEDGE_INDEX_REL = ".opencode/runtime/eta_mu_knowledge.v1.jsonl"
ETA_MU_KNOWLEDGE_ARCHIVE_REL = ".opencode/knowledge/archive"
ETA_MU_REGISTRY_REL = ".Π/ημ_registry.jsonl"
ETA_MU_EMBEDDINGS_DB_REL = ".opencode/runtime/eta_mu_embeddings.v1.jsonl"
ETA_MU_GRAPH_MOVES_REL = ".opencode/runtime/eta_mu_graph_moves.v1.json"
PRESENCE_ACCOUNTS_LOG_REL = ".opencode/runtime/presence_accounts.v1.jsonl"
SIMULATION_METADATA_LOG_REL = ".opencode/runtime/simulation_metadata.v1.jsonl"
IMAGE_COMMENTS_LOG_REL = ".opencode/runtime/image_comments.v1.jsonl"
WIKIMEDIA_STREAM_LOG_REL = ".opencode/runtime/wikimedia_stream.v1.jsonl"
NWS_ALERTS_LOG_REL = ".opencode/runtime/nws_alerts.v1.jsonl"
SWPC_ALERTS_LOG_REL = ".opencode/runtime/swpc_alerts.v1.jsonl"
GIBS_LAYERS_LOG_REL = ".opencode/runtime/gibs_layers.v1.jsonl"
EONET_EVENTS_LOG_REL = ".opencode/runtime/eonet_events.v1.jsonl"
EMSC_STREAM_LOG_REL = ".opencode/runtime/emsc_stream.v1.jsonl"

# ============================================================================
# CANONICAL UNIFIED MODEL RECORD TYPES (v2)
# ============================================================================
#
# The unified model has exactly four primitive types:
# - Presence: AI physics-based agent with spec embedding, need, priority, mass
# - Nexus: Graph node/particle representing a resource with embedding, capacity, demand, role
# - Daimoi: Free particle with carrier embedding, seed embedding, type distribution, owner
# - Field: Shared global scalar field (demand, flow, entropy, graph)
#
# See specs/drafts/part64-deep-research-09-unified-nexus-graph.md
# See specs/drafts/part64-deep-research-10-shared-fields-daimoi-dynamics.md
# See specs/drafts/part64-deep-research-11-model-audit-alien-concepts.md

# Canonical unified graph (replaces file_graph, crawler_graph, logical_graph)
NEXUS_GRAPH_RECORD = "ημ.nexus-graph.v1"
NEXUS_GRAPH_SCHEMA_VERSION = "nexus.graph.v1"

# Canonical daimon (replaces multiple particle types)
DAIMON_RECORD = "ημ.daimon.v1"
DAIMON_SCHEMA_VERSION = "daimon.v1"
DAIMOI_PACKET_RECORD = "ημ.daimoi-packet.v1"

# Canonical presence (unified agent type)
PRESENCE_RECORD = "ημ.presence.v1"
PRESENCE_SCHEMA_VERSION = "presence.v1"
USER_PRESENCE_ID = "presence.user.operator"
USER_PRESENCE_LABEL_EN = "User Presence"
USER_PRESENCE_LABEL_JA = "操作者プレゼンス"
USER_PRESENCE_DEFAULT_X = max(
    0.0,
    min(1.0, float(os.getenv("USER_PRESENCE_DEFAULT_X", "0.5") or "0.5")),
)
USER_PRESENCE_DEFAULT_Y = max(
    0.0,
    min(1.0, float(os.getenv("USER_PRESENCE_DEFAULT_Y", "0.72") or "0.72")),
)
USER_PRESENCE_DRIFT_ALPHA = max(
    0.01,
    min(0.35, float(os.getenv("USER_PRESENCE_DRIFT_ALPHA", "0.06") or "0.06")),
)
USER_PRESENCE_EVENT_TTL_SECONDS = max(
    2.0,
    float(os.getenv("USER_PRESENCE_EVENT_TTL_SECONDS", "18.0") or "18.0"),
)
USER_PRESENCE_MAX_EVENTS = max(
    8,
    int(os.getenv("USER_PRESENCE_MAX_EVENTS", "96") or "96"),
)

# Shared fields (bounded global registry)
SHARED_FIELD_RECORD = "ημ.shared-field.v1"
FIELD_REGISTRY_RECORD = "ημ.field-registry.v1"
FIELD_REGISTRY_SCHEMA_VERSION = "field.registry.v1"

# Braid diagnostics (field-derived)
BRAID_DIAGNOSTICS_RECORD = "ημ.braid-diagnostics.v1"

# Ledger events
LEDGER_EVENT_RECORD = "ημ.ledger-event.v1"

# Field types (bounded set)
FIELD_KINDS = ("demand", "flow", "entropy", "graph")
MAX_FIELD_COUNT = len(FIELD_KINDS)  # Bounded!

# ============================================================================
# LEGACY RECORD TYPES (to be deprecated, migrated to unified model)
# ============================================================================

# Record Types
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
PRESENCE_ACCOUNT_RECORD = "ημ.presence-account.v1"
SIMULATION_METADATA_RECORD = "ημ.simulation-metadata.v1"
IMAGE_COMMENT_RECORD = "ημ.image-comment.v1"

# Weaver Parameters
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
).strip().lower() in {"1", "true", "yes", "on"}
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

# Active Search
WEAVER_ACTIVE_SEARCH_ENABLED = str(
    os.getenv("WEAVER_ACTIVE_SEARCH_ENABLED", "1") or "1"
).strip().lower() in {"1", "true", "yes", "on"}
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
WEAVER_ACTIVE_SEARCH_COOLDOWN_SECONDS = max(
    5.0,
    float(os.getenv("WEAVER_ACTIVE_SEARCH_COOLDOWN_SECONDS", "240.0") or "240.0"),
)
WEAVER_ACTIVE_SEARCH_MAX_RECENT = max(
    4,
    int(os.getenv("WEAVER_ACTIVE_SEARCH_MAX_RECENT", "20") or "20"),
)

# Multimodal Presence
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

# Truth Binding
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

# File and Ingest
ETA_MU_TEXT_SUFFIXES = {
    ".graphql",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".less",
    ".lisp",
    ".log",
    ".md",
    ".ndjson",
    ".php",
    ".pl",
    ".pm",
    ".proto",
    ".ps1",
    ".py",
    ".rs",
    ".rtf",
    ".sass",
    ".scss",
    ".sexp",
    ".sh",
    ".sql",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
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
ETA_MU_INGEST_INCLUDE_AUDIO_MIME = {
    "audio/aac",
    "audio/flac",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/x-aiff",
    "audio/x-m4a",
    "audio/x-ms-wma",
}
ETA_MU_INGEST_INCLUDE_PDF_MIME = {
    "application/pdf",
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
    ".graphql",
    ".h",
    ".hpp",
    ".html",
    ".hy",
    ".java",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".kt",
    ".less",
    ".lisp",
    ".log",
    ".markdown",
    ".md",
    ".org",
    ".php",
    ".pl",
    ".pm",
    ".proto",
    ".py",
    ".rb",
    ".rs",
    ".sass",
    ".scm",
    ".scss",
    ".sh",
    ".sibilant",
    ".sql",
    ".ss",
    ".t",
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
ETA_MU_INGEST_INCLUDE_AUDIO_EXT = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".wav",
    ".wma",
}
ETA_MU_INGEST_INCLUDE_PDF_EXT = {
    ".pdf",
}
ETA_MU_INGEST_EXCLUDE_REL_PATHS = {
    "__pycache__",
    ".cache",
    ".DS_Store",
    ".git",
    ".gradle",
    ".idea",
    ".next",
    ".npm",
    ".nuxt",
    ".pytest_cache",
    ".sass-cache",
    ".svelte-kit",
    ".vite",
    ".vscode",
    ".yarn",
    "_notes",
    "_rejected",
    "bin",
    "bower_components",
    "build",
    "dist",
    "node_modules",
    "obj",
    "out",
    "target",
    "vendor",
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
ETA_MU_INGEST_MAX_AUDIO_BYTES = max(
    4096,
    int(os.getenv("ETA_MU_INGEST_MAX_AUDIO_BYTES", "100000000") or "100000000"),
)
ETA_MU_INGEST_MAX_PDF_BYTES = max(
    4096,
    int(os.getenv("ETA_MU_INGEST_MAX_PDF_BYTES", "50000000") or "50000000"),
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

# Vecstore Spaces
ETA_MU_INGEST_SPACE_TEXT_ID = "ημ.text.v1"
ETA_MU_INGEST_SPACE_IMAGE_ID = "ημ.image.v1"
ETA_MU_INGEST_SPACE_AUDIO_ID = "ημ.audio.v1"
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
ETA_MU_INGEST_AUDIO_MODEL = str(
    os.getenv("ETA_MU_AUDIO_EMBED_MODEL", "nomic-embed-text") or "nomic-embed-text"
).strip()
ETA_MU_INGEST_AUDIO_MODEL_DIGEST = str(
    os.getenv("ETA_MU_AUDIO_EMBED_DIGEST", "none") or "none"
).strip()
ETA_MU_INGEST_AUDIO_DIMS = max(
    64,
    int(os.getenv("ETA_MU_AUDIO_EMBED_DIMS", "768") or "768"),
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
ETA_MU_INGEST_CONTRACT_ID = "ημ.ingest.text+image+audio+pdf.v2"
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
ETA_MU_DOCMETA_RECORD = "ημ.docmeta.v1"
ETA_MU_DOCMETA_REL = ".opencode/runtime/eta_mu_docmeta.v1.jsonl"
ETA_MU_DOCMETA_CYCLE_SECONDS = max(
    0.0,
    float(os.getenv("ETA_MU_DOCMETA_CYCLE_SECONDS", "20") or "20"),
)
ETA_MU_DOCMETA_MAX_PER_CYCLE = max(
    1,
    int(os.getenv("ETA_MU_DOCMETA_MAX_PER_CYCLE", "4") or "4"),
)
ETA_MU_DOCMETA_TEXT_CHAR_LIMIT = max(
    200,
    int(os.getenv("ETA_MU_DOCMETA_TEXT_CHAR_LIMIT", "2600") or "2600"),
)
ETA_MU_DOCMETA_SUMMARY_CHAR_LIMIT = max(
    80,
    int(os.getenv("ETA_MU_DOCMETA_SUMMARY_CHAR_LIMIT", "280") or "280"),
)
ETA_MU_DOCMETA_TAG_LIMIT = max(
    2,
    int(os.getenv("ETA_MU_DOCMETA_TAG_LIMIT", "8") or "8"),
)
ETA_MU_DOCMETA_LLM_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("ETA_MU_DOCMETA_LLM_TIMEOUT_SECONDS", "0.8") or "0.8"),
)
ETA_MU_DOCMETA_LLM_MODEL = str(os.getenv("ETA_MU_DOCMETA_LLM_MODEL", "") or "").strip()

# File Graph
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
ETA_MU_FILE_GRAPH_TAG_LIMIT = max(
    8,
    int(os.getenv("ETA_MU_FILE_GRAPH_TAG_LIMIT", "320") or "320"),
)
ETA_MU_FILE_GRAPH_TAG_EDGE_LIMIT = max(
    16,
    int(os.getenv("ETA_MU_FILE_GRAPH_TAG_EDGE_LIMIT", "4800") or "4800"),
)
ETA_MU_FILE_GRAPH_TAG_PAIR_EDGE_LIMIT = max(
    8,
    int(os.getenv("ETA_MU_FILE_GRAPH_TAG_PAIR_EDGE_LIMIT", "1600") or "1600"),
)

# Council and Study
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

# Chat Training
CHAT_TRAINING_RECORD = "ημ.chat-training.v1"
CHAT_TRAINING_LOG_REL = ".opencode/runtime/chat_training.v1.jsonl"
CHAT_TRAINING_HISTORY_LIMIT = max(
    8,
    int(os.getenv("CHAT_TRAINING_HISTORY_LIMIT", "600") or "600"),
)
CHAT_TRAINING_CONTEXT_LIMIT = max(
    1,
    min(8, int(os.getenv("CHAT_TRAINING_CONTEXT_LIMIT", "3") or "3")),
)
CHAT_TRAINING_TEXT_LIMIT = max(
    256,
    int(os.getenv("CHAT_TRAINING_TEXT_LIMIT", "4000") or "4000"),
)

# Image Commentary
IMAGE_COMMENTARY_MODEL = str(
    os.getenv("IMAGE_COMMENTARY_MODEL", "qwen3-vl:2b-instruct")
    or "qwen3-vl:2b-instruct"
).strip()
IMAGE_COMMENTARY_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("IMAGE_COMMENTARY_TIMEOUT_SECONDS", "30") or "30"),
)
IMAGE_COMMENTARY_MAX_BYTES = max(
    1024,
    int(os.getenv("IMAGE_COMMENTARY_MAX_BYTES", "20000000") or "20000000"),
)

# World Log
WORLD_LOG_EVENT_LIMIT = max(
    24,
    min(800, int(os.getenv("WORLD_LOG_EVENT_LIMIT", "180") or "180")),
)
WORLD_LOG_RELATION_LIMIT = max(
    1,
    min(8, int(os.getenv("WORLD_LOG_RELATION_LIMIT", "4") or "4")),
)
WIKIMEDIA_EVENTSTREAMS_ENABLED = str(
    os.getenv("WIKIMEDIA_EVENTSTREAMS_ENABLED", "1") or "1"
).strip().lower() in {"1", "true", "yes", "on"}
WIKIMEDIA_EVENTSTREAMS_STREAMS = str(
    os.getenv(
        "WIKIMEDIA_EVENTSTREAMS_STREAMS",
        "recentchange,revision-create,page-create,page-delete,page-undelete",
    )
    or "recentchange,revision-create,page-create,page-delete,page-undelete"
).strip()
WIKIMEDIA_EVENTSTREAMS_BASE_URL = str(
    os.getenv(
        "WIKIMEDIA_EVENTSTREAMS_BASE_URL",
        "https://stream.wikimedia.org/v2/stream",
    )
    or "https://stream.wikimedia.org/v2/stream"
).strip()
WIKIMEDIA_EVENTSTREAMS_POLL_INTERVAL_SECONDS = max(
    1.0,
    float(os.getenv("WIKIMEDIA_EVENTSTREAMS_POLL_INTERVAL_SECONDS", "4.0") or "4.0"),
)
WIKIMEDIA_EVENTSTREAMS_FETCH_TIMEOUT_SECONDS = max(
    0.1,
    float(os.getenv("WIKIMEDIA_EVENTSTREAMS_FETCH_TIMEOUT_SECONDS", "0.8") or "0.8"),
)
WIKIMEDIA_EVENTSTREAMS_MAX_BYTES = max(
    2048,
    int(os.getenv("WIKIMEDIA_EVENTSTREAMS_MAX_BYTES", "262144") or "262144"),
)
WIKIMEDIA_EVENTSTREAMS_MAX_EVENTS_PER_POLL = max(
    1,
    int(os.getenv("WIKIMEDIA_EVENTSTREAMS_MAX_EVENTS_PER_POLL", "80") or "80"),
)
WIKIMEDIA_EVENTSTREAMS_RATE_LIMIT_PER_POLL = max(
    1,
    int(os.getenv("WIKIMEDIA_EVENTSTREAMS_RATE_LIMIT_PER_POLL", "28") or "28"),
)
WIKIMEDIA_EVENTSTREAMS_DEDUPE_TTL_SECONDS = max(
    15.0,
    float(os.getenv("WIKIMEDIA_EVENTSTREAMS_DEDUPE_TTL_SECONDS", "900") or "900"),
)
WIKIMEDIA_EVENTSTREAMS_DETAIL_CHAR_LIMIT = max(
    96,
    int(os.getenv("WIKIMEDIA_EVENTSTREAMS_DETAIL_CHAR_LIMIT", "220") or "220"),
)
NWS_ALERTS_ENABLED = str(
    os.getenv("NWS_ALERTS_ENABLED", "0") or "0"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
NWS_ALERTS_ENDPOINT = str(
    os.getenv("NWS_ALERTS_ENDPOINT", "https://api.weather.gov/alerts/active")
    or "https://api.weather.gov/alerts/active"
).strip()
NWS_ALERTS_POLL_INTERVAL_SECONDS = max(
    2.0,
    float(os.getenv("NWS_ALERTS_POLL_INTERVAL_SECONDS", "24.0") or "24.0"),
)
NWS_ALERTS_FETCH_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("NWS_ALERTS_FETCH_TIMEOUT_SECONDS", "3.0") or "3.0"),
)
NWS_ALERTS_MAX_BYTES = max(
    2048,
    int(os.getenv("NWS_ALERTS_MAX_BYTES", "262144") or "262144"),
)
NWS_ALERTS_MAX_ALERTS_PER_POLL = max(
    1,
    int(os.getenv("NWS_ALERTS_MAX_ALERTS_PER_POLL", "120") or "120"),
)
NWS_ALERTS_RATE_LIMIT_PER_POLL = max(
    1,
    int(os.getenv("NWS_ALERTS_RATE_LIMIT_PER_POLL", "24") or "24"),
)
NWS_ALERTS_DEDUPE_TTL_SECONDS = max(
    15.0,
    float(os.getenv("NWS_ALERTS_DEDUPE_TTL_SECONDS", "900") or "900"),
)
NWS_ALERTS_DETAIL_CHAR_LIMIT = max(
    96,
    int(os.getenv("NWS_ALERTS_DETAIL_CHAR_LIMIT", "240") or "240"),
)
SWPC_ALERTS_ENABLED = str(
    os.getenv("SWPC_ALERTS_ENABLED", "0") or "0"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SWPC_ALERTS_ENDPOINT = str(
    os.getenv(
        "SWPC_ALERTS_ENDPOINT", "https://services.swpc.noaa.gov/products/alerts.json"
    )
    or "https://services.swpc.noaa.gov/products/alerts.json"
).strip()
SWPC_ALERTS_POLL_INTERVAL_SECONDS = max(
    2.0,
    float(os.getenv("SWPC_ALERTS_POLL_INTERVAL_SECONDS", "28.0") or "28.0"),
)
SWPC_ALERTS_FETCH_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("SWPC_ALERTS_FETCH_TIMEOUT_SECONDS", "3.0") or "3.0"),
)
SWPC_ALERTS_MAX_BYTES = max(
    2048,
    int(os.getenv("SWPC_ALERTS_MAX_BYTES", "262144") or "262144"),
)
SWPC_ALERTS_MAX_ALERTS_PER_POLL = max(
    1,
    int(os.getenv("SWPC_ALERTS_MAX_ALERTS_PER_POLL", "120") or "120"),
)
SWPC_ALERTS_RATE_LIMIT_PER_POLL = max(
    1,
    int(os.getenv("SWPC_ALERTS_RATE_LIMIT_PER_POLL", "24") or "24"),
)
SWPC_ALERTS_DEDUPE_TTL_SECONDS = max(
    15.0,
    float(os.getenv("SWPC_ALERTS_DEDUPE_TTL_SECONDS", "900") or "900"),
)
SWPC_ALERTS_DETAIL_CHAR_LIMIT = max(
    96,
    int(os.getenv("SWPC_ALERTS_DETAIL_CHAR_LIMIT", "240") or "240"),
)
GIBS_LAYERS_ENABLED = str(
    os.getenv("GIBS_LAYERS_ENABLED", "0") or "0"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
GIBS_LAYERS_CAPABILITIES_ENDPOINT = str(
    os.getenv(
        "GIBS_LAYERS_CAPABILITIES_ENDPOINT",
        "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities",
    )
    or "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities"
).strip()
GIBS_LAYERS_TARGETS = str(
    os.getenv(
        "GIBS_LAYERS_TARGETS",
        "MODIS_Terra_CorrectedReflectance_TrueColor,MODIS_Aqua_CorrectedReflectance_TrueColor,VIIRS_SNPP_CorrectedReflectance_TrueColor",
    )
    or "MODIS_Terra_CorrectedReflectance_TrueColor,MODIS_Aqua_CorrectedReflectance_TrueColor,VIIRS_SNPP_CorrectedReflectance_TrueColor"
).strip()
GIBS_LAYERS_POLL_INTERVAL_SECONDS = max(
    2.0,
    float(os.getenv("GIBS_LAYERS_POLL_INTERVAL_SECONDS", "60.0") or "60.0"),
)
GIBS_LAYERS_FETCH_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("GIBS_LAYERS_FETCH_TIMEOUT_SECONDS", "4.0") or "4.0"),
)
GIBS_LAYERS_MAX_BYTES = max(
    4096,
    int(os.getenv("GIBS_LAYERS_MAX_BYTES", "524288") or "524288"),
)
GIBS_LAYERS_MAX_LAYERS_PER_POLL = max(
    1,
    int(os.getenv("GIBS_LAYERS_MAX_LAYERS_PER_POLL", "24") or "24"),
)
GIBS_LAYERS_RATE_LIMIT_PER_POLL = max(
    1,
    int(os.getenv("GIBS_LAYERS_RATE_LIMIT_PER_POLL", "8") or "8"),
)
GIBS_LAYERS_DEDUPE_TTL_SECONDS = max(
    15.0,
    float(os.getenv("GIBS_LAYERS_DEDUPE_TTL_SECONDS", "21600") or "21600"),
)
GIBS_LAYERS_DETAIL_CHAR_LIMIT = max(
    96,
    int(os.getenv("GIBS_LAYERS_DETAIL_CHAR_LIMIT", "260") or "260"),
)
EONET_EVENTS_ENABLED = str(
    os.getenv("EONET_EVENTS_ENABLED", "0") or "0"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
EONET_EVENTS_ENDPOINT = str(
    os.getenv(
        "EONET_EVENTS_ENDPOINT", "https://eonet.gsfc.nasa.gov/api/v3/events?status=open"
    )
    or "https://eonet.gsfc.nasa.gov/api/v3/events?status=open"
).strip()
EONET_EVENTS_POLL_INTERVAL_SECONDS = max(
    2.0,
    float(os.getenv("EONET_EVENTS_POLL_INTERVAL_SECONDS", "45.0") or "45.0"),
)
EONET_EVENTS_FETCH_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("EONET_EVENTS_FETCH_TIMEOUT_SECONDS", "4.0") or "4.0"),
)
EONET_EVENTS_MAX_BYTES = max(
    2048,
    int(os.getenv("EONET_EVENTS_MAX_BYTES", "262144") or "262144"),
)
EONET_EVENTS_MAX_EVENTS_PER_POLL = max(
    1,
    int(os.getenv("EONET_EVENTS_MAX_EVENTS_PER_POLL", "120") or "120"),
)
EONET_EVENTS_RATE_LIMIT_PER_POLL = max(
    1,
    int(os.getenv("EONET_EVENTS_RATE_LIMIT_PER_POLL", "24") or "24"),
)
EONET_EVENTS_DEDUPE_TTL_SECONDS = max(
    15.0,
    float(os.getenv("EONET_EVENTS_DEDUPE_TTL_SECONDS", "21600") or "21600"),
)
EONET_EVENTS_DETAIL_CHAR_LIMIT = max(
    96,
    int(os.getenv("EONET_EVENTS_DETAIL_CHAR_LIMIT", "260") or "260"),
)
GIBS_LAYERS_TILE_MATRIX_SET = str(
    os.getenv("GIBS_LAYERS_TILE_MATRIX_SET", "250m") or "250m"
).strip()
GIBS_LAYERS_TILE_MATRIX = max(
    0,
    int(os.getenv("GIBS_LAYERS_TILE_MATRIX", "2") or "2"),
)
GIBS_LAYERS_TILE_ROW = max(
    0,
    int(os.getenv("GIBS_LAYERS_TILE_ROW", "1") or "1"),
)
GIBS_LAYERS_TILE_COL = max(
    0,
    int(os.getenv("GIBS_LAYERS_TILE_COL", "1") or "1"),
)
EMSC_STREAM_ENABLED = str(
    os.getenv("EMSC_STREAM_ENABLED", "0") or "0"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
EMSC_STREAM_URL = str(
    os.getenv(
        "EMSC_STREAM_URL",
        "ws://www.seismicportal.eu/standing_order/websocket",
    )
    or "ws://www.seismicportal.eu/standing_order/websocket"
).strip()
EMSC_STREAM_POLL_INTERVAL_SECONDS = max(
    1.0,
    float(os.getenv("EMSC_STREAM_POLL_INTERVAL_SECONDS", "6.0") or "6.0"),
)
EMSC_STREAM_FETCH_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("EMSC_STREAM_FETCH_TIMEOUT_SECONDS", "1.4") or "1.4"),
)
EMSC_STREAM_MAX_BYTES = max(
    2048,
    int(os.getenv("EMSC_STREAM_MAX_BYTES", "262144") or "262144"),
)
EMSC_STREAM_MAX_EVENTS_PER_POLL = max(
    1,
    int(os.getenv("EMSC_STREAM_MAX_EVENTS_PER_POLL", "60") or "60"),
)
EMSC_STREAM_RATE_LIMIT_PER_POLL = max(
    1,
    int(os.getenv("EMSC_STREAM_RATE_LIMIT_PER_POLL", "20") or "20"),
)
EMSC_STREAM_DEDUPE_TTL_SECONDS = max(
    15.0,
    float(os.getenv("EMSC_STREAM_DEDUPE_TTL_SECONDS", "900") or "900"),
)
EMSC_STREAM_DETAIL_CHAR_LIMIT = max(
    96,
    int(os.getenv("EMSC_STREAM_DETAIL_CHAR_LIMIT", "220") or "220"),
)

# Resource Monitoring
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

# Docker Autorestart
DOCKER_AUTORESTART_ENABLED = str(
    os.getenv("DOCKER_AUTORESTART_ENABLED", "1") or "1"
).strip().lower() in {"1", "true", "yes", "on"}
DOCKER_AUTORESTART_REQUIRE_COUNCIL = str(
    os.getenv("DOCKER_AUTORESTART_REQUIRE_COUNCIL", "1") or "1"
).strip().lower() in {"1", "true", "yes", "on"}
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

# Presence and Fields
FIELD_TO_PRESENCE = {
    "f1": "receipt_river",
    "f2": "witness_thread",
    "f3": "anchor_registry",
    "f4": "keeper_of_receipts",
    "f5": "fork_tax_canticle",
    "f6": "mage_of_receipts",
    "f7": "gates_of_truth",
    "f8": "anchor_registry",
    # Philosophical concept fields
    "f9": "principle_good",
    "f10": "principle_evil",
    "f11": "principle_right",
    "f12": "principle_wrong",
    "f13": "state_dead",
    "f14": "state_living",
    # Chaos field - transcends and perturbs all other fields
    "f15": "chaos_butterfly",
    # Core resource fields
    "f16": "presence.core.cpu",
    "f17": "presence.core.ram",
    "f18": "presence.core.disk",
    "f19": "presence.core.network",
    "f20": "presence.core.gpu",
    "f21": "presence.core.npu",
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
        "documentation",
        "doc",
        "guide",
        "manual",
        "handbook",
        "overview",
        "summary",
        "readme",
        "wiki",
        "knowledge",
        "reference",
        "taxonomy",
        "ontology",
        "schema",
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
        "creative",
        "synthesis",
        "art",
        "design",
        "visual",
        "illustration",
        "narrative",
        "fiction",
        "poetry",
        "compose",
        "imagine",
        "sketch",
        "draft",
        "concept",
        "story",
        "lyrics",
        "song",
        "muse",
        "aesthetic",
        "expression",
        "craft",
        "artifact",
        "render",
        "canvas",
        "palette",
        "harmony",
        "melody",
        "rhythm",
        "texture",
        "form",
        "structure",
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
        "verify",
        "test",
        "assert",
        "check",
        "audit",
        "inspect",
        "review",
        "certify",
        "guarantee",
        "warrant",
        "requirement",
        "specification",
        "constraint",
        "invariant",
        "axiom",
        "law",
        "rule",
        "standard",
        "compliance",
        "governance",
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
        "config",
        "configuration",
        "deploy",
        "deployment",
        "pipeline",
        "workflow",
        "automation",
        "orchestration",
        "schedule",
        "cron",
        "job",
        "service",
        "daemon",
        "server",
        "environment",
        "setup",
        "install",
        "bootstrap",
        "init",
        "system",
        "infrastructure",
        "devops",
        "ci",
        "cd",
    },
    # Philosophical concept fields with rich semantic keyword sets
    "f9": {
        # The Good - virtue, benevolence, altruism domain
        "good",
        "virtue",
        "benevolent",
        "altruism",
        "compassion",
        "kindness",
        "generous",
        "healing",
        "nurture",
        "protect",
        "innocent",
        "moral",
        "excellence",
        "righteous",
        "benefit",
        "selfless",
        "care",
        "love",
        "empathy",
        "conscience",
        "honor",
        "integrity",
        "pure",
        "intent",
        "light",
        "hope",
        "redemption",
        "salvation",
        "grace",
        "blessing",
        "harmony",
        "peace",
        "mercy",
        "flourish",
        "wellbeing",
        "optimal",
        "altruistic",
        "charity",
        "gentle",
        "helpful",
        "nourish",
        "restore",
        "save",
        "uplift",
        "virtuous",
        "worthy",
        "noble",
        "elevated",
        "sacred",
        "holy",
        "divine",
        "blessed",
        "benevolence",
        "goodness",
        "decency",
        "humanity",
        "warmth",
        "tenderness",
        "pity",
        "sympathy",
        "understanding",
        "forgiveness",
        "magnanimity",
        "beneficence",
        "philanthropy",
        "humanitarian",
        "humane",
        "clement",
        "lenient",
        "merciful",
        "compassionate",
    },
    "f10": {
        # The Evil - malevolence, corruption, harm domain
        "evil",
        "malevolent",
        "corrupt",
        "selfish",
        "cruel",
        "harm",
        "destroy",
        "deception",
        "manipulation",
        "exploit",
        "suffering",
        "inflict",
        "decay",
        "vice",
        "sin",
        "wicked",
        "malice",
        "spite",
        "hatred",
        "greed",
        "lust",
        "power",
        "dominate",
        "oppress",
        "tyranny",
        "violence",
        "betrayal",
        "treachery",
        "poison",
        "entropy",
        "chaos",
        "void",
        "dark",
        "despair",
        "damnation",
        "corrupt",
        "vile",
        "foul",
        "nefarious",
        "heinous",
        "atrocious",
        "abominable",
        "detestable",
        "loathsome",
        "odious",
        "repugnant",
        "revolting",
        "repellent",
        "disgusting",
        "distasteful",
        "offensive",
        "objectionable",
        "hateful",
        "maleficent",
        "malignant",
        "malign",
        "baleful",
        "sinister",
        "iniquitous",
        "depraved",
        "degenerate",
        "debased",
        "degraded",
        "perverted",
        "warped",
        "twisted",
        "sadistic",
        "brutal",
        "savage",
        "ferocious",
        "fierce",
        "ruthless",
        "merciless",
        "pitiless",
        "remorseless",
        "heartless",
        "callous",
        "cold",
        "inhuman",
        "monstrous",
        "demonic",
        "diabolical",
        "fiendish",
        "devilish",
        "impish",
        "infernal",
        "hellish",
    },
    "f11": {
        # The Right - justice, correctness, moral truth domain
        "right",
        "justice",
        "correct",
        "moral",
        "truth",
        "righteous",
        "proper",
        "ethical",
        "duty",
        "responsibility",
        "fair",
        "equity",
        "deserve",
        "merit",
        "lawful",
        "legitimate",
        "valid",
        "order",
        "alignment",
        "clarity",
        "certainty",
        "principled",
        "stand",
        "defend",
        "uphold",
        "standard",
        "correctness",
        "accuracy",
        "validity",
        "soundness",
        "authenticity",
        "genuineness",
        "legitimacy",
        "lawfulness",
        "legality",
        "constitutionality",
        "rightfulness",
        "justness",
        "fairness",
        "equitableness",
        "impartiality",
        "objectivity",
        "disinterestedness",
        "evenhandedness",
        "honesty",
        "probity",
        "rectitude",
        "uprightness",
        "righteousness",
        "integrity",
        "honor",
        "decorum",
        "propriety",
        "seemliness",
        "fitness",
        "appropriateness",
        "suitability",
        "aptness",
        "fittingness",
        "meetness",
        "due",
        "owing",
        "appropriate",
        "fitting",
        "suitable",
        "proper",
        "apt",
        "becoming",
        "seemly",
        "decorous",
        "meet",
        "commensurate",
    },
    "f12": {
        # The Wrong - injustice, error, moral failure domain
        "wrong",
        "injustice",
        "error",
        "fail",
        "immoral",
        "false",
        "improper",
        "unethical",
        "neglect",
        "irresponsible",
        "unfair",
        "inequity",
        "undeserved",
        "corrupt",
        "lawless",
        "illegitimate",
        "invalid",
        "disorder",
        "misalignment",
        "confusion",
        "failure",
        "unprincipled",
        "tolerate",
        "injustice",
        "lower",
        "standard",
        "mistaken",
        "erroneous",
        "incorrect",
        "inaccurate",
        "inexact",
        "imprecise",
        "fallacious",
        "unsound",
        "untrue",
        "false",
        "unfounded",
        "groundless",
        "baseless",
        "unjustified",
        "unwarranted",
        "indefensible",
        "unjust",
        "inequitable",
        "biased",
        "prejudiced",
        "partial",
        "partisan",
        "discriminatory",
        "preferential",
        "unjustified",
        "unwarranted",
        "undeserved",
        "unmerited",
        "unearned",
        "unjust",
        "inequitable",
        "biased",
        "prejudiced",
        "partial",
        "partisan",
        "discriminatory",
        "preferential",
        "wrongful",
        "unlawful",
        "illegal",
        "illegitimate",
        "illicit",
        "criminal",
        "felonious",
        "delinquent",
        "culpable",
        "guilty",
        "blameworthy",
        "reprehensible",
        "disgraceful",
        "shameful",
        "dishonorable",
        "ignoble",
        "improper",
        "indecent",
        "indecorous",
        "unseemly",
        "unsuitable",
        "inappropriate",
        "unfitting",
        "unbecoming",
    },
    "f13": {
        # The Dead - finality, stillness, end domain
        "dead",
        "death",
        "final",
        "stillness",
        "silence",
        "end",
        "terminate",
        "cease",
        "nonexistence",
        "oblivion",
        "rest",
        "peace",
        "complete",
        "closure",
        "memory",
        "legacy",
        "remains",
        "ashes",
        "dust",
        "entropy",
        "void",
        "absence",
        "null",
        "zero",
        "transcend",
        "release",
        "sleep",
        "dark",
        "quiet",
        "mortality",
        "mortal",
        "perish",
        "expire",
        "decease",
        "demise",
        "dying",
        "fatality",
        "loss",
        "bereavement",
        "grief",
        "mourning",
        "lamentation",
        "elegy",
        "dirge",
        "requiem",
        "threnody",
        "corpse",
        "cadaver",
        "body",
        "carcass",
        "remains",
        "relic",
        "fossil",
        "skeleton",
        "skull",
        "bones",
        "cinerary",
        "funeral",
        "burial",
        "interment",
        "entombment",
        "inhumation",
        "cremation",
        "incineration",
        "immolation",
        "extinction",
        "annihilation",
        "obliteration",
        "eradication",
        "elimination",
        "extermination",
        "destruction",
        "ruin",
        "ruination",
        "decay",
        "decomposition",
        "putrefaction",
        "corruption",
        "rotting",
        "disintegration",
        "dissolution",
        "dissolving",
        "melting",
        "fading",
        "waning",
        "ebbing",
        "declining",
        "dwindling",
        "subsiding",
        "abating",
        "diminishing",
        "lessening",
        "decreasing",
        "weakening",
        "failing",
        "deteriorating",
        "degenerating",
    },
    "f14": {
        # The Living - vitality, growth, existence domain
        "living",
        "life",
        "alive",
        "vital",
        "growth",
        "change",
        "adapt",
        "reproduce",
        "metabolism",
        "conscious",
        "aware",
        "sensation",
        "experience",
        "joy",
        "pain",
        "desire",
        "will",
        "agency",
        "choice",
        "emerge",
        "become",
        "potential",
        "possible",
        "future",
        "hope",
        "struggle",
        "survive",
        "resilient",
        "flourish",
        "bloom",
        "pulse",
        "breathe",
        "heartbeat",
        "spark",
        "animate",
        "being",
        "exist",
        "presence",
        "vital",
        "vivid",
        "vigorous",
        "energetic",
        "dynamic",
        "active",
        "lively",
        "spirited",
        "animated",
        "vivacious",
        "vibrant",
        "vital",
        "thriving",
        "flourishing",
        "prospering",
        "blooming",
        "burgeoning",
        "growing",
        "developing",
        "evolving",
        "progressing",
        "advancing",
        "maturing",
        "ripening",
        "bearing",
        "fruit",
        "generative",
        "fertile",
        "fecund",
        "prolific",
        "productive",
        "creative",
        "generative",
        "birth",
        "born",
        "creation",
        "created",
        "making",
        "made",
        "formation",
        "forming",
        "emergence",
        "emerging",
        "appearance",
        "appearing",
        "arising",
        "rising",
        "dawning",
        "beginning",
        "starting",
        "commencing",
        "originating",
        "initiating",
        "launching",
        "birth",
        "nativity",
        "genesis",
        "origin",
        "source",
        "root",
        "seed",
        "germ",
        "embryo",
        "fetus",
        "ovum",
        "egg",
        "spawn",
        "issue",
        "offspring",
        "progeny",
        "descendant",
        "heir",
        "successor",
        "replacement",
        "substitute",
        "surrogate",
        "proxy",
        "agent",
        "actor",
        "doer",
        "performer",
        "player",
        "participant",
        "member",
        "part",
        "portion",
        "share",
        "stake",
        "interest",
        "investment",
        "commitment",
        "dedication",
        "devotion",
        "devoted",
        "faithful",
        "loyal",
        "true",
        "trustworthy",
        "reliable",
        "dependable",
        "responsible",
        "accountable",
        "answerable",
        "liable",
        "subject",
        "open",
        "susceptible",
        "vulnerable",
        "exposed",
        "sensitive",
        "responsive",
        "reactive",
        "alive",
        "alert",
        "awake",
        "watchful",
        "vigilant",
        "attentive",
        "observant",
        "mindful",
        "aware",
        "conscious",
        "cognizant",
        "sensible",
        "sentient",
        "feeling",
        "sensing",
        "perceiving",
        "discerning",
        "recognizing",
        "knowing",
        "understanding",
        "comprehending",
        "apprehending",
        "grasping",
        "seizing",
        "catching",
        "capturing",
        "trapping",
        "snaring",
        "entangling",
        "involving",
        "implicating",
        "incriminating",
        "accusing",
        "charging",
        "indicting",
        "arraigning",
        "prosecuting",
        "trying",
        "testing",
        "proving",
        "demonstrating",
        "showing",
        "displaying",
        "exhibiting",
        "presenting",
        "offering",
        "giving",
        "bestowing",
        "granting",
        "conferring",
        "awarding",
        "accord",
    },
    # f15: Chaos Butterfly - noise, unpredictability, perturbation
    "f15": {
        "chaos",
        "butterfly",
        "flutter",
        "noise",
        "random",
        "unpredictable",
        "turbulence",
        "perturbation",
        "disruption",
        "disorder",
        "entropy",
        "instability",
        "fluctuation",
        "oscillation",
        "vibration",
        "resonance",
        "interference",
        "distortion",
        "scramble",
        "jumble",
        "tangle",
        "mess",
        "muddle",
        "confusion",
        "befuddle",
        "bewilder",
        "perplex",
        "puzzle",
        "mystify",
        "obscure",
        "cloud",
        "blur",
        "distort",
        "warp",
        "twist",
        "contort",
        "bend",
        "curve",
        "swirl",
        "twirl",
        "spin",
        "rotate",
        "gyrate",
        "whirl",
        "circulate",
        "cycle",
        "loop",
        "spiral",
        "helix",
        "vortex",
        "maelstrom",
        "whirlpool",
        "eddy",
        "current",
        "flow",
        "stream",
        "torrent",
        "cascade",
        "avalanche",
        "landslide",
        "deluge",
        "flood",
        "inundation",
        "surge",
        "rush",
        "gush",
        "spurt",
        "squirt",
        "jet",
        "spray",
        "mist",
        "fog",
        "haze",
        "smog",
        "smoke",
        "steam",
        "vapor",
        "gas",
        "aerosol",
        "particulate",
        "dust",
        "ash",
        "soot",
        "grime",
        "dirt",
        "filth",
        "pollution",
        "contamination",
        "taint",
        "corruption",
        "infection",
        "disease",
        "plague",
        "pestilence",
        "blight",
        "scourge",
        "curse",
        "hex",
        "jinx",
        "spell",
        "charm",
        "enchantment",
        "glamour",
        "illusion",
        "mirage",
        "hallucination",
        "phantasm",
        "apparition",
        "specter",
        "ghost",
        "phantom",
        "wraith",
        "shade",
        "shadow",
        "darkness",
        "void",
        "abyss",
        "chasm",
        "gulf",
        "rift",
        "breach",
        "gap",
        "opening",
        "hole",
        "puncture",
        "tear",
        "rip",
        "rupture",
        "break",
        "fracture",
        "crack",
        "fissure",
        "crevice",
        "cleft",
        "split",
        "cleave",
        "divide",
        "separate",
        "sunder",
        "sever",
        "detach",
        "disconnect",
        "disjoin",
        "disengage",
        "disassociate",
        "dissociate",
        "disaffiliate",
        "alienate",
        "estrange",
        "isolate",
        "insulate",
        "seclude",
        "sequester",
        "segregate",
        "partition",
        "compartment",
        "section",
        "segment",
        "fragment",
        "shard",
        "shatter",
        "smash",
        "crush",
        "grind",
        "crumble",
        "disintegrate",
        "dissolve",
        "melt",
        "liquefy",
        "vaporize",
        "evaporate",
        "sublimate",
        "transmute",
        "transform",
        "metamorphose",
        "mutate",
        "morph",
        "change",
        "shift",
        "alter",
        "modify",
        "vary",
        "diversify",
        "differentiate",
        "deviate",
        "diverge",
        "branch",
        "fork",
        "bifurcate",
        "split",
        "divide",
    },
    "f16": {
        "cpu",
        "core",
        "compute",
        "cycle",
        "thread",
        "load",
        "process",
        "task",
        "run",
        "execute",
    },
    "f17": {
        "ram",
        "memory",
        "heap",
        "stack",
        "cache",
        "buffer",
        "store",
        "allocation",
        "capacity",
    },
    "f18": {
        "disk",
        "storage",
        "io",
        "read",
        "write",
        "throughput",
        "latency",
        "volume",
        "mount",
    },
    "f19": {
        "network",
        "bandwidth",
        "packet",
        "stream",
        "connection",
        "port",
        "socket",
        "traffic",
    },
    "f20": {
        "gpu",
        "graphics",
        "render",
        "shader",
        "texture",
        "frame",
        "vram",
        "compute",
        "tensor",
    },
    "f21": {
        "npu",
        "neural",
        "inference",
        "model",
        "weight",
        "activation",
        "token",
        "embedding",
    },
}

# PromptDB
PROMPTDB_PACKET_GLOBS = ("*.intent.lisp", "*.packet.lisp")
PROMPTDB_CONTRACT_GLOBS = ("*.contract.lisp",)
PROMPTDB_REFRESH_DEBOUNCE_SECONDS = max(
    0.0,
    float(os.getenv("PROMPTDB_REFRESH_DEBOUNCE_SECONDS", "1.2") or "1.2"),
)
PROMPTDB_OPEN_QUESTIONS_PACKET = "diagrams/part64_runtime_system.packet.lisp"

# Task Queue
TASK_QUEUE_EVENT_VERSION = "eta-mu.task-queue/v1"

# Weaver Service
WEAVER_AUTOSTART = str(
    os.getenv("WEAVER_AUTOSTART", "1") or "1"
).strip().lower() not in {"0", "false", "off", "no"}
WEAVER_PORT = int(os.getenv("WEAVER_PORT", "8793") or "8793")
WEAVER_HOST_ENV = str(os.getenv("WEAVER_HOST", "") or "").strip()

# TTS
TTS_BASE_URL = str(os.getenv("TTS_BASE_URL", "http://127.0.0.1:8788") or "").rstrip("/")

# Presence Profiles
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
    "ja": "ヘルスの番人 - NPU0",
}

RESOURCE_CORE_PROFILE = {
    "id": "resource_core",
    "en": "Resource Core",
    "ja": "リソース・コア",
}


# Stopwords
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

# Presence Aliases
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

# UI Projection Perspetives
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

# Projection Field Schemas
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

# Projection Elements
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

# Shared Locks and Caches
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
_ETA_MU_DOCMETA_LOCK = threading.Lock()
_ETA_MU_DOCMETA_CACHE: dict[str, Any] = {
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
_WEAVER_ACTIVE_SEARCH_LOCK = threading.Lock()
_WEAVER_ACTIVE_SEARCH_STATE: dict[str, Any] = {
    "enabled": WEAVER_ACTIVE_SEARCH_ENABLED,
    "running": False,
    "last_tick_monotonic": 0.0,
    "last_tick_at": "",
    "last_success_at": "",
    "recent_seeded": {},
    "recent_decisions": [],
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
_PRESENCE_ACCOUNTS_LOCK = threading.Lock()
_PRESENCE_ACCOUNTS_CACHE: dict[str, Any] = {
    "path": "",
    "mtime_ns": 0,
    "entries": [],
}
_IMAGE_COMMENTS_LOCK = threading.Lock()
_IMAGE_COMMENTS_CACHE: dict[str, Any] = {
    "path": "",
    "mtime_ns": 0,
    "entries": [],
}
_DAIMO_DYNAMICS_LOCK = threading.Lock()
_DAIMO_DYNAMICS_CACHE: dict[str, Any] = {
    "entities": {},
    "last_gc_monotonic": 0.0,
}
_USER_PRESENCE_INPUT_LOCK = threading.Lock()
_USER_PRESENCE_INPUT_CACHE: dict[str, Any] = {
    "target_x": USER_PRESENCE_DEFAULT_X,
    "target_y": USER_PRESENCE_DEFAULT_Y,
    "anchor_x": USER_PRESENCE_DEFAULT_X,
    "anchor_y": USER_PRESENCE_DEFAULT_Y,
    "latest_message": "",
    "latest_target": "",
    "last_input_unix": 0.0,
    "last_pointer_unix": 0.0,
    "last_input_monotonic": 0.0,
    "last_pointer_monotonic": 0.0,
    "seq": 0,
    "events": [],
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
_MYCELIAL_ECHO_CACHE_LOCK = threading.Lock()
_MYCELIAL_ECHO_CACHE: dict[str, Any] = {
    "limit": 0,
    "checked_monotonic": 0.0,
    "docs": [],
}
_SIMULATION_METADATA_LOCK = threading.Lock()
_SIMULATION_METADATA_CACHE: dict[str, Any] = {
    "path": "",
    "mtime_ns": 0,
    "entries": [],
}
