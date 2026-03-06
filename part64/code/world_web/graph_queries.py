from __future__ import annotations

import hashlib
import ast
import json
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .control_budget_policy import build_control_budget_snapshot
from .control_budget_strategies import apply_threat_compute_budget_policy
from .graph_query_policy import (
    build_unknown_graph_query_result,
    normalize_graph_query_name,
)
from .threat_radar_strategy import (
    apply_threat_signal_strategy,
    build_threat_llm_fallback,
    resolve_threat_proximity_strategy,
    resolve_threat_risk_level,
    resolve_threat_scoring_mode,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _clamp01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, _safe_float(value, default)))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


_GITHUB_THREAT_TERMS: set[str] = {
    "0day",
    "auth",
    "credential",
    "cve",
    "dos",
    "exploit",
    "hotfix",
    "injection",
    "leak",
    "malware",
    "phishing",
    "privilege",
    "rce",
    "secret",
    "security",
    "supply",
    "token",
    "vuln",
    "xss",
    "xxe",
}


_HORMUZ_THREAT_LABEL_SCORES: dict[str, int] = {
    "military_activity": 3,
    "electronic_interference": 2,
    "boarding_seizure_risk": 2,
}

_GLOBAL_GEO_KEYWORD_SCORES: dict[str, int] = {
    "hormuz": 4,
    "strait of hormuz": 4,
    "red sea": 3,
    "gulf of aden": 3,
    "persian gulf": 3,
    "shipping": 3,
    "maritime": 2,
    "naval": 2,
    "boarding": 3,
    "seizure": 3,
    "detention": 2,
    "drone": 3,
    "missile": 3,
    "strike": 3,
    "military": 2,
    "conflict": 2,
    "sanction": 2,
    "embargo": 2,
    "ceasefire": 2,
    "attack": 2,
    "terror": 2,
    "cyber": 2,
    "critical infrastructure": 3,
    "electronic interference": 3,
}

_GLOBAL_GEO_DOMAIN_WEIGHTS: dict[str, int] = {
    "ukmto.org": 3,
    "maritime.dot.gov": 2,
    "imo.org": 2,
    "reuters.com": 2,
    "apnews.com": 2,
    "aljazeera.com": 2,
    "bbc.com": 2,
}

_GLOBAL_GEO_SLUG_STOPWORDS: set[str] = {
    "advisory",
    "article",
    "news",
    "update",
    "report",
    "story",
    "index",
    "html",
    "media",
    "products",
    "watch",
}

_PROXIMITY_TERM_RE = re.compile(r"[a-z][a-z0-9_.-]{2,}")
_PROXIMITY_STOPWORDS: set[str] = {
    "about",
    "after",
    "against",
    "allow",
    "build",
    "change",
    "closed",
    "commit",
    "context",
    "default",
    "details",
    "discussion",
    "docs",
    "entry",
    "feature",
    "first",
    "fix",
    "github",
    "have",
    "https",
    "improve",
    "issue",
    "items",
    "lane",
    "latest",
    "merge",
    "minor",
    "notes",
    "open",
    "patch",
    "pull",
    "readme",
    "refactor",
    "release",
    "request",
    "review",
    "score",
    "signal",
    "state",
    "summary",
    "tests",
    "thread",
    "title",
    "update",
    "work",
}

_PROXIMITY_HMM_STATES: tuple[str, ...] = (
    "background",
    "emerging",
    "active",
    "critical",
)

_PROXIMITY_HMM_PRIOR: tuple[float, ...] = (0.90, 0.08, 0.02, 0.00)

_PROXIMITY_HMM_TRANSITIONS: tuple[tuple[float, ...], ...] = (
    (0.86, 0.14, 0.00, 0.00),
    (0.10, 0.75, 0.14, 0.01),
    (0.03, 0.14, 0.70, 0.13),
    (0.01, 0.03, 0.18, 0.78),
)

_PROXIMITY_HMM_MEANS: tuple[float, ...] = (0.08, 0.34, 0.62, 0.88)
_PROXIMITY_HMM_STDDEV = 0.20

_CYBER_REGIME_STATES: tuple[str, ...] = (
    "baseline",
    "elevated_chatter",
    "active_exploitation_wave",
    "supply_chain_campaign",
    "geopolitical_targeting_shift",
)
_CYBER_REGIME_PRIOR: tuple[float, ...] = (0.78, 0.14, 0.04, 0.02, 0.02)
_CYBER_REGIME_TRANSITIONS: tuple[tuple[float, ...], ...] = (
    (0.84, 0.11, 0.03, 0.01, 0.01),
    (0.17, 0.66, 0.10, 0.03, 0.04),
    (0.08, 0.12, 0.64, 0.12, 0.04),
    (0.06, 0.09, 0.14, 0.66, 0.05),
    (0.10, 0.14, 0.09, 0.04, 0.63),
)
_CYBER_REGIME_MEANS: tuple[float, ...] = (0.09, 0.28, 0.63, 0.78, 0.56)
_CYBER_REGIME_STDDEV = 0.19


_THREAT_RADAR_CLASSIFIER_ENABLED = str(
    os.getenv("THREAT_RADAR_CLASSIFIER_ENABLED", "1") or "1"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_THREAT_RADAR_CLASSIFIER_VERSION = (
    str(
        os.getenv("THREAT_RADAR_CLASSIFIER_VERSION", "github_linear_v1")
        or "github_linear_v1"
    ).strip()
    or "github_linear_v1"
)

_GITHUB_LABEL_FUNCTIONS_VERSION = "github_lf_v1"

_GITHUB_LINEAR_CLASSIFIER_BIAS = _safe_float(
    os.getenv("THREAT_RADAR_GITHUB_LINEAR_BIAS", "-2.40") or "-2.40",
    -2.40,
)

_GITHUB_LINEAR_CLASSIFIER_WEIGHTS: dict[str, float] = {
    "is_advisory": _safe_float(
        os.getenv("THREAT_RADAR_GITHUB_W_IS_ADVISORY", "3.40") or "3.40", 3.40
    ),
    "cve_count": _safe_float(
        os.getenv("THREAT_RADAR_GITHUB_W_CVE_COUNT", "1.80") or "1.80", 1.80
    ),
    "changes_dependency": _safe_float(
        os.getenv("THREAT_RADAR_GITHUB_W_CHANGES_DEP", "1.10") or "1.10", 1.10
    ),
    "security_label": _safe_float(
        os.getenv("THREAT_RADAR_GITHUB_W_SECURITY_LABEL", "0.70") or "0.70", 0.70
    ),
    "mentions_security_term": _safe_float(
        os.getenv("THREAT_RADAR_GITHUB_W_MENTION_TERM", "0.45") or "0.45", 0.45
    ),
    "title_security_term": _safe_float(
        os.getenv("THREAT_RADAR_GITHUB_W_TITLE_TERM", "0.55") or "0.55", 0.55
    ),
    "open_state": _safe_float(
        os.getenv("THREAT_RADAR_GITHUB_W_OPEN_STATE", "0.25") or "0.25", 0.25
    ),
    "pr_merged": _safe_float(
        os.getenv("THREAT_RADAR_GITHUB_W_PR_MERGED", "0.30") or "0.30", 0.30
    ),
    "importance_scaled": _safe_float(
        os.getenv("THREAT_RADAR_GITHUB_W_IMPORTANCE", "0.90") or "0.90", 0.90
    ),
    "security_term_density": _safe_float(
        os.getenv("THREAT_RADAR_GITHUB_W_TERM_DENSITY", "0.80") or "0.80", 0.80
    ),
}


_THREAT_RADAR_LLM_ENABLED = str(
    os.getenv("THREAT_RADAR_LLM_ENABLED", "0") or "0"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_THREAT_RADAR_LLM_MODEL = (
    str(
        os.getenv(
            "THREAT_RADAR_LLM_MODEL",
            os.getenv("TEXT_GENERATION_MODEL", "qwen3-vl:4b-instruct")
            or "qwen3-vl:4b-instruct",
        )
        or "qwen3-vl:4b-instruct"
    ).strip()
    or "qwen3-vl:4b-instruct"
)
_THREAT_RADAR_LLM_TIMEOUT_SEC = max(
    0.5,
    min(
        45.0,
        _safe_float(
            os.getenv("THREAT_RADAR_LLM_TIMEOUT_SEC", "3") or "3",
            3.0,
        ),
    ),
)
_THREAT_RADAR_LLM_MAX_ITEMS = max(
    1,
    min(
        24,
        _safe_int(os.getenv("THREAT_RADAR_LLM_MAX_ITEMS", "6") or "6", 6),
    ),
)
_THREAT_RADAR_LLM_MAX_TOKENS = max(
    128,
    min(
        2048,
        _safe_int(os.getenv("THREAT_RADAR_LLM_MAX_TOKENS", "768") or "768", 768),
    ),
)
_THREAT_RADAR_LLM_CACHE_TTL_SEC = max(
    2.0,
    min(
        3600.0,
        _safe_float(os.getenv("THREAT_RADAR_LLM_CACHE_TTL_SEC", "300") or "300", 300.0),
    ),
)
_THREAT_RADAR_LLM_CACHE_MAX = max(
    32,
    min(
        8192,
        _safe_int(os.getenv("THREAT_RADAR_LLM_CACHE_MAX", "1024") or "1024", 1024),
    ),
)
_THREAT_RADAR_LLM_CACHE_LOCK = threading.Lock()
_THREAT_RADAR_LLM_CACHE: dict[str, dict[str, Any]] = {}

_WORLD_WATCHLIST_PATH = (
    Path(__file__).resolve().parents[2]
    / "world_state"
    / "config"
    / "world_watchlist.json"
)
_WEAVER_SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[2]
    / "world_state"
    / "web_graph_weaver.snapshot.json"
)
_WEAVER_FETCHED_SOURCE_CACHE_LOCK = threading.Lock()
_WEAVER_FETCHED_SOURCE_CACHE: dict[str, Any] = {
    "mtime": -1.0,
    "rows": [],
}


def _normalize_source_url(url: str) -> str:
    target = str(url or "").strip()
    if not target:
        return ""
    try:
        parsed = urlparse(target)
    except Exception:
        return ""
    scheme = str(parsed.scheme or "").strip().lower()
    if scheme not in {"http", "https"}:
        return ""
    host = str(parsed.netloc or "").strip().lower()
    if not host:
        return ""
    if scheme == "http" and host.endswith(":80"):
        host = host[: -len(":80")]
    elif scheme == "https" and host.endswith(":443"):
        host = host[: -len(":443")]

    path = re.sub(r"/{2,}", "/", str(parsed.path or "/"))
    if not path:
        path = "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    query_rows = parse_qsl(str(parsed.query or ""), keep_blank_values=True)
    query_rows.sort(key=lambda row: (str(row[0]), str(row[1])))
    query = urlencode(query_rows, doseq=True)

    return urlunparse((scheme, host, path, "", query, ""))


def _normalize_epoch_seconds(value: Any) -> float:
    raw_value = max(0.0, _safe_float(value, 0.0))
    if raw_value <= 0.0:
        return 0.0
    if raw_value > 10_000_000_000.0:
        raw_value /= 1000.0
    return round(raw_value, 6)


def _load_weaver_fetched_source_rows() -> list[dict[str, Any]]:
    path = _WEAVER_SNAPSHOT_PATH
    if not path.exists() or not path.is_file():
        return []

    try:
        mtime = float(path.stat().st_mtime)
    except Exception:
        mtime = -1.0

    with _WEAVER_FETCHED_SOURCE_CACHE_LOCK:
        cache_mtime = _safe_float(_WEAVER_FETCHED_SOURCE_CACHE.get("mtime", -1.0), -1.0)
        if mtime >= 0.0 and abs(cache_mtime - mtime) < 1e-9:
            cached_rows = _WEAVER_FETCHED_SOURCE_CACHE.get("rows", [])
            if isinstance(cached_rows, list):
                return [row for row in cached_rows if isinstance(row, dict)]

    try:
        payload = json.loads(path.read_text("utf-8"))
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []

    graph_payload = (
        payload.get("graph", {}) if isinstance(payload.get("graph", {}), dict) else {}
    )
    node_rows = (
        graph_payload.get("nodes", [])
        if isinstance(graph_payload.get("nodes", []), list)
        else []
    )

    dedupe_rows: dict[str, dict[str, Any]] = {}
    for node in node_rows:
        if not isinstance(node, dict):
            continue
        kind = str(node.get("kind", "") or "").strip().lower()
        if kind != "url":
            continue
        raw_url = str(
            node.get("url", "") or node.get("canonical_url", "") or ""
        ).strip()
        normalized_url = _normalize_source_url(raw_url)
        if not normalized_url:
            continue
        status = str(node.get("status", "") or "").strip().lower()
        fetched_ts = _normalize_epoch_seconds(
            node.get("fetched_at", node.get("last_visited_at", 0.0))
        )
        if status != "fetched" and fetched_ts <= 0.0:
            continue

        row = {
            "url": raw_url,
            "normalized_url": normalized_url,
            "fetched_ts": fetched_ts,
            "title": str(
                node.get("title", "") or node.get("feed_entry_title", "") or ""
            ).strip(),
            "summary": str(
                node.get("analysis_summary", "")
                or node.get("feed_entry_summary", "")
                or node.get("text_excerpt", "")
                or ""
            ).strip(),
            "text_excerpt": str(node.get("text_excerpt", "") or "").strip(),
        }
        existing = dedupe_rows.get(normalized_url)
        if not isinstance(existing, dict):
            dedupe_rows[normalized_url] = row
            continue

        existing_fetched_ts = _safe_float(existing.get("fetched_ts", 0.0), 0.0)
        if fetched_ts > existing_fetched_ts:
            dedupe_rows[normalized_url] = row

    rows = sorted(
        [row for row in dedupe_rows.values() if isinstance(row, dict)],
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("normalized_url", "")),
        ),
    )

    with _WEAVER_FETCHED_SOURCE_CACHE_LOCK:
        _WEAVER_FETCHED_SOURCE_CACHE["mtime"] = mtime
        _WEAVER_FETCHED_SOURCE_CACHE["rows"] = rows

    return rows


def _load_world_watchlist_sources() -> list[dict[str, Any]]:
    path = _WORLD_WATCHLIST_PATH
    if not path.exists() or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text("utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    if payload.get("enabled", True) is False:
        return []

    rows: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    domains = (
        payload.get("domains", [])
        if isinstance(payload.get("domains", []), list)
        else []
    )
    for domain_row in domains:
        if not isinstance(domain_row, dict):
            continue
        if domain_row.get("enabled", True) is False:
            continue
        domain_id = str(domain_row.get("id", "") or "").strip().lower()
        seed_rows = (
            domain_row.get("seed_urls", [])
            if isinstance(domain_row.get("seed_urls", []), list)
            else []
        )
        for seed_row in seed_rows:
            url = ""
            kind = ""
            title = ""
            source_type = ""
            if isinstance(seed_row, str):
                url = str(seed_row).strip()
            elif isinstance(seed_row, dict):
                url = str(seed_row.get("url", "") or "").strip()
                kind = str(seed_row.get("kind", "") or "").strip().lower()
                title = str(seed_row.get("title", "") or "").strip()
                source_type = str(seed_row.get("source_type", "") or "").strip().lower()

            if not url:
                continue
            normalized = url.lower()
            if normalized in seen_urls:
                continue
            if not (
                normalized.startswith("https://") or normalized.startswith("http://")
            ):
                continue
            seen_urls.add(normalized)
            rows.append(
                {
                    "url": url,
                    "kind": kind,
                    "title": title,
                    "source_type": source_type,
                    "domain_id": domain_id,
                }
            )

    rows.sort(
        key=lambda row: (
            str(row.get("domain_id", "")),
            str(row.get("kind", "")),
            str(row.get("url", "")),
        )
    )
    return rows


def _hormuz_watchlist_sources(kind_filter: str = "") -> list[dict[str, Any]]:
    rows = _load_world_watchlist_sources()
    filtered: list[dict[str, Any]] = []
    kind_token = str(kind_filter or "").strip().lower()
    for row in rows:
        if not isinstance(row, dict):
            continue
        domain_id = str(row.get("domain_id", "") or "").strip().lower()
        kind = str(row.get("kind", "") or "").strip().lower()
        url = str(row.get("url", "") or "").strip()
        if not url:
            continue
        if domain_id != "hormuz" and not kind.startswith("maritime:"):
            continue
        if kind_token and kind != kind_token:
            continue
        filtered.append(
            {
                "url": url,
                "kind": kind,
                "title": str(row.get("title", "") or "").strip(),
                "source_type": str(row.get("source_type", "") or "").strip(),
                "domain_id": domain_id,
            }
        )
    return filtered


def _node_role(node: dict[str, Any]) -> str:
    role = str(node.get("role", "") or "").strip().lower()
    if role:
        return role
    web_role = str(node.get("web_node_role", "") or "").strip().lower()
    if web_role:
        return web_role
    return (
        str(node.get("node_type", "unknown") or "unknown").strip().lower() or "unknown"
    )


def _node_extension(node: dict[str, Any]) -> dict[str, Any]:
    extension = node.get("extension", {}) if isinstance(node, dict) else {}
    return extension if isinstance(extension, dict) else {}


def _node_value(node: dict[str, Any], key: str, default: Any = "") -> Any:
    if key in node and node.get(key) is not None:
        value = node.get(key)
        if isinstance(value, str):
            if value.strip():
                return value
        else:
            return value
    extension = _node_extension(node)
    if key in extension:
        return extension.get(key)
    return default


def _node_list(node: dict[str, Any], key: str) -> list[Any]:
    value = _node_value(node, key, [])
    return value if isinstance(value, list) else []


def _node_canonical_url(node: dict[str, Any]) -> str:
    for key in ("canonical_url", "url", "href", "source_url"):
        value = str(_node_value(node, key, "") or "").strip()
        if value:
            return value
    return ""


def _node_title(node: dict[str, Any]) -> str:
    for key in ("title", "label", "name"):
        value = str(_node_value(node, key, "") or "").strip()
        if value:
            return value
    return ""


def _graph_rows(
    nexus_graph: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph = nexus_graph if isinstance(nexus_graph, dict) else {}
    nodes = [row for row in graph.get("nodes", []) if isinstance(row, dict)]
    edges = [row for row in graph.get("edges", []) if isinstance(row, dict)]
    return nodes, edges


def _node_index(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in nodes:
        node_id = str(row.get("id", "")).strip()
        if node_id and node_id not in by_id:
            by_id[node_id] = row
    return by_id


def _query_overview(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    role_counts: dict[str, int] = {}
    for node in nodes:
        role = _node_role(node)
        role_counts[role] = role_counts.get(role, 0) + 1

    degree: dict[str, int] = {}
    for edge in edges:
        source_id = str(edge.get("source", "")).strip()
        target_id = str(edge.get("target", "")).strip()
        if source_id:
            degree[source_id] = degree.get(source_id, 0) + 1
        if target_id:
            degree[target_id] = degree.get(target_id, 0) + 1

    top_degree_nodes = sorted(
        [{"id": node_id, "degree": count} for node_id, count in degree.items()],
        key=lambda row: (-_safe_int(row.get("degree", 0), 0), str(row.get("id", ""))),
    )[:24]

    edge_kind_counts: dict[str, int] = {}
    for edge in edges:
        kind = str(edge.get("kind", "relates") or "relates").strip().lower()
        edge_kind_counts[kind] = edge_kind_counts.get(kind, 0) + 1

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "role_counts": role_counts,
        "edge_kind_counts": edge_kind_counts,
        "top_degree_nodes": top_degree_nodes,
    }


def _query_neighbors(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    node_id = str(args.get("node_id", "")).strip()
    if not node_id and nodes:
        node_id = str(nodes[0].get("id", "")).strip()
    edge_role = str(args.get("edge_role", "")).strip().lower()

    rows: list[dict[str, Any]] = []
    for edge in edges:
        kind = str(edge.get("kind", "relates") or "relates").strip().lower()
        if edge_role and kind != edge_role:
            continue
        source_id = str(edge.get("source", "")).strip()
        target_id = str(edge.get("target", "")).strip()
        if source_id == node_id and target_id:
            rows.append({"node_id": target_id, "edge_kind": kind, "direction": "out"})
        elif target_id == node_id and source_id:
            rows.append({"node_id": source_id, "edge_kind": kind, "direction": "in"})

    rows.sort(
        key=lambda row: (
            str(row.get("direction", "")),
            str(row.get("edge_kind", "")),
            str(row.get("node_id", "")),
        )
    )
    return {
        "node_id": node_id,
        "edge_role": edge_role,
        "neighbor_count": len(rows),
        "neighbors": rows[:128],
    }


def _query_role_slice(
    nodes: list[dict[str, Any]], args: dict[str, Any]
) -> dict[str, Any]:
    role = str(args.get("role", "file") or "file").strip().lower()
    rows = [
        {
            "id": str(node.get("id", "")).strip(),
            "label": str(node.get("label", "") or "").strip(),
            "importance": round(_safe_float(node.get("importance", 0.0), 0.0), 6),
        }
        for node in nodes
        if _node_role(node) == role
    ]
    rows.sort(
        key=lambda row: (
            -_safe_float(row.get("importance", 0.0), 0.0),
            str(row.get("id", "")),
        )
    )
    return {
        "role": role,
        "count": len(rows),
        "nodes": rows[:128],
    }


def _query_search(nodes: list[dict[str, Any]], args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("q", args.get("query", "")) or "").strip().lower()
    if not query:
        return {"query": query, "count": 0, "nodes": []}

    rows: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        label = str(node.get("label", "") or "").strip()
        canonical_url = _node_canonical_url(node)
        title = _node_title(node)
        path = str(_node_value(node, "path", "") or "").strip()
        source_uri = str(_node_value(node, "source_uri", "") or "").strip()
        head = str(_node_value(node, "head", "") or "").strip()
        tags = [
            str(item).strip() for item in _node_list(node, "tags") if str(item).strip()
        ]
        refs = [
            str(item).strip() for item in _node_list(node, "refs") if str(item).strip()
        ]
        explicit_id = str(_node_value(node, "explicit_id", "") or "").strip()
        haystack = " ".join(
            [
                node_id.lower(),
                label.lower(),
                canonical_url.lower(),
                title.lower(),
                path.lower(),
                source_uri.lower(),
                head.lower(),
                explicit_id.lower(),
                " ".join(tag.lower() for tag in tags),
                " ".join(ref.lower() for ref in refs),
            ]
        ).strip()
        if query in haystack:
            rows.append(
                {
                    "id": node_id,
                    "role": _node_role(node),
                    "label": label,
                    "canonical_url": canonical_url,
                    "title": title,
                    "path": path,
                    "tags": tags[:8],
                }
            )

    rows.sort(key=lambda row: (str(row.get("role", "")), str(row.get("id", ""))))
    return {"query": query, "count": len(rows), "nodes": rows[:128]}


def _query_url_status(
    nodes: list[dict[str, Any]], args: dict[str, Any]
) -> dict[str, Any]:
    target = str(
        args.get("target", args.get("url", args.get("url_id", ""))) or ""
    ).strip()
    target_lower = target.lower()

    found: dict[str, Any] | None = None
    for node in nodes:
        role = _node_role(node)
        if role != "web:url":
            continue
        node_id = str(node.get("id", "")).strip()
        canonical_url = _node_canonical_url(node)
        if target and target_lower not in {node_id.lower(), canonical_url.lower()}:
            continue
        found = {
            "id": node_id,
            "canonical_url": canonical_url,
            "next_allowed_fetch_ts": round(
                max(0.0, _safe_float(node.get("next_allowed_fetch_ts", 0.0), 0.0)),
                6,
            ),
            "fail_count": max(0, _safe_int(node.get("fail_count", 0), 0)),
            "last_fetch_ts": round(
                max(0.0, _safe_float(node.get("last_fetch_ts", 0.0), 0.0)), 6
            ),
            "last_status": str(
                node.get("last_status", node.get("status", "")) or ""
            ).strip(),
            "title": _node_title(node),
        }
        break

    return {"target": target, "found": found}


def _query_resource_for_url(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    target = str(
        args.get("target", args.get("url_id", args.get("url", ""))) or ""
    ).strip()
    if not target:
        return {"target": "", "count": 0, "resources": []}

    node_by_id = _node_index(nodes)
    target_url_id = ""
    for node in nodes:
        if _node_role(node) != "web:url":
            continue
        node_id = str(node.get("id", "")).strip()
        canonical_url = _node_canonical_url(node)
        if target.lower() in {node_id.lower(), canonical_url.lower()}:
            target_url_id = node_id
            break

    if not target_url_id:
        return {"target": target, "count": 0, "resources": []}

    resources: list[dict[str, Any]] = []
    for edge in edges:
        if str(edge.get("kind", "")).strip().lower() != "web:source_of":
            continue
        source_id = str(edge.get("source", "")).strip()
        target_id = str(edge.get("target", "")).strip()
        if target_id != target_url_id:
            continue
        source_node = node_by_id.get(source_id, {})
        resources.append(
            {
                "id": source_id,
                "canonical_url": _node_canonical_url(source_node),
                "fetched_ts": round(
                    max(0.0, _safe_float(source_node.get("fetched_ts", 0.0), 0.0)), 6
                ),
                "content_hash": str(source_node.get("content_hash", "") or "").strip(),
            }
        )

    resources.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("id", "")),
        )
    )
    return {
        "target": target_url_id,
        "count": len(resources),
        "resources": resources[:64],
    }


def _query_recently_updated(
    nodes: list[dict[str, Any]], args: dict[str, Any]
) -> dict[str, Any]:
    limit = max(1, min(128, _safe_int(args.get("limit", 24), 24)))
    rows: list[dict[str, Any]] = []
    for node in nodes:
        role = _node_role(node)
        fetched_ts = max(
            0.0,
            _safe_float(node.get("fetched_ts", node.get("last_fetch_ts", 0.0)), 0.0),
        )
        if fetched_ts <= 0.0:
            continue
        rows.append(
            {
                "id": str(node.get("id", "")).strip(),
                "role": role,
                "canonical_url": _node_canonical_url(node),
                "fetched_ts": round(fetched_ts, 6),
            }
        )
    rows.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("id", "")),
        )
    )
    return {"count": len(rows), "nodes": rows[:limit]}


def _simulation_sections(
    simulation: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    payload = simulation if isinstance(simulation, dict) else {}
    dynamics = (
        payload.get("presence_dynamics", {})
        if isinstance(payload.get("presence_dynamics", {}), dict)
        else {}
    )
    crawler_graph = (
        payload.get("crawler_graph", {})
        if isinstance(payload.get("crawler_graph", {}), dict)
        else {}
    )
    field_particles = [
        row for row in dynamics.get("field_particles", []) if isinstance(row, dict)
    ]
    outcome_rows = [
        row
        for row in dynamics.get("daimoi_outcome_trails", [])
        if isinstance(row, dict)
    ]
    return dynamics, crawler_graph, field_particles, outcome_rows


def _truthy_query_flag(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    token = "" if value is None else str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _threat_compute_budget_from_simulation(
    simulation: dict[str, Any] | None,
) -> dict[str, Any]:
    dynamics, _crawler_graph, _field_particles, _outcome_rows = _simulation_sections(
        simulation
    )
    snapshot = build_control_budget_snapshot(dynamics)
    llm_item_cap_default = max(1, _safe_int(_THREAT_RADAR_LLM_MAX_ITEMS, 6))
    return apply_threat_compute_budget_policy(
        snapshot=snapshot,
        llm_item_cap_default=llm_item_cap_default,
    )


def _query_explain_daimoi(
    simulation: dict[str, Any] | None,
    args: dict[str, Any],
) -> dict[str, Any]:
    dynamics, _, field_particles, outcome_rows = _simulation_sections(simulation)
    target_id = str(
        args.get("daimoi_id", args.get("id", args.get("target", ""))) or ""
    ).strip()

    live_state: dict[str, Any] | None = None
    if target_id:
        for row in field_particles:
            if str(row.get("id", "") or "").strip() == target_id:
                live_state = row
                break
    elif field_particles:
        live_state = field_particles[0]
        target_id = str(live_state.get("id", "") or "").strip()

    matched_outcomes: list[dict[str, Any]] = []
    for row in outcome_rows:
        row_daimoi_id = str(row.get("daimoi_id", "") or "").strip()
        if target_id and row_daimoi_id != target_id:
            continue
        matched_outcomes.append(row)

    matched_outcomes.sort(
        key=lambda row: (
            _safe_int(row.get("tick", row.get("seq", 0)), 0),
            _safe_int(row.get("seq", 0), 0),
        )
    )
    last_outcome = matched_outcomes[-1] if matched_outcomes else None
    last_outcome_kind = (
        str(last_outcome.get("outcome", "") or "").strip().lower()
        if isinstance(last_outcome, dict)
        else ""
    )

    status = "unknown"
    if isinstance(live_state, dict) and live_state:
        status = "alive"
    elif last_outcome_kind in {"food", "death"}:
        status = last_outcome_kind

    reinforcement_direction = "none"
    if last_outcome_kind == "food":
        reinforcement_direction = "forward"
    elif last_outcome_kind == "death":
        reinforcement_direction = "inverse"

    return {
        "daimoi_id": target_id,
        "status": status,
        "live_state": (
            {
                "present": True,
                "owner": str(
                    live_state.get(
                        "presence_id",
                        live_state.get(
                            "owner_presence_id", live_state.get("owner", "")
                        ),
                    )
                    or ""
                ).strip(),
                "graph_node_id": str(live_state.get("graph_node_id", "") or "").strip(),
                "top_job": str(live_state.get("top_job", "") or "").strip(),
                "collision_count": max(
                    0,
                    _safe_int(
                        live_state.get(
                            "collision_count", live_state.get("collisions", 0)
                        ),
                        0,
                    ),
                ),
                "message_probability": round(
                    max(
                        0.0,
                        _safe_float(live_state.get("message_probability", 0.0), 0.0),
                    ),
                    6,
                ),
            }
            if isinstance(live_state, dict)
            else {"present": False}
        ),
        "last_outcome": (
            {
                "outcome": last_outcome_kind,
                "reason": str(last_outcome.get("reason", "") or "").strip(),
                "target_id": str(last_outcome.get("graph_node_id", "") or "").strip(),
                "intensity": round(
                    max(0.0, _safe_float(last_outcome.get("intensity", 0.0), 0.0)),
                    6,
                ),
                "tick": max(
                    0,
                    _safe_int(last_outcome.get("tick", last_outcome.get("seq", 0)), 0),
                ),
                "trail_steps": max(
                    0,
                    _safe_int(last_outcome.get("trail_steps", 0), 0),
                ),
                "ts": str(last_outcome.get("ts", "") or "").strip(),
            }
            if isinstance(last_outcome, dict)
            else None
        ),
        "reinforcement": {
            "direction": reinforcement_direction,
            "intensity": round(
                max(
                    0.0,
                    _safe_float(
                        (last_outcome or {}).get("intensity", 0.0)
                        if isinstance(last_outcome, dict)
                        else 0.0,
                        0.0,
                    ),
                ),
                6,
            ),
        },
        "outcome_count": len(matched_outcomes),
        "tick": max(0, _safe_int(dynamics.get("tick", 0), 0)),
    }


def _query_recent_outcomes(
    simulation: dict[str, Any] | None,
    args: dict[str, Any],
) -> dict[str, Any]:
    _, _, _, outcome_rows = _simulation_sections(simulation)
    window_ticks = max(1, min(20_000, _safe_int(args.get("window_ticks", 360), 360)))
    limit = max(1, min(256, _safe_int(args.get("limit", 48), 48)))

    normalized_rows = [
        {
            "tick": max(0, _safe_int(row.get("tick", row.get("seq", 0)), 0)),
            "seq": max(0, _safe_int(row.get("seq", 0), 0)),
            "daimoi_id": str(row.get("daimoi_id", "") or "").strip(),
            "outcome": str(row.get("outcome", "") or "").strip().lower(),
            "reason": str(row.get("reason", "") or "").strip(),
            "target_id": str(row.get("graph_node_id", "") or "").strip(),
            "intensity": round(
                max(0.0, _safe_float(row.get("intensity", 0.0), 0.0)), 6
            ),
            "ts": str(row.get("ts", "") or "").strip(),
        }
        for row in outcome_rows
    ]
    normalized_rows.sort(
        key=lambda row: (
            _safe_int(row.get("tick", 0), 0),
            _safe_int(row.get("seq", 0), 0),
        )
    )

    max_tick = max(
        [max(0, _safe_int(row.get("tick", 0), 0)) for row in normalized_rows],
        default=0,
    )
    min_tick = max(0, max_tick - window_ticks + 1)
    windowed = [
        row
        for row in normalized_rows
        if max(0, _safe_int(row.get("tick", 0), 0)) >= min_tick
    ]
    return {
        "window_ticks": int(window_ticks),
        "count": len(windowed),
        "outcomes": windowed[-limit:],
    }


def _query_crawler_status(simulation: dict[str, Any] | None) -> dict[str, Any]:
    _, crawler_graph, _, _ = _simulation_sections(simulation)
    stats = crawler_graph.get("stats", {}) if isinstance(crawler_graph, dict) else {}
    status = crawler_graph.get("status", {}) if isinstance(crawler_graph, dict) else {}
    nodes = [row for row in crawler_graph.get("nodes", []) if isinstance(row, dict)]
    url_nodes = [row for row in nodes if _node_role(row) == "web:url"]
    now_epoch = time.time()

    cooldown_active = 0
    fail_total = 0
    for row in url_nodes:
        next_allowed = max(0.0, _safe_float(row.get("next_allowed_fetch_ts", 0.0), 0.0))
        if next_allowed > now_epoch:
            cooldown_active += 1
        fail_total += max(0, _safe_int(row.get("fail_count", 0), 0))

    last_fetch_rows = [
        {
            "url_id": str(row.get("id", "") or "").strip(),
            "canonical_url": _node_canonical_url(row),
            "title": _node_title(row),
            "last_fetch_ts": round(
                max(0.0, _safe_float(row.get("last_fetch_ts", 0.0), 0.0)),
                6,
            ),
            "last_status": str(row.get("last_status", "") or "").strip(),
            "next_allowed_fetch_ts": round(
                max(0.0, _safe_float(row.get("next_allowed_fetch_ts", 0.0), 0.0)),
                6,
            ),
        }
        for row in url_nodes
    ]
    last_fetch_rows.sort(
        key=lambda row: (
            -_safe_float(row.get("last_fetch_ts", 0.0), 0.0),
            str(row.get("url_id", "")),
        )
    )

    queue_length = max(
        0,
        _safe_int(
            status.get(
                "queue_length", status.get("pending_count", status.get("queue", 0))
            ),
            0,
        ),
    )

    arxiv_abs_rows = [
        row for row in url_nodes if "arxiv.org/abs/" in _node_canonical_url(row)
    ]
    arxiv_abs_fetched = [
        row
        for row in arxiv_abs_rows
        if str(row.get("last_status", "") or "").strip().lower() == "ok"
        and _safe_float(row.get("last_fetch_ts", 0.0), 0.0) > 0.0
    ]
    arxiv_recent_fetches = [
        row
        for row in last_fetch_rows
        if "arxiv.org/abs/" in str(row.get("canonical_url", ""))
    ]

    return {
        "queue_length": int(queue_length),
        "cooldown_active": int(cooldown_active),
        "url_node_count": len(url_nodes),
        "resource_node_count": max(
            0,
            _safe_int(
                stats.get(
                    "web_role_counts",
                    {},
                ).get("web:resource", 0)
                if isinstance(stats.get("web_role_counts", {}), dict)
                else 0,
                0,
            ),
        ),
        "fail_count_total": int(fail_total),
        "event_count": max(0, _safe_int(stats.get("event_count", 0), 0)),
        "arxiv_abs_count": len(arxiv_abs_rows),
        "arxiv_abs_fetched": len(arxiv_abs_fetched),
        "arxiv_recent_fetches": arxiv_recent_fetches[:8],
        "last_fetches": last_fetch_rows[:8],
    }


def _query_arxiv_papers(
    simulation: dict[str, Any] | None,
    args: dict[str, Any],
) -> dict[str, Any]:
    _, crawler_graph, _, _ = _simulation_sections(simulation)
    nodes = [row for row in crawler_graph.get("nodes", []) if isinstance(row, dict)]
    limit = max(
        1,
        min(
            64,
            _safe_int(args.get("limit", args.get("n", args.get("count", 8))), 8),
        ),
    )

    papers: list[dict[str, Any]] = []
    for node in nodes:
        if _node_role(node) != "web:url":
            continue
        canonical_url = _node_canonical_url(node)
        if "arxiv.org/abs/" not in canonical_url:
            continue
        arxiv_id = ""
        if "/abs/" in canonical_url:
            arxiv_id = (
                canonical_url.split("/abs/", 1)[1].split("?", 1)[0].split("#", 1)[0]
            )

        last_status = str(node.get("last_status", node.get("status", "")) or "").strip()
        last_fetch_ts = max(
            0.0,
            _safe_float(node.get("last_fetch_ts", node.get("fetched_ts", 0.0)), 0.0),
        )
        fetched = last_status.lower() == "ok" and last_fetch_ts > 0.0
        papers.append(
            {
                "url_id": str(node.get("id", "") or "").strip(),
                "arxiv_id": arxiv_id,
                "canonical_url": canonical_url,
                "title": _node_title(node),
                "last_status": last_status,
                "last_fetch_ts": round(last_fetch_ts, 6),
                "fetched": bool(fetched),
            }
        )

    papers.sort(
        key=lambda row: (
            0 if bool(row.get("fetched", False)) else 1,
            -_safe_float(row.get("last_fetch_ts", 0.0), 0.0),
            str(row.get("canonical_url", "")),
        )
    )
    fetched_count = sum(1 for row in papers if bool(row.get("fetched", False)))
    return {
        "count_total": len(papers),
        "count_fetched": fetched_count,
        "papers": papers[:limit],
    }


def _clamp_score_100(value: Any) -> int:
    numeric = _safe_float(value, 0.0)
    if numeric < 0.0:
        return 0
    if numeric > 100.0:
        return 100
    return int(round(numeric))


def _threat_row_identity(domain: str, row: dict[str, Any]) -> str:
    clean_domain = str(domain or "generic").strip().lower() or "generic"
    canonical_url = str(row.get("canonical_url", "") or "").strip().lower()
    kind = str(row.get("kind", "") or "").strip().lower()
    title = str(row.get("title", "") or "").strip()
    repo = str(row.get("repo", "") or "").strip().lower()
    number = max(0, _safe_int(row.get("number", 0), 0))

    if clean_domain == "github" and repo and number > 0:
        return f"github:{repo}#{number}"
    if canonical_url:
        return f"{clean_domain}:{canonical_url}"
    if kind and title:
        return f"{clean_domain}:{kind}:{title.lower()}"
    digest = hashlib.sha1(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{clean_domain}:{digest}"


def _threat_llm_candidate_payload(domain: str, row: dict[str, Any]) -> dict[str, Any]:
    base = {
        "id": _threat_row_identity(domain, row),
        "domain": str(domain or "generic").strip().lower(),
        "kind": str(row.get("kind", "") or "").strip().lower(),
        "title": str(row.get("title", "") or "").strip()[:280],
        "canonical_url": str(row.get("canonical_url", "") or "").strip(),
        "state": str(row.get("state", "") or "").strip().lower(),
        "repo": str(row.get("repo", "") or "").strip(),
        "number": max(0, _safe_int(row.get("number", 0), 0)),
        "signals": [
            str(token).strip().lower()
            for token in (
                row.get("signals", [])
                if isinstance(row.get("signals", []), list)
                else []
            )
            if str(token).strip()
        ][:12],
        "labels": [
            str(token).strip().lower()
            for token in (
                row.get("labels", []) if isinstance(row.get("labels", []), list) else []
            )
            if str(token).strip()
        ][:12],
        "cves": [
            str(token).strip().upper()
            for token in (
                row.get("cves", []) if isinstance(row.get("cves", []), list) else []
            )
            if str(token).strip()
        ][:12],
        "summary": str(row.get("summary", "") or "").strip()[:360],
        "text_excerpt": str(row.get("text_excerpt", "") or "").strip()[:420],
        "deterministic_score": max(0, _safe_int(row.get("deterministic_score", 0), 0)),
    }
    return base


def _threat_llm_prompt(domain: str, items: list[dict[str, Any]]) -> str:
    safe_domain = str(domain or "generic").strip().lower() or "generic"
    payload = {
        "domain": safe_domain,
        "items": items,
    }
    rubric = [
        "You are a threat scoring analyst.",
        "Evaluate each item and return strict JSON only (no markdown, no prose).",
        "Output schema:",
        '{"record":"eta-mu.threat-metrics.v1","domain":"<domain>","items":[{"id":"...","overall_score":0-100,"confidence":0-100,"severity":0-100,"immediacy":0-100,"impact":0-100,"exploitability":0-100,"credibility":0-100,"exposure":0-100,"novelty":0-100,"operational_risk":0-100,"rationale":"<=180 chars"}]}',
        "Scoring guidance:",
        "- Use higher overall_score for credible, severe, and near-term threats.",
        "- Penalize weak evidence and vague titles with lower confidence and credibility.",
        "- Keep rationale short and concrete.",
        "Only include IDs provided in input.",
        "INPUT_JSON:",
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    ]
    return "\n".join(rubric)


def _extract_first_json_object(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, count=1, flags=re.IGNORECASE).strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    candidates: list[str] = [text]
    first_open = text.find("{")
    last_close = text.rfind("}")
    if first_open >= 0 and last_close > first_open:
        candidates.append(text[first_open : last_close + 1])

    first_list_open = text.find("[")
    last_list_close = text.rfind("]")
    if first_list_open >= 0 and last_list_close > first_list_open:
        candidates.append(text[first_list_open : last_list_close + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"items": payload}

        try:
            repaired = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if isinstance(repaired, dict):
            return repaired
        if isinstance(repaired, list):
            return {"items": repaired}
    return None


def _threat_llm_metrics_from_payload(
    payload: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], str]:
    rows = (
        payload.get("items", []) if isinstance(payload.get("items", []), list) else []
    )
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("id", "") or "").strip()
        if not item_id:
            continue

        metrics = {
            "overall_score": _clamp_score_100(row.get("overall_score", 0)),
            "confidence": _clamp_score_100(row.get("confidence", 0)),
            "severity": _clamp_score_100(row.get("severity", 0)),
            "immediacy": _clamp_score_100(row.get("immediacy", 0)),
            "impact": _clamp_score_100(row.get("impact", 0)),
            "exploitability": _clamp_score_100(row.get("exploitability", 0)),
            "credibility": _clamp_score_100(row.get("credibility", 0)),
            "exposure": _clamp_score_100(row.get("exposure", 0)),
            "novelty": _clamp_score_100(row.get("novelty", 0)),
            "operational_risk": _clamp_score_100(row.get("operational_risk", 0)),
            "rationale": str(row.get("rationale", "") or "").strip()[:180],
        }
        by_id[item_id] = metrics

    model_name = str(payload.get("model", "") or "").strip()
    return by_id, model_name


def _threat_llm_metrics(
    *,
    domain: str,
    rows: list[dict[str, Any]],
    max_items: int | None = None,
) -> dict[str, Any]:
    if not _THREAT_RADAR_LLM_ENABLED:
        return {
            "applied": False,
            "enabled": False,
            "model": _THREAT_RADAR_LLM_MODEL,
            "error": "disabled",
            "metrics": {},
        }

    item_cap = _THREAT_RADAR_LLM_MAX_ITEMS
    if max_items is not None:
        item_cap = max(0, min(_THREAT_RADAR_LLM_MAX_ITEMS, _safe_int(max_items, 0)))
    candidates = [_threat_llm_candidate_payload(domain, row) for row in rows[:item_cap]]
    if not candidates:
        return {
            "applied": False,
            "enabled": True,
            "model": _THREAT_RADAR_LLM_MODEL,
            "error": "no_candidates",
            "metrics": {},
        }

    cache_key = _canonical_hash(
        {
            "domain": str(domain or "").strip().lower(),
            "model": _THREAT_RADAR_LLM_MODEL,
            "candidates": candidates,
        }
    )
    now_epoch = time.time()
    with _THREAT_RADAR_LLM_CACHE_LOCK:
        cached = _THREAT_RADAR_LLM_CACHE.get(cache_key)
        if isinstance(cached, dict):
            cached_ts = _safe_float(cached.get("ts", 0.0), 0.0)
            if now_epoch - cached_ts <= _THREAT_RADAR_LLM_CACHE_TTL_SEC:
                return {
                    "applied": bool(cached.get("applied", False)),
                    "enabled": True,
                    "model": str(
                        cached.get("model", _THREAT_RADAR_LLM_MODEL)
                        or _THREAT_RADAR_LLM_MODEL
                    ),
                    "error": str(cached.get("error", "") or ""),
                    "metrics": dict(cached.get("metrics", {}))
                    if isinstance(cached.get("metrics", {}), dict)
                    else {},
                    "cache": "hit",
                }

    try:
        try:
            from .ai import _ollama_generate_text_remote  # type: ignore
        except Exception:
            from ai import _ollama_generate_text_remote  # type: ignore
    except Exception:
        return {
            "applied": False,
            "enabled": True,
            "model": _THREAT_RADAR_LLM_MODEL,
            "error": "llm_import_failed",
            "metrics": {},
        }

    prompt = _threat_llm_prompt(domain, candidates)
    model_name = _THREAT_RADAR_LLM_MODEL
    error_text = ""
    metrics_by_id: dict[str, dict[str, Any]] = {}
    applied = False

    try:
        text, resolved_model = _ollama_generate_text_remote(
            prompt,
            model=_THREAT_RADAR_LLM_MODEL,
            timeout_s=_THREAT_RADAR_LLM_TIMEOUT_SEC,
            max_tokens=_THREAT_RADAR_LLM_MAX_TOKENS,
        )
        model_name = str(resolved_model or model_name).strip() or model_name
        payload = _extract_first_json_object(str(text or ""))
        if not isinstance(payload, dict):
            error_text = "llm_invalid_json"
        else:
            metrics_by_id, payload_model = _threat_llm_metrics_from_payload(payload)
            if payload_model:
                model_name = payload_model
            applied = len(metrics_by_id) > 0
            if not applied:
                error_text = "llm_empty_metrics"
    except Exception as exc:
        error_text = f"llm_error:{exc.__class__.__name__}"

    with _THREAT_RADAR_LLM_CACHE_LOCK:
        _THREAT_RADAR_LLM_CACHE[cache_key] = {
            "ts": now_epoch,
            "applied": bool(applied),
            "model": model_name,
            "error": error_text,
            "metrics": metrics_by_id,
        }
        if len(_THREAT_RADAR_LLM_CACHE) > _THREAT_RADAR_LLM_CACHE_MAX:
            overflow = len(_THREAT_RADAR_LLM_CACHE) - _THREAT_RADAR_LLM_CACHE_MAX
            oldest = sorted(
                _THREAT_RADAR_LLM_CACHE.items(),
                key=lambda item: _safe_float(item[1].get("ts", 0.0), 0.0),
            )[: max(0, overflow)]
            for key, _value in oldest:
                _THREAT_RADAR_LLM_CACHE.pop(key, None)

    return {
        "applied": bool(applied),
        "enabled": True,
        "model": model_name,
        "error": error_text,
        "metrics": metrics_by_id,
        "cache": "miss",
    }


def _blend_llm_score(
    deterministic_score: int, llm_overall: Any, *, max_score: int
) -> int:
    base = max(0, _safe_int(deterministic_score, 0))
    overall = _clamp_score_100(llm_overall)
    llm_normalized = int(round((overall / 100.0) * max(1, int(max_score))))
    return max(0, int(round((base * 0.35) + (llm_normalized * 0.65))))


def _github_resource_rows(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in nodes:
        if _node_role(node) != "web:resource":
            continue
        kind = str(_node_value(node, "kind", "") or "").strip().lower()
        if not kind.startswith("github:"):
            continue

        labels = [
            str(item).strip()
            for item in _node_list(node, "labels")
            if str(item).strip()
        ]
        authors = [
            str(item).strip()
            for item in _node_list(node, "authors")
            if str(item).strip()
        ]
        atoms = [row for row in _node_list(node, "atoms") if isinstance(row, dict)]
        rows.append(
            {
                "id": str(node.get("id", "") or "").strip(),
                "kind": kind,
                "canonical_url": _node_canonical_url(node),
                "title": _node_title(node),
                "repo": str(_node_value(node, "repo", "") or "").strip(),
                "number": max(0, _safe_int(_node_value(node, "number", 0), 0)),
                "labels": labels,
                "authors": authors,
                "updated_at": str(_node_value(node, "updated_at", "") or "").strip(),
                "summary": str(_node_value(node, "summary", "") or "").strip(),
                "text_excerpt": str(
                    _node_value(node, "text_excerpt", "") or ""
                ).strip(),
                "fetched_ts": round(
                    max(0.0, _safe_float(_node_value(node, "fetched_ts", 0.0), 0.0)),
                    6,
                ),
                "content_hash": str(
                    _node_value(node, "content_hash", "") or ""
                ).strip(),
                "importance_score": max(
                    0,
                    _safe_int(_node_value(node, "importance_score", 0), 0),
                ),
                "atoms": atoms,
                "state": str(_node_value(node, "state", "") or "").strip().lower(),
                "merged_at": str(_node_value(node, "merged_at", "") or "").strip(),
            }
        )

    rows.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("id", "")),
        )
    )
    return rows


def _is_github_like_url(url: str) -> bool:
    lowered = str(url or "").strip().lower()
    if not lowered:
        return False
    host = _url_host_token(lowered)
    if host:
        if host == "github.com" or host.endswith(".github.com"):
            return True
        if host == "githubusercontent.com" or host.endswith(".githubusercontent.com"):
            return True
        if host == "githubassets.com" or host.endswith(".githubassets.com"):
            return True
    return "github.com/" in lowered or "githubusercontent.com/" in lowered


def _url_host_token(url: str) -> str:
    target = str(url or "").strip()
    if not target:
        return ""
    try:
        host = urlparse(target).netloc.strip().lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _url_slug_tokens(url: str) -> list[str]:
    target = str(url or "").strip()
    if not target:
        return []
    try:
        path = urlparse(target).path.strip().lower()
    except Exception:
        path = ""
    if not path:
        return []
    tokens = re.findall(r"[a-z0-9][a-z0-9-]{2,48}", path)
    output: list[str] = []
    for token in tokens:
        if token.isdigit():
            continue
        if token in _GLOBAL_GEO_SLUG_STOPWORDS:
            continue
        output.append(token)
    return output


def _web_edge_counters(
    edges: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int], dict[str, str], set[str]]:
    inbound_by_url: dict[str, int] = {}
    links_by_resource: dict[str, int] = {}
    source_url_by_resource: dict[str, str] = {}
    fetched_url_ids: set[str] = set()

    for edge in edges:
        kind = str(edge.get("kind", "") or "").strip().lower()
        source_id = str(edge.get("source", "") or "").strip()
        target_id = str(edge.get("target", "") or "").strip()
        if kind == "web:links_to" and target_id:
            inbound_by_url[target_id] = inbound_by_url.get(target_id, 0) + 1
            if source_id:
                links_by_resource[source_id] = links_by_resource.get(source_id, 0) + 1
        elif kind == "web:source_of" and source_id and target_id:
            source_url_by_resource[source_id] = target_id
            fetched_url_ids.add(target_id)

    return (inbound_by_url, links_by_resource, source_url_by_resource, fetched_url_ids)


def _geopolitical_resource_rows(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    inbound_by_url, links_by_resource, source_url_by_resource, _ = _web_edge_counters(
        edges
    )

    rows: list[dict[str, Any]] = []
    for node in nodes:
        if _node_role(node) != "web:resource":
            continue
        kind = str(_node_value(node, "kind", "") or "").strip().lower()
        canonical_url = _node_canonical_url(node)
        if not canonical_url:
            continue
        if kind.startswith("github:") or _is_github_like_url(canonical_url):
            continue

        resource_id = str(node.get("id", "") or "").strip()
        source_url_id = str(
            _node_value(
                node, "source_url_id", source_url_by_resource.get(resource_id, "")
            )
            or ""
        ).strip()
        fetched_ts = round(
            max(0.0, _safe_float(_node_value(node, "fetched_ts", 0.0), 0.0)),
            6,
        )
        rows.append(
            {
                "id": resource_id,
                "kind": kind,
                "domain": _url_host_token(canonical_url),
                "canonical_url": canonical_url,
                "title": _node_title(node),
                "summary": str(_node_value(node, "summary", "") or "").strip(),
                "text_excerpt": str(
                    _node_value(node, "text_excerpt", "") or ""
                ).strip(),
                "fetched_ts": fetched_ts,
                "importance_score": max(
                    0,
                    _safe_int(_node_value(node, "importance_score", 0), 0),
                ),
                "atoms": [
                    row for row in _node_list(node, "atoms") if isinstance(row, dict)
                ],
                "source_url_id": source_url_id,
                "inbound_links": max(0, inbound_by_url.get(source_url_id, 0)),
                "link_count": max(0, links_by_resource.get(resource_id, 0)),
            }
        )

    rows.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("canonical_url", "")),
        )
    )
    return rows


def _geopolitical_row_score(row: dict[str, Any]) -> tuple[int, list[str]]:
    text_parts = [
        str(row.get("title", "") or ""),
        str(row.get("summary", "") or ""),
        str(row.get("text_excerpt", "") or ""),
        str(row.get("kind", "") or ""),
        str(row.get("canonical_url", "") or ""),
    ]
    merged_text = "\n".join(text_parts).strip().lower()

    score = 0
    signals: set[str] = set()
    for token, weight in _GLOBAL_GEO_KEYWORD_SCORES.items():
        if token in merged_text:
            score += max(0, int(weight))
            signals.add(token)

    for atom in row.get("atoms", []):
        if not isinstance(atom, dict):
            continue
        atom_kind = str(atom.get("kind", "") or "").strip().lower()
        atom_label = str(atom.get("label", "") or "").strip().lower()
        if atom_label in _HORMUZ_THREAT_LABEL_SCORES:
            score += max(
                0, _safe_int(_HORMUZ_THREAT_LABEL_SCORES.get(atom_label, 0), 0)
            )
            signals.add(atom_label)
        elif atom_label:
            signals.add(atom_label)
        if atom_kind == "hazard":
            score += 1

    domain = str(row.get("domain", "") or "").strip().lower()
    domain_weight = 0
    for candidate, weight in _GLOBAL_GEO_DOMAIN_WEIGHTS.items():
        if domain == candidate or domain.endswith(f".{candidate}"):
            domain_weight = max(domain_weight, max(0, int(weight)))
    score += domain_weight

    score += min(3, max(0, _safe_int(row.get("importance_score", 0), 0)) // 2)
    score += min(2, max(0, _safe_int(row.get("inbound_links", 0), 0)) // 2)
    score += min(2, max(0, _safe_int(row.get("link_count", 0), 0)) // 3)
    if str(row.get("kind", "") or "").strip().lower().startswith("maritime:"):
        score += 1
        signals.add("maritime")

    return (max(0, int(score)), sorted(signals))


def _query_geopolitical_news_radar(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    args: dict[str, Any],
    simulation: dict[str, Any] | None,
) -> dict[str, Any]:
    resources = _geopolitical_resource_rows(nodes, edges)
    domain_filter = str(args.get("domain", "") or "").strip().lower()
    kind_filter = str(args.get("kind", "") or "").strip().lower()
    window_ticks = max(
        60,
        min(120_000, _safe_int(args.get("window_ticks", 10_080), 10_080)),
    )
    limit = max(1, min(256, _safe_int(args.get("limit", 48), 48)))
    include_provisional = _truthy_query_flag(
        args.get("include_provisional", "1"),
        default=True,
    )
    include_watchlist_source_evidence = _truthy_query_flag(
        args.get("include_watchlist_source_evidence", "0"),
        default=False,
    )
    configured_sources_raw = _load_world_watchlist_sources()
    configured_feed_source_urls: set[str] = set()
    for source_row in configured_sources_raw:
        if not isinstance(source_row, dict):
            continue
        source_url = str(source_row.get("url", "") or "").strip()
        if not source_url:
            continue
        source_kind = str(source_row.get("kind", "") or "").strip().lower()
        source_type = str(source_row.get("source_type", "") or "").strip().lower()
        if not (
            source_kind.startswith("feed:")
            or source_type in {"rss", "atom", "jsonfeed", "feed"}
        ):
            continue
        normalized_source_url = _normalize_source_url(source_url)
        if normalized_source_url:
            configured_feed_source_urls.add(normalized_source_url)

    latest_fetched_ts = max(
        [_safe_float(row.get("fetched_ts", 0.0), 0.0) for row in resources],
        default=0.0,
    )
    min_fetched_ts = max(0.0, latest_fetched_ts - float(window_ticks))

    rows: list[dict[str, Any]] = []
    domain_stats: dict[str, dict[str, Any]] = {}
    slug_counts: dict[str, int] = {}
    for row in resources:
        domain = str(row.get("domain", "") or "").strip().lower()
        kind = str(row.get("kind", "") or "").strip().lower()
        canonical_url = str(row.get("canonical_url", "") or "").strip()
        normalized_canonical_url = _normalize_source_url(canonical_url)
        if kind.startswith("feed:"):
            continue
        if (
            normalized_canonical_url
            and normalized_canonical_url in configured_feed_source_urls
        ):
            continue
        if domain_filter and domain_filter not in domain:
            continue
        if kind_filter and kind != kind_filter:
            continue

        fetched_ts = max(0.0, _safe_float(row.get("fetched_ts", 0.0), 0.0))
        if min_fetched_ts > 0.0 and fetched_ts > 0.0 and fetched_ts < min_fetched_ts:
            continue

        score, signals = _geopolitical_row_score(row)
        if score <= 1:
            continue

        risk_score = max(1, min(10, score))
        if risk_score >= 8:
            risk_level = "critical"
        elif risk_score >= 6:
            risk_level = "high"
        elif risk_score >= 4:
            risk_level = "medium"
        else:
            risk_level = "low"

        rows.append(
            {
                "kind": kind,
                "domain": domain,
                "title": row.get("title", ""),
                "canonical_url": row.get("canonical_url", ""),
                "fetched_ts": fetched_ts,
                "signals": signals,
                "labels": signals,
                "summary": row.get("summary", ""),
                "text_excerpt": row.get("text_excerpt", ""),
                "deterministic_score": int(score),
                "risk_score": int(risk_score),
                "risk_level": risk_level,
            }
        )

        if domain:
            current = domain_stats.get(domain, {})
            current_count = _safe_int(current.get("count", 0), 0)
            if current_count <= 0:
                domain_stats[domain] = {
                    "domain": domain,
                    "count": 1,
                    "max_risk_score": int(risk_score),
                    "url": row.get("canonical_url", ""),
                    "title": row.get("title", ""),
                    "kind": kind,
                }
            else:
                current["count"] = current_count + 1
                current["max_risk_score"] = max(
                    _safe_int(current.get("max_risk_score", 0), 0),
                    int(risk_score),
                )
                domain_stats[domain] = current

        for token in _url_slug_tokens(str(row.get("canonical_url", "") or "")):
            slug_counts[token] = slug_counts.get(token, 0) + 1

    rows.sort(
        key=lambda row: (
            -_safe_int(row.get("risk_score", 0), 0),
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("canonical_url", "")),
        )
    )

    level_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    kind_risk: dict[str, int] = {}
    signal_counts: dict[str, int] = {}
    for row in rows:
        level = str(row.get("risk_level", "low") or "low").strip().lower()
        if level in level_counts:
            level_counts[level] += 1
        kind = str(row.get("kind", "") or "").strip()
        if kind:
            kind_risk[kind] = max(
                _safe_int(kind_risk.get(kind, 0), 0),
                _safe_int(row.get("risk_score", 0), 0),
            )
        for signal in row.get("signals", []):
            token = str(signal or "").strip().lower()
            if token:
                signal_counts[token] = signal_counts.get(token, 0) + 1

    _, _, _, fetched_url_ids = _web_edge_counters(edges)
    inbound_by_url, _, _, _ = _web_edge_counters(edges)
    frontier_rows: list[dict[str, Any]] = []
    for node in nodes:
        if _node_role(node) != "web:url":
            continue
        url_id = str(node.get("id", "") or "").strip()
        if not url_id or url_id in fetched_url_ids:
            continue
        canonical_url = _node_canonical_url(node)
        if not canonical_url or _is_github_like_url(canonical_url):
            continue
        domain = _url_host_token(canonical_url)
        if domain_filter and domain_filter not in domain:
            continue
        inbound = max(0, _safe_int(inbound_by_url.get(url_id, 0), 0))
        if inbound <= 0:
            continue
        frontier_rows.append(
            {
                "url_id": url_id,
                "canonical_url": canonical_url,
                "domain": domain,
                "inbound_links": inbound,
                "last_status": str(_node_value(node, "last_status", "") or "").strip(),
                "fail_count": max(0, _safe_int(_node_value(node, "fail_count", 0), 0)),
                "next_allowed_fetch_ts": round(
                    max(
                        0.0,
                        _safe_float(
                            _node_value(node, "next_allowed_fetch_ts", 0.0), 0.0
                        ),
                    ),
                    6,
                ),
            }
        )
        for token in _url_slug_tokens(canonical_url):
            slug_counts[token] = slug_counts.get(token, 0) + 1

    frontier_rows.sort(
        key=lambda row: (
            -_safe_int(row.get("inbound_links", 0), 0),
            _safe_int(row.get("fail_count", 0), 0),
            str(row.get("canonical_url", "")),
        )
    )

    hot_domains = sorted(
        [
            {
                "domain": domain,
                "count": _safe_int(row.get("count", 0), 0),
                "max_risk_score": _safe_int(row.get("max_risk_score", 0), 0),
            }
            for domain, row in domain_stats.items()
        ],
        key=lambda row: (
            -_safe_int(row.get("max_risk_score", 0), 0),
            -_safe_int(row.get("count", 0), 0),
            str(row.get("domain", "")),
        ),
    )

    hot_kinds = sorted(
        [{"kind": kind, "max_risk_score": score} for kind, score in kind_risk.items()],
        key=lambda row: (
            -_safe_int(row.get("max_risk_score", 0), 0),
            str(row.get("kind", "")),
        ),
    )

    discovered_sources = sorted(
        [
            {
                "url": str(row.get("url", "") or "").strip(),
                "kind": str(row.get("kind", "") or "").strip(),
                "title": str(row.get("title", "") or "").strip(),
                "source_type": "crawl",
                "domain_id": str(row.get("domain", "") or "").strip(),
                "count": _safe_int(row.get("count", 0), 0),
            }
            for row in domain_stats.values()
            if str(row.get("url", "") or "").strip()
        ],
        key=lambda row: (
            -_safe_int(row.get("count", 0), 0),
            str(row.get("domain_id", "")),
        ),
    )

    configured_sources: list[dict[str, Any]] = []
    for row in configured_sources_raw:
        if not isinstance(row, dict):
            continue
        source_url = str(row.get("url", "") or "").strip()
        if not source_url or _is_github_like_url(source_url):
            continue
        source_kind = str(row.get("kind", "") or "").strip().lower()
        if kind_filter and source_kind != kind_filter:
            continue
        source_domain = _url_host_token(source_url)
        if domain_filter and domain_filter not in source_domain:
            continue
        configured_sources.append(
            {
                "url": source_url,
                "kind": source_kind,
                "title": str(row.get("title", "") or "").strip(),
                "source_type": str(row.get("source_type", "") or "").strip()
                or "watchlist",
                "domain_id": str(row.get("domain_id", "") or "").strip()
                or source_domain,
                "count": 0,
            }
        )

    source_by_url: dict[str, dict[str, Any]] = {}
    for row in configured_sources:
        source_url = str(row.get("url", "") or "").strip()
        if not source_url:
            continue
        source_by_url[source_url] = dict(row)

    for row in discovered_sources:
        source_url = str(row.get("url", "") or "").strip()
        if not source_url:
            continue
        current = source_by_url.get(source_url)
        if not isinstance(current, dict):
            source_by_url[source_url] = dict(row)
            continue
        merged = dict(current)
        merged["count"] = max(
            _safe_int(current.get("count", 0), 0),
            _safe_int(row.get("count", 0), 0),
        )
        if str(row.get("kind", "") or "").strip():
            merged["kind"] = str(row.get("kind", "") or "").strip()
        if str(row.get("title", "") or "").strip():
            merged["title"] = str(row.get("title", "") or "").strip()
        if str(row.get("domain_id", "") or "").strip():
            merged["domain_id"] = str(row.get("domain_id", "") or "").strip()
        merged["source_type"] = (
            str(row.get("source_type", "") or "").strip()
            or str(merged.get("source_type", "") or "watchlist").strip()
        )
        source_by_url[source_url] = merged

    configured_source_by_normalized: dict[str, dict[str, Any]] = {}
    for row in configured_sources:
        source_url = str(row.get("url", "") or "").strip()
        normalized_url = _normalize_source_url(source_url)
        if not normalized_url or normalized_url in configured_source_by_normalized:
            continue
        configured_source_by_normalized[normalized_url] = row

    sources = sorted(
        list(source_by_url.values()),
        key=lambda row: (
            -_safe_int(row.get("count", 0), 0),
            str(row.get("domain_id", "")),
            str(row.get("url", "")),
        ),
    )

    if not frontier_rows and not rows and configured_sources:
        for row in configured_sources:
            source_url = str(row.get("url", "") or "").strip()
            if not source_url:
                continue
            frontier_rows.append(
                {
                    "url_id": f"watch:{hashlib.sha1(source_url.encode('utf-8')).hexdigest()[:12]}",
                    "canonical_url": source_url,
                    "domain": _url_host_token(source_url),
                    "inbound_links": 0,
                    "last_status": "watchlist",
                    "fail_count": 0,
                    "next_allowed_fetch_ts": 0.0,
                }
            )
        frontier_rows.sort(
            key=lambda row: (
                str(row.get("domain", "")),
                str(row.get("canonical_url", "")),
            )
        )

    if (
        include_watchlist_source_evidence
        and not rows
        and configured_source_by_normalized
    ):
        weaver_source_rows = _load_weaver_fetched_source_rows()
        evidence_rows: list[dict[str, Any]] = []
        for weaver_row in weaver_source_rows:
            if not isinstance(weaver_row, dict):
                continue
            normalized_url = str(weaver_row.get("normalized_url", "") or "").strip()
            if not normalized_url:
                continue
            source_row = configured_source_by_normalized.get(normalized_url)
            if not isinstance(source_row, dict):
                continue

            source_url = str(source_row.get("url", "") or "").strip()
            source_kind = str(source_row.get("kind", "") or "").strip().lower()
            source_type = str(source_row.get("source_type", "") or "").strip().lower()
            source_domain = _url_host_token(source_url)
            source_title = str(source_row.get("title", "") or "").strip()
            fetched_ts = _normalize_epoch_seconds(weaver_row.get("fetched_ts", 0.0))
            if fetched_ts <= 0.0:
                continue

            summary_text = str(weaver_row.get("summary", "") or "").strip()
            text_excerpt = str(weaver_row.get("text_excerpt", "") or "").strip()
            title_text = (
                source_title
                or str(weaver_row.get("title", "") or "").strip()
                or source_url
            )
            source_type_token = re.sub(r"[^a-z0-9]+", "_", source_type).strip("_")
            signals = ["crawl_evidence", "watchlist_source"]
            if source_type_token:
                signals.append(f"{source_type_token}_fetched")
            if source_kind.startswith("feed:"):
                signals.append("feed_source")
            signals = sorted({token for token in signals if token})

            evidence_rows.append(
                {
                    "kind": source_kind or "global:watchlist_source",
                    "domain": source_domain,
                    "title": title_text,
                    "canonical_url": source_url,
                    "fetched_ts": fetched_ts,
                    "signals": signals,
                    "labels": list(signals),
                    "summary": summary_text
                    or "Watchlist source fetched; no elevated threat signals yet.",
                    "text_excerpt": text_excerpt,
                    "deterministic_score": 2,
                    "risk_score": 2,
                    "risk_level": "low",
                    "source_type": source_type,
                    "provisional": False,
                }
            )

            existing_source = source_by_url.get(source_url)
            if isinstance(existing_source, dict):
                source_next = dict(existing_source)
                source_next["count"] = max(_safe_int(source_next.get("count", 0), 0), 1)
                source_by_url[source_url] = source_next

        deduped_evidence: dict[str, dict[str, Any]] = {}
        for row in evidence_rows:
            canonical_url = str(row.get("canonical_url", "") or "").strip()
            if not canonical_url:
                continue
            current = deduped_evidence.get(canonical_url)
            if not isinstance(current, dict):
                deduped_evidence[canonical_url] = row
                continue
            current_fetched = _safe_float(current.get("fetched_ts", 0.0), 0.0)
            next_fetched = _safe_float(row.get("fetched_ts", 0.0), 0.0)
            if next_fetched > current_fetched:
                deduped_evidence[canonical_url] = row

        evidence_sorted = sorted(
            deduped_evidence.values(),
            key=lambda row: (
                -_safe_float(row.get("fetched_ts", 0.0), 0.0),
                str(row.get("canonical_url", "")),
            ),
        )
        if evidence_sorted:
            rows = evidence_sorted[:limit]
            level_counts = {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": len(rows),
            }
            signal_counts = {}
            kind_risk = {}

            for row in rows:
                domain = str(row.get("domain", "") or "").strip()
                kind = str(row.get("kind", "") or "").strip()
                risk_score = _safe_int(row.get("risk_score", 0), 0)
                canonical_url = str(row.get("canonical_url", "") or "").strip()
                if domain:
                    current = domain_stats.get(domain, {})
                    current_count = _safe_int(current.get("count", 0), 0)
                    if current_count <= 0:
                        domain_stats[domain] = {
                            "domain": domain,
                            "count": 1,
                            "max_risk_score": risk_score,
                            "url": canonical_url,
                            "title": str(row.get("title", "") or "").strip(),
                            "kind": kind,
                        }
                    else:
                        current["count"] = current_count + 1
                        current["max_risk_score"] = max(
                            _safe_int(current.get("max_risk_score", 0), 0),
                            risk_score,
                        )
                        domain_stats[domain] = current
                if kind:
                    kind_risk[kind] = max(
                        _safe_int(kind_risk.get(kind, 0), 0), risk_score
                    )
                for signal in row.get("signals", []):
                    token = str(signal or "").strip().lower()
                    if token:
                        signal_counts[token] = signal_counts.get(token, 0) + 1
                for token in _url_slug_tokens(canonical_url):
                    slug_counts[token] = slug_counts.get(token, 0) + 1

            hot_domains = sorted(
                [
                    {
                        "domain": domain,
                        "count": _safe_int(row.get("count", 0), 0),
                        "max_risk_score": _safe_int(row.get("max_risk_score", 0), 0),
                    }
                    for domain, row in domain_stats.items()
                ],
                key=lambda row: (
                    -_safe_int(row.get("max_risk_score", 0), 0),
                    -_safe_int(row.get("count", 0), 0),
                    str(row.get("domain", "")),
                ),
            )
            hot_kinds = sorted(
                [
                    {"kind": kind, "max_risk_score": score}
                    for kind, score in kind_risk.items()
                ],
                key=lambda row: (
                    -_safe_int(row.get("max_risk_score", 0), 0),
                    str(row.get("kind", "")),
                ),
            )
            sources = sorted(
                list(source_by_url.values()),
                key=lambda row: (
                    -_safe_int(row.get("count", 0), 0),
                    str(row.get("domain_id", "")),
                    str(row.get("url", "")),
                ),
            )

    if include_provisional and not rows and configured_sources:
        fallback_rows: list[dict[str, Any]] = []
        fallback_limit = max(1, min(limit, len(configured_sources)))
        for source_row in configured_sources[:fallback_limit]:
            source_url = str(source_row.get("url", "") or "").strip()
            if not source_url:
                continue
            source_kind = str(source_row.get("kind", "") or "").strip().lower()
            source_type = str(source_row.get("source_type", "") or "").strip().lower()
            source_domain = _url_host_token(source_url)
            source_title = str(source_row.get("title", "") or "").strip()
            source_type_token = re.sub(r"[^a-z0-9]+", "_", source_type).strip("_")

            fallback_signals = ["watchlist_seed", "pending_fetch"]
            if source_type_token:
                fallback_signals.append(f"{source_type_token}_seed")

            fallback_rows.append(
                {
                    "kind": source_kind or "global:watchlist_source",
                    "domain": source_domain,
                    "title": source_title or source_url,
                    "canonical_url": source_url,
                    "fetched_ts": 0.0,
                    "signals": sorted(set(fallback_signals)),
                    "labels": sorted(set(fallback_signals)),
                    "summary": "Configured watchlist source awaiting fresh crawl evidence.",
                    "text_excerpt": "",
                    "deterministic_score": 2,
                    "risk_score": 2,
                    "risk_level": "low",
                    "source_type": source_type,
                    "provisional": True,
                }
            )

            if source_domain:
                domain_current = domain_stats.get(source_domain, {})
                domain_count = _safe_int(domain_current.get("count", 0), 0)
                if domain_count <= 0:
                    domain_stats[source_domain] = {
                        "domain": source_domain,
                        "count": 1,
                        "max_risk_score": 2,
                        "url": source_url,
                        "title": source_title,
                        "kind": source_kind,
                    }
                else:
                    domain_current["count"] = domain_count + 1
                    domain_current["max_risk_score"] = max(
                        _safe_int(domain_current.get("max_risk_score", 0), 0),
                        2,
                    )
                    domain_stats[source_domain] = domain_current

            if source_kind:
                kind_risk[source_kind] = max(
                    _safe_int(kind_risk.get(source_kind, 0), 0), 2
                )

            for token in fallback_signals:
                signal_counts[token] = signal_counts.get(token, 0) + 1

            for token in _url_slug_tokens(source_url):
                slug_counts[token] = slug_counts.get(token, 0) + 1

        rows = fallback_rows
        level_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": len(rows),
        }

        hot_domains = sorted(
            [
                {
                    "domain": domain,
                    "count": _safe_int(row.get("count", 0), 0),
                    "max_risk_score": _safe_int(row.get("max_risk_score", 0), 0),
                }
                for domain, row in domain_stats.items()
            ],
            key=lambda row: (
                -_safe_int(row.get("max_risk_score", 0), 0),
                -_safe_int(row.get("count", 0), 0),
                str(row.get("domain", "")),
            ),
        )

        hot_kinds = sorted(
            [
                {"kind": kind, "max_risk_score": score}
                for kind, score in kind_risk.items()
            ],
            key=lambda row: (
                -_safe_int(row.get("max_risk_score", 0), 0),
                str(row.get("kind", "")),
            ),
        )

    signal_families = sorted(
        [{"signal": token, "count": count} for token, count in signal_counts.items()],
        key=lambda row: (
            -_safe_int(row.get("count", 0), 0),
            str(row.get("signal", "")),
        ),
    )

    slug_rows = sorted(
        [{"slug": token, "count": count} for token, count in slug_counts.items()],
        key=lambda row: (
            -_safe_int(row.get("count", 0), 0),
            str(row.get("slug", "")),
        ),
    )

    crawler_graph = (
        simulation.get("crawler_graph", {}) if isinstance(simulation, dict) else {}
    )
    crawler_graph = crawler_graph if isinstance(crawler_graph, dict) else {}
    crawler_status = (
        crawler_graph.get("status", {})
        if isinstance(crawler_graph.get("status", {}), dict)
        else {}
    )
    crawler_events = (
        crawler_graph.get("events", [])
        if isinstance(crawler_graph.get("events", []), list)
        else []
    )
    traversal_events = [
        {
            "id": str(row.get("id", "") or "").strip(),
            "kind": str(row.get("kind", row.get("event", "")) or "").strip(),
            "ts": row.get("ts", row.get("tick", "")),
            "url_id": str(row.get("url_id", "") or "").strip(),
            "status": str(row.get("status", "") or "").strip(),
            "reason": str(row.get("reason", "") or "").strip(),
        }
        for row in crawler_events[-24:]
        if isinstance(row, dict)
    ]

    provisional_count = sum(
        1
        for row in rows
        if isinstance(row, dict) and bool(row.get("provisional", False))
    )
    non_provisional_count = max(0, len(rows) - provisional_count)
    seed_only = len(rows) > 0 and provisional_count == len(rows)

    return {
        "domain": domain_filter,
        "kind": kind_filter,
        "window_ticks": int(window_ticks),
        "count": len(rows),
        "critical_count": int(level_counts["critical"]),
        "high_count": int(level_counts["high"]),
        "medium_count": int(level_counts["medium"]),
        "low_count": int(level_counts["low"]),
        "provisional_count": int(provisional_count),
        "non_provisional_count": int(non_provisional_count),
        "hot_domains": hot_domains[:12],
        "hot_kinds": hot_kinds[:12],
        "signal_families": signal_families[:16],
        "scoring": {
            "mode": "deterministic_discovery",
            "llm_enabled": False,
            "llm_applied": False,
            "llm_model": "",
            "llm_error": "",
        },
        "source_count": len(sources),
        "sources": sources[:96],
        "discovery": {
            "domain_count": len(domain_stats),
            "candidate_url_count": len(frontier_rows),
            "frontier_candidates": frontier_rows[:64],
            "slug_count": len(slug_rows),
            "slugs": slug_rows[:32],
        },
        "traversal": {
            "queue_length": max(0, _safe_int(crawler_status.get("queue_length", 0), 0)),
            "active_fetches": max(
                0,
                _safe_int(crawler_status.get("active_fetches", 0), 0),
            ),
            "event_count": len(crawler_events),
            "recent_events": traversal_events,
        },
        "quality": {
            "seed_only": bool(seed_only),
            "needs_crawl_evidence": bool(seed_only or len(rows) == 0),
        },
        "threats": rows[:limit],
    }


def _hormuz_resource_rows(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in nodes:
        if _node_role(node) != "web:resource":
            continue
        kind = str(_node_value(node, "kind", "") or "").strip().lower()
        if not kind.startswith("maritime:"):
            continue

        rows.append(
            {
                "id": str(node.get("id", "") or "").strip(),
                "kind": kind,
                "canonical_url": _node_canonical_url(node),
                "title": _node_title(node),
                "summary": str(_node_value(node, "summary", "") or "").strip(),
                "text_excerpt": str(
                    _node_value(node, "text_excerpt", "") or ""
                ).strip(),
                "fetched_ts": round(
                    max(0.0, _safe_float(_node_value(node, "fetched_ts", 0.0), 0.0)),
                    6,
                ),
                "content_hash": str(
                    _node_value(node, "content_hash", "") or ""
                ).strip(),
                "importance_score": max(
                    0,
                    _safe_int(_node_value(node, "importance_score", 0), 0),
                ),
                "atoms": [
                    row for row in _node_list(node, "atoms") if isinstance(row, dict)
                ],
            }
        )

    rows.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("id", "")),
        )
    )
    return rows


def _hormuz_threat_score(row: dict[str, Any]) -> tuple[int, list[str], list[str]]:
    labels: set[str] = set()
    for atom in row.get("atoms", []):
        if not isinstance(atom, dict):
            continue
        atom_kind = str(atom.get("kind", "") or "").strip().lower()
        if atom_kind and atom_kind not in {"hazard", "status", "impact"}:
            continue
        atom_region = str(atom.get("region", "") or "").strip().lower()
        if atom_region and "hormuz" not in atom_region:
            continue
        label = str(atom.get("label", "") or "").strip().lower()
        if label in _HORMUZ_THREAT_LABEL_SCORES:
            labels.add(label)

    score = 0
    for label in labels:
        score += _safe_int(_HORMUZ_THREAT_LABEL_SCORES.get(label, 0), 0)

    signals = sorted(labels)
    return int(score), signals, sorted(labels)


def _query_hormuz_threat_radar(
    nodes: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    resources = _hormuz_resource_rows(nodes)
    kind_filter = str(args.get("kind", "") or "").strip().lower()
    window_ticks = max(1, min(20_000, _safe_int(args.get("window_ticks", 1440), 1440)))
    limit = max(1, min(128, _safe_int(args.get("limit", 24), 24)))
    watch_sources = _hormuz_watchlist_sources(kind_filter)

    latest_fetched_ts = max(
        [_safe_float(row.get("fetched_ts", 0.0), 0.0) for row in resources],
        default=0.0,
    )
    min_fetched_ts = max(0.0, latest_fetched_ts - float(window_ticks))

    rows: list[dict[str, Any]] = []
    for row in resources:
        row_kind = str(row.get("kind", "") or "").strip().lower()
        if kind_filter and row_kind != kind_filter:
            continue

        fetched_ts = max(0.0, _safe_float(row.get("fetched_ts", 0.0), 0.0))
        if min_fetched_ts > 0.0 and fetched_ts > 0.0 and fetched_ts < min_fetched_ts:
            continue

        deterministic_score, signals, labels = _hormuz_threat_score(row)
        if deterministic_score <= 0:
            continue

        threat_identity = _threat_row_identity(
            "hormuz",
            {
                "kind": row_kind,
                "canonical_url": row.get("canonical_url", ""),
                "title": row.get("title", ""),
            },
        )

        rows.append(
            {
                "_threat_id": threat_identity,
                "kind": row_kind,
                "title": row.get("title", ""),
                "canonical_url": row.get("canonical_url", ""),
                "fetched_ts": fetched_ts,
                "signals": signals,
                "labels": labels,
                "summary": row.get("summary", ""),
                "text_excerpt": row.get("text_excerpt", ""),
                "deterministic_score": int(deterministic_score),
            }
        )

    llm = _threat_llm_metrics(domain="hormuz", rows=rows)
    llm_metrics = (
        llm.get("metrics", {}) if isinstance(llm.get("metrics", {}), dict) else {}
    )
    llm_model = str(
        llm.get("model", _THREAT_RADAR_LLM_MODEL) or _THREAT_RADAR_LLM_MODEL
    )
    for row in rows:
        threat_id = str(row.get("_threat_id", "") or "")
        deterministic_score = max(0, _safe_int(row.get("deterministic_score", 0), 0))
        metrics = llm_metrics.get(threat_id, {}) if threat_id else {}
        if isinstance(metrics, dict) and metrics:
            final_score = _blend_llm_score(
                deterministic_score,
                metrics.get("overall_score", 0),
                max_score=10,
            )
            row["llm_score"] = int(
                round(
                    (
                        max(
                            0.0,
                            min(
                                100.0, _safe_float(metrics.get("overall_score", 0), 0.0)
                            ),
                        )
                        / 100.0
                    )
                    * 10.0
                )
            )
            row["threat_metrics"] = dict(metrics)
            row["llm_model"] = llm_model
        else:
            final_score = deterministic_score
        row["risk_score"] = int(final_score)
        if final_score >= 7:
            row["risk_level"] = "critical"
        elif final_score >= 5:
            row["risk_level"] = "high"
        elif final_score >= 3:
            row["risk_level"] = "medium"
        else:
            row["risk_level"] = "low"
        row.pop("_threat_id", None)

    rows.sort(
        key=lambda row: (
            -_safe_int(row.get("risk_score", 0), 0),
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("canonical_url", "")),
        )
    )

    level_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    kind_risk: dict[str, int] = {}
    for row in rows:
        level = str(row.get("risk_level", "low") or "low").strip().lower()
        if level in level_counts:
            level_counts[level] += 1
        kind = str(row.get("kind", "") or "").strip()
        if not kind:
            continue
        score = _safe_int(row.get("risk_score", 0), 0)
        previous = _safe_int(kind_risk.get(kind, 0), 0)
        if score > previous:
            kind_risk[kind] = score

    hot_kinds = sorted(
        kind_risk.items(),
        key=lambda item: (-_safe_int(item[1], 0), str(item[0])),
    )

    return {
        "kind": kind_filter,
        "window_ticks": int(window_ticks),
        "count": len(rows),
        "critical_count": int(level_counts["critical"]),
        "high_count": int(level_counts["high"]),
        "medium_count": int(level_counts["medium"]),
        "low_count": int(level_counts["low"]),
        "hot_kinds": [
            {"kind": kind, "max_risk_score": score} for kind, score in hot_kinds[:8]
        ],
        "scoring": {
            "mode": (
                "llm_blend"
                if bool(llm.get("enabled", False)) and bool(llm.get("applied", False))
                else "deterministic"
            ),
            "llm_enabled": bool(llm.get("enabled", False)),
            "llm_applied": bool(llm.get("applied", False)),
            "llm_model": llm_model,
            "llm_error": str(llm.get("error", "") or ""),
        },
        "source_count": len(watch_sources),
        "sources": watch_sources[:64],
        "threats": rows[:limit],
    }


def _query_github_status(
    nodes: list[dict[str, Any]],
    simulation: dict[str, Any] | None,
) -> dict[str, Any]:
    resources = _github_resource_rows(nodes)
    repo_set = {
        str(row.get("repo", "")).strip()
        for row in resources
        if str(row.get("repo", "")).strip()
    }

    now_epoch = time.time()
    github_url_nodes = []
    for node in nodes:
        if _node_role(node) != "web:url":
            continue
        canonical_url = _node_canonical_url(node).lower()
        source_hint = str(_node_value(node, "source_hint", "") or "").strip().lower()
        if source_hint == "github" or _is_github_like_url(canonical_url):
            github_url_nodes.append(node)

    cooldown_active = 0
    for node in github_url_nodes:
        next_allowed = max(
            0.0,
            _safe_float(_node_value(node, "next_allowed_fetch_ts", 0.0), 0.0),
        )
        if next_allowed > now_epoch:
            cooldown_active += 1

    crawler_graph = (
        simulation.get("crawler_graph", {}) if isinstance(simulation, dict) else {}
    )
    status = crawler_graph.get("status", {}) if isinstance(crawler_graph, dict) else {}
    github_status = status.get("github", {}) if isinstance(status, dict) else {}
    github_status = github_status if isinstance(github_status, dict) else {}

    events = crawler_graph.get("events", []) if isinstance(crawler_graph, dict) else []
    github_events = [
        row
        for row in events
        if isinstance(row, dict)
        and str(row.get("event", row.get("kind", "")))
        .strip()
        .lower()
        .startswith("github_")
    ]

    monitored_repos = [
        str(item).strip()
        for item in github_status.get("monitored_repos", [])
        if str(item).strip()
    ]
    if not monitored_repos:
        monitored_repos = sorted(repo_set)

    return {
        "monitored_repos": monitored_repos,
        "queue_length": max(0, _safe_int(github_status.get("queue_length", 0), 0)),
        "active_fetches": max(0, _safe_int(github_status.get("active_fetches", 0), 0)),
        "cooldown_blocks": max(
            0,
            _safe_int(
                github_status.get(
                    "cooldown_blocks",
                    github_status.get("cooldown_active", cooldown_active),
                ),
                cooldown_active,
            ),
        ),
        "url_node_count": len(github_url_nodes),
        "resource_node_count": len(resources),
        "event_count": len(github_events),
        "last_fetches": [
            {
                "repo": row.get("repo", ""),
                "kind": row.get("kind", ""),
                "canonical_url": row.get("canonical_url", ""),
                "title": row.get("title", ""),
                "fetched_ts": row.get("fetched_ts", 0.0),
                "content_hash": row.get("content_hash", ""),
            }
            for row in resources[:8]
        ],
    }


def _query_github_repo_summary(
    nodes: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    resources = _github_resource_rows(nodes)
    repo = str(args.get("repo", "") or "").strip()
    if not repo and resources:
        repo = str(resources[0].get("repo", "") or "").strip()

    filtered = [
        row
        for row in resources
        if not repo or str(row.get("repo", "")).strip().lower() == repo.lower()
    ]

    ranked = list(filtered)
    ranked.sort(
        key=lambda row: (
            -_safe_int(row.get("importance_score", 0), 0),
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("canonical_url", "")),
        )
    )

    return {
        "repo": repo,
        "count": len(filtered),
        "last_updated": str(filtered[0].get("updated_at", "")) if filtered else "",
        "recent_resources": [
            {
                "id": row.get("id", ""),
                "kind": row.get("kind", ""),
                "canonical_url": row.get("canonical_url", ""),
                "title": row.get("title", ""),
                "number": row.get("number", 0),
                "updated_at": row.get("updated_at", ""),
                "fetched_ts": row.get("fetched_ts", 0.0),
                "importance_score": row.get("importance_score", 0),
            }
            for row in filtered[:24]
        ],
        "top_items": [
            {
                "id": row.get("id", ""),
                "kind": row.get("kind", ""),
                "canonical_url": row.get("canonical_url", ""),
                "title": row.get("title", ""),
                "importance_score": row.get("importance_score", 0),
                "content_hash": row.get("content_hash", ""),
            }
            for row in ranked[:16]
        ],
    }


def _query_github_find(
    nodes: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    term = (
        str(args.get("term", args.get("q", args.get("query", ""))) or "")
        .strip()
        .lower()
    )
    repo = str(args.get("repo", "") or "").strip().lower()
    limit = max(1, min(128, _safe_int(args.get("limit", 24), 24)))

    if not term:
        return {"term": term, "repo": repo, "count": 0, "matches": []}

    matches: list[dict[str, Any]] = []
    for row in _github_resource_rows(nodes):
        row_repo = str(row.get("repo", "")).strip().lower()
        if repo and row_repo != repo:
            continue

        title = str(row.get("title", "") or "")
        canonical_url = str(row.get("canonical_url", "") or "")
        haystack = " ".join(
            [
                title.lower(),
                canonical_url.lower(),
                str(row.get("kind", "")).lower(),
            ]
        )

        row_matched = term in haystack
        if row_matched:
            matches.append(
                {
                    "repo": row.get("repo", ""),
                    "kind": row.get("kind", ""),
                    "canonical_url": canonical_url,
                    "title": title,
                    "res_id": row.get("id", ""),
                    "atom": None,
                    "fetched_ts": row.get("fetched_ts", 0.0),
                }
            )

        for atom in row.get("atoms", []):
            if not isinstance(atom, dict):
                continue
            atom_blob = json.dumps(atom, ensure_ascii=False, sort_keys=True).lower()
            if term not in atom_blob:
                continue
            matches.append(
                {
                    "repo": row.get("repo", ""),
                    "kind": row.get("kind", ""),
                    "canonical_url": canonical_url,
                    "title": title,
                    "res_id": row.get("id", ""),
                    "atom": atom,
                    "fetched_ts": row.get("fetched_ts", 0.0),
                }
            )

    matches.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("repo", "")),
            str(row.get("canonical_url", "")),
            json.dumps(row.get("atom") or {}, ensure_ascii=False, sort_keys=True),
        )
    )

    return {
        "term": term,
        "repo": repo,
        "count": len(matches),
        "matches": matches[:limit],
    }


def _query_github_recent_changes(
    nodes: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    window_ticks = max(1, min(20_000, _safe_int(args.get("window_ticks", 360), 360)))
    limit = max(1, min(128, _safe_int(args.get("limit", 32), 32)))

    changes: list[dict[str, Any]] = []
    for row in _github_resource_rows(nodes):
        atoms = row.get("atoms", []) if isinstance(row.get("atoms", []), list) else []
        atom_kinds = {
            str(atom.get("kind", "")).strip().lower()
            for atom in atoms
            if isinstance(atom, dict)
        }
        change_kind = ""
        if row.get("kind") == "github:pr" and str(row.get("merged_at", "")).strip():
            change_kind = "pr_merged"
        elif "changes_dependency" in atom_kinds:
            change_kind = "dependency_change"
        elif row.get("kind") == "github:advisory" or "references_cve" in atom_kinds:
            change_kind = "security_signal"
        elif row.get("kind") in {"github:release", "github:issue"}:
            change_kind = str(row.get("kind", "")).replace("github:", "")

        if not change_kind:
            continue

        changes.append(
            {
                "change_kind": change_kind,
                "repo": row.get("repo", ""),
                "kind": row.get("kind", ""),
                "number": row.get("number", 0),
                "canonical_url": row.get("canonical_url", ""),
                "title": row.get("title", ""),
                "fetched_ts": row.get("fetched_ts", 0.0),
                "content_hash": row.get("content_hash", ""),
            }
        )

    changes.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("canonical_url", "")),
        )
    )
    return {
        "window_ticks": int(window_ticks),
        "count": len(changes),
        "changes": changes[:limit],
    }


def _proximity_seed_terms(args: dict[str, Any]) -> list[str]:
    raw = str(args.get("seed_set", "") or "").strip().lower()
    if not raw or raw in {"default", "security", "github_security"}:
        return sorted(_GITHUB_THREAT_TERMS)

    tokens: list[str] = []
    seen: set[str] = set()
    for chunk in re.split(r"[,\s]+", raw):
        token = str(chunk or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens or sorted(_GITHUB_THREAT_TERMS)


def _tokenize_proximity_terms(text: str) -> list[str]:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return []

    terms: list[str] = []
    seen: set[str] = set()
    for token in _PROXIMITY_TERM_RE.findall(lowered):
        if token in _PROXIMITY_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _normalize_distribution(values: list[float]) -> list[float]:
    if not values:
        return []
    cleaned = [max(0.0, _safe_float(value, 0.0)) for value in values]
    total = sum(cleaned)
    if total <= 0.0:
        uniform = 1.0 / float(len(cleaned))
        return [uniform for _ in cleaned]
    return [value / total for value in cleaned]


def _gaussian_likelihood(score: float, mean: float, stddev: float) -> float:
    sigma = max(1e-6, _safe_float(stddev, 0.2))
    delta = _safe_float(score, 0.0) - _safe_float(mean, 0.0)
    exponent = -((delta * delta) / (2.0 * sigma * sigma))
    return max(1e-9, math.exp(exponent))


def _proximity_bin_index(
    fetched_ts: float,
    *,
    min_fetched_ts: float,
    window_ticks: int,
    state_bins: int,
) -> int:
    bins = max(1, _safe_int(state_bins, 6))
    window = max(1.0, _safe_float(window_ticks, 1440.0))
    offset = max(0.0, _safe_float(fetched_ts, 0.0) - max(0.0, min_fetched_ts))
    ratio = max(0.0, min(0.999999, offset / window))
    return max(0, min(bins - 1, int(ratio * float(bins))))


def _proximity_state_posterior(
    bin_rows: list[dict[str, Any]],
    *,
    min_fetched_ts: float,
    bin_width: float,
) -> dict[str, Any]:
    states = list(_PROXIMITY_HMM_STATES)
    n_states = len(states)
    if not bin_rows or n_states <= 0:
        return {
            "state": "background",
            "p_background": 1.0,
            "p_emerging": 0.0,
            "p_active": 0.0,
            "p_critical": 0.0,
            "last_transition_bin": 0,
            "last_transition_ts": round(max(0.0, _safe_float(min_fetched_ts, 0.0)), 6),
            "state_path": ["background"],
        }

    observations: list[float] = []
    prev_count = 0
    for row in bin_rows:
        count = max(0, _safe_int(row.get("count", 0), 0))
        source_count = max(0, _safe_int(row.get("source_count", 0), 0))
        seed_hits = max(0, _safe_int(row.get("seed_hits", 0), 0))
        cve_hits = max(0, _safe_int(row.get("cve_hits", 0), 0))

        count_norm = max(0.0, min(1.0, float(count) / 3.0))
        source_norm = max(0.0, min(1.0, float(source_count) / 3.0))
        seed_norm = max(0.0, min(1.0, float(seed_hits) / float(max(1, count))))
        cve_norm = max(0.0, min(1.0, float(cve_hits) / float(max(1, count))))
        burst_norm = max(
            0.0,
            min(
                1.0,
                float(max(0, count - prev_count)) / float(max(1, prev_count + 1)),
            ),
        )
        obs_score = (
            (count_norm * 0.30)
            + (seed_norm * 0.28)
            + (source_norm * 0.20)
            + (burst_norm * 0.14)
            + (cve_norm * 0.08)
        )
        observations.append(max(0.0, min(1.0, obs_score)))
        prev_count = count

    prior = list(_PROXIMITY_HMM_PRIOR)[:n_states]
    if len(prior) != n_states:
        prior = [1.0 / float(n_states) for _ in range(n_states)]
    alpha_prev = _normalize_distribution(prior)
    viterbi_prev = [math.log(max(1e-9, value)) for value in alpha_prev]
    backpointers: list[list[int]] = []

    for obs in observations:
        emissions = [
            _gaussian_likelihood(obs, _PROXIMITY_HMM_MEANS[i], _PROXIMITY_HMM_STDDEV)
            for i in range(n_states)
        ]
        alpha_next: list[float] = []
        viterbi_next: list[float] = []
        back_row: list[int] = []

        for state_idx in range(n_states):
            trans_total = 0.0
            best_prev_idx = 0
            best_prev_score = -1e18

            for prev_idx in range(n_states):
                trans_prob = max(
                    1e-9,
                    _safe_float(
                        _PROXIMITY_HMM_TRANSITIONS[prev_idx][state_idx],
                        0.0,
                    ),
                )
                trans_total += alpha_prev[prev_idx] * trans_prob
                candidate = viterbi_prev[prev_idx] + math.log(trans_prob)
                if candidate > best_prev_score:
                    best_prev_score = candidate
                    best_prev_idx = prev_idx

            emission_prob = max(1e-9, emissions[state_idx])
            alpha_next.append(emission_prob * trans_total)
            viterbi_next.append(best_prev_score + math.log(emission_prob))
            back_row.append(best_prev_idx)

        alpha_prev = _normalize_distribution(alpha_next)
        viterbi_prev = viterbi_next
        backpointers.append(back_row)

    final_state_idx = max(range(n_states), key=lambda idx: viterbi_prev[idx])
    path_idx = [0 for _ in observations]
    if observations:
        path_idx[-1] = final_state_idx
        for step in range(len(observations) - 1, 0, -1):
            prev_state_idx = backpointers[step][path_idx[step]]
            path_idx[step - 1] = prev_state_idx

    last_transition_bin = 0
    for idx in range(1, len(path_idx)):
        if path_idx[idx] != path_idx[idx - 1]:
            last_transition_bin = idx

    transition_ts = max(
        0.0,
        _safe_float(min_fetched_ts, 0.0)
        + (float(last_transition_bin + 1) * max(1e-6, _safe_float(bin_width, 1.0))),
    )

    posterior = alpha_prev if alpha_prev else [1.0, 0.0, 0.0, 0.0]
    while len(posterior) < 4:
        posterior.append(0.0)
    posterior = _normalize_distribution(posterior[:4])

    return {
        "state": states[max(range(n_states), key=lambda idx: posterior[idx])],
        "p_background": round(posterior[0], 6),
        "p_emerging": round(posterior[1], 6),
        "p_active": round(posterior[2], 6),
        "p_critical": round(posterior[3], 6),
        "last_transition_bin": int(last_transition_bin),
        "last_transition_ts": round(transition_ts, 6),
        "state_path": [states[idx] for idx in path_idx] if path_idx else [states[0]],
    }


def _cyber_regime_state_posterior(
    bin_rows: list[dict[str, Any]],
    *,
    min_fetched_ts: float,
    bin_width: float,
) -> dict[str, Any]:
    states = list(_CYBER_REGIME_STATES)
    n_states = len(states)
    if not bin_rows or n_states <= 0:
        return {
            "state": "baseline",
            "posterior": {
                state: (1.0 if idx == 0 else 0.0) for idx, state in enumerate(states)
            },
            "p_baseline": 1.0,
            "p_elevated_chatter": 0.0,
            "p_active_exploitation_wave": 0.0,
            "p_supply_chain_campaign": 0.0,
            "p_geopolitical_targeting_shift": 0.0,
            "last_transition_bin": 0,
            "last_transition_ts": round(max(0.0, _safe_float(min_fetched_ts, 0.0)), 6),
            "state_path": ["baseline"],
            "observation_mean": 0.0,
            "observation_peak": 0.0,
        }

    observations: list[float] = []
    prev_count = 0
    for row in bin_rows:
        count = max(0, _safe_int(row.get("count", 0), 0))
        critical_count = max(0, _safe_int(row.get("critical_count", 0), 0))
        high_count = max(0, _safe_int(row.get("high_count", 0), 0))
        avg_risk = max(0.0, min(14.0, _safe_float(row.get("avg_risk", 0.0), 0.0)))
        corroborated_count = max(0, _safe_int(row.get("corroborated_count", 0), 0))
        source_tier_high_count = max(
            0,
            _safe_int(row.get("source_tier_high_count", 0), 0),
        )
        proximity_critical_count = max(
            0,
            _safe_int(row.get("proximity_critical_count", 0), 0),
        )

        count_norm = max(0.0, min(1.0, float(count) / 6.0))
        critical_norm = max(0.0, min(1.0, float(critical_count) / 3.0))
        high_norm = max(0.0, min(1.0, float(high_count) / 4.0))
        risk_norm = max(0.0, min(1.0, avg_risk / 14.0))
        corroboration_norm = max(
            0.0,
            min(1.0, float(corroborated_count) / float(max(1, count))),
        )
        source_norm = max(
            0.0,
            min(1.0, float(source_tier_high_count) / float(max(1, count))),
        )
        proximity_norm = max(
            0.0,
            min(1.0, float(proximity_critical_count) / float(max(1, count))),
        )
        burst_norm = max(
            0.0,
            min(
                1.0,
                float(max(0, count - prev_count)) / float(max(1, prev_count + 1)),
            ),
        )
        obs_score = (
            (count_norm * 0.16)
            + (critical_norm * 0.24)
            + (high_norm * 0.14)
            + (risk_norm * 0.18)
            + (corroboration_norm * 0.12)
            + (source_norm * 0.08)
            + (proximity_norm * 0.04)
            + (burst_norm * 0.04)
        )
        observations.append(max(0.0, min(1.0, obs_score)))
        prev_count = count

    prior = list(_CYBER_REGIME_PRIOR)[:n_states]
    if len(prior) != n_states:
        prior = [1.0 / float(n_states) for _ in range(n_states)]
    alpha_prev = _normalize_distribution(prior)
    viterbi_prev = [math.log(max(1e-9, value)) for value in alpha_prev]
    backpointers: list[list[int]] = []

    for obs in observations:
        emissions = [
            _gaussian_likelihood(obs, _CYBER_REGIME_MEANS[i], _CYBER_REGIME_STDDEV)
            for i in range(n_states)
        ]
        alpha_next: list[float] = []
        viterbi_next: list[float] = []
        back_row: list[int] = []

        for state_idx in range(n_states):
            trans_total = 0.0
            best_prev_idx = 0
            best_prev_score = -1e18

            for prev_idx in range(n_states):
                trans_prob = max(
                    1e-9,
                    _safe_float(_CYBER_REGIME_TRANSITIONS[prev_idx][state_idx], 0.0),
                )
                trans_total += alpha_prev[prev_idx] * trans_prob
                candidate = viterbi_prev[prev_idx] + math.log(trans_prob)
                if candidate > best_prev_score:
                    best_prev_score = candidate
                    best_prev_idx = prev_idx

            emission_prob = max(1e-9, emissions[state_idx])
            alpha_next.append(emission_prob * trans_total)
            viterbi_next.append(best_prev_score + math.log(emission_prob))
            back_row.append(best_prev_idx)

        alpha_prev = _normalize_distribution(alpha_next)
        viterbi_prev = viterbi_next
        backpointers.append(back_row)

    final_state_idx = max(range(n_states), key=lambda idx: viterbi_prev[idx])
    path_idx = [0 for _ in observations]
    if observations:
        path_idx[-1] = final_state_idx
        for step in range(len(observations) - 1, 0, -1):
            prev_state_idx = backpointers[step][path_idx[step]]
            path_idx[step - 1] = prev_state_idx

    last_transition_bin = 0
    for idx in range(1, len(path_idx)):
        if path_idx[idx] != path_idx[idx - 1]:
            last_transition_bin = idx

    transition_ts = max(
        0.0,
        _safe_float(min_fetched_ts, 0.0)
        + (float(last_transition_bin + 1) * max(1e-6, _safe_float(bin_width, 1.0))),
    )

    posterior = alpha_prev if alpha_prev else [1.0] + [0.0 for _ in range(n_states - 1)]
    while len(posterior) < n_states:
        posterior.append(0.0)
    posterior = _normalize_distribution(posterior[:n_states])

    posterior_named = {
        state: round(posterior[idx], 6) for idx, state in enumerate(states)
    }

    return {
        "state": states[max(range(n_states), key=lambda idx: posterior[idx])],
        "posterior": posterior_named,
        "p_baseline": posterior_named.get("baseline", 0.0),
        "p_elevated_chatter": posterior_named.get("elevated_chatter", 0.0),
        "p_active_exploitation_wave": posterior_named.get(
            "active_exploitation_wave",
            0.0,
        ),
        "p_supply_chain_campaign": posterior_named.get("supply_chain_campaign", 0.0),
        "p_geopolitical_targeting_shift": posterior_named.get(
            "geopolitical_targeting_shift",
            0.0,
        ),
        "last_transition_bin": int(last_transition_bin),
        "last_transition_ts": round(transition_ts, 6),
        "state_path": [states[idx] for idx in path_idx] if path_idx else [states[0]],
        "observation_mean": round(
            sum(observations) / float(len(observations)) if observations else 0.0,
            6,
        ),
        "observation_peak": round(max(observations) if observations else 0.0, 6),
    }


def _cyber_regime_policy(posterior: dict[str, Any]) -> dict[str, Any]:
    state = str(posterior.get("state", "baseline") or "baseline").strip().lower()
    p_active_wave = max(
        0.0,
        min(1.0, _safe_float(posterior.get("p_active_exploitation_wave", 0.0), 0.0)),
    )
    p_supply_chain = max(
        0.0,
        min(1.0, _safe_float(posterior.get("p_supply_chain_campaign", 0.0), 0.0)),
    )
    p_geo_shift = max(
        0.0,
        min(
            1.0,
            _safe_float(posterior.get("p_geopolitical_targeting_shift", 0.0), 0.0),
        ),
    )
    p_elevated = max(
        0.0,
        min(1.0, _safe_float(posterior.get("p_elevated_chatter", 0.0), 0.0)),
    )

    if state in {"active_exploitation_wave", "supply_chain_campaign"}:
        threshold = 6
        crawl_multiplier = 1.8
        query_expansion_multiplier = 1.8
    elif state == "geopolitical_targeting_shift":
        threshold = 7
        crawl_multiplier = 1.5
        query_expansion_multiplier = 1.5
    elif state == "elevated_chatter":
        threshold = 7
        crawl_multiplier = 1.25
        query_expansion_multiplier = 1.25
    else:
        threshold = 8
        crawl_multiplier = 1.0
        query_expansion_multiplier = 1.0

    pressure = max(
        0.0,
        min(
            1.0,
            (p_active_wave * 0.40)
            + (p_supply_chain * 0.25)
            + (p_geo_shift * 0.20)
            + (p_elevated * 0.15),
        ),
    )
    weighted_threshold = max(
        4,
        min(
            10,
            int(
                round(
                    (8.0 * max(0.0, 1.0 - pressure))
                    + (6.0 * min(1.0, p_active_wave + p_supply_chain))
                    + (7.0 * min(1.0, p_geo_shift + p_elevated))
                )
            ),
        ),
    )
    threshold = min(threshold, weighted_threshold)

    return {
        "state": state,
        "risk_score_threshold": int(threshold),
        "crawl_budget_multiplier": round(crawl_multiplier, 6),
        "query_expansion_multiplier": round(query_expansion_multiplier, 6),
        "pressure": round(pressure, 6),
    }


def _query_cyber_regime_state(
    nodes: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    repo_filter = str(args.get("repo", "") or "").strip().lower()
    window_ticks = max(1, min(20_000, _safe_int(args.get("window_ticks", 1440), 1440)))
    state_bins = max(3, min(48, _safe_int(args.get("state_bins", 8), 8)))
    threat_limit = max(32, min(512, _safe_int(args.get("threat_limit", 256), 256)))
    min_weak_label_score = max(
        -16,
        min(16, _safe_int(args.get("min_weak_label_score", 1), 1)),
    )

    github_payload = _query_github_threat_radar(
        nodes,
        {
            "repo": repo_filter,
            "window_ticks": window_ticks,
            "state_bins": state_bins,
            "limit": threat_limit,
            "llm_enabled": False,
            "use_daimoi_budget": args.get("use_daimoi_budget", True),
            "_threat_compute_budget": args.get("_threat_compute_budget", {}),
            "min_weak_label_score": int(min_weak_label_score),
            "proximity_seed_set": str(
                args.get("proximity_seed_set", args.get("seed_set", "default"))
                or "default"
            ).strip(),
        },
    )

    threat_rows = (
        github_payload.get("threats", [])
        if isinstance(github_payload.get("threats", []), list)
        else []
    )
    if threat_rows:
        latest_fetched_ts = max(
            _safe_float(row.get("fetched_ts", 0.0), 0.0)
            for row in threat_rows
            if isinstance(row, dict)
        )
    else:
        latest_fetched_ts = 0.0
    min_fetched_ts = max(0.0, latest_fetched_ts - float(window_ticks))
    bin_width = float(window_ticks) / float(max(1, state_bins))

    bin_rows: list[dict[str, Any]] = [
        {
            "count": 0,
            "critical_count": 0,
            "high_count": 0,
            "avg_risk": 0.0,
            "corroborated_count": 0,
            "source_tier_high_count": 0,
            "proximity_critical_count": 0,
        }
        for _ in range(state_bins)
    ]
    risk_sums = [0.0 for _ in range(state_bins)]

    for row in threat_rows:
        if not isinstance(row, dict):
            continue
        fetched_ts = max(0.0, _safe_float(row.get("fetched_ts", 0.0), 0.0))
        if min_fetched_ts > 0.0 and fetched_ts > 0.0 and fetched_ts < min_fetched_ts:
            continue

        bin_idx = _proximity_bin_index(
            fetched_ts,
            min_fetched_ts=min_fetched_ts,
            window_ticks=window_ticks,
            state_bins=state_bins,
        )
        if bin_idx < 0 or bin_idx >= state_bins:
            continue

        deterministic_score = max(
            0,
            _safe_int(
                row.get("deterministic_score", row.get("risk_score", 0)),
                0,
            ),
        )
        bucket = bin_rows[bin_idx]
        bucket["count"] = _safe_int(bucket.get("count", 0), 0) + 1
        if deterministic_score >= 11:
            bucket["critical_count"] = _safe_int(bucket.get("critical_count", 0), 0) + 1
        if deterministic_score >= 8:
            bucket["high_count"] = _safe_int(bucket.get("high_count", 0), 0) + 1
        if _safe_int(row.get("corroboration_count", 1), 1) >= 2:
            bucket["corroborated_count"] = (
                _safe_int(bucket.get("corroborated_count", 0), 0) + 1
            )
        if _safe_int(row.get("source_tier", 1), 1) >= 3:
            bucket["source_tier_high_count"] = (
                _safe_int(bucket.get("source_tier_high_count", 0), 0) + 1
            )
        if _safe_float(row.get("proximity_p_critical_max", 0.0), 0.0) >= 0.55:
            bucket["proximity_critical_count"] = (
                _safe_int(bucket.get("proximity_critical_count", 0), 0) + 1
            )
        risk_sums[bin_idx] += float(deterministic_score)

    for idx in range(state_bins):
        count = max(0, _safe_int(bin_rows[idx].get("count", 0), 0))
        if count <= 0:
            bin_rows[idx]["avg_risk"] = 0.0
        else:
            bin_rows[idx]["avg_risk"] = round(risk_sums[idx] / float(count), 6)

    posterior = _cyber_regime_state_posterior(
        bin_rows,
        min_fetched_ts=min_fetched_ts,
        bin_width=bin_width,
    )
    policy = _cyber_regime_policy(posterior)

    return {
        "repo": repo_filter,
        "window_ticks": int(window_ticks),
        "state_bins": int(state_bins),
        "count": int(max(0, _safe_int(github_payload.get("count", 0), 0))),
        "critical_count": int(
            max(0, _safe_int(github_payload.get("critical_count", 0), 0))
        ),
        "high_count": int(max(0, _safe_int(github_payload.get("high_count", 0), 0))),
        "state": str(posterior.get("state", "baseline") or "baseline").strip(),
        "posterior": (
            posterior.get("posterior", {})
            if isinstance(posterior.get("posterior", {}), dict)
            else {}
        ),
        "p_baseline": _safe_float(posterior.get("p_baseline", 0.0), 0.0),
        "p_elevated_chatter": _safe_float(
            posterior.get("p_elevated_chatter", 0.0),
            0.0,
        ),
        "p_active_exploitation_wave": _safe_float(
            posterior.get("p_active_exploitation_wave", 0.0),
            0.0,
        ),
        "p_supply_chain_campaign": _safe_float(
            posterior.get("p_supply_chain_campaign", 0.0),
            0.0,
        ),
        "p_geopolitical_targeting_shift": _safe_float(
            posterior.get("p_geopolitical_targeting_shift", 0.0),
            0.0,
        ),
        "last_transition_bin": max(
            0,
            _safe_int(posterior.get("last_transition_bin", 0), 0),
        ),
        "last_transition_ts": round(
            max(0.0, _safe_float(posterior.get("last_transition_ts", 0.0), 0.0)),
            6,
        ),
        "state_path": (
            posterior.get("state_path", [])
            if isinstance(posterior.get("state_path", []), list)
            else []
        ),
        "observation_mean": round(
            max(0.0, _safe_float(posterior.get("observation_mean", 0.0), 0.0)),
            6,
        ),
        "observation_peak": round(
            max(0.0, _safe_float(posterior.get("observation_peak", 0.0), 0.0)),
            6,
        ),
        "policy": policy,
        "bins": bin_rows,
        "scoring": github_payload.get("scoring", {}),
        "min_weak_label_score": int(min_weak_label_score),
    }


def _query_cyber_risk_radar(
    nodes: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    limit = max(1, min(128, _safe_int(args.get("limit", 24), 24)))
    min_weak_label_score = max(
        -16,
        min(16, _safe_int(args.get("min_weak_label_score", 1), 1)),
    )
    apply_regime_threshold = str(
        args.get("apply_regime_threshold", "1") or "1"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    regime_payload = _query_cyber_regime_state(
        nodes,
        {
            **args,
            "min_weak_label_score": int(min_weak_label_score),
        },
    )
    github_payload = _query_github_threat_radar(
        nodes,
        {
            "repo": str(args.get("repo", "") or "").strip().lower(),
            "window_ticks": max(
                1,
                min(20_000, _safe_int(args.get("window_ticks", 1440), 1440)),
            ),
            "state_bins": max(3, min(48, _safe_int(args.get("state_bins", 8), 8))),
            "limit": max(16, min(512, _safe_int(args.get("threat_limit", 256), 256))),
            "llm_enabled": args.get("llm_enabled", True),
            "use_daimoi_budget": args.get("use_daimoi_budget", True),
            "_threat_compute_budget": args.get("_threat_compute_budget", {}),
            "min_weak_label_score": int(min_weak_label_score),
            "proximity_seed_set": str(
                args.get("proximity_seed_set", args.get("seed_set", "default"))
                or "default"
            ).strip(),
        },
    )

    policy = regime_payload.get("policy", {})
    threshold = max(
        0,
        _safe_int(
            policy.get("risk_score_threshold", 8) if isinstance(policy, dict) else 8,
            8,
        ),
    )
    threshold_base = int(threshold)
    threshold_fallback_applied = False
    threshold_fallback_reason = ""
    regime_state = str(regime_payload.get("state", "baseline") or "baseline").strip()
    regime_pressure = max(
        0.0,
        min(
            1.0,
            _safe_float(
                policy.get("pressure", 0.0) if isinstance(policy, dict) else 0.0,
                0.0,
            ),
        ),
    )

    raw_rows = (
        github_payload.get("threats", [])
        if isinstance(github_payload.get("threats", []), list)
        else []
    )

    def _derived_weak_label_score(row: dict[str, Any]) -> int:
        if "weak_label_score" in row:
            return int(_safe_int(row.get("weak_label_score", 0), 0))
        label_token = str(row.get("weak_label", "") or "").strip().lower()
        if label_token == "security_likely":
            return 4
        if label_token == "security_possible":
            return 1
        if label_token == "low_security_relevance":
            return -2
        return 0

    def _passes_security_gate(row: dict[str, Any]) -> bool:
        has_weak_label_metadata = "weak_label_score" in row or bool(
            str(row.get("weak_label", "") or "").strip()
        )
        if not has_weak_label_metadata:
            return True
        return bool(_derived_weak_label_score(row) >= int(min_weak_label_score))

    rows_all: list[dict[str, Any]] = []
    security_gate_filtered_count = 0
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        score = max(0, _safe_int(row.get("risk_score", 0), 0))
        passes_security_gate = _passes_security_gate(row)
        weak_label_score = _derived_weak_label_score(row)
        if not passes_security_gate:
            security_gate_filtered_count += 1
        enriched = dict(row)
        enriched["regime_state"] = regime_state
        enriched["regime_pressure"] = round(regime_pressure, 6)
        enriched["security_label_score"] = int(weak_label_score)
        enriched["passes_security_gate"] = bool(passes_security_gate)
        enriched["regime_risk_score_threshold"] = int(threshold)
        enriched["passes_regime_threshold"] = bool(
            score >= threshold and passes_security_gate
        )
        rows_all.append(enriched)

    rows = list(rows_all)

    if apply_regime_threshold:
        rows = [row for row in rows if bool(row.get("passes_regime_threshold", False))]

    if apply_regime_threshold and not rows:
        fallback_threshold = max(4, int(threshold_base) - 3)
        fallback_rows: list[dict[str, Any]] = []
        for row in rows_all:
            if not bool(row.get("passes_security_gate", False)):
                continue
            score = max(0, _safe_int(row.get("risk_score", 0), 0))
            if score < fallback_threshold:
                continue
            fallback_row = dict(row)
            fallback_row["regime_risk_score_threshold"] = int(fallback_threshold)
            fallback_row["passes_regime_threshold"] = True
            fallback_rows.append(fallback_row)

        if fallback_rows:
            rows = fallback_rows
            threshold = int(fallback_threshold)
            threshold_fallback_applied = True
            threshold_fallback_reason = "empty_after_regime_threshold"

    rows.sort(
        key=lambda row: (
            -_safe_int(row.get("risk_score", 0), 0),
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("canonical_url", "")),
        )
    )

    return {
        "repo": str(github_payload.get("repo", "") or "").strip().lower(),
        "window_ticks": int(
            max(1, _safe_int(github_payload.get("window_ticks", 1440), 1440))
        ),
        "count": int(len(rows)),
        "critical_count": int(
            sum(1 for row in rows if _safe_int(row.get("risk_score", 0), 0) >= 11)
        ),
        "high_count": int(
            sum(1 for row in rows if 8 <= _safe_int(row.get("risk_score", 0), 0) < 11)
        ),
        "medium_count": int(
            sum(1 for row in rows if 5 <= _safe_int(row.get("risk_score", 0), 0) < 8)
        ),
        "low_count": int(
            sum(1 for row in rows if _safe_int(row.get("risk_score", 0), 0) < 5)
        ),
        "apply_regime_threshold": bool(apply_regime_threshold),
        "regime": {
            "state": regime_state,
            "posterior": regime_payload.get("posterior", {}),
            "policy": policy,
            "observation_mean": regime_payload.get("observation_mean", 0.0),
            "observation_peak": regime_payload.get("observation_peak", 0.0),
        },
        "scoring": {
            "mode": "cyber_regime_context",
            "risk_score_threshold": int(threshold),
            "risk_score_threshold_base": int(threshold_base),
            "risk_score_threshold_fallback_applied": bool(threshold_fallback_applied),
            "risk_score_threshold_fallback_reason": str(
                threshold_fallback_reason or ""
            ),
            "regime_state": regime_state,
            "regime_pressure": round(regime_pressure, 6),
            "min_weak_label_score": int(min_weak_label_score),
            "security_gate_filtered_count": int(max(0, security_gate_filtered_count)),
            "base": github_payload.get("scoring", {}),
        },
        "threats": rows[:limit],
    }


def _query_proximity_radar(
    nodes: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    resources = _github_resource_rows(nodes)
    repo_filter = str(args.get("repo", "") or "").strip().lower()
    window_ticks = max(1, min(20_000, _safe_int(args.get("window_ticks", 1440), 1440)))
    limit = max(1, min(512, _safe_int(args.get("limit", 24), 24)))
    state_bins = max(3, min(24, _safe_int(args.get("state_bins", 6), 6)))
    seed_terms = _proximity_seed_terms(args)

    latest_fetched_ts = max(
        [_safe_float(row.get("fetched_ts", 0.0), 0.0) for row in resources],
        default=0.0,
    )
    min_fetched_ts = max(0.0, latest_fetched_ts - float(window_ticks))
    split_ts = min_fetched_ts + (float(window_ticks) * 0.5)
    bin_width = float(window_ticks) / float(max(1, state_bins))

    term_stats: dict[str, dict[str, Any]] = {}
    for row in resources:
        row_repo = str(row.get("repo", "") or "").strip().lower()
        if repo_filter and row_repo != repo_filter:
            continue

        fetched_ts = max(0.0, _safe_float(row.get("fetched_ts", 0.0), 0.0))
        if min_fetched_ts > 0.0 and fetched_ts > 0.0 and fetched_ts < min_fetched_ts:
            continue

        title = str(row.get("title", "") or "").strip()
        summary = str(row.get("summary", "") or "").strip()
        excerpt = str(row.get("text_excerpt", "") or "").strip()
        corpus = "\n".join(token for token in (title, summary, excerpt) if token)
        corpus_lower = corpus.lower()

        row_terms: set[str] = set(_tokenize_proximity_terms(corpus))
        row_terms.update(
            token.upper().lower()
            for token in re.findall(r"cve-\d{4}-\d{4,7}", corpus_lower)
            if token
        )

        for atom in row.get("atoms", []):
            if not isinstance(atom, dict):
                continue
            atom_kind = str(atom.get("kind", "") or "").strip().lower()
            if atom_kind == "mentions":
                token = str(atom.get("term", "") or "").strip().lower()
                if token:
                    row_terms.add(token)
            elif atom_kind == "references_cve":
                token = str(atom.get("cve_id", "") or "").strip().lower()
                if token:
                    row_terms.add(token)
            elif atom_kind == "changes_dependency":
                token = str(atom.get("dep_name", "") or "").strip().lower()
                if token:
                    row_terms.add(token)

        if not row_terms:
            continue

        row_seed_hits: set[str] = set()
        for seed in seed_terms:
            if seed in row_terms or seed in corpus_lower:
                row_seed_hits.add(seed)

        row_has_cve = any(term.startswith("cve-") for term in row_terms)

        source_id = (
            str(row.get("canonical_url", "") or "").strip()
            or str(row.get("id", "") or "").strip()
        )
        if not source_id:
            source_id = f"row:{len(term_stats)}"
        bin_idx = _proximity_bin_index(
            fetched_ts,
            min_fetched_ts=min_fetched_ts,
            window_ticks=window_ticks,
            state_bins=state_bins,
        )

        for term in sorted(row_terms):
            if term in _PROXIMITY_STOPWORDS:
                continue
            if len(term) > 64:
                continue
            if term.startswith("http"):
                continue

            stats = term_stats.setdefault(
                term,
                {
                    "term": term,
                    "recent": 0,
                    "previous": 0,
                    "sources": set(),
                    "repos": set(),
                    "seed_cooc": {},
                    "last_seen_ts": 0.0,
                    "evidence": [],
                    "bin_rows": {},
                },
            )
            if fetched_ts >= split_ts:
                stats["recent"] = _safe_int(stats.get("recent", 0), 0) + 1
            else:
                stats["previous"] = _safe_int(stats.get("previous", 0), 0) + 1

            sources = stats.get("sources")
            if isinstance(sources, set):
                sources.add(source_id)
            repos = stats.get("repos")
            if isinstance(repos, set):
                repo_name = str(row.get("repo", "") or "").strip().lower()
                if repo_name:
                    repos.add(repo_name)

            stats["last_seen_ts"] = max(
                _safe_float(stats.get("last_seen_ts", 0.0), 0.0),
                fetched_ts,
            )

            raw_seed_cooc = stats.get("seed_cooc")
            seed_cooc: dict[str, int] = {}
            if isinstance(raw_seed_cooc, dict):
                for seed_key, seed_value in raw_seed_cooc.items():
                    key = str(seed_key or "").strip().lower()
                    if key:
                        seed_cooc[key] = _safe_int(seed_value, 0)
            for seed in sorted(row_seed_hits):
                seed_cooc[seed] = _safe_int(seed_cooc.get(seed, 0), 0) + 1
            stats["seed_cooc"] = seed_cooc

            raw_bin_rows = stats.get("bin_rows")
            bin_rows = raw_bin_rows if isinstance(raw_bin_rows, dict) else {}
            raw_bin = bin_rows.get(bin_idx)
            if isinstance(raw_bin, dict):
                bin_row = dict(raw_bin)
            else:
                bin_row = {
                    "count": 0,
                    "seed_hits": 0,
                    "cve_hits": 0,
                    "sources": set(),
                }

            raw_sources = bin_row.get("sources")
            sources_set: set[str] = set()
            if isinstance(raw_sources, set):
                for token in raw_sources:
                    cleaned = str(token or "").strip()
                    if cleaned:
                        sources_set.add(cleaned)
            elif isinstance(raw_sources, list):
                for token in raw_sources:
                    cleaned = str(token or "").strip()
                    if cleaned:
                        sources_set.add(cleaned)
            sources_set.add(source_id)

            bin_row["count"] = _safe_int(bin_row.get("count", 0), 0) + 1
            bin_row["seed_hits"] = _safe_int(bin_row.get("seed_hits", 0), 0) + len(
                row_seed_hits
            )
            cve_bump = 0
            if term.startswith("cve-"):
                cve_bump = 1
            elif row_has_cve and term in row_seed_hits:
                cve_bump = 1
            bin_row["cve_hits"] = _safe_int(bin_row.get("cve_hits", 0), 0) + cve_bump
            bin_row["sources"] = sources_set
            bin_rows[bin_idx] = bin_row
            stats["bin_rows"] = bin_rows

            evidence = stats.get("evidence")
            if isinstance(evidence, list) and len(evidence) < 4:
                evidence.append(
                    {
                        "repo": str(row.get("repo", "") or "").strip(),
                        "canonical_url": str(
                            row.get("canonical_url", "") or ""
                        ).strip(),
                        "title": title,
                        "fetched_ts": round(fetched_ts, 6),
                    }
                )

    rows: list[dict[str, Any]] = []
    flattened_hits: list[dict[str, Any]] = []
    state_counts = {state: 0 for state in _PROXIMITY_HMM_STATES}
    for term, stats in term_stats.items():
        recent = max(0, _safe_int(stats.get("recent", 0), 0))
        previous = max(0, _safe_int(stats.get("previous", 0), 0))
        total = recent + previous
        if total <= 0:
            continue

        sources = stats.get("sources")
        source_count = len(sources) if isinstance(sources, set) else 0
        if source_count <= 0:
            continue

        raw_seed_cooc = stats.get("seed_cooc")
        seed_cooc: dict[str, int] = {}
        if isinstance(raw_seed_cooc, dict):
            for seed_key, seed_value in raw_seed_cooc.items():
                key = str(seed_key or "").strip().lower()
                if key:
                    seed_cooc[key] = _safe_int(seed_value, 0)
        top_seed = ""
        top_seed_count = 0
        if seed_cooc:
            top_seed, top_seed_count = sorted(
                (
                    (str(seed), _safe_int(count, 0))
                    for seed, count in seed_cooc.items()
                    if str(seed).strip()
                ),
                key=lambda item: (-item[1], item[0]),
            )[0]

        embed_score = max(0.0, min(1.0, float(top_seed_count) / float(max(1, total))))
        graph_score = max(
            0.0,
            min(
                1.0,
                float(
                    len(
                        [
                            seed
                            for seed, count in seed_cooc.items()
                            if _safe_int(count, 0) > 0
                        ]
                    )
                )
                / 4.0,
            ),
        )
        burst_raw = (float(recent) - float(previous)) / float(max(1, previous))
        burst_score = max(0.0, min(1.0, burst_raw / 3.0))
        source_diversity = max(0.0, min(1.0, float(source_count) / 4.0))
        semantic_agree = embed_score >= 0.50
        graph_agree = graph_score >= 0.35
        temporal_agree = burst_score >= 0.35 or source_diversity >= 0.50
        agreement_axes: list[str] = []
        if semantic_agree:
            agreement_axes.append("semantic")
        if graph_agree:
            agreement_axes.append("graph")
        if temporal_agree:
            agreement_axes.append("temporal")
        agreement_count = len(agreement_axes)
        promotion_gate_passed = agreement_count >= 2

        combined_score_raw = (
            (embed_score * 0.35)
            + (graph_score * 0.35)
            + (burst_score * 0.20)
            + (source_diversity * 0.10)
        )
        if not promotion_gate_passed:
            combined_score_raw = min(combined_score_raw, 0.59)

        raw_bin_rows = stats.get("bin_rows")
        bin_rows = raw_bin_rows if isinstance(raw_bin_rows, dict) else {}
        bin_series: list[dict[str, Any]] = []
        for idx in range(state_bins):
            raw_bin = bin_rows.get(idx)
            if not isinstance(raw_bin, dict):
                bin_series.append(
                    {
                        "count": 0,
                        "source_count": 0,
                        "seed_hits": 0,
                        "cve_hits": 0,
                    }
                )
                continue

            raw_sources = raw_bin.get("sources")
            if isinstance(raw_sources, set):
                source_bin_count = len(raw_sources)
            elif isinstance(raw_sources, list):
                source_bin_count = len(
                    {
                        str(token or "").strip()
                        for token in raw_sources
                        if str(token or "").strip()
                    }
                )
            else:
                source_bin_count = 0

            bin_series.append(
                {
                    "count": max(0, _safe_int(raw_bin.get("count", 0), 0)),
                    "source_count": max(0, _safe_int(source_bin_count, 0)),
                    "seed_hits": max(0, _safe_int(raw_bin.get("seed_hits", 0), 0)),
                    "cve_hits": max(0, _safe_int(raw_bin.get("cve_hits", 0), 0)),
                }
            )

        state_summary = _proximity_state_posterior(
            bin_series,
            min_fetched_ts=min_fetched_ts,
            bin_width=bin_width,
        )
        p_active = max(
            0.0, min(1.0, _safe_float(state_summary.get("p_active", 0.0), 0.0))
        )
        p_critical = max(
            0.0, min(1.0, _safe_float(state_summary.get("p_critical", 0.0), 0.0))
        )
        state_weight = max(0.0, min(1.0, (p_active * 0.50) + (p_critical * 1.00)))
        combined_score = max(
            0.0,
            min(1.0, (combined_score_raw * 0.65) + (state_weight * 0.35)),
        )
        if not promotion_gate_passed:
            p_critical = min(p_critical, 0.54)
            p_active = min(p_active, 0.74)
            combined_score = min(combined_score, 0.74)

        state_name = str(state_summary.get("state", "background") or "background")
        if state_name in state_counts:
            state_counts[state_name] = _safe_int(state_counts.get(state_name, 0), 0) + 1

        last_seen_ts = round(
            max(0.0, _safe_float(stats.get("last_seen_ts", 0.0), 0.0)), 6
        )
        hits = [
            {
                "kind": "embed",
                "term_id": term,
                "seed_id": top_seed,
                "score": round(embed_score, 6),
                "ts": last_seen_ts,
            },
            {
                "kind": "graph",
                "term_id": term,
                "seed_id": top_seed,
                "score": round(graph_score, 6),
                "ts": last_seen_ts,
            },
            {
                "kind": "burst",
                "term_id": term,
                "seed_id": "timeline",
                "score": round(burst_score, 6),
                "ts": last_seen_ts,
            },
        ]
        for hit in hits:
            hit["atom"] = "proximity_hit"
            hit["state"] = state_name
            hit["p_critical"] = round(p_critical, 6)
            hit["promotion_gate_passed"] = bool(promotion_gate_passed)

        rows.append(
            {
                "term": term,
                "score": round(combined_score, 6),
                "score_raw": round(combined_score_raw, 6),
                "embed_top1": top_seed,
                "embed_score": round(embed_score, 6),
                "ppr_score": round(graph_score, 6),
                "burst_score": round(burst_score, 6),
                "source_diversity": int(source_count),
                "promotion_gate_passed": bool(promotion_gate_passed),
                "agreement_count": int(agreement_count),
                "agreement_axes": agreement_axes,
                "recent_count": int(recent),
                "previous_count": int(previous),
                "last_seen_ts": last_seen_ts,
                "state": state_name,
                "p_background": _safe_float(
                    state_summary.get("p_background", 0.0), 0.0
                ),
                "p_emerging": _safe_float(state_summary.get("p_emerging", 0.0), 0.0),
                "p_active": p_active,
                "p_critical": p_critical,
                "last_transition_bin": max(
                    0, _safe_int(state_summary.get("last_transition_bin", 0), 0)
                ),
                "last_transition_ts": round(
                    max(
                        0.0,
                        _safe_float(state_summary.get("last_transition_ts", 0.0), 0.0),
                    ),
                    6,
                ),
                "state_path": (
                    state_summary.get("state_path", [])
                    if isinstance(state_summary.get("state_path", []), list)
                    else []
                ),
                "top_evidence": sorted(
                    stats.get("evidence", []),
                    key=lambda item: (
                        -_safe_float(item.get("fetched_ts", 0.0), 0.0),
                        str(item.get("canonical_url", "")),
                    ),
                )[:3],
                "proximity_hits": hits,
            }
        )
        flattened_hits.extend(hits)

    rows.sort(
        key=lambda row: (
            -_safe_float(row.get("score", 0.0), 0.0),
            -_safe_int(row.get("recent_count", 0), 0),
            -_safe_int(row.get("source_diversity", 0), 0),
            str(row.get("term", "")),
        )
    )

    selected = rows[:limit]
    selected_terms = {str(row.get("term", "") or "") for row in selected}
    filtered_hits = [
        hit
        for hit in flattened_hits
        if str(hit.get("term_id", "") or "") in selected_terms
    ]
    filtered_hits.sort(
        key=lambda row: (
            str(row.get("term_id", "")),
            str(row.get("kind", "")),
            -_safe_float(row.get("score", 0.0), 0.0),
        )
    )

    active_or_higher_count = _safe_int(state_counts.get("active", 0), 0) + _safe_int(
        state_counts.get("critical", 0),
        0,
    )
    promotion_gate_pass_count = sum(
        1 for row in rows if bool(row.get("promotion_gate_passed", False))
    )

    return {
        "window_ticks": int(window_ticks),
        "state_bins": int(state_bins),
        "repo": repo_filter,
        "seed_set": seed_terms,
        "count": len(rows),
        "promotion_gate_pass_count": int(promotion_gate_pass_count),
        "state_counts": state_counts,
        "active_or_higher_count": int(active_or_higher_count),
        "terms": selected,
        "proximity_hits": filtered_hits,
    }


def _query_entity_risk_state(
    nodes: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    proximity_payload = _query_proximity_radar(nodes, args)
    terms = (
        proximity_payload.get("terms", [])
        if isinstance(proximity_payload.get("terms", []), list)
        else []
    )

    entities: list[dict[str, Any]] = []
    for row in terms:
        if not isinstance(row, dict):
            continue
        entities.append(
            {
                "term": str(row.get("term", "") or "").strip(),
                "state": str(row.get("state", "background") or "background").strip(),
                "p_active": round(
                    max(0.0, min(1.0, _safe_float(row.get("p_active", 0.0), 0.0))),
                    6,
                ),
                "p_critical": round(
                    max(0.0, min(1.0, _safe_float(row.get("p_critical", 0.0), 0.0))),
                    6,
                ),
                "score": round(
                    max(0.0, min(1.0, _safe_float(row.get("score", 0.0), 0.0))),
                    6,
                ),
                "promotion_gate_passed": bool(row.get("promotion_gate_passed", False)),
                "agreement_count": int(
                    max(0, _safe_int(row.get("agreement_count", 0), 0))
                ),
                "agreement_axes": (
                    row.get("agreement_axes", [])
                    if isinstance(row.get("agreement_axes", []), list)
                    else []
                ),
                "last_transition_ts": round(
                    max(
                        0.0,
                        _safe_float(row.get("last_transition_ts", 0.0), 0.0),
                    ),
                    6,
                ),
                "top_evidence": (
                    row.get("top_evidence", [])
                    if isinstance(row.get("top_evidence", []), list)
                    else []
                )[:2],
            }
        )

    return {
        "window_ticks": int(
            max(1, _safe_int(proximity_payload.get("window_ticks", 1440), 1440))
        ),
        "state_bins": int(max(1, _safe_int(proximity_payload.get("state_bins", 6), 6))),
        "repo": str(proximity_payload.get("repo", "") or "").strip().lower(),
        "count": int(max(0, _safe_int(proximity_payload.get("count", 0), 0))),
        "promotion_gate_pass_count": int(
            max(0, _safe_int(proximity_payload.get("promotion_gate_pass_count", 0), 0))
        ),
        "state_counts": (
            proximity_payload.get("state_counts", {})
            if isinstance(proximity_payload.get("state_counts", {}), dict)
            else {}
        ),
        "active_or_higher_count": int(
            max(0, _safe_int(proximity_payload.get("active_or_higher_count", 0), 0))
        ),
        "entities": entities,
    }


def _sigmoid(value: float) -> float:
    bounded = max(-60.0, min(60.0, _safe_float(value, 0.0)))
    return 1.0 / (1.0 + math.exp(-bounded))


def _github_linear_classifier_score(feature_map: dict[str, float]) -> tuple[float, int]:
    logit = _safe_float(_GITHUB_LINEAR_CLASSIFIER_BIAS, -2.40)
    for name, weight in _GITHUB_LINEAR_CLASSIFIER_WEIGHTS.items():
        logit += _safe_float(weight, 0.0) * _safe_float(feature_map.get(name, 0.0), 0.0)
    probability = _sigmoid(logit)
    score = max(0, min(14, int(round(probability * 14.0))))
    return probability, score


def _github_threat_features(
    row: dict[str, Any],
) -> tuple[dict[str, float], list[str], list[str]]:
    atoms = row.get("atoms", []) if isinstance(row.get("atoms", []), list) else []
    atom_kinds: set[str] = set()
    mention_terms: set[str] = set()
    labels: set[str] = set()
    cves: set[str] = set()

    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        kind = str(atom.get("kind", "") or "").strip().lower()
        if not kind:
            continue
        atom_kinds.add(kind)
        if kind == "mentions":
            term = str(atom.get("term", "") or "").strip().lower()
            if term:
                mention_terms.add(term)
        elif kind == "has_label":
            label = str(atom.get("label", "") or "").strip().lower()
            if label:
                labels.add(label)
        elif kind == "references_cve":
            cve_id = str(atom.get("cve_id", "") or "").strip().upper()
            if cve_id:
                cves.add(cve_id)

    kind = str(row.get("kind", "") or "").strip().lower()
    state = str(row.get("state", "") or "").strip().lower()
    title = str(row.get("title", "") or "").strip().lower()
    importance_score = max(0, _safe_int(row.get("importance_score", 0), 0))

    title_term_hits = sum(
        1 for token in _GITHUB_THREAT_TERMS if token and token in title
    )
    mention_term_hits = sum(
        1 for term in mention_terms if term and term in _GITHUB_THREAT_TERMS
    )

    feature_map = {
        "is_advisory": 1.0 if kind == "github:advisory" else 0.0,
        "cve_count": float(min(3, len(cves))),
        "changes_dependency": 1.0 if "changes_dependency" in atom_kinds else 0.0,
        "security_label": (
            1.0
            if any(label in {"security", "hotfix", "bug"} for label in labels)
            else 0.0
        ),
        "mentions_security_term": 1.0 if mention_term_hits > 0 else 0.0,
        "title_security_term": 1.0 if title_term_hits > 0 else 0.0,
        "open_state": 1.0 if state == "open" else 0.0,
        "pr_merged": (
            1.0 if kind == "github:pr" and "pr_merged" in atom_kinds else 0.0
        ),
        "importance_scaled": min(1.0, float(importance_score) / 10.0),
        "security_term_density": min(
            1.0, float(mention_term_hits + min(1, title_term_hits)) / 4.0
        ),
    }

    signals: list[str] = []
    if feature_map["is_advisory"] > 0.0:
        signals.append("github_advisory")
    if feature_map["cve_count"] > 0.0:
        signals.append("references_cve")
    if feature_map["changes_dependency"] > 0.0:
        signals.append("changes_dependency")
    if feature_map["security_label"] > 0.0:
        signals.append("security_label")
    if feature_map["mentions_security_term"] > 0.0:
        signals.append("mentions_security_term")
    if feature_map["title_security_term"] > 0.0:
        signals.append("title_security_term")
    if feature_map["open_state"] > 0.0:
        signals.append("open_state")
    if feature_map["pr_merged"] > 0.0:
        signals.append("pr_merged")
    if feature_map["importance_scaled"] > 0.0:
        signals.append("importance_boost")

    return feature_map, sorted(set(signals)), sorted(cves)


def _github_threat_score(
    row: dict[str, Any],
) -> tuple[int, int, float, list[str], list[str]]:
    feature_map, signals, cves = _github_threat_features(row)
    importance_score = max(0, _safe_int(row.get("importance_score", 0), 0))

    heuristic_score = 0
    if feature_map["is_advisory"] > 0.0:
        heuristic_score += 8
    if feature_map["cve_count"] > 0.0:
        heuristic_score += 8 + min(4, max(0, len(cves) - 1))
    if feature_map["changes_dependency"] > 0.0:
        heuristic_score += 4
    if feature_map["security_label"] > 0.0:
        heuristic_score += 2
    if feature_map["mentions_security_term"] > 0.0:
        heuristic_score += 2
    if feature_map["title_security_term"] > 0.0:
        heuristic_score += 2
    if feature_map["open_state"] > 0.0:
        heuristic_score += 1
    if feature_map["pr_merged"] > 0.0:
        heuristic_score += 2
    if importance_score > 0:
        heuristic_score += min(4, max(1, int(round(importance_score / 2.0))))

    classifier_probability, classifier_score = _github_linear_classifier_score(
        feature_map
    )
    return (
        int(heuristic_score),
        int(classifier_score),
        classifier_probability,
        signals,
        cves,
    )


def _github_weak_label_registry(
    row: dict[str, Any],
    *,
    feature_map: dict[str, float],
    cves: list[str],
    source_tier: int,
    corroboration_count: int,
    proximity_p_active_max: float,
    proximity_p_critical_max: float,
) -> dict[str, Any]:
    votes: list[dict[str, Any]] = []

    def _add_vote(name: str, vote: int, reason: str) -> None:
        token = str(name or "").strip().lower()
        if not token:
            return
        votes.append(
            {
                "name": token,
                "vote": int(vote),
                "reason": str(reason or "").strip(),
            }
        )

    kind = str(row.get("kind", "") or "").strip().lower()
    state = str(row.get("state", "") or "").strip().lower()

    if kind == "github:advisory":
        _add_vote(
            "lf_advisory_channel",
            3,
            "resource originates from github advisory channel",
        )
    if cves:
        _add_vote(
            "lf_cve_reference",
            2,
            "resource references at least one CVE identifier",
        )
    if _safe_float(feature_map.get("changes_dependency", 0.0), 0.0) > 0.0:
        _add_vote(
            "lf_dependency_delta",
            1,
            "dependency or manifest changes were detected",
        )
    if _safe_float(feature_map.get("security_label", 0.0), 0.0) > 0.0:
        _add_vote(
            "lf_security_label",
            1,
            "security-focused label is present",
        )
    if _safe_float(feature_map.get("mentions_security_term", 0.0), 0.0) > 0.0:
        _add_vote(
            "lf_security_lexicon",
            1,
            "security lexicon appears in text atoms",
        )
    if state == "open" and cves:
        _add_vote(
            "lf_open_with_cve",
            1,
            "open resource still carrying CVE reference",
        )
    if source_tier >= 3:
        _add_vote(
            "lf_high_credibility_source",
            1,
            "source tier is release/advisory class",
        )
    if corroboration_count >= 2:
        _add_vote(
            "lf_cross_source_corroborated",
            1,
            "same CVE appears across multiple sources",
        )
    if proximity_p_critical_max >= 0.55:
        _add_vote(
            "lf_proximity_critical",
            1,
            "proximity engine indicates critical context",
        )
    elif proximity_p_active_max >= 0.55:
        _add_vote(
            "lf_proximity_active",
            1,
            "proximity engine indicates active context",
        )

    if (
        not cves
        and _safe_float(feature_map.get("security_label", 0.0), 0.0) <= 0.0
        and _safe_float(feature_map.get("mentions_security_term", 0.0), 0.0) <= 0.0
        and _safe_float(feature_map.get("title_security_term", 0.0), 0.0) <= 0.0
    ):
        _add_vote(
            "lf_low_signal_noise",
            -2,
            "resource lacks explicit security indicator atoms",
        )
    elif (
        not cves
        and _safe_float(feature_map.get("security_label", 0.0), 0.0) <= 0.0
        and _safe_float(feature_map.get("mentions_security_term", 0.0), 0.0) <= 0.0
    ):
        _add_vote(
            "lf_low_signal_weak",
            -1,
            "resource has weak direct security evidence",
        )

    raw_score = int(sum(_safe_int(vote.get("vote", 0), 0) for vote in votes))
    confidence = _sigmoid(float(raw_score) / 2.5)
    if raw_score >= 4:
        label = "security_likely"
    elif raw_score >= 1:
        label = "security_possible"
    elif raw_score <= -2:
        label = "low_security_relevance"
    else:
        label = "uncertain"

    votes.sort(
        key=lambda row: (
            -abs(_safe_int(row.get("vote", 0), 0)),
            -_safe_int(row.get("vote", 0), 0),
            str(row.get("name", "")),
        )
    )
    return {
        "version": _GITHUB_LABEL_FUNCTIONS_VERSION,
        "label": str(label),
        "score": int(raw_score),
        "confidence": round(max(0.0, min(1.0, confidence)), 6),
        "votes": votes[:8],
    }


def _github_source_tier(kind: str) -> int:
    token = str(kind or "").strip().lower()
    if token == "github:advisory":
        return 4
    if token == "github:release":
        return 3
    if token == "github:pr":
        return 2
    if token == "github:issue":
        return 1
    return 1


def _github_source_weight(tier: int) -> float:
    normalized = max(1, min(4, _safe_int(tier, 1)))
    mapping = {
        1: 0.35,
        2: 0.55,
        3: 0.75,
        4: 1.0,
    }
    return _safe_float(mapping.get(normalized, 0.35), 0.35)


def _github_row_proximity_terms(row: dict[str, Any]) -> list[str]:
    title = str(row.get("title", "") or "").strip()
    summary = str(row.get("summary", "") or "").strip()
    excerpt = str(row.get("text_excerpt", "") or "").strip()
    corpus = "\n".join(token for token in (title, summary, excerpt) if token)
    corpus_lower = corpus.lower()

    terms: set[str] = set(_tokenize_proximity_terms(corpus))
    terms.update(
        token.lower()
        for token in re.findall(r"cve-\d{4}-\d{4,7}", corpus_lower)
        if token
    )

    atoms = row.get("atoms", []) if isinstance(row.get("atoms", []), list) else []
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        atom_kind = str(atom.get("kind", "") or "").strip().lower()
        if atom_kind == "mentions":
            token = str(atom.get("term", "") or "").strip().lower()
            if token:
                terms.add(token)
        elif atom_kind == "references_cve":
            token = str(atom.get("cve_id", "") or "").strip().lower()
            if token:
                terms.add(token)
        elif atom_kind == "changes_dependency":
            token = str(atom.get("dep_name", "") or "").strip().lower()
            if token:
                terms.add(token)

    filtered = [
        term
        for term in sorted(terms)
        if term and len(term) <= 64 and term not in _PROXIMITY_STOPWORDS
    ]
    return filtered[:64]


def _query_github_threat_radar(
    nodes: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    resources = _github_resource_rows(nodes)
    repo_filter = str(args.get("repo", "") or "").strip().lower()
    window_ticks = max(1, min(20_000, _safe_int(args.get("window_ticks", 1440), 1440)))
    limit = max(1, min(128, _safe_int(args.get("limit", 24), 24)))
    min_weak_label_score = max(
        -16,
        min(16, _safe_int(args.get("min_weak_label_score", -999), -999)),
    )
    weak_label_filter_enabled = bool(min_weak_label_score > -900)
    weak_label_filtered_out = 0

    latest_fetched_ts = max(
        [_safe_float(row.get("fetched_ts", 0.0), 0.0) for row in resources],
        default=0.0,
    )
    min_fetched_ts = max(0.0, latest_fetched_ts - float(window_ticks))

    proximity_seed_set = str(
        args.get("proximity_seed_set", args.get("seed_set", "default")) or "default"
    ).strip()
    llm_requested = _truthy_query_flag(args.get("llm_enabled", "1"), default=True)
    use_daimoi_budget = _truthy_query_flag(
        args.get("use_daimoi_budget", args.get("compute_budget_bound", "1")),
        default=True,
    )
    raw_budget = args.get("_threat_compute_budget", {})
    compute_budget = raw_budget if isinstance(raw_budget, dict) else {}
    compute_budget_active = bool(use_daimoi_budget and compute_budget)
    allow_classifier = bool(
        compute_budget.get("allow_classifier", True) if compute_budget_active else True
    )
    allow_llm = bool(
        compute_budget.get("allow_llm", True) if compute_budget_active else True
    )
    llm_item_cap = max(
        0,
        _safe_int(
            compute_budget.get("llm_item_cap", _THREAT_RADAR_LLM_MAX_ITEMS)
            if compute_budget_active
            else _THREAT_RADAR_LLM_MAX_ITEMS,
            _THREAT_RADAR_LLM_MAX_ITEMS,
        ),
    )
    classifier_enabled = bool(_THREAT_RADAR_CLASSIFIER_ENABLED and allow_classifier)
    llm_allowed = bool(llm_requested and allow_llm and llm_item_cap > 0)
    proximity_payload = _query_proximity_radar(
        nodes,
        {
            "window_ticks": window_ticks,
            "repo": repo_filter,
            "seed_set": proximity_seed_set,
            "limit": 512,
            "state_bins": max(3, min(24, _safe_int(args.get("state_bins", 6), 6))),
        },
    )
    proximity_rows = (
        proximity_payload.get("terms", [])
        if isinstance(proximity_payload.get("terms", []), list)
        else []
    )
    proximity_index: dict[str, dict[str, Any]] = {}
    for token in proximity_rows:
        if not isinstance(token, dict):
            continue
        term = str(token.get("term", "") or "").strip().lower()
        if term and term not in proximity_index:
            proximity_index[term] = token

    filtered_resources: list[tuple[dict[str, Any], float]] = []
    for row in resources:
        row_repo = str(row.get("repo", "") or "").strip().lower()
        if repo_filter and row_repo != repo_filter:
            continue

        fetched_ts = max(0.0, _safe_float(row.get("fetched_ts", 0.0), 0.0))
        if min_fetched_ts > 0.0 and fetched_ts > 0.0 and fetched_ts < min_fetched_ts:
            continue

        filtered_resources.append((row, fetched_ts))

    cve_to_sources: dict[str, set[str]] = {}
    for row, _fetched_ts in filtered_resources:
        source_key = (
            str(row.get("canonical_url", "") or "").strip()
            or str(row.get("id", "") or "").strip()
        )
        if not source_key:
            continue
        atoms = row.get("atoms", []) if isinstance(row.get("atoms", []), list) else []
        cve_ids = {
            str(atom.get("cve_id", "") or "").strip().upper()
            for atom in atoms
            if isinstance(atom, dict)
            and str(atom.get("kind", "") or "").strip().lower() == "references_cve"
            and str(atom.get("cve_id", "") or "").strip()
        }
        for cve_id in cve_ids:
            bucket = cve_to_sources.get(cve_id)
            if not isinstance(bucket, set):
                bucket = set()
                cve_to_sources[cve_id] = bucket
            bucket.add(source_key)

    rows: list[dict[str, Any]] = []
    for row, fetched_ts in filtered_resources:
        feature_map, _, _ = _github_threat_features(row)
        (
            heuristic_score,
            classifier_score,
            classifier_probability,
            signals,
            cves,
        ) = _github_threat_score(row)
        deterministic_score = (
            classifier_score if classifier_enabled else heuristic_score
        )
        corroboration_count = max(
            [len(cve_to_sources.get(cve_id, set())) for cve_id in cves],
            default=1,
        )
        source_tier = _github_source_tier(str(row.get("kind", "") or ""))
        source_weight = _github_source_weight(source_tier)
        source_tier_boost = 1 if source_tier >= 3 else 0
        corroboration_boost = 0
        if corroboration_count >= 2:
            corroboration_boost = 1
        if corroboration_count >= 4:
            corroboration_boost = 2

        deterministic_score = min(
            14,
            max(
                0,
                _safe_int(deterministic_score, 0)
                + _safe_int(source_tier_boost, 0)
                + _safe_int(corroboration_boost, 0),
            ),
        )

        row_terms = _github_row_proximity_terms(row)
        proximity_hits: list[dict[str, Any]] = []
        p_active_max = 0.0
        p_critical_max = 0.0
        gated_hit_count = 0
        for term in row_terms:
            hit = proximity_index.get(term)
            if not isinstance(hit, dict):
                continue
            p_active = max(0.0, min(1.0, _safe_float(hit.get("p_active", 0.0), 0.0)))
            p_critical = max(
                0.0,
                min(1.0, _safe_float(hit.get("p_critical", 0.0), 0.0)),
            )
            gate_passed = bool(hit.get("promotion_gate_passed", False))
            if gate_passed:
                gated_hit_count += 1
                p_active_max = max(p_active_max, p_active)
                p_critical_max = max(p_critical_max, p_critical)
            proximity_hits.append(
                {
                    "term": term,
                    "state": str(
                        hit.get("state", "background") or "background"
                    ).strip(),
                    "p_active": round(p_active, 6),
                    "p_critical": round(p_critical, 6),
                    "promotion_gate_passed": gate_passed,
                    "score": round(
                        max(0.0, min(1.0, _safe_float(hit.get("score", 0.0), 0.0))),
                        6,
                    ),
                }
            )

        proximity_hits.sort(
            key=lambda item: (
                -_safe_float(item.get("p_critical", 0.0), 0.0),
                -_safe_float(item.get("p_active", 0.0), 0.0),
                str(item.get("term", "")),
            )
        )

        proximity_strategy = resolve_threat_proximity_strategy(
            p_active_max=p_active_max,
            p_critical_max=p_critical_max,
        )
        proximity_boost = int(proximity_strategy.get("boost", 0) or 0)
        deterministic_score = min(14, max(0, deterministic_score + proximity_boost))

        signals = apply_threat_signal_strategy(
            signals=signals,
            source_tier_boost=source_tier_boost,
            corroboration_boost=corroboration_boost,
            proximity_signal=str(proximity_strategy.get("signal", "") or ""),
        )

        weak_label = _github_weak_label_registry(
            row,
            feature_map=feature_map,
            cves=cves,
            source_tier=source_tier,
            corroboration_count=corroboration_count,
            proximity_p_active_max=p_active_max,
            proximity_p_critical_max=p_critical_max,
        )
        weak_label_score_value = int(_safe_int(weak_label.get("score", 0), 0))

        if weak_label_filter_enabled and weak_label_score_value < int(
            min_weak_label_score
        ):
            weak_label_filtered_out += 1
            continue

        if deterministic_score <= 0:
            continue

        threat_identity = _threat_row_identity(
            "github",
            {
                "repo": row.get("repo", ""),
                "number": row.get("number", 0),
                "kind": row.get("kind", ""),
                "canonical_url": row.get("canonical_url", ""),
                "title": row.get("title", ""),
            },
        )

        rows.append(
            {
                "_threat_id": threat_identity,
                "repo": row.get("repo", ""),
                "kind": row.get("kind", ""),
                "number": row.get("number", 0),
                "title": row.get("title", ""),
                "canonical_url": row.get("canonical_url", ""),
                "content_hash": row.get("content_hash", ""),
                "fetched_ts": fetched_ts,
                "updated_at": row.get("updated_at", ""),
                "state": row.get("state", ""),
                "importance_score": row.get("importance_score", 0),
                "signals": signals,
                "cves": cves,
                "summary": row.get("summary", ""),
                "text_excerpt": row.get("text_excerpt", ""),
                "deterministic_score": int(deterministic_score),
                "deterministic_score_legacy": int(heuristic_score),
                "classifier_score": int(classifier_score),
                "classifier_probability": round(
                    max(0.0, min(1.0, _safe_float(classifier_probability, 0.0))),
                    6,
                ),
                "classifier_version": _THREAT_RADAR_CLASSIFIER_VERSION,
                "source_tier": int(source_tier),
                "source_weight": round(max(0.0, min(1.0, source_weight)), 6),
                "corroboration_count": int(max(1, corroboration_count)),
                "proximity_p_active_max": round(max(0.0, min(1.0, p_active_max)), 6),
                "proximity_p_critical_max": round(
                    max(0.0, min(1.0, p_critical_max)),
                    6,
                ),
                "proximity_boost": int(proximity_boost),
                "proximity_gated_hits": int(max(0, gated_hit_count)),
                "proximity_terms": proximity_hits[:4],
                "weak_label": str(weak_label.get("label", "uncertain") or "uncertain"),
                "weak_label_score": int(weak_label_score_value),
                "weak_label_confidence": round(
                    max(
                        0.0,
                        min(1.0, _safe_float(weak_label.get("confidence", 0.0), 0.0)),
                    ),
                    6,
                ),
                "weak_label_votes": (
                    weak_label.get("votes", [])
                    if isinstance(weak_label.get("votes", []), list)
                    else []
                )[:8],
            }
        )

    dedupe_count = 0
    if rows:
        best_by_key: dict[str, dict[str, Any]] = {}
        for row in rows:
            content_hash = str(row.get("content_hash", "") or "").strip().lower()
            canonical_url = str(row.get("canonical_url", "") or "").strip()
            dedupe_key = (
                content_hash
                or canonical_url
                or str(row.get("_threat_id", "") or "").strip()
            )
            if not dedupe_key:
                dedupe_key = f"row:{len(best_by_key)}"

            existing = best_by_key.get(dedupe_key)
            if not isinstance(existing, dict):
                best_by_key[dedupe_key] = row
                continue

            row_score = _safe_int(row.get("deterministic_score", 0), 0)
            existing_score = _safe_int(existing.get("deterministic_score", 0), 0)
            row_ts = _safe_float(row.get("fetched_ts", 0.0), 0.0)
            existing_ts = _safe_float(existing.get("fetched_ts", 0.0), 0.0)
            keep_row = False
            if row_score > existing_score:
                keep_row = True
            elif row_score == existing_score and row_ts > existing_ts:
                keep_row = True
            elif (
                row_score == existing_score
                and row_ts == existing_ts
                and str(row.get("canonical_url", ""))
                < str(existing.get("canonical_url", ""))
            ):
                keep_row = True

            if keep_row:
                best_by_key[dedupe_key] = row

        deduped_rows = sorted(
            best_by_key.values(),
            key=lambda item: (
                -_safe_int(item.get("deterministic_score", 0), 0),
                -_safe_float(item.get("fetched_ts", 0.0), 0.0),
                str(item.get("canonical_url", "")),
            ),
        )
        dedupe_count = max(0, len(rows) - len(deduped_rows))
        rows = deduped_rows

    if llm_allowed:
        llm = _threat_llm_metrics(domain="github", rows=rows, max_items=llm_item_cap)
    else:
        llm = build_threat_llm_fallback(
            llm_requested=llm_requested,
            allow_llm=allow_llm,
            llm_item_cap=llm_item_cap,
            llm_model=_THREAT_RADAR_LLM_MODEL,
        )
    llm_metrics = (
        llm.get("metrics", {}) if isinstance(llm.get("metrics", {}), dict) else {}
    )
    llm_model = str(
        llm.get("model", _THREAT_RADAR_LLM_MODEL) or _THREAT_RADAR_LLM_MODEL
    )
    for row in rows:
        threat_id = str(row.get("_threat_id", "") or "")
        deterministic_score = max(0, _safe_int(row.get("deterministic_score", 0), 0))
        metrics = llm_metrics.get(threat_id, {}) if threat_id else {}
        if isinstance(metrics, dict) and metrics:
            final_score = _blend_llm_score(
                deterministic_score,
                metrics.get("overall_score", 0),
                max_score=14,
            )
            row["llm_score"] = int(
                round(
                    (
                        max(
                            0.0,
                            min(
                                100.0, _safe_float(metrics.get("overall_score", 0), 0.0)
                            ),
                        )
                        / 100.0
                    )
                    * 14.0
                )
            )
            row["threat_metrics"] = dict(metrics)
            row["llm_model"] = llm_model
        else:
            final_score = deterministic_score
        row["risk_score"] = int(final_score)
        row["risk_level"] = resolve_threat_risk_level(int(final_score))
        row.pop("_threat_id", None)

    rows.sort(
        key=lambda row: (
            -_safe_int(row.get("risk_score", 0), 0),
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("canonical_url", "")),
        )
    )

    level_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    repo_risk: dict[str, int] = {}
    for row in rows:
        level = str(row.get("risk_level", "low") or "low").strip().lower()
        if level in level_counts:
            level_counts[level] += 1
        repo = str(row.get("repo", "") or "").strip()
        if not repo:
            continue
        score = _safe_int(row.get("risk_score", 0), 0)
        previous = _safe_int(repo_risk.get(repo, 0), 0)
        if score > previous:
            repo_risk[repo] = score

    repo_hotlist = sorted(
        repo_risk.items(),
        key=lambda item: (-_safe_int(item[1], 0), str(item[0])),
    )

    weak_label_counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("weak_label", "") or "").strip()
        if not label:
            continue
        weak_label_counts[label] = weak_label_counts.get(label, 0) + 1

    return {
        "repo": repo_filter,
        "window_ticks": int(window_ticks),
        "count": len(rows),
        "dedupe_count": int(dedupe_count),
        "critical_count": int(level_counts["critical"]),
        "high_count": int(level_counts["high"]),
        "medium_count": int(level_counts["medium"]),
        "low_count": int(level_counts["low"]),
        "hot_repos": [
            {"repo": repo, "max_risk_score": score} for repo, score in repo_hotlist[:16]
        ],
        "scoring": {
            "mode": resolve_threat_scoring_mode(
                llm_enabled=bool(llm.get("enabled", False)),
                llm_applied=bool(llm.get("applied", False)),
                classifier_enabled=classifier_enabled,
            ),
            "classifier_enabled": bool(classifier_enabled),
            "classifier_requested": bool(_THREAT_RADAR_CLASSIFIER_ENABLED),
            "classifier_version": _THREAT_RADAR_CLASSIFIER_VERSION,
            "source_weighting": True,
            "corroboration_enabled": True,
            "proximity_state_influence": True,
            "proximity_seed_set": proximity_seed_set,
            "proximity_terms_indexed": len(proximity_index),
            "weak_supervision_enabled": True,
            "weak_supervision_version": _GITHUB_LABEL_FUNCTIONS_VERSION,
            "weak_label_counts": weak_label_counts,
            "weak_label_filter_enabled": bool(weak_label_filter_enabled),
            "weak_label_min_score": int(
                min_weak_label_score if weak_label_filter_enabled else 0
            ),
            "weak_label_filtered_out": int(max(0, weak_label_filtered_out)),
            "llm_requested": bool(llm_requested),
            "llm_allowed": bool(llm_allowed),
            "llm_enabled": bool(llm.get("enabled", False)),
            "llm_applied": bool(llm.get("applied", False)),
            "llm_model": llm_model,
            "llm_error": str(llm.get("error", "") or ""),
            "compute_budget": {
                "bound": bool(compute_budget_active),
                "mode": str(
                    compute_budget.get("mode", "") if compute_budget_active else ""
                ),
                "ratio": round(
                    max(
                        0.0,
                        min(
                            1.0,
                            _safe_float(
                                compute_budget.get("ratio", 0.0)
                                if compute_budget_active
                                else 0.0,
                                0.0,
                            ),
                        ),
                    ),
                    6,
                ),
                "queue_ratio": round(
                    max(
                        0.0,
                        min(
                            1.0,
                            _safe_float(
                                compute_budget.get("queue_ratio", 0.0)
                                if compute_budget_active
                                else 0.0,
                                0.0,
                            ),
                        ),
                    ),
                    6,
                ),
                "allow_classifier": bool(allow_classifier),
                "allow_llm": bool(allow_llm),
                "llm_item_cap": int(llm_item_cap),
                "reason": str(
                    compute_budget.get("reason", "") if compute_budget_active else ""
                ),
            },
        },
        "threats": rows[:limit],
    }


def _query_multi_threat_radar(
    nodes: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    window_ticks = max(1, min(20_000, _safe_int(args.get("window_ticks", 1440), 1440)))
    limit = max(1, min(128, _safe_int(args.get("limit", 24), 24)))
    per_domain_limit = max(1, min(96, _safe_int(args.get("per_domain_limit", 16), 16)))

    github_result = _query_github_threat_radar(
        nodes,
        {
            "window_ticks": window_ticks,
            "limit": per_domain_limit,
            "llm_enabled": args.get("llm_enabled", True),
            "use_daimoi_budget": args.get("use_daimoi_budget", True),
            "_threat_compute_budget": args.get("_threat_compute_budget", {}),
        },
    )
    hormuz_result = _query_hormuz_threat_radar(
        nodes,
        {
            "window_ticks": window_ticks,
            "limit": per_domain_limit,
        },
    )

    merged: list[dict[str, Any]] = []
    for row in github_result.get("threats", []):
        if not isinstance(row, dict):
            continue
        merged.append({"domain": "github", **row})
    for row in hormuz_result.get("threats", []):
        if not isinstance(row, dict):
            continue
        merged.append({"domain": "hormuz", **row})

    merged.sort(
        key=lambda row: (
            -_safe_int(row.get("risk_score", 0), 0),
            str(row.get("domain", "")),
            str(row.get("canonical_url", "")),
        )
    )

    return {
        "window_ticks": int(window_ticks),
        "count": max(0, _safe_int(github_result.get("count", 0), 0))
        + max(0, _safe_int(hormuz_result.get("count", 0), 0)),
        "critical_count": max(0, _safe_int(github_result.get("critical_count", 0), 0))
        + max(0, _safe_int(hormuz_result.get("critical_count", 0), 0)),
        "high_count": max(0, _safe_int(github_result.get("high_count", 0), 0))
        + max(0, _safe_int(hormuz_result.get("high_count", 0), 0)),
        "medium_count": max(0, _safe_int(github_result.get("medium_count", 0), 0))
        + max(0, _safe_int(hormuz_result.get("medium_count", 0), 0)),
        "low_count": max(0, _safe_int(github_result.get("low_count", 0), 0))
        + max(0, _safe_int(hormuz_result.get("low_count", 0), 0)),
        "domains": {
            "github": github_result,
            "hormuz": hormuz_result,
        },
        "threats": merged[:limit],
        "scoring": {
            "mode": "multi_domain",
            "github": github_result.get("scoring", {}),
            "hormuz": hormuz_result.get("scoring", {}),
        },
    }


def _query_web_resource_summary(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    args: dict[str, Any],
) -> dict[str, Any]:
    target = str(
        args.get(
            "target",
            args.get(
                "res_id",
                args.get("resource_id", args.get("url_id", args.get("url", ""))),
            ),
        )
        or ""
    ).strip()
    if not target:
        return {"target": "", "found": None, "link_count": 0, "links": []}

    node_by_id = _node_index(nodes)
    target_lower = target.lower()
    resource_node: dict[str, Any] | None = None

    for node in nodes:
        if _node_role(node) != "web:resource":
            continue
        node_id = str(node.get("id", "") or "").strip()
        canonical_url = str(node.get("canonical_url", "") or "").strip()
        if target_lower in {node_id.lower(), canonical_url.lower()}:
            resource_node = node
            break

    if resource_node is None:
        matched_url_id = ""
        for node in nodes:
            if _node_role(node) != "web:url":
                continue
            node_id = str(node.get("id", "") or "").strip()
            canonical_url = str(node.get("canonical_url", "") or "").strip()
            if target_lower in {node_id.lower(), canonical_url.lower()}:
                matched_url_id = node_id
                break
        if matched_url_id:
            for edge in edges:
                kind = str(edge.get("kind", "") or "").strip().lower()
                if kind != "web:source_of":
                    continue
                if str(edge.get("target", "") or "").strip() != matched_url_id:
                    continue
                source_id = str(edge.get("source", "") or "").strip()
                candidate = node_by_id.get(source_id)
                if (
                    isinstance(candidate, dict)
                    and _node_role(candidate) == "web:resource"
                ):
                    resource_node = candidate
                    break

    if not isinstance(resource_node, dict):
        return {"target": target, "found": None, "link_count": 0, "links": []}

    resource_id = str(resource_node.get("id", "") or "").strip()
    links: list[dict[str, Any]] = []
    for edge in edges:
        kind = str(edge.get("kind", "") or "").strip().lower()
        if kind != "web:links_to":
            continue
        source_id = str(edge.get("source", "") or "").strip()
        if source_id != resource_id:
            continue
        target_id = str(edge.get("target", "") or "").strip()
        target_node = node_by_id.get(target_id, {})
        links.append(
            {
                "url_id": target_id,
                "canonical_url": str(
                    target_node.get("canonical_url", "") or ""
                ).strip(),
                "last_status": str(target_node.get("last_status", "") or "").strip(),
                "next_allowed_fetch_ts": round(
                    max(
                        0.0,
                        _safe_float(target_node.get("next_allowed_fetch_ts", 0.0), 0.0),
                    ),
                    6,
                ),
            }
        )
    links.sort(
        key=lambda row: (str(row.get("canonical_url", "")), str(row.get("url_id", "")))
    )

    return {
        "target": target,
        "found": {
            "res_id": resource_id,
            "canonical_url": str(resource_node.get("canonical_url", "") or "").strip(),
            "fetched_ts": round(
                max(0.0, _safe_float(resource_node.get("fetched_ts", 0.0), 0.0)),
                6,
            ),
            "content_hash": str(resource_node.get("content_hash", "") or "").strip(),
            "summary": str(resource_node.get("summary", "") or "").strip(),
            "text_excerpt": str(resource_node.get("text_excerpt", "") or "").strip(),
            "conversation_comment_count": max(
                0,
                _safe_int(resource_node.get("conversation_comment_count", 0), 0),
            ),
            "commit_count": max(0, _safe_int(resource_node.get("commit_count", 0), 0)),
            "source_url_id": str(resource_node.get("source_url_id", "") or "").strip(),
        },
        "link_count": len(links),
        "links": links[:128],
    }


def _query_graph_summary(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    simulation: dict[str, Any] | None,
    args: dict[str, Any],
) -> dict[str, Any]:
    scope = str(args.get("scope", "all") or "all").strip().lower()
    top_n = max(1, min(64, _safe_int(args.get("n", 12), 12)))
    overview = _query_overview(nodes, edges)
    top_degree_nodes = list(overview.get("top_degree_nodes", []))[:top_n]

    filtered_nodes = nodes
    if scope in {"web", "crawler"}:
        filtered_nodes = [
            row
            for row in nodes
            if _node_role(row) in {"web:url", "web:resource"}
            or str(row.get("crawler_kind", "")).strip()
        ]
    elif scope in {"nexus", "graph"}:
        filtered_nodes = [row for row in nodes if str(row.get("id", "")).strip()]

    role_counts: dict[str, int] = {}
    for node in filtered_nodes:
        role = _node_role(node)
        role_counts[role] = role_counts.get(role, 0) + 1

    _, _, _, outcome_rows = _simulation_sections(simulation)
    outcome_counts = {"food": 0, "death": 0}
    for row in outcome_rows:
        outcome_kind = str(row.get("outcome", "") or "").strip().lower()
        if outcome_kind in outcome_counts:
            outcome_counts[outcome_kind] += 1

    return {
        "scope": scope,
        "node_count": len(filtered_nodes),
        "edge_count": len(edges),
        "role_counts": role_counts,
        "edge_kind_counts": overview.get("edge_kind_counts", {}),
        "top_nodes": top_degree_nodes,
        "outcomes": outcome_counts,
    }


def run_named_graph_query(
    nexus_graph: dict[str, Any] | None,
    query_name: str,
    *,
    args: dict[str, Any] | None = None,
    simulation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes, edges = _graph_rows(nexus_graph)
    query = normalize_graph_query_name(query_name)
    query_args = dict(args) if isinstance(args, dict) else {}

    threat_query_args = dict(query_args)
    if isinstance(simulation, dict):
        threat_query_args["_threat_compute_budget"] = (
            _threat_compute_budget_from_simulation(simulation)
        )

    handlers = {
        "overview": lambda: _query_overview(nodes, edges),
        "graph_summary": lambda: _query_graph_summary(
            nodes,
            edges,
            simulation,
            query_args,
        ),
        "neighbors": lambda: _query_neighbors(nodes, edges, query_args),
        "role_slice": lambda: _query_role_slice(nodes, query_args),
        "search": lambda: _query_search(nodes, query_args),
        "url_status": lambda: _query_url_status(nodes, query_args),
        "resource_for_url": lambda: _query_resource_for_url(nodes, edges, query_args),
        "recently_updated": lambda: _query_recently_updated(nodes, query_args),
        "explain_daimoi": lambda: _query_explain_daimoi(simulation, query_args),
        "recent_outcomes": lambda: _query_recent_outcomes(simulation, query_args),
        "crawler_status": lambda: _query_crawler_status(simulation),
        "arxiv_papers": lambda: _query_arxiv_papers(simulation, query_args),
        "github_status": lambda: _query_github_status(nodes, simulation),
        "github_repo_summary": lambda: _query_github_repo_summary(nodes, query_args),
        "github_find": lambda: _query_github_find(nodes, query_args),
        "github_recent_changes": lambda: _query_github_recent_changes(
            nodes,
            query_args,
        ),
        "proximity_radar": lambda: _query_proximity_radar(
            nodes,
            query_args,
        ),
        "entity_risk_state": lambda: _query_entity_risk_state(
            nodes,
            query_args,
        ),
        "cyber_regime_state": lambda: _query_cyber_regime_state(
            nodes,
            threat_query_args,
        ),
        "cyber_risk_radar": lambda: _query_cyber_risk_radar(
            nodes,
            threat_query_args,
        ),
        "github_threat_radar": lambda: _query_github_threat_radar(
            nodes,
            threat_query_args,
        ),
        "geopolitical_news_radar": lambda: _query_geopolitical_news_radar(
            nodes,
            edges,
            threat_query_args,
            simulation,
        ),
        "hormuz_threat_radar": lambda: _query_hormuz_threat_radar(
            nodes,
            threat_query_args,
        ),
        "multi_threat_radar": lambda: _query_multi_threat_radar(
            nodes,
            threat_query_args,
        ),
        "web_resource_summary": lambda: _query_web_resource_summary(
            nodes,
            edges,
            query_args,
        ),
    }

    if query in handlers:
        result = handlers[query]()
    else:
        result = build_unknown_graph_query_result(query)

    payload = {
        "record": "eta-mu.graph-query.v1",
        "schema_version": "graph.query.v1",
        "generated_at": _now_iso(),
        "query": query,
        "args": query_args,
        "result": result,
    }
    payload["snapshot_hash"] = _canonical_hash(payload)
    return payload


def _facts_counts(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    nodes_by_role: dict[str, int] = {}
    edges_by_kind: dict[str, int] = {}
    for node in nodes:
        role = _node_role(node)
        nodes_by_role[role] = nodes_by_role.get(role, 0) + 1
    for edge in edges:
        kind = str(edge.get("kind", "relates") or "relates").strip().lower()
        edges_by_kind[kind] = edges_by_kind.get(kind, 0) + 1
    return {
        "nodes_by_role": nodes_by_role,
        "edges_by_kind": edges_by_kind,
    }


def _facts_web_section(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, Any]:
    node_by_id = _node_index(nodes)
    inbound_links: dict[str, int] = {}
    source_of_count: dict[str, int] = {}
    links_to_count: dict[str, int] = {}

    for edge in edges:
        kind = str(edge.get("kind", "")).strip().lower()
        source_id = str(edge.get("source", "")).strip()
        target_id = str(edge.get("target", "")).strip()
        if kind == "web:links_to" and target_id:
            inbound_links[target_id] = inbound_links.get(target_id, 0) + 1
            links_to_count[source_id] = links_to_count.get(source_id, 0) + 1
        if kind == "web:source_of" and source_id:
            source_of_count[source_id] = source_of_count.get(source_id, 0) + 1

    now_epoch = time.time()
    urls: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []

    for node in nodes:
        role = _node_role(node)
        if role == "web:url":
            node_id = str(node.get("id", "")).strip()
            next_allowed = max(
                0.0, _safe_float(node.get("next_allowed_fetch_ts", 0.0), 0.0)
            )
            urls.append(
                {
                    "url_id": node_id,
                    "canonical_url": str(node.get("canonical_url", "") or "").strip(),
                    "next_allowed_fetch_ts": round(next_allowed, 6),
                    "cooldown_active": bool(next_allowed > now_epoch),
                    "fail_count": max(0, _safe_int(node.get("fail_count", 0), 0)),
                    "last_status": str(node.get("last_status", "") or "").strip(),
                    "last_fetch_ts": round(
                        max(0.0, _safe_float(node.get("last_fetch_ts", 0.0), 0.0)), 6
                    ),
                    "inbound_links": max(0, inbound_links.get(node_id, 0)),
                }
            )
        elif role == "web:resource":
            node_id = str(node.get("id", "")).strip()
            resources.append(
                {
                    "res_id": node_id,
                    "canonical_url": str(node.get("canonical_url", "") or "").strip(),
                    "fetched_ts": round(
                        max(0.0, _safe_float(node.get("fetched_ts", 0.0), 0.0)), 6
                    ),
                    "content_hash": str(node.get("content_hash", "") or "").strip(),
                    "link_count": max(0, links_to_count.get(node_id, 0)),
                    "source_count": max(0, source_of_count.get(node_id, 0)),
                }
            )

    urls.sort(
        key=lambda row: (
            -_safe_int(row.get("inbound_links", 0), 0),
            -_safe_float(row.get("last_fetch_ts", 0.0), 0.0),
            str(row.get("url_id", "")),
        )
    )
    resources.sort(
        key=lambda row: (
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            str(row.get("res_id", "")),
        )
    )

    return {
        "urls": urls[:64],
        "resources": resources[:64],
        "resource_lookup": {
            str(row.get("res_id", "")): dict(row)
            for row in resources
            if str(row.get("res_id", ""))
        },
        "node_lookup": node_by_id,
    }


def _facts_github_section(
    nodes: list[dict[str, Any]],
    simulation: dict[str, Any],
    *,
    invariants: list[dict[str, Any]],
) -> dict[str, Any]:
    resources = _github_resource_rows(nodes)

    crawler_graph = (
        simulation.get("crawler_graph", {}) if isinstance(simulation, dict) else {}
    )
    status = crawler_graph.get("status", {}) if isinstance(crawler_graph, dict) else {}
    github_status = status.get("github", {}) if isinstance(status, dict) else {}
    github_status = github_status if isinstance(github_status, dict) else {}

    monitored_repos = [
        str(item).strip()
        for item in github_status.get("monitored_repos", [])
        if str(item).strip()
    ]
    if not monitored_repos:
        monitored_repos = sorted(
            {
                str(row.get("repo", "")).strip()
                for row in resources
                if str(row.get("repo", "")).strip()
            }
        )

    atom_counts: dict[str, int] = {}
    atom_lookup: dict[str, dict[str, Any]] = {}
    for row in resources:
        for atom in (
            row.get("atoms", []) if isinstance(row.get("atoms", []), list) else []
        ):
            if not isinstance(atom, dict):
                continue
            token = _canonical_json(atom)
            atom_counts[token] = atom_counts.get(token, 0) + 1
            if token not in atom_lookup:
                atom_lookup[token] = atom

    top_atoms = sorted(
        atom_counts.items(),
        key=lambda item: (-_safe_int(item[1], 0), str(item[0])),
    )

    github_invariants = [
        row
        for row in invariants
        if isinstance(row, dict)
        and str(row.get("kind", "")).strip().lower().startswith("github_")
    ]

    return {
        "monitored_repos": monitored_repos,
        "recent_resources": [
            {
                "repo": row.get("repo", ""),
                "kind": row.get("kind", ""),
                "number": row.get("number", 0),
                "canonical_url": row.get("canonical_url", ""),
                "title": row.get("title", ""),
                "updated_at": row.get("updated_at", ""),
                "fetched_ts": row.get("fetched_ts", 0.0),
                "content_hash": row.get("content_hash", ""),
                "importance_score": row.get("importance_score", 0),
            }
            for row in resources[:24]
        ],
        "top_atoms": [
            {
                "atom": atom_lookup.get(token, {}),
                "count": count,
            }
            for token, count in top_atoms[:20]
        ],
        "invariants_violations": github_invariants,
    }


def _facts_recent_events(
    simulation: dict[str, Any], *, limit: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dynamics = (
        simulation.get("presence_dynamics", {}) if isinstance(simulation, dict) else {}
    )
    if isinstance(dynamics, dict):
        trails = dynamics.get("daimoi_outcome_trails", [])
        if isinstance(trails, list):
            for row in trails[-limit:]:
                if not isinstance(row, dict):
                    continue
                rows.append(
                    {
                        "id": f"daimoi:{_safe_int(row.get('seq', 0), 0)}",
                        "kind": "daimoi_outcome",
                        "ts": str(row.get("ts", "") or ""),
                        "outcome": str(row.get("outcome", "") or "").strip().lower(),
                        "target": str(row.get("graph_node_id", "") or "").strip(),
                        "reason": str(row.get("reason", "") or "").strip(),
                    }
                )

    crawler_graph = (
        simulation.get("crawler_graph", {}) if isinstance(simulation, dict) else {}
    )
    crawler_events = (
        crawler_graph.get("events", []) if isinstance(crawler_graph, dict) else []
    )
    if isinstance(crawler_events, list):
        for row in crawler_events[-limit:]:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "id": str(row.get("id", "") or "").strip(),
                    "kind": str(row.get("kind", "") or "").strip(),
                    "ts": str(row.get("ts", "") or ""),
                    "url_id": str(row.get("url_id", "") or "").strip(),
                    "reason": str(row.get("reason", "") or "").strip(),
                    "status": str(row.get("status", "") or "").strip(),
                }
            )

    rows.sort(
        key=lambda row: (
            str(row.get("ts", "")),
            str(row.get("kind", "")),
            str(row.get("id", "")),
        )
    )
    return rows[-limit:]


def _facts_invariants(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    node_by_id = _node_index(nodes)
    url_ids = {
        str(node.get("id", "")).strip()
        for node in nodes
        if _node_role(node) == "web:url" and str(node.get("id", "")).strip()
    }
    violations: list[dict[str, Any]] = []
    allowed_github_kinds = {
        "github:repo",
        "github:pr",
        "github:issue",
        "github:release",
        "github:advisory",
        "github:compare",
        "github:diff",
        "github:file",
    }

    source_edges_by_resource: dict[str, int] = {}
    for edge in edges:
        kind = str(edge.get("kind", "")).strip().lower()
        source_id = str(edge.get("source", "")).strip()
        target_id = str(edge.get("target", "")).strip()
        if kind == "web:source_of":
            source_edges_by_resource[source_id] = (
                source_edges_by_resource.get(source_id, 0) + 1
            )
            if target_id not in url_ids:
                violations.append(
                    {
                        "kind": "web_source_target_not_url",
                        "resource_id": source_id,
                        "target_id": target_id,
                    }
                )
        if kind == "web:links_to" and target_id not in url_ids:
            violations.append(
                {
                    "kind": "web_links_target_not_url",
                    "source_id": source_id,
                    "target_id": target_id,
                }
            )

    for node in nodes:
        role = _node_role(node)
        node_id = str(node.get("id", "")).strip()
        if role == "web:resource":
            source_count = source_edges_by_resource.get(node_id, 0)
            if source_count != 1:
                violations.append(
                    {
                        "kind": "resource_source_of_count_invalid",
                        "resource_id": node_id,
                        "count": source_count,
                    }
                )
            canonical_url = _node_canonical_url(node)
            kind_value = str(_node_value(node, "kind", "") or "").strip().lower()
            is_github_resource = _is_github_like_url(
                canonical_url
            ) or kind_value.startswith("github:")
            if is_github_resource and kind_value not in allowed_github_kinds:
                violations.append(
                    {
                        "kind": "github_resource_kind_invalid",
                        "resource_id": node_id,
                        "value": kind_value,
                    }
                )
        if role == "web:url":
            canonical_url = _node_canonical_url(node)
            if not canonical_url:
                violations.append(
                    {"kind": "url_missing_canonical_url", "url_id": node_id}
                )
            next_allowed = _safe_float(
                _node_value(node, "next_allowed_fetch_ts", 0.0), 0.0
            )
            fail_count = _safe_int(_node_value(node, "fail_count", 0), 0)
            if next_allowed < 0.0 or fail_count < 0:
                violations.append(
                    {
                        "kind": "url_cooldown_invalid",
                        "url_id": node_id,
                        "next_allowed_fetch_ts": round(next_allowed, 6),
                        "fail_count": fail_count,
                    }
                )

    # deterministic dedupe
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in sorted(
        violations,
        key=lambda item: _canonical_json(item if isinstance(item, dict) else {}),
    ):
        token = _canonical_json(row)
        if token in seen:
            continue
        seen.add(token)
        deduped.append(row)
    return deduped


def _facts_table_rows(
    simulation: dict[str, Any],
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    dynamics, _, field_particles, outcome_rows = _simulation_sections(simulation)
    node_by_id = _node_index(nodes)
    tick_hint = max(0, _safe_int(dynamics.get("tick", 0), 0))

    node_table = [
        {
            "node_id": str(node.get("id", "") or "").strip(),
            "role": _node_role(node),
            "status": str(node.get("status", "") or "").strip(),
            "label": str(node.get("label", "") or "").strip(),
            "canonical_url": str(node.get("canonical_url", "") or "").strip(),
            "importance": round(_safe_float(node.get("importance", 0.0), 0.0), 6),
        }
        for node in nodes
        if str(node.get("id", "") or "").strip()
    ]
    node_table.sort(key=lambda row: str(row.get("node_id", "")))

    edge_table = [
        {
            "src": str(edge.get("source", "") or "").strip(),
            "role": str(edge.get("kind", "") or "").strip().lower() or "relates",
            "dst": str(edge.get("target", "") or "").strip(),
            "weight": round(_safe_float(edge.get("weight", 0.0), 0.0), 6),
        }
        for edge in edges
        if str(edge.get("source", "") or "").strip()
        and str(edge.get("target", "") or "").strip()
    ]
    edge_table.sort(
        key=lambda row: (
            str(row.get("src", "")),
            str(row.get("role", "")),
            str(row.get("dst", "")),
        )
    )

    daimoi_table = [
        {
            "daimoi_id": str(row.get("id", "") or "").strip(),
            "owner": str(
                row.get(
                    "presence_id",
                    row.get("owner_presence_id", row.get("owner", "")),
                )
                or ""
            ).strip(),
            "intent": str(row.get("top_job", "") or "").strip(),
            "born_tick": 0,
            "ttl": max(0, _safe_int(row.get("ttl", row.get("ttl_steps", 0)), 0)),
            "age": max(0, _safe_int(row.get("age", 0), 0)),
            "graph_node_id": str(row.get("graph_node_id", "") or "").strip(),
            "route_node_id": str(row.get("route_node_id", "") or "").strip(),
            "message_probability": round(
                max(0.0, _safe_float(row.get("message_probability", 0.0), 0.0)),
                6,
            ),
        }
        for row in field_particles
        if str(row.get("id", "") or "").strip()
    ]
    daimoi_table.sort(key=lambda row: str(row.get("daimoi_id", "")))

    event_collision_table = []
    for row in field_particles:
        daimoi_id = str(row.get("id", "") or "").strip()
        if not daimoi_id:
            continue
        collisions = max(
            0,
            _safe_int(row.get("collision_count", row.get("collisions", 0)), 0),
        )
        if collisions <= 0:
            continue
        event_collision_table.append(
            {
                "tick": max(0, _safe_int(row.get("age", tick_hint), tick_hint)),
                "daimoi_id": daimoi_id,
                "target_id": str(row.get("graph_node_id", "") or "").strip(),
                "collision_count": collisions,
            }
        )
    event_collision_table.sort(
        key=lambda row: (
            _safe_int(row.get("tick", 0), 0),
            str(row.get("daimoi_id", "")),
        )
    )

    outcome_table: list[dict[str, Any]] = []
    for row in outcome_rows:
        outcome_kind = str(row.get("outcome", "") or "").strip().lower()
        if outcome_kind not in {"food", "death"}:
            continue
        outcome_table.append(
            {
                "tick": max(0, _safe_int(row.get("tick", row.get("seq", 0)), 0)),
                "daimoi_id": str(row.get("daimoi_id", "") or "").strip(),
                "target_id": str(row.get("graph_node_id", "") or "").strip(),
                "reason": str(row.get("reason", "") or "").strip(),
                "intensity": round(
                    max(0.0, _safe_float(row.get("intensity", 0.0), 0.0)),
                    6,
                ),
                "outcome": outcome_kind,
            }
        )
    outcome_table.sort(
        key=lambda row: (
            _safe_int(row.get("tick", 0), 0),
            str(row.get("daimoi_id", "")),
            str(row.get("outcome", "")),
        )
    )

    event_timeout_table = [
        {
            "tick": max(0, _safe_int(row.get("tick", 0), 0)),
            "daimoi_id": str(row.get("daimoi_id", "") or "").strip(),
            "reason": str(row.get("reason", "") or "").strip(),
        }
        for row in outcome_table
        if str(row.get("outcome", "")).strip() == "death"
        and any(
            token in str(row.get("reason", "")).strip().lower()
            for token in ("timeout", "ttl")
        )
    ]

    heartbeat = (
        dynamics.get("resource_heartbeat", {})
        if isinstance(dynamics.get("resource_heartbeat", {}), dict)
        else {}
    )
    devices = heartbeat.get("devices", {}) if isinstance(heartbeat, dict) else {}
    capacity_table = []
    if isinstance(devices, dict):
        for device_id, payload in devices.items():
            device_row = payload if isinstance(payload, dict) else {}
            utilization = max(
                0.0,
                min(
                    100.0,
                    _safe_float(
                        device_row.get(
                            "utilization", device_row.get("usage_percent", 0.0)
                        ),
                        0.0,
                    ),
                ),
            )
            capacity_table.append(
                {
                    "tick": tick_hint,
                    "target_id": f"device:{str(device_id).strip().lower()}",
                    "cap": 100.0,
                    "used": round(utilization, 6),
                }
            )
    capacity_table.sort(key=lambda row: str(row.get("target_id", "")))

    web_url_table = [
        {
            "url_id": str(node.get("id", "") or "").strip(),
            "canonical_url": str(node.get("canonical_url", "") or "").strip(),
            "next_allowed_fetch_ts": round(
                max(0.0, _safe_float(node.get("next_allowed_fetch_ts", 0.0), 0.0)),
                6,
            ),
            "last_fetch_ts": round(
                max(0.0, _safe_float(node.get("last_fetch_ts", 0.0), 0.0)),
                6,
            ),
            "fail_count": max(0, _safe_int(node.get("fail_count", 0), 0)),
            "last_status": str(node.get("last_status", "") or "").strip(),
        }
        for node in nodes
        if _node_role(node) == "web:url"
    ]
    web_url_table.sort(key=lambda row: str(row.get("url_id", "")))

    web_resource_table = [
        {
            "res_id": str(node.get("id", "") or "").strip(),
            "canonical_url": str(node.get("canonical_url", "") or "").strip(),
            "content_hash": str(node.get("content_hash", "") or "").strip(),
            "fetched_ts": round(
                max(0.0, _safe_float(node.get("fetched_ts", 0.0), 0.0)),
                6,
            ),
            "source_url_id": str(node.get("source_url_id", "") or "").strip(),
        }
        for node in nodes
        if _node_role(node) == "web:resource"
    ]
    web_resource_table.sort(key=lambda row: str(row.get("res_id", "")))

    web_link_table = []
    for edge in edges:
        if str(edge.get("kind", "") or "").strip().lower() != "web:links_to":
            continue
        source_id = str(edge.get("source", "") or "").strip()
        target_id = str(edge.get("target", "") or "").strip()
        if not source_id or not target_id:
            continue
        source_node = node_by_id.get(source_id, {})
        target_node = node_by_id.get(target_id, {})
        if (
            _node_role(source_node) != "web:resource"
            or _node_role(target_node) != "web:url"
        ):
            continue
        web_link_table.append({"res_id": source_id, "url_id": target_id})
    web_link_table.sort(
        key=lambda row: (str(row.get("res_id", "")), str(row.get("url_id", "")))
    )

    food_table = [
        {
            "tick": max(0, _safe_int(row.get("tick", 0), 0)),
            "daimoi_id": str(row.get("daimoi_id", "") or "").strip(),
            "target_id": str(row.get("target_id", "") or "").strip(),
        }
        for row in outcome_table
        if str(row.get("outcome", "")).strip() == "food"
    ]
    death_table = [
        {
            "tick": max(0, _safe_int(row.get("tick", 0), 0)),
            "daimoi_id": str(row.get("daimoi_id", "") or "").strip(),
        }
        for row in outcome_table
        if str(row.get("outcome", "")).strip() == "death"
    ]

    return {
        "node": node_table,
        "edge": edge_table,
        "daimoi": daimoi_table,
        "event_collision": event_collision_table,
        "event_timeout": event_timeout_table,
        "capacity": capacity_table,
        "web_url": web_url_table,
        "web_resource": web_resource_table,
        "web_link": web_link_table,
        "food": food_table,
        "death": death_table,
    }


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        _canonical_json(row if isinstance(row, dict) else {})
        for row in rows
        if isinstance(row, dict)
    ]
    path.write_text(
        ("\n".join(lines) + "\n") if lines else "",
        "utf-8",
    )


def build_facts_snapshot(
    simulation: dict[str, Any] | None,
    *,
    part_root: Path | str | None = None,
) -> dict[str, Any]:
    payload = simulation if isinstance(simulation, dict) else {}
    nexus_graph = (
        payload.get("nexus_graph", {})
        if isinstance(payload.get("nexus_graph", {}), dict)
        else {}
    )
    nodes, edges = _graph_rows(nexus_graph)
    counts = _facts_counts(nodes, edges)
    web_section = _facts_web_section(nodes, edges)
    invariants = _facts_invariants(nodes, edges)
    github_section = _facts_github_section(nodes, payload, invariants=invariants)
    recent_events = _facts_recent_events(payload, limit=48)

    dynamics = (
        payload.get("presence_dynamics", {})
        if isinstance(payload.get("presence_dynamics", {}), dict)
        else {}
    )
    outcomes = (
        dynamics.get("daimoi_outcome_summary", {})
        if isinstance(dynamics.get("daimoi_outcome_summary", {}), dict)
        else {}
    )
    tables = _facts_table_rows(payload, nodes=nodes, edges=edges)

    facts: dict[str, Any] = {
        "record": "eta-mu.facts-snapshot.v1",
        "schema_version": "facts.snapshot.v1",
        "ts": _now_iso(),
        "tick": max(0, _safe_int(dynamics.get("tick", 0), 0)),
        "counts": counts,
        "recent_events": recent_events,
        "web": {
            "urls": web_section.get("urls", []),
            "resources": web_section.get("resources", []),
        },
        "github": github_section,
        "dynamics": {
            "daimoi_outcomes": {
                "food": max(0, _safe_int(outcomes.get("food", 0), 0)),
                "death": max(0, _safe_int(outcomes.get("death", 0), 0)),
                "total": max(0, _safe_int(outcomes.get("total", 0), 0)),
            }
        },
        "tables": {
            table_name: {
                "row_count": len(table_rows),
                "path": "",
            }
            for table_name, table_rows in tables.items()
        },
        "invariants_violations": invariants,
    }

    snapshot_hash = _canonical_hash(facts)
    facts["snapshot_hash"] = snapshot_hash

    snapshot_path = ""
    if part_root is not None:
        base = Path(part_root)
        ts_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = base / "world_state" / "facts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ts_token}_{snapshot_hash[:12]}.json"
        out_path.write_text(
            json.dumps(facts, ensure_ascii=False, sort_keys=True, indent=2), "utf-8"
        )
        snapshot_path = str(out_path)
        table_dir = out_dir / "current"
        table_dir.mkdir(parents=True, exist_ok=True)
        table_meta = facts.get("tables", {})
        if isinstance(table_meta, dict):
            for table_name, table_rows in tables.items():
                table_path = table_dir / f"{table_name}.jsonl"
                _write_jsonl_rows(table_path, table_rows)
                row_meta = table_meta.get(table_name)
                if isinstance(row_meta, dict):
                    row_meta["path"] = str(table_path)
    facts["snapshot_path"] = snapshot_path
    return facts
