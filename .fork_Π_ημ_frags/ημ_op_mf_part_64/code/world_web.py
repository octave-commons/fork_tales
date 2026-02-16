from __future__ import annotations

import argparse
import base64
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
import wave
import webbrowser
from array import array
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha1
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, quote, urlparse, unquote
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
MAX_SIM_POINTS = 512
SIM_TICK_SECONDS = float(os.getenv("SIM_TICK_SECONDS", "0.2") or "0.2")
CATALOG_REFRESH_SECONDS = float(os.getenv("CATALOG_REFRESH_SECONDS", "1.5") or "1.5")
CATALOG_BROADCAST_HEARTBEAT_SECONDS = float(
    os.getenv("CATALOG_BROADCAST_HEARTBEAT_SECONDS", "6.0") or "6.0"
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
        "title": "Presence Music Command Center",
        "binds_to": ["/api/catalog", "/api/task/queue"],
        "field_bindings": {"f1": 0.27, "f3": 0.2, "f6": 0.31, "f8": 0.22},
        "presence": "anchor_registry",
        "tags": ["music", "command", "council"],
        "lane": "council",
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


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


class RuntimeInfluenceTracker:
    CLICK_WINDOW_SECONDS = 45.0
    FILE_WINDOW_SECONDS = 120.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._click_events: list[dict[str, Any]] = []
        self._file_events: list[dict[str, Any]] = []
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

    def snapshot(self, queue_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        now = time.time()
        queue_snapshot = queue_snapshot or {}
        with self._lock:
            self._prune(now)
            click_rows = list(self._click_events)
            file_rows = list(self._file_events)
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

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "clicks_45s": clicks_recent,
            "file_changes_120s": file_changes_recent,
            "recent_click_targets": recent_targets,
            "recent_file_paths": recent_file_paths,
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
    if suffix in {".md", ".txt", ".json", ".jsonl", ".ndjson"}:
        return "text"
    return "file"


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

    candidate = (vault_root / relative).resolve()
    vault_resolved = vault_root.resolve()
    if candidate == vault_resolved or vault_resolved in candidate.parents:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def collect_catalog(part_root: Path, vault_root: Path) -> dict[str, Any]:
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

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "part_roots": [str(p) for p in discover_part_roots(vault_root, part_root)],
        "counts": dict(counts),
        "canonical_terms": [{"en": en, "ja": ja} for en, ja in CANONICAL_TERMS],
        "entity_manifest": ENTITY_MANIFEST,
        "named_fields": build_named_field_overlays(ENTITY_MANIFEST),
        "ui_default_perspective": PROJECTION_DEFAULT_PERSPECTIVE,
        "ui_perspectives": projection_perspective_options(),
        "cover_fields": cover_fields,
        "promptdb": promptdb_index,
        "items": items,
    }


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

    asks: list[str] = []
    if "?" in clean_text:
        asks.append(clean_text)
    if "why" in tokens:
        asks.append("Why is this state true in the current catalog?")

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

    facts = [
        f"presence_id={presence_id}",
        f"catalog_items={len(catalog.get('items', []))}",
        f"promptdb_packets={int(catalog.get('promptdb', {}).get('packet_count', 0))}",
    ]

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
    artifact_hashes = [
        {"path": str(path), "sha256": _sha256_for_path(path)}
        for path in artifact_paths
        if path.exists() and path.is_file()
    ]
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
    points: list[dict[str, float]] = []
    items = catalog.get("items", [])

    for idx, item in enumerate(items[:MAX_SIM_POINTS]):
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

    river_flow_rate = round(
        1.2 + (audio_ratio * 4.4) + (file_ratio * 7.2) + (click_ratio * 2.6), 3
    )
    river_turbulence = round(_clamp01((file_ratio * 0.72) + (click_ratio * 0.4)), 4)

    manifest_lookup = {
        str(item.get("id", "")): item for item in ENTITY_MANIFEST if item.get("id")
    }
    impact_order = [*CANONICAL_NAMED_FIELD_IDS, FILE_SENTINEL_PROFILE["id"]]
    base_file = {
        "receipt_river": 0.94,
        "witness_thread": 0.38,
        "fork_tax_canticle": 0.84,
        "mage_of_receipts": 0.88,
        "keeper_of_receipts": 0.9,
        "anchor_registry": 0.64,
        "gates_of_truth": 0.73,
        "file_sentinel": 1.0,
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
    }

    presence_impacts: list[dict[str, Any]] = []
    for presence_id in impact_order:
        if presence_id == FILE_SENTINEL_PROFILE["id"]:
            meta = FILE_SENTINEL_PROFILE
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
        total_influence = _clamp01((file_influence * 0.63) + (click_influence * 0.37))
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
        elif presence_id == "fork_tax_canticle":
            notes_en = "Fork tax pressure rises with unresolved file drift."
            notes_ja = "未解決のファイルドリフトでフォーク税圧は上がる。"
        elif presence_id == "witness_thread":
            notes_en = "Mouse touches tighten witness linkage across presences."
            notes_ja = "マウスの接触はプレゼンス間の証人連結を強める。"
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
        "click_events": clicks_recent,
        "file_events": file_changes_recent,
        "recent_click_targets": list(influence.get("recent_click_targets", []))[:6],
        "recent_file_paths": list(influence.get("recent_file_paths", []))[:8],
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
        "entities": entity_states,
        "echoes": echo_particles,
        "presence_dynamics": presence_dynamics,
        "myth": myth_summary or {},
        "world": world_summary or {},
    }


def _ollama_endpoint() -> tuple[str, str, str, float]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    endpoint = (
        base_url if base_url.endswith("/api/generate") else f"{base_url}/api/generate"
    )
    embeddings_endpoint = (
        base_url
        if base_url.endswith("/api/embeddings")
        else f"{base_url}/api/embeddings"
    )
    model = os.getenv("OLLAMA_MODEL", "qwen3-vl:4b-instruct")
    timeout_s = float(os.getenv("OLLAMA_TIMEOUT_SEC", "30"))
    return endpoint, embeddings_endpoint, model, timeout_s


def _ollama_embed(text: str, model: str | None = None) -> list[float] | None:
    _, endpoint, default_model, default_timeout = _ollama_endpoint()
    chosen_model = model or default_model or "nomic-embed-text"

    payload = {
        "model": chosen_model,
        "prompt": text,
    }

    req = Request(
        endpoint,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )

    try:
        with urlopen(req, timeout=default_timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
            embedding = raw.get("embedding")
            if isinstance(embedding, list):
                return embedding
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        pass

    return None


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


def _ollama_generate_text(
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
    return {
        str(item.get("id", "")): item
        for item in ENTITY_MANIFEST
        if item.get("id") and item.get("id") != "core_pulse"
    }


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

    if presence_ids:
        for item in presence_ids:
            key = str(item).strip()
            if key in lookup:
                selected.append(lookup[key])
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
    ]
    for token, key in hints:
        if token in normalized and key in lookup and lookup[key] not in selected:
            selected.append(lookup[key])

    if not selected:
        fallback_order = ["receipt_river", "witness_thread", "gates_of_truth"]
        for key in fallback_order:
            if key in lookup:
                selected.append(lookup[key])

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
    task_queue = TaskQueue(
        queue_log_path,
        receipts_path,
        owner="Err",
        host=f"{host}:{port}",
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
            cached_catalog["presence_runtime"] = _INFLUENCE_TRACKER.snapshot(
                queue_snapshot=initial_queue_snapshot
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
                        refreshed_catalog["presence_runtime"] = (
                            _INFLUENCE_TRACKER.snapshot(
                                queue_snapshot=refreshed_queue_snapshot
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
                        queue_snapshot=current_queue_snapshot
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

                # Proxy to TTS Sidecar (Tier 0/1)
                tts_url = f"http://127.0.0.1:8788/tts?text={quote(text)}&speed={speed}"
                try:
                    with urlopen(tts_url, timeout=30) as resp:
                        self._send_bytes(resp.read(), "audio/wav")
                except Exception as e:
                    self._send_bytes(
                        json.dumps({"ok": False, "error": str(e)}).encode("utf-8"),
                        "application/json",
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )
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
                influence_snapshot = _INFLUENCE_TRACKER.snapshot(
                    queue_snapshot=queue_snapshot
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
                influence_snapshot = _INFLUENCE_TRACKER.snapshot(
                    queue_snapshot=queue_snapshot
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
                myth_summary = _MYTH_TRACKER.snapshot(catalog)
                world_summary = _LIFE_TRACKER.snapshot(
                    catalog, myth_summary, ENTITY_MANIFEST
                )
                influence_snapshot = _INFLUENCE_TRACKER.snapshot(
                    queue_snapshot=queue_snapshot
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

            if parsed.path == "/api/input-stream":
                req = self._read_json_body() or {}
                stream_type = str(req.get("type", "unknown"))
                data = req.get("data", {})

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

                body = json.dumps(
                    {
                        "ok": True,
                        "force": force_vector,
                        "embedding_dim": len(embedding) if embedding else 0,
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
                    queue_snapshot=queue_snapshot
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
