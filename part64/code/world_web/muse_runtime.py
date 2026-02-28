from __future__ import annotations

import json
import math
import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any


MUSE_RUNTIME_RECORD = "eta-mu.muse-runtime.snapshot.v1"
MUSE_RUNTIME_SCHEMA_VERSION = "muse.runtime.v1"
MUSE_EVENT_RECORD = "eta-mu.muse-event.v1"
MUSE_EVENT_SCHEMA_VERSION = "muse.events.v1"
MUSE_CONTEXT_MANIFEST_RECORD = "eta-mu.muse-context-manifest.v1"
MUSE_CONTEXT_MANIFEST_SCHEMA_VERSION = "muse.context-manifest.v1"
MUSE_RESOURCE_NODE_RECORD = "eta-mu.resource-node.v1"
MUSE_DAIMON_RECORD = "eta-mu.muse-daimon.v1"
MUSE_GPU_CLAIM_RECORD = "eta-mu.muse-gpu-claim.v1"
MUSE_STATE_FILE_RECORD = "eta-mu.muse-runtime-state.v1"

DEFAULT_MUSE_ID = "witness_thread"
DEFAULT_MUSE_LABEL = "Witness Thread"
BOOTSTRAP_MUSE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "witness_thread",
        "label": "Witness Thread",
        "anchor": {"x": 0.5, "y": 0.5, "zoom": 1.0, "kind": "bootstrap"},
    },
    {
        "id": "chaos",
        "label": "Chaos",
        "anchor": {"x": 0.18, "y": 0.23, "zoom": 1.0, "kind": "bootstrap"},
    },
    {
        "id": "stability",
        "label": "Stability",
        "anchor": {"x": 0.5, "y": 0.22, "zoom": 1.0, "kind": "bootstrap"},
    },
    {
        "id": "symmetry",
        "label": "Symmetry",
        "anchor": {"x": 0.82, "y": 0.23, "zoom": 1.0, "kind": "bootstrap"},
    },
)
DEFAULT_AUDIO_INTENT_ENABLED = True
DEFAULT_AUDIO_MIN_SCORE = 0.55
DEFAULT_AUDIO_DAIMOI_FANOUT = 3
DEFAULT_AUDIO_MAX_CANDIDATES = 32
DEFAULT_AUDIO_TARGET_PRESENCE_ID = "receipt_river"
DEFAULT_IMAGE_TARGET_PRESENCE_ID = "mage_of_receipts"
_AUDIO_SUFFIXES = (".mp3", ".wav", ".ogg", ".m4a", ".flac")
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg")
_AUDIO_ACTION_TOKENS = {
    "play",
    "queue",
    "spin",
    "hear",
    "listen",
    "sing",
    "start",
}
_IMAGE_ACTION_TOKENS = {
    "open",
    "view",
    "show",
    "display",
    "render",
    "load",
}
_AUDIO_HINT_TOKENS = {
    "song",
    "track",
    "music",
    "audio",
    "mp3",
    "wav",
    "playlist",
    "beat",
    "anthem",
}
_IMAGE_HINT_TOKENS = {
    "image",
    "photo",
    "picture",
    "art",
    "cover",
    "png",
    "jpg",
    "jpeg",
    "webp",
    "gif",
    "svg",
}
_AUDIO_STOP_TOKENS = {
    "a",
    "an",
    "the",
    "to",
    "for",
    "please",
    "now",
    "with",
    "from",
    "my",
    "that",
    "this",
    "use",
    "song",
    "track",
    "music",
    "audio",
    "play",
    "queue",
    "listen",
    "hear",
}
_IMAGE_STOP_TOKENS = {
    "a",
    "an",
    "the",
    "to",
    "for",
    "please",
    "now",
    "with",
    "from",
    "my",
    "that",
    "this",
    "use",
    "image",
    "photo",
    "picture",
    "open",
    "view",
    "show",
    "display",
}
DEFAULT_TURNS_PER_MINUTE = 24
DEFAULT_MAX_PARTICLES_PER_TURN = 18
DEFAULT_MAX_EVENTS = 2400
DEFAULT_MAX_HISTORY_MESSAGES = 320
DEFAULT_MAX_MANIFESTS_PER_MUSE = 128


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


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _hash_unit(value: str) -> float:
    token = sha1(value.encode("utf-8")).hexdigest()[:10]
    raw = int(token, 16)
    return raw / float(0xFFFFFFFFFF)


def _slugify(text: str, *, fallback: str = "muse") -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        raw = fallback
    raw = raw.replace(" ", "_")
    raw = re.sub(r"[^a-z0-9_]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    if not raw:
        return fallback
    if raw[0].isdigit():
        raw = f"muse_{raw}"
    return raw


def _token_cost(text: str) -> int:
    clean = str(text or "")
    if not clean:
        return 8
    return max(8, int(math.ceil(len(clean) / 4.0)))


def _normalize_visibility(raw: Any) -> str:
    text = str(raw or "public").strip().lower()
    if text in {"public", "private", "sealed"}:
        return text
    return "public"


def _seed_to_int(seed: str) -> int:
    return int(sha1(seed.encode("utf-8")).hexdigest()[:16], 16)


def _normalized_anchor(raw: Any) -> dict[str, Any]:
    payload = raw if isinstance(raw, dict) else {}
    return {
        "x": round(_clamp01(_safe_float(payload.get("x", 0.5), 0.5)), 6),
        "y": round(_clamp01(_safe_float(payload.get("y", 0.5), 0.5)), 6),
        "zoom": max(0.05, _safe_float(payload.get("zoom", 1.0), 1.0)),
        "kind": str(payload.get("kind", "viewport") or "viewport").strip()
        or "viewport",
    }


def _default_state() -> dict[str, Any]:
    return {
        "record": MUSE_STATE_FILE_RECORD,
        "schema_version": MUSE_RUNTIME_SCHEMA_VERSION,
        "seq": 0,
        "muses": {},
        "messages": {},
        "manifests": {},
        "events": [],
        "idempotency": {},
        "rate_windows": {},
    }


class InMemoryMuseStorage:
    backend_name = "memory"

    def __init__(self) -> None:
        self._state = _default_state()
        self._lock = threading.Lock()

    def load_state(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state, ensure_ascii=False))

    def save_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._state = json.loads(json.dumps(state, ensure_ascii=False))

    def reset(self) -> None:
        with self._lock:
            self._state = _default_state()


class JsonFileMuseStorage:
    backend_name = "json"

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def load_state(self) -> dict[str, Any]:
        with self._lock:
            if not self._path.exists() or not self._path.is_file():
                return _default_state()
            try:
                text = self._path.read_text("utf-8")
                payload = json.loads(text)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                return _default_state()
            return _default_state()

    def save_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(".tmp")
            tmp_path.write_text(
                json.dumps(state, ensure_ascii=False, separators=(",", ":")),
                "utf-8",
            )
            tmp_path.replace(self._path)

    def reset(self) -> None:
        with self._lock:
            if self._path.exists():
                self._path.unlink()


class MuseRuntimeManager:
    def __init__(
        self,
        *,
        storage: Any,
        enabled: bool,
        backend_name: str,
        turns_per_minute: int,
        max_particles_per_turn: int,
        max_events: int,
        max_history_messages: int,
        max_manifests_per_muse: int,
        audio_intent_enabled: bool,
        audio_min_score: float,
        audio_daimoi_fanout: int,
        audio_max_candidates: int,
        audio_target_presence_id: str,
        image_target_presence_id: str,
    ) -> None:
        self._storage = storage
        self._enabled = bool(enabled and storage is not None)
        self._backend_name = str(backend_name or "memory")
        self._turns_per_minute = max(1, int(turns_per_minute))
        self._max_particles_per_turn = max(1, int(max_particles_per_turn))
        self._max_events = max(128, int(max_events))
        self._max_history_messages = max(24, int(max_history_messages))
        self._max_manifests_per_muse = max(8, int(max_manifests_per_muse))
        self._audio_intent_enabled = bool(audio_intent_enabled)
        self._audio_min_score = max(0.01, min(2.0, float(audio_min_score)))
        self._audio_daimoi_fanout = max(1, min(12, int(audio_daimoi_fanout)))
        self._audio_max_candidates = max(4, min(96, int(audio_max_candidates)))
        self._audio_target_presence_id = (
            str(audio_target_presence_id or DEFAULT_AUDIO_TARGET_PRESENCE_ID).strip()
            or DEFAULT_AUDIO_TARGET_PRESENCE_ID
        )
        self._image_target_presence_id = (
            str(image_target_presence_id or DEFAULT_IMAGE_TARGET_PRESENCE_ID).strip()
            or DEFAULT_IMAGE_TARGET_PRESENCE_ID
        )
        self._lock = threading.Lock()
        self._state = self._coerce_state(storage.load_state() if self._enabled else {})
        self._ensure_bootstrap_muses_locked()

    def reset(self) -> None:
        with self._lock:
            self._state = _default_state()
            self._ensure_bootstrap_muses_locked()
            if self._enabled and self._storage is not None:
                self._storage.save_state(self._state)

    def list_muses(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                self._public_muse_row(muse) for muse in self._state["muses"].values()
            ]
            rows.sort(key=lambda row: str(row.get("created_at", "")))
            return rows

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = [
                self._public_muse_row(muse)
                for muse in self._state["muses"].values()
                if isinstance(muse, dict)
            ]
            rows.sort(key=lambda row: str(row.get("created_at", "")))
            return {
                "record": MUSE_RUNTIME_RECORD,
                "schema_version": MUSE_RUNTIME_SCHEMA_VERSION,
                "enabled": self._enabled,
                "backend": self._backend_name,
                "generated_at": _now_iso(),
                "muse_count": len(self._state["muses"]),
                "event_seq": _safe_int(self._state.get("seq", 0), 0),
                "muses": rows,
            }

    def create_muse(
        self,
        *,
        muse_id: str,
        label: str,
        anchor: dict[str, Any] | None,
        user_intent_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            payload = self._create_muse_locked(
                muse_id=muse_id,
                label=label,
                anchor=anchor,
                user_intent_id=user_intent_id,
            )
            self._persist_locked()
            return payload

    def set_pause(
        self,
        muse_id: str,
        *,
        paused: bool,
        reason: str,
        user_intent_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            muse = self._state["muses"].get(str(muse_id).strip())
            if muse is None:
                return {"ok": False, "error": "muse_not_found"}
            muse["status"] = "paused" if paused else "active"
            kind = "muse.paused" if paused else "muse.resumed"
            self._emit_event_locked(
                kind=kind,
                status="ok",
                muse_id=str(muse_id),
                payload={
                    "reason": str(reason or ""),
                    "user_intent_id": str(user_intent_id or ""),
                },
            )
            self._persist_locked()
            return {"ok": True, "muse": self._public_muse_row(muse)}

    def pin_node(
        self,
        muse_id: str,
        *,
        node_id: str,
        user_intent_id: str,
        reason: str,
    ) -> dict[str, Any]:
        with self._lock:
            result = self._pin_node_locked(
                muse_id,
                node_id=node_id,
                user_intent_id=user_intent_id,
                reason=reason,
            )
            if bool(result.get("ok", False)):
                self._persist_locked()
            return result

    def unpin_node(
        self,
        muse_id: str,
        *,
        node_id: str,
        user_intent_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            result = self._unpin_node_locked(
                muse_id,
                node_id=node_id,
                user_intent_id=user_intent_id,
            )
            if bool(result.get("ok", False)):
                self._persist_locked()
            return result

    def bind_nexus(
        self,
        muse_id: str,
        *,
        nexus_id: str,
        user_intent_id: str,
        reason: str,
    ) -> dict[str, Any]:
        with self._lock:
            muse = self._state["muses"].get(str(muse_id).strip())
            clean_nexus = str(nexus_id or "").strip()
            if muse is None:
                return {"ok": False, "error": "muse_not_found"}
            if not clean_nexus:
                return {"ok": False, "error": "invalid_nexus_id"}
            muse["active_nexus_id"] = clean_nexus
            self._emit_event_locked(
                kind="muse.bind.nexus",
                status="ok",
                muse_id=str(muse_id),
                payload={
                    "nexus_id": clean_nexus,
                    "reason": str(reason or ""),
                    "user_intent_id": str(user_intent_id or ""),
                },
            )
            self._persist_locked()
            return {
                "ok": True,
                "muse": self._public_muse_row(muse),
            }

    def sync_workspace_pins(
        self,
        muse_id: str,
        *,
        pinned_node_ids: list[str],
        user_intent_id: str,
        reason: str,
    ) -> dict[str, Any]:
        with self._lock:
            muse = self._state["muses"].get(str(muse_id).strip())
            if muse is None:
                return {"ok": False, "error": "muse_not_found"}
            next_ids = [
                str(item).strip()
                for item in (
                    pinned_node_ids if isinstance(pinned_node_ids, list) else []
                )
                if str(item).strip()
            ]
            deduped_next: list[str] = []
            for item in next_ids:
                if item not in deduped_next:
                    deduped_next.append(item)

            current = [
                str(item) for item in muse.get("pinned_node_ids", []) if str(item)
            ]
            current_set = set(current)
            next_set = set(deduped_next)

            added = sorted(next_set - current_set)
            removed = sorted(current_set - next_set)
            for node_id in added:
                self._pin_node_locked(
                    muse_id,
                    node_id=node_id,
                    user_intent_id=user_intent_id,
                    reason=reason,
                )
            for node_id in removed:
                self._unpin_node_locked(
                    muse_id,
                    node_id=node_id,
                    user_intent_id=user_intent_id,
                )
            muse["pinned_node_ids"] = deduped_next[:48]
            self._persist_locked()
            return {
                "ok": True,
                "muse": self._public_muse_row(muse),
                "added": added,
                "removed": removed,
            }

    def get_context_manifest(self, muse_id: str, turn_id: str) -> dict[str, Any] | None:
        with self._lock:
            manifests = self._state["manifests"].get(str(muse_id).strip(), {})
            if not isinstance(manifests, dict):
                return None
            payload = manifests.get(str(turn_id).strip())
            if not isinstance(payload, dict):
                return None
            return json.loads(json.dumps(payload, ensure_ascii=False))

    def list_events(
        self,
        *,
        muse_id: str = "",
        since_seq: int = 0,
        limit: int = 64,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._state.get("events", [])
            if not isinstance(rows, list):
                return []
            clean_id = str(muse_id or "").strip()
            clean_seq = max(0, int(since_seq))
            clean_limit = max(1, min(512, int(limit)))
            filtered: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if clean_id and str(row.get("muse_id", "")).strip() != clean_id:
                    continue
                if _safe_int(row.get("seq", 0), 0) <= clean_seq:
                    continue
                filtered.append(row)
            return [
                json.loads(json.dumps(item, ensure_ascii=False))
                for item in filtered[-clean_limit:]
            ]

    def send_message(
        self,
        *,
        muse_id: str,
        text: str,
        mode: str,
        token_budget: int,
        idempotency_key: str,
        graph_revision: str,
        surrounding_nodes: list[dict[str, Any]],
        tool_callback: Any | None,
        reply_builder: Any | None,
        seed: str,
    ) -> dict[str, Any]:
        with self._lock:
            clean_muse_id = str(muse_id or "").strip()
            muse = self._state["muses"].get(clean_muse_id)
            if muse is None:
                return {"ok": False, "error": "muse_not_found", "status_code": 404}

            clean_text = str(text or "").strip()
            if not clean_text:
                return {
                    "ok": False,
                    "error": "empty_message",
                    "status_code": 400,
                }

            if str(muse.get("status", "active")) == "paused":
                self._emit_event_locked(
                    kind="muse.rejected",
                    status="blocked",
                    muse_id=clean_muse_id,
                    payload={
                        "reason": "paused",
                    },
                )
                self._persist_locked()
                return {
                    "ok": False,
                    "error": "muse_paused",
                    "status_code": 409,
                }

            now_ms = _now_ms()
            window_rows = self._state["rate_windows"].setdefault(clean_muse_id, [])
            if not isinstance(window_rows, list):
                window_rows = []
            horizon = now_ms - 60000
            fresh_window = [
                _safe_int(item, 0)
                for item in window_rows
                if _safe_int(item, 0) >= horizon
            ]
            if len(fresh_window) >= self._turns_per_minute:
                self._state["rate_windows"][clean_muse_id] = fresh_window
                self._emit_event_locked(
                    kind="muse.rate_limited",
                    status="blocked",
                    muse_id=clean_muse_id,
                    payload={
                        "limit": self._turns_per_minute,
                        "window_seconds": 60,
                    },
                )
                self._persist_locked()
                return {
                    "ok": False,
                    "error": "rate_limited",
                    "status_code": 429,
                }

            dedupe_map = self._state["idempotency"].setdefault(clean_muse_id, {})
            if not isinstance(dedupe_map, dict):
                dedupe_map = {}
            clean_key = str(idempotency_key or "").strip()
            if clean_key and clean_key in dedupe_map:
                cached = dedupe_map.get(clean_key)
                if isinstance(cached, dict):
                    self._emit_event_locked(
                        kind="muse.message.deduped",
                        status="ok",
                        muse_id=clean_muse_id,
                        payload={
                            "idempotency_key": clean_key,
                            "turn_id": str(cached.get("turn_id", "")),
                        },
                    )
                    self._persist_locked()
                    return json.loads(json.dumps(cached, ensure_ascii=False))

            turn_id = f"turn:{sha1(f'{clean_muse_id}|{now_ms}|{clean_text}'.encode('utf-8')).hexdigest()[:16]}"
            message_node = self._append_message_node_locked(
                muse_id=clean_muse_id,
                turn_id=turn_id,
                role="user",
                text=clean_text,
            )
            self._emit_event_locked(
                kind="muse.message.received",
                status="ok",
                muse_id=clean_muse_id,
                turn_id=turn_id,
                payload={
                    "message_node_id": str(message_node.get("id", "")),
                },
            )
            self._emit_event_locked(
                kind="resource.message.created",
                status="ok",
                muse_id=clean_muse_id,
                turn_id=turn_id,
                payload={
                    "resource_node_id": str(message_node.get("id", "")),
                    "role": "user",
                },
            )

            clean_mode = str(mode or "stochastic").strip().lower()
            if clean_mode not in {"deterministic", "stochastic"}:
                clean_mode = "stochastic"
            safe_budget = max(320, min(8192, int(token_budget)))
            resolved_seed = str(seed or "").strip()
            if not resolved_seed:
                resolved_seed = sha1(
                    f"{clean_muse_id}|{turn_id}|{clean_text}".encode("utf-8")
                ).hexdigest()[:16]
            rng = random.Random(_seed_to_int(resolved_seed))

            manifest = self._assemble_context_manifest_locked(
                muse=muse,
                turn_id=turn_id,
                graph_revision=graph_revision,
                mode=clean_mode,
                token_budget=safe_budget,
                seed=resolved_seed,
                surrounding_nodes=surrounding_nodes,
            )
            self._emit_event_locked(
                kind="muse.context.assembled",
                status="ok",
                muse_id=clean_muse_id,
                turn_id=turn_id,
                payload={
                    "manifest_id": str(manifest.get("id", "")),
                    "explicit_count": len(manifest.get("explicit_selected", [])),
                    "surround_count": len(manifest.get("surround_selected", [])),
                },
            )
            self._emit_event_locked(
                kind="muse.tet.compiled",
                status="ok",
                muse_id=clean_muse_id,
                turn_id=turn_id,
                payload={
                    "tet_count": len(manifest.get("tet_units", [])),
                },
            )

            tool_rows = self._run_tool_requests_locked(
                clean_text,
                muse_id=clean_muse_id,
                turn_id=turn_id,
                tool_callback=tool_callback,
            )
            media_action = self._resolve_media_command_action_locked(
                text=clean_text,
                muse=muse,
                turn_id=turn_id,
                manifest=manifest,
                surrounding_nodes=surrounding_nodes,
            )
            daimoi_rows = self._emit_daimoi_locked(
                muse=muse,
                manifest=manifest,
                turn_id=turn_id,
                rng=rng,
            )
            if isinstance(media_action, dict) and media_action:
                media_action = self._route_media_action_with_daimoi_locked(
                    muse_id=clean_muse_id,
                    turn_id=turn_id,
                    action=media_action,
                    daimoi_rows=daimoi_rows,
                )
            field_deltas = self._field_deltas_from_daimoi_locked(
                muse=muse,
                turn_id=turn_id,
                daimoi_rows=daimoi_rows,
            )
            gpu_claim = self._claim_gpu_locked(
                muse=muse,
                turn_id=turn_id,
                manifest=manifest,
                rng=rng,
                surrounding_nodes=surrounding_nodes,
            )

            context_lines = [
                "Muse context manifest:",
                f"muse_id={clean_muse_id}",
                f"turn_id={turn_id}",
                f"mode={clean_mode}",
                f"graph_revision={graph_revision}",
                f"tet_units={len(manifest.get('tet_units', []))}",
                f"gpu_claim={str(gpu_claim.get('status', 'released'))}",
            ]
            for tet in manifest.get("tet_units", [])[:12]:
                if not isinstance(tet, dict):
                    continue
                context_lines.append(
                    "- "
                    + str(tet.get("node_id", ""))
                    + " | "
                    + str(tet.get("kind", "resource"))
                    + " | "
                    + str(tet.get("text", ""))[:180]
                )
            if tool_rows:
                context_lines.append("tool_results:")
                for item in tool_rows[:4]:
                    context_lines.append(
                        "- "
                        + str(item.get("tool", "tool"))
                        + " => "
                        + str(item.get("summary", ""))[:200]
                    )
            if isinstance(media_action, dict) and media_action:
                context_lines.append("media_action:")
                context_lines.append(
                    "- status="
                    + str(media_action.get("status", ""))
                    + " kind="
                    + str(media_action.get("media_kind", ""))
                    + " target="
                    + str(media_action.get("selected_node_id", ""))
                    + " collisions="
                    + str(media_action.get("collision_count", 0))
                )
            context_block = "\n".join(context_lines)

            history_rows = self._history_messages_locked(clean_muse_id, limit=14)
            model_messages = [
                {"role": str(row.get("role", "user")), "text": str(row.get("text", ""))}
                for row in history_rows
            ]

            assistant_text = ""
            model_name = None
            reply_mode = "canonical"
            fallback_used = False
            grounded_row = self._grounded_reply_from_tool_rows_locked(tool_rows)
            grounded_receipts: dict[str, Any] = {}
            if isinstance(grounded_row, dict):
                assistant_text = str(grounded_row.get("reply", "")).strip()
                grounded_receipts = (
                    dict(grounded_row.get("receipts", {}))
                    if isinstance(grounded_row.get("receipts", {}), dict)
                    else {}
                )
                model_name = "tool-grounded"
                reply_mode = "grounded"
            elif callable(reply_builder):
                try:
                    built = reply_builder(
                        messages=model_messages,
                        context_block=context_block,
                        mode=clean_mode,
                        muse_id=clean_muse_id,
                        turn_id=turn_id,
                    )
                    if isinstance(built, dict):
                        assistant_text = str(built.get("reply", "")).strip()
                        model_name = built.get("model")
                        reply_mode = str(built.get("mode", "canonical") or "canonical")
                except Exception:
                    assistant_text = ""
            if not assistant_text:
                fallback_used = True
                assistant_text = (
                    "Muse signal stabilized. I translated your message into bounded field deltas "
                    "and recorded a replayable context manifest."
                )
            if (
                isinstance(media_action, dict)
                and str(media_action.get("status", "")) == "requested"
            ):
                selected_label = str(media_action.get("selected_label", "")).strip()
                media_kind = str(media_action.get("media_kind", "")).strip().lower()
                if selected_label and media_kind == "audio":
                    assistant_text += f" Queued audio: {selected_label}."
                elif selected_label and media_kind == "image":
                    assistant_text += f" Opened image target: {selected_label}."

            assistant_node = self._append_message_node_locked(
                muse_id=clean_muse_id,
                turn_id=turn_id,
                role="assistant",
                text=assistant_text,
            )
            self._emit_event_locked(
                kind="resource.message.appended",
                status="ok",
                muse_id=clean_muse_id,
                turn_id=turn_id,
                payload={
                    "resource_node_id": str(assistant_node.get("id", "")),
                    "role": "assistant",
                },
            )
            self._emit_event_locked(
                kind="muse.response.generated",
                status="ok",
                muse_id=clean_muse_id,
                turn_id=turn_id,
                payload={
                    "mode": reply_mode,
                    "model": model_name,
                    "fallback": fallback_used,
                    "assistant_node_id": str(assistant_node.get("id", "")),
                },
            )
            if grounded_receipts:
                self._emit_event_locked(
                    kind="muse_job_completed",
                    status="ok",
                    muse_id=clean_muse_id,
                    turn_id=turn_id,
                    payload={
                        "snapshot_hash": str(
                            grounded_receipts.get("snapshot_hash", "")
                        ).strip(),
                        "queries_used": list(
                            grounded_receipts.get("queries_used", [])
                            if isinstance(
                                grounded_receipts.get("queries_used", []), list
                            )
                            else []
                        )[:8],
                    },
                )
            self._emit_event_locked(
                kind="muse.turn.completed",
                status="ok",
                muse_id=clean_muse_id,
                turn_id=turn_id,
                payload={
                    "manifest_id": str(manifest.get("id", "")),
                    "daimoi": len(daimoi_rows),
                    "field_deltas": len(field_deltas),
                    "tool_results": len(tool_rows),
                    "media_actions": 1
                    if isinstance(media_action, dict) and media_action
                    else 0,
                    "audio_actions": 1
                    if isinstance(media_action, dict)
                    and media_action
                    and str(media_action.get("media_kind", "")).strip().lower()
                    == "audio"
                    else 0,
                },
            )

            self._release_gpu_claim_locked(muse=muse, turn_id=turn_id, claim=gpu_claim)

            fresh_window.append(now_ms)
            self._state["rate_windows"][clean_muse_id] = fresh_window[
                -self._turns_per_minute :
            ]

            response_payload = {
                "ok": True,
                "record": "eta-mu.muse-turn.v1",
                "schema_version": MUSE_RUNTIME_SCHEMA_VERSION,
                "generated_at": _now_iso(),
                "muse": self._public_muse_row(muse),
                "turn_id": turn_id,
                "reply": assistant_text,
                "mode": reply_mode,
                "model": model_name,
                "fallback": fallback_used,
                "manifest": manifest,
                "daimoi": daimoi_rows,
                "field_deltas": field_deltas,
                "gpu_claim": gpu_claim,
                "tool_results": tool_rows,
                "media_actions": [media_action]
                if isinstance(media_action, dict) and media_action
                else [],
                "audio_actions": [media_action]
                if isinstance(media_action, dict)
                and media_action
                and str(media_action.get("media_kind", "")).strip().lower() == "audio"
                else [],
                "grounded_receipts": grounded_receipts,
                "messages": [message_node, assistant_node],
            }
            if clean_key:
                dedupe_map[clean_key] = response_payload
                if len(dedupe_map) > 128:
                    oldest_keys = sorted(dedupe_map.keys())[: len(dedupe_map) - 128]
                    for key in oldest_keys:
                        dedupe_map.pop(key, None)
                self._state["idempotency"][clean_muse_id] = dedupe_map
            self._persist_locked()
            return response_payload

    def _history_messages_locked(
        self, muse_id: str, *, limit: int
    ) -> list[dict[str, Any]]:
        rows = self._state["messages"].get(muse_id, [])
        if not isinstance(rows, list):
            return []
        clean_limit = max(1, min(128, int(limit)))
        trimmed = rows[-clean_limit:]
        return [dict(row) for row in trimmed if isinstance(row, dict)]

    def _append_message_node_locked(
        self,
        *,
        muse_id: str,
        turn_id: str,
        role: str,
        text: str,
    ) -> dict[str, Any]:
        rows = self._state["messages"].setdefault(muse_id, [])
        if not isinstance(rows, list):
            rows = []
        node_id = f"resource:message:{sha1(f'{muse_id}|{turn_id}|{role}|{text}|{_now_ms()}'.encode('utf-8')).hexdigest()[:16]}"
        row = {
            "record": MUSE_RESOURCE_NODE_RECORD,
            "id": node_id,
            "muse_id": muse_id,
            "turn_id": turn_id,
            "role": role,
            "text": text,
            "created_at": _now_iso(),
            "kind": "message",
            "visibility": "private",
            "x": _safe_float(
                self._state["muses"][muse_id].get("anchor", {}).get("x", 0.5), 0.5
            ),
            "y": _safe_float(
                self._state["muses"][muse_id].get("anchor", {}).get("y", 0.5), 0.5
            ),
        }
        rows.append(row)
        if len(rows) > self._max_history_messages:
            self._state["messages"][muse_id] = rows[-self._max_history_messages :]
        else:
            self._state["messages"][muse_id] = rows
        return dict(row)

    def _assemble_context_manifest_locked(
        self,
        *,
        muse: dict[str, Any],
        turn_id: str,
        graph_revision: str,
        mode: str,
        token_budget: int,
        seed: str,
        surrounding_nodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        muse_id = str(muse.get("id", "")).strip()
        anchor = _normalized_anchor(muse.get("anchor", {}))
        alpha = 3.2
        tau = 0.82
        explicit_budget = int(token_budget * 0.6)
        surround_budget = max(0, token_budget - explicit_budget)

        candidate_by_id: dict[str, dict[str, Any]] = {}
        clean_surrounding = (
            surrounding_nodes if isinstance(surrounding_nodes, list) else []
        )
        for row in clean_surrounding:
            if not isinstance(row, dict):
                continue
            node_id = str(row.get("id", "")).strip()
            if not node_id:
                continue
            if node_id in candidate_by_id:
                continue
            candidate_by_id[node_id] = {
                "id": node_id,
                "kind": str(row.get("kind", "resource") or "resource").strip()
                or "resource",
                "label": str(row.get("label", node_id) or node_id),
                "text": str(
                    row.get("text", row.get("summary", row.get("label", ""))) or ""
                ),
                "x": round(_clamp01(_safe_float(row.get("x", 0.5), 0.5)), 6),
                "y": round(_clamp01(_safe_float(row.get("y", 0.5), 0.5)), 6),
                "visibility": _normalize_visibility(row.get("visibility", "public")),
                "tags": [
                    str(item)
                    for item in (
                        row.get("tags", []) if isinstance(row.get("tags"), list) else []
                    )
                    if str(item).strip()
                ],
                "embedding_ref": str(row.get("embedding_ref", "") or "").strip(),
                "ts": str(row.get("ts", "") or "").strip(),
            }

        for row in self._history_messages_locked(muse_id, limit=10):
            node_id = str(row.get("id", "")).strip()
            if not node_id:
                continue
            candidate_by_id[node_id] = {
                "id": node_id,
                "kind": "message",
                "label": str(row.get("role", "message")),
                "text": str(row.get("text", "")),
                "x": round(
                    _clamp01(_safe_float(row.get("x", anchor["x"]), anchor["x"])), 6
                ),
                "y": round(
                    _clamp01(_safe_float(row.get("y", anchor["y"]), anchor["y"])), 6
                ),
                "visibility": "private",
                "tags": ["chat-history"],
                "embedding_ref": "",
                "ts": str(row.get("created_at", "")),
            }

        explicit_ids: list[str] = []
        for node_id in (
            muse.get("pinned_node_ids", [])
            if isinstance(muse.get("pinned_node_ids"), list)
            else []
        ):
            clean_node = str(node_id).strip()
            if not clean_node:
                continue
            if clean_node not in explicit_ids:
                explicit_ids.append(clean_node)
            candidate_by_id.setdefault(
                clean_node,
                {
                    "id": clean_node,
                    "kind": "resource",
                    "label": clean_node,
                    "text": clean_node,
                    "x": anchor["x"],
                    "y": anchor["y"],
                    "visibility": "private",
                    "tags": ["explicit-pin"],
                    "embedding_ref": "",
                    "ts": "",
                },
            )

        active_nexus = str(muse.get("active_nexus_id", "")).strip()
        if active_nexus:
            if active_nexus not in explicit_ids:
                explicit_ids.append(active_nexus)
            candidate_by_id.setdefault(
                active_nexus,
                {
                    "id": active_nexus,
                    "kind": "nexus",
                    "label": active_nexus,
                    "text": active_nexus,
                    "x": anchor["x"],
                    "y": anchor["y"],
                    "visibility": "private",
                    "tags": ["active-nexus"],
                    "embedding_ref": "",
                    "ts": "",
                },
            )

        recent_message_ids = [
            str(row.get("id", "")).strip()
            for row in self._history_messages_locked(muse_id, limit=6)
            if str(row.get("id", "")).strip()
        ]
        for node_id in recent_message_ids:
            if node_id not in explicit_ids:
                explicit_ids.append(node_id)

        tet_units: list[dict[str, Any]] = []
        explicit_selected: list[str] = []
        surround_selected: list[str] = []
        surround_candidates: list[str] = []
        dropped: list[dict[str, str]] = []

        explicit_tokens = 0
        for node_id in explicit_ids:
            node = candidate_by_id.get(node_id)
            if not isinstance(node, dict):
                continue
            visibility = _normalize_visibility(node.get("visibility", "public"))
            if visibility == "sealed":
                dropped.append({"node_id": node_id, "reason": "sealed_visibility"})
                continue
            text = str(node.get("text", "") or "")[:420]
            token_cost = _token_cost(text)
            if explicit_tokens + token_cost > explicit_budget and explicit_selected:
                dropped.append(
                    {"node_id": node_id, "reason": "explicit_budget_exceeded"}
                )
                continue
            explicit_tokens += token_cost
            explicit_selected.append(node_id)
            distance = math.hypot(
                _safe_float(node.get("x", anchor["x"]), anchor["x"]) - anchor["x"],
                _safe_float(node.get("y", anchor["y"]), anchor["y"]) - anchor["y"],
            )
            tet_units.append(
                {
                    "tet_id": f"tet:{sha1(f'{turn_id}|{node_id}'.encode('utf-8')).hexdigest()[:14]}",
                    "node_id": node_id,
                    "kind": str(node.get("kind", "resource") or "resource"),
                    "distance": round(distance, 6),
                    "recency_ms": self._recency_ms(node.get("ts", "")),
                    "token_cost": token_cost,
                    "text": text,
                    "embedding_ref": str(node.get("embedding_ref", "") or ""),
                    "provenance_event_id": "",
                }
            )

        scored_rows: list[tuple[dict[str, Any], float]] = []
        for node_id, node in candidate_by_id.items():
            if node_id in explicit_selected:
                continue
            visibility = _normalize_visibility(node.get("visibility", "public"))
            if visibility == "sealed":
                dropped.append({"node_id": node_id, "reason": "sealed_visibility"})
                continue
            distance = math.hypot(
                _safe_float(node.get("x", anchor["x"]), anchor["x"]) - anchor["x"],
                _safe_float(node.get("y", anchor["y"]), anchor["y"]) - anchor["y"],
            )
            prox = math.exp(-alpha * distance)
            recency_ms = self._recency_ms(node.get("ts", ""))
            recency = 1.0 / (1.0 + (recency_ms / 90000.0)) if recency_ms >= 0 else 0.0
            affinity = 0.0
            tags = [
                str(item).strip() for item in node.get("tags", []) if str(item).strip()
            ]
            if muse_id in tags:
                affinity += 0.22
            if active_nexus and active_nexus in tags:
                affinity += 0.22
            if active_nexus and node_id == active_nexus:
                affinity += 0.28
            score = (0.56 * prox) + (0.22 * recency) + (0.22 * affinity)
            scored_rows.append((node, score))

        scored_rows.sort(
            key=lambda item: (
                -item[1],
                math.hypot(
                    _safe_float(item[0].get("x", anchor["x"]), anchor["x"])
                    - anchor["x"],
                    _safe_float(item[0].get("y", anchor["y"]), anchor["y"])
                    - anchor["y"],
                ),
                str(item[0].get("id", "")),
            )
        )
        surround_candidates = [str(item[0].get("id", "")) for item in scored_rows]

        selected_surround_rows: list[tuple[dict[str, Any], float]] = []
        if mode == "deterministic":
            selected_surround_rows = scored_rows
        else:
            pool = list(scored_rows)
            while pool:
                weights = []
                for node, score in pool:
                    weights.append(math.exp(score / max(0.1, tau)))
                total = sum(weights)
                if total <= 0:
                    break
                mark = (
                    random.Random(
                        _seed_to_int(seed + str(len(selected_surround_rows)))
                    ).random()
                    * total
                )
                cursor = 0.0
                chosen_index = 0
                for idx, weight in enumerate(weights):
                    cursor += weight
                    if cursor >= mark:
                        chosen_index = idx
                        break
                selected_surround_rows.append(pool.pop(chosen_index))

        surround_tokens = 0
        for node, score in selected_surround_rows:
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            text = str(node.get("text", "") or "")[:420]
            token_cost = _token_cost(text)
            if surround_tokens + token_cost > surround_budget:
                dropped.append(
                    {"node_id": node_id, "reason": "surround_budget_exceeded"}
                )
                continue
            surround_tokens += token_cost
            surround_selected.append(node_id)
            distance = math.hypot(
                _safe_float(node.get("x", anchor["x"]), anchor["x"]) - anchor["x"],
                _safe_float(node.get("y", anchor["y"]), anchor["y"]) - anchor["y"],
            )
            tet_units.append(
                {
                    "tet_id": f"tet:{sha1(f'{turn_id}|{node_id}'.encode('utf-8')).hexdigest()[:14]}",
                    "node_id": node_id,
                    "kind": str(node.get("kind", "resource") or "resource"),
                    "distance": round(distance, 6),
                    "recency_ms": self._recency_ms(node.get("ts", "")),
                    "token_cost": token_cost,
                    "text": text,
                    "embedding_ref": str(node.get("embedding_ref", "") or ""),
                    "score": round(score, 6),
                    "provenance_event_id": "",
                }
            )

        manifest_id = f"manifest:{sha1(f'{muse_id}|{turn_id}|{seed}'.encode('utf-8')).hexdigest()[:18]}"
        manifest = {
            "record": MUSE_CONTEXT_MANIFEST_RECORD,
            "schema_version": MUSE_CONTEXT_MANIFEST_SCHEMA_VERSION,
            "id": manifest_id,
            "muse_id": muse_id,
            "turn_id": turn_id,
            "graph_revision": str(graph_revision or ""),
            "scorer_version": "muse.proximity.v1",
            "rng_seed": seed,
            "mode": mode,
            "token_budget": token_budget,
            "explicit_selected": explicit_selected,
            "surround_candidates": surround_candidates,
            "surround_selected": surround_selected,
            "tet_units": tet_units,
            "dropped": dropped,
            "generated_at": _now_iso(),
        }
        manifests = self._state["manifests"].setdefault(muse_id, {})
        if not isinstance(manifests, dict):
            manifests = {}
        manifests[turn_id] = manifest
        if len(manifests) > self._max_manifests_per_muse:
            keys = sorted(manifests.keys())
            for key in keys[: len(manifests) - self._max_manifests_per_muse]:
                manifests.pop(key, None)
        self._state["manifests"][muse_id] = manifests
        return manifest

    def _emit_daimoi_locked(
        self,
        *,
        muse: dict[str, Any],
        manifest: dict[str, Any],
        turn_id: str,
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        muse_id = str(muse.get("id", "")).strip()
        anchor = _normalized_anchor(muse.get("anchor", {}))
        tet_units = [
            item for item in manifest.get("tet_units", []) if isinstance(item, dict)
        ]
        if not tet_units:
            return []

        sample_count = min(self._max_particles_per_turn, max(1, len(tet_units)))
        source_units = tet_units[:]
        rng.shuffle(source_units)
        selected_units = source_units[:sample_count]

        rows: list[dict[str, Any]] = []
        for idx, tet in enumerate(selected_units):
            source_id = str(tet.get("node_id", "")).strip() or f"node:{idx}"
            embedding_seed = f"{muse_id}|{turn_id}|{source_id}|{idx}"
            emb = self._embedding_preview(embedding_seed)
            jitter_x = (rng.random() - 0.5) * 0.06
            jitter_y = (rng.random() - 0.5) * 0.06
            row = {
                "record": MUSE_DAIMON_RECORD,
                "id": f"daimon:{sha1(embedding_seed.encode('utf-8')).hexdigest()[:16]}",
                "muse_id": muse_id,
                "turn_id": turn_id,
                "manifest_id": str(manifest.get("id", "")),
                "source_node_id": source_id,
                "embedding": emb,
                "x": round(_clamp01(anchor["x"] + jitter_x), 6),
                "y": round(_clamp01(anchor["y"] + jitter_y), 6),
                "ttl": int(3 + math.floor(rng.random() * 4)),
                "energy": round(0.34 + (rng.random() * 0.58), 6),
                "quarantine_turns": 1,
                "created_at": _now_iso(),
            }
            rows.append(row)

        self._emit_event_locked(
            kind="muse.daimoi.emitted",
            status="ok",
            muse_id=muse_id,
            turn_id=turn_id,
            payload={
                "count": len(rows),
                "manifest_id": str(manifest.get("id", "")),
            },
        )
        return rows

    def _field_deltas_from_daimoi_locked(
        self,
        *,
        muse: dict[str, Any],
        turn_id: str,
        daimoi_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        muse_id = str(muse.get("id", "")).strip()
        deltas: list[dict[str, Any]] = []
        for row in daimoi_rows[:16]:
            energy = _clamp01(_safe_float(row.get("energy", 0.0), 0.0))
            ttl = max(1, _safe_int(row.get("ttl", 1), 1))
            deltas.append(
                {
                    "id": f"delta:{sha1((str(row.get('id', '')) + turn_id).encode('utf-8')).hexdigest()[:14]}",
                    "muse_id": muse_id,
                    "turn_id": turn_id,
                    "x": round(_clamp01(_safe_float(row.get("x", 0.5), 0.5)), 6),
                    "y": round(_clamp01(_safe_float(row.get("y", 0.5), 0.5)), 6),
                    "intensity": round(min(1.0, energy * 0.92), 6),
                    "radius": round(0.05 + (energy * 0.12), 6),
                    "ttl": ttl,
                }
            )
        self._emit_event_locked(
            kind="field.delta.applied",
            status="ok",
            muse_id=muse_id,
            turn_id=turn_id,
            payload={
                "count": len(deltas),
            },
        )
        return deltas

    def _claim_gpu_locked(
        self,
        *,
        muse: dict[str, Any],
        turn_id: str,
        manifest: dict[str, Any],
        rng: random.Random,
        surrounding_nodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        muse_id = str(muse.get("id", "")).strip()
        self._emit_event_locked(
            kind="muse.gpu.claim.requested",
            status="ok",
            muse_id=muse_id,
            turn_id=turn_id,
            payload={
                "manifest_id": str(manifest.get("id", "")),
            },
        )

        candidate_devices: dict[str, float] = {}
        for row in surrounding_nodes if isinstance(surrounding_nodes, list) else []:
            if not isinstance(row, dict):
                continue
            if str(row.get("kind", "")).strip() != "device":
                continue
            node_id = str(row.get("id", "")).strip().lower()
            if "gpu" not in node_id and "npu" not in node_id:
                continue
            util = _clamp01(_safe_float(row.get("utilization", 0.0), 0.0))
            candidate_devices[node_id] = util

        if not candidate_devices:
            self._emit_event_locked(
                kind="muse.gpu.claim.rejected",
                status="blocked",
                muse_id=muse_id,
                turn_id=turn_id,
                payload={
                    "reason": "no_gpu_devices_visible",
                },
            )
            return {
                "record": MUSE_GPU_CLAIM_RECORD,
                "status": "rejected",
                "reason": "no_gpu_devices_visible",
                "claim_id": "",
                "device": "",
            }

        best_device = ""
        best_score = -1.0
        best_util = 1.0
        for device, util in candidate_devices.items():
            attract = 0.4 + (rng.random() * 0.6)
            score = ((1.0 - util) * 0.72) + (attract * 0.28)
            if score > best_score:
                best_score = score
                best_device = device
                best_util = util

        if not best_device or best_util >= 0.97:
            self._emit_event_locked(
                kind="muse.gpu.claim.rejected",
                status="blocked",
                muse_id=muse_id,
                turn_id=turn_id,
                payload={
                    "reason": "gpu_pressure_high",
                    "utilization": round(best_util, 6),
                },
            )
            return {
                "record": MUSE_GPU_CLAIM_RECORD,
                "status": "rejected",
                "reason": "gpu_pressure_high",
                "claim_id": "",
                "device": best_device,
                "utilization": round(best_util, 6),
            }

        claim_id = f"gpu-claim:{sha1(f'{muse_id}|{turn_id}|{best_device}'.encode('utf-8')).hexdigest()[:16]}"
        claim = {
            "record": MUSE_GPU_CLAIM_RECORD,
            "status": "granted",
            "claim_id": claim_id,
            "device": best_device,
            "utilization": round(best_util, 6),
            "influence": round(best_score, 6),
            "granted_at": _now_iso(),
        }
        muse["gpu_state"] = {
            "status": "claimed",
            "claim_id": claim_id,
            "device": best_device,
            "updated_at": _now_iso(),
        }
        self._emit_event_locked(
            kind="muse.gpu.claim.granted",
            status="ok",
            muse_id=muse_id,
            turn_id=turn_id,
            payload={
                "claim_id": claim_id,
                "device": best_device,
                "utilization": round(best_util, 6),
            },
        )
        return claim

    def _release_gpu_claim_locked(
        self,
        *,
        muse: dict[str, Any],
        turn_id: str,
        claim: dict[str, Any],
    ) -> None:
        if not isinstance(claim, dict):
            return
        if str(claim.get("status", "")) != "granted":
            return
        muse_id = str(muse.get("id", "")).strip()
        muse["gpu_state"] = {
            "status": "released",
            "claim_id": "",
            "device": str(claim.get("device", "")),
            "updated_at": _now_iso(),
        }
        self._emit_event_locked(
            kind="muse.gpu.claim.released",
            status="ok",
            muse_id=muse_id,
            turn_id=turn_id,
            payload={
                "claim_id": str(claim.get("claim_id", "")),
                "device": str(claim.get("device", "")),
            },
        )

    def _run_tool_requests_locked(
        self,
        text: str,
        *,
        muse_id: str,
        turn_id: str,
        tool_callback: Any | None,
    ) -> list[dict[str, Any]]:
        requested = self._tool_requests(text)
        if not requested:
            return []
        if not callable(tool_callback):
            return []
        rows: list[dict[str, Any]] = []
        for tool_name in requested[:3]:
            is_grounding_tool = (
                tool_name == "facts_snapshot"
                or tool_name.startswith("graph:")
                or tool_name.startswith("graph_query:")
                or tool_name == "graph_query"
            )
            if is_grounding_tool:
                self._emit_event_locked(
                    kind="muse_job_enqueued",
                    status="ok",
                    muse_id=muse_id,
                    turn_id=turn_id,
                    payload={"tool": tool_name},
                )
                self._emit_event_locked(
                    kind="muse_job_started",
                    status="ok",
                    muse_id=muse_id,
                    turn_id=turn_id,
                    payload={"tool": tool_name},
                )
            self._emit_event_locked(
                kind="muse.tool.requested",
                status="ok",
                muse_id=muse_id,
                turn_id=turn_id,
                payload={"tool": tool_name},
            )
            try:
                result = tool_callback(tool_name=tool_name)
            except Exception as exc:
                result = {"ok": False, "error": f"tool_failed:{exc.__class__.__name__}"}
            summary = ""
            if isinstance(result, dict):
                if bool(result.get("ok", False)):
                    summary = str(result.get("summary", "ok")).strip() or "ok"
                else:
                    summary = str(result.get("error", "error")).strip() or "error"
            row = {
                "tool": tool_name,
                "result": result if isinstance(result, dict) else {"ok": False},
                "summary": summary,
            }
            rows.append(row)
            self._emit_event_locked(
                kind="muse.tool.result",
                status="ok" if bool(row["result"].get("ok", False)) else "blocked",
                muse_id=muse_id,
                turn_id=turn_id,
                payload={
                    "tool": tool_name,
                    "summary": summary[:180],
                },
            )
            if is_grounding_tool:
                self._emit_event_locked(
                    kind="muse_job_completed",
                    status="ok" if bool(row["result"].get("ok", False)) else "blocked",
                    muse_id=muse_id,
                    turn_id=turn_id,
                    payload={
                        "tool": tool_name,
                        "snapshot_hash": str(
                            row["result"].get("snapshot_hash", "")
                        ).strip(),
                        "query": str(row["result"].get("query", "")).strip(),
                    },
                )
        return rows

    def _tool_requests(self, text: str) -> list[str]:
        raw_text = str(text or "").strip()
        normalized = raw_text.lower()
        requested: list[str] = []

        def _append_tool(name: str) -> None:
            clean = str(name or "").strip()
            if not clean:
                return
            if clean not in requested:
                requested.append(clean)

        if not normalized:
            return requested
        if normalized.startswith("/facts") or " facts " in f" {normalized} ":
            _append_tool("facts_snapshot")
        if "/graph" in normalized:
            graph_tail = str(normalized.split("/graph", 1)[1]).strip()
            _append_tool(f"graph_query:{graph_tail or 'overview'}")
        else:
            daimoi_id = ""
            daimoi_match = re.search(r"\bdaimoi[:\s]+([a-z0-9._:-]+)", normalized)
            if daimoi_match:
                daimoi_id = str(daimoi_match.group(1) or "").strip()
            if not daimoi_id:
                for token in re.findall(r"[a-z0-9._:-]+", normalized):
                    if token.startswith("field:") or token.startswith("daimoi:"):
                        daimoi_id = token
                        break

            asks_why_daimoi = (
                ("why" in normalized and "daimoi" in normalized)
                or "explain daimoi" in normalized
                or "daimoi die" in normalized
                or "daimoi died" in normalized
                or "daimoi death" in normalized
            )
            if asks_why_daimoi and daimoi_id:
                _append_tool(f"graph_query:explain_daimoi {daimoi_id}")

            asks_recent_outcomes = (
                "recent outcomes" in normalized
                or "recent outcome" in normalized
                or ("food" in normalized and "death" in normalized)
                or "wins and losses" in normalized
            )
            if asks_recent_outcomes:
                _append_tool("graph_query:recent_outcomes 360 24")

            asks_crawler_status = (
                "crawler status" in normalized
                or "crawler queue" in normalized
                or "cooldown" in normalized
            )
            if asks_crawler_status:
                _append_tool("graph_query:crawler_status")

            asks_arxiv_papers = (
                "arxiv" in normalized
                and (
                    "paper" in normalized
                    or "crawl" in normalized
                    or "crawled" in normalized
                    or "fetched" in normalized
                    or "fetch" in normalized
                )
            ) or ("papers" in normalized and "crawled" in normalized)
            if asks_arxiv_papers:
                _append_tool("graph_query:arxiv_papers 8")

            asks_web_resource_summary = (
                "crawler learn" in normalized
                or "what got crawled" in normalized
                or "web resource" in normalized
            )
            if asks_web_resource_summary:
                target = ""
                url_match = re.search(r"https?://\S+", raw_text)
                if url_match:
                    target = str(url_match.group(0) or "").strip()
                if target:
                    _append_tool(f"graph_query:web_resource_summary {target}")
                else:
                    _append_tool("graph_query:web_resource_summary")

            if "graph summary" in normalized or "graph overview" in normalized:
                _append_tool("graph_query:graph_summary all 12")
        if normalized.startswith("/study") or "stability" in normalized:
            _append_tool("study_snapshot")
        if normalized.startswith("/drift") or "drift" in normalized:
            _append_tool("drift_scan")
        if "push truth" in normalized or "push-truth" in normalized:
            _append_tool("push_truth_dry_run")
        return requested

    def _grounded_reply_from_tool_rows_locked(
        self, tool_rows: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not isinstance(tool_rows, list) or not tool_rows:
            return None

        facts_hash = ""
        facts_path = ""
        facts_nodes = 0
        facts_edges = 0
        graph_receipts: list[str] = []
        query_names: list[str] = []
        grounded = False
        crawler_summary: dict[str, Any] = {}
        arxiv_summary: dict[str, Any] = {}

        for row in tool_rows:
            if not isinstance(row, dict):
                continue
            tool_name = str(row.get("tool", "")).strip().lower()
            result = row.get("result", {})
            if not isinstance(result, dict):
                continue
            if tool_name == "facts_snapshot" and bool(result.get("ok", False)):
                grounded = True
                facts_hash = str(result.get("snapshot_hash", "")).strip()
                facts_path = str(result.get("snapshot_path", "")).strip()
                facts_nodes = max(0, _safe_int(result.get("node_count", 0), 0))
                facts_edges = max(0, _safe_int(result.get("edge_count", 0), 0))
            if (
                tool_name.startswith("graph:") or tool_name.startswith("graph_query:")
            ) and bool(result.get("ok", False)):
                grounded = True
                query_name = str(result.get("query", "")).strip()
                if query_name and query_name not in query_names:
                    query_names.append(query_name)
                query_count = max(0, _safe_int(result.get("result_count", 0), 0))
                graph_receipts.append(f"{query_name}:{query_count}")
                query_payload = result.get("result", {})
                if query_name == "crawler_status" and isinstance(query_payload, dict):
                    crawler_summary = {
                        "queue_length": max(
                            0, _safe_int(query_payload.get("queue_length", 0), 0)
                        ),
                        "url_node_count": max(
                            0, _safe_int(query_payload.get("url_node_count", 0), 0)
                        ),
                        "resource_node_count": max(
                            0, _safe_int(query_payload.get("resource_node_count", 0), 0)
                        ),
                        "arxiv_abs_fetched": max(
                            0, _safe_int(query_payload.get("arxiv_abs_fetched", 0), 0)
                        ),
                        "arxiv_recent_fetches": [
                            str(item.get("canonical_url", "")).strip()
                            for item in query_payload.get("arxiv_recent_fetches", [])
                            if isinstance(item, dict)
                            and str(item.get("canonical_url", "")).strip()
                        ][:4],
                    }
                if query_name == "arxiv_papers" and isinstance(query_payload, dict):
                    arxiv_summary = {
                        "count_total": max(
                            0, _safe_int(query_payload.get("count_total", 0), 0)
                        ),
                        "count_fetched": max(
                            0, _safe_int(query_payload.get("count_fetched", 0), 0)
                        ),
                        "papers": [
                            {
                                "title": str(item.get("title", "")).strip(),
                                "url": str(item.get("canonical_url", "")).strip(),
                                "fetched": bool(item.get("fetched", False)),
                                "status": str(item.get("last_status", "")).strip(),
                            }
                            for item in query_payload.get("papers", [])
                            if isinstance(item, dict)
                            and str(item.get("canonical_url", "")).strip()
                        ],
                    }

        if not grounded:
            return None

        facts_lines: list[str] = []
        derivation_lines: list[str] = []
        unknown_lines: list[str] = []
        if facts_hash:
            facts_line = (
                f"Facts grounded at {facts_hash}. "
                f"nodes={facts_nodes} edges={facts_edges}."
            )
            if facts_path:
                facts_line += f" snapshot={facts_path}."
            facts_lines.append(facts_line)
        if arxiv_summary:
            total = max(0, _safe_int(arxiv_summary.get("count_total", 0), 0))
            fetched = max(0, _safe_int(arxiv_summary.get("count_fetched", 0), 0))
            derivation_lines.append(
                f"arXiv crawl currently tracks {fetched} fetched papers out of {total} discovered."
            )
            paper_lines: list[str] = []
            for row in arxiv_summary.get("papers", []):
                if not isinstance(row, dict):
                    continue
                url = str(row.get("url", "")).strip()
                if not url:
                    continue
                title = str(row.get("title", "")).strip()
                if title and title.lower() not in {"arxiv.org", "arxiv"}:
                    paper_lines.append(f"{title} ({url})")
                else:
                    paper_lines.append(url)
                if len(paper_lines) >= 4:
                    break
            if paper_lines:
                derivation_lines.append(
                    "Recent arXiv papers: " + "; ".join(paper_lines) + "."
                )
        elif crawler_summary:
            queue_length = max(0, _safe_int(crawler_summary.get("queue_length", 0), 0))
            url_node_count = max(
                0, _safe_int(crawler_summary.get("url_node_count", 0), 0)
            )
            resource_node_count = max(
                0, _safe_int(crawler_summary.get("resource_node_count", 0), 0)
            )
            arxiv_abs_fetched = max(
                0, _safe_int(crawler_summary.get("arxiv_abs_fetched", 0), 0)
            )
            derivation_lines.append(
                "Crawler status: "
                + f"queue={queue_length}, url_nodes={url_node_count}, resources={resource_node_count}, "
                + f"arXiv_fetched={arxiv_abs_fetched}."
            )
            recent_rows = [
                str(row).strip()
                for row in crawler_summary.get("arxiv_recent_fetches", [])
                if str(row).strip()
            ][:4]
            if recent_rows:
                derivation_lines.append(
                    "Recent arXiv fetches: " + ", ".join(recent_rows) + "."
                )
        elif graph_receipts:
            derivation_lines.append(
                "Graph queries: " + ", ".join(graph_receipts[:6]) + "."
            )
        if not (facts_lines or derivation_lines):
            unknown_lines.append(
                "Grounding tools ran, but no resolvable facts were returned. unknown."
            )

        reply_parts: list[str] = []
        reply_parts.append("FACTS:")
        reply_parts.extend(
            facts_lines
            if facts_lines
            else ["No direct facts snapshot was available for this turn."]
        )
        reply_parts.append("DERIVATIONS:")
        reply_parts.extend(
            derivation_lines
            if derivation_lines
            else [
                "No higher-level derivation could be computed from current tool output."
            ]
        )
        reply_parts.append("UNKNOWN:")
        reply_parts.extend(
            unknown_lines
            if unknown_lines
            else ["No unresolved grounding gaps detected in this turn."]
        )

        receipts = {
            "snapshot_hash": facts_hash,
            "snapshot_path": facts_path,
            "queries_used": query_names,
        }
        return {
            "reply": " ".join(reply_parts).strip(),
            "receipts": receipts,
        }

    def _resolve_media_command_action_locked(
        self,
        *,
        text: str,
        muse: dict[str, Any],
        turn_id: str,
        manifest: dict[str, Any],
        surrounding_nodes: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self._audio_intent_enabled:
            return None
        normalized = str(text or "").strip().lower()
        if not normalized:
            return None

        token_list = re.findall(r"[a-z0-9_./:-]+", normalized)
        token_set = set(token_list)

        audio_suffix_hit = any(
            token.endswith(_AUDIO_SUFFIXES)
            or token in {"mp3", "wav", "ogg", "m4a", "flac"}
            for token in token_set
        )
        image_suffix_hit = any(
            token.endswith(_IMAGE_SUFFIXES)
            or token in {"png", "jpg", "jpeg", "webp", "gif", "svg", "bmp"}
            for token in token_set
        )
        audio_action_hit = bool(token_set & _AUDIO_ACTION_TOKENS)
        image_action_hit = bool(token_set & _IMAGE_ACTION_TOKENS)
        audio_hint_hit = bool(token_set & _AUDIO_HINT_TOKENS)
        image_hint_hit = bool(token_set & _IMAGE_HINT_TOKENS)

        explicit_audio = (
            normalized.startswith("/play")
            or normalized.startswith("play ")
            or ("play" in token_set and bool(token_set & _AUDIO_HINT_TOKENS))
        )
        explicit_image = (
            normalized.startswith("/image")
            or normalized.startswith("/open-image")
            or normalized.startswith("open image")
            or (("open" in token_set or "show" in token_set) and image_hint_hit)
        )

        audio_signal = 0
        image_signal = 0
        if audio_action_hit:
            audio_signal += 2
        if audio_hint_hit:
            audio_signal += 3
        if audio_suffix_hit:
            audio_signal += 3
        if explicit_audio:
            audio_signal += 4
        if image_action_hit:
            image_signal += 2
        if image_hint_hit:
            image_signal += 3
        if image_suffix_hit:
            image_signal += 3
        if explicit_image:
            image_signal += 4

        if audio_signal <= 0 and image_signal <= 0:
            return None

        requested_kind = ""
        if audio_signal > image_signal:
            requested_kind = "audio"
        elif image_signal > audio_signal:
            requested_kind = "image"

        strict_kind = ""
        if explicit_audio and not explicit_image:
            strict_kind = "audio"
        elif explicit_image and not explicit_audio:
            strict_kind = "image"

        muse_id = str(muse.get("id", "")).strip()
        query_tokens_by_kind = {
            "audio": [
                token
                for token in token_list
                if token not in _AUDIO_STOP_TOKENS and len(token) > 1
            ][:12],
            "image": [
                token
                for token in token_list
                if token not in _IMAGE_STOP_TOKENS and len(token) > 1
            ][:12],
        }
        explicit_selected = {
            str(item).strip()
            for item in manifest.get("explicit_selected", [])
            if str(item).strip()
        }
        tet_distance: dict[str, float] = {}
        for tet in manifest.get("tet_units", []) if isinstance(manifest, dict) else []:
            if not isinstance(tet, dict):
                continue
            node_id = str(tet.get("node_id", "")).strip()
            if not node_id:
                continue
            tet_distance[node_id] = _clamp01(_safe_float(tet.get("distance", 1.0), 1.0))

        classifier_bias = {"audio": 0.0, "image": 0.0}
        focus_node_bias = {"audio": {}, "image": {}}
        classifier_presence_ids: list[str] = []
        for raw_row in surrounding_nodes if isinstance(surrounding_nodes, list) else []:
            if not isinstance(raw_row, dict):
                continue
            row_id = str(
                raw_row.get(
                    "id", raw_row.get("node_id", raw_row.get("source_rel_path", ""))
                )
                or ""
            ).strip()
            row_kind = str(raw_row.get("kind", "") or "").strip().lower()
            presence_type = str(raw_row.get("presence_type", "") or "").strip().lower()
            tags = {
                str(item).strip().lower()
                for item in (
                    raw_row.get("tags", [])
                    if isinstance(raw_row.get("tags"), list)
                    else []
                )
                if str(item).strip()
            }
            seed_terms = {
                str(item).strip().lower()
                for item in (
                    raw_row.get("seed_terms", [])
                    if isinstance(raw_row.get("seed_terms"), list)
                    else []
                )
                if str(item).strip()
            }
            if not seed_terms:
                inferred_terms = re.findall(
                    r"[a-z0-9_./:-]+",
                    " ".join(
                        [
                            str(raw_row.get("label", "") or ""),
                            str(raw_row.get("text", "") or ""),
                            str(raw_row.get("source_rel_path", "") or ""),
                        ]
                    ).lower(),
                )
                seed_terms = {token for token in inferred_terms if len(token) > 1}

            focus_ids = [
                str(item).strip()
                for item in (
                    raw_row.get("focus_node_ids", [])
                    if isinstance(raw_row.get("focus_node_ids"), list)
                    else []
                )
                if str(item).strip()
            ]
            default_kind = (
                str(
                    raw_row.get("default_media_kind", raw_row.get("media_kind", ""))
                    or ""
                )
                .strip()
                .lower()
            )
            if default_kind not in {"audio", "image"}:
                continue

            overlap = len(seed_terms.intersection(token_set))
            is_classifier_row = bool(
                row_kind == "classifier"
                or presence_type
                in {"classifier", "modality_classifier", "baseline_classifier"}
                or "classifier" in tags
            )
            if is_classifier_row:
                classifier_bias[default_kind] += 0.14 + min(0.42, overlap * 0.1)
                if row_id and row_id not in classifier_presence_ids:
                    classifier_presence_ids.append(row_id)

            if focus_ids:
                focus_boost = 0.32 + min(1.4, overlap * 0.28)
                bias_map = focus_node_bias[default_kind]
                for focus_id in focus_ids:
                    current = _safe_float(bias_map.get(focus_id, 0.0), 0.0)
                    bias_map[focus_id] = current + focus_boost
            elif "concept-seed" in tags and overlap > 0:
                classifier_bias[default_kind] += min(0.24, overlap * 0.08)

        if not strict_kind and not requested_kind:
            audio_bias = _safe_float(classifier_bias.get("audio", 0.0), 0.0)
            image_bias = _safe_float(classifier_bias.get("image", 0.0), 0.0)
            if audio_bias > image_bias + 0.04:
                requested_kind = "audio"
            elif image_bias > audio_bias + 0.04:
                requested_kind = "image"

        candidates: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for raw_row in surrounding_nodes if isinstance(surrounding_nodes, list) else []:
            if not isinstance(raw_row, dict):
                continue
            node_id = str(
                raw_row.get(
                    "id", raw_row.get("node_id", raw_row.get("source_rel_path", ""))
                )
                or ""
            ).strip()
            if not node_id or node_id in seen_ids:
                continue
            seen_ids.add(node_id)

            kind = str(raw_row.get("kind", "resource") or "resource").strip().lower()
            label = str(raw_row.get("label", node_id) or node_id).strip() or node_id
            text_blob = str(raw_row.get("text", "") or "")
            source_rel_path = str(raw_row.get("source_rel_path", "") or "").strip()
            raw_url = str(raw_row.get("url", "") or "").strip()
            tags = [
                str(item).strip().lower()
                for item in (
                    raw_row.get("tags", [])
                    if isinstance(raw_row.get("tags"), list)
                    else []
                )
                if str(item).strip()
            ]
            joined = " ".join(
                [
                    node_id,
                    label,
                    text_blob,
                    source_rel_path,
                    raw_url,
                    " ".join(tags),
                ]
            ).lower()
            corpus_tokens = set(re.findall(r"[a-z0-9_./:-]+", joined))

            kind_audio = (
                kind == "audio"
                or kind.startswith("audio/")
                or "music" in kind
                or "song" in kind
            )
            suffix_audio = any(
                str(field).strip().lower().endswith(_AUDIO_SUFFIXES)
                for field in (node_id, label, source_rel_path, raw_url)
                if str(field).strip()
            )
            path_audio = "/artifacts/audio/" in joined or "artifacts/audio/" in joined
            lexical_audio = bool(
                {"song", "track", "music", "audio", "bpm"} & corpus_tokens
            )
            is_audio_candidate = bool(
                kind_audio or suffix_audio or path_audio or lexical_audio
            )

            kind_image = (
                kind == "image"
                or kind.startswith("image/")
                or kind == "cover_art"
                or "image" in kind
            )
            suffix_image = any(
                str(field).strip().lower().endswith(_IMAGE_SUFFIXES)
                for field in (node_id, label, source_rel_path, raw_url)
                if str(field).strip()
            )
            path_image = "/artifacts/images/" in joined or "artifacts/images/" in joined
            lexical_image = bool(
                {
                    "image",
                    "photo",
                    "picture",
                    "cover",
                    "png",
                    "jpg",
                    "jpeg",
                    "webp",
                    "gif",
                    "svg",
                }
                & corpus_tokens
            )
            is_image_candidate = bool(
                kind_image or suffix_image or path_image or lexical_image
            )

            if strict_kind == "audio":
                is_image_candidate = False
            elif strict_kind == "image":
                is_audio_candidate = False

            if requested_kind == "audio" and not strict_kind:
                is_image_candidate = False
            elif requested_kind == "image" and not strict_kind:
                is_audio_candidate = False

            resolved_url = raw_url
            if resolved_url and not resolved_url.startswith(
                ("http://", "https://", "/")
            ):
                resolved_url = "/" + resolved_url.lstrip("./")
            if not resolved_url and source_rel_path:
                clean_rel = source_rel_path.lstrip("/")
                if clean_rel.startswith("library/"):
                    resolved_url = "/" + clean_rel
                else:
                    resolved_url = "/library/" + clean_rel

            if is_audio_candidate:
                score = 0.0
                if kind_audio:
                    score += 1.0
                if suffix_audio:
                    score += 0.72
                if path_audio:
                    score += 0.42
                if node_id in explicit_selected:
                    score += 0.78
                if "workspace-pin" in tags:
                    score += 0.34
                if muse_id.lower() in tags:
                    score += 0.24
                if node_id in tet_distance:
                    score += (1.0 - tet_distance[node_id]) * 0.26
                overlap = len(
                    corpus_tokens.intersection(set(query_tokens_by_kind["audio"]))
                )
                score += min(0.84, overlap * 0.18)
                score += min(0.55, _safe_float(classifier_bias.get("audio", 0.0), 0.0))
                score += min(
                    0.9,
                    _safe_float(
                        focus_node_bias["audio"].get(node_id, 0.0)
                        if isinstance(focus_node_bias.get("audio"), dict)
                        else 0.0,
                        0.0,
                    ),
                )
                score += _hash_unit(f"audio|{muse_id}|{node_id}") * 0.04
                candidates.append(
                    {
                        "media_kind": "audio",
                        "intent": "play_music",
                        "node_id": node_id,
                        "label": label,
                        "kind": kind,
                        "score": score,
                        "url": resolved_url,
                        "source_rel_path": source_rel_path,
                        "target_presence_id": self._audio_target_presence_id,
                    }
                )

            if is_image_candidate:
                score = 0.0
                if kind_image:
                    score += 1.0
                if suffix_image:
                    score += 0.72
                if path_image:
                    score += 0.42
                if node_id in explicit_selected:
                    score += 0.78
                if "workspace-pin" in tags:
                    score += 0.34
                if muse_id.lower() in tags:
                    score += 0.24
                if node_id in tet_distance:
                    score += (1.0 - tet_distance[node_id]) * 0.26
                overlap = len(
                    corpus_tokens.intersection(set(query_tokens_by_kind["image"]))
                )
                score += min(0.84, overlap * 0.18)
                score += min(0.55, _safe_float(classifier_bias.get("image", 0.0), 0.0))
                score += min(
                    0.9,
                    _safe_float(
                        focus_node_bias["image"].get(node_id, 0.0)
                        if isinstance(focus_node_bias.get("image"), dict)
                        else 0.0,
                        0.0,
                    ),
                )
                score += _hash_unit(f"image|{muse_id}|{node_id}") * 0.04
                candidates.append(
                    {
                        "media_kind": "image",
                        "intent": "open_image",
                        "node_id": node_id,
                        "label": label,
                        "kind": kind,
                        "score": score,
                        "url": resolved_url,
                        "source_rel_path": source_rel_path,
                        "target_presence_id": self._image_target_presence_id,
                    }
                )

        candidates.sort(
            key=lambda row: (
                -_safe_float(row.get("score", 0.0), 0.0),
                str(row.get("media_kind", "")),
                str(row.get("node_id", "")),
            )
        )
        candidates = candidates[: self._audio_max_candidates]

        if strict_kind:
            fallback_candidates = [
                row
                for row in candidates
                if str(row.get("media_kind", "")) == strict_kind
            ]
            candidates = fallback_candidates

        resolved_kind = requested_kind or strict_kind
        if not resolved_kind and candidates:
            resolved_kind = str(candidates[0].get("media_kind", "")).strip().lower()

        if resolved_kind == "image":
            query_tokens = query_tokens_by_kind["image"]
        else:
            query_tokens = query_tokens_by_kind["audio"]

        action = {
            "record": "eta-mu.media-command-action.v1",
            "schema_version": "muse.media-command.v1",
            "muse_id": muse_id,
            "turn_id": turn_id,
            "intent": "command",
            "command": resolved_kind or "media",
            "media_kind": resolved_kind or "",
            "query": " ".join(query_tokens)[:120],
            "candidate_count": len(candidates),
            "target_presence_id": (
                self._audio_target_presence_id
                if resolved_kind != "image"
                else self._image_target_presence_id
            ),
            "selected_node_id": "",
            "selected_label": "",
            "selected_kind": "",
            "selected_url": "",
            "selected_source_rel_path": "",
            "score": 0.0,
            "collision_count": 0,
            "status": "blocked",
            "reason": "no_media_candidates",
            "created_at": _now_iso(),
            "classifier_presence_ids": classifier_presence_ids,
            "classifier_default_kind": (
                "audio"
                if _safe_float(classifier_bias.get("audio", 0.0), 0.0)
                > _safe_float(classifier_bias.get("image", 0.0), 0.0)
                else "image"
                if _safe_float(classifier_bias.get("image", 0.0), 0.0)
                > _safe_float(classifier_bias.get("audio", 0.0), 0.0)
                else ""
            ),
            "classifier_bias": {
                "audio": round(_safe_float(classifier_bias.get("audio", 0.0), 0.0), 6),
                "image": round(_safe_float(classifier_bias.get("image", 0.0), 0.0), 6),
            },
        }

        self._emit_event_locked(
            kind="muse.media.intent.detected",
            status="ok",
            muse_id=muse_id,
            turn_id=turn_id,
            payload={
                "media_kind": resolved_kind,
                "query": str(action.get("query", "")),
                "classifier_default_kind": str(
                    action.get("classifier_default_kind", "")
                ),
                "classifier_presence_count": len(classifier_presence_ids),
            },
        )
        if resolved_kind == "audio":
            self._emit_event_locked(
                kind="muse.audio.intent.detected",
                status="ok",
                muse_id=muse_id,
                turn_id=turn_id,
                payload={
                    "query": str(action.get("query", "")),
                },
            )
        if resolved_kind == "image":
            self._emit_event_locked(
                kind="muse.image.intent.detected",
                status="ok",
                muse_id=muse_id,
                turn_id=turn_id,
                payload={
                    "query": str(action.get("query", "")),
                },
            )

        if not candidates:
            reason = "no_media_candidates"
            if strict_kind == "audio":
                reason = "no_audio_candidates"
            elif strict_kind == "image":
                reason = "no_image_candidates"
            elif resolved_kind == "audio":
                reason = "no_audio_candidates"
            elif resolved_kind == "image":
                reason = "no_image_candidates"
            action["reason"] = reason
            blocked_kind = resolved_kind or strict_kind or "media"
            if blocked_kind == "audio":
                self._emit_event_locked(
                    kind="audio.play.blocked",
                    status="blocked",
                    muse_id=muse_id,
                    turn_id=turn_id,
                    payload={
                        "reason": reason,
                        "candidate_count": 0,
                    },
                )
            elif blocked_kind == "image":
                self._emit_event_locked(
                    kind="image.open.blocked",
                    status="blocked",
                    muse_id=muse_id,
                    turn_id=turn_id,
                    payload={
                        "reason": reason,
                        "candidate_count": 0,
                    },
                )
            return action

        selected = candidates[0]
        selected_score = _safe_float(selected.get("score", 0.0), 0.0)
        if selected_score < self._audio_min_score:
            action["reason"] = "confidence_too_low"
            action["score"] = round(selected_score, 6)
            selected_kind = str(selected.get("media_kind", "")).strip().lower()
            if selected_kind == "audio":
                self._emit_event_locked(
                    kind="audio.play.blocked",
                    status="blocked",
                    muse_id=muse_id,
                    turn_id=turn_id,
                    payload={
                        "reason": "confidence_too_low",
                        "candidate_count": len(candidates),
                        "score": round(selected_score, 6),
                        "threshold": round(self._audio_min_score, 6),
                    },
                )
            if selected_kind == "image":
                self._emit_event_locked(
                    kind="image.open.blocked",
                    status="blocked",
                    muse_id=muse_id,
                    turn_id=turn_id,
                    payload={
                        "reason": "confidence_too_low",
                        "candidate_count": len(candidates),
                        "score": round(selected_score, 6),
                        "threshold": round(self._audio_min_score, 6),
                    },
                )
            return action

        action["status"] = "requested"
        action["reason"] = ""
        action["media_kind"] = str(selected.get("media_kind", "")).strip().lower()
        action["command"] = action["media_kind"]
        action["intent"] = str(selected.get("intent", "command"))
        action["target_presence_id"] = str(selected.get("target_presence_id", ""))
        action["selected_node_id"] = str(selected.get("node_id", ""))
        action["selected_label"] = str(selected.get("label", ""))
        action["selected_kind"] = str(selected.get("kind", ""))
        action["selected_url"] = str(selected.get("url", ""))
        action["selected_source_rel_path"] = str(selected.get("source_rel_path", ""))
        action["score"] = round(selected_score, 6)
        return action

    def _route_media_action_with_daimoi_locked(
        self,
        *,
        muse_id: str,
        turn_id: str,
        action: dict[str, Any],
        daimoi_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if str(action.get("status", "")) != "requested":
            return action

        media_kind = str(action.get("media_kind", "")).strip().lower()
        target_node_id = str(action.get("selected_node_id", "")).strip()
        if not target_node_id:
            action["status"] = "blocked"
            action["reason"] = "missing_target_node"
            if media_kind == "image":
                self._emit_event_locked(
                    kind="image.open.blocked",
                    status="blocked",
                    muse_id=muse_id,
                    turn_id=turn_id,
                    payload={
                        "reason": "missing_target_node",
                    },
                )
            else:
                self._emit_event_locked(
                    kind="audio.play.blocked",
                    status="blocked",
                    muse_id=muse_id,
                    turn_id=turn_id,
                    payload={
                        "reason": "missing_target_node",
                    },
                )
            return action

        ranked_rows = sorted(
            [row for row in daimoi_rows if isinstance(row, dict)],
            key=lambda row: (
                -_safe_float(row.get("energy", 0.0), 0.0),
                str(row.get("id", "")),
            ),
        )
        selected_rows = ranked_rows[: self._audio_daimoi_fanout]
        collisions = 0
        daimon_ids: list[str] = []
        target_presence_id = str(
            action.get(
                "target_presence_id",
                self._audio_target_presence_id
                if media_kind != "image"
                else self._image_target_presence_id,
            )
        ).strip() or (
            self._audio_target_presence_id
            if media_kind != "image"
            else self._image_target_presence_id
        )

        for row in selected_rows:
            if media_kind == "image":
                row["intent"] = "image.open"
                row["collision"] = "image-resource"
            else:
                row["intent"] = "audio.play"
                row["collision"] = "audio-resource"
            row["target_node_id"] = target_node_id
            row["target_presence_id"] = target_presence_id
            daimon_id = str(row.get("id", "")).strip()
            if daimon_id:
                daimon_ids.append(daimon_id)
            collisions += 1

        action["collision_count"] = collisions
        action["daimon_ids"] = daimon_ids
        if collisions <= 0:
            action["status"] = "blocked"
            action["reason"] = "no_daimoi_available"
            blocked_kind = (
                "image.open.blocked" if media_kind == "image" else "audio.play.blocked"
            )
            self._emit_event_locked(
                kind=blocked_kind,
                status="blocked",
                muse_id=muse_id,
                turn_id=turn_id,
                payload={
                    "reason": "no_daimoi_available",
                    "target_node_id": target_node_id,
                },
            )
            return action

        self._emit_event_locked(
            kind="muse.daimoi.media.collided",
            status="ok",
            muse_id=muse_id,
            turn_id=turn_id,
            payload={
                "count": collisions,
                "target_node_id": target_node_id,
                "target_presence_id": target_presence_id,
                "media_kind": media_kind,
                "daimon_ids": daimon_ids[:8],
            },
        )
        if media_kind == "image":
            self._emit_event_locked(
                kind="muse.daimoi.image.collided",
                status="ok",
                muse_id=muse_id,
                turn_id=turn_id,
                payload={
                    "count": collisions,
                    "target_node_id": target_node_id,
                    "target_presence_id": target_presence_id,
                    "daimon_ids": daimon_ids[:8],
                },
            )
            self._emit_event_locked(
                kind="image.open.requested",
                status="ok",
                muse_id=muse_id,
                turn_id=turn_id,
                payload={
                    "target_node_id": target_node_id,
                    "target_presence_id": target_presence_id,
                    "url": str(action.get("selected_url", "")),
                    "label": str(action.get("selected_label", "")),
                    "source_rel_path": str(action.get("selected_source_rel_path", "")),
                    "collision_count": collisions,
                },
            )
        else:
            self._emit_event_locked(
                kind="muse.daimoi.audio.collided",
                status="ok",
                muse_id=muse_id,
                turn_id=turn_id,
                payload={
                    "count": collisions,
                    "target_node_id": target_node_id,
                    "target_presence_id": target_presence_id,
                    "daimon_ids": daimon_ids[:8],
                },
            )
            self._emit_event_locked(
                kind="audio.play.requested",
                status="ok",
                muse_id=muse_id,
                turn_id=turn_id,
                payload={
                    "target_node_id": target_node_id,
                    "target_presence_id": target_presence_id,
                    "url": str(action.get("selected_url", "")),
                    "label": str(action.get("selected_label", "")),
                    "source_rel_path": str(action.get("selected_source_rel_path", "")),
                    "collision_count": collisions,
                },
            )
        return action

    def _pin_node_locked(
        self,
        muse_id: str,
        *,
        node_id: str,
        user_intent_id: str,
        reason: str,
    ) -> dict[str, Any]:
        muse = self._state["muses"].get(str(muse_id).strip())
        clean_node = str(node_id or "").strip()
        if muse is None:
            return {"ok": False, "error": "muse_not_found"}
        if not clean_node:
            return {"ok": False, "error": "invalid_node_id"}
        pins = [str(item) for item in muse.get("pinned_node_ids", []) if str(item)]
        if clean_node in pins:
            return {
                "ok": True,
                "muse": self._public_muse_row(muse),
                "changed": False,
            }
        pins.append(clean_node)
        muse["pinned_node_ids"] = pins[:48]
        self._emit_event_locked(
            kind="muse.pin.node",
            status="ok",
            muse_id=str(muse_id),
            payload={
                "node_id": clean_node,
                "reason": str(reason or ""),
                "user_intent_id": str(user_intent_id or ""),
            },
        )
        return {
            "ok": True,
            "muse": self._public_muse_row(muse),
            "changed": True,
        }

    def _create_muse_locked(
        self,
        *,
        muse_id: str,
        label: str,
        anchor: dict[str, Any] | None,
        user_intent_id: str,
    ) -> dict[str, Any]:
        created_at = _now_iso()
        clean_id = _slugify(
            muse_id, fallback=_slugify(label or "muse", fallback="muse")
        )
        if not clean_id:
            clean_id = f"muse_{_safe_int(time.time(), 0)}"
        if clean_id in self._state["muses"]:
            return {
                "ok": False,
                "error": "muse_already_exists",
                "muse": self._public_muse_row(self._state["muses"][clean_id]),
            }

        clean_label = str(label or "").strip() or clean_id.replace("_", " ").title()
        muse = {
            "id": clean_id,
            "label": clean_label,
            "presence_type": "muse",
            "created_at": created_at,
            "status": "active",
            "anchor": _normalized_anchor(anchor),
            "panel_id": f"nexus.ui.chat.{clean_id}",
            "pinned_node_ids": [],
            "active_nexus_id": "",
            "gpu_budget": {
                "max_claims_per_turn": 1,
                "max_claims_per_hour": 120,
            },
            "gpu_state": {
                "status": "released",
                "claim_id": "",
                "device": "",
                "updated_at": created_at,
            },
        }
        self._state["muses"][clean_id] = muse
        self._state["messages"][clean_id] = []
        self._state["manifests"][clean_id] = {}
        self._state["idempotency"][clean_id] = {}
        self._state["rate_windows"][clean_id] = []

        self._emit_event_locked(
            kind="muse.created",
            status="ok",
            muse_id=clean_id,
            payload={
                "label": clean_label,
                "anchor": dict(muse["anchor"]),
                "user_intent_id": str(user_intent_id or ""),
            },
        )
        self._emit_event_locked(
            kind="muse.anchored",
            status="ok",
            muse_id=clean_id,
            payload={
                "anchor": dict(muse["anchor"]),
            },
        )
        self._emit_event_locked(
            kind="muse.panel.opened",
            status="ok",
            muse_id=clean_id,
            payload={
                "panel_id": str(muse.get("panel_id", "")),
            },
        )
        return {
            "ok": True,
            "muse": self._public_muse_row(muse),
        }

    def _unpin_node_locked(
        self,
        muse_id: str,
        *,
        node_id: str,
        user_intent_id: str,
    ) -> dict[str, Any]:
        muse = self._state["muses"].get(str(muse_id).strip())
        clean_node = str(node_id or "").strip()
        if muse is None:
            return {"ok": False, "error": "muse_not_found"}
        if not clean_node:
            return {"ok": False, "error": "invalid_node_id"}
        pins = [str(item) for item in muse.get("pinned_node_ids", []) if str(item)]
        if clean_node not in pins:
            return {
                "ok": True,
                "muse": self._public_muse_row(muse),
                "changed": False,
            }
        muse["pinned_node_ids"] = [item for item in pins if item != clean_node]
        self._emit_event_locked(
            kind="muse.unpin.node",
            status="ok",
            muse_id=str(muse_id),
            payload={
                "node_id": clean_node,
                "user_intent_id": str(user_intent_id or ""),
            },
        )
        return {
            "ok": True,
            "muse": self._public_muse_row(muse),
            "changed": True,
        }

    def _public_muse_row(self, muse: dict[str, Any]) -> dict[str, Any]:
        muse_id = str(muse.get("id", "")).strip()
        messages = self._state["messages"].get(muse_id, [])
        history_count = len(messages) if isinstance(messages, list) else 0
        return {
            "id": muse_id,
            "label": str(muse.get("label", muse_id) or muse_id),
            "presence_type": str(muse.get("presence_type", "muse") or "muse"),
            "created_at": str(muse.get("created_at", "")),
            "status": str(muse.get("status", "active")),
            "anchor": _normalized_anchor(muse.get("anchor", {})),
            "panel_id": str(muse.get("panel_id", f"nexus.ui.chat.{muse_id}")),
            "active_nexus_id": str(muse.get("active_nexus_id", "")),
            "pinned_node_ids": [
                str(item)
                for item in (
                    muse.get("pinned_node_ids", [])
                    if isinstance(muse.get("pinned_node_ids"), list)
                    else []
                )
                if str(item).strip()
            ],
            "chat_history_count": history_count,
            "gpu_state": dict(muse.get("gpu_state", {})),
        }

    def _emit_event_locked(
        self,
        *,
        kind: str,
        status: str,
        muse_id: str,
        payload: dict[str, Any] | None = None,
        turn_id: str = "",
    ) -> dict[str, Any]:
        seq = _safe_int(self._state.get("seq", 0), 0) + 1
        self._state["seq"] = seq
        event = {
            "record": MUSE_EVENT_RECORD,
            "schema_version": MUSE_EVENT_SCHEMA_VERSION,
            "event_id": f"mev:{seq}",
            "seq": seq,
            "kind": str(kind or "muse.event"),
            "status": str(status or "ok"),
            "muse_id": str(muse_id or ""),
            "turn_id": str(turn_id or ""),
            "ts": _now_iso(),
            "payload": payload if isinstance(payload, dict) else {},
        }
        events = self._state.get("events", [])
        if not isinstance(events, list):
            events = []
        events.append(event)
        if len(events) > self._max_events:
            events = events[-self._max_events :]
        self._state["events"] = events
        return event

    def _persist_locked(self) -> None:
        if not self._enabled or self._storage is None:
            return
        self._storage.save_state(self._state)

    def _coerce_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = _default_state()
        if not isinstance(payload, dict):
            return state
        state["record"] = str(payload.get("record", MUSE_STATE_FILE_RECORD))
        state["schema_version"] = str(
            payload.get("schema_version", MUSE_RUNTIME_SCHEMA_VERSION)
        )
        state["seq"] = _safe_int(payload.get("seq", 0), 0)
        for key in ("muses", "messages", "manifests", "idempotency", "rate_windows"):
            value = payload.get(key)
            state[key] = value if isinstance(value, dict) else {}
        events = payload.get("events")
        state["events"] = events if isinstance(events, list) else []
        return state

    def _ensure_bootstrap_muses_locked(self) -> None:
        for spec in BOOTSTRAP_MUSE_SPECS:
            muse_id = str(spec.get("id", "")).strip()
            if not muse_id:
                continue
            existing = self._state["muses"].get(muse_id)
            if isinstance(existing, dict):
                existing.setdefault("presence_type", "muse")
                existing.setdefault("panel_id", f"nexus.ui.chat.{muse_id}")
                if not str(existing.get("label", "")).strip():
                    existing["label"] = str(spec.get("label", muse_id) or muse_id)
                self._state["messages"].setdefault(muse_id, [])
                self._state["manifests"].setdefault(muse_id, {})
                self._state["idempotency"].setdefault(muse_id, {})
                self._state["rate_windows"].setdefault(muse_id, [])
                continue
            self._create_muse_locked(
                muse_id=muse_id,
                label=str(spec.get("label", muse_id) or muse_id),
                anchor=spec.get("anchor", {}),
                user_intent_id="bootstrap",
            )

    def _embedding_preview(self, seed: str) -> list[float]:
        vector: list[float] = []
        for index in range(6):
            unit = _hash_unit(f"{seed}|{index}")
            vector.append(round((unit * 2.0) - 1.0, 6))
        return vector

    def _recency_ms(self, ts_value: Any) -> int:
        text = str(ts_value or "").strip()
        if not text:
            return -1
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta = now - parsed.astimezone(timezone.utc)
            return max(0, int(delta.total_seconds() * 1000))
        except Exception:
            return -1


_MUSE_RUNTIME_MANAGER_LOCK = threading.Lock()
_MUSE_RUNTIME_MANAGER: MuseRuntimeManager | None = None
_MUSE_RUNTIME_MANAGER_SIGNATURE = ""


def _muse_runtime_signature_from_env() -> str:
    return "|".join(
        [
            str(os.getenv("MUSE_RUNTIME_BACKEND", "memory") or "memory").strip(),
            str(os.getenv("MUSE_RUNTIME_STATE_PATH", "") or "").strip(),
            str(os.getenv("MUSE_RUNTIME_TURNS_PER_MINUTE", "24") or "24").strip(),
            str(os.getenv("MUSE_RUNTIME_MAX_PARTICLES_PER_TURN", "18") or "18").strip(),
            str(os.getenv("MUSE_RUNTIME_MAX_EVENTS", "2400") or "2400").strip(),
            str(os.getenv("MUSE_RUNTIME_AUDIO_INTENT_ENABLED", "1") or "1").strip(),
            str(
                os.getenv("MUSE_RUNTIME_AUDIO_MIN_SCORE", str(DEFAULT_AUDIO_MIN_SCORE))
                or str(DEFAULT_AUDIO_MIN_SCORE)
            ).strip(),
            str(
                os.getenv(
                    "MUSE_RUNTIME_AUDIO_DAIMOI_FANOUT", str(DEFAULT_AUDIO_DAIMOI_FANOUT)
                )
                or str(DEFAULT_AUDIO_DAIMOI_FANOUT)
            ).strip(),
            str(
                os.getenv(
                    "MUSE_RUNTIME_AUDIO_MAX_CANDIDATES",
                    str(DEFAULT_AUDIO_MAX_CANDIDATES),
                )
                or str(DEFAULT_AUDIO_MAX_CANDIDATES)
            ).strip(),
            str(
                os.getenv(
                    "MUSE_RUNTIME_AUDIO_TARGET_PRESENCE",
                    DEFAULT_AUDIO_TARGET_PRESENCE_ID,
                )
                or DEFAULT_AUDIO_TARGET_PRESENCE_ID
            ).strip(),
            str(
                os.getenv(
                    "MUSE_RUNTIME_IMAGE_TARGET_PRESENCE",
                    DEFAULT_IMAGE_TARGET_PRESENCE_ID,
                )
                or DEFAULT_IMAGE_TARGET_PRESENCE_ID
            ).strip(),
        ]
    )


def _build_muse_runtime_manager_from_env() -> MuseRuntimeManager:
    backend = (
        str(os.getenv("MUSE_RUNTIME_BACKEND", "memory") or "memory").strip().lower()
    )
    turns_per_minute = _safe_int(
        os.getenv("MUSE_RUNTIME_TURNS_PER_MINUTE", str(DEFAULT_TURNS_PER_MINUTE)),
        DEFAULT_TURNS_PER_MINUTE,
    )
    max_particles_per_turn = _safe_int(
        os.getenv(
            "MUSE_RUNTIME_MAX_PARTICLES_PER_TURN",
            str(DEFAULT_MAX_PARTICLES_PER_TURN),
        ),
        DEFAULT_MAX_PARTICLES_PER_TURN,
    )
    max_events = _safe_int(
        os.getenv("MUSE_RUNTIME_MAX_EVENTS", str(DEFAULT_MAX_EVENTS)),
        DEFAULT_MAX_EVENTS,
    )

    storage: Any
    backend_name = backend
    if backend in {"json", "persistent", "file"}:
        raw_path = str(
            os.getenv(
                "MUSE_RUNTIME_STATE_PATH",
                "./part64/world_state/muse_runtime_state.json",
            )
            or "./part64/world_state/muse_runtime_state.json"
        ).strip()
        storage = JsonFileMuseStorage(Path(raw_path))
        backend_name = "json"
    else:
        storage = InMemoryMuseStorage()
        backend_name = "memory"

    return MuseRuntimeManager(
        storage=storage,
        enabled=True,
        backend_name=backend_name,
        turns_per_minute=turns_per_minute,
        max_particles_per_turn=max_particles_per_turn,
        max_events=max_events,
        max_history_messages=_safe_int(
            os.getenv(
                "MUSE_RUNTIME_MAX_HISTORY_MESSAGES",
                str(DEFAULT_MAX_HISTORY_MESSAGES),
            ),
            DEFAULT_MAX_HISTORY_MESSAGES,
        ),
        max_manifests_per_muse=_safe_int(
            os.getenv(
                "MUSE_RUNTIME_MAX_MANIFESTS_PER_MUSE",
                str(DEFAULT_MAX_MANIFESTS_PER_MUSE),
            ),
            DEFAULT_MAX_MANIFESTS_PER_MUSE,
        ),
        audio_intent_enabled=_safe_bool(
            os.getenv("MUSE_RUNTIME_AUDIO_INTENT_ENABLED", "1"),
            DEFAULT_AUDIO_INTENT_ENABLED,
        ),
        audio_min_score=_safe_float(
            os.getenv(
                "MUSE_RUNTIME_AUDIO_MIN_SCORE",
                str(DEFAULT_AUDIO_MIN_SCORE),
            ),
            DEFAULT_AUDIO_MIN_SCORE,
        ),
        audio_daimoi_fanout=_safe_int(
            os.getenv(
                "MUSE_RUNTIME_AUDIO_DAIMOI_FANOUT",
                str(DEFAULT_AUDIO_DAIMOI_FANOUT),
            ),
            DEFAULT_AUDIO_DAIMOI_FANOUT,
        ),
        audio_max_candidates=_safe_int(
            os.getenv(
                "MUSE_RUNTIME_AUDIO_MAX_CANDIDATES",
                str(DEFAULT_AUDIO_MAX_CANDIDATES),
            ),
            DEFAULT_AUDIO_MAX_CANDIDATES,
        ),
        audio_target_presence_id=str(
            os.getenv(
                "MUSE_RUNTIME_AUDIO_TARGET_PRESENCE",
                DEFAULT_AUDIO_TARGET_PRESENCE_ID,
            )
            or DEFAULT_AUDIO_TARGET_PRESENCE_ID
        ).strip()
        or DEFAULT_AUDIO_TARGET_PRESENCE_ID,
        image_target_presence_id=str(
            os.getenv(
                "MUSE_RUNTIME_IMAGE_TARGET_PRESENCE",
                DEFAULT_IMAGE_TARGET_PRESENCE_ID,
            )
            or DEFAULT_IMAGE_TARGET_PRESENCE_ID
        ).strip()
        or DEFAULT_IMAGE_TARGET_PRESENCE_ID,
    )


def get_muse_runtime_manager() -> MuseRuntimeManager:
    global _MUSE_RUNTIME_MANAGER, _MUSE_RUNTIME_MANAGER_SIGNATURE
    signature = _muse_runtime_signature_from_env()
    with _MUSE_RUNTIME_MANAGER_LOCK:
        if (
            _MUSE_RUNTIME_MANAGER is None
            or _MUSE_RUNTIME_MANAGER_SIGNATURE != signature
        ):
            _MUSE_RUNTIME_MANAGER = _build_muse_runtime_manager_from_env()
            _MUSE_RUNTIME_MANAGER_SIGNATURE = signature
        return _MUSE_RUNTIME_MANAGER


def reset_muse_runtime_state_for_tests() -> None:
    global _MUSE_RUNTIME_MANAGER, _MUSE_RUNTIME_MANAGER_SIGNATURE
    with _MUSE_RUNTIME_MANAGER_LOCK:
        if _MUSE_RUNTIME_MANAGER is not None:
            _MUSE_RUNTIME_MANAGER.reset()
        _MUSE_RUNTIME_MANAGER = None
        _MUSE_RUNTIME_MANAGER_SIGNATURE = ""
