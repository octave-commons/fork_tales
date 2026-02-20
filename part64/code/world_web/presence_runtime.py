from __future__ import annotations

import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any


PRESENCE_RUNTIME_RECORD = "eta-mu.presence-runtime.snapshot.v1"
PRESENCE_RUNTIME_EVENT_RECORD = "eta-mu.presence-event.v1"
PRESENCE_RUNTIME_SCHEMA_VERSION = "presence.redis.v1"
PRESENCE_RUNTIME_STREAM_RECORD = "eta-mu.presence-stream.v1"
PRESENCE_RUNTIME_BUS_STREAM = "bus:events"

_LEASE_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if (not current) or current == ARGV[1] then
  redis.call('PSETEX', KEYS[1], tonumber(ARGV[2]), ARGV[1])
  return {1, current or ''}
end
return {0, current}
"""

_PRESENCE_CAS_SCRIPT = """
local expected = tonumber(ARGV[1])
local next_ver = tonumber(ARGV[2])
local now_iso = ARGV[3]
local event_json = ARGV[4]
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if current ~= expected then
  return {0, tostring(current)}
end
for i = 5, #ARGV, 2 do
  local field = ARGV[i]
  local value = ARGV[i + 1]
  if field and value then
    redis.call('HSET', KEYS[2], field, value)
  end
end
redis.call('HSET', KEYS[2], 'updated_at', now_iso)
redis.call('SET', KEYS[1], tostring(next_ver))
redis.call('XADD', KEYS[3], '*', 'event', event_json)
return {1, tostring(next_ver)}
"""

_DAIMO_UPSERT_SCRIPT = """
local expected = tonumber(ARGV[1])
local owner = ARGV[2]
local did = ARGV[3]
local now_iso = ARGV[4]
local event_json = ARGV[5]
local current_owner = redis.call('HGET', KEYS[1], 'owner')
if current_owner and current_owner ~= owner then
  return {0, 'owner_conflict', current_owner}
end
local current_ver = tonumber(redis.call('HGET', KEYS[1], 'ver') or '0')
if expected >= 0 and current_ver ~= expected then
  return {0, 'cas_conflict', tostring(current_ver)}
end
for i = 6, #ARGV, 2 do
  local field = ARGV[i]
  local value = ARGV[i + 1]
  if field and value then
    redis.call('HSET', KEYS[1], field, value)
  end
end
redis.call('HSET', KEYS[1], 'owner', owner, 'updated_at', now_iso)
local next_ver = tonumber(redis.call('HINCRBY', KEYS[1], 'ver', 1))
redis.call('SADD', KEYS[2], did)
redis.call('XADD', KEYS[3], '*', 'event', event_json)
return {1, tostring(next_ver), owner}
"""

_DAIMO_HANDOFF_SCRIPT = """
local did = ARGV[1]
local from_owner = ARGV[2]
local to_owner = ARGV[3]
local now_iso = ARGV[4]
local event_json = ARGV[5]
local current_owner = redis.call('HGET', KEYS[1], 'owner')
if not current_owner then
  return {0, 'missing'}
end
if current_owner ~= from_owner then
  return {0, current_owner}
end
redis.call('HSET', KEYS[1], 'owner', to_owner, 'updated_at', now_iso)
local next_ver = tonumber(redis.call('HINCRBY', KEYS[1], 'ver', 1))
redis.call('SREM', KEYS[2], did)
redis.call('SADD', KEYS[3], did)
redis.call('XADD', KEYS[4], '*', 'event', event_json)
redis.call('XADD', KEYS[5], '*', 'event', event_json)
return {1, tostring(next_ver)}
"""


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


def _stable_instance_id() -> str:
    host = socket.gethostname().strip() or "host"
    return f"{host}:{os.getpid()}"


def _presence_lease_key(presence_id: str) -> str:
    return f"p:{presence_id}:lease"


def _presence_vars_key(presence_id: str) -> str:
    return f"p:{presence_id}:vars"


def _presence_ver_key(presence_id: str) -> str:
    return f"p:{presence_id}:ver"


def _presence_daimoi_key(presence_id: str) -> str:
    return f"p:{presence_id}:daimoi"


def _presence_inbox_key(presence_id: str) -> str:
    return f"p:{presence_id}:inbox"


def _daimoi_state_key(daimoi_id: str) -> str:
    return f"d:{daimoi_id}:state"


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


class InMemoryPresenceStorage:
    backend_name = "memory"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_event_id = 1
        self._presence_leases: dict[str, dict[str, Any]] = {}
        self._presence_vars: dict[str, dict[str, str]] = {}
        self._presence_ver: dict[str, int] = {}
        self._presence_daimoi: dict[str, set[str]] = {}
        self._daimoi_state: dict[str, dict[str, Any]] = {}
        self._bus_events: list[dict[str, Any]] = []
        self._inbox_events: dict[str, list[dict[str, Any]]] = {}

    def reset(self) -> None:
        with self._lock:
            self._next_event_id = 1
            self._presence_leases = {}
            self._presence_vars = {}
            self._presence_ver = {}
            self._presence_daimoi = {}
            self._daimoi_state = {}
            self._bus_events = []
            self._inbox_events = {}

    def seed_lease(self, presence_id: str, holder: str, *, expires_ms: int) -> None:
        with self._lock:
            self._presence_leases[str(presence_id)] = {
                "instance_id": str(holder),
                "expires_ms": int(expires_ms),
            }

    def acquire_or_renew_lease(
        self,
        presence_id: str,
        instance_id: str,
        ttl_ms: int,
        *,
        now_ms: int,
    ) -> tuple[bool, str]:
        pid = str(presence_id).strip()
        if not pid:
            return False, ""
        with self._lock:
            lease = self._presence_leases.get(pid)
            holder = str((lease or {}).get("instance_id", "")).strip()
            expires = _safe_int((lease or {}).get("expires_ms"), 0)
            if lease is None or expires <= now_ms or holder == instance_id:
                self._presence_leases[pid] = {
                    "instance_id": instance_id,
                    "expires_ms": now_ms + max(250, int(ttl_ms)),
                }
                return True, (holder or instance_id)
            return False, holder

    def get_presence_ver(self, presence_id: str) -> int:
        with self._lock:
            return _safe_int(self._presence_ver.get(str(presence_id), 0), 0)

    def get_presence_vars(self, presence_id: str) -> dict[str, str]:
        with self._lock:
            return dict(self._presence_vars.get(str(presence_id).strip(), {}))

    def cas_update_presence_vars(
        self,
        presence_id: str,
        *,
        expected_ver: int,
        vars_patch: dict[str, Any],
        now_iso: str,
        event_row: dict[str, Any],
    ) -> tuple[bool, int]:
        pid = str(presence_id).strip()
        if not pid:
            return False, 0
        with self._lock:
            current = _safe_int(self._presence_ver.get(pid, 0), 0)
            if current != int(expected_ver):
                return False, current
            merged = dict(self._presence_vars.get(pid, {}))
            for key, value in vars_patch.items():
                merged[str(key)] = _stringify(value)
            merged["updated_at"] = str(now_iso)
            self._presence_vars[pid] = merged
            next_ver = current + 1
            self._presence_ver[pid] = next_ver
            self._append_bus_event_locked(dict(event_row))
            return True, next_ver

    def get_daimoi_owner_ver(self, daimoi_id: str) -> tuple[str, int]:
        did = str(daimoi_id).strip()
        if not did:
            return "", 0
        with self._lock:
            row = self._daimoi_state.get(did, {})
            return (
                str(row.get("owner", "")).strip(),
                _safe_int(row.get("ver", 0), 0),
            )

    def upsert_daimoi(
        self,
        *,
        owner_presence_id: str,
        daimoi_id: str,
        expected_ver: int,
        state_patch: dict[str, Any],
        now_iso: str,
        event_row: dict[str, Any],
    ) -> tuple[bool, str, int]:
        owner = str(owner_presence_id).strip()
        did = str(daimoi_id).strip()
        if not owner or not did:
            return False, "invalid", 0
        with self._lock:
            existing = dict(self._daimoi_state.get(did, {}))
            current_owner = str(existing.get("owner", "")).strip()
            if current_owner and current_owner != owner:
                return False, "owner_conflict", _safe_int(existing.get("ver", 0), 0)
            current_ver = _safe_int(existing.get("ver", 0), 0)
            if int(expected_ver) >= 0 and current_ver != int(expected_ver):
                return False, "cas_conflict", current_ver
            next_ver = current_ver + 1
            merged = {
                **existing,
                **{str(k): _stringify(v) for k, v in state_patch.items()},
                "owner": owner,
                "updated_at": str(now_iso),
                "ver": next_ver,
            }
            self._daimoi_state[did] = merged
            self._presence_daimoi.setdefault(owner, set()).add(did)
            self._append_bus_event_locked(dict(event_row))
            return True, "ok", next_ver

    def handoff_daimoi(
        self,
        *,
        daimoi_id: str,
        from_owner_presence_id: str,
        to_owner_presence_id: str,
        now_iso: str,
        event_row: dict[str, Any],
    ) -> tuple[bool, str, int]:
        did = str(daimoi_id).strip()
        from_owner = str(from_owner_presence_id).strip()
        to_owner = str(to_owner_presence_id).strip()
        if not did or not from_owner or not to_owner:
            return False, "invalid", 0
        with self._lock:
            existing = dict(self._daimoi_state.get(did, {}))
            current_owner = str(existing.get("owner", "")).strip()
            if not current_owner:
                return False, "missing", 0
            if current_owner != from_owner:
                return False, current_owner, _safe_int(existing.get("ver", 0), 0)
            next_ver = _safe_int(existing.get("ver", 0), 0) + 1
            existing["owner"] = to_owner
            existing["updated_at"] = str(now_iso)
            existing["ver"] = next_ver
            self._daimoi_state[did] = existing
            self._presence_daimoi.setdefault(from_owner, set()).discard(did)
            self._presence_daimoi.setdefault(to_owner, set()).add(did)
            payload = dict(event_row)
            self._append_bus_event_locked(payload)
            self._inbox_events.setdefault(to_owner, []).append(
                {
                    "stream_id": self._next_stream_id_locked(),
                    "event": payload,
                }
            )
            return True, "ok", next_ver

    def append_event(
        self, event_row: dict[str, Any], *, inbox_presence_id: str = ""
    ) -> None:
        with self._lock:
            payload = dict(event_row)
            self._append_bus_event_locked(payload)
            target = str(inbox_presence_id).strip()
            if target:
                self._inbox_events.setdefault(target, []).append(
                    {
                        "stream_id": self._next_stream_id_locked(),
                        "event": payload,
                    }
                )

    def list_bus_events(self, *, limit: int = 64) -> list[dict[str, Any]]:
        with self._lock:
            if limit <= 0:
                return []
            return [dict(row) for row in self._bus_events[-limit:]]

    def inbox_events(
        self, presence_id: str, *, limit: int = 64
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._inbox_events.get(str(presence_id).strip(), [])
            if limit <= 0:
                return []
            return [dict(row) for row in rows[-limit:]]

    def presence_daimoi_ids(self, presence_id: str) -> set[str]:
        with self._lock:
            return set(self._presence_daimoi.get(str(presence_id).strip(), set()))

    def _next_stream_id_locked(self) -> str:
        raw = self._next_event_id
        self._next_event_id += 1
        return f"{_now_ms()}-{raw}"

    def _append_bus_event_locked(self, event_row: dict[str, Any]) -> None:
        payload = dict(event_row)
        payload.setdefault("record", PRESENCE_RUNTIME_EVENT_RECORD)
        payload.setdefault("schema_version", PRESENCE_RUNTIME_SCHEMA_VERSION)
        payload.setdefault("ts", _now_iso())
        self._bus_events.append(
            {
                "stream_id": self._next_stream_id_locked(),
                "event": payload,
            }
        )


class RedisPresenceStorage:
    backend_name = "redis"

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._lease_script = self._redis.register_script(_LEASE_SCRIPT)
        self._presence_cas_script = self._redis.register_script(_PRESENCE_CAS_SCRIPT)
        self._daimoi_upsert_script = self._redis.register_script(_DAIMO_UPSERT_SCRIPT)
        self._daimoi_handoff_script = self._redis.register_script(_DAIMO_HANDOFF_SCRIPT)

    def acquire_or_renew_lease(
        self,
        presence_id: str,
        instance_id: str,
        ttl_ms: int,
        *,
        now_ms: int,
    ) -> tuple[bool, str]:
        del now_ms
        pid = str(presence_id).strip()
        if not pid:
            return False, ""
        result = self._lease_script(
            keys=[_presence_lease_key(pid)],
            args=[instance_id, str(max(250, int(ttl_ms)))],
        )
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            return False, ""
        ok = _safe_int(result[0], 0) == 1
        holder = str(result[1] or "").strip()
        return ok, (holder or instance_id)

    def get_presence_ver(self, presence_id: str) -> int:
        pid = str(presence_id).strip()
        if not pid:
            return 0
        raw = self._redis.get(_presence_ver_key(pid))
        return _safe_int(raw, 0)

    def get_presence_vars(self, presence_id: str) -> dict[str, str]:
        pid = str(presence_id).strip()
        if not pid:
            return {}
        return self._redis.hgetall(_presence_vars_key(pid))

    def cas_update_presence_vars(
        self,
        presence_id: str,
        *,
        expected_ver: int,
        vars_patch: dict[str, Any],
        now_iso: str,
        event_row: dict[str, Any],
    ) -> tuple[bool, int]:
        pid = str(presence_id).strip()
        if not pid:
            return False, 0
        next_ver = int(expected_ver) + 1
        args: list[str] = [
            str(int(expected_ver)),
            str(next_ver),
            str(now_iso),
            json.dumps(event_row, ensure_ascii=False, separators=(",", ":")),
        ]
        for key, value in vars_patch.items():
            args.append(str(key))
            args.append(_stringify(value))
        result = self._presence_cas_script(
            keys=[
                _presence_ver_key(pid),
                _presence_vars_key(pid),
                PRESENCE_RUNTIME_BUS_STREAM,
            ],
            args=args,
        )
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            return False, self.get_presence_ver(pid)
        ok = _safe_int(result[0], 0) == 1
        ver = _safe_int(result[1], self.get_presence_ver(pid))
        return ok, ver

    def get_daimoi_owner_ver(self, daimoi_id: str) -> tuple[str, int]:
        did = str(daimoi_id).strip()
        if not did:
            return "", 0
        owner, ver = self._redis.hmget(_daimoi_state_key(did), ["owner", "ver"])
        return str(owner or "").strip(), _safe_int(ver, 0)

    def upsert_daimoi(
        self,
        *,
        owner_presence_id: str,
        daimoi_id: str,
        expected_ver: int,
        state_patch: dict[str, Any],
        now_iso: str,
        event_row: dict[str, Any],
    ) -> tuple[bool, str, int]:
        owner = str(owner_presence_id).strip()
        did = str(daimoi_id).strip()
        if not owner or not did:
            return False, "invalid", 0
        args: list[str] = [
            str(int(expected_ver)),
            owner,
            did,
            str(now_iso),
            json.dumps(event_row, ensure_ascii=False, separators=(",", ":")),
        ]
        for key, value in state_patch.items():
            args.append(str(key))
            args.append(_stringify(value))
        result = self._daimoi_upsert_script(
            keys=[
                _daimoi_state_key(did),
                _presence_daimoi_key(owner),
                PRESENCE_RUNTIME_BUS_STREAM,
            ],
            args=args,
        )
        if not isinstance(result, (list, tuple)) or len(result) < 3:
            return False, "unknown", 0
        ok = _safe_int(result[0], 0) == 1
        reason = str(result[1] or "") if not ok else "ok"
        ver = _safe_int(result[1 if ok else 2], 0)
        return ok, reason, ver

    def handoff_daimoi(
        self,
        *,
        daimoi_id: str,
        from_owner_presence_id: str,
        to_owner_presence_id: str,
        now_iso: str,
        event_row: dict[str, Any],
    ) -> tuple[bool, str, int]:
        did = str(daimoi_id).strip()
        from_owner = str(from_owner_presence_id).strip()
        to_owner = str(to_owner_presence_id).strip()
        if not did or not from_owner or not to_owner:
            return False, "invalid", 0
        result = self._daimoi_handoff_script(
            keys=[
                _daimoi_state_key(did),
                _presence_daimoi_key(from_owner),
                _presence_daimoi_key(to_owner),
                _presence_inbox_key(to_owner),
                PRESENCE_RUNTIME_BUS_STREAM,
            ],
            args=[
                did,
                from_owner,
                to_owner,
                str(now_iso),
                json.dumps(event_row, ensure_ascii=False, separators=(",", ":")),
            ],
        )
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            return False, "unknown", 0
        ok = _safe_int(result[0], 0) == 1
        reason = "ok" if ok else str(result[1] or "")
        ver = _safe_int(result[1], 0) if ok else 0
        return ok, reason, ver

    def append_event(
        self, event_row: dict[str, Any], *, inbox_presence_id: str = ""
    ) -> None:
        payload = {
            "record": PRESENCE_RUNTIME_STREAM_RECORD,
            "event": json.dumps(event_row, ensure_ascii=False, separators=(",", ":")),
        }
        self._redis.xadd(PRESENCE_RUNTIME_BUS_STREAM, payload)
        target = str(inbox_presence_id).strip()
        if target:
            self._redis.xadd(_presence_inbox_key(target), payload)

    def list_bus_events(self, *, limit: int = 64) -> list[dict[str, Any]]:
        rows = self._redis.xrevrange(
            PRESENCE_RUNTIME_BUS_STREAM, count=max(1, int(limit))
        )
        rows.reverse()
        parsed: list[dict[str, Any]] = []
        for stream_id, values in rows:
            raw_event = ""
            if isinstance(values, dict):
                raw_event = str(values.get("event", ""))
            event_payload: dict[str, Any] = {}
            if raw_event:
                try:
                    parsed_event = json.loads(raw_event)
                    if isinstance(parsed_event, dict):
                        event_payload = parsed_event
                except json.JSONDecodeError:
                    event_payload = {"raw": raw_event}
            parsed.append({"stream_id": str(stream_id), "event": event_payload})
        return parsed

    def inbox_events(
        self, presence_id: str, *, limit: int = 64
    ) -> list[dict[str, Any]]:
        rows = self._redis.xrevrange(
            _presence_inbox_key(str(presence_id).strip()),
            count=max(1, int(limit)),
        )
        rows.reverse()
        parsed: list[dict[str, Any]] = []
        for stream_id, values in rows:
            raw_event = ""
            if isinstance(values, dict):
                raw_event = str(values.get("event", ""))
            event_payload: dict[str, Any] = {}
            if raw_event:
                try:
                    parsed_event = json.loads(raw_event)
                    if isinstance(parsed_event, dict):
                        event_payload = parsed_event
                except json.JSONDecodeError:
                    event_payload = {"raw": raw_event}
            parsed.append({"stream_id": str(stream_id), "event": event_payload})
        return parsed

    def presence_daimoi_ids(self, presence_id: str) -> set[str]:
        rows = self._redis.smembers(_presence_daimoi_key(str(presence_id).strip()))
        return {str(row) for row in rows}


class PresenceRuntimeManager:
    def __init__(
        self,
        *,
        storage: Any | None,
        enabled: bool,
        backend_name: str,
        instance_id: str,
        lease_ttl_ms: int,
        max_writes_per_presence: int,
        fallback_reason: str = "",
    ) -> None:
        self._storage = storage
        self._enabled = bool(enabled and storage is not None)
        self._backend_name = str(backend_name)
        self._instance_id = str(instance_id)
        self._lease_ttl_ms = max(250, int(lease_ttl_ms))
        self._max_writes_per_presence = max(1, int(max_writes_per_presence))
        self._fallback_reason = str(fallback_reason or "")
        self._lock = threading.Lock()
        self._presence_paused: dict[str, bool] = {}
        self._particle_fingerprint: dict[str, str] = {}
        self._last_particle_xy: dict[str, tuple[float, float, float]] = {}

    def reset(self) -> None:
        with self._lock:
            self._presence_paused = {}
            self._particle_fingerprint = {}
            self._last_particle_xy = {}
            storage = self._storage
            if storage is not None and hasattr(storage, "reset"):
                storage.reset()

    def get_state(self, presence_id: str) -> dict[str, Any]:
        if not self._enabled or self._storage is None:
            return {}
        raw = self._storage.get_presence_vars(presence_id)
        if not isinstance(raw, dict):
            return {}
        # Parse JSON values if possible, otherwise string
        parsed = {}
        for k, v in raw.items():
            try:
                parsed[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                parsed[k] = v
        return parsed

    def sync(
        self,
        *,
        field_particles: list[dict[str, Any]],
        presence_impacts: list[dict[str, Any]],
        queue_ratio: float,
        resource_ratio: float,
    ) -> dict[str, Any]:
        now_iso = _now_iso()
        now_ms = _now_ms()
        now_secs = time.time()

        if not self._enabled or self._storage is None:
            return {
                "record": PRESENCE_RUNTIME_RECORD,
                "schema_version": PRESENCE_RUNTIME_SCHEMA_VERSION,
                "enabled": False,
                "backend": self._backend_name,
                "instance_id": self._instance_id,
                "generated_at": now_iso,
                "streams": {
                    "bus": PRESENCE_RUNTIME_BUS_STREAM,
                    "inbox_template": "p:{pid}:inbox",
                },
                "counts": {
                    "presences": 0,
                    "active_writers": 0,
                    "paused": 0,
                    "resumed": 0,
                    "presence_updates": 0,
                    "daimoi_updates": 0,
                    "handoffs": 0,
                    "deduped": 0,
                    "rate_limited": 0,
                    "compliance_blocked": 0,
                    "events_emitted": 0,
                },
                "fallback_reason": self._fallback_reason,
            }

        presence_ids: set[str] = {
            str(row.get("id", "")).strip()
            for row in (presence_impacts if isinstance(presence_impacts, list) else [])
            if isinstance(row, dict) and str(row.get("id", "")).strip()
        }
        particles_by_presence: dict[str, list[dict[str, Any]]] = {}
        compliance_blocked = 0

        for particle in field_particles if isinstance(field_particles, list) else []:
            if not isinstance(particle, dict):
                continue
            particle_id = str(particle.get("id", "")).strip()
            presence_id = str(particle.get("presence_id", "")).strip()
            if not particle_id or not presence_id:
                compliance_blocked += 1
                self._emit_event(
                    {
                        "kind": "presence.compliance.blocked",
                        "status": "blocked",
                        "reason": "missing_particle_identity",
                        "ts": now_iso,
                    }
                )
                continue
            presence_ids.add(presence_id)
            particles_by_presence.setdefault(presence_id, []).append(particle)

        leases_granted = 0
        paused_count = 0
        resumed_count = 0
        active_writers: set[str] = set()

        for presence_id in sorted(pid for pid in presence_ids if pid):
            ok, holder = self._storage.acquire_or_renew_lease(
                presence_id,
                self._instance_id,
                self._lease_ttl_ms,
                now_ms=now_ms,
            )
            if ok:
                active_writers.add(presence_id)
                leases_granted += 1
                self._emit_event(
                    {
                        "kind": "presence.lease.granted",
                        "status": "ok",
                        "presence_id": presence_id,
                        "holder": self._instance_id,
                        "ttl_ms": self._lease_ttl_ms,
                        "ts": now_iso,
                    }
                )
                if self._presence_paused.get(presence_id, False):
                    self._presence_paused[presence_id] = False
                    resumed_count += 1
                    self._emit_event(
                        {
                            "kind": "presence.writer.resumed",
                            "status": "ok",
                            "presence_id": presence_id,
                            "reason": "lease_recovered",
                            "ts": now_iso,
                        }
                    )
            else:
                self._emit_event(
                    {
                        "kind": "presence.lease.denied",
                        "status": "blocked",
                        "presence_id": presence_id,
                        "holder": holder,
                        "ts": now_iso,
                    }
                )
                if not self._presence_paused.get(presence_id, False):
                    self._presence_paused[presence_id] = True
                    paused_count += 1
                    self._emit_event(
                        {
                            "kind": "presence.writer.paused",
                            "status": "blocked",
                            "presence_id": presence_id,
                            "holder": holder,
                            "ts": now_iso,
                        }
                    )

        presence_updates = 0
        daimoi_updates = 0
        handoffs = 0
        deduped = 0
        rate_limited = 0
        emitted_events = 0

        for presence_id in sorted(active_writers):
            rows = particles_by_presence.get(presence_id, [])
            count = len(rows)
            if count:
                centroid_x = round(
                    sum(_clamp01(_safe_float(row.get("x", 0.5), 0.5)) for row in rows)
                    / float(count),
                    6,
                )
                centroid_y = round(
                    sum(_clamp01(_safe_float(row.get("y", 0.5), 0.5)) for row in rows)
                    / float(count),
                    6,
                )
                average_size = round(
                    sum(
                        max(0.6, _safe_float(row.get("size", 1.0), 1.0)) for row in rows
                    )
                    / float(count),
                    6,
                )
            else:
                centroid_x = 0.5
                centroid_y = 0.5
                average_size = 0.0

            vars_patch = {
                "record": PRESENCE_RUNTIME_SCHEMA_VERSION,
                "particle_count": count,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "avg_size": average_size,
                "queue_ratio": round(_clamp01(_safe_float(queue_ratio, 0.0)), 6),
                "resource_ratio": round(_clamp01(_safe_float(resource_ratio, 0.0)), 6),
                "updated_at": now_iso,
            }
            # Inject wallet for persistence
            if "resource_wallet" in vars_patch:
                pass  # Already included if passed in impacts?

            # We need to ensure wallet from impact object is saved to vars_patch
            # Find the impact object
            impact = next(
                (row for row in presence_impacts if str(row.get("id")) == presence_id),
                None,
            )
            if impact and "resource_wallet" in impact:
                vars_patch["resource_wallet"] = impact["resource_wallet"]

            expected_ver = self._storage.get_presence_ver(presence_id)
            update_event = {
                "kind": "presence.vars.updated",
                "status": "ok",
                "presence_id": presence_id,
                "expected_ver": expected_ver,
                "ts": now_iso,
            }
            ok, next_ver = self._storage.cas_update_presence_vars(
                presence_id,
                expected_ver=expected_ver,
                vars_patch=vars_patch,
                now_iso=now_iso,
                event_row=update_event,
            )
            if not ok:
                retry_expected = self._storage.get_presence_ver(presence_id)
                retry_event = {
                    "kind": "presence.vars.cas-retry",
                    "status": "retry",
                    "presence_id": presence_id,
                    "expected_ver": retry_expected,
                    "ts": now_iso,
                }
                ok, next_ver = self._storage.cas_update_presence_vars(
                    presence_id,
                    expected_ver=retry_expected,
                    vars_patch=vars_patch,
                    now_iso=now_iso,
                    event_row=retry_event,
                )
                if not ok:
                    compliance_blocked += 1
                    self._emit_event(
                        {
                            "kind": "presence.compliance.blocked",
                            "status": "blocked",
                            "presence_id": presence_id,
                            "reason": "presence_vars_cas_conflict",
                            "observed_ver": next_ver,
                            "ts": now_iso,
                        }
                    )
            if ok:
                presence_updates += 1

            writes_for_presence = 0
            dedupe_for_presence = 0
            rate_for_presence = 0

            for row in rows:
                daimoi_id = str(row.get("id", "")).strip()
                if not daimoi_id:
                    compliance_blocked += 1
                    self._emit_event(
                        {
                            "kind": "presence.compliance.blocked",
                            "status": "blocked",
                            "presence_id": presence_id,
                            "reason": "missing_daimoi_id",
                            "ts": now_iso,
                        }
                    )
                    continue

                if writes_for_presence >= self._max_writes_per_presence:
                    rate_limited += 1
                    rate_for_presence += 1
                    continue

                x = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
                y = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
                size = max(0.6, _safe_float(row.get("size", 1.0), 1.0))
                r = _clamp01(_safe_float(row.get("r", 0.4), 0.4))
                g = _clamp01(_safe_float(row.get("g", 0.4), 0.4))
                b = _clamp01(_safe_float(row.get("b", 0.4), 0.4))

                fingerprint = "|".join(
                    [
                        f"{presence_id}",
                        f"{x:.5f}",
                        f"{y:.5f}",
                        f"{size:.5f}",
                        f"{r:.5f}",
                        f"{g:.5f}",
                        f"{b:.5f}",
                    ]
                )
                if self._particle_fingerprint.get(daimoi_id) == fingerprint:
                    deduped += 1
                    dedupe_for_presence += 1
                    continue

                prev_xy = self._last_particle_xy.get(daimoi_id)
                if prev_xy is None:
                    vel_x = 0.0
                    vel_y = 0.0
                else:
                    px, py, pts = prev_xy
                    dt = max(0.001, now_secs - max(0.0, _safe_float(pts, 0.0)))
                    vel_x = (x - _clamp01(_safe_float(px, x))) / dt
                    vel_y = (y - _clamp01(_safe_float(py, y))) / dt

                owner, owner_ver = self._storage.get_daimoi_owner_ver(daimoi_id)
                if owner and owner != presence_id:
                    handoff_event = {
                        "kind": "daimoi.owner.handoff",
                        "status": "ok",
                        "presence_id": presence_id,
                        "from_owner": owner,
                        "to_owner": presence_id,
                        "daimoi_id": daimoi_id,
                        "reason": "presence_domain_shift",
                        "ts": now_iso,
                    }
                    handoff_ok, handoff_reason, handoff_ver = (
                        self._storage.handoff_daimoi(
                            daimoi_id=daimoi_id,
                            from_owner_presence_id=owner,
                            to_owner_presence_id=presence_id,
                            now_iso=now_iso,
                            event_row=handoff_event,
                        )
                    )
                    if not handoff_ok:
                        compliance_blocked += 1
                        self._emit_event(
                            {
                                "kind": "presence.compliance.blocked",
                                "status": "blocked",
                                "presence_id": presence_id,
                                "daimoi_id": daimoi_id,
                                "reason": f"handoff_failed:{handoff_reason}",
                                "ts": now_iso,
                            }
                        )
                        continue
                    handoffs += 1
                    owner_ver = handoff_ver

                state_patch = {
                    "record": PRESENCE_RUNTIME_SCHEMA_VERSION,
                    "owner": presence_id,
                    "presence_id": presence_id,
                    "x": round(x, 6),
                    "y": round(y, 6),
                    "vel_x": round(vel_x, 6),
                    "vel_y": round(vel_y, 6),
                    "size": round(size, 6),
                    "mass": round(max(0.35, size * 0.82), 6),
                    "radius": round(max(0.2, size * 0.48), 6),
                    "r": round(r, 6),
                    "g": round(g, 6),
                    "b": round(b, 6),
                    "updated_at": now_iso,
                }
                upsert_event = {
                    "kind": "daimoi.state.updated",
                    "status": "ok",
                    "presence_id": presence_id,
                    "daimoi_id": daimoi_id,
                    "expected_ver": owner_ver,
                    "ts": now_iso,
                }
                upsert_ok, upsert_reason, upsert_ver = self._storage.upsert_daimoi(
                    owner_presence_id=presence_id,
                    daimoi_id=daimoi_id,
                    expected_ver=owner_ver,
                    state_patch=state_patch,
                    now_iso=now_iso,
                    event_row=upsert_event,
                )
                if not upsert_ok and upsert_reason == "cas_conflict":
                    _, retry_ver = self._storage.get_daimoi_owner_ver(daimoi_id)
                    retry_event = {
                        "kind": "daimoi.state.cas-retry",
                        "status": "retry",
                        "presence_id": presence_id,
                        "daimoi_id": daimoi_id,
                        "expected_ver": retry_ver,
                        "ts": now_iso,
                    }
                    upsert_ok, upsert_reason, upsert_ver = self._storage.upsert_daimoi(
                        owner_presence_id=presence_id,
                        daimoi_id=daimoi_id,
                        expected_ver=retry_ver,
                        state_patch=state_patch,
                        now_iso=now_iso,
                        event_row=retry_event,
                    )

                if upsert_ok:
                    daimoi_updates += 1
                    writes_for_presence += 1
                    self._particle_fingerprint[daimoi_id] = fingerprint
                    self._last_particle_xy[daimoi_id] = (x, y, now_secs)
                    continue

                compliance_blocked += 1
                self._emit_event(
                    {
                        "kind": "presence.compliance.blocked",
                        "status": "blocked",
                        "presence_id": presence_id,
                        "daimoi_id": daimoi_id,
                        "reason": f"daimoi_write_failed:{upsert_reason}",
                        "observed_ver": upsert_ver,
                        "ts": now_iso,
                    }
                )

            if dedupe_for_presence > 0:
                self._emit_event(
                    {
                        "kind": "daimoi.write.deduped",
                        "status": "ok",
                        "presence_id": presence_id,
                        "count": dedupe_for_presence,
                        "ts": now_iso,
                    }
                )
            if rate_for_presence > 0:
                self._emit_event(
                    {
                        "kind": "presence.write.rate-limited",
                        "status": "blocked",
                        "presence_id": presence_id,
                        "dropped": rate_for_presence,
                        "limit": self._max_writes_per_presence,
                        "ts": now_iso,
                    }
                )

        if hasattr(self._storage, "list_bus_events"):
            bus_rows = self._storage.list_bus_events(limit=16)
            emitted_events = len(bus_rows)

        paused_total = len(
            [pid for pid, paused in self._presence_paused.items() if bool(paused)]
        )

        return {
            "record": PRESENCE_RUNTIME_RECORD,
            "schema_version": PRESENCE_RUNTIME_SCHEMA_VERSION,
            "enabled": True,
            "backend": self._backend_name,
            "instance_id": self._instance_id,
            "generated_at": now_iso,
            "lease_ttl_ms": self._lease_ttl_ms,
            "streams": {
                "bus": PRESENCE_RUNTIME_BUS_STREAM,
                "inbox_template": "p:{pid}:inbox",
            },
            "counts": {
                "presences": len([pid for pid in presence_ids if pid]),
                "active_writers": len(active_writers),
                "leases_granted": leases_granted,
                "paused": paused_count,
                "paused_total": paused_total,
                "resumed": resumed_count,
                "presence_updates": presence_updates,
                "daimoi_updates": daimoi_updates,
                "handoffs": handoffs,
                "deduped": deduped,
                "rate_limited": rate_limited,
                "compliance_blocked": compliance_blocked,
                "events_emitted": emitted_events,
            },
            "fallback_reason": self._fallback_reason,
        }

    def _emit_event(
        self, event_row: dict[str, Any], *, inbox_presence_id: str = ""
    ) -> None:
        if self._storage is None:
            return
        payload = dict(event_row)
        payload.setdefault("record", PRESENCE_RUNTIME_EVENT_RECORD)
        payload.setdefault("schema_version", PRESENCE_RUNTIME_SCHEMA_VERSION)
        payload.setdefault("ts", _now_iso())
        try:
            self._storage.append_event(payload, inbox_presence_id=inbox_presence_id)
        except Exception:
            return


_RUNTIME_MANAGER_LOCK = threading.Lock()
_RUNTIME_MANAGER: PresenceRuntimeManager | None = None
_RUNTIME_MANAGER_SIGNATURE = ""


def _presence_runtime_signature_from_env() -> str:
    return "|".join(
        [
            str(os.getenv("PRESENCE_RUNTIME_BACKEND", "memory") or "memory").strip(),
            str(os.getenv("PRESENCE_RUNTIME_REDIS_URL", "") or "").strip(),
            str(os.getenv("REDIS_URL", "") or "").strip(),
            str(os.getenv("PRESENCE_RUNTIME_INSTANCE_ID", "") or "").strip(),
            str(os.getenv("PRESENCE_RUNTIME_LEASE_MS", "5000") or "5000").strip(),
            str(
                os.getenv("PRESENCE_RUNTIME_MAX_WRITES_PER_PRESENCE", "64") or "64"
            ).strip(),
        ]
    )


def _build_runtime_manager_from_env() -> PresenceRuntimeManager:
    backend = str(os.getenv("PRESENCE_RUNTIME_BACKEND", "memory") or "memory").strip()
    backend = backend.lower()
    instance_id = str(os.getenv("PRESENCE_RUNTIME_INSTANCE_ID", "") or "").strip()
    if not instance_id:
        instance_id = _stable_instance_id()
    lease_ttl_ms = _safe_int(os.getenv("PRESENCE_RUNTIME_LEASE_MS", "5000"), 5000)
    max_writes = _safe_int(
        os.getenv("PRESENCE_RUNTIME_MAX_WRITES_PER_PRESENCE", "64"), 64
    )

    if backend in {"disabled", "off", "0", "false", "none"}:
        return PresenceRuntimeManager(
            storage=None,
            enabled=False,
            backend_name="disabled",
            instance_id=instance_id,
            lease_ttl_ms=lease_ttl_ms,
            max_writes_per_presence=max_writes,
            fallback_reason="runtime_disabled",
        )

    if backend == "redis":
        redis_url = str(
            os.getenv("PRESENCE_RUNTIME_REDIS_URL", "")
            or os.getenv("REDIS_URL", "")
            or "redis://127.0.0.1:6379/0"
        ).strip()
        try:
            import redis as redis_lib  # type: ignore[import-not-found]

            client = redis_lib.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            return PresenceRuntimeManager(
                storage=RedisPresenceStorage(client),
                enabled=True,
                backend_name="redis",
                instance_id=instance_id,
                lease_ttl_ms=lease_ttl_ms,
                max_writes_per_presence=max_writes,
            )
        except Exception as exc:
            fallback = InMemoryPresenceStorage()
            return PresenceRuntimeManager(
                storage=fallback,
                enabled=True,
                backend_name="memory",
                instance_id=instance_id,
                lease_ttl_ms=lease_ttl_ms,
                max_writes_per_presence=max_writes,
                fallback_reason=f"redis_unavailable:{exc.__class__.__name__}",
            )

    fallback = InMemoryPresenceStorage()
    return PresenceRuntimeManager(
        storage=fallback,
        enabled=True,
        backend_name="memory",
        instance_id=instance_id,
        lease_ttl_ms=lease_ttl_ms,
        max_writes_per_presence=max_writes,
    )


def get_presence_runtime_manager() -> PresenceRuntimeManager:
    global _RUNTIME_MANAGER, _RUNTIME_MANAGER_SIGNATURE
    signature = _presence_runtime_signature_from_env()
    with _RUNTIME_MANAGER_LOCK:
        if _RUNTIME_MANAGER is None or _RUNTIME_MANAGER_SIGNATURE != signature:
            _RUNTIME_MANAGER = _build_runtime_manager_from_env()
            _RUNTIME_MANAGER_SIGNATURE = signature
        return _RUNTIME_MANAGER


def reset_presence_runtime_state_for_tests() -> None:
    global _RUNTIME_MANAGER, _RUNTIME_MANAGER_SIGNATURE
    with _RUNTIME_MANAGER_LOCK:
        if _RUNTIME_MANAGER is not None:
            _RUNTIME_MANAGER.reset()
        _RUNTIME_MANAGER = None
        _RUNTIME_MANAGER_SIGNATURE = ""


def sync_presence_runtime_state(
    *,
    field_particles: list[dict[str, Any]],
    presence_impacts: list[dict[str, Any]],
    queue_ratio: float,
    resource_ratio: float,
) -> dict[str, Any]:
    manager = get_presence_runtime_manager()
    return manager.sync(
        field_particles=field_particles,
        presence_impacts=presence_impacts,
        queue_ratio=queue_ratio,
        resource_ratio=resource_ratio,
    )


def simulation_fingerprint(simulation: dict[str, Any]) -> str:
    if not isinstance(simulation, dict):
        return ""
    rows: list[str] = []
    for key in (
        "timestamp",
        "total",
        "audio",
        "image",
        "video",
        "perspective",
    ):
        rows.append(f"{key}:{simulation.get(key)}")
    dynamics = simulation.get("presence_dynamics", {})
    if isinstance(dynamics, dict):
        runtime = dynamics.get("distributed_runtime", {})
        runtime_counts = runtime.get("counts", {}) if isinstance(runtime, dict) else {}
        rows.append(
            "runtime:"
            + ",".join(
                [
                    str(runtime.get("backend", "")),
                    str(runtime_counts.get("daimoi_updates", 0)),
                    str(runtime_counts.get("handoffs", 0)),
                    str(runtime_counts.get("deduped", 0)),
                ]
            )
        )
        rows.append(
            "particles:"
            + str(len(dynamics.get("field_particles", [])))
            + ":"
            + str(dynamics.get("field_particles_record", ""))
        )
    rows.append("points:" + str(len(simulation.get("points", []))))
    rows.append("daimoi:" + str(simulation.get("daimoi", {}).get("record", "")))
    return sha1("\n".join(rows).encode("utf-8")).hexdigest()
