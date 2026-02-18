# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass
class SessionAffinityEntry:
    provider: str
    session_id: str
    credential: str
    updated_at: float
    hit_count: int = 0


class SessionAffinityCache:
    """
    In-memory affinity cache that keeps session -> credential bindings stable.

    This cache is intentionally process-local and conservative:
    - idle entries are evicted after idle_ttl_seconds
    - total entries are capped by max_entries (oldest evicted first)
    """

    def __init__(self, max_entries: int = 2000, idle_ttl_seconds: int = 7200):
        self.max_entries = max(1, int(max_entries))
        self.idle_ttl_seconds = max(60, int(idle_ttl_seconds))
        self._entries: Dict[Tuple[str, str], SessionAffinityEntry] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize(value: str | None) -> str:
        if value is None:
            return ""
        return str(value).strip()

    async def get(
        self,
        provider: str,
        session_id: str,
        allowed_credentials: Optional[Iterable[str]] = None,
    ) -> Optional[str]:
        normalized_provider = self._normalize(provider).lower()
        normalized_session = self._normalize(session_id)
        if not normalized_provider or not normalized_session:
            return None

        now = time.time()
        allowed = set(allowed_credentials) if allowed_credentials is not None else None

        async with self._lock:
            self._prune_unlocked(now)
            key = (normalized_provider, normalized_session)
            entry = self._entries.get(key)
            if not entry:
                return None

            if allowed is not None and entry.credential not in allowed:
                del self._entries[key]
                return None

            entry.updated_at = now
            entry.hit_count += 1
            return entry.credential

    async def set(self, provider: str, session_id: str, credential: str) -> bool:
        normalized_provider = self._normalize(provider).lower()
        normalized_session = self._normalize(session_id)
        normalized_credential = self._normalize(credential)
        if (
            not normalized_provider
            or not normalized_session
            or not normalized_credential
        ):
            return False

        now = time.time()
        async with self._lock:
            self._prune_unlocked(now)
            self._entries[(normalized_provider, normalized_session)] = (
                SessionAffinityEntry(
                    provider=normalized_provider,
                    session_id=normalized_session,
                    credential=normalized_credential,
                    updated_at=now,
                )
            )
            self._enforce_capacity_unlocked()
            return True

    async def clear(
        self,
        provider: str,
        session_id: str,
        credential: str | None = None,
    ) -> bool:
        normalized_provider = self._normalize(provider).lower()
        normalized_session = self._normalize(session_id)
        normalized_credential = self._normalize(credential)
        if not normalized_provider or not normalized_session:
            return False

        key = (normalized_provider, normalized_session)
        async with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return False
            if normalized_credential and entry.credential != normalized_credential:
                return False
            del self._entries[key]
            return True

    async def clear_credential(self, provider: str, credential: str) -> int:
        normalized_provider = self._normalize(provider).lower()
        normalized_credential = self._normalize(credential)
        if not normalized_provider or not normalized_credential:
            return 0

        removed = 0
        async with self._lock:
            keys_to_remove = [
                key
                for key, entry in self._entries.items()
                if entry.provider == normalized_provider
                and entry.credential == normalized_credential
            ]
            for key in keys_to_remove:
                del self._entries[key]
                removed += 1
        return removed

    async def snapshot(
        self,
        provider: str | None = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        normalized_provider = self._normalize(provider).lower()
        safe_limit = max(1, min(int(limit), self.max_entries))
        now = time.time()

        async with self._lock:
            self._prune_unlocked(now)
            rows = list(self._entries.values())
            if normalized_provider:
                rows = [row for row in rows if row.provider == normalized_provider]

            rows.sort(key=lambda row: row.updated_at, reverse=True)
            rows = rows[:safe_limit]
            return {
                "enabled": True,
                "provider": normalized_provider or None,
                "entry_count": len(rows),
                "max_entries": self.max_entries,
                "idle_ttl_seconds": self.idle_ttl_seconds,
                "sessions": [
                    {
                        "provider": row.provider,
                        "session_id": row.session_id,
                        "credential": row.credential,
                        "updated_at": row.updated_at,
                        "hit_count": row.hit_count,
                    }
                    for row in rows
                ],
            }

    async def size(self) -> int:
        async with self._lock:
            self._prune_unlocked(time.time())
            return len(self._entries)

    def _prune_unlocked(self, now: float) -> None:
        expiry = now - self.idle_ttl_seconds
        stale_keys = [
            key for key, entry in self._entries.items() if entry.updated_at < expiry
        ]
        for key in stale_keys:
            del self._entries[key]
        self._enforce_capacity_unlocked()

    def _enforce_capacity_unlocked(self) -> None:
        if len(self._entries) <= self.max_entries:
            return

        ordered = sorted(self._entries.items(), key=lambda row: row[1].updated_at)
        overflow = len(self._entries) - self.max_entries
        for key, _ in ordered[:overflow]:
            if key in self._entries:
                del self._entries[key]
