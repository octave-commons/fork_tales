# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import dotenv_values, set_key, unset_key

ACCOUNT_SCHEMA_VERSION = "openai-codex-accounts.v1"
EVENT_SCHEMA_VERSION = "openai-codex-event.v1"
EVENT_LOG_FILENAME = "openai_codex_events.v1.jsonl"

SUPPORTED_PROVIDER_ALIASES = {
    "openai": "openai",
    "openai_codex": "openai",
    "codex": "openai",
}

ENV_KEY_PREFIX = {
    "openai": "OPENAI_API_KEY",
}


def normalize_provider(provider: str | None) -> str:
    value = (provider or "openai").strip().lower()
    return SUPPORTED_PROVIDER_ALIASES.get(value, "")


def provider_env_prefix(provider: str) -> str:
    prefix = ENV_KEY_PREFIX.get(provider)
    if not prefix:
        raise ValueError(f"Unsupported provider: {provider}")
    return prefix


def mask_api_key(api_key: str) -> str:
    value = api_key.strip()
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def account_id_for_key(provider: str, api_key: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return f"{provider}:{digest}"


def parse_provider_api_keys(env_file: Path, provider: str) -> List[Tuple[str, str]]:
    if not env_file.is_file():
        return []

    prefix = provider_env_prefix(provider)
    parsed = dotenv_values(str(env_file))
    indexed_rows: List[Tuple[int, str, str]] = []

    for key, raw_value in parsed.items():
        if key is None:
            continue
        value = str(raw_value or "").strip()
        if not value:
            continue

        if key == prefix:
            indexed_rows.append((0, key, value))
            continue

        if key.startswith(f"{prefix}_"):
            suffix = key[len(prefix) + 1 :]
            if suffix.isdigit():
                indexed_rows.append((int(suffix), key, value))

    indexed_rows.sort(key=lambda row: row[0])
    return [(key, value) for _, key, value in indexed_rows]


def upsert_provider_api_key(
    env_file: Path, provider: str, api_key: str
) -> Tuple[str, bool]:
    env_file.parent.mkdir(parents=True, exist_ok=True)
    if not env_file.exists():
        env_file.touch()

    current_rows = parse_provider_api_keys(env_file, provider)
    for env_key, existing_value in current_rows:
        if existing_value == api_key:
            return env_key, False

    prefix = provider_env_prefix(provider)
    used_indices = {0}
    for env_key, _ in current_rows:
        if env_key == prefix:
            used_indices.add(0)
            continue
        suffix = env_key[len(prefix) + 1 :]
        if suffix.isdigit():
            used_indices.add(int(suffix))

    next_index = max(used_indices) + 1 if used_indices else 1
    new_key = f"{prefix}_{next_index}"
    set_key(str(env_file), new_key, api_key)
    return new_key, True


def remove_provider_api_key(env_file: Path, provider: str, api_key: str) -> List[str]:
    if not env_file.is_file():
        return []

    removed: List[str] = []
    for env_key, value in parse_provider_api_keys(env_file, provider):
        if value != api_key:
            continue
        unset_key(str(env_file), env_key)
        removed.append(env_key)
    return removed


def is_valid_openai_api_key_shape(api_key: str) -> bool:
    value = api_key.strip()
    if len(value) < 20:
        return False
    if value.startswith("sk-"):
        return True
    return False


class OpenAICodexEventLog:
    def __init__(self, path: Path, max_entries: int = 400):
        self.path = path
        self.max_entries = max_entries
        self._events: List[Dict[str, Any]] = []
        self._loaded = False
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._events = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        self._events.append(parsed)
        except OSError:
            self._events = []

        if len(self._events) > self.max_entries:
            self._events = self._events[-self.max_entries :]
        self._loaded = True

    async def append_event(
        self,
        action: str,
        outcome: str,
        detail: str,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "id": f"evt_{int(time.time() * 1000)}",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": action,
            "outcome": outcome,
            "detail": detail,
            "meta": meta or {},
        }
        async with self._lock:
            await self._ensure_loaded()
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(event, separators=(",", ":"), sort_keys=False)
                    )
                    handle.write("\n")
            except OSError:
                pass
            self._events.append(event)
            if len(self._events) > self.max_entries:
                self._events = self._events[-self.max_entries :]
        return event

    async def list_events(self, limit: int = 120) -> List[Dict[str, Any]]:
        async with self._lock:
            await self._ensure_loaded()
            safe_limit = max(1, min(limit, self.max_entries))
            return list(reversed(self._events[-safe_limit:]))
