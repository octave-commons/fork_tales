from __future__ import annotations

import threading
import time
from typing import Any, Callable

from .metrics import _json_deep_clone


_STUDY_RESPONSE_CACHE_LOCK = threading.Lock()
_STUDY_RESPONSE_CACHE: dict[str, dict[str, Any]] = {}


def get_or_build_study_snapshot_response(
    *,
    cache_key: str,
    max_age_seconds: float,
    builder: Callable[[], dict[str, Any]],
    wait_timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    safe_key = str(cache_key or "").strip()
    safe_max_age = max(0.0, float(max_age_seconds or 0.0))
    safe_wait_timeout = max(0.1, float(wait_timeout_seconds or 0.1))
    if not safe_key or safe_max_age <= 0.0:
        return builder()

    now_monotonic = time.monotonic()
    owner = False
    wait_event: threading.Event | None = None

    with _STUDY_RESPONSE_CACHE_LOCK:
        entry = _STUDY_RESPONSE_CACHE.get(safe_key)
        if isinstance(entry, dict):
            checked = float(entry.get("checked_monotonic", 0.0) or 0.0)
            payload = entry.get("payload")
            if (
                isinstance(payload, dict)
                and not bool(entry.get("building", False))
                and (now_monotonic - checked) <= safe_max_age
            ):
                return _json_deep_clone(payload)
            existing_event = entry.get("event")
            if bool(entry.get("building", False)) and isinstance(
                existing_event, threading.Event
            ):
                wait_event = existing_event

        if wait_event is None:
            wait_event = threading.Event()
            _STUDY_RESPONSE_CACHE[safe_key] = {
                "checked_monotonic": now_monotonic,
                "payload": None,
                "building": True,
                "event": wait_event,
            }
            owner = True

    if owner:
        try:
            payload = builder()
        except Exception:
            with _STUDY_RESPONSE_CACHE_LOCK:
                current = _STUDY_RESPONSE_CACHE.get(safe_key)
                if isinstance(current, dict) and current.get("event") is wait_event:
                    _STUDY_RESPONSE_CACHE.pop(safe_key, None)
                    wait_event.set()
            raise

        with _STUDY_RESPONSE_CACHE_LOCK:
            _STUDY_RESPONSE_CACHE[safe_key] = {
                "checked_monotonic": time.monotonic(),
                "payload": _json_deep_clone(payload),
                "building": False,
                "event": None,
            }
            wait_event.set()
        return payload

    wait_event.wait(timeout=safe_wait_timeout)
    with _STUDY_RESPONSE_CACHE_LOCK:
        entry = _STUDY_RESPONSE_CACHE.get(safe_key)
        if isinstance(entry, dict) and not bool(entry.get("building", False)):
            payload = entry.get("payload")
            if isinstance(payload, dict):
                return _json_deep_clone(payload)
    return builder()
