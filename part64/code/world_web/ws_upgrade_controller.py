from __future__ import annotations

from http import HTTPStatus
from typing import Any, Callable


def upgrade_websocket_or_respond(
    *,
    ws_key: str,
    try_acquire_client_slot: Callable[[], bool],
    runtime_ws_client_snapshot: Callable[[], dict[str, Any]],
    send_json: Callable[..., None],
    send_bytes: Callable[..., None],
    send_upgrade_response: Callable[[str], None],
    release_client_slot: Callable[[], None],
) -> bool:
    key = str(ws_key or "").strip()
    if not key:
        send_bytes(
            b"missing websocket key",
            "text/plain; charset=utf-8",
            status=HTTPStatus.BAD_REQUEST,
        )
        return False

    if not try_acquire_client_slot():
        send_json(
            {
                "ok": False,
                "error": "websocket_capacity_reached",
                "record": "eta-mu.runtime-health.v1",
                "websocket": runtime_ws_client_snapshot(),
            },
            status=HTTPStatus.SERVICE_UNAVAILABLE,
        )
        return False

    try:
        send_upgrade_response(key)
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        release_client_slot()
        return False
    return True
