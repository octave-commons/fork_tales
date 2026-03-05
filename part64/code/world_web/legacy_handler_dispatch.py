from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, cast


@dataclass
class DirectDispatchResponse:
    body: bytes
    content_type: str
    status_code: int
    extra_headers: dict[str, str]


def dispatch_via_legacy_handler(
    *,
    handler_class: Any,
    method: str,
    path: str,
    headers: dict[str, str],
    body: bytes,
) -> DirectDispatchResponse | None:
    method_key = str(method or "GET").strip().upper() or "GET"
    if method_key not in {"GET", "POST", "DELETE"}:
        return None

    request_headers = {str(key): str(value) for key, value in dict(headers).items()}
    if body and "Content-Length" not in request_headers:
        request_headers["Content-Length"] = str(len(body))

    handler = handler_class.__new__(handler_class)
    handler.path = str(path or "/")
    handler.command = method_key
    handler.headers = cast(Any, request_headers)
    handler.rfile = io.BytesIO(body)

    captured: dict[str, Any] = {}

    def _capture_send_bytes(
        payload: bytes,
        content_type: str,
        status: int = 200,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        status_raw = int(getattr(status, "value", status))
        captured["body"] = bytes(payload)
        captured["content_type"] = str(content_type or "application/octet-stream")
        captured["status_code"] = status_raw
        captured["extra_headers"] = {
            str(key): str(value) for key, value in (extra_headers or {}).items()
        }

    def _capture_send_json(payload: dict[str, Any], status: int = 200) -> None:
        _capture_send_bytes(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            ),
            "application/json; charset=utf-8",
            status=status,
        )

    handler._send_bytes = _capture_send_bytes
    handler._send_json = _capture_send_json

    if method_key == "GET":
        handler.do_GET()
    elif method_key == "POST":
        handler.do_POST()
    elif method_key == "DELETE":
        handler.do_DELETE()

    if not captured:
        return None

    return DirectDispatchResponse(
        body=bytes(captured.get("body", b"")),
        content_type=str(
            captured.get("content_type", "application/json; charset=utf-8")
        ),
        status_code=int(captured.get("status_code", 200)),
        extra_headers={
            str(key): str(value)
            for key, value in dict(captured.get("extra_headers", {})).items()
        },
    )
