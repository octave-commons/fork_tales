from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from code.world_web import muse_mvc_controller as muse_mvc_controller_module


class _FakeMuseManager:
    def __init__(self) -> None:
        self.send_calls: list[dict[str, Any]] = []
        self.events_calls: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        return {"ok": True, "muses": []}

    def list_events(
        self, *, muse_id: str, since_seq: int, limit: int
    ) -> list[dict[str, Any]]:
        self.events_calls.append(
            {
                "muse_id": muse_id,
                "since_seq": since_seq,
                "limit": limit,
            }
        )
        return []

    def get_context_manifest(self, muse_id: str, turn_id: str) -> dict[str, Any] | None:
        _ = (muse_id, turn_id)
        return {"id": "manifest:test"}

    def create_muse(self, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    def set_pause(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    def pin_node(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    def unpin_node(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    def bind_nexus(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    def sync_workspace_pins(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    def send_message(self, **kwargs: Any) -> dict[str, Any]:
        self.send_calls.append(dict(kwargs))
        return {"ok": True, "turn_id": "turn:test"}


class _FakeHandler:
    def __init__(self) -> None:
        self.manager = _FakeMuseManager()
        self._muse_tool_cache: dict[str, Any] = {}

    def _muse_manager(self) -> _FakeMuseManager:
        return self.manager

    def _muse_threat_radar_status(self) -> dict[str, Any]:
        return {"status": "ok"}

    def _muse_threat_radar_tick(self, *, force: bool, reason: str) -> dict[str, Any]:
        _ = (force, reason)
        return {"ok": True}

    def _runtime_catalog_base(self) -> dict[str, Any]:
        return {"generated_at": "2026-03-04T00:00:00+00:00"}

    def _get_active_threat_nodes(self, radar: str) -> list[dict[str, Any]]:
        _ = radar
        return []

    def _muse_tool_callback(self, **_kwargs: Any) -> dict[str, Any]:
        return {"ok": True}

    def _muse_reply_builder(self, **_kwargs: Any) -> dict[str, Any]:
        return {"reply": "ok", "mode": "canonical", "model": "unit"}


def _server_module() -> Any:
    return SimpleNamespace(
        _safe_float=lambda value, default=0.0: (
            float(value)
            if str(value).strip() not in {"", "none", "null"}
            else float(default)
        ),
        _safe_bool_query=lambda value, default=False: (
            str(value or "").strip().lower() in {"1", "true", "yes", "on"}
            if str(value or "").strip()
            else bool(default)
        ),
    )


def test_post_message_normalizes_muse_id_before_dispatch() -> None:
    handler = _FakeHandler()
    captured: dict[str, Any] = {}

    def _send_json(payload: dict[str, Any], status: int = 200) -> None:
        captured["payload"] = payload
        captured["status"] = status

    handled = muse_mvc_controller_module.handle_muse_post_route(
        handler=handler,
        path="/api/muse/message",
        read_json_body=lambda: {
            "muse_id": " Chaos ",
            "text": "status update",
            "mode": "deterministic",
            "token_budget": 1024,
        },
        send_json=_send_json,
        headers={},
        server_module=_server_module(),
    )

    assert handled is True
    assert int(captured.get("status", 0)) == 200
    assert handler.manager.send_calls
    assert handler.manager.send_calls[0].get("muse_id") == "chaos"


def test_get_events_normalizes_muse_id_filter() -> None:
    handler = _FakeHandler()
    captured: dict[str, Any] = {}

    def _send_json(payload: dict[str, Any], status: int = 200) -> None:
        captured["payload"] = payload
        captured["status"] = status

    handled = muse_mvc_controller_module.handle_muse_get_route(
        handler=handler,
        path="/api/muse/events",
        params={"muse_id": ["Witness-Thread"], "since_seq": ["0"], "limit": ["12"]},
        send_json=_send_json,
        server_module=_server_module(),
    )

    assert handled is True
    assert int(captured.get("status", 0)) == 200
    assert handler.manager.events_calls
    assert handler.manager.events_calls[0].get("muse_id") == "witness_thread"
