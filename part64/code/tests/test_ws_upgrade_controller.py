from __future__ import annotations

from typing import Any

from code.world_web import simulation_docker_ws_controller as docker_ws_module
from code.world_web import ws_upgrade_controller as ws_upgrade_module


def test_ws_upgrade_rejects_missing_key() -> None:
    sent: dict[str, Any] = {}

    def _send_bytes(
        body: bytes,
        content_type: str,
        status: int = 200,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        sent["body"] = body
        sent["content_type"] = content_type
        sent["status"] = status
        sent["extra_headers"] = dict(extra_headers or {})

    result = ws_upgrade_module.upgrade_websocket_or_respond(
        ws_key="",
        try_acquire_client_slot=lambda: True,
        runtime_ws_client_snapshot=lambda: {"active": 0},
        send_json=lambda *_args, **_kwargs: None,
        send_bytes=_send_bytes,
        send_upgrade_response=lambda _key: None,
        release_client_slot=lambda: None,
    )

    assert result is False
    assert sent.get("status") == 400
    assert sent.get("body") == b"missing websocket key"


def test_ws_upgrade_rejects_when_capacity_full() -> None:
    sent: dict[str, Any] = {}

    def _send_json(payload: dict[str, Any], status: int = 200) -> None:
        sent["payload"] = payload
        sent["status"] = status

    result = ws_upgrade_module.upgrade_websocket_or_respond(
        ws_key="key",
        try_acquire_client_slot=lambda: False,
        runtime_ws_client_snapshot=lambda: {"active": 24},
        send_json=_send_json,
        send_bytes=lambda *_args, **_kwargs: None,
        send_upgrade_response=lambda _key: None,
        release_client_slot=lambda: None,
    )

    assert result is False
    assert sent.get("status") == 503
    payload = sent.get("payload", {})
    assert payload.get("error") == "websocket_capacity_reached"


def test_ws_upgrade_releases_slot_on_upgrade_failure() -> None:
    released = {"count": 0}

    def _release() -> None:
        released["count"] += 1

    result = ws_upgrade_module.upgrade_websocket_or_respond(
        ws_key="key",
        try_acquire_client_slot=lambda: True,
        runtime_ws_client_snapshot=lambda: {"active": 1},
        send_json=lambda *_args, **_kwargs: None,
        send_bytes=lambda *_args, **_kwargs: None,
        send_upgrade_response=lambda _key: (_ for _ in ()).throw(BrokenPipeError()),
        release_client_slot=_release,
    )

    assert result is False
    assert released["count"] == 1


def test_ws_upgrade_succeeds_with_valid_key() -> None:
    called: dict[str, Any] = {}

    result = ws_upgrade_module.upgrade_websocket_or_respond(
        ws_key="valid-key",
        try_acquire_client_slot=lambda: True,
        runtime_ws_client_snapshot=lambda: {"active": 1},
        send_json=lambda *_args, **_kwargs: None,
        send_bytes=lambda *_args, **_kwargs: None,
        send_upgrade_response=lambda key: called.update({"key": key}),
        release_client_slot=lambda: called.update({"released": True}),
    )

    assert result is True
    assert called.get("key") == "valid-key"
    assert called.get("released") is None


def test_docker_ws_stream_sends_initial_snapshot_and_releases_slot(
    monkeypatch: Any,
) -> None:
    sent_rows: list[dict[str, Any]] = []
    state: dict[str, Any] = {"released": 0, "consumed": 0, "force_flags": []}

    class _FakePoll:
        def register(self, _connection: Any, _events: int) -> None:
            return None

        def poll(self, _timeout: int) -> list[tuple[int, int]]:
            return [(1, 1)]

    monkeypatch.setattr(docker_ws_module.select, "poll", lambda: _FakePoll())

    def _collect_docker(**kwargs: Any) -> dict[str, Any]:
        state["force_flags"].append(bool(kwargs.get("force_refresh", False)))
        return {"fingerprint": "fp-1"}

    def _consume(_connection: Any) -> bool:
        state["consumed"] += 1
        return False

    def _release() -> None:
        state["released"] += 1

    docker_ws_module.handle_docker_websocket_stream(
        connection=object(),
        send_ws=lambda payload: sent_rows.append(dict(payload)),
        collect_docker_simulation_snapshot=_collect_docker,
        consume_ws_client_frame=_consume,
        release_client_slot=_release,
        docker_refresh_seconds=30.0,
        docker_heartbeat_seconds=30.0,
    )

    assert state["released"] == 1
    assert state["consumed"] == 1
    assert state["force_flags"] == [True]
    assert sent_rows
    assert sent_rows[0].get("type") == "docker_simulations"


def test_docker_ws_stream_emits_error_event_when_initial_snapshot_fails(
    monkeypatch: Any,
) -> None:
    sent_rows: list[dict[str, Any]] = []
    released = {"count": 0}

    class _FakePoll:
        def register(self, _connection: Any, _events: int) -> None:
            return None

        def poll(self, _timeout: int) -> list[tuple[int, int]]:
            return [(1, 1)]

    monkeypatch.setattr(docker_ws_module.select, "poll", lambda: _FakePoll())

    def _collect_docker(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    docker_ws_module.handle_docker_websocket_stream(
        connection=object(),
        send_ws=lambda payload: sent_rows.append(dict(payload)),
        collect_docker_simulation_snapshot=_collect_docker,
        consume_ws_client_frame=lambda _connection: False,
        release_client_slot=lambda: released.update({"count": released["count"] + 1}),
        docker_refresh_seconds=30.0,
        docker_heartbeat_seconds=30.0,
    )

    assert released["count"] == 1
    assert sent_rows
    assert sent_rows[0].get("type") == "docker_simulations_error"
