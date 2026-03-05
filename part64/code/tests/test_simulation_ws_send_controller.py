from __future__ import annotations

from typing import Any

from code.world_web import simulation_ws_send_controller as ws_send_module


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def test_ws_send_controller_plain_event_path_uses_send_ws_event() -> None:
    sent_events: list[tuple[dict[str, Any], str]] = []
    sent_text: list[str] = []

    controller = ws_send_module.SimulationWsSendController(
        chunk_enabled=False,
        wire_mode="json",
        chunk_chars=256,
        stream_particle_max=240,
        sim_tick_seconds=0.1,
        json_compact=lambda payload: '{"ok":true}',
        simulation_ws_chunk_plan=lambda *_args, **_kwargs: ([], None),
        simulation_ws_chunk_messages=lambda *_args, **_kwargs: [],
        send_ws_event=lambda payload, mode: sent_events.append((dict(payload), mode)),
        send_ws_text=lambda payload_text: sent_text.append(payload_text),
        ws_clamp01=_clamp01,
    )

    controller.send({"type": "ping"})

    assert sent_events == [({"type": "ping"}, "json")]
    assert sent_text == []
    assert controller.chunk_message_seq == 0


def test_ws_send_controller_json_chunk_path_emits_chunk_rows() -> None:
    sent_events: list[tuple[dict[str, Any], str]] = []

    def _chunk_plan(
        *_args: Any, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], str | None]:
        message_seq = int(kwargs.get("message_seq", 0))
        return ([{"type": "ws_chunk", "message_seq": message_seq}], None)

    controller = ws_send_module.SimulationWsSendController(
        chunk_enabled=True,
        wire_mode="json",
        chunk_chars=32,
        stream_particle_max=180,
        sim_tick_seconds=0.1,
        json_compact=lambda payload: str(payload),
        simulation_ws_chunk_plan=_chunk_plan,
        simulation_ws_chunk_messages=lambda *_args, **_kwargs: [],
        send_ws_event=lambda payload, mode: sent_events.append((dict(payload), mode)),
        send_ws_text=lambda _payload_text: None,
        ws_clamp01=_clamp01,
    )

    controller.send({"type": "catalog"})
    controller.send({"type": "simulation"})

    assert controller.chunk_message_seq == 2
    assert sent_events[0][0].get("message_seq") == 1
    assert sent_events[1][0].get("message_seq") == 2


def test_ws_send_controller_arr_chunk_path_uses_chunk_messages() -> None:
    sent_events: list[tuple[dict[str, Any], str]] = []

    def _chunk_messages(
        _payload: dict[str, Any],
        *,
        chunk_chars: int,
        message_seq: int,
    ) -> list[dict[str, Any]]:
        assert chunk_chars == 64
        return [{"type": "ws_chunk", "message_seq": message_seq}]

    controller = ws_send_module.SimulationWsSendController(
        chunk_enabled=True,
        wire_mode="arr",
        chunk_chars=64,
        stream_particle_max=220,
        sim_tick_seconds=0.1,
        json_compact=lambda payload: str(payload),
        simulation_ws_chunk_plan=lambda *_args, **_kwargs: ([], None),
        simulation_ws_chunk_messages=_chunk_messages,
        send_ws_event=lambda payload, mode: sent_events.append((dict(payload), mode)),
        send_ws_text=lambda _payload_text: None,
        ws_clamp01=_clamp01,
    )

    controller.send({"type": "delta"})

    assert controller.chunk_message_seq == 1
    assert sent_events == [({"type": "ws_chunk", "message_seq": 1}, "arr")]


def test_ws_send_controller_pressure_reduces_particle_cap_under_sustained_load() -> (
    None
):
    sent_events: list[tuple[dict[str, Any], str]] = []

    controller = ws_send_module.SimulationWsSendController(
        chunk_enabled=False,
        wire_mode="json",
        chunk_chars=128,
        stream_particle_max=300,
        sim_tick_seconds=0.1,
        json_compact=lambda _payload: "x" * 250000,
        simulation_ws_chunk_plan=lambda *_args, **_kwargs: ([], None),
        simulation_ws_chunk_messages=lambda *_args, **_kwargs: [],
        send_ws_event=lambda payload, mode: sent_events.append((dict(payload), mode)),
        send_ws_text=lambda _payload_text: None,
        ws_clamp01=_clamp01,
        network_particle_cap=300,
    )

    for _ in range(20):
        controller.send({"type": "simulation"})

    assert sent_events
    assert controller.send_pressure >= 0.92
    assert controller.network_particle_cap < 300


def test_ws_send_controller_pressure_increases_cap_when_pressure_low() -> None:
    controller = ws_send_module.SimulationWsSendController(
        chunk_enabled=False,
        wire_mode="json",
        chunk_chars=128,
        stream_particle_max=300,
        sim_tick_seconds=0.1,
        json_compact=lambda _payload: "ok",
        simulation_ws_chunk_plan=lambda *_args, **_kwargs: ([], None),
        simulation_ws_chunk_messages=lambda *_args, **_kwargs: [],
        send_ws_event=lambda _payload, _mode: None,
        send_ws_text=lambda _payload_text: None,
        ws_clamp01=_clamp01,
        network_particle_cap=120,
    )
    controller.send_pressure = 0.0

    controller.send({"type": "simulation"})

    assert controller.network_particle_cap == 132
