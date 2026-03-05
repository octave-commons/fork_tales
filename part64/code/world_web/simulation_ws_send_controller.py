from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SimulationWsSendController:
    chunk_enabled: bool
    wire_mode: str
    chunk_chars: int
    stream_particle_max: int
    sim_tick_seconds: float
    json_compact: Callable[[Any], str]
    simulation_ws_chunk_plan: Callable[..., tuple[list[dict[str, Any]], str | None]]
    simulation_ws_chunk_messages: Callable[..., list[dict[str, Any]]]
    send_ws_event: Callable[[dict[str, Any], str], None]
    send_ws_text: Callable[[str], None]
    ws_clamp01: Callable[[float], float]
    chunk_message_seq: int = 0
    send_ema_ms: float = 0.0
    send_ema_bytes: float = 0.0
    send_pressure: float = 0.0
    network_particle_cap: int = 0

    def __post_init__(self) -> None:
        if self.network_particle_cap <= 0:
            self.network_particle_cap = max(24, int(self.stream_particle_max))

    def send(self, payload: dict[str, Any]) -> None:
        send_started = time.perf_counter()
        payload_chars = 0
        try:
            payload_chars = len(self.json_compact(payload))
        except Exception:
            payload_chars = 0

        if self.chunk_enabled:
            self.chunk_message_seq += 1
            if self.wire_mode == "json":
                chunk_rows, payload_text = self.simulation_ws_chunk_plan(
                    payload,
                    chunk_chars=self.chunk_chars,
                    message_seq=self.chunk_message_seq,
                )
                if chunk_rows:
                    for chunk_row in chunk_rows:
                        self.send_ws_event(chunk_row, self.wire_mode)
                elif isinstance(payload_text, str):
                    self.send_ws_text(payload_text)
                else:
                    self.send_ws_event(payload, self.wire_mode)
            else:
                chunk_rows = self.simulation_ws_chunk_messages(
                    payload,
                    chunk_chars=self.chunk_chars,
                    message_seq=self.chunk_message_seq,
                )
                if chunk_rows:
                    for chunk_row in chunk_rows:
                        self.send_ws_event(chunk_row, self.wire_mode)
                else:
                    self.send_ws_event(payload, self.wire_mode)
        else:
            self.send_ws_event(payload, self.wire_mode)

        elapsed_ms = max(0.01, (time.perf_counter() - send_started) * 1000.0)
        ema_alpha = 0.18
        self.send_ema_ms = (
            elapsed_ms
            if self.send_ema_ms <= 0.0
            else ((self.send_ema_ms * (1.0 - ema_alpha)) + (elapsed_ms * ema_alpha))
        )
        if payload_chars > 0:
            self.send_ema_bytes = (
                float(payload_chars)
                if self.send_ema_bytes <= 0.0
                else (
                    (self.send_ema_bytes * (1.0 - ema_alpha))
                    + (float(payload_chars) * ema_alpha)
                )
            )

        tick_budget_ms_ref = max(8.0, float(self.sim_tick_seconds) * 1000.0)
        ms_pressure = self.send_ema_ms / tick_budget_ms_ref
        byte_pressure = (
            self.send_ema_bytes / 180000.0 if self.send_ema_bytes > 0.0 else 0.0
        )
        instantaneous_pressure = max(ms_pressure, byte_pressure)
        self.send_pressure = self.ws_clamp01(
            (self.send_pressure * 0.85) + (instantaneous_pressure * 0.15)
        )
        if self.send_pressure >= 0.92:
            self.network_particle_cap = max(
                24,
                int(math.floor(self.network_particle_cap * 0.88)),
            )
        elif self.send_pressure <= 0.35:
            self.network_particle_cap = min(
                int(self.stream_particle_max),
                self.network_particle_cap + 12,
            )
