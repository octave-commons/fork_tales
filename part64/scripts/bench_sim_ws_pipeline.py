#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import socket
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen


PART_ROOT = Path(__file__).resolve().parents[1]
if str(PART_ROOT) not in sys.path:
    sys.path.insert(0, str(PART_ROOT))
if "code" in sys.modules and not hasattr(sys.modules["code"], "__path__"):
    del sys.modules["code"]

from code.world_web.projection import build_ui_projection  # type: ignore
from code.world_web.simulation import (  # type: ignore
    build_simulation_delta,
    build_simulation_state,
)


_WS_WIRE_ARRAY_SCHEMA = "eta-mu.ws.arr.v1"
_WS_PACK_TAG_OBJECT = -1
_WS_PACK_TAG_ARRAY = -2
_WS_PACK_TAG_STRING = -3
_WS_PACK_TAG_BOOL = -4
_WS_PACK_TAG_NULL = -5


def _json_compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _simulation_ws_trim_simulation_payload(
    simulation: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(simulation, dict):
        return {}
    trimmed = dict(simulation)
    for key in (
        "field_particles",
        "nexus_graph",
        "logical_graph",
        "pain_field",
        "file_graph",
        "crawler_graph",
    ):
        trimmed.pop(key, None)
    return trimmed


def websocket_frame_text(message: str) -> bytes:
    payload = message.encode("utf-8")
    length = len(payload)
    header = bytearray([0x81])
    if length <= 125:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + payload


def _ws_decode_packed_node(node: Any, key_table: list[str]) -> Any:
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return node
    if not isinstance(node, list) or not node:
        return None
    tag = int(node[0]) if isinstance(node[0], (int, float)) else None
    if tag is None:
        return None
    if tag == _WS_PACK_TAG_NULL:
        return None
    if tag == _WS_PACK_TAG_BOOL:
        return int(node[1]) != 0 if len(node) > 1 else False
    if tag == _WS_PACK_TAG_STRING:
        return str(node[1]) if len(node) > 1 else ""
    if tag == _WS_PACK_TAG_ARRAY:
        return [_ws_decode_packed_node(item, key_table) for item in node[1:]]
    if tag != _WS_PACK_TAG_OBJECT:
        return None

    output: dict[str, Any] = {}
    index = 1
    while index + 1 < len(node):
        slot = node[index]
        if isinstance(slot, (int, float)) and not isinstance(slot, bool):
            slot_index = int(slot)
            if 0 <= slot_index < len(key_table):
                output[key_table[slot_index]] = _ws_decode_packed_node(
                    node[index + 1], key_table
                )
        index += 2
    return output


def _ws_decode_message(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        return payload
    if (
        not isinstance(payload, list)
        or len(payload) < 3
        or str(payload[0]) != _WS_WIRE_ARRAY_SCHEMA
    ):
        return None
    key_table_raw = payload[1]
    key_table = (
        [str(item) for item in key_table_raw if isinstance(item, str)]
        if isinstance(key_table_raw, list)
        else []
    )
    decoded = _ws_decode_packed_node(payload[2], key_table)
    return decoded if isinstance(decoded, dict) else None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * max(0.0, min(1.0, p))
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] + ((values[upper] - values[lower]) * weight)


def _summarize_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    ordered = sorted(values)
    return {
        "count": float(len(values)),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(ordered, 0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _summarize_count(values: list[int]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    as_float = [float(v) for v in values]
    ordered = sorted(as_float)
    return {
        "count": float(len(values)),
        "mean": statistics.fmean(as_float),
        "median": statistics.median(as_float),
        "p95": _percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _safe_iso_age_ms(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - parsed).total_seconds() * 1000.0


def _recv_exact(connection: socket.socket, size: int) -> bytes | None:
    if size <= 0:
        return b""
    chunks = bytearray()
    while len(chunks) < size:
        try:
            chunk = connection.recv(size - len(chunks))
        except socket.timeout:
            return None
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def _read_ws_frame(connection: socket.socket) -> tuple[int, bytes] | None:
    header = _recv_exact(connection, 2)
    if header is None:
        return None
    b0, b1 = header[0], header[1]
    opcode = b0 & 0x0F
    masked = (b1 & 0x80) != 0
    payload_len = b1 & 0x7F
    if payload_len == 126:
        extra = _recv_exact(connection, 2)
        if extra is None:
            return None
        payload_len = struct.unpack("!H", extra)[0]
    elif payload_len == 127:
        extra = _recv_exact(connection, 8)
        if extra is None:
            return None
        payload_len = struct.unpack("!Q", extra)[0]
    mask_key = b""
    if masked:
        mask_key = _recv_exact(connection, 4) or b""
        if len(mask_key) != 4:
            return None
    payload = _recv_exact(connection, int(payload_len))
    if payload is None:
        return None
    if masked:
        payload = bytes(
            byte ^ mask_key[index % 4] for index, byte in enumerate(payload)
        )
    return opcode, payload


def _send_ws_control(
    connection: socket.socket, opcode: int, payload: bytes = b""
) -> None:
    mask_key = os.urandom(4)
    masked_payload = bytes(
        value ^ mask_key[index % 4] for index, value in enumerate(payload)
    )
    frame = bytearray()
    frame.append(0x80 | (opcode & 0x0F))
    length = len(masked_payload)
    if length <= 125:
        frame.append(0x80 | length)
    elif length <= 65535:
        frame.append(0x80 | 126)
        frame.extend(struct.pack("!H", length))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack("!Q", length))
    frame.extend(mask_key)
    frame.extend(masked_payload)
    connection.sendall(bytes(frame))


def _connect_websocket(
    *,
    host: str,
    port: int,
    path: str,
    timeout_seconds: float,
) -> socket.socket:
    connection = socket.create_connection((host, port), timeout=timeout_seconds)
    connection.settimeout(timeout_seconds)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Origin: http://{host}:{port}\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )
    connection.sendall(request.encode("ascii"))

    response = bytearray()
    while b"\r\n\r\n" not in response:
        chunk = connection.recv(1024)
        if not chunk:
            raise RuntimeError("websocket handshake failed: no response")
        response.extend(chunk)
        if len(response) > 65536:
            raise RuntimeError("websocket handshake failed: oversized response")

    header_text = bytes(response).decode("utf-8", errors="replace")
    status_line = header_text.split("\r\n", 1)[0]
    if " 101 " not in f" {status_line} ":
        raise RuntimeError(f"websocket handshake failed: {status_line}")

    expected_accept = base64.b64encode(
        hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
        ).digest()
    ).decode("ascii")
    accept_line = ""
    for line in header_text.split("\r\n"):
        if line.lower().startswith("sec-websocket-accept:"):
            accept_line = line.split(":", 1)[1].strip()
            break
    if expected_accept != accept_line:
        raise RuntimeError("websocket handshake failed: invalid accept key")
    return connection


def _wait_for_health(*, host: str, port: int, timeout_seconds: float) -> None:
    deadline = time.monotonic() + max(1.0, timeout_seconds)
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=1.0)
            sock.close()
            return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"timed out waiting for runtime socket: {host}:{port}")


def _fetch_catalog_snapshot(
    *,
    host: str,
    port: int,
    perspective: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    url = f"http://{host}:{port}/api/catalog?perspective={quote(perspective)}"
    with urlopen(url, timeout=max(1.0, timeout_seconds)) as response:
        status = int(getattr(response, "status", 0) or 0)
        if status < 200 or status >= 300:
            raise RuntimeError(f"catalog request failed with status {status}")
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("catalog payload is not a JSON object")
    return payload


def benchmark_raw_pipeline(
    *,
    catalog_seed: dict[str, Any],
    perspective: str,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    catalog = catalog_seed if isinstance(catalog_seed, dict) else {}
    previous_simulation: dict[str, Any] | None = None

    build_ms: list[float] = []
    sim_only_ms: list[float] = []
    projection_only_ms: list[float] = []
    encode_ms: list[float] = []
    delta_build_ms: list[float] = []
    total_ms: list[float] = []
    simulation_event_bytes: list[int] = []
    simulation_frame_bytes: list[int] = []
    delta_event_bytes: list[int] = []
    delta_frame_bytes: list[int] = []
    delta_changed_keys: list[int] = []

    rounds = max(1, warmup + iterations)
    for round_index in range(rounds):
        started = time.perf_counter()
        simulation = build_simulation_state(catalog)
        after_sim = time.perf_counter()
        projection = build_ui_projection(
            catalog,
            simulation,
            perspective=perspective,
        )
        simulation["projection"] = projection
        simulation["perspective"] = perspective
        after_build = time.perf_counter()

        simulation_trim = _simulation_ws_trim_simulation_payload(simulation)
        simulation_event = {
            "type": "simulation",
            "simulation": simulation_trim,
            "projection": projection,
        }
        simulation_json = _json_compact(simulation_event)
        simulation_frame = websocket_frame_text(simulation_json)
        after_encode = time.perf_counter()

        delta_json_bytes = 0
        delta_frame_len = 0
        delta_key_count = 0
        if previous_simulation is not None:
            delta = build_simulation_delta(previous_simulation, simulation_trim)
            if bool(delta.get("has_changes", False)):
                changed_keys = delta.get("changed_keys", [])
                delta_key_count = (
                    len(changed_keys) if isinstance(changed_keys, list) else 0
                )
                delta_event = {
                    "type": "simulation_delta",
                    "delta": delta,
                }
                delta_json = _json_compact(delta_event)
                delta_frame = websocket_frame_text(delta_json)
                delta_json_bytes = len(delta_json.encode("utf-8"))
                delta_frame_len = len(delta_frame)
        previous_simulation = simulation_trim
        finished = time.perf_counter()

        if round_index < warmup:
            continue

        build_ms.append((after_build - started) * 1000.0)
        sim_only_ms.append((after_sim - started) * 1000.0)
        projection_only_ms.append((after_build - after_sim) * 1000.0)
        encode_ms.append((after_encode - after_build) * 1000.0)
        delta_build_ms.append((finished - after_encode) * 1000.0)
        total_ms.append((finished - started) * 1000.0)
        simulation_event_bytes.append(len(simulation_json.encode("utf-8")))
        simulation_frame_bytes.append(len(simulation_frame))
        if delta_json_bytes > 0:
            delta_event_bytes.append(delta_json_bytes)
            delta_frame_bytes.append(delta_frame_len)
            delta_changed_keys.append(delta_key_count)

    mean_total_ms = statistics.fmean(total_ms) if total_ms else 0.0
    effective_hz = (1000.0 / mean_total_ms) if mean_total_ms > 1e-9 else 0.0
    return {
        "iterations": max(1, iterations),
        "warmup": max(0, warmup),
        "catalog_item_count": len(catalog.get("items", []))
        if isinstance(catalog.get("items", []), list)
        else 0,
        "sim_only": _summarize_ms(sim_only_ms),
        "projection_only": _summarize_ms(projection_only_ms),
        "build_total": _summarize_ms(build_ms),
        "encode_only": _summarize_ms(encode_ms),
        "delta_build_only": _summarize_ms(delta_build_ms),
        "encode_and_frame": _summarize_ms(
            [e + d for e, d in zip(encode_ms, delta_build_ms)]
        ),
        "total": _summarize_ms(total_ms),
        "simulation_event_bytes": _summarize_count(simulation_event_bytes),
        "simulation_frame_bytes": _summarize_count(simulation_frame_bytes),
        "delta_event_bytes": _summarize_count(delta_event_bytes),
        "delta_frame_bytes": _summarize_count(delta_frame_bytes),
        "delta_changed_keys": _summarize_count(delta_changed_keys),
        "delta_emit_ratio": (
            float(len(delta_event_bytes)) / float(len(total_ms)) if total_ms else 0.0
        ),
        "effective_pipeline_hz": effective_hz,
    }


def benchmark_websocket_stream(
    *,
    host: str,
    port: int,
    perspective: str,
    delta_stream_mode: str,
    ws_wire_mode: str,
    duration_seconds: float,
    read_timeout_seconds: float,
    max_messages: int,
) -> dict[str, Any]:
    ws_path = (
        f"/ws?perspective={quote(perspective)}"
        f"&delta_stream={quote(delta_stream_mode)}"
        f"&wire={quote(ws_wire_mode)}"
    )
    connection = _connect_websocket(
        host=host,
        port=port,
        path=ws_path,
        timeout_seconds=read_timeout_seconds,
    )
    connection.settimeout(max(0.2, read_timeout_seconds))

    started = time.perf_counter()
    last_recv_by_type: dict[str, float] = {}
    intervals_by_type: dict[str, list[float]] = {}
    age_by_type: dict[str, list[float]] = {}
    type_counts: dict[str, int] = {}
    type_bytes: dict[str, int] = {}
    delta_changed_keys: list[int] = []
    simulation_tick_intervals_ms: list[float] = []
    simulation_tick_count = 0
    last_tick_timestamp = ""
    last_tick_received_at: float | None = None
    all_bytes = 0
    all_messages = 0

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - started
            if elapsed >= max(0.5, duration_seconds):
                break
            if max_messages > 0 and all_messages >= max_messages:
                break

            frame = _read_ws_frame(connection)
            if frame is None:
                continue
            opcode, payload = frame
            if opcode == 0x8:
                break
            if opcode == 0x9:
                _send_ws_control(connection, 0xA, payload[:125])
                continue
            if opcode != 0x1:
                continue

            received_at = time.perf_counter()
            all_messages += 1
            payload_size = len(payload)
            all_bytes += payload_size

            try:
                parsed_message = json.loads(payload.decode("utf-8"))
            except Exception:
                continue
            message = _ws_decode_message(parsed_message)
            if not isinstance(message, dict):
                continue

            msg_type = str(message.get("type", "unknown")).strip() or "unknown"
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
            type_bytes[msg_type] = type_bytes.get(msg_type, 0) + payload_size

            previous_recv = last_recv_by_type.get(msg_type)
            if previous_recv is not None:
                intervals_by_type.setdefault(msg_type, []).append(
                    (received_at - previous_recv) * 1000.0
                )
            last_recv_by_type[msg_type] = received_at

            timestamp_value: Any = None
            if msg_type == "simulation":
                simulation = message.get("simulation", {})
                if isinstance(simulation, dict):
                    timestamp_value = simulation.get("timestamp")
            elif msg_type == "simulation_delta":
                delta = message.get("delta", {})
                if isinstance(delta, dict):
                    patch = delta.get("patch", {})
                    if isinstance(patch, dict):
                        timestamp_value = patch.get("timestamp")
                    changed_keys = delta.get("changed_keys", [])
                    if isinstance(changed_keys, list):
                        delta_changed_keys.append(len(changed_keys))

            age_ms = _safe_iso_age_ms(timestamp_value)
            if age_ms is not None:
                age_by_type.setdefault(msg_type, []).append(age_ms)

            timestamp_text = str(timestamp_value or "").strip()
            if timestamp_text and timestamp_text != last_tick_timestamp:
                if last_tick_received_at is not None:
                    simulation_tick_intervals_ms.append(
                        (received_at - last_tick_received_at) * 1000.0
                    )
                last_tick_timestamp = timestamp_text
                last_tick_received_at = received_at
                simulation_tick_count += 1
    finally:
        try:
            _send_ws_control(connection, 0x8, b"")
        except Exception:
            pass
        connection.close()

    elapsed_seconds = max(1e-9, time.perf_counter() - started)
    interval_summary = {
        key: _summarize_ms(value) for key, value in sorted(intervals_by_type.items())
    }
    age_summary = {
        key: _summarize_ms(value) for key, value in sorted(age_by_type.items())
    }
    return {
        "duration_seconds": elapsed_seconds,
        "message_count": all_messages,
        "bytes_total": all_bytes,
        "messages_per_second": float(all_messages) / elapsed_seconds,
        "bytes_per_second": float(all_bytes) / elapsed_seconds,
        "type_counts": type_counts,
        "type_bytes": type_bytes,
        "interval_ms": interval_summary,
        "simulation_tick_count": simulation_tick_count,
        "simulation_tick_interval_ms": _summarize_ms(simulation_tick_intervals_ms),
        "age_ms": age_summary,
        "delta_changed_keys": _summarize_count(delta_changed_keys),
    }


def _derive_comparison(raw: dict[str, Any], ws: dict[str, Any]) -> dict[str, Any]:
    raw_total_mean = float((raw.get("total", {}) or {}).get("mean_ms", 0.0) or 0.0)
    ws_tick_interval_mean = float(
        ((ws.get("simulation_tick_interval_ms", {}) or {}).get("mean_ms", 0.0) or 0.0)
    )
    interval_map = (
        ws.get("interval_ms", {}) if isinstance(ws.get("interval_ms"), dict) else {}
    )
    ws_sim_interval_mean = float(
        ((interval_map.get("simulation", {}) or {}).get("mean_ms", 0.0) or 0.0)
    )
    ws_delta_interval_mean = float(
        ((interval_map.get("simulation_delta", {}) or {}).get("mean_ms", 0.0) or 0.0)
    )
    interval_source = "simulation"
    if ws_tick_interval_mean > 0.0:
        ws_sim_interval_mean = ws_tick_interval_mean
        interval_source = "simulation_tick"
    elif ws_sim_interval_mean <= 0.0 and ws_delta_interval_mean > 0.0:
        ws_sim_interval_mean = ws_delta_interval_mean
        interval_source = "simulation_delta"
    elif ws_sim_interval_mean <= 0.0:
        interval_source = "none"
    ws_sim_rate = 1000.0 / ws_sim_interval_mean if ws_sim_interval_mean > 1e-9 else 0.0
    raw_rate = 1000.0 / raw_total_mean if raw_total_mean > 1e-9 else 0.0
    headroom_ms = ws_sim_interval_mean - raw_total_mean
    headroom_ratio = (
        (headroom_ms / ws_sim_interval_mean) if ws_sim_interval_mean > 1e-9 else 0.0
    )

    type_counts = (
        ws.get("type_counts", {}) if isinstance(ws.get("type_counts"), dict) else {}
    )
    simulation_count = int(type_counts.get("simulation", 0) or 0)
    delta_count = int(type_counts.get("simulation_delta", 0) or 0)
    duplicate_ratio = (
        float(delta_count) / float(simulation_count) if simulation_count > 0 else 0.0
    )

    bottleneck = "undetermined"
    if raw_total_mean > 0.0 and ws_sim_interval_mean > 0.0:
        if raw_total_mean < ws_sim_interval_mean * 0.7:
            bottleneck = "ws_tick_interval_or_transport"
        else:
            bottleneck = "simulation_compute_or_projection"

    return {
        "raw_pipeline_mean_ms": raw_total_mean,
        "raw_pipeline_hz": raw_rate,
        "ws_simulation_interval_mean_ms": ws_sim_interval_mean,
        "ws_interval_source": interval_source,
        "ws_simulation_hz": ws_sim_rate,
        "ws_simulation_tick_count": int(ws.get("simulation_tick_count", 0) or 0),
        "interval_headroom_ms": headroom_ms,
        "interval_headroom_ratio": headroom_ratio,
        "simulation_message_count": simulation_count,
        "delta_message_count": delta_count,
        "delta_per_simulation_ratio": duplicate_ratio,
        "inferred_bottleneck": bottleneck,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark raw simulation pipeline against websocket stream output"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18787)
    parser.add_argument(
        "--part-root",
        type=Path,
        default=PART_ROOT,
        help="Part root for world runtime (defaults to part64)",
    )
    parser.add_argument(
        "--vault-root",
        type=Path,
        default=PART_ROOT.parent,
        help="Vault root (defaults to repository root)",
    )
    parser.add_argument("--perspective", default="hybrid")
    parser.add_argument(
        "--delta-stream",
        default="world",
        help="Websocket delta stream mode: world or workers",
    )
    parser.add_argument(
        "--ws-wire",
        default="json",
        help="Websocket wire mode: json or arr",
    )
    parser.add_argument("--raw-iterations", type=int, default=24)
    parser.add_argument("--raw-warmup", type=int, default=4)
    parser.add_argument("--ws-seconds", type=float, default=10.0)
    parser.add_argument(
        "--ws-max-messages",
        type=int,
        default=0,
        help="Stop websocket sample after N messages (0 = disabled)",
    )
    parser.add_argument(
        "--reuse-server",
        action="store_true",
        help="Use an existing runtime server instead of spawning one",
    )
    parser.add_argument(
        "--boot-timeout",
        type=float,
        default=45.0,
        help="Seconds to wait for spawned server health",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=2.5,
        help="Websocket frame read timeout seconds",
    )
    parser.add_argument(
        "--sim-tick-seconds",
        type=float,
        default=-1.0,
        help="Override SIM_TICK_SECONDS for spawned server (-1 keeps env default)",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file path",
    )
    return parser.parse_args(argv)


from code.world_web.metrics import _resource_monitor_snapshot  # type: ignore


def benchmark_pure_sim(
    *,
    catalog_seed: dict[str, Any],
    iterations: int,
    warmup: int,
    part_root: Path,
) -> dict[str, Any]:
    catalog = catalog_seed if isinstance(catalog_seed, dict) else {}
    sim_ms: list[float] = []
    metrics_ms: list[float] = []

    rounds = max(1, warmup + iterations)
    for round_index in range(rounds):
        # Measure metrics cost explicitly
        m_start = time.perf_counter()
        _ = _resource_monitor_snapshot(part_root=part_root)
        m_end = time.perf_counter()

        started = time.perf_counter()
        _ = build_simulation_state(catalog)
        finished = time.perf_counter()

        if round_index < warmup:
            continue
        sim_ms.append((finished - started) * 1000.0)
        metrics_ms.append((m_end - m_start) * 1000.0)

    return {
        "sim": _summarize_ms(sim_ms),
        "metrics": _summarize_ms(metrics_ms),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    part_root = args.part_root.expanduser().resolve()
    vault_root = args.vault_root.expanduser().resolve()

    # We need a server running or at least a catalog snapshot
    # If reuse_server is false, we spawn one just to get the catalog
    # then we can run pure sim benchmarks in this process

    runtime_process: subprocess.Popen[str] | None = None
    if not args.reuse_server:
        env = os.environ.copy()
        env.setdefault("WEAVER_AUTOSTART", "0")
        if float(args.sim_tick_seconds) > 0.0:
            env["SIM_TICK_SECONDS"] = str(float(args.sim_tick_seconds))
        world_web_entry = (PART_ROOT / "code" / "world_web.py").resolve()
        command = [
            sys.executable,
            str(world_web_entry),
            "--host",
            str(args.host),
            "--port",
            str(int(args.port)),
            "--part-root",
            str(part_root),
            "--vault-root",
            str(vault_root),
        ]
        runtime_process = subprocess.Popen(
            command,
            cwd=str(part_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_health(
                host=str(args.host),
                port=int(args.port),
                timeout_seconds=float(args.boot_timeout),
            )
        except Exception:
            output = ""
            if runtime_process.stdout is not None:
                try:
                    output = runtime_process.stdout.read()[:4000]
                except Exception:
                    output = ""
            runtime_process.terminate()
            raise RuntimeError(
                "failed to boot runtime server for websocket benchmark"
                + (f": {output}" if output else "")
            )

    raw: dict[str, Any]
    ws: dict[str, Any]
    try:
        _wait_for_health(
            host=str(args.host),
            port=int(args.port),
            timeout_seconds=float(args.boot_timeout),
        )
        catalog_snapshot = _fetch_catalog_snapshot(
            host=str(args.host),
            port=int(args.port),
            perspective=str(args.perspective),
            timeout_seconds=max(30.0, float(args.boot_timeout)),
        )

        raw = benchmark_raw_pipeline(
            catalog_seed=catalog_snapshot,
            perspective=str(args.perspective),
            iterations=max(1, int(args.raw_iterations)),
            warmup=max(0, int(args.raw_warmup)),
        )

        pure_sim = benchmark_pure_sim(
            catalog_seed=catalog_snapshot,
            iterations=max(1, int(args.raw_iterations) * 2),
            warmup=max(0, int(args.raw_warmup)),
            part_root=part_root,
        )

        ws = benchmark_websocket_stream(
            host=str(args.host),
            port=int(args.port),
            perspective=str(args.perspective),
            delta_stream_mode=str(args.delta_stream),
            ws_wire_mode=str(args.ws_wire),
            duration_seconds=max(0.5, float(args.ws_seconds)),
            read_timeout_seconds=max(0.2, float(args.read_timeout)),
            max_messages=max(0, int(args.ws_max_messages)),
        )
    finally:
        if runtime_process is not None:
            runtime_process.terminate()
            try:
                runtime_process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                runtime_process.kill()

    comparison = _derive_comparison(raw, ws)
    report = {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "host": str(args.host),
            "port": int(args.port),
            "part_root": str(part_root),
            "vault_root": str(vault_root),
            "perspective": str(args.perspective),
            "delta_stream": str(args.delta_stream),
            "ws_wire": str(args.ws_wire),
            "raw_iterations": int(args.raw_iterations),
            "raw_warmup": int(args.raw_warmup),
            "ws_seconds": float(args.ws_seconds),
            "ws_max_messages": int(args.ws_max_messages),
            "reuse_server": bool(args.reuse_server),
            "sim_tick_seconds_override": (
                float(args.sim_tick_seconds)
                if float(args.sim_tick_seconds) > 0.0
                else None
            ),
        },
        "raw_pipeline": raw,
        "websocket_stream": ws,
        "comparison": comparison,
    }

    output_path: Path | None = None
    if args.output is not None:
        resolved_output_path = args.output.expanduser()
        if not resolved_output_path.is_absolute():
            resolved_output_path = (Path.cwd() / resolved_output_path).resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        output_path = resolved_output_path

    if args.json:
        print(json.dumps(report, indent=2))
        if output_path is not None:
            print(f"wrote report: {output_path}")
        return 0

    raw_mean = float((raw.get("total", {}) or {}).get("mean_ms", 0.0) or 0.0)
    ws_interval = float(comparison.get("ws_simulation_interval_mean_ms", 0.0) or 0.0)
    ws_interval_source = str(comparison.get("ws_interval_source", "none") or "none")
    ws_counts = (
        ws.get("type_counts", {}) if isinstance(ws.get("type_counts"), dict) else {}
    )
    print(
        "raw pipeline: "
        f"total={raw_mean:.2f}ms "
        f"sim={float((raw.get('sim_only', {}) or {}).get('mean_ms', 0.0) or 0.0):.2f}ms "
        f"proj={float((raw.get('projection_only', {}) or {}).get('mean_ms', 0.0) or 0.0):.2f}ms "
        f"encode={float((raw.get('encode_only', {}) or {}).get('mean_ms', 0.0) or 0.0):.2f}ms "
        f"delta={float((raw.get('delta_build_only', {}) or {}).get('mean_ms', 0.0) or 0.0):.2f}ms "
        f"effective={float(raw.get('effective_pipeline_hz', 0.0) or 0.0):.2f}Hz"
    )
    print(
        "pure sim: "
        f"mean={float(pure_sim.get('sim', {}).get('mean_ms', 0.0)):.2f}ms "
        f"metrics_overhead={float(pure_sim.get('metrics', {}).get('mean_ms', 0.0)):.2f}ms"
    )
    print(
        "websocket stream: "
        f"simulation={int(ws_counts.get('simulation', 0) or 0)} "
        f"delta={int(ws_counts.get('simulation_delta', 0) or 0)} "
        f"ticks={int(comparison.get('ws_simulation_tick_count', 0) or 0)} "
        f"sim_interval_mean={ws_interval:.2f}ms({ws_interval_source}) "
        f"messages_per_second={float(ws.get('messages_per_second', 0.0) or 0.0):.2f} "
        f"bytes_per_second={float(ws.get('bytes_per_second', 0.0) or 0.0):.0f}"
    )
    print(
        "comparison: "
        f"headroom={float(comparison.get('interval_headroom_ms', 0.0) or 0.0):.2f}ms "
        f"({float(comparison.get('interval_headroom_ratio', 0.0) or 0.0) * 100.0:+.1f}%) "
        f"bottleneck={comparison.get('inferred_bottleneck', 'undetermined')}"
    )
    if output_path is not None:
        print(f"wrote report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
