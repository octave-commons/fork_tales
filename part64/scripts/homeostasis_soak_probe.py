from __future__ import annotations

import argparse
import base64
import json
import os
import random
import socket
import statistics
import struct
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from http.client import HTTPConnection
from typing import Any


def _safe_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not (number == number) or number in {float("inf"), float("-inf")}:
        return fallback
    return number


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    q_clamped = max(0.0, min(1.0, q))
    index = q_clamped * (len(ordered) - 1)
    lo = int(index)
    hi = min(len(ordered) - 1, lo + 1)
    frac = index - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _http_get(
    *,
    host: str,
    port: int,
    path: str,
    timeout_seconds: float,
    read_limit: int,
) -> tuple[bool, int, float, str]:
    started = time.monotonic()
    conn = HTTPConnection(host, port, timeout=timeout_seconds)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        body = response.read(read_limit)
        _ = len(body)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return (200 <= int(response.status) < 300, int(response.status), elapsed_ms, "")
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return (False, 0, elapsed_ms, str(exc))
    finally:
        conn.close()


def _http_post_json(
    *,
    host: str,
    port: int,
    path: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    read_limit: int,
) -> tuple[bool, int, float, str]:
    started = time.monotonic()
    conn = HTTPConnection(host, port, timeout=timeout_seconds)
    try:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        conn.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        _ = response.read(read_limit)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return (200 <= int(response.status) < 300, int(response.status), elapsed_ms, "")
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return (False, 0, elapsed_ms, str(exc))
    finally:
        conn.close()


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    if size <= 0:
        return b""
    data = bytearray()
    while len(data) < size:
        try:
            chunk = sock.recv(size - len(data))
        except socket.timeout:
            if not data:
                raise
            continue
        if not chunk:
            raise EOFError("socket closed")
        data.extend(chunk)
    return bytes(data)


def _read_ws_frame(sock: socket.socket) -> tuple[int, bool, bytes]:
    header = _recv_exact(sock, 2)
    first, second = header
    is_final = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    payload_len = second & 0x7F

    if payload_len == 126:
        payload_len = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif payload_len == 127:
        payload_len = struct.unpack("!Q", _recv_exact(sock, 8))[0]

    mask_key = _recv_exact(sock, 4) if masked else b""
    payload = _recv_exact(sock, payload_len) if payload_len else b""
    if masked and payload:
        payload = bytes(
            value ^ mask_key[index % 4] for index, value in enumerate(payload)
        )
    return opcode, is_final, payload


def _send_ws_client_frame(
    sock: socket.socket, opcode: int, payload: bytes = b""
) -> None:
    frame_head = bytearray([0x80 | (opcode & 0x0F)])
    payload_len = len(payload)
    mask = os.urandom(4)

    if payload_len <= 125:
        frame_head.append(0x80 | payload_len)
    elif payload_len < 65536:
        frame_head.append(0x80 | 126)
        frame_head.extend(struct.pack("!H", payload_len))
    else:
        frame_head.append(0x80 | 127)
        frame_head.extend(struct.pack("!Q", payload_len))

    masked_payload = bytes(
        value ^ mask[index % 4] for index, value in enumerate(payload)
    )
    sock.sendall(bytes(frame_head) + mask + masked_payload)


def _connect_ws(
    *, host: str, port: int, path: str, timeout_seconds: float
) -> socket.socket:
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout_seconds)
    sock.connect((host, port))
    sock.sendall(request.encode("ascii"))

    response = b""
    while b"\r\n\r\n" not in response:
        response += sock.recv(4096)
    status_line = response.split(b"\r\n", 1)[0].decode("latin1", errors="ignore")
    if "101" not in status_line:
        raise RuntimeError(f"websocket handshake failed: {status_line}")

    sock.settimeout(2.0)
    return sock


@dataclass
class RequestBucket:
    total: int = 0
    ok: int = 0
    failures: int = 0
    timeouts: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    statuses: Counter = field(default_factory=Counter)

    def record(self, *, ok: bool, status: int, latency_ms: float, error: str) -> None:
        self.total += 1
        self.latencies_ms.append(latency_ms)
        if status > 0:
            self.statuses[str(status)] += 1
        if ok:
            self.ok += 1
            return
        self.failures += 1
        if "timed out" in error.lower() or "timeout" in error.lower():
            self.timeouts += 1


@dataclass
class GuardSample:
    phase: str
    ts_mono: float
    mode: str
    cpu_utilization: float
    memory_pressure: float
    reasons: list[str]


@dataclass
class CoreTickSample:
    phase: str
    ts_mono: float
    tick_elapsed_ms: float
    slack_ms: float
    ingestion_pressure: float
    ws_particle_max: int
    payload_mode: str


@dataclass
class ProbeState:
    phase_name: str
    run_start_mono: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker_stop_event: threading.Event = field(default_factory=threading.Event)
    request_buckets: dict[str, dict[str, RequestBucket]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    guard_samples: list[GuardSample] = field(default_factory=list)
    core_ticks: list[CoreTickSample] = field(default_factory=list)
    ws_types: Counter = field(default_factory=Counter)
    ws_workers: Counter = field(default_factory=Counter)
    ws_parse_errors: int = 0
    ws_connect_error: str = ""
    ws_stop_reason: str = ""

    def current_phase(self) -> str:
        with self.lock:
            return self.phase_name

    def set_phase(self, value: str) -> None:
        with self.lock:
            self.phase_name = value

    def record_request(
        self,
        *,
        endpoint_name: str,
        ok: bool,
        status: int,
        latency_ms: float,
        error: str,
    ) -> None:
        phase = self.current_phase()
        with self.lock:
            phase_buckets = self.request_buckets.get(phase)
            if not isinstance(phase_buckets, dict):
                phase_buckets = {}
                self.request_buckets[phase] = phase_buckets
            bucket = phase_buckets.get(endpoint_name)
            if not isinstance(bucket, RequestBucket):
                bucket = RequestBucket()
                phase_buckets[endpoint_name] = bucket
            bucket.record(ok=ok, status=status, latency_ms=latency_ms, error=error)

    def record_guard(self, sample: GuardSample) -> None:
        with self.lock:
            self.guard_samples.append(sample)

    def record_core_tick(self, sample: CoreTickSample) -> None:
        with self.lock:
            self.core_ticks.append(sample)

    def record_ws_type(self, ws_type: str) -> None:
        with self.lock:
            self.ws_types[ws_type] += 1

    def record_ws_worker(self, worker_id: str) -> None:
        with self.lock:
            self.ws_workers[worker_id] += 1


def _simulation_reader(
    state: ProbeState,
    *,
    host: str,
    port: int,
    read_limit: int,
    timeout_seconds: float,
) -> None:
    while not state.worker_stop_event.is_set():
        ok, status, latency_ms, error = _http_get(
            host=host,
            port=port,
            path="/api/simulation?perspective=hybrid&compact=true",
            timeout_seconds=timeout_seconds,
            read_limit=read_limit,
        )
        state.record_request(
            endpoint_name="simulation",
            ok=ok,
            status=status,
            latency_ms=latency_ms,
            error=error,
        )
        time.sleep(0.15)


def _catalog_reader(
    state: ProbeState,
    *,
    host: str,
    port: int,
    read_limit: int,
    timeout_seconds: float,
) -> None:
    while not state.worker_stop_event.is_set():
        ok, status, latency_ms, error = _http_get(
            host=host,
            port=port,
            path="/api/catalog",
            timeout_seconds=timeout_seconds,
            read_limit=read_limit,
        )
        state.record_request(
            endpoint_name="catalog",
            ok=ok,
            status=status,
            latency_ms=latency_ms,
            error=error,
        )
        time.sleep(0.2)


def _witness_writer(
    state: ProbeState,
    *,
    host: str,
    port: int,
    timeout_seconds: float,
) -> None:
    while not state.worker_stop_event.is_set():
        payload = {
            "type": random.choice(["touch", "focus", "hover"]),
            "target": random.choice(["particle_field", "receipt_panel", "catalog"]),
        }
        ok, status, latency_ms, error = _http_post_json(
            host=host,
            port=port,
            path="/api/witness",
            payload=payload,
            timeout_seconds=timeout_seconds,
            read_limit=512,
        )
        state.record_request(
            endpoint_name="witness",
            ok=ok,
            status=status,
            latency_ms=latency_ms,
            error=error,
        )
        time.sleep(0.12)


def _user_input_writer(
    state: ProbeState,
    *,
    host: str,
    port: int,
    timeout_seconds: float,
) -> None:
    queries = [
        "homeostasis signal",
        "load balancing witness",
        "resource pressure adaptation",
        "tick governor response",
        "stream cadence probe",
    ]
    while not state.worker_stop_event.is_set():
        query = random.choice(queries)
        payload = {
            "kind": "search",
            "target": "simulation",
            "message": query,
            "embed_daimoi": True,
            "meta": {"query": query},
        }
        ok, status, latency_ms, error = _http_post_json(
            host=host,
            port=port,
            path="/api/presence/user/input",
            payload=payload,
            timeout_seconds=timeout_seconds,
            read_limit=1024,
        )
        state.record_request(
            endpoint_name="user_input",
            ok=ok,
            status=status,
            latency_ms=latency_ms,
            error=error,
        )
        time.sleep(0.25)


def _runtime_health_poller(
    state: ProbeState,
    *,
    host: str,
    port: int,
    timeout_seconds: float,
    interval_seconds: float,
) -> None:
    while not state.stop_event.is_set():
        started = time.monotonic()
        conn = HTTPConnection(host, port, timeout=timeout_seconds)
        ok = False
        status = 0
        error = ""
        body = b""
        try:
            conn.request("GET", "/api/runtime/health")
            response = conn.getresponse()
            status = int(response.status)
            body = response.read(16 * 1024)
            ok = 200 <= status < 300
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        finally:
            conn.close()

        elapsed_ms = (time.monotonic() - started) * 1000.0
        state.record_request(
            endpoint_name="runtime_health",
            ok=ok,
            status=status,
            latency_ms=elapsed_ms,
            error=error,
        )

        if ok:
            try:
                payload = json.loads(body.decode("utf-8"))
                guard = payload.get("guard", {}) if isinstance(payload, dict) else {}
                reasons = guard.get("reasons", [])
                if not isinstance(reasons, list):
                    reasons = []
                sample = GuardSample(
                    phase=state.current_phase(),
                    ts_mono=time.monotonic(),
                    mode=str(guard.get("mode", "")),
                    cpu_utilization=_safe_float(guard.get("cpu_utilization", 0.0), 0.0),
                    memory_pressure=_safe_float(guard.get("memory_pressure", 0.0), 0.0),
                    reasons=[str(item) for item in reasons],
                )
                state.record_guard(sample)
            except Exception:
                pass
        time.sleep(interval_seconds)


def _ws_monitor(
    state: ProbeState,
    *,
    host: str,
    port: int,
    ws_path: str,
) -> None:
    try:
        sock = _connect_ws(host=host, port=port, path=ws_path, timeout_seconds=8.0)
    except Exception as exc:  # noqa: BLE001
        state.ws_connect_error = str(exc)
        return

    try:
        fragment_opcode: int | None = None
        fragment_parts: list[bytes] = []
        chunk_buffers: dict[str, dict[str, Any]] = {}

        def _record_row(decoded_row: dict[str, Any]) -> None:
            ws_type = str(decoded_row.get("type", "") or "")
            state.record_ws_type(ws_type)

            if ws_type == "ws_chunk":
                chunk_id = str(decoded_row.get("chunk_id", "") or "")
                chunk_total = int(_safe_float(decoded_row.get("chunk_total", 0), 0.0))
                chunk_index = int(_safe_float(decoded_row.get("chunk_index", -1), -1.0))
                chunk_payload = decoded_row.get("payload", "")
                if (
                    not chunk_id
                    or chunk_total <= 0
                    or chunk_index < 0
                    or chunk_index >= chunk_total
                    or not isinstance(chunk_payload, str)
                ):
                    return

                if len(chunk_buffers) > 128:
                    chunk_buffers.clear()

                entry = chunk_buffers.get(chunk_id)
                if (
                    not isinstance(entry, dict)
                    or int(entry.get("total", 0)) != chunk_total
                ):
                    entry = {
                        "total": chunk_total,
                        "parts": [None] * chunk_total,
                        "received": 0,
                    }
                    chunk_buffers[chunk_id] = entry

                parts = entry.get("parts", [])
                if not isinstance(parts, list) or len(parts) != chunk_total:
                    return
                if parts[chunk_index] is None:
                    parts[chunk_index] = chunk_payload
                    entry["received"] = int(entry.get("received", 0)) + 1

                if int(entry.get("received", 0)) < chunk_total:
                    return
                if any(not isinstance(part, str) for part in parts):
                    return

                chunk_buffers.pop(chunk_id, None)
                assembled = "".join(str(part) for part in parts)
                try:
                    chunk_row = json.loads(assembled)
                except Exception:
                    state.ws_parse_errors += 1
                    return
                if isinstance(chunk_row, dict):
                    _record_row(chunk_row)
                return

            if ws_type != "simulation_delta":
                return

            worker_id = str(decoded_row.get("worker_id", "") or "")
            if worker_id:
                state.record_ws_worker(worker_id)

            if worker_id != "sim-core":
                return

            delta = decoded_row.get("delta", {})
            patch = delta.get("patch", {}) if isinstance(delta, dict) else {}
            if not isinstance(patch, dict):
                patch = {}

            tick_sample = CoreTickSample(
                phase=state.current_phase(),
                ts_mono=time.monotonic(),
                tick_elapsed_ms=_safe_float(
                    patch.get("tick_elapsed_ms", float("nan")), float("nan")
                ),
                slack_ms=_safe_float(patch.get("slack_ms", float("nan")), float("nan")),
                ingestion_pressure=_safe_float(
                    patch.get("ingestion_pressure", float("nan")),
                    float("nan"),
                ),
                ws_particle_max=int(
                    _safe_float(patch.get("ws_particle_max", 0.0), 0.0)
                ),
                payload_mode=str(patch.get("particle_payload_mode", "")),
            )
            state.record_core_tick(tick_sample)

        while not state.stop_event.is_set():
            try:
                opcode, is_final, payload = _read_ws_frame(sock)
            except socket.timeout:
                continue
            except Exception as exc:  # noqa: BLE001
                state.ws_stop_reason = str(exc)
                break

            if opcode == 0x8:
                break
            if opcode == 0x9:
                _send_ws_client_frame(sock, 0xA, payload[:125])
                continue

            message_payload: bytes | None = None
            if opcode == 0x1:
                if is_final:
                    message_payload = payload
                else:
                    fragment_opcode = opcode
                    fragment_parts = [payload]
                    continue
            elif opcode == 0x0:
                if fragment_opcode is None:
                    continue
                fragment_parts.append(payload)
                if not is_final:
                    continue
                if fragment_opcode == 0x1:
                    message_payload = b"".join(fragment_parts)
                fragment_opcode = None
                fragment_parts = []
            else:
                fragment_opcode = None
                fragment_parts = []
                continue

            try:
                row = json.loads(message_payload.decode("utf-8"))
            except Exception:
                state.ws_parse_errors += 1
                continue
            if not isinstance(row, dict):
                continue
            _record_row(row)
    finally:
        try:
            _send_ws_client_frame(sock, 0x8, b"\x03\xe8")
        except Exception:
            pass
        sock.close()


def _start_workers(
    state: ProbeState,
    *,
    host: str,
    port: int,
    sim_workers: int,
    catalog_workers: int,
    witness_workers: int,
    input_workers: int,
) -> list[threading.Thread]:
    threads: list[threading.Thread] = []

    for _ in range(sim_workers):
        thread = threading.Thread(
            target=_simulation_reader,
            args=(state,),
            kwargs={
                "host": host,
                "port": port,
                "read_limit": 128 * 1024,
                "timeout_seconds": 14.0,
            },
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for _ in range(catalog_workers):
        thread = threading.Thread(
            target=_catalog_reader,
            args=(state,),
            kwargs={
                "host": host,
                "port": port,
                "read_limit": 32 * 1024,
                "timeout_seconds": 10.0,
            },
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for _ in range(witness_workers):
        thread = threading.Thread(
            target=_witness_writer,
            args=(state,),
            kwargs={"host": host, "port": port, "timeout_seconds": 6.0},
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for _ in range(input_workers):
        thread = threading.Thread(
            target=_user_input_writer,
            args=(state,),
            kwargs={"host": host, "port": port, "timeout_seconds": 8.0},
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    return threads


def _summarize_requests(bucket: RequestBucket | None) -> dict[str, Any]:
    if not isinstance(bucket, RequestBucket):
        return {
            "total": 0,
            "ok": 0,
            "failures": 0,
            "timeouts": 0,
            "latency_ms": {"median": 0.0, "p95": 0.0, "max": 0.0},
            "status_codes": {},
        }
    latencies = bucket.latencies_ms
    return {
        "total": bucket.total,
        "ok": bucket.ok,
        "failures": bucket.failures,
        "timeouts": bucket.timeouts,
        "latency_ms": {
            "median": round(statistics.median(latencies), 2) if latencies else 0.0,
            "p95": round(_percentile(latencies, 0.95), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "status_codes": dict(bucket.statuses),
    }


def _summarize_phase(
    *,
    state: ProbeState,
    phase_name: str,
    phase_duration_seconds: float,
) -> dict[str, Any]:
    with state.lock:
        phase_buckets = state.request_buckets.get(phase_name, {})
        phase_guards = [
            sample for sample in state.guard_samples if sample.phase == phase_name
        ]
        phase_ticks = [
            sample for sample in state.core_ticks if sample.phase == phase_name
        ]

    request_summary = {
        endpoint: _summarize_requests(bucket)
        for endpoint, bucket in sorted(phase_buckets.items())
    }

    tick_times = [sample.ts_mono for sample in phase_ticks]
    tick_intervals = [
        tick_times[index] - tick_times[index - 1]
        for index in range(1, len(tick_times))
        if tick_times[index] >= tick_times[index - 1]
    ]
    elapsed_values = [
        sample.tick_elapsed_ms
        for sample in phase_ticks
        if sample.tick_elapsed_ms == sample.tick_elapsed_ms
    ]
    slack_values = [
        sample.slack_ms for sample in phase_ticks if sample.slack_ms == sample.slack_ms
    ]
    pressure_values = [
        sample.ingestion_pressure
        for sample in phase_ticks
        if sample.ingestion_pressure == sample.ingestion_pressure
    ]
    ws_particle_values = [
        sample.ws_particle_max for sample in phase_ticks if sample.ws_particle_max > 0
    ]
    payload_modes = sorted(
        {sample.payload_mode for sample in phase_ticks if sample.payload_mode}
    )

    mode_counts = Counter(sample.mode for sample in phase_guards if sample.mode)
    guard_cpu = [sample.cpu_utilization for sample in phase_guards]
    guard_mem = [sample.memory_pressure for sample in phase_guards]
    guard_reasons = Counter(
        reason for sample in phase_guards for reason in sample.reasons
    )

    return {
        "duration_s": round(phase_duration_seconds, 2),
        "requests": request_summary,
        "sim_core_ticks": {
            "count": len(phase_ticks),
            "fps": round(len(phase_ticks) / max(0.001, phase_duration_seconds), 3),
            "interval_s": {
                "median": round(statistics.median(tick_intervals), 4)
                if tick_intervals
                else 0.0,
                "p95": round(_percentile(tick_intervals, 0.95), 4)
                if tick_intervals
                else 0.0,
            },
            "tick_elapsed_ms": {
                "median": round(statistics.median(elapsed_values), 4)
                if elapsed_values
                else 0.0,
                "max": round(max(elapsed_values), 4) if elapsed_values else 0.0,
            },
            "slack_ms": {
                "median": round(statistics.median(slack_values), 4)
                if slack_values
                else 0.0,
                "min": round(min(slack_values), 4) if slack_values else 0.0,
            },
            "ingestion_pressure": {
                "median": round(statistics.median(pressure_values), 4)
                if pressure_values
                else 0.0,
                "max": round(max(pressure_values), 4) if pressure_values else 0.0,
            },
            "ws_particle_max": {
                "min": int(min(ws_particle_values)) if ws_particle_values else 0,
                "max": int(max(ws_particle_values)) if ws_particle_values else 0,
            },
            "payload_modes": payload_modes,
        },
        "runtime_guard": {
            "samples": len(phase_guards),
            "mode_counts": dict(mode_counts),
            "cpu_utilization": {
                "median": round(statistics.median(guard_cpu), 2) if guard_cpu else 0.0,
                "max": round(max(guard_cpu), 2) if guard_cpu else 0.0,
            },
            "memory_pressure": {
                "median": round(statistics.median(guard_mem), 4) if guard_mem else 0.0,
                "max": round(max(guard_mem), 4) if guard_mem else 0.0,
            },
            "reason_counts": dict(guard_reasons),
        },
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    phases = [
        {
            "name": "baseline",
            "duration_s": float(args.baseline_seconds),
            "sim_workers": int(args.baseline_sim_workers),
            "catalog_workers": 0,
            "witness_workers": 0,
            "input_workers": 0,
        },
        {
            "name": "soak",
            "duration_s": float(args.soak_seconds),
            "sim_workers": int(args.soak_sim_workers),
            "catalog_workers": int(args.soak_catalog_workers),
            "witness_workers": int(args.soak_witness_workers),
            "input_workers": int(args.soak_input_workers),
        },
        {
            "name": "recovery",
            "duration_s": float(args.recovery_seconds),
            "sim_workers": int(args.recovery_sim_workers),
            "catalog_workers": 0,
            "witness_workers": 0,
            "input_workers": 0,
        },
    ]

    state = ProbeState(
        phase_name=phases[0]["name"],
        run_start_mono=time.monotonic(),
    )

    ws_thread = threading.Thread(
        target=_ws_monitor,
        args=(state,),
        kwargs={
            "host": args.host,
            "port": int(args.port),
            "ws_path": args.ws_path,
        },
        daemon=True,
    )
    ws_thread.start()

    guard_thread = threading.Thread(
        target=_runtime_health_poller,
        args=(state,),
        kwargs={
            "host": args.host,
            "port": int(args.port),
            "timeout_seconds": 8.0,
            "interval_seconds": float(args.runtime_health_interval_seconds),
        },
        daemon=True,
    )
    guard_thread.start()

    phase_actual_durations: dict[str, float] = {}

    for phase in phases:
        phase_name = str(phase["name"])
        state.set_phase(phase_name)
        phase_started = time.monotonic()

        workers = _start_workers(
            state,
            host=args.host,
            port=int(args.port),
            sim_workers=int(phase["sim_workers"]),
            catalog_workers=int(phase["catalog_workers"]),
            witness_workers=int(phase["witness_workers"]),
            input_workers=int(phase["input_workers"]),
        )

        phase_end = phase_started + float(phase["duration_s"])
        while time.monotonic() < phase_end:
            time.sleep(0.2)

        state.worker_stop_event.set()
        for worker in workers:
            worker.join(timeout=3.0)
        state.worker_stop_event.clear()

        phase_actual_durations[phase_name] = time.monotonic() - phase_started

    state.stop_event.set()
    ws_thread.join(timeout=4.0)
    guard_thread.join(timeout=4.0)

    total_duration = time.monotonic() - state.run_start_mono

    phase_summaries: dict[str, Any] = {}
    for phase in phases:
        phase_name = str(phase["name"])
        phase_summaries[phase_name] = _summarize_phase(
            state=state,
            phase_name=phase_name,
            phase_duration_seconds=phase_actual_durations.get(
                phase_name,
                float(phase["duration_s"]),
            ),
        )

    target_fps = float(args.target_fps)
    soak_fps = _safe_float(
        phase_summaries.get("soak", {}).get("sim_core_ticks", {}).get("fps", 0.0),
        0.0,
    )

    overall = {
        "duration_s": round(total_duration, 2),
        "target_fps": target_fps,
        "soak_fps": round(soak_fps, 3),
        "soak_target_met": soak_fps >= (target_fps * 0.9),
        "ws_types": dict(state.ws_types),
        "ws_workers": dict(state.ws_workers),
        "ws_parse_errors": int(state.ws_parse_errors),
        "ws_connect_error": state.ws_connect_error,
        "ws_stop_reason": state.ws_stop_reason,
    }

    return {
        "ok": True,
        "record": "eta-mu.homeostasis-soak-probe.v1",
        "host": args.host,
        "port": int(args.port),
        "phases": phase_summaries,
        "overall": overall,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run phased external soak against eta-mu runtime and report "
            "WS cadence + guard adaptation."
        )
    )
    parser.add_argument("--host", default="eta-mu-system")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--ws-path",
        default=(
            "/ws?perspective=hybrid&delta_stream=workers&wire=json"
            "&simulation_payload=trimmed&particle_payload=lite"
            "&ws_chunk=1&catalog_events=0&skip_catalog_bootstrap=1"
        ),
    )
    parser.add_argument("--baseline-seconds", type=float, default=30.0)
    parser.add_argument("--baseline-sim-workers", type=int, default=0)
    parser.add_argument("--soak-seconds", type=float, default=90.0)
    parser.add_argument("--recovery-seconds", type=float, default=35.0)
    parser.add_argument("--recovery-sim-workers", type=int, default=0)
    parser.add_argument("--soak-sim-workers", type=int, default=0)
    parser.add_argument("--soak-catalog-workers", type=int, default=1)
    parser.add_argument("--soak-witness-workers", type=int, default=0)
    parser.add_argument("--soak-input-workers", type=int, default=0)
    parser.add_argument("--target-fps", type=float, default=12.0)
    parser.add_argument("--runtime-health-interval-seconds", type=float, default=8.0)
    args = parser.parse_args()

    result = run_probe(args)
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
