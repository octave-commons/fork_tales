from __future__ import annotations

import select
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SharedSimulationStreamFrame:
    seq: int
    payload: dict[str, Any]
    fingerprint: str


@dataclass
class SharedSimulationStreamLease:
    stream: "_SharedSimulationStream"

    def wait_for_frame(
        self,
        last_seq: int,
        *,
        timeout_seconds: float,
    ) -> SharedSimulationStreamFrame | None:
        return self.stream.wait_for_frame(last_seq, timeout_seconds=timeout_seconds)

    def close(self) -> None:
        self.stream.release()


class _SharedSimulationStream:
    def __init__(
        self,
        *,
        key: str,
        collect_frame: Callable[[], tuple[dict[str, Any], str] | None],
        refresh_seconds: float,
        heartbeat_seconds: float,
        release_callback: Callable[[str], None],
        idle_shutdown_seconds: float,
    ) -> None:
        self.key = key
        self.collect_frame = collect_frame
        self.refresh_seconds = max(0.1, float(refresh_seconds))
        self.heartbeat_seconds = max(0.5, float(heartbeat_seconds))
        self.idle_shutdown_seconds = max(0.5, float(idle_shutdown_seconds))
        self._release_callback = release_callback

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._subscriber_count = 0
        self._last_unsubscribed_monotonic = 0.0
        self._last_refresh_monotonic = 0.0
        self._last_publish_monotonic = 0.0
        self._frame: SharedSimulationStreamFrame | None = None
        self._seq = 0
        self._producer_thread: threading.Thread | None = None

    def acquire(self) -> SharedSimulationStreamLease:
        with self._lock:
            self._subscriber_count += 1
            self._last_unsubscribed_monotonic = 0.0
            if self._producer_thread is None or not self._producer_thread.is_alive():
                self._producer_thread = threading.Thread(
                    target=self._run_producer,
                    name=f"simulation-ws-shared:{self.key}",
                    daemon=True,
                )
                self._producer_thread.start()
        return SharedSimulationStreamLease(stream=self)

    def release(self) -> None:
        with self._lock:
            if self._subscriber_count > 0:
                self._subscriber_count -= 1
            if self._subscriber_count == 0:
                self._last_unsubscribed_monotonic = time.monotonic()
            self._condition.notify_all()

    def wait_for_frame(
        self,
        last_seq: int,
        *,
        timeout_seconds: float,
    ) -> SharedSimulationStreamFrame | None:
        timeout = max(0.0, float(timeout_seconds))
        deadline = time.monotonic() + timeout
        with self._lock:
            while True:
                frame = self._frame
                if frame is not None and frame.seq != int(last_seq):
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)

    def _run_producer(self) -> None:
        try:
            while True:
                now_monotonic = time.monotonic()
                with self._lock:
                    subscriber_count = self._subscriber_count
                    last_unsubscribed = self._last_unsubscribed_monotonic
                    last_refresh = self._last_refresh_monotonic
                if subscriber_count <= 0:
                    if (
                        last_unsubscribed > 0.0
                        and now_monotonic - last_unsubscribed
                        >= self.idle_shutdown_seconds
                    ):
                        break
                    time.sleep(0.05)
                    continue

                refresh_due = (
                    last_refresh <= 0.0
                    or now_monotonic - last_refresh >= self.refresh_seconds
                )
                if not refresh_due:
                    time.sleep(0.01)
                    continue

                collected_payload: dict[str, Any] | None = None
                collected_fingerprint = ""
                try:
                    collected = self.collect_frame()
                    if isinstance(collected, tuple) and len(collected) >= 2:
                        payload_candidate, fingerprint_candidate = collected
                        if isinstance(payload_candidate, dict):
                            collected_payload = payload_candidate
                            collected_fingerprint = str(fingerprint_candidate or "")
                except Exception:
                    collected_payload = None

                with self._lock:
                    self._last_refresh_monotonic = now_monotonic
                    if not isinstance(collected_payload, dict):
                        continue
                    fingerprint_changed = (
                        self._frame is None
                        or collected_fingerprint != self._frame.fingerprint
                    )
                    heartbeat_due = (
                        self._last_publish_monotonic <= 0.0
                        or now_monotonic - self._last_publish_monotonic
                        >= self.heartbeat_seconds
                    )
                    if not (fingerprint_changed or heartbeat_due):
                        continue
                    self._seq += 1
                    self._frame = SharedSimulationStreamFrame(
                        seq=self._seq,
                        payload=collected_payload,
                        fingerprint=collected_fingerprint,
                    )
                    self._last_publish_monotonic = now_monotonic
                    self._condition.notify_all()
        finally:
            self._release_callback(self.key)


class SharedSimulationStreamRegistry:
    def __init__(self) -> None:
        self._streams: dict[str, _SharedSimulationStream] = {}
        self._lock = threading.Lock()

    def subscribe(
        self,
        *,
        stream_key: str,
        collect_frame: Callable[[], tuple[dict[str, Any], str] | None],
        refresh_seconds: float,
        heartbeat_seconds: float,
        idle_shutdown_seconds: float = 3.0,
    ) -> SharedSimulationStreamLease:
        with self._lock:
            stream = self._streams.get(stream_key)
            if stream is None:
                stream = _SharedSimulationStream(
                    key=stream_key,
                    collect_frame=collect_frame,
                    refresh_seconds=refresh_seconds,
                    heartbeat_seconds=heartbeat_seconds,
                    release_callback=self._release_stream,
                    idle_shutdown_seconds=idle_shutdown_seconds,
                )
                self._streams[stream_key] = stream
        return stream.acquire()

    def _release_stream(self, stream_key: str) -> None:
        with self._lock:
            stream = self._streams.get(stream_key)
            if stream is None:
                return
            with stream._lock:
                if stream._subscriber_count <= 0:
                    self._streams.pop(stream_key, None)

    def clear(self) -> None:
        with self._lock:
            self._streams.clear()


_SHARED_STREAM_REGISTRY = SharedSimulationStreamRegistry()


def reset_shared_simulation_stream_registry_for_tests() -> None:
    _SHARED_STREAM_REGISTRY.clear()


def handle_shared_simulation_websocket_stream(
    *,
    connection: Any,
    send_ws: Callable[[dict[str, Any]], None],
    consume_ws_client_frame: Callable[[Any], bool],
    release_client_slot: Callable[[], None],
    stream_key: str,
    collect_frame: Callable[[], tuple[dict[str, Any], str] | None],
    refresh_seconds: float,
    heartbeat_seconds: float,
) -> None:
    lease = _SHARED_STREAM_REGISTRY.subscribe(
        stream_key=stream_key,
        collect_frame=collect_frame,
        refresh_seconds=refresh_seconds,
        heartbeat_seconds=heartbeat_seconds,
    )
    poll = select.poll()
    poll.register(connection, select.POLLIN)
    last_seq = 0
    try:
        while True:
            frame = lease.wait_for_frame(last_seq, timeout_seconds=0.25)
            if frame is not None and frame.seq != last_seq:
                send_ws(frame.payload)
                last_seq = frame.seq
            ready = poll.poll(10)
            if ready and not consume_ws_client_frame(connection):
                break
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        pass
    finally:
        lease.close()
        release_client_slot()
