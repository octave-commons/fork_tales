from __future__ import annotations

import select
import time
from typing import Any, Callable


def handle_docker_websocket_stream(
    *,
    connection: Any,
    send_ws: Callable[[dict[str, Any]], None],
    collect_docker_simulation_snapshot: Callable[..., dict[str, Any]],
    consume_ws_client_frame: Callable[[Any], bool],
    release_client_slot: Callable[[], None],
    docker_refresh_seconds: float,
    docker_heartbeat_seconds: float,
) -> None:
    poll = select.poll()
    poll.register(connection, select.POLLIN)

    last_docker_refresh = time.monotonic()
    last_docker_broadcast = last_docker_refresh
    last_docker_fingerprint = ""

    try:
        try:
            docker_snapshot = collect_docker_simulation_snapshot(force_refresh=True)
            last_docker_fingerprint = str(docker_snapshot.get("fingerprint", "") or "")
            send_ws({"type": "docker_simulations", "docker": docker_snapshot})
        except Exception as exc:
            send_ws(
                {
                    "type": "docker_simulations_error",
                    "error": exc.__class__.__name__,
                }
            )

        while True:
            now_monotonic = time.monotonic()
            if now_monotonic - last_docker_refresh >= docker_refresh_seconds:
                docker_snapshot = collect_docker_simulation_snapshot()
                docker_fingerprint = str(docker_snapshot.get("fingerprint", "") or "")
                docker_changed = docker_fingerprint != last_docker_fingerprint
                docker_heartbeat_due = (
                    now_monotonic - last_docker_broadcast >= docker_heartbeat_seconds
                )
                if docker_changed or docker_heartbeat_due:
                    send_ws({"type": "docker_simulations", "docker": docker_snapshot})
                    last_docker_broadcast = now_monotonic
                    last_docker_fingerprint = docker_fingerprint
                last_docker_refresh = now_monotonic

            ready = poll.poll(10)
            if ready and not consume_ws_client_frame(connection):
                break
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        pass
    finally:
        release_client_slot()
