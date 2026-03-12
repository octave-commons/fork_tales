from __future__ import annotations

from typing import Any, Callable


def stream_muse_bootstrap_events(
    *,
    handler: Any,
    send_ws: Callable[[dict[str, Any]], None],
    muse_event_seq: int,
    enabled: bool,
    event_limit: int = 96,
) -> int:
    if not enabled:
        return int(muse_event_seq)

    muse_bootstrap_events = handler._muse_manager().list_events(
        since_seq=0,
        limit=max(1, int(event_limit)),
    )
    if muse_bootstrap_events:
        muse_event_seq = max(
            int(muse_event_seq),
            max(
                int(row.get("seq", 0))
                for row in muse_bootstrap_events
                if isinstance(row, dict)
            ),
        )
        send_ws(
            {
                "type": "muse_events",
                "events": muse_bootstrap_events,
                "since_seq": 0,
                "next_seq": muse_event_seq,
            }
        )
    return int(muse_event_seq)


def maybe_send_muse_events(
    *,
    handler: Any,
    send_ws: Callable[[dict[str, Any]], None],
    now_monotonic: float,
    muse_event_seq: int,
    last_muse_poll: float,
    server_module: Any,
    event_limit: int = 96,
) -> tuple[int, float]:
    if (
        float(now_monotonic) - float(last_muse_poll)
        < server_module._SIMULATION_WS_MUSE_POLL_SECONDS
    ):
        return int(muse_event_seq), float(last_muse_poll)

    try:
        handler._muse_threat_radar_tick(
            now_monotonic=now_monotonic,
            force=False,
            reason="ws.poll",
        )
    except Exception:
        pass

    previous_muse_seq = int(muse_event_seq)
    muse_events = handler._muse_manager().list_events(
        since_seq=previous_muse_seq,
        limit=max(1, int(event_limit)),
    )
    if muse_events:
        muse_event_seq = max(
            previous_muse_seq,
            max(int(row.get("seq", 0)) for row in muse_events if isinstance(row, dict)),
        )
        send_ws(
            {
                "type": "muse_events",
                "events": muse_events,
                "since_seq": previous_muse_seq,
                "next_seq": muse_event_seq,
            }
        )
    return int(muse_event_seq), float(now_monotonic)
