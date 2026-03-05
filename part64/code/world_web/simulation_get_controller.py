from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable


@dataclass(frozen=True)
class SimulationGetDependencies:
    default_perspective: str
    full_async_rebuild_enabled: bool
    cache_seconds: float
    compact_stale_fallback_seconds: float
    disk_cold_start_seconds: float
    full_async_stale_max_age_seconds: float
    disk_cache_seconds: float
    stale_fallback_seconds: float
    disk_fallback_max_age_seconds: float
    compact_build_wait_seconds: float
    build_wait_seconds: float
    build_lock_acquire_timeout_seconds: float
    simulation_build_lock: Any
    normalize_projection_perspective: Callable[[str], str]
    normalize_payload_mode: Callable[[Any], str]
    safe_bool_query: Callable[[str, bool], bool]
    safe_float: Callable[[Any, float], float]
    json_compact: Callable[[Any], str]
    simulation_http_request_profile: Callable[..., Any]
    simulation_http_schedule_full_async_refresh: Callable[
        ..., tuple[bool, dict[str, Any]]
    ]
    simulation_http_send_response: Callable[..., None]
    simulation_http_refresh_retry_payload: Callable[..., tuple[int, dict[str, Any]]]
    simulation_http_stale_or_disk_body: Callable[..., tuple[bytes | None, str]]
    simulation_http_fallback_headers: Callable[..., dict[str, str]]
    simulation_http_error_payload: Callable[..., dict[str, Any]]
    simulation_http_acquire_build_lock_or_respond: Callable[..., bool]
    simulation_http_send_inflight_cached_response_if_any: Callable[..., bool]
    simulation_http_full_async_refresh_headers: Callable[..., dict[str, str]]
    simulation_http_compact_stale_fallback_body: Callable[..., tuple[bytes | None, str]]
    simulation_http_is_cold_start: Callable[[], bool]
    simulation_http_cache_key: Callable[..., str]
    simulation_http_cached_body: Callable[..., bytes | None]
    simulation_http_disk_cache_load: Callable[..., bytes | None]
    simulation_http_compact_cached_body: Callable[..., bytes | None]
    simulation_http_cache_store: Callable[..., None]
    simulation_http_compact_cache_store: Callable[..., None]
    simulation_http_compact_response_body: Callable[..., bytes]
    simulation_http_compact_simulation_payload: Callable[..., dict[str, Any]]
    simulation_http_disk_cache_store: Callable[..., None]
    simulation_http_failure_backoff_snapshot: Callable[[], tuple[float, str, int]]
    simulation_http_failure_clear: Callable[[], None]
    simulation_http_failure_record: Callable[[str], None]
    simulation_http_wait_for_exact_cache: Callable[..., bytes | None]
    simulation_http_full_async_refresh_snapshot: Callable[[], dict[str, Any]]
    simulation_ws_decode_cached_payload: Callable[[bytes], Any]
    simulation_ws_payload_missing_graph_payload: Callable[[dict[str, Any]], bool]


def handle_simulation_get(
    *,
    params: dict[str, list[str]],
    part_root: Any,
    runtime_catalog: Callable[
        ..., tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any], Any]
    ],
    runtime_simulation: Callable[..., tuple[dict[str, Any], dict[str, Any]]],
    schedule_full_simulation_async_refresh: Callable[..., tuple[bool, dict[str, Any]]],
    send_json: Callable[..., None],
    send_bytes: Callable[..., None],
    dependencies: SimulationGetDependencies,
) -> None:
    request_profile = dependencies.simulation_http_request_profile(
        params,
        default_perspective=dependencies.default_perspective,
        full_async_rebuild_enabled=dependencies.full_async_rebuild_enabled,
        normalize_projection_perspective=dependencies.normalize_projection_perspective,
        normalize_payload_mode=dependencies.normalize_payload_mode,
        safe_bool_query=dependencies.safe_bool_query,
    )
    perspective = request_profile.perspective
    compact_response_mode = request_profile.compact_response_mode
    full_async_preferred = request_profile.full_async_preferred
    cache_profile = request_profile.cache_profile
    cache_perspective = request_profile.cache_perspective
    cache_key = ""

    def _schedule_full_async_refresh(*, trigger: str) -> tuple[bool, dict[str, Any]]:
        return dependencies.simulation_http_schedule_full_async_refresh(
            full_async_preferred=full_async_preferred,
            cache_key=cache_key,
            cache_perspective=cache_perspective,
            perspective=perspective,
            trigger=trigger,
            schedule_refresh=schedule_full_simulation_async_refresh,
            refresh_snapshot=dependencies.simulation_http_full_async_refresh_snapshot,
        )

    def _graph_safe_cached_body(body: bytes | None) -> bytes | None:
        if body is None:
            return None
        if not compact_response_mode:
            return body
        decoded_payload = dependencies.simulation_ws_decode_cached_payload(body)
        if isinstance(
            decoded_payload, dict
        ) and dependencies.simulation_ws_payload_missing_graph_payload(decoded_payload):
            return None
        return body

    def _graph_safe_cached_body_reader(**kwargs: Any) -> bytes | None:
        return _graph_safe_cached_body(
            dependencies.simulation_http_cached_body(**kwargs)
        )

    def _graph_safe_disk_cache_loader(*args: Any, **kwargs: Any) -> bytes | None:
        return _graph_safe_cached_body(
            dependencies.simulation_http_disk_cache_load(*args, **kwargs)
        )

    def _graph_safe_compact_cached_body_reader(**kwargs: Any) -> bytes | None:
        return _graph_safe_cached_body(
            dependencies.simulation_http_compact_cached_body(**kwargs)
        )

    def _send_simulation_response(
        body: bytes,
        *,
        cache_key_for_compact: str = "",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        dependencies.simulation_http_send_response(
            body,
            compact_response_mode=compact_response_mode,
            cache_key_for_compact=cache_key_for_compact,
            cache_seconds=dependencies.cache_seconds,
            compact_cached_body_reader=_graph_safe_compact_cached_body_reader,
            compact_cache_store=dependencies.simulation_http_compact_cache_store,
            compact_response_body_builder=dependencies.simulation_http_compact_response_body,
            send_bytes=send_bytes,
            content_type="application/json; charset=utf-8",
            extra_headers=extra_headers,
        )

    if compact_response_mode:
        cached_compact_by_perspective = _graph_safe_compact_cached_body_reader(
            perspective=cache_perspective,
            max_age_seconds=dependencies.cache_seconds,
        )
        if cached_compact_by_perspective is not None:
            send_bytes(
                cached_compact_by_perspective,
                "application/json; charset=utf-8",
                extra_headers={"X-Eta-Mu-Simulation-Fallback": "compact-cache"},
            )
            return

        stale_compact_body, stale_compact_source = (
            dependencies.simulation_http_compact_stale_fallback_body(
                part_root=part_root,
                perspective=cache_perspective,
                max_age_seconds=dependencies.compact_stale_fallback_seconds,
            )
        )
        stale_compact_body = _graph_safe_cached_body(stale_compact_body)
        if stale_compact_body is not None:
            fallback_source = (
                str(stale_compact_source or "stale-cache").strip().lower()
                or "stale-cache"
            )
            _send_simulation_response(
                stale_compact_body,
                cache_key_for_compact=(
                    f"{cache_perspective}|{fallback_source}|compact-fallback|simulation"
                ),
                extra_headers={
                    "X-Eta-Mu-Simulation-Fallback": f"{fallback_source}-compact"
                },
            )
            return

    try:
        if compact_response_mode and dependencies.simulation_http_is_cold_start():
            cold_disk_body = _graph_safe_disk_cache_loader(
                part_root,
                perspective=cache_perspective,
                max_age_seconds=dependencies.disk_cold_start_seconds,
            )
            if cold_disk_body is not None:
                dependencies.simulation_http_cache_store(
                    f"{cache_perspective}|disk-cold-start|simulation",
                    cold_disk_body,
                )
                _send_simulation_response(
                    cold_disk_body,
                    cache_key_for_compact=(
                        f"{cache_perspective}|disk-cold-start|simulation"
                    ),
                    extra_headers={
                        "X-Eta-Mu-Simulation-Fallback": "disk-cache-cold-start"
                    },
                )
                return

        catalog, queue_snapshot, _, influence_snapshot, _ = runtime_catalog(
            perspective=perspective,
            include_projection=False,
            include_runtime_fields=False,
            allow_inline_collect=False,
        )
        if compact_response_mode and (
            str(catalog.get("runtime_state", "") or "").strip().lower() == "fallback"
        ):
            try:
                (
                    refreshed_catalog,
                    refreshed_queue_snapshot,
                    _,
                    refreshed_influence_snapshot,
                    _,
                ) = runtime_catalog(
                    perspective=perspective,
                    include_projection=False,
                    include_runtime_fields=False,
                    allow_inline_collect=True,
                    strict_collect=True,
                )
                catalog = refreshed_catalog
                queue_snapshot = refreshed_queue_snapshot
                influence_snapshot = refreshed_influence_snapshot
            except Exception:
                pass

        user_inputs_recent = int(
            dependencies.safe_float(influence_snapshot.get("user_inputs_120s", 0), 0.0)
        )
        cache_key = dependencies.simulation_http_cache_key(
            perspective=perspective,
            catalog=catalog,
            queue_snapshot=queue_snapshot,
            influence_snapshot=influence_snapshot,
        )
        cache_key = f"{cache_key}|profile:{cache_profile}"
        cached_body = _graph_safe_cached_body_reader(
            cache_key=cache_key,
            perspective=cache_perspective,
            max_age_seconds=dependencies.cache_seconds,
            require_exact_key=True,
        )
        if cached_body is not None:
            _send_simulation_response(cached_body, cache_key_for_compact=cache_key)
            return

        if full_async_preferred:
            stale_full_body, stale_source = (
                dependencies.simulation_http_stale_or_disk_body(
                    part_root=part_root,
                    cache_perspective=cache_perspective,
                    stale_max_age_seconds=dependencies.full_async_stale_max_age_seconds,
                    disk_max_age_seconds=dependencies.disk_cache_seconds,
                    cache_key=cache_key,
                    cached_body_reader=_graph_safe_cached_body_reader,
                    disk_cache_loader=_graph_safe_disk_cache_loader,
                    cache_store=dependencies.simulation_http_cache_store,
                )
            )

            refresh_scheduled, refresh_snapshot = _schedule_full_async_refresh(
                trigger="full-cache-miss"
            )
            refresh_headers = dependencies.simulation_http_full_async_refresh_headers(
                refresh_snapshot,
                scheduled=refresh_scheduled,
            )
            if stale_full_body is not None:
                response_headers = dependencies.simulation_http_fallback_headers(
                    fallback=stale_source,
                    error="full_async_refresh",
                )
                response_headers.update(refresh_headers)
                _send_simulation_response(
                    stale_full_body,
                    cache_key_for_compact=cache_key,
                    extra_headers=response_headers,
                )
                return

            response_status, response_payload = (
                dependencies.simulation_http_refresh_retry_payload(
                    refresh_snapshot=refresh_snapshot,
                    refresh_scheduled=refresh_scheduled,
                    perspective=perspective,
                    safe_float=dependencies.safe_float,
                )
            )
            send_json(response_payload, status=response_status)
            return

        if compact_response_mode and user_inputs_recent <= 0:
            disk_cached_body = _graph_safe_disk_cache_loader(
                part_root,
                perspective=cache_perspective,
                max_age_seconds=max(
                    dependencies.stale_fallback_seconds,
                    dependencies.disk_fallback_max_age_seconds,
                ),
            )
            if disk_cached_body is not None:
                dependencies.simulation_http_cache_store(cache_key, disk_cached_body)
                _send_simulation_response(
                    disk_cached_body,
                    cache_key_for_compact=cache_key,
                    extra_headers={"X-Eta-Mu-Simulation-Fallback": "disk-cache"},
                )
                return

        backoff_remaining, backoff_error, backoff_streak = (
            dependencies.simulation_http_failure_backoff_snapshot()
        )
        if backoff_remaining > 0.0:
            stale_body, stale_source = dependencies.simulation_http_stale_or_disk_body(
                part_root=part_root,
                cache_perspective=cache_perspective,
                stale_max_age_seconds=dependencies.stale_fallback_seconds,
                disk_max_age_seconds=dependencies.disk_cache_seconds,
                cache_key=cache_key,
                cached_body_reader=_graph_safe_cached_body_reader,
                disk_cache_loader=_graph_safe_disk_cache_loader,
                cache_store=dependencies.simulation_http_cache_store,
            )
            if stale_body is not None:
                _send_simulation_response(
                    stale_body,
                    cache_key_for_compact=cache_key,
                    extra_headers=dependencies.simulation_http_fallback_headers(
                        fallback=(
                            "failure-backoff"
                            if stale_source == "stale-cache"
                            else stale_source
                        ),
                        error=backoff_error or "build_failure",
                        backoff_remaining_seconds=backoff_remaining,
                        backoff_streak=backoff_streak,
                    ),
                )
                return

        lock_acquired = dependencies.simulation_http_acquire_build_lock_or_respond(
            build_lock=dependencies.simulation_build_lock,
            compact_response_mode=compact_response_mode,
            compact_build_wait_seconds=dependencies.compact_build_wait_seconds,
            build_wait_seconds=dependencies.build_wait_seconds,
            cache_key=cache_key,
            cache_perspective=cache_perspective,
            stale_fallback_seconds=dependencies.stale_fallback_seconds,
            disk_cache_seconds=dependencies.disk_cache_seconds,
            build_lock_acquire_timeout_seconds=(
                dependencies.build_lock_acquire_timeout_seconds
            ),
            part_root=part_root,
            wait_for_exact_cache=dependencies.simulation_http_wait_for_exact_cache,
            cached_body_reader=_graph_safe_cached_body_reader,
            disk_cache_loader=_graph_safe_disk_cache_loader,
            cache_store=dependencies.simulation_http_cache_store,
            send_simulation_response=_send_simulation_response,
            fallback_headers_builder=dependencies.simulation_http_fallback_headers,
            error_payload_builder=dependencies.simulation_http_error_payload,
            send_json=send_json,
            perspective=perspective,
            service_unavailable_status=int(HTTPStatus.SERVICE_UNAVAILABLE),
        )
        if not lock_acquired:
            return

        try:
            if dependencies.simulation_http_send_inflight_cached_response_if_any(
                cache_key=cache_key,
                cache_perspective=cache_perspective,
                cache_seconds=dependencies.cache_seconds,
                cached_body_reader=_graph_safe_cached_body_reader,
                send_simulation_response=_send_simulation_response,
                fallback_headers_builder=dependencies.simulation_http_fallback_headers,
            ):
                return

            simulation, projection = runtime_simulation(
                catalog,
                queue_snapshot,
                influence_snapshot,
                perspective=perspective,
                include_unified_graph=not compact_response_mode,
                include_particle_dynamics=not compact_response_mode,
            )
            simulation["projection"] = projection
            response_body = dependencies.json_compact(simulation).encode("utf-8")
            dependencies.simulation_http_cache_store(cache_key, response_body)
            if compact_response_mode:
                compact_cache_key = f"{cache_key}|compact"
                compact_payload = (
                    dependencies.simulation_http_compact_simulation_payload(simulation)
                )
                dependencies.simulation_http_compact_cache_store(
                    compact_cache_key,
                    dependencies.json_compact(compact_payload).encode("utf-8"),
                )
            dependencies.simulation_http_disk_cache_store(
                part_root,
                perspective=cache_perspective,
                body=response_body,
            )
            dependencies.simulation_http_failure_clear()
            _send_simulation_response(response_body, cache_key_for_compact=cache_key)
        finally:
            if lock_acquired:
                dependencies.simulation_build_lock.release()
    except Exception as exc:
        dependencies.simulation_http_failure_record(exc.__class__.__name__)
        stale_body, stale_source = dependencies.simulation_http_stale_or_disk_body(
            part_root=part_root,
            cache_perspective=cache_perspective,
            stale_max_age_seconds=dependencies.stale_fallback_seconds,
            disk_max_age_seconds=dependencies.disk_cache_seconds,
            cache_key=cache_key,
            cached_body_reader=_graph_safe_cached_body_reader,
            disk_cache_loader=_graph_safe_disk_cache_loader,
            cache_store=dependencies.simulation_http_cache_store,
        )
        if stale_body is not None:
            _send_simulation_response(
                stale_body,
                cache_key_for_compact=cache_key,
                extra_headers=dependencies.simulation_http_fallback_headers(
                    fallback=stale_source,
                    error=exc.__class__.__name__,
                ),
            )
            return
        send_json(
            dependencies.simulation_http_error_payload(
                perspective=perspective,
                error="simulation_unavailable",
                detail=exc.__class__.__name__,
            ),
            status=HTTPStatus.SERVICE_UNAVAILABLE,
        )
