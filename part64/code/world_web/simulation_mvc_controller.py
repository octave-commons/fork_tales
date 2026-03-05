from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable

from . import server as server_module
from . import simulation_get_controller as simulation_get_controller_module
from . import (
    simulation_management_controller as simulation_management_controller_module,
)
from . import simulation_post_controller as simulation_post_controller_module
from . import simulation_status_command_utils as simulation_status_command_utils_module
from .simulation_mvc_models import (
    SimulationBootstrapRequestModel,
    SimulationControllerResponseModel,
    SimulationGetQueryModel,
    SimulationInstancePathModel,
    SimulationInstancesSpawnRequestModel,
    SimulationRefreshRequestModel,
    SimulationRefreshStatusQueryModel,
)


def _coerce_status_code(status: Any, fallback: int) -> int:
    try:
        return int(getattr(status, "value", status))
    except Exception:
        return int(fallback)


def _capture_json_response(
    *,
    command: Callable[[Callable[..., None]], None],
    fallback_error: str,
) -> SimulationControllerResponseModel:
    captured_payload: dict[str, Any] | None = None
    captured_status = int(HTTPStatus.OK)

    def _send_json(payload: dict[str, Any], status: int = 200) -> None:
        nonlocal captured_payload, captured_status
        captured_payload = dict(payload) if isinstance(payload, dict) else {}
        captured_status = _coerce_status_code(status, int(HTTPStatus.OK))

    command(_send_json)
    if captured_payload is None:
        return SimulationControllerResponseModel(
            payload={"ok": False, "error": fallback_error},
            status_code=int(HTTPStatus.INTERNAL_SERVER_ERROR),
        )
    return SimulationControllerResponseModel(
        payload=captured_payload,
        status_code=captured_status,
    )


@dataclass
class SimulationMvcController:
    handler_class: Any

    def _build_handler(self) -> Any:
        return self.handler_class.__new__(self.handler_class)

    def handle_refresh_post(
        self,
        request_model: SimulationRefreshRequestModel,
    ) -> SimulationControllerResponseModel:
        handler = self._build_handler()

        def _command(send_json: Callable[..., None]) -> None:
            simulation_post_controller_module.handle_simulation_refresh_post(
                req=dict(request_model.payload),
                default_perspective=server_module.PROJECTION_DEFAULT_PERSPECTIVE,
                normalize_projection_perspective=server_module.normalize_projection_perspective,
                safe_bool_query=server_module._safe_bool_query,
                full_async_refresh_enabled=server_module._SIMULATION_HTTP_FULL_ASYNC_REBUILD_ENABLED,
                full_async_refresh_snapshot=server_module._simulation_http_full_async_refresh_snapshot,
                full_async_refresh_cancel=server_module._simulation_http_full_async_refresh_cancel,
                schedule_full_simulation_async_refresh=handler._schedule_full_simulation_async_refresh,
                safe_float=server_module._safe_float,
                send_json=send_json,
            )

        return _capture_json_response(
            command=_command,
            fallback_error="simulation_refresh_no_response",
        )

    def handle_simulation_get(
        self,
        query_model: SimulationGetQueryModel,
    ) -> SimulationControllerResponseModel:
        handler = self._build_handler()
        captured_status = int(HTTPStatus.OK)
        captured_payload: Any | None = None
        captured_headers: dict[str, str] = {}
        captured_raw_body: bytes | None = None
        captured_content_type: str | None = None

        def _capture_send_json(payload: Any, status: int = 200) -> None:
            nonlocal captured_status, captured_payload
            nonlocal captured_headers, captured_raw_body, captured_content_type
            captured_status = _coerce_status_code(status, int(HTTPStatus.OK))
            captured_payload = payload
            captured_headers = {}
            captured_raw_body = None
            captured_content_type = None

        def _capture_send_bytes(
            payload: bytes,
            content_type: str,
            status: int = 200,
            *,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            nonlocal captured_status, captured_payload
            nonlocal captured_headers, captured_raw_body, captured_content_type
            captured_status = _coerce_status_code(status, int(HTTPStatus.OK))
            captured_payload = None
            captured_headers = {
                str(key): str(value) for key, value in dict(extra_headers or {}).items()
            }
            captured_raw_body = bytes(payload)
            captured_content_type = str(content_type or "application/octet-stream")

        simulation_get_controller_module.handle_simulation_get(
            params=dict(query_model.params),
            part_root=handler.part_root,
            runtime_catalog=handler._runtime_catalog,
            runtime_simulation=handler._runtime_simulation,
            schedule_full_simulation_async_refresh=handler._schedule_full_simulation_async_refresh,
            send_json=_capture_send_json,
            send_bytes=_capture_send_bytes,
            dependencies=handler._simulation_get_dependencies(),
        )

        if captured_raw_body is not None:
            return SimulationControllerResponseModel(
                payload=None,
                status_code=int(captured_status),
                headers=dict(captured_headers),
                raw_body=bytes(captured_raw_body),
                content_type=str(captured_content_type or "application/octet-stream"),
            )

        if captured_payload is None:
            return SimulationControllerResponseModel(
                payload={"ok": False, "error": "simulation_get_no_response"},
                status_code=int(HTTPStatus.INTERNAL_SERVER_ERROR),
            )

        return SimulationControllerResponseModel(
            payload=captured_payload,
            status_code=int(captured_status),
        )

    def handle_bootstrap_post(
        self,
        request_model: SimulationBootstrapRequestModel,
    ) -> SimulationControllerResponseModel:
        handler = self._build_handler()

        def _command(send_json: Callable[..., None]) -> None:
            simulation_post_controller_module.handle_simulation_bootstrap_post(
                req=dict(request_model.payload),
                default_perspective=server_module.PROJECTION_DEFAULT_PERSPECTIVE,
                normalize_projection_perspective=server_module.normalize_projection_perspective,
                safe_bool_query=server_module._safe_bool_query,
                run_simulation_bootstrap=handler._run_simulation_bootstrap,
                simulation_bootstrap_job_start=server_module._simulation_bootstrap_job_start,
                simulation_bootstrap_job_mark_phase=server_module._simulation_bootstrap_job_mark_phase,
                simulation_bootstrap_job_complete=server_module._simulation_bootstrap_job_complete,
                simulation_bootstrap_job_fail=server_module._simulation_bootstrap_job_fail,
                send_json=send_json,
            )

        return _capture_json_response(
            command=_command,
            fallback_error="simulation_bootstrap_no_response",
        )

    def handle_refresh_status_get(
        self,
        query_model: SimulationRefreshStatusQueryModel,
    ) -> SimulationControllerResponseModel:
        handler = self._build_handler()
        refresh_context = (
            simulation_status_command_utils_module.simulation_refresh_status_context(
                query_model.params,
                default_perspective=server_module.PROJECTION_DEFAULT_PERSPECTIVE,
                normalize_projection_perspective=(
                    server_module.normalize_projection_perspective
                ),
            )
        )
        perspective = refresh_context.perspective
        cache_perspective = refresh_context.cache_perspective
        refresh_snapshot = server_module._simulation_http_full_async_refresh_snapshot()
        refresh_public = simulation_status_command_utils_module.simulation_refresh_public_snapshot(
            refresh_snapshot,
            perspective=perspective,
            throttle_remaining_seconds=(
                lambda snapshot, perspective_key: (
                    server_module._simulation_http_full_async_throttle_remaining_seconds(
                        snapshot,
                        perspective=perspective_key,
                    )
                )
            ),
            safe_float=server_module._safe_float,
        )
        availability = simulation_status_command_utils_module.simulation_refresh_status_availability(
            part_root=handler.part_root,
            cache_perspective=cache_perspective,
            cache_seconds=server_module._SIMULATION_HTTP_CACHE_SECONDS,
            stale_max_age_seconds=(
                server_module._SIMULATION_HTTP_FULL_ASYNC_STALE_MAX_AGE_SECONDS
            ),
            disk_cache_seconds=server_module._SIMULATION_HTTP_DISK_CACHE_SECONDS,
            cached_body_reader=server_module._simulation_http_cached_body,
            disk_cache_has_payload=server_module._simulation_http_disk_cache_has_payload,
        )

        payload = simulation_status_command_utils_module.simulation_refresh_status_payload(
            perspective=perspective,
            full_async_enabled=server_module._SIMULATION_HTTP_FULL_ASYNC_REBUILD_ENABLED,
            cache_seconds=server_module._SIMULATION_HTTP_CACHE_SECONDS,
            stale_max_age_seconds=(
                server_module._SIMULATION_HTTP_FULL_ASYNC_STALE_MAX_AGE_SECONDS
            ),
            lock_timeout_seconds=(
                server_module._SIMULATION_HTTP_FULL_ASYNC_LOCK_TIMEOUT_SECONDS
            ),
            max_running_seconds=server_module._SIMULATION_HTTP_FULL_ASYNC_MAX_RUNNING_SECONDS,
            start_min_interval_seconds=(
                server_module._SIMULATION_HTTP_FULL_ASYNC_START_MIN_INTERVAL_SECONDS
            ),
            availability=availability,
            refresh_public=refresh_public,
        )
        return SimulationControllerResponseModel(
            payload=payload,
            status_code=int(HTTPStatus.OK),
        )

    def handle_bootstrap_status_get(self) -> SimulationControllerResponseModel:
        payload = (
            simulation_status_command_utils_module.simulation_bootstrap_status_payload(
                job_snapshot=server_module._simulation_bootstrap_job_snapshot(),
                report=server_module._simulation_bootstrap_snapshot_report(),
            )
        )
        return SimulationControllerResponseModel(
            payload=payload,
            status_code=int(HTTPStatus.OK),
        )

    def handle_presets_get(self) -> SimulationControllerResponseModel:
        handler = self._build_handler()
        payload, status_code = (
            simulation_management_controller_module.simulation_presets_get_response(
                part_root=handler.part_root
            )
        )
        return SimulationControllerResponseModel(
            payload=payload,
            status_code=int(status_code),
        )

    def handle_instances_get(self) -> SimulationControllerResponseModel:
        handler = self._build_handler()
        payload, status_code = (
            simulation_management_controller_module.simulation_instances_list_response(
                part_root=handler.part_root
            )
        )
        return SimulationControllerResponseModel(
            payload=payload,
            status_code=int(status_code),
        )

    def handle_instances_spawn_post(
        self,
        request_model: SimulationInstancesSpawnRequestModel,
    ) -> SimulationControllerResponseModel:
        handler = self._build_handler()
        payload, status_code = (
            simulation_management_controller_module.simulation_instances_spawn_response(
                part_root=handler.part_root,
                req_payload=dict(request_model.payload),
            )
        )
        return SimulationControllerResponseModel(
            payload=payload,
            status_code=int(status_code),
        )

    def handle_instance_delete(
        self,
        path_model: SimulationInstancePathModel,
    ) -> SimulationControllerResponseModel:
        handler = self._build_handler()
        payload, status_code = (
            simulation_management_controller_module.simulation_instance_delete_response(
                part_root=handler.part_root,
                instance_id=path_model.instance_id,
            )
        )
        return SimulationControllerResponseModel(
            payload=payload,
            status_code=int(status_code),
        )
