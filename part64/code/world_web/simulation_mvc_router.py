from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, Request

from .simulation_mvc_models import (
    SimulationBootstrapRequestModel,
    SimulationControllerResponseModel,
    SimulationGetQueryModel,
    SimulationInstancePathModel,
    SimulationInstancesSpawnRequestModel,
    SimulationRefreshRequestModel,
    SimulationRefreshStatusQueryModel,
)
from .simulation_mvc_views import render_simulation_controller_response

router = APIRouter()


def _controller_from_request(request: Request) -> Any:
    return getattr(request.app.state, "simulation_mvc_controller", None)


async def _payload_dict_from_request(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _controller_missing_response() -> SimulationControllerResponseModel:
    return SimulationControllerResponseModel(
        payload={"ok": False, "error": "simulation_controller_unavailable"},
        status_code=int(HTTPStatus.SERVICE_UNAVAILABLE),
    )


@router.post("/api/simulation/refresh")
async def post_simulation_refresh(request: Request) -> Any:
    controller = _controller_from_request(request)
    if controller is None:
        return render_simulation_controller_response(_controller_missing_response())

    request_payload = await _payload_dict_from_request(request)
    request_model = SimulationRefreshRequestModel.from_raw(request_payload)
    response_model = controller.handle_refresh_post(request_model)
    return render_simulation_controller_response(response_model)


@router.get("/api/simulation")
async def get_simulation(request: Request) -> Any:
    controller = _controller_from_request(request)
    if controller is None:
        return render_simulation_controller_response(_controller_missing_response())

    query_model = SimulationGetQueryModel.from_raw(request.query_params)
    response_model = controller.handle_simulation_get(query_model)
    return render_simulation_controller_response(response_model)


@router.get("/api/simulation/refresh-status")
async def get_simulation_refresh_status(request: Request) -> Any:
    controller = _controller_from_request(request)
    if controller is None:
        return render_simulation_controller_response(_controller_missing_response())

    query_model = SimulationRefreshStatusQueryModel.from_raw(request.query_params)
    response_model = controller.handle_refresh_status_get(query_model)
    return render_simulation_controller_response(response_model)


@router.post("/api/simulation/bootstrap")
async def post_simulation_bootstrap(request: Request) -> Any:
    controller = _controller_from_request(request)
    if controller is None:
        return render_simulation_controller_response(_controller_missing_response())

    request_payload = await _payload_dict_from_request(request)
    request_model = SimulationBootstrapRequestModel.from_raw(request_payload)
    response_model = controller.handle_bootstrap_post(request_model)
    return render_simulation_controller_response(response_model)


@router.get("/api/simulation/presets")
async def get_simulation_presets(request: Request) -> Any:
    controller = _controller_from_request(request)
    if controller is None:
        return render_simulation_controller_response(_controller_missing_response())

    response_model = controller.handle_presets_get()
    return render_simulation_controller_response(response_model)


@router.get("/api/simulation/instances")
async def get_simulation_instances(request: Request) -> Any:
    controller = _controller_from_request(request)
    if controller is None:
        return render_simulation_controller_response(_controller_missing_response())

    response_model = controller.handle_instances_get()
    return render_simulation_controller_response(response_model)


@router.post("/api/simulation/instances/spawn")
async def post_simulation_instances_spawn(request: Request) -> Any:
    controller = _controller_from_request(request)
    if controller is None:
        return render_simulation_controller_response(_controller_missing_response())

    request_payload = await _payload_dict_from_request(request)
    request_model = SimulationInstancesSpawnRequestModel.from_raw(request_payload)
    response_model = controller.handle_instances_spawn_post(request_model)
    return render_simulation_controller_response(response_model)


@router.delete("/api/simulation/instances/{instance_id}")
async def delete_simulation_instance(request: Request, instance_id: str) -> Any:
    controller = _controller_from_request(request)
    if controller is None:
        return render_simulation_controller_response(_controller_missing_response())

    path_model = SimulationInstancePathModel.from_raw(instance_id)
    response_model = controller.handle_instance_delete(path_model)
    return render_simulation_controller_response(response_model)


@router.get("/api/simulation/bootstrap")
async def get_simulation_bootstrap_status(request: Request) -> Any:
    controller = _controller_from_request(request)
    if controller is None:
        return render_simulation_controller_response(_controller_missing_response())

    response_model = controller.handle_bootstrap_status_get()
    return render_simulation_controller_response(response_model)
