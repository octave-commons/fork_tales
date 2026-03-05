from __future__ import annotations

from typing import Any, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from code.world_web import simulation_mvc_models as models_module
from code.world_web import simulation_mvc_router as router_module


class _FakeSimulationController:
    def __init__(self) -> None:
        self.simulation_get_calls: list[dict[str, list[str]]] = []
        self.refresh_calls: list[dict[str, Any]] = []
        self.bootstrap_calls: list[dict[str, Any]] = []
        self.refresh_status_calls: list[dict[str, list[str]]] = []
        self.bootstrap_status_calls = 0
        self.presets_calls = 0
        self.instances_calls = 0
        self.instances_spawn_calls: list[dict[str, Any]] = []
        self.instance_delete_calls: list[str] = []

    def handle_refresh_post(
        self,
        request_model: models_module.SimulationRefreshRequestModel,
    ) -> models_module.SimulationControllerResponseModel:
        self.refresh_calls.append(dict(request_model.payload))
        return models_module.SimulationControllerResponseModel(
            payload={"ok": True, "record": "mvc.refresh"},
            status_code=202,
        )

    def handle_simulation_get(
        self,
        query_model: models_module.SimulationGetQueryModel,
    ) -> models_module.SimulationControllerResponseModel:
        self.simulation_get_calls.append(dict(query_model.params))
        return models_module.SimulationControllerResponseModel(
            payload={"ok": True, "record": "mvc.simulation"},
            status_code=200,
        )

    def handle_bootstrap_post(
        self,
        request_model: models_module.SimulationBootstrapRequestModel,
    ) -> models_module.SimulationControllerResponseModel:
        self.bootstrap_calls.append(dict(request_model.payload))
        return models_module.SimulationControllerResponseModel(
            payload={"ok": True, "record": "mvc.bootstrap"},
            status_code=200,
        )

    def handle_refresh_status_get(
        self,
        query_model: models_module.SimulationRefreshStatusQueryModel,
    ) -> models_module.SimulationControllerResponseModel:
        self.refresh_status_calls.append(dict(query_model.params))
        return models_module.SimulationControllerResponseModel(
            payload={"ok": True, "record": "mvc.refresh-status"},
            status_code=200,
        )

    def handle_bootstrap_status_get(
        self,
    ) -> models_module.SimulationControllerResponseModel:
        self.bootstrap_status_calls += 1
        return models_module.SimulationControllerResponseModel(
            payload={"ok": True, "record": "mvc.bootstrap-status"},
            status_code=200,
        )

    def handle_presets_get(self) -> models_module.SimulationControllerResponseModel:
        self.presets_calls += 1
        return models_module.SimulationControllerResponseModel(
            payload={"ok": True, "record": "mvc.presets"},
            status_code=200,
        )

    def handle_instances_get(self) -> models_module.SimulationControllerResponseModel:
        self.instances_calls += 1
        return models_module.SimulationControllerResponseModel(
            payload=[{"id": "sim-1"}],
            status_code=200,
        )

    def handle_instances_spawn_post(
        self,
        request_model: models_module.SimulationInstancesSpawnRequestModel,
    ) -> models_module.SimulationControllerResponseModel:
        self.instances_spawn_calls.append(dict(request_model.payload))
        return models_module.SimulationControllerResponseModel(
            payload={"ok": True, "record": "mvc.instances-spawn"},
            status_code=200,
        )

    def handle_instance_delete(
        self,
        path_model: models_module.SimulationInstancePathModel,
    ) -> models_module.SimulationControllerResponseModel:
        self.instance_delete_calls.append(path_model.instance_id)
        return models_module.SimulationControllerResponseModel(
            payload={"ok": True, "record": "mvc.instance-delete"},
            status_code=200,
        )


def _build_app(controller: Any | None) -> FastAPI:
    app = FastAPI()
    app.include_router(router_module.router)
    app.state.simulation_mvc_controller = controller
    return app


def test_simulation_mvc_refresh_route_uses_controller_payload() -> None:
    controller = _FakeSimulationController()
    client = TestClient(_build_app(controller))

    response = client.post(
        "/api/simulation/refresh",
        json={"action": "start", "perspective": "hybrid"},
    )

    assert response.status_code == 202
    assert response.json().get("record") == "mvc.refresh"
    assert controller.refresh_calls == [{"action": "start", "perspective": "hybrid"}]


def test_simulation_mvc_simulation_get_route_uses_controller_query() -> None:
    controller = _FakeSimulationController()
    client = TestClient(_build_app(controller))

    response = client.get("/api/simulation?payload=trimmed&perspective=hybrid")

    assert response.status_code == 200
    assert response.json().get("record") == "mvc.simulation"
    assert controller.simulation_get_calls == [
        {"payload": ["trimmed"], "perspective": ["hybrid"]}
    ]


def test_simulation_mvc_bootstrap_route_uses_controller_payload() -> None:
    controller = _FakeSimulationController()
    client = TestClient(_build_app(controller))

    response = client.post(
        "/api/simulation/bootstrap",
        json={"wait": True, "perspective": "hybrid"},
    )

    assert response.status_code == 200
    assert response.json().get("record") == "mvc.bootstrap"
    assert controller.bootstrap_calls == [{"wait": True, "perspective": "hybrid"}]


def test_simulation_mvc_refresh_route_handles_invalid_json_body() -> None:
    controller = _FakeSimulationController()
    client = TestClient(_build_app(controller))

    response = client.post(
        "/api/simulation/refresh",
        data=cast(Any, "{not-json"),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 202
    assert response.json().get("record") == "mvc.refresh"
    assert controller.refresh_calls == [{}]


def test_simulation_mvc_refresh_status_route_uses_controller_query() -> None:
    controller = _FakeSimulationController()
    client = TestClient(_build_app(controller))

    response = client.get("/api/simulation/refresh-status?perspective=hybrid")

    assert response.status_code == 200
    assert response.json().get("record") == "mvc.refresh-status"
    assert controller.refresh_status_calls == [{"perspective": ["hybrid"]}]


def test_simulation_mvc_bootstrap_status_route_uses_controller() -> None:
    controller = _FakeSimulationController()
    client = TestClient(_build_app(controller))

    response = client.get("/api/simulation/bootstrap")

    assert response.status_code == 200
    assert response.json().get("record") == "mvc.bootstrap-status"
    assert controller.bootstrap_status_calls == 1


def test_simulation_mvc_presets_route_uses_controller() -> None:
    controller = _FakeSimulationController()
    client = TestClient(_build_app(controller))

    response = client.get("/api/simulation/presets")

    assert response.status_code == 200
    assert response.json().get("record") == "mvc.presets"
    assert controller.presets_calls == 1


def test_simulation_mvc_instances_route_uses_controller() -> None:
    controller = _FakeSimulationController()
    client = TestClient(_build_app(controller))

    response = client.get("/api/simulation/instances")

    assert response.status_code == 200
    assert response.json() == [{"id": "sim-1"}]
    assert controller.instances_calls == 1


def test_simulation_mvc_instances_spawn_route_uses_controller_payload() -> None:
    controller = _FakeSimulationController()
    client = TestClient(_build_app(controller))

    response = client.post(
        "/api/simulation/instances/spawn",
        json={"preset_id": "fast"},
    )

    assert response.status_code == 200
    assert response.json().get("record") == "mvc.instances-spawn"
    assert controller.instances_spawn_calls == [{"preset_id": "fast"}]


def test_simulation_mvc_instances_delete_route_uses_controller_path() -> None:
    controller = _FakeSimulationController()
    client = TestClient(_build_app(controller))

    response = client.delete("/api/simulation/instances/sim-1")

    assert response.status_code == 200
    assert response.json().get("record") == "mvc.instance-delete"
    assert controller.instance_delete_calls == ["sim-1"]


def test_simulation_mvc_refresh_route_returns_503_without_controller() -> None:
    client = TestClient(_build_app(None))

    response = client.post("/api/simulation/refresh", json={"action": "start"})

    assert response.status_code == 503
    payload = response.json()
    assert payload.get("ok") is False
    assert payload.get("error") == "simulation_controller_unavailable"


def test_simulation_mvc_refresh_status_route_returns_503_without_controller() -> None:
    client = TestClient(_build_app(None))

    response = client.get("/api/simulation/refresh-status")

    assert response.status_code == 503
    payload = response.json()
    assert payload.get("ok") is False
    assert payload.get("error") == "simulation_controller_unavailable"


def test_simulation_mvc_instances_route_returns_503_without_controller() -> None:
    client = TestClient(_build_app(None))

    response = client.get("/api/simulation/instances")

    assert response.status_code == 503
    payload = response.json()
    assert payload.get("ok") is False
    assert payload.get("error") == "simulation_controller_unavailable"


def test_simulation_mvc_simulation_get_route_returns_503_without_controller() -> None:
    client = TestClient(_build_app(None))

    response = client.get("/api/simulation?payload=trimmed")

    assert response.status_code == 503
    payload = response.json()
    assert payload.get("ok") is False
    assert payload.get("error") == "simulation_controller_unavailable"
