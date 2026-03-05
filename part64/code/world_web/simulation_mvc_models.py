from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _coerce_payload_dict(raw_payload: Any) -> dict[str, Any]:
    if not isinstance(raw_payload, dict):
        return {}
    return {str(key): value for key, value in raw_payload.items()}


def _coerce_query_params(raw_query: Any) -> dict[str, list[str]]:
    pairs: list[tuple[str, str]] = []
    if hasattr(raw_query, "multi_items"):
        try:
            pairs = [
                (str(key), str(value))
                for key, value in raw_query.multi_items()  # type: ignore[attr-defined]
            ]
        except Exception:
            pairs = []
    elif isinstance(raw_query, dict):
        for key, value in raw_query.items():
            if isinstance(value, list):
                pairs.extend((str(key), str(row)) for row in value)
            else:
                pairs.append((str(key), str(value)))

    query: dict[str, list[str]] = {}
    for key, value in pairs:
        query.setdefault(key, []).append(value)
    return query


@dataclass(frozen=True)
class SimulationRefreshRequestModel:
    payload: dict[str, Any]

    @classmethod
    def from_raw(cls, raw_payload: Any) -> "SimulationRefreshRequestModel":
        return cls(payload=_coerce_payload_dict(raw_payload))


@dataclass(frozen=True)
class SimulationBootstrapRequestModel:
    payload: dict[str, Any]

    @classmethod
    def from_raw(cls, raw_payload: Any) -> "SimulationBootstrapRequestModel":
        return cls(payload=_coerce_payload_dict(raw_payload))


@dataclass(frozen=True)
class SimulationInstancesSpawnRequestModel:
    payload: dict[str, Any]

    @classmethod
    def from_raw(cls, raw_payload: Any) -> "SimulationInstancesSpawnRequestModel":
        return cls(payload=_coerce_payload_dict(raw_payload))


@dataclass(frozen=True)
class SimulationInstancePathModel:
    instance_id: str

    @classmethod
    def from_raw(cls, raw_instance_id: Any) -> "SimulationInstancePathModel":
        return cls(instance_id=str(raw_instance_id or "").strip())


@dataclass(frozen=True)
class SimulationControllerResponseModel:
    payload: Any
    status_code: int
    headers: dict[str, str] | None = None
    raw_body: bytes | None = None
    content_type: str | None = None


@dataclass(frozen=True)
class SimulationGetQueryModel:
    params: dict[str, list[str]]

    @classmethod
    def from_raw(cls, raw_query: Any) -> "SimulationGetQueryModel":
        return cls(params=_coerce_query_params(raw_query))


@dataclass(frozen=True)
class SimulationRefreshStatusQueryModel:
    params: dict[str, list[str]]

    @classmethod
    def from_raw(cls, raw_query: Any) -> "SimulationRefreshStatusQueryModel":
        return cls(params=_coerce_query_params(raw_query))
