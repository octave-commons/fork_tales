from __future__ import annotations

from fastapi.responses import Response
from fastapi.responses import JSONResponse

from .simulation_mvc_models import SimulationControllerResponseModel


def render_simulation_controller_response(
    response_model: SimulationControllerResponseModel,
) -> Response:
    response_headers = {
        str(key): str(value)
        for key, value in dict(response_model.headers or {}).items()
    }
    if response_model.raw_body is not None:
        if response_model.content_type:
            response_headers.setdefault(
                "Content-Type", str(response_model.content_type)
            )
        return Response(
            content=bytes(response_model.raw_body),
            status_code=int(response_model.status_code),
            headers=response_headers,
        )

    return JSONResponse(
        content=response_model.payload,
        status_code=int(response_model.status_code),
        headers=response_headers,
    )
