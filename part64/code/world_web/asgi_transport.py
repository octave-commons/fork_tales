from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response

from . import legacy_handler_dispatch as legacy_handler_dispatch_module
from .server import _schedule_simulation_http_warmup, create_http_server
from . import simulation_mvc_controller as simulation_mvc_controller_module
from . import simulation_mvc_router as simulation_mvc_router_module

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

_ASGI_PROXY_CONNECT_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("WORLD_WEB_ASGI_PROXY_CONNECT_TIMEOUT_SECONDS", "5.0") or "5.0"),
)
_ASGI_PROXY_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("WORLD_WEB_ASGI_PROXY_TIMEOUT_SECONDS", "120.0") or "120.0"),
)
_ASGI_PROXY_SIMULATION_FULL_TIMEOUT_SECONDS = max(
    2.0,
    float(
        os.getenv("WORLD_WEB_ASGI_PROXY_SIMULATION_FULL_TIMEOUT_SECONDS", "45.0")
        or "45.0"
    ),
)
_ASGI_KEEPALIVE_SECONDS = max(
    1,
    int(float(os.getenv("WORLD_WEB_ASGI_KEEPALIVE_SECONDS", "25") or "25")),
)
_ASGI_BACKLOG = max(
    64,
    int(float(os.getenv("WORLD_WEB_ASGI_BACKLOG", "512") or "512")),
)
_ASGI_LIMIT_CONCURRENCY = max(
    0,
    int(float(os.getenv("WORLD_WEB_ASGI_LIMIT_CONCURRENCY", "384") or "384")),
)
_ASGI_WS_OPEN_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("WORLD_WEB_ASGI_WS_OPEN_TIMEOUT_SECONDS", "8.0") or "8.0"),
)
_ASGI_WS_CLOSE_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("WORLD_WEB_ASGI_WS_CLOSE_TIMEOUT_SECONDS", "4.0") or "4.0"),
)
_ASGI_WS_PING_INTERVAL_SECONDS = max(
    0.0,
    float(os.getenv("WORLD_WEB_ASGI_WS_PING_INTERVAL_SECONDS", "12.0") or "12.0"),
)
_ASGI_WS_PING_TIMEOUT_SECONDS = max(
    0.2,
    float(os.getenv("WORLD_WEB_ASGI_WS_PING_TIMEOUT_SECONDS", "12.0") or "12.0"),
)
_ASGI_NATIVE_DISPATCH_PATHS = {
    "/",
    "/healthz",
    "/api/catalog",
    "/api/simulation/refresh",
    "/api/simulation/refresh-status",
    "/api/ui/projection",
}
_ASGI_NATIVE_DISPATCH_SIMULATION_PAYLOADS = {"trimmed", "lite", "compact"}


def _pick_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    return max(1, port)


@dataclass
class _LegacyProxyRuntime:
    host: str
    port: int
    server: Any
    handler_class: Any
    thread: threading.Thread
    client: httpx.AsyncClient


def _proxy_request_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() in _HOP_BY_HOP_HEADERS:
            continue
        headers[key] = value

    headers["x-forwarded-proto"] = request.url.scheme
    client = request.client
    if client is not None and client.host:
        existing = headers.get("x-forwarded-for", "").strip()
        headers["x-forwarded-for"] = (
            f"{existing}, {client.host}" if existing else client.host
        )
    return headers


def _proxy_response_headers(headers: httpx.Headers) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in _HOP_BY_HOP_HEADERS:
            continue
        out[key] = value
    return out


def _resolve_legacy_port(legacy_port: int) -> int:
    if int(legacy_port) > 0:
        return int(legacy_port)
    return _pick_loopback_port()


def _safe_bool_string(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _native_dispatch_enabled_for_request(request: Request) -> bool:
    path = str(request.url.path or "").strip() or "/"
    if path.startswith("/api/weaver/"):
        return True
    if path in _ASGI_NATIVE_DISPATCH_PATHS:
        return True
    if path != "/api/simulation":
        return False

    payload = str(request.query_params.get("payload", "full") or "full").strip().lower()
    compact = _safe_bool_string(str(request.query_params.get("compact", "") or ""))
    return compact or payload in _ASGI_NATIVE_DISPATCH_SIMULATION_PAYLOADS


def create_asgi_transport_app(
    *,
    part_root: Path,
    vault_root: Path,
    host: str,
    port: int,
    legacy_host: str = "127.0.0.1",
    legacy_port: int = 0,
) -> FastAPI:
    part_root_resolved = part_root.resolve()
    vault_root_resolved = vault_root.resolve()
    legacy_bind_port = _resolve_legacy_port(int(legacy_port))
    host_label = f"{host}:{int(port)}"

    app = FastAPI(title="eta-mu-world-asgi-transport")
    app.include_router(simulation_mvc_router_module.router)

    @app.on_event("startup")
    async def _startup() -> None:
        legacy_server = create_http_server(
            part_root_resolved,
            vault_root_resolved,
            legacy_host,
            legacy_bind_port,
            host_label=host_label,
        )
        backend_port = int(legacy_server.server_address[1])
        server_thread = threading.Thread(
            target=legacy_server.serve_forever,
            kwargs={"poll_interval": 0.25},
            daemon=True,
            name="eta-mu-legacy-http",
        )
        server_thread.start()

        timeout = httpx.Timeout(
            timeout=_ASGI_PROXY_TIMEOUT_SECONDS,
            connect=_ASGI_PROXY_CONNECT_TIMEOUT_SECONDS,
            read=_ASGI_PROXY_TIMEOUT_SECONDS,
            write=_ASGI_PROXY_TIMEOUT_SECONDS,
        )
        client = httpx.AsyncClient(
            base_url=f"http://{legacy_host}:{backend_port}",
            timeout=timeout,
            follow_redirects=False,
        )
        app.state.legacy_runtime = _LegacyProxyRuntime(
            host=legacy_host,
            port=backend_port,
            server=legacy_server,
            handler_class=legacy_server.RequestHandlerClass,
            thread=server_thread,
            client=client,
        )
        app.state.simulation_mvc_controller = (
            simulation_mvc_controller_module.SimulationMvcController(
                handler_class=legacy_server.RequestHandlerClass,
            )
        )
        _schedule_simulation_http_warmup(host=legacy_host, port=backend_port)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        runtime = cast(
            _LegacyProxyRuntime | None, getattr(app.state, "legacy_runtime", None)
        )
        if runtime is None:
            return
        with contextlib.suppress(Exception):
            await runtime.client.aclose()
        with contextlib.suppress(Exception):
            runtime.server.shutdown()
        with contextlib.suppress(Exception):
            runtime.server.server_close()
        with contextlib.suppress(Exception):
            runtime.thread.join(timeout=3.0)

    @app.websocket("/ws")
    async def _proxy_ws(websocket: WebSocket) -> None:
        runtime = cast(_LegacyProxyRuntime, app.state.legacy_runtime)
        backend_url = f"ws://{runtime.host}:{runtime.port}/ws"
        query = str(websocket.url.query or "").strip()
        if query:
            backend_url = f"{backend_url}?{query}"

        accepted = False
        try:
            async with websockets.connect(
                backend_url,
                open_timeout=_ASGI_WS_OPEN_TIMEOUT_SECONDS,
                close_timeout=_ASGI_WS_CLOSE_TIMEOUT_SECONDS,
                ping_interval=(
                    _ASGI_WS_PING_INTERVAL_SECONDS
                    if _ASGI_WS_PING_INTERVAL_SECONDS > 0.0
                    else None
                ),
                ping_timeout=_ASGI_WS_PING_TIMEOUT_SECONDS,
                max_size=None,
            ) as upstream:
                await websocket.accept()
                accepted = True

                async def _client_to_upstream() -> None:
                    while True:
                        event = await websocket.receive()
                        event_type = str(event.get("type", ""))
                        if event_type == "websocket.disconnect":
                            break
                        data = event.get("bytes")
                        text = event.get("text")
                        if isinstance(data, (bytes, bytearray)):
                            await upstream.send(bytes(data))
                        elif isinstance(text, str):
                            await upstream.send(text)

                async def _upstream_to_client() -> None:
                    while True:
                        message = await upstream.recv()
                        if isinstance(message, (bytes, bytearray)):
                            await websocket.send_bytes(bytes(message))
                        else:
                            await websocket.send_text(str(message))

                client_task = asyncio.create_task(_client_to_upstream())
                upstream_task = asyncio.create_task(_upstream_to_client())
                done, pending = await asyncio.wait(
                    {client_task, upstream_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                for task in done:
                    task.result()
        except Exception:
            if not accepted:
                with contextlib.suppress(Exception):
                    await websocket.accept()
            with contextlib.suppress(Exception):
                await websocket.close(code=1011)

    @app.api_route(
        "/", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
    )
    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def _proxy_http(path: str, request: Request) -> Response:
        runtime = cast(_LegacyProxyRuntime, app.state.legacy_runtime)
        upstream_path = f"/{path}" if path else "/"
        if request.url.query:
            upstream_path = f"{upstream_path}?{request.url.query}"

        payload = await request.body()
        request_headers = _proxy_request_headers(request)

        if _native_dispatch_enabled_for_request(request):
            direct = await asyncio.to_thread(
                legacy_handler_dispatch_module.dispatch_via_legacy_handler,
                handler_class=runtime.handler_class,
                method=request.method,
                path=upstream_path,
                headers=request_headers,
                body=payload,
            )
            if direct is not None:
                response_headers = dict(direct.extra_headers)
                response_headers["Content-Type"] = direct.content_type
                response_body = b"" if request.method.upper() == "HEAD" else direct.body
                return Response(
                    content=response_body,
                    status_code=direct.status_code,
                    headers=response_headers,
                )

        try:
            per_request_timeout: httpx.Timeout | None = None
            if str(request.url.path or "") == "/api/simulation":
                payload_mode = (
                    str(request.query_params.get("payload", "full") or "full")
                    .strip()
                    .lower()
                )
                compact_mode = _safe_bool_string(
                    str(request.query_params.get("compact", "") or "")
                )
                if (payload_mode in {"", "full"}) and not compact_mode:
                    per_request_timeout = httpx.Timeout(
                        timeout=_ASGI_PROXY_SIMULATION_FULL_TIMEOUT_SECONDS,
                        connect=_ASGI_PROXY_CONNECT_TIMEOUT_SECONDS,
                        read=_ASGI_PROXY_SIMULATION_FULL_TIMEOUT_SECONDS,
                        write=_ASGI_PROXY_SIMULATION_FULL_TIMEOUT_SECONDS,
                    )

            upstream = await runtime.client.request(
                request.method,
                upstream_path,
                content=payload,
                headers=request_headers,
                timeout=per_request_timeout,
            )
        except httpx.TimeoutException as exc:
            payload = {
                "ok": False,
                "error": "asgi_upstream_timeout",
                "detail": f"{exc.__class__.__name__}: {exc}",
                "upstream_path": upstream_path,
            }
            return Response(
                content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                status_code=504,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
        except httpx.HTTPError as exc:
            payload = {
                "ok": False,
                "error": "asgi_upstream_error",
                "detail": f"{exc.__class__.__name__}: {exc}",
                "upstream_path": upstream_path,
            }
            return Response(
                content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                status_code=502,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_proxy_response_headers(upstream.headers),
        )

    return app


def run_asgi_transport(
    *,
    part_root: Path,
    vault_root: Path,
    host: str,
    port: int,
    legacy_port: int = 0,
) -> None:
    import uvicorn

    app = create_asgi_transport_app(
        part_root=part_root,
        vault_root=vault_root,
        host=host,
        port=port,
        legacy_port=legacy_port,
    )
    limit_concurrency = _ASGI_LIMIT_CONCURRENCY if _ASGI_LIMIT_CONCURRENCY > 0 else None
    uvicorn.run(
        app,
        host=host,
        port=int(port),
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=_ASGI_KEEPALIVE_SECONDS,
        limit_concurrency=limit_concurrency,
        backlog=_ASGI_BACKLOG,
        ws="websockets",
    )
