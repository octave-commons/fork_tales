"""Runtime IO helpers for multipart parsing and weaver bring-up."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


def weaver_probe_host(bind_host: str) -> str:
    host = bind_host.strip().lower()
    if not host or host in {"0.0.0.0", "::", "localhost"}:
        return "127.0.0.1"
    return bind_host


def weaver_health_check(host: str, port: int, timeout_s: float = 0.8) -> bool:
    target = f"http://{host}:{port}/healthz"
    req = Request(target, method="GET")
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return int(getattr(resp, "status", 0)) == 200
    except Exception:
        return False


def ensure_weaver_service(
    part_root: Path,
    world_host: str,
    *,
    weaver_autostart: bool,
    weaver_host_env: str,
    weaver_port: int,
    weaver_probe_host_fn: Callable[[str], str],
    weaver_health_check_fn: Callable[[str, int], bool],
) -> None:
    del part_root
    if not weaver_autostart:
        return
    probe_host = weaver_probe_host_fn(weaver_host_env or world_host)
    if weaver_health_check_fn(probe_host, weaver_port):
        return

    script_path = (
        Path(__file__).resolve().parent.parent / "web_graph_weaver.js"
    ).resolve()
    if not script_path.exists() or not script_path.is_file():
        return
    node_binary = shutil.which("node")
    if not node_binary:
        return

    env = os.environ.copy()
    env.setdefault("WEAVER_HOST", weaver_host_env)
    env.setdefault("WEAVER_PORT", str(weaver_port))
    try:
        subprocess.Popen(
            [node_binary, str(script_path)],
            cwd=str(script_path.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        return


def parse_multipart_form(raw_body: bytes, content_type: str) -> dict[str, Any] | None:
    match = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type, re.I)
    if match is None:
        return None
    boundary_token = (match.group(1) or match.group(2) or "").strip()
    if not boundary_token:
        return None

    delimiter = b"--" + boundary_token.encode("utf-8", errors="ignore")
    data: dict[str, Any] = {}
    for part in raw_body.split(delimiter):
        chunk = part.strip()
        if not chunk or chunk == b"--":
            continue
        if chunk.endswith(b"--"):
            chunk = chunk[:-2].strip()
        head, sep, body = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        if body.endswith(b"\r\n"):
            body = body[:-2]

        disposition = ""
        part_content_type = ""
        for line in head.decode("utf-8", errors="ignore").split("\r\n"):
            low = line.lower()
            if low.startswith("content-disposition:"):
                disposition = line.split(":", 1)[1].strip()
            elif low.startswith("content-type:"):
                part_content_type = line.split(":", 1)[1].strip()

        name_match = re.search(r'name="([^"]+)"', disposition)
        if name_match is None:
            continue
        field_name = name_match.group(1)
        file_match = re.search(r'filename="([^"]*)"', disposition)
        if file_match is not None:
            data[field_name] = {
                "filename": file_match.group(1),
                "content_type": part_content_type,
                "value": body,
            }
        else:
            data[field_name] = body.decode("utf-8", errors="ignore")
    return data


def resolve_artifact_path(part_root: Path, request_path: str) -> Path | None:
    parsed = urlparse(request_path)
    raw_path = unquote(parsed.path)
    if not raw_path.startswith("/artifacts/"):
        return None

    relative = raw_path.removeprefix("/")
    if not relative:
        return None

    candidate = (part_root / relative).resolve()
    artifacts_root = (part_root / "artifacts").resolve()
    if artifacts_root == candidate or artifacts_root in candidate.parents:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None
