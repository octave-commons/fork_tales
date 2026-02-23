#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from http.client import HTTPConnection

# Resolve paths relative to script
SCRIPT_PATH = Path(__file__).resolve()
PART_ROOT = SCRIPT_PATH.parent.parent
PRESETS_PATH = PART_ROOT / "world_state" / "sim_presets.json"
COMPOSE_FILE = PART_ROOT / "docker-compose.sim-slice-bench.yml"
DOCKER_SOCKET = "/var/run/docker.sock"


def _load_proxy_api_key() -> str:
    direct = str(os.getenv("OPENVINO_EMBED_API_KEY", "") or "").strip()
    if direct:
        return direct
    proxy_direct = str(os.getenv("PROXY_API_KEY", "") or "").strip()
    if proxy_direct:
        return proxy_direct

    proxy_env = PART_ROOT.parent / "docker-llm-proxy" / ".env"
    if not proxy_env.exists() or not proxy_env.is_file():
        return ""
    try:
        rows = proxy_env.read_text("utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    for row in rows:
        text = row.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("PROXY_API_KEY="):
            return text.split("=", 1)[1].strip().strip('"').strip("'")
        if text.startswith("OPENVINO_EMBED_API_KEY="):
            return text.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


class UnixSocketHTTPConnection(HTTPConnection):
    def __init__(self, socket_path: str, timeout: float = 10):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def docker_api_get(path: str) -> Any:
    if not Path(DOCKER_SOCKET).exists():
        return None
    conn = UnixSocketHTTPConnection(DOCKER_SOCKET)
    try:
        conn.request("GET", path)
        res = conn.getresponse()
        if res.status < 400:
            return json.loads(res.read().decode("utf-8"))
    except Exception:
        pass
    finally:
        conn.close()
    return None


def docker_api_post(path: str, body: Any = None) -> Any:
    if not Path(DOCKER_SOCKET).exists():
        return None
    conn = UnixSocketHTTPConnection(DOCKER_SOCKET)
    try:
        headers = {"Content-Type": "application/json"}
        conn.request(
            "POST", path, body=json.dumps(body) if body else None, headers=headers
        )
        res = conn.getresponse()
        return res.status < 400
    except Exception:
        pass
    finally:
        conn.close()
    return False


def docker_api_delete(path: str) -> bool:
    if not Path(DOCKER_SOCKET).exists():
        return False
    conn = UnixSocketHTTPConnection(DOCKER_SOCKET)
    try:
        conn.request("DELETE", path)
        res = conn.getresponse()
        return res.status < 400
    except Exception:
        pass
    finally:
        conn.close()
    return False


class SimulationManager:
    def __init__(self):
        self.presets = self._load_presets()

    def _load_presets(self) -> list[dict[str, Any]]:
        if not PRESETS_PATH.exists():
            return []
        try:
            with open(PRESETS_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("presets", [])
        except Exception:
            return []

    def get_presets(self) -> list[dict[str, Any]]:
        return self.presets

    def list_active_simulations(self) -> list[dict[str, Any]]:
        # Use Docker API via socket
        containers = docker_api_get("/containers/json")
        if not containers:
            return []

        results = []
        for c in containers:
            labels = c.get("Labels", {})
            if labels.get("io.fork_tales.simulation") != "true":
                continue

            # Extract port
            port = None
            for p in c.get("Ports", []):
                if p.get("PrivatePort") == 8787:
                    port = p.get("PublicPort")
                    break

            name = c.get("Names", [""])[0].lstrip("/")
            results.append(
                {
                    "id": c.get("Id")[:12],
                    "name": name,
                    "status": c.get("Status"),
                    "port": port,
                    "backend": labels.get(
                        "io.fork_tales.simulation.variant", "unknown"
                    ),
                    "preset": labels.get("io.fork_tales.simulation.preset", "none"),
                    "role": labels.get("io.fork_tales.simulation.role", "unknown"),
                }
            )
        return results

    def stop_simulation(self, container_id: str) -> bool:
        # Try to stop via API
        if docker_api_post(f"/containers/{container_id}/stop"):
            return docker_api_delete(f"/containers/{container_id}")
        return False

    def spawn_simulation(self, preset_id: str, port: int) -> dict[str, Any]:
        # Still use subprocess for spawning because docker run --label is easier than full API payload
        # but we need to find image name
        preset = next((p for p in self.presets if p["id"] == preset_id), None)
        if not preset:
            return {"ok": False, "error": f"Preset {preset_id} not found"}

        instance_name = f"sim-dynamic-{preset_id}-{int(time.time())}"
        image_name = "part64-eta-mu-local"  # Default fallback

        # We can't use 'docker run' if docker CLI is missing or old.
        # But wait, if I can't use 'docker run', I HAVE to use the API for spawning too.

        # Let's try API create + start.
        extra_env = [
            "CHROMA_HOST=chroma",
            "CHROMA_PORT=8000",
            "TEXT_GENERATION_BACKEND=vllm",
            "TEXT_GENERATION_BASE_URL=http://host.docker.internal:18000",
            "TEXT_GENERATION_MODEL=qwen3-vl:4b-instruct",
            "TEXT_GENERATION_DEVICE=GPU",
            "ETA_MU_IMAGE_VISION_ENABLED=1",
            "ETA_MU_IMAGE_VISION_BASE_URL=http://host.docker.internal:18000",
            "OPENVINO_EMBED_MODEL=nomic-embed-text",
            "OPENVINO_EMBED_DEVICE=NPU",
            "EMBEDDINGS_BACKEND=openvino",
            "CDB_EMBED_IN_C=1",
            "CDB_EMBED_REQUIRE_C=1",
            "CDB_EMBED_DEVICE=NPU",
            "CDB_EMBED_GPU_REQUIRE_CUDA=1",
            "CDB_EMBED_STRICT_DEVICE=1",
            "CDB_EMBED_PRELOAD_ORT_CORE=1",
        ]

        config = {
            "Image": image_name,
            "Labels": {
                "io.fork_tales.simulation": "true",
                "io.fork_tales.simulation.role": "experiment",
                "io.fork_tales.simulation.variant": preset.get("backend", "unknown"),
                "io.fork_tales.simulation.preset": preset_id,
                "io.fork_tales.simulation.instance": instance_name,
            },
            "HostConfig": {
                "PortBindings": {"8787/tcp": [{"HostPort": str(port)}]},
                "Binds": [f"{PART_ROOT}:/app", f"{PART_ROOT.parent}:/vault"],
                "NetworkMode": "eta-mu-sim-net",
                "ExtraHosts": ["host.docker.internal:host-gateway"],
            },
            "Env": [f"{k}={v}" for k, v in preset.get("env", {}).items()] + extra_env,
            "Cmd": [
                "pm2-runtime",
                "start",
                "ecosystem.bench.config.cjs",
                "--only",
                "eta-mu-world",
            ],
        }

        create_res = docker_api_post(
            f"/containers/create?name={instance_name}", body=config
        )
        if isinstance(create_res, dict) and "Id" in create_res:
            cid = create_res["Id"]
            if docker_api_post(f"/containers/{cid}/start"):
                return {
                    "ok": True,
                    "id": cid[:12],
                    "name": instance_name,
                    "port": port,
                    "preset": preset_id,
                }

        return {"ok": False, "error": "API spawn failed"}


def main():
    manager = SimulationManager()
    parser = argparse.ArgumentParser(description="Manage simulation instances")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list-presets")
    subparsers.add_parser("list-active")

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("id", help="Container ID or name")

    spawn_parser = subparsers.add_parser("spawn")
    spawn_parser.add_argument("--preset", required=True)
    spawn_parser.add_argument("--port", type=int, required=True)

    args = parser.parse_args()

    if args.command == "list-presets":
        print(json.dumps(manager.get_presets(), indent=2))
    elif args.command == "list-active":
        print(json.dumps(manager.list_active_simulations(), indent=2))
    elif args.command == "stop":
        if manager.stop_simulation(args.id):
            print(json.dumps({"ok": True}))
        else:
            print(json.dumps({"ok": False}), file=sys.stderr)
            sys.exit(1)
    elif args.command == "spawn":
        res = manager.spawn_simulation(args.preset, args.port)
        print(json.dumps(res, indent=2))
        if not res.get("ok"):
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
