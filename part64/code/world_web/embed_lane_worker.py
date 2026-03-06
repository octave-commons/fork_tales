#!/usr/bin/env python3
"""JSONL sidecar worker for embedding and cosine matrix requests."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def _bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _ld_library_path_preview(limit: int = 5) -> str:
    entries = [
        entry for entry in str(os.getenv("LD_LIBRARY_PATH", "")).split(":") if entry
    ]
    if not entries:
        return ""
    preview = entries[: max(1, int(limit))]
    return ":".join(preview)


def _diag_base(source: str) -> dict[str, Any]:
    entries = [
        entry for entry in str(os.getenv("LD_LIBRARY_PATH", "")).split(":") if entry
    ]
    return {
        "source": str(source or ""),
        "cdb_ort_gpu_capi_dir": str(os.getenv("CDB_ORT_GPU_CAPI_DIR", "") or ""),
        "cdb_ort_gpu_include_dir": str(os.getenv("CDB_ORT_GPU_INCLUDE_DIR", "") or ""),
        "cdb_ort_capi_dir": str(os.getenv("CDB_ORT_CAPI_DIR", "") or ""),
        "cdb_ort_include_dir": str(os.getenv("CDB_ORT_INCLUDE_DIR", "") or ""),
        "nvidia_visible_devices": str(os.getenv("NVIDIA_VISIBLE_DEVICES", "") or ""),
        "cuda_visible_devices": str(os.getenv("CUDA_VISIBLE_DEVICES", "") or ""),
        "cdb_embed_device": str(os.getenv("CDB_EMBED_DEVICE", "") or ""),
        "ld_library_path_entries": int(len(entries)),
        "ld_library_path_preview": _ld_library_path_preview(),
        "sidecar_stderr_path": str(
            os.getenv("CDB_EMBED_SIDECAR_STDERR_PATH", "") or ""
        ),
    }


def _sanitize_rows(payload: Any, *, expected_dim: int) -> list[list[float]]:
    out: list[list[float]] = []
    if not isinstance(payload, list):
        return out
    for row in payload:
        if not isinstance(row, list) or len(row) != expected_dim:
            return []
        try:
            out.append([float(item) for item in row])
        except Exception:
            return []
    return out


def _ort_gpu_python_roots() -> list[Path]:
    roots: list[Path] = []
    explicit = str(os.getenv("CDB_ORT_GPU_PYTHON_DIR", "") or "").strip()
    if explicit:
        roots.append(Path(explicit).expanduser())

    capi = str(os.getenv("CDB_ORT_GPU_CAPI_DIR", "") or "").strip()
    if capi:
        capi_path = Path(capi).expanduser()
        roots.append(capi_path.parent.parent)
        roots.append(capi_path.parent)

    roots.append(Path("/opt/ort-gpu"))

    ordered: list[Path] = []
    seen: set[str] = set()
    for candidate in roots:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        marker = str(resolved)
        if marker in seen:
            continue
        seen.add(marker)
        if (resolved / "onnxruntime" / "__init__.py").exists():
            ordered.append(resolved)
    return ordered


def _import_onnxruntime_for_device(device: str) -> Any:
    normalized = str(device or "").strip().upper()
    if normalized != "GPU":
        return importlib.import_module("onnxruntime")

    if "onnxruntime" in sys.modules:
        module = sys.modules.get("onnxruntime")
        if module is not None:
            return module

    for root in _ort_gpu_python_roots():
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            ort = importlib.import_module("onnxruntime")
        except Exception:
            sys.modules.pop("onnxruntime", None)
            continue
        return ort

    return importlib.import_module("onnxruntime")


class _CosineSession:
    def __init__(self, *, device: str, expected_dim: int) -> None:
        self._device = str(device or "CPU").strip().upper()
        self._expected_dim = int(expected_dim)
        self._error = ""
        self._provider = ""
        self._ort_module = ""
        self._model_path = ""
        self._enabled = _bool_env("CDB_COSINE_GPU_MATRIX_ENABLED", True)
        self._session: Any = None
        self._input_left = "left"
        self._input_right = "right"
        self._output = "scores"

    def _resolve_model_path(self) -> Path:
        explicit = str(os.getenv("CDB_COSINE_MATRIX_MODEL_PATH", "") or "").strip()
        if explicit:
            return Path(explicit).expanduser().resolve()
        return Path(__file__).resolve().parent / "native" / "cosine_matrix_dynamic.onnx"

    def _ensure(self) -> bool:
        if self._session is not None:
            return True
        if self._device == "GPU" and not self._enabled:
            self._error = "cosine_gpu_disabled"
            return False
        try:
            import numpy as np  # type: ignore

            ort = _import_onnxruntime_for_device(self._device)
        except Exception as exc:
            self._error = f"cosine_import_failed:{exc}"
            return False

        model_path = self._resolve_model_path()
        self._model_path = str(model_path)
        if not model_path.exists() or not model_path.is_file():
            self._error = f"cosine_model_missing:{model_path}"
            return False

        providers: list[str]
        provider_options: list[dict[str, str]] | None = None
        if self._device == "GPU":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self._device == "NPU":
            providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{"device_type": "NPU"}, {}]
        else:
            providers = ["CPUExecutionProvider"]

        try:
            if provider_options is not None:
                session = ort.InferenceSession(
                    str(model_path),
                    providers=providers,
                    provider_options=provider_options,
                )
            else:
                session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as exc:
            self._error = f"cosine_session_create_failed:{exc}"
            return False

        active = [str(item) for item in session.get_providers()]
        if self._device == "GPU":
            if not active or active[0] != "CUDAExecutionProvider":
                self._error = "cosine_cuda_provider_unavailable:" + ",".join(active)
                return False
        if self._device == "NPU":
            if not active or active[0] != "OpenVINOExecutionProvider":
                self._error = "cosine_openvino_npu_unavailable:" + ",".join(active)
                return False

        self._provider = active[0] if active else "unknown"
        self._ort_module = str(getattr(ort, "__file__", "") or "")
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) >= 2:
            self._input_left = str(inputs[0].name)
            self._input_right = str(inputs[1].name)
        if outputs:
            self._output = str(outputs[0].name)
        self._session = session
        self._np = np
        return True

    def diag(self) -> dict[str, Any]:
        return {
            "cosine_source": "onnxruntime",
            "cosine_provider": str(self._provider or ""),
            "cosine_model_path": str(self._model_path or ""),
            "cosine_ort_module": str(self._ort_module or ""),
            "cosine_gpu_enabled": bool(self._enabled),
            "cosine_device": str(self._device or ""),
        }

    def error(self) -> str:
        return str(self._error or "")

    def run(
        self, left: list[list[float]], right: list[list[float]]
    ) -> tuple[list[float], int, int] | None:
        if not self._ensure():
            return None

        rows = len(left)
        cols = len(right)
        if rows <= 0 or cols <= 0:
            self._error = "cosine_empty"
            return None
        if len(left[0]) != self._expected_dim or len(right[0]) != self._expected_dim:
            self._error = "cosine_invalid_dim"
            return None

        left_np = self._np.asarray(left, dtype=self._np.float32)
        right_np = self._np.asarray(right, dtype=self._np.float32)
        try:
            out = self._session.run(
                [self._output],
                {
                    self._input_left: left_np,
                    self._input_right: right_np,
                },
            )[0]
        except Exception as exc:
            self._error = f"cosine_run_failed:{exc}"
            return None
        try:
            flat = [float(item) for item in out.reshape(-1).tolist()]
        except Exception as exc:
            self._error = f"cosine_output_invalid:{exc}"
            return None
        return flat, rows, cols


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedding sidecar worker")
    parser.add_argument("--device", default="GPU")
    args = parser.parse_args()

    device = str(args.device or "GPU").strip().upper()
    os.environ["CDB_EMBED_LANE_WORKER"] = "1"
    os.environ["CDB_EMBED_GPU_SIDECAR_ENABLED"] = "0"
    os.environ["CDB_EMBED_GPU_USE_SIDECAR_FOR_EXPLICIT"] = "0"
    os.environ["CDB_EMBED_DEVICE"] = device

    from . import c_double_buffer_backend as backend

    if device == "GPU":
        try:
            backend._prepare_gpu_cuda_env()
        except Exception:
            pass
    elif device == "NPU":
        try:
            backend._prepare_npu_level_zero_env()
        except Exception:
            pass

    expected_dim = int(getattr(backend, "_CDB_EMBED_DIM", 24))
    cosine = _CosineSession(device=device, expected_dim=expected_dim)

    for raw_line in sys.stdin:
        line = str(raw_line or "").strip()
        if not line:
            continue
        req: dict[str, Any]
        req_id = ""
        cmd = ""
        try:
            req = json.loads(line)
            req_id = str(req.get("id", "") or "")
            cmd = str(req.get("cmd", "") or "").strip().lower()
        except Exception as exc:
            payload = {
                "id": req_id,
                "error": f"invalid_json:{exc}",
                "diag": _diag_base("worker"),
            }
            sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
            sys.stdout.flush()
            continue

        if cmd == "shutdown":
            payload = {"id": req_id, "ok": True}
            sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
            sys.stdout.flush()
            break

        if cmd == "embed":
            text = str(req.get("text", "") or "")
            vector = backend.embed_text_24_local(text, requested_device=device)
            if isinstance(vector, list) and len(vector) == expected_dim:
                payload = {"id": req_id, "vector": vector}
            else:
                snap = backend.embed_runtime_snapshot()
                diag = _diag_base("embed")
                if isinstance(snap, dict):
                    diag["selected_device"] = str(snap.get("selected_device", "") or "")
                    diag["cpu_fallback"] = bool(snap.get("cpu_fallback", False))
                    diag["cpu_fallback_detail"] = str(
                        snap.get("cpu_fallback_detail", "") or ""
                    )
                message = "embed_failed"
                if isinstance(snap, dict):
                    message = str(snap.get("error", "") or "").strip() or "embed_failed"
                payload = {"id": req_id, "error": message, "diag": diag}
            sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
            sys.stdout.flush()
            continue

        if cmd == "cosine_matrix":
            left = _sanitize_rows(req.get("left"), expected_dim=expected_dim)
            right = _sanitize_rows(req.get("right"), expected_dim=expected_dim)
            if not left or not right:
                payload = {
                    "id": req_id,
                    "error": "invalid_cosine_payload",
                    "diag": {**_diag_base("cosine"), **cosine.diag()},
                }
                sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
                sys.stdout.flush()
                continue

            result = cosine.run(left, right)
            if result is None:
                payload = {
                    "id": req_id,
                    "error": cosine.error() or "cosine_failed",
                    "diag": {**_diag_base("cosine"), **cosine.diag()},
                }
                sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
                sys.stdout.flush()
                continue

            matrix, rows, cols = result
            payload = {
                "id": req_id,
                "matrix": matrix,
                "rows": rows,
                "cols": cols,
            }
            sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
            sys.stdout.flush()
            continue

        payload = {
            "id": req_id,
            "error": f"unknown_command:{cmd}",
            "diag": _diag_base("worker"),
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
