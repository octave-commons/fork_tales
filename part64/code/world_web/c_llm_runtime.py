from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


_PART_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TRTLLM_ENV_WRAPPER = _PART_ROOT / "scripts" / "trtllm_env.sh"
_DEFAULT_TRTLLM_TEXT_SCRIPT = _PART_ROOT / "scripts" / "trtllm_text_generate.py"


def _coerce_timeout(timeout_s: float | None) -> float:
    try:
        value = float(timeout_s) if timeout_s is not None else 30.0
    except Exception:
        value = 30.0
    return max(0.2, min(180.0, value))


def _coerce_int_env(name: str, fallback: int, *, low: int, high: int) -> int:
    try:
        value = int(float(str(os.getenv(name, str(fallback)) or str(fallback)).strip()))
    except Exception:
        value = fallback
    return max(low, min(high, value))


def _coerce_float_env(name: str, fallback: float, *, low: float, high: float) -> float:
    try:
        value = float(str(os.getenv(name, str(fallback)) or str(fallback)).strip())
    except Exception:
        value = fallback
    return max(low, min(high, value))


def _resolve_candidate_paths(values: list[str]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for value in values:
        raw = str(value or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _resolve_engine_dir(model: str) -> tuple[Path | None, str]:
    model_candidate = ""
    trimmed_model = str(model or "").strip()
    if trimmed_model and ("/" in trimmed_model or trimmed_model.startswith(".")):
        model_candidate = trimmed_model

    candidates = _resolve_candidate_paths(
        [
            str(os.getenv("C_LLM_ENGINE_DIR", "") or ""),
            str(os.getenv("QWEN3VL_ENGINE_DIR", "") or ""),
            str(os.getenv("TRTLLM_ENGINE_DIR", "") or ""),
            model_candidate,
        ]
    )
    if not candidates:
        return None, "c_llm_engine_dir_missing"

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve(), ""
    return None, "c_llm_engine_dir_not_found"


def _resolve_tokenizer_dir(model: str) -> str:
    model_candidate = ""
    trimmed_model = str(model or "").strip()
    if trimmed_model and ("/" in trimmed_model or trimmed_model.startswith(".")):
        model_candidate = trimmed_model

    candidates = _resolve_candidate_paths(
        [
            str(os.getenv("C_LLM_TOKENIZER_DIR", "") or ""),
            str(os.getenv("QWEN3VL_CHECKPOINT_DIR", "") or ""),
            str(os.getenv("TRTLLM_TOKENIZER_DIR", "") or ""),
            model_candidate,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def _resolve_runner_paths() -> tuple[Path | None, Path | None, str]:
    wrapper = Path(
        str(
            os.getenv("C_LLM_TRTLLM_ENV_WRAPPER", str(_DEFAULT_TRTLLM_ENV_WRAPPER))
            or str(_DEFAULT_TRTLLM_ENV_WRAPPER)
        )
    ).expanduser()
    script = Path(
        str(os.getenv("C_LLM_TRTLLM_SCRIPT", str(_DEFAULT_TRTLLM_TEXT_SCRIPT)) or "")
    ).expanduser()

    if not wrapper.exists():
        return None, None, "c_llm_runtime_wrapper_missing"
    if not wrapper.is_file():
        return None, None, "c_llm_runtime_wrapper_invalid"
    if not os.access(wrapper, os.X_OK):
        return None, None, "c_llm_runtime_wrapper_not_executable"

    if not script.exists():
        return None, None, "c_llm_runtime_script_missing"
    if not script.is_file():
        return None, None, "c_llm_runtime_script_invalid"
    return wrapper.resolve(), script.resolve(), ""


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _runner_command(
    *,
    wrapper: Path,
    script: Path,
    engine_dir: Path,
    prompt: str,
    model: str,
    tokenizer_dir: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    command = [
        str(wrapper),
        sys.executable,
        str(script),
        "--engine-dir",
        str(engine_dir),
        "--prompt",
        prompt,
        "--model",
        str(model or "").strip(),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        f"{temperature:.6f}",
        "--top-p",
        f"{top_p:.6f}",
    ]
    if tokenizer_dir:
        command.extend(["--tokenizer", tokenizer_dir])
    return command


def _run_trtllm_subprocess(
    *,
    prompt: str,
    model: str,
    timeout_s: float | None,
) -> tuple[str | None, str]:
    engine_dir, engine_error = _resolve_engine_dir(model)
    if engine_dir is None:
        return None, engine_error

    wrapper, script, runner_error = _resolve_runner_paths()
    if wrapper is None or script is None:
        return None, runner_error

    tokenizer_dir = _resolve_tokenizer_dir(model)
    max_tokens = _coerce_int_env("C_LLM_MAX_TOKENS", 96, low=1, high=2048)
    temperature = _coerce_float_env("C_LLM_TEMPERATURE", 0.2, low=0.0, high=2.0)
    top_p = _coerce_float_env("C_LLM_TOP_P", 0.95, low=0.01, high=1.0)

    command = _runner_command(
        wrapper=wrapper,
        script=script,
        engine_dir=engine_dir,
        prompt=prompt,
        model=model,
        tokenizer_dir=tokenizer_dir,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    environment = os.environ.copy()
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    environment.setdefault("HF_DATASETS_OFFLINE", "1")

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_coerce_timeout(timeout_s),
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return None, "c_llm_runtime_timeout"
    except Exception:
        return None, "c_llm_runtime_exec_failed"

    payload = _parse_json_payload(completed.stdout)
    if isinstance(payload, dict):
        payload_error = str(payload.get("error", "") or "").strip()
        payload_text = str(payload.get("text", "") or "").strip()
        payload_ok = bool(payload.get("ok", False))
        if payload_ok and payload_text:
            return payload_text, ""
        if payload_error:
            return None, payload_error

    if completed.returncode != 0:
        return None, f"c_llm_runtime_exit_{completed.returncode}"
    return None, "c_llm_generation_failed"


def generate_text_local(
    prompt: str,
    *,
    model: str,
    timeout_s: float | None = None,
) -> tuple[str | None, str]:
    prompt_text = str(prompt or "").strip()
    if not prompt_text:
        return "", ""

    backend = (
        str(
            os.getenv("C_LLM_RUNTIME_BACKEND", "trtllm_subprocess")
            or "trtllm_subprocess"
        )
        .strip()
        .lower()
    )
    if backend in {"", "none", "off", "disabled"}:
        return None, "c_llm_runtime_disabled"

    if backend in {"trtllm", "trtllm_subprocess"}:
        return _run_trtllm_subprocess(
            prompt=prompt_text,
            model=str(model or "").strip(),
            timeout_s=timeout_s,
        )

    return None, "c_llm_runtime_backend_unsupported"
