from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from code.world_web import c_llm_runtime


def _write_executable(path: Path) -> None:
    path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)


def test_generate_text_local_requires_engine_dir(monkeypatch: Any) -> None:
    monkeypatch.delenv("C_LLM_ENGINE_DIR", raising=False)
    monkeypatch.delenv("QWEN3VL_ENGINE_DIR", raising=False)
    monkeypatch.delenv("TRTLLM_ENGINE_DIR", raising=False)

    text, error = c_llm_runtime.generate_text_local(
        "hello",
        model="qwen3-vl:2b-instruct",
    )

    assert text is None
    assert error == "c_llm_engine_dir_missing"


def test_generate_text_local_invokes_trtllm_runner(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir(parents=True)
    wrapper = tmp_path / "trtllm_env.sh"
    _write_executable(wrapper)
    script = tmp_path / "trtllm_text_generate.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setenv("C_LLM_ENGINE_DIR", str(engine_dir))
    monkeypatch.setenv("C_LLM_TRTLLM_ENV_WRAPPER", str(wrapper))
    monkeypatch.setenv("C_LLM_TRTLLM_SCRIPT", str(script))
    monkeypatch.setenv("C_LLM_TOKENIZER_DIR", str(tmp_path))
    monkeypatch.setenv("C_LLM_MAX_TOKENS", "77")
    monkeypatch.setenv("C_LLM_TEMPERATURE", "0.3")
    monkeypatch.setenv("C_LLM_TOP_P", "0.7")

    observed: dict[str, Any] = {}

    def _fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        observed["cmd"] = cmd
        observed["timeout"] = timeout
        observed["hf_offline"] = env.get("HF_HUB_OFFLINE")
        observed["transformers_offline"] = env.get("TRANSFORMERS_OFFLINE")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='{"ok": true, "text": "runner-output", "error": ""}\n',
            stderr="",
        )

    monkeypatch.setattr(c_llm_runtime.subprocess, "run", _fake_run)

    text, error = c_llm_runtime.generate_text_local(
        "hello runner",
        model="qwen3-vl:2b-instruct",
        timeout_s=7.5,
    )

    assert text == "runner-output"
    assert error == ""
    assert observed.get("timeout") == 7.5
    command = observed.get("cmd")
    assert isinstance(command, list)
    assert command[0] == str(wrapper)
    assert command[1] == sys.executable
    assert command[2] == str(script)
    assert observed.get("hf_offline") == "1"
    assert observed.get("transformers_offline") == "1"


def test_generate_text_local_propagates_runner_error(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir(parents=True)
    wrapper = tmp_path / "trtllm_env.sh"
    _write_executable(wrapper)
    script = tmp_path / "trtllm_text_generate.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    monkeypatch.setenv("C_LLM_ENGINE_DIR", str(engine_dir))
    monkeypatch.setenv("C_LLM_TRTLLM_ENV_WRAPPER", str(wrapper))
    monkeypatch.setenv("C_LLM_TRTLLM_SCRIPT", str(script))

    def _fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, timeout, env
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=6,
            stdout=(
                '{"ok": false, "text": "", "error": "trtllm_qwen3_vl_unsupported"}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(c_llm_runtime.subprocess, "run", _fake_run)

    text, error = c_llm_runtime.generate_text_local(
        "hello runner",
        model="qwen3-vl:2b-instruct",
    )

    assert text is None
    assert error == "trtllm_qwen3_vl_unsupported"
