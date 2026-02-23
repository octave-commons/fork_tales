#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _emit(payload: dict[str, Any], exit_code: int) -> int:
    print(json.dumps(payload, ensure_ascii=True), flush=True)
    return int(exit_code)


def _error_payload(*, error: str, model: str, detail: str = "") -> dict[str, Any]:
    return {
        "ok": False,
        "text": "",
        "error": str(error or "trtllm_generate_failed"),
        "model": str(model or "").strip(),
        "backend": "trtllm",
        "detail": str(detail or "")[:400],
    }


def _classify_error(exc: Exception) -> str:
    detail = f"{exc.__class__.__name__}:{exc}".lower()
    if "qwen3_vl" in detail or "qwen3vl" in detail:
        return "trtllm_qwen3_vl_unsupported"
    if "qwen3vlforconditionalgeneration" in detail:
        return "trtllm_qwen3_vl_unsupported"
    if "libmpi.so.40" in detail:
        return "trtllm_mpi_runtime_missing"
    if "tokenizer" in detail and "not found" in detail:
        return "trtllm_tokenizer_missing"
    if "engine" in detail and "not found" in detail:
        return "trtllm_engine_missing"
    return "trtllm_generate_failed"


def _extract_text(result: Any) -> str:
    item = result
    if isinstance(item, list):
        if not item:
            return ""
        item = item[0]
    outputs = getattr(item, "outputs", None)
    if isinstance(outputs, list):
        for candidate in outputs:
            text = str(getattr(candidate, "text", "") or "").strip()
            if text:
                return text
    text = str(getattr(item, "text", "") or "").strip()
    if text:
        return text
    return ""


def _engine_artifacts_present(engine_dir: Path) -> bool:
    try:
        for _ in engine_dir.glob("*.engine"):
            return True
        for _ in engine_dir.glob("**/*.engine"):
            return True
    except Exception:
        return False
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local TensorRT-LLM single prompt runner"
    )
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    engine_dir = Path(str(args.engine_dir or "")).expanduser()
    if not engine_dir.is_dir():
        return _emit(
            _error_payload(
                error="trtllm_engine_dir_not_found",
                model=args.model,
                detail=str(engine_dir),
            ),
            2,
        )

    if not _engine_artifacts_present(engine_dir):
        return _emit(
            _error_payload(
                error="trtllm_engine_not_built",
                model=args.model,
                detail=(
                    "missing *.engine artifact under --engine-dir; "
                    "build TensorRT-LLM engine first"
                ),
            ),
            2,
        )

    tokenizer = str(args.tokenizer or "").strip()
    if tokenizer:
        tokenizer_path = Path(tokenizer).expanduser()
        if not tokenizer_path.exists():
            return _emit(
                _error_payload(
                    error="trtllm_tokenizer_missing",
                    model=args.model,
                    detail=str(tokenizer_path),
                ),
                3,
            )
        tokenizer = str(tokenizer_path)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    try:
        from tensorrt_llm import LLM, SamplingParams
    except Exception as exc:
        return _emit(
            _error_payload(
                error=_classify_error(exc),
                model=args.model,
                detail=f"{exc.__class__.__name__}:{exc}",
            ),
            4,
        )

    llm = None
    try:
        llm_kwargs: dict[str, Any] = {"model": str(engine_dir)}
        if tokenizer:
            llm_kwargs["tokenizer"] = tokenizer
        llm = LLM(**llm_kwargs)
        sampling = SamplingParams(
            max_tokens=max(1, int(args.max_tokens)),
            temperature=max(0.0, float(args.temperature)),
            top_p=max(0.01, min(1.0, float(args.top_p))),
        )
        output = llm.generate(
            str(args.prompt or ""),
            sampling_params=sampling,
            use_tqdm=False,
        )
        text = _extract_text(output)
        if not text:
            return _emit(
                _error_payload(
                    error="trtllm_empty_output",
                    model=args.model,
                    detail="no generated text",
                ),
                5,
            )
        return _emit(
            {
                "ok": True,
                "text": text,
                "error": "",
                "model": str(args.model or "").strip(),
                "backend": "trtllm",
            },
            0,
        )
    except Exception as exc:
        return _emit(
            _error_payload(
                error=_classify_error(exc),
                model=args.model,
                detail=f"{exc.__class__.__name__}:{exc}",
            ),
            6,
        )
    finally:
        if llm is not None:
            try:
                llm.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
