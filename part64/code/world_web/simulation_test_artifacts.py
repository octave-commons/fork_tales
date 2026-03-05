"""Helpers for test failure and coverage artifact ingestion.

Extracted from simulation.py to reduce file size and local complexity.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse


def coerce_test_failure_rows(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _add(item: Any) -> None:
        if isinstance(item, dict):
            rows.append(dict(item))
        elif str(item).strip():
            rows.append({"name": str(item).strip(), "status": "failed"})

    if isinstance(payload, list):
        for item in payload:
            _add(item)
    elif isinstance(payload, dict):
        for key in ("failures", "failed_tests", "failing_tests", "tests"):
            if isinstance(payload.get(key), list):
                for item in payload[key]:
                    if key == "tests" and isinstance(item, dict):
                        status = str(
                            item.get("status") or item.get("outcome") or ""
                        ).lower()
                        if status in {"failed", "error", "failing", "xfailed"}:
                            row = dict(item)
                            row.setdefault("status", status or "failed")
                            row.setdefault(
                                "name",
                                str(
                                    item.get("nodeid")
                                    or item.get("test")
                                    or item.get("id")
                                    or ""
                                ),
                            )
                            rows.append(row)
                    else:
                        _add(item)
                if rows:
                    return rows
        name = str(
            payload.get("name") or payload.get("test") or payload.get("nodeid") or ""
        ).strip()
        if name:
            row = dict(payload)
            row.setdefault("status", "failed")
            rows.append(row)
    return rows


def parse_test_failures_text(raw_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        name, sep, covered = line.partition("|")
        if name.strip():
            row: dict[str, Any] = {"name": name.strip(), "status": "failed"}
            if sep:
                row["covered_files"] = [
                    token.strip()
                    for token in re.split(r"[,\s]+", covered.strip())
                    if token.strip()
                ]
            rows.append(row)
    return rows


def load_test_failures_from_path(candidate: Path) -> list[dict[str, Any]]:
    try:
        text = candidate.read_text("utf-8")
    except Exception:
        return []
    suffix = candidate.suffix.lower()
    if suffix in {".json", ".jsonl", ".ndjson"}:
        if suffix == ".json":
            try:
                return coerce_test_failure_rows(json.loads(text))
            except Exception:
                return []
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            try:
                rows.extend(coerce_test_failure_rows(json.loads(line)))
            except Exception:
                continue
        return rows
    return parse_test_failures_text(text)


def line_hits_to_spans(hits: list[tuple[int, int]]) -> list[dict[str, Any]]:
    sorted_hits = sorted(
        [(line, count) for line, count in hits if count > 0], key=lambda row: row[0]
    )
    if not sorted_hits:
        return []
    spans: list[dict[str, Any]] = []
    start_line, prev_line, total_hits = (
        sorted_hits[0][0],
        sorted_hits[0][0],
        sorted_hits[0][1],
    )
    for line, count in sorted_hits[1:]:
        if line <= prev_line + 1:
            prev_line = line
            total_hits += count
        else:
            spans.append(
                {
                    "start_line": start_line,
                    "end_line": prev_line,
                    "hits": total_hits,
                }
            )
            start_line, prev_line, total_hits = line, line, count
    spans.append({"start_line": start_line, "end_line": prev_line, "hits": total_hits})
    return spans


def normalize_coverage_source_path(
    raw: str,
    part_root: Path,
    vault_root: Path,
    *,
    normalize_path_for_file_id: Callable[[str], str],
) -> str:
    source = str(raw or "").strip()
    if source.startswith("file://"):
        source = unquote(urlparse(source).path)
    path_obj = Path(source.strip())
    if path_obj.is_absolute():
        try:
            resolved = path_obj.resolve(strict=False)
        except Exception:
            resolved = path_obj
        for root in (part_root, vault_root):
            try:
                return normalize_path_for_file_id(
                    str(resolved.relative_to(root.resolve()))
                )
            except Exception:
                continue
    return normalize_path_for_file_id(source)


def parse_lcov_payload(
    text: str,
    part_root: Path,
    vault_root: Path,
    *,
    normalize_path_for_file_id: Callable[[str], str],
    file_id_for_path: Callable[[str], str],
    safe_float: Callable[[Any, float], float],
    clamp01: Callable[[float], float],
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    by_test_sets: dict[str, set[str]] = defaultdict(set)
    by_test_spans: dict[str, list[dict[str, Any]]] = defaultdict(list)
    current_test = ""
    current_source = ""
    da_found = da_hit = lines_found = lines_hit = 0
    current_hits: list[tuple[int, int]] = []

    def _flush() -> None:
        nonlocal current_source, da_found, da_hit, lines_found, lines_hit, current_hits
        if not current_source:
            return
        normalized = normalize_coverage_source_path(
            current_source,
            part_root,
            vault_root,
            normalize_path_for_file_id=normalize_path_for_file_id,
        )
        if normalized:
            entry = files.setdefault(
                normalized,
                {
                    "file_id": file_id_for_path(normalized),
                    "lines_found": 0,
                    "lines_hit": 0,
                    "tests": [],
                },
            )
            entry["lines_found"] += lines_found or da_found
            entry["lines_hit"] += lines_hit or da_hit
            if current_test.strip():
                by_test_sets[current_test.strip()].add(normalized)
                if current_test.strip() not in entry["tests"]:
                    entry["tests"].append(current_test.strip())
                for span in line_hits_to_spans(current_hits):
                    by_test_spans[current_test.strip()].append(
                        {
                            "file": normalized,
                            "start_line": span["start_line"],
                            "end_line": span["end_line"],
                            "hits": span["hits"],
                            "weight": 1.0,
                        }
                    )
        current_source = ""
        da_found = da_hit = lines_found = lines_hit = 0
        current_hits = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("TN:"):
            current_test = line[3:].strip()
        elif line.startswith("SF:"):
            _flush()
            current_source = line[3:].strip()
        elif line == "end_of_record":
            _flush()
        elif current_source:
            if line.startswith("DA:"):
                parts = line[3:].split(",")
                line_number = int(safe_float(parts[0], 0.0))
                hit_count = int(safe_float(parts[1], 0.0))
                current_hits.append((line_number, hit_count))
                da_found += 1
                da_hit += 1 if hit_count > 0 else 0
            elif line.startswith("LF:"):
                lines_found = int(safe_float(line[3:], 0.0))
            elif line.startswith("LH:"):
                lines_hit = int(safe_float(line[3:], 0.0))
    _flush()

    file_payload = {
        key: {
            **value,
            "line_rate": round(clamp01(value["lines_hit"] / value["lines_found"]), 6)
            if value["lines_found"] > 0
            else 0.0,
        }
        for key, value in files.items()
    }
    return {
        "record": "ημ.test-coverage.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "lcov",
        "files": file_payload,
        "by_test": {key: sorted(list(value)) for key, value in by_test_sets.items()},
        "by_test_spans": dict(by_test_spans),
        "hottest_files": sorted(
            file_payload.keys(),
            key=lambda key: (file_payload[key].get("line_rate", 1.0), key),
        ),
    }


def load_test_coverage_from_path(
    candidate: Path,
    part_root: Path,
    vault_root: Path,
    *,
    normalize_path_for_file_id: Callable[[str], str],
    file_id_for_path: Callable[[str], str],
    safe_float: Callable[[Any, float], float],
    clamp01: Callable[[float], float],
) -> dict[str, Any]:
    try:
        text = candidate.read_text("utf-8")
    except Exception:
        return {}
    if candidate.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    if candidate.name.lower().endswith(".info") and "lcov" in candidate.name.lower():
        return parse_lcov_payload(
            text,
            part_root,
            vault_root,
            normalize_path_for_file_id=normalize_path_for_file_id,
            file_id_for_path=file_id_for_path,
            safe_float=safe_float,
            clamp01=clamp01,
        )
    return {}


def extract_coverage_spans(
    raw: Any,
    *,
    normalize_path_for_file_id: Callable[[str], str],
    safe_float: Callable[[Any, float], float],
    safe_int: Callable[[Any, int], int],
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []

    def _walk(item: Any, fallback_path: str, fallback_weight: float) -> None:
        if isinstance(item, str):
            normalized = normalize_path_for_file_id(item)
            if normalized:
                spans.append(
                    {
                        "path": normalized,
                        "start_line": 1,
                        "end_line": 1,
                        "symbol": "",
                        "weight": fallback_weight,
                    }
                )
        elif isinstance(item, list):
            for sub_item in item:
                _walk(sub_item, fallback_path, fallback_weight)
        elif isinstance(item, dict):
            path_value = next(
                (
                    item.get(key)
                    for key in ("file", "path", "source")
                    if isinstance(item.get(key), str)
                ),
                fallback_path,
            )
            weight_value = safe_float(
                next(
                    (
                        item.get(key)
                        for key in ("w", "weight")
                        if item.get(key) is not None
                    ),
                    fallback_weight,
                ),
                fallback_weight,
            )
            for key in ("spans", "files", "coverage"):
                if item.get(key):
                    _walk(item[key], path_value, weight_value)
            if path_value and not any(
                item.get(key) for key in ("spans", "files", "coverage")
            ):
                spans.append(
                    {
                        "path": normalize_path_for_file_id(path_value),
                        "start_line": int(safe_int(item.get("start_line", 1), 1)),
                        "end_line": int(safe_int(item.get("end_line", 1), 1)),
                        "symbol": str(item.get("symbol", "")),
                        "weight": weight_value,
                    }
                )

    _walk(raw, "", 1.0)
    return spans


def load_test_signal_artifacts(
    part_root: Path,
    vault_root: Path,
    *,
    normalize_path_for_file_id: Callable[[str], str],
    file_id_for_path: Callable[[str], str],
    safe_float: Callable[[Any, float], float],
    safe_int: Callable[[Any, int], int],
    clamp01: Callable[[float], float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for candidate in [
        part_root / "world_state" / "failing_tests.txt",
        part_root / "world_state" / "failing_tests.json",
        part_root / ".opencode" / "runtime" / "failing_tests.json",
        vault_root / ".opencode" / "runtime" / "failing_tests.json",
    ]:
        if candidate.exists():
            rows = load_test_failures_from_path(candidate)
            if rows:
                failures = rows
                break

    coverage: dict[str, Any] = {}
    for candidate in [
        part_root / "coverage" / "lcov.info",
        part_root / "world_state" / "test_coverage.json",
    ]:
        if candidate.exists():
            payload = load_test_coverage_from_path(
                candidate,
                part_root,
                vault_root,
                normalize_path_for_file_id=normalize_path_for_file_id,
                file_id_for_path=file_id_for_path,
                safe_float=safe_float,
                clamp01=clamp01,
            )
            if payload:
                coverage = payload
                break

    return failures, coverage
