from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


TRAINING_EXAMPLE_RECORD = "training.example.v1"
REQUIRED_SNAPSHOT_KEYS = (
    "L0.sim",
    "L1.health",
    "L2.gov",
    "L3.dialog",
    "L4.context",
    "L6.model",
)
ALLOWED_CONFIDENCE = {"soft", "strong", "hard"}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                count += 1
    return count


def describe_jsonl(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "records": count_jsonl_records(resolved),
    }


def load_manifest_for_dataset(train_jsonl_path: Path) -> Tuple[Path, Dict[str, Any]]:
    manifest_path = train_jsonl_path.resolve().with_name("manifest.json")
    if not manifest_path.exists() or not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest is not an object: {manifest_path}")
    return manifest_path, payload


def verify_dataset_checksums(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    checksums = manifest.get("dataset_checksums")
    if not isinstance(checksums, dict):
        raise ValueError("Manifest missing dataset_checksums")

    verified: Dict[str, Dict[str, Any]] = {}
    for key in ("train", "val", "training_examples"):
        entry = checksums.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"dataset_checksums.{key} missing or invalid")

        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"dataset_checksums.{key}.path missing")

        path = Path(path_value).resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"dataset_checksums.{key}.path not found: {path}")

        expected_sha = str(entry.get("sha256", "")).strip().lower()
        actual_sha = sha256_file(path)
        if expected_sha and expected_sha != actual_sha:
            raise ValueError(
                f"Checksum mismatch for {key}: expected={expected_sha} actual={actual_sha}"
            )

        verified[key] = {
            "path": str(path),
            "sha256": actual_sha,
            "records": count_jsonl_records(path),
            "bytes": path.stat().st_size,
        }

    return verified


def verify_training_examples(
    training_examples_path: Path,
    *,
    expected_taxonomy: Optional[str],
) -> Dict[str, Any]:
    resolved = training_examples_path.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"training_examples file not found: {resolved}")

    total = 0
    confidence_counts: Dict[str, int] = {key: 0 for key in sorted(ALLOWED_CONFIDENCE)}
    event_counts: Dict[str, int] = {}

    with resolved.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid training example JSON line {line_no}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(f"training example line {line_no} is not an object")

            record = str(row.get("record", "")).strip()
            if record != TRAINING_EXAMPLE_RECORD:
                raise ValueError(
                    f"training example line {line_no} has invalid record={record}"
                )

            snapshots = row.get("snapshots")
            if not isinstance(snapshots, dict):
                raise ValueError(f"training example line {line_no} missing snapshots")
            for key in REQUIRED_SNAPSHOT_KEYS:
                value = snapshots.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"training example line {line_no} missing snapshot key: {key}"
                    )

            label = row.get("label")
            if not isinstance(label, dict):
                raise ValueError(f"training example line {line_no} missing label")

            taxonomy = str(label.get("taxonomy", "")).strip()
            if not taxonomy:
                raise ValueError(
                    f"training example line {line_no} missing label.taxonomy"
                )
            if expected_taxonomy and taxonomy != expected_taxonomy:
                raise ValueError(
                    f"training example line {line_no} taxonomy mismatch: "
                    f"expected={expected_taxonomy} actual={taxonomy}"
                )

            for required in ("destination", "intent"):
                value = str(label.get(required, "")).strip()
                if not value:
                    raise ValueError(
                        f"training example line {line_no} missing label.{required}"
                    )

            confidence = str(label.get("confidence", "")).strip().lower()
            if confidence not in ALLOWED_CONFIDENCE:
                raise ValueError(
                    f"training example line {line_no} invalid confidence={confidence}"
                )

            event = str(row.get("event", "")).strip()
            if not event:
                raise ValueError(f"training example line {line_no} missing event")

            confidence_counts[confidence] += 1
            event_counts[event] = event_counts.get(event, 0) + 1
            total += 1

    if total <= 0:
        raise ValueError("training_examples file has zero records")

    return {
        "path": str(resolved),
        "total": total,
        "confidence_counts": confidence_counts,
        "event_counts": event_counts,
    }


def append_registry_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
