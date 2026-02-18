from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from constitution_runtime import describe_jsonl


FIELD_TO_PRESENCE: Dict[str, str] = {
    "f1": "receipt_river",
    "f2": "witness_thread",
    "f3": "anchor_registry",
    "f4": "keeper_of_receipts",
    "f5": "fork_tax_canticle",
    "f6": "mage_of_receipts",
    "f7": "gates_of_truth",
    "f8": "anchor_registry",
}

FIELD_NAMES: Dict[str, str] = {
    "f1": "artifact_flux",
    "f2": "witness_tension",
    "f3": "coherence_focus",
    "f4": "drift_pressure",
    "f5": "fork_tax_balance",
    "f6": "curiosity_drive",
    "f7": "gate_pressure",
    "f8": "council_heat",
}

TRAINING_TAXONOMY_VERSION = "ημ.ui.training.constitution.v1"
TRAINING_EXAMPLE_RECORD = "training.example.v1"

ETA_MU_FIELD_KEYWORDS: Dict[str, set[str]] = {
    "f1": {
        "audio",
        "wav",
        "mp3",
        "m4a",
        "ogg",
        "flac",
        "image",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "cover",
        "art",
        "video",
        "render",
        "mix",
        "stream",
    },
    "f2": {
        "witness",
        "thread",
        "touch",
        "lineage",
        "trace",
        "collapse",
        "observer",
        "entangle",
    },
    "f3": {
        "coherence",
        "focus",
        "atlas",
        "center",
        "balance",
        "catalog",
        "index",
        "reference",
    },
    "f4": {
        "drift",
        "delta",
        "change",
        "patch",
        "churn",
        "error",
        "fail",
        "stale",
        "todo",
        "tmp",
    },
    "f5": {
        "fork",
        "tax",
        "debt",
        "canticle",
        "payment",
        "paid",
        "balance",
        "settle",
    },
    "f6": {
        "prompt",
        "intent",
        "packet",
        "lisp",
        "story",
        "lyrics",
        "song",
        "note",
        "readme",
        "text",
        "memory",
        "research",
    },
    "f7": {
        "gate",
        "truth",
        "receipt",
        "proof",
        "contract",
        "policy",
        "validate",
        "manifest",
        "ledger",
    },
    "f8": {
        "queue",
        "decision",
        "council",
        "runtime",
        "task",
        "ops",
        "status",
        "process",
        "pm2",
    },
}

TEXT_SUFFIXES = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".env",
    ".sh",
    ".bash",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".lisp",
    ".clj",
    ".cljs",
    ".css",
    ".html",
    ".xml",
    ".sql",
    ".proto",
    ".mjs",
    ".cjs",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    ".clj-kondo",
    "dist",
    "build",
    "tts_cache",
    "training-output",
    "training-data",
}

SKIP_PATH_SUBSTRINGS = {
    "/.git/",
    "/node_modules/",
    "/__pycache__/",
    "/part64/world_state/tts_cache/",
    "/.μη_ports/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training JSONL from system file corpus + simulation labels"
    )
    parser.add_argument("--vault-root", type=str, default="/vault")
    parser.add_argument("--out-dir", type=str, default="/data/qwen3_vl")
    parser.add_argument(
        "--knowledge-index",
        type=str,
        default=".opencode/runtime/eta_mu_knowledge.v1.jsonl",
    )
    parser.add_argument(
        "--decision-ledger",
        type=str,
        default="part64/world_state/decision_ledger.jsonl",
    )
    parser.add_argument("--max-file-samples", type=int, default=1400)
    parser.add_argument("--max-knowledge-samples", type=int, default=1400)
    parser.add_argument("--max-bytes-per-file", type=int, default=300_000)
    parser.add_argument("--max-chars", type=int, default=2200)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _clean_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_:-]+", text.lower())


def _normalize_scores(
    scores: Dict[str, float], fallback: str = "f6"
) -> Dict[str, float]:
    sanitized = {k: max(0.0, float(scores.get(k, 0.0))) for k in FIELD_TO_PRESENCE}
    total = sum(sanitized.values())
    if total <= 0.0:
        sanitized = {k: 0.0 for k in FIELD_TO_PRESENCE}
        sanitized[fallback] = 1.0
        return sanitized
    return {k: round(v / total, 4) for k, v in sanitized.items()}


def _dominant_field(scores: Dict[str, float]) -> Tuple[str, float]:
    if not scores:
        return "f6", 1.0
    field_id = max(scores.keys(), key=lambda key: float(scores.get(key, 0.0)))
    return field_id, float(scores.get(field_id, 0.0))


def _inverse_presence_map() -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = defaultdict(list)
    for field_id, presence_id in FIELD_TO_PRESENCE.items():
        mapping[presence_id].append(field_id)
    return dict(mapping)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_touch_prior(ledger_path: Path) -> Dict[str, float]:
    inverse = _inverse_presence_map()
    counts = {field_id: 1.0 for field_id in FIELD_TO_PRESENCE}

    for row in _load_jsonl(ledger_path):
        event = str(row.get("event", "")).strip().lower()
        if event not in {"touch", "fork_tax_payment"}:
            continue
        target = str(row.get("target", "")).strip()
        if not target:
            continue
        fields = inverse.get(target)
        if not fields:
            continue

        share = 1.0 / float(len(fields))
        for field_id in fields:
            counts[field_id] += share

    return _normalize_scores(counts, fallback="f3")


def _looks_textual(sample: bytes) -> bool:
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    control_count = sum(1 for byte in sample if byte < 9 or (13 < byte < 32))
    ratio = control_count / max(1, len(sample))
    return ratio <= 0.03


def _read_excerpt(path: Path, max_chars: int) -> str:
    try:
        with path.open("rb") as handle:
            payload = handle.read(max_chars * 4)
    except OSError:
        return ""

    if not _looks_textual(payload):
        return ""

    text = payload.decode("utf-8", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:max_chars].strip()


def _infer_language(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".json": "json",
        ".jsonl": "json",
        ".md": "markdown",
        ".txt": "text",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".lisp": "lisp",
        ".clj": "clojure",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
    }
    return mapping.get(suffix, suffix.replace(".", "") or "text")


def _infer_role(rel_path: str) -> str:
    lower = rel_path.lower()
    if "/tests/" in lower or lower.endswith("_test.py") or lower.endswith(".spec.ts"):
        return "test"
    if lower.endswith("readme.md") or "/docs/" in lower or "/specs/" in lower:
        return "docs"
    if "docker" in lower or "compose" in lower or lower.endswith(".env"):
        return "ops"
    if "/frontend/" in lower or lower.endswith(".tsx") or lower.endswith(".css"):
        return "frontend"
    if "/code/" in lower or lower.endswith(".py") or lower.endswith(".ts"):
        return "runtime"
    return "knowledge"


def _infer_scores(
    *,
    rel_path: str,
    kind: str,
    text_excerpt: str,
    touch_prior: Dict[str, float],
) -> Dict[str, float]:
    scores = {field_id: 0.0 for field_id in FIELD_TO_PRESENCE}
    kind_key = kind.strip().lower()

    if kind_key in {"audio", "image", "video"}:
        scores["f1"] += 0.42
        scores["f6"] += 0.08
    elif kind_key == "text":
        scores["f6"] += 0.46
        scores["f3"] += 0.2
        scores["f7"] += 0.12
    else:
        scores["f4"] += 0.24
        scores["f8"] += 0.18

    rel_lower = rel_path.lower()
    if rel_lower.endswith(".zip"):
        scores["f5"] += 0.24
        scores["f4"] += 0.12
    if rel_lower.endswith(".lisp"):
        scores["f6"] += 0.24
        scores["f7"] += 0.1
    if rel_lower.endswith(".md") or rel_lower.endswith(".txt"):
        scores["f6"] += 0.18
        scores["f3"] += 0.1
    if "/tests/" in rel_lower:
        scores["f7"] += 0.2
        scores["f8"] += 0.08
    if "world_state" in rel_lower:
        scores["f8"] += 0.22
        scores["f4"] += 0.08

    tokens = _clean_tokens(f"{rel_path} {text_excerpt}")
    for token in tokens:
        for field_id, keywords in ETA_MU_FIELD_KEYWORDS.items():
            if token in keywords:
                scores[field_id] += 0.07

    for field_id, keywords in ETA_MU_FIELD_KEYWORDS.items():
        if any(keyword in rel_lower for keyword in keywords):
            scores[field_id] += 0.12

    # Blend in simulation touch priors so the live simulation influences labels.
    for field_id in scores:
        scores[field_id] += float(touch_prior.get(field_id, 0.0)) * 0.35

    return _normalize_scores(scores, fallback="f6" if kind_key == "text" else "f1")


def _iter_text_corpus(
    vault_root: Path,
    *,
    max_bytes: int,
) -> Iterable[Path]:
    for root, dirnames, filenames in os.walk(vault_root):
        rel_dir = Path(root).relative_to(vault_root).as_posix()

        pruned: List[str] = []
        for dirname in dirnames:
            if dirname in SKIP_DIR_NAMES:
                continue
            rel = f"/{(Path(rel_dir) / dirname).as_posix()}/"
            if any(token in rel for token in SKIP_PATH_SUBSTRINGS):
                continue
            pruned.append(dirname)
        dirnames[:] = pruned

        for filename in filenames:
            path = Path(root) / filename
            rel = f"/{path.relative_to(vault_root).as_posix()}"
            if any(token in rel for token in SKIP_PATH_SUBSTRINGS):
                continue
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size <= 0 or size > max_bytes:
                continue
            yield path


def _build_file_corpus_rows(
    *,
    vault_root: Path,
    touch_prior: Dict[str, float],
    max_samples: int,
    max_bytes: int,
    max_chars: int,
    seed: int,
) -> List[Dict[str, Any]]:
    all_paths = list(_iter_text_corpus(vault_root, max_bytes=max_bytes))
    rng = random.Random(seed)
    rng.shuffle(all_paths)

    rows: List[Dict[str, Any]] = []
    for path in all_paths:
        rel_path = path.relative_to(vault_root).as_posix()
        excerpt = _read_excerpt(path, max_chars=max_chars)
        if not excerpt:
            continue

        scores = _infer_scores(
            rel_path=rel_path,
            kind="text",
            text_excerpt=excerpt,
            touch_prior=touch_prior,
        )
        dominant_field, dominant_weight = _dominant_field(scores)
        presence_id = FIELD_TO_PRESENCE.get(dominant_field, "anchor_registry")
        language = _infer_language(path)
        role = _infer_role(rel_path)

        response_obj = {
            "task": "field_structure_label",
            "source": "file_corpus",
            "path": rel_path,
            "kind": "text",
            "language": language,
            "role": role,
            "dominant_field": dominant_field,
            "dominant_field_name": FIELD_NAMES.get(dominant_field, dominant_field),
            "dominant_presence": presence_id,
            "dominant_weight": round(dominant_weight, 4),
            "field_scores": scores,
        }

        prompt = (
            "You are labeling Promethean system corpus artifacts.\n"
            "Infer structural training labels for the file snippet below.\n"
            f"Path: {rel_path}\n"
            f"Language: {language}\n"
            f"Role: {role}\n"
            "Snippet:\n"
            f"{excerpt}\n\n"
            "Return a JSON object with dominant_field, dominant_presence, field_scores, language, and role."
        )

        rows.append(
            {
                "prompt": prompt,
                "response": json.dumps(
                    response_obj, ensure_ascii=False, sort_keys=True
                ),
                "metadata": {
                    "source": "file_corpus",
                    "path": rel_path,
                    "dominant_field": dominant_field,
                },
            }
        )

        if len(rows) >= max_samples:
            break

    return rows


def _resolve_knowledge_path(vault_root: Path, row: Dict[str, Any]) -> Optional[Path]:
    for key in ("source_rel_path", "archived_rel_path"):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = (vault_root / value).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _knowledge_scores(
    row: Dict[str, Any], touch_prior: Dict[str, float]
) -> Dict[str, float]:
    payload = row.get("field_scores")
    if isinstance(payload, dict):
        try:
            parsed = {k: float(payload.get(k, 0.0)) for k in FIELD_TO_PRESENCE}
            return _normalize_scores(parsed, fallback="f6")
        except (TypeError, ValueError):
            pass

    rel_path = str(row.get("source_rel_path") or row.get("name") or "")
    text_excerpt = str(row.get("text_excerpt") or "")
    kind = str(row.get("kind") or "text")
    return _infer_scores(
        rel_path=rel_path,
        kind=kind,
        text_excerpt=text_excerpt,
        touch_prior=touch_prior,
    )


def _build_knowledge_rows(
    *,
    vault_root: Path,
    knowledge_index: Path,
    touch_prior: Dict[str, float],
    max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    knowledge_rows = _load_jsonl(knowledge_index)
    rng = random.Random(seed + 17)
    rng.shuffle(knowledge_rows)

    # Prefer image artifacts first so vision loops always get usable supervision.
    prioritized_rows = sorted(
        knowledge_rows,
        key=lambda row: 0 if str(row.get("kind", "")).strip().lower() == "image" else 1,
    )

    rows: List[Dict[str, Any]] = []
    for row in prioritized_rows:
        kind = str(row.get("kind") or "text")
        rel_path = str(row.get("source_rel_path") or row.get("name") or "")
        excerpt = str(row.get("text_excerpt") or "")

        scores = _knowledge_scores(row, touch_prior)
        dominant_field, dominant_weight = _dominant_field(scores)
        presence_id = str(
            row.get("dominant_presence")
            or FIELD_TO_PRESENCE.get(dominant_field, "anchor_registry")
        )

        response_obj = {
            "task": "field_structure_label",
            "source": "simulation_knowledge",
            "path": rel_path,
            "kind": kind,
            "dominant_field": dominant_field,
            "dominant_field_name": FIELD_NAMES.get(dominant_field, dominant_field),
            "dominant_presence": presence_id,
            "dominant_weight": round(dominant_weight, 4),
            "field_scores": scores,
        }

        top_fields = row.get("top_fields")
        if isinstance(top_fields, list):
            response_obj["top_fields"] = top_fields

        prompt = (
            "You are labeling simulation-derived knowledge artifacts for Promethean training.\n"
            f"Artifact path: {rel_path or '[unknown]'}\n"
            f"Artifact kind: {kind}\n"
            f"Excerpt: {excerpt if excerpt else '[none]'}\n"
            "Infer structured field labels and return JSON with dominant_field, dominant_presence, and field_scores."
        )

        sample: Dict[str, Any] = {
            "prompt": prompt,
            "response": json.dumps(response_obj, ensure_ascii=False, sort_keys=True),
            "metadata": {
                "source": "simulation_knowledge",
                "path": rel_path,
                "dominant_field": dominant_field,
            },
        }

        resolved_path = _resolve_knowledge_path(vault_root, row)
        if resolved_path and resolved_path.suffix.lower() in IMAGE_SUFFIXES:
            sample["image"] = str(resolved_path)
            if "<image>" not in sample["prompt"]:
                sample["prompt"] = "<image>\n" + sample["prompt"]

        rows.append(sample)
        if len(rows) >= max_samples:
            break

    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_rows(
    rows: List[Dict[str, Any]], val_ratio: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    if not shuffled:
        return [], []

    val_count = int(round(len(shuffled) * max(0.0, min(0.4, val_ratio))))
    val_count = max(1, val_count)
    val_count = min(val_count, max(1, len(shuffled) // 4))

    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]
    if not train_rows:
        train_rows = val_rows[:]
    return train_rows, val_rows


def _manifest(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_source = Counter()
    by_field = Counter()
    image_count = 0

    for row in rows:
        meta = row.get("metadata")
        if isinstance(meta, dict):
            by_source[str(meta.get("source", "unknown"))] += 1
            by_field[str(meta.get("dominant_field", "unknown"))] += 1
        if isinstance(row.get("image"), str):
            image_count += 1

    return {
        "total_samples": len(rows),
        "image_samples": image_count,
        "source_counts": dict(by_source),
        "field_counts": dict(by_field),
    }


def _confidence_from_weight(weight: float) -> str:
    value = float(weight)
    if value >= 0.8:
        return "hard"
    if value >= 0.55:
        return "strong"
    return "soft"


def _event_for_sample(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    dominant_field = str(metadata.get("dominant_field", "f6"))
    if isinstance(row.get("image"), str):
        return "vault.move"
    if dominant_field in {"f7", "f8"}:
        return "ui.reject_proposal"
    if dominant_field in {"f4", "f5"}:
        return "ui.lock"
    return "ui.accept_proposal"


def _parse_response_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row.get("response")
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _snapshot_pointers(
    vault_root: Path, knowledge_index: Path, decision_ledger: Path
) -> Dict[str, str]:
    return {
        "L0.sim": str(decision_ledger),
        "L1.health": str(
            (vault_root / "part64/world_state/pm2-world-out.log").resolve()
        ),
        "L2.gov": str(
            (vault_root / "contracts/ημ_ui_training_constitution.v1.lith").resolve()
        ),
        "L3.dialog": str((vault_root / "receipts.log").resolve()),
        "L4.context": str(knowledge_index),
        "L6.model": str(
            (vault_root / "docker-llm-proxy/training/tf_qwen3_vl/config.yaml").resolve()
        ),
    }


def _build_training_example_rows(
    *,
    rows: List[Dict[str, Any]],
    split: str,
    snapshots: Dict[str, str],
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        payload = _parse_response_payload(row)

        dominant_field = str(
            payload.get("dominant_field") or metadata.get("dominant_field") or "f6"
        )
        if dominant_field not in FIELD_NAMES:
            dominant_field = "f6"

        dominant_weight = float(payload.get("dominant_weight", 0.5))
        field_name = FIELD_NAMES.get(dominant_field, dominant_field)

        sample_path = str(metadata.get("path") or payload.get("path") or "")
        sample_source = str(
            metadata.get("source") or payload.get("source") or "unknown"
        )

        examples.append(
            {
                "record": TRAINING_EXAMPLE_RECORD,
                "event": _event_for_sample(row),
                "split": split,
                "snapshots": dict(snapshots),
                "label": {
                    "taxonomy": TRAINING_TAXONOMY_VERSION,
                    "destination": f"zone/{field_name}",
                    "intent": f"field/{field_name}",
                    "confidence": _confidence_from_weight(dominant_weight),
                },
                "source": {
                    "path": sample_path,
                    "source": sample_source,
                    "dominant_field": dominant_field,
                },
            }
        )

    return examples


def main() -> None:
    args = parse_args()

    vault_root = Path(args.vault_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    knowledge_index = (vault_root / args.knowledge_index).resolve()
    decision_ledger = (vault_root / args.decision_ledger).resolve()

    touch_prior = _load_touch_prior(decision_ledger)

    knowledge_rows = _build_knowledge_rows(
        vault_root=vault_root,
        knowledge_index=knowledge_index,
        touch_prior=touch_prior,
        max_samples=max(0, args.max_knowledge_samples),
        seed=args.seed,
    )

    file_rows = _build_file_corpus_rows(
        vault_root=vault_root,
        touch_prior=touch_prior,
        max_samples=max(0, args.max_file_samples),
        max_bytes=max(1_024, args.max_bytes_per_file),
        max_chars=max(200, args.max_chars),
        seed=args.seed,
    )

    rows = knowledge_rows + file_rows
    train_rows, val_rows = _split_rows(rows, args.val_ratio, args.seed)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    training_examples_path = out_dir / "training_examples.v1.jsonl"
    manifest_path = out_dir / "manifest.json"

    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    snapshots = _snapshot_pointers(vault_root, knowledge_index, decision_ledger)
    training_examples_train = _build_training_example_rows(
        rows=train_rows,
        split="train",
        snapshots=snapshots,
    )
    training_examples_val = _build_training_example_rows(
        rows=val_rows,
        split="val",
        snapshots=snapshots,
    )
    training_examples_rows = training_examples_train + training_examples_val
    _write_jsonl(training_examples_path, training_examples_rows)

    dataset_checksums = {
        "train": describe_jsonl(train_path),
        "val": describe_jsonl(val_path),
        "training_examples": describe_jsonl(training_examples_path),
    }

    manifest = {
        "vault_root": str(vault_root),
        "knowledge_index": str(knowledge_index),
        "decision_ledger": str(decision_ledger),
        "constitution": str(
            (vault_root / "contracts/ημ_ui_training_constitution.v1.lith").resolve()
        ),
        "training_example_record": TRAINING_EXAMPLE_RECORD,
        "constitution_id": TRAINING_TAXONOMY_VERSION,
        "training_examples_path": str(training_examples_path),
        "dataset_checksums": dataset_checksums,
        "determinism": {
            "seed": int(args.seed),
            "val_ratio": float(args.val_ratio),
            "split_policy": "deterministic-random-shuffle",
        },
        "touch_prior": touch_prior,
        "train": _manifest(train_rows),
        "val": _manifest(val_rows),
        "training_examples": {
            "total": len(training_examples_rows),
            "train": len(training_examples_train),
            "val": len(training_examples_val),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "ok": True,
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "train_path": str(train_path),
                "val_path": str(val_path),
                "training_examples_path": str(training_examples_path),
                "manifest_path": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
