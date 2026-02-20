from __future__ import annotations
import os
import re
import json
import hashlib
import fnmatch
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from hashlib import sha1
from .constants import (
    ETA_MU_INBOX_DIRNAME,
    ETA_MU_KNOWLEDGE_INDEX_REL,
    ETA_MU_KNOWLEDGE_ARCHIVE_REL,
    ETA_MU_DOCMETA_REL,
    ETA_MU_REGISTRY_REL,
    ETA_MU_EMBEDDINGS_DB_REL,
    ETA_MU_GRAPH_MOVES_REL,
    PRESENCE_ACCOUNTS_LOG_REL,
    IMAGE_COMMENTS_LOG_REL,
    STUDY_SNAPSHOT_LOG_REL,
    WIKIMEDIA_STREAM_LOG_REL,
    NWS_ALERTS_LOG_REL,
    SWPC_ALERTS_LOG_REL,
    GIBS_LAYERS_LOG_REL,
    EONET_EVENTS_LOG_REL,
    EMSC_STREAM_LOG_REL,
)


def _eta_mu_substrate_root(vault_root: Path) -> Path:
    primary = vault_root.resolve()
    primary_roots: list[Path] = [primary, *primary.parents]
    cwd = Path.cwd().resolve()
    fallback_roots: list[Path] = [cwd, *cwd.parents]

    seen: set[Path] = set()

    def _dedupe(roots: list[Path]) -> list[Path]:
        ordered: list[Path] = []
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            ordered.append(root)
        return ordered

    primary_lineage = _dedupe(primary_roots)
    fallback_lineage = _dedupe(fallback_roots)

    for root in primary_lineage:
        inbox = root / ETA_MU_INBOX_DIRNAME
        if inbox.exists() and inbox.is_dir():
            return root
    for root in primary_lineage:
        opencode_dir = root / ".opencode"
        if opencode_dir.exists() and opencode_dir.is_dir():
            return root

    for root in fallback_lineage:
        inbox = root / ETA_MU_INBOX_DIRNAME
        if inbox.exists() and inbox.is_dir():
            return root
    for root in fallback_lineage:
        opencode_dir = root / ".opencode"
        if opencode_dir.exists() and opencode_dir.is_dir():
            return root

    return primary


def _eta_mu_inbox_root(vault_root: Path) -> Path:
    base = _eta_mu_substrate_root(vault_root)
    return (base / ETA_MU_INBOX_DIRNAME).resolve()


def _eta_mu_knowledge_archive_root(vault_root: Path) -> Path:
    base = _eta_mu_substrate_root(vault_root)
    return (base / ETA_MU_KNOWLEDGE_ARCHIVE_REL).resolve()


def _eta_mu_knowledge_index_path(vault_root: Path) -> Path:
    base = _eta_mu_substrate_root(vault_root)
    return (base / ETA_MU_KNOWLEDGE_INDEX_REL).resolve()


def _eta_mu_docmeta_path(vault_root: Path) -> Path:
    base = _eta_mu_substrate_root(vault_root)
    return (base / ETA_MU_DOCMETA_REL).resolve()


def _eta_mu_registry_path(vault_root: Path) -> Path:
    base = _eta_mu_substrate_root(vault_root)
    return (base / ETA_MU_REGISTRY_REL).resolve()


def _eta_mu_output_root(vault_root: Path) -> Path:
    base = _eta_mu_substrate_root(vault_root)
    return (base / ".Π").resolve()


def _study_snapshot_log_path(vault_root: Path) -> Path:
    return (vault_root / STUDY_SNAPSHOT_LOG_REL).resolve()


def _embeddings_db_path(vault_root: Path) -> Path:
    base = vault_root.resolve()
    return (base / ETA_MU_EMBEDDINGS_DB_REL).resolve()


def _file_graph_moves_path(vault_root: Path) -> Path:
    base = vault_root.resolve()
    return (base / ETA_MU_GRAPH_MOVES_REL).resolve()


def _presence_accounts_log_path(vault_root: Path) -> Path:
    from .constants import PRESENCE_ACCOUNTS_LOG_REL

    base = vault_root.resolve()
    return (base / PRESENCE_ACCOUNTS_LOG_REL).resolve()


def _simulation_metadata_log_path(vault_root: Path) -> Path:
    from .constants import SIMULATION_METADATA_LOG_REL

    base = vault_root.resolve()
    return (base / SIMULATION_METADATA_LOG_REL).resolve()


def _image_comments_log_path(vault_root: Path) -> Path:
    base = vault_root.resolve()
    return (base / IMAGE_COMMENTS_LOG_REL).resolve()


def _wikimedia_stream_log_path(vault_root: Path) -> Path:
    base = vault_root.resolve()
    return (base / WIKIMEDIA_STREAM_LOG_REL).resolve()


def _nws_alerts_log_path(vault_root: Path) -> Path:
    base = vault_root.resolve()
    return (base / NWS_ALERTS_LOG_REL).resolve()


def _swpc_alerts_log_path(vault_root: Path) -> Path:
    base = vault_root.resolve()
    return (base / SWPC_ALERTS_LOG_REL).resolve()


def _gibs_layers_log_path(vault_root: Path) -> Path:
    base = vault_root.resolve()
    return (base / GIBS_LAYERS_LOG_REL).resolve()


def _eonet_events_log_path(vault_root: Path) -> Path:
    base = vault_root.resolve()
    return (base / EONET_EVENTS_LOG_REL).resolve()


def _emsc_stream_log_path(vault_root: Path) -> Path:
    base = vault_root.resolve()
    return (base / EMSC_STREAM_LOG_REL).resolve()


def _safe_rel_path(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.name
    return str(rel).replace("\\", "/")


def discover_part_roots(vault_root: Path, part_root: Path) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    candidates = [part_root, *sorted(vault_root.glob("ημ_op_mf_part_*"))]
    for candidate in candidates:
        if not candidate.is_dir():
            continue

        direct_manifest = candidate / "manifest.json"
        nested_manifest = candidate / candidate.name / "manifest.json"
        resolved: Path | None = None
        if direct_manifest.exists():
            resolved = candidate.resolve()
        elif nested_manifest.exists():
            resolved = (candidate / candidate.name).resolve()

        if resolved is not None and resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)

    return roots


def _part_label(part_root: Path, manifest: dict[str, Any]) -> str:
    if "part" in manifest:
        return str(manifest["part"])
    if "name" in manifest:
        return str(manifest["name"])
    return part_root.name


def _locate_receipts_log(vault_root: Path, part_root: Path) -> Path | None:
    vault_base = vault_root.resolve()
    part_base = part_root.resolve()
    bases = [vault_base, part_base]

    cwd_base = Path.cwd().resolve()
    if any(
        cwd_base == base or cwd_base in base.parents or base in cwd_base.parents
        for base in bases
    ):
        bases.append(cwd_base)

    candidates: list[Path] = []
    for base in bases:
        candidates.append(base / "receipts.log")
        for parent in base.parents:
            candidates.append(parent / "receipts.log")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _parse_receipt_line(line: str) -> dict[str, str]:
    row: dict[str, str] = {}
    parts = [part.strip() for part in line.split(" | ") if part.strip()]
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
        elif ":" in part:
            key, value = part.split(":", 1)
        else:
            continue
        row[key.strip()] = value.strip()
    return row


def _split_receipt_refs(refs_value: str) -> list[str]:
    return [item.strip() for item in str(refs_value or "").split(",") if item.strip()]


def _eta_mu_inbox_rel_path(path: Path, inbox_root: Path) -> str:
    try:
        rel = path.relative_to(inbox_root)
    except ValueError:
        try:
            rel = path.resolve().relative_to(inbox_root.resolve())
        except ValueError:
            return path.name
    return str(rel).replace("\\", "/")


def _eta_mu_is_excluded_inbox_rel(rel_path: str) -> bool:
    from .constants import (
        ETA_MU_INGEST_EXCLUDE_REL_PATHS,
        ETA_MU_INGEST_EXCLUDE_GLOBS,
    )

    normalized = str(rel_path).strip().replace("\\", "/")
    if not normalized:
        return True

    parts = [token for token in normalized.split("/") if token]
    if any(token in ETA_MU_INGEST_EXCLUDE_REL_PATHS for token in parts):
        return True

    if any(
        fnmatch.fnmatch(normalized, pattern) for pattern in ETA_MU_INGEST_EXCLUDE_GLOBS
    ):
        return True
    return False


def _eta_mu_scan_candidates(inbox_root: Path) -> list[Path]:
    from .constants import (
        ETA_MU_INGEST_MAX_SCAN_FILES,
        ETA_MU_INGEST_MAX_SCAN_DEPTH,
    )

    if not inbox_root.exists() or not inbox_root.is_dir():
        return []

    candidates: list[Path] = []
    for path in inbox_root.rglob("*"):
        if not path.is_file():
            continue
        rel = _eta_mu_inbox_rel_path(path, inbox_root)
        if _eta_mu_is_excluded_inbox_rel(rel):
            continue
        depth = max(0, len([token for token in rel.split("/") if token]) - 1)
        if depth > ETA_MU_INGEST_MAX_SCAN_DEPTH:
            continue
        candidates.append(path)
        if len(candidates) >= ETA_MU_INGEST_MAX_SCAN_FILES:
            break
    return candidates


def _eta_mu_rejected_target_path(
    *,
    inbox_root: Path,
    source_path: Path,
    source_hash: str,
) -> Path:
    rel = _eta_mu_inbox_rel_path(source_path, inbox_root)
    parts = [token for token in rel.split("/") if token]
    if not parts:
        parts = [source_path.name]
    leaf = _sanitize_archive_name(parts[-1])
    prefix = source_hash[:12] if source_hash else "unknown"
    stamped = f"{prefix}_{leaf}" if leaf else prefix
    return (inbox_root / "_rejected" / stamped).resolve()


def _sanitize_archive_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "artifact"


def _archive_member_name(name: str) -> str:
    return f"payload/{_sanitize_archive_name(name)}"


def _archive_container_id(archive_rel_path: str) -> str:
    digest = sha1(archive_rel_path.encode("utf-8")).hexdigest()[:14]
    return f"archive:{digest}"


def _eta_mu_iso_compact(ts: datetime | None = None) -> str:
    stamp = ts or datetime.now(timezone.utc)
    return stamp.strftime("%Y%m%dT%H%M%SZ")


def _eta_mu_write_sexp_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "(artifact " + _eta_mu_sexp_atom(payload) + ")\n"
    path.write_text(content, encoding="utf-8")


def _eta_mu_sexp_atom(value: Any) -> str:
    import math

    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "nil"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return "0.0"
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        inner = " ".join(_eta_mu_sexp_atom(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, dict):
        parts: list[str] = []
        for key in sorted(value.keys()):
            raw_key = str(key)
            if re.fullmatch(r"[A-Za-z0-9_.+\-/]+", raw_key):
                key_form = f":{raw_key}"
            else:
                key_form = json.dumps(raw_key, ensure_ascii=False)
            parts.append(f"({key_form} {_eta_mu_sexp_atom(value[key])})")
        return f"(map {' '.join(parts)})"
    return json.dumps(str(value), ensure_ascii=False)


def _cleanup_empty_inbox_dirs(inbox_root: Path) -> None:
    for path in sorted(inbox_root.rglob("*"), reverse=True):
        if not path.is_dir():
            continue
        try:
            path.rmdir()
        except OSError:
            continue


def _ensure_receipts_log_path(vault_root: Path, part_root: Path) -> Path:
    located = _locate_receipts_log(vault_root, part_root)
    if located is not None:
        return located
    fallback = (vault_root / "receipts.log").resolve()
    fallback.parent.mkdir(parents=True, exist_ok=True)
    if not fallback.exists():
        fallback.touch()
    return fallback


def _append_receipt_line(
    path: Path,
    *,
    kind: str,
    origin: str,
    owner: str,
    dod: str,
    pi: str,
    host: str,
    manifest: str,
    refs: list[str],
    note: str = "",
) -> None:
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).isoformat()
    kind_value = kind if str(kind).startswith(":") else f":{kind}"
    refs_value = ",".join(sorted({ref for ref in refs if ref}))
    fields = [
        f"ts={ts}",
        f"kind={kind_value}",
        f"origin={origin}",
        f"owner={owner}",
        f"dod={dod}",
        f"pi={pi}",
        f"host={host}",
        f"manifest={manifest}",
        f"refs={refs_value}",
    ]
    if note:
        fields.append(f"note={note}")
    line = " | ".join(fields)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
