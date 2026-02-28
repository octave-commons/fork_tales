from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse, urlunparse


_ALLOWED_GITHUB_HOSTS: set[str] = {
    "github.com",
    "api.github.com",
    "raw.githubusercontent.com",
}

_SECURITY_LABELS: set[str] = {"security", "bug", "hotfix"}

_LOCKFILE_AND_MANIFEST_NAMES: set[str] = {
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "requirements.txt",
    "pipfile",
    "pipfile.lock",
    "poetry.lock",
    "cargo.lock",
    "go.mod",
    "go.sum",
}

_SECURITY_TERMS: set[str] = {
    "cve",
    "xxe",
    "token",
    "auth",
    "oauth",
    "exploit",
    "leak",
    "credential",
    "secret",
    "security",
}


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _normalized_segments(path: str) -> list[str]:
    normalized = str(path or "").replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return [segment for segment in normalized.split("/") if segment]


def _canonical_html_from_repo_parts(owner: str, repo: str, tail: list[str]) -> str:
    clean_owner = _safe_str(owner)
    clean_repo = _safe_str(repo).removesuffix(".git")
    if not clean_owner or not clean_repo:
        return ""

    base = f"https://github.com/{clean_owner}/{clean_repo}"
    if not tail:
        return base

    head = _safe_str(tail[0]).lower()
    if head in {"pull", "pulls"}:
        if len(tail) >= 2 and str(tail[1]).isdigit():
            suffix = "/".join(
                _safe_str(piece) for piece in tail[2:] if _safe_str(piece)
            )
            if suffix:
                return f"{base}/pull/{int(str(tail[1]))}/{suffix}"
            return f"{base}/pull/{int(str(tail[1]))}"
        return f"{base}/pulls"

    if head == "issues":
        if len(tail) >= 2 and str(tail[1]).isdigit():
            suffix = "/".join(
                _safe_str(piece) for piece in tail[2:] if _safe_str(piece)
            )
            if suffix:
                return f"{base}/issues/{int(str(tail[1]))}/{suffix}"
            return f"{base}/issues/{int(str(tail[1]))}"
        return f"{base}/issues"

    if head == "releases":
        if len(tail) >= 3 and _safe_str(tail[1]).lower() in {"tag", "tags"}:
            tag = "/".join(_safe_str(piece) for piece in tail[2:] if _safe_str(piece))
            if tag:
                return f"{base}/releases/tag/{tag}"
        return f"{base}/releases"

    if head == "compare" and len(tail) >= 2:
        compare_spec = "/".join(
            _safe_str(piece) for piece in tail[1:] if _safe_str(piece)
        )
        if compare_spec:
            return f"{base}/compare/{compare_spec}"

    suffix = "/".join(_safe_str(piece) for piece in tail if _safe_str(piece))
    return f"{base}/{suffix}" if suffix else base


def canonical_github_url(raw_url: str) -> str:
    """Normalize GitHub URLs to deterministic canonical forms."""
    text = _safe_str(raw_url)
    if not text:
        return ""

    try:
        parsed = urlparse(text)
    except Exception:
        return ""

    scheme = _safe_str(parsed.scheme).lower()
    if scheme not in {"http", "https"}:
        return ""

    host = _safe_str(parsed.hostname).lower()
    if host not in _ALLOWED_GITHUB_HOSTS:
        return ""

    segments = _normalized_segments(parsed.path)
    if not segments:
        return ""

    if host == "github.com":
        if len(segments) < 2:
            return ""
        return _canonical_html_from_repo_parts(segments[0], segments[1], segments[2:])

    if host == "api.github.com":
        if len(segments) < 3 or _safe_str(segments[0]).lower() != "repos":
            return ""
        owner = segments[1]
        repo = segments[2]
        tail = segments[3:]
        return _canonical_html_from_repo_parts(owner, repo, tail)

    # raw.githubusercontent.com
    if len(segments) < 4:
        return ""
    owner = _safe_str(segments[0])
    repo = _safe_str(segments[1]).removesuffix(".git")
    ref = _safe_str(segments[2])
    path_tail = "/".join(_safe_str(piece) for piece in segments[3:] if _safe_str(piece))
    if not owner or not repo or not ref or not path_tail:
        return ""
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path_tail}"


def extract_repo_from_canonical(
    canonical_url: str, payload: dict[str, Any] | None = None
) -> str:
    payload_obj = payload if isinstance(payload, dict) else {}
    for key in ("full_name", "repository", "repo"):
        value = payload_obj.get(key)
        if isinstance(value, str) and "/" in value:
            return _safe_str(value)
        if isinstance(value, dict):
            inner = _safe_str(value.get("full_name", ""))
            if "/" in inner:
                return inner

    parsed = urlparse(_safe_str(canonical_url))
    if _safe_str(parsed.hostname).lower() not in {
        "github.com",
        "raw.githubusercontent.com",
    }:
        return ""
    segments = _normalized_segments(parsed.path)
    if len(segments) < 2:
        return ""
    return f"{segments[0]}/{segments[1].removesuffix('.git')}"


def _labels_from_payload(payload: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for row in (
        payload.get("labels", []) if isinstance(payload.get("labels", []), list) else []
    ):
        if isinstance(row, dict):
            token = _safe_str(row.get("name", "")).lower()
        else:
            token = _safe_str(row).lower()
        if token:
            labels.append(token)
    return labels


def _touched_files_from_payload(payload: dict[str, Any]) -> list[str]:
    touched: list[str] = []

    explicit = payload.get("filenames_touched", [])
    if isinstance(explicit, list):
        for row in explicit:
            token = _safe_str(row)
            if token:
                touched.append(token)

    files_payload = payload.get("files", [])
    if isinstance(files_payload, list):
        for row in files_payload:
            if not isinstance(row, dict):
                continue
            token = _safe_str(row.get("filename", ""))
            if token:
                touched.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in touched:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:200]


def _candidate_terms(config_seeds: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for row in (
        config_seeds.get("keywords", []) if isinstance(config_seeds, dict) else []
    ):
        token = _safe_str(row).lower()
        if not token or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _candidate_file_patterns(config_seeds: dict[str, Any]) -> set[str]:
    patterns: set[str] = set(_LOCKFILE_AND_MANIFEST_NAMES)
    values = (
        config_seeds.get("file_patterns", []) if isinstance(config_seeds, dict) else []
    )
    if isinstance(values, list):
        for row in values:
            token = _safe_str(row).lower()
            if token:
                patterns.add(token)
    return patterns


def _canonical_atom_key(atom: dict[str, Any]) -> str:
    return json.dumps(atom, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dedupe_sort_atoms(
    atoms: list[dict[str, Any]], *, max_atoms: int
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for atom in sorted(atoms, key=_canonical_atom_key):
        token = _canonical_atom_key(atom)
        if token in seen:
            continue
        seen.add(token)
        deduped.append(atom)
        if len(deduped) >= max_atoms:
            break
    return deduped


def extract_diff_keyword_hits(
    file_rows: list[dict[str, Any]] | Any,
    *,
    keywords: list[str],
    max_matches: int = 24,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if not isinstance(file_rows, list) or not file_rows:
        return matches

    terms = [token for token in keywords if token]
    if not terms:
        return matches

    for row in file_rows[:200]:
        if not isinstance(row, dict):
            continue
        filename = _safe_str(row.get("filename", ""))
        patch = _safe_str(row.get("patch", "")).lower()
        if not filename:
            continue
        for term in terms:
            if term in patch:
                matches.append(
                    {
                        "kind": "diff_keyword_match",
                        "file": filename,
                        "term": term,
                    }
                )

    return _dedupe_sort_atoms(matches, max_atoms=max(1, int(max_matches)))


def extract_github_atoms(
    canonical_url: str,
    payload: dict[str, Any],
    config_seeds: dict[str, Any],
    *,
    max_atoms: int = 50,
) -> list[dict[str, Any]]:
    """Extract bounded, deterministic observation atoms."""
    if not isinstance(payload, dict):
        return []

    repo = extract_repo_from_canonical(canonical_url, payload)
    atoms: list[dict[str, Any]] = []

    text_parts: list[str] = []
    for key in (
        "title",
        "body",
        "name",
        "tag_name",
        "message",
    ):
        token = _safe_str(payload.get(key, ""))
        if token:
            text_parts.append(token)
    corpus = "\n".join(text_parts)
    corpus_lower = corpus.lower()

    for term in _candidate_terms(config_seeds):
        if term and term in corpus_lower:
            atoms.append(
                {
                    "kind": "mentions",
                    "repo": repo,
                    "term": term,
                }
            )

    cve_tokens = sorted(
        {
            match.upper()
            for match in re.findall(r"cve-\d{4}-\d{4,7}", corpus_lower)
            if match
        }
    )
    for cve_id in cve_tokens:
        atoms.append(
            {
                "kind": "references_cve",
                "repo": repo,
                "cve_id": cve_id,
            }
        )

    for label in _labels_from_payload(payload):
        if label in _SECURITY_LABELS:
            atoms.append(
                {
                    "kind": "has_label",
                    "repo": repo,
                    "label": label,
                }
            )

    touched_files = _touched_files_from_payload(payload)
    pattern_set = _candidate_file_patterns(config_seeds)
    for filename in touched_files:
        atoms.append(
            {
                "kind": "mentions_file",
                "repo": repo,
                "path": filename,
            }
        )
        prefix = _safe_str(filename.split("/", 1)[0]).lower()
        if prefix:
            atoms.append(
                {
                    "kind": "touches_path",
                    "repo": repo,
                    "path_prefix": prefix,
                }
            )

        leaf = _safe_str(filename.split("/")[-1]).lower()
        if leaf in pattern_set:
            atoms.append(
                {
                    "kind": "changes_dependency",
                    "repo": repo,
                    "dep_name": leaf,
                }
            )

    number_value = payload.get("number")
    if isinstance(number_value, int):
        state = _safe_str(payload.get("state", "")).lower()
        if state:
            atoms.append(
                {
                    "kind": "pr_state",
                    "repo": repo,
                    "pr_number": number_value,
                    "state": state,
                }
            )
        merged_at = _safe_str(payload.get("merged_at", ""))
        if merged_at:
            atoms.append(
                {
                    "kind": "pr_merged",
                    "repo": repo,
                    "pr_number": number_value,
                    "merged_at": merged_at,
                }
            )

    return _dedupe_sort_atoms(atoms, max_atoms=max(1, int(max_atoms)))


def compute_importance_score(
    payload: dict[str, Any],
    atoms: list[dict[str, Any]],
    *,
    touched_files: list[str] | None = None,
) -> int:
    """Compute deterministic importance score from payload and atoms."""
    if not isinstance(payload, dict):
        return 0

    score = 0
    atom_kinds = {str(row.get("kind", "")).strip().lower() for row in atoms}

    mentioned_terms = {
        str(row.get("term", "")).strip().lower()
        for row in atoms
        if str(row.get("kind", "")).strip().lower() == "mentions"
    }

    file_rows = (
        touched_files
        if isinstance(touched_files, list)
        else _touched_files_from_payload(payload)
    )
    file_rows_lower = [
        str(path).strip().lower() for path in file_rows if str(path).strip()
    ]

    if any(
        path.split("/")[-1] in _LOCKFILE_AND_MANIFEST_NAMES for path in file_rows_lower
    ):
        score += 3

    if "references_cve" in atom_kinds or any(
        term in _SECURITY_TERMS for term in mentioned_terms
    ):
        score += 3

    if any(
        any(
            token in path
            for token in ("auth", "token", "credential", "secret", "oauth")
        )
        for path in file_rows_lower
    ):
        score += 2

    comments = int(payload.get("comments", 0) or 0)
    reactions_total = 0
    reactions = payload.get("reactions", {})
    if isinstance(reactions, dict):
        for value in reactions.values():
            if isinstance(value, int):
                reactions_total += max(0, value)
    if comments >= 6 or reactions_total >= 10:
        score += 2

    state = _safe_str(payload.get("state", "")).lower()
    merged_at = _safe_str(payload.get("merged_at", ""))
    if state in {"closed", "merged"} or merged_at:
        score += 1

    if "has_label" in atom_kinds:
        score += 1

    return int(score)
