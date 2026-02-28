from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from .github_extract import (
    canonical_github_url,
    compute_importance_score,
    extract_github_atoms,
    extract_repo_from_canonical,
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _github_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "User-Agent": "fork-tales-part64-github-crawler/1.0",
        "Accept": "application/vnd.github+json",
    }
    token = _safe_str(os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN"))
    if token:
        # Do not log this header.
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _resource_kind_from_canonical(canonical_url: str) -> str:
    url = canonical_url.lower()
    if "raw.githubusercontent.com/" in url:
        return "github:file"
    if "/pull/" in url:
        return "github:pr"
    if "/issues/" in url:
        return "github:issue"
    if "/releases/tag/" in url or url.endswith("/releases"):
        return "github:release"
    if "/compare/" in url:
        return "github:compare"
    return "github:repo"


def _label_names(payload: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    rows = (
        payload.get("labels", []) if isinstance(payload.get("labels", []), list) else []
    )
    for row in rows:
        if isinstance(row, dict):
            token = _safe_str(row.get("name", ""))
        else:
            token = _safe_str(row)
        if token:
            labels.append(token)
    return labels


def _author_names(payload: dict[str, Any]) -> list[str]:
    authors: list[str] = []
    for key in ("user", "author"):
        row = payload.get(key)
        if isinstance(row, dict):
            token = _safe_str(row.get("login", "") or row.get("name", ""))
            if token:
                authors.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for row in authors:
        if row in seen:
            continue
        seen.add(row)
        deduped.append(row)
    return deduped


def _decode_payload(body: bytes, content_type: str) -> Any:
    if "json" in content_type.lower():
        try:
            return json.loads(body.decode("utf-8", errors="replace"))
        except (ValueError, json.JSONDecodeError):
            return {}
    return body.decode("utf-8", errors="replace")


def _read_response(
    opener: Callable[..., Any],
    request: urllib.request.Request,
    *,
    timeout_s: float,
) -> tuple[int, dict[str, str], bytes]:
    response = opener(request, timeout=timeout_s)
    try:
        status = _safe_int(getattr(response, "status", 0), 0)
        headers = dict(getattr(response, "headers", {}) or {})
        body = response.read()
        if not isinstance(body, (bytes, bytearray)):
            body = bytes(str(body or ""), "utf-8")
        return status, headers, bytes(body)
    finally:
        close_method = getattr(response, "close", None)
        if callable(close_method):
            close_method()


def fetch_github_api(
    url: str,
    config_seeds: dict[str, Any] | None = None,
    *,
    opener: Callable[..., Any] | None = None,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Fetch GitHub endpoint and return structured payload plus extraction hints."""
    config = config_seeds if isinstance(config_seeds, dict) else {}
    canonical = canonical_github_url(url)
    if not canonical:
        return {
            "ok": False,
            "url": _safe_str(url),
            "canonical_url": "",
            "error": "invalid_url",
            "status": 0,
            "duration_ms": 0,
        }

    opener_fn = opener or urllib.request.urlopen
    request = urllib.request.Request(url, headers=_github_headers(), method="GET")
    started = time.time()

    try:
        status, headers, body = _read_response(
            opener_fn,
            request,
            timeout_s=max(1.0, float(timeout_s)),
        )
        content_type = _safe_str(headers.get("Content-Type", "application/json"))
        payload = _decode_payload(body, content_type)
        content_hash = hashlib.sha256(body).hexdigest()

        resource: dict[str, Any] = {
            "kind": _resource_kind_from_canonical(canonical),
            "title": "",
            "number": 0,
            "repo": "",
            "labels": [],
            "authors": [],
            "updated_at": "",
            "importance_score": 0,
            "content_hash": content_hash,
            "text_excerpt_hash": "",
        }
        atoms: list[dict[str, Any]] = []

        if isinstance(payload, dict):
            text_excerpt = " ".join(
                [
                    _safe_str(payload.get("title", "")),
                    _safe_str(payload.get("body", "")),
                ]
            ).strip()
            text_excerpt_hash = (
                hashlib.sha256(text_excerpt.encode("utf-8")).hexdigest()
                if text_excerpt
                else ""
            )

            atoms = extract_github_atoms(canonical, payload, config)
            touched_files = payload.get("filenames_touched", [])
            score = compute_importance_score(
                payload,
                atoms,
                touched_files=touched_files
                if isinstance(touched_files, list)
                else None,
            )

            resource.update(
                {
                    "title": _safe_str(payload.get("title", payload.get("name", "")))[
                        :240
                    ],
                    "number": _safe_int(payload.get("number", 0), 0),
                    "repo": extract_repo_from_canonical(canonical, payload),
                    "labels": _label_names(payload),
                    "authors": _author_names(payload),
                    "updated_at": _safe_str(
                        payload.get("updated_at", payload.get("published_at", ""))
                    ),
                    "importance_score": int(score),
                    "text_excerpt_hash": text_excerpt_hash,
                }
            )

        return {
            "ok": True,
            "url": _safe_str(url),
            "canonical_url": canonical,
            "api_endpoint": _safe_str(url),
            "status": status,
            "duration_ms": int(max(0.0, (time.time() - started) * 1000.0)),
            "payload_bytes": len(body),
            "rate_limit_remaining": _safe_int(
                headers.get("X-RateLimit-Remaining", 5000), 5000
            ),
            "rate_limit_reset": _safe_int(headers.get("X-RateLimit-Reset", 0), 0),
            "payload": payload,
            "resource": resource,
            "atoms": atoms,
        }

    except urllib.error.HTTPError as exc:
        reset_at = 0
        try:
            reset_at = _safe_int((exc.headers or {}).get("X-RateLimit-Reset", 0), 0)
        except Exception:
            reset_at = 0
        return {
            "ok": False,
            "url": _safe_str(url),
            "canonical_url": canonical,
            "api_endpoint": _safe_str(url),
            "error": "http_error",
            "status": _safe_int(getattr(exc, "code", 0), 0),
            "duration_ms": int(max(0.0, (time.time() - started) * 1000.0)),
            "rate_limit_reset": reset_at,
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": _safe_str(url),
            "canonical_url": canonical,
            "api_endpoint": _safe_str(url),
            "error": f"{exc.__class__.__name__}:{exc}",
            "status": 0,
            "duration_ms": int(max(0.0, (time.time() - started) * 1000.0)),
        }
