"""GitHub conversation fetch/format helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable
import urllib.error
from urllib.request import Request, urlopen


def github_conversation_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "User-Agent": "fork-tales-part64-github-conversation/1.0",
        "Accept": "application/vnd.github+json",
    }
    token = str(os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_conversation_fetch_json(
    url: str,
    *,
    timeout_s: float,
    safe_float: Callable[[Any, float], float],
    headers_builder: Callable[[], dict[str, str]],
) -> tuple[bool, Any, int, str]:
    request = Request(url, headers=headers_builder(), method="GET")
    try:
        with urlopen(request, timeout=max(2.0, float(timeout_s))) as response:
            status = int(safe_float(getattr(response, "status", 200), 200.0))
            payload_bytes = response.read()
            if not isinstance(payload_bytes, (bytes, bytearray)):
                payload_bytes = bytes(str(payload_bytes or ""), "utf-8")
            payload_text = bytes(payload_bytes).decode("utf-8", errors="replace")
            try:
                payload = json.loads(payload_text)
            except Exception:
                payload = {}
            return True, payload, status, ""
    except urllib.error.HTTPError as exc:
        status = int(safe_float(getattr(exc, "code", 0), 0.0))
        detail = ""
        try:
            detail_bytes = exc.read()
            if isinstance(detail_bytes, (bytes, bytearray)):
                detail = bytes(detail_bytes).decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        detail = str(detail or "").strip()
        if len(detail) > 280:
            detail = f"{detail[:279].rstrip()}…"
        return False, {}, status, detail or f"http_error:{status}"
    except Exception as exc:
        return False, {}, 0, f"{exc.__class__.__name__}:{exc}"


def github_conversation_comment_rows(
    payload: Any,
    *,
    channel: str,
    max_comments: int,
    max_body_chars: int,
) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    entries: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        body = str(row.get("body", "") or "").strip()
        if not body:
            continue
        if len(body) > max_body_chars:
            body = f"{body[: max(1, max_body_chars - 1)].rstrip()}…"
        user_row = row.get("user", {}) if isinstance(row.get("user", {}), dict) else {}
        author = str(user_row.get("login", "") or "").strip() or "unknown"
        created_at = str(
            row.get("created_at", "") or row.get("submitted_at", "") or ""
        ).strip()
        updated_at = str(
            row.get("updated_at", "") or row.get("submitted_at", "") or ""
        ).strip()
        html_url = str(row.get("html_url", "") or "").strip()
        entry_id = str(row.get("id", "") or "").strip()
        entries.append(
            {
                "id": entry_id,
                "channel": channel,
                "author": author,
                "created_at": created_at,
                "updated_at": updated_at,
                "url": html_url,
                "body": body,
            }
        )
        if len(entries) >= max_comments:
            break
    entries.sort(
        key=lambda row: (
            str(row.get("created_at", "")),
            str(row.get("id", "")),
            str(row.get("channel", "")),
        )
    )
    return entries[:max_comments]


def github_conversation_markdown(
    *,
    repo: str,
    number: int,
    kind: str,
    title: str,
    state: str,
    html_url: str,
    root_body: str,
    comments: list[dict[str, Any]],
    max_markdown_chars: int,
) -> str:
    heading = title or f"{kind} #{number}"
    lines: list[str] = [
        f"# {heading}",
        "",
        f"- Repo: {repo}",
        f"- Kind: {kind}",
        f"- Number: {max(0, int(number))}",
        f"- State: {state or 'unknown'}",
    ]
    if html_url:
        lines.append(f"- URL: {html_url}")

    lines.extend(["", "## Root", "", root_body or "_No root message body returned._"])
    lines.extend(["", "## Conversation Chain", ""])
    if comments:
        for index, row in enumerate(comments, start=1):
            author = str(row.get("author", "unknown") or "unknown").strip() or "unknown"
            created_at = str(row.get("created_at", "") or "").strip() or "unknown-time"
            channel = str(row.get("channel", "comment") or "comment").strip()
            body = str(row.get("body", "") or "").strip()
            url = str(row.get("url", "") or "").strip()
            lines.append(f"### {index}. @{author} · {created_at} · {channel}")
            if url:
                lines.append(f"- Link: {url}")
            lines.append("")
            lines.append(body or "_empty comment body_")
            lines.append("")
    else:
        lines.append("_No comments returned for this item._")

    markdown = "\n".join(lines).strip()
    if len(markdown) > max_markdown_chars:
        markdown = f"{markdown[: max(1, max_markdown_chars - 1)].rstrip()}…"
    return f"{markdown}\n"


def github_conversation_payload(
    *,
    repo: str,
    number: int,
    kind: str,
    max_comments: int,
    max_root_body_chars: int,
    max_comment_body_chars: int,
    max_markdown_chars: int,
    include_review_comments: bool,
    timeout_s: float,
    fetch_json: Callable[[str], tuple[bool, Any, int, str]],
    comment_rows: Callable[..., list[dict[str, Any]]],
    markdown_builder: Callable[..., str],
) -> dict[str, Any]:
    normalized_kind = str(kind or "").strip().lower()
    if normalized_kind not in {"github:issue", "github:pr"}:
        normalized_kind = "github:issue"

    safe_number = max(1, int(number))
    if normalized_kind == "github:pr":
        item_url = f"https://api.github.com/repos/{repo}/pulls/{safe_number}"
    else:
        item_url = f"https://api.github.com/repos/{repo}/issues/{safe_number}"

    ok_item, item_payload, item_status, item_error = fetch_json(item_url)
    if not ok_item or not isinstance(item_payload, dict):
        return {
            "ok": False,
            "error": item_error or "github_item_fetch_failed",
            "status": item_status,
            "repo": repo,
            "number": safe_number,
            "kind": normalized_kind,
        }

    title = str(item_payload.get("title", "") or item_payload.get("name", "")).strip()
    state = str(item_payload.get("state", "") or "").strip()
    html_url = str(item_payload.get("html_url", "") or "").strip()
    root_body = str(item_payload.get("body", "") or "").strip()
    if len(root_body) > max_root_body_chars:
        root_body = f"{root_body[: max(1, max_root_body_chars - 1)].rstrip()}…"

    comments: list[dict[str, Any]] = []
    issue_comments_url = f"https://api.github.com/repos/{repo}/issues/{safe_number}/comments?per_page=100"
    ok_issue_comments, issue_comments_payload, _, _ = fetch_json(issue_comments_url)
    if ok_issue_comments:
        comments.extend(
            comment_rows(
                issue_comments_payload,
                channel="issue-comment",
                max_comments=max_comments,
                max_body_chars=max_comment_body_chars,
            )
        )

    if normalized_kind == "github:pr" and include_review_comments:
        reviews_url = f"https://api.github.com/repos/{repo}/pulls/{safe_number}/reviews?per_page=100"
        ok_reviews, reviews_payload, _, _ = fetch_json(reviews_url)
        if ok_reviews and len(comments) < max_comments:
            remaining = max(1, max_comments - len(comments))
            comments.extend(
                comment_rows(
                    reviews_payload,
                    channel="review",
                    max_comments=remaining,
                    max_body_chars=max_comment_body_chars,
                )
            )

        review_comments_url = f"https://api.github.com/repos/{repo}/pulls/{safe_number}/comments?per_page=100"
        ok_review_comments, review_comments_payload, _, _ = fetch_json(
            review_comments_url
        )
        if ok_review_comments and len(comments) < max_comments:
            remaining = max(1, max_comments - len(comments))
            comments.extend(
                comment_rows(
                    review_comments_payload,
                    channel="review-comment",
                    max_comments=remaining,
                    max_body_chars=max_comment_body_chars,
                )
            )

    deduped_comments: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for row in comments:
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("id", "")),
            str(row.get("channel", "")),
            str(row.get("created_at", "")),
            str(row.get("body", "")),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_comments.append(row)
        if len(deduped_comments) >= max_comments:
            break

    deduped_comments.sort(
        key=lambda row: (
            str(row.get("created_at", "")),
            str(row.get("id", "")),
            str(row.get("channel", "")),
        )
    )
    markdown = markdown_builder(
        repo=repo,
        number=safe_number,
        kind=normalized_kind,
        title=title,
        state=state,
        html_url=html_url,
        root_body=root_body,
        comments=deduped_comments,
        max_markdown_chars=max_markdown_chars,
    )

    token_configured = bool(
        str(os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()
    )
    return {
        "ok": True,
        "record": "eta-mu.github-conversation.v1",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "repo": repo,
        "number": safe_number,
        "kind": normalized_kind,
        "title": title,
        "state": state,
        "url": html_url,
        "root_body": root_body,
        "comment_count": len(deduped_comments),
        "comments": deduped_comments,
        "markdown": markdown,
        "token_configured": token_configured,
    }
