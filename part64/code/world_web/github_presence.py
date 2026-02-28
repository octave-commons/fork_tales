from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .github_extract import (
    canonical_github_url,
    compute_importance_score,
    extract_diff_keyword_hits,
    extract_github_atoms,
)
from .github_fetcher import fetch_github_api

_LOGGER = logging.getLogger(__name__)

_MAX_EVENT_ROWS = 480
_MAX_RESOURCES = 640
_MAX_TRACE_EDGES = 2400
_MAX_CONVERSATION_ROWS = 180
_MAX_CONVERSATION_BODY_CHARS = 6000
_MAX_CONVERSATION_TEXT_CHARS = 48000
_MAX_SUMMARY_CHARS = 320
_MAX_EXCERPT_CHARS = 9000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    token = _safe_str(value).lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _bounded_text(value: Any, limit: int) -> str:
    text = _safe_str(value)
    if len(text) <= max(0, int(limit)):
        return text
    return text[: max(0, int(limit))]


def _canonical_repo_name(value: Any) -> str:
    repo = _safe_str(value).strip("/")
    if not repo or "/" not in repo:
        return ""
    owner, name = repo.split("/", 1)
    owner = _safe_str(owner)
    name = _safe_str(name)
    if not owner or not name:
        return ""
    return f"{owner}/{name}"


def _url_id(canonical_url: str) -> str:
    clean = _safe_str(canonical_url)
    if not clean:
        return ""
    return "url:" + hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]


def _resource_id(canonical_url: str, content_hash: str) -> str:
    basis = _safe_str(content_hash).lower() or _safe_str(canonical_url)
    if not basis:
        return ""
    return "res:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _hash_event_id(row: dict[str, Any]) -> str:
    canonical = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return "evt:" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def _position_from_token(token: str) -> tuple[float, float]:
    digest = hashlib.sha1(_safe_str(token).encode("utf-8")).digest()
    x = round((int.from_bytes(digest[0:2], "big") / 65535.0), 4)
    y = round((int.from_bytes(digest[2:4], "big") / 65535.0), 4)
    return x, y


class GithubPresence:
    def __init__(
        self,
        part_root: Path,
        *,
        fetcher: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.part_root = self._resolve_part_root(Path(part_root))
        self.world_state_root = self.part_root / "world_state"
        self.config_path = self.world_state_root / "config" / "github_seeds.json"
        self.state_path = self.world_state_root / "github_crawler_state.json"
        self.log_path = self.world_state_root / "github_crawler_events.jsonl"
        self.graph_path = self.world_state_root / "github_crawler_graph.json"
        self._fetcher = fetcher or fetch_github_api
        self.config = self._load_config()
        self.state = self._load_state()

    def _resolve_part_root(self, candidate: Path) -> Path:
        resolved = candidate.resolve()
        if (resolved / "world_state").exists():
            return resolved
        if (resolved / "part64" / "world_state").exists():
            return (resolved / "part64").resolve()
        return resolved

    def _load_config(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "enabled": True,
            "repos": [],
            "keywords": [],
            "file_patterns": [
                "package.json",
                "pnpm-lock.yaml",
                "requirements.txt",
                "Cargo.lock",
            ],
            "max_repos": 8,
            "max_items_per_repo": 5,
            "max_github_fetches_per_tick": 2,
            "max_concurrent_github_fetches": 2,
            "repo_cooldown_s": 300,
            "url_cooldown_s": 1800,
            "sweep_interval_seconds": 600,
            "endpoint_order": ["pulls", "issues", "releases"],
        }
        if not self.config_path.exists():
            return defaults
        try:
            payload = json.loads(self.config_path.read_text("utf-8"))
            if not isinstance(payload, dict):
                return defaults
        except Exception:
            return defaults

        merged = dict(defaults)
        merged.update(payload)
        merged["repos"] = [
            repo
            for repo in (
                _canonical_repo_name(row) for row in payload.get("repos", []) if row
            )
            if repo
        ]
        return merged

    def _default_state(self) -> dict[str, Any]:
        return {
            "record": "eta-mu.github-crawler-state.v1",
            "schema_version": "github.crawler.state.v1",
            "last_sweep_ts": 0.0,
            "config_checked_ts": 0.0,
            "global_rate_limit_reset": 0.0,
            "repo_cooldowns": {},
            "repo_endpoint_cursor": {},
            "url_state": {},
            "resources": {},
            "trigger_edges": [],
            "cooldown_blocks": [],
            "recent_events": [],
            "metrics": {
                "scheduled": 0,
                "fetched": 0,
                "failed": 0,
                "cooldown_blocked": 0,
                "atoms_emitted": 0,
            },
        }

    def _load_state(self) -> dict[str, Any]:
        state = self._default_state()
        if not self.state_path.exists():
            return state
        try:
            payload = json.loads(self.state_path.read_text("utf-8"))
            if isinstance(payload, dict):
                state.update(payload)
        except Exception:
            return state

        for key, default_value in self._default_state().items():
            if key not in state:
                state[key] = default_value
        return state

    def _save_state(self) -> None:
        self._trim_state()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, sort_keys=True, indent=2),
            "utf-8",
        )

    def _trim_state(self) -> None:
        resources_obj = self.state.get("resources", {})
        if isinstance(resources_obj, dict) and len(resources_obj) > _MAX_RESOURCES:
            rows = [
                value for value in resources_obj.values() if isinstance(value, dict)
            ]
            rows.sort(
                key=lambda row: (
                    -_safe_float(row.get("fetched_ts", 0.0), 0.0),
                    _safe_str(row.get("id", "")),
                )
            )
            keep = rows[:_MAX_RESOURCES]
            self.state["resources"] = {
                _safe_str(row.get("id", "")): row
                for row in keep
                if _safe_str(row.get("id", ""))
            }

        for key in ("trigger_edges", "cooldown_blocks", "recent_events"):
            rows = self.state.get(key, [])
            if not isinstance(rows, list):
                self.state[key] = []
                continue
            limit = _MAX_TRACE_EDGES if key != "recent_events" else _MAX_EVENT_ROWS
            if len(rows) > limit:
                self.state[key] = rows[-limit:]

    def _append_event(self, event_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = {
            "record": "eta-mu.github-crawler-event.v1",
            "schema_version": "github.crawler.event.v1",
            "ts": _now_iso(),
            "event": _safe_str(event_name),
            **payload,
        }
        row.setdefault("id", _hash_event_id(row))

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        events = self.state.get("recent_events", [])
        if not isinstance(events, list):
            events = []
        events.append(row)
        if len(events) > _MAX_EVENT_ROWS:
            events = events[-_MAX_EVENT_ROWS:]
        self.state["recent_events"] = events
        return row

    def _url_state_row(self, canonical_url: str) -> dict[str, Any]:
        url_state = self.state.get("url_state", {})
        if not isinstance(url_state, dict):
            url_state = {}
            self.state["url_state"] = url_state
        row = url_state.get(canonical_url)
        if not isinstance(row, dict):
            row = {
                "canonical_url": canonical_url,
                "next_allowed_fetch_ts": 0.0,
                "last_fetch_ts": 0.0,
                "fail_count": 0,
                "last_status": "",
                "source_hint": "github",
            }
            url_state[canonical_url] = row
        return row

    def _record_trace_edge(
        self, kind: str, source_url: str, target_url: str, ts_epoch: float
    ) -> None:
        source_id = _url_id(source_url)
        target_id = _url_id(target_url)
        if not source_id or not target_id:
            return
        bucket_name = (
            "trigger_edges" if kind == "crawl:triggered_fetch" else "cooldown_blocks"
        )
        bucket = self.state.get(bucket_name, [])
        if not isinstance(bucket, list):
            bucket = []
        row = {
            "id": "edge:"
            + hashlib.sha1(
                f"{kind}|{source_id}|{target_id}|{int(ts_epoch)}".encode("utf-8")
            ).hexdigest()[:16],
            "kind": kind,
            "source": source_id,
            "target": target_id,
            "ts": round(max(0.0, float(ts_epoch)), 6),
        }
        bucket.append(row)
        if len(bucket) > _MAX_TRACE_EDGES:
            bucket = bucket[-_MAX_TRACE_EDGES:]
        self.state[bucket_name] = bucket

    def _schedule_fetch(
        self,
        *,
        source_url: str,
        target_url: str,
        repo: str,
        now_ts: float,
    ) -> tuple[bool, str, float]:
        canonical_target = canonical_github_url(target_url)
        if not canonical_target:
            return False, "", 0.0

        row = self._url_state_row(canonical_target)
        next_allowed = max(0.0, _safe_float(row.get("next_allowed_fetch_ts", 0.0), 0.0))
        if now_ts < next_allowed:
            self.state.setdefault("metrics", {}).setdefault("cooldown_blocked", 0)
            self.state["metrics"]["cooldown_blocked"] += 1
            self._record_trace_edge(
                "crawl:cooldown_block",
                source_url,
                canonical_target,
                now_ts,
            )
            self._append_event(
                "github_fetch_scheduled",
                {
                    "repo": repo,
                    "url": target_url,
                    "canonical_url": canonical_target,
                    "scheduled": False,
                    "reason": "cooldown_active",
                    "next_allowed_fetch_ts": round(next_allowed, 6),
                },
            )
            return False, canonical_target, next_allowed

        self.state.setdefault("metrics", {}).setdefault("scheduled", 0)
        self.state["metrics"]["scheduled"] += 1
        self._record_trace_edge(
            "crawl:triggered_fetch",
            source_url,
            canonical_target,
            now_ts,
        )
        self._append_event(
            "github_fetch_scheduled",
            {
                "repo": repo,
                "url": target_url,
                "canonical_url": canonical_target,
                "scheduled": True,
                "reason": "ready",
            },
        )
        return True, canonical_target, 0.0

    def _set_fetch_success(self, canonical_url: str, *, now_ts: float) -> None:
        cooldown_s = max(
            30.0, _safe_float(self.config.get("url_cooldown_s", 1800), 1800.0)
        )
        row = self._url_state_row(canonical_url)
        row["last_fetch_ts"] = round(max(0.0, now_ts), 6)
        row["last_status"] = "ok"
        row["fail_count"] = 0
        row["next_allowed_fetch_ts"] = round(max(0.0, now_ts + cooldown_s), 6)
        row["source_hint"] = "github"

    def _set_fetch_failure(
        self,
        canonical_url: str,
        *,
        now_ts: float,
        status: int,
    ) -> float:
        row = self._url_state_row(canonical_url)
        fail_count = max(0, _safe_int(row.get("fail_count", 0), 0)) + 1
        row["fail_count"] = fail_count
        row["last_status"] = f"http_{status}" if status > 0 else "error"
        row["last_fetch_ts"] = round(max(0.0, now_ts), 6)

        base_backoff = 60.0 if status in {400, 404} else 300.0
        if status in {403, 429}:
            base_backoff = 900.0
        backoff = min(6.0 * 3600.0, base_backoff * (2 ** max(0, fail_count - 1)))
        row["next_allowed_fetch_ts"] = round(max(0.0, now_ts + backoff), 6)
        return float(backoff)

    def _endpoint_order(self) -> list[str]:
        order = self.config.get("endpoint_order", ["pulls", "issues", "releases"])
        if not isinstance(order, list):
            return ["pulls", "issues", "releases"]
        cleaned = []
        seen: set[str] = set()
        for row in order:
            token = _safe_str(row).lower()
            if token not in {"pulls", "issues", "releases"} or token in seen:
                continue
            seen.add(token)
            cleaned.append(token)
        return cleaned or ["pulls", "issues", "releases"]

    def _next_endpoint_kind(self, repo: str) -> str:
        cursors = self.state.get("repo_endpoint_cursor", {})
        if not isinstance(cursors, dict):
            cursors = {}
            self.state["repo_endpoint_cursor"] = cursors
        index = max(0, _safe_int(cursors.get(repo, 0), 0))
        order = self._endpoint_order()
        kind = order[index % len(order)]
        cursors[repo] = index + 1
        return kind

    def _endpoint_url(self, repo: str, endpoint_kind: str) -> str:
        per_page = max(
            1, min(50, _safe_int(self.config.get("max_items_per_repo", 5), 5))
        )
        if endpoint_kind == "issues":
            return (
                f"https://api.github.com/repos/{repo}/issues"
                f"?state=all&sort=updated&direction=desc&per_page={per_page}"
            )
        if endpoint_kind == "releases":
            return f"https://api.github.com/repos/{repo}/releases?per_page={per_page}"
        return (
            f"https://api.github.com/repos/{repo}/pulls"
            f"?state=all&sort=updated&direction=desc&per_page={per_page}"
        )

    def _extract_candidates(
        self,
        repo: str,
        endpoint_kind: str,
        payload_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for row in payload_rows:
            if not isinstance(row, dict):
                continue

            if endpoint_kind == "issues" and isinstance(row.get("pull_request"), dict):
                # GitHub issues endpoint includes PRs; skip in issue sweep.
                continue

            canonical = canonical_github_url(_safe_str(row.get("html_url", "")))
            api_url = _safe_str(row.get("url", ""))
            if not canonical or not api_url:
                continue

            if endpoint_kind == "pulls":
                kind = "github:pr"
            elif endpoint_kind == "issues":
                kind = "github:issue"
            else:
                kind = "github:release"

            atoms = extract_github_atoms(canonical, row, self.config)
            score = compute_importance_score(row, atoms)
            candidates.append(
                {
                    "repo": repo,
                    "kind": kind,
                    "canonical_url": canonical,
                    "api_url": api_url,
                    "number": max(0, _safe_int(row.get("number", 0), 0)),
                    "title": _safe_str(row.get("title", row.get("name", "")))[:240],
                    "updated_at": _safe_str(
                        row.get("updated_at", row.get("published_at", ""))
                    ),
                    "labels": [
                        _safe_str(lbl.get("name", ""))
                        for lbl in row.get("labels", [])
                        if isinstance(lbl, dict) and _safe_str(lbl.get("name", ""))
                    ],
                    "authors": [
                        _safe_str(
                            (
                                row.get("user", {})
                                if isinstance(row.get("user", {}), dict)
                                else {}
                            ).get("login", "")
                        )
                    ],
                    "importance_score": int(score),
                    "atoms": atoms,
                }
            )

        deduped: dict[str, dict[str, Any]] = {}
        for row in candidates:
            key = _safe_str(row.get("canonical_url", ""))
            if not key:
                continue
            existing = deduped.get(key)
            if existing is None or _safe_int(
                row.get("importance_score", 0), 0
            ) > _safe_int(existing.get("importance_score", 0), 0):
                deduped[key] = row

        rows = list(deduped.values())
        rows.sort(
            key=lambda item: (
                -_safe_int(item.get("importance_score", 0), 0),
                _safe_str(item.get("updated_at", "")),
                _safe_str(item.get("canonical_url", "")),
            )
        )
        max_items = max(1, _safe_int(self.config.get("max_items_per_repo", 5), 5))
        return rows[:max_items]

    def _discover_links(self, payload: dict[str, Any]) -> list[str]:
        links: list[str] = []
        text = "\n".join(
            [
                _safe_str(payload.get("body", "")),
                _safe_str(payload.get("title", "")),
            ]
        )
        for match in set(re.findall(r"https?://[^\s\]\)\"'>]+", text)):
            canonical = canonical_github_url(match)
            if canonical:
                links.append(canonical)
        links.sort()
        return links[:64]

    def _conversation_context_enabled(self) -> bool:
        return _safe_bool(self.config.get("include_conversation_context", True), True)

    def _pr_commit_context_enabled(self) -> bool:
        return _safe_bool(self.config.get("include_pr_commit_context", True), True)

    def _max_conversation_rows(self) -> int:
        configured = _safe_int(
            self.config.get("max_conversation_rows", _MAX_CONVERSATION_ROWS),
            _MAX_CONVERSATION_ROWS,
        )
        return max(8, min(_MAX_CONVERSATION_ROWS, configured))

    def _max_conversation_body_chars(self) -> int:
        configured = _safe_int(
            self.config.get(
                "max_conversation_body_chars", _MAX_CONVERSATION_BODY_CHARS
            ),
            _MAX_CONVERSATION_BODY_CHARS,
        )
        return max(280, min(_MAX_CONVERSATION_BODY_CHARS, configured))

    def _conversation_rows_from_payload(
        self,
        payload_rows: Any,
        *,
        channel: str,
        max_rows: int,
        max_body_chars: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not isinstance(payload_rows, list):
            return rows

        for item in payload_rows:
            if not isinstance(item, dict):
                continue
            body = _bounded_text(item.get("body", ""), max_body_chars)
            if not body:
                continue
            user_row = item.get("user", {})
            author = (
                _safe_str(user_row.get("login", ""))
                if isinstance(user_row, dict)
                else ""
            )
            created_at = _safe_str(item.get("created_at", item.get("submitted_at", "")))
            rows.append(
                {
                    "channel": channel,
                    "author": author,
                    "created_at": created_at,
                    "body": body,
                }
            )
            if len(rows) >= max_rows:
                break
        return rows[:max_rows]

    def _fetch_structured_rows(
        self,
        *,
        repo: str,
        source_url: str,
        target_url: str,
        now_ts: float,
        fetch_budget: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if fetch_budget <= 0:
            return [], fetch_budget
        ok, _, result = self._fetch_with_schedule(
            repo=repo,
            source_url=source_url,
            target_url=target_url,
            now_ts=now_ts,
        )
        fetch_budget -= 1
        if not ok or not isinstance(result, dict):
            return [], fetch_budget
        payload_rows = result.get("payload", [])
        if not isinstance(payload_rows, list):
            return [], fetch_budget
        return [row for row in payload_rows if isinstance(row, dict)], fetch_budget

    def _resource_text_context(
        self,
        payload: dict[str, Any],
        *,
        conversation_rows: list[dict[str, Any]],
        commit_rows: list[dict[str, Any]],
        diff_keyword_hits: list[dict[str, Any]],
    ) -> tuple[str, str, str]:
        title = _bounded_text(payload.get("title", payload.get("name", "")), 240)
        body = _bounded_text(payload.get("body", ""), _MAX_CONVERSATION_TEXT_CHARS)

        markdown_lines: list[str] = []
        if title:
            markdown_lines.append(f"# {title}")
        if body:
            markdown_lines.append("")
            markdown_lines.append("## Root Body")
            markdown_lines.append(body)

        if conversation_rows:
            markdown_lines.append("")
            markdown_lines.append("## Conversation")
            for index, row in enumerate(conversation_rows, start=1):
                channel = _safe_str(row.get("channel", "comment"))
                author = _safe_str(row.get("author", "unknown")) or "unknown"
                created_at = _safe_str(row.get("created_at", ""))
                stamp = f" @ {created_at}" if created_at else ""
                markdown_lines.append("")
                markdown_lines.append(f"### {index}. {channel} by {author}{stamp}")
                markdown_lines.append(
                    _bounded_text(row.get("body", ""), _MAX_CONVERSATION_BODY_CHARS)
                )

        if commit_rows:
            markdown_lines.append("")
            markdown_lines.append("## Commits")
            for row in commit_rows:
                sha = _safe_str(row.get("sha", ""))
                short_sha = sha[:12] if sha else "unknown"
                author = _safe_str(row.get("author", "")) or "unknown"
                message = _safe_str(row.get("message", ""))
                if message:
                    markdown_lines.append(f"- `{short_sha}` {author}: {message}")

        if diff_keyword_hits:
            markdown_lines.append("")
            markdown_lines.append("## Diff Signals")
            for row in diff_keyword_hits[:24]:
                if not isinstance(row, dict):
                    continue
                file_name = _safe_str(row.get("file", ""))
                term = _safe_str(row.get("term", ""))
                if file_name and term:
                    markdown_lines.append(f"- `{file_name}` matched `{term}`")

        conversation_markdown = _bounded_text(
            "\n".join(markdown_lines).strip(), _MAX_CONVERSATION_TEXT_CHARS
        )

        summary_parts = [title]
        if body:
            summary_parts.append(body.splitlines()[0])
        if conversation_rows:
            summary_parts.append(f"{len(conversation_rows)} discussion entries")
        if commit_rows:
            summary_parts.append(f"{len(commit_rows)} commit entries")
        summary = _bounded_text(
            " | ".join(part for part in summary_parts if part), _MAX_SUMMARY_CHARS
        )

        excerpt_parts: list[str] = []
        if title:
            excerpt_parts.append(title)
        if body:
            excerpt_parts.append(body)
        for row in conversation_rows[:48]:
            excerpt_parts.append(_safe_str(row.get("body", "")))
        for row in commit_rows[:32]:
            excerpt_parts.append(_safe_str(row.get("message", "")))
        for row in diff_keyword_hits[:24]:
            if isinstance(row, dict):
                excerpt_parts.append(
                    f"{_safe_str(row.get('file', ''))} {_safe_str(row.get('term', ''))}".strip()
                )
        text_excerpt = _bounded_text(
            "\n".join(part for part in excerpt_parts if part), _MAX_EXCERPT_CHARS
        )
        return conversation_markdown, summary, text_excerpt

    def _fetch_item_context(
        self,
        *,
        repo: str,
        kind: str,
        payload: dict[str, Any],
        source_url: str,
        now_ts: float,
        fetch_budget: int,
    ) -> tuple[dict[str, Any], int]:
        if fetch_budget <= 0:
            return {
                "conversation_rows": [],
                "commit_rows": [],
            }, fetch_budget

        if not self._conversation_context_enabled():
            return {
                "conversation_rows": [],
                "commit_rows": [],
            }, fetch_budget

        number = _safe_int(payload.get("number", 0), 0)
        if number <= 0:
            return {
                "conversation_rows": [],
                "commit_rows": [],
            }, fetch_budget

        max_rows = self._max_conversation_rows()
        max_body_chars = self._max_conversation_body_chars()
        conversation_rows: list[dict[str, Any]] = []
        commit_rows: list[dict[str, Any]] = []

        issue_comments_url = (
            f"https://api.github.com/repos/{repo}/issues/{number}/comments?per_page=100"
        )
        issue_rows, fetch_budget = self._fetch_structured_rows(
            repo=repo,
            source_url=source_url,
            target_url=issue_comments_url,
            now_ts=now_ts,
            fetch_budget=fetch_budget,
        )
        if issue_rows:
            remaining = max(0, max_rows - len(conversation_rows))
            conversation_rows.extend(
                self._conversation_rows_from_payload(
                    issue_rows,
                    channel="issue-comment",
                    max_rows=remaining,
                    max_body_chars=max_body_chars,
                )
            )

        kind_token = _safe_str(kind)
        if kind_token == "github:pr":
            if fetch_budget > 0:
                review_rows, fetch_budget = self._fetch_structured_rows(
                    repo=repo,
                    source_url=source_url,
                    target_url=f"https://api.github.com/repos/{repo}/pulls/{number}/reviews?per_page=100",
                    now_ts=now_ts,
                    fetch_budget=fetch_budget,
                )
                if review_rows:
                    remaining = max(0, max_rows - len(conversation_rows))
                    conversation_rows.extend(
                        self._conversation_rows_from_payload(
                            review_rows,
                            channel="review",
                            max_rows=remaining,
                            max_body_chars=max_body_chars,
                        )
                    )

            if fetch_budget > 0:
                review_comment_rows, fetch_budget = self._fetch_structured_rows(
                    repo=repo,
                    source_url=source_url,
                    target_url=f"https://api.github.com/repos/{repo}/pulls/{number}/comments?per_page=100",
                    now_ts=now_ts,
                    fetch_budget=fetch_budget,
                )
                if review_comment_rows:
                    remaining = max(0, max_rows - len(conversation_rows))
                    conversation_rows.extend(
                        self._conversation_rows_from_payload(
                            review_comment_rows,
                            channel="review-comment",
                            max_rows=remaining,
                            max_body_chars=max_body_chars,
                        )
                    )

            if fetch_budget > 0 and self._pr_commit_context_enabled():
                commit_payload_rows, fetch_budget = self._fetch_structured_rows(
                    repo=repo,
                    source_url=source_url,
                    target_url=f"https://api.github.com/repos/{repo}/pulls/{number}/commits?per_page=100",
                    now_ts=now_ts,
                    fetch_budget=fetch_budget,
                )
                for row in commit_payload_rows[:64]:
                    sha = _safe_str(row.get("sha", ""))
                    commit = row.get("commit", {})
                    if not isinstance(commit, dict):
                        commit = {}
                    message = _bounded_text(
                        commit.get("message", ""),
                        max_body_chars,
                    )
                    author_obj = commit.get("author", {})
                    author = (
                        _safe_str(author_obj.get("name", ""))
                        if isinstance(author_obj, dict)
                        else ""
                    )
                    if not message:
                        continue
                    commit_rows.append(
                        {
                            "sha": sha,
                            "author": author,
                            "message": message,
                        }
                    )

        deduped_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in conversation_rows:
            key = "|".join(
                [
                    _safe_str(row.get("channel", "")),
                    _safe_str(row.get("author", "")),
                    _safe_str(row.get("created_at", "")),
                    _safe_str(row.get("body", ""))[:128],
                ]
            )
            if not key or key in seen:
                continue
            seen.add(key)
            deduped_rows.append(row)
            if len(deduped_rows) >= max_rows:
                break
        deduped_rows.sort(
            key=lambda row: (
                _safe_str(row.get("created_at", "")),
                _safe_str(row.get("channel", "")),
                _safe_str(row.get("author", "")),
            )
        )

        return {
            "conversation_rows": deduped_rows,
            "commit_rows": commit_rows[:64],
        }, fetch_budget

    def _store_resource(
        self,
        *,
        repo: str,
        source_canonical_url: str,
        fetch_result: dict[str, Any],
        payload: dict[str, Any],
        kind_hint: str,
        atoms: list[dict[str, Any]],
        filenames_touched: list[str],
        diff_keyword_hits: list[dict[str, Any]],
        conversation_context: dict[str, Any] | None,
        now_ts: float,
    ) -> dict[str, Any] | None:
        canonical_url = _safe_str(fetch_result.get("canonical_url", ""))
        if not canonical_url:
            return None

        content_hash = _safe_str(
            (
                fetch_result.get("resource", {})
                if isinstance(fetch_result.get("resource", {}), dict)
                else {}
            ).get("content_hash", "")
        )
        res_id = _resource_id(canonical_url, content_hash)
        source_url_id = _url_id(source_canonical_url)
        if not res_id or not source_url_id:
            return None

        score = compute_importance_score(
            payload, atoms, touched_files=filenames_touched
        )
        title = _safe_str(payload.get("title", payload.get("name", "")))[:240]
        updated_at = _safe_str(
            payload.get("updated_at", payload.get("published_at", ""))
        )
        labels = [
            _safe_str(lbl.get("name", ""))
            for lbl in payload.get("labels", [])
            if isinstance(lbl, dict) and _safe_str(lbl.get("name", ""))
        ]
        authors = []
        user_row = payload.get("user", {})
        if isinstance(user_row, dict):
            login = _safe_str(user_row.get("login", ""))
            if login:
                authors.append(login)

        links_to = self._discover_links(payload)
        for link in links_to:
            self._url_state_row(link)

        context = (
            dict(conversation_context)
            if isinstance(conversation_context, dict)
            else {"conversation_rows": [], "commit_rows": []}
        )
        conversation_rows = [
            row for row in context.get("conversation_rows", []) if isinstance(row, dict)
        ]
        commit_rows = [
            row for row in context.get("commit_rows", []) if isinstance(row, dict)
        ]
        conversation_markdown, context_summary, context_excerpt = (
            self._resource_text_context(
                payload,
                conversation_rows=conversation_rows,
                commit_rows=commit_rows,
                diff_keyword_hits=diff_keyword_hits,
            )
        )
        context_hash = (
            hashlib.sha256(context_excerpt.encode("utf-8")).hexdigest()
            if context_excerpt
            else ""
        )

        row = {
            "id": res_id,
            "canonical_url": canonical_url,
            "fetched_ts": round(max(0.0, now_ts), 6),
            "content_hash": content_hash,
            "title": title,
            "kind": _safe_str(kind_hint),
            "repo": repo,
            "number": max(0, _safe_int(payload.get("number", 0), 0)),
            "labels": labels[:24],
            "authors": authors[:8],
            "updated_at": updated_at,
            "importance_score": int(score),
            "source_url_id": source_url_id,
            "source_canonical_url": source_canonical_url,
            "api_endpoint": _safe_str(fetch_result.get("url", "")),
            "atoms": atoms[:50],
            "filenames_touched": filenames_touched[:200],
            "diff_keyword_hits": diff_keyword_hits[:24],
            "state": _safe_str(payload.get("state", "")),
            "merged_at": _safe_str(payload.get("merged_at", "")),
            "links_to": links_to,
            "summary": context_summary,
            "text_excerpt": context_excerpt,
            "text_excerpt_hash": context_hash
            or _safe_str(
                (
                    fetch_result.get("resource", {})
                    if isinstance(fetch_result.get("resource", {}), dict)
                    else {}
                ).get("text_excerpt_hash", "")
            ),
            "conversation_markdown": conversation_markdown,
            "conversation_comment_count": len(conversation_rows),
            "conversation_rows": conversation_rows[
                : max(8, self._max_conversation_rows())
            ],
            "commit_count": len(commit_rows),
            "commit_rows": commit_rows[:64],
        }

        resources = self.state.get("resources", {})
        if not isinstance(resources, dict):
            resources = {}
        resources[res_id] = row
        self.state["resources"] = resources
        self._trim_state()
        return row

    def _fetch_with_schedule(
        self,
        *,
        repo: str,
        source_url: str,
        target_url: str,
        now_ts: float,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        should_fetch, canonical_target, _ = self._schedule_fetch(
            source_url=source_url,
            target_url=target_url,
            repo=repo,
            now_ts=now_ts,
        )
        if not should_fetch:
            return False, canonical_target, None

        self._append_event(
            "github_fetch_started",
            {
                "repo": repo,
                "url": target_url,
                "canonical_url": canonical_target,
            },
        )

        result = self._fetcher(target_url, self.config)
        if not isinstance(result, dict):
            result = {
                "ok": False,
                "error": "invalid_fetcher_result",
                "status": 0,
                "canonical_url": canonical_target,
                "duration_ms": 0,
            }

        if bool(result.get("ok", False)):
            self.state.setdefault("metrics", {}).setdefault("fetched", 0)
            self.state["metrics"]["fetched"] += 1
            self._set_fetch_success(canonical_target, now_ts=now_ts)
            remaining = _safe_int(result.get("rate_limit_remaining", 5000), 5000)
            if remaining < 5:
                reset_at = max(
                    now_ts + 60.0,
                    _safe_float(result.get("rate_limit_reset", 0), 0.0),
                )
                self.state["global_rate_limit_reset"] = round(reset_at, 6)
            self._append_event(
                "github_fetch_completed",
                {
                    "repo": repo,
                    "url": target_url,
                    "canonical_url": canonical_target,
                    "status": _safe_int(result.get("status", 200), 200),
                    "duration_ms": _safe_int(result.get("duration_ms", 0), 0),
                    "content_hash": _safe_str(
                        (
                            result.get("resource", {})
                            if isinstance(result.get("resource", {}), dict)
                            else {}
                        ).get("content_hash", "")
                    ),
                },
            )
            return True, canonical_target, result

        self.state.setdefault("metrics", {}).setdefault("failed", 0)
        self.state["metrics"]["failed"] += 1
        status = _safe_int(result.get("status", 0), 0)
        backoff = self._set_fetch_failure(
            canonical_target, now_ts=now_ts, status=status
        )
        if status in {403, 429}:
            reset_at = max(
                now_ts + backoff,
                _safe_float(result.get("rate_limit_reset", 0), 0.0),
            )
            self.state["global_rate_limit_reset"] = round(reset_at, 6)

        self._append_event(
            "github_fetch_failed",
            {
                "repo": repo,
                "url": target_url,
                "canonical_url": canonical_target,
                "status": status,
                "error": _safe_str(result.get("error", ""))[:200],
                "backoff_s": round(backoff, 3),
                "next_allowed_fetch_ts": round(now_ts + backoff, 6),
            },
        )
        return False, canonical_target, result

    def _fetch_pr_files(
        self,
        *,
        repo: str,
        pr_number: int,
        source_url: str,
        now_ts: float,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        if pr_number <= 0:
            return [], []

        files_url = (
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files?per_page=200"
        )
        ok, _, result = self._fetch_with_schedule(
            repo=repo,
            source_url=source_url,
            target_url=files_url,
            now_ts=now_ts,
        )
        if not ok or not isinstance(result, dict):
            return [], []

        payload_rows = result.get("payload", [])
        if not isinstance(payload_rows, list):
            return [], []

        filenames: list[str] = []
        for row in payload_rows[:200]:
            if not isinstance(row, dict):
                continue
            filename = _safe_str(row.get("filename", ""))
            if filename:
                filenames.append(filename)
        deduped: list[str] = []
        seen: set[str] = set()
        for row in filenames:
            key = row.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)

        keywords = [
            _safe_str(token).lower()
            for token in self.config.get("keywords", [])
            if _safe_str(token)
        ]
        diff_hits = extract_diff_keyword_hits(
            payload_rows, keywords=keywords, max_matches=24
        )
        return deduped[:200], diff_hits

    def _sweep_repo(
        self,
        repo: str,
        *,
        now_ts: float,
        fetch_budget: int,
    ) -> int:
        if fetch_budget <= 0:
            return fetch_budget

        endpoint_kind = self._next_endpoint_kind(repo)
        list_url = self._endpoint_url(repo, endpoint_kind)
        source_url = f"https://github.com/{repo}"

        ok, list_canonical, list_result = self._fetch_with_schedule(
            repo=repo,
            source_url=source_url,
            target_url=list_url,
            now_ts=now_ts,
        )
        fetch_budget -= 1
        if not ok or not isinstance(list_result, dict):
            return fetch_budget

        payload_rows = list_result.get("payload", [])
        if not isinstance(payload_rows, list):
            payload_rows = []
        payload_rows = [row for row in payload_rows if isinstance(row, dict)]

        candidates = self._extract_candidates(repo, endpoint_kind, payload_rows)
        for candidate in candidates:
            if fetch_budget <= 0:
                break
            api_url = _safe_str(candidate.get("api_url", ""))
            if not api_url:
                continue

            ok_item, item_canonical, item_result = self._fetch_with_schedule(
                repo=repo,
                source_url=list_canonical,
                target_url=api_url,
                now_ts=now_ts,
            )
            fetch_budget -= 1
            if not ok_item or not isinstance(item_result, dict):
                continue

            payload = item_result.get("payload", {})
            if not isinstance(payload, dict):
                continue

            filenames_touched: list[str] = []
            diff_keyword_hits: list[dict[str, Any]] = []
            if _safe_str(candidate.get("kind", "")) == "github:pr" and fetch_budget > 0:
                pr_number = _safe_int(
                    payload.get("number", candidate.get("number", 0)), 0
                )
                touched, hits = self._fetch_pr_files(
                    repo=repo,
                    pr_number=pr_number,
                    source_url=item_canonical,
                    now_ts=now_ts,
                )
                fetch_budget -= 1
                if touched or hits:
                    filenames_touched = touched
                    diff_keyword_hits = hits
                    payload = dict(payload)
                    payload["filenames_touched"] = list(filenames_touched)
                    payload["diff_keyword_hits"] = list(diff_keyword_hits)

            conversation_context, fetch_budget = self._fetch_item_context(
                repo=repo,
                kind=_safe_str(candidate.get("kind", "github:repo")),
                payload=payload,
                source_url=item_canonical,
                now_ts=now_ts,
                fetch_budget=fetch_budget,
            )

            atoms = extract_github_atoms(item_canonical, payload, self.config)
            resource_row = self._store_resource(
                repo=repo,
                source_canonical_url=list_canonical,
                fetch_result=item_result,
                payload=payload,
                kind_hint=_safe_str(candidate.get("kind", "github:repo")),
                atoms=atoms,
                filenames_touched=filenames_touched,
                diff_keyword_hits=diff_keyword_hits,
                conversation_context=conversation_context,
                now_ts=now_ts,
            )

            if resource_row is not None:
                self.state.setdefault("metrics", {}).setdefault("atoms_emitted", 0)
                self.state["metrics"]["atoms_emitted"] += len(atoms)
                self._append_event(
                    "github_extract_atoms",
                    {
                        "repo": repo,
                        "canonical_url": _safe_str(
                            resource_row.get("canonical_url", "")
                        ),
                        "res_id": _safe_str(resource_row.get("id", "")),
                        "atoms_count": len(atoms),
                        "top_atoms": atoms[:8],
                    },
                )

        repo_cooldown_s = max(
            10.0, _safe_float(self.config.get("repo_cooldown_s", 300), 300.0)
        )
        repo_cooldowns = self.state.get("repo_cooldowns", {})
        if not isinstance(repo_cooldowns, dict):
            repo_cooldowns = {}
            self.state["repo_cooldowns"] = repo_cooldowns
        repo_cooldowns[repo] = round(now_ts + repo_cooldown_s, 6)
        return fetch_budget

    def tick(
        self,
        *,
        slack_ms: float = 100.0,
        now_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        if _safe_float(slack_ms, 0.0) < 0.0:
            return []

        now_value = max(0.0, _safe_float(now_ts, time.time()))

        config_check_age = now_value - _safe_float(
            self.state.get("config_checked_ts", 0.0), 0.0
        )
        if config_check_age >= 300.0:
            self.config = self._load_config()
            self.state["config_checked_ts"] = round(now_value, 6)

        if not bool(self.config.get("enabled", True)):
            self._save_state()
            self.graph_path.write_text(
                json.dumps(
                    self.graph_snapshot(), ensure_ascii=False, sort_keys=True, indent=2
                ),
                "utf-8",
            )
            return []

        if now_value < _safe_float(self.state.get("global_rate_limit_reset", 0.0), 0.0):
            self._save_state()
            self.graph_path.write_text(
                json.dumps(
                    self.graph_snapshot(), ensure_ascii=False, sort_keys=True, indent=2
                ),
                "utf-8",
            )
            return []

        repos = [
            _canonical_repo_name(row)
            for row in self.config.get("repos", [])
            if _canonical_repo_name(row)
        ]
        if not repos:
            self._save_state()
            self.graph_path.write_text(
                json.dumps(
                    self.graph_snapshot(), ensure_ascii=False, sort_keys=True, indent=2
                ),
                "utf-8",
            )
            return []

        max_repos = max(1, _safe_int(self.config.get("max_repos", 8), 8))
        fetch_budget = max(
            1, _safe_int(self.config.get("max_github_fetches_per_tick", 2), 2)
        )

        repo_cooldowns = self.state.get("repo_cooldowns", {})
        if not isinstance(repo_cooldowns, dict):
            repo_cooldowns = {}
            self.state["repo_cooldowns"] = repo_cooldowns

        touched_rows: list[dict[str, Any]] = []
        for repo in sorted(repos)[:max_repos]:
            if fetch_budget <= 0:
                break
            if now_value < _safe_float(repo_cooldowns.get(repo, 0.0), 0.0):
                continue
            before = fetch_budget
            fetch_budget = self._sweep_repo(
                repo, now_ts=now_value, fetch_budget=fetch_budget
            )
            if fetch_budget != before:
                touched_rows.append({"repo": repo, "consumed": before - fetch_budget})

        self.state["last_sweep_ts"] = round(now_value, 6)
        self._save_state()

        graph_payload = self.graph_snapshot()
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph_path.write_text(
            json.dumps(graph_payload, ensure_ascii=False, sort_keys=True, indent=2),
            "utf-8",
        )
        return touched_rows

    def _url_nodes_for_graph(self) -> list[dict[str, Any]]:
        url_state = self.state.get("url_state", {})
        if not isinstance(url_state, dict):
            return []

        nodes: list[dict[str, Any]] = []
        for canonical_url, row in sorted(
            url_state.items(), key=lambda item: str(item[0])
        ):
            if not isinstance(row, dict):
                continue
            url_id = _url_id(canonical_url)
            if not url_id:
                continue
            x, y = _position_from_token(canonical_url)
            nodes.append(
                {
                    "id": url_id,
                    "node_id": url_id,
                    "node_type": "web:url",
                    "crawler_kind": "url",
                    "web_node_role": "web:url",
                    "resource_kind": "link",
                    "modality": "web",
                    "label": canonical_url,
                    "x": x,
                    "y": y,
                    "hue": 212,
                    "importance": 0.58,
                    "url": canonical_url,
                    "canonical_url": canonical_url,
                    "next_allowed_fetch_ts": round(
                        max(
                            0.0, _safe_float(row.get("next_allowed_fetch_ts", 0.0), 0.0)
                        ),
                        6,
                    ),
                    "last_fetch_ts": round(
                        max(0.0, _safe_float(row.get("last_fetch_ts", 0.0), 0.0)),
                        6,
                    ),
                    "fail_count": max(0, _safe_int(row.get("fail_count", 0), 0)),
                    "last_status": _safe_str(row.get("last_status", "")),
                    "source_hint": "github",
                }
            )
        return nodes

    def _resource_nodes_for_graph(self) -> list[dict[str, Any]]:
        resources = self.state.get("resources", {})
        if not isinstance(resources, dict):
            return []

        rows = [row for row in resources.values() if isinstance(row, dict)]
        rows.sort(
            key=lambda row: (
                -_safe_float(row.get("fetched_ts", 0.0), 0.0),
                _safe_str(row.get("id", "")),
            )
        )

        nodes: list[dict[str, Any]] = []
        for row in rows:
            canonical_url = _safe_str(row.get("canonical_url", ""))
            res_id = _safe_str(row.get("id", ""))
            if not canonical_url or not res_id:
                continue
            x, y = _position_from_token(res_id)
            nodes.append(
                {
                    "id": res_id,
                    "node_id": res_id,
                    "node_type": "web:resource",
                    "crawler_kind": "resource",
                    "web_node_role": "web:resource",
                    "resource_kind": "text",
                    "modality": "web",
                    "label": _safe_str(row.get("title", "")) or canonical_url,
                    "x": x,
                    "y": y,
                    "hue": 24,
                    "importance": round(
                        max(
                            0.1,
                            min(
                                1.0,
                                0.2
                                + (_safe_int(row.get("importance_score", 0), 0) * 0.08),
                            ),
                        ),
                        4,
                    ),
                    "canonical_url": canonical_url,
                    "fetched_ts": round(
                        max(0.0, _safe_float(row.get("fetched_ts", 0.0), 0.0)),
                        6,
                    ),
                    "content_hash": _safe_str(row.get("content_hash", "")),
                    "text_excerpt_hash": _safe_str(row.get("text_excerpt_hash", "")),
                    "text_excerpt": _bounded_text(
                        row.get("text_excerpt", ""), _MAX_EXCERPT_CHARS
                    ),
                    "summary": _bounded_text(
                        row.get("summary", ""), _MAX_SUMMARY_CHARS
                    ),
                    "title": _safe_str(row.get("title", "")),
                    "source_url_id": _safe_str(row.get("source_url_id", "")),
                    "kind": _safe_str(row.get("kind", "github:repo")),
                    "repo": _safe_str(row.get("repo", "")),
                    "number": max(0, _safe_int(row.get("number", 0), 0)),
                    "labels": [
                        _safe_str(token)
                        for token in row.get("labels", [])
                        if _safe_str(token)
                    ][:24],
                    "authors": [
                        _safe_str(token)
                        for token in row.get("authors", [])
                        if _safe_str(token)
                    ][:12],
                    "updated_at": _safe_str(row.get("updated_at", "")),
                    "importance_score": max(
                        0, _safe_int(row.get("importance_score", 0), 0)
                    ),
                    "atoms": [
                        atom for atom in row.get("atoms", []) if isinstance(atom, dict)
                    ][:50],
                    "filenames_touched": [
                        _safe_str(path)
                        for path in row.get("filenames_touched", [])
                        if _safe_str(path)
                    ][:200],
                    "diff_keyword_hits": [
                        hit
                        for hit in row.get("diff_keyword_hits", [])
                        if isinstance(hit, dict)
                    ][:24],
                    "conversation_markdown": _bounded_text(
                        row.get("conversation_markdown", ""),
                        _MAX_CONVERSATION_TEXT_CHARS,
                    ),
                    "conversation_comment_count": max(
                        0,
                        _safe_int(row.get("conversation_comment_count", 0), 0),
                    ),
                    "conversation_rows": [
                        item
                        for item in row.get("conversation_rows", [])
                        if isinstance(item, dict)
                    ][: max(8, self._max_conversation_rows())],
                    "commit_count": max(0, _safe_int(row.get("commit_count", 0), 0)),
                    "commit_rows": [
                        item
                        for item in row.get("commit_rows", [])
                        if isinstance(item, dict)
                    ][:64],
                    "links_to": [
                        _safe_str(link)
                        for link in row.get("links_to", [])
                        if _safe_str(link)
                    ][:64],
                    "api_endpoint": _safe_str(row.get("api_endpoint", "")),
                    "state": _safe_str(row.get("state", "")),
                    "merged_at": _safe_str(row.get("merged_at", "")),
                }
            )
        return nodes

    def _edges_for_graph(
        self, resource_nodes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        def _append_edge(
            source_id: str, target_id: str, kind: str, weight: float = 1.0
        ) -> None:
            key = (source_id, target_id, kind)
            if key in seen or not source_id or not target_id or source_id == target_id:
                return
            seen.add(key)
            edge_id = (
                "crawl-link:"
                + hashlib.sha1(
                    f"{source_id}|{target_id}|{kind}".encode("utf-8")
                ).hexdigest()[:14]
            )
            edges.append(
                {
                    "id": edge_id,
                    "source": source_id,
                    "target": target_id,
                    "field": "",
                    "weight": round(max(0.0, min(1.0, weight)), 4),
                    "kind": kind,
                }
            )

        for row in resource_nodes:
            if not isinstance(row, dict):
                continue
            res_id = _safe_str(row.get("id", ""))
            source_url_id = _safe_str(row.get("source_url_id", ""))
            if res_id and source_url_id:
                _append_edge(res_id, source_url_id, "web:source_of", 1.0)
            for link in (
                row.get("links_to", [])
                if isinstance(row.get("links_to", []), list)
                else []
            ):
                target_id = _url_id(_safe_str(link))
                if res_id and target_id:
                    _append_edge(res_id, target_id, "web:links_to", 0.42)

        for bucket_name in ("trigger_edges", "cooldown_blocks"):
            for row in (
                self.state.get(bucket_name, [])
                if isinstance(self.state.get(bucket_name, []), list)
                else []
            ):
                if not isinstance(row, dict):
                    continue
                _append_edge(
                    _safe_str(row.get("source", "")),
                    _safe_str(row.get("target", "")),
                    _safe_str(row.get("kind", "")) or "crawl:triggered_fetch",
                    0.32,
                )
        return edges

    def graph_snapshot(self) -> dict[str, Any]:
        url_nodes = self._url_nodes_for_graph()
        resource_nodes = self._resource_nodes_for_graph()
        crawler_nodes = [*url_nodes, *resource_nodes]
        edges = self._edges_for_graph(resource_nodes)

        kind_counts: dict[str, int] = {}
        resource_kind_counts: dict[str, int] = {}
        web_role_counts: dict[str, int] = {}
        for row in crawler_nodes:
            kind = _safe_str(row.get("crawler_kind", "")) or "unknown"
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            resource_kind = _safe_str(row.get("resource_kind", "")) or "unknown"
            resource_kind_counts[resource_kind] = (
                resource_kind_counts.get(resource_kind, 0) + 1
            )
            web_role = _safe_str(row.get("web_node_role", ""))
            if web_role:
                web_role_counts[web_role] = web_role_counts.get(web_role, 0) + 1

        web_edge_kind_counts: dict[str, int] = {}
        for edge in edges:
            kind = _safe_str(edge.get("kind", "")).lower()
            if kind.startswith("web:") or kind.startswith("crawl:"):
                web_edge_kind_counts[kind] = web_edge_kind_counts.get(kind, 0) + 1

        events = [
            row for row in self.state.get("recent_events", []) if isinstance(row, dict)
        ]
        events.sort(
            key=lambda row: (
                _safe_str(row.get("ts", "")),
                _safe_str(row.get("event", "")),
                _safe_str(row.get("id", "")),
            )
        )

        monitored_repos = sorted(
            {
                _canonical_repo_name(row)
                for row in self.config.get("repos", [])
                if _canonical_repo_name(row)
            }
        )

        status = {
            "queue_length": 0,
            "pending_count": 0,
            "active_fetches": 0,
            "cooldown_blocks": max(
                0,
                _safe_int(
                    (
                        self.state.get("metrics", {})
                        if isinstance(self.state.get("metrics", {}), dict)
                        else {}
                    ).get("cooldown_blocked", 0),
                    0,
                ),
            ),
            "last_sweep_ts": round(
                max(0.0, _safe_float(self.state.get("last_sweep_ts", 0.0), 0.0)),
                6,
            ),
            "monitored_repos": monitored_repos,
            "global_rate_limit_reset": round(
                max(
                    0.0,
                    _safe_float(self.state.get("global_rate_limit_reset", 0.0), 0.0),
                ),
                6,
            ),
            "enabled": bool(self.config.get("enabled", True)),
        }

        return {
            "record": "eta-mu.github-crawler-graph.v1",
            "schema_version": "crawler.github-graph.v1",
            "generated_at": _now_iso(),
            "source": {
                "service": "github-presence",
                "endpoint": "https://api.github.com",
            },
            "status": status,
            "events": events[-200:],
            "nodes": crawler_nodes,
            "field_nodes": [],
            "crawler_nodes": crawler_nodes,
            "edges": edges,
            "stats": {
                "field_count": 0,
                "crawler_count": len(crawler_nodes),
                "edge_count": len(edges),
                "kind_counts": kind_counts,
                "resource_kind_counts": resource_kind_counts,
                "field_counts": {},
                "web_role_counts": web_role_counts,
                "web_edge_kind_counts": web_edge_kind_counts,
                "event_count": len(events[-200:]),
                "nodes_total": len(crawler_nodes),
                "edges_total": len(edges),
                "url_nodes_total": web_role_counts.get("web:url", 0),
                "github_resource_count": web_role_counts.get("web:resource", 0),
            },
        }


def run_github_presence_tick(
    part_root: Path,
    *,
    slack_ms: float | None = None,
    now_ts: float | None = None,
) -> dict[str, Any]:
    presence = GithubPresence(part_root)
    target_slack = _safe_float(
        slack_ms,
        _safe_float(os.getenv("GITHUB_PRESENCE_SLACK_MS", "120"), 120.0),
    )
    try:
        presence.tick(slack_ms=target_slack, now_ts=now_ts)
    except Exception as exc:
        _LOGGER.warning("github presence tick failed: %s", exc)
    return presence.graph_snapshot()


def merge_crawler_graph_with_github(
    base_crawler_graph: dict[str, Any] | None,
    github_graph: dict[str, Any] | None,
) -> dict[str, Any]:
    base = dict(base_crawler_graph) if isinstance(base_crawler_graph, dict) else {}
    github = dict(github_graph) if isinstance(github_graph, dict) else {}
    if not github:
        return base

    field_nodes = [row for row in base.get("field_nodes", []) if isinstance(row, dict)]
    crawler_nodes = [
        row for row in base.get("crawler_nodes", []) if isinstance(row, dict)
    ]
    crawler_nodes.sort(
        key=lambda row: (
            max(
                0,
                _safe_int(
                    row.get("depth", row.get("crawl_depth", 99)),
                    99,
                ),
            ),
            0
            if _safe_str(row.get("status", "")).strip().lower()
            in {"fetched", "duplicate", "ok"}
            else 1,
            len(_safe_str(row.get("canonical_url", row.get("url", "")))),
            _safe_str(row.get("id", "")),
        )
    )
    edges = [row for row in base.get("edges", []) if isinstance(row, dict)]
    events = [row for row in base.get("events", []) if isinstance(row, dict)]

    node_ids = {
        _safe_str(row.get("id", ""))
        for row in crawler_nodes
        if _safe_str(row.get("id", ""))
    }
    github_new_nodes: list[dict[str, Any]] = []
    for row in (
        github.get("crawler_nodes", [])
        if isinstance(github.get("crawler_nodes", []), list)
        else []
    ):
        if not isinstance(row, dict):
            continue
        node_id = _safe_str(row.get("id", ""))
        if not node_id or node_id in node_ids:
            continue
        node_ids.add(node_id)
        github_new_nodes.append(row)

    github_new_nodes.sort(
        key=lambda row: (
            0
            if _safe_str(row.get("web_node_role", "")).strip().lower() == "web:resource"
            else 1,
            -_safe_float(row.get("fetched_ts", 0.0), 0.0),
            _safe_str(row.get("id", "")),
        )
    )
    github_priority_limit = max(24, min(160, len(github_new_nodes)))
    github_priority_nodes = github_new_nodes[:github_priority_limit]
    github_overflow_nodes = github_new_nodes[github_priority_limit:]
    crawler_nodes = [*github_priority_nodes, *crawler_nodes, *github_overflow_nodes]

    edge_keys = {
        (
            _safe_str(row.get("source", "")),
            _safe_str(row.get("target", "")),
            _safe_str(row.get("kind", "")).lower(),
        )
        for row in edges
        if _safe_str(row.get("source", "")) and _safe_str(row.get("target", ""))
    }
    for row in (
        github.get("edges", []) if isinstance(github.get("edges", []), list) else []
    ):
        if not isinstance(row, dict):
            continue
        source_id = _safe_str(row.get("source", ""))
        target_id = _safe_str(row.get("target", ""))
        kind = _safe_str(row.get("kind", "")).lower()
        if not source_id or not target_id or not kind:
            continue
        key = (source_id, target_id, kind)
        if key in edge_keys:
            continue
        edge_keys.add(key)
        edges.append(row)

    events.extend(
        [row for row in github.get("events", []) if isinstance(row, dict)]
        if isinstance(github.get("events", []), list)
        else []
    )
    events.sort(
        key=lambda row: (
            _safe_str(row.get("ts", "")),
            _safe_str(row.get("event", row.get("kind", ""))),
            _safe_str(row.get("id", "")),
        )
    )
    if len(events) > 240:
        events = events[-240:]

    merged = dict(base)
    merged["field_nodes"] = field_nodes
    merged["crawler_nodes"] = crawler_nodes
    merged["nodes"] = [*field_nodes, *crawler_nodes]
    merged["edges"] = edges
    merged["events"] = events

    status = base.get("status", {}) if isinstance(base.get("status", {}), dict) else {}
    github_status = (
        github.get("status", {}) if isinstance(github.get("status", {}), dict) else {}
    )
    merged_status = dict(status)
    merged_status["github"] = github_status
    merged["status"] = merged_status

    kind_counts: dict[str, int] = {}
    resource_kind_counts: dict[str, int] = {}
    web_role_counts: dict[str, int] = {}
    for row in crawler_nodes:
        if not isinstance(row, dict):
            continue
        kind = _safe_str(row.get("crawler_kind", "")) or "unknown"
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        resource_kind = _safe_str(row.get("resource_kind", "")) or "unknown"
        resource_kind_counts[resource_kind] = (
            resource_kind_counts.get(resource_kind, 0) + 1
        )
        web_role = _safe_str(row.get("web_node_role", ""))
        if web_role:
            web_role_counts[web_role] = web_role_counts.get(web_role, 0) + 1

    web_edge_kind_counts: dict[str, int] = {}
    for row in edges:
        if not isinstance(row, dict):
            continue
        kind = _safe_str(row.get("kind", "")).lower()
        if kind.startswith("web:") or kind.startswith("crawl:"):
            web_edge_kind_counts[kind] = web_edge_kind_counts.get(kind, 0) + 1

    stats = base.get("stats", {}) if isinstance(base.get("stats", {}), dict) else {}
    merged_stats = dict(stats)
    merged_stats.update(
        {
            "field_count": len(field_nodes),
            "crawler_count": len(crawler_nodes),
            "edge_count": len(edges),
            "kind_counts": kind_counts,
            "resource_kind_counts": resource_kind_counts,
            "web_role_counts": web_role_counts,
            "web_edge_kind_counts": web_edge_kind_counts,
            "event_count": len(events),
            "nodes_total": len(crawler_nodes),
            "edges_total": len(edges),
            "url_nodes_total": web_role_counts.get("web:url", 0),
            "github_resource_count": sum(
                1
                for row in crawler_nodes
                if _safe_str(row.get("web_node_role", "")) == "web:resource"
                and _safe_str(row.get("kind", "")).startswith("github:")
            ),
        }
    )
    merged["stats"] = merged_stats
    return merged
