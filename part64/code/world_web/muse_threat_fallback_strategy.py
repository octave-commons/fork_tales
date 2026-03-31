from __future__ import annotations

import re
from typing import Any


THREAT_FOCUSED_MUSE_IDS: set[str] = {
    "witness_thread",
    "chaos",
    "github_security_review",
}

THREAT_QUERY_TOKENS: set[str] = {
    "threat",
    "threats",
    "risk",
    "risks",
    "cve",
    "security",
    "exploit",
    "active",
    "alert",
    "alerts",
}

_THREAT_FALLBACK_HEADERS: dict[str, str] = {
    "chaos": "Active global threat snapshot (model fallback):",
    "github_security_review": "Active GitHub threat snapshot (model fallback):",
    "witness_thread": "Active security threat snapshot (model fallback):",
}


def _collect_threat_fallback_lines(
    tet_units: list[dict[str, Any]] | None,
) -> tuple[list[str], list[str]]:
    threat_lines: list[str] = []
    source_lines: list[str] = []
    rows = tet_units if isinstance(tet_units, list) else []
    for tet in rows:
        if not isinstance(tet, dict):
            continue
        kind = str(tet.get("kind", "") or "").strip().lower()
        text = str(tet.get("text", "") or "").strip()
        if kind == "threat":
            title_match = re.search(r"Threat:\s*([^\n]+)", text, flags=re.IGNORECASE)
            risk_match = re.search(
                r"Risk:\s*([A-Za-z]+)\s*\((\d+)\)",
                text,
                flags=re.IGNORECASE,
            )
            title = (
                title_match.group(1).strip()
                if title_match
                else text.splitlines()[0].strip()
            )
            if not title:
                title = "Unlabeled threat"
            risk_level = risk_match.group(1).upper() if risk_match else "WATCH"
            risk_score = risk_match.group(2) if risk_match else "0"
            threat_lines.append(f"- {risk_level}({risk_score}) {title[:120]}")
        elif kind == "threat-source":
            source_match = re.search(r"Watch source:\s*([^\.\n]+)", text)
            label = source_match.group(1).strip() if source_match else ""
            if not label:
                label = text.splitlines()[0].strip()[:96]
            if label:
                source_lines.append(f"- {label}")
    return threat_lines, source_lines


def build_muse_threat_fallback_reply(
    *,
    muse_id: str,
    user_text: str,
    manifest: dict[str, Any],
) -> str:
    clean_muse_id = str(muse_id or "").strip().lower()
    if clean_muse_id not in THREAT_FOCUSED_MUSE_IDS:
        return ""

    user_tokens = {
        token
        for token in re.findall(r"[a-z0-9_\-]+", str(user_text or "").lower())
        if token
    }
    threat_prompted = bool(user_tokens & THREAT_QUERY_TOKENS)
    threat_lines, source_lines = _collect_threat_fallback_lines(
        manifest.get("tet_units", []) if isinstance(manifest, dict) else []
    )

    if not threat_lines and not source_lines:
        return ""
    if not threat_prompted and not threat_lines:
        return ""

    header = _THREAT_FALLBACK_HEADERS.get(
        clean_muse_id,
        "Active security threat snapshot (model fallback):",
    )
    lines = [header]
    if threat_lines:
        lines.append("Top threats:")
        lines.extend(threat_lines[:3])
    if source_lines:
        lines.append("Hot sources:")
        lines.extend(source_lines[:3])
    return "\n".join(lines)
