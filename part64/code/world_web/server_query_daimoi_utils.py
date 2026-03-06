from __future__ import annotations

import hashlib
from typing import Any, Callable


def normalize_query_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    return " ".join(raw.split())[:220]


def query_variant_terms(query_text: str) -> list[str]:
    base = normalize_query_text(query_text)
    if not base:
        return []

    lowered = base.lower()
    token_rows = [
        token
        for token in "".join(
            ch if (ch.isalnum() or ch.isspace()) else " " for ch in lowered
        ).split()
        if token
    ]

    variants: list[str] = []
    for candidate in (
        base,
        lowered,
        " ".join(token_rows),
        " ".join(token_rows[:4]),
        " ".join(token_rows[-4:]),
    ):
        clean = normalize_query_text(candidate)
        if clean and clean not in variants:
            variants.append(clean)
        if len(variants) >= 6:
            break
    return variants


def build_search_daimoi_meta(
    query_text: str,
    *,
    target: str,
    model: str | None,
    entity_manifest: list[dict[str, Any]],
    normalize_embedding_vector: Callable[[Any], list[float] | None],
    ollama_embed: Callable[..., Any],
) -> dict[str, Any]:
    variants = query_variant_terms(query_text)
    if not variants:
        return {}

    target_text = str(target or "").strip().lower()
    target_presence_ids: list[str] = []
    for row in entity_manifest:
        if not isinstance(row, dict):
            continue
        presence_id = str(row.get("id", "") or "").strip()
        if not presence_id:
            continue
        if (
            presence_id.lower() in target_text
            and presence_id not in target_presence_ids
        ):
            target_presence_ids.append(presence_id)

    component_rows: list[dict[str, Any]] = []
    for index, term in enumerate(variants[:6]):
        component_id = hashlib.sha1(f"{term}|{index}".encode("utf-8")).hexdigest()[:12]
        embedding = normalize_embedding_vector(ollama_embed(term, model=model))
        component: dict[str, Any] = {
            "component_id": f"query:{component_id}",
            "component_type": "query-term",
            "kind": "search",
            "text": term,
            "weight": round(max(0.2, 1.0 - (index * 0.12)), 6),
            "variant_rank": index,
            "embedding_dim": 0,
        }
        if embedding:
            component["embedding_dim"] = len(embedding)
            component["embedding_preview"] = [
                round(float(value), 6) for value in embedding[:8]
            ]
        component_rows.append(component)

    return {
        "record": "ημ.user-search-daimoi.v1",
        "schema_version": "user.search.daimoi.v1",
        "query": variants[0],
        "variant_count": len(variants),
        "embed_model": str(model or "").strip(),
        "target_presence_ids": target_presence_ids,
        "components": component_rows,
    }
