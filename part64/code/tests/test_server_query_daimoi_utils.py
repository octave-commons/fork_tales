from __future__ import annotations

from code.world_web.server_query_daimoi_utils import (
    build_search_daimoi_meta,
    normalize_query_text,
    query_variant_terms,
)


def test_query_variant_terms_normalizes_and_dedupes() -> None:
    assert normalize_query_text("  hello   world  ") == "hello world"
    variants = query_variant_terms("Hello, world!! signals")
    assert variants[0] == "Hello, world!! signals"
    assert "hello world signals" in variants


def test_build_search_daimoi_meta_builds_components_and_targets() -> None:
    payload = build_search_daimoi_meta(
        "fork tax search",
        target="receipt_river and witness_thread",
        model="stub-model",
        entity_manifest=[
            {"id": "receipt_river"},
            {"id": "witness_thread"},
            {"id": "chaos"},
        ],
        normalize_embedding_vector=lambda value: (
            list(value) if isinstance(value, list) else []
        ),
        ollama_embed=lambda text, model=None: [0.25, 0.5, len(str(text)) / 100.0],
    )
    assert payload["query"] == "fork tax search"
    assert payload["embed_model"] == "stub-model"
    assert payload["target_presence_ids"] == ["receipt_river", "witness_thread"]
    components = payload["components"]
    assert isinstance(components, list)
    assert len(components) >= 1
    assert components[0]["component_type"] == "query-term"
    assert components[0]["embedding_dim"] == 3
