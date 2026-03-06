from __future__ import annotations

from code.world_web.graph_query_policy import (
    SUPPORTED_GRAPH_QUERY_NAMES,
    build_unknown_graph_query_result,
    normalize_graph_query_name,
)


def test_normalize_graph_query_name_maps_common_aliases() -> None:
    assert normalize_graph_query_name("") == "overview"
    assert normalize_graph_query_name("summary") == "overview"
    assert normalize_graph_query_name("recently_touched") == "recently_updated"
    assert normalize_graph_query_name("global") == "geopolitical_news_radar"
    assert normalize_graph_query_name("threats") == "multi_threat_radar"


def test_build_unknown_graph_query_result_uses_supported_registry() -> None:
    payload = build_unknown_graph_query_result("mystery_query")
    assert payload["error"] == "unknown_query"
    assert payload["query"] == "mystery_query"
    assert payload["supported"] == list(SUPPORTED_GRAPH_QUERY_NAMES)
    assert "overview" in payload["supported"]
    assert "multi_threat_radar" in payload["supported"]
