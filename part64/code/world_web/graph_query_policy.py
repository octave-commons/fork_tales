from __future__ import annotations

from typing import Any


GRAPH_QUERY_ALIASES: dict[str, str] = {
    "summary": "overview",
    "stats": "overview",
    "neighbor": "neighbors",
    "node_neighbors": "neighbors",
    "roles": "role_slice",
    "role": "role_slice",
    "recent": "recently_updated",
    "recently_touched": "recently_updated",
    "touched": "recently_updated",
    "explain": "explain_daimoi",
    "outcomes": "recent_outcomes",
    "crawler": "crawler_status",
    "arxiv": "arxiv_papers",
    "arxiv_papers": "arxiv_papers",
    "resource_summary": "web_resource_summary",
    "graph_summary": "graph_summary",
    "github": "github_status",
    "github_repo": "github_repo_summary",
    "github_search": "github_find",
    "github_recent": "github_recent_changes",
    "proximity": "proximity_radar",
    "proximity_radar": "proximity_radar",
    "emerging_terms": "proximity_radar",
    "entity_risk_state": "entity_risk_state",
    "entity_state": "entity_risk_state",
    "cyber_regime": "cyber_regime_state",
    "cyber_regime_state": "cyber_regime_state",
    "regime": "cyber_regime_state",
    "cyber_risk_radar": "cyber_risk_radar",
    "regime_radar": "cyber_risk_radar",
    "local": "github_threat_radar",
    "local_threat_radar": "github_threat_radar",
    "global": "geopolitical_news_radar",
    "global_feed": "geopolitical_news_radar",
    "geopolitics": "geopolitical_news_radar",
    "geopolitical": "geopolitical_news_radar",
    "geopolitical_news": "geopolitical_news_radar",
    "geopolitical_news_radar": "geopolitical_news_radar",
    "github_threats": "github_threat_radar",
    "threat_radar": "multi_threat_radar",
    "threats": "multi_threat_radar",
    "multi_threat_radar": "multi_threat_radar",
    "hormuz": "hormuz_threat_radar",
    "hormuz_threats": "hormuz_threat_radar",
    "maritime_threat_radar": "hormuz_threat_radar",
}


SUPPORTED_GRAPH_QUERY_NAMES: tuple[str, ...] = (
    "overview",
    "graph_summary",
    "neighbors",
    "search",
    "url_status",
    "resource_for_url",
    "recently_updated",
    "role_slice",
    "explain_daimoi",
    "recent_outcomes",
    "crawler_status",
    "arxiv_papers",
    "github_status",
    "github_repo_summary",
    "github_find",
    "github_recent_changes",
    "proximity_radar",
    "entity_risk_state",
    "cyber_regime_state",
    "cyber_risk_radar",
    "github_threat_radar",
    "geopolitical_news_radar",
    "hormuz_threat_radar",
    "multi_threat_radar",
    "web_resource_summary",
)


def normalize_graph_query_name(name: Any) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return "overview"
    return GRAPH_QUERY_ALIASES.get(text, text)


def build_unknown_graph_query_result(query: str) -> dict[str, Any]:
    return {
        "error": "unknown_query",
        "query": str(query or ""),
        "supported": list(SUPPORTED_GRAPH_QUERY_NAMES),
    }
