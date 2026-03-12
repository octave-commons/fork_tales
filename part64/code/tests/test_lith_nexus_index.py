from __future__ import annotations

import tempfile
from pathlib import Path

from code.world_web.chamber import collect_promptdb_packets
from code.world_web.graph_queries import run_named_graph_query
from code.world_web.lith_nexus_index import (
    build_promptdb_snapshot_from_lith_index,
    collect_lith_nexus_index,
)
from code.world_web.simulation import _build_logical_graph
from code.world_web.simulation_nexus import _build_canonical_nexus_graph


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _fixture_repo(root: Path) -> None:
    _write(
        root / ".opencode" / "promptdb" / "demo.intent.lisp",
        """
(packet
  (v "opencode.packet/v1")
  (id "demo:packet")
  (kind :intent)
  (title "Demo Packet")
  (tags [:demo :nexus])
  (routing (target :eta-mu-world) (handler :orchestrate) (mode :apply))
  (refs ["contracts/demo.contract.lisp" "https://example.org/demo"]))
        """,
    )
    _write(
        root / ".opencode" / "promptdb" / "contracts" / "demo.contract.lisp",
        '(contract "demo.contract/v1" (title "Demo Contract") (tags [:demo]))',
    )
    _write(
        root / ".opencode" / "promptdb" / "facts" / "2026-03" / "demo-fact.lisp",
        ';; fact\n(fact (ctx 世) (claim "demo claim") (source (path "contracts/demo.contract.lisp")))',
    )
    _write(
        root / ".opencode" / "protocol" / "demo.v1.lisp",
        "(protocol demo.v1 (required (demo :bool)))",
    )
    _write(
        root / "contracts" / "demo.contract.lisp",
        '(contract "contracts.demo/v1" (title "Contracts Demo") (tags [:contract :demo]))',
    )
    _write(
        root / "manifest.lith",
        '(manifest (id "demo:manifest") (title "Demo Manifest") (tags [:manifest]))',
    )
    _write(
        root / "specs" / "demo.md",
        """
# Demo Spec

```lith
(fact (ctx 世) (claim "spec fact") (source (path ".opencode/promptdb/demo.intent.lisp")))
```
        """,
    )


def test_collect_lith_nexus_index_builds_first_class_forms() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _fixture_repo(root)

        index = collect_lith_nexus_index(root, include_text=False)

        stats = index.get("stats", {})
        assert stats.get("packet_count") == 1
        assert stats.get("contract_count") == 2
        assert stats.get("fact_count") == 2
        assert stats.get("form_count") == 7
        assert not index.get("errors")

        roles = {row.get("kind") for row in index.get("nodes", [])}
        assert {
            "file",
            "form",
            "packet",
            "contract",
            "fact",
            "protocol",
            "spec",
            "tag",
        } <= roles

        edge_kinds = {row.get("kind") for row in index.get("edges", [])}
        assert {
            "contains",
            "declares",
            "derived_from",
            "tagged",
            "depends_on",
            "references",
        } <= edge_kinds

        packet = next(
            row for row in index.get("packets", []) if row.get("id") == "demo:packet"
        )
        assert packet.get("routing", {}).get("handler") == ":orchestrate"


def test_promptdb_snapshot_and_canonical_graph_include_lith_nodes() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _fixture_repo(root)

        index = collect_lith_nexus_index(root, include_text=False)
        promptdb = build_promptdb_snapshot_from_lith_index(index)
        assert promptdb.get("packet_count") == 1
        assert promptdb.get("contract_count") == 1

        legacy_promptdb = collect_promptdb_packets(root)
        assert legacy_promptdb.get("packet_count") == 1
        assert legacy_promptdb.get("contract_count") == 1

        logical = _build_logical_graph(
            {
                "file_graph": {},
                "truth_state": {},
                "lith_nexus": index,
                "test_failures": [],
                "world_log": {},
            }
        )
        assert any(node.get("kind") == "packet" for node in logical.get("nodes", []))
        assert any(node.get("kind") == "fact" for node in logical.get("nodes", []))
        assert any(
            edge.get("kind") == "derived_from" for edge in logical.get("edges", [])
        )

        nexus = _build_canonical_nexus_graph(
            file_graph={},
            crawler_graph={},
            logical_graph=logical,
            include_crawler=False,
            include_logical=True,
        )
        assert any(node.get("role") == "packet" for node in nexus.get("nodes", []))
        assert any(node.get("role") == "fact" for node in nexus.get("nodes", []))
        assert any(
            isinstance(edge.get("provenance"), dict) for edge in nexus.get("edges", [])
        )

        search = run_named_graph_query(nexus, "search", args={"q": "demo:packet"})
        ids = {row.get("id") for row in search.get("result", {}).get("nodes", [])}
        assert "demo:packet" in ids
