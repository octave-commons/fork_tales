from __future__ import annotations

import json
import os
import tempfile
import wave
import zipfile
from array import array
from pathlib import Path
from typing import Any

import code.world_web as world_web_module
import code.world_web.ai as ai_module

from code.world_web import build_simulation_state, collect_catalog


def _create_fixture_tree(root: Path) -> None:
    manifest = {
        "part": 64,
        "seed_label": "eta_mu_part_64",
        "files": [
            {"path": "artifacts/audio/test.wav", "role": "audio/canonical"},
            {"path": "world_state/constraints.md", "role": "world_state"},
        ],
    }
    (root / "artifacts" / "audio").mkdir(parents=True)
    (root / "world_state").mkdir(parents=True)
    sample = array("h", [0, 1200, -1200, 300, -300, 0])
    with wave.open(str(root / "artifacts" / "audio" / "test.wav"), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(sample.tobytes())
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "world_state" / "constraints.md").write_text(
        "# Constraints\n\n- C-64-world-snapshot: active\n",
        encoding="utf-8",
    )


def test_eta_mu_inbox_is_ingested_and_graphed() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        part.mkdir(parents=True)
        _create_fixture_tree(part)

        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        (inbox / "new_witness_note.md").write_text(
            "Witness thread continuity, gate proof, and fork tax notes.",
            encoding="utf-8",
        )
        (inbox / "new_receipt.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        catalog = collect_catalog(part, vault)
        inbox_state = catalog.get("eta_mu_inbox", {})
        assert inbox_state.get("is_empty") is True
        assert inbox_state.get("pending_count") == 0
        assert inbox_state.get("processed_count", 0) >= 2

        index_path = vault / ".opencode" / "runtime" / "eta_mu_knowledge.v1.jsonl"
        assert index_path.exists()
        entries = [
            json.loads(line)
            for line in index_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        assert any(row.get("name") == "new_witness_note.md" for row in entries)

        note_entry = next(
            row for row in entries if row.get("name") == "new_witness_note.md"
        )
        assert note_entry.get("archive_kind") == "zip"
        assert str(note_entry.get("archived_rel_path", "")).endswith(".zip")
        assert str(note_entry.get("archive_member_path", "")).startswith("payload/")
        assert note_entry.get("archive_manifest_path") == "manifest.json"
        assert isinstance(note_entry.get("archive_container_id"), str)
        assert note_entry.get("archive_container_id")
        assert any(
            isinstance(link, dict) and link.get("kind") == "stored_in_archive"
            for link in note_entry.get("embedding_links", [])
        )

        note_zip_path = vault / str(note_entry.get("archived_rel_path", ""))
        assert note_zip_path.exists()
        with zipfile.ZipFile(note_zip_path, "r") as archive_zip:
            members = set(archive_zip.namelist())
            assert str(note_entry.get("archive_member_path", "")) in members
            assert "manifest.json" in members
            manifest_payload = json.loads(
                archive_zip.read("manifest.json").decode("utf-8")
            )
            assert manifest_payload.get("record") == "ημ.archive-manifest.v1"
            assert manifest_payload.get("archive_rel_path") == note_entry.get(
                "archived_rel_path"
            )
            assert str(manifest_payload.get("source_rel_path", "")).endswith(
                "new_witness_note.md"
            )

        file_graph = catalog.get("file_graph", {})
        assert file_graph.get("record") == "ημ.file-graph.v1"
        assert file_graph.get("stats", {}).get("file_count", 0) >= 2
        assert len(file_graph.get("edges", [])) >= 2
        assert file_graph.get("stats", {}).get("archive_count", 0) >= 2
        assert any(
            str(node.get("url", "")).startswith("/library/.opencode/knowledge/archive/")
            for node in file_graph.get("file_nodes", [])
        )
        note_node = next(
            node
            for node in file_graph.get("file_nodes", [])
            if node.get("name") == "new_witness_note.md"
        )
        assert note_node.get("archive_kind") == "zip"
        assert note_node.get("resource_kind") == "text"
        assert note_node.get("modality") == "text"
        assert note_node.get("archive_member_path") == note_entry.get(
            "archive_member_path"
        )
        assert note_node.get("archive_container_id") == note_entry.get(
            "archive_container_id"
        )
        assert isinstance(note_node.get("embed_layer_points", []), list)
        assert note_node.get("embed_layer_count", 0) >= 1
        embed_layers = file_graph.get("embed_layers", [])
        assert isinstance(embed_layers, list)
        assert len(embed_layers) >= 1
        assert file_graph.get("stats", {}).get("embed_layer_count", 0) >= 1
        assert (
            file_graph.get("stats", {}).get("resource_kind_counts", {}).get("text", 0)
            >= 1
        )
        organizer = file_graph.get("organizer_presence", {})
        assert organizer.get("id") == "presence:file_organizer"
        concept_presences = file_graph.get("concept_presences", [])
        assert isinstance(concept_presences, list)
        assert len(concept_presences) >= 1
        assert any(
            str(edge.get("kind", "")) == "spawns_presence"
            for edge in file_graph.get("edges", [])
            if isinstance(edge, dict)
        )
        assert isinstance(note_node.get("concept_presence_id", ""), str)
        assert note_node.get("organized_by") == "file_organizer"


def test_eta_mu_detect_modality_supports_pdf() -> None:
    modality, reason = ai_module._eta_mu_detect_modality(
        path=Path("/tmp/sample.pdf"),
        mime="application/pdf",
    )
    assert modality == "pdf"
    assert reason == "mime-pdf"


def test_eta_mu_inbox_pdf_is_ingested_with_text_embedding(monkeypatch: Any) -> None:
    monkeypatch.setattr(world_web_module, "ETA_MU_INBOX_DEBOUNCE_SECONDS", 0.0)

    from code.world_web import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "_eta_mu_pdf_derive_segments",
        lambda **kwargs: [
            {
                "id": "pdf-0001",
                "start": 1,
                "end": 1,
                "unit": "page",
                "text": f"pdf page 1 text from {kwargs.get('source_rel_path', '')}",
            }
        ],
    )

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        part.mkdir(parents=True)
        _create_fixture_tree(part)

        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        (inbox / "paper.pdf").write_bytes(b"%PDF-1.4\n%stub\n")

        catalog = collect_catalog(part, vault)
        inbox_state = catalog.get("eta_mu_inbox", {})
        assert inbox_state.get("processed_count", 0) >= 1

        index_path = vault / ".opencode" / "runtime" / "eta_mu_knowledge.v1.jsonl"
        entries = [
            json.loads(line)
            for line in index_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        pdf_entry = next(row for row in entries if row.get("name") == "paper.pdf")
        assert pdf_entry.get("kind") == "pdf"
        assert "pdf page 1 text" in str(pdf_entry.get("text_excerpt", "")).lower()

        file_graph = catalog.get("file_graph", {})
        pdf_node = next(
            node
            for node in file_graph.get("file_nodes", [])
            if node.get("name") == "paper.pdf"
        )
        assert pdf_node.get("resource_kind") == "pdf"
        assert pdf_node.get("modality") == "text"


def test_docmeta_tags_create_first_class_tag_nodes(monkeypatch: Any) -> None:
    monkeypatch.setattr(world_web_module, "ETA_MU_INBOX_DEBOUNCE_SECONDS", 0.0)

    from code.world_web import catalog as catalog_module

    monkeypatch.setattr(catalog_module, "ETA_MU_DOCMETA_CYCLE_SECONDS", 0.0)
    monkeypatch.setattr(catalog_module, "ETA_MU_DOCMETA_MAX_PER_CYCLE", 256)

    monkeypatch.setattr(
        catalog_module,
        "_ollama_generate_text",
        lambda *_args, **_kwargs: ("", ""),
    )

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        part.mkdir(parents=True)
        _create_fixture_tree(part)

        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        (inbox / "new_witness_note.md").write_text(
            "Witness thread continuity, gate proof, and fork tax notes.",
            encoding="utf-8",
        )
        (inbox / "new_anchor_note.md").write_text(
            "Anchor registry updates tie proofs to runtime gates.",
            encoding="utf-8",
        )

        catalog = collect_catalog(part, vault)
        file_graph = catalog.get("file_graph", {})
        assert isinstance(file_graph, dict)

        tag_nodes = file_graph.get("tag_nodes", [])
        assert isinstance(tag_nodes, list)
        assert len(tag_nodes) >= 1

        stats = file_graph.get("stats", {})
        assert isinstance(stats, dict)
        assert stats.get("tag_count", 0) == len(tag_nodes)
        assert stats.get("docmeta_enriched_count", 0) >= 1
        assert stats.get("tag_edge_count", 0) >= 1

        note_node = next(
            node
            for node in file_graph.get("file_nodes", [])
            if node.get("name") == "new_witness_note.md"
        )
        assert str(note_node.get("summary", "")).strip()
        assert len(note_node.get("tags", [])) >= 1
        assert len(note_node.get("labels", [])) >= 1

        note_id = str(note_node.get("id", "")).strip()
        assert note_id
        assert any(
            isinstance(edge, dict)
            and str(edge.get("kind", "")) == "labeled_as"
            and str(edge.get("source", "")) == note_id
            for edge in file_graph.get("edges", [])
        )

        simulation = build_simulation_state(catalog)
        logical_graph = simulation.get("logical_graph", {})
        assert isinstance(logical_graph, dict)
        assert any(
            isinstance(node, dict) and str(node.get("kind", "")) == "tag"
            for node in logical_graph.get("nodes", [])
        )
        assert any(
            isinstance(edge, dict)
            and str(edge.get("kind", "")) in {"labeled_as", "relates_tag"}
            for edge in logical_graph.get("edges", [])
        )


def test_docmeta_index_is_idempotent_and_refreshes_on_source_change(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(world_web_module, "ETA_MU_INBOX_DEBOUNCE_SECONDS", 0.0)

    from code.world_web import catalog as catalog_module

    monkeypatch.setattr(catalog_module, "ETA_MU_DOCMETA_CYCLE_SECONDS", 0.0)
    monkeypatch.setattr(catalog_module, "ETA_MU_DOCMETA_MAX_PER_CYCLE", 256)
    monkeypatch.setattr(
        catalog_module,
        "_ollama_generate_text",
        lambda *_args, **_kwargs: (
            '{"summary":"auto metadata summary","tags":["witness","gate"]}',
            "stub-llm",
        ),
    )

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        part.mkdir(parents=True)
        _create_fixture_tree(part)

        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        note_name = "docmeta_refresh_note.md"
        (inbox / note_name).write_text(
            "Witness thread continuity, gate proof, and fork tax notes.",
            encoding="utf-8",
        )

        collect_catalog(part, vault)
        docmeta_path = vault / ".opencode" / "runtime" / "eta_mu_docmeta.v1.jsonl"
        assert docmeta_path.exists()
        first_rows = [
            json.loads(line)
            for line in docmeta_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        first_count = len(first_rows)
        assert first_count >= 1

        collect_catalog(part, vault)
        second_rows = [
            json.loads(line)
            for line in docmeta_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        second_count = len(second_rows)
        assert second_count == first_count

        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / note_name).write_text(
            "Witness thread continuity with updated anchor registry links.",
            encoding="utf-8",
        )
        collect_catalog(part, vault)

        third_rows = [
            json.loads(line)
            for line in docmeta_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        assert len(third_rows) > second_count
        latest = next(
            row
            for row in sorted(
                third_rows,
                key=lambda item: str(item.get("generated_at", "")),
                reverse=True,
            )
            if str(row.get("source_rel_path", "")).endswith(note_name)
        )
        assert str(latest.get("summary", "")).strip()
        assert latest.get("strategy") in {"llm", "heuristic"}
        assert len(latest.get("tags", [])) >= 1


def test_eta_mu_inbox_registry_skips_unchanged_duplicate(monkeypatch: Any) -> None:
    monkeypatch.setattr(world_web_module, "ETA_MU_INBOX_DEBOUNCE_SECONDS", 0.0)

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        part.mkdir(parents=True)
        _create_fixture_tree(part)

        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        note_body = "Witness thread continuity, gate proof, and fork tax notes."
        (inbox / "new_witness_note.md").write_text(note_body, encoding="utf-8")

        first_catalog = collect_catalog(part, vault)
        first_inbox = first_catalog.get("eta_mu_inbox", {})
        assert first_inbox.get("processed_count", 0) >= 1
        assert first_inbox.get("skipped_count", 0) == 0

        index_path = vault / ".opencode" / "runtime" / "eta_mu_knowledge.v1.jsonl"
        assert index_path.exists()

        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / "new_witness_note.md").write_text(note_body, encoding="utf-8")

        second_catalog = collect_catalog(part, vault)
        second_inbox = second_catalog.get("eta_mu_inbox", {})
        assert second_inbox.get("processed_count", 0) == 0
        assert second_inbox.get("skipped_count", 0) >= 1
        assert second_inbox.get("is_empty") is True

        entries = [
            json.loads(line)
            for line in index_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        note_entries = [
            row for row in entries if row.get("name") == "new_witness_note.md"
        ]
        assert len(note_entries) == 1

        registry_path = vault / ".Π" / "ημ_registry.jsonl"
        assert registry_path.exists()
        registry_rows = [
            json.loads(line)
            for line in registry_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        note_registry_rows = [
            row
            for row in registry_rows
            if row.get("source_name") == "new_witness_note.md"
        ]
        assert any(row.get("event") == "ingested" for row in note_registry_rows)

        ingested_keys = {
            str(row.get("registry_key", ""))
            for row in note_registry_rows
            if row.get("event") == "ingested"
        }
        assert any(
            row.get("event") == "skipped"
            and row.get("reason") == "duplicate_unchanged"
            and str(row.get("registry_key", "")) in ingested_keys
            for row in note_registry_rows
        )


def test_eta_mu_ingest_rejects_unsupported_payloads(monkeypatch: Any) -> None:
    monkeypatch.setattr(world_web_module, "ETA_MU_INBOX_DEBOUNCE_SECONDS", 0.0)

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        part.mkdir(parents=True)
        _create_fixture_tree(part)

        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        (inbox / "unsupported.bin").write_bytes(b"\x00\x01\x02\x03")

        catalog = collect_catalog(part, vault)
        inbox_state = catalog.get("eta_mu_inbox", {})

        assert inbox_state.get("rejected_count", 0) >= 1
        assert inbox_state.get("is_empty") is True

        rejected_root = inbox / "_rejected"
        assert rejected_root.exists()
        assert any(path.is_file() for path in rejected_root.rglob("*"))

        registry_path = vault / ".Π" / "ημ_registry.jsonl"
        assert registry_path.exists()
        registry_rows = [
            json.loads(line)
            for line in registry_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        rejected_rows = [
            row for row in registry_rows if row.get("source_name") == "unsupported.bin"
        ]
        assert any(row.get("event") == "rejected" for row in rejected_rows)
        assert any(row.get("status") == "reject" for row in rejected_rows)


def test_eta_mu_ingest_writes_manifest_stats_snapshot_artifacts(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(world_web_module, "ETA_MU_INBOX_DEBOUNCE_SECONDS", 0.0)

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        part.mkdir(parents=True)
        _create_fixture_tree(part)

        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        (inbox / "ingest_note.md").write_text(
            "Witness thread continuity and gate proof.",
            encoding="utf-8",
        )
        (inbox / "ingest_image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        catalog = collect_catalog(part, vault)
        inbox_state = catalog.get("eta_mu_inbox", {})

        assert inbox_state.get("processed_count", 0) >= 1
        artifacts = inbox_state.get("artifacts", {})
        for key in ("manifest", "stats", "snapshot"):
            rel_path = str(artifacts.get(key, ""))
            assert rel_path
            artifact_path = vault / rel_path
            assert artifact_path.exists()
            assert artifact_path.read_text("utf-8").startswith("(artifact ")

        registry_path = vault / ".Π" / "ημ_registry.jsonl"
        assert registry_path.exists()
        registry_rows = [
            json.loads(line)
            for line in registry_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        note_rows = [
            row for row in registry_rows if row.get("source_name") == "ingest_note.md"
        ]
        assert any(row.get("event") == "ingested" for row in note_rows)
        assert any(str(row.get("idempotence_key", "")).strip() for row in note_rows)
        assert any(str(row.get("embed_id", "")).strip() for row in note_rows)


def test_embeddings_db_upsert_query_delete_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        upsert = world_web_module._embedding_db_upsert(
            vault,
            entry_id="demo-emb-1",
            text="fork tax witness continuity",
            embedding=[1.0, 0.0, 0.0],
            metadata={"kind": "demo"},
            model="test-embed",
        )
        assert upsert.get("ok") is True
        assert upsert.get("status") == "upserted"

        status = world_web_module._embedding_db_status(vault)
        assert status.get("ok") is True
        assert status.get("entry_count") == 1
        assert status.get("dimensions") == [3]

        listing = world_web_module._embedding_db_list(vault, limit=10)
        assert listing.get("ok") is True
        assert listing.get("count") == 1
        assert listing.get("entries", [])[0].get("id") == "demo-emb-1"

        query = world_web_module._embedding_db_query(
            vault,
            query_embedding=[0.95, 0.05, 0.0],
            top_k=3,
        )
        assert query.get("ok") is True
        assert query.get("match_count") == 1
        first = query.get("results", [])[0]
        assert first.get("id") == "demo-emb-1"
        assert float(first.get("score", 0.0)) > 0.9

        deleted = world_web_module._embedding_db_delete(vault, entry_id="demo-emb-1")
        assert deleted.get("ok") is True
        assert deleted.get("status") == "deleted"

        query_after = world_web_module._embedding_db_query(
            vault,
            query_embedding=[0.95, 0.05, 0.0],
            top_k=3,
        )
        assert query_after.get("ok") is True
        assert query_after.get("match_count") == 0


def test_embeddings_db_upsert_is_idempotent_for_same_payload() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        first = world_web_module._embedding_db_upsert(
            vault,
            entry_id="same-id",
            text="anchor registry",
            embedding=[0.2, 0.8, 0.1],
            metadata={"source": "unit"},
            model="test-embed",
        )
        second = world_web_module._embedding_db_upsert(
            vault,
            entry_id="same-id",
            text="anchor registry",
            embedding=[0.2, 0.8, 0.1],
            metadata={"source": "unit"},
            model="test-embed",
        )

        assert first.get("ok") is True
        assert first.get("status") == "upserted"
        assert second.get("ok") is True
        assert second.get("status") == "unchanged"

        db_path = world_web_module._embeddings_db_path(vault)
        assert db_path.exists()
        rows = [
            json.loads(line)
            for line in db_path.read_text("utf-8").splitlines()
            if line.strip()
        ]
        upsert_rows = [
            row
            for row in rows
            if row.get("record") == "ημ.embedding-db.v1"
            and row.get("event") == "upsert"
            and row.get("id") == "same-id"
        ]
        assert len(upsert_rows) == 1


def test_ollama_embed_remote_falls_back_to_api_embed(monkeypatch: Any) -> None:
    calls: list[dict[str, str]] = []

    def fake_c_embed(text: str, *, requested_device: str | None = None) -> list[float]:
        calls.append({"text": text, "device": str(requested_device or "")})
        return [0.1, 0.2, 0.3]

    monkeypatch.setitem(
        world_web_module._ollama_embed_remote.__globals__,
        "_c_embed_text_24",
        fake_c_embed,
    )
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "openvino")

    vector = world_web_module._ollama_embed_remote("witness probe")

    assert vector == [0.1, 0.2, 0.3]
    assert calls == [{"text": "witness probe", "device": "NPU"}]


def test_ollama_embed_remote_prefers_explicit_model_over_env(monkeypatch: Any) -> None:
    calls: list[dict[str, str]] = []

    def fake_c_embed(text: str, *, requested_device: str | None = None) -> list[float]:
        calls.append({"text": text, "device": str(requested_device or "")})
        return [0.4, 0.5, 0.6]

    monkeypatch.setitem(
        world_web_module._ollama_embed_remote.__globals__,
        "_c_embed_text_24",
        fake_c_embed,
    )
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "torch")

    vector = world_web_module._ollama_embed_remote("gate probe", model="arg-model")

    assert vector == [0.4, 0.5, 0.6]
    assert calls == [{"text": "gate probe", "device": "GPU"}]


def test_ollama_embed_remote_force_nomic_with_small_context(monkeypatch: Any) -> None:
    seen_rows: list[dict[str, str]] = []

    def fake_c_embed(text: str, *, requested_device: str | None = None) -> list[float]:
        seen_rows.append({"text": text, "device": str(requested_device or "")})
        return [0.7, 0.8, 0.9]

    monkeypatch.setitem(
        world_web_module._ollama_embed_remote.__globals__,
        "_c_embed_text_24",
        fake_c_embed,
    )
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "openvino")
    monkeypatch.setenv("OLLAMA_EMBED_MAX_CHARS", "6")

    vector = world_web_module._ollama_embed_remote("abcdefghi", model="arg-model")

    assert vector == [0.7, 0.8, 0.9]
    assert seen_rows == [{"text": "abcdef", "device": "NPU"}]


def test_ollama_embed_remote_includes_gpu_options_when_configured(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, str]] = []

    def fake_c_embed(text: str, *, requested_device: str | None = None) -> list[float]:
        calls.append({"text": text, "device": str(requested_device or "")})
        return [0.3, 0.4, 0.5]

    monkeypatch.setitem(
        world_web_module._ollama_embed_remote.__globals__,
        "_c_embed_text_24",
        fake_c_embed,
    )
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "torch")

    vector = world_web_module._ollama_embed_remote("gpu configured")

    assert vector == [0.3, 0.4, 0.5]
    assert calls == [{"text": "gpu configured", "device": "GPU"}]


def test_ollama_generate_remote_includes_gpu_options_when_configured(
    monkeypatch: Any,
) -> None:
    class _FakeResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self._payload).encode("utf-8")

    calls: list[dict[str, Any]] = []

    def fake_urlopen(req: Any, timeout: float = 0.0) -> _FakeResponse:
        payload = json.loads((req.data or b"{}").decode("utf-8"))
        calls.append(
            {
                "url": str(req.full_url),
                "timeout": timeout,
                "payload": payload,
            }
        )
        return _FakeResponse(
            {
                "model": "qwen3-vl:4b-instruct",
                "choices": [{"message": {"content": "gpu response"}}],
            }
        )

    monkeypatch.setattr(world_web_module, "urlopen", fake_urlopen)
    monkeypatch.setenv("TEXT_GENERATION_BASE_URL", "http://vllm.local:8000")
    monkeypatch.setenv("TEXT_GENERATION_MODEL", "qwen3-vl:4b-instruct")
    monkeypatch.setenv("TEXT_GENERATION_DEVICE", "GPU")

    text, model = world_web_module._ollama_generate_text_remote("gpu prompt")

    assert text == "gpu response"
    assert model == "qwen3-vl:4b-instruct"
    assert len(calls) == 1
    assert calls[0]["url"] == "http://vllm.local:8000/v1/chat/completions"
    assert calls[0]["payload"].get("extra_body", {}).get("device") == "GPU"


def test_resource_auto_embedding_order_prefers_hardware_backends(
    monkeypatch: Any,
) -> None:
    order = world_web_module._resource_auto_embedding_order(
        snapshot={
            "devices": {
                "npu0": {"status": "ok"},
                "cpu": {"utilization": 22.0},
                "gpu1": {"utilization": 31.0},
            }
        }
    )

    assert order[0] == "openvino"
    assert "torch" in order


def test_effective_request_embed_model_honors_force_nomic(monkeypatch: Any) -> None:
    from code.world_web import server as server_module

    monkeypatch.setenv("OLLAMA_EMBED_FORCE_NOMIC", "1")
    assert (
        server_module._effective_request_embed_model("qwen3-embedding:8b")
        == "nomic-embed-text"
    )

    monkeypatch.setenv("OLLAMA_EMBED_FORCE_NOMIC", "0")
    assert (
        server_module._effective_request_embed_model("qwen3-embedding:8b")
        == "qwen3-embedding:8b"
    )


def test_openvino_embed_uses_local_endpoint_device_and_normalization(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, str]] = []

    def fake_c_embed(text: str, *, requested_device: str | None = None) -> list[float]:
        calls.append({"text": text, "device": str(requested_device or "")})
        return [0.6, 0.8]

    monkeypatch.setitem(
        world_web_module._openvino_embed.__globals__,
        "_c_embed_text_24",
        fake_c_embed,
    )
    monkeypatch.setenv("OPENVINO_EMBED_DEVICE", "NPU")
    monkeypatch.setenv("OPENVINO_EMBED_MAX_CHARS", "5")

    vector = world_web_module._openvino_embed("abcdefghi")

    assert vector is not None
    assert len(calls) == 1
    assert calls[0]["text"] == "abcde"
    assert calls[0]["device"] == "NPU"
    assert abs(vector[0] - 0.6) < 1e-6
    assert abs(vector[1] - 0.8) < 1e-6


def test_apply_embedding_provider_options_supports_gpu_and_npu_presets(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("EMBEDDINGS_BACKEND", "auto")
    monkeypatch.setenv("OLLAMA_EMBED_FORCE_NOMIC", "0")
    monkeypatch.setenv("OPENVINO_EMBED_DEVICE", "CPU")

    gpu_result = world_web_module._apply_embedding_provider_options(
        {
            "preset": "gpu_local",
            "cuda_model": "nomic-ai/nomic-embed-text-v1.5",
        }
    )
    assert gpu_result.get("ok") is True
    assert os.getenv("EMBEDDINGS_BACKEND") == "torch"
    assert os.getenv("TORCH_EMBED_MODEL") == "nomic-ai/nomic-embed-text-v1.5"

    npu_result = world_web_module._apply_embedding_provider_options(
        {
            "preset": "npu_local",
            "openvino_endpoint": "http://ov.local:18000/v1/embeddings",
            "openvino_bearer_token": "ov-token",
        }
    )
    assert npu_result.get("ok") is True
    assert os.getenv("EMBEDDINGS_BACKEND") == "openvino"
    assert os.getenv("OPENVINO_EMBED_DEVICE") == "NPU"
    assert os.getenv("OPENVINO_EMBED_BEARER_TOKEN") == "ov-token"
    options = npu_result.get("options", {}).get("config", {})
    assert options.get("openvino_auth_mode") == "bearer"
    assert options.get("openvino_auth_header_name") == "Authorization"


def test_eta_mu_space_forms_support_layered_collections(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        world_web_module, "ETA_MU_INGEST_VECSTORE_LAYER_MODE", "signature"
    )
    monkeypatch.setattr(
        world_web_module, "ETA_MU_INGEST_VECSTORE_COLLECTION", "eta_mu_nexus_v1"
    )

    spaces = world_web_module._eta_mu_space_forms()

    text_collection = str(spaces.get("text", {}).get("collection", ""))
    image_collection = str(spaces.get("image", {}).get("collection", ""))
    assert text_collection.startswith("eta_mu_nexus_v1__")
    assert image_collection.startswith("eta_mu_nexus_v1__")
    assert text_collection != image_collection

    vecstore = spaces.get("vecstore", {})
    assert vecstore.get("layer_mode") == "signature"
    collections = vecstore.get("collections", {})
    assert collections.get("text") == text_collection
    assert collections.get("image") == image_collection


def test_docmeta_record_rejects_path_only_llm_summary(monkeypatch: Any) -> None:
    from code.world_web import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "_ollama_generate_text",
        lambda *_args, **_kwargs: (
            '{"summary":"demo.zip .opencode/knowledge/archive/2026/02/16/demo.zip", "tags":["archive","file"]}',
            "stub-llm",
        ),
    )

    with tempfile.TemporaryDirectory() as td:
        record = catalog_module._build_docmeta_record(
            {
                "id": "knowledge.demo",
                "source_hash": "abc123",
                "source_rel_path": ".ημ/demo.zip",
                "archive_rel_path": ".opencode/knowledge/archive/demo.zip",
                "kind": "file",
                "name": "demo.zip",
                "dominant_field": "f3",
                "text_excerpt": ".opencode/knowledge/archive/2026/02/16/demo.zip",
            },
            Path(td),
        )

    summary = str(record.get("summary", ""))
    assert summary
    assert "opencode/knowledge/archive" not in summary.lower()
    assert catalog_module._docmeta_summary_is_usable(summary) is True


def test_docmeta_record_keeps_concrete_llm_summary(monkeypatch: Any) -> None:
    from code.world_web import catalog as catalog_module

    monkeypatch.setattr(
        catalog_module,
        "_ollama_generate_text",
        lambda *_args, **_kwargs: (
            '{"summary":"Release note describing token validation hardening and dependency update impact.", "tags":["security_update","token_validation","dependency_update","release_note"]}',
            "stub-llm",
        ),
    )

    with tempfile.TemporaryDirectory() as td:
        record = catalog_module._build_docmeta_record(
            {
                "id": "knowledge.release-note",
                "source_hash": "def456",
                "source_rel_path": ".ημ/release-note.md",
                "archive_rel_path": ".opencode/knowledge/archive/release-note.zip",
                "kind": "text",
                "name": "release-note.md",
                "dominant_field": "f7",
                "text_excerpt": "Updated token validation and parser dependency for security release.",
            },
            Path(td),
        )

    assert record.get("strategy") == "llm"
    assert "token validation" in str(record.get("summary", "")).lower()
    assert len(record.get("tags", [])) >= 3


def test_embedding_caption_normalizer_strips_labels_and_bounds_words() -> None:
    raw = (
        "Caption: A complex observatory interior with instrument panels, warning lights, "
        "status displays, and operators monitoring security telemetry across multiple monitors "
        "during an incident response drill."
    )
    normalized = ai_module._normalize_embedding_caption_text(raw)

    assert normalized
    assert not normalized.lower().startswith("caption:")
    assert len(normalized.split()) <= 28
