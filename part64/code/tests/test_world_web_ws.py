from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path
from typing import Any, cast

from code.world_web import websocket_accept_value, websocket_frame_text


class _MockWebSocketTransport:
    def __init__(self, payload: bytes) -> None:
        self._buffer = payload
        self.sent: list[bytes] = []

    def recv(self, size: int) -> bytes:
        if not self._buffer:
            return b""
        chunk = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return chunk

    def sendall(self, payload: bytes) -> None:
        self.sent.append(payload)

def _masked_ws_frame(opcode: int, payload: bytes) -> bytes:
    mask = bytes([0x23, 0x45, 0x67, 0x89])
    masked_payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return bytes([0x80 | (opcode & 0x0F), 0x80 | len(payload)]) + mask + masked_payload

def test_websocket_helpers() -> None:
    accept = websocket_accept_value("dGhlIHNhbXBsZSBub25jZQ==")
    assert accept == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="

    frame = websocket_frame_text('{"ok":true}')
    assert frame[0] == 0x81
    assert frame[1] == len('{"ok":true}')

def test_simulation_ws_normalize_delta_stream_mode_aliases() -> None:
    from code.world_web import server as server_module

    assert (
        server_module._simulation_ws_normalize_delta_stream_mode("workers") == "workers"
    )
    assert (
        server_module._simulation_ws_normalize_delta_stream_mode("thread") == "workers"
    )
    assert server_module._simulation_ws_normalize_delta_stream_mode("world") == "world"

def test_simulation_ws_normalize_payload_mode_aliases() -> None:
    from code.world_web import server as server_module

    assert server_module._simulation_ws_normalize_payload_mode("full") == "full"
    assert server_module._simulation_ws_normalize_payload_mode("debug") == "full"
    assert server_module._simulation_ws_normalize_payload_mode("trimmed") == "trimmed"
    assert server_module._simulation_ws_normalize_payload_mode("") == "trimmed"

def test_simulation_ws_chunk_messages_round_trip() -> None:
    from code.world_web import server as server_module

    payload: dict[str, Any] = {
        "type": "simulation",
        "simulation": {
            "timestamp": "2026-02-23T00:00:00Z",
            "points": [
                {
                    "id": f"pt:{index}",
                    "x": round(index / 300.0, 6),
                    "y": round((300 - index) / 300.0, 6),
                }
                for index in range(320)
            ],
        },
    }

    rows = server_module._simulation_ws_chunk_messages(
        payload,
        chunk_chars=320,
        message_seq=11,
    )
    assert len(rows) > 1
    assert all(str(row.get("type", "")) == "ws_chunk" for row in rows)
    assert all(str(row.get("chunk_payload_type", "")) == "simulation" for row in rows)

    ordered_rows = sorted(rows, key=lambda row: int(row.get("chunk_index", -1)))
    merged_text = "".join(str(row.get("payload", "")) for row in ordered_rows)
    assert json.loads(merged_text) == payload

def test_simulation_ws_chunk_plan_reuses_small_payload_text() -> None:
    from code.world_web import server as server_module

    payload: dict[str, Any] = {
        "type": "simulation_delta",
        "delta": {"timestamp": "2026-02-23T00:00:00Z", "changed_keys": ["x"]},
    }

    rows, payload_text = server_module._simulation_ws_chunk_plan(
        payload,
        chunk_chars=4096,
        message_seq=5,
    )

    assert rows == []
    assert isinstance(payload_text, str)
    assert json.loads(payload_text) == payload

def test_simulation_ws_chunk_plan_emits_chunks_for_large_payload() -> None:
    from code.world_web import server as server_module

    payload: dict[str, Any] = {
        "type": "simulation",
        "simulation": {
            "timestamp": "2026-02-23T00:00:00Z",
            "points": [
                {
                    "id": f"pt:{index}",
                    "x": round(index / 300.0, 6),
                    "y": round((300 - index) / 300.0, 6),
                }
                for index in range(320)
            ],
        },
    }

    rows, payload_text = server_module._simulation_ws_chunk_plan(
        payload,
        chunk_chars=320,
        message_seq=12,
    )

    assert len(rows) > 1
    assert payload_text is None
    merged_text = "".join(
        str(row.get("payload", ""))
        for row in sorted(rows, key=lambda row: int(row.get("chunk_index", -1)))
    )
    assert json.loads(merged_text) == payload

def test_simulation_ws_chunk_plan_keeps_medium_delta_payload_unchunked(
    monkeypatch: Any,
) -> None:
    from code.world_web import server as server_module

    monkeypatch.setattr(server_module, "_SIMULATION_WS_CHUNK_DELTA_MIN_CHARS", 50000)
    payload: dict[str, Any] = {
        "type": "simulation_delta",
        "delta": {
            "timestamp": "2026-02-23T00:00:00Z",
            "patch": {
                "presence_dynamics": {
                    "field_particles": [
                        {
                            "id": f"dm:{index}",
                            "presence_id": "witness_thread",
                            "x": round(index / 240.0, 6),
                            "y": round((240 - index) / 240.0, 6),
                        }
                        for index in range(240)
                    ]
                }
            },
        },
    }

    rows, payload_text = server_module._simulation_ws_chunk_plan(
        payload,
        chunk_chars=4096,
        message_seq=33,
    )

    assert rows == []
    assert isinstance(payload_text, str)
    assert len(payload_text) > 4096
    assert len(payload_text) < 50000

def test_simulation_ws_compact_graph_payload_keeps_node_labels() -> None:
    from code.world_web import server as server_module

    simulation_payload: dict[str, Any] = {
        "timestamp": "2026-02-23T00:00:00Z",
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "nodes": [
                {
                    "id": "file:abc",
                    "node_type": "file",
                    "label": "evidence-note.md",
                    "name": "evidence-note.md",
                    "source_rel_path": ".ημ/evidence-note.md",
                    "x": 0.42,
                    "y": 0.33,
                    "hue": 205,
                    "importance": 0.51,
                }
            ],
            "file_nodes": [
                {
                    "id": "file:abc",
                    "node_type": "file",
                    "label": "evidence-note.md",
                    "name": "evidence-note.md",
                    "source_rel_path": ".ημ/evidence-note.md",
                    "x": 0.42,
                    "y": 0.33,
                    "hue": 205,
                    "importance": 0.51,
                }
            ],
            "edges": [],
            "stats": {"file_count": 1, "edge_count": 0},
        },
        "crawler_graph": {
            "record": "ημ.crawler-graph.v1",
            "nodes": [
                {
                    "id": "crawler:def",
                    "node_type": "crawler",
                    "label": "example.org",
                    "crawler_kind": "domain",
                    "domain": "example.org",
                    "x": 0.6,
                    "y": 0.21,
                    "hue": 176,
                    "importance": 0.48,
                }
            ],
            "crawler_nodes": [
                {
                    "id": "crawler:def",
                    "node_type": "crawler",
                    "label": "example.org",
                    "crawler_kind": "domain",
                    "domain": "example.org",
                    "x": 0.6,
                    "y": 0.21,
                    "hue": 176,
                    "importance": 0.48,
                }
            ],
            "edges": [],
            "stats": {"crawler_count": 1, "edge_count": 0},
        },
    }

    compact = server_module._simulation_ws_compact_graph_payload(simulation_payload)
    file_graph = compact.get("file_graph", {})
    crawler_graph = compact.get("crawler_graph", {})
    assert isinstance(file_graph, dict)
    assert isinstance(crawler_graph, dict)
    assert file_graph.get("nodes", [])[0].get("label") == "evidence-note.md"
    assert (
        file_graph.get("nodes", [])[0].get("source_rel_path") == ".ημ/evidence-note.md"
    )
    assert crawler_graph.get("nodes", [])[0].get("label") == "example.org"

def test_simulation_ws_compact_graph_payload_assume_trimmed_reuses_graph_refs() -> None:
    from code.world_web import server as server_module

    file_graph = {"record": "ημ.file-graph.v1", "nodes": [{"id": "file:1"}]}
    crawler_graph = {
        "record": "ημ.crawler-graph.v1",
        "nodes": [{"id": "crawler:1"}],
    }
    simulation_payload = {
        "file_graph": file_graph,
        "crawler_graph": crawler_graph,
    }

    compact = server_module._simulation_ws_compact_graph_payload(
        simulation_payload,
        assume_trimmed=True,
    )
    assert compact.get("file_graph") is file_graph
    assert compact.get("crawler_graph") is crawler_graph

def test_simulation_ws_load_cached_payload_trimmed_includes_compact_graphs() -> None:
    from code.world_web import server as server_module

    cached_payload = {
        "timestamp": "2026-02-23T00:00:00Z",
        "total": 2,
        "points": [{"id": "pt:1"}, {"id": "pt:2"}],
        "presence_dynamics": {
            "field_particles": [{"id": "dm:1", "presence_id": "witness_thread"}],
        },
        "file_graph": {
            "record": "ημ.file-graph.v1",
            "nodes": [
                {
                    "id": "file:abc",
                    "node_type": "file",
                    "label": "evidence-note.md",
                    "name": "evidence-note.md",
                    "source_rel_path": ".ημ/evidence-note.md",
                    "x": 0.42,
                    "y": 0.33,
                    "hue": 205,
                    "importance": 0.51,
                }
            ],
            "file_nodes": [
                {
                    "id": "file:abc",
                    "node_type": "file",
                    "label": "evidence-note.md",
                    "name": "evidence-note.md",
                    "source_rel_path": ".ημ/evidence-note.md",
                    "x": 0.42,
                    "y": 0.33,
                    "hue": 205,
                    "importance": 0.51,
                }
            ],
            "edges": [],
            "stats": {"file_count": 1, "edge_count": 0},
        },
        "crawler_graph": {
            "record": "ημ.crawler-graph.v1",
            "nodes": [
                {
                    "id": "crawler:def",
                    "node_type": "crawler",
                    "label": "example.org",
                    "crawler_kind": "domain",
                    "domain": "example.org",
                    "x": 0.6,
                    "y": 0.21,
                    "hue": 176,
                    "importance": 0.48,
                }
            ],
            "crawler_nodes": [
                {
                    "id": "crawler:def",
                    "node_type": "crawler",
                    "label": "example.org",
                    "crawler_kind": "domain",
                    "domain": "example.org",
                    "x": 0.6,
                    "y": 0.21,
                    "hue": 176,
                    "importance": 0.48,
                }
            ],
            "edges": [],
            "stats": {"crawler_count": 1, "edge_count": 0},
        },
        "projection": {"perspective": "hybrid"},
    }

    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        monkeypatch_payload = json.dumps(cached_payload).encode("utf-8")

        original_compact_cached_body = (
            server_module._simulation_http_compact_cached_body
        )
        original_cached_body = server_module._simulation_http_cached_body
        original_disk_cache_load = server_module._simulation_http_disk_cache_load
        try:
            server_module._simulation_http_compact_cached_body = lambda **kwargs: None  # type: ignore[assignment]
            server_module._simulation_http_cached_body = lambda **kwargs: (
                monkeypatch_payload
            )  # type: ignore[assignment]
            server_module._simulation_http_disk_cache_load = lambda *args, **kwargs: (
                None
            )  # type: ignore[assignment]
            loaded = server_module._simulation_ws_load_cached_payload(
                part_root=part_root,
                perspective="hybrid",
                payload_mode="trimmed",
            )
        finally:
            server_module._simulation_http_compact_cached_body = (
                original_compact_cached_body  # type: ignore[assignment]
            )
            server_module._simulation_http_cached_body = original_cached_body  # type: ignore[assignment]
            server_module._simulation_http_disk_cache_load = original_disk_cache_load  # type: ignore[assignment]

    assert loaded is not None
    simulation_payload, projection = loaded
    assert projection == {"perspective": "hybrid"}
    assert (
        simulation_payload.get("file_graph", {}).get("nodes", [])[0].get("label")
        == "evidence-note.md"
    )
    assert (
        simulation_payload.get("crawler_graph", {}).get("nodes", [])[0].get("label")
        == "example.org"
    )

def test_simulation_ws_load_cached_payload_trimmed_prefers_compact_cache() -> None:
    from code.world_web import server as server_module

    compact_payload = {
        "timestamp": "2026-02-23T00:00:00Z",
        "total": 1,
        "points": [{"id": "pt:1"}],
        "presence_dynamics": {
            "field_particles": [{"id": "dm:1", "presence_id": "witness_thread"}],
        },
        "file_graph": {"record": "ημ.file-graph.v1", "nodes": [{"id": "file:abc"}]},
        "crawler_graph": {
            "record": "ημ.crawler-graph.v1",
            "nodes": [{"id": "crawler:def"}],
        },
        "projection": {"perspective": "hybrid"},
    }

    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        compact_body = json.dumps(compact_payload).encode("utf-8")
        fallback_calls = {"full_cache": 0}

        original_compact_cached_body = (
            server_module._simulation_http_compact_cached_body
        )
        original_cached_body = server_module._simulation_http_cached_body
        original_disk_cache_load = server_module._simulation_http_disk_cache_load
        try:
            server_module._simulation_http_compact_cached_body = lambda **kwargs: (
                compact_body
            )  # type: ignore[assignment]

            def _full_cache_miss(**kwargs: Any) -> None:
                fallback_calls["full_cache"] += 1
                return None

            server_module._simulation_http_cached_body = _full_cache_miss  # type: ignore[assignment]
            server_module._simulation_http_disk_cache_load = lambda *args, **kwargs: (
                None
            )  # type: ignore[assignment]
            loaded = server_module._simulation_ws_load_cached_payload(
                part_root=part_root,
                perspective="hybrid",
                payload_mode="trimmed",
            )
        finally:
            server_module._simulation_http_compact_cached_body = (
                original_compact_cached_body  # type: ignore[assignment]
            )
            server_module._simulation_http_cached_body = original_cached_body  # type: ignore[assignment]
            server_module._simulation_http_disk_cache_load = original_disk_cache_load  # type: ignore[assignment]

    assert fallback_calls["full_cache"] == 0
    assert loaded is not None
    simulation_payload, projection = loaded
    assert projection.get("perspective") == "hybrid"
    assert simulation_payload.get("file_graph", {}).get("record") == "ημ.file-graph.v1"

def test_simulation_ws_load_cached_payload_full_falls_back_when_cache_bytes_invalid(
    monkeypatch: Any,
) -> None:
    from code.world_web import server as server_module

    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        cache_path = server_module._simulation_http_disk_cache_path(part_root, "hybrid")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_payload = {
            "timestamp": "2026-02-23T06:00:00Z",
            "total": 2,
            "points": [{"id": "a"}, {"id": "b"}],
            "presence_dynamics": {
                "field_particles": [{"id": "dm-1"}],
            },
            "projection": {"perspective": "hybrid"},
        }
        cache_path.write_text(json.dumps(cache_payload), encoding="utf-8")

        monkeypatch.setattr(
            server_module,
            "_simulation_http_cached_body",
            lambda **kwargs: b"{invalid-json",
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_disk_cache_load",
            lambda *args, **kwargs: None,
        )

        cached_payload = server_module._simulation_ws_load_cached_payload(
            part_root=part_root,
            perspective="hybrid",
            payload_mode="full",
        )
        assert cached_payload is not None
        simulation_payload, projection = cached_payload
        assert simulation_payload.get("total") == 2
        assert len(simulation_payload.get("points", [])) == 2
        assert simulation_payload.get("projection") is None
        assert projection == {"perspective": "hybrid"}

def test_simulation_ws_load_cached_payload_full_prefers_non_sparse_disk_payload(
    monkeypatch: Any,
) -> None:
    from code.world_web import server as server_module

    in_memory_sparse = {
        "timestamp": "2026-02-23T06:01:00Z",
        "total": 0,
        "points": [],
        "presence_dynamics": {
            "field_particles": [],
        },
        "projection": {"perspective": "in-memory"},
    }

    with tempfile.TemporaryDirectory() as td:
        part_root = Path(td)
        cache_path = server_module._simulation_http_disk_cache_path(part_root, "hybrid")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        disk_payload = {
            "timestamp": "2026-02-23T05:59:00Z",
            "total": 5,
            "points": [{"id": f"pt:{index}"} for index in range(5)],
            "presence_dynamics": {
                "field_particles": [{"id": "dm-1"}, {"id": "dm-2"}],
            },
            "projection": {"perspective": "hybrid"},
        }
        cache_path.write_text(json.dumps(disk_payload), encoding="utf-8")

        monkeypatch.setattr(
            server_module,
            "_simulation_http_cached_body",
            lambda **kwargs: json.dumps(in_memory_sparse).encode("utf-8"),
        )
        monkeypatch.setattr(
            server_module,
            "_simulation_http_disk_cache_load",
            lambda *args, **kwargs: None,
        )

        cached_payload = server_module._simulation_ws_load_cached_payload(
            part_root=part_root,
            perspective="hybrid",
            payload_mode="full",
        )
        assert cached_payload is not None
        simulation_payload, projection = cached_payload
        assert simulation_payload.get("total") == 5
        assert len(simulation_payload.get("points", [])) == 5
        assert (
            len(
                simulation_payload.get("presence_dynamics", {}).get(
                    "field_particles", []
                )
            )
            == 2
        )
        assert projection == {"perspective": "hybrid"}

def test_ws_wire_mode_normalization_aliases() -> None:
    from code.world_web import server as server_module

    assert server_module._normalize_ws_wire_mode("arr") == "arr"
    assert server_module._normalize_ws_wire_mode("packed") == "arr"
    assert server_module._normalize_ws_wire_mode("json") == "json"

def test_ws_pack_message_uses_array_numeric_wire_nodes() -> None:
    from code.world_web import server as server_module

    payload = {
        "type": "simulation_delta",
        "ok": True,
        "nullable": None,
        "delta": {
            "patch": {
                "timestamp": "2026-02-21T18:10:00Z",
                "values": [1, "alpha", 2.5, False, None],
            }
        },
    }

    packed = server_module._ws_pack_message(payload)
    assert isinstance(packed, list)
    assert packed[0] == server_module._WS_WIRE_ARRAY_SCHEMA
    assert isinstance(packed[1], list)
    assert isinstance(packed[2], list)

    def assert_only_arrays_strings_numbers(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                assert_only_arrays_strings_numbers(item)
            return
        assert isinstance(value, (str, int, float))
        assert not isinstance(value, bool)

    assert_only_arrays_strings_numbers(packed)

    key_table = cast(list[str], packed[1])

    def decode_node(node: Any) -> Any:
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            return node
        assert isinstance(node, list)
        assert node
        tag = int(cast(int | float, node[0]))
        if tag == server_module._WS_PACK_TAG_NULL:
            return None
        if tag == server_module._WS_PACK_TAG_BOOL:
            return int(cast(int | float, node[1])) != 0
        if tag == server_module._WS_PACK_TAG_STRING:
            return str(node[1])
        if tag == server_module._WS_PACK_TAG_ARRAY:
            return [decode_node(item) for item in node[1:]]
        if tag == server_module._WS_PACK_TAG_OBJECT:
            out: dict[str, Any] = {}
            index = 1
            while index + 1 < len(node):
                key_slot = int(cast(int | float, node[index]))
                key_name = key_table[key_slot]
                out[key_name] = decode_node(node[index + 1])
                index += 2
            return out
        raise AssertionError(f"unexpected node tag: {tag}")

    decoded = decode_node(packed[2])
    assert decoded == payload

def test_simulation_ws_split_delta_by_worker_splits_presence_dynamics() -> None:
    from code.world_web import server as server_module

    delta = {
        "patch": {
            "timestamp": "2026-02-21T17:55:00Z",
            "total": 2,
            "daimoi": {"active": 1},
            "presence_dynamics": {
                "field_particles": [{"id": "dm-1"}],
                "graph_node_positions": {"node-1": {"x": 0.3, "y": 0.7}},
                "presence_anchor_positions": {"witness_thread": {"x": 0.6, "y": 0.4}},
                "resource_heartbeat": {"devices": {"cpu": {"utilization": 12.0}}},
                "user_presence": {"id": "user"},
            },
        },
        "changed_keys": [
            "timestamp",
            "total",
            "daimoi",
            "presence_dynamics",
        ],
    }

    rows = server_module._simulation_ws_split_delta_by_worker(delta)
    by_worker = {
        str(row.get("worker_id", "")): row
        for row in rows
        if isinstance(row, dict) and row.get("worker_id")
    }

    assert "sim-core" in by_worker
    assert "sim-daimoi" in by_worker
    assert "sim-particles" in by_worker
    assert "sim-resource" in by_worker
    assert "sim-interaction" in by_worker

    assert by_worker["sim-core"]["patch"].get("total") == 2
    assert by_worker["sim-daimoi"]["patch"].get("daimoi") == {"active": 1}
    assert by_worker["sim-particles"]["patch"].get("presence_dynamics", {}).get(
        "field_particles", []
    ) == [{"id": "dm-1"}]
    assert by_worker["sim-particles"]["patch"].get("presence_dynamics", {}).get(
        "graph_node_positions", {}
    ) == {"node-1": {"x": 0.3, "y": 0.7}}
    assert by_worker["sim-particles"]["patch"].get("presence_dynamics", {}).get(
        "presence_anchor_positions", {}
    ) == {"witness_thread": {"x": 0.6, "y": 0.4}}
    assert by_worker["sim-resource"]["patch"].get("presence_dynamics", {}).get(
        "resource_heartbeat", {}
    ) == {"devices": {"cpu": {"utilization": 12.0}}}
    assert by_worker["sim-interaction"]["patch"].get("presence_dynamics", {}).get(
        "user_presence", {}
    ) == {"id": "user"}

    assert by_worker["sim-daimoi"]["patch"].get("timestamp") == "2026-02-21T17:55:00Z"

def test_simulation_ws_split_delta_by_worker_routes_tick_telemetry_to_core() -> None:
    from code.world_web import server as server_module

    delta = {
        "patch": {
            "timestamp": "2026-02-21T18:20:00Z",
            "tick_elapsed_ms": 4.2,
            "slack_ms": 7.8,
            "ingestion_pressure": 0.63,
            "ws_particle_max": 640,
            "particle_payload_mode": "lite",
            "graph_node_positions_truncated": True,
            "graph_node_positions_total": 2048,
            "presence_anchor_positions_truncated": True,
            "presence_anchor_positions_total": 1024,
        },
        "changed_keys": [
            "timestamp",
            "tick_elapsed_ms",
            "slack_ms",
            "ingestion_pressure",
            "ws_particle_max",
            "particle_payload_mode",
            "graph_node_positions_truncated",
            "graph_node_positions_total",
            "presence_anchor_positions_truncated",
            "presence_anchor_positions_total",
        ],
    }

    rows = server_module._simulation_ws_split_delta_by_worker(delta)
    by_worker = {
        str(row.get("worker_id", "")): row
        for row in rows
        if isinstance(row, dict) and row.get("worker_id")
    }

    assert "sim-core" in by_worker
    assert "sim-misc" not in by_worker
    core_patch = by_worker["sim-core"].get("patch", {})
    assert core_patch.get("tick_elapsed_ms") == 4.2
    assert core_patch.get("particle_payload_mode") == "lite"
    assert core_patch.get("graph_node_positions_total") == 2048
    assert core_patch.get("presence_anchor_positions_total") == 1024

def test_consume_ws_client_frame_replies_to_ping() -> None:
    from code.world_web import server as server_module

    transport = _MockWebSocketTransport(_masked_ws_frame(0x9, b"ping"))
    keep_open = server_module._consume_ws_client_frame(cast(Any, transport))

    assert keep_open is True
    assert transport.sent == [b"\x8a\x04ping"]

def test_consume_ws_client_frame_handles_close() -> None:
    from code.world_web import server as server_module

    payload = struct.pack("!H", 1000)
    transport = _MockWebSocketTransport(_masked_ws_frame(0x8, payload))
    keep_open = server_module._consume_ws_client_frame(cast(Any, transport))

    assert keep_open is False
    assert transport.sent == [b"\x88\x02" + payload]

def test_simulation_ws_trim_payload_drops_field_registry_and_heat_values() -> None:
    from code.world_web import server as server_module

    payload = {
        "presence_dynamics": {"field_particles": [{"id": "p:1"}]},
        "field_particles": [{"id": "legacy:root"}],
        "nexus_graph": {"nodes": [1]},
        "logical_graph": {"nodes": [1]},
        "pain_field": {"node_heat": [1]},
        "heat_values": {"regions": [1]},
        "file_graph": {"file_nodes": [1]},
        "crawler_graph": {"crawler_nodes": [1]},
        "field_registry": {"fields": {"demand": {}}},
        "truth_state": {"gate": {"blocked": True}},
    }

    trimmed = server_module._simulation_ws_trim_simulation_payload(payload)
    assert "field_particles" not in trimmed
    assert "nexus_graph" not in trimmed
    assert "logical_graph" not in trimmed
    assert "pain_field" not in trimmed
    assert "heat_values" not in trimmed
    assert "file_graph" not in trimmed
    assert "crawler_graph" not in trimmed
    assert "field_registry" not in trimmed
    assert trimmed.get("truth_state", {}).get("gate", {}).get("blocked") is True
