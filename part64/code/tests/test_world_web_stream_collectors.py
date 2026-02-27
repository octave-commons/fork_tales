from __future__ import annotations

import json
import tempfile
import wave
from array import array
from pathlib import Path
from typing import Any

import code.world_web as world_web_module

from code.world_web import build_world_log_payload


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


def test_world_log_payload_tracks_pending_eta_mu_and_embeddings() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "ημ_op_mf_part_64"
        _create_fixture_tree(part)

        (vault / "receipts.log").write_text(
            (
                "ts=2026-02-15T00:00:00Z | kind=:decision | origin=unit-test | owner=Err | "
                "dod=world-log-check | pi=part64 | host=127.0.0.1:8787 | manifest=manifest.lith | "
                "refs=.opencode/promptdb/00_wire_world.intent.lisp"
            ),
            encoding="utf-8",
        )
        inbox = vault / ".ημ"
        inbox.mkdir(parents=True)
        (inbox / "pending_note.md").write_text(
            "pending note waiting for ingest",
            encoding="utf-8",
        )

        payload = build_world_log_payload(part, vault, limit=80)
        assert payload.get("ok") is True
        assert payload.get("record") == "ημ.world-log.v1"
        assert int(payload.get("count", 0)) >= 1
        assert int(payload.get("pending_inbox", 0)) >= 1

        events = payload.get("events", [])
        pending_events = [
            row
            for row in events
            if isinstance(row, dict) and str(row.get("kind", "")) == "eta_mu.pending"
        ]
        assert pending_events
        sample = pending_events[0]
        assert str(sample.get("embedding_id", "")).startswith("world-event:")
        assert isinstance(sample.get("x"), float)
        assert isinstance(sample.get("y"), float)
        assert isinstance(sample.get("relations", []), list)

        listing = world_web_module._embedding_db_list(vault, limit=500)
        ids = {
            str(row.get("id", ""))
            for row in listing.get("entries", [])
            if isinstance(row, dict)
        }
        assert str(sample.get("embedding_id", "")) in ids

def test_wikimedia_stream_collect_tracks_dedupe_rate_limit_and_parse_errors(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._WIKIMEDIA_STREAM_LOCK:
            chamber_module._WIKIMEDIA_STREAM_CACHE.clear()
            chamber_module._WIKIMEDIA_STREAM_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "WIKIMEDIA_EVENTSTREAMS_ENABLED", True)
        monkeypatch.setattr(
            chamber_module, "WIKIMEDIA_EVENTSTREAMS_POLL_INTERVAL_SECONDS", 0.0
        )
        monkeypatch.setattr(
            chamber_module, "WIKIMEDIA_EVENTSTREAMS_RATE_LIMIT_PER_POLL", 1
        )
        monkeypatch.setattr(
            chamber_module, "WIKIMEDIA_EVENTSTREAMS_STREAMS", "recentchange,page-create"
        )

        def _fake_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "events": [
                    {
                        "meta": {
                            "id": "evt-1",
                            "stream": "recentchange",
                            "dt": "2026-02-18T00:00:00Z",
                        },
                        "type": "edit",
                        "title": "Alpha",
                        "comment": "first",
                        "wiki": "enwiki",
                        "server_url": "https://en.wikipedia.org",
                    },
                    {
                        "meta": {
                            "id": "evt-1",
                            "stream": "recentchange",
                            "dt": "2026-02-18T00:00:00Z",
                        },
                        "type": "edit",
                        "title": "Alpha",
                        "comment": "duplicate",
                        "wiki": "enwiki",
                        "server_url": "https://en.wikipedia.org",
                    },
                    {
                        "meta": {
                            "id": "evt-2",
                            "stream": "page-create",
                            "dt": "2026-02-18T00:00:01Z",
                        },
                        "type": "create",
                        "title": "Beta",
                        "comment": "second",
                        "wiki": "enwiki",
                        "server_url": "https://en.wikipedia.org",
                    },
                ],
                "parse_errors": 2,
                "bytes_read": 321,
            }

        monkeypatch.setattr(chamber_module, "_wikimedia_fetch_sse_events", _fake_fetch)

        chamber_module._collect_wikimedia_stream_rows(vault)
        log_path, rows = chamber_module._load_wikimedia_stream_rows(vault, limit=200)
        assert log_path.endswith("wikimedia_stream.v1.jsonl")

        kinds = {str(row.get("kind", "")) for row in rows if isinstance(row, dict)}
        assert "wikimedia.stream.connected" in kinds
        assert "wikimedia.stream.poll" in kinds
        assert "wikimedia.stream.parse-error" in kinds
        assert "wikimedia.stream.dedupe" in kinds
        assert "wikimedia.stream.rate-limit" in kinds

        accepted = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("record", "")) == chamber_module.WIKIMEDIA_EVENT_RECORD
        ]
        assert len(accepted) == 1

def test_wikimedia_stream_collect_emits_pause_resume_events(monkeypatch: Any) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._WIKIMEDIA_STREAM_LOCK:
            chamber_module._WIKIMEDIA_STREAM_CACHE.clear()
            chamber_module._WIKIMEDIA_STREAM_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "WIKIMEDIA_EVENTSTREAMS_ENABLED", False)
        chamber_module._collect_wikimedia_stream_rows(vault)

        monkeypatch.setattr(chamber_module, "WIKIMEDIA_EVENTSTREAMS_ENABLED", True)
        monkeypatch.setattr(
            chamber_module, "WIKIMEDIA_EVENTSTREAMS_POLL_INTERVAL_SECONDS", 0.0
        )

        def _empty_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "events": [],
                "parse_errors": 0,
                "bytes_read": 0,
            }

        monkeypatch.setattr(chamber_module, "_wikimedia_fetch_sse_events", _empty_fetch)
        chamber_module._collect_wikimedia_stream_rows(vault)

        _, rows = chamber_module._load_wikimedia_stream_rows(vault, limit=200)
        kinds = [str(row.get("kind", "")) for row in rows if isinstance(row, dict)]
        assert "wikimedia.stream.paused" in kinds
        assert "wikimedia.stream.resumed" in kinds

def test_nws_alert_stream_collect_tracks_dedupe_rate_limit_and_parse_errors(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._NWS_ALERTS_LOCK:
            chamber_module._NWS_ALERTS_CACHE.clear()
            chamber_module._NWS_ALERTS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "NWS_ALERTS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "NWS_ALERTS_POLL_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(chamber_module, "NWS_ALERTS_RATE_LIMIT_PER_POLL", 1)

        def _fake_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "alerts": [
                    {
                        "id": "https://api.weather.gov/alerts/ALPHA",
                        "properties": {
                            "event": "Severe Thunderstorm Warning",
                            "headline": "Severe thunderstorm warning",
                            "areaDesc": "King County",
                            "severity": "Severe",
                            "urgency": "Immediate",
                            "certainty": "Observed",
                            "status": "Actual",
                            "sent": "2026-02-18T00:00:00Z",
                        },
                    },
                    {
                        "id": "https://api.weather.gov/alerts/ALPHA",
                        "properties": {
                            "event": "Severe Thunderstorm Warning",
                            "headline": "duplicate",
                            "areaDesc": "King County",
                            "status": "Actual",
                            "sent": "2026-02-18T00:00:00Z",
                        },
                    },
                    {
                        "id": "https://api.weather.gov/alerts/BETA",
                        "properties": {
                            "event": "Flood Watch",
                            "headline": "Flood watch",
                            "areaDesc": "Pierce County",
                            "status": "Actual",
                            "sent": "2026-02-18T00:01:00Z",
                        },
                    },
                ],
                "parse_errors": 1,
                "bytes_read": 432,
            }

        monkeypatch.setattr(chamber_module, "_nws_fetch_active_alerts", _fake_fetch)

        chamber_module._collect_nws_alert_rows(vault)
        log_path, rows = chamber_module._load_nws_alert_rows(vault, limit=220)
        assert log_path.endswith("nws_alerts.v1.jsonl")

        kinds = {str(row.get("kind", "")) for row in rows if isinstance(row, dict)}
        assert "nws.alerts.connected" in kinds
        assert "nws.alerts.poll" in kinds
        assert "nws.alerts.parse-error" in kinds
        assert "nws.alerts.dedupe" in kinds
        assert "nws.alerts.rate-limit" in kinds

        accepted = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("record", "")) == chamber_module.NWS_ALERT_RECORD
        ]
        assert len(accepted) == 1

def test_nws_alert_stream_collect_emits_pause_resume_events(monkeypatch: Any) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._NWS_ALERTS_LOCK:
            chamber_module._NWS_ALERTS_CACHE.clear()
            chamber_module._NWS_ALERTS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "NWS_ALERTS_ENABLED", False)
        chamber_module._collect_nws_alert_rows(vault)

        monkeypatch.setattr(chamber_module, "NWS_ALERTS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "NWS_ALERTS_POLL_INTERVAL_SECONDS", 0.0)

        def _empty_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "alerts": [],
                "parse_errors": 0,
                "bytes_read": 0,
            }

        monkeypatch.setattr(chamber_module, "_nws_fetch_active_alerts", _empty_fetch)
        chamber_module._collect_nws_alert_rows(vault)

        _, rows = chamber_module._load_nws_alert_rows(vault, limit=220)
        kinds = [str(row.get("kind", "")) for row in rows if isinstance(row, dict)]
        assert "nws.alerts.paused" in kinds
        assert "nws.alerts.resumed" in kinds

def test_swpc_alert_stream_collect_tracks_dedupe_rate_limit_and_parse_errors(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._SWPC_ALERTS_LOCK:
            chamber_module._SWPC_ALERTS_CACHE.clear()
            chamber_module._SWPC_ALERTS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "SWPC_ALERTS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "SWPC_ALERTS_POLL_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(chamber_module, "SWPC_ALERTS_RATE_LIMIT_PER_POLL", 1)

        def _fake_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "alerts": [
                    {
                        "message_type": "ALERT",
                        "message_code": "ALTK08",
                        "serial_number": "101",
                        "issue_datetime": "2026-02-18T00:00:00Z",
                        "message": "Geomagnetic conditions expected.",
                    },
                    {
                        "message_type": "ALERT",
                        "message_code": "ALTK08",
                        "serial_number": "101",
                        "issue_datetime": "2026-02-18T00:00:00Z",
                        "message": "duplicate",
                    },
                    {
                        "message_type": "WATCH",
                        "message_code": "WATA20",
                        "serial_number": "102",
                        "issue_datetime": "2026-02-18T00:01:00Z",
                        "message": "Solar flare watch.",
                    },
                ],
                "parse_errors": 1,
                "bytes_read": 384,
            }

        monkeypatch.setattr(chamber_module, "_swpc_fetch_alert_rows", _fake_fetch)

        chamber_module._collect_swpc_alert_rows(vault)
        log_path, rows = chamber_module._load_swpc_alert_rows(vault, limit=220)
        assert log_path.endswith("swpc_alerts.v1.jsonl")

        kinds = {str(row.get("kind", "")) for row in rows if isinstance(row, dict)}
        assert "swpc.alerts.connected" in kinds
        assert "swpc.alerts.poll" in kinds
        assert "swpc.alerts.parse-error" in kinds
        assert "swpc.alerts.dedupe" in kinds
        assert "swpc.alerts.rate-limit" in kinds

        accepted = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("record", "")) == chamber_module.SWPC_ALERT_RECORD
        ]
        assert len(accepted) == 1

def test_swpc_alert_stream_collect_emits_pause_resume_events(monkeypatch: Any) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._SWPC_ALERTS_LOCK:
            chamber_module._SWPC_ALERTS_CACHE.clear()
            chamber_module._SWPC_ALERTS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "SWPC_ALERTS_ENABLED", False)
        chamber_module._collect_swpc_alert_rows(vault)

        monkeypatch.setattr(chamber_module, "SWPC_ALERTS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "SWPC_ALERTS_POLL_INTERVAL_SECONDS", 0.0)

        def _empty_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "alerts": [],
                "parse_errors": 0,
                "bytes_read": 0,
            }

        monkeypatch.setattr(chamber_module, "_swpc_fetch_alert_rows", _empty_fetch)
        chamber_module._collect_swpc_alert_rows(vault)

        _, rows = chamber_module._load_swpc_alert_rows(vault, limit=220)
        kinds = [str(row.get("kind", "")) for row in rows if isinstance(row, dict)]
        assert "swpc.alerts.paused" in kinds
        assert "swpc.alerts.resumed" in kinds

def test_eonet_event_stream_collect_tracks_dedupe_rate_limit_and_parse_errors(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._EONET_EVENTS_LOCK:
            chamber_module._EONET_EVENTS_CACHE.clear()
            chamber_module._EONET_EVENTS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "EONET_EVENTS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "EONET_EVENTS_POLL_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(chamber_module, "EONET_EVENTS_RATE_LIMIT_PER_POLL", 1)

        def _fake_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "events": [
                    {
                        "id": "EONET-1",
                        "title": "Volcano unrest",
                        "link": "https://eonet.gsfc.nasa.gov/api/v3/events/EONET-1",
                        "categories": [{"id": "volcanoes", "title": "Volcanoes"}],
                        "geometry": [
                            {
                                "date": "2026-02-18T00:00:00Z",
                                "type": "Point",
                                "coordinates": [14.0, 40.8],
                            }
                        ],
                        "sources": [{"id": "USGS", "url": "https://example.test/usgs"}],
                    },
                    {
                        "id": "EONET-1",
                        "title": "Volcano unrest duplicate",
                        "categories": [{"id": "volcanoes", "title": "Volcanoes"}],
                        "geometry": [
                            {
                                "date": "2026-02-18T00:00:00Z",
                                "type": "Point",
                                "coordinates": [14.0, 40.8],
                            }
                        ],
                    },
                    {
                        "id": "EONET-2",
                        "title": "Wildfire activity",
                        "link": "https://eonet.gsfc.nasa.gov/api/v3/events/EONET-2",
                        "categories": [{"id": "wildfires", "title": "Wildfires"}],
                        "geometry": [
                            {
                                "date": "2026-02-18T00:05:00Z",
                                "type": "Point",
                                "coordinates": [-120.0, 45.0],
                            }
                        ],
                    },
                ],
                "parse_errors": 2,
                "bytes_read": 512,
            }

        monkeypatch.setattr(chamber_module, "_eonet_fetch_events", _fake_fetch)

        chamber_module._collect_eonet_event_rows(vault)
        log_path, rows = chamber_module._load_eonet_event_rows(vault, limit=220)
        assert log_path.endswith("eonet_events.v1.jsonl")

        kinds = {str(row.get("kind", "")) for row in rows if isinstance(row, dict)}
        assert "eonet.events.connected" in kinds
        assert "eonet.events.poll" in kinds
        assert "eonet.events.parse-error" in kinds
        assert "eonet.events.dedupe" in kinds
        assert "eonet.events.rate-limit" in kinds

        accepted = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("record", "")) == chamber_module.EONET_EVENT_RECORD
        ]
        assert len(accepted) == 1

def test_eonet_event_stream_collect_emits_pause_resume_events(monkeypatch: Any) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._EONET_EVENTS_LOCK:
            chamber_module._EONET_EVENTS_CACHE.clear()
            chamber_module._EONET_EVENTS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "EONET_EVENTS_ENABLED", False)
        chamber_module._collect_eonet_event_rows(vault)

        monkeypatch.setattr(chamber_module, "EONET_EVENTS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "EONET_EVENTS_POLL_INTERVAL_SECONDS", 0.0)

        def _empty_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "events": [],
                "parse_errors": 0,
                "bytes_read": 0,
            }

        monkeypatch.setattr(chamber_module, "_eonet_fetch_events", _empty_fetch)
        chamber_module._collect_eonet_event_rows(vault)

        _, rows = chamber_module._load_eonet_event_rows(vault, limit=220)
        kinds = [str(row.get("kind", "")) for row in rows if isinstance(row, dict)]
        assert "eonet.events.paused" in kinds
        assert "eonet.events.resumed" in kinds

def test_gibs_capabilities_parser_extracts_target_layer_metadata() -> None:
    from code.world_web import chamber as chamber_module

    capabilities_xml = """
<Capabilities xmlns="http://www.opengis.net/wmts/1.0">
  <Contents>
    <Layer>
      <Title>MODIS Terra True Color</Title>
      <Identifier>MODIS_Terra_CorrectedReflectance_TrueColor</Identifier>
      <Format>image/jpeg</Format>
      <Dimension>
        <Identifier>Time</Identifier>
        <Default>2026-02-18</Default>
        <Value>2026-02-16/2026-02-18/P1D</Value>
      </Dimension>
      <TileMatrixSetLink>
        <TileMatrixSet>250m</TileMatrixSet>
      </TileMatrixSetLink>
    </Layer>
    <Layer>
      <Title>Ignored Layer</Title>
      <Identifier>Other_Layer</Identifier>
      <Format>image/png</Format>
      <TileMatrixSetLink>
        <TileMatrixSet>500m</TileMatrixSet>
      </TileMatrixSetLink>
    </Layer>
  </Contents>
</Capabilities>
"""

    layers, parse_errors = chamber_module._gibs_layers_from_capabilities_xml(
        capabilities_xml,
        target_layers=["MODIS_Terra_CorrectedReflectance_TrueColor"],
        max_layers=4,
    )
    assert parse_errors == 0
    assert len(layers) == 1
    layer = layers[0]
    assert layer.get("layer_id") == "MODIS_Terra_CorrectedReflectance_TrueColor"
    assert layer.get("title") == "MODIS Terra True Color"
    assert layer.get("default_time") == "2026-02-18"
    assert layer.get("latest_time") == "2026-02-18"
    assert layer.get("tile_matrix_set") == "250m"

def test_gibs_layer_stream_collect_tracks_dedupe_rate_limit_and_parse_errors(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._GIBS_LAYERS_LOCK:
            chamber_module._GIBS_LAYERS_CACHE.clear()
            chamber_module._GIBS_LAYERS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "GIBS_LAYERS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "GIBS_LAYERS_POLL_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(chamber_module, "GIBS_LAYERS_RATE_LIMIT_PER_POLL", 1)
        monkeypatch.setattr(
            chamber_module,
            "GIBS_LAYERS_CAPABILITIES_ENDPOINT",
            "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities",
        )

        def _fake_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "layers": [
                    {
                        "layer_id": "MODIS_Terra_CorrectedReflectance_TrueColor",
                        "title": "MODIS Terra True Color",
                        "default_time": "2026-02-18",
                        "latest_time": "2026-02-18",
                        "tile_matrix_set": "250m",
                        "formats": ["image/jpeg"],
                    },
                    {
                        "layer_id": "MODIS_Terra_CorrectedReflectance_TrueColor",
                        "title": "duplicate",
                        "default_time": "2026-02-18",
                        "latest_time": "2026-02-18",
                        "tile_matrix_set": "250m",
                        "formats": ["image/jpeg"],
                    },
                    {
                        "layer_id": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
                        "title": "VIIRS SNPP True Color",
                        "default_time": "2026-02-18",
                        "latest_time": "2026-02-18",
                        "tile_matrix_set": "250m",
                        "formats": ["image/png"],
                    },
                ],
                "parse_errors": 1,
                "bytes_read": 654,
            }

        monkeypatch.setattr(
            chamber_module, "_gibs_fetch_capabilities_layers", _fake_fetch
        )

        chamber_module._collect_gibs_layer_rows(vault)
        log_path, rows = chamber_module._load_gibs_layer_rows(vault, limit=220)
        assert log_path.endswith("gibs_layers.v1.jsonl")

        kinds = {str(row.get("kind", "")) for row in rows if isinstance(row, dict)}
        assert "gibs.layers.connected" in kinds
        assert "gibs.layers.poll" in kinds
        assert "gibs.layers.parse-error" in kinds
        assert "gibs.layers.dedupe" in kinds
        assert "gibs.layers.rate-limit" in kinds

        accepted = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("record", "")) == chamber_module.GIBS_LAYER_RECORD
        ]
        assert len(accepted) == 1
        refs = accepted[0].get("refs", [])
        assert isinstance(refs, list)
        assert any("/wmts/epsg4326/best/" in str(value) for value in refs)

def test_gibs_layer_stream_collect_emits_pause_resume_and_compliance_events(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._GIBS_LAYERS_LOCK:
            chamber_module._GIBS_LAYERS_CACHE.clear()
            chamber_module._GIBS_LAYERS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "GIBS_LAYERS_ENABLED", False)
        chamber_module._collect_gibs_layer_rows(vault)

        monkeypatch.setattr(chamber_module, "GIBS_LAYERS_ENABLED", True)
        monkeypatch.setattr(chamber_module, "GIBS_LAYERS_POLL_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(
            chamber_module,
            "GIBS_LAYERS_CAPABILITIES_ENDPOINT",
            "http://example.com/wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities",
        )

        calls = {"count": 0}

        def _should_not_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            calls["count"] += 1
            return {
                "ok": True,
                "error": "",
                "layers": [],
                "parse_errors": 0,
                "bytes_read": 0,
            }

        monkeypatch.setattr(
            chamber_module,
            "_gibs_fetch_capabilities_layers",
            _should_not_fetch,
        )

        chamber_module._collect_gibs_layer_rows(vault)

        _, rows = chamber_module._load_gibs_layer_rows(vault, limit=220)
        kinds = [str(row.get("kind", "")) for row in rows if isinstance(row, dict)]
        assert "gibs.layers.paused" in kinds
        assert "gibs.layers.resumed" in kinds
        assert "gibs.layers.compliance" in kinds
        compliance_rows = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("kind", "")) == "gibs.layers.compliance"
        ]
        assert compliance_rows
        assert str(compliance_rows[-1].get("status", "")) == "blocked"
        assert calls["count"] == 0

def test_emsc_stream_collect_tracks_dedupe_rate_limit_and_parse_errors(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._EMSC_STREAM_LOCK:
            chamber_module._EMSC_STREAM_CACHE.clear()
            chamber_module._EMSC_STREAM_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "EMSC_STREAM_ENABLED", True)
        monkeypatch.setattr(chamber_module, "EMSC_STREAM_POLL_INTERVAL_SECONDS", 0.0)
        monkeypatch.setattr(chamber_module, "EMSC_STREAM_RATE_LIMIT_PER_POLL", 1)

        def _fake_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "events": [
                    {
                        "action": "create",
                        "data": {
                            "properties": {
                                "unid": "eq-1",
                                "time": "2026-02-18T00:00:00Z",
                                "mag": 4.8,
                                "flynn_region": "Ionian Sea",
                            },
                            "geometry": {"coordinates": [20.2, 37.8, 10.0]},
                        },
                    },
                    {
                        "action": "create",
                        "data": {
                            "properties": {
                                "unid": "eq-1",
                                "time": "2026-02-18T00:00:00Z",
                                "mag": 4.8,
                                "flynn_region": "Ionian Sea",
                            },
                            "geometry": {"coordinates": [20.2, 37.8, 10.0]},
                        },
                    },
                    {
                        "action": "create",
                        "data": {
                            "properties": {
                                "unid": "eq-2",
                                "time": "2026-02-18T00:00:03Z",
                                "mag": 5.1,
                                "flynn_region": "Aegean Sea",
                            },
                            "geometry": {"coordinates": [25.0, 38.5, 8.0]},
                        },
                    },
                ],
                "parse_errors": 2,
                "bytes_read": 512,
            }

        monkeypatch.setattr(chamber_module, "_emsc_fetch_ws_events", _fake_fetch)

        chamber_module._collect_emsc_stream_rows(vault)
        log_path, rows = chamber_module._load_emsc_stream_rows(vault, limit=220)
        assert log_path.endswith("emsc_stream.v1.jsonl")

        kinds = {str(row.get("kind", "")) for row in rows if isinstance(row, dict)}
        assert "emsc.stream.connected" in kinds
        assert "emsc.stream.poll" in kinds
        assert "emsc.stream.parse-error" in kinds
        assert "emsc.stream.dedupe" in kinds
        assert "emsc.stream.rate-limit" in kinds

        accepted = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("record", "")) == chamber_module.EMSC_EVENT_RECORD
        ]
        assert len(accepted) == 1

def test_emsc_stream_collect_emits_pause_resume_events(monkeypatch: Any) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)

        with chamber_module._EMSC_STREAM_LOCK:
            chamber_module._EMSC_STREAM_CACHE.clear()
            chamber_module._EMSC_STREAM_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        monkeypatch.setattr(chamber_module, "EMSC_STREAM_ENABLED", False)
        chamber_module._collect_emsc_stream_rows(vault)

        monkeypatch.setattr(chamber_module, "EMSC_STREAM_ENABLED", True)
        monkeypatch.setattr(chamber_module, "EMSC_STREAM_POLL_INTERVAL_SECONDS", 0.0)

        def _empty_fetch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "error": "",
                "events": [],
                "parse_errors": 0,
                "bytes_read": 0,
            }

        monkeypatch.setattr(chamber_module, "_emsc_fetch_ws_events", _empty_fetch)
        chamber_module._collect_emsc_stream_rows(vault)

        _, rows = chamber_module._load_emsc_stream_rows(vault, limit=220)
        kinds = [str(row.get("kind", "")) for row in rows if isinstance(row, dict)]
        assert "emsc.stream.paused" in kinds
        assert "emsc.stream.resumed" in kinds

def test_world_log_payload_includes_nws_and_emsc_rows_and_embeddings(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "eta_mu_part_64"
        _create_fixture_tree(part)

        chamber_module._append_nws_alert_rows(
            vault,
            [
                {
                    "record": chamber_module.NWS_ALERT_RECORD,
                    "id": "nws:test-1",
                    "ts": "2026-02-18T00:00:00Z",
                    "source": "nws_alerts",
                    "kind": "nws.alert.flood-watch",
                    "status": "actual",
                    "title": "Flood watch",
                    "detail": "fixture alert",
                    "refs": ["https://api.weather.gov/alerts/TEST"],
                    "tags": ["nws", "flood-watch"],
                    "meta": {},
                }
            ],
        )
        chamber_module._append_emsc_stream_rows(
            vault,
            [
                {
                    "record": chamber_module.EMSC_EVENT_RECORD,
                    "id": "emsc:test-1",
                    "ts": "2026-02-18T00:00:01Z",
                    "source": "emsc_stream",
                    "kind": "emsc.earthquake",
                    "status": "recorded",
                    "title": "M 4.9 earthquake",
                    "detail": "fixture quake",
                    "refs": [
                        "https://www.seismicportal.eu/eventdetails.html?unid=eq-1"
                    ],
                    "tags": ["emsc", "earthquake"],
                    "meta": {},
                }
            ],
        )

        monkeypatch.setattr(
            chamber_module, "_collect_emsc_stream_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_nws_alert_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_wikimedia_stream_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_gibs_layer_rows", lambda _vault: None
        )

        payload = build_world_log_payload(part, vault, limit=80)
        events = payload.get("events", [])
        nws_events = [
            row
            for row in events
            if isinstance(row, dict) and str(row.get("source", "")) == "nws_alerts"
        ]
        emsc_events = [
            row
            for row in events
            if isinstance(row, dict) and str(row.get("source", "")) == "emsc_stream"
        ]
        assert nws_events
        assert emsc_events
        assert str(nws_events[0].get("embedding_id", "")).startswith("world-event:")
        assert str(emsc_events[0].get("embedding_id", "")).startswith("world-event:")

def test_world_log_payload_includes_swpc_and_eonet_rows_and_embeddings(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "eta_mu_part_64"
        _create_fixture_tree(part)

        chamber_module._append_swpc_alert_rows(
            vault,
            [
                {
                    "record": chamber_module.SWPC_ALERT_RECORD,
                    "id": "swpc:test-1",
                    "ts": "2026-02-18T00:02:00Z",
                    "source": "swpc_alerts",
                    "kind": "swpc.alert.alert",
                    "status": "actual",
                    "title": "ALERT ALTK08 #101",
                    "detail": "fixture swpc alert",
                    "refs": ["https://services.swpc.noaa.gov/products/alerts.json"],
                    "tags": ["swpc", "space-weather"],
                    "meta": {},
                }
            ],
        )
        chamber_module._append_eonet_event_rows(
            vault,
            [
                {
                    "record": chamber_module.EONET_EVENT_RECORD,
                    "id": "eonet:test-1",
                    "ts": "2026-02-18T00:03:00Z",
                    "source": "nasa_eonet",
                    "kind": "eonet.event.wildfires",
                    "status": "open",
                    "title": "Wildfire activity",
                    "detail": "fixture eonet event",
                    "refs": ["https://eonet.gsfc.nasa.gov/api/v3/events/EONET-1"],
                    "tags": ["eonet", "wildfires"],
                    "meta": {},
                }
            ],
        )

        monkeypatch.setattr(
            chamber_module, "_collect_swpc_alert_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_eonet_event_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_wikimedia_stream_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_gibs_layer_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_nws_alert_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_emsc_stream_rows", lambda _vault: None
        )

        payload = build_world_log_payload(part, vault, limit=80)
        events = payload.get("events", [])
        swpc_events = [
            row
            for row in events
            if isinstance(row, dict) and str(row.get("source", "")) == "swpc_alerts"
        ]
        eonet_events = [
            row
            for row in events
            if isinstance(row, dict) and str(row.get("source", "")) == "nasa_eonet"
        ]
        assert swpc_events
        assert eonet_events
        assert str(swpc_events[0].get("embedding_id", "")).startswith("world-event:")
        assert str(eonet_events[0].get("embedding_id", "")).startswith("world-event:")

def test_world_log_payload_includes_wikimedia_rows_and_embeddings(
    monkeypatch: Any,
) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "eta_mu_part_64"
        _create_fixture_tree(part)

        with chamber_module._WIKIMEDIA_STREAM_LOCK:
            chamber_module._WIKIMEDIA_STREAM_CACHE.clear()
            chamber_module._WIKIMEDIA_STREAM_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        chamber_module._append_wikimedia_stream_rows(
            vault,
            [
                {
                    "record": chamber_module.WIKIMEDIA_EVENT_RECORD,
                    "id": "wikimedia:test-1",
                    "ts": "2026-02-18T00:00:00Z",
                    "source": "wikimedia_eventstreams",
                    "kind": "wikimedia.edit",
                    "status": "recorded",
                    "title": "Alpha",
                    "detail": "fixture event",
                    "refs": ["https://en.wikipedia.org/wiki/Alpha"],
                    "tags": ["enwiki", "edit"],
                    "meta": {},
                }
            ],
        )

        monkeypatch.setattr(
            chamber_module, "_collect_wikimedia_stream_rows", lambda _vault: None
        )
        monkeypatch.setattr(
            chamber_module, "_collect_gibs_layer_rows", lambda _vault: None
        )

        payload = build_world_log_payload(part, vault, limit=60)
        events = payload.get("events", [])
        wiki_events = [
            row
            for row in events
            if isinstance(row, dict)
            and str(row.get("source", "")) == "wikimedia_eventstreams"
        ]
        assert wiki_events
        sample = wiki_events[0]
        assert str(sample.get("embedding_id", "")).startswith("world-event:")

def test_world_log_payload_includes_gibs_rows_and_embeddings(monkeypatch: Any) -> None:
    from code.world_web import chamber as chamber_module

    with tempfile.TemporaryDirectory() as td:
        vault = Path(td)
        part = vault / "eta_mu_part_64"
        _create_fixture_tree(part)

        with chamber_module._GIBS_LAYERS_LOCK:
            chamber_module._GIBS_LAYERS_CACHE.clear()
            chamber_module._GIBS_LAYERS_CACHE.update(
                {
                    "last_poll_monotonic": 0.0,
                    "connected": False,
                    "paused": False,
                    "seen_ids": {},
                }
            )

        chamber_module._append_gibs_layer_rows(
            vault,
            [
                {
                    "record": chamber_module.GIBS_LAYER_RECORD,
                    "id": "gibs:test-1",
                    "ts": "2026-02-18T00:00:00Z",
                    "source": "nasa_gibs",
                    "kind": "gibs.layer.modis-terra-correctedreflectance-truecolor",
                    "status": "recorded",
                    "title": "MODIS Terra True Color",
                    "detail": "fixture layer event",
                    "refs": [
                        "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/2026-02-18/250m/2/1/1.jpg"
                    ],
                    "tags": ["gibs", "nasa", "satellite"],
                    "meta": {},
                }
            ],
        )

        monkeypatch.setattr(
            chamber_module, "_collect_gibs_layer_rows", lambda _vault: None
        )

        payload = build_world_log_payload(part, vault, limit=60)
        events = payload.get("events", [])
        gibs_events = [
            row
            for row in events
            if isinstance(row, dict) and str(row.get("source", "")) == "nasa_gibs"
        ]
        assert gibs_events
        sample = gibs_events[0]
        assert str(sample.get("embedding_id", "")).startswith("world-event:")
