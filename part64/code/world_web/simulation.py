from __future__ import annotations
import os
import time
import math
import hashlib
import threading
import colorsys
import base64
import struct
import json
import re
import socket
import sys
import io
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import defaultdict
from array import array
from hashlib import sha1
from urllib.parse import urlparse, unquote
from urllib.request import Request, urlopen
from urllib.error import URLError

from .constants import (
    DAIMO_PROFILE_DEFS,
    DAIMO_FORCE_KAPPA,
    DAIMO_DAMPING,
    DAIMO_DT_SECONDS,
    DAIMO_MAX_TRACKED_ENTITIES,
    ENTITY_MANIFEST,
    _DAIMO_DYNAMICS_LOCK,
    _DAIMO_DYNAMICS_CACHE,
    _MIX_CACHE_LOCK,
    _MIX_CACHE,
    CANONICAL_NAMED_FIELD_IDS,
    FIELD_TO_PRESENCE,
    MAX_SIM_POINTS,
    WS_MAGIC,
    WEAVER_HOST_ENV,
    WEAVER_PORT,
    WEAVER_GRAPH_HEALTH_TIMEOUT_SECONDS,
    WEAVER_GRAPH_NODE_LIMIT,
    WEAVER_GRAPH_EDGE_LIMIT,
    WEAVER_GRAPH_FETCH_TIMEOUT_SECONDS,
    WEAVER_GRAPH_CACHE_SECONDS,
    ETA_MU_FIELD_KEYWORDS,
    ETA_MU_FILE_GRAPH_RECORD,
    ETA_MU_CRAWLER_GRAPH_RECORD,
    FILE_SENTINEL_PROFILE,
    FILE_ORGANIZER_PROFILE,
    HEALTH_SENTINEL_CPU_PROFILE,
    HEALTH_SENTINEL_GPU1_PROFILE,
    HEALTH_SENTINEL_GPU2_PROFILE,
    HEALTH_SENTINEL_NPU0_PROFILE,
    _WEAVER_GRAPH_CACHE_LOCK,
    _WEAVER_GRAPH_CACHE,
)
from .metrics import (
    _safe_float,
    _safe_int,
    _clamp01,
    _stable_ratio,
    _normalize_field_scores,
    _resource_monitor_snapshot,
    _INFLUENCE_TRACKER,
)
from .paths import _safe_rel_path, _eta_mu_substrate_root
from .db import (
    _normalize_embedding_vector,
    _load_embeddings_db_state,
    _get_chroma_collection,
    _cosine_similarity,
    _load_eta_mu_knowledge_entries,
)


def _world_web_symbol(name: str, default: Any) -> Any:
    module = sys.modules.get("code.world_web")
    if module is None:
        return default
    return getattr(module, name, default)


def _normalize_path_for_file_id(path_like: str) -> str:
    raw = str(path_like or "").strip().replace("\\", "/")
    if not raw:
        return ""
    parts: list[str] = []
    for token in raw.split("/"):
        piece = token.strip()
        if not piece or piece == ".":
            continue
        if piece == "..":
            if parts:
                parts.pop()
            continue
        parts.append(piece)
    return "/".join(parts)


def _file_id_for_path(path_like: str) -> str:
    norm = _normalize_path_for_file_id(path_like)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest() if norm else ""


def _stable_entity_id(prefix: str, seed: str, width: int = 20) -> str:
    token = hashlib.sha256(seed.encode("utf-8")).hexdigest()[: max(8, width)]
    return f"{prefix}:{token}"


def _field_scores_from_position(
    x: float,
    y: float,
    field_anchors: dict[str, tuple[float, float]],
) -> dict[str, float]:
    raw: dict[str, float] = {}
    for field_id in FIELD_TO_PRESENCE:
        anchor_x, anchor_y = field_anchors.get(field_id, (0.5, 0.5))
        dx, dy = x - anchor_x, y - anchor_y
        distance = math.sqrt((dx * dx) + (dy * dy))
        raw[field_id] = 1.0 / (0.04 + (distance * distance * 6.0))
    return _normalize_field_scores(raw)


def _daimoi_softmax_weights(
    rows: list[tuple[str, float]], *, temperature: float
) -> dict[str, float]:
    if not rows:
        return {}
    temp = max(0.05, _safe_float(temperature, 0.42))
    max_score = max(_safe_float(score, 0.0) for _, score in rows)
    expo = [
        (eid, math.exp((_safe_float(s, 0.0) - max_score) / temp)) for eid, s in rows
    ]
    total = sum(v for _, v in expo)
    return (
        {eid: v / total for eid, v in expo}
        if total > 0.0
        else {eid: 1.0 / len(expo) for eid, _ in expo}
    )


def _build_daimoi_state(
    heat_values: dict[str, Any],
    pain_field: dict[str, Any],
    *,
    queue_ratio: float = 0.0,
    resource_ratio: float = 0.0,
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    node_heat_rows = (
        pain_field.get("node_heat", []) if isinstance(pain_field, dict) else []
    )
    relations: dict[str, list[dict[str, Any]]] = {
        "霊/attend": [],
        "霊/push": [],
        "霊/link": [],
        "霊/bind": [],
    }

    if not node_heat_rows:
        return {
            "record": "ημ.daimoi.v1",
            "generated_at": generated_at,
            "glyph": "霊",
            "active": False,
            "pressure": {
                "queue_ratio": round(_clamp01(_safe_float(queue_ratio)), 4),
                "resource_ratio": round(_clamp01(_safe_float(resource_ratio)), 4),
            },
            "daimoi": [],
            "relations": relations,
            "entities": [],
            "physics": {
                "kappa": round(DAIMO_FORCE_KAPPA, 6),
                "damping": round(DAIMO_DAMPING, 6),
                "dt": round(DAIMO_DT_SECONDS, 6),
            },
        }

    region_heat, region_centers = {}, {}
    for row in heat_values.get("regions", []):
        rid = str(row.get("region_id", "")).strip()
        if rid:
            region_heat[rid] = _clamp01(
                _safe_float(row.get("value", row.get("heat", 0.0)))
            )
            region_centers[rid] = (
                _clamp01(_safe_float(row.get("x", 0.5))),
                _clamp01(_safe_float(row.get("y", 0.5))),
            )
    for row in heat_values.get("facts", []):
        rid = str(row.get("region_id", "")).strip()
        if rid:
            region_heat[rid] = max(
                region_heat.get(rid, 0.0), _clamp01(_safe_float(row.get("value")))
            )
            region_centers.setdefault(rid, (0.5, 0.5))

    entity_manifest_by_id = {
        str(row.get("id")): row
        for row in ENTITY_MANIFEST
        if str(row.get("id", "")).strip()
    }
    field_anchors = {}
    for fid, pid in FIELD_TO_PRESENCE.items():
        c = region_centers.get(fid)
        if c:
            field_anchors[fid] = c
        else:
            e = entity_manifest_by_id.get(pid, {})
            field_anchors[fid] = (
                _clamp01(_safe_float(e.get("x", 0.5))),
                _clamp01(_safe_float(e.get("y", 0.5))),
            )

    locate_by_entity = defaultdict(dict)
    for row in heat_values.get("locate", []):
        eid, rid = str(row.get("entity_id")), str(row.get("region_id"))
        if eid and rid:
            locate_by_entity[eid][rid] = max(
                locate_by_entity[eid].get(rid, 0.0),
                _clamp01(_safe_float(row.get("weight"))),
            )

    entities = []
    for row in node_heat_rows[:DAIMO_MAX_TRACKED_ENTITIES]:
        eid = str(row.get("node_id"))
        if not eid:
            continue
        x, y, h = (
            _clamp01(_safe_float(row.get("x", 0.5))),
            _clamp01(_safe_float(row.get("y", 0.5))),
            _clamp01(_safe_float(row.get("heat", 0.0))),
        )
        locate = dict(locate_by_entity.get(eid, {}))
        if not locate:
            locate = _field_scores_from_position(x, y, field_anchors)
        score = sum(
            _clamp01(_safe_float(w)) * _clamp01(_safe_float(region_heat.get(rid, 0.0)))
            for rid, w in locate.items()
        ) or (h * 0.1)
        entities.append(
            {
                "id": eid,
                "x": x,
                "y": y,
                "heat": h,
                "score": score,
                "mass": max(0.35, 0.8 + ((1.0 - h) * 2.2)),
                "locate": locate,
            }
        )

    entities.sort(key=lambda r: (-_safe_float(r.get("score")), str(r.get("id"))))
    entity_by_id = {str(r["id"]): r for r in entities}
    pressure = _clamp01(
        (_clamp01(_safe_float(queue_ratio)) * 0.58)
        + (_clamp01(_safe_float(resource_ratio)) * 0.42)
    )
    budget_scale = max(0.4, 1.0 - (pressure * 0.5))

    daimo_rows, force_by_entity = [], defaultdict(lambda: [0.0, 0.0])
    for idx, profile in enumerate(DAIMO_PROFILE_DEFS):
        did = str(profile.get("id", f"daimo:{idx}"))
        ctx, dw = (
            str(profile.get("ctx", "世")),
            _clamp01(_safe_float(profile.get("w", 0.88))),
        )
        budget = max(
            1, int(round(_safe_float(profile.get("base_budget", 6.0)) * budget_scale))
        )
        temp, top_k = (
            max(0.05, _safe_float(profile.get("temperature", 0.42))),
            max(1, min(6, budget // 2)),
        )

        scored = []
        for e_idx, e in enumerate(entities[:64]):
            eid, eh = e["id"], e["heat"]
            gain = {
                "主": 1.08 + eh * 0.08,
                "己": 0.95 + (1 - eh) * 0.08,
                "汝": 0.98 + abs(e["x"] - 0.5) * 0.06,
                "彼": 0.98 + abs(e["y"] - 0.5) * 0.06,
            }.get(ctx, 1.02 + pressure * 0.08)
            s = max(
                0.0,
                (e["score"] * gain)
                + (_stable_ratio(f"{did}|{eid}", e_idx) - 0.5) * 0.12,
            )
            if s > 0.0:
                scored.append((eid, s))

        scored.sort(key=lambda r: (-r[1], r[0]))
        top = scored[:top_k]
        attn = _daimoi_softmax_weights(top, temperature=temp)
        counts = {"attend": 0, "push": 0, "bind": 0, "link": 0}

        for eid, escore in top:
            aw = _clamp01(_safe_float(attn.get(eid)))
            if aw <= 0.0:
                continue
            counts["attend"] += 1
            relations["霊/attend"].append(
                {
                    "id": _stable_entity_id("edge", f"{did}|{eid}|霊/attend"),
                    "rel": "霊/attend",
                    "daimo_id": did,
                    "entity_id": eid,
                    "w": round(aw, 6),
                    "score": round(escore, 6),
                }
            )
            e = entity_by_id[eid]
            vx = vy = 0.0
            best_r, best_s = "", 0.0
            for rid, lw in e["locate"].items():
                sig = _clamp01(_safe_float(lw)) * region_heat.get(str(rid), 0.0)
                if sig <= 0.0:
                    continue
                bx, by = field_anchors.get(str(rid), (0.5, 0.5))
                dx, dy = bx - e["x"], by - e["y"]
                mag = math.sqrt(dx**2 + dy**2)
                if mag > 1e-8:
                    vx += sig * (dx / mag)
                    vy += sig * (dy / mag)
                if sig > best_s:
                    best_s, best_r = sig, str(rid)
            v_mag = math.sqrt(vx**2 + vy**2)
            dx, dy = (vx / v_mag, vy / v_mag) if v_mag > 1e-8 else (0.0, 0.0)
            fx, fy = DAIMO_FORCE_KAPPA * dw * aw * dx, DAIMO_FORCE_KAPPA * dw * aw * dy
            if abs(fx) + abs(fy) > 1e-10:
                force_by_entity[eid][0] += fx
                force_by_entity[eid][1] += fy
            counts["push"] += 1
            relations["霊/push"].append(
                {
                    "id": _stable_entity_id("edge", f"{did}|{eid}|霊/push"),
                    "rel": "霊/push",
                    "daimo_id": did,
                    "entity_id": eid,
                    "region_id": best_r,
                    "fx": round(fx, 8),
                    "fy": round(fy, 8),
                    "w": round(aw, 6),
                }
            )

        if len(top) >= 2:
            counts["link"] += 1
            relations["霊/link"].append(
                {
                    "id": _stable_entity_id(
                        "edge", f"{did}|{top[0][0]}|{top[1][0]}|霊/link"
                    ),
                    "rel": "霊/link",
                    "daimo_id": did,
                    "entity_a": top[0][0],
                    "entity_b": top[1][0],
                    "w": round(math.sqrt(attn[top[0][0]] * attn[top[1][0]]), 6),
                }
            )

        daimo_rows.append(
            {
                "id": did,
                "name": str(profile.get("name", did)),
                "ctx": ctx,
                "state": "idle"
                if not top
                else ("move" if counts["push"] > 0 else "seek"),
                "budget": float(budget),
                "w": round(dw, 4),
                "at_iso": generated_at,
                "emitted": {**counts, "total": sum(counts.values())},
            }
        )

    e_rows = [
        {
            "id": eid,
            "x": round(e["x"], 4),
            "y": round(e["y"], 4),
            "heat": round(e["heat"], 4),
            "score": round(e["score"], 6),
            "mass": round(e["mass"], 6),
            "force": {
                "fx": round(force_by_entity[eid][0], 8),
                "fy": round(force_by_entity[eid][1], 8),
                "magnitude": round(
                    math.sqrt(
                        force_by_entity[eid][0] ** 2 + force_by_entity[eid][1] ** 2
                    ),
                    8,
                ),
            },
        }
        for eid, e in entity_by_id.items()
    ]
    return {
        "record": "ημ.daimoi.v1",
        "generated_at": generated_at,
        "glyph": "霊",
        "active": bool(relations["霊/attend"]),
        "pressure": {
            "queue_ratio": round(_clamp01(_safe_float(queue_ratio)), 4),
            "resource_ratio": round(_clamp01(_safe_float(resource_ratio)), 4),
            "blend": round(pressure, 4),
        },
        "daimoi": daimo_rows,
        "relations": relations,
        "entities": e_rows,
        "physics": {
            "kappa": round(DAIMO_FORCE_KAPPA, 6),
            "damping": round(DAIMO_DAMPING, 6),
            "dt": round(DAIMO_DT_SECONDS, 6),
        },
    }


def _apply_daimoi_dynamics_to_pain_field(
    pain_field: dict[str, Any], daimoi_state: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(pain_field, dict):
        return {}
    node_heat_rows = pain_field.get("node_heat", [])
    relations = (
        daimoi_state.get("relations", {}) if isinstance(daimoi_state, dict) else {}
    )
    push_rows = relations.get("霊/push", []) if isinstance(relations, dict) else []
    force_by_entity = defaultdict(lambda: [0.0, 0.0])
    for row in push_rows:
        eid = str(row.get("entity_id")).strip()
        if eid:
            force_by_entity[eid][0] += _safe_float(row.get("fx"))
            force_by_entity[eid][1] += _safe_float(row.get("fy"))

    physics = daimoi_state.get("physics", {}) if isinstance(daimoi_state, dict) else {}
    dt, damping = (
        max(0.02, min(0.4, _safe_float(physics.get("dt", DAIMO_DT_SECONDS)))),
        max(0.0, min(0.99, _safe_float(physics.get("damping", DAIMO_DAMPING)))),
    )
    updated_rows, active_ids, now_mono = [], set(), time.monotonic()

    with _DAIMO_DYNAMICS_LOCK:
        cache = _DAIMO_DYNAMICS_CACHE.get("entities", {})
        for row in node_heat_rows:
            eid = str(row.get("node_id")).strip()
            if not eid:
                updated_rows.append(dict(row))
                continue
            active_ids.add(eid)
            bx, by, h = (
                _clamp01(_safe_float(row.get("x", 0.5))),
                _clamp01(_safe_float(row.get("y", 0.5))),
                _clamp01(_safe_float(row.get("heat", 0.0))),
            )
            mass, c = max(0.35, 0.7 + ((1.0 - h) * 2.0)), cache.get(eid, {})
            px, py, pvx, pvy = (
                _clamp01(_safe_float(c.get("x", bx))),
                _clamp01(_safe_float(c.get("y", by))),
                _safe_float(c.get("vx")),
                _safe_float(c.get("vy")),
            )
            fx, fy = (
                force_by_entity[eid][0] + (bx - px) * 0.18,
                force_by_entity[eid][1] + (by - py) * 0.18,
            )
            nvx, nvy = (
                (pvx * damping) + ((dt / mass) * fx),
                (pvy * damping) + ((dt / mass) * fy),
            )
            nx, ny = _clamp01(px + (dt * nvx)), _clamp01(py + (dt * nvy))
            cache[eid] = {"x": nx, "y": ny, "vx": nvx, "vy": nvy, "ts": now_mono}
            updated_rows.append(
                {
                    **row,
                    "x": round(nx, 4),
                    "y": round(ny, 4),
                    "vx": round(nvx, 6),
                    "vy": round(nvy, 6),
                    "speed": round(math.sqrt(nvx**2 + nvy**2), 6),
                }
            )
        stale = now_mono - 120.0
        for k in list(cache.keys()):
            if k not in active_ids and _safe_float(cache[k].get("ts")) < stale:
                cache.pop(k, None)
        _DAIMO_DYNAMICS_CACHE["entities"] = cache

    return {
        **pain_field,
        "node_heat": updated_rows,
        "motion": {
            "record": "ημ.daimoi-motion.v1",
            "glyph": "霊",
            "active": bool(force_by_entity),
            "dt": round(dt, 6),
            "damping": round(damping, 6),
            "entity_count": len(updated_rows),
            "forced_entities": len(force_by_entity),
        },
    }


def _load_test_failures_from_path(candidate: Path) -> list[dict[str, Any]]:
    try:
        text = candidate.read_text("utf-8")
    except:
        return []
    if candidate.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
        if candidate.suffix.lower() == ".json":
            try:
                return _coerce_test_failure_rows(json.loads(text))
            except:
                return []
        rows = []
        for line in text.splitlines():
            try:
                rows.extend(_coerce_test_failure_rows(json.loads(line)))
            except:
                continue
        return rows
    return _parse_test_failures_text(text)


def _coerce_test_failure_rows(payload: Any) -> list[dict[str, Any]]:
    rows = []

    def _add(item):
        if isinstance(item, dict):
            rows.append(dict(item))
        elif str(item).strip():
            rows.append({"name": str(item).strip(), "status": "failed"})

    if isinstance(payload, list):
        for i in payload:
            _add(i)
    elif isinstance(payload, dict):
        for k in ("failures", "failed_tests", "failing_tests", "tests"):
            if isinstance(payload.get(k), list):
                for i in payload[k]:
                    if k == "tests" and isinstance(i, dict):
                        s = str(i.get("status") or i.get("outcome") or "").lower()
                        if s in {"failed", "error", "failing", "xfailed"}:
                            r = dict(i)
                            r.setdefault("status", s or "failed")
                            r.setdefault(
                                "name",
                                str(
                                    i.get("nodeid")
                                    or i.get("test")
                                    or i.get("id")
                                    or ""
                                ),
                            )
                            rows.append(r)
                    else:
                        _add(i)
                if rows:
                    return rows
        n = str(
            payload.get("name") or payload.get("test") or payload.get("nodeid") or ""
        ).strip()
        if n:
            r = dict(payload)
            r.setdefault("status", "failed")
            rows.append(r)
    return rows


def _parse_test_failures_text(raw_text: str) -> list[dict[str, Any]]:
    rows = []
    for line in raw_text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        n, sep, c = line.partition("|")
        if n.strip():
            row: dict[str, Any] = {"name": n.strip(), "status": "failed"}
            if sep:
                row["covered_files"] = [
                    t.strip() for t in re.split(r"[,\s]+", c.strip()) if t.strip()
                ]
            rows.append(row)
    return rows


def _load_test_coverage_from_path(
    candidate: Path, part_root: Path, vault_root: Path
) -> dict[str, Any]:
    try:
        text = candidate.read_text("utf-8")
    except:
        return {}
    if candidate.suffix.lower() == ".json":
        try:
            p = json.loads(text)
            return p if isinstance(p, dict) else {}
        except:
            return {}
    if candidate.name.lower().endswith(".info") and "lcov" in candidate.name.lower():
        return _parse_lcov_payload(text, part_root, vault_root)
    return {}


def _parse_lcov_payload(text: str, part_root: Path, vault_root: Path) -> dict[str, Any]:
    files, by_test_sets, by_test_spans = {}, defaultdict(set), defaultdict(list)
    cur_t = cur_s = ""
    da_f = da_h = lf = lh = 0
    cur_hits = []

    def _flush():
        nonlocal cur_s, da_f, da_h, lf, lh, cur_hits
        if not cur_s:
            return
        n = _normalize_coverage_source_path(cur_s, part_root, vault_root)
        if n:
            e = files.setdefault(
                n,
                {
                    "file_id": _file_id_for_path(n),
                    "lines_found": 0,
                    "lines_hit": 0,
                    "tests": [],
                },
            )
            e["lines_found"] += lf or da_f
            e["lines_hit"] += lh or da_h
            if cur_t.strip():
                by_test_sets[cur_t.strip()].add(n)
                if cur_t.strip() not in e["tests"]:
                    e["tests"].append(cur_t.strip())
                for sp in _line_hits_to_spans(cur_hits):
                    by_test_spans[cur_t.strip()].append(
                        {
                            "file": n,
                            "start_line": sp["start_line"],
                            "end_line": sp["end_line"],
                            "hits": sp["hits"],
                            "weight": 1.0,
                        }
                    )
        cur_s = ""
        da_f = da_h = lf = lh = 0
        cur_hits = []

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("TN:"):
            cur_t = line[3:].strip()
        elif line.startswith("SF:"):
            _flush()
            cur_s = line[3:].strip()
        elif line == "end_of_record":
            _flush()
        elif cur_s:
            if line.startswith("DA:"):
                p = line[3:].split(",")
                ln, h = int(_safe_float(p[0])), int(_safe_float(p[1]))
                cur_hits.append((ln, h))
                da_f += 1
                da_h += 1 if h > 0 else 0
            elif line.startswith("LF:"):
                lf = int(_safe_float(line[3:]))
            elif line.startswith("LH:"):
                lh = int(_safe_float(line[3:]))
    _flush()
    f_pay = {
        k: {
            **v,
            "line_rate": round(_clamp01(v["lines_hit"] / v["lines_found"]), 6)
            if v["lines_found"] > 0
            else 0.0,
        }
        for k, v in files.items()
    }
    return {
        "record": "ημ.test-coverage.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "lcov",
        "files": f_pay,
        "by_test": {k: sorted(list(v)) for k, v in by_test_sets.items()},
        "by_test_spans": dict(by_test_spans),
        "hottest_files": sorted(
            f_pay.keys(), key=lambda k: (f_pay[k].get("line_rate", 1.0), k)
        ),
    }


def _normalize_coverage_source_path(raw: str, part_root: Path, vault_root: Path) -> str:
    s = str(raw or "").strip()
    if s.startswith("file://"):
        s = unquote(urlparse(s).path)
    p = Path(s.strip())
    if p.is_absolute():
        try:
            r = p.resolve(strict=False)
        except:
            r = p
        for root in (part_root, vault_root):
            try:
                return _normalize_path_for_file_id(str(r.relative_to(root.resolve())))
            except:
                continue
    return _normalize_path_for_file_id(s)


def _line_hits_to_spans(hits: list[tuple[int, int]]) -> list[dict[str, Any]]:
    sorted_hits = sorted([(ln, h) for ln, h in hits if h > 0], key=lambda r: r[0])
    if not sorted_hits:
        return []
    spans, sl, pl, t = [], sorted_hits[0][0], sorted_hits[0][0], sorted_hits[0][1]
    for ln, h in sorted_hits[1:]:
        if ln <= pl + 1:
            pl, t = ln, t + h
        else:
            spans.append({"start_line": sl, "end_line": pl, "hits": t})
            sl, pl, t = ln, ln, h
    spans.append({"start_line": sl, "end_line": pl, "hits": t})
    return spans


def _extract_coverage_spans(raw: Any) -> list[dict[str, Any]]:
    spans = []

    def _walk(item, fp, fw):
        if isinstance(item, str):
            n = _normalize_path_for_file_id(item)
            if n:
                spans.append(
                    {
                        "path": n,
                        "start_line": 1,
                        "end_line": 1,
                        "symbol": "",
                        "weight": fw,
                    }
                )
        elif isinstance(item, list):
            for s in item:
                _walk(s, fp, fw)
        elif isinstance(item, dict):
            p = next(
                (
                    item.get(k)
                    for k in ("file", "path", "source")
                    if isinstance(item.get(k), str)
                ),
                fp,
            )
            w = _safe_float(
                next(
                    (item.get(k) for k in ("w", "weight") if item.get(k) is not None),
                    fw,
                )
            )
            for k in ("spans", "files", "coverage"):
                if item.get(k):
                    _walk(item[k], p, w)
            if p and not any(item.get(k) for k in ("spans", "files", "coverage")):
                spans.append(
                    {
                        "path": _normalize_path_for_file_id(p),
                        "start_line": int(_safe_int(item.get("start_line", 1))),
                        "end_line": int(_safe_int(item.get("end_line", 1))),
                        "symbol": str(item.get("symbol", "")),
                        "weight": w,
                    }
                )

    _walk(raw, "", 1.0)
    return spans


def _load_test_signal_artifacts(
    part_root: Path, vault_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    f: list[dict[str, Any]] = []
    for c in [
        part_root / "world_state" / "failing_tests.txt",
        part_root / "world_state" / "failing_tests.json",
        part_root / ".opencode" / "runtime" / "failing_tests.json",
        vault_root / ".opencode" / "runtime" / "failing_tests.json",
    ]:
        if c.exists():
            rows = _load_test_failures_from_path(c)
            if rows:
                f = rows
                break
    cov: dict[str, Any] = {}
    for c in [
        part_root / "coverage" / "lcov.info",
        part_root / "world_state" / "test_coverage.json",
    ]:
        if c.exists():
            p = _load_test_coverage_from_path(c, part_root, vault_root)
            if p:
                cov = p
                break
    return f, cov


def _build_logical_graph(catalog: dict[str, Any]) -> dict[str, Any]:
    file_graph = catalog.get("file_graph") if isinstance(catalog, dict) else {}
    truth_state = catalog.get("truth_state") if isinstance(catalog, dict) else {}
    if not isinstance(file_graph, dict):
        file_graph = {}
    if not isinstance(truth_state, dict):
        truth_state = {}

    file_nodes_raw = file_graph.get("file_nodes", [])
    if not isinstance(file_nodes_raw, list):
        file_nodes_raw = []
    file_edges_raw = file_graph.get("edges", [])
    if not isinstance(file_edges_raw, list):
        file_edges_raw = []
    tag_nodes_raw = file_graph.get("tag_nodes", [])
    if not isinstance(tag_nodes_raw, list):
        fallback_graph_nodes = file_graph.get("nodes", [])
        if isinstance(fallback_graph_nodes, list):
            tag_nodes_raw = [
                row
                for row in fallback_graph_nodes
                if isinstance(row, dict)
                and str(row.get("node_type", "")).strip().lower() == "tag"
            ]
        else:
            tag_nodes_raw = []

    claims_raw = truth_state.get("claims", [])
    if not isinstance(claims_raw, list) or not claims_raw:
        claim_single = truth_state.get("claim", {})
        if isinstance(claim_single, dict) and claim_single:
            claims_raw = [claim_single]
        else:
            claims_raw = []

    proof = truth_state.get("proof", {})
    if not isinstance(proof, dict):
        proof = {}
    proof_entries = proof.get("entries", [])
    if not isinstance(proof_entries, list):
        proof_entries = []
    required_kinds = proof.get("required_kinds", [])
    if not isinstance(required_kinds, list):
        required_kinds = []

    gate = truth_state.get("gate", {})
    if not isinstance(gate, dict):
        gate = {}

    graph_nodes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    joins_source_to_file: dict[str, str] = {}
    file_path_to_node: dict[str, str] = {}
    file_id_to_node: dict[str, str] = {}
    file_graph_node_to_logical: dict[str, str] = {}
    tag_graph_node_to_logical: dict[str, str] = {}
    tag_token_to_logical: dict[str, str] = {}

    test_failures = (
        catalog.get("test_failures", []) if isinstance(catalog, dict) else []
    )
    if not isinstance(test_failures, list):
        test_failures = []

    for file_node in file_nodes_raw:
        if not isinstance(file_node, dict):
            continue
        source_rel_path = str(
            file_node.get("source_rel_path")
            or file_node.get("archived_rel_path")
            or file_node.get("archive_rel_path")
            or file_node.get("name")
            or ""
        )
        normalized_path = _normalize_path_for_file_id(source_rel_path)
        if not normalized_path:
            continue
        file_id = _file_id_for_path(normalized_path)
        if not file_id:
            continue
        node_id = f"logical:file:{file_id[:24]}"
        source_uri = f"library:/{normalized_path}"
        file_path_to_node[normalized_path] = node_id
        file_id_to_node[file_id] = node_id
        joins_source_to_file[source_uri] = file_id
        file_graph_node_id = str(file_node.get("id", "")).strip()
        if file_graph_node_id:
            file_graph_node_to_logical[file_graph_node_id] = node_id
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "file",
                "label": str(
                    file_node.get("label") or file_node.get("name") or normalized_path
                ),
                "file_id": file_id,
                "source_uri": source_uri,
                "path": normalized_path,
                "x": round(_clamp01(_safe_float(file_node.get("x", 0.5), 0.5)), 4),
                "y": round(_clamp01(_safe_float(file_node.get("y", 0.5), 0.5)), 4),
                "confidence": 1.0,
                "provenance": {
                    "source_uri": source_uri,
                    "file_id": file_id,
                },
            }
        )

    for idx, tag_node in enumerate(tag_nodes_raw):
        if not isinstance(tag_node, dict):
            continue
        raw_tag = str(
            tag_node.get("tag")
            or tag_node.get("node_id")
            or tag_node.get("label")
            or ""
        ).strip()
        normalized_tag = re.sub(r"\s+", "_", raw_tag.lower())
        normalized_tag = re.sub(r"[^a-z0-9_]+", "", normalized_tag)
        normalized_tag = normalized_tag.strip("_")
        if not normalized_tag:
            continue
        node_id = (
            "logical:tag:"
            + hashlib.sha256(normalized_tag.encode("utf-8")).hexdigest()[:22]
        )
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "tag",
                "label": str(tag_node.get("label") or raw_tag or normalized_tag),
                "status": "active",
                "confidence": round(
                    _clamp01(
                        min(
                            1.0,
                            _safe_float(tag_node.get("member_count", 1), 1.0) / 8.0,
                        )
                    ),
                    4,
                ),
                "x": round(_clamp01(_safe_float(tag_node.get("x", 0.5), 0.5)), 4),
                "y": round(_clamp01(_safe_float(tag_node.get("y", 0.5), 0.5)), 4),
                "provenance": {
                    "tag": normalized_tag,
                    "member_count": int(
                        _safe_float(tag_node.get("member_count", 0), 0.0)
                    ),
                },
            }
        )
        graph_tag_id = str(tag_node.get("id", "")).strip()
        if graph_tag_id:
            tag_graph_node_to_logical[graph_tag_id] = node_id
        tag_token_to_logical[normalized_tag] = node_id

    tag_edge_seen: set[tuple[str, str, str]] = set()
    for edge in file_edges_raw:
        if not isinstance(edge, dict):
            continue
        kind = str(edge.get("kind", "")).strip().lower()
        if kind not in {"labeled_as", "relates_tag"}:
            continue
        source_key = str(edge.get("source", "")).strip()
        target_key = str(edge.get("target", "")).strip()
        source_id = file_graph_node_to_logical.get(
            source_key
        ) or tag_graph_node_to_logical.get(source_key)
        target_id = file_graph_node_to_logical.get(
            target_key
        ) or tag_graph_node_to_logical.get(target_key)
        if not source_id or not target_id or source_id == target_id:
            continue
        edge_key = (source_id, target_id, kind)
        if edge_key in tag_edge_seen:
            continue
        tag_edge_seen.add(edge_key)
        graph_edges.append(
            {
                "id": "logical:edge:tag:"
                + hashlib.sha256(
                    f"{source_id}|{target_id}|{kind}".encode("utf-8")
                ).hexdigest()[:20],
                "source": source_id,
                "target": target_id,
                "kind": kind,
                "weight": round(
                    _clamp01(_safe_float(edge.get("weight", 0.55), 0.55)), 4
                ),
            }
        )

    for file_node in file_nodes_raw:
        if not isinstance(file_node, dict):
            continue
        source_logical_id = file_graph_node_to_logical.get(str(file_node.get("id", "")))
        if not source_logical_id:
            continue
        tags_raw = file_node.get("tags", [])
        if not isinstance(tags_raw, list):
            continue
        for tag_raw in tags_raw:
            normalized_tag = re.sub(r"\s+", "_", str(tag_raw or "").strip().lower())
            normalized_tag = re.sub(r"[^a-z0-9_]+", "", normalized_tag).strip("_")
            if not normalized_tag:
                continue
            target_logical_id = tag_token_to_logical.get(normalized_tag)
            if not target_logical_id:
                continue
            edge_key = (source_logical_id, target_logical_id, "labeled_as")
            if edge_key in tag_edge_seen:
                continue
            tag_edge_seen.add(edge_key)
            graph_edges.append(
                {
                    "id": "logical:edge:tag:"
                    + hashlib.sha256(
                        f"{source_logical_id}|{target_logical_id}|fallback".encode(
                            "utf-8"
                        )
                    ).hexdigest()[:20],
                    "source": source_logical_id,
                    "target": target_logical_id,
                    "kind": "labeled_as",
                    "weight": 0.58,
                }
            )

    for idx, row in enumerate(test_failures):
        if not isinstance(row, dict):
            continue
        test_name = str(row.get("name") or row.get("test") or "").strip()
        if not test_name:
            continue
        test_id_seed = f"test:{test_name}|{idx}"
        test_node_id = f"logical:test:{hashlib.sha256(test_id_seed.encode('utf-8')).hexdigest()[:24]}"
        graph_nodes.append(
            {
                "id": test_node_id,
                "kind": "test",
                "label": test_name,
                "glyph": "試",
                "status": str(row.get("status", "failed")),
                "x": 0.5,
                "y": 0.5,
                "confidence": 1.0,
            }
        )

        covered_files = row.get("covered_files", [])
        if isinstance(covered_files, list):
            for path_item in covered_files:
                normalized_path = _normalize_path_for_file_id(str(path_item))
                target_node_id = file_path_to_node.get(normalized_path)
                if not target_node_id:
                    file_id = _file_id_for_path(normalized_path)
                    target_node_id = file_id_to_node.get(file_id)
                if target_node_id:
                    graph_edges.append(
                        {
                            "source": test_node_id,
                            "target": target_node_id,
                            "kind": "covers",
                            "weight": 0.8,
                        }
                    )

    world_log = catalog.get("world_log") if isinstance(catalog, dict) else {}
    world_log_events = (
        world_log.get("events", []) if isinstance(world_log, dict) else []
    )
    if not isinstance(world_log_events, list):
        world_log_events = []

    event_node_by_event_id: dict[str, str] = {}
    event_relation_pairs: set[tuple[str, str]] = set()
    event_link_count = 0
    event_relation_count = 0

    for idx, event in enumerate(world_log_events[:120]):
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            continue

        node_id = (
            "logical:event:" + hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:22]
        )
        event_node_by_event_id[event_id] = node_id

        x = _clamp01(
            _safe_float(
                event.get("x", _stable_ratio(event_id, idx * 11 + 3)),
                _stable_ratio(event_id, idx * 11 + 3),
            )
        )
        y = _clamp01(
            _safe_float(
                event.get("y", _stable_ratio(event_id, idx * 11 + 7)),
                _stable_ratio(event_id, idx * 11 + 7),
            )
        )
        importance = _clamp01(_safe_float(event.get("dominant_weight", 0.62), 0.62))

        graph_nodes.append(
            {
                "id": node_id,
                "kind": "event",
                "label": str(event.get("title") or event.get("kind") or event_id),
                "status": str(event.get("status", "recorded") or "recorded"),
                "confidence": round(importance, 4),
                "x": round(x, 4),
                "y": round(y, 4),
                "provenance": {
                    "event_id": event_id,
                    "source": str(event.get("source", "")),
                    "event_kind": str(event.get("kind", "")),
                    "ts": str(event.get("ts", "")),
                    "embedding_id": str(event.get("embedding_id", "")),
                    "refs": [
                        str(item) for item in event.get("refs", []) if str(item).strip()
                    ],
                },
            }
        )

        refs = [str(item) for item in event.get("refs", []) if str(item).strip()]
        for ref in refs[:6]:
            normalized_ref = _normalize_path_for_file_id(ref)
            if not normalized_ref:
                continue
            file_node_id = file_path_to_node.get(normalized_ref)
            if not file_node_id:
                file_id = _file_id_for_path(normalized_ref)
                file_node_id = file_id_to_node.get(file_id)
            if not file_node_id:
                continue
            graph_edges.append(
                {
                    "id": "logical:edge:mentions:"
                    + hashlib.sha256(
                        f"{node_id}|{file_node_id}|{normalized_ref}".encode("utf-8")
                    ).hexdigest()[:20],
                    "source": node_id,
                    "target": file_node_id,
                    "kind": "mentions",
                    "weight": 0.66,
                }
            )
            event_link_count += 1

    for event in world_log_events[:120]:
        if not isinstance(event, dict):
            continue
        source_event_id = str(event.get("id", "")).strip()
        source_node_id = event_node_by_event_id.get(source_event_id)
        if not source_node_id:
            continue
        relations_raw = event.get("relations", [])
        if not isinstance(relations_raw, list):
            continue
        for relation in relations_raw:
            if not isinstance(relation, dict):
                continue
            target_event_id = str(relation.get("event_id", "")).strip()
            target_node_id = event_node_by_event_id.get(target_event_id)
            if not target_node_id or target_node_id == source_node_id:
                continue
            pair = (
                source_event_id
                if source_event_id < target_event_id
                else target_event_id,
                target_event_id
                if source_event_id < target_event_id
                else source_event_id,
            )
            if pair in event_relation_pairs:
                continue
            event_relation_pairs.add(pair)
            graph_edges.append(
                {
                    "id": "logical:edge:correlates:"
                    + hashlib.sha256(
                        f"{source_node_id}|{target_node_id}".encode("utf-8")
                    ).hexdigest()[:20],
                    "source": source_node_id,
                    "target": target_node_id,
                    "kind": "correlates",
                    "weight": round(
                        _clamp01(_safe_float(relation.get("score", 0.4), 0.4)),
                        4,
                    ),
                }
            )
            event_relation_count += 1

    rule_nodes_by_kind: dict[str, str] = {}
    for idx, kind in enumerate(required_kinds):
        kind_text = str(kind).strip()
        if not kind_text:
            continue
        node_id = (
            f"logical:rule:{hashlib.sha256(kind_text.encode('utf-8')).hexdigest()[:20]}"
        )
        x = 0.18 + (_stable_ratio(kind_text, idx) * 0.2)
        y = 0.2 + (_stable_ratio(kind_text, idx + 19) * 0.35)
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "rule",
                "label": kind_text,
                "x": round(_clamp01(x), 4),
                "y": round(_clamp01(y), 4),
                "confidence": 1.0,
                "provenance": {"required_kind": kind_text},
            }
        )
        rule_nodes_by_kind[kind_text] = node_id

    fact_nodes: list[str] = []
    for idx, claim in enumerate(claims_raw):
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("id") or f"claim:{idx}")
        claim_text = str(claim.get("text") or claim_id)
        status = str(claim.get("status", "undecided")).strip() or "undecided"
        kappa = round(_clamp01(_safe_float(claim.get("kappa", 0.0), 0.0)), 4)
        node_id = (
            f"logical:fact:{hashlib.sha256(claim_id.encode('utf-8')).hexdigest()[:22]}"
        )
        orbit = 0.14 + (_stable_ratio(claim_id, idx) * 0.09)
        angle = _stable_ratio(claim_id, idx + 7) * math.tau
        x = 0.72 + math.cos(angle) * orbit
        y = 0.5 + math.sin(angle) * orbit
        proof_refs = [
            str(item).strip()
            for item in claim.get("proof_refs", [])
            if str(item).strip()
        ]
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "fact",
                "label": claim_text,
                "status": status,
                "confidence": kappa,
                "x": round(_clamp01(x), 4),
                "y": round(_clamp01(y), 4),
                "provenance": {
                    "claim_id": claim_id,
                    "proof_refs": proof_refs,
                },
            }
        )
        fact_nodes.append(node_id)

        for ref in proof_refs:
            normalized_ref = _normalize_path_for_file_id(ref)
            file_node_id = file_path_to_node.get(normalized_ref)
            if not file_node_id:
                continue
            graph_edges.append(
                {
                    "id": f"logical:edge:prove:{hashlib.sha256((file_node_id + node_id + ref).encode('utf-8')).hexdigest()[:20]}",
                    "source": file_node_id,
                    "target": node_id,
                    "kind": "proves",
                    "weight": 1.0,
                }
            )

    derivation_nodes: list[str] = []
    for idx, entry in enumerate(proof_entries):
        if not isinstance(entry, dict):
            continue
        ref = str(entry.get("ref", "")).strip()
        kind = str(entry.get("kind", "")).strip()
        present = bool(entry.get("present", False))
        detail = str(entry.get("detail", "")).strip()
        base = f"{kind}|{ref}|{idx}"
        node_id = f"logical:derivation:{hashlib.sha256(base.encode('utf-8')).hexdigest()[:20]}"
        x = 0.42 + (_stable_ratio(base, idx) * 0.22)
        y = 0.42 + (_stable_ratio(base, idx + 27) * 0.3)
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "derivation",
                "label": detail or ref or kind or f"derivation-{idx + 1}",
                "status": "present" if present else "missing",
                "confidence": 1.0 if present else 0.0,
                "x": round(_clamp01(x), 4),
                "y": round(_clamp01(y), 4),
                "provenance": {
                    "kind": kind,
                    "ref": ref,
                    "present": present,
                },
            }
        )
        derivation_nodes.append(node_id)

        rule_node = rule_nodes_by_kind.get(kind)
        if rule_node:
            graph_edges.append(
                {
                    "id": f"logical:edge:rule:{hashlib.sha256((rule_node + node_id).encode('utf-8')).hexdigest()[:20]}",
                    "source": rule_node,
                    "target": node_id,
                    "kind": "requires",
                    "weight": 0.9,
                }
            )

        if fact_nodes:
            target_fact = fact_nodes[idx % len(fact_nodes)]
            graph_edges.append(
                {
                    "id": f"logical:edge:derive:{hashlib.sha256((node_id + target_fact).encode('utf-8')).hexdigest()[:20]}",
                    "source": node_id,
                    "target": target_fact,
                    "kind": "derives",
                    "weight": 0.82 if present else 0.36,
                }
            )

        normalized_ref = _normalize_path_for_file_id(ref)
        file_node_id = file_path_to_node.get(normalized_ref)
        if file_node_id:
            graph_edges.append(
                {
                    "id": f"logical:edge:source:{hashlib.sha256((file_node_id + node_id + normalized_ref).encode('utf-8')).hexdigest()[:20]}",
                    "source": file_node_id,
                    "target": node_id,
                    "kind": "source",
                    "weight": 0.92,
                }
            )

    gate_target = str(gate.get("target") or "push-truth")
    gate_node_id = (
        f"logical:gate:{hashlib.sha256(gate_target.encode('utf-8')).hexdigest()[:20]}"
    )
    graph_nodes.append(
        {
            "id": gate_node_id,
            "kind": "gate",
            "label": gate_target,
            "status": "blocked" if bool(gate.get("blocked", True)) else "ready",
            "confidence": 1.0,
            "x": 0.76,
            "y": 0.54,
            "provenance": {"target": gate_target},
        }
    )

    for fact_id in fact_nodes:
        graph_edges.append(
            {
                "id": f"logical:edge:gate:{hashlib.sha256((fact_id + gate_node_id).encode('utf-8')).hexdigest()[:20]}",
                "source": fact_id,
                "target": gate_node_id,
                "kind": "feeds",
                "weight": 0.74,
            }
        )

    contradiction_nodes = 0
    gate_reasons = [
        str(item).strip() for item in gate.get("reasons", []) if str(item).strip()
    ]
    for idx, reason in enumerate(gate_reasons[:6]):
        node_id = f"logical:contradiction:{hashlib.sha256(reason.encode('utf-8')).hexdigest()[:20]}"
        x = 0.86 + (_stable_ratio(reason, idx) * 0.1)
        y = 0.42 + (_stable_ratio(reason, idx + 33) * 0.24)
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "contradiction",
                "label": reason,
                "status": "active",
                "confidence": 1.0,
                "x": round(_clamp01(x), 4),
                "y": round(_clamp01(y), 4),
                "provenance": {"reason": reason},
            }
        )
        graph_edges.append(
            {
                "id": f"logical:edge:block:{hashlib.sha256((node_id + gate_node_id).encode('utf-8')).hexdigest()[:20]}",
                "source": node_id,
                "target": gate_node_id,
                "kind": "blocks",
                "weight": 1.0,
            }
        )
        contradiction_nodes += 1

    for node in graph_nodes:
        if node.get("kind") != "fact" or str(node.get("status")) != "refuted":
            continue
        reason = str(node.get("label", "refuted-fact"))
        node_id = f"logical:contradiction:{hashlib.sha256((reason + ':fact').encode('utf-8')).hexdigest()[:20]}"
        x = _clamp01(_safe_float(node.get("x", 0.5), 0.5) + 0.08)
        y = _clamp01(_safe_float(node.get("y", 0.5), 0.5) + 0.04)
        graph_nodes.append(
            {
                "id": node_id,
                "kind": "contradiction",
                "label": reason,
                "status": "refuted",
                "confidence": 1.0,
                "x": round(x, 4),
                "y": round(y, 4),
                "provenance": {"from_fact": str(node.get("id", ""))},
            }
        )
        graph_edges.append(
            {
                "id": f"logical:edge:contradict:{hashlib.sha256((str(node.get('id')) + node_id).encode('utf-8')).hexdigest()[:20]}",
                "source": str(node.get("id", "")),
                "target": node_id,
                "kind": "contradicts",
                "weight": 1.0,
            }
        )
        contradiction_nodes += 1

    return {
        "record": "ημ.logical-graph.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nodes": graph_nodes,
        "edges": graph_edges,
        "joins": {
            "file_ids": sorted(file_id_to_node.keys()),
            "file_index": {
                path: _file_id_for_path(path)
                for path in sorted(file_path_to_node.keys())
            },
            "source_to_file": dict(sorted(joins_source_to_file.items())),
        },
        "stats": {
            "file_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "file"]
            ),
            "tag_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "tag"]
            ),
            "fact_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "fact"]
            ),
            "event_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "event"]
            ),
            "rule_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "rule"]
            ),
            "derivation_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "derivation"]
            ),
            "contradiction_nodes": contradiction_nodes,
            "gate_nodes": len(
                [node for node in graph_nodes if node.get("kind") == "gate"]
            ),
            "tag_edges": len(
                [
                    edge
                    for edge in graph_edges
                    if str(edge.get("kind", "")).strip().lower()
                    in {"labeled_as", "relates_tag"}
                ]
            ),
            "event_links": event_link_count,
            "event_relations": event_relation_count,
            "edge_count": len(graph_edges),
        },
    }


def _build_pain_field(
    catalog: dict[str, Any], logical_graph: dict[str, Any]
) -> dict[str, Any]:
    failures_raw = catalog.get("test_failures", []) if isinstance(catalog, dict) else []
    coverage_raw = catalog.get("test_coverage", {}) if isinstance(catalog, dict) else {}
    if not isinstance(failures_raw, list):
        failures_raw = []
    if not isinstance(coverage_raw, dict):
        coverage_raw = {}

    nodes = logical_graph.get("nodes", []) if isinstance(logical_graph, dict) else []
    edges = logical_graph.get("edges", []) if isinstance(logical_graph, dict) else []
    joins = logical_graph.get("joins", {}) if isinstance(logical_graph, dict) else {}
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    if not isinstance(joins, dict):
        joins = {}

    node_by_id = {
        str(node.get("id", "")): node
        for node in nodes
        if isinstance(node, dict) and str(node.get("id", "")).strip()
    }
    file_index = joins.get("file_index", {})
    if not isinstance(file_index, dict):
        file_index = {}
    file_id_to_path: dict[str, str] = {}
    for path_key, file_id_value in file_index.items():
        normalized_path = _normalize_path_for_file_id(str(path_key))
        file_id_key = str(file_id_value).strip()
        if normalized_path and file_id_key:
            file_id_to_path[file_id_key] = normalized_path
    file_id_to_node = {
        str(node.get("file_id", "")): str(node.get("id", ""))
        for node in nodes
        if isinstance(node, dict)
        and str(node.get("kind", "")) == "file"
        and str(node.get("file_id", "")).strip()
    }

    region_rows: list[dict[str, Any]] = []
    region_by_id: dict[str, dict[str, Any]] = {}
    region_by_file_id: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("kind", "")).strip() != "file":
            continue
        file_id = str(node.get("file_id", "")).strip()
        node_id = str(node.get("id", "")).strip()
        if not file_id or not node_id:
            continue
        region_key = str(node.get("path") or node.get("label") or file_id)
        region_seed = f"world-web|node|{node_id}|{region_key}"
        region_id = _stable_entity_id("region", region_seed)
        region_row: dict[str, Any] = {
            "region_id": region_id,
            "region_kind": "node",
            "region_key": region_key,
            "node_id": node_id,
            "file_id": file_id,
            "x": round(_clamp01(_safe_float(node.get("x", 0.5), 0.5)), 4),
            "y": round(_clamp01(_safe_float(node.get("y", 0.5), 0.5)), 4),
            "label": str(node.get("label", "")),
        }
        region_rows.append(region_row)
        region_by_id[region_id] = region_row
        region_by_file_id[file_id] = region_id

    region_rows.sort(key=lambda row: str(row.get("region_id", "")))

    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target:
            continue
        weight = _clamp01(_safe_float(edge.get("weight", 0.4), 0.4))
        adjacency[source].append((target, weight))
        adjacency[target].append((source, weight * 0.92))

    coverage_by_test = coverage_raw.get("by_test", {})
    if not isinstance(coverage_by_test, dict):
        coverage_by_test = {}

    coverage_by_test_spans = coverage_raw.get("by_test_spans", {})
    if not isinstance(coverage_by_test_spans, dict):
        coverage_by_test_spans = {}

    coverage_by_test_lower: dict[str, Any] = {}
    for key, value in coverage_by_test.items():
        normalized_key = str(key).strip().lower()
        if not normalized_key:
            continue
        coverage_by_test_lower[normalized_key] = value

    coverage_by_test_spans_lower: dict[str, Any] = {}
    for key, value in coverage_by_test_spans.items():
        normalized_key = str(key).strip().lower()
        if not normalized_key:
            continue
        coverage_by_test_spans_lower[normalized_key] = value

    hottest_files_raw = coverage_raw.get("hottest_files", [])
    hottest_files: list[str] = []
    if isinstance(hottest_files_raw, list):
        hottest_files = [str(path) for path in hottest_files_raw if str(path).strip()]
    if not hottest_files:
        files_metrics = coverage_raw.get("files", {})
        if isinstance(files_metrics, dict):
            scored_paths: list[tuple[str, float, float]] = []
            for path_key, metrics in files_metrics.items():
                path_text = str(path_key).strip()
                if not path_text:
                    continue
                line_rate = _clamp01(
                    _safe_float(
                        metrics.get("line_rate", 0.0)
                        if isinstance(metrics, dict)
                        else 0.0,
                        0.0,
                    )
                )
                lines_found = _safe_float(
                    metrics.get("lines_found", 0.0)
                    if isinstance(metrics, dict)
                    else 0.0,
                    0.0,
                )
                uncovered = max(0.0, 1.0 - line_rate)
                scored_paths.append((path_text, uncovered, lines_found))
            hottest_files = [
                path
                for path, _, _ in sorted(
                    scored_paths,
                    key=lambda row: (-row[1], -row[2], row[0]),
                )
            ]

    hottest_file_rank: dict[str, int] = {}
    for index, path_key in enumerate(hottest_files):
        normalized = _normalize_path_for_file_id(path_key)
        if normalized and normalized not in hottest_file_rank:
            hottest_file_rank[normalized] = index

    failing_tests: list[dict[str, Any]] = []
    test_span_weights: dict[tuple[str, str], float] = {}
    span_region_weights: dict[str, dict[str, float]] = defaultdict(dict)
    span_rows_by_id: dict[str, dict[str, Any]] = {}
    region_heat_raw: dict[str, float] = defaultdict(float)
    seeded_node_heat: dict[str, float] = defaultdict(float)

    for idx, row in enumerate(failures_raw):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "failed")).strip().lower()
        if status not in {"failed", "error", "xfailed", "failing"}:
            continue
        test_name = str(
            row.get("name") or row.get("test") or row.get("nodeid") or f"test-{idx + 1}"
        ).strip()
        if not test_name:
            continue
        message = str(row.get("message") or row.get("error") or "")

        coverage_sources: list[Any] = []
        for key in (
            "covered_spans",
            "spans",
            "coverage_spans",
            "covered_files",
            "files",
            "coverage",
        ):
            value = row.get(key)
            if value is not None:
                coverage_sources.append(value)

        from_coverage_spans = coverage_by_test_spans.get(test_name)
        if from_coverage_spans is None:
            from_coverage_spans = coverage_by_test_spans_lower.get(test_name.lower())
        if from_coverage_spans is not None:
            coverage_sources.append(from_coverage_spans)

        from_coverage = coverage_by_test.get(test_name)
        if from_coverage is None:
            from_coverage = coverage_by_test_lower.get(test_name.lower())
        if from_coverage is not None:
            coverage_sources.append(from_coverage)

        if not coverage_sources and hottest_files:
            coverage_sources.append(hottest_files[:3])

        normalized_spans: list[dict[str, Any]] = []
        for source in coverage_sources:
            normalized_spans.extend(_extract_coverage_spans(source))
        if not normalized_spans:
            continue

        severity = max(0.0, _safe_float(row.get("severity", 1.0), 1.0))
        signal_w = _clamp01(
            _safe_float(
                row.get(
                    "signal_w",
                    row.get("signal_weight", row.get("signal/w", 1.0)),
                ),
                1.0,
            )
        )
        suite_name = str(
            row.get("suite") or row.get("module") or row.get("file") or ""
        ).strip()
        runner_name = str(
            row.get("runner") or row.get("framework") or row.get("tool") or ""
        ).strip()
        test_id = _stable_entity_id(
            "test",
            f"{test_name}|{suite_name}|{runner_name}",
        )

        span_weights_for_test: dict[str, float] = defaultdict(float)
        covered_paths: set[str] = set()
        covered_file_ids: set[str] = set()

        for span in normalized_spans:
            path_value = _normalize_path_for_file_id(str(span.get("path") or ""))
            if not path_value:
                continue
            file_id = _file_id_for_path(path_value)
            if not file_id:
                continue

            start_line = max(1, _safe_int(span.get("start_line", 1), 1))
            end_line = max(
                start_line,
                _safe_int(span.get("end_line", start_line), start_line),
            )
            symbol = str(span.get("symbol", "")).strip()
            weight_raw = max(0.0, _safe_float(span.get("weight", 1.0), 1.0))
            if weight_raw <= 0.0:
                weight_raw = 1.0

            span_id = _stable_entity_id(
                "span",
                f"{file_id}|{start_line}|{end_line}|{symbol}",
            )
            span_weights_for_test[span_id] += weight_raw
            covered_paths.add(path_value)
            covered_file_ids.add(file_id)

            span_row = span_rows_by_id.get(span_id)
            if span_row is None:
                span_row = {
                    "id": span_id,
                    "file_id": file_id,
                    "path": path_value,
                    "start_line": start_line,
                    "end_line": end_line,
                    "symbol": symbol,
                }
                span_rows_by_id[span_id] = span_row

            region_id = region_by_file_id.get(file_id, "")
            if region_id:
                span_region_weights.setdefault(span_id, {})[region_id] = max(
                    span_region_weights.get(span_id, {}).get(region_id, 0.0),
                    1.0,
                )

        if not span_weights_for_test:
            continue

        total_span_weight = sum(span_weights_for_test.values())
        if total_span_weight <= 0.0:
            total_span_weight = float(len(span_weights_for_test))

        region_ids_for_test: set[str] = set()
        for span_id, raw_weight in sorted(span_weights_for_test.items()):
            edge_weight = (
                raw_weight / total_span_weight if total_span_weight > 0 else 0.0
            )
            if edge_weight <= 0.0:
                continue
            test_span_key = (test_id, span_id)
            test_span_weights[test_span_key] = max(
                test_span_weights.get(test_span_key, 0.0),
                edge_weight,
            )

            for region_id, region_weight in sorted(
                span_region_weights.get(span_id, {}).items()
            ):
                region_ids_for_test.add(region_id)
                contrib = severity * signal_w * edge_weight * max(0.0, region_weight)
                if contrib <= 0.0:
                    continue
                region_heat_raw[region_id] += contrib
                region_info = region_by_id.get(region_id, {})
                node_id = str(region_info.get("node_id", "")).strip()
                if node_id:
                    seeded_node_heat[node_id] += contrib

        span_ids_sorted = sorted(span_weights_for_test.keys())
        normalized_files = sorted(covered_paths)
        file_ids = sorted(covered_file_ids)
        if not normalized_files and hottest_files:
            normalized_files = [
                _normalize_path_for_file_id(path)
                for path in hottest_files[:3]
                if _normalize_path_for_file_id(path)
            ]

        failing_tests.append(
            {
                "id": test_id,
                "name": test_name,
                "status": status,
                "message": message,
                "severity": round(severity, 4),
                "signal_w": round(signal_w, 4),
                "failure_glyph": "破",
                "covered_files": normalized_files,
                "file_ids": file_ids,
                "span_ids": span_ids_sorted,
                "region_ids": sorted(region_ids_for_test),
            }
        )

    node_heat: dict[str, float] = {
        node_id: _clamp01(_safe_float(heat, 0.0))
        for node_id, heat in seeded_node_heat.items()
        if _safe_float(heat, 0.0) > 0.0
    }

    hop_decay = 0.58
    max_hops = 4
    current_frontier = sorted(node_heat.items(), key=lambda row: row[0])
    for _hop in range(max_hops):
        next_frontier: list[tuple[str, float]] = []
        for node_id, heat in current_frontier:
            if heat <= 0.02:
                continue
            for neighbor_id, edge_weight in adjacency.get(node_id, []):
                next_heat = _clamp01(heat * hop_decay * max(0.1, edge_weight))
                if next_heat <= 0.01:
                    continue
                if next_heat <= node_heat.get(neighbor_id, 0.0) + 0.004:
                    continue
                node_heat[neighbor_id] = next_heat
                next_frontier.append((neighbor_id, next_heat))
        current_frontier = next_frontier
        if not current_frontier:
            break

    def _heat_sort_key(item: tuple[str, float]) -> tuple[float, int, str]:
        node_id, heat_value = item
        node = node_by_id.get(node_id, {})
        if not isinstance(node, dict):
            node = {}
        file_id = str(node.get("file_id", "")).strip()
        path_value = file_id_to_path.get(file_id, "")
        if not path_value:
            path_value = _normalize_path_for_file_id(str(node.get("path", "")))
        rank = hottest_file_rank.get(path_value, 1_000_000)
        return (-_clamp01(_safe_float(heat_value, 0.0)), rank, node_id)

    heat_nodes: list[dict[str, Any]] = []
    for node_id, heat in sorted(node_heat.items(), key=_heat_sort_key):
        node = node_by_id.get(node_id, {})
        if not isinstance(node, dict):
            node = {}
        node_file_id = str(node.get("file_id", "")).strip()
        node_path = _normalize_path_for_file_id(str(node.get("path", "")))
        if node_file_id and node_path and node_file_id not in file_id_to_path:
            file_id_to_path[node_file_id] = node_path
        heat_nodes.append(
            {
                "node_id": node_id,
                "kind": str(node.get("kind", "unknown")),
                "heat": round(_clamp01(heat), 4),
                "x": round(_clamp01(_safe_float(node.get("x", 0.5), 0.5)), 4),
                "y": round(_clamp01(_safe_float(node.get("y", 0.5), 0.5)), 4),
                "file_id": str(node.get("file_id", "")),
                "label": str(node.get("label", "")),
            }
        )

    debug_target: dict[str, Any] = {
        "meaning": "DEBUG",
        "glyph": "診",
        "grounded": False,
        "source": "none",
        "node_id": "",
        "file_id": "",
        "region_id": "",
        "path": "",
        "label": "",
        "heat": 0.0,
        "x": 0.5,
        "y": 0.5,
        "reason": "no-active-failure-signal",
    }

    hottest_node = next(
        (
            row
            for row in heat_nodes
            if str(row.get("node_id", "")).strip()
            or str(row.get("file_id", "")).strip()
        ),
        None,
    )
    if isinstance(hottest_node, dict):
        node_id = str(hottest_node.get("node_id", "")).strip()
        file_id = str(hottest_node.get("file_id", "")).strip()
        node = node_by_id.get(node_id, {}) if node_id else {}
        if not isinstance(node, dict):
            node = {}

        path_value = file_id_to_path.get(file_id, "")
        if not path_value:
            path_value = _normalize_path_for_file_id(str(node.get("path", "")))
        if file_id and path_value and file_id not in file_id_to_path:
            file_id_to_path[file_id] = path_value

        label_value = str(hottest_node.get("label", "")).strip()
        if not label_value:
            label_value = str(node.get("label", "")).strip()
        if not label_value and path_value:
            label_value = Path(path_value).name

        debug_target = {
            "meaning": "DEBUG",
            "glyph": "診",
            "grounded": True,
            "source": "pain_field.max_heat",
            "node_id": node_id,
            "file_id": file_id,
            "region_id": region_by_file_id.get(file_id, ""),
            "path": path_value,
            "label": label_value,
            "heat": round(_clamp01(_safe_float(hottest_node.get("heat", 0.0), 0.0)), 4),
            "x": round(_clamp01(_safe_float(hottest_node.get("x", 0.5), 0.5)), 4),
            "y": round(_clamp01(_safe_float(hottest_node.get("y", 0.5), 0.5)), 4),
            "reason": "points-to-hottest-file",
        }
    elif hottest_files:
        fallback_path = _normalize_path_for_file_id(str(hottest_files[0]))
        fallback_file_id = _file_id_for_path(fallback_path) if fallback_path else ""
        fallback_node_id = file_id_to_node.get(fallback_file_id, "")
        fallback_node = node_by_id.get(fallback_node_id, {}) if fallback_node_id else {}
        if not isinstance(fallback_node, dict):
            fallback_node = {}
        if (
            fallback_file_id
            and fallback_path
            and fallback_file_id not in file_id_to_path
        ):
            file_id_to_path[fallback_file_id] = fallback_path

        label_value = str(fallback_node.get("label", "")).strip()
        if not label_value and fallback_path:
            label_value = Path(fallback_path).name

        debug_target = {
            "meaning": "DEBUG",
            "glyph": "診",
            "grounded": bool(fallback_path),
            "source": "coverage.hottest_files",
            "node_id": fallback_node_id,
            "file_id": fallback_file_id,
            "region_id": region_by_file_id.get(fallback_file_id, ""),
            "path": fallback_path,
            "label": label_value,
            "heat": 0.0,
            "x": round(_clamp01(_safe_float(fallback_node.get("x", 0.5), 0.5)), 4),
            "y": round(_clamp01(_safe_float(fallback_node.get("y", 0.5), 0.5)), 4),
            "reason": "fallback-to-coverage-hottest-file",
        }

    heat_regions: list[dict[str, Any]] = []
    for region_id, raw_heat in sorted(
        region_heat_raw.items(), key=lambda row: (-row[1], row[0])
    ):
        if raw_heat <= 0.0:
            continue
        region = region_by_id.get(region_id, {})
        heat_regions.append(
            {
                "region_id": region_id,
                "node_id": str(region.get("node_id", "")),
                "file_id": str(region.get("file_id", "")),
                "heat": round(_clamp01(_safe_float(raw_heat, 0.0)), 4),
                "heat_raw": round(max(0.0, _safe_float(raw_heat, 0.0)), 6),
                "glyph": "熱",
            }
        )

    span_rows = sorted(
        span_rows_by_id.values(),
        key=lambda row: (
            str(row.get("path", "")),
            int(row.get("start_line", 0)),
            int(row.get("end_line", 0)),
            str(row.get("id", "")),
        ),
    )

    test_covers_span_rows: list[dict[str, Any]] = []
    for (test_id, span_id), weight in sorted(
        test_span_weights.items(), key=lambda row: (row[0][0], row[0][1])
    ):
        test_covers_span_rows.append(
            {
                "id": _stable_entity_id(
                    "edge", f"{test_id}|{span_id}|覆/test-covers-span"
                ),
                "rel": "覆/test-covers-span",
                "test_id": test_id,
                "span_id": span_id,
                "w": round(_clamp01(_safe_float(weight, 0.0)), 6),
            }
        )

    span_maps_region_rows: list[dict[str, Any]] = []
    for span_id, region_weights in sorted(span_region_weights.items()):
        for region_id, weight in sorted(region_weights.items()):
            span_maps_region_rows.append(
                {
                    "id": _stable_entity_id(
                        "edge", f"{span_id}|{region_id}|覆/span-maps-to-region"
                    ),
                    "rel": "覆/span-maps-to-region",
                    "span_id": span_id,
                    "region_id": region_id,
                    "w": round(_clamp01(_safe_float(weight, 0.0)), 6),
                }
            )

    max_heat = max((row.get("heat", 0.0) for row in heat_nodes), default=0.0)
    return {
        "record": "ημ.pain-field.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active": bool(failing_tests),
        "decay": hop_decay,
        "hops": max_hops,
        "failing_tests": failing_tests,
        "spans": span_rows,
        "regions": region_rows,
        "relations": {
            "覆/test-covers-span": test_covers_span_rows,
            "覆/span-maps-to-region": span_maps_region_rows,
        },
        "heat_regions": heat_regions,
        "glyphs": {
            "locus": "址",
            "heat": "熱",
            "coverage": "覆",
            "failure": "破",
            "debug": "診",
        },
        "debug": debug_target,
        "grounded_meanings": {"DEBUG": debug_target},
        "node_heat": heat_nodes,
        "max_heat": round(_clamp01(_safe_float(max_heat, 0.0)), 4),
        "join_key": "file_id=sha256(normalized_path)",
        "region_join_key": "region_id=sha256(world|region_kind|region_key)",
    }


def _materialize_heat_values(
    catalog: dict[str, Any], pain_field: dict[str, Any]
) -> dict[str, Any]:
    named_fields = catalog.get("named_fields", []) if isinstance(catalog, dict) else []
    if not isinstance(named_fields, list):
        named_fields = []

    by_presence: dict[str, dict[str, Any]] = {}
    for row in named_fields:
        if not isinstance(row, dict):
            continue
        presence_id = str(row.get("id", "")).strip()
        if presence_id:
            by_presence[presence_id] = row

    for entity in ENTITY_MANIFEST:
        if not isinstance(entity, dict):
            continue
        presence_id = str(entity.get("id", "")).strip()
        if presence_id and presence_id not in by_presence:
            by_presence[presence_id] = entity

    field_anchors: dict[str, tuple[float, float]] = {}
    region_meta: dict[str, dict[str, Any]] = {}
    for field_id, presence_id in FIELD_TO_PRESENCE.items():
        item = by_presence.get(presence_id, {})
        x = _clamp01(_safe_float(item.get("x", 0.5), 0.5))
        y = _clamp01(_safe_float(item.get("y", 0.5), 0.5))
        field_anchors[field_id] = (x, y)
        region_meta[field_id] = {
            "region_id": field_id,
            "presence_id": presence_id,
            "en": str(item.get("en", presence_id)),
            "ja": str(item.get("ja", "")),
            "x": round(x, 4),
            "y": round(y, 4),
        }

    node_heat_rows = (
        pain_field.get("node_heat", []) if isinstance(pain_field, dict) else []
    )
    if not isinstance(node_heat_rows, list):
        node_heat_rows = []

    region_heat_raw: dict[str, float] = {field_id: 0.0 for field_id in field_anchors}
    locate_rows: list[dict[str, Any]] = []
    for row in node_heat_rows[:240]:
        if not isinstance(row, dict):
            continue
        entity_id = str(row.get("node_id", "")).strip()
        if not entity_id:
            continue
        heat = _clamp01(_safe_float(row.get("heat", 0.0), 0.0))
        if heat <= 0.0:
            continue

        x = _clamp01(_safe_float(row.get("x", 0.5), 0.5))
        y = _clamp01(_safe_float(row.get("y", 0.5), 0.5))
        locate_scores = _field_scores_from_position(x, y, field_anchors)
        ranked_scores = sorted(
            locate_scores.items(),
            key=lambda item: (-_safe_float(item[1], 0.0), item[0]),
        )

        for field_id, locate_weight in ranked_scores:
            region_heat_raw[field_id] += heat * _clamp01(
                _safe_float(locate_weight, 0.0)
            )

        for field_id, locate_weight in ranked_scores[:4]:
            locate_rows.append(
                {
                    "kind": "址",
                    "entity_id": entity_id,
                    "region_id": field_id,
                    "weight": round(_clamp01(_safe_float(locate_weight, 0.0)), 4),
                }
            )

    max_raw_heat = max(region_heat_raw.values(), default=0.0)
    regions: list[dict[str, Any]] = []
    for rank, (field_id, raw_heat) in enumerate(
        sorted(region_heat_raw.items(), key=lambda item: (-item[1], item[0])),
        start=1,
    ):
        value = 0.0
        if max_raw_heat > 0.0:
            value = _clamp01(raw_heat / max_raw_heat)
        meta = region_meta.get(
            field_id,
            {
                "region_id": field_id,
                "presence_id": FIELD_TO_PRESENCE.get(field_id, ""),
                "en": field_id,
                "ja": "",
                "x": 0.5,
                "y": 0.5,
            },
        )
        regions.append(
            {
                **meta,
                "rank": rank,
                "raw": round(max(0.0, raw_heat), 6),
                "value": round(value, 4),
            }
        )

    facts = [
        {
            "kind": "熱/value",
            "region_id": row.get("region_id", ""),
            "value": row.get("value", 0.0),
            "raw": row.get("raw", 0.0),
        }
        for row in regions
    ]
    return {
        "record": "ημ.heat-values.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active": max_raw_heat > 0.0,
        "source": "pain_field.node_heat",
        "regions": regions,
        "facts": facts,
        "locate": locate_rows,
        "max_raw": round(max(0.0, max_raw_heat), 6),
    }


def build_named_field_overlays(
    entity_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for entity in entity_manifest:
        key = str(entity.get("id", "")).strip()
        if key:
            by_id[key] = entity

    overlays: list[dict[str, Any]] = []
    canonical_presence_ids = [
        "receipt_river",
        "witness_thread",
        "fork_tax_canticle",
        "mage_of_receipts",
        "keeper_of_receipts",
        "anchor_registry",
        "gates_of_truth",
    ]
    for idx, field_id in enumerate(canonical_presence_ids):
        item = by_id.get(field_id)
        if item is None:
            continue

        hue = int(item.get("hue", 200))
        overlays.append(
            {
                "id": field_id,
                "en": str(item.get("en", field_id.replace("_", " ").title())),
                "ja": str(item.get("ja", "")),
                "type": str(item.get("type", "flow")),
                "x": float(item.get("x", 0.5)),
                "y": float(item.get("y", 0.5)),
                "freq": float(item.get("freq", 220.0)),
                "hue": hue,
                "gradient": {
                    "mode": "radial",
                    "radius": round(0.2 + (idx % 3) * 0.035, 3),
                    "stops": [
                        {
                            "offset": 0.0,
                            "color": f"hsla({hue}, 88%, 74%, 0.36)",
                        },
                        {
                            "offset": 0.52,
                            "color": f"hsla({hue}, 76%, 58%, 0.2)",
                        },
                        {
                            "offset": 1.0,
                            "color": f"hsla({(hue + 28) % 360}, 72%, 44%, 0.0)",
                        },
                    ],
                },
                "motion": {
                    "drift_hz": round(0.07 + idx * 0.013, 3),
                    "wobble_px": 5 + (idx % 4) * 3,
                },
            }
        )

    return overlays


def _mix_fingerprint(catalog: dict[str, Any]) -> str:
    rows: list[str] = []
    for item in catalog.get("items", []):
        rel_path = str(item.get("rel_path", ""))
        if rel_path.lower().endswith(".wav"):
            rows.append(
                "|".join(
                    [
                        rel_path,
                        str(item.get("bytes", 0)),
                        str(item.get("mtime_utc", "")),
                    ]
                )
            )
    rows.sort()
    return sha1("\n".join(rows).encode("utf-8")).hexdigest()


def _collect_mix_sources(catalog: dict[str, Any], vault_root: Path) -> list[Path]:
    paths: list[Path] = []
    for item in catalog.get("items", []):
        rel_path = str(item.get("rel_path", ""))
        if not rel_path.lower().endswith(".wav"):
            continue
        candidate = (vault_root / rel_path).resolve()
        if candidate.exists() and candidate.is_file():
            paths.append(candidate)
    return paths


def _mix_wav_sources(sources: list[Path]) -> tuple[bytes, dict[str, Any]]:
    if not sources:
        return b"", {"sources": 0, "sample_rate": 0, "duration_seconds": 0.0}

    sample_rate = 44100
    clips: list[tuple[array, int]] = []
    max_frames = 0

    for src in sources:
        with wave.open(str(src), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            if sampwidth != 2:
                continue
            if channels not in (1, 2):
                continue

            frames_raw = wf.readframes(wf.getnframes())
            pcm = array("h")
            pcm.frombytes(frames_raw)
            frames = len(pcm) // channels
            if frames == 0:
                continue

            sample_rate = framerate
            clips.append((pcm, channels))
            if frames > max_frames:
                max_frames = frames

    if not clips or max_frames == 0:
        return b"", {"sources": 0, "sample_rate": 0, "duration_seconds": 0.0}

    gain = 1.0 / max(1, len(clips))
    mix = [0] * (max_frames * 2)

    for pcm, channels in clips:
        if channels == 1:
            frame_count = len(pcm)
            for i in range(frame_count):
                value = int(pcm[i] * gain)
                idx = i * 2
                mix[idx] += value
                mix[idx + 1] += value
            continue

        frame_count = len(pcm) // 2
        for i in range(frame_count):
            src_idx = i * 2
            dst_idx = i * 2
            mix[dst_idx] += int(pcm[src_idx] * gain)
            mix[dst_idx + 1] += int(pcm[src_idx + 1] * gain)

    out = array("h")
    for value in mix:
        if value > 32767:
            out.append(32767)
        elif value < -32768:
            out.append(-32768)
        else:
            out.append(value)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf_out:
        wf_out.setnchannels(2)
        wf_out.setsampwidth(2)
        wf_out.setframerate(sample_rate)
        wf_out.writeframes(out.tobytes())

    meta = {
        "sources": len(clips),
        "sample_rate": sample_rate,
        "duration_seconds": round(max_frames / sample_rate, 3),
    }
    return buffer.getvalue(), meta


def build_mix_stream(
    catalog: dict[str, Any], vault_root: Path
) -> tuple[bytes, dict[str, Any]]:
    fingerprint = _mix_fingerprint(catalog)
    with _MIX_CACHE_LOCK:
        if _MIX_CACHE["fingerprint"] == fingerprint and _MIX_CACHE["wav"]:
            return _MIX_CACHE["wav"], _MIX_CACHE["meta"]

    sources = _collect_mix_sources(catalog, vault_root)
    wav, meta = _mix_wav_sources(sources)
    meta["fingerprint"] = fingerprint

    with _MIX_CACHE_LOCK:
        _MIX_CACHE["fingerprint"] = fingerprint
        _MIX_CACHE["wav"] = wav
        _MIX_CACHE["meta"] = meta
    return wav, meta


def websocket_accept_value(client_key: str) -> str:
    digest = sha1((client_key + WS_MAGIC).encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def websocket_frame_text(message: str) -> bytes:
    payload = message.encode("utf-8")
    length = len(payload)
    header = bytearray([0x81])
    if length <= 125:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + payload


def _weaver_probe_host(bind_host: str) -> str:
    host = str(bind_host or "127.0.0.1").strip()
    return "127.0.0.1" if host == "0.0.0.0" else host


def _weaver_health_check(host: str, port: int, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=float(timeout_s)):
            return True
    except:
        return False


def _weaver_service_base_url() -> str:
    from .constants import WEAVER_HOST_ENV, WEAVER_PORT

    return f"http://{_weaver_probe_host(WEAVER_HOST_ENV or '127.0.0.1')}:{WEAVER_PORT}"


def _read_weaver_snapshot_file(part_root: Path) -> dict[str, Any] | None:
    p = (part_root / "world_state" / "web_graph_weaver.snapshot.json").resolve()
    if not p.exists() or not p.is_file():
        return None
    try:
        pay = json.loads(p.read_text("utf-8"))
        if isinstance(pay, dict):
            return {
                "ok": True,
                "graph": pay.get("graph", {}),
                "status": pay.get("status", {}),
                "source": str(p),
            }
    except:
        pass
    return None


def _fetch_weaver_graph_payload(part_root: Path) -> dict[str, Any]:
    def _graph_node_count(graph_payload: dict[str, Any]) -> int:
        counts = graph_payload.get("counts", {})
        if isinstance(counts, dict):
            from_counts = int(_safe_float(counts.get("nodes_total", 0), 0.0))
            if from_counts > 0:
                return from_counts
        nodes = graph_payload.get("nodes", [])
        return len(nodes) if isinstance(nodes, list) else 0

    base = _weaver_service_base_url()
    parsed = urlparse(base)
    host = parsed.hostname or "127.0.0.1"
    if _weaver_health_check(
        host,
        WEAVER_PORT,
        timeout_s=WEAVER_GRAPH_HEALTH_TIMEOUT_SECONDS,
    ):
        try:
            with urlopen(
                Request(
                    f"{base}/api/weaver/graph?node_limit={WEAVER_GRAPH_NODE_LIMIT}&edge_limit={WEAVER_GRAPH_EDGE_LIMIT}",
                    method="GET",
                ),
                timeout=WEAVER_GRAPH_FETCH_TIMEOUT_SECONDS,
            ) as response:
                graph_payload = json.loads(
                    response.read().decode("utf-8", errors="ignore")
                )
            with urlopen(
                Request(f"{base}/api/weaver/status", method="GET"),
                timeout=WEAVER_GRAPH_FETCH_TIMEOUT_SECONDS,
            ) as response:
                status_payload = json.loads(
                    response.read().decode("utf-8", errors="ignore")
                )

            graph = (
                graph_payload.get("graph", {})
                if isinstance(graph_payload, dict)
                else {}
            )
            status = status_payload if isinstance(status_payload, dict) else {}
            if isinstance(graph, dict):
                live_nodes = _graph_node_count(graph)
                if live_nodes <= 0:
                    fallback = _read_weaver_snapshot_file(part_root)
                    if fallback is not None:
                        fallback_graph = fallback.get("graph", {})
                        if isinstance(fallback_graph, dict):
                            fallback_nodes = _graph_node_count(fallback_graph)
                            if fallback_nodes > 0:
                                fallback_status = fallback.get("status", {})
                                merged_status: dict[str, Any] = {}
                                if isinstance(fallback_status, dict):
                                    merged_status.update(fallback_status)
                                if isinstance(status, dict):
                                    merged_status.update(status)
                                return {
                                    "ok": True,
                                    "graph": fallback_graph,
                                    "status": merged_status,
                                    "source": str(fallback.get("source", "")),
                                }
                return {
                    "ok": True,
                    "graph": graph,
                    "status": status,
                    "source": f"{base}/api/weaver/graph",
                }
        except Exception:
            pass

    fallback = _read_weaver_snapshot_file(part_root)
    if fallback is not None:
        return fallback

    return {
        "ok": False,
        "graph": {"nodes": [], "edges": [], "counts": {}},
        "status": {},
        "source": "",
    }


def _json_deep_clone(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _clean_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_-]+", text.lower()) if token]


def _document_layout_range_from_importance(importance: float) -> float:
    normalized = _clamp01(_safe_float(importance, 0.2))
    return 0.018 + (normalized * 0.055)


def _document_layout_tokens(node: dict[str, Any]) -> list[str]:
    values: list[str] = []
    tags = node.get("tags", [])
    labels = node.get("labels", [])
    if isinstance(tags, list):
        values.extend(str(tag) for tag in tags)
    if isinstance(labels, list):
        values.extend(str(label) for label in labels)
    values.extend(
        [
            str(node.get("summary", "")),
            str(node.get("text_excerpt", "")),
            str(node.get("source_rel_path", "")),
            str(node.get("archived_rel_path", "")),
            str(node.get("archive_rel_path", "")),
            str(node.get("name", "")),
            str(node.get("kind", "")),
            str(node.get("dominant_field", "")),
            str(node.get("vecstore_collection", "")),
        ]
    )

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in _clean_tokens(value):
            if len(token) < 3:
                continue
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
            if len(deduped) >= 80:
                return deduped
    return deduped


def _document_layout_text_density(node: dict[str, Any], tokens: list[str]) -> float:
    token_density = min(1.0, len(tokens) / 42.0)
    summary_len = len(str(node.get("summary", "")).strip())
    excerpt_len = len(str(node.get("text_excerpt", "")).strip())
    label_len = len(str(node.get("name", "")).strip()) + len(
        str(node.get("label", "")).strip()
    )
    char_density = min(1.0, (summary_len + excerpt_len + label_len) / 760.0)

    tags = node.get("tags", [])
    labels = node.get("labels", [])
    embedding_links = node.get("embedding_links", [])
    tag_count = len(tags) if isinstance(tags, list) else 0
    label_count = len(labels) if isinstance(labels, list) else 0
    link_count = len(embedding_links) if isinstance(embedding_links, list) else 0
    layer_count = _safe_int(node.get("embed_layer_count", 0), 0)
    structural = min(
        1.0,
        (tag_count * 0.08)
        + (label_count * 0.05)
        + (layer_count * 0.22)
        + (link_count * 0.04),
    )

    density = 0.2 + (token_density * 0.42) + (char_density * 0.26) + (structural * 0.36)
    return max(0.12, min(1.9, density))


def _document_layout_semantic_vector(
    node: dict[str, Any],
    tokens: list[str],
    *,
    dimensions: int = 8,
) -> list[float]:
    if dimensions <= 0:
        return []

    raw_tokens = list(tokens)
    if not raw_tokens:
        raw_tokens = _clean_tokens(
            " ".join(
                [
                    str(node.get("dominant_field", "")),
                    str(node.get("kind", "")),
                    str(node.get("vecstore_collection", "")),
                    str(node.get("name", "")),
                    str(node.get("label", "")),
                ]
            )
        )
    if not raw_tokens:
        raw_tokens = ["eta", "mu", "field"]

    accum = [0.0 for _ in range(dimensions)]
    for token_index, token in enumerate(raw_tokens[:96]):
        weight = 0.8 + min(1.6, len(token) / 6.5)
        digest = hashlib.sha1(
            f"{token}|{token_index}|{dimensions}".encode("utf-8")
        ).digest()
        for axis in range(dimensions):
            byte = digest[axis % len(digest)]
            signed = (float(byte) / 127.5) - 1.0
            accum[axis] += signed * weight

    field_token = str(node.get("dominant_field", "")).strip()
    kind_token = str(node.get("kind", "")).strip().lower()
    for marker, gain in ((field_token, 0.36), (kind_token, 0.22)):
        if not marker:
            continue
        digest = hashlib.sha1(f"marker:{marker}".encode("utf-8")).digest()
        for axis in range(dimensions):
            byte = digest[(axis * 3) % len(digest)]
            signed = (float(byte) / 127.5) - 1.0
            accum[axis] += signed * gain

    magnitude = math.sqrt(sum(value * value for value in accum))
    if magnitude <= 1e-8:
        fallback = [0.0 for _ in range(dimensions)]
        fallback[0] = 1.0
        return fallback
    return [value / magnitude for value in accum]


def _semantic_vector_blend(
    base: list[float], target: list[float], blend: float
) -> list[float]:
    if not base and not target:
        return []
    if not base:
        return list(target)
    if not target:
        return list(base)

    mix = max(0.0, min(1.0, _safe_float(blend, 0.5)))
    size = min(len(base), len(target))
    if size <= 0:
        return list(base)

    merged = [(base[i] * (1.0 - mix)) + (target[i] * mix) for i in range(size)]
    magnitude = math.sqrt(sum(value * value for value in merged))
    if magnitude <= 1e-8:
        return [0.0 for _ in range(size)]
    return [value / magnitude for value in merged]


def _semantic_vector_cosine(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size <= 0:
        return 0.0
    dot = sum(left[i] * right[i] for i in range(size))
    left_mag = sum(left[i] * left[i] for i in range(size))
    right_mag = sum(right[i] * right[i] for i in range(size))
    if left_mag <= 1e-12 or right_mag <= 1e-12:
        return 0.0
    cosine = dot / math.sqrt(left_mag * right_mag)
    return max(-1.0, min(1.0, cosine))


def _semantic_vector_hue(vector: list[float]) -> float:
    if not vector:
        return 210.0
    vx = _safe_float(vector[0], 0.0)
    vy = _safe_float(vector[1], 0.0) if len(vector) > 1 else 0.0
    if abs(vx) <= 1e-8 and abs(vy) <= 1e-8:
        return 210.0
    return (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0


def _document_layout_similarity(
    left_node: dict[str, Any],
    right_node: dict[str, Any],
    left_tokens: list[str],
    right_tokens: list[str],
) -> float:
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    overlap = len(left_set.intersection(right_set))
    union = max(1, len(left_set) + len(right_set) - overlap)
    token_jaccard = overlap / float(union)

    same_field = (
        1.0
        if str(left_node.get("dominant_field", "")).strip()
        and str(left_node.get("dominant_field", "")).strip()
        == str(right_node.get("dominant_field", "")).strip()
        else 0.0
    )
    same_kind = (
        1.0
        if str(left_node.get("kind", "")).strip().lower()
        and str(left_node.get("kind", "")).strip().lower()
        == str(right_node.get("kind", "")).strip().lower()
        else 0.0
    )
    left_collection = str(left_node.get("vecstore_collection", "")).strip()
    right_collection = str(right_node.get("vecstore_collection", "")).strip()
    same_collection = (
        1.0 if left_collection and left_collection == right_collection else 0.0
    )

    score = (
        (token_jaccard * 0.78)
        + (same_field * 0.12)
        + (same_kind * 0.06)
        + (same_collection * 0.04)
    )
    if token_jaccard < 0.05 and same_field <= 0.0 and same_kind <= 0.0:
        score *= 0.45
    return _clamp01(score)


def _document_layout_is_embedded(node: dict[str, Any]) -> bool:
    if _safe_int(node.get("embed_layer_count", 0), 0) > 0:
        return True

    layer_points = node.get("embed_layer_points", [])
    if isinstance(layer_points, list):
        for row in layer_points:
            if not isinstance(row, dict):
                continue
            if bool(row.get("active", True)):
                return True

    if str(node.get("vecstore_collection", "")).strip():
        return True

    embedding_links = node.get("embedding_links", [])
    if isinstance(embedding_links, list) and len(embedding_links) > 0:
        return True

    return False


def _apply_file_graph_document_similarity_layout(
    file_graph: dict[str, Any], *, now: float | None = None
) -> list[dict[str, float]]:
    file_nodes_raw = file_graph.get("file_nodes", [])
    if not isinstance(file_nodes_raw, list) or len(file_nodes_raw) <= 0:
        file_graph["embedding_particles"] = []
        return []

    entries: list[dict[str, Any]] = []
    for index, node in enumerate(file_nodes_raw):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", "")).strip() or f"file:{index}"
        x = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
        y = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
        importance = _clamp01(_safe_float(node.get("importance", 0.2), 0.2))
        local_range = _document_layout_range_from_importance(importance)
        tokens = _document_layout_tokens(node)
        semantic_vector = _document_layout_semantic_vector(node, tokens)
        text_density = _document_layout_text_density(node, tokens)
        entries.append(
            {
                "id": node_id,
                "index": len(entries),
                "node": node,
                "x": x,
                "y": y,
                "importance": importance,
                "range": local_range,
                "embedded": _document_layout_is_embedded(node),
                "tokens": tokens,
                "vector": semantic_vector,
                "text_density": text_density,
            }
        )

    if not entries:
        file_graph["embedding_particles"] = []
        return []

    cell_size = 0.08
    grid: dict[tuple[int, int], list[int]] = {}
    for index, entry in enumerate(entries):
        gx = int(entry["x"] / cell_size)
        gy = int(entry["y"] / cell_size)
        grid.setdefault((gx, gy), []).append(index)

    offsets: list[list[float]] = [[0.0, 0.0] for _ in entries]
    if len(entries) > 1:
        for index, left in enumerate(entries):
            gx = int(left["x"] / cell_size)
            gy = int(left["y"] / cell_size)
            radius_cells = max(1, int(math.ceil(left["range"] / cell_size)))

            for oy in range(-radius_cells, radius_cells + 1):
                for ox in range(-radius_cells, radius_cells + 1):
                    bucket = grid.get((gx + ox, gy + oy), [])
                    for other_index in bucket:
                        if other_index <= index:
                            continue
                        right = entries[other_index]
                        pair_range = max(left["range"], right["range"])
                        dx = right["x"] - left["x"]
                        dy = right["y"] - left["y"]
                        distance = math.sqrt((dx * dx) + (dy * dy))
                        if distance <= 1e-8 or distance > pair_range:
                            continue

                        similarity = _document_layout_similarity(
                            left["node"],
                            right["node"],
                            left.get("tokens", []),
                            right.get("tokens", []),
                        )
                        semantic_signed = max(
                            -1.0, min(1.0, (similarity - 0.52) / 0.48)
                        )
                        mixed_embedding = bool(left["embedded"]) != bool(
                            right["embedded"]
                        )
                        signed_similarity = (
                            -max(0.46, abs(semantic_signed) * 0.72)
                            if mixed_embedding
                            else semantic_signed
                        )
                        if abs(signed_similarity) < 0.22:
                            continue

                        falloff = _clamp01(1.0 - (distance / max(pair_range, 1e-6)))
                        importance_mix = (
                            left["importance"] + right["importance"]
                        ) * 0.5
                        density_mix = (
                            _safe_float(left.get("text_density"), 0.45)
                            + _safe_float(right.get("text_density"), 0.45)
                        ) * 0.5
                        strength = (
                            falloff
                            * abs(signed_similarity)
                            * (1.2 if mixed_embedding else 1.0)
                            * (0.00145 + (importance_mix * 0.0022))
                            * (0.66 + (density_mix * 0.3))
                        )
                        if strength <= 0.0:
                            continue

                        ux = dx / distance
                        uy = dy / distance
                        direction = 1.0 if signed_similarity >= 0.0 else -1.0
                        fx = ux * strength * direction
                        fy = uy * strength * direction

                        offsets[index][0] += fx
                        offsets[index][1] += fy
                        offsets[other_index][0] -= fx
                        offsets[other_index][1] -= fy

    embedding_particle_points: list[dict[str, float]] = []
    embedding_particle_nodes: list[dict[str, float | str]] = []
    embedded_entries = [entry for entry in entries if bool(entry.get("embedded"))]
    if embedded_entries:
        now_seconds = _safe_float(now, time.time()) if now is not None else time.time()
        particle_count = max(6, min(42, int(round(len(embedded_entries) * 1.8))))
        particles: list[dict[str, Any]] = []

        for index in range(particle_count):
            source = embedded_entries[index % len(embedded_entries)]
            seed = f"{source['id']}|particle|{index}"
            phase = (_stable_ratio(seed, 17) * math.tau) + (
                now_seconds * (0.28 + (_stable_ratio(seed, 23) * 0.52))
            )
            orbit = 0.006 + (
                _stable_ratio(seed, 31)
                * max(0.018, _safe_float(source["range"], 0.03) * 0.64)
            )
            x = _clamp01(_safe_float(source["x"], 0.5) + math.cos(phase) * orbit)
            y = _clamp01(_safe_float(source["y"], 0.5) + math.sin(phase) * orbit)
            particles.append(
                {
                    "id": f"embed-particle:{index}",
                    "x": x,
                    "y": y,
                    "vx": 0.0,
                    "vy": 0.0,
                    "vector": list(source.get("vector", [])),
                    "text_density": _safe_float(source.get("text_density"), 0.45),
                    "focus_x": x,
                    "focus_y": y,
                    "cohesion": 0.0,
                    "drift": (_stable_ratio(seed, 41) * 2.0) - 1.0,
                }
            )

        for _ in range(4):
            particle_forces: list[list[float]] = [[0.0, 0.0] for _ in particles]

            for particle_index, particle in enumerate(particles):
                influence_total = 0.0
                avg_x = 0.0
                avg_y = 0.0
                avg_vector = [0.0 for _ in particle.get("vector", [])]
                doc_radius = 0.22

                for entry in embedded_entries:
                    dx = _safe_float(entry["x"], 0.5) - _safe_float(particle["x"], 0.5)
                    dy = _safe_float(entry["y"], 0.5) - _safe_float(particle["y"], 0.5)
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance > doc_radius:
                        continue
                    if distance <= 1e-8:
                        jitter = (
                            _stable_ratio(
                                f"{particle['id']}|{entry['id']}|jitter",
                                particle_index + 1,
                            )
                            - 0.5
                        ) * 0.0012
                        dx += jitter
                        dy -= jitter
                        distance = max(1e-6, math.sqrt((dx * dx) + (dy * dy)))

                    similarity = _semantic_vector_cosine(
                        particle.get("vector", []),
                        entry.get("vector", []),
                    )
                    distance_weight = _clamp01(1.0 - (distance / doc_radius))
                    density_weight = 0.24 + (
                        _safe_float(entry.get("text_density"), 0.45) * 0.92
                    )
                    similarity_weight = 0.28 + ((similarity + 1.0) * 0.36)
                    influence_weight = (
                        distance_weight
                        * distance_weight
                        * density_weight
                        * similarity_weight
                    )
                    if influence_weight <= 0.0:
                        continue

                    influence_total += influence_weight
                    avg_x += _safe_float(entry["x"], 0.5) * influence_weight
                    avg_y += _safe_float(entry["y"], 0.5) * influence_weight
                    entry_vector = entry.get("vector", [])
                    for axis in range(min(len(avg_vector), len(entry_vector))):
                        avg_vector[axis] += (
                            _safe_float(entry_vector[axis], 0.0) * influence_weight
                        )

                    direction = 1.0 if similarity >= 0.0 else -1.0
                    force_strength = (
                        (0.00072 + (abs(similarity) * 0.0024))
                        * distance_weight
                        * density_weight
                    )
                    particle_forces[particle_index][0] += (
                        (dx / distance) * force_strength * direction
                    )
                    particle_forces[particle_index][1] += (
                        (dy / distance) * force_strength * direction
                    )

                if influence_total > 0.0:
                    target_x = avg_x / influence_total
                    target_y = avg_y / influence_total
                    particle["focus_x"] = target_x
                    particle["focus_y"] = target_y
                    particle["cohesion"] = _clamp01(
                        (_safe_float(particle.get("cohesion", 0.0), 0.0) * 0.55)
                        + min(1.0, influence_total * 0.48)
                    )
                    pull_strength = min(0.0052, 0.0012 + (influence_total * 0.0019))
                    particle_forces[particle_index][0] += (
                        target_x - _safe_float(particle["x"], 0.5)
                    ) * pull_strength
                    particle_forces[particle_index][1] += (
                        target_y - _safe_float(particle["y"], 0.5)
                    ) * pull_strength

                    if avg_vector:
                        avg_magnitude = math.sqrt(
                            sum(value * value for value in avg_vector)
                        )
                        if avg_magnitude > 1e-8:
                            normalized_avg = [
                                value / avg_magnitude for value in avg_vector
                            ]
                            particle["vector"] = _semantic_vector_blend(
                                list(particle.get("vector", [])),
                                normalized_avg,
                                0.26,
                            )
                else:
                    particle["cohesion"] = _clamp01(
                        _safe_float(particle.get("cohesion", 0.0), 0.0) * 0.86
                    )

            for left_index in range(len(particles)):
                left = particles[left_index]
                for right_index in range(left_index + 1, len(particles)):
                    right = particles[right_index]
                    dx = _safe_float(right["x"], 0.5) - _safe_float(left["x"], 0.5)
                    dy = _safe_float(right["y"], 0.5) - _safe_float(left["y"], 0.5)
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance > 0.2:
                        continue
                    if distance <= 1e-8:
                        jitter = (
                            _stable_ratio(
                                f"{left['id']}|{right['id']}|pair", left_index + 3
                            )
                            - 0.5
                        ) * 0.001
                        dx += jitter
                        dy -= jitter
                        distance = max(1e-6, math.sqrt((dx * dx) + (dy * dy)))

                    similarity = _semantic_vector_cosine(
                        left.get("vector", []),
                        right.get("vector", []),
                    )
                    falloff = _clamp01(1.0 - (distance / 0.2))
                    pair_strength = (0.00044 + (abs(similarity) * 0.00186)) * falloff
                    direction = 1.0 if similarity >= 0.0 else -1.0
                    fx = (dx / distance) * pair_strength * direction
                    fy = (dy / distance) * pair_strength * direction

                    particle_forces[left_index][0] += fx
                    particle_forces[left_index][1] += fy
                    particle_forces[right_index][0] -= fx
                    particle_forces[right_index][1] -= fy

            for particle_index, particle in enumerate(particles):
                drift_phase = (
                    now_seconds
                    * (0.62 + abs(_safe_float(particle.get("drift", 0.0), 0.0)) * 0.42)
                ) + (particle_index * 0.41)
                particle_forces[particle_index][0] += math.cos(drift_phase) * 0.00021
                particle_forces[particle_index][1] += math.sin(drift_phase) * 0.00017

                vx = (
                    _safe_float(particle.get("vx", 0.0), 0.0)
                    + particle_forces[particle_index][0]
                ) * 0.84
                vy = (
                    _safe_float(particle.get("vy", 0.0), 0.0)
                    + particle_forces[particle_index][1]
                ) * 0.84
                speed = math.sqrt((vx * vx) + (vy * vy))
                speed_limit = 0.0062 + (
                    _safe_float(particle.get("text_density", 0.45), 0.45) * 0.0024
                )
                if speed > speed_limit and speed > 1e-8:
                    scale = speed_limit / speed
                    vx *= scale
                    vy *= scale

                particle["vx"] = vx
                particle["vy"] = vy
                particle["x"] = _clamp01(_safe_float(particle.get("x", 0.5), 0.5) + vx)
                particle["y"] = _clamp01(_safe_float(particle.get("y", 0.5), 0.5) + vy)

        for entry in embedded_entries:
            entry_index = int(entry.get("index", 0))
            if entry_index < 0 or entry_index >= len(offsets):
                continue
            influence_x = 0.0
            influence_y = 0.0
            influence_radius = max(
                0.08,
                min(0.26, (_safe_float(entry.get("range", 0.03), 0.03) * 2.4) + 0.05),
            )

            for particle in particles:
                dx = _safe_float(particle.get("x", 0.5), 0.5) - _safe_float(
                    entry.get("x", 0.5), 0.5
                )
                dy = _safe_float(particle.get("y", 0.5), 0.5) - _safe_float(
                    entry.get("y", 0.5), 0.5
                )
                distance = math.sqrt((dx * dx) + (dy * dy))
                if distance > influence_radius:
                    continue
                if distance <= 1e-8:
                    continue

                similarity = _semantic_vector_cosine(
                    entry.get("vector", []),
                    particle.get("vector", []),
                )
                falloff = _clamp01(1.0 - (distance / influence_radius))
                density_mix = 0.58 + (
                    (
                        _safe_float(entry.get("text_density"), 0.45)
                        + _safe_float(particle.get("text_density"), 0.45)
                    )
                    * 0.24
                )
                strength = (
                    (0.00016 + (abs(similarity) * 0.00052)) * falloff * density_mix
                )
                direction = 1.0 if similarity >= 0.0 else -1.0
                influence_x += (dx / distance) * strength * direction
                influence_y += (dy / distance) * strength * direction

            max_influence = 0.0032 + (
                _safe_float(entry.get("importance", 0.2), 0.2) * 0.0048
            )
            offsets[entry_index][0] += max(
                -max_influence, min(max_influence, influence_x)
            )
            offsets[entry_index][1] += max(
                -max_influence, min(max_influence, influence_y)
            )

        for particle in particles[:48]:
            hue = _semantic_vector_hue(list(particle.get("vector", [])))
            cohesion = _clamp01(_safe_float(particle.get("cohesion", 0.0), 0.0))
            saturation = max(0.52, min(0.92, 0.64 + (cohesion * 0.2)))
            value = max(0.72, min(0.98, 0.84 + (cohesion * 0.14)))
            r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
                (hue % 360.0) / 360.0,
                saturation,
                value,
            )
            size = (
                1.8
                + (_safe_float(particle.get("text_density", 0.45), 0.45) * 1.1)
                + (cohesion * 1.8)
            )
            x_norm = _clamp01(_safe_float(particle.get("x", 0.5), 0.5))
            y_norm = _clamp01(_safe_float(particle.get("y", 0.5), 0.5))
            embedding_particle_points.append(
                {
                    "x": round((x_norm * 2.0) - 1.0, 5),
                    "y": round(1.0 - (y_norm * 2.0), 5),
                    "size": round(size, 5),
                    "r": round(r_raw, 5),
                    "g": round(g_raw, 5),
                    "b": round(b_raw, 5),
                }
            )
            embedding_particle_nodes.append(
                {
                    "id": str(particle.get("id", "")),
                    "x": round(x_norm, 5),
                    "y": round(y_norm, 5),
                    "hue": round(hue, 4),
                    "cohesion": round(cohesion, 5),
                    "text_density": round(
                        _safe_float(particle.get("text_density", 0.45), 0.45), 5
                    ),
                }
            )

    file_graph["embedding_particles"] = embedding_particle_nodes

    position_by_id: dict[str, tuple[float, float]] = {}
    for index, entry in enumerate(entries):
        max_offset = 0.008 + (entry["importance"] * 0.014)
        offset_x = max(-max_offset, min(max_offset, offsets[index][0]))
        offset_y = max(-max_offset, min(max_offset, offsets[index][1]))
        x = round(_clamp01(entry["x"] + offset_x), 6)
        y = round(_clamp01(entry["y"] + offset_y), 6)
        entry["node"]["x"] = x
        entry["node"]["y"] = y
        position_by_id[entry["id"]] = (x, y)

    graph_nodes = file_graph.get("nodes", [])
    if isinstance(graph_nodes, list):
        for node in graph_nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("node_type", "")).strip().lower() != "file":
                continue
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            position = position_by_id.get(node_id)
            if position is None:
                continue
            node["x"] = position[0]
            node["y"] = position[1]

    return embedding_particle_points


def _build_backend_field_particles(
    *,
    file_graph: dict[str, Any] | None,
    presence_impacts: list[dict[str, Any]],
    resource_heartbeat: dict[str, Any],
    compute_jobs: list[dict[str, Any]],
    now: float,
) -> list[dict[str, float | str]]:
    if not presence_impacts:
        return []

    file_nodes_raw = (
        file_graph.get("file_nodes", []) if isinstance(file_graph, dict) else []
    )
    file_nodes = [row for row in file_nodes_raw if isinstance(row, dict)]
    embedding_nodes_raw = (
        file_graph.get("embedding_particles", [])
        if isinstance(file_graph, dict)
        else []
    )
    embedding_nodes = [row for row in embedding_nodes_raw if isinstance(row, dict)]

    manifest_by_id = {
        str(row.get("id", "")).strip(): row
        for row in ENTITY_MANIFEST
        if str(row.get("id", "")).strip()
    }
    presence_to_field: dict[str, str] = {}
    for field_id, presence_id in FIELD_TO_PRESENCE.items():
        pid = str(presence_id).strip()
        if pid and pid not in presence_to_field:
            presence_to_field[pid] = str(field_id).strip()

    devices = (
        resource_heartbeat.get("devices", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    if not isinstance(devices, dict):
        devices = {}
    resource_pressure = 0.0
    for device_key in ("cpu", "gpu1", "gpu2", "npu0"):
        row = devices.get(device_key, {})
        util = _safe_float(
            (row if isinstance(row, dict) else {}).get("utilization", 0.0), 0.0
        )
        resource_pressure = max(resource_pressure, _clamp01(util / 100.0))

    compute_pressure = _clamp01(len(compute_jobs) / 24.0)

    field_particles: list[dict[str, float | str]] = []
    now_mono = time.monotonic()
    live_ids: set[str] = set()

    def _node_field_similarity(
        node: dict[str, Any], target_field_id: str, target_presence_id: str
    ) -> float:
        if not target_field_id:
            return 0.0
        score = 0.0
        field_scores = node.get("field_scores", {})
        if isinstance(field_scores, dict):
            score = _clamp01(_safe_float(field_scores.get(target_field_id, 0.0), 0.0))
        dominant_field = str(node.get("dominant_field", "")).strip()
        if dominant_field and dominant_field == target_field_id:
            score = max(score, 0.85)
        dominant_presence = str(node.get("dominant_presence", "")).strip()
        if dominant_presence and dominant_presence == target_presence_id:
            score = max(score, 1.0)
        return _clamp01(score)

    with _DAIMO_DYNAMICS_LOCK:
        particle_cache = _DAIMO_DYNAMICS_CACHE.get("field_particles", {})
        if not isinstance(particle_cache, dict):
            particle_cache = {}

        for impact in presence_impacts:
            presence_id = str(impact.get("id", "")).strip()
            if not presence_id:
                continue

            presence_meta = manifest_by_id.get(presence_id, {})
            anchor_x = _clamp01(
                _safe_float(
                    presence_meta.get("x", _stable_ratio(f"{presence_id}|anchor", 3)),
                    _stable_ratio(f"{presence_id}|anchor", 3),
                )
            )
            anchor_y = _clamp01(
                _safe_float(
                    presence_meta.get("y", _stable_ratio(f"{presence_id}|anchor", 9)),
                    _stable_ratio(f"{presence_id}|anchor", 9),
                )
            )
            base_hue = _safe_float(presence_meta.get("hue", 200.0), 200.0)
            target_field_id = presence_to_field.get(presence_id, "")
            presence_role, particle_mode = _particle_role_and_mode_for_presence(
                presence_id
            )

            affected_by = (
                impact.get("affected_by", {}) if isinstance(impact, dict) else {}
            )
            affects = impact.get("affects", {}) if isinstance(impact, dict) else {}
            file_influence = _clamp01(
                _safe_float(
                    (affected_by if isinstance(affected_by, dict) else {}).get(
                        "files", 0.0
                    ),
                    0.0,
                )
            )
            world_influence = _clamp01(
                _safe_float(
                    (affects if isinstance(affects, dict) else {}).get("world", 0.0),
                    0.0,
                )
            )
            ledger_influence = _clamp01(
                _safe_float(
                    (affects if isinstance(affects, dict) else {}).get("ledger", 0.0),
                    0.0,
                )
            )

            node_signals: list[dict[str, float]] = []
            cluster_map: dict[tuple[int, int], dict[str, float]] = {}
            cluster_bucket_size = 0.18
            local_density_score = 0.0
            for node in file_nodes:
                nx = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
                ny = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
                field_similarity = _node_field_similarity(
                    node, target_field_id, presence_id
                )
                embed_signal = _clamp01(
                    (_safe_float(node.get("embed_layer_count", 0.0), 0.0) / 3.0)
                    + (
                        0.35
                        if str(node.get("vecstore_collection", "")).strip()
                        else 0.0
                    )
                )
                signed_similarity = max(
                    -1.0,
                    min(
                        1.0,
                        (field_similarity * 0.72) + (embed_signal * 0.34) - 0.43,
                    ),
                )
                node_importance = _clamp01(
                    _safe_float(node.get("importance", 0.25), 0.25)
                )
                distance_to_anchor = math.sqrt(
                    ((nx - anchor_x) * (nx - anchor_x))
                    + ((ny - anchor_y) * (ny - anchor_y))
                )
                anchor_proximity = _clamp01(1.0 - (distance_to_anchor / 0.55))
                relevance = (
                    (abs(signed_similarity) * 0.62)
                    + (node_importance * 0.24)
                    + (anchor_proximity * 0.14)
                )
                if relevance < 0.12 and anchor_proximity <= 0.04:
                    continue

                if distance_to_anchor <= 0.24:
                    local_density_score += _clamp01(
                        1.0 - (distance_to_anchor / 0.24)
                    ) * (0.35 + (node_importance * 0.65))

                node_signals.append(
                    {
                        "x": nx,
                        "y": ny,
                        "signed": signed_similarity,
                        "importance": node_importance,
                        "relevance": relevance,
                    }
                )

                cluster_key = (
                    int(nx / cluster_bucket_size),
                    int(ny / cluster_bucket_size),
                )
                cluster_weight = (
                    0.24 + (node_importance * 0.64) + (abs(signed_similarity) * 0.82)
                )
                cluster_row = cluster_map.setdefault(
                    cluster_key,
                    {
                        "xw": 0.0,
                        "yw": 0.0,
                        "signed": 0.0,
                        "weight_raw": 0.0,
                    },
                )
                cluster_row["xw"] += nx * cluster_weight
                cluster_row["yw"] += ny * cluster_weight
                cluster_row["signed"] += signed_similarity * cluster_weight
                cluster_row["weight_raw"] += cluster_weight

            if len(node_signals) > 140:
                node_signals.sort(
                    key=lambda row: _safe_float(row.get("relevance", 0.0), 0.0),
                    reverse=True,
                )
                node_signals = node_signals[:140]

            clusters: list[dict[str, float]] = []
            for cluster_row in cluster_map.values():
                weight_raw = _safe_float(cluster_row.get("weight_raw", 0.0), 0.0)
                if weight_raw <= 1e-8:
                    continue
                clusters.append(
                    {
                        "x": _clamp01(
                            _safe_float(cluster_row.get("xw", 0.0), 0.0) / weight_raw
                        ),
                        "y": _clamp01(
                            _safe_float(cluster_row.get("yw", 0.0), 0.0) / weight_raw
                        ),
                        "signed": max(
                            -1.0,
                            min(
                                1.0,
                                _safe_float(cluster_row.get("signed", 0.0), 0.0)
                                / weight_raw,
                            ),
                        ),
                        "weight_raw": weight_raw,
                        "weight": 0.0,
                    }
                )
            clusters.sort(
                key=lambda row: _safe_float(row.get("weight_raw", 0.0), 0.0),
                reverse=True,
            )
            if len(clusters) > 8:
                clusters = clusters[:8]

            cluster_weight_total = 0.0
            for row in clusters:
                cluster_weight_total += _safe_float(row.get("weight_raw", 0.0), 0.0)
            if cluster_weight_total > 1e-8:
                for row in clusters:
                    row["weight"] = _clamp01(
                        _safe_float(row.get("weight_raw", 0.0), 0.0)
                        / cluster_weight_total
                    )

            local_density_ratio = _clamp01(local_density_score / 3.0)
            cluster_ratio = _clamp01(len(clusters) / 6.0)

            field_center_x = anchor_x
            field_center_y = anchor_y
            if clusters:
                primary_cluster = clusters[0]
                cluster_pull = _clamp01(
                    0.22
                    + (local_density_ratio * 0.42)
                    + (file_influence * 0.28)
                    + (cluster_ratio * 0.2)
                )
                field_center_x = _clamp01(
                    (anchor_x * (1.0 - cluster_pull))
                    + (
                        _safe_float(primary_cluster.get("x", anchor_x), anchor_x)
                        * cluster_pull
                    )
                )
                field_center_y = _clamp01(
                    (anchor_y * (1.0 - cluster_pull))
                    + (
                        _safe_float(primary_cluster.get("y", anchor_y), anchor_y)
                        * cluster_pull
                    )
                )

            raw_count = (
                4.0
                + (world_influence * 4.0)
                + (file_influence * 4.2)
                + (local_density_ratio * 8.6)
                + (cluster_ratio * 2.2)
                - (resource_pressure * 1.2)
            )
            particle_count = max(4, min(22, int(round(raw_count))))

            short_range_radius = 0.16 + (local_density_ratio * 0.04)
            interaction_radius = 0.36
            long_range_radius = 0.92

            for local_index in range(particle_count):
                particle_id = f"field:{presence_id}:{local_index}"
                live_ids.add(particle_id)
                cache_row = particle_cache.get(particle_id, {})
                if not isinstance(cache_row, dict):
                    cache_row = {}

                seed_ratio = _stable_ratio(f"{particle_id}|seed", local_index + 11)
                base_angle = (seed_ratio * math.tau) + (
                    now * (0.09 + (world_influence * 0.22) + (compute_pressure * 0.08))
                )
                orbit_span = max(0.018, 0.085 - (local_density_ratio * 0.045))
                base_orbit = 0.008 + (
                    _stable_ratio(f"{particle_id}|orbit", local_index + 19) * orbit_span
                )
                home_x = _clamp01(field_center_x + (math.cos(base_angle) * base_orbit))
                home_y = _clamp01(
                    field_center_y + (math.sin(base_angle) * base_orbit * 0.82)
                )

                px = _clamp01(_safe_float(cache_row.get("x", home_x), home_x))
                py = _clamp01(_safe_float(cache_row.get("y", home_y), home_y))
                pvx = _safe_float(cache_row.get("vx", 0.0), 0.0)
                pvy = _safe_float(cache_row.get("vy", 0.0), 0.0)

                fx = (home_x - px) * (0.18 + (ledger_influence * 0.18))
                fy = (home_y - py) * (0.18 + (ledger_influence * 0.18))

                for node in node_signals:
                    dx = _safe_float(node.get("x", 0.5), 0.5) - px
                    dy = _safe_float(node.get("y", 0.5), 0.5) - py
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance <= 1e-8 or distance > interaction_radius:
                        continue

                    signed_similarity = max(
                        -1.0,
                        min(1.0, _safe_float(node.get("signed", 0.0), 0.0)),
                    )
                    if abs(signed_similarity) <= 0.03:
                        continue
                    node_importance = _clamp01(
                        _safe_float(node.get("importance", 0.25), 0.25)
                    )

                    if distance <= short_range_radius:
                        falloff = _clamp01(1.0 - (distance / short_range_radius))
                        strength = (
                            (0.00125 + (node_importance * 0.00245))
                            * (falloff * falloff)
                            * (0.78 + (abs(signed_similarity) * 0.94))
                            * (0.72 + (file_influence * 0.58))
                        )
                    else:
                        transition = max(1e-8, interaction_radius - short_range_radius)
                        band = _clamp01((interaction_radius - distance) / transition)
                        strength = (
                            (0.00024 + (node_importance * 0.00082))
                            * band
                            * (0.46 + (abs(signed_similarity) * 0.54))
                        )

                    direction = 1.0 if signed_similarity >= 0.0 else -1.0
                    ux = dx / distance
                    uy = dy / distance
                    fx += ux * strength * direction
                    fy += uy * strength * direction

                for cluster in clusters:
                    dx = _safe_float(cluster.get("x", 0.5), 0.5) - px
                    dy = _safe_float(cluster.get("y", 0.5), 0.5) - py
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance <= short_range_radius or distance > long_range_radius:
                        continue

                    cluster_signed = max(
                        -1.0,
                        min(1.0, _safe_float(cluster.get("signed", 0.0), 0.0)),
                    )
                    if abs(cluster_signed) <= 0.04:
                        continue
                    cluster_weight = _clamp01(
                        _safe_float(cluster.get("weight", 0.0), 0.0)
                    )
                    range_span = max(1e-8, long_range_radius - short_range_radius)
                    falloff = _clamp01((long_range_radius - distance) / range_span)
                    strength = (
                        (0.00012 + (cluster_weight * 0.00044))
                        * falloff
                        * (0.54 + (abs(cluster_signed) * 0.56))
                        * (0.6 + (cluster_ratio * 0.5))
                    )
                    direction = 1.0 if cluster_signed >= 0.0 else -1.0
                    ux = dx / distance
                    uy = dy / distance
                    fx += ux * strength * direction
                    fy += uy * strength * direction

                for embed in embedding_nodes:
                    ex = _clamp01(_safe_float(embed.get("x", 0.5), 0.5))
                    ey = _clamp01(_safe_float(embed.get("y", 0.5), 0.5))
                    dx = ex - px
                    dy = ey - py
                    distance = math.sqrt((dx * dx) + (dy * dy))
                    if distance <= 1e-8 or distance > 0.23:
                        continue
                    falloff = _clamp01(1.0 - (distance / 0.23))
                    if falloff <= 0.0:
                        continue
                    cohesion = _clamp01(_safe_float(embed.get("cohesion", 0.0), 0.0))
                    density = _clamp01(
                        _safe_float(embed.get("text_density", 0.45), 0.45)
                    )
                    signed = (
                        (file_influence * 0.74)
                        + (cohesion * 0.52)
                        + (density * 0.26)
                        - 0.58
                    )
                    direction = 1.0 if signed >= 0.0 else -1.0
                    strength = (0.00042 + (abs(signed) * 0.00108)) * (falloff * falloff)
                    ux = dx / distance
                    uy = dy / distance
                    fx += ux * strength * direction
                    fy += uy * strength * direction

                jitter_angle = (now * (0.34 + (compute_pressure * 0.4))) + (
                    local_index * 0.93
                )
                jitter_power = (
                    0.00006
                    + ((1.0 - resource_pressure) * 0.0001)
                    + (local_density_ratio * 0.00005)
                )
                fx += math.cos(jitter_angle) * jitter_power
                fy += math.sin(jitter_angle) * jitter_power

                damping = max(0.74, 0.91 - (resource_pressure * 0.13))
                vx = (pvx * damping) + fx
                vy = (pvy * damping) + fy
                speed = math.sqrt((vx * vx) + (vy * vy))
                speed_limit = (
                    0.0042
                    + ((1.0 - resource_pressure) * 0.0021)
                    + (local_density_ratio * 0.0018)
                )
                if speed > speed_limit and speed > 1e-8:
                    scale = speed_limit / speed
                    vx *= scale
                    vy *= scale

                nx = _clamp01(px + vx)
                ny = _clamp01(py + vy)
                particle_cache[particle_id] = {
                    "x": nx,
                    "y": ny,
                    "vx": vx,
                    "vy": vy,
                    "ts": now_mono,
                }

                saturation = max(
                    0.32,
                    min(
                        0.58,
                        0.4 + (world_influence * 0.16) + (local_density_ratio * 0.06),
                    ),
                )
                value = max(
                    0.38,
                    min(
                        0.68,
                        0.48
                        + (ledger_influence * 0.12)
                        + (local_density_ratio * 0.06)
                        - (resource_pressure * 0.12),
                    ),
                )
                r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
                    (base_hue % 360.0) / 360.0,
                    saturation,
                    value,
                )
                particle_size = (
                    0.9
                    + (world_influence * 1.0)
                    + (file_influence * 0.8)
                    + (local_density_ratio * 0.9)
                )

                field_particles.append(
                    {
                        "id": particle_id,
                        "presence_id": presence_id,
                        "presence_role": presence_role,
                        "particle_mode": particle_mode,
                        "x": round(nx, 5),
                        "y": round(ny, 5),
                        "size": round(particle_size, 5),
                        "r": round(_clamp01(r_raw), 5),
                        "g": round(_clamp01(g_raw), 5),
                        "b": round(_clamp01(b_raw), 5),
                    }
                )

        stale_before = now_mono - 180.0
        for pid in list(particle_cache.keys()):
            if pid in live_ids:
                continue
            row = particle_cache.get(pid, {})
            ts_value = _safe_float(
                (row if isinstance(row, dict) else {}).get("ts", 0.0), 0.0
            )
            if ts_value < stale_before:
                particle_cache.pop(pid, None)

        _DAIMO_DYNAMICS_CACHE["field_particles"] = particle_cache

    field_particles.sort(
        key=lambda row: (
            str(row.get("presence_id", "")),
            str(row.get("id", "")),
        )
    )
    return field_particles


_PARTICLE_ROLE_BY_PRESENCE: dict[str, str] = {
    "witness_thread": "crawl-routing",
    "keeper_of_receipts": "file-analysis",
    "mage_of_receipts": "image-captioning",
    "anchor_registry": "council-orchestration",
    "gates_of_truth": "compliance-gating",
}


def _particle_role_and_mode_for_presence(presence_id: str) -> tuple[str, str]:
    clean_presence_id = str(presence_id).strip()
    if not clean_presence_id:
        return "neutral", "neutral"
    role = str(_PARTICLE_ROLE_BY_PRESENCE.get(clean_presence_id, "")).strip()
    if not role:
        return "neutral", "neutral"
    return role, "role-bound"


def _dominant_eta_mu_field(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "f6", 1.0
    dominant_field = max(
        scores.keys(),
        key=lambda key: _safe_float(scores.get(key, 0.0), 0.0),
    )
    return dominant_field, _safe_float(scores.get(dominant_field, 0.0), 0.0)


def _infer_weaver_field_scores(node: dict[str, Any]) -> dict[str, float]:
    scores = {field_id: 0.0 for field_id in FIELD_TO_PRESENCE}
    kind = str(node.get("kind", "")).strip().lower()
    url = str(node.get("url", "") or node.get("label", "")).strip().lower()
    domain = str(node.get("domain", "")).strip().lower()
    title = str(node.get("title", "")).strip().lower()
    content_type = str(node.get("content_type", "")).strip().lower()

    if kind == "url":
        scores["f2"] += 0.24
        scores["f6"] += 0.24
        scores["f3"] += 0.12
    elif kind == "domain":
        scores["f2"] += 0.32
        scores["f8"] += 0.22
        scores["f3"] += 0.1
    elif kind == "content":
        if (
            content_type.startswith("image/")
            or content_type.startswith("audio/")
            or content_type.startswith("video/")
        ):
            scores["f1"] += 0.5
        else:
            scores["f6"] += 0.42
            scores["f3"] += 0.18

    combined = " ".join(filter(None, [url, domain, title, content_type]))
    tokens = _clean_tokens(combined)
    for token in tokens:
        for field_id, keywords in ETA_MU_FIELD_KEYWORDS.items():
            if token in keywords:
                scores[field_id] += 0.06

    for needle in ("policy", "privacy", "terms", "robots", "compliance", "license"):
        if needle in combined:
            scores["f7"] += 0.15
    for needle in ("blog", "news", "article", "docs", "wiki", "readme"):
        if needle in combined:
            scores["f6"] += 0.12
            scores["f3"] += 0.08
    for needle in ("status", "dashboard", "metrics", "api", "admin"):
        if needle in combined:
            scores["f8"] += 0.11

    total = sum(max(0.0, value) for value in scores.values())
    if total <= 0.0:
        fallback = "f2" if kind in {"domain", "url"} else "f6"
        scores[fallback] = 1.0
        return scores

    normalized: dict[str, float] = {}
    for field_id, value in scores.items():
        normalized[field_id] = round(max(0.0, value) / total, 4)
    return normalized


def _crawler_node_importance(node: dict[str, Any], dominant_weight: float) -> float:
    kind = str(node.get("kind", "")).strip().lower()
    if kind == "domain":
        return _clamp01(0.35 + (dominant_weight * 0.55))
    if kind == "content":
        return _clamp01(0.28 + (dominant_weight * 0.5))

    depth = _safe_float(node.get("depth", 0), 0.0)
    status = str(node.get("status", "")).strip().lower()
    compliance = str(node.get("compliance", "")).strip().lower()
    score = 0.22 + (dominant_weight * 0.5)
    score += _clamp01(1.0 - (depth / 8.0)) * 0.18
    if status in {"fetched", "duplicate"}:
        score += 0.08
    if compliance in {"allowed", "pending"}:
        score += 0.05
    return _clamp01(score)


def _build_weaver_field_graph_uncached(
    part_root: Path,
    vault_root: Path,
    *,
    fetcher: Any | None = None,
) -> dict[str, Any]:
    del vault_root
    source_fetcher = fetcher or _world_web_symbol(
        "_fetch_weaver_graph_payload", _fetch_weaver_graph_payload
    )
    source_payload = source_fetcher(part_root)
    graph_payload = (
        source_payload.get("graph", {}) if isinstance(source_payload, dict) else {}
    )
    status_payload = (
        source_payload.get("status", {}) if isinstance(source_payload, dict) else {}
    )
    if not isinstance(graph_payload, dict):
        graph_payload = {}
    if not isinstance(status_payload, dict):
        status_payload = {}

    raw_nodes = graph_payload.get("nodes", [])
    raw_edges = graph_payload.get("edges", [])
    if not isinstance(raw_nodes, list):
        raw_nodes = []
    if not isinstance(raw_edges, list):
        raw_edges = []

    entity_lookup = {
        str(entity.get("id", "")): entity
        for entity in ENTITY_MANIFEST
        if str(entity.get("id", "")).strip()
    }

    field_nodes: list[dict[str, Any]] = []
    for field_id in CANONICAL_NAMED_FIELD_IDS:
        entity = entity_lookup.get(field_id)
        if entity is None:
            continue
        mapped_field = next(
            (
                key
                for key, presence_id in FIELD_TO_PRESENCE.items()
                if presence_id == field_id
            ),
            "f3",
        )
        field_nodes.append(
            {
                "id": f"crawler-field:{field_id}",
                "node_id": field_id,
                "node_type": "field",
                "field": mapped_field,
                "label": str(entity.get("en", field_id)),
                "label_ja": str(entity.get("ja", "")),
                "x": round(_safe_float(entity.get("x", 0.5), 0.5), 4),
                "y": round(_safe_float(entity.get("y", 0.5), 0.5), 4),
                "hue": int(_safe_float(entity.get("hue", 200), 200.0)),
            }
        )

    crawler_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_id_map: dict[str, str] = {}
    kind_counts: dict[str, int] = defaultdict(int)
    field_counts: dict[str, int] = defaultdict(int)

    for index, node in enumerate(raw_nodes[:WEAVER_GRAPH_NODE_LIMIT]):
        if not isinstance(node, dict):
            continue
        original_id = str(node.get("id", "")).strip()
        if not original_id:
            continue
        scores = _infer_weaver_field_scores(node)
        dominant_field, dominant_weight = _dominant_eta_mu_field(scores)
        dominant_presence = FIELD_TO_PRESENCE.get(dominant_field, "anchor_registry")
        anchor = entity_lookup.get(dominant_presence, {"x": 0.5, "y": 0.5, "hue": 200})
        seed = sha1(f"crawler|{original_id}|{index}".encode("utf-8")).digest()
        angle = (int.from_bytes(seed[0:2], "big") / 65535.0) * math.tau
        radius = 0.05 + (int.from_bytes(seed[2:4], "big") / 65535.0) * 0.2
        jitter_x = ((seed[4] / 255.0) - 0.5) * 0.042
        jitter_y = ((seed[5] / 255.0) - 0.5) * 0.042
        x = _clamp01(
            _safe_float(anchor.get("x", 0.5), 0.5) + math.cos(angle) * radius + jitter_x
        )
        y = _clamp01(
            _safe_float(anchor.get("y", 0.5), 0.5) + math.sin(angle) * radius + jitter_y
        )
        kind = str(node.get("kind", "url")).strip().lower() or "url"
        if kind == "domain":
            hue = 176
        elif kind == "content":
            hue = 22
        else:
            hue = int(_safe_float(anchor.get("hue", 200), 200.0))

        graph_node_id = f"crawler:{sha1(original_id.encode('utf-8')).hexdigest()[:16]}"
        node_id_map[original_id] = graph_node_id
        kind_counts[kind] += 1
        field_counts[dominant_field] += 1
        importance = _crawler_node_importance(node, dominant_weight)
        label = str(
            node.get("title", "")
            or node.get("domain", "")
            or node.get("label", "")
            or original_id
        )
        crawler_nodes.append(
            {
                "id": graph_node_id,
                "node_id": original_id,
                "node_type": "crawler",
                "crawler_kind": kind,
                "label": label,
                "x": round(x, 4),
                "y": round(y, 4),
                "hue": int(hue),
                "importance": round(importance, 4),
                "url": str(node.get("url", "") or ""),
                "domain": str(node.get("domain", "") or ""),
                "title": str(node.get("title", "") or ""),
                "status": str(node.get("status", "") or ""),
                "content_type": str(node.get("content_type", "") or ""),
                "compliance": str(node.get("compliance", "") or ""),
                "dominant_field": dominant_field,
                "dominant_presence": dominant_presence,
                "field_scores": {
                    key: round(_safe_float(value, 0.0), 4)
                    for key, value in scores.items()
                },
            }
        )

        ranked = sorted(
            [
                (str(field), _safe_float(weight, 0.0))
                for field, weight in scores.items()
            ],
            key=lambda row: row[1],
            reverse=True,
        )
        for edge_index, (field_id, weight) in enumerate(ranked[:2]):
            if weight <= 0:
                continue
            target_presence = FIELD_TO_PRESENCE.get(field_id, dominant_presence)
            if target_presence not in entity_lookup:
                continue
            edges.append(
                {
                    "id": f"crawler-edge:{graph_node_id}:{field_id}:{edge_index}",
                    "source": graph_node_id,
                    "target": f"crawler-field:{target_presence}",
                    "field": field_id,
                    "weight": round(_clamp01(weight), 4),
                    "kind": "categorizes",
                }
            )

    for edge in raw_edges[:WEAVER_GRAPH_EDGE_LIMIT]:
        if not isinstance(edge, dict):
            continue
        source_id = node_id_map.get(str(edge.get("source", "")).strip())
        target_id = node_id_map.get(str(edge.get("target", "")).strip())
        if not source_id or not target_id:
            continue
        kind = str(edge.get("kind", "hyperlink") or "hyperlink")
        if kind == "domain_membership":
            weight = 0.25
        elif kind == "content_membership":
            weight = 0.22
        elif kind == "canonical_redirect":
            weight = 0.34
        else:
            weight = 0.28
        edges.append(
            {
                "id": f"crawl-link:{str(edge.get('id', '')) or sha1((source_id + target_id + kind).encode('utf-8')).hexdigest()[:14]}",
                "source": source_id,
                "target": target_id,
                "field": "",
                "weight": round(weight, 4),
                "kind": kind,
            }
        )

    graph_counts = graph_payload.get("counts", {})
    if not isinstance(graph_counts, dict):
        graph_counts = {}
    nodes = [*field_nodes, *crawler_nodes]
    return {
        "record": ETA_MU_CRAWLER_GRAPH_RECORD,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "endpoint": str(source_payload.get("source", "")),
            "service": "web-graph-weaver",
        },
        "status": status_payload,
        "nodes": nodes,
        "field_nodes": field_nodes,
        "crawler_nodes": crawler_nodes,
        "edges": edges,
        "stats": {
            "field_count": len(field_nodes),
            "crawler_count": len(crawler_nodes),
            "edge_count": len(edges),
            "kind_counts": dict(kind_counts),
            "field_counts": dict(field_counts),
            "nodes_total": int(
                _safe_float(
                    graph_counts.get("nodes_total", len(crawler_nodes)),
                    float(len(crawler_nodes)),
                )
            ),
            "edges_total": int(
                _safe_float(
                    graph_counts.get("edges_total", len(edges)), float(len(edges))
                )
            ),
            "url_nodes_total": int(
                _safe_float(
                    graph_counts.get("url_nodes_total", kind_counts.get("url", 0)),
                    float(kind_counts.get("url", 0)),
                )
            ),
        },
    }


def build_weaver_field_graph(part_root: Path, vault_root: Path) -> dict[str, Any]:
    fetcher = _world_web_symbol(
        "_fetch_weaver_graph_payload", _fetch_weaver_graph_payload
    )
    if fetcher is not _fetch_weaver_graph_payload:
        return _build_weaver_field_graph_uncached(
            part_root,
            vault_root,
            fetcher=fetcher,
        )

    substrate_root = _eta_mu_substrate_root(vault_root)
    cache_key = f"{part_root.resolve()}|{substrate_root}|{_weaver_service_base_url()}"
    now_monotonic = time.monotonic()
    with _WEAVER_GRAPH_CACHE_LOCK:
        cached_key = str(_WEAVER_GRAPH_CACHE.get("key", ""))
        cached_snapshot = _WEAVER_GRAPH_CACHE.get("snapshot")
        elapsed = now_monotonic - float(
            _WEAVER_GRAPH_CACHE.get("checked_monotonic", 0.0)
        )
        if (
            cached_snapshot is not None
            and cached_key == cache_key
            and elapsed < WEAVER_GRAPH_CACHE_SECONDS
        ):
            return _json_deep_clone(cached_snapshot)

    snapshot = _build_weaver_field_graph_uncached(part_root, vault_root)
    with _WEAVER_GRAPH_CACHE_LOCK:
        _WEAVER_GRAPH_CACHE["key"] = cache_key
        _WEAVER_GRAPH_CACHE["snapshot"] = _json_deep_clone(snapshot)
        _WEAVER_GRAPH_CACHE["checked_monotonic"] = now_monotonic
    return snapshot


def build_simulation_state(
    catalog: dict[str, Any],
    myth_summary: dict[str, Any] | None = None,
    world_summary: dict[str, Any] | None = None,
    *,
    influence_snapshot: dict[str, Any] | None = None,
    queue_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = time.time()
    resource_budget_snapshot = _resource_monitor_snapshot()
    budget_devices = (
        resource_budget_snapshot.get("devices", {})
        if isinstance(resource_budget_snapshot, dict)
        else {}
    )
    budget_cpu = _safe_float(
        (budget_devices.get("cpu", {}) if isinstance(budget_devices, dict) else {}).get(
            "utilization", 0.0
        ),
        0.0,
    )
    if budget_cpu >= 90.0:
        sim_point_budget = max(256, int(MAX_SIM_POINTS * 0.55))
    elif budget_cpu >= 78.0:
        sim_point_budget = max(320, int(MAX_SIM_POINTS * 0.74))
    else:
        sim_point_budget = MAX_SIM_POINTS

    points: list[dict[str, float]] = []
    embedding_particle_points_raw: list[dict[str, float]] = []
    emitted_embedding_particles: list[dict[str, float]] = []
    field_particle_points_raw: list[dict[str, float | str]] = []
    emitted_field_particles: list[dict[str, float | str]] = []
    items = catalog.get("items", [])
    file_graph = catalog.get("file_graph") if isinstance(catalog, dict) else None
    if isinstance(file_graph, dict):
        file_graph = _json_deep_clone(file_graph)
        embedding_particle_points_raw = _apply_file_graph_document_similarity_layout(
            file_graph,
            now=now,
        )
    crawler_graph = catalog.get("crawler_graph") if isinstance(catalog, dict) else None
    truth_state = catalog.get("truth_state") if isinstance(catalog, dict) else None
    logical_graph = catalog.get("logical_graph") if isinstance(catalog, dict) else None
    if not isinstance(logical_graph, dict):
        logical_graph = _build_logical_graph(
            catalog if isinstance(catalog, dict) else {}
        )
    pain_field = catalog.get("pain_field") if isinstance(catalog, dict) else None
    if not isinstance(pain_field, dict):
        pain_field = _build_pain_field(
            catalog if isinstance(catalog, dict) else {}, logical_graph
        )
    heat_values = _materialize_heat_values(
        catalog if isinstance(catalog, dict) else {}, pain_field
    )
    graph_file_nodes = (
        file_graph.get("file_nodes", []) if isinstance(file_graph, dict) else []
    )
    graph_crawler_nodes = (
        crawler_graph.get("crawler_nodes", [])
        if isinstance(crawler_graph, dict)
        else []
    )

    for idx, item in enumerate(items[:sim_point_budget]):
        key = f"{item.get('rel_path', '')}|{item.get('part', '')}|{item.get('kind', '')}|{idx}".encode(
            "utf-8"
        )
        digest = sha1(key).digest()

        x = (int.from_bytes(digest[0:2], "big") / 65535.0) * 2.0 - 1.0
        base_y = (int.from_bytes(digest[2:4], "big") / 65535.0) * 2.0 - 1.0
        phase = (digest[4] / 255.0) * math.tau
        speed = 0.4 + (digest[5] / 255.0) * 0.9
        wobble = math.sin(now * speed + phase) * 0.11
        y = max(-1.0, min(1.0, base_y + wobble))

        size = 2.8 + (digest[6] / 255.0) * 9.0
        r = 0.2 + (digest[7] / 255.0) * 0.75
        g = 0.2 + (digest[8] / 255.0) * 0.75
        b = 0.2 + (digest[9] / 255.0) * 0.75

        kind = str(item.get("kind", ""))
        if kind == "audio":
            size += 2.2
            r = min(1.0, r + 0.18)
            g = min(1.0, g + 0.16)
        elif kind == "video":
            b = min(1.0, b + 0.2)
        elif kind == "image":
            g = min(1.0, g + 0.1)

        points.append(
            {
                "x": round(x, 5),
                "y": round(y, 5),
                "size": round(size, 5),
                "r": round(r, 5),
                "g": round(g, 5),
                "b": round(b, 5),
            }
        )

    remaining_capacity = max(0, sim_point_budget - len(points))
    for node in list(graph_file_nodes)[:remaining_capacity]:
        if not isinstance(node, dict):
            continue
        x_norm = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
        y_norm = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
        hue = _safe_float(node.get("hue", 200), 200.0)
        importance = _clamp01(_safe_float(node.get("importance", 0.4), 0.4))
        r_raw, g_raw, b_raw = colorsys.hsv_to_rgb((hue % 360.0) / 360.0, 0.58, 0.95)
        points.append(
            {
                "x": round((x_norm * 2.0) - 1.0, 5),
                "y": round(1.0 - (y_norm * 2.0), 5),
                "size": round(2.6 + (importance * 6.2), 5),
                "r": round(r_raw, 5),
                "g": round(g_raw, 5),
                "b": round(b_raw, 5),
            }
        )

    remaining_capacity = max(0, sim_point_budget - len(points))
    for particle in embedding_particle_points_raw[:remaining_capacity]:
        if not isinstance(particle, dict):
            continue
        particle_row = {
            "x": round(_safe_float(particle.get("x", 0.0), 0.0), 5),
            "y": round(_safe_float(particle.get("y", 0.0), 0.0), 5),
            "size": round(max(0.4, _safe_float(particle.get("size", 1.0), 1.0)), 5),
            "r": round(_clamp01(_safe_float(particle.get("r", 0.5), 0.5)), 5),
            "g": round(_clamp01(_safe_float(particle.get("g", 0.5), 0.5)), 5),
            "b": round(_clamp01(_safe_float(particle.get("b", 0.5), 0.5)), 5),
        }
        points.append(particle_row)
        emitted_embedding_particles.append(dict(particle_row))

    remaining_capacity = max(0, sim_point_budget - len(points))
    for node in list(graph_crawler_nodes)[:remaining_capacity]:
        if not isinstance(node, dict):
            continue
        x_norm = _clamp01(_safe_float(node.get("x", 0.5), 0.5))
        y_norm = _clamp01(_safe_float(node.get("y", 0.5), 0.5))
        hue = _safe_float(node.get("hue", 180), 180.0)
        importance = _clamp01(_safe_float(node.get("importance", 0.3), 0.3))
        crawler_kind = str(node.get("crawler_kind", "url")).strip().lower()
        saturation = 0.66 if crawler_kind == "url" else 0.52
        value = 0.96 if crawler_kind == "url" else 0.9
        r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
            (hue % 360.0) / 360.0, saturation, value
        )
        points.append(
            {
                "x": round((x_norm * 2.0) - 1.0, 5),
                "y": round(1.0 - (y_norm * 2.0), 5),
                "size": round(2.2 + (importance * 5.0), 5),
                "r": round(r_raw, 5),
                "g": round(g_raw, 5),
                "b": round(b_raw, 5),
            }
        )

    truth_claims = (
        truth_state.get("claims", []) if isinstance(truth_state, dict) else []
    )
    if not isinstance(truth_claims, list):
        truth_claims = []
    truth_guard = truth_state.get("guard", {}) if isinstance(truth_state, dict) else {}
    if not isinstance(truth_guard, dict):
        truth_guard = {}
    truth_gate = truth_state.get("gate", {}) if isinstance(truth_state, dict) else {}
    if not isinstance(truth_gate, dict):
        truth_gate = {}
    truth_gate_blocked = bool(truth_gate.get("blocked", True))
    truth_guard_pass = bool(truth_guard.get("passes", False))

    remaining_capacity = max(0, sim_point_budget - len(points))
    if remaining_capacity > 0 and truth_claims:
        claim_x = 0.76
        claim_y = 0.54
        for claim_index, claim in enumerate(truth_claims[: min(3, remaining_capacity)]):
            if not isinstance(claim, dict):
                continue
            kappa = _clamp01(_safe_float(claim.get("kappa", 0.0), 0.0))
            status = str(claim.get("status", "undecided")).strip().lower()
            if status == "proved":
                hue = 136.0
            elif status == "refuted":
                hue = 12.0
            else:
                hue = 52.0
            if truth_guard_pass:
                hue = 150.0
            elif truth_gate_blocked:
                hue = max(0.0, hue - 12.0)

            orbit = 0.012 + (claim_index * 0.014)
            phase = now * (0.45 + claim_index * 0.11)
            x_norm = _clamp01(claim_x + (math.cos(phase) * orbit))
            y_norm = _clamp01(claim_y + (math.sin(phase) * orbit))
            saturation = 0.72 if status == "proved" else 0.78
            value = 0.96 if status == "proved" else 0.88
            r_raw, g_raw, b_raw = colorsys.hsv_to_rgb(
                (hue % 360.0) / 360.0, saturation, value
            )
            points.append(
                {
                    "x": round((x_norm * 2.0) - 1.0, 5),
                    "y": round(1.0 - (y_norm * 2.0), 5),
                    "size": round(3.2 + (kappa * 5.8), 5),
                    "r": round(r_raw, 5),
                    "g": round(g_raw, 5),
                    "b": round(b_raw, 5),
                }
            )

    counts = catalog.get("counts", {})

    entity_states = []
    for e in ENTITY_MANIFEST:
        base_seed = int(sha1(e["id"].encode("utf-8")).hexdigest()[:8], 16)
        t = now + (base_seed % 1000)
        bpm = 60 + (math.sin(t * 0.1) * 20) + ((base_seed % 20) - 10)

        vitals = {}
        for k, unit in e.get("flavor_vitals", {}).items():
            val_seed = (base_seed + hash(k)) % 1000
            val = abs(
                math.sin(t * (0.05 + (val_seed % 10) / 100)) * (100 + (val_seed % 50))
            )
            if unit == "%":
                val = val % 100
            vitals[k] = f"{val:.1f}{unit}"

        entity_states.append(
            {
                "id": e["id"],
                "bpm": round(bpm, 1),
                "stability": round(90 + math.sin(t * 0.02) * 9, 1),
                "resonance": round(e["freq"] + math.sin(t) * 2, 1),
                "vitals": vitals,
            }
        )

    echo_particles = []
    collection = _get_chroma_collection()
    if collection:
        try:
            results = collection.get(limit=12)
            docs = results.get("documents", [])
            for i, doc in enumerate(docs):
                seed = int(sha1(doc.encode("utf-8")).hexdigest()[:8], 16)
                t_off = now + (seed % 500)
                echo_particles.append(
                    {
                        "id": f"echo_{i}",
                        "text": doc[:24] + "...",
                        "x": 0.5 + math.sin(t_off * 0.15) * 0.35,
                        "y": 0.5 + math.cos(t_off * 0.12) * 0.35,
                        "hue": (200 + (seed % 100)) % 360,
                        "life": 0.5 + math.sin(t_off * 0.5) * 0.5,
                    }
                )
        except Exception:
            pass

    queue_snapshot = queue_snapshot or {}
    influence = influence_snapshot or _INFLUENCE_TRACKER.snapshot(
        queue_snapshot=queue_snapshot
    )

    clicks_recent = int(influence.get("clicks_45s", 0))
    file_changes_recent = int(influence.get("file_changes_120s", 0))
    queue_pending_count = int(queue_snapshot.get("pending_count", 0))
    queue_event_count = int(queue_snapshot.get("event_count", 0))

    audio_count = int(counts.get("audio", 0))
    audio_ratio = _clamp01(audio_count / 12.0)
    click_ratio = _clamp01(clicks_recent / 18.0)
    file_ratio = _clamp01(file_changes_recent / 24.0)
    queue_ratio = _clamp01((queue_pending_count + queue_event_count * 0.25) / 16.0)
    resource_heartbeat = (
        influence.get("resource_heartbeat", {}) if isinstance(influence, dict) else {}
    )
    if not isinstance(resource_heartbeat, dict) or not resource_heartbeat:
        resource_heartbeat = resource_budget_snapshot
    resource_devices = (
        resource_heartbeat.get("devices", {})
        if isinstance(resource_heartbeat, dict)
        else {}
    )
    resource_cpu_util = _safe_float(
        (
            resource_devices.get("cpu", {})
            if isinstance(resource_devices, dict)
            else {}
        ).get("utilization", 0.0),
        0.0,
    )
    resource_gpu_util = _safe_float(
        (
            resource_devices.get("gpu1", {})
            if isinstance(resource_devices, dict)
            else {}
        ).get("utilization", 0.0),
        0.0,
    )
    resource_npu_util = _safe_float(
        (
            resource_devices.get("npu0", {})
            if isinstance(resource_devices, dict)
            else {}
        ).get("utilization", 0.0),
        0.0,
    )
    resource_ratio = _clamp01(
        max(resource_cpu_util, resource_gpu_util, resource_npu_util) / 100.0
    )

    river_flow_rate = round(
        1.2 + (audio_ratio * 4.4) + (file_ratio * 7.2) + (click_ratio * 2.6), 3
    )
    river_turbulence = round(_clamp01((file_ratio * 0.72) + (click_ratio * 0.4)), 4)

    manifest_lookup = {
        str(item.get("id", "")): item for item in ENTITY_MANIFEST if item.get("id")
    }
    impact_order = [
        *CANONICAL_NAMED_FIELD_IDS,
        FILE_SENTINEL_PROFILE["id"],
        FILE_ORGANIZER_PROFILE["id"],
        HEALTH_SENTINEL_CPU_PROFILE["id"],
        HEALTH_SENTINEL_GPU1_PROFILE["id"],
        HEALTH_SENTINEL_GPU2_PROFILE["id"],
        HEALTH_SENTINEL_NPU0_PROFILE["id"],
    ]
    base_file = {
        "receipt_river": 0.94,
        "witness_thread": 0.38,
        "fork_tax_canticle": 0.84,
        "mage_of_receipts": 0.88,
        "keeper_of_receipts": 0.9,
        "anchor_registry": 0.64,
        "gates_of_truth": 0.73,
        "file_sentinel": 1.0,
        "file_organizer": 0.86,
        "health_sentinel_cpu": 0.58,
        "health_sentinel_gpu1": 0.54,
        "health_sentinel_gpu2": 0.5,
        "health_sentinel_npu0": 0.52,
    }
    base_click = {
        "receipt_river": 0.52,
        "witness_thread": 0.94,
        "fork_tax_canticle": 0.66,
        "mage_of_receipts": 0.57,
        "keeper_of_receipts": 0.61,
        "anchor_registry": 0.83,
        "gates_of_truth": 0.8,
        "file_sentinel": 0.55,
        "file_organizer": 0.62,
        "health_sentinel_cpu": 0.44,
        "health_sentinel_gpu1": 0.36,
        "health_sentinel_gpu2": 0.34,
        "health_sentinel_npu0": 0.32,
    }
    base_emit = {
        "receipt_river": 0.95,
        "witness_thread": 0.71,
        "fork_tax_canticle": 0.79,
        "mage_of_receipts": 0.73,
        "keeper_of_receipts": 0.81,
        "anchor_registry": 0.68,
        "gates_of_truth": 0.75,
        "file_sentinel": 0.82,
        "file_organizer": 0.78,
        "health_sentinel_cpu": 0.68,
        "health_sentinel_gpu1": 0.74,
        "health_sentinel_gpu2": 0.7,
        "health_sentinel_npu0": 0.77,
    }
    base_resource = {
        "receipt_river": 0.2,
        "witness_thread": 0.15,
        "fork_tax_canticle": 0.31,
        "mage_of_receipts": 0.28,
        "keeper_of_receipts": 0.26,
        "anchor_registry": 0.24,
        "gates_of_truth": 0.33,
        "file_sentinel": 0.44,
        "file_organizer": 0.36,
        "health_sentinel_cpu": 0.92,
        "health_sentinel_gpu1": 0.96,
        "health_sentinel_gpu2": 0.9,
        "health_sentinel_npu0": 0.95,
    }

    presence_impacts: list[dict[str, Any]] = []
    for presence_id in impact_order:
        if presence_id == FILE_SENTINEL_PROFILE["id"]:
            meta = FILE_SENTINEL_PROFILE
        elif presence_id == FILE_ORGANIZER_PROFILE["id"]:
            meta = FILE_ORGANIZER_PROFILE
        else:
            meta = manifest_lookup.get(
                presence_id,
                {
                    "id": presence_id,
                    "en": presence_id.replace("_", " ").title(),
                    "ja": "",
                },
            )

        file_influence = _clamp01(
            (file_ratio * float(base_file.get(presence_id, 0.5))) + (queue_ratio * 0.22)
        )
        click_influence = _clamp01(
            click_ratio * float(base_click.get(presence_id, 0.5))
        )
        resource_influence = _clamp01(
            resource_ratio * float(base_resource.get(presence_id, 0.22))
        )
        total_influence = _clamp01(
            (file_influence * 0.52)
            + (click_influence * 0.28)
            + (resource_influence * 0.2)
        )
        emits_flow = _clamp01(
            (total_influence * 0.72)
            + (audio_ratio * float(base_emit.get(presence_id, 0.5)) * 0.35)
        )

        if presence_id == "receipt_river":
            notes_en = (
                "River flow accelerates when files move and witnesses touch the field."
            )
            notes_ja = "ファイル変化と触れた証人で、川の流れは加速する。"
        elif presence_id == "file_sentinel":
            notes_en = "Auto-committing ghost stages proof paths before the gate asks."
            notes_ja = "自動コミットの幽霊は、門に問われる前に証明経路を段取る。"
        elif presence_id == "file_organizer":
            notes_en = "Organizer presence groups files into concept clusters from embedding space."
            notes_ja = "分類師プレゼンスは埋め込み空間から概念クラスタを編成する。"
        elif presence_id == "fork_tax_canticle":
            notes_en = "Fork tax pressure rises with unresolved file drift."
            notes_ja = "未解決のファイルドリフトでフォーク税圧は上がる。"
        elif presence_id == "witness_thread":
            notes_en = "Mouse touches tighten witness linkage across presences."
            notes_ja = "マウスの接触はプレゼンス間の証人連結を強める。"
        elif presence_id == "health_sentinel_cpu":
            notes_en = (
                "CPU sentinel throttles particle budgets when host pressure rises."
            )
            notes_ja = "CPU哨戒はホスト圧上昇時に粒子予算を絞る。"
        elif presence_id == "health_sentinel_gpu1":
            notes_en = (
                "GPU1 sentinel maps throughput and thermals into backend selection."
            )
            notes_ja = "GPU1哨戒は処理量と熱をバックエンド選択へ写像する。"
        elif presence_id == "health_sentinel_gpu2":
            notes_en = "GPU2 sentinel absorbs burst load to keep field vectors stable."
            notes_ja = "GPU2哨戒は突発負荷を吸収し、場のベクトルを安定化する。"
        elif presence_id == "health_sentinel_npu0":
            notes_en = (
                "NPU sentinel tracks efficient inferencing for embedding pathways."
            )
            notes_ja = "NPU哨戒は埋め込み経路の効率推論を監視する。"
        else:
            notes_en = "Presence responds to blended file and witness pressure."
            notes_ja = "このプレゼンスはファイル圧と証人圧の混合に応答する。"

        presence_impacts.append(
            {
                "id": presence_id,
                "en": str(meta.get("en", "Presence")),
                "ja": str(meta.get("ja", "プレゼンス")),
                "affected_by": {
                    "files": round(file_influence, 4),
                    "clicks": round(click_influence, 4),
                    "resource": round(resource_influence, 4),
                },
                "affects": {
                    "world": round(emits_flow, 4),
                    "ledger": round(_clamp01(total_influence * 0.86), 4),
                },
                "notes_en": notes_en,
                "notes_ja": notes_ja,
            }
        )

    witness_meta = manifest_lookup.get(
        "witness_thread",
        {
            "id": "witness_thread",
            "en": "Witness Thread",
            "ja": "証人の糸",
        },
    )
    witness_impact = next(
        (item for item in presence_impacts if item.get("id") == "witness_thread"), None
    )
    lineage: list[dict[str, str]] = []
    seen_lineage_refs: set[str] = set()

    for target in list(influence.get("recent_click_targets", []))[:6]:
        ref = str(target).strip()
        if not ref or ref in seen_lineage_refs:
            continue
        seen_lineage_refs.add(ref)
        lineage.append(
            {
                "kind": "touch",
                "ref": ref,
                "why_en": "Witness touch linked this target into continuity.",
                "why_ja": "証人の接触がこの対象を連続線へ接続した。",
            }
        )

    for path in list(influence.get("recent_file_paths", []))[:8]:
        ref = str(path).strip()
        if not ref or ref in seen_lineage_refs:
            continue
        seen_lineage_refs.add(ref)
        lineage.append(
            {
                "kind": "file",
                "ref": ref,
                "why_en": "File drift supplied provenance for witness continuity.",
                "why_ja": "ファイルドリフトが証人連続性の来歴を供給した。",
            }
        )

    if not lineage:
        lineage.append(
            {
                "kind": "idle",
                "ref": "awaiting-touch",
                "why_en": "No recent witness touch; continuity waits for the next trace.",
                "why_ja": "直近の証人接触なし。次の痕跡を待機中。",
            }
        )

    linked_presence_ids = [
        str(item.get("id", ""))
        for item in sorted(
            [row for row in presence_impacts if row.get("id") != "witness_thread"],
            key=lambda row: float(row.get("affected_by", {}).get("clicks", 0.0)),
            reverse=True,
        )
        if str(item.get("id", "")).strip()
    ][:4]

    witness_thread_state = {
        "id": str(witness_meta.get("id", "witness_thread")),
        "en": str(witness_meta.get("en", "Witness Thread")),
        "ja": str(witness_meta.get("ja", "証人の糸")),
        "continuity_index": round(
            _clamp01((click_ratio * 0.54) + (file_ratio * 0.3) + (queue_ratio * 0.16)),
            4,
        ),
        "click_pressure": round(click_ratio, 4),
        "file_pressure": round(file_ratio, 4),
        "linked_presences": linked_presence_ids,
        "lineage": lineage[:6],
        "notes_en": str(
            (witness_impact or {}).get(
                "notes_en",
                "Witness Thread binds touch and file drift into explicit continuity.",
            )
        ),
        "notes_ja": str(
            (witness_impact or {}).get(
                "notes_ja",
                "証人の糸は接触とファイルドリフトを明示的な連続性へ束ねる。",
            )
        ),
    }

    fork_tax = dict(influence.get("fork_tax", {}))
    if not fork_tax:
        fork_tax = {
            "law_en": "Pay the fork tax; annotate every drift with proof.",
            "law_ja": "フォーク税は法。",
            "debt": 0.0,
            "paid": 0.0,
            "balance": 0.0,
            "paid_ratio": 1.0,
        }
    if not str(fork_tax.get("law_ja", "")).strip():
        fork_tax["law_ja"] = "フォーク税は法。"

    ghost = dict(influence.get("ghost", {}))
    ghost.setdefault("id", FILE_SENTINEL_PROFILE["id"])
    ghost.setdefault("en", FILE_SENTINEL_PROFILE["en"])
    ghost.setdefault("ja", FILE_SENTINEL_PROFILE["ja"])
    ghost["auto_commit_pulse"] = round(
        _clamp01(
            float(ghost.get("auto_commit_pulse", 0.0))
            + (file_ratio * 0.12)
            + (queue_ratio * 0.08)
        ),
        4,
    )
    ghost["actions_60s"] = int((file_changes_recent * 0.5) + (queue_event_count * 0.8))
    ghost["status_en"] = str(ghost.get("status_en", "gate idle"))
    ghost["status_ja"] = str(ghost.get("status_ja", "門前で待機中"))

    ds = _build_daimoi_state(
        heat_values,
        pain_field,
        queue_ratio=queue_ratio,
        resource_ratio=resource_ratio,
    )

    compute_jobs_raw = influence.get("compute_jobs", [])
    compute_jobs = compute_jobs_raw if isinstance(compute_jobs_raw, list) else []
    compute_summary_raw = influence.get("compute_summary", {})
    compute_summary = (
        compute_summary_raw if isinstance(compute_summary_raw, dict) else {}
    )
    compute_jobs_count = int(
        _safe_float(influence.get("compute_jobs_180s", len(compute_jobs)), 0.0)
    )

    field_particle_points_raw = _build_backend_field_particles(
        file_graph=file_graph,
        presence_impacts=presence_impacts,
        resource_heartbeat=resource_heartbeat,
        compute_jobs=compute_jobs,
        now=now,
    )

    normalized_field_particles: list[dict[str, float | str]] = []
    for particle in field_particle_points_raw:
        if not isinstance(particle, dict):
            continue
        x_norm = _clamp01(_safe_float(particle.get("x", 0.5), 0.5))
        y_norm = _clamp01(_safe_float(particle.get("y", 0.5), 0.5))
        normalized_field_particles.append(
            {
                "id": str(particle.get("id", "")),
                "presence_id": str(particle.get("presence_id", "")),
                "presence_role": str(particle.get("presence_role", "neutral")),
                "particle_mode": str(particle.get("particle_mode", "neutral")),
                "x": round(x_norm, 5),
                "y": round(y_norm, 5),
                "size": round(max(0.6, _safe_float(particle.get("size", 1.0), 1.0)), 5),
                "r": round(_clamp01(_safe_float(particle.get("r", 0.4), 0.4)), 5),
                "g": round(_clamp01(_safe_float(particle.get("g", 0.4), 0.4)), 5),
                "b": round(_clamp01(_safe_float(particle.get("b", 0.4), 0.4)), 5),
            }
        )

    remaining_capacity = max(0, sim_point_budget - len(points))
    for particle in normalized_field_particles[:remaining_capacity]:
        points.append(
            {
                "x": round((_safe_float(particle.get("x", 0.5), 0.5) * 2.0) - 1.0, 5),
                "y": round(1.0 - (_safe_float(particle.get("y", 0.5), 0.5) * 2.0), 5),
                "size": round(max(0.6, _safe_float(particle.get("size", 1.0), 1.0)), 5),
                "r": round(_clamp01(_safe_float(particle.get("r", 0.4), 0.4)), 5),
                "g": round(_clamp01(_safe_float(particle.get("g", 0.4), 0.4)), 5),
                "b": round(_clamp01(_safe_float(particle.get("b", 0.4), 0.4)), 5),
            }
        )

    emitted_field_particles = normalized_field_particles

    presence_dynamics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "simulation_budget": {
            "point_limit": int(sim_point_budget),
            "point_limit_max": int(MAX_SIM_POINTS),
            "cpu_utilization": round(resource_cpu_util, 2),
        },
        "click_events": clicks_recent,
        "file_events": file_changes_recent,
        "recent_click_targets": list(influence.get("recent_click_targets", []))[:6],
        "recent_file_paths": list(influence.get("recent_file_paths", []))[:8],
        "resource_heartbeat": resource_heartbeat,
        "compute_jobs_180s": max(0, compute_jobs_count),
        "compute_summary": compute_summary,
        "compute_jobs": compute_jobs[:32],
        "field_particles_record": "ημ.field-particles.v1",
        "field_particles": emitted_field_particles,
        "river_flow": {
            "unit": "m3/s",
            "rate": river_flow_rate,
            "turbulence": river_turbulence,
        },
        "ghost": ghost,
        "fork_tax": fork_tax,
        "witness_thread": witness_thread_state,
        "presence_impacts": presence_impacts,
    }

    default_truth_state = {
        "record": "ημ.truth-state.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim": {"status": "undecided"},
        "claims": [],
        "proof": {"entries": []},
        "guard": {},
        "gate": {},
    }

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(points),
        "audio": int(counts.get("audio", 0)),
        "image": int(counts.get("image", 0)),
        "video": int(counts.get("video", 0)),
        "points": points,
        "embedding_particles": emitted_embedding_particles,
        "field_particles": emitted_field_particles,
        "file_graph": file_graph
        if isinstance(file_graph, dict)
        else {
            "record": ETA_MU_FILE_GRAPH_RECORD,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "inbox": {
                "record": "ημ.inbox.v1",
                "path": "",
                "pending_count": 0,
                "processed_count": 0,
                "failed_count": 0,
                "is_empty": True,
                "knowledge_entries": 0,
                "last_ingested_at": "",
                "errors": [],
            },
            "nodes": [],
            "field_nodes": [],
            "file_nodes": [],
            "embedding_particles": [],
            "edges": [],
            "stats": {
                "field_count": 0,
                "file_count": 0,
                "edge_count": 0,
                "kind_counts": {},
                "field_counts": {},
                "knowledge_entries": 0,
            },
        },
        "crawler_graph": crawler_graph
        if isinstance(crawler_graph, dict)
        else {
            "record": ETA_MU_CRAWLER_GRAPH_RECORD,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": {"endpoint": "", "service": "web-graph-weaver"},
            "status": {},
            "nodes": [],
            "field_nodes": [],
            "crawler_nodes": [],
            "edges": [],
            "stats": {
                "field_count": 0,
                "crawler_count": 0,
                "edge_count": 0,
                "kind_counts": {},
                "field_counts": {},
                "nodes_total": 0,
                "edges_total": 0,
                "url_nodes_total": 0,
            },
        },
        "truth_state": truth_state
        if isinstance(truth_state, dict)
        else default_truth_state,
        "logical_graph": logical_graph,
        "pain_field": pain_field,
        "heat_values": heat_values,
        "daimoi": ds,
        "entities": entity_states,
        "echoes": echo_particles,
        "fork_tax": fork_tax,
        "ghost": ghost,
        "presence_dynamics": presence_dynamics,
        "myth": myth_summary or {},
        "world": world_summary or {},
    }
