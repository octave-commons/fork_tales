"""Catalog stream chunking helpers for websocket/http streaming paths."""

from __future__ import annotations

import copy
from typing import Any, Iterator


def catalog_stream_chunk_rows(
    value: Any,
    *,
    default_chunk_rows: int,
    safe_float: Any,
) -> int:
    return max(
        1,
        min(
            2048,
            int(safe_float(value, float(default_chunk_rows))),
        ),
    )


def catalog_stream_get_path_value(
    payload: dict[str, Any], path: tuple[str, ...]
) -> Any:
    cursor: Any = payload
    for part in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
    return cursor


def catalog_stream_set_path_value(
    payload: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    if not path:
        return
    cursor: dict[str, Any] = payload
    for part in path[:-1]:
        nested = cursor.get(part)
        if not isinstance(nested, dict):
            nested = {}
            cursor[part] = nested
        cursor = nested
    cursor[path[-1]] = value


def catalog_stream_meta(
    catalog: dict[str, Any],
    *,
    section_paths: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, Any]:
    if not isinstance(catalog, dict):
        return {}
    meta = copy.deepcopy(catalog)
    for section_name, path in section_paths:
        rows = catalog_stream_get_path_value(catalog, path)
        if isinstance(rows, list):
            catalog_stream_set_path_value(
                meta,
                path,
                {
                    "streamed": True,
                    "section": section_name,
                    "count": len(rows),
                },
            )
    return meta


def catalog_stream_iter_rows(
    catalog: dict[str, Any],
    *,
    chunk_rows: int,
    section_paths: tuple[tuple[str, tuple[str, ...]], ...],
    default_chunk_rows: int,
    safe_float: Any,
) -> Iterator[dict[str, Any]]:
    chunk_size = catalog_stream_chunk_rows(
        chunk_rows,
        default_chunk_rows=default_chunk_rows,
        safe_float=safe_float,
    )
    catalog_payload = catalog if isinstance(catalog, dict) else {}
    section_stats: dict[str, dict[str, int]] = {}
    yield {
        "type": "meta",
        "record": "eta-mu.catalog.stream.meta.v1",
        "schema_version": "catalog.stream.meta.v1",
        "catalog": catalog_stream_meta(catalog_payload, section_paths=section_paths),
    }

    for section_name, path in section_paths:
        rows = catalog_stream_get_path_value(catalog_payload, path)
        if not isinstance(rows, list):
            continue
        total = len(rows)
        chunk_count = 0
        for offset in range(0, total, chunk_size):
            chunk = rows[offset : offset + chunk_size]
            yield {
                "type": "rows",
                "record": "eta-mu.catalog.stream.rows.v1",
                "schema_version": "catalog.stream.rows.v1",
                "section": section_name,
                "offset": offset,
                "rows": chunk,
            }
            chunk_count += 1
        section_stats[section_name] = {
            "total": total,
            "chunks": chunk_count,
        }

    yield {
        "type": "done",
        "ok": True,
        "record": "eta-mu.catalog.stream.done.v1",
        "schema_version": "catalog.stream.done.v1",
        "chunk_rows": chunk_size,
        "sections": section_stats,
    }
