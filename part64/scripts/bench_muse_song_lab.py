#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


AUDIO_SUFFIXES = (".mp3", ".wav", ".ogg", ".m4a", ".flac")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg")


@dataclass
class TaskSpec:
    task_id: str
    muse_id: str
    prompt: str
    mode: str


@dataclass
class RunSample:
    runtime: str
    task_id: str
    ok: bool
    latency_ms: float
    requested: bool
    blocked: bool
    action_status: str
    selected_label: str
    selected_node_id: str
    media_kind: str
    collisions: int


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _tokenize(text: str) -> list[str]:
    token = ""
    output: list[str] = []
    for char in str(text).lower():
        if char.isalnum() or char in {"_", ".", "/", "-"}:
            token += char
            continue
        if token:
            output.append(token)
            token = ""
    if token:
        output.append(token)
    return output


def _request_json(url: str, timeout_seconds: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout_seconds) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise RuntimeError(f"expected object payload from {url}")
    return data


def _post_json(
    url: str, body: dict[str, Any], timeout_seconds: float
) -> dict[str, Any]:
    payload_bytes = json.dumps(body).encode("utf-8")
    request = Request(
        url=url,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise RuntimeError(f"expected object payload from {url}")
        return data
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {"ok": False, "error": f"http_{exc.code}", "detail": text[:240]}
    except TimeoutError:
        return {"ok": False, "error": "timeout"}
    except URLError as exc:
        return {"ok": False, "error": f"url_error:{exc.reason}"}
    except Exception as exc:
        return {"ok": False, "error": f"request_failed:{exc.__class__.__name__}"}


def _parse_runtime_arg(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "runtime must be in label=http://host:port format"
        )
    label, url = raw.split("=", 1)
    clean_label = label.strip()
    clean_url = url.strip().rstrip("/")
    if not clean_label or not clean_url.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError(
            "runtime must be in label=http://host:port format"
        )
    return clean_label, clean_url


def _load_regimen(path: Path) -> tuple[list[TaskSpec], int]:
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("regimen must be a JSON object")
    rounds = max(1, int(payload.get("rounds", 3) or 3))
    tasks_raw = payload.get("tasks", [])
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise RuntimeError("regimen tasks must be a non-empty list")
    tasks: list[TaskSpec] = []
    for idx, row in enumerate(tasks_raw):
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("id", f"task_{idx + 1:02d}")).strip()
        muse_id = str(row.get("muse_id", "chaos")).strip() or "chaos"
        prompt = str(row.get("prompt", "")).strip()
        mode = str(row.get("mode", "deterministic")).strip().lower()
        if mode not in {"deterministic", "stochastic"}:
            mode = "deterministic"
        if not task_id or not prompt:
            continue
        tasks.append(
            TaskSpec(task_id=task_id, muse_id=muse_id, prompt=prompt, mode=mode)
        )
    if not tasks:
        raise RuntimeError("no valid tasks in regimen")
    return tasks, rounds


def _looks_audio(row: dict[str, Any]) -> bool:
    kind = str(row.get("kind", "")).strip().lower()
    source_rel_path = str(row.get("source_rel_path", "")).strip().lower()
    url = str(row.get("url", "")).strip().lower()
    label = str(row.get("label", "")).strip().lower()
    if kind == "audio" or kind.startswith("audio/"):
        return True
    fields = [source_rel_path, url, label, str(row.get("id", "")).strip().lower()]
    if any(field.endswith(AUDIO_SUFFIXES) for field in fields if field):
        return True
    joined = " ".join(fields)
    return (
        "artifacts/audio/" in joined
        or " song" in f" {joined}"
        or " music" in f" {joined}"
    )


def _looks_image(row: dict[str, Any]) -> bool:
    kind = str(row.get("kind", "")).strip().lower()
    source_rel_path = str(row.get("source_rel_path", "")).strip().lower()
    url = str(row.get("url", "")).strip().lower()
    label = str(row.get("label", "")).strip().lower()
    if kind == "image" or kind.startswith("image/") or kind == "cover_art":
        return True
    fields = [source_rel_path, url, label, str(row.get("id", "")).strip().lower()]
    if any(field.endswith(IMAGE_SUFFIXES) for field in fields if field):
        return True
    joined = " ".join(fields)
    return (
        "artifacts/images/" in joined
        or " image" in f" {joined}"
        or " photo" in f" {joined}"
    )


def _build_media_surrounding_pool(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    graph = catalog.get("file_graph", {}) if isinstance(catalog, dict) else {}
    file_nodes = graph.get("file_nodes", []) if isinstance(graph, dict) else []
    if not isinstance(file_nodes, list):
        return []
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in file_nodes:
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("id", raw.get("node_id", ""))).strip()
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        row = {
            "id": node_id,
            "kind": str(raw.get("kind", "resource") or "resource"),
            "label": str(
                raw.get("source_rel_path", raw.get("label", raw.get("name", node_id)))
                or node_id
            ),
            "text": str(
                raw.get("summary", raw.get("text_excerpt", raw.get("label", node_id)))
                or node_id
            ),
            "x": _clamp(float(raw.get("x", 0.5) or 0.5), 0.0, 1.0),
            "y": _clamp(float(raw.get("y", 0.5) or 0.5), 0.0, 1.0),
            "visibility": "public",
            "tags": [
                str(raw.get("dominant_presence", "")).strip(),
                str(raw.get("dominant_field", "")).strip(),
            ],
            "source_rel_path": str(raw.get("source_rel_path", "") or "").strip(),
            "url": str(raw.get("url", "") or "").strip(),
            "importance": float(raw.get("importance", 0.0) or 0.0),
            "embed_layer_count": float(raw.get("embed_layer_count", 0.0) or 0.0),
        }
        if row["source_rel_path"] and not row["url"]:
            clean_rel = row["source_rel_path"].lstrip("/")
            row["url"] = f"/library/{clean_rel}"
        if _looks_audio(row) or _looks_image(row):
            pool.append(row)
    pool.sort(
        key=lambda item: (
            -float(item.get("importance", 0.0)),
            -float(item.get("embed_layer_count", 0.0)),
            str(item.get("id", "")),
        )
    )
    return pool


def _select_task_surrounding_nodes(
    *,
    pool: list[dict[str, Any]],
    task: TaskSpec,
    max_nodes: int,
) -> list[dict[str, Any]]:
    muse_id = task.muse_id.strip().lower()
    prompt_tokens = set(_tokenize(task.prompt))
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in pool:
        row_id = str(row.get("id", "")).strip()
        if not row_id:
            continue
        tags = [
            str(item).strip().lower()
            for item in (
                row.get("tags", []) if isinstance(row.get("tags"), list) else []
            )
            if str(item).strip()
        ]
        label_tokens = set(
            _tokenize(
                " ".join(
                    [
                        str(row.get("label", "")),
                        str(row.get("text", "")),
                        str(row.get("source_rel_path", "")),
                        str(row.get("url", "")),
                    ]
                )
            )
        )
        overlap = len(prompt_tokens.intersection(label_tokens))
        score = float(row.get("importance", 0.0) or 0.0)
        if muse_id and muse_id in tags:
            score += 0.5
        score += min(1.2, overlap * 0.2)
        kind = str(row.get("kind", "")).strip().lower()
        if kind.startswith("audio"):
            score += 0.4
        if kind.startswith("image") or kind == "cover_art":
            score += 0.4
        scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("id", ""))))
    return [dict(item[1]) for item in scored[: max(1, max_nodes)]]


def _sample_from_response(
    *, runtime: str, task_id: str, response: dict[str, Any], latency_ms: float
) -> RunSample:
    actions = response.get("media_actions", response.get("audio_actions", []))
    rows = (
        [row for row in actions if isinstance(row, dict)]
        if isinstance(actions, list)
        else []
    )
    first = rows[0] if rows else {}
    status = str(first.get("status", "")).strip() if isinstance(first, dict) else ""
    requested = status == "requested"
    blocked = status == "blocked"
    return RunSample(
        runtime=runtime,
        task_id=task_id,
        ok=bool(response.get("ok", False)),
        latency_ms=latency_ms,
        requested=requested,
        blocked=blocked,
        action_status=status or "none",
        selected_label=str(first.get("selected_label", "")).strip()
        if isinstance(first, dict)
        else "",
        selected_node_id=str(first.get("selected_node_id", "")).strip()
        if isinstance(first, dict)
        else "",
        media_kind=str(first.get("media_kind", "")).strip()
        if isinstance(first, dict)
        else "",
        collisions=int(first.get("collision_count", 0) or 0)
        if isinstance(first, dict)
        else 0,
    )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] + ((values[upper] - values[lower]) * weight)


def _summarize(
    runtime: str, samples: list[RunSample], tasks: list[TaskSpec]
) -> dict[str, Any]:
    durations = sorted(sample.latency_ms for sample in samples)
    ok_count = sum(1 for sample in samples if sample.ok)
    requested_count = sum(1 for sample in samples if sample.requested)
    blocked_count = sum(1 for sample in samples if sample.blocked)
    unique_targets = sorted(
        {
            sample.selected_node_id
            for sample in samples
            if sample.requested and sample.selected_node_id
        }
    )
    requested_audio = sum(
        1 for sample in samples if sample.requested and sample.media_kind == "audio"
    )
    requested_image = sum(
        1 for sample in samples if sample.requested and sample.media_kind == "image"
    )
    collision_rows = [sample.collisions for sample in samples if sample.requested]
    mean_latency = statistics.fmean(durations) if durations else 0.0
    p95_latency = _percentile(durations, 0.95)
    collision_mean = statistics.fmean(collision_rows) if collision_rows else 0.0
    request_rate = requested_count / max(1, len(samples))
    blocked_rate = blocked_count / max(1, len(samples))

    task_rows: list[dict[str, Any]] = []
    for task in tasks:
        per_task = [sample for sample in samples if sample.task_id == task.task_id]
        if not per_task:
            continue
        task_requested = sum(1 for sample in per_task if sample.requested)
        task_rows.append(
            {
                "task_id": task.task_id,
                "samples": len(per_task),
                "requested": task_requested,
                "request_rate": task_requested / max(1, len(per_task)),
                "prompt": task.prompt,
            }
        )

    print(
        f"{runtime}: n={len(samples)} ok={ok_count} requested={requested_count} blocked={blocked_count} "
        f"mean={mean_latency:.2f}ms p95={p95_latency:.2f}ms collisions_mean={collision_mean:.2f} "
        f"audio={requested_audio} image={requested_image} unique_targets={len(unique_targets)}"
    )
    for task_row in task_rows:
        if int(task_row.get("samples", 0)) <= 0:
            continue
        print(
            "  - "
            f"{task_row['task_id']}: requested={int(task_row['requested'])}/{int(task_row['samples'])} "
            f"prompt='{str(task_row['prompt'])[:56]}'"
        )

    return {
        "runtime": runtime,
        "samples": len(samples),
        "ok_count": ok_count,
        "requested_count": requested_count,
        "blocked_count": blocked_count,
        "request_rate": request_rate,
        "blocked_rate": blocked_rate,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "collision_mean": collision_mean,
        "requested_audio": requested_audio,
        "requested_image": requested_image,
        "unique_target_count": len(unique_targets),
        "unique_target_ids": unique_targets,
        "tasks": task_rows,
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run parallel muse song-task comparison across multiple simulation runtimes"
    )
    parser.add_argument(
        "--runtime",
        action="append",
        default=[],
        help="Runtime endpoint in label=http://host:port format (repeatable)",
    )
    parser.add_argument(
        "--regimen",
        default="world_state/muse_song_training_regime.json",
        help="Path to JSON training regime/task battery",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="Override rounds from regimen (0 keeps regimen default)",
    )
    parser.add_argument(
        "--max-surrounding",
        type=int,
        default=28,
        help="Max surrounding media nodes sent per task",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional JSON report output path",
    )
    args = parser.parse_args()

    runtimes: list[tuple[str, str]] = []
    if args.runtime:
        for raw in args.runtime:
            runtimes.append(_parse_runtime_arg(str(raw)))
    else:
        runtimes = [
            ("song-baseline", "http://127.0.0.1:19877"),
            ("song-chaos", "http://127.0.0.1:19878"),
            ("song-stability", "http://127.0.0.1:19879"),
        ]

    regimen_path = Path(str(args.regimen)).resolve()
    tasks, regimen_rounds = _load_regimen(regimen_path)
    rounds = max(1, int(args.rounds)) if int(args.rounds) > 0 else regimen_rounds
    timeout_seconds = max(1.0, float(args.timeout))
    report_out = (
        Path(str(args.json_out)).resolve() if str(args.json_out).strip() else None
    )

    all_samples: list[RunSample] = []
    runtime_summaries: list[dict[str, Any]] = []
    for runtime_name, runtime_base in runtimes:
        try:
            catalog = _request_json(f"{runtime_base}/api/catalog", timeout_seconds)
        except Exception as exc:
            print(f"{runtime_name}: catalog probe failed ({exc.__class__.__name__})")
            continue
        media_pool = _build_media_surrounding_pool(catalog)
        graph_revision = str(catalog.get("generated_at", "") or "")
        print(
            f"{runtime_name}: media_pool={len(media_pool)} graph_revision={graph_revision or '(none)'}"
        )
        runtime_samples: list[RunSample] = []
        for round_index in range(rounds):
            for task in tasks:
                surrounding_nodes = _select_task_surrounding_nodes(
                    pool=media_pool,
                    task=task,
                    max_nodes=max(4, int(args.max_surrounding)),
                )
                idempotency_key = f"song-lab:{runtime_name}:{task.task_id}:{round_index}:{int(time.time() * 1000)}"
                started = time.perf_counter()
                response = _post_json(
                    f"{runtime_base}/api/muse/message",
                    {
                        "muse_id": task.muse_id,
                        "text": task.prompt,
                        "mode": task.mode,
                        "token_budget": 2048,
                        "idempotency_key": idempotency_key,
                        "graph_revision": graph_revision,
                        "surrounding_nodes": surrounding_nodes,
                        "seed": f"song-lab|{runtime_name}|{task.task_id}|{round_index}",
                    },
                    timeout_seconds,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                sample = _sample_from_response(
                    runtime=runtime_name,
                    task_id=task.task_id,
                    response=response,
                    latency_ms=elapsed_ms,
                )
                runtime_samples.append(sample)
                all_samples.append(sample)
        runtime_summaries.append(_summarize(runtime_name, runtime_samples, tasks))

    if not all_samples:
        print("no samples captured")
        return 1

    print("comparison:")
    by_runtime: dict[str, list[RunSample]] = {}
    for sample in all_samples:
        by_runtime.setdefault(sample.runtime, []).append(sample)
    ranking = sorted(
        by_runtime.items(),
        key=lambda item: (
            -(sum(1 for row in item[1] if row.requested) / max(1, len(item[1]))),
            statistics.fmean([row.latency_ms for row in item[1]]) if item[1] else 0.0,
        ),
    )
    for runtime_name, rows in ranking:
        request_rate = sum(1 for row in rows if row.requested) / max(1, len(rows))
        mean_latency = (
            statistics.fmean([row.latency_ms for row in rows]) if rows else 0.0
        )
        print(
            f"  - {runtime_name}: request_rate={request_rate * 100:.1f}% mean_latency={mean_latency:.2f}ms"
        )

    if report_out is not None:
        ranked_rows: list[dict[str, Any]] = []
        for runtime_name, rows in ranking:
            request_rate = sum(1 for row in rows if row.requested) / max(1, len(rows))
            mean_latency = (
                statistics.fmean([row.latency_ms for row in rows]) if rows else 0.0
            )
            ranked_rows.append(
                {
                    "runtime": runtime_name,
                    "request_rate": request_rate,
                    "mean_latency_ms": mean_latency,
                    "samples": len(rows),
                }
            )

        report = {
            "ok": True,
            "record": "eta-mu.song-lab-benchmark.v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "regimen": str(regimen_path),
            "rounds": rounds,
            "runtimes": [{"label": label, "url": url} for label, url in runtimes],
            "runtime_summaries": runtime_summaries,
            "ranking": ranked_rows,
            "samples": [
                {
                    "runtime": row.runtime,
                    "task_id": row.task_id,
                    "ok": row.ok,
                    "latency_ms": round(row.latency_ms, 3),
                    "requested": row.requested,
                    "blocked": row.blocked,
                    "action_status": row.action_status,
                    "selected_label": row.selected_label,
                    "selected_node_id": row.selected_node_id,
                    "media_kind": row.media_kind,
                    "collisions": row.collisions,
                }
                for row in all_samples
            ],
        }
        _save_json(report_out, report)
        print(f"json_report={report_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
