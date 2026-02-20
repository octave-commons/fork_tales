#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class TaskSpec:
    task_id: str
    prompt: str
    mode: str
    expected_media_kind: str
    expected_node_id: str


@dataclass
class TrialSample:
    round_index: int
    task_id: str
    latency_ms: float
    ok: bool
    requested: bool
    blocked: bool
    correct_kind: bool
    correct_node: bool
    routed_correctly: bool
    selected_media_kind: str
    selected_node_id: str
    status: str


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _hash_unit(seed: str) -> float:
    raw = hash(seed) & 0xFFFFFFFF
    return float(raw) / 4294967295.0


def _request_json(url: str, timeout_seconds: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout_seconds) as response:
        payload = response.read().decode("utf-8")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise RuntimeError(f"expected object payload from {url}")
    return decoded


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
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise RuntimeError(f"expected object payload from {url}")
        return decoded
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            decoded = json.loads(text)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
        return {"ok": False, "error": f"http_{exc.code}", "detail": text[:260]}
    except TimeoutError:
        return {"ok": False, "error": "timeout"}
    except URLError as exc:
        return {"ok": False, "error": f"url_error:{exc.reason}"}
    except Exception as exc:
        return {"ok": False, "error": f"request_failed:{exc.__class__.__name__}"}


def _load_circumstances(path: Path) -> tuple[dict[str, Any], list[TaskSpec], int]:
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("circumstances file must be a JSON object")

    rounds = max(1, int(payload.get("rounds", 4) or 4))
    tasks_raw = payload.get("tasks", [])
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise RuntimeError("circumstances.tasks must be a non-empty list")

    tasks: list[TaskSpec] = []
    for index, row in enumerate(tasks_raw):
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("id", f"task_{index + 1:02d}")).strip()
        prompt = str(row.get("prompt", "")).strip()
        mode = str(row.get("mode", "deterministic")).strip().lower()
        expected_media_kind = (
            str(row.get("expected_media_kind", "audio")).strip().lower() or "audio"
        )
        expected_node_id = str(row.get("expected_node_id", "")).strip()
        if mode not in {"deterministic", "stochastic"}:
            mode = "deterministic"
        if not task_id or not prompt or not expected_node_id:
            continue
        tasks.append(
            TaskSpec(
                task_id=task_id,
                prompt=prompt,
                mode=mode,
                expected_media_kind=expected_media_kind,
                expected_node_id=expected_node_id,
            )
        )

    if not tasks:
        raise RuntimeError("no valid tasks found in circumstances")
    return payload, tasks, rounds


def _xy(seed: str) -> tuple[float, float]:
    return (
        round(_clamp(0.08 + (_hash_unit(seed + "|x") * 0.84), 0.0, 1.0), 6),
        round(_clamp(0.08 + (_hash_unit(seed + "|y") * 0.84), 0.0, 1.0), 6),
    )


def _copy_node_with_position(row: dict[str, Any], *, seed: str) -> dict[str, Any]:
    node = dict(row)
    x, y = _xy(seed)
    node["x"] = _clamp(_safe_float(node.get("x"), x), 0.0, 1.0)
    node["y"] = _clamp(_safe_float(node.get("y"), y), 0.0, 1.0)
    node["visibility"] = str(node.get("visibility", "public") or "public")
    tags = node.get("tags", [])
    node["tags"] = [
        str(item).strip()
        for item in (tags if isinstance(tags, list) else [])
        if str(item).strip()
    ]
    return node


def _build_classifier_and_seed_nodes(
    circumstances: dict[str, Any],
    *,
    round_index: int,
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []

    classifier = circumstances.get("classifier_presence", {})
    if isinstance(classifier, dict):
        classifier_id = str(
            classifier.get("id", "presence.modality.baseline")
            or "presence.modality.baseline"
        ).strip()
        classifier_kind = (
            str(classifier.get("default_media_kind", "audio") or "audio")
            .strip()
            .lower()
            or "audio"
        )
        x, y = _xy(f"classifier|{classifier_id}|{round_index}")
        nodes.append(
            {
                "id": classifier_id,
                "kind": "presence",
                "presence_type": "classifier",
                "label": str(
                    classifier.get("label", "Baseline Modality Classifier")
                    or "Baseline Modality Classifier"
                ),
                "text": "base/default presence modality classifier for routing",
                "default_media_kind": classifier_kind,
                "seed_terms": [
                    str(item).strip()
                    for item in (
                        classifier.get("seed_terms", [])
                        if isinstance(classifier.get("seed_terms"), list)
                        else []
                    )
                    if str(item).strip()
                ],
                "x": x,
                "y": y,
                "visibility": "public",
                "tags": ["classifier", classifier_kind],
            }
        )

    seeds = (
        circumstances.get("concept_seeds", [])
        if isinstance(circumstances.get("concept_seeds"), list)
        else []
    )
    for index, raw in enumerate(seeds):
        if not isinstance(raw, dict):
            continue
        row = _copy_node_with_position(raw, seed=f"seed|{index}|{round_index}")
        row["kind"] = str(row.get("kind", "daimon") or "daimon")
        row["presence_type"] = str(
            row.get("presence_type", "concept_seed") or "concept_seed"
        )
        row["media_kind"] = str(row.get("media_kind", "") or "").strip().lower()
        row["focus_node_ids"] = [
            str(item).strip()
            for item in (
                row.get("focus_node_ids", [])
                if isinstance(row.get("focus_node_ids"), list)
                else []
            )
            if str(item).strip()
        ]
        row["seed_terms"] = [
            str(item).strip()
            for item in (
                row.get("seed_terms", [])
                if isinstance(row.get("seed_terms"), list)
                else []
            )
            if str(item).strip()
        ]
        tags = [str(item).strip() for item in row.get("tags", []) if str(item).strip()]
        if "concept-seed" not in tags:
            tags.append("concept-seed")
        row["tags"] = tags
        nodes.append(row)

    return nodes


def _build_noise_nodes(
    circumstances: dict[str, Any], *, rng: random.Random
) -> list[dict[str, Any]]:
    noise_cfg = (
        circumstances.get("noise", {})
        if isinstance(circumstances.get("noise"), dict)
        else {}
    )
    audio_count = max(0, min(96, int(noise_cfg.get("audio_distractors", 18) or 18)))
    image_count = max(0, min(96, int(noise_cfg.get("image_distractors", 18) or 18)))
    nexus_count = max(0, min(144, int(noise_cfg.get("nexus_presence_noise", 24) or 24)))

    presences = [
        "receipt_river",
        "anchor_registry",
        "gates_of_truth",
        "keeper_of_receipts",
        "witness_thread",
        "health_sentinel_cpu",
        "health_sentinel_ram",
        "health_sentinel_disk",
        "mage_of_receipts",
        "chaos",
        "stability",
        "symmetry",
    ]
    nexus_ids = [
        "nexus.ui.chat.witness_thread",
        "nexus.ui.chat.chaos",
        "nexus.ui.chat.stability",
        "nexus.ui.chat.symmetry",
        "nexus.ui.projection_ledger",
        "nexus.ui.world_log",
        "nexus.ui.omni_archive",
    ]

    nodes: list[dict[str, Any]] = []
    for index in range(audio_count):
        owner = rng.choice(presences)
        x, y = rng.random(), rng.random()
        nodes.append(
            {
                "id": f"noise:audio:{owner}:{index:03d}",
                "kind": "audio" if rng.random() < 0.8 else "resource",
                "label": f"{owner} loop {index:03d}",
                "text": f"non-target song candidate from {owner} via {rng.choice(nexus_ids)}",
                "x": round(x, 6),
                "y": round(y, 6),
                "source_rel_path": f"artifacts/audio/noise_{owner}_{index:03d}.mp3",
                "url": f"/library/artifacts/audio/noise_{owner}_{index:03d}.mp3",
                "tags": [owner, "noise", "audio"],
            }
        )

    for index in range(image_count):
        owner = rng.choice(presences)
        x, y = rng.random(), rng.random()
        nodes.append(
            {
                "id": f"noise:image:{owner}:{index:03d}",
                "kind": "image" if rng.random() < 0.8 else "resource",
                "label": f"{owner} frame {index:03d}",
                "text": f"non-target image candidate from {owner} via {rng.choice(nexus_ids)}",
                "x": round(x, 6),
                "y": round(y, 6),
                "source_rel_path": f"artifacts/images/noise_{owner}_{index:03d}.png",
                "url": f"/library/artifacts/images/noise_{owner}_{index:03d}.png",
                "tags": [owner, "noise", "image"],
            }
        )

    for index in range(nexus_count):
        owner = rng.choice(presences)
        nexus = rng.choice(nexus_ids)
        x, y = rng.random(), rng.random()
        nodes.append(
            {
                "id": f"noise:nexus:{index:03d}",
                "kind": "nexus" if rng.random() < 0.7 else "presence",
                "label": f"{nexus}:{owner}:{index:03d}",
                "text": f"semantic landscape drift around {owner} and {nexus}",
                "x": round(x, 6),
                "y": round(y, 6),
                "tags": ["noise", owner, nexus],
            }
        )

    rng.shuffle(nodes)
    return nodes


def _build_surrounding_nodes(
    circumstances: dict[str, Any],
    *,
    round_index: int,
    task: TaskSpec,
    pinned_node_ids: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    corpus_rows = (
        circumstances.get("corpus_nodes", [])
        if isinstance(circumstances.get("corpus_nodes"), list)
        else []
    )
    for index, raw in enumerate(corpus_rows):
        if not isinstance(raw, dict):
            continue
        row = _copy_node_with_position(
            raw, seed=f"corpus|{task.task_id}|{round_index}|{index}"
        )
        tags = [str(item).strip() for item in row.get("tags", []) if str(item).strip()]
        if (
            str(row.get("id", "")).strip() in pinned_node_ids
            and "workspace-pin" not in tags
        ):
            tags.append("workspace-pin")
        row["tags"] = tags
        nodes.append(row)

    nodes.extend(
        _build_classifier_and_seed_nodes(circumstances, round_index=round_index)
    )
    nodes.extend(_build_noise_nodes(circumstances, rng=rng))
    return nodes


def _ensure_muse(
    runtime_base: str,
    *,
    muse_id: str,
    label: str,
    timeout_seconds: float,
) -> None:
    payload = _post_json(
        f"{runtime_base}/api/muse/create",
        {
            "muse_id": muse_id,
            "label": label,
            "anchor": {"x": 0.62, "y": 0.34, "zoom": 1.0, "kind": "semantic-training"},
            "user_intent_id": "semantic-training-bootstrap",
        },
        timeout_seconds,
    )
    if bool(payload.get("ok", False)):
        return
    if str(payload.get("error", "")) == "muse_already_exists":
        return
    raise RuntimeError(f"failed to ensure muse '{muse_id}': {payload}")


def _sync_pins(
    runtime_base: str,
    *,
    muse_id: str,
    pinned_node_ids: set[str],
    timeout_seconds: float,
) -> None:
    _post_json(
        f"{runtime_base}/api/muse/sync-pins",
        {
            "muse_id": muse_id,
            "pinned_node_ids": sorted(pinned_node_ids),
            "reason": "semantic-training",
            "user_intent_id": "semantic-training-pins",
        },
        timeout_seconds,
    )


def _extract_action(payload: dict[str, Any]) -> dict[str, Any]:
    actions = payload.get("media_actions", payload.get("audio_actions", []))
    rows = (
        [row for row in actions if isinstance(row, dict)]
        if isinstance(actions, list)
        else []
    )
    return dict(rows[0]) if rows else {}


def _score_trial(
    *,
    task: TaskSpec,
    payload: dict[str, Any],
    action: dict[str, Any],
) -> tuple[bool, bool, bool]:
    selected_kind = str(action.get("media_kind", "")).strip().lower()
    selected_node_id = str(action.get("selected_node_id", "")).strip()

    correct_kind = selected_kind == task.expected_media_kind
    correct_node = selected_node_id == task.expected_node_id

    expected_intent = (
        "audio.play" if task.expected_media_kind == "audio" else "image.open"
    )
    routed_correctly = False
    for row in (
        payload.get("daimoi", []) if isinstance(payload.get("daimoi"), list) else []
    ):
        if not isinstance(row, dict):
            continue
        if (
            str(row.get("intent", "")).strip() == expected_intent
            and str(row.get("target_node_id", "")).strip() == task.expected_node_id
        ):
            routed_correctly = True
            break
    if int(action.get("collision_count", 0) or 0) <= 0:
        routed_correctly = False
    return correct_kind, correct_node, routed_correctly


def _mean_rate(rows: list[bool]) -> float:
    if not rows:
        return 0.0
    return statistics.fmean(1.0 if item else 0.0 for item in rows)


def _chart_bar(value: float, *, width: int = 24) -> str:
    value = _clamp(value, 0.0, 1.0)
    filled = int(round(value * width))
    return ("#" * filled) + ("." * max(0, width - filled))


def _save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")


def _post_meta_run(
    runtime_base: str,
    *,
    owner: str,
    scenario_path: Path,
    report_path: Path,
    report: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    summary = (
        report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    )
    payload = {
        "run_type": "training",
        "status": "completed",
        "title": "Muse semantic routing training cycle",
        "owner": str(owner or "Err").strip() or "Err",
        "objective": "Improve modality and target routing under dense non-target semantic noise.",
        "model_ref": "muse-runtime.media-classifier.v1",
        "dataset_ref": f"file://{scenario_path}",
        "notes": f"report={report_path}",
        "tags": [
            "muse-routing",
            "semantic-landscape",
            "classifier",
            "multiverse-gateway",
        ],
        "targets": [str(report.get("muse_id", "")), "docker-gateway"],
        "links": [str(report_path)],
        "metrics": {
            "samples": int(summary.get("samples", 0) or 0),
            "rounds": int(summary.get("rounds", 0) or 0),
            "requested_rate": round(
                _safe_float(summary.get("requested_rate", 0.0), 0.0), 6
            ),
            "modality_accuracy": round(
                _safe_float(summary.get("modality_accuracy", 0.0), 0.0), 6
            ),
            "target_accuracy": round(
                _safe_float(summary.get("target_accuracy", 0.0), 0.0), 6
            ),
            "routed_accuracy": round(
                _safe_float(summary.get("routed_accuracy", 0.0), 0.0), 6
            ),
            "blocked_rate": round(
                _safe_float(summary.get("blocked_rate", 0.0), 0.0), 6
            ),
        },
    }
    return _post_json(f"{runtime_base}/api/meta/runs", payload, timeout_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Iterative semantic training circumstances for muse routing with concept-seed noise and classifier baseline"
    )
    parser.add_argument(
        "--runtime",
        default="http://127.0.0.1:8787",
        help="Runtime base URL (gateway-compatible)",
    )
    parser.add_argument(
        "--circumstances",
        default="world_state/muse_semantic_training_circumstances.json",
        help="Path to semantic training circumstances JSON",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="Override rounds (0 uses circumstances default)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=35.0,
        help="HTTP timeout seconds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4207,
        help="Deterministic random seed for distractor generation",
    )
    parser.add_argument(
        "--owner",
        default="Err",
        help="Owner label for meta run logging",
    )
    parser.add_argument(
        "--output",
        default="../.opencode/runtime/muse_semantic_training.latest.json",
        help="Output report path",
    )
    parser.add_argument(
        "--no-meta-run",
        action="store_true",
        help="Skip posting aggregate run metrics to /api/meta/runs",
    )
    args = parser.parse_args()

    runtime_base = str(args.runtime or "http://127.0.0.1:8787").strip().rstrip("/")
    if not runtime_base.startswith(("http://", "https://")):
        raise SystemExit("--runtime must start with http:// or https://")

    timeout_seconds = max(1.0, float(args.timeout))
    circumstances_path = Path(str(args.circumstances)).resolve()
    circumstances, tasks, default_rounds = _load_circumstances(circumstances_path)
    rounds = max(1, int(args.rounds)) if int(args.rounds) > 0 else default_rounds

    muse_block = (
        circumstances.get("muse", {})
        if isinstance(circumstances.get("muse"), dict)
        else {}
    )
    muse_id = str(
        muse_block.get("id", "lumen_classifier_muse") or "lumen_classifier_muse"
    ).strip()
    muse_label = str(
        muse_block.get("label", "Lumen Classifier Muse") or "Lumen Classifier Muse"
    ).strip()

    _request_json(f"{runtime_base}/api/catalog", timeout_seconds)
    _ensure_muse(
        runtime_base, muse_id=muse_id, label=muse_label, timeout_seconds=timeout_seconds
    )

    pinned_node_ids: set[str] = set()
    _sync_pins(
        runtime_base,
        muse_id=muse_id,
        pinned_node_ids=pinned_node_ids,
        timeout_seconds=timeout_seconds,
    )

    samples: list[TrialSample] = []
    round_rows: list[dict[str, Any]] = []

    for round_index in range(1, rounds + 1):
        rng = random.Random(int(args.seed) + (round_index * 1000))
        round_samples: list[TrialSample] = []
        for task in tasks:
            surrounding_nodes = _build_surrounding_nodes(
                circumstances,
                round_index=round_index,
                task=task,
                pinned_node_ids=pinned_node_ids,
                rng=rng,
            )

            started = time.perf_counter()
            response = _post_json(
                f"{runtime_base}/api/muse/message",
                {
                    "muse_id": muse_id,
                    "text": task.prompt,
                    "mode": task.mode,
                    "token_budget": 2048,
                    "idempotency_key": f"semantic-training:{muse_id}:{task.task_id}:{round_index}:{int(time.time() * 1000)}",
                    "graph_revision": f"semantic-training:{round_index}",
                    "surrounding_nodes": surrounding_nodes,
                    "seed": f"semantic-training|{round_index}|{task.task_id}",
                },
                timeout_seconds,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            action = _extract_action(response)
            status = str(action.get("status", "none") or "none").strip().lower()

            correct_kind, correct_node, routed_correctly = _score_trial(
                task=task,
                payload=response,
                action=action,
            )
            if correct_node:
                pinned_node_ids.add(task.expected_node_id)

            sample = TrialSample(
                round_index=round_index,
                task_id=task.task_id,
                latency_ms=elapsed_ms,
                ok=bool(response.get("ok", False)),
                requested=status == "requested",
                blocked=status == "blocked",
                correct_kind=correct_kind,
                correct_node=correct_node,
                routed_correctly=routed_correctly,
                selected_media_kind=str(action.get("media_kind", "") or "")
                .strip()
                .lower(),
                selected_node_id=str(action.get("selected_node_id", "") or "").strip(),
                status=status,
            )
            samples.append(sample)
            round_samples.append(sample)

        _sync_pins(
            runtime_base,
            muse_id=muse_id,
            pinned_node_ids=pinned_node_ids,
            timeout_seconds=timeout_seconds,
        )

        requested_rate = _mean_rate([row.requested for row in round_samples])
        modality_accuracy = _mean_rate([row.correct_kind for row in round_samples])
        target_accuracy = _mean_rate([row.correct_node for row in round_samples])
        routed_accuracy = _mean_rate([row.routed_correctly for row in round_samples])
        blocked_rate = _mean_rate([row.blocked for row in round_samples])
        round_rows.append(
            {
                "round": round_index,
                "samples": len(round_samples),
                "requested_rate": round(requested_rate, 6),
                "modality_accuracy": round(modality_accuracy, 6),
                "target_accuracy": round(target_accuracy, 6),
                "routed_accuracy": round(routed_accuracy, 6),
                "blocked_rate": round(blocked_rate, 6),
                "pinned_node_count": len(pinned_node_ids),
            }
        )

    requested_rate = _mean_rate([row.requested for row in samples])
    modality_accuracy = _mean_rate([row.correct_kind for row in samples])
    target_accuracy = _mean_rate([row.correct_node for row in samples])
    routed_accuracy = _mean_rate([row.routed_correctly for row in samples])
    blocked_rate = _mean_rate([row.blocked for row in samples])
    latency_ms = [row.latency_ms for row in samples]

    report = {
        "ok": True,
        "record": "eta-mu.muse-semantic-training-report.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": runtime_base,
        "circumstances_path": str(circumstances_path),
        "muse_id": muse_id,
        "muse_label": muse_label,
        "summary": {
            "samples": len(samples),
            "rounds": rounds,
            "requested_rate": round(requested_rate, 6),
            "modality_accuracy": round(modality_accuracy, 6),
            "target_accuracy": round(target_accuracy, 6),
            "routed_accuracy": round(routed_accuracy, 6),
            "blocked_rate": round(blocked_rate, 6),
            "latency_mean_ms": round(
                statistics.fmean(latency_ms) if latency_ms else 0.0, 3
            ),
            "latency_p95_ms": round(
                sorted(latency_ms)[int(max(0, (len(latency_ms) - 1) * 0.95))], 3
            )
            if latency_ms
            else 0.0,
            "pinned_node_count": len(pinned_node_ids),
        },
        "rounds": round_rows,
        "tasks": [
            {
                "id": task.task_id,
                "prompt": task.prompt,
                "expected_media_kind": task.expected_media_kind,
                "expected_node_id": task.expected_node_id,
            }
            for task in tasks
        ],
        "samples": [
            {
                "round": sample.round_index,
                "task_id": sample.task_id,
                "latency_ms": round(sample.latency_ms, 3),
                "ok": sample.ok,
                "requested": sample.requested,
                "blocked": sample.blocked,
                "correct_kind": sample.correct_kind,
                "correct_node": sample.correct_node,
                "routed_correctly": sample.routed_correctly,
                "selected_media_kind": sample.selected_media_kind,
                "selected_node_id": sample.selected_node_id,
                "status": sample.status,
            }
            for sample in samples
        ],
    }

    output_path = Path(str(args.output)).resolve()
    _save_report(output_path, report)

    print(f"runtime={runtime_base}")
    print(f"muse={muse_id}")
    print(f"report={output_path}")
    print(
        "summary: "
        f"requested={requested_rate * 100:.1f}% "
        f"modality={modality_accuracy * 100:.1f}% "
        f"target={target_accuracy * 100:.1f}% "
        f"routed={routed_accuracy * 100:.1f}% "
        f"blocked={blocked_rate * 100:.1f}%"
    )
    for row in round_rows:
        print(
            f"  round {int(row['round']):02d} "
            f"target={_chart_bar(float(row['target_accuracy']))} {float(row['target_accuracy']) * 100:5.1f}% "
            f"routed={_chart_bar(float(row['routed_accuracy']))} {float(row['routed_accuracy']) * 100:5.1f}% "
            f"pins={int(row['pinned_node_count'])}"
        )

    if not args.no_meta_run:
        meta_result = _post_meta_run(
            runtime_base,
            owner=str(args.owner),
            scenario_path=circumstances_path,
            report_path=output_path,
            report=report,
            timeout_seconds=timeout_seconds,
        )
        if bool(meta_result.get("ok", False)):
            print("meta_run=posted")
        else:
            print(f"meta_run=failed error={meta_result.get('error', 'unknown')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
