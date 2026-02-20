from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    part_root = Path(__file__).resolve().parents[2]
    script_path = part_root / "scripts" / "muse_semantic_training_lab.py"
    spec = importlib.util.spec_from_file_location(
        "muse_semantic_training_lab", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_circumstances_returns_tasks_and_rounds() -> None:
    module = _load_module()
    part_root = Path(__file__).resolve().parents[2]
    circumstances_path = (
        part_root / "world_state" / "muse_semantic_training_circumstances.json"
    )
    payload, tasks, rounds = module._load_circumstances(circumstances_path)
    assert isinstance(payload, dict)
    assert rounds >= 1
    assert len(tasks) >= 2
    assert all(str(task.expected_node_id).strip() for task in tasks)


def test_build_surrounding_nodes_includes_classifier_and_noise() -> None:
    module = _load_module()
    part_root = Path(__file__).resolve().parents[2]
    circumstances_path = (
        part_root / "world_state" / "muse_semantic_training_circumstances.json"
    )
    payload, tasks, _ = module._load_circumstances(circumstances_path)
    rng = module.random.Random(42)
    rows = module._build_surrounding_nodes(
        payload,
        round_index=1,
        task=tasks[0],
        pinned_node_ids={tasks[0].expected_node_id},
        rng=rng,
    )
    assert len(rows) > 50
    classifier_rows = [
        row for row in rows if str(row.get("presence_type", "")).strip() == "classifier"
    ]
    assert len(classifier_rows) == 1
    assert str(classifier_rows[0].get("default_media_kind", "")) in {"audio", "image"}


def test_score_trial_checks_kind_node_and_routed_daimoi() -> None:
    module = _load_module()
    task = module.TaskSpec(
        task_id="test-task",
        prompt="Play Music",
        mode="deterministic",
        expected_media_kind="audio",
        expected_node_id="target:audio:fork-canticle",
    )
    payload = {
        "daimoi": [
            {
                "intent": "audio.play",
                "target_node_id": "target:audio:fork-canticle",
            }
        ]
    }
    action = {
        "media_kind": "audio",
        "selected_node_id": "target:audio:fork-canticle",
        "collision_count": 2,
    }
    correct_kind, correct_node, routed = module._score_trial(
        task=task,
        payload=payload,
        action=action,
    )
    assert correct_kind is True
    assert correct_node is True
    assert routed is True
