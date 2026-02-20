from __future__ import annotations

import tempfile
from pathlib import Path

from code.world_web import meta_ops


def test_create_and_list_meta_notes() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault_root = Path(td)

        created = meta_ops.create_meta_note(
            vault_root,
            text="Chaos container hit repeated timeout during catalog probe.",
            owner="Err",
            category="failure",
            severity="warning",
            tags=["chaos", "healthcheck"],
            targets=["eta-mu-song-chaos"],
        )
        assert created["ok"] is True
        assert created["note"]["category"] == "failure"

        _ = meta_ops.create_meta_note(
            vault_root,
            text="Queue evaluation objective after stabilizing memory envelope.",
            owner="Err",
            category="action",
            severity="info",
            tags=["evaluation"],
            targets=["eta-mu-song-chaos"],
        )

        listed = meta_ops.list_meta_notes(
            vault_root,
            limit=8,
            tag="healthcheck",
            target="eta-mu-song-chaos",
        )
        assert listed["ok"] is True
        assert listed["count"] == 1
        assert listed["notes"][0]["severity"] == "warning"


def test_create_and_list_meta_runs() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault_root = Path(td)

        invalid = meta_ops.create_meta_run(
            vault_root,
            run_type="invalid-kind",
            title="bad run",
        )
        assert invalid["ok"] is False

        created = meta_ops.create_meta_run(
            vault_root,
            run_type="training",
            status="running",
            title="Chaos fine-tune cycle 04",
            model_ref="qwen3-vl:4b-instruct",
            dataset_ref="dataset://song-lab/chaos-2026-02-19",
            tags=["chaos", "training"],
            targets=["eta-mu-song-chaos"],
            metrics={"loss": 0.129, "steps": 4200},
            owner="Err",
        )
        assert created["ok"] is True
        assert created["run"]["run_type"] == "training"
        assert created["run"]["status"] == "running"

        listed = meta_ops.list_meta_runs(
            vault_root,
            limit=8,
            run_type="training",
            status="running",
            target="eta-mu-song-chaos",
        )
        assert listed["ok"] is True
        assert listed["count"] == 1
        assert listed["runs"][0]["model_ref"] == "qwen3-vl:4b-instruct"


def test_build_meta_overview_promotes_failure_signals() -> None:
    with tempfile.TemporaryDirectory() as td:
        vault_root = Path(td)
        _ = meta_ops.create_meta_note(
            vault_root,
            text="OOM spike observed in chaos run.",
            category="failure",
            severity="critical",
            targets=["eta-mu-song-chaos"],
        )
        _ = meta_ops.create_meta_run(
            vault_root,
            run_type="evaluation",
            status="planned",
            title="Chaos regression eval",
            model_ref="qwen3-vl:4b-instruct",
            dataset_ref="dataset://song-lab/eval-2026-02-19",
        )

        overview = meta_ops.build_meta_overview(
            vault_root,
            docker_snapshot={
                "summary": {
                    "running_simulations": 2,
                    "strict_simulations": 2,
                },
                "simulations": [
                    {
                        "id": "a" * 64,
                        "name": "eta-mu-song-chaos",
                        "service": "eta-mu-song-chaos",
                        "project": "song-lab",
                        "state": "running",
                        "status": "Up 3 minutes",
                        "lifecycle": {
                            "stability": "failing",
                            "health_status": "unhealthy",
                            "restart_count": 4,
                            "oom_killed": True,
                            "signals": [
                                "oom_killed",
                                "health_unhealthy",
                                "restarted",
                            ],
                        },
                        "resources": {
                            "pressure": {
                                "state": "critical",
                            }
                        },
                    },
                    {
                        "id": "b" * 64,
                        "name": "eta-mu-song-stability",
                        "service": "eta-mu-song-stability",
                        "project": "song-lab",
                        "state": "running",
                        "status": "Up 3 minutes",
                        "lifecycle": {
                            "stability": "healthy",
                            "health_status": "healthy",
                            "restart_count": 0,
                            "oom_killed": False,
                            "signals": [],
                        },
                        "resources": {
                            "pressure": {
                                "state": "ok",
                            }
                        },
                    },
                ],
            },
            queue_snapshot={
                "pending_count": 1,
                "event_count": 8,
            },
            notes_limit=6,
            runs_limit=6,
        )

        assert overview["ok"] is True
        assert overview["failures"]["failing_count"] == 1
        assert overview["failures"]["failing"][0]["name"] == "eta-mu-song-chaos"
        assert overview["runs"]["active_count"] == 1
        assert overview["notes"]["count"] == 1
        assert len(overview["suggestions"]) >= 1
