#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


SERVICE_BY_ALIAS = {
    "baseline": "eta-mu-song-baseline",
    "chaos": "eta-mu-song-chaos",
    "stability": "eta-mu-song-stability",
}


def _run(command: list[str], *, cwd: Path) -> int:
    print("$", " ".join(shlex.quote(part) for part in command))
    completed = subprocess.run(command, cwd=str(cwd))
    return int(completed.returncode)


def _parse_aliases(raw: str) -> list[str]:
    aliases = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    deduped: list[str] = []
    for alias in aliases:
        if alias not in SERVICE_BY_ALIAS:
            raise ValueError(f"unsupported runtime alias: {alias}")
        if alias not in deduped:
            deduped.append(alias)
    return deduped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage throttled parallel muse-song simulation runtimes"
    )
    parser.add_argument(
        "command",
        choices=["start", "stop", "status", "bench", "train", "eval"],
        help="Operation to run",
    )
    parser.add_argument(
        "--runtimes",
        default="baseline,chaos,stability",
        help="Comma-separated runtime aliases (baseline,chaos,stability)",
    )
    parser.add_argument(
        "--compose-file",
        default="docker-compose.muse-song-lab.yml",
        help="Compose file path relative to part64 root",
    )
    parser.add_argument(
        "--regimen",
        default="world_state/muse_song_training_regime.json",
        help="Training regime JSON path (for bench)",
    )
    parser.add_argument(
        "--circumstances",
        default="world_state/muse_semantic_training_circumstances.json",
        help="Semantic training circumstances JSON path (for train)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="Override regimen rounds for bench (0 uses file default)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Benchmark HTTP timeout seconds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4207,
        help="Deterministic seed base for train command",
    )
    parser.add_argument(
        "--output",
        default="../.opencode/runtime/sim_learning_eval.latest.json",
        help="Evaluation report output path (for eval)",
    )
    parser.add_argument(
        "--keep-other-stacks",
        action="store_true",
        help="Do not auto-stop sim-slice bench stack before starting song lab",
    )
    args = parser.parse_args()

    part_root = Path(__file__).resolve().parents[1]
    compose_file = part_root / str(args.compose_file)
    if not compose_file.exists():
        raise SystemExit(f"compose file not found: {compose_file}")

    aliases = _parse_aliases(args.runtimes)
    services = [SERVICE_BY_ALIAS[alias] for alias in aliases]

    if args.command == "start":
        if not args.keep_other_stacks:
            _run(
                [
                    "docker",
                    "compose",
                    "-f",
                    "docker-compose.sim-slice-bench.yml",
                    "down",
                    "--remove-orphans",
                ],
                cwd=part_root,
            )
        command = [
            "docker",
            "compose",
            "-f",
            str(compose_file.name),
            "up",
            "-d",
            "--build",
            "chroma",
            *services,
        ]
        return _run(command, cwd=part_root)

    if args.command == "stop":
        return _run(
            [
                "docker",
                "compose",
                "-f",
                str(compose_file.name),
                "down",
                "--remove-orphans",
            ],
            cwd=part_root,
        )

    if args.command == "status":
        ps_code = _run(
            ["docker", "compose", "-f", str(compose_file.name), "ps"],
            cwd=part_root,
        )
        if ps_code != 0:
            return ps_code
        container_names = [f"part64-{SERVICE_BY_ALIAS[alias]}-1" for alias in aliases]
        return _run(
            ["docker", "stats", "--no-stream", *container_names],
            cwd=part_root,
        )

    if args.command == "bench":
        runtime_args: list[str] = []
        for alias in aliases:
            offset = {"baseline": 0, "chaos": 1, "stability": 2}[alias]
            runtime_args.extend(
                ["--runtime", f"song-{alias}=http://127.0.0.1:{19877 + offset}"]
            )
        command = [
            "python",
            "scripts/bench_muse_song_lab.py",
            "--regimen",
            str(args.regimen),
            "--timeout",
            str(float(args.timeout)),
            *runtime_args,
        ]
        if int(args.rounds) > 0:
            command.extend(["--rounds", str(int(args.rounds))])
        return _run(command, cwd=part_root)

    if args.command == "eval":
        runtime_args: list[str] = []
        for alias in aliases:
            offset = {"baseline": 0, "chaos": 1, "stability": 2}[alias]
            runtime_args.extend(
                ["--runtime", f"song-{alias}=http://127.0.0.1:{19877 + offset}"]
            )
        command = [
            "python",
            "scripts/eval_sim_learning_suite.py",
            "--circumstances",
            str(args.circumstances),
            "--regimen",
            str(args.regimen),
            "--timeout",
            str(float(args.timeout)),
            "--seed",
            str(int(args.seed)),
            "--output",
            str(args.output),
            *runtime_args,
        ]
        if int(args.rounds) > 0:
            command.extend(
                [
                    "--training-rounds",
                    str(int(args.rounds)),
                    "--song-rounds",
                    str(int(args.rounds)),
                ]
            )
        return _run(command, cwd=part_root)

    exit_code = 0
    for index, alias in enumerate(aliases):
        offset = {"baseline": 0, "chaos": 1, "stability": 2}[alias]
        runtime_url = f"http://127.0.0.1:{19877 + offset}"
        command = [
            "python",
            "scripts/muse_semantic_training_lab.py",
            "--runtime",
            runtime_url,
            "--circumstances",
            str(args.circumstances),
            "--timeout",
            str(float(args.timeout)),
            "--seed",
            str(int(args.seed) + (index * 97)),
        ]
        if int(args.rounds) > 0:
            command.extend(["--rounds", str(int(args.rounds))])
        rc = _run(command, cwd=part_root)
        if rc != 0:
            exit_code = rc
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
