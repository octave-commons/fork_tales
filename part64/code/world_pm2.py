from __future__ import annotations

import argparse
import shutil
import subprocess
import webbrowser
from pathlib import Path


def run_pm2(args: list[str], cwd: Path) -> int:
    cmd = ["pm2", *args]
    result = subprocess.run(cmd, cwd=cwd, check=False)
    return result.returncode


def require_pm2() -> None:
    if shutil.which("pm2") is None:
        raise SystemExit("pm2 is not installed or not on PATH")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control eta-mu world PM2 daemon")
    parser.add_argument(
        "command", choices=["start", "stop", "restart", "status", "open"]
    )
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--name", default="eta-mu-world")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "ecosystem.config.cjs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_pm2()

    part_root = Path(__file__).resolve().parents[1]
    config = args.config.resolve()

    if args.command == "start":
        code = run_pm2(["start", str(config)], cwd=part_root)
        if code == 0:
            webbrowser.open(f"http://{args.host}:{args.port}/")
        return code

    if args.command == "stop":
        return run_pm2(["stop", args.name], cwd=part_root)

    if args.command == "restart":
        return run_pm2(["restart", args.name], cwd=part_root)

    if args.command == "status":
        return run_pm2(["status", args.name], cwd=part_root)

    webbrowser.open(f"http://{args.host}:{args.port}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
