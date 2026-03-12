from __future__ import annotations

import argparse
import json
from pathlib import Path

from .lith_nexus_snapshot import build_lith_nexus_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Lith Nexus snapshot helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser("snapshot", help="emit lith nexus snapshot")
    snapshot_parser.add_argument(
        "--repo-root", default=".", help="repository root to index"
    )
    snapshot_parser.add_argument(
        "--include-text",
        action="store_true",
        help="include canonical/text payloads in index nodes",
    )

    args = parser.parse_args()

    if args.command == "snapshot":
        payload = build_lith_nexus_snapshot(
            Path(args.repo_root), include_text=bool(args.include_text)
        )
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
