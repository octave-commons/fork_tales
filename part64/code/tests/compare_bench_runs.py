from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def main() -> None:
    runs_root = Path("runs")
    rows: list[dict[str, Any]] = []

    for run_json in sorted(runs_root.glob("npu_spec_*/run.json")):
        try:
            payload = json.loads(run_json.read_text(encoding="utf-8"))
        except Exception:
            continue

        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        rows.append(
            {
                "tag": str(payload.get("tag", run_json.parent.name)),
                "device_requested": str(payload.get("device_requested", "")),
                "device_selected": str(payload.get("device_selected", "")),
                "timing": str(payload.get("timing", "")),
                "dim": int(payload.get("dim", 0) or 0),
                "compile_ms": _safe_float(summary.get("compile_ms", 0.0)),
                "first_infer_us": _safe_float(summary.get("first_infer_us", 0.0)),
                "p99_us": _safe_float(summary.get("p99_us", 0.0)),
                "mean_us": _safe_float(summary.get("mean_us", 0.0)),
                "max_us": _safe_float(summary.get("max_us", 0.0)),
                "run_path": str(run_json.parent),
            }
        )

    rows.sort(key=lambda row: (row["p99_us"], row["mean_us"]))

    winners: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"timing={row['timing']}|dim={row['dim']}"
        current = winners.get(key)
        if current is None or row["p99_us"] < current["p99_us"]:
            winners[key] = row

    compare = {
        "count": len(rows),
        "rows": rows,
        "winners_by_mode": winners,
    }

    out_path = runs_root / "compare.json"
    out_path.write_text(json.dumps(compare, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    for key, row in winners.items():
        print(
            f"{key}: winner={row['tag']} selected={row['device_selected']} p99_us={row['p99_us']:.2f}"
        )


if __name__ == "__main__":
    main()
