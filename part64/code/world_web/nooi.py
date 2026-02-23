from __future__ import annotations

import math
from array import array
from typing import Any

from .constants import DAIMO_DT_SECONDS

# Nooi Field Constants
NOOI_GRID_COLS = 64
NOOI_GRID_ROWS = 64
NOOI_LAYERS = 8
NOOI_DECAY_BASE = 0.985
NOOI_DEPOSIT_ALPHA = 0.15
NOOI_CELL_SIZE_X = 1.0 / NOOI_GRID_COLS
NOOI_CELL_SIZE_Y = 1.0 / NOOI_GRID_ROWS


class NooiField:
    def __init__(self, cols: int = NOOI_GRID_COLS, rows: int = NOOI_GRID_ROWS) -> None:
        self.cols = cols
        self.rows = rows
        self.layers: list[list[float]] = [
            [0.0] * (cols * rows * 2) for _ in range(NOOI_LAYERS)
        ]
        self.layer_decay: list[float] = [
            1.0 - (0.01 * (i + 1)) for i in range(NOOI_LAYERS)
        ]

    def decay(self, dt: float) -> None:
        """Apply decay to all field layers."""
        for l_idx in range(NOOI_LAYERS):
            decay_factor = pow(self.layer_decay[l_idx], dt * 60.0)  # Scale to ~60fps
            layer = self.layers[l_idx]
            for i in range(len(layer)):
                layer[i] *= decay_factor
                if abs(layer[i]) < 1e-6:
                    layer[i] = 0.0

    def deposit(
        self,
        x: float,
        y: float,
        vx: float,
        vy: float,
        layer_weights: list[float] | None = None,
    ) -> None:
        """Deposit vector at position (x, y) into field layers."""
        if x < 0.0 or x >= 1.0 or y < 0.0 or y >= 1.0:
            return

        col = int(x * self.cols)
        row = int(y * self.rows)
        idx = (row * self.cols + col) * 2

        # Normalize direction
        mag = math.sqrt(vx * vx + vy * vy)
        if mag < 1e-6:
            return
        dx, dy = vx / mag, vy / mag

        weights = layer_weights or [1.0] * NOOI_LAYERS
        for l_idx in range(min(NOOI_LAYERS, len(weights))):
            w = weights[l_idx]
            if w > 0:
                self.layers[l_idx][idx] += dx * w * NOOI_DEPOSIT_ALPHA
                self.layers[l_idx][idx + 1] += dy * w * NOOI_DEPOSIT_ALPHA

    def sample_vector(self, x: float, y: float) -> tuple[float, float]:
        """Sample aggregated field vector at normalized coordinates."""
        if self.cols <= 0 or self.rows <= 0:
            return (0.0, 0.0)
        col = int(min(self.cols - 1, max(0, x * self.cols)))
        row = int(min(self.rows - 1, max(0, y * self.rows)))
        idx = (row * self.cols + col) * 2
        vx = 0.0
        vy = 0.0
        for layer in self.layers:
            vx += layer[idx]
            vy += layer[idx + 1]
        return (vx, vy)

    def get_grid_snapshot(
        self, particles: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Return the aggregated field grid for the frontend."""
        # 1. Compute persistent vector field
        agg_layer = [0.0] * (self.cols * self.rows * 2)
        for layer in self.layers:
            for i in range(len(layer)):
                agg_layer[i] += layer[i]

        # 2. Compute instantaneous stats from particles (if provided)
        cell_stats: dict[int, dict[str, Any]] = {}
        if particles:
            for p in particles:
                px = p.get("x", 0.5)
                py = p.get("y", 0.5)
                # Ensure float compatibility
                try:
                    fpx = float(px) if px is not None else 0.5
                    fpy = float(py) if py is not None else 0.5
                except (ValueError, TypeError):
                    continue

                c = int(fpx * self.cols)
                r = int(fpy * self.rows)
                if c < 0 or c >= self.cols or r < 0 or r >= self.rows:
                    continue

                idx = r * self.cols + c
                stats = cell_stats.get(idx)
                if stats is None:
                    stats = {"occupancy": 0, "presence_counts": {}}
                    cell_stats[idx] = stats

                stats["occupancy"] += 1

                owner_raw = p.get("owner", "")
                pid = str(owner_raw).strip() if owner_raw else ""
                if pid:
                    stats["presence_counts"][pid] = (
                        stats["presence_counts"].get(pid, 0) + 1
                    )

        cells = []
        max_mag = 0.0

        # First pass to find max magnitude for normalization if needed,
        # or just use it for metadata
        for i in range(0, len(agg_layer), 2):
            vx = agg_layer[i]
            vy = agg_layer[i + 1]
            mag = math.sqrt(vx * vx + vy * vy)
            if mag > max_mag:
                max_mag = mag

        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                vec_idx = idx * 2
                vx = agg_layer[vec_idx]
                vy = agg_layer[vec_idx + 1]
                mag = math.sqrt(vx * vx + vy * vy)

                stats = cell_stats.get(idx, {})
                occupancy = stats.get("occupancy", 0)

                # Dominant presence
                dominant = ""
                if occupancy > 0:
                    counts = stats.get("presence_counts", {})
                    if counts:
                        dominant = max(counts, key=counts.get)  # type: ignore

                # Only include significant cells
                if mag > 0.01 or occupancy > 0:
                    intensity = min(1.0, mag + (occupancy * 0.1))
                    cells.append(
                        {
                            "id": f"{c}_{r}",
                            "col": c,
                            "row": r,
                            "x": (c + 0.5) * NOOI_CELL_SIZE_X,
                            "y": (r + 0.5) * NOOI_CELL_SIZE_Y,
                            "vx": round(vx, 4),
                            "vy": round(vy, 4),
                            "vector_magnitude": round(mag, 4),
                            "occupancy": occupancy,
                            "occupancy_ratio": min(1.0, occupancy / 10.0),
                            "influence": round(mag, 4),
                            "intensity": round(intensity, 4),
                            "message": 0,
                            "route": 0,
                            "dominant_presence_id": dominant,
                        }
                    )

        return {
            "record": "ημ.nooi-field-grid.v1",
            "schema_version": "nooi.field.v1",
            "cols": self.cols,
            "rows": self.rows,
            "cells": cells,
        }
