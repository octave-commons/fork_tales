import io
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

# Attempt to import PIL for wireframe rendering
try:
    from PIL import Image, ImageDraw, ImageFont

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Attempt to import tensorflow for NPU/Lite delegation
try:
    import tensorflow as tf  # type: ignore

    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

_LOGGER = logging.getLogger("ux_critic")


def render_wireframe(
    projection: Dict[str, Any], width: int = 1280, height: int = 800
) -> Optional[bytes]:
    """
    Renders a wireframe representation of the current UI projection.
    Returns JPEG bytes or None if PIL is missing.
    """
    if not _PIL_AVAILABLE:
        _LOGGER.warning("PIL not available; cannot render wireframe.")
        return None

    layout = projection.get("layout", {})
    rects = layout.get("rects", {})

    # Create canvas (dark theme background)
    img = Image.new("RGB", (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Grid lines (subtle)
    cols = 12
    rows = 12
    cell_w = width / cols
    cell_h = height / rows

    for i in range(cols + 1):
        x = int(i * cell_w)
        draw.line([(x, 0), (x, height)], fill=(50, 50, 50), width=1)
    for i in range(rows + 1):
        y = int(i * cell_h)
        draw.line([(0, y), (width, y)], fill=(50, 50, 50), width=1)

    # Render panels
    for element_id, rect in rects.items():
        # rect is normalized 0..1
        rx = int(rect.get("x", 0) * width)
        ry = int(rect.get("y", 0) * height)
        rw = int(rect.get("w", 0.1) * width)
        rh = int(rect.get("h", 0.1) * height)

        # Color based on element ID hash or type
        hue = hash(element_id) % 360
        fill_color = f"hsl({hue}, 40%, 20%)"
        outline_color = f"hsl({hue}, 80%, 60%)"

        # Draw box
        # PIL doesn't support HSL directly strings in all versions, simplifying to RGB approximation or safe colors
        # Fallback to gray/colored boxes
        r = hash(element_id) & 0xFF
        g = (hash(element_id) >> 8) & 0xFF
        b = (hash(element_id) >> 16) & 0xFF

        draw.rectangle(
            [rx, ry, rx + rw, ry + rh],
            fill=(r // 4, g // 4, b // 4),
            outline=(r, g, b),
            width=2,
        )

        # Draw label
        label = element_id.split(".")[-1]
        draw.text((rx + 5, ry + 5), label, fill=(220, 220, 220))
        draw.text(
            (rx + 5, ry + 20),
            f"{int(rect.get('w', 0) * 100)}% x {int(rect.get('h', 0) * 100)}%",
            fill=(150, 150, 150),
        )

    # Output to buffer
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def critique_ux(projection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the UX layout using heuristics and (simulated) VLM intuition.
    """
    layout = projection.get("layout", {})
    rects = layout.get("rects", {})

    critique = {
        "score": 0.0,
        "issues": [],
        "suggestions": [],
        "latency_ms": 0.0,  # Placeholder for NPU latency
    }

    # 1. Overlap Detection (Heuristic)
    # Simple N^2 check for significant overlaps
    items = list(rects.items())
    overlaps = []
    total_area = 0.0

    for i in range(len(items)):
        id1, r1 = items[i]
        total_area += r1.get("w", 0) * r1.get("h", 0)

        for j in range(i + 1, len(items)):
            id2, r2 = items[j]

            # Intersection
            x_left = max(r1["x"], r2["x"])
            y_top = max(r1["y"], r2["y"])
            x_right = min(r1["x"] + r1["w"], r2["x"] + r2["w"])
            y_bottom = min(r1["y"] + r1["h"], r2["y"] + r2["h"])

            if x_right > x_left and y_bottom > y_top:
                # Overlap area
                area = (x_right - x_left) * (y_bottom - y_top)
                if area > 0.01:  # Threshold
                    overlaps.append(
                        f"{id1.split('.')[-1]} overlaps {id2.split('.')[-1]}"
                    )

    if overlaps:
        critique["score"] -= 0.2 * len(overlaps)
        critique["issues"].extend(overlaps)

    # 2. Balance / Symmetry (Simulated VLM Insight)
    # Calculate center of mass
    cx, cy = 0.0, 0.0
    n = len(rects)
    if n > 0:
        for r in rects.values():
            cx += r["x"] + r["w"] / 2
            cy += r["y"] + r["h"] / 2
        cx /= n
        cy /= n

        dist_from_center = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
        if dist_from_center > 0.1:
            critique["issues"].append(
                "Layout is unbalanced (center of mass off-center)"
            )
            critique["score"] -= 0.1
        else:
            critique["score"] += 0.1

    # 3. TensorFlow NPU Simulation (Placeholder for actual TFLite call)
    # In a real scenario, we would:
    #   interpreter = tf.lite.Interpreter(model_path="ux_critic_float16.tflite")
    #   interpreter.allocate_tensors()
    #   interpreter.set_tensor(input_details[0]['index'], wireframe_bytes)
    #   interpreter.invoke()
    #   score = interpreter.get_tensor(output_details[0]['index'])

    critique["score"] = max(0.0, min(1.0, 0.8 + critique["score"]))

    if critique["score"] < 0.6:
        critique["suggestions"].append("Spread panels out to reduce density.")
    elif critique["score"] > 0.9:
        critique["suggestions"].append("Layout is highly coherent.")

    return critique
