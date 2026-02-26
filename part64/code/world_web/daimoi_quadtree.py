from __future__ import annotations

from typing import Any, Callable


def rect_intersects_circle(
    bounds: tuple[float, float, float, float],
    x: float,
    y: float,
    radius: float,
) -> bool:
    x0, y0, x1, y1 = bounds
    nearest_x = min(max(x, x0), x1)
    nearest_y = min(max(y, y0), y1)
    dx = x - nearest_x
    dy = y - nearest_y
    return (dx * dx) + (dy * dy) <= (radius * radius)


def quadtree_build(
    items: list[dict[str, Any]],
    *,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    depth: int = 0,
    max_items: int = 24,
    max_depth: int = 7,
    clamp01: Callable[[Any], float],
    safe_float: Callable[[Any, float], float],
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "bounds": bounds,
        "items": list(items),
        "children": None,
    }
    if depth >= max_depth or len(items) <= max_items:
        return node

    x0, y0, x1, y1 = bounds
    mx = (x0 + x1) * 0.5
    my = (y0 + y1) * 0.5
    quadrants = [
        (x0, y0, mx, my),
        (mx, y0, x1, my),
        (x0, my, mx, y1),
        (mx, my, x1, y1),
    ]
    buckets: list[list[dict[str, Any]]] = [[], [], [], []]
    spill: list[dict[str, Any]] = []

    for item in items:
        ix = clamp01(safe_float(item.get("x", 0.5), 0.5))
        iy = clamp01(safe_float(item.get("y", 0.5), 0.5))
        assigned = False
        for index, (qx0, qy0, qx1, qy1) in enumerate(quadrants):
            if qx0 <= ix < qx1 and qy0 <= iy < qy1:
                buckets[index].append(item)
                assigned = True
                break
        if not assigned:
            spill.append(item)

    child_nodes: list[dict[str, Any]] = []
    for bucket, qbounds in zip(buckets, quadrants):
        if bucket:
            child_nodes.append(
                quadtree_build(
                    bucket,
                    bounds=qbounds,
                    depth=depth + 1,
                    max_items=max_items,
                    max_depth=max_depth,
                    clamp01=clamp01,
                    safe_float=safe_float,
                )
            )

    if not child_nodes:
        return node

    node["items"] = spill
    node["children"] = child_nodes
    return node


def quadtree_query_radius(
    node: dict[str, Any],
    x: float,
    y: float,
    radius: float,
    out: list[dict[str, Any]],
    *,
    rect_intersects_circle_fn: Callable[
        [tuple[float, float, float, float], float, float, float], bool
    ] = rect_intersects_circle,
) -> None:
    if not node:
        return
    bounds = node.get("bounds", (0.0, 0.0, 1.0, 1.0))
    if not isinstance(bounds, tuple) or len(bounds) != 4:
        return
    if not rect_intersects_circle_fn(bounds, x, y, radius):
        return

    items = node.get("items", [])
    if isinstance(items, list) and items:
        out.extend(item for item in items if isinstance(item, dict))

    children = node.get("children")
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                quadtree_query_radius(
                    child,
                    x,
                    y,
                    radius,
                    out,
                    rect_intersects_circle_fn=rect_intersects_circle_fn,
                )


def quadtree_semantic_aggregate(
    node: dict[str, Any],
    *,
    vector_dims: int,
    clamp01: Callable[[Any], float],
    safe_float: Callable[[Any, float], float],
    safe_int: Callable[[Any, int], int],
    finite_float: Callable[[Any, float], float],
) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {
            "weight": 0.0,
            "count": 0,
            "cx": 0.5,
            "cy": 0.5,
            "vector_sum": [0.0] * max(1, int(vector_dims)),
            "single_id": "",
        }

    dims = max(1, int(vector_dims))
    weight_sum = 0.0
    count_sum = 0
    weighted_x = 0.0
    weighted_y = 0.0
    vector_sum = [0.0] * dims
    single_id = ""

    items = node.get("items", [])
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            source_weight = max(0.0, safe_float(item.get("weight", 1.0), 1.0))
            if source_weight <= 1e-8:
                continue
            sx = clamp01(safe_float(item.get("x", 0.5), 0.5))
            sy = clamp01(safe_float(item.get("y", 0.5), 0.5))
            source_vector_raw = item.get("vector", [])
            source_vector = (
                source_vector_raw if isinstance(source_vector_raw, list) else []
            )
            source_id = str(item.get("id", "")).strip()
            weighted_x += sx * source_weight
            weighted_y += sy * source_weight
            for index in range(min(dims, len(source_vector))):
                vector_sum[index] += (
                    finite_float(source_vector[index], 0.0) * source_weight
                )
            weight_sum += source_weight
            count_sum += 1
            single_id = source_id if count_sum == 1 else ""

    children = node.get("children")
    if isinstance(children, list):
        for child in children:
            if not isinstance(child, dict):
                continue
            child_agg = quadtree_semantic_aggregate(
                child,
                vector_dims=dims,
                clamp01=clamp01,
                safe_float=safe_float,
                safe_int=safe_int,
                finite_float=finite_float,
            )
            child_weight = max(0.0, safe_float(child_agg.get("weight", 0.0), 0.0))
            if child_weight <= 1e-8:
                continue
            child_count = max(0, safe_int(child_agg.get("count", 0), 0))
            child_cx = clamp01(safe_float(child_agg.get("cx", 0.5), 0.5))
            child_cy = clamp01(safe_float(child_agg.get("cy", 0.5), 0.5))
            child_vector_sum_raw = child_agg.get("vector_sum", [])
            child_vector_sum = (
                child_vector_sum_raw if isinstance(child_vector_sum_raw, list) else []
            )
            weighted_x += child_cx * child_weight
            weighted_y += child_cy * child_weight
            for index in range(min(dims, len(child_vector_sum))):
                vector_sum[index] += finite_float(child_vector_sum[index], 0.0)
            previous_count = count_sum
            count_sum += child_count
            if previous_count == 0 and child_count == 1:
                single_id = str(child_agg.get("single_id", "")).strip()
            elif count_sum > 1:
                single_id = ""
            weight_sum += child_weight

    center_x = 0.5
    center_y = 0.5
    if weight_sum > 1e-8:
        center_x = clamp01(weighted_x / weight_sum)
        center_y = clamp01(weighted_y / weight_sum)
    aggregate = {
        "weight": weight_sum,
        "count": count_sum,
        "cx": center_x,
        "cy": center_y,
        "vector_sum": vector_sum,
        "single_id": single_id if count_sum == 1 else "",
    }
    node["semantic_agg"] = aggregate
    return aggregate
