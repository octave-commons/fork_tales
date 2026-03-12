from __future__ import annotations

import copy
import math
from datetime import datetime, timezone
from typing import Any


def _config_numeric_scalar(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    return None


def _config_numeric_only(value: Any) -> Any:
    scalar = _config_numeric_scalar(value)
    if scalar is not None:
        return scalar

    if isinstance(value, dict):
        nested: dict[str, Any] = {}
        for key in sorted(value.keys(), key=lambda item: str(item)):
            numeric_value = _config_numeric_only(value[key])
            if numeric_value is None:
                continue
            nested[str(key)] = numeric_value
        return nested or None

    if isinstance(value, (list, tuple, set)):
        nested_list: list[Any] = []
        sequence = (
            sorted(value, key=lambda item: str(item))
            if isinstance(value, set)
            else value
        )
        for item in sequence:
            numeric_value = _config_numeric_only(item)
            if numeric_value is None:
                continue
            nested_list.append(numeric_value)
        return nested_list or None

    return None


def _config_numeric_leaf_count(value: Any) -> int:
    scalar = _config_numeric_scalar(value)
    if scalar is not None:
        return 1
    if isinstance(value, dict):
        return sum(_config_numeric_leaf_count(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(_config_numeric_leaf_count(item) for item in value)
    return 0


def _config_collect_module_constants(
    module: Any,
    *,
    prefixes: tuple[str, ...],
    exact_names: tuple[str, ...],
) -> tuple[dict[str, Any], int]:
    names = sorted(str(name) for name in vars(module).keys())
    selected: dict[str, Any] = {}
    numeric_leaf_count = 0
    for name in names:
        include = name in exact_names
        if not include:
            include = any(name.startswith(prefix) for prefix in prefixes)
        if not include:
            continue
        numeric_value = _config_numeric_only(getattr(module, name, None))
        if numeric_value is None:
            continue
        selected[name] = numeric_value
        numeric_leaf_count += _config_numeric_leaf_count(numeric_value)
    return selected, numeric_leaf_count


def _config_collect_selected_names(
    module: Any,
    *,
    prefixes: tuple[str, ...],
    exact_names: tuple[str, ...],
) -> list[str]:
    names = sorted(str(name) for name in vars(module).keys())
    selected: list[str] = []
    for name in names:
        include = name in exact_names
        if not include:
            include = any(name.startswith(prefix) for prefix in prefixes)
        if not include:
            continue
        numeric_value = _config_numeric_only(getattr(module, name, None))
        if numeric_value is None:
            continue
        selected.append(name)
    return selected


def config_capture_runtime_baseline(
    config_module_specs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    baseline: dict[str, dict[str, Any]] = {}
    for module_name, spec in config_module_specs.items():
        module = spec.get("module")
        if module is None:
            continue
        selected_names = _config_collect_selected_names(
            module,
            prefixes=tuple(spec.get("prefixes", ())),
            exact_names=tuple(spec.get("exact_names", ())),
        )
        module_baseline: dict[str, Any] = {}
        for name in selected_names:
            module_baseline[name] = copy.deepcopy(getattr(module, name, None))
        baseline[module_name] = module_baseline
    return baseline


def config_normalize_path_tokens(path_raw: Any) -> list[str]:
    if path_raw is None:
        return []
    if isinstance(path_raw, list):
        return [str(item).strip() for item in path_raw if str(item).strip()]
    if isinstance(path_raw, str):
        clean = path_raw.strip()
        if not clean:
            return []
        normalized = clean.replace("[", ".").replace("]", "")
        return [token for token in normalized.split(".") if token]
    return []


def _config_resolve_dict_key(container: dict[Any, Any], token: str) -> Any:
    if token in container:
        return token
    for key in container.keys():
        if str(key) == token:
            return key
    raise KeyError(token)


def _config_parse_list_index(token: str, length: int) -> int:
    try:
        index = int(token)
    except Exception as exc:
        raise ValueError(f"invalid_index:{token}") from exc
    if index < 0:
        index = length + index
    if index < 0 or index >= length:
        raise IndexError(f"index_out_of_range:{token}")
    return index


def _config_coerce_numeric_like(reference: Any, value: float | int) -> float | int:
    if isinstance(reference, bool):
        return int(round(float(value)))
    if isinstance(reference, int):
        return int(round(float(value)))
    return float(value)


def _config_clamp_scalar_update(
    *,
    module_name: str,
    key_name: str,
    value: float | int,
    scalar_limits: dict[tuple[str, str], tuple[float, float]],
) -> float | int:
    limits = scalar_limits.get((module_name, key_name))
    if limits is None:
        return value
    lower, upper = limits
    return max(lower, min(upper, float(value)))


def _config_get_at_path(root: Any, path_tokens: list[str]) -> Any:
    value = root
    for token in path_tokens:
        if isinstance(value, dict):
            value = value[_config_resolve_dict_key(value, token)]
            continue
        if isinstance(value, (list, tuple)):
            index = _config_parse_list_index(token, len(value))
            value = value[index]
            continue
        raise TypeError(f"non_container_at:{token}")
    return value


def _config_set_at_path(
    root: Any, path_tokens: list[str], new_scalar: float | int
) -> Any:
    if not path_tokens:
        scalar = _config_numeric_scalar(root)
        if scalar is None:
            raise TypeError("target_not_numeric")
        return _config_coerce_numeric_like(root, new_scalar)

    token = path_tokens[0]
    tail = path_tokens[1:]
    if isinstance(root, dict):
        key = _config_resolve_dict_key(root, token)
        updated = dict(root)
        updated[key] = _config_set_at_path(root[key], tail, new_scalar)
        return updated

    if isinstance(root, list):
        index = _config_parse_list_index(token, len(root))
        updated_list = list(root)
        updated_list[index] = _config_set_at_path(root[index], tail, new_scalar)
        return updated_list

    if isinstance(root, tuple):
        index = _config_parse_list_index(token, len(root))
        updated_list = list(root)
        updated_list[index] = _config_set_at_path(root[index], tail, new_scalar)
        return tuple(updated_list)

    raise TypeError(f"non_container_at:{token}")


def config_apply_update(
    *,
    module_name: str,
    key_name: str,
    path_tokens: list[str],
    value: Any,
    config_module_specs: dict[str, dict[str, Any]],
    config_runtime_edit_lock: Any,
    scalar_limits: dict[tuple[str, str], tuple[float, float]],
) -> dict[str, Any]:
    requested_module = str(module_name or "").strip().lower()
    requested_key = str(key_name or "").strip()
    if not requested_module:
        return {"ok": False, "error": "module_required"}
    if not requested_key:
        return {"ok": False, "error": "key_required"}
    if requested_module not in config_module_specs:
        return {
            "ok": False,
            "error": "unknown_module",
            "module": requested_module,
            "available_modules": sorted(config_module_specs.keys()),
        }

    next_scalar = _config_numeric_scalar(value)
    if next_scalar is None and isinstance(value, str):
        text = value.strip()
        if text:
            try:
                next_scalar = float(text)
            except Exception:
                next_scalar = None
    if next_scalar is None:
        return {
            "ok": False,
            "error": "numeric_value_required",
            "module": requested_module,
            "key": requested_key,
        }
    next_scalar_value = _config_clamp_scalar_update(
        module_name=requested_module,
        key_name=requested_key,
        value=next_scalar,
        scalar_limits=scalar_limits,
    )

    spec = config_module_specs[requested_module]
    module = spec.get("module")
    if module is None:
        return {
            "ok": False,
            "error": "module_unavailable",
            "module": requested_module,
        }

    selected_names = set(
        _config_collect_selected_names(
            module,
            prefixes=tuple(spec.get("prefixes", ())),
            exact_names=tuple(spec.get("exact_names", ())),
        )
    )
    if requested_key not in selected_names:
        return {
            "ok": False,
            "error": "unknown_constant",
            "module": requested_module,
            "key": requested_key,
        }

    try:
        with config_runtime_edit_lock:
            current_value = copy.deepcopy(getattr(module, requested_key, None))
            previous_leaf = _config_get_at_path(current_value, path_tokens)
            updated_value = _config_set_at_path(
                current_value, path_tokens, next_scalar_value
            )
            setattr(module, requested_key, updated_value)
            current_after = copy.deepcopy(getattr(module, requested_key, None))
            current_leaf = _config_get_at_path(current_after, path_tokens)
    except Exception as exc:
        return {
            "ok": False,
            "error": "config_update_failed",
            "detail": f"{exc.__class__.__name__}: {exc}",
            "module": requested_module,
            "key": requested_key,
            "path": path_tokens,
        }

    return {
        "ok": True,
        "record": "eta-mu.runtime-config.update.v1",
        "schema_version": "runtime.config.update.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "module": requested_module,
        "key": requested_key,
        "path": path_tokens,
        "previous": _config_numeric_only(previous_leaf),
        "current": _config_numeric_only(current_leaf),
    }


def config_reset_updates(
    *,
    module_name: str = "",
    key_name: str = "",
    path_tokens: list[str] | None = None,
    config_module_specs: dict[str, dict[str, Any]],
    config_runtime_baseline: dict[str, dict[str, Any]],
    config_runtime_edit_lock: Any,
) -> dict[str, Any]:
    requested_module = str(module_name or "").strip().lower()
    requested_key = str(key_name or "").strip()
    normalized_path = path_tokens or []

    if normalized_path and not requested_key:
        return {"ok": False, "error": "key_required_for_path_reset"}

    available_modules = sorted(config_module_specs.keys())
    if requested_module and requested_module not in config_module_specs:
        return {
            "ok": False,
            "error": "unknown_module",
            "module": requested_module,
            "available_modules": available_modules,
        }

    module_names = [requested_module] if requested_module else available_modules
    applied: list[dict[str, Any]] = []

    with config_runtime_edit_lock:
        for module_item in module_names:
            spec = config_module_specs[module_item]
            module = spec.get("module")
            if module is None:
                continue
            baseline_module = config_runtime_baseline.get(module_item, {})
            if not baseline_module:
                continue
            selected_names = set(
                _config_collect_selected_names(
                    module,
                    prefixes=tuple(spec.get("prefixes", ())),
                    exact_names=tuple(spec.get("exact_names", ())),
                )
            )
            key_names = (
                [requested_key] if requested_key else sorted(baseline_module.keys())
            )
            for key_item in key_names:
                if key_item not in selected_names or key_item not in baseline_module:
                    continue

                baseline_value = copy.deepcopy(baseline_module[key_item])
                if normalized_path:
                    try:
                        baseline_leaf = _config_get_at_path(
                            baseline_value, normalized_path
                        )
                        baseline_scalar = _config_numeric_scalar(baseline_leaf)
                        if baseline_scalar is None:
                            continue
                        current_value = copy.deepcopy(getattr(module, key_item, None))
                        updated_value = _config_set_at_path(
                            current_value, normalized_path, baseline_scalar
                        )
                    except Exception:
                        continue
                else:
                    updated_value = baseline_value

                setattr(module, key_item, updated_value)
                applied.append(
                    {
                        "module": module_item,
                        "key": key_item,
                        "path": list(normalized_path),
                    }
                )

    if requested_key and not applied:
        return {
            "ok": False,
            "error": "unknown_constant",
            "module": requested_module,
            "key": requested_key,
        }

    return {
        "ok": True,
        "record": "eta-mu.runtime-config.reset.v1",
        "schema_version": "runtime.config.reset.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "module": requested_module,
        "key": requested_key,
        "path": normalized_path,
        "reset_count": len(applied),
        "resets": applied[:128],
    }


def config_payload(
    *,
    module_filter: str = "",
    config_module_specs: dict[str, dict[str, Any]],
    runtime_version_snapshot: Any,
) -> dict[str, Any]:
    requested = str(module_filter or "").strip().lower()
    available_modules = sorted(config_module_specs.keys())
    if requested and requested not in config_module_specs:
        return {
            "ok": False,
            "error": "unknown_module",
            "requested_module": requested,
            "available_modules": available_modules,
        }

    module_names = [requested] if requested else available_modules
    modules_payload: dict[str, Any] = {}
    total_constants = 0
    total_numeric_leaf_count = 0
    for module_name in module_names:
        spec = config_module_specs[module_name]
        constants, leaf_count = _config_collect_module_constants(
            spec["module"],
            prefixes=tuple(spec["prefixes"]),
            exact_names=tuple(spec["exact_names"]),
        )
        modules_payload[module_name] = {
            "constants": copy.deepcopy(constants),
            "constant_count": len(constants),
            "numeric_leaf_count": leaf_count,
        }
        total_constants += len(constants)
        total_numeric_leaf_count += leaf_count

    return {
        "ok": True,
        "record": "eta-mu.runtime-config.v1",
        "schema_version": "runtime.config.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_config_version": int(runtime_version_snapshot()),
        "available_modules": available_modules,
        "requested_module": requested,
        "module_count": len(modules_payload),
        "constant_count": total_constants,
        "numeric_leaf_count": total_numeric_leaf_count,
        "modules": modules_payload,
    }
