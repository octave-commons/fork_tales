# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Set, cast


lib_logger = logging.getLogger("proxy_app.smart_gateway")


TRUE_VALUES = {"1", "true", "yes", "on"}
SUPPORTED_MODES = {"smart", "direct", "hardway"}
SUPPORTED_HARDWARE = {"gpu", "npu", "openvino", "tensorflow", "cpu"}


def _as_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in TRUE_VALUES


def _normalize_word(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


class SmartGateway:
    """
    Field-aware model router for OpenAI-compatible chat requests.

    Request extension contract:
    {
      "model": "provider/model",
      "messages": [...],
      "gateway": {
        "mode": "smart|direct|hardway",
        "field": "code|train|vision|retrieval|ops|general",
        "hardware": "gpu|npu|openvino|tensorflow|cpu",
        "direct_model": "provider/model",
        "hardway_model": "provider/model",
        "fields": {"code": 0.9, "train": 0.2}
      }
    }
    """

    def __init__(self) -> None:
        self.enabled = _as_bool(os.getenv("SMART_GATEWAY_ENABLED"), default=True)
        self.default_mode = _normalize_word(
            os.getenv("SMART_GATEWAY_DEFAULT_MODE", "smart")
        )
        if self.default_mode not in SUPPORTED_MODES:
            self.default_mode = "smart"

        self.public_provider_tag = os.getenv(
            "PROMETHEAN_PROVIDER_TAG", "octave-commons"
        ).strip()
        self.internal_provider_tag = os.getenv(
            "PROMETHEAN_PROVIDER_INTERNAL", "octave_commons"
        ).strip()
        self.default_model_name = os.getenv(
            "PROMETHEAN_MODEL_NAME", "promethean"
        ).strip()
        self.default_public_model = (
            f"{self.public_provider_tag}/{self.default_model_name}"
        )

        self.advertise_default_model = _as_bool(
            os.getenv("SMART_GATEWAY_ADVERTISE_DEFAULT_MODEL"), default=True
        )

        self.provider_aliases = self._load_provider_aliases()
        self.reverse_provider_aliases = {
            internal: public for public, internal in self.provider_aliases.items()
        }

        self.field_routes = self._load_field_routes()

    def _load_provider_aliases(self) -> Dict[str, str]:
        aliases: Dict[str, str] = {
            _normalize_word(self.public_provider_tag): _normalize_word(
                self.internal_provider_tag
            )
        }

        raw = os.getenv("SMART_GATEWAY_PROVIDER_ALIASES", "")
        if not raw.strip():
            return aliases

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for public, internal in parsed.items():
                    public_key = _normalize_word(str(public))
                    internal_key = _normalize_word(str(internal))
                    if public_key and internal_key:
                        aliases[public_key] = internal_key
            else:
                lib_logger.warning(
                    "SMART_GATEWAY_PROVIDER_ALIASES must be a JSON object."
                )
        except json.JSONDecodeError as exc:
            lib_logger.warning(
                "Failed to parse SMART_GATEWAY_PROVIDER_ALIASES: %s", exc
            )

        return aliases

    def _default_field_routes(self) -> Dict[str, Dict[str, str]]:
        provider = self.public_provider_tag
        base = self.default_public_model
        return {
            "general": {
                "default": base,
                "hardway": f"{provider}/promethean-hardway",
                "gpu": f"gpu/promethean-gpu",
                "npu": f"npu/promethean-npu",
                "openvino": f"openvino/promethean-openvino",
                "tensorflow": f"tensorflow/promethean-train",
            },
            "code": {
                "default": base,
                "hardway": f"{provider}/promethean-code-hardway",
                "gpu": f"gpu/promethean-code-gpu",
                "npu": f"npu/promethean-code-npu",
                "openvino": f"openvino/promethean-code-edge",
                "tensorflow": f"tensorflow/promethean-code-train",
            },
            "train": {
                "default": f"tensorflow/promethean-train",
                "hardway": f"{provider}/promethean-train-hardway",
                "gpu": f"gpu/promethean-train-gpu",
                "npu": f"npu/promethean-train-npu",
                "openvino": f"openvino/promethean-train-openvino",
            },
            "vision": {
                "default": f"{provider}/promethean-vision",
                "gpu": f"gpu/promethean-vision-gpu",
                "openvino": f"openvino/promethean-vision-edge",
                "tensorflow": "tensorflow/qwen3-vl-2b-image",
                "hardway": f"{provider}/promethean-vision-hardway",
            },
            "retrieval": {
                "default": f"{provider}/promethean-rag",
                "gpu": f"gpu/promethean-rag-gpu",
                "npu": f"npu/promethean-rag-npu",
                "hardway": f"{provider}/promethean-rag-hardway",
            },
            "ops": {
                "default": base,
                "hardway": f"{provider}/promethean-ops-hardway",
                "gpu": f"gpu/promethean-ops-gpu",
            },
        }

    def _load_field_routes(self) -> Dict[str, Dict[str, str]]:
        routes = self._default_field_routes()
        raw = os.getenv("SMART_GATEWAY_FIELD_ROUTES_JSON", "")
        if not raw.strip():
            return routes

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                lib_logger.warning(
                    "SMART_GATEWAY_FIELD_ROUTES_JSON must be a JSON object."
                )
                return routes

            for field_name, cfg in parsed.items():
                field_key = _normalize_word(str(field_name))
                if not field_key:
                    continue
                if field_key not in routes:
                    routes[field_key] = {}
                if isinstance(cfg, dict):
                    for route_key, model in cfg.items():
                        route_name = _normalize_word(str(route_key))
                        model_name = str(model).strip()
                        if route_name and model_name:
                            routes[field_key][route_name] = model_name
        except json.JSONDecodeError as exc:
            lib_logger.warning(
                "Failed to parse SMART_GATEWAY_FIELD_ROUTES_JSON: %s", exc
            )

        return routes

    def _normalize_provider(self, provider: str) -> str:
        key = _normalize_word(provider)
        if key in self.provider_aliases:
            return self.provider_aliases[key]
        return key

    def _publicize_provider(self, provider: str) -> str:
        key = _normalize_word(provider)
        if key in self.reverse_provider_aliases:
            return self.reverse_provider_aliases[key]
        return key

    def _ensure_model_format(self, model: str) -> str:
        candidate = str(model or "").strip()
        if not candidate:
            return self.default_public_model
        return candidate

    def to_internal_model(self, model: str) -> str:
        candidate = self._ensure_model_format(model)
        if "/" not in candidate:
            return candidate
        provider, name = candidate.split("/", 1)
        internal_provider = self._normalize_provider(provider)
        return f"{internal_provider}/{name}"

    def to_public_model(self, model: str) -> str:
        candidate = self._ensure_model_format(model)
        if "/" not in candidate:
            return candidate
        provider, name = candidate.split("/", 1)
        public_provider = self._publicize_provider(provider)
        return f"{public_provider}/{name}"

    def _extract_field(self, payload: Dict[str, Any], gateway: Dict[str, Any]) -> str:
        field = gateway.get("field") or payload.get("field")
        if field:
            return _normalize_word(str(field))

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            metadata_field = metadata.get("field")
            if metadata_field:
                return _normalize_word(str(metadata_field))

        weighted_fields = gateway.get("fields")
        if isinstance(weighted_fields, dict) and weighted_fields:
            try:
                ranked = sorted(
                    weighted_fields.items(),
                    key=lambda item: float(item[1]),
                    reverse=True,
                )
                return _normalize_word(str(ranked[0][0]))
            except Exception:
                first_key = next(iter(weighted_fields.keys()))
                return _normalize_word(str(first_key))

        if isinstance(weighted_fields, list) and weighted_fields:
            return _normalize_word(str(weighted_fields[0]))

        return "general"

    def _extract_mode(self, gateway: Dict[str, Any]) -> str:
        if gateway.get("hardway") is True:
            return "hardway"
        if gateway.get("direct") is True:
            return "direct"

        mode = _normalize_word(gateway.get("mode") or self.default_mode)
        if mode in SUPPORTED_MODES:
            return mode
        return self.default_mode

    def _extract_hardware(
        self, payload: Dict[str, Any], gateway: Dict[str, Any]
    ) -> str:
        candidates: List[Any] = []
        candidates.append(gateway.get("hardware"))
        candidates.append(payload.get("hardware"))

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            candidates.append(metadata.get("hardware"))

        for candidate in candidates:
            if isinstance(candidate, list):
                for item in candidate:
                    normalized = _normalize_word(str(item))
                    if normalized in SUPPORTED_HARDWARE:
                        return normalized
            elif isinstance(candidate, str):
                for token in candidate.split(","):
                    normalized = _normalize_word(token)
                    if normalized in SUPPORTED_HARDWARE:
                        return normalized

        return ""

    def _routes_for_field(self, field: str) -> Dict[str, str]:
        if field in self.field_routes:
            return self.field_routes[field]
        return self.field_routes.get("general", {})

    def _candidate_models(self, field: str) -> List[str]:
        routes = self._routes_for_field(field)
        ordered_keys = ["default", "gpu", "npu", "openvino", "tensorflow", "hardway"]

        candidates: List[str] = []
        for key in ordered_keys:
            if key in routes and routes[key] not in candidates:
                candidates.append(routes[key])

        for model in routes.values():
            if model not in candidates:
                candidates.append(model)

        return candidates

    def _resolve_model_for_mode(
        self,
        mode: str,
        field: str,
        hardware: str,
        requested_model: str,
        gateway: Dict[str, Any],
    ) -> tuple[str, str]:
        routes = self._routes_for_field(field)

        if mode == "direct":
            model = (
                gateway.get("direct_model") or gateway.get("model") or requested_model
            )
            return self._ensure_model_format(str(model)), "direct mode"

        if mode == "hardway":
            hardway_model = gateway.get("hardway_model") or routes.get("hardway")
            if hardway_model:
                return self._ensure_model_format(str(hardway_model)), "hardway route"
            return self._ensure_model_format(requested_model), "hardway fallback"

        # smart mode
        if hardware and hardware in routes:
            return (
                self._ensure_model_format(str(routes[hardware])),
                f"field={field}, hardware={hardware}",
            )

        if "default" in routes:
            return self._ensure_model_format(str(routes["default"])), f"field={field}"

        return self._ensure_model_format(requested_model), "requested model fallback"

    def _model_provider(self, model: str) -> str:
        candidate = self._ensure_model_format(model)
        if "/" not in candidate:
            return ""
        return _normalize_word(candidate.split("/", 1)[0])

    def resolve_for_request(
        self,
        payload: Dict[str, Any],
        available_providers: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        gateway_obj = payload.get("gateway")
        gateway: Dict[str, Any] = (
            cast(Dict[str, Any], gateway_obj) if isinstance(gateway_obj, dict) else {}
        )

        requested_model = self._ensure_model_format(
            str(payload.get("model") or self.default_public_model)
        )
        mode = self._extract_mode(gateway)
        field = self._extract_field(payload, gateway)
        hardware = self._extract_hardware(payload, gateway)

        has_gateway_directive = bool(gateway) or bool(payload.get("route_via_fields"))
        model_is_promethean_alias = self._model_provider(requested_model) in {
            _normalize_word(self.public_provider_tag),
            _normalize_word(self.internal_provider_tag),
        }

        should_apply = self.enabled and (
            has_gateway_directive or model_is_promethean_alias
        )

        if should_apply:
            selected_public_model, reason = self._resolve_model_for_mode(
                mode, field, hardware, requested_model, gateway
            )
        else:
            selected_public_model = requested_model
            reason = "gateway bypass"

        selected_internal_model = self.to_internal_model(selected_public_model)
        provider_available = True

        if available_providers is not None:
            provider_set: Set[str] = {
                _normalize_word(str(p)) for p in available_providers
            }
            selected_provider = self._model_provider(selected_internal_model)
            if selected_provider not in provider_set:
                provider_available = False
                requested_internal_model = self.to_internal_model(requested_model)
                requested_provider = self._model_provider(requested_internal_model)
                if requested_provider in provider_set:
                    selected_internal_model = requested_internal_model
                    selected_public_model = self.to_public_model(
                        requested_internal_model
                    )
                    reason = (
                        f"{reason}; fallback to requested model because provider "
                        f"'{selected_provider}' has no credentials"
                    )

        return {
            "enabled": self.enabled,
            "applied": should_apply,
            "mode": mode,
            "field": field,
            "hardware": hardware or None,
            "requested_model": requested_model,
            "resolved_model_public": selected_public_model,
            "resolved_model": selected_internal_model,
            "reason": reason,
            "candidate_models": self._candidate_models(field),
            "provider_available": provider_available,
        }

    def apply_route(
        self, payload: Dict[str, Any], decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        routed = copy.deepcopy(payload)
        routed.pop("gateway", None)
        routed.pop("field", None)
        routed.pop("fields", None)
        routed.pop("hardware", None)
        routed.pop("route_via_fields", None)
        routed["model"] = decision.get(
            "resolved_model", self.to_internal_model(payload.get("model", ""))
        )
        return routed

    def present_models(self, model_ids: List[str]) -> List[str]:
        visible: List[str] = []
        seen = set()
        for model_id in model_ids:
            public_model = self.to_public_model(model_id)
            if public_model not in seen:
                visible.append(public_model)
                seen.add(public_model)

        if self.advertise_default_model and self.default_public_model not in seen:
            visible.insert(0, self.default_public_model)

        return visible

    def describe(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "default_mode": self.default_mode,
            "modes": sorted(SUPPORTED_MODES),
            "hardware_options": sorted(SUPPORTED_HARDWARE),
            "default_model": self.default_public_model,
            "provider_aliases": self.provider_aliases,
            "field_routes": self.field_routes,
            "advertise_default_model": self.advertise_default_model,
        }
