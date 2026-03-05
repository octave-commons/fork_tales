from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class WorldRuntimeController:
    part_root: Path
    vault_root: Path
    task_queue: Any
    council_chamber: Any
    myth_tracker: Any
    life_tracker: Any
    collect_catalog_fast: Callable[[], dict[str, Any]]
    schedule_runtime_catalog_refresh: Callable[[], None]
    schedule_runtime_inbox_sync: Callable[[], None]
    collect_runtime_catalog_isolated: Callable[
        [Path, Path], tuple[dict[str, Any] | None, str]
    ]
    runtime_catalog_fallback: Callable[[Path, Path], dict[str, Any]]
    runtime_catalog_cache_lock: Any
    runtime_catalog_collect_lock: Any
    runtime_catalog_cache: dict[str, Any]
    runtime_catalog_cache_seconds: float
    runtime_eta_mu_sync_enabled: bool
    resource_monitor_snapshot: Callable[..., dict[str, Any]]
    influence_tracker: Any
    muse_runtime_snapshot: Callable[[], dict[str, Any]]
    attach_ui_projection: Callable[..., Any]
    simulation_http_trim_catalog: Callable[[dict[str, Any]], dict[str, Any]]
    collect_docker_simulation_snapshot: Callable[[], dict[str, Any]]
    build_simulation_state: Callable[..., dict[str, Any]]
    build_ui_projection: Callable[..., dict[str, Any]]
    entity_manifest: list[dict[str, Any]]

    def runtime_catalog_base(
        self,
        *,
        allow_inline_collect: bool = True,
        strict_collect: bool = False,
    ) -> dict[str, Any]:
        now_monotonic = time.monotonic()
        with self.runtime_catalog_cache_lock:
            cached_catalog = self.runtime_catalog_cache.get("catalog")
            refreshed_monotonic = float(
                self.runtime_catalog_cache.get("refreshed_monotonic", 0.0)
            )
            inbox_sync_snapshot = self.runtime_catalog_cache.get("inbox_sync_snapshot")

        cached_runtime_state = ""
        if isinstance(cached_catalog, dict):
            cached_runtime_state = (
                str(cached_catalog.get("runtime_state", "") or "").strip().lower()
            )
        cached_catalog_is_fallback = (
            isinstance(cached_catalog, dict) and cached_runtime_state == "fallback"
        )
        if strict_collect and cached_catalog_is_fallback:
            cached_catalog = None

        if not isinstance(cached_catalog, dict):
            if not allow_inline_collect and not strict_collect:
                with self.runtime_catalog_cache_lock:
                    recached_catalog = self.runtime_catalog_cache.get("catalog")
                    if isinstance(recached_catalog, dict):
                        recached_runtime_state = (
                            str(recached_catalog.get("runtime_state", "") or "")
                            .strip()
                            .lower()
                        )
                        if not (
                            strict_collect and recached_runtime_state == "fallback"
                        ):
                            return dict(recached_catalog)
                    inbox_sync_snapshot = self.runtime_catalog_cache.get(
                        "inbox_sync_snapshot"
                    )

                fallback_catalog = self.runtime_catalog_fallback(
                    self.part_root,
                    self.vault_root,
                )
                if isinstance(inbox_sync_snapshot, dict):
                    fallback_catalog["eta_mu_inbox"] = dict(inbox_sync_snapshot)
                with self.runtime_catalog_cache_lock:
                    self.runtime_catalog_cache["catalog"] = dict(fallback_catalog)
                    self.runtime_catalog_cache["refreshed_monotonic"] = time.monotonic()
                    self.runtime_catalog_cache["last_error"] = (
                        "catalog_bootstrap_deferred"
                    )
                self.schedule_runtime_catalog_refresh()
                return dict(fallback_catalog)

            with self.runtime_catalog_collect_lock:
                with self.runtime_catalog_cache_lock:
                    recached_catalog = self.runtime_catalog_cache.get("catalog")
                    if isinstance(recached_catalog, dict):
                        recached_runtime_state = (
                            str(recached_catalog.get("runtime_state", "") or "")
                            .strip()
                            .lower()
                        )
                        if not (
                            strict_collect and recached_runtime_state == "fallback"
                        ):
                            return dict(recached_catalog)
                    inbox_sync_snapshot = self.runtime_catalog_cache.get(
                        "inbox_sync_snapshot"
                    )

                fresh_catalog, isolated_error = self.collect_runtime_catalog_isolated(
                    self.part_root,
                    self.vault_root,
                )
                try:
                    if fresh_catalog is None and allow_inline_collect:
                        fresh_catalog = self.collect_catalog_fast()
                    if fresh_catalog is None and strict_collect:
                        raise RuntimeError(
                            isolated_error or "catalog_collect_unavailable"
                        )
                    if fresh_catalog is None:
                        raise RuntimeError(
                            isolated_error or "catalog_collect_unavailable"
                        )
                    cache_error = isolated_error
                    if cache_error == "catalog_subprocess_disabled":
                        cache_error = ""
                    if isinstance(inbox_sync_snapshot, dict):
                        fresh_catalog["eta_mu_inbox"] = dict(inbox_sync_snapshot)
                    with self.runtime_catalog_cache_lock:
                        self.runtime_catalog_cache["catalog"] = fresh_catalog
                        self.runtime_catalog_cache["refreshed_monotonic"] = (
                            time.monotonic()
                        )
                        self.runtime_catalog_cache["last_error"] = cache_error
                    if self.runtime_eta_mu_sync_enabled:
                        self.schedule_runtime_inbox_sync()
                    return dict(fresh_catalog)
                except Exception as exc:
                    with self.runtime_catalog_cache_lock:
                        self.runtime_catalog_cache["last_error"] = (
                            f"catalog_inline_failed:{exc.__class__.__name__}"
                        )
                    if strict_collect:
                        raise

        cache_age = now_monotonic - refreshed_monotonic
        if cache_age >= self.runtime_catalog_cache_seconds:
            self.schedule_runtime_catalog_refresh()

        if isinstance(cached_catalog, dict):
            return dict(cached_catalog)
        fallback_catalog = self.runtime_catalog_fallback(
            self.part_root, self.vault_root
        )
        if isinstance(inbox_sync_snapshot, dict):
            fallback_catalog["eta_mu_inbox"] = dict(inbox_sync_snapshot)
        with self.runtime_catalog_cache_lock:
            self.runtime_catalog_cache["catalog"] = dict(fallback_catalog)
            self.runtime_catalog_cache["refreshed_monotonic"] = time.monotonic()
        return fallback_catalog

    def runtime_catalog(
        self,
        *,
        perspective: str,
        include_projection: bool = True,
        include_runtime_fields: bool = True,
        allow_inline_collect: bool = True,
        strict_collect: bool = False,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        catalog = self.runtime_catalog_base(
            allow_inline_collect=allow_inline_collect,
            strict_collect=strict_collect,
        )
        queue_snapshot = self.task_queue.snapshot(include_pending=False)
        council_snapshot = self.council_chamber.snapshot(include_decisions=False)
        catalog["task_queue"] = queue_snapshot
        catalog["council"] = council_snapshot

        resource_snapshot: dict[str, Any] = {}
        influence_snapshot: dict[str, Any] = {}
        if include_runtime_fields or include_projection:
            resource_snapshot = self.resource_monitor_snapshot(part_root=self.part_root)
            self.influence_tracker.record_resource_heartbeat(
                resource_snapshot,
                source="runtime.catalog",
            )
            influence_snapshot = self.influence_tracker.snapshot(
                queue_snapshot=queue_snapshot,
                part_root=self.part_root,
            )
        if include_runtime_fields:
            catalog["presence_runtime"] = influence_snapshot
            catalog["muse_runtime"] = self.muse_runtime_snapshot()

        if include_projection:
            self.attach_ui_projection(
                catalog,
                perspective=perspective,
                queue_snapshot=queue_snapshot,
                influence_snapshot=influence_snapshot,
            )
        return (
            catalog,
            queue_snapshot,
            council_snapshot,
            influence_snapshot,
            resource_snapshot,
        )

    def runtime_simulation(
        self,
        catalog: dict[str, Any],
        queue_snapshot: dict[str, Any],
        influence_snapshot: dict[str, Any],
        *,
        perspective: str,
        include_unified_graph: bool = True,
        include_particle_dynamics: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        simulation_catalog = self.simulation_http_trim_catalog(catalog)
        try:
            myth_summary = self.myth_tracker.snapshot(simulation_catalog)
        except Exception:
            myth_summary = {}
        try:
            world_summary = self.life_tracker.snapshot(
                simulation_catalog,
                myth_summary,
                self.entity_manifest,
            )
        except Exception:
            world_summary = {}

        docker_snapshot = self.collect_docker_simulation_snapshot()
        simulation = self.build_simulation_state(
            simulation_catalog,
            myth_summary,
            world_summary,
            influence_snapshot=influence_snapshot,
            queue_snapshot=queue_snapshot,
            docker_snapshot=docker_snapshot,
            include_unified_graph=include_unified_graph,
            include_particle_dynamics=include_particle_dynamics,
        )

        def _graph_node_count(graph_payload: Any) -> int:
            if not isinstance(graph_payload, dict):
                return 0
            for key in ("file_nodes", "crawler_nodes", "nodes"):
                rows = graph_payload.get(key)
                if isinstance(rows, list) and rows:
                    return len(rows)
            return 0

        if not include_unified_graph:
            catalog_file_graph = simulation_catalog.get("file_graph")
            catalog_crawler_graph = simulation_catalog.get("crawler_graph")
            if (
                _graph_node_count(simulation.get("file_graph")) <= 0
                and _graph_node_count(catalog_file_graph) > 0
            ):
                simulation["file_graph"] = copy.deepcopy(catalog_file_graph)
            if (
                _graph_node_count(simulation.get("crawler_graph")) <= 0
                and _graph_node_count(catalog_crawler_graph) > 0
            ):
                simulation["crawler_graph"] = copy.deepcopy(catalog_crawler_graph)

        projection = self.build_ui_projection(
            simulation_catalog,
            simulation,
            perspective=perspective,
            queue_snapshot=queue_snapshot,
            influence_snapshot=influence_snapshot,
        )
        simulation["projection"] = projection
        simulation["perspective"] = perspective
        return simulation, projection
