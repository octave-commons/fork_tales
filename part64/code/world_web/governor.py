from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Deque, Dict, Iterable, Literal

TICK_MS = 83.33
TICK_SEC = TICK_MS / 1000.0

PacketDeadline = Literal["tick", "best-effort"]
FidelitySignal = Literal["increase", "decrease", "hold"]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


class LaneType:
    CPU = "cpu"
    RTX = "rtx"
    ARC = "arc"
    NPU = "npu"


LANE_ORDER: tuple[str, ...] = (
    LaneType.CPU,
    LaneType.RTX,
    LaneType.ARC,
    LaneType.NPU,
)


@dataclass
class Packet:
    id: str
    work: float
    value: float
    deadline: PacketDeadline
    executable: Callable[[], Any]
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    lane_efficiency: Dict[str, float] = field(default_factory=dict)
    cost_vector: Dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    family: str = "other"
    allow_degrade: bool = False
    degraded_executable: Callable[[], Any] | None = None
    degrade_work_factor: float = 0.6
    degrade_value_factor: float = 0.92

    def resource_cost(self, key: str) -> float:
        if key == "mem":
            if "mem" in self.cost_vector:
                return max(0.0, _safe_float(self.cost_vector.get("mem"), 0.0))
            return max(0.0, _safe_float(self.memory_usage, 0.0))
        if key == "disk":
            if "disk" in self.cost_vector:
                return max(0.0, _safe_float(self.cost_vector.get("disk"), 0.0))
            return max(0.0, _safe_float(self.disk_usage, 0.0))
        return max(0.0, _safe_float(self.cost_vector.get(key, 0.0), 0.0))

    def lane_factor(self, lane_id: str) -> float:
        return max(0.0, _safe_float(self.lane_efficiency.get(lane_id, 0.0), 0.0))

    @property
    def is_simulation(self) -> bool:
        if self.family == "simulation":
            return True
        for tag in self.tags:
            clean = str(tag).strip().lower()
            if "sim" in clean or clean == "tick":
                return True
        return False


@dataclass
class LaneState:
    capacity: float
    overhead_ms: float
    queue_work: float = 0.0
    lambda_val: float = 0.0
    last_utilization: float = 0.0
    last_queue_delay_ms: float = 0.0


@dataclass
class StockState:
    pressure: float = 0.0
    lambda_val: float = 0.0
    last_pressure: float = 0.0


@dataclass
class TickResult:
    receipts: list[dict[str, Any]]
    slack_ms: float
    budget_ms: float
    overhead_ms: float
    deadline_miss: bool
    fidelity_signal: FidelitySignal
    ingestion_pressure: float
    control_plane_ms: float
    required_executed: int
    required_deferred: int
    required_downgraded: int


class TickGovernor:
    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.lanes: dict[str, LaneState] = {
            LaneType.CPU: LaneState(capacity=5000.0, overhead_ms=0.05),
            LaneType.RTX: LaneState(capacity=45000.0, overhead_ms=0.25),
            LaneType.ARC: LaneState(capacity=28000.0, overhead_ms=0.22),
            LaneType.NPU: LaneState(capacity=18000.0, overhead_ms=0.35),
        }

        self.stock: dict[str, StockState] = {
            "mem": StockState(),
            "disk": StockState(),
        }

        self.ingestion: dict[str, float] = {
            "queue_depth": 0.0,
            "bytes_pending": 0.0,
            "embedding_backlog": 0.0,
            "disk_queue_depth": 0.0,
            "pressure": 0.0,
        }

        self.control: dict[str, float] = {
            "slack_ms_ema": 0.0,
            "last_tick_overhead_ms": 0.0,
            "last_budget_ms": TICK_MS,
            "last_control_plane_ms": 0.0,
            "last_deadline_miss": 0.0,
            "last_required_deferred": 0.0,
            "last_required_downgraded": 0.0,
        }

        self._filler_queue: Deque[Packet] = deque()
        self.max_filler_size = 256
        self.max_filler_packets_per_tick = 32
        self.max_packets_per_tick = 1024
        self.control_plane_limit_ratio = 0.2
        self.target_lane_utilization = 0.9
        self.lambda_floor = 0.0
        self.lambda_ceiling = 24.0

        self.p_I = 0.0
        self._tick_counter = 0
        self._receipt_counter = 0

    @property
    def filler_queue(self) -> list[Packet]:
        with self._lock:
            return list(self._filler_queue)

    def update_ingestion_status(
        self,
        queue_depth: int,
        bytes_pending: int,
        *,
        embedding_backlog: int = 0,
        disk_queue_depth: int = 0,
    ) -> None:
        with self._lock:
            queue_value = max(0.0, _safe_float(queue_depth, 0.0))
            bytes_value = max(0.0, _safe_float(bytes_pending, 0.0))
            embed_value = max(0.0, _safe_float(embedding_backlog, 0.0))
            disk_value = max(0.0, _safe_float(disk_queue_depth, 0.0))

            self.ingestion["queue_depth"] = queue_value
            self.ingestion["bytes_pending"] = bytes_value
            self.ingestion["embedding_backlog"] = embed_value
            self.ingestion["disk_queue_depth"] = disk_value

            queue_norm = _clamp(queue_value / 1200.0, 0.0, 1.0)
            bytes_norm = _clamp(bytes_value / (256.0 * 1024.0 * 1024.0), 0.0, 1.0)
            embed_norm = _clamp(embed_value / 2400.0, 0.0, 1.0)
            disk_norm = _clamp(disk_value / 320.0, 0.0, 1.0)
            self.p_I = max(queue_norm, bytes_norm, embed_norm, disk_norm)
            self.ingestion["pressure"] = self.p_I

    def update_stock_pressure(
        self,
        *,
        mem_pressure: float | None = None,
        disk_pressure: float | None = None,
    ) -> None:
        with self._lock:
            if mem_pressure is not None:
                self.stock["mem"].pressure = _clamp(
                    _safe_float(mem_pressure, 0.0),
                    0.0,
                    1.5,
                )
            if disk_pressure is not None:
                self.stock["disk"].pressure = _clamp(
                    _safe_float(disk_pressure, 0.0),
                    0.0,
                    1.5,
                )

    def tick(
        self,
        packets: Iterable[Packet],
        *,
        dt_ms: float = TICK_MS,
        overhead_ms: float = 0.0,
    ) -> TickResult:
        with self._lock:
            self._tick_counter += 1
            tick_id = self._tick_counter
            tick_start = time.perf_counter()

            tick_window_ms = max(1.0, _safe_float(dt_ms, TICK_MS))
            control_limit_ms = max(0.2, tick_window_ms * self.control_plane_limit_ratio)

            packet_rows = [row for row in packets if isinstance(row, Packet)]
            if len(packet_rows) > self.max_packets_per_tick:
                packet_rows = packet_rows[: self.max_packets_per_tick]

            required: list[Packet] = []
            optional: list[Packet] = []
            for packet in packet_rows:
                if packet.deadline == "tick":
                    required.append(packet)
                else:
                    optional.append(packet)
            self._enqueue_filler(optional)

            packetization_ms = self._elapsed_ms(tick_start)
            unavoidable_overhead_ms = max(0.0, _safe_float(overhead_ms, 0.0))
            unavoidable_overhead_ms += packetization_ms
            budget_ms = max(0.0, tick_window_ms - unavoidable_overhead_ms)

            receipts: list[dict[str, Any]] = []
            lane_tick_ms = {lane_id: 0.0 for lane_id in self.lanes}

            predicted_spent_ms = 0.0
            required_misses = 0
            required_executed = 0
            required_deferred = 0
            required_downgraded = 0
            executed_wall_ms = 0.0

            required_sorted = sorted(required, key=self._packet_priority, reverse=True)
            for packet in required_sorted:
                if (
                    self._control_plane_elapsed_ms(
                        tick_start=tick_start,
                        executed_wall_ms=executed_wall_ms,
                    )
                    >= control_limit_ms
                ):
                    required_misses += 1
                    required_deferred += 1
                    continue

                remaining_budget_ms = max(0.0, budget_ms - predicted_spent_ms)
                dispatch_choice = self._choose_required_dispatch(
                    packet,
                    remaining_budget_ms=remaining_budget_ms,
                )
                if dispatch_choice is None:
                    required_misses += 1
                    required_deferred += 1
                    continue

                dispatch_packet, lane_id, predicted_ms, dispatch_mode = dispatch_choice
                receipt = self._dispatch(
                    dispatch_packet,
                    lane_id,
                    predicted_ms=predicted_ms,
                    tick_id=tick_id,
                    dispatch_mode=dispatch_mode,
                    source_packet_id=packet.id,
                )
                receipts.append(receipt)
                lane_tick_ms[lane_id] += predicted_ms
                predicted_spent_ms += predicted_ms
                executed_wall_ms += max(
                    0.0,
                    _safe_float(receipt.get("exec_ms", 0.0), 0.0),
                )
                required_executed += 1
                if dispatch_mode == "degraded":
                    required_downgraded += 1

            remaining_budget_ms = max(0.0, budget_ms - predicted_spent_ms)
            (
                filler_receipts,
                filler_lane_ms,
                filler_spent_ms,
                executed_wall_ms,
            ) = self._drain_filler(
                remaining_budget_ms=remaining_budget_ms,
                tick_id=tick_id,
                tick_start=tick_start,
                control_limit_ms=control_limit_ms,
                executed_wall_ms=executed_wall_ms,
            )
            receipts.extend(filler_receipts)
            for lane_id, duration_ms in filler_lane_ms.items():
                lane_tick_ms[lane_id] += duration_ms
            predicted_spent_ms += filler_spent_ms

            tick_elapsed_ms = self._elapsed_ms(tick_start)
            control_plane_ms = max(0.0, tick_elapsed_ms - executed_wall_ms)

            model_slack_ms = budget_ms - predicted_spent_ms
            wall_slack_ms = tick_window_ms - tick_elapsed_ms
            slack_ms = min(model_slack_ms, wall_slack_ms)
            deadline_miss = bool(required_misses > 0 or slack_ms < 0.0)

            self._update_shadow_prices(
                dt_ms=tick_window_ms,
                lane_tick_ms=lane_tick_ms,
                required_misses=required_misses,
                deadline_miss=deadline_miss,
            )

            alpha = 0.2
            previous_ema = _safe_float(self.control.get("slack_ms_ema", 0.0), 0.0)
            slack_ema = (alpha * slack_ms) + ((1.0 - alpha) * previous_ema)
            self.control["slack_ms_ema"] = slack_ema
            self.control["last_tick_overhead_ms"] = unavoidable_overhead_ms
            self.control["last_budget_ms"] = budget_ms
            self.control["last_control_plane_ms"] = control_plane_ms
            self.control["last_deadline_miss"] = 1.0 if deadline_miss else 0.0
            self.control["last_required_deferred"] = float(required_deferred)
            self.control["last_required_downgraded"] = float(required_downgraded)

            fidelity_signal = self._fidelity_signal(
                dt_ms=tick_window_ms,
                slack_ema=slack_ema,
                deadline_miss=deadline_miss,
            )

            return TickResult(
                receipts=receipts,
                slack_ms=slack_ms,
                budget_ms=budget_ms,
                overhead_ms=unavoidable_overhead_ms,
                deadline_miss=deadline_miss,
                fidelity_signal=fidelity_signal,
                ingestion_pressure=self.p_I,
                control_plane_ms=control_plane_ms,
                required_executed=required_executed,
                required_deferred=required_deferred,
                required_downgraded=required_downgraded,
            )

    def _elapsed_ms(self, started: float) -> float:
        return (time.perf_counter() - started) * 1000.0

    def _control_plane_elapsed_ms(
        self, *, tick_start: float, executed_wall_ms: float
    ) -> float:
        return max(0.0, self._elapsed_ms(tick_start) - max(0.0, executed_wall_ms))

    def _queue_delay_ms(self, lane_id: str) -> float:
        lane = self.lanes[lane_id]
        if lane.capacity <= 0.0:
            return float("inf")
        return max(0.0, (lane.queue_work / lane.capacity) * 1000.0)

    def _predict_duration_ms(self, packet: Packet, lane_id: str) -> float:
        lane = self.lanes[lane_id]
        factor = packet.lane_factor(lane_id)
        if factor <= 0.0:
            return float("inf")

        effective_capacity = lane.capacity * factor
        if effective_capacity <= 0.0:
            return float("inf")

        execution_ms = (max(0.0, packet.work) / effective_capacity) * 1000.0
        return lane.overhead_ms + execution_ms + self._queue_delay_ms(lane_id)

    def _ingestion_reservation_cost(self, packet: Packet, lane_id: str) -> float:
        if self.p_I <= 0.0 or not packet.is_simulation:
            return 0.0
        if lane_id == LaneType.CPU:
            return 0.0

        multipliers = {
            LaneType.RTX: 0.055,
            LaneType.ARC: 0.045,
            LaneType.NPU: 0.08,
        }
        multiplier = multipliers.get(lane_id, 0.0)
        return max(0.0, packet.work) * self.p_I * multiplier

    def _score_packet_lane(
        self, packet: Packet, lane_id: str, predicted_ms: float
    ) -> float:
        lane = self.lanes[lane_id]
        lane_seconds = max(0.0, predicted_ms / 1000.0)
        lane_cost = lane.lambda_val * lane_seconds
        lane_cost += packet.resource_cost(lane_id)

        mem_cost = self.stock["mem"].lambda_val * _clamp(
            packet.resource_cost("mem"), 0.0, 1.0
        )
        disk_cost = self.stock["disk"].lambda_val * _clamp(
            packet.resource_cost("disk"),
            0.0,
            1.0,
        )

        reservation_cost = self._ingestion_reservation_cost(packet, lane_id)
        return packet.value - (lane_cost + mem_cost + disk_cost + reservation_cost)

    def _route_packet(self, packet: Packet) -> tuple[str | None, float, float]:
        best_lane: str | None = None
        best_pred_ms = float("inf")
        best_score = -float("inf")

        for lane_id in LANE_ORDER:
            predicted_ms = self._predict_duration_ms(packet, lane_id)
            if math.isinf(predicted_ms):
                continue

            score = self._score_packet_lane(packet, lane_id, predicted_ms)
            if score > best_score + 1e-9:
                best_lane = lane_id
                best_pred_ms = predicted_ms
                best_score = score
                continue
            if abs(score - best_score) <= 1e-9 and predicted_ms < best_pred_ms:
                best_lane = lane_id
                best_pred_ms = predicted_ms
                best_score = score

        return best_lane, best_pred_ms, best_score

    def _best_packet_score(self, packet: Packet) -> float:
        _, _, score = self._route_packet(packet)
        return score

    def _packet_priority(self, packet: Packet) -> float:
        base = self._best_packet_score(packet)
        if packet.deadline == "tick":
            return base + 25.0
        return base

    def _build_degraded_packet(self, packet: Packet) -> Packet | None:
        if not packet.allow_degrade:
            return None

        factor = _clamp(_safe_float(packet.degrade_work_factor, 0.6), 0.1, 1.0)
        value_factor = _clamp(_safe_float(packet.degrade_value_factor, 0.92), 0.1, 1.0)
        executable = packet.degraded_executable or packet.executable

        if factor >= 0.999 and packet.degraded_executable is None:
            return None

        scaled_cost_vector = {
            key: _clamp(_safe_float(value, 0.0) * factor, 0.0, 1.0)
            for key, value in packet.cost_vector.items()
        }
        tags = list(packet.tags)
        if "governor:degraded" not in tags:
            tags.append("governor:degraded")

        degraded = replace(
            packet,
            work=max(1.0, max(0.0, packet.work) * factor),
            value=max(0.0, packet.value * value_factor),
            executable=executable,
            memory_usage=max(0.0, packet.memory_usage * factor),
            disk_usage=max(0.0, packet.disk_usage * factor),
            cost_vector=scaled_cost_vector,
            tags=tags,
        )
        return degraded

    def _choose_required_dispatch(
        self,
        packet: Packet,
        *,
        remaining_budget_ms: float,
    ) -> tuple[Packet, str, float, str] | None:
        if remaining_budget_ms <= 0.0:
            return None

        lane_id, predicted_ms, _ = self._route_packet(packet)
        if (
            lane_id is not None
            and not math.isinf(predicted_ms)
            and self._is_affordable(packet)
            and predicted_ms <= remaining_budget_ms
        ):
            return packet, lane_id, predicted_ms, "full"

        degraded = self._build_degraded_packet(packet)
        if degraded is None:
            return None

        degraded_lane, degraded_ms, _ = self._route_packet(degraded)
        if degraded_lane is None or math.isinf(degraded_ms):
            return None
        if not self._is_affordable(degraded):
            return None
        if degraded_ms > remaining_budget_ms:
            return None
        return degraded, degraded_lane, degraded_ms, "degraded"

    def _is_affordable(self, packet: Packet) -> bool:
        mem_pressure = self.stock["mem"].pressure + _clamp(
            packet.resource_cost("mem"), 0.0, 1.0
        )
        disk_pressure = self.stock["disk"].pressure + _clamp(
            packet.resource_cost("disk"),
            0.0,
            1.0,
        )
        return mem_pressure < 1.0 and disk_pressure < 1.0

    def _dispatch(
        self,
        packet: Packet,
        lane_id: str,
        *,
        predicted_ms: float,
        tick_id: int,
        dispatch_mode: str = "full",
        source_packet_id: str | None = None,
    ) -> dict[str, Any]:
        lane = self.lanes[lane_id]
        lane.queue_work += max(0.0, _safe_float(packet.work, 0.0))

        started = time.perf_counter()
        status = "ok"
        error = ""
        try:
            packet.executable()
        except Exception as exc:  # pragma: no cover - exercised via status path
            status = "error"
            error = f"{exc.__class__.__name__}:{exc}"[:160]
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        self._receipt_counter += 1
        packet_id = source_packet_id or packet.id
        receipt: dict[str, Any] = {
            "receipt_id": self._receipt_counter,
            "tick_id": tick_id,
            "packet_id": packet_id,
            "lane": lane_id,
            "deadline": packet.deadline,
            "mode": dispatch_mode,
            "status": status,
            "pred_ms": round(predicted_ms, 4),
            "exec_ms": round(elapsed_ms, 4),
        }
        if packet.id != packet_id:
            receipt["variant_packet_id"] = packet.id
        if error:
            receipt["error"] = error
        return receipt

    def _enqueue_filler(self, packets: list[Packet]) -> None:
        if not packets:
            return
        merged = list(self._filler_queue)
        merged.extend(packets)
        merged.sort(key=self._packet_priority, reverse=True)
        if len(merged) > self.max_filler_size:
            merged = merged[: self.max_filler_size]
        self._filler_queue = deque(merged)

    def _drain_filler(
        self,
        *,
        remaining_budget_ms: float,
        tick_id: int,
        tick_start: float,
        control_limit_ms: float,
        executed_wall_ms: float,
    ) -> tuple[list[dict[str, Any]], dict[str, float], float, float]:
        if remaining_budget_ms <= 0.0 or not self._filler_queue:
            return [], {lane_id: 0.0 for lane_id in self.lanes}, 0.0, executed_wall_ms

        candidates = sorted(
            list(self._filler_queue), key=self._packet_priority, reverse=True
        )
        self._filler_queue.clear()

        receipts: list[dict[str, Any]] = []
        lane_tick_ms = {lane_id: 0.0 for lane_id in self.lanes}
        spent_ms = 0.0
        kept: list[Packet] = []

        admitted = 0
        for packet in candidates:
            if admitted >= self.max_filler_packets_per_tick:
                kept.append(packet)
                continue
            if (
                self._control_plane_elapsed_ms(
                    tick_start=tick_start,
                    executed_wall_ms=executed_wall_ms,
                )
                >= control_limit_ms
            ):
                kept.append(packet)
                continue

            lane_id, predicted_ms, _ = self._route_packet(packet)
            if lane_id is None or math.isinf(predicted_ms):
                continue
            if not self._is_affordable(packet):
                kept.append(packet)
                continue
            if predicted_ms > remaining_budget_ms:
                kept.append(packet)
                continue

            receipt = self._dispatch(
                packet,
                lane_id,
                predicted_ms=predicted_ms,
                tick_id=tick_id,
                dispatch_mode="filler",
            )
            receipts.append(receipt)
            lane_tick_ms[lane_id] += predicted_ms
            spent_ms += predicted_ms
            remaining_budget_ms -= predicted_ms
            executed_wall_ms += max(
                0.0,
                _safe_float(receipt.get("exec_ms", 0.0), 0.0),
            )
            admitted += 1

        if kept:
            kept.sort(key=self._packet_priority, reverse=True)
            if len(kept) > self.max_filler_size:
                kept = kept[: self.max_filler_size]
            self._filler_queue = deque(kept)

        return receipts, lane_tick_ms, spent_ms, executed_wall_ms

    def _bump_lane_lambda(self, lane_id: str, delta: float) -> None:
        state = self.lanes[lane_id]
        state.lambda_val = _clamp(
            state.lambda_val + delta,
            self.lambda_floor,
            self.lambda_ceiling,
        )

    def _update_shadow_prices(
        self,
        *,
        dt_ms: float,
        lane_tick_ms: dict[str, float],
        required_misses: int,
        deadline_miss: bool,
    ) -> None:
        dt_seconds = max(0.001, dt_ms / 1000.0)

        for lane_id, lane in self.lanes.items():
            lane_time_ms = max(0.0, _safe_float(lane_tick_ms.get(lane_id, 0.0), 0.0))
            utilization = _clamp(lane_time_ms / max(1.0, dt_ms), 0.0, 2.0)
            queue_capacity = lane.capacity * dt_seconds
            queue_ratio = lane.queue_work / max(1e-6, queue_capacity)

            delta = 0.18 * (utilization - self.target_lane_utilization)
            delta += 0.08 * max(0.0, queue_ratio - 1.0)
            if deadline_miss:
                delta += 0.08
            if required_misses > 0:
                delta += min(0.6, 0.04 * required_misses)

            lane.lambda_val = _clamp(
                lane.lambda_val + delta,
                self.lambda_floor,
                self.lambda_ceiling,
            )

            lane.last_utilization = utilization

            drained_work = lane.capacity * dt_seconds
            lane.queue_work = max(0.0, lane.queue_work - drained_work)
            lane.last_queue_delay_ms = self._queue_delay_ms(lane_id)

        if deadline_miss:
            culprit_lane = max(
                LANE_ORDER,
                key=lambda lane_id: _safe_float(lane_tick_ms.get(lane_id, 0.0), 0.0),
            )
            self._bump_lane_lambda(culprit_lane, 0.75)
            for lane_id in LANE_ORDER:
                self._bump_lane_lambda(lane_id, 0.14)

        mem_state = self.stock["mem"]
        disk_state = self.stock["disk"]

        mem_slope = mem_state.pressure - mem_state.last_pressure
        mem_delta = 0.36 * (mem_state.pressure - 0.8)
        if mem_slope > 0.02:
            mem_delta += min(3.0, mem_slope * 18.0)
        if deadline_miss and mem_state.pressure > 0.8:
            mem_delta += 0.2
        mem_state.lambda_val = _clamp(
            mem_state.lambda_val + mem_delta,
            self.lambda_floor,
            self.lambda_ceiling,
        )
        mem_state.last_pressure = mem_state.pressure

        disk_slope = disk_state.pressure - disk_state.last_pressure
        disk_delta = 0.28 * (disk_state.pressure - 0.8)
        if disk_slope > 0.02:
            disk_delta += min(2.0, disk_slope * 10.0)
        if deadline_miss and disk_state.pressure > 0.8:
            disk_delta += 0.16
        disk_state.lambda_val = _clamp(
            disk_state.lambda_val + disk_delta,
            self.lambda_floor,
            self.lambda_ceiling,
        )
        disk_state.last_pressure = disk_state.pressure

        if self.p_I > 0.5:
            self._bump_lane_lambda(LaneType.NPU, 1.1 * self.p_I)
            self._bump_lane_lambda(LaneType.RTX, 0.7 * self.p_I)
            self._bump_lane_lambda(LaneType.ARC, 0.55 * self.p_I)

    def _fidelity_signal(
        self,
        *,
        dt_ms: float,
        slack_ema: float,
        deadline_miss: bool,
    ) -> FidelitySignal:
        if deadline_miss:
            return "decrease"
        if self.p_I >= 0.65:
            return "decrease"
        if slack_ema < (0.04 * dt_ms):
            return "decrease"
        if self.p_I <= 0.25 and slack_ema > (0.18 * dt_ms):
            return "increase"
        return "hold"


_GOVERNOR = TickGovernor()


def get_governor() -> TickGovernor:
    return _GOVERNOR
