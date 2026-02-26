import pytest

from code.world_web.governor import LaneType, Packet, TickGovernor


def noop() -> None:
    return None


@pytest.fixture
def governor() -> TickGovernor:
    return TickGovernor()


def test_budget_subtracts_overhead(governor: TickGovernor) -> None:
    packet = Packet(
        id="required-1",
        work=60.0,
        value=100.0,
        deadline="tick",
        executable=noop,
        lane_efficiency={LaneType.CPU: 1.0},
    )

    result = governor.tick([packet], dt_ms=100.0, overhead_ms=12.5)

    assert result.budget_ms == pytest.approx(87.5, abs=1.0)
    assert result.overhead_ms >= 12.5
    assert result.slack_ms <= result.budget_ms
    assert result.required_executed == 1
    assert result.required_deferred == 0


def test_required_packet_defers_when_budget_is_exhausted(
    governor: TickGovernor,
) -> None:
    calls = {"ran": 0}

    def run() -> None:
        calls["ran"] += 1

    packet = Packet(
        id="required-defer",
        work=2200.0,
        value=100.0,
        deadline="tick",
        executable=run,
        lane_efficiency={LaneType.CPU: 1.0},
        family="simulation",
    )

    result = governor.tick([packet], dt_ms=24.0)

    assert calls["ran"] == 0
    assert result.required_executed == 0
    assert result.required_deferred == 1
    assert result.deadline_miss
    packet_ids = {str(row.get("packet_id", "")) for row in result.receipts}
    assert "required-defer" not in packet_ids


def test_required_packet_uses_degraded_variant_to_fit_budget(
    governor: TickGovernor,
) -> None:
    calls = {"full": 0, "degraded": 0}

    def run_full() -> None:
        calls["full"] += 1

    def run_degraded() -> None:
        calls["degraded"] += 1

    packet = Packet(
        id="required-degraded",
        work=500.0,
        value=100.0,
        deadline="tick",
        executable=run_full,
        lane_efficiency={LaneType.CPU: 1.0},
        family="simulation",
        allow_degrade=True,
        degraded_executable=run_degraded,
        degrade_work_factor=0.3,
        degrade_value_factor=0.9,
    )

    result = governor.tick([packet], dt_ms=45.0)

    assert calls["full"] == 0
    assert calls["degraded"] == 1
    assert result.required_executed == 1
    assert result.required_deferred == 0
    assert result.required_downgraded == 1
    assert not result.deadline_miss
    receipt = next(
        row
        for row in result.receipts
        if str(row.get("packet_id", "")) == "required-degraded"
    )
    assert str(receipt.get("mode", "")) == "degraded"


def test_route_prefers_lane_with_lower_shadow_cost(governor: TickGovernor) -> None:
    governor.lanes[LaneType.CPU].lambda_val = 12.0
    governor.lanes[LaneType.RTX].lambda_val = 0.0

    packet = Packet(
        id="route-1",
        work=200.0,
        value=20.0,
        deadline="tick",
        executable=noop,
        lane_efficiency={
            LaneType.CPU: 1.0,
            LaneType.RTX: 1.0,
        },
        family="simulation",
    )

    lane_id, _, _ = governor._route_packet(packet)
    assert lane_id == LaneType.RTX


def test_ingestion_pressure_reserves_accelerators(governor: TickGovernor) -> None:
    governor.update_ingestion_status(
        queue_depth=9000,
        bytes_pending=900 * 1024 * 1024,
        embedding_backlog=5000,
        disk_queue_depth=900,
    )
    governor.tick([], dt_ms=40.0)

    packet = Packet(
        id="sim-lane-choice",
        work=600.0,
        value=12.0,
        deadline="tick",
        executable=noop,
        lane_efficiency={
            LaneType.CPU: 1.0,
            LaneType.RTX: 1.0,
            LaneType.ARC: 1.0,
            LaneType.NPU: 1.0,
        },
        family="simulation",
    )

    lane_id, _, _ = governor._route_packet(packet)
    assert lane_id == LaneType.CPU


def test_filler_only_runs_with_slack(governor: TickGovernor) -> None:
    filler_packet = Packet(
        id="filler-big",
        work=2500.0,
        value=5.0,
        deadline="best-effort",
        executable=noop,
        lane_efficiency={LaneType.CPU: 1.0},
        family="filler",
    )

    first = governor.tick([filler_packet], dt_ms=5.0)
    assert len(first.receipts) == 0
    assert len(governor.filler_queue) == 1

    second = governor.tick([], dt_ms=700.0)
    packet_ids = {str(row.get("packet_id", "")) for row in second.receipts}
    assert "filler-big" in packet_ids
    assert len(governor.filler_queue) == 0


def test_filler_queue_is_bounded(governor: TickGovernor) -> None:
    packets = [
        Packet(
            id=f"fill-{index}",
            work=2000.0,
            value=float(index),
            deadline="best-effort",
            executable=noop,
            lane_efficiency={LaneType.CPU: 1.0},
            family="filler",
        )
        for index in range(governor.max_filler_size + 40)
    ]

    governor.tick(packets, dt_ms=2.0)
    assert len(governor.filler_queue) <= governor.max_filler_size


def test_deadline_miss_raises_lane_lambda(governor: TickGovernor) -> None:
    before = governor.lanes[LaneType.CPU].lambda_val
    packet = Packet(
        id="required-too-large",
        work=5000.0,
        value=100.0,
        deadline="tick",
        executable=noop,
        lane_efficiency={LaneType.CPU: 1.0},
        family="simulation",
    )

    result = governor.tick([packet], dt_ms=40.0)

    assert result.deadline_miss
    assert governor.lanes[LaneType.CPU].lambda_val > before


def test_memory_pressure_slope_raises_mem_lambda(governor: TickGovernor) -> None:
    governor.update_stock_pressure(mem_pressure=0.45, disk_pressure=0.1)
    governor.tick([], dt_ms=30.0)
    low_mem_lambda = governor.stock["mem"].lambda_val

    governor.update_stock_pressure(mem_pressure=0.95, disk_pressure=0.1)
    governor.tick([], dt_ms=30.0)

    assert governor.stock["mem"].lambda_val > low_mem_lambda
