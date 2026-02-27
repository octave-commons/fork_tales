---
title: "Falsifiable Claims for Nexus System"
summary: "Defines testable claims for homeostasis, routing resilience, and auditable constraint enforcement."
category: "claims"
created_at: "2026-02-26T19:07:03"
original_filename: "2026.02.26.19.07.03.md"
original_relpath: "docs/notes/claims/2026.02.26.19.07.03.md"
tags:
  - claims
  - validation
  - simulation
---

# CLAIMS

*Drafted with the assistance of an AI.*

This document states the **minimum set of falsifiable claims** for the Fork Tales / Nexus system. Each claim is written to be testable in a simulator and to remain meaningful even if the story framing changes.

## Definitions (short)

* **Nexus Graph**: the canonical shared workspace graph (nodes/edges) representing resources, relationships, and state summaries.
* **Presence**: a persistent actor (process) with local priorities that can write/read the Nexus Graph and influence fields.
* **Daimoi**: messenger particles that move, interact, and are evaluated by outcomes (food/death).
* **Nooi Field**: a decaying **signal vector field** (trail field) deposited into by motion and optionally reinforced by outcomes.
* **Food**: a daimoi reaches a target by successfully colliding with a **Nexus** or **Presence** within a deadline.
* **Death**: a daimoi fails to reach a target within a deadline (timeout).
* **Homeostasis**: maintaining bounded compute/memory and bounded tick latency under variable load while preserving useful behavior.

---

## Claim 1 — Homeostasis under load without loss of purpose

**Claim:** Under variable load and bursty input, the system maintains a stable working set (bounded active graph size + bounded active particle count) **while preserving task performance**, without requiring a monolithic model.

**Operationally:** The Tick Governor + compaction + scheduling keep the sim running at a target tick rate, and the system continues to route daimoi toward successful collisions (food) rather than devolving into random drift.

**How to measure (sim harness):**

Run a city/infra scenario where the input event rate changes over time (low → surge → low) and measure:

1. **Tick stability**: median and p95 tick time; percent of ticks meeting the target budget.
2. **Working-set bounds**: max and p95 of active node count, active edge count, and active particle count.
3. **Throughput under stress**: food rate (successes per minute) during surge.
4. **Graceful degradation**: quality curve vs load (food rate declines smoothly, not cliff-falls).

Compare against at least one baseline:

* baseline A: same system with compaction disabled.
* baseline B: same system with governor disabled (or slack ignored).

**What would falsify it:**

* Tick time becomes unstable (runaway p95) or collapses under surge.
* Working set grows without bound or oscillates chaotically.
* Food rate drops to near-zero under load even though resources exist.
* Degradation is cliff-like (e.g., sudden failure) rather than gradual.

---

## Claim 2 — Trail field + semantic attraction yields faster, more resilient routing on a changing graph

**Claim:** The combined dynamics of (a) semantic attraction/repulsion and (b) the Nooi trail field produce **faster adaptation** to graph changes than either dynamic alone.

**Operationally:** When topology changes (edges blocked, nodes removed, capacities changed), daimoi find usable paths again quickly because the trail field carries learned directional bias and semantic attraction keeps motion coherent.

**How to measure (sim harness):**

Use scenarios with controlled graph perturbations:

* road/rail closure events,
* capacity reductions,
* partial comms degradation,
* added incidents (new high-priority attractors).

Track:

1. **Recovery time**: time from perturbation to returning to ≥X% of pre-perturbation food rate.
2. **Path churn**: how many distinct routes are used; whether hot paths stabilize.
3. **Exploration vs exploitation balance**: entropy/diversity of movement vs convergence.

Compare against ablations:

* ablation A: semantic attraction only (trail field off).
* ablation B: trail field only (semantic attraction off).
* ablation C: neither (random drift with only boundary forces).

**What would falsify it:**

* Combined system does not recover faster than the ablations.
* Trail field causes lock-in to bad routes (recovery slower or fails).
* Semantic attraction causes brittle clustering (cannot reroute when topology changes).

---

## Claim 3 — A formal constraint layer makes decisions auditable and prevents unsafe/invalid actions

**Claim:** Adding a Prolog/Datalog constraint layer (LLM optional) makes the system’s decisions **auditable**, and enforces hard safety/feasibility constraints such that invalid plans are rejected consistently.

**Operationally:** Particles/exploration can propose candidate actions/routes, but the constraint layer acts as a deterministic judge that:

* rejects illegal actions (role/authority),
* rejects infeasible dispatches (capacity, unavailable resources),
* rejects forbidden communications (channel/policy),
* produces a queryable proof/explanation trail.

**How to measure (sim harness):**

Introduce deliberate “temptations” in scenarios:

* dispatch that would overcommit resources,
* comms that violate policy,
* routing through blocked segments,
* conflicting actions for a single responder.

Measure:

1. **Invalid action rejection rate**: should be ~100% for explicitly forbidden actions.
2. **False rejections**: rate of rejecting valid actions (should be low, explainable).
3. **Auditability**: ability to produce a machine-checkable explanation (which rule caused rejection).
4. **Latency overhead**: added time per decision step.

Compare:

* baseline A: heuristic rule checks embedded in code without a logic engine.
* baseline B: “LLM-only” plan generation without a judge (if applicable).

**What would falsify it:**

* Invalid actions routinely pass.
* The logic layer frequently rejects valid actions without a stable reason.
* Explanations are missing or non-deterministic.
* Overhead makes real-time operation impossible at target scale.

---

## Notes on scope and honesty

* These claims do **not** claim general intelligence.
* These claims are about **resilience**, **bounded computation**, and **coordination under uncertainty**.
* Story framing ("model collapse") is treated as a *stress-test narrative*, not required for the claims to be true.

## Minimal scenario set (recommended)

1. Chemical spill near a hospital (single incident, high severity).
2. Earthquake with partial infrastructure outages (roads/rails).
3. Multi-incident surge (competing priorities under scarcity).

Each scenario should be runnable with controlled seeds and produce a short metrics report.
