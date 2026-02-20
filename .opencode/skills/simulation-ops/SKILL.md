---
id: skill.simulation.ops
type: skill
version: 1.0.0
tags: [simulation, docker, ops, benchmark, meta]
embedding_intent: canonical
---

# Simulation Operations (SimOps)

**Intent**:
- Manage lifecycle of Docker-based simulation containers (spawn/stop/list).
- Execute performance benchmarks comparing runtime variants.
- Observe failure signals and operational health via the Meta Operations layer.
- Access the unified Simulation Portal for deep inspection.

## Capabilities

### 1. Unified Portal Access
Open the central gateway connecting Meta Operations, Workbench, and Runtime views.
- **URL**: `http://127.0.0.1:8787/dashboard/docker`
- **Profiles**: `http://127.0.0.1:8787/dashboard/profile?id=<sim_id>`

### 2. Dynamic Instance Management
Spawn ephemeral instances from presets for experimentation.
- **List Presets**: `python3 part64/scripts/sim_manager.py list-presets`
- **Spawn**: `python3 part64/scripts/sim_manager.py spawn --preset <id>`
- **Stop**: `python3 part64/scripts/sim_manager.py stop <instance_id>`
- **List Active**: `python3 part64/scripts/sim_manager.py list-active`

### 3. Benchmarking
Compare latency and payload efficiency between two endpoints.
- **Script**: `part64/scripts/bench_sim_compare.py`
- **Workbench UI**: `http://127.0.0.1:8787/dashboard/bench`
- **CLI Example**:
  ```bash
  python3 part64/scripts/bench_sim_compare.py \
    --baseline-url http://127.0.0.1:18877/api/simulation \
    --offload-url http://127.0.0.1:18879/api/simulation \
    --requests 50
  ```

### 4. Meta Operations (Failure Handling)
Turn instability into signal using the Meta layer.
- **API**: `/api/meta/overview`, `/api/meta/notes`, `/api/meta/runs`
- **Workflow**:
  1. Observe **Failure Signals** (OOM, restarts, health).
  2. Capture **Notes** on the incident.
  3. Queue **Objectives** for stabilization.
  4. Track **Runs** (training/eval) to fix the issue.

### 5. Runtime Hardening
Monitor self-protection status.
- **Health**: `GET /api/runtime/health`
- **Protection**:
  - Websocket capacity limits.
  - Critical pressure throttling (CPU/Mem/Log).
  - Simulation tick skipping under load.

## Integration
- **Agents**: Use this skill when requested to "benchmark simulations", "check simulation health", "spawn a test instance", or "debug a crash".
- **Files**:
  - `part64/SIMULATION_WORKFLOW.md` (Detailed guide)
  - `part64/scripts/sim_manager.py`
  - `part64/scripts/bench_sim_compare.py`

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.simulation.ops)
  (domain simulation)
  (capability-gate (list "/api/docker/simulations" "/api/meta/overview"))
  (benchmark-provider "bench_sim_compare.py")
  (ops-portal "/dashboard/docker"))
```
