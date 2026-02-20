# C Sim Slice Worker (Template)

This is a tiny first slice for moving simulation work into C microservices via Redis.

Current slice:
- input: `cpu_utilization`, `max_sim_points`
- output: `sim_point_budget`

Redis contract (request):
- stream: `SIM_SLICE_REDIS_JOBS_STREAM` (default `eta_mu:sim_slice_jobs`)
- fields:
  - `slice=sim_point_budget.v1`
  - `job_id=<uuid>`
  - `cpu_utilization=<float>`
  - `max_sim_points=<int>`
  - `reply_key=<redis-key>`
  - `submitted_ms=<epoch-ms>`

Redis contract (response):
- key: `reply_key`
- value JSON:
  - `{"job_id":"...","sim_point_budget":612,"source":"c-budget-worker.v1"}`

## Build

Requires `hiredis` development headers/libs.

```bash
cc -O2 -Wall -Wextra -std=c11 code/world_web/sim_slice_budget_worker.c -lhiredis -o code/world_web/sim_slice_budget_worker
```

## Run

```bash
SIM_SLICE_REDIS_HOST=127.0.0.1 \
SIM_SLICE_REDIS_PORT=6379 \
SIM_SLICE_REDIS_JOBS_STREAM=eta_mu:sim_slice_jobs \
SIM_SLICE_WORKER_NAME=c-budget-worker.v1 \
code/world_web/sim_slice_budget_worker
```

Enable Python offload path:

```bash
SIM_SLICE_OFFLOAD_MODE=redis
```
