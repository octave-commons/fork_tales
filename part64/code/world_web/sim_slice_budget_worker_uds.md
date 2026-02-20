# C Sim Slice Worker (UDS)

This worker is a direct-IPC alternative to Redis for hot-path simulation slices.

Current slice:
- input: `cpu_utilization`, `max_sim_points`
- output: `sim_point_budget`

Protocol:
- transport: Unix domain socket (`AF_UNIX`, stream)
- request: single-line JSON ending with `\n`
  - `{"slice":"sim_point_budget.v1","job_id":"...","cpu_utilization":85.0,"max_sim_points":1000}`
- response: single-line JSON ending with `\n`
  - `{"job_id":"...","sim_point_budget":677,"source":"c-uds-worker.v1"}`

## Build

```bash
cc -O2 -Wall -Wextra -std=c11 code/world_web/sim_slice_budget_worker_uds.c -o code/world_web/sim_slice_budget_worker_uds
```

## Run

```bash
SIM_SLICE_UDS_PATH=/tmp/eta_mu_sim_slice.sock \
SIM_SLICE_WORKER_NAME=c-uds-worker.v1 \
code/world_web/sim_slice_budget_worker_uds
```

Enable Python offload path:

```bash
SIM_SLICE_OFFLOAD_MODE=uds
SIM_SLICE_ASYNC=1
SIM_SLICE_UDS_PATH=/tmp/eta_mu_sim_slice.sock
```
