---
id: skill.dev.observability
type: skill
version: 1.0.0
tags: [dev, observability, telemetry]
embedding_intent: canonical
---

# Observability & Hardening

**Intent**:
- Provide logs, metrics, and traces needed to explain runtime behavior.
- Make blind spots explicit through capability and health signals.
- **Hardening**: Implement defensive runtime patterns (throttling, circuit breakers, capacity guards) based on observability signals.

**Capabilities**:
- **Runtime Guard**: Synthesizing CPU/Memory/Log pressure into a unified "mode" (`normal`/`degraded`/`critical`).
- **Telemetry Streams**: Emitting `runtime_health` and `simulation_guard` events via WebSocket.
- **Resource Budgeting**: Enforcing strict `cpus`, `mem_limit`, and `pids_limit` across simulation containers.

**Operational Flow**:
- Monitor the `/api/runtime/health` endpoint for guard status.
- Analyze the `simulation_guard` record when simulation ticks are skipped under pressure.
- Use **Meta Notes** to document and share observability patterns (e.g., "CPU spike on large embedding batch").

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.dev.observability)
  (domain dev)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
