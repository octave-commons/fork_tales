---
id: skill.health.telemetry.collect
type: skill
version: 1.0.0
tags: [health, telemetry, collect]
embedding_intent: canonical
---

# Health Telemetry Collection

**Intent**:
- Collect host/device health metrics at controlled cadence.
- Emit explicit capability and sampling-gap events.
- Use read-only collection paths.

**Operational Anchors**:
- Timestamp each sample with source and interval.
- Emit `telemetry_error` when data cannot be collected.
- Avoid synthetic or guessed values.

**Device Specifics**:
- **NVIDIA GPU**: Utilization, Memory, Temp, and Power via `nvidia-smi` (or containerized equivalent).
- **Intel GPU/NPU**: Frequency and Utilization via vendor-specific sysfs or drivers.
- **Docker Stats**: Direct retrieval of per-container resource metrics (`CPU%`, `MemoryUsage`, `PIDs`) from the Docker socket.

**Cadence Guidance**:
- Standard heartbeat: 2-5 seconds.
- Historical log sync: 60 seconds.
- High-pressure mode: Aggressive sampling to identify transient spikes.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.health.telemetry.collect)
  (domain health)
  (operational-cadence continuous)
  (provenance canonical-part64))
```
