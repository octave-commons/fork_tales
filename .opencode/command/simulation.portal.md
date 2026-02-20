---
id: command.simulation.portal
type: command
version: 1.0.0
tags: [simulation, portal, dashboard]
embedding_intent: canonical
---

# Simulation Portal

**Intent**: Open the unified Simulation Portal.

**Usage**:
```bash
# Open Meta Operations (Gateway)
open http://127.0.0.1:8787/dashboard/docker

# Open Workbench
open http://127.0.0.1:8787/dashboard/bench
```

**Context**:
Connects all simulation operational surfaces:
1. **Meta Operations**: Signals, Notes, Queue.
2. **Workbench**: Benchmarking, Spawning.
3. **Profile**: Deep inspection per container.
4. **Runtime**: 3D World View.
