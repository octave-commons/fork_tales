# Simulation Workflow Guide

This document explains how to create, manage, benchmark, and observe simulations using the new **Simulation Portal** and **Meta Operations** layer.

## 1. Accessing the Portal

The unified portal connects all operational views. Access the gateway at:

**[http://127.0.0.1:8787/dashboard/docker](http://127.0.0.1:8787/dashboard/docker)** (Meta Operations Dashboard)

From there, you can navigate using the top bar:
- **Meta Operations**: Failure signals, notes, objectives, and run tracking.
- **Workbench**: Benchmarking and dynamic instance spawning.
- **Runtime**: The main 3D world view.

## 2. Creating New Simulations

To add a persistent simulation service that appears in the dashboard, add it to your `docker-compose.yml` (or a specialized override file).

### Requirements for Auto-Discovery

1.  **Labels**: You must tag the service so the runtime discovers it.
    ```yaml
    labels:
      - "io.fork_tales.simulation=true"
      - "io.fork_tales.simulation.role=experiment" # or 'core', 'benchmark', etc.
    ```
2.  **Network**: Connect it to the shared simulation network.
    ```yaml
    networks:
      - eta-mu-sim-net
    ```
3.  **Ports (Optional)**: If you want to access its world API directly from the host, map port 8787 to a unique host port.
    ```yaml
    ports:
      - "19880:8787"
    ```

### Example Service Definition

```yaml
  eta-mu-experiment-alpha:
    image: fork-tales/eta-mu:latest
    container_name: eta-mu-experiment-alpha
    restart: unless-stopped
    labels:
      - "io.fork_tales.simulation=true"
      - "io.fork_tales.simulation.role=experiment"
    networks:
      - eta-mu-sim-net
    ports:
      - "19880:8787"
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 4096M
          pids: 120
    environment:
      - SIM_TICK_RATE=15
```

Once added, run `docker compose up -d` and refresh the **Meta Operations** dashboard. The new container will appear in the "Running Simulations" list.

## 3. Benchmarking & Dynamic Instances

Use the **Workbench** (`/dashboard/bench`) to compare performance side-by-side.

### Spawning Temporary Instances
You don't always need to edit `docker-compose.yml`. You can spawn ephemeral instances from presets:
1.  Go to **Workbench**.
2.  In the **Presets** sidebar, click **Launch** on a configuration (e.g., `standard-local` or `high-perf-cdb`).
3.  The system will spawn a container on an available port (range 18900+).
4.  These instances are visible in the "Active Instances" list and can be stopped with one click.

### Running a Benchmark
1.  Select a **Baseline Instance** (e.g., your stable `eta-mu-local`).
2.  Select an **Experiment Instance** (e.g., a new `offload-c-worker` spawn).
3.  Click **Start Comparison**.
4.  The tool runs a concurrent warmup followed by a latency/payload measurement series.
5.  Results appear in a table showing Mean/P95 latency, Payload size, and relative Efficiency.

## 4. Deep Inspection (Profile View)

Click any simulation name in the dashboard lists to open its **Profile**:
- **Live Resource Bars**: See exact CPU/Memory/PID pressure relative to limits.
- **Live Preview**: Interact with that specific world's 3D view in an embedded frame.
- **Configuration**: Verify image tags, network attachment, and restart counts.

## 5. Meta-Cognition Loop (Handling Instability)

When simulations fail or degrade (e.g., OOM kills, health check timeouts), use the **Meta Operations** view:

1.  **Observe Signals**: The "Failure Signals" feed shows degraded/failing containers.
2.  **Capture Note**: Use the "Capture Note" form to record what you see (e.g., "OOM spike during graph ingest").
3.  **Queue Objective**: If a fix is needed, enqueue it (e.g., "Stabilize memory under heavy ingest").
4.  **Track Run**: If you start a training or evaluation run to fix it, log it in "Track Run" so metrics (accuracy, latency) appear in the **Training Charts**.

## 6. Process Hardening & Health

The runtime now protects itself against overload:
- **WebSocket Throttling**: If `eta-mu-system` is under critical pressure (CPU > 92% or Memory > 90%), it will:
    - Slow down dashboard updates.
    - Skip expensive simulation ticks.
    - Reject new WebSocket connections (HTTP 503).
- **Health Endpoint**: Monitor `/api/runtime/health` for machine-readable guard status.

Always check the **Resource Budget** bars on the dashboard profile to ensure your simulations aren't hitting hard limits.

## 7. Lith Probabilistic ECS DSL

The system includes a high-level orchestration DSL (**ημ.lith.lang.v0_1**) for managing entities, components, and contracts with uncertainty.

### Core Forms
- **entity**: Define a stable identity.
  `(entity {:in :sim :id :e/duck :type :agent})`
- **attach**: Attach a component payload.
  `(attach {:in :sim :e :e/duck :c :World.Pos :v {:x 1.2 :y -3.4}})`
- **obs**: Emit probabilistic signals.
  `(obs {:ctx :presence/witness_thread :about {:e :e/node-123} :signal {:kind :related} :p 0.62})`
- **system**: Declare a read/write contract.
- **promise**: A time-bounded promise about state transitions.

### Execution
The DSL executes in a 5-stage cycle:
1. **Ingest**: Observations arrive.
2. **Reconcile**: Belief policies combine conflicting evidence.
3. **Plan**: Systems propose future states.
4. **Commit**: Intents apply if permitted.
5. **Audit**: Contracts are verified.

### Presence Overlap
Presences are first-class entities. "Mind overlap" between entities is computed as the cosine similarity of their presence attachment weights.
