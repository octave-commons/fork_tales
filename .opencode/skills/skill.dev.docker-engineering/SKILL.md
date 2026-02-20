---
id: skill.dev.docker-engineering
type: skill
version: 1.0.0
tags: [dev, docker, containers, orchestration, security]
embedding_intent: canonical
---

# Docker & Container Engineering

**Intent**:
- Design, build, and optimize containerized runtimes and simulation environments.
- Manage Docker network topology and cross-container awareness.
- Enforce strict resource constraints and security boundaries.

## Capabilities

### 1. Image Engineering
Build deterministic, optimized, and secure container images.
- **Multistage Builds**: Separating build-time dependencies from runtime artifacts to minimize attack surface and image size.
- **Layer Optimization**: Ordering instructions to maximize build cache efficiency.
- **Base Image Hardening**: Using minimal bases (e.g., Alpine, distroless) and pinning exact versions.

### 2. Orchestration & Compose
Maintain complex multi-service stacks.
- **Dependency Management**: Using `depends_on` with `service_healthy` conditions.
- **Override Patterns**: Using `docker-compose.override.yml` for environment-specific tuning.
- **Scaling**: Managing replica counts and load-balancing through the nginx gateway.

### 3. Network Topology
Configure isolated and shared networks for simulation clusters.
- **Driver Selection**: Bridged networks for local isolation; host networking for high-performance telemetry.
- **Alias Management**: Using network aliases for stable inter-container discovery (e.g., `weaver` alias in `eta-mu-sim-net`).
- **Isolation**: Ensuring simulations are partitioned from the host where required.

### 4. Resource & cgroups Control
Implement strict resource limits to prevent noisy-neighbor effects and OOM cascades.
- **Limits**: Setting `cpus`, `mem_limit`, `memswap_limit`, and `pids_limit`.
- **Reservations**: Ensuring critical system services (e.g., `eta-mu-system`) have reserved resources.
- **Monitoring**: Consuming `/containers/{id}/stats` for live telemetry.

### 5. Docker Socket & API Interaction
Safe interaction with the host Docker daemon.
- **Volume Mounts**: Managing `/var/run/docker.sock` access for discovery services.
- **Event Streaming**: Consuming the Docker events API to trigger real-time dashboard updates on container lifecycle changes.
- **Security**: Using read-only mounts for the socket where possible.

## Operational Guidance
- **Discovery**: Prefer label-based discovery (`io.fork_tales.*`) over name-matching.
- **Signals**: Map container exit codes to Meta failure categories (e.g., Exit 137 -> OOM).
- **Cleanup**: Proactively manage dangling volumes and images in benchmark workers.

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.dev.docker-engineering)
  (domain infra)
  (security-gate "/var/run/docker.sock")
  (discovery-label "io.fork_tales.simulation")
  (network-backbone "eta-mu-sim-net"))
```
