# Specification: Nexus–Daimoi Simulation Mathematical Alignment

## Overview
This specification defines the architectural and mathematical requirements for the Nexus–Daimoi hybrid simulation runtime. It establishes a formal mapping between the "Compact Mathematical Specification" and the `part64/code` implementation, focusing on aligning the C-based double-buffer backend with the core physical primitives.

## 1. Mathematical Foundations

### 1.1 State Spaces
- **Field Space ($\mathbb{R}^2$)**: Particles and Nexus entities occupy a continuous 2D coordinate system $[0, 1]$.
- **Semantic Space ($\mathbb{R}^{24}$)**: Entities carry a 24-dimensional embedding vector used for similarity calculations and force modulation.
- **Intent Set ($J$)**: A discrete index set of job keys (e.g., `deliver_message`, `invoke_truth_gate`).

### 1.2 Entity Mapping
| Mathematical Entity | Code Implementation | Key Properties |
| :--- | :--- | :--- |
| **Nexus** ($n$) | `is_nexus: true` particles | `mass`, `radius`, `preferred_x/y`, `source_node_id` |
| **Presence** ($p$) | `presence_impacts` entries | `lens_embedding` (from domain), `resource_wallet` |
| **Daimon** ($i$) | `field_particles` | `alpha_pkg` (intent counts), `owner_presence_id`, `velocity` |

## 2. Nooi Field (Sparse Vector Field)
The field consists of discrete cells ($64 \times 64$) with 8 layers of vectors.

- **Decay**: $\mathbf{F}_\ell \leftarrow (1 - \delta \Delta t) \mathbf{F}_\ell$.
- **Deposit**: Particles deposit their normalized velocity $\hat{v}_i$ into the layer corresponding to their owner index: $\ell = o_i \pmod 8$.
- **Implementation**: Handled in `nooi.py` and synchronized to the C-backend via `cdb_engine_update_nooi`.

## 3. Motion Dynamics Alignment

### 3.1 Field Acceleration ($\mathbf{a}^{field}$)
Particles sample the field layer $\ell$ at their current cell.
- **Current Status**: Compliant.
- **Requirement**: Maintain scaling factor $\alpha_\ell$ to ensure field forces are observable relative to noise.

### 3.2 Semantic Gravity ($\mathbf{a}^{grav}$) - [GAP]
The spec requires a $1/r^2$ attraction toward all Nexus nodes, weighted by semantic affinity $\kappa(i,n)$.
- **Current State**: Implementation uses Quadtree-bounded nearby node attraction.
- **Alignment Requirement**: Implement a global (or wider-radius) $1/r^2$ gravity term in `c_double_buffer_sim.c` where $\kappa(i,n) = \text{sim}(u_i, e_n)$.

### 3.3 Noise ($\mathbf{a}^{noise}$)
- **Current State**: Compliant. Uses multi-octave Simplex noise.
- **Chaos Agent**: "Chaos Butterflies" provide high-amplitude noise injection into the local field.

### 3.4 Integration & Friction - [GAP]
- **Spec**: $v_i \leftarrow (1 - \gamma_i \Delta t)v_i + \Delta t(\mathbf{a}_{total})$.
- **Current State**: Uses a hard `speed_cap` for stability.
- **Alignment Requirement**: Introduce explicit $\gamma$ (friction) coefficients for Nexus vs. Daimon entities to allow for natural damping and overshooting.

## 4. Graph Elastic Edges - [GAP]
Edges between Nexus nodes must have semantic elasticity.
- **Spec**: $k_e = k_0 (1 - sim(e_{n_a}, e_{n_b}))^\beta$.
- **Current State**: Uses hardcoded `group_id` matching.
- **Alignment Requirement**: Refactor `force_system_worker` to compute edge tension based on the cosine similarity of the linked Nexus embeddings.

## 5. Collision & Wallet Exchange
- **Saturation ($S_n$)**: Nexus nodes deflect if their load/capacity ratio $\ge 1$.
- **Probabilistic Exchange**: Implementation uses `_dirichlet_transfer` to blend intent distributions ($\alpha$) between colliding entities. This is a "soft" version of the spec's integer quanta exchange.

## 6. Closed Loop Lifecycle
1. **Spending**: Presence drains resource wallet to maintain state.
2. **Emission**: Presence emits new daimoi based on resource availability ($\mu_p$).
3. **Motion**: Field, Gravity, and Noise drive displacement.
4. **Exchange**: Collisions update intent distributions.
5. **Collapse**: Energy depletion triggers observation/ledger events.

## Roadmap & Sub-tasks

### Phase 1: C-Backend Physics Refactor
- [ ] **Task 1.1**: Implement $\gamma$ friction damping in `integrate_system_worker` and remove hard speed caps.
- [ ] **Task 1.2**: Implement global $1/r^2$ Semantic Gravity in `force_system_worker`.
- [ ] **Task 1.3**: Refactor Edge Force to use cosine similarity of embeddings instead of `group_id`.

### Phase 2: Runtime Stabilization
- [ ] **Task 2.1**: Tune $\delta$ (field decay) and $\lambda$ (noise) to prevent field stagnation or explosion.
- [ ] **Task 2.2**: Verify Dirichlet transfer stability during high-frequency collisions.

### Phase 3: Hardware Perceptual Link
- [ ] **Task 3.1**: Ensure `resource_heartbeat` continues to drive the Emission Rate ($\mu_p$) to maintain the closed-loop economy.

## Definition of Done
- [ ] Simulation motion profiles demonstrate observable semantic clustering.
- [ ] C-backend follows the $v \leftarrow (1 - \gamma)v + a \Delta t$ damping model.
- [ ] Edge tension is dynamically calculated from embedding similarity.
- [ ] All LSP errors in `daimoi_probabilistic.py` are resolved.
