---
source: user-session-2026-02-20
section: Shared field model and daimon dynamics
status: inprogress
reviewed_on: 2026-02-20
last_progress: 2026-02-20
---

# Part64 Deep Research Spec 10 - Shared Fields and Daimoi Dynamics

## Priority
- high

## Complexity
- high (simulation math + attribution + runtime diagnostics)

## Intent
- Move from presence-coupled field assumptions to a bounded shared-field stack.
- Keep presences as contributors to fields, not field owners.
- Make daimon motion, assignment, and attribution fully explainable from ledger events.

## Requirements
- Maintain a fixed global field registry (`demand`, `flow`, `entropy`, `graph`) with bounded count.
- Forbid per-presence field objects in runtime payload and simulation loops.
- Compute total potential as weighted sum of shared fields and use it for daimon transport.
- Keep owner, task pointer, and type distribution explicit on every daimon packet.
- Every braid diagnostic must be decomposable into presence + node + event contributions.
- Emit ledger events for emission, transport, collision, assignment, handoff, spawn, prune, and anchor actions.

## Progress (2026-02-20)
- [x] Added `FIELD_KINDS` and `MAX_FIELD_COUNT` constants in `constants.py`.
- [x] Added canonical `FieldRegistry`, `SharedField`, `SharedFieldSample`, `SharedFieldContributor` TypeScript interfaces.
- [x] Added `_build_field_registry` function in `simulation.py` that builds bounded field registry.
- [x] Field registry explicitly tracks `bounded: true` and `field_count: 4`.
- [x] Field registry includes weights for combining fields: `{demand: 0.4, flow: 0.2, entropy: 0.15, graph: 0.25}`.
- [x] Demand field samples derived from gravity data.
- [x] Graph field samples derived from node prices.
- [x] Test `test_canonical_field_registry_is_bounded` enforces bounded invariant.

## Current evidence in code
- Gravity and routing are currently derived with per-source input vectors in `part64/code/world_web/c_double_buffer_backend.py:687`.
- Runtime output exposes scalar gravity plus resource maps in `part64/code/world_web/c_double_buffer_backend.py:1935`.
- Drift terms currently combine gravity/cost/valve terms in `part64/code/world_web/c_double_buffer_backend.py:260`.
- Daimoi generation and routing integration are spread across `part64/code/world_web/simulation.py` and `part64/code/world_web/daimoi_probabilistic.py`.

## Gaps vs target model
- No explicit shared field registry with schema/versioning and bounded cardinality.
- Attribution decomposition is partial and not normalized across diagnostics.
- Daimoi owner/purpose labels are not uniformly required by contract.
- No formal no-per-presence-field guardrail in tests.

## Mathematical contract (ascii form)
- `K(x, e_p) = exp(-||x - e_p||^2 / (2 * alpha^2))`
- `F_demand(x,t) = sum_p (pi_p(t) * m_p(t) * D_p(t) * K(x, e_p))`
- `F_flow(x,t) = sum_i (||x_i(t)-x_i(t-dt)|| * K(x, x_i(t)))`
- `F_entropy(x,t) = sum_i (H(q_i(t)) * K(x, x_i(t)))`
- `F_graph(x,t) = sum_v (gamma_v(t) * K(x, c_v(t)))`
- `Phi_total(x,t) = w_d*F_demand + w_f*F_flow + w_e*F_entropy + w_g*F_graph`
- `x_i(t+dt) = x_i(t) + eta * grad(Phi_total(x_i,t)) * dt + sqrt(2*kappa*dt) * eps`

## Files planned
- `part64/code/world_web/c_double_buffer_backend.py` (shared field registry and potential assembly)
- `part64/code/world_web/simulation.py` (daimon transport loop and event emission)
- `part64/code/world_web/presence_runtime.py` (presence contribution hooks)
- `part64/code/world_web/projection.py` (diagnostic thread projections)
- `part64/frontend/src/types/index.ts` (shared field payload types)
- `part64/frontend/src/components/Simulation/Canvas.tsx` (field visualization binding)
- `part64/code/tests/test_c_double_buffer_backend.py` (shared-field invariants)
- `part64/code/tests/test_daimoi_probabilistic.py` (motion/collision invariants)
- `part64/code/tests/test_world_web_pm2.py` (payload and attribution schema checks)

## Subtasks
1. Add field registry contract and serialization schema.
2. Refactor field computation to aggregate all presence contributions into shared fields.
3. Add daimon transport update using `grad(Phi_total)` with deterministic fallback.
4. Standardize daimon metadata (`owner`, `type_dist`, `task_id`, `top_k_targets`).
5. Add contribution accounting for every braid metric.
6. Add regression tests for bounded field count and attribution explainability.

## Risks
- Numerical instability from gradient updates without tuned `dt`/`eta` bounds.
- Performance regressions if field updates are not incremental.
- Attribution overhead can increase payload size and websocket traffic.

## Existing issues
- No open GitHub issues found (`gh issue list --state all --limit 10 --json number,title,state` returned `[]`).

## Existing PRs
- No open GitHub PRs found (`gh pr list --state all --limit 10 --json number,title,state` returned `[]`).

## Definition of done
- Runtime exposes one bounded shared-field registry with no per-presence field surfaces.
- Daimoi transport uses shared potential and is deterministic under fixed seeds.
- Braid metrics are attributable to ranked presence/node/event contributors.
- Backend tests covering field cardinality, transport, and attribution pass.
