# C Simulation Performance Optimization Spec (v1)

## 1. Purpose

Reduce computational cost of the C-backed particle simulation (`libc_double_buffer_sim.so`) by targeting identified hot paths, preserving behavioral semantics and test coverage.

## 2. Scope

### In scope
- `force_system_worker` (all-pairs semantic gravity, elastic edges)
- `chaos_system_worker` (simplex noise, trigonometric harmonics)
- `cdb_graph_runtime_accumulate_gravity` (bounded Dijkstra gravity propagation)
- `cdb_graph_route_step` (per-particle edge scanning and stochastic routing)
- API boundary costs (`update_nooi`, `update_embeddings`, `snapshot`)
- Python bridge (`c_double_buffer_backend.py`) routing mode selection

### Out of scope
- Frontend rendering performance
- Network/WebSocket throughput
- NPU/GPU acceleration paths (separate spec)
- Audio synthesis pipelines

## 3. Baseline Measurements (Current)

| Component | Configuration | Metric | Value |
|-----------|--------------|--------|-------|
| force worker | count=200, sleeps=0 | fps | ~21,394 |
| force worker | count=400, sleeps=0 | fps | ~3,317 |
| force worker | count=800, sleeps=0 | fps | ~1,396 |
| chaos worker | count=200, sleeps=0 | fps | ~67,867 |
| chaos worker | count=400, sleeps=0 | fps | ~25,007 |
| chaos worker | count=800, sleeps=0 | fps | ~10,592 |
| integrate worker | count=200, sleeps=0 | fps | ~550,384 |
| integrate worker | count=400, sleeps=0 | fps | ~381,908 |
| integrate worker | count=800, sleeps=0 | fps | ~128,052 |
| semantic worker | count=200, sleeps=0 | fps | ~1,154,162 |
| semantic worker | count=400, sleeps=0 | fps | ~585,008 |
| semantic worker | count=800, sleeps=0 | fps | ~279,354 |
| graph maps | nodes=220, edges=1320, sources=12 | ms/call | ~0.11 |
| graph maps | nodes=420, edges=2520, sources=12 | ms/call | ~0.19 |
| graph route | nodes=280, fanout=6, particles=800 | ms/call | ~0.65 |
| graph route | nodes=280, fanout=10, particles=1600 | ms/call | ~1.52 |
| update_nooi | count=900 | ms/call | ~0.0096 |
| update_embeddings | count=900 | ms/call | ~0.0041 |
| snapshot | count=900 | ms/call | ~0.0086 |
| full pipeline | python benchmark | cdb mean | ~45.78ms |

### Key observations
1. Force worker shows superlinear scaling due to nested O(N^2) loops for nexus pairs
2. Chaos worker scales quadratically due to simplex calls per particle per frame
3. Graph maps scales linearly with source_count (bounded Dijkstra per source)
4. Graph route scans all edges per particle (O(P * E) worst case)

## 4. Hot Path Identification

### 4.1 force_system_worker (HIGHEST PRIORITY)
- **Location**: `c_double_buffer_sim.c:995-1101`
- **Inner loop 1**: Semantic gravity toward all Nexus entities (line 1043-1059)
  - O(N * N_nexus) with 24D dot product per pair
  - `dot_product_24` (line 987-993): 24 multiply-accumulate operations
- **Inner loop 2**: Elastic edges for Nexus-Nexus links (line 1065-1087)
  - O(N_nexus * N_group_members)
  - `sqrtf`, `powf`, `clamp01` per pair
- **Nooi field lookup**: Per-particle grid cell access (line 1021-1038)

### 4.2 cdb_graph_runtime_accumulate_gravity
- **Location**: `c_double_buffer_sim.c:220-336`
- **Dijkstra inner loops**:
  - Node scan for minimum (line 274-287): O(V) per iteration
  - Edge relaxation (line 294-312): O(E) per iteration
  - Gravity accumulation (line 315-330): O(V) per source
- **Total per source**: O(V^2 + E) naive, bounded by `bounded_radius`

### 4.3 cdb_graph_route_step
- **Location**: `c_double_buffer_sim.c:776-941`
- **Edge scanning**: Full edge array scan per particle (line 825-852)
- **Exploration scan**: Second pass when exploring (line 900-923)
- **Per-particle ops**: Hash mixing, expf, tanhf

### 4.4 chaos_system_worker
- **Location**: `c_double_buffer_sim.c:1103-1158`
- **Simplex noise**: 2 calls per particle (lines 1134-1143)
  - `simplex_noise_2d` (line 168-218): hash, gradient, interpolation
- **Trigonometric harmonics**: sinf, cosf per particle

### 4.5 Python bridge routing mode selection
- **Location**: `c_double_buffer_backend.py:2244+`
- **Fallback risk**: Resource-aware routing may bypass native `cdb_graph_route_step`
- **Metric**: `resource_routing_mode` in payload summary

## 5. Optimization Strategies

### 5.1 force_system_worker (P0)
| Strategy | Complexity | Impact | Risk |
|----------|------------|--------|------|
| Spatial hash / cell grid | Medium | High | Medium |
| Precompute normalized embeddings | Low | Medium | Low |
| Cache nexus list + group adjacency | Low | Medium | Low |
| Batched SIMD dot products | High | High | Medium |

**Recommended first step**: Cache `nexus_indices[]` and per-group member lists, precompute normalized embeddings.

### 5.2 cdb_graph_runtime_accumulate_gravity (P0)
| Strategy | Complexity | Impact | Risk |
|----------|------------|--------|------|
| CSR adjacency + heap Dijkstra | Medium | High | Low |
| Multi-source Dijkstra | Medium | High | Medium |
| Edge cost caching across frames | Low | Medium | Low |

**Recommended first step**: Convert edge storage to CSR format with per-node edge ranges.

### 5.3 cdb_graph_route_step (P0)
| Strategy | Complexity | Impact | Risk |
|----------|------------|--------|------|
| CSR edge ranges per node | Low | High | Low |
| Early exit on candidate threshold | Low | Medium | Low |
| SIMD score computation | Medium | Medium | Medium |

**Recommended first step**: CSR edge ranges (same as 5.2, shared data structure).

### 5.4 chaos_system_worker (P1)
| Strategy | Complexity | Impact | Risk |
|----------|------------|--------|------|
| Decimated simplex + interpolation | Medium | Medium | Medium |
| Precomputed noise tiles | Medium | Medium | Low |
| Lower harmonic count | Low | Low | Low |

**Recommended first step**: Skip simplex on alternate frames with linear interpolation.

### 5.5 Native resource-aware routing (P1)
| Strategy | Complexity | Impact | Risk |
|----------|------------|--------|------|
| Port resource logic to C | High | Medium | Medium |
| Add native routing mode flag | Low | Low | Low |

**Recommended first step**: Add telemetry for Python fallback frequency.

## 6. Implementation Phases

### Phase 1: Baseline + Kernel Bench (1 day)
- [ ] Create `part64/scripts/bench_cdb_kernels.py`
- [ ] Record baseline artifacts in `perf/baselines/`
- [ ] Add CI perf gate placeholder

### Phase 2: CSR Edge Storage + Route Optimization (2 days)
- [ ] Add CSR structures to `CDBEngine` (node_offsets, sorted edges)
- [ ] Refactor `cdb_graph_route_step` to use edge ranges
- [ ] Refactor `cdb_graph_runtime_accumulate_gravity` to use adjacency
- [ ] Run tests + benchmarks

### Phase 3: Force Worker Caching (2 days)
- [ ] Add `nexus_indices`, `nexus_count` arrays
- [ ] Add per-group member lists
- [ ] Precompute normalized embeddings (or cache magnitudes)
- [ ] Run tests + benchmarks

### Phase 4: Chaos Optimization (1 day)
- [ ] Add frame-decimated simplex mode
- [ ] Add linear interpolation for skipped frames
- [ ] Run tests + benchmarks

### Phase 5: Telemetry + Guardrails (1 day)
- [ ] Add routing mode telemetry counter
- [ ] Add perf regression thresholds to CI
- [ ] Document tuning knobs

## 7. Acceptance Criteria

### Must have
- [ ] All tests pass: `python -m pytest part64/code/tests/test_c_double_buffer_backend.py -q`
- [ ] No payload schema regressions (frontend still renders)
- [ ] Kernel benchmarks improve >=30% for target configuration:
  - nodes=220, fanout=6, sources=12, particles=800
- [ ] Full pipeline benchmark improves >=20%:
  - `python part64/scripts/bench_cdb_vs_python.py --iterations 30`
- [ ] Receipts logged for runtime changes

### Should have
- [ ] Force worker fps at count=800 improves >=50%
- [ ] Route step ms/call at 800 particles improves >=30%
- [ ] No new memory allocations in hot loops (valgrind check)

### Nice to have
- [ ] SIMD-accelerated dot products
- [ ] Native resource-aware routing

## 8. Test Plan

### Unit tests
- Existing: `test_c_double_buffer_backend.py`
- Add: CSR edge range correctness tests
- Add: Force worker nexus list cache validity

### Integration tests
- Existing: `test_world_web_pm2.py`
- Add: Perf regression threshold check (fail if >10% slower)

### Stress tests
- Run with `count=1200` for 60 seconds, verify no crashes
- Run with all sleeps=0, verify frame counters advance correctly

### Visual regression
- Compare particle position distributions before/after
- Use fixed seed, compare snapshots at frame 100, 1000

## 9. Rollback Plan

If optimization causes regressions:
1. Feature flags in `c_double_buffer_sim.c`:
   - `CDB_USE_CSR_EDGES=0` to disable CSR path
   - `CDB_CACHE_NEXUS_LIST=0` to disable force cache
   - `CDB_DECIMATE_SIMPLEX=0` to disable chaos decimation
2. Environment variable overrides in Python bridge
3. Revert to previous `.so` via git checkout

## 10. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Dynamics drift after force refactor | Medium | High | A/B snapshot comparison, feature flag |
| Memory growth from caches | Low | Medium | Pool allocations, limit cache sizes |
| Thread safety regression | Low | High | Single-writer discipline, stress test |
| Perf variance hiding regressions | Medium | Medium | Fixed seeds, warmup, median/p95 gating |
| Python fallback silently reappears | Medium | Low | Telemetry counter, explicit assert |

## 11. Related Documents

- `specs/npu-benchmark-spec.md` - NPU embedding benchmark protocol
- `specs/drafts/part64-deep-research-02-graph-runtime-gravity-pricing.md`
- `specs/drafts/part64-deep-research-04-daimoi-packets-routing.md`
- `part64/code/world_web/native/c_double_buffer_sim.md`

## 12. Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-02-20 | Muse - Alignment | Initial spec from hot path analysis |
