# C Double-Buffer Simulation Core

This core implements the simulation pattern requested for high-throughput updates:

- no strings in the hot loop,
- numeric ids for ownership/group routing,
- bitfields for booleans,
- single-writer ownership per variable set,
- double-buffered component arrays (`readable`/`writable`) with atomic swap.

## Component Ownership

- `force` system thread owns `acceleration` buffers.
- `chaos` system thread owns `noise` buffers.
- `integrate` system thread owns `position` and `velocity` buffers.

Each system reads only readable buffers and writes only its owned writable buffer.
When a pass is complete, the system flips its readable index atomically.

## Data Layout

The core uses struct-of-arrays style storage:

- dynamic vec2 buffers: `position`, `velocity`, `acceleration`, `noise`
- static numeric arrays: `owner_id`, `group_id`, `mass`, `radius`, `flags`

Force-side spatial acceleration uses a frame-local Barnes-Hut quadtree
backed by engine-owned arrays:

- quadtree clustering nodes (AABB bounds, aggregate mass, semantic centroid)
- per-particle linked lists for leaf membership
- optional parallel force workers over disjoint particle ranges

The force pass now computes:

- collision response from quadtree-pruned neighborhood queries
- inverse-square semantic gravity using Barnes-Hut approximation
- nexus in-group clustering spring using Barnes-Hut node aggregation

Flags are bitfields:

- `0x1`: nexus
- `0x2`: chaos
- `0x4`: active

## API

Shared library entrypoints:

- `cdb_engine_create(count, seed)`
- `cdb_engine_start(engine)`
- `cdb_engine_stop(engine)`
- `cdb_engine_snapshot(engine, ...)`
- `cdb_engine_destroy(engine)`

Additional native utility kernels used by Python hot paths:

- `cdb_resolve_semantic_collisions(...)` (parallel collision resolution over stream rows)

The Python bridge (`c_double_buffer_backend.py`) compiles/loads this library and
maps snapshots into existing `field_particles` payload rows.

## Tunables

Key runtime env controls for the C force pass:

- `CDB_FORCE_WORKERS` (parallel worker count for force ranges)
- `CDB_BH_THETA` (Barnes-Hut opening angle)
- `CDB_BH_LEAF_CAP` (max particles per quadtree leaf)
- `CDB_BH_MAX_DEPTH` (quadtree depth cap)
- `CDB_COLLISION_SPRING` (collision correction stiffness)
- `CDB_CLUSTER_THETA` (Barnes-Hut opening angle for in-group clustering)
- `CDB_CLUSTER_REST_LENGTH` (rest length for in-group clustering spring)
- `CDB_CLUSTER_STIFFNESS` (stiffness multiplier for in-group clustering spring)

Collision helper tunables:

- `CDB_COLLISION_WORKERS` (worker count for `cdb_resolve_semantic_collisions`; falls back to force worker config)
