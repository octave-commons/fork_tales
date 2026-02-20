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

The Python bridge (`c_double_buffer_backend.py`) compiles/loads this library and
maps snapshots into existing `field_particles` payload rows.
