# NPU Benchmark Spec (C Runtime, Embeddings + MRL)

## 1. Purpose

Define a **reproducible, allocation-free, in-process** benchmarking protocol for **NPU-backed embedding inference** on a laptop where **microseconds matter**.

This spec focuses on:

* **Latency** (especially **p99 / max**) and **jitter**
* **Time-to-first-inference (TTFI)** and warm-up behavior
* **Dispatch/copy overhead** (host↔device) that often dominates at batch=1
* **Correctness** and **no-silent-fallback** to CPU
* **MRL dimension ladder** operating points (quality vs dim), but the *bench* is NPU-centric.

---

## 2. Scope

### In scope

* Local **embedding model inference** in a **C runtime** using an NPU-capable backend.
* NPU device enumeration, selection, verification.
* Bench harness design requirements.
* Output formats for automated comparison.

### Out of scope

* Model training.
* Networked inference.
* Production deployment architecture (except what’s required to mimic the sim’s hot loop).

---

## 3. Definitions

* **Backend**: inference runtime path (e.g., ORT+EP, OpenVINO direct, vendor SDK).
* **EP**: Execution Provider (ORT term).
* **TTFI**: time from process start (or model load call) to first valid embedding output.
* **Hot call**: the exact function you call from the simulation tick.
* **Silent fallback**: backend selects CPU for unsupported ops without failing or warning.

---

## 4. Assumptions & Hard Constraints

1. **No network** and no cross-process IPC in the hot path.
2. Hot call must be **allocation-free** (no malloc/free, no new/delete, no hidden reallocs).
3. Inputs/outputs are **pre-allocated** and reused.
4. Benchmark must produce **raw per-iteration timings**.
5. Benchmark must validate **device usage** (NPU) and detect fallback.
6. Measurement must include both:

   * **Model-call-only latency** (pure inference)
   * **Boundary latency** (preprocess/tokenize if applicable + inference + postprocess/normalize)

---

## 5. Benchmark Harness Requirements

### 5.1 Executable interface

Single binary `embed_bench` with CLI:

* Model/config:

  * `--model <path>` (ONNX / IR / vendor format)
  * `--tokenizer <path>` (optional; can be disabled)

* Backend/device:

  * `--backend <ort|openvino|vendor>`
  * `--device <NPU|CPU|GPU|AUTO>` (for comparative runs; NPU-focused)
  * `--verify-device <strict|warn|off>`

* Workload:

  * `--inputs <file>` (newline-delimited text) or `--tokens <file>` (pre-tokenized)
  * `--n <iters>`
  * `--warmup <iters>`
  * `--batch <1|...>` (default 1)
  * `--dim <d>` (MRL truncation dim; may be 0 meaning “full”)

* Execution control:

  * `--threads <T>` (intra-op / inter-op as supported)
  * `--pin <0|1>` (pin benchmark threads)
  * `--priority <normal|high|realtime>` (best-effort; OS dependent)
  * `--power <ac|battery|ignore>` (record only)

* Output:

  * `--out <dir>` (writes JSON + CSV)
  * `--tag <string>` (run label)

Exit codes:

* `0`: success
* `2`: device verification failed (strict)
* `3`: correctness failed
* `4`: runtime error

### 5.2 Core API inside the binary

The harness must expose an internal interface used by all backends:

* `bench_init(config) -> ctx`
* `bench_prepare_inputs(ctx, dataset)`
* `bench_warmup(ctx)`
* `bench_run(ctx) -> results`
* `bench_destroy(ctx)`

Hot call form (what sim would call):

* `embed_hot(ctx, input_ref, out_embedding_ptr, dim)`

Constraints:

* `embed_hot` MUST NOT allocate.
* `out_embedding_ptr` points to caller-owned memory.

---

## 6. Device Enumeration, Selection, Verification (NPU)

### 6.1 Enumerate

On startup, the harness MUST print and record:

* available devices
* selected device
* runtime versions

### 6.2 Select

Selection rules:

* If `--device NPU` specified: force NPU selection.
* If device unavailable: fail with exit code `4` unless `--device AUTO`.

### 6.3 Verify (no silent fallback)

Verification modes:

* `strict`: if NPU not actually used for compute, exit `2`.
* `warn`: record a warning and continue.
* `off`: do not verify (not recommended).

Verification mechanisms (use as many as backend provides):

* Backend-reported device for compiled model / session
* Runtime logs indicating placement
* Profiling/tracing: operator placement per node

Additionally, include a “fallback detector” heuristic:

* If selected NPU but measured latency matches CPU baseline within small epsilon AND profiling indicates CPU kernels, flag.

---

## 7. Workload Specification

### 7.1 Input datasets

Benchmark must support two modes:

**A) Pre-tokenized (preferred for micro-latency isolation)**

* Input file provides `input_ids`, `attention_mask` (and optional `token_type_ids`).
* This isolates tokenizer costs and focuses on inference.

**B) Raw text**

* Harness performs tokenization (must be benchmarked separately and recorded).

Dataset requirements:

* Must represent real sim distribution.
* Include duplicates to measure caching potential.
* Provide summary stats (min/mean/max token length).

### 7.2 Batch

Default `batch=1`.
If batch>1 is used, record batch size and ensure sim can actually benefit (no hiding overhead behind batching unless that’s the plan).

---

## 8. Timing & Measurement Protocol

### 8.1 Clocks

Use monotonic high-resolution clock.
Record timing overhead by measuring an empty loop and subtract only if necessary (microsecond regime).

### 8.2 Warm-up

Warm-up stages:

1. session/model compile
2. first call (records first-run latency)
3. warmup iterations (discarded) until stable

Record:

* compile time
* first inference time
* warm-up iterations count

### 8.3 Sampling

For `--n iters`:

* choose inputs deterministically (round-robin) unless `--shuffle` is added
* record per-iteration latency in **nanoseconds**

### 8.4 What to time (two modes)

The harness must support timing modes:

* `--timing model_only`: start timer immediately before runtime `Run`/`Infer` call, stop after output is ready.
* `--timing boundary`: include tokenization (if enabled) + inference + embedding normalization + truncation.

For NPU, also record (if measurable):

* host→device copy time
* device→host copy time

---

## 9. Output Artifacts

Each run produces:

### 9.1 `run.json`

Contains:

* run metadata (tag, timestamp, git sha if available)
* machine info (CPU model, RAM, OS, power state)
* runtime info (backend versions)
* device info (selected device, enumerated devices)
* model info (path, opset, input shapes)
* config (threads, pinning, batch, dim)
* dataset stats (token lengths, count)
* summary metrics:

  * compile_ms
  * first_infer_us
  * p50_us, p90_us, p99_us, max_us
  * mean_us, stddev_us
  * spike_rate (fraction over threshold)

### 9.2 `latency.csv`

Columns:

* `iter, input_index, tokens_len, latency_ns`

### 9.3 `profile.json` (optional but recommended)

Operator placement and/or runtime profiling trace.

---

## 10. Correctness Checks

Before recording timings:

* run a correctness pass on a small fixed set:

  * verify output shape
  * verify normalization (if enabled)
  * verify deterministic behavior within tolerance

For MRL truncation:

* ensure `dim <= full_dim`
* `embedding_prefix = embedding[0:dim]`
* normalization rule is explicit and consistent:

  * recommended: normalize **after** truncation for cosine workflows

If correctness fails, exit `3`.

---

## 11. MRL Dimension Ladder Bench (NPU-specific)

Because MRL is a first-class constraint, every NPU bench run MUST specify one of:

* `--dim 0` meaning full
* `--dim <d>` meaning prefix dim

Recommended standard ladder for each model:

* Full, 512, 384, 256, 192, 128

NPU benchmark outputs must include results for at least:

* Full and the intended operating dim `d*`

Optional: export a “dim sweep” mode:

* `--dim-sweep full,512,256,128`
* runs same benchmark for each dim and writes a `sweep.json` summary

---

## 12. Environmental Controls (Reducing Jitter)

Record (and where possible control):

* CPU governor / performance mode
* pin benchmark thread(s)
* set thread priority
* disable background updates if possible
* run on AC power for stable clocks

Thermal protocol:

* record temperature/frequency if accessible
* abort or mark run invalid if throttling detected (policy choice)

---

## 13. Comparative Baselines (Required)

Even though this is an NPU spec, each session of benchmarking MUST include at least one baseline:

* CPU backend, same model, same dataset, same timing mode.

Rationale: distinguishes “NPU not actually used” and quantifies dispatch overhead.

---

## 14. Acceptance Criteria

A candidate NPU configuration is acceptable only if:

1. **Device verification passes** (no silent fallback).
2. Under realistic sim load (or a synthetic contention proxy), it delivers:

   * lowest (or near-lowest) **p99** among candidates, OR
   * a justified trade where CPU headroom improves sim p99 more than embedding p99 worsens.
3. TTFI is within your operational budget, OR model caching/persistence strategy exists.
4. For the chosen operating dim `d*`, your MRL quality harness shows acceptable degradation.

---

## 15. Failure Modes & Diagnostics

If NPU underperforms or jitters:

* check for host↔device copy overhead
* check thread contention with sim
* check runtime falling back to CPU
* check model not supported by NPU compiler (partial placement)
* check power state / throttling

Diagnostics required in logs:

* selected device
* placement summary (how many nodes on NPU vs CPU)
* warnings from compiler/runtime

---

## 16. Deliverables

By the end of NPU selection, you should have:

* a `runs/` folder of `run.json` + `latency.csv` for each configuration
* a `compare.json` summarizing winners by p99 and jitter
* chosen:

  * primary backend/device
  * fallback backend/device
  * default dim `d*` and optional dim ladder policy

---

## 17. Minimal First Run (Checklist)

1. Prepare dataset (pre-tokenized preferred).
2. Run CPU baseline: `--device CPU --timing model_only`.
3. Run NPU: `--device NPU --verify-device strict`.
4. Repeat both with `--timing boundary`.
5. Repeat both at `--dim full` and `--dim d*`.
6. Compare p99 and max; inspect placement logs.
