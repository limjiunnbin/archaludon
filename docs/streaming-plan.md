# Plan: compute-to-compute streaming (direct unit-to-unit dataflow)

## Goal

Add a new kind of data path: a **compute unit → compute unit** link where the
producer's result is streamed directly into a consumer compute unit for the next
computation, **without writing back to internal memory** (UB / L0C / etc.). The
producer and consumer may run at the same or different rates.

This document is the design; phase 1 (instruction + engine) is what gets
implemented first.

## The core modeling decision

Today the engine treats every dependency as "wait for full completion": in
`sim/engine.py` phase 2, a consumer can only start once every `dep.retire_time`
is set and `<= now`. Producer output also lands in a storage buffer (`dst`), and
that buffer's occupancy is the backpressure mechanism.

A compute→compute stream breaks both assumptions:

- **Overlap, not serialize.** The consumer starts while the producer is still
  running, offset by a fill latency — not after it retires.
- **No buffer slot.** The result never lands in UB/L0C, so there is no storage
  `dst` to occupy. The link is a small hardware FIFO instead.
- **Rate coupling.** Producer and consumer run concurrently; the slower one
  governs. Different `throughput_ops_per_cycle` on the two units already gives
  different per-op costs, so the rate difference falls out of the existing cost
  model — only the *timing* has to reflect overlap.

### Timing model (v1)

For a streamed producer P and consumer C:

```
Cs (consumer start) = max(C's unit free, P.start + L_fill, other non-stream deps retired)
Ce (consumer end)   = max(Cs + Cc, P.end + L_drain)
```

where `Cc` is the consumer's `Op` cost and `L_fill` / `L_drain` are pipeline
fill / drain latencies (default 0 = ideal overlap; configurable per link).

Regime checks:

- **Equal rates:** `Ce ≈ P.start + max(Cp, Cc)` — pipelined cost is `max`, not
  the sum.
- **Consumer faster (`Cc < Cp`):** `P.end + L_drain` dominates → consumer is
  producer-paced, finishes just after the producer.
- **Consumer slower (`Cc > Cp`):** `Cs + Cc` dominates → consumer is the
  bottleneck.

Chains (A→B→C streaming) work recursively, because `start` / `end` are fixed when
an instruction starts, so B's `end` is already known when C starts.

### Explicit non-goal for v1

v1 does **not** model a slow consumer applying finite-FIFO backpressure that
stalls the producer's *unit* (and thus the producer's next instruction). The
streamed producer's `dst` is a non-blocking sink, so it frees its unit at
`P.end`. This is accurate for downstream timing; it is only optimistic about when
the producer's unit becomes reusable. Phase 2 models the link as a depth-`D` FIFO
that backpressures the producer the same way storage buffers do today.

## Code changes

### Core (phase 1)

1. **`sim/instruction.py` — express a streaming edge.**
   Add `stream_deps: list["Instruction"]` alongside `deps`, plus
   `stream_latency: float = 0.0` (the `L_fill` / `L_drain` term). A consumer
   lists memory-resident inputs in `deps` (unchanged) and streamed inputs in
   `stream_deps`. `cost()` is unchanged.

2. **`sim/engine.py` — overlap + non-blocking link.** Three edits in
   `simulate()`:
   - *Start condition (phase 2):* additionally require, for each `s` in
     `head.stream_deps`, that `s.start_time is not None and now >= s.start_time +
     head.stream_latency`. Keep the existing `deps` retire check for non-streamed
     inputs.
   - *End time (phase 2):* `head.end_time = max(now + head.cost(),
     max(s.end_time + head.stream_latency for s in head.stream_deps))`.
     `s.end_time` is always known here (set at the producer's start).
   - *Slot accounting:* the loop that frees a producer's buffer slot when the
     consumer starts reading runs for `deps` only, never `stream_deps`. Streamed
     producers carry `dst=None`, so phase 1 retires them at `end_time` without a
     buffer.
   - Extend the deadlock message to surface instructions stuck on a `stream_dep`
     that never started.

### Declaration / tooling (phase 1.5) — **done**

Items 3–6 are implemented (`DataPath.stream`/`stream_latency`/`fifo_depth`,
builder/loader round-trip, validator compute→compute + acyclic checks, distinct
visualizer rendering, and a `cube_to_vector` stream link in `npu.yaml`).
`DataPath.stream` is inert spec metadata for now — the simulator still reads
`Instruction.stream_deps`; item 7 (MLIR lowering) is what connects the two.

3. **`arch/pipe.py` (`DataPath`) + `arch/builder.py` + `arch/loader.py`.**
   Add to `DataPath`: `stream: bool = False`, `stream_latency: float = 0.0`, and
   optional `fifo_depth: int = 1` (reserved for phase 2). Teach `_build_paths`
   and `Module.connect` to accept them. A bare compute→compute `DataPath` is
   already legal — no `Pipe.validate` change needed.

4. **`arch/validator.py`.** Check that any `stream=True` path is
   compute→compute (warn/error if `dst` is a `StorageUnit`), and add a cycle
   check over streaming edges.

5. **`arch/visualize.py`.** Render streaming edges distinctly (bold/solid arrow
   vs. the dotted direct wire).

6. **`specs/npu.yaml`.** Add e.g.
   `- name: cube_to_vector, src: Cube, dst: Vector, stream: true` so a matmul
   result can stream into Vector for a fused bias/activation without round-trips
   through L0C→UB.

### Optional (later)

7. **`frontend/linalg_lowering.py`.** — **done.** When a matmul's result feeds a
   single elementwise consumer and the spec declares a Cube→Vector stream link,
   `lower()` streams the matmul result into the consumer's Vector op
   (`stream_deps=[mm]`) and drops the matmul's L0C→GM store and the consumer's
   reload. Fusion is matmul→`add` only (add→add still routes through GM).
   Limitation: the IR doesn't model `func` returns, so a matmul whose result is
   also returned would be fused and lose its GM copy — acceptable for v1.

8. **`cost/cost.py`.** If `L_fill` should be derived from tile size rather than a
   constant, add a `tiles(unit, t)` helper and set `L_fill = Cp / tiles`. Out of
   scope for v1.

## Examples + tests

- New example `examples/run_stream_fusion.py`: Cube→Vector streamed (matmul then
  activation) vs. the unfused version through L0C/UB, printing both cycle counts.
- `tests/sim/`: equal-rate overlap gives `max` not sum; faster-consumer is
  producer-bound; slower-consumer is consumer-bound; a 3-stage stream chain; and
  an assertion that the streamed result occupies no storage buffer.
- `tests/arch/`: compute→compute `stream` path builds and validates; a stream
  path to storage warns; visualizer renders the stream edge.

## Suggested order

Land items 1 + 2 first — that alone makes streaming simulable and testable with
hand-built schedules. Then 3–6 for declarative specs and visualization, then 7 if
automatic fusion from MLIR is wanted. Phase 2 (finite-FIFO producer backpressure)
only if a slow consumer stalling the producer's reuse must be modeled.
