# Examples

Every script in `examples/` builds an instruction list by hand and runs it through
`Sim`. Run any with `.venv/bin/python examples/<name>.py`. The cycle numbers below
are what the scripts produce on `specs/npu.yaml` (or the 2-core spec).

For the underlying mechanics see [Simulator](simulator.md); for the per-op cycle
math see [Cost model](cost-model.md).

## run_add.py

`c = a + b` over 1024 fp32 — the basic load / compute / store pipeline.

```
total cycles: 200
  MTE2: 0..64    GM->UB a
  MTE2: 64..128  GM->UB b
  Vector: 128..136  c = a + b
  MTE3: 136..200    UB->GM c
```

Two loads serialize on MTE2, the add waits for both, the store waits for the add.
MTE2 dominates (128 of 200 cycles) — this is a memory-bound workload.

## run_matmul.py

`C = A @ B`, 64×64×64 fp32, through the Cube datapath: GM→L1 (MTE2), L1→L0A/L0B
(MTE1), Cube into L0C, FixPipe L0C→GM.

```
matmul 64x64 @ 64x64 fp32 -> 768 cycles
  MTE2: 0..256      GM->L1 A
  MTE2: 256..512    GM->L1 B
  MTE1: 256..320    L1->L0A A
  MTE1: 512..576    L1->L0B B
  Cube: 576..640    Cube C = A@B
  FixPipe: 640..768 L0C->GM C
```

Note MTE1 starts at 256 — in parallel with MTE2 still moving B — because the two
movers are independent channels. The Cube cost (64 cycles) uses the matmul caveat:
a `(64, 64, 64)` meta tensor whose `numel` equals the MAC count (see
[Cost model](cost-model.md#the-matmul-caveat)).

## run_softmax.py

softmax over 1024 fp32, decomposed into three Vector ops (exp, sum, div).

```
softmax over 1024 fp32 -> 152 cycles
  MTE2: 0..64    GM->UB x
  Vector: 64..72   exp(x)
  Vector: 72..80   sum(exp)
  Vector: 80..88   y = exp / sum
  MTE3: 88..152    UB->GM y
```

The three Vector ops serialize on the single Vector channel (8 cycles each). Load
and store bookend them on MTE2/MTE3.

## run_parallel_add.py

`g = (a + b) + (d + e)` on one AICore. The two inner adds are structurally
independent, but they share the single Vector unit, so they take turns.

```
total cycles: 336
  MTE2:    0..64    GM->UB a
  MTE2:   64..128   GM->UB b
  MTE2:  128..192   GM->UB d
  Vector:  128..136   c = a + b      <- Vector and MTE2 firing at the same time
  MTE2:  192..256   GM->UB e
  Vector:  256..264   f = d + e
  Vector:  264..272   g = c + f
  MTE3:  272..336   UB->GM g
```

This shows move/compute overlap (at t=128, MTE2 loads `d` while Vector adds `c`),
but the adds can't truly parallelize on one Vector unit.

## run_parallel_add_2core.py

The same kind of work split across two AICores: `c = a + b` on AICore0 and
`f = d + e` on AICore1, using `specs/npu-2core.yaml`.

```
total cycles: 200  (single-core baseline for one add: 200; for two serialized: 400)
  AICore0.MTE2          0..64    A0: GM->UB a
  AICore1.MTE2          0..64    A1: GM->UB d     <- both cores' MTE2 at once
  AICore0.MTE2         64..128   A0: GM->UB b
  AICore1.MTE2         64..128   A1: GM->UB e
  AICore0.Vector      128..136   A0: c = a + b    <- both cores' Vector at once
  AICore1.Vector      128..136   A1: f = d + e
  AICore0.MTE3        136..200   A0: UB->GM c
  AICore1.MTE3        136..200   A1: UB->GM f
```

Each replica has its own MTE2/Vector/MTE3, so the two additions run fully in
parallel — a clean 2× speedup. Bump the spec's `count` from 2 to 20 for an Ascend
910B shape with no other changes. See [Chip specs](chip-specs.md#multi-core-with-count).

## run_vector_stall.py

Demonstrates **backpressure** using the `sim/backpressure.py` engine (not the
default `Sim`). A burst of Vector ops writes results into UB (capacity 2 results);
MTE3 drains UB to GM 8× slower than Vector produces. UB fills, so a finished Vector
op can't retire and Vector stalls — throttled to the drain rate.

```
per-instruction (start .. exec-end .. retire):
  Vector     0..8    ret 8     v0 = op(x0)
  MTE3       8..72   ret 72    store v0->GM
  Vector     8..16   ret 16    v1 = op(x1)
  MTE3      72..136  ret 136   store v1->GM
  Vector    16..24   ret 24    v2 = op(x2)
  MTE3     136..200  ret 200   store v2->GM
  Vector    24..32   ret 72    v3 = op(x3)   <-- STALLED
  MTE3     200..264  ret 264   store v3->GM

total cycles: 264
bandwidth utilization: 96.97%
```

v3 finishes executing at cycle 32 but can't retire until 72 — UB is full until MTE3
drains a slot. MTE3 is the saturated bottleneck (97%). See
[Simulator › Backpressure engine](simulator.md#backpressure-engine-simbackpressurepy).

## visualize_npu.py

Renders `specs/npu.yaml` to Graphviz (`examples/npu.dot`, plus a `.png`
if `dot` is installed). See [Getting started](getting-started.md#visualize-a-chip).
