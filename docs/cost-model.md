# Cost model

The cost model is two pure functions in `src/arch_sim/cost/cost.py`. Each takes an
architecture unit and a tensor, and returns a cycle count.

```python
def Op(unit: ComputeUnit, t: torch.Tensor) -> float:
    return t.numel() / unit.throughput_ops_per_cycle

def Move(pipe: Pipe, t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / pipe.bandwidth
```

- **`Op`** prices a compute operation: number of elements divided by the unit's
  throughput. It uses **elements**.
- **`Move`** prices a data transfer through a pipe: number of bytes divided by the
  pipe's bandwidth. It uses **bytes** (`numel * element_size`).

The simulator's `Instruction.cost()` dispatches automatically: a `Pipe` target
calls `Move`, anything else (a `ComputeUnit`) calls `Op`.

## Unit conventions

The two denominators come straight from the [chip spec](chip-specs.md) and are in
different units — keep them straight:

- `ComputeUnit.throughput_ops_per_cycle` → **elements per cycle**.
- `Pipe.bandwidth` → **bytes per cycle**.

So a 1024-element fp32 tensor (4096 bytes) costs `1024 / 128 = 8` cycles on a
Vector unit (throughput 128) but `4096 / 64 = 64` cycles on MTE2 (bandwidth 64).
Same tensor, different denominators. dtype affects `Move` (via `element_size`) but
not `Op` — fp16 halves the move cost while leaving the element count unchanged.

## Meta tensors

Tensors are `torch.Tensor` on the **`meta` device**. A meta tensor has a shape and
dtype but no backing storage, so it costs nothing to create even at huge sizes:

```python
import torch
t = torch.empty(1_000_000_000, dtype=torch.float32, device="meta")
t.numel()         # 1_000_000_000  — works without allocating 4 GB
t.element_size()  # 4
```

The cost functions read only shape and dtype (`numel()`, `element_size()`, and —
for tiled units — `shape`/`dim()`), so meta tensors are exactly the right
descriptor: they carry what the model needs and nothing else. Always build workload
tensors with `device="meta"`.

## Matmul: tile-quantized cost

`Op` uses plain `numel` for elementwise ops (add, exp, …), but a matmul's real cost
is M·N·K multiply-accumulates, and the Cube works on fixed tiles — every dimension
is **padded up to the tile boundary** before the MACs are counted. `Op` models this
when the unit has an `operand_shape` (the tile):

```python
# Op pads each dim of t up to the corresponding operand_shape dim, then divides.
work = product(ceil(dim / tile_dim) * tile_dim for dim, tile_dim in zip(t.shape, operand_shape))
return work / unit.throughput_ops_per_cycle
```

So you pass the matmul's iteration space `(M, N, K)` as the workload tensor and the
Cube's `operand_shape` (e.g. `[16, 16, 16]`) supplies the tile:

```python
cube = ComputeUnit(operand_shape=(16, 16, 16), throughput_ops_per_cycle=4096)
Op(cube, meta(64, 64, 64))  #  64^3 / 4096 = 64   (already tile-aligned)
Op(cube, meta(17, 17, 17))  #  32^3 / 4096 = 8    (17 rounds up to 32 on each axis)
```

The padding is the point: a 17×17×17 matmul costs as much as 32×32×32, because the
Cube can't do a partial tile. Units without an `operand_shape` (Vector, Scalar) keep
the plain `numel / throughput` path. The tensor rank must match the `operand_shape`
rank, or `Op` raises.

Not yet modeled (see PLAN.md): pipeline fill latency and dtype-dependent throughput.
