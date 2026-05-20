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

The cost functions only ever read `numel()` and `element_size()`, so meta tensors
are exactly the right descriptor: they carry the shape/dtype the model needs and
nothing else. Always build workload tensors with `device="meta"`.

## The matmul caveat

`Op` currently assumes one operation per element (`numel`). That's right for
elementwise ops (add, exp, …) but wrong for matmul, whose true cost is M·N·K
multiply-accumulates, not M·N output elements.

The examples work around this by handing `Op` a tensor whose `numel` equals the
MAC count. For a 64×64×64 matmul:

```python
work = torch.empty(64, 64, 64, dtype=torch.float32, device="meta")  # numel = 262144
# Op(cube, work) = 262144 / 4096 = 64 cycles
```

The clean fix (not yet implemented) is to make `Op` dispatch on `unit.operation`:
for `matmul`, derive cycles from the operand shapes (M·N·K) so the natural `(M, N)`
output tensor can be passed instead. See PLAN.md.
