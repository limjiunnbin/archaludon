# archaludon

A cycle simulator for Ascend-like NPU architectures. You describe a
chip declaratively (storage, compute units, and queued data movers), hand it a
schedule of instructions, and it reports when each instruction runs and how many
cycles the whole thing takes.

Full documentation is in **[docs/](docs/README.md)**.

## Status

The architecture description layer, the cost model, and the channel-based
simulator are implemented. The MLIR/linalg frontend that would turn a tensor
program into instructions automatically is still stubbed out — for now you build
the instruction list by hand (see the examples). The roadmap is in
[PLAN.md](PLAN.md).

## Setup

Requires Python 3.10+. `torch` is a dependency (used only as a shape/dtype
descriptor via meta tensors — no tensors are allocated).

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Quick start

```bash
.venv/bin/python examples/run_add.py
```

```
total cycles: 200
  MTE2: 0..64    GM->UB a
  MTE2: 64..128  GM->UB b
  Vector: 128..136  c = a + b
  MTE3: 136..200    UB->GM c
```

## How it works

**Architecture** is a tree of `Module`s containing units, connected by
`DataPath`s. Units are `StorageUnit`, `ComputeUnit`, `ControlUnit`, and `Pipe`
(a queued mover — MTE1/2/3, FixPipe). A `DataPath` with no engine is a direct
wire; one carried by a `Pipe` shares that pipe's bandwidth. Specs are written in
YAML (`specs/npu.yaml`) or built with the fluent `Module.add_*` API.

**Cost** is two pure functions in `arch_sim/cost/cost.py`:
- `Op(unit, tensor)` = `tensor.numel() / unit.throughput_ops_per_cycle`
- `Move(pipe, tensor)` = `tensor.numel() * tensor.element_size() / pipe.bandwidth`

Tensors are `torch.Tensor` on the `meta` device, so only their shape and dtype
matter.

**Simulation** gives every `Pipe` and `ComputeUnit` its own FIFO `Channel`.
Each `Instruction` carries its target unit, a tensor, a list of `deps`, and
gets `start_time` / `end_time` filled in as it runs. The engine repeatedly pops
the channel head with the smallest ready start time; an instruction starts at
`max(channel's last end, max(dep end times))`. Total cycles = `max(end_time)`.

```python
import torch
from arch_sim.arch import load
from arch_sim.sim import Instruction, Sim

chip = load("specs/npu.yaml")
mte2, mte3, vec = (chip.find(f"AICore.{n}") for n in ("MTE2", "MTE3", "Vector"))

def meta(): return torch.empty(1024, dtype=torch.float32, device="meta")
a, b, c = meta(), meta(), meta()

move_a = Instruction(unit=mte2, tensor=a)
move_b = Instruction(unit=mte2, tensor=b)
add    = Instruction(unit=vec, tensor=c, deps=[move_a, move_b])
store  = Instruction(unit=mte3, tensor=c, deps=[add])

sim = Sim(chip, [move_a, move_b, add, store])
sim.run()
print(sim.total_cycles())  # 200
```

## Examples

| Script | What it shows | Cycles |
|---|---|---|
| `run_add.py` | `c = a + b`, the basic move/compute/move pipeline | 200 |
| `run_matmul.py` | 64×64×64 matmul through L1/L0A/L0B/Cube/FixPipe | 768 |
| `run_softmax.py` | softmax as three serialized Vector ops | 152 |
| `run_parallel_add.py` | `g = (a+b)+(d+e)` — move/compute overlap on one core | 336 |
| `run_parallel_add_2core.py` | two adds on two AICores, truly in parallel | 200 |

Run any with `.venv/bin/python examples/<script>.py`.

## Multi-core

Add `count: N` to a module entry and the builder replicates it (`AICore0`..,
`AICore19` for `count: 20`), including the top-level paths that reference it.
`specs/npu-2core.yaml` is the single-core spec plus one line:

```yaml
- name: AICore
  count: 2
  kind: module
  ...
```

Bumping that to `count: 20` gives an Ascend 910B-shaped chip with no other changes.

## Visualization

```bash
.venv/bin/python examples/visualize_npu.py
```

Writes a Graphviz `.dot` file (and a `.png` if the `dot` binary is installed).
Pipes render as edge labels rather than nodes.

## Tests

```bash
.venv/bin/pytest tests/
```

## Layout

```
src/arch_sim/
  arch/    architecture DSL: units, pipes, YAML loader, validator, visualizer
  cost/    cost primitives (Op, Move)
  sim/     channel-based simulator (Instruction, Channel, Sim, engine)
  ir/ frontend/ report/   stubs — see PLAN.md
specs/     example chip specs
examples/  hand-built schedules
```
