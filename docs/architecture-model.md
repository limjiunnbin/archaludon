# Architecture model

A chip is a **tree of `Module`s containing units, connected by `DataPath`s**. The
DSL lives in `src/arch_sim/arch/`. You can build a chip in Python with the fluent
API or load it from YAML (see [Chip specs](chip-specs.md)).

## BaseUnit

Every element is a `BaseUnit` (`arch/base.py`). It carries a `name`, a `parent`
backreference, and a `kind` (read from a `KIND` class variable on each subclass —
there's no stored kind field). Notable behavior:

- **Identity equality.** `__eq__` is `is` and `__hash__` is `id`. Two units with
  the same name are distinct objects, and sets/dicts of units key on identity.
  This is why the simulator can group instructions by `id(unit)`.
- **`qualified_name()`** walks parents to produce a dotted path, e.g.
  `Chip.AICore.MTE2`.

## Module

A `Module` (`arch/base.py`) is a `BaseUnit` that holds `children` (other units,
including nested modules) and `paths` (the `DataPath`s wiring its children
together). The chip root is a `Module`; each `AICore` is a nested `Module`.

Useful methods:

- `add_storage`, `add_compute`, `add_control`, `add_pipe`, `add_module` — the
  fluent builders.
- `connect(src, dst, *, engine=None, direction=..., bandwidth=..., name=...)` —
  create a `DataPath`. If an `engine` (a `Pipe`) is given, `connect` calls
  `engine.validate(path)` and raises on a violation.
- `find("Dotted.Path")` — resolve a unit by dotted name.
- `walk()` — depth-first iterator over the module and all descendants.

## Unit kinds

| Kind | Class | Represents | Key fields |
|---|---|---|---|
| `storage` | `StorageUnit` | a memory (GM, L1, UB, …) | `capacity_bytes`, `banks`, `read_ports`, `write_ports` |
| `compute` | `ComputeUnit` | a functional unit (Cube, Vector, Scalar) | `operation`, `throughput_ops_per_cycle`, `operand_shape` |
| `control` | `ControlUnit` | instruction issue logic | `instruction_queue_depth` |
| `pipe` | `Pipe` | a queued data mover (MTE1/2/3, FixPipe) | `allowed_src/dst_kinds`, `allowed_src/dst_names`, `bandwidth`, `queue_depth` |

`StorageUnit.can_serve(reads, writes)` checks a request against the port budget.
`ComputeUnit.throughput_ops_per_cycle` and `Pipe.bandwidth` feed the
[cost model](cost-model.md).

## DataPath: wires vs pipes

A `DataPath` (`arch/pipe.py`) connects a `src` unit to a `dst` unit. The crucial
distinction is its `engine` field:

- **`engine is None`** → a direct wire with its own `bandwidth`.
- **`engine` is a `Pipe`** → the transfer is *carried by* that pipe and shares its
  bandwidth and queue. Every `DataPath` referencing the same `Pipe` contends for
  it. This is the hook the simulator uses: one FIFO channel per pipe.

A path also has a `direction` (`uni` or `bi`) and an optional `name`.

## Pipe validation

A `Pipe` constrains which `(src, dst)` pairs it may carry, mirroring real NPU
movers (e.g. MTE2 reads GM, MTE1 reads L1). The constraints:

- `allowed_src_kinds` / `allowed_dst_kinds` — by `UnitKind`.
- `allowed_src_names` / `allowed_dst_names` — by exact unit name.

`Pipe.validate(path)` returns `(ok, reason)`. `Module.connect` enforces it at
construction time, and `arch/validator.py`'s `validate(root)` re-checks every path
across the whole chip (along with name collisions, dangling paths, orphan storage,
and port oversubscription) and returns a `ValidationReport` rather than raising.

## Two ways to build a chip

**Fluent (Python):**

```python
from arch_sim.arch import Module, UnitKind

chip = Module("Chip")
gm = chip.add_storage("GM", capacity_bytes=8 << 30)
core = chip.add_module("AICore")
ub = core.add_storage("UB", capacity_bytes=256 << 10)
mte2 = core.add_pipe("MTE2", allowed_src_kinds=[UnitKind.STORAGE],
                     allowed_dst_kinds=[UnitKind.STORAGE],
                     allowed_src_names=["GM"], allowed_dst_names=["UB"], bandwidth=64)
chip.connect(gm, ub, engine=mte2, name="gm_to_ub")
```

**YAML** — `load("specs/npu.yaml")`. The two paths are equivalent; the loader
sits on top of the same fluent API. See [Chip specs](chip-specs.md).
