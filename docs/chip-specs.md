# Chip specs (YAML)

A chip spec is a YAML mapping describing a `Module` tree. `specs/npu.yaml` is
the reference single-core chip; `specs/npu-2core.yaml` is the same chip with
two AICores. The loader (`arch/loader.py`) and builder (`arch/builder.py`) turn
this YAML into the [architecture model](architecture-model.md).

## Top-level structure

```yaml
name: Chip
kind: module
children:
  - ...        # units and nested modules
paths:
  - ...        # DataPaths between children (resolved by dotted name)
```

The top-level entry must be `kind: module`. A module has `children` and `paths`.

## Unit entries

Every child has a `name` and a `kind`. Fields by kind:

### storage

```yaml
- name: UB
  kind: storage
  capacity_bytes: 262144
  banks: 4          # default 1
  read_ports: 2     # default 1
  write_ports: 2    # default 1
```

### compute

```yaml
- name: Cube
  kind: compute
  operation: matmul              # free-form label; default "generic"
  throughput_ops_per_cycle: 4096 # feeds cost.Op; default 1.0
  operand_shape: [16, 16, 16]    # optional fixed SIMD shape
```

### control

```yaml
- name: ScalarCtrl
  kind: control
  instruction_queue_depth: 16    # default 1
```

### pipe

```yaml
- name: MTE2
  kind: pipe
  allowed_src_kinds: [storage]
  allowed_dst_kinds: [storage]
  allowed_src_names: [GM]               # optional; omit to allow any name
  allowed_dst_names: [L1Buffer, UB]     # optional
  bandwidth: 64                          # bytes per cycle; feeds cost.Move
  queue_depth: 1                         # default 1; currently metadata only
```

`allowed_*_kinds` / `allowed_*_names` are the validation constraints (see
[Pipe validation](architecture-model.md#pipe-validation)).

### module (nesting)

```yaml
- name: AICore
  kind: module
  children: [...]
  paths: [...]
```

## Paths

Paths connect two units. `src`, `dst`, and `engine` are resolved by name —
unqualified (a sibling of the current module) or dotted from the root.

```yaml
paths:
  - name: gm_to_l1          # optional
    src: GM
    dst: AICore.L1Buffer
    engine: AICore.MTE2     # optional; a Pipe carrying this transfer
    direction: uni          # uni (default) or bi
    bandwidth: 0            # only meaningful for engine-less wires
```

A path with no `engine` is a direct wire; one with an `engine` is carried by that
pipe and shares its bandwidth/queue. The builder raises if `engine` doesn't name a
`Pipe`, and `connect` raises if the pipe's constraints reject the `(src, dst)`.

## Multi-core with `count`

Add `count: N` to a module entry and the builder replicates it N times. This is
how you express an Ascend 910B (20 AICores) without writing 20 blocks.

```yaml
- name: AICore
  count: 2
  kind: module
  children: [...]
  paths: [...]
```

What replication does:

- **Names** get a zero-padded numeric suffix sized to `count - 1`: `count: 2` →
  `AICore0`, `AICore1`; `count: 20` → `AICore00` … `AICore19`.
- **Internal structure** is deep-copied per replica — each `AICore0` gets its own
  `UB`, `Cube`, MTEs, and its own internal paths pointing at *its* children.
- **Top-level paths** that reference the counted module at the head of a dotted ref
  are replicated once per replica. `GM -> AICore.L1Buffer (engine: AICore.MTE2)`
  becomes `GM -> AICore0.L1Buffer (engine: AICore0.MTE2)` and the `AICore1`
  equivalent. Path names get a `_<replica>` suffix.

Because each replica has independent units, the simulator treats them as parallel
hardware — see [run_parallel_add_2core](examples.md#run_parallel_add_2corepy).

## Loading, validating, dumping, visualizing

```python
from arch_sim.arch import load, loads, dump, validate, to_dot, visualize

chip = load("specs/npu.yaml")   # from a file
chip = loads(yaml_text)              # from a string

report = validate(chip)              # never raises; collects errors + warnings
assert report.ok

text = dump(chip)                    # Module tree -> YAML (round-trips structurally)

dot = to_dot(chip)                   # Graphviz DOT string
visualize(chip, "chip.png")          # writes .dot, and .png if `dot` is installed
```

`dump` produces a fully-expanded spec — if you loaded a `count: 20` chip, the dump
contains 20 explicit AICore entries (the `count` shorthand is not reconstructed).
