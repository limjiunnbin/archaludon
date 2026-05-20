# Getting started

## Prerequisites

- Python 3.10 or newer.
- `torch` (a runtime dependency). It's used only as a shape/dtype descriptor via
  meta tensors — no tensor data is ever allocated. See [Cost model](cost-model.md).

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

The `-e` (editable) install puts `arch_sim` on the path so the examples and tests
resolve it from any directory. Note that the system `python3` will *not* see the
package — always use `.venv/bin/python`.

## Run an example

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

See [Examples](examples.md) for a walkthrough of every script.

## Run the tests

```bash
.venv/bin/pytest tests/
```

Run a single test with `-k`:

```bash
.venv/bin/pytest tests/sim/test_engine.py -k c_equals_a_plus_b
```

## Visualize a chip

```bash
.venv/bin/python examples/visualize_npu.py
```

This writes a Graphviz `.dot` file, and a `.png` too if the `dot` binary is on
your `PATH`. If it isn't, paste the `.dot` contents into
<https://dreampuf.github.io/GraphvizOnline/>. Pipes render as edge labels, not
nodes. See [Chip specs](chip-specs.md) for the visualization API.

## A note on the NumPy warning

`torch` prints `Failed to initialize NumPy` if NumPy isn't installed. It's
harmless — nothing in archaludon uses NumPy. Install it (`pip install numpy`) to
silence the warning if you like.
