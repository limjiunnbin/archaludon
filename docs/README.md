# archaludon wiki

archaludon is a cycle simulator for Ascend-like NPU architectures. You
describe a chip declaratively (memories, compute units, and queued data movers),
hand the simulator a list of instructions with dependencies, and it reports when
each instruction runs and how many cycles the whole thing takes.

The mental model has three layers: an **architecture** (a tree of units connected
by data paths), a **cost model** (two functions that price a unit of work in
cycles), and a **simulator** (one FIFO channel per unit, advanced by a
dependency-aware event walk).

## Pages

- **[Getting started](getting-started.md)** — install, run an example, run tests.
- **[Architecture model](architecture-model.md)** — the DSL: modules, units, pipes, and data paths.
- **[Chip specs](chip-specs.md)** — the YAML format, every field, and `count: N` for multi-core chips.
- **[Cost model](cost-model.md)** — how compute and movement cycles are computed.
- **[Simulator](simulator.md)** — instructions, channels, the engine loop, and what is (and isn't) modeled.
- **[Examples](examples.md)** — a walkthrough of every script in `examples/`, with traces.
- **[Glossary](glossary.md)** — NPU terminology and simulator terms.

## Related files at the repo root

- **[README.md](../README.md)** — the project's top-level readme.
- **[PLAN.md](../PLAN.md)** — implementation roadmap and open design decisions for the unbuilt parts (`ir/`, `frontend/`, `report/`).
- **[CLAUDE.md](../CLAUDE.md)** — orientation for contributors and coding agents.
