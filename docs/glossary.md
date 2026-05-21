# Glossary

Terms used across the wiki, the specs, and the code. The hardware terms follow
NPU / Ascend naming; the storage capacities and bandwidths quoted are from
`specs/npu.yaml` and are illustrative, not official.

## Hardware (modeled in the specs)

- **AICore** — one compute core: its own buffers (L1/L0/UB), compute units
  (Cube/Vector/Scalar), and movers (MTE1/2/3, FixPipe). Modeled as a nested
  `Module`. A chip has one or more (use `count: N` for many).
- **GM** — Global Memory. The large off-core memory shared by all AICores; the
  source/sink for data entering and leaving a core.
- **L1Buffer** — on-core L1 scratchpad; staging area between GM and the L0 buffers.
- **L0A / L0B** — input buffers feeding the Cube (the matmul operands).
- **L0C** — the Cube's accumulator / output buffer.
- **UB** — Unified Buffer; the working memory for the Vector and Scalar units.
- **Cube** — the matrix-multiply unit (systolic array). High throughput; `matmul`.
- **Vector** — the SIMD vector ALU for elementwise ops, reductions, activations.
- **Scalar** — the scalar unit; low throughput, used for control/address math.
- **MTE1 / MTE2 / MTE3** — Memory Transfer Engines (DMA movers). Roughly: MTE2
  GM→on-core, MTE1 L1→L0, MTE3 on-core→GM. Modeled as `Pipe`s.
- **FixPipe** — the post-Cube mover that writes L0C out to UB or GM (handles the
  fixup/requantize path in real hardware). Modeled as a `Pipe`.

## Architecture model (the DSL)

- **Module** — a container of child units and the data paths between them. The
  chip and each AICore are modules. See [Architecture model](architecture-model.md).
- **Unit** — any architectural element (`BaseUnit`): storage, compute, control,
  pipe, or a nested module.
- **Pipe** — a queued data mover with a bandwidth, a queue depth, and constraints
  on which `(src, dst)` it can carry. Generalizes the MTEs and FixPipe.
- **DataPath** — a connection between two units. A *wire* if it has no engine; a
  *pipe-carried* transfer if it references a `Pipe`.
- **kind** — the unit's type tag (`storage`, `compute`, `control`, `pipe`,
  `module`), used both in YAML and for pipe validation.

## Simulator

- **Instruction** — one unit of work: a target unit, a tensor, dependencies, and
  start/end times. See [Simulator](simulator.md).
- **Channel** — a per-unit FIFO of instructions. Work within a channel is serial.
- **dep** — a dependency: another instruction whose output this one needs. Drives
  cross-channel ordering.
- **cycle** — the simulator's time unit. Costs come from the
  [cost model](cost-model.md); total cycles = `max(end_time)` over all instructions.
- **meta tensor** — a `torch.Tensor` on the `meta` device: shape and dtype with no
  storage. Used purely as a workload descriptor.
- **bandwidth** — a pipe's transfer rate in **bytes per cycle**.
- **throughput** — a compute unit's rate in **elements per cycle**
  (`throughput_ops_per_cycle`).

## Backpressure (the `sim/backpressure.py` engine)

- **backpressure** — a slow consumer forcing its producer to stall once the buffer
  between them fills. Propagates one hop upstream at a time.
- **queue_depth** — a unit's bounded buffer/queue capacity in slots. Used by the
  backpressure engine; ignored by the default `Sim`.
- **destination / `dst`** — the buffer an instruction's output lands in. `None` =
  an unbounded sink (no backpressure on that output).
- **retire** — when an instruction leaves its unit: execution done **and** its
  destination has a free slot. `retire_time` records it; total cycles =
  `max(retire_time)`.
- **blocking-after-service (BAS)** — the discipline the engine uses: a unit finishes
  executing but stays blocked (can't start the next instruction) until the current
  one can retire.
