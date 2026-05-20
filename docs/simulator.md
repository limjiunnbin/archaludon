# Simulator

The simulator lives in `src/arch_sim/sim/`. It models a chip as a set of FIFO
channels — one per pipe and compute unit — and advances them with a
dependency-aware event walk. Schedules are built by hand: you construct the
instruction list and their dependencies yourself (there's no scheduler).

## Instruction

`Instruction` (`sim/instruction.py`) is one unit of work:

```python
@dataclass
class Instruction:
    unit: ComputeUnit | Pipe
    tensor: torch.Tensor
    deps: list[Instruction] = []
    label: str = ""
    start_time: float | None = None   # filled in by the sim
    end_time: float | None = None     # filled in by the sim

    def cost(self) -> float:
        # Pipe -> cost.Move, ComputeUnit -> cost.Op
```

`deps` are other instructions whose output this one consumes. `start_time` and
`end_time` are blank until the instruction runs.

## Channel

`Channel` (`sim/channel.py`) is a FIFO bound to one unit, plus `last_end` (the
cycle that unit last finished work):

- `enqueue(instr)` — append to the queue.
- `head_ready_at()` — returns `None` if the queue is empty *or* the head has a dep
  whose `end_time` is still unset; otherwise the tentative start,
  `max(last_end, max(dep.end_time))`.
- `pop()` — finalize the head: `start_time = max(last_end, dep_end)`,
  `end_time = start_time + cost()`, advance `last_end`, return it.

A single channel is strictly sequential: `last_end` chains forward, so the next
instruction on a unit can't begin before the previous one ends.

## The engine loop

`run(channels)` (`sim/engine.py`) repeats:

1. Ask every channel `head_ready_at()`.
2. Drop the ones that return `None` (empty or blocked on deps).
3. If none remain: if any channel still has queued work, raise a **deadlock**
   error; otherwise return the trace.
4. Otherwise sort by tentative start and `pop()` the smallest.

Picking the earliest-startable head each step makes this equivalent to a
discrete-event simulation, without maintaining a global event queue. Two facts
make it correct: ordering *within* a unit comes from the channel FIFO, and
ordering *across* units comes from `Instruction.deps`.

`total_cycles(trace)` returns `max(end_time)`.

## Sim

`Sim` (`sim/engine.py`) is the wrapper you use:

```python
sim = Sim(chip, [move_a, move_b, add, store_c])
sim.run()                  # groups instructions into per-unit channels, runs the engine
sim.total_cycles()         # max end_time
```

`run()` groups instructions by `id(instr.unit)` into one `Channel` per unit. The
order of the instruction list becomes the FIFO order on each unit, so put a unit's
instructions in program order.

## Pipelining: across pipes, not within one

Independent channels advance independently, so the simulator overlaps work on
different units for free. But two instructions on the *same* pipe serialize,
because a pipe has one FIFO.

Two unrelated additions (`c = a + b`, `f = d + e`) on a single AICore illustrate
both: the four loads share the one MTE2 pipe and run back-to-back, while the adds
(Vector) and stores (MTE3) overlap with later loads.

```
MTE2:   [a 0-64][b 64-128][d 128-192][e 192-256]      ← single load pipe, the bottleneck
Vector:                   [c 128-136]      [f 256-264]
MTE3:                          [store c 136-200]   [store f 264-328]
```

`load d` starts at 128 — while `c = a + b` is still computing and before `c` is
stored — because it has no dependency on them, only on MTE2 being free. To make
the two additions run *fully* in parallel (loads included) you need two load
pipes, i.e. two AICores. See [run_parallel_add_2core](examples.md#run_parallel_add_2corepy).

## What is and isn't modeled

Modeled:
- Sequential execution within a unit, parallel execution across units.
- Data dependencies via `deps`.
- Cycle costs from the [cost model](cost-model.md).
- Deadlock detection (queued work with no runnable head).

Not modeled (see PLAN.md open decisions):
- **Intra-pipe contention** beyond the FIFO — no fair-share bandwidth splitting, no
  multiple physical DMA channels per pipe.
- **Shared-resource contention** — e.g. GM bandwidth shared across cores; each pipe
  is independent.
- **`queue_depth` / backpressure** — channels are unbounded; `queue_depth` is
  metadata. A full-queue stall on the issue side isn't simulated.
- **Flag synchronization** (`SET_FLAG` / `WAIT_FLAG`) — ordering is via `deps`;
  hardware-style flags would be a later trace-fidelity feature.
