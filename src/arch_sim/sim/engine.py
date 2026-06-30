"""The simulator: per-AICore in-order dispatcher + per-unit FIFOs + BAS backpressure.

Each instruction's `unit.parent` (the module containing the unit, e.g. an AICore)
owns a **dispatcher** that issues that group's instructions in program order into
the target unit's bounded inflight queue. When a target queue is full
(`queue_depth` instructions in-flight), the dispatcher stalls — and because issue
is sequential within a group, that stall **blocks subsequent issues to other
units too** (head-of-line blocking). Different AICores have independent
dispatchers, so they don't block each other.

Per unit, execution is **blocking-after-service** (BAS): the head of the inflight
queue runs to completion, then can only retire (and free its slot) when its
destination buffer (`dst`) has room. So a full destination throttles its producer
and backpressure propagates upstream.

There is no separate "unbounded" engine. To model an effectively-limitless
buffer, set a large `queue_depth` (`UNBOUNDED`). `dst=None` is a sink outside the
modeled region — capacity `UNBOUNDED`, never blocks.
"""
from __future__ import annotations

from collections import defaultdict

from arch_sim.arch import Module

from .instruction import Instruction

# Finite stand-in for "effectively unbounded". Hardware queues aren't infinite,
# so we use a very large depth rather than a literal infinity.
UNBOUNDED = 1 << 30


def _capacity(unit) -> int:
    if unit is None:
        return UNBOUNDED
    return getattr(unit, "queue_depth", 1)


def _has_free_slot(dst, occ: dict[int, int]) -> bool:
    if dst is None:
        return True  # sink outside the modeled region
    return occ[id(dst)] < _capacity(dst)


def simulate(instructions: list[Instruction]) -> float:
    """Run the sim. Fills start_time / end_time / retire_time in place; returns total cycles."""
    # Group instructions by dispatcher = unit.parent. Each group keeps its
    # program order (the order in which its instructions appeared in `instructions`).
    groups: dict[int, list[Instruction]] = defaultdict(list)
    group_name: dict[int, str] = {}
    for instr in instructions:
        parent = instr.unit.parent
        gid = id(parent) if parent is not None else 0
        groups[gid].append(instr)
        if gid not in group_name:
            group_name[gid] = parent.qualified_name() if parent is not None else "<root>"

    # Unique units in first-seen order.
    units: list = []
    seen: set[int] = set()
    for instr in instructions:
        if id(instr.unit) not in seen:
            seen.add(id(instr.unit))
            units.append(instr.unit)

    # Per-group dispatcher index.
    prog_idx: dict[int, int] = {g: 0 for g in groups}
    # Per-unit state.
    inflight: dict[int, list[Instruction]] = {id(u): [] for u in units}
    free_at: dict[int, float] = {id(u): 0.0 for u in units}
    occ: dict[int, int] = defaultdict(int)

    remaining = len(instructions)
    now = 0.0

    while remaining > 0:
        progress = True
        while progress:
            progress = False

            # 1) Retire: head of each unit's inflight queue if started + done + dst has room.
            for u in units:
                q = inflight[id(u)]
                if not q:
                    continue
                head = q[0]
                if (head.start_time is not None
                        and head.retire_time is None
                        and head.end_time <= now
                        and _has_free_slot(head.dst, occ)):
                    head.retire_time = now
                    if head.dst is not None:
                        occ[id(head.dst)] += 1
                    free_at[id(u)] = now
                    q.pop(0)
                    remaining -= 1
                    progress = True

            # 2) Start: head of each unit's inflight queue if not yet started, unit free,
            #    memory deps retired, and streamed producers already flowing.
            for u in units:
                q = inflight[id(u)]
                if not q or q[0].start_time is not None:
                    continue
                head = q[0]
                if free_at[id(u)] > now:
                    continue
                if not all(d.retire_time is not None and d.retire_time <= now for d in head.deps):
                    continue
                # A streamed producer need only have *started* (plus the link's fill
                # latency) — producer and consumer overlap instead of serializing.
                if not all(s.start_time is not None and s.start_time + head.stream_latency <= now
                           for s in head.stream_deps):
                    continue
                head.start_time = now
                # End is the later of (a) this unit's own execution and (b) being fed
                # by its slowest streamed producer (it can't finish before the producer
                # has streamed its last element, plus drain). end_time of a streamed
                # producer is known here because it was set when the producer started.
                end = now + head.cost()
                for s in head.stream_deps:
                    end = max(end, s.end_time + head.stream_latency)
                head.end_time = end
                for d in head.deps:  # reading inputs frees the producers' slots
                    if d.dst is not None:
                        occ[id(d.dst)] -= 1
                progress = True

            # 3) Issue: per dispatcher group, walk in program order while the target queue has room.
            for gid, prog in groups.items():
                while prog_idx[gid] < len(prog):
                    candidate = prog[prog_idx[gid]]
                    if len(inflight[id(candidate.unit)]) >= _capacity(candidate.unit):
                        break  # head-of-line: this group is stalled until the target retires
                    inflight[id(candidate.unit)].append(candidate)
                    prog_idx[gid] += 1
                    progress = True

        if remaining == 0:
            break

        # Advance to the next event: an execution-end among running heads, or the
        # moment a stalled head becomes eligible to start (e.g. a stream's fill
        # latency elapses). Without the latter, a sub-execution-step fill latency
        # would be rounded up to the next execution-end.
        future = []
        for u in units:
            q = inflight[id(u)]
            if not q:
                continue
            head = q[0]
            if head.start_time is not None:
                if head.retire_time is None and head.end_time > now:
                    future.append(head.end_time)
                continue
            # Not started: if it's blocked only on time (unit-free, deps retired,
            # streamed producers already flowing), schedule a wake at that time.
            if free_at[id(u)] > now and free_at[id(u)] not in (None,):
                future.append(free_at[id(u)])
            if (all(d.retire_time is not None for d in head.deps)
                    and all(s.start_time is not None for s in head.stream_deps)
                    and head.stream_deps):
                ready = max(s.start_time + head.stream_latency for s in head.stream_deps)
                if ready > now:
                    future.append(ready)
        if not future:
            stuck_units = [u.name for u in units if inflight[id(u)]]
            stuck_groups = [
                f"{group_name[gid]}@{prog_idx[gid]}/{len(prog)}"
                for gid, prog in groups.items()
                if prog_idx[gid] < len(prog)
            ]
            # Surface heads waiting on a streamed producer that never started
            # (e.g. a cyclic or mis-ordered stream edge).
            stream_waits = [
                head.label or head.unit.name
                for u in units
                for head in (inflight[id(u)][:1])
                if head.start_time is None
                and any(s.start_time is None for s in head.stream_deps)
            ]
            raise RuntimeError(
                f"deadlock: units={stuck_units} groups={stuck_groups} "
                f"stream_waits={stream_waits}"
            )
        now = min(future)

    return total_cycles(instructions)


def total_cycles(instructions: list[Instruction]) -> float:
    """Total simulated cycles = latest retirement (falling back to execution end)."""
    times = [
        i.retire_time if i.retire_time is not None else i.end_time
        for i in instructions
        if i.retire_time is not None or i.end_time is not None
    ]
    return max(times, default=0.0)


class Sim:
    def __init__(self, module: Module, instructions: list[Instruction]):
        self.module = module
        self.instructions = instructions

    def run(self) -> list[Instruction]:
        simulate(self.instructions)
        return self.instructions

    def total_cycles(self) -> float:
        return total_cycles(self.instructions)
