"""Backpressure-aware engine: bounded destination buffers, blocking-after-service.

An instruction retires only when its execution is done AND its destination buffer
(`dst`) has a free slot. A unit is held until its instruction retires (BAS), so a
full destination throttles the producer. `dst=None` is an unbounded sink.

This is a separate engine from `engine.run` — that one ignores `dst`/capacity.
"""
from __future__ import annotations

from collections import defaultdict

from .instruction import Instruction


def _capacity(unit) -> float:
    return getattr(unit, "queue_depth", 1)


def simulate(instructions: list[Instruction]) -> float:
    """Run with backpressure. Fills start_time / end_time / retire_time; returns total cycles."""
    prog: dict[int, list[Instruction]] = defaultdict(list)
    units: list = []
    for instr in instructions:
        if id(instr.unit) not in prog:
            units.append(instr.unit)
        prog[id(instr.unit)].append(instr)

    idx = {id(u): 0 for u in units}
    free_at = {id(u): 0.0 for u in units}
    executing: dict[int, Instruction | None] = {id(u): None for u in units}
    occ: dict[int, int] = defaultdict(int)

    remaining = len(instructions)
    now = 0.0

    while remaining > 0:
        progress = True
        while progress:
            progress = False
            for u in units:
                instr = executing[id(u)]
                if instr is not None and instr.retire_time is None and instr.end_time <= now:
                    if instr.dst is None or occ[id(instr.dst)] < _capacity(instr.dst):
                        instr.retire_time = now
                        if instr.dst is not None:
                            occ[id(instr.dst)] += 1
                        free_at[id(u)] = now
                        executing[id(u)] = None
                        remaining -= 1
                        progress = True
            for u in units:
                if executing[id(u)] is None and idx[id(u)] < len(prog[id(u)]):
                    instr = prog[id(u)][idx[id(u)]]
                    deps_ready = all(
                        d.retire_time is not None and d.retire_time <= now for d in instr.deps
                    )
                    if free_at[id(u)] <= now and deps_ready:
                        instr.start_time = now
                        instr.end_time = now + instr.cost()
                        for d in instr.deps:
                            if d.dst is not None:
                                occ[id(d.dst)] -= 1
                        executing[id(u)] = instr
                        idx[id(u)] += 1
                        progress = True

        if remaining == 0:
            break

        future = [
            executing[id(u)].end_time
            for u in units
            if executing[id(u)] is not None
            and executing[id(u)].retire_time is None
            and executing[id(u)].end_time > now
        ]
        if not future:
            stuck = [
                u.name
                for u in units
                if executing[id(u)] is not None or idx[id(u)] < len(prog[id(u)])
            ]
            raise RuntimeError(f"backpressure deadlock: units stuck: {stuck}")
        now = min(future)

    return max((i.retire_time for i in instructions if i.retire_time is not None), default=0.0)
