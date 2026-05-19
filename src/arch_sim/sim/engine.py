from __future__ import annotations

from arch_sim.arch import Module

from .channel import Channel
from .instruction import Instruction


def run(channels: list[Channel]) -> list[Instruction]:
    """Walk channels, popping the head with smallest tentative start_time each step."""
    trace: list[Instruction] = []
    while True:
        ready = [(ch.head_ready_at(), ch) for ch in channels]
        ready = [(t, ch) for t, ch in ready if t is not None]
        if not ready:
            stuck = [ch.unit.name for ch in channels if ch.queue]
            if stuck:
                raise RuntimeError(f"deadlock: channels stuck waiting on deps: {stuck}")
            return trace
        ready.sort(key=lambda pair: pair[0])
        trace.append(ready[0][1].pop())


def total_cycles(trace: list[Instruction]) -> float:
    return max((i.end_time for i in trace if i.end_time is not None), default=0.0)


class Sim:
    def __init__(self, module: Module, instructions: list[Instruction]):
        self.module = module
        self.instructions = instructions

    def run(self) -> list[Instruction]:
        channels: dict[int, Channel] = {}
        for instr in self.instructions:
            key = id(instr.unit)
            if key not in channels:
                channels[key] = Channel(instr.unit)
            channels[key].enqueue(instr)
        return run(list(channels.values()))

    def total_cycles(self) -> float:
        return total_cycles(self.instructions)
