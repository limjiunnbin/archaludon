from __future__ import annotations

from typing import Optional, Union

from arch_sim.arch import ComputeUnit, Pipe

from .instruction import Instruction

class Channel:
    def __init__(self, unit: Union[ComputeUnit, Pipe]):
        self.unit = unit
        self.queue: list[Instruction] = []
        self.last_end: float = 0.0

    def enqueue(self, instr: Instruction) -> None:
        self.queue.append(instr)

    def head_ready_at(self) -> Optional[float]:
        if not self.queue:
            return None
        instr = self.queue[0]
        for d in instr.deps:
            if d.end_time is None:
                return None
        dep_end = max((d.end_time for d in instr.deps), default=0.0)
        return max(self.last_end, dep_end)

    def pop(self) -> Instruction:
        instr = self.queue.pop(0)
        dep_end = max((d.end_time for d in instr.deps), default=0.0)
        instr.start_time = max(self.last_end, dep_end)
        instr.end_time = instr.start_time + instr.cost()
        self.last_end = instr.end_time
        return instr
