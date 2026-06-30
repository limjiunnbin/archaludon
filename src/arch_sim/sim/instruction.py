from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from arch_sim.arch import BaseUnit, ComputeUnit, Pipe
from arch_sim.cost import Move, Op


@dataclass
class Instruction:
    unit: Union[ComputeUnit, Pipe]
    tensor: torch.Tensor
    deps: list["Instruction"] = field(default_factory=list)
    label: str = ""
    # Destination buffer the output lands in. None = sink outside the modeled region.
    dst: Optional[BaseUnit] = None
    # Streamed producers feeding this instruction directly (compute->compute), with
    # no write-back to internal memory. Unlike `deps` (which gate on full retirement),
    # a stream_dep only needs to have *started* (plus `stream_latency`) before this
    # instruction can begin, so producer and consumer overlap. See docs/streaming-plan.md.
    stream_deps: list["Instruction"] = field(default_factory=list)
    # Pipeline fill/drain latency on the streaming link, in cycles (0 = ideal overlap).
    stream_latency: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None  # execution end (start + cost)
    retire_time: Optional[float] = None  # execution done + destination ready

    def cost(self) -> float:
        if isinstance(self.unit, Pipe):
            return Move(self.unit, self.tensor)
        return Op(self.unit, self.tensor)
