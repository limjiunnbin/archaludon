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
    start_time: Optional[float] = None
    end_time: Optional[float] = None  # execution end (start + cost)
    retire_time: Optional[float] = None  # execution done + destination ready

    def cost(self) -> float:
        if isinstance(self.unit, Pipe):
            return Move(self.unit, self.tensor)
        return Op(self.unit, self.tensor)
