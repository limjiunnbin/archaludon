from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from arch_sim.arch import ComputeUnit, Pipe
from arch_sim.cost import Move, Op


@dataclass
class Instruction:
    unit: Union[ComputeUnit, Pipe]
    tensor: torch.Tensor
    deps: list["Instruction"] = field(default_factory=list)
    label: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def cost(self) -> float:
        if isinstance(self.unit, Pipe):
            return Move(self.unit, self.tensor)
        return Op(self.unit, self.tensor)
