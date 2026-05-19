"""Cost primitives: cycles for compute ops and pipe moves."""
from __future__ import annotations

import torch

from arch_sim.arch import ComputeUnit, Pipe


def Op(unit: ComputeUnit, t: torch.Tensor) -> float:
    return t.numel() / unit.throughput_ops_per_cycle


def Move(pipe: Pipe, t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / pipe.bandwidth
