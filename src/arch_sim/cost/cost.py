"""Cost primitives: cycles for compute ops and pipe moves."""
from __future__ import annotations

import torch

from arch_sim.arch import ComputeUnit, Pipe


def Op(unit: ComputeUnit, t: torch.Tensor) -> float:
    tile = unit.operand_shape
    if tile is None:
        work = t.numel()
    else:
        if t.dim() != len(tile):
            raise ValueError(
                f"{unit.name}: operand rank {t.dim()} != operand_shape rank {len(tile)}"
            )
        work = 1
        for dim, ts in zip(t.shape, tile):
            work *= ((dim + ts - 1) // ts) * ts  # pad dim up to the tile boundary
    return work / unit.throughput_ops_per_cycle


def Move(pipe: Pipe, t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / pipe.bandwidth
