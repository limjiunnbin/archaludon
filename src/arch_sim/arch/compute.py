"""Compute units: matmul cubes, vector ALUs, scalar units."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from .base import BaseUnit, UnitKind


@dataclass(eq=False)
class ComputeUnit(BaseUnit):
    """A functional unit that consumes operands and produces results."""

    KIND: ClassVar[UnitKind] = UnitKind.COMPUTE

    operation: str = "generic"
    throughput_ops_per_cycle: float = 1.0
    # Fixed operand shape for SIMD-like units (e.g. (16,16,16) for a Cube). None = flexible.
    operand_shape: Optional[tuple[int, ...]] = None
    queue_depth: int = 1
