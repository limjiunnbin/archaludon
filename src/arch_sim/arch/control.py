"""Control units: instruction fetch/issue logic. Minimal for now; extend later."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import BaseUnit, UnitKind


@dataclass(eq=False)
class ControlUnit(BaseUnit):
    """Issues instructions to compute and DMA units."""

    KIND: ClassVar[UnitKind] = UnitKind.CONTROL

    instruction_queue_depth: int = 1
