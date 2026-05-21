"""Storage units: scratchpads, buffers, register-file-like memories."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import BaseUnit, UnitKind


@dataclass(eq=False)
class StorageUnit(BaseUnit):
    """A memory buffer with a capacity, banking, and port budget."""

    KIND: ClassVar[UnitKind] = UnitKind.STORAGE

    capacity_bytes: int = 0
    banks: int = 1
    read_ports: int = 1
    write_ports: int = 1
    # How many live results this buffer can hold (slots) for backpressure modeling.
    queue_depth: int = 1

    def can_serve(self, num_reads: int, num_writes: int) -> bool:
        """True if the requested simultaneous accesses fit within the port budget."""
        if num_reads < 0 or num_writes < 0:
            raise ValueError("port counts must be non-negative")
        return num_reads <= self.read_ports and num_writes <= self.write_ports
