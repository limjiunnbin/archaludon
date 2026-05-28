from __future__ import annotations

from dataclasses import dataclass, field

from .ops import TensorOp


@dataclass
class Graph:
    """An ordered list of TensorOps (already in a valid execution order)."""

    ops: list[TensorOp] = field(default_factory=list)

    def add(self, op: TensorOp) -> TensorOp:
        self.ops.append(op)
        return op
