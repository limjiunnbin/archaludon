from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class TensorOp:
    """A high-level tensor operation (one linalg op). Operands/results are meta tensors."""

    kind: str  # "matmul" | "add"
    ins: list[torch.Tensor]
    out: torch.Tensor
    name: str = ""
    # Producer op for each input (None = a graph input, not produced by another op).
    in_sources: list[Optional["TensorOp"]] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
