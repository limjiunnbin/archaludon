"""Pipes: queued movers (DMA-style engines and FixPipe) that carry DataPaths."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Optional

from .base import BaseUnit, UnitKind


class Direction(Enum):
    """Whether a DataPath flows one way or both."""

    UNI = "uni"
    BI = "bi"


@dataclass(eq=False)
class DataPath:
    """A connection between two units, optionally carried by a Pipe."""

    src: BaseUnit
    dst: BaseUnit
    direction: Direction = Direction.UNI
    bandwidth: float = 0.0
    engine: Optional["Pipe"] = None
    name: Optional[str] = None

    def endpoints(self) -> tuple[BaseUnit, BaseUnit]:
        return self.src, self.dst


@dataclass(eq=False)
class Pipe(BaseUnit):
    """A queued mover (MTE1/2/3, FixPipe, …) that constrains which (src, dst) units a DataPath can join."""

    KIND: ClassVar[UnitKind] = UnitKind.PIPE

    allowed_src_kinds: list[UnitKind] = field(default_factory=list)
    allowed_dst_kinds: list[UnitKind] = field(default_factory=list)
    allowed_src_names: Optional[list[str]] = None
    allowed_dst_names: Optional[list[str]] = None
    bandwidth: float = 0.0
    queue_depth: int = 1

    def validate(self, path: DataPath) -> tuple[bool, str]:
        """Check whether this pipe can legally carry the given DataPath."""
        if self.allowed_src_kinds and path.src.kind not in self.allowed_src_kinds:
            return False, f"src kind {path.src.kind.value!r} not allowed by {self.name}"
        if self.allowed_dst_kinds and path.dst.kind not in self.allowed_dst_kinds:
            return False, f"dst kind {path.dst.kind.value!r} not allowed by {self.name}"
        if self.allowed_src_names is not None and path.src.name not in self.allowed_src_names:
            return False, f"src {path.src.name!r} not in allowed_src_names of {self.name}"
        if self.allowed_dst_names is not None and path.dst.name not in self.allowed_dst_names:
            return False, f"dst {path.dst.name!r} not in allowed_dst_names of {self.name}"
        return True, ""
