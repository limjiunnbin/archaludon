"""Core architecture DSL primitives: UnitKind, BaseUnit, Module."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Optional

if TYPE_CHECKING:
    from .compute import ComputeUnit
    from .control import ControlUnit
    from .dma import DataPath, DMAEngine, Direction
    from .storage import StorageUnit


class UnitKind(Enum):
    STORAGE = "storage"
    COMPUTE = "compute"
    CONTROL = "control"
    DMA = "dma"
    EXTERNAL = "external"
    MODULE = "module"


@dataclass(eq=False)
class BaseUnit:
    """An addressable architectural element. Subclasses set KIND."""

    KIND: ClassVar[UnitKind] = UnitKind.EXTERNAL

    name: str
    parent: Optional["Module"] = field(default=None, init=False, repr=False)

    @property
    def kind(self) -> UnitKind:
        """The unit's kind (read from the subclass ClassVar)."""
        return type(self).KIND

    def qualified_name(self) -> str:
        """Dotted path from the root module down to this unit."""
        parts = [self.name]
        p = self.parent
        while p is not None:
            parts.append(p.name)
            p = p.parent
        return ".".join(reversed(parts))

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


@dataclass(eq=False)
class Module(BaseUnit):
    """A container of child units and the DataPaths that connect them."""

    KIND: ClassVar[UnitKind] = UnitKind.MODULE

    children: list[BaseUnit] = field(default_factory=list)
    paths: list["DataPath"] = field(default_factory=list)

    def add_child(self, unit: BaseUnit) -> BaseUnit:
        """Attach a pre-built unit as a child of this module."""
        unit.parent = self
        self.children.append(unit)
        return unit

    def add_module(self, name: str) -> "Module":
        return self.add_child(Module(name=name))  # type: ignore[return-value]

    def add_storage(self, name: str, **kwargs: Any) -> "StorageUnit":
        from .storage import StorageUnit

        return self.add_child(StorageUnit(name=name, **kwargs))  # type: ignore[return-value]

    def add_compute(self, name: str, **kwargs: Any) -> "ComputeUnit":
        from .compute import ComputeUnit

        return self.add_child(ComputeUnit(name=name, **kwargs))  # type: ignore[return-value]

    def add_control(self, name: str, **kwargs: Any) -> "ControlUnit":
        from .control import ControlUnit

        return self.add_child(ControlUnit(name=name, **kwargs))  # type: ignore[return-value]

    def add_dma(self, name: str, **kwargs: Any) -> "DMAEngine":
        from .dma import DMAEngine

        return self.add_child(DMAEngine(name=name, **kwargs))  # type: ignore[return-value]

    def connect(
        self,
        src: BaseUnit,
        dst: BaseUnit,
        *,
        engine: Optional["DMAEngine"] = None,
        direction: Optional["Direction"] = None,
        bandwidth: float = 0.0,
        name: Optional[str] = None,
    ) -> "DataPath":
        """Build a DataPath between two units. Raises on DMA kind/name violation."""
        from .dma import DataPath, Direction

        if direction is None:
            direction = Direction.UNI
        path = DataPath(
            src=src,
            dst=dst,
            direction=direction,
            bandwidth=bandwidth,
            engine=engine,
            name=name,
        )
        if engine is not None:
            ok, reason = engine.validate(path)
            if not ok:
                raise ValueError(f"DMA validation failed: {reason}")
        self.paths.append(path)
        return path

    def walk(self) -> Iterator[BaseUnit]:
        """Yield this module and all descendants, depth-first."""
        yield self
        for c in self.children:
            if isinstance(c, Module):
                yield from c.walk()
            else:
                yield c

    def find(self, dotted: str) -> BaseUnit:
        """Resolve a dotted name relative to this module."""
        cur: BaseUnit = self
        for part in dotted.split("."):
            if not isinstance(cur, Module):
                raise KeyError(f"cannot descend into non-module {cur.name!r}")
            for child in cur.children:
                if child.name == part:
                    cur = child
                    break
            else:
                raise KeyError(f"no child named {part!r} under {cur.name!r}")
        return cur
