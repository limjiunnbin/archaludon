"""Architecture DSL: describe accelerators declaratively in Python or YAML."""
from .base import BaseUnit, Module, UnitKind
from .compute import ComputeUnit
from .control import ControlUnit
from .dma import DataPath, Direction, DMAEngine
from .loader import dump, load, loads
from .storage import StorageUnit
from .validator import ValidationReport, validate
from .visualize import to_dot, visualize

__all__ = [
    "BaseUnit",
    "Module",
    "UnitKind",
    "StorageUnit",
    "ComputeUnit",
    "ControlUnit",
    "DMAEngine",
    "DataPath",
    "Direction",
    "ValidationReport",
    "load",
    "loads",
    "dump",
    "validate",
    "to_dot",
    "visualize",
]
