"""Whole-system architecture checks. Returns a ValidationReport, never raises."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .base import BaseUnit, Module, UnitKind
from .storage import StorageUnit


@dataclass
class ValidationReport:
    """Result of validating a Module tree."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok


def validate(root: Module) -> ValidationReport:
    """Run all structural checks against a Module tree."""
    report = ValidationReport()
    units = set(_walk(root))
    _check_name_collisions(root, report)
    _check_pipe_paths(root, units, report)
    _check_orphans(root, report)
    _check_port_oversubscription(root, report)
    return report


def _walk(mod: Module) -> Iterator[BaseUnit]:
    yield mod
    for c in mod.children:
        if isinstance(c, Module):
            yield from _walk(c)
        else:
            yield c


def _modules(root: Module) -> Iterator[Module]:
    yield root
    for c in root.children:
        if isinstance(c, Module):
            yield from _modules(c)


def _check_name_collisions(root: Module, report: ValidationReport) -> None:
    for mod in _modules(root):
        seen: set[str] = set()
        for c in mod.children:
            if c.name in seen:
                report.errors.append(
                    f"name collision in {mod.qualified_name()!r}: {c.name!r}"
                )
            seen.add(c.name)


def _check_pipe_paths(root: Module, units: set[BaseUnit], report: ValidationReport) -> None:
    for mod in _modules(root):
        for path in mod.paths:
            label = path.name or f"{path.src.name}->{path.dst.name}"
            if path.src not in units:
                report.errors.append(f"path {label!r}: src not in module tree")
            if path.dst not in units:
                report.errors.append(f"path {label!r}: dst not in module tree")
            if path.engine is not None:
                if path.engine not in units:
                    report.errors.append(f"path {label!r}: engine not in module tree")
                else:
                    ok, reason = path.engine.validate(path)
                    if not ok:
                        report.errors.append(f"path {label!r}: {reason}")


def _check_orphans(root: Module, report: ValidationReport) -> None:
    """Storage/compute units that no DataPath touches are likely a spec mistake."""
    referenced: set[BaseUnit] = set()
    for mod in _modules(root):
        for path in mod.paths:
            referenced.add(path.src)
            referenced.add(path.dst)
    for unit in _walk(root):
        if unit.kind in (UnitKind.STORAGE, UnitKind.COMPUTE) and unit not in referenced:
            report.warnings.append(
                f"unit {unit.qualified_name()} is not referenced by any DataPath"
            )


def _check_port_oversubscription(root: Module, report: ValidationReport) -> None:
    """Best-effort: warn if a storage unit shows up as src/dst on more paths than ports."""
    src_count: dict[BaseUnit, int] = {}
    dst_count: dict[BaseUnit, int] = {}
    for mod in _modules(root):
        for path in mod.paths:
            src_count[path.src] = src_count.get(path.src, 0) + 1
            dst_count[path.dst] = dst_count.get(path.dst, 0) + 1
    for unit, n in src_count.items():
        if isinstance(unit, StorageUnit) and n > unit.read_ports:
            report.warnings.append(
                f"{unit.qualified_name()}: {n} outgoing paths exceed {unit.read_ports} read ports"
            )
    for unit, n in dst_count.items():
        if isinstance(unit, StorageUnit) and n > unit.write_ports:
            report.warnings.append(
                f"{unit.qualified_name()}: {n} incoming paths exceed {unit.write_ports} write ports"
            )
