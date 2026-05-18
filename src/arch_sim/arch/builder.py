"""Declarative builder: turn a nested-dict spec into a Module tree.

The fluent API (Module.add_storage, Module.connect, etc.) lives on Module itself.
This file handles dict-based construction, which is what the YAML loader sits on
top of.
"""
from __future__ import annotations

from typing import Any, Optional

from .base import BaseUnit, Module, UnitKind
from .dma import Direction, DMAEngine


def build(spec: dict[str, Any]) -> Module:
    """Construct a Module tree from a parsed-YAML-style dict."""
    if spec.get("kind", "module") != "module":
        raise ValueError("top-level spec must be a module")
    root = _build_units(spec)
    _build_paths(spec, root, current=root)
    return root


def _build_units(spec: dict[str, Any]) -> Module:
    mod = Module(name=spec["name"])
    for child in spec.get("children", []):
        kind = child.get("kind", "module")
        if kind == "module":
            mod.add_child(_build_units(child))
        elif kind == "storage":
            mod.add_storage(
                child["name"],
                capacity_bytes=child.get("capacity_bytes", 0),
                banks=child.get("banks", 1),
                read_ports=child.get("read_ports", 1),
                write_ports=child.get("write_ports", 1),
            )
        elif kind == "compute":
            shape = child.get("operand_shape")
            mod.add_compute(
                child["name"],
                operation=child.get("operation", "generic"),
                throughput_ops_per_cycle=child.get("throughput_ops_per_cycle", 1.0),
                operand_shape=tuple(shape) if shape is not None else None,
            )
        elif kind == "control":
            mod.add_control(
                child["name"],
                instruction_queue_depth=child.get("instruction_queue_depth", 1),
            )
        elif kind == "dma":
            mod.add_dma(
                child["name"],
                allowed_src_kinds=[UnitKind(k) for k in child.get("allowed_src_kinds", [])],
                allowed_dst_kinds=[UnitKind(k) for k in child.get("allowed_dst_kinds", [])],
                allowed_src_names=list(child["allowed_src_names"]) if "allowed_src_names" in child else None,
                allowed_dst_names=list(child["allowed_dst_names"]) if "allowed_dst_names" in child else None,
                bandwidth=child.get("bandwidth", 0.0),
            )
        else:
            raise ValueError(f"unknown unit kind: {kind!r}")
    return mod


def _build_paths(spec: dict[str, Any], root: Module, current: Module) -> None:
    for child in spec.get("children", []):
        if child.get("kind", "module") == "module":
            child_mod = current.find(child["name"])
            assert isinstance(child_mod, Module)
            _build_paths(child, root, child_mod)
    for p in spec.get("paths", []):
        src = _resolve(root, current, p["src"])
        dst = _resolve(root, current, p["dst"])
        engine: Optional[DMAEngine] = None
        if "engine" in p and p["engine"] is not None:
            resolved = _resolve(root, current, p["engine"])
            if not isinstance(resolved, DMAEngine):
                raise ValueError(f"engine {p['engine']!r} is not a DMAEngine")
            engine = resolved
        current.connect(
            src,
            dst,
            engine=engine,
            direction=Direction(p.get("direction", "uni")),
            bandwidth=p.get("bandwidth", 0.0),
            name=p.get("name"),
        )


def _resolve(root: Module, current: Module, name: str) -> BaseUnit:
    """Look up a unit by name: first as a child of `current`, then dotted from root."""
    if "." not in name:
        for c in current.children:
            if c.name == name:
                return c
    return root.find(name)
