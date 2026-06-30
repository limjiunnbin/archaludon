"""YAML loading and dumping. Round-trips a Module tree via dotted-path references."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import BaseUnit, Module
from .builder import build
from .compute import ComputeUnit
from .control import ControlUnit
from .pipe import Pipe
from .storage import StorageUnit


def load(path: str | Path) -> Module:
    """Load a Module tree from a YAML file."""
    return loads(Path(path).read_text())


def loads(text: str) -> Module:
    """Load a Module tree from a YAML string."""
    spec = yaml.safe_load(text)
    if not isinstance(spec, dict):
        raise ValueError("top-level YAML must be a mapping")
    return build(spec)


def dump(root: Module) -> str:
    """Serialize a Module tree to a YAML string."""
    text: str = yaml.safe_dump(_module_to_dict(root, root), sort_keys=False)
    return text


def _qualified_from(root: Module, target: BaseUnit) -> str:
    parts: list[str] = []
    cur: BaseUnit | None = target
    while cur is not None and cur is not root:
        parts.append(cur.name)
        cur = cur.parent
    if cur is None:
        raise ValueError(f"target {target.qualified_name()!r} is not under root {root.name!r}")
    return ".".join(reversed(parts))


def _module_to_dict(mod: Module, root: Module) -> dict[str, Any]:
    out: dict[str, Any] = {"name": mod.name, "kind": "module"}
    if mod.children:
        out["children"] = [_unit_to_dict(c, root) for c in mod.children]
    if mod.paths:
        paths: list[dict[str, Any]] = []
        for p in mod.paths:
            entry: dict[str, Any] = {
                "src": _qualified_from(root, p.src),
                "dst": _qualified_from(root, p.dst),
            }
            if p.direction.value != "uni":
                entry["direction"] = p.direction.value
            if p.engine is not None:
                entry["engine"] = _qualified_from(root, p.engine)
            if p.bandwidth:
                entry["bandwidth"] = p.bandwidth
            if p.stream:
                entry["stream"] = p.stream
            if p.stream_latency:
                entry["stream_latency"] = p.stream_latency
            if p.fifo_depth != 1:
                entry["fifo_depth"] = p.fifo_depth
            if p.name:
                entry["name"] = p.name
            paths.append(entry)
        out["paths"] = paths
    return out


def _unit_to_dict(u: BaseUnit, root: Module) -> dict[str, Any]:
    if isinstance(u, Module):
        return _module_to_dict(u, root)
    d: dict[str, Any] = {"name": u.name, "kind": u.kind.value}
    if isinstance(u, StorageUnit):
        d["capacity_bytes"] = u.capacity_bytes
        d["banks"] = u.banks
        d["read_ports"] = u.read_ports
        d["write_ports"] = u.write_ports
        if u.queue_depth != 1:
            d["queue_depth"] = u.queue_depth
    elif isinstance(u, ComputeUnit):
        d["operation"] = u.operation
        d["throughput_ops_per_cycle"] = u.throughput_ops_per_cycle
        if u.operand_shape is not None:
            d["operand_shape"] = list(u.operand_shape)
        if u.queue_depth != 1:
            d["queue_depth"] = u.queue_depth
    elif isinstance(u, ControlUnit):
        d["instruction_queue_depth"] = u.instruction_queue_depth
    elif isinstance(u, Pipe):
        d["allowed_src_kinds"] = [k.value for k in u.allowed_src_kinds]
        d["allowed_dst_kinds"] = [k.value for k in u.allowed_dst_kinds]
        if u.allowed_src_names is not None:
            d["allowed_src_names"] = list(u.allowed_src_names)
        if u.allowed_dst_names is not None:
            d["allowed_dst_names"] = list(u.allowed_dst_names)
        if u.bandwidth:
            d["bandwidth"] = u.bandwidth
        if u.queue_depth != 1:
            d["queue_depth"] = u.queue_depth
    return d
