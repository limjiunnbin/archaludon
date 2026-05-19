"""Declarative builder: turn a nested-dict spec into a Module tree.

The fluent API (Module.add_storage, Module.connect, etc.) lives on Module itself.
This file handles dict-based construction, which is what the YAML loader sits on
top of.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

from .base import BaseUnit, Module, UnitKind
from .pipe import Direction, Pipe


def build(spec: dict[str, Any]) -> Module:
    """Construct a Module tree from a parsed-YAML-style dict."""
    if spec.get("kind", "module") != "module":
        raise ValueError("top-level spec must be a module")
    spec = expand_counts(spec)
    root = _build_units(spec)
    _build_paths(spec, root, current=root)
    return root


def expand_counts(spec: dict[str, Any]) -> dict[str, Any]:
    """Recursively replace `count: N` module entries with N deep-copied replicas.

    Replica names are zero-padded numeric suffixes (count=2 -> '0'/'1';
    count=20 -> '00'..'19'). Paths at the same level that reference a counted
    module at the head of a dotted ref are replicated once per replica.
    """
    children: list[dict[str, Any]] = []
    counted: dict[str, list[str]] = {}

    for child in spec.get("children", []):
        if child.get("kind", "module") == "module":
            child = expand_counts(child)
        count = child.get("count", 1)
        if count <= 1:
            children.append(child)
            continue
        if child.get("kind", "module") != "module":
            raise ValueError(f"`count` only supported on module entries, got kind={child.get('kind')!r}")
        base = child["name"]
        width = max(1, len(str(count - 1)))
        replicas: list[str] = []
        for i in range(count):
            replica = copy.deepcopy(child)
            replica.pop("count")
            replica["name"] = f"{base}{i:0{width}d}"
            children.append(replica)
            replicas.append(replica["name"])
        counted[base] = replicas

    new_spec = {**spec, "children": children}
    if "paths" in new_spec and counted:
        new_paths: list[dict[str, Any]] = []
        for p in new_spec["paths"]:
            orig = _path_references(p, counted)
            if orig is None:
                new_paths.append(p)
            else:
                for replica in counted[orig]:
                    new_paths.append(_substitute_head(p, orig, replica))
        new_spec["paths"] = new_paths
    return new_spec


def _path_references(p: dict[str, Any], counted: dict[str, list[str]]) -> Optional[str]:
    for ref in (p.get("src"), p.get("dst"), p.get("engine")):
        if ref is None:
            continue
        head = ref.split(".", 1)[0]
        if head in counted:
            return head
    return None


def _substitute_head(p: dict[str, Any], orig: str, replica: str) -> dict[str, Any]:
    def sub(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        parts = s.split(".", 1)
        if parts[0] == orig:
            parts[0] = replica
            return ".".join(parts)
        return s

    new_p = {**p, "src": sub(p["src"]), "dst": sub(p["dst"])}
    if p.get("engine") is not None:
        new_p["engine"] = sub(p["engine"])
    if "name" in p:
        new_p["name"] = f"{p['name']}_{replica}"
    return new_p


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
        elif kind == "pipe":
            mod.add_pipe(
                child["name"],
                allowed_src_kinds=[UnitKind(k) for k in child.get("allowed_src_kinds", [])],
                allowed_dst_kinds=[UnitKind(k) for k in child.get("allowed_dst_kinds", [])],
                allowed_src_names=list(child["allowed_src_names"]) if "allowed_src_names" in child else None,
                allowed_dst_names=list(child["allowed_dst_names"]) if "allowed_dst_names" in child else None,
                bandwidth=child.get("bandwidth", 0.0),
                queue_depth=child.get("queue_depth", 1),
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
        engine: Optional[Pipe] = None
        if "engine" in p and p["engine"] is not None:
            resolved = _resolve(root, current, p["engine"])
            if not isinstance(resolved, Pipe):
                raise ValueError(f"engine {p['engine']!r} is not a Pipe")
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
