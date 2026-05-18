"""Render a Module tree to Graphviz DOT (and optionally an image)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterator

from .base import BaseUnit, Module, UnitKind
from .compute import ComputeUnit
from .control import ControlUnit
from .dma import DMAEngine, Direction
from .storage import StorageUnit


_COLOR = {
    UnitKind.STORAGE: "lightblue",
    UnitKind.COMPUTE: "steelblue",
    UnitKind.CONTROL: "lightgray",
    UnitKind.DMA: "khaki",
    UnitKind.MODULE: "white",
    UnitKind.EXTERNAL: "white",
}

_SHAPE = {
    UnitKind.STORAGE: "box",
    UnitKind.COMPUTE: "ellipse",
    UnitKind.CONTROL: "octagon",
    UnitKind.DMA: "cds",
    UnitKind.MODULE: "box",
    UnitKind.EXTERNAL: "box",
}


def visualize(root: Module, path: str | Path) -> Path:
    """Write DOT for `root` to `path`. If `dot` is on PATH, also render to the same stem.

    Format is inferred from `path`'s suffix; `.dot` writes DOT only.
    Returns the rendered image path if produced, else the `.dot` path.
    """
    out = Path(path)
    dot_path = out.with_suffix(".dot")
    dot_src = to_dot(root)
    dot_path.write_text(dot_src)

    if out.suffix in ("", ".dot"):
        return dot_path

    dot_bin = shutil.which("dot")
    if dot_bin is None:
        print(
            f"wrote {dot_path}; install graphviz (`apt install graphviz`) to render, "
            f"or paste into https://dreampuf.github.io/GraphvizOnline/"
        )
        return dot_path

    fmt = out.suffix.lstrip(".")
    subprocess.run([dot_bin, f"-T{fmt}", "-o", str(out), str(dot_path)], check=True)
    return out


def to_dot(root: Module) -> str:
    """Return a Graphviz DOT representation of `root`."""
    lines: list[str] = [
        "digraph arch {",
        "  rankdir=LR;",
        "  newrank=true;",
        "  compound=true;",
        '  node [style=filled, fontname="Helvetica"];',
        '  edge [fontname="Helvetica", fontsize=10];',
    ]
    _emit_module(root, lines, depth=1)
    _emit_paths(root, lines)
    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_module(mod: Module, lines: list[str], depth: int) -> None:
    indent = "  " * depth
    lines.append(f"{indent}subgraph cluster_{_safe_id(mod)} {{")
    lines.append(f'{indent}  label="{mod.name}";')
    lines.append(f"{indent}  style=dashed;")
    for c in mod.children:
        if isinstance(c, Module):
            _emit_module(c, lines, depth + 1)
        elif isinstance(c, DMAEngine):
            # DMA engines appear as edge labels, not as nodes.
            continue
        else:
            lines.append(
                f"{indent}  {_safe_id(c)} ["
                f'label="{_node_label(c)}", '
                f"shape={_SHAPE.get(c.kind, 'box')}, "
                f"fillcolor={_COLOR.get(c.kind, 'white')}"
                f"];"
            )
    lines.append(f"{indent}}}")


def _emit_paths(root: Module, lines: list[str]) -> None:
    for mod in _modules(root):
        for p in mod.paths:
            attrs: list[str] = []
            if p.engine is not None:
                attrs.append(f'label="{p.engine.name}"')
            if p.direction == Direction.BI:
                attrs.append("dir=both")
            tail = f" [{', '.join(attrs)}]" if attrs else ""
            lines.append(f"  {_safe_id(p.src)} -> {_safe_id(p.dst)}{tail};")


def _modules(root: Module) -> Iterator[Module]:
    yield root
    for c in root.children:
        if isinstance(c, Module):
            yield from _modules(c)


def _safe_id(u: BaseUnit) -> str:
    return f"n{id(u)}"


def _node_label(u: BaseUnit) -> str:
    if isinstance(u, StorageUnit):
        return f"{u.name}\\n{u.capacity_bytes}B"
    if isinstance(u, ComputeUnit):
        return f"{u.name}\\n[{u.operation}]"
    if isinstance(u, ControlUnit):
        return f"{u.name}\\n(ctrl)"
    return u.name
