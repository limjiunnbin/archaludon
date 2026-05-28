"""Parse MLIR linalg text into an `ir.Graph`, using the real MLIR bindings shipped
with `iree-base-compiler`. Supported ops: linalg.matmul, linalg.add."""
from __future__ import annotations

from pathlib import Path

import torch

from arch_sim.ir import Graph, TensorOp

_DTYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
}

# linalg op name -> (TensorOp kind, number of input operands before the outs/init)
_SUPPORTED = {
    "linalg.matmul": ("matmul", 2),
    "linalg.add": ("add", 2),
}


def _meta(rtt) -> torch.Tensor:
    et = str(rtt.element_type)
    if et not in _DTYPES:
        raise ValueError(f"unsupported element type {et!r}")
    return torch.empty(*rtt.shape, dtype=_DTYPES[et], device="meta")


def parse(text: str) -> Graph:
    """Parse an MLIR module string into an `ir.Graph`."""
    from iree.compiler import ir as mlir
    from iree.compiler.dialects import func, linalg  # noqa: F401  (registers the dialects)

    graph = Graph()
    with mlir.Context():
        module = mlir.Module.parse(text)
        produced: dict = {}  # mlir Value -> TensorOp that produces it
        for fn in module.body.operations:
            if fn.operation.name != "func.func":
                continue
            for op in fn.regions[0].blocks[0].operations:
                opname = op.operation.name
                if opname.startswith("linalg.") and opname not in _SUPPORTED:
                    raise NotImplementedError(f"unsupported linalg op: {opname}")
                spec = _SUPPORTED.get(opname)
                if spec is None:
                    continue
                kind, n_in = spec
                in_vals = list(op.operands)[:n_in]
                ins = [_meta(mlir.RankedTensorType(v.type)) for v in in_vals]
                out = _meta(mlir.RankedTensorType(op.results[0].type))
                top = TensorOp(
                    kind=kind,
                    ins=ins,
                    out=out,
                    in_sources=[produced.get(v) for v in in_vals],
                )
                produced[op.results[0]] = top
                graph.add(top)
    return graph


def parse_file(path: str | Path) -> Graph:
    return parse(Path(path).read_text())
