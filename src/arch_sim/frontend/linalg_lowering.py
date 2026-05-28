"""Lower an `ir.Graph` to `Instruction`s on the NPU memory hierarchy.

v1 is assume-fits: each op becomes one compute instruction plus the loads/stores
that feed it, with no tiling. If a tensor exceeds the destination buffer's
`capacity_bytes`, we raise (tiling is not implemented). Cross-op dataflow goes
through GM: a consumer's load depends on the producer's store.
"""
from __future__ import annotations

import torch

from arch_sim.arch import Module
from arch_sim.ir import Graph, TensorOp
from arch_sim.sim import Instruction


def _check_fits(t: torch.Tensor, buf) -> None:
    nbytes = t.numel() * t.element_size()
    cap = getattr(buf, "capacity_bytes", 0)
    if cap and nbytes > cap:
        raise ValueError(
            f"{buf.qualified_name()}: tensor of {nbytes} B exceeds capacity {cap} B "
            f"(tiling not implemented)"
        )


def lower(graph: Graph, chip: Module, core: str = "AICore") -> list[Instruction]:
    c = chip.find(core)
    mte1, mte2, mte3 = c.find("MTE1"), c.find("MTE2"), c.find("MTE3")
    cube, vector, fixpipe = c.find("Cube"), c.find("Vector"), c.find("FixPipe")
    ub, l1 = c.find("UB"), c.find("L1Buffer")
    l0a, l0b, l0c = c.find("L0A"), c.find("L0B"), c.find("L0C")

    instrs: list[Instruction] = []
    store_of: dict[int, Instruction] = {}  # id(TensorOp) -> the store that lands its output in GM

    def src_dep(op: TensorOp, i: int) -> list[Instruction]:
        src = op.in_sources[i] if i < len(op.in_sources) else None
        return [store_of[id(src)]] if src is not None else []

    for op in graph.ops:
        if op.kind == "add":
            a, b = op.ins[0], op.ins[1]
            for t in (a, b, op.out):
                _check_fits(t, ub)
            la = Instruction(unit=mte2, tensor=a, deps=src_dep(op, 0), label="GM->UB a")
            lb = Instruction(unit=mte2, tensor=b, deps=src_dep(op, 1), label="GM->UB b")
            compute = Instruction(unit=vector, tensor=op.out, deps=[la, lb], label="add")
            store = Instruction(unit=mte3, tensor=op.out, deps=[compute], label="UB->GM")
            instrs += [la, lb, compute, store]
            store_of[id(op)] = store

        elif op.kind == "matmul":
            a, b = op.ins[0], op.ins[1]
            m, k = int(a.shape[0]), int(a.shape[1])
            n = int(op.out.shape[1])
            _check_fits(a, l1)
            _check_fits(b, l1)
            _check_fits(a, l0a)
            _check_fits(b, l0b)
            _check_fits(op.out, l0c)
            # Cube cost is tile-quantized over the (M, N, K) iteration space.
            work = torch.empty(m, n, k, dtype=op.out.dtype, device="meta")
            a_l1 = Instruction(unit=mte2, tensor=a, deps=src_dep(op, 0), label="GM->L1 A")
            b_l1 = Instruction(unit=mte2, tensor=b, deps=src_dep(op, 1), label="GM->L1 B")
            a_l0 = Instruction(unit=mte1, tensor=a, deps=[a_l1], label="L1->L0A A")
            b_l0 = Instruction(unit=mte1, tensor=b, deps=[b_l1], label="L1->L0B B")
            mm = Instruction(unit=cube, tensor=work, deps=[a_l0, b_l0], label="Cube C=A@B")
            store = Instruction(unit=fixpipe, tensor=op.out, deps=[mm], label="L0C->GM C")
            instrs += [a_l1, b_l1, a_l0, b_l0, mm, store]
            store_of[id(op)] = store

        else:
            raise NotImplementedError(f"lowering for kind {op.kind!r}")

    return instrs
