"""Lower an `ir.Graph` to `Instruction`s on the NPU memory hierarchy.

v1 is assume-fits: each op becomes one compute instruction plus the loads/stores
that feed it, with no tiling. If a tensor exceeds the destination buffer's
`capacity_bytes`, we raise (tiling is not implemented). Cross-op dataflow goes
through GM: a consumer's load depends on the producer's store.

Streaming fusion: when a matmul's result feeds a single elementwise consumer and
the spec declares a Cube->Vector stream link (`DataPath.stream`), the matmul
result is streamed straight into the consumer's Vector op (`stream_deps`) instead
of being stored to and reloaded from GM. See docs/streaming-plan.md item 7.
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


def _find_stream_link(chip: Module, src_unit, dst_unit):
    """The streaming DataPath from `src_unit` to `dst_unit` in the spec, or None."""
    for unit in chip.walk():
        if not isinstance(unit, Module):
            continue
        for p in unit.paths:
            if getattr(p, "stream", False) and p.src is src_unit and p.dst is dst_unit:
                return p
    return None


def _fusion_targets(graph: Graph, stream_link) -> dict[int, TensorOp]:
    """Map id(matmul op) -> the single elementwise op that consumes its result.

    A matmul is eligible to stream into its consumer when a Cube->Vector stream
    link exists and the matmul's result feeds exactly one op, which is an
    elementwise `add`. The matmul result then never lands in GM.

    Limitation: the IR does not model `func` returns, so a matmul whose result is
    *also* returned would still be fused and lose its GM copy. Acceptable for v1
    (assume-fits, single-consumer); revisit when returns are tracked.
    """
    if stream_link is None:
        return {}
    consumers: dict[int, list[TensorOp]] = {}
    for op in graph.ops:
        for src in op.in_sources:
            if src is not None:
                consumers.setdefault(id(src), []).append(op)
    targets: dict[int, TensorOp] = {}
    for op in graph.ops:
        if op.kind == "matmul":
            cons = consumers.get(id(op), [])
            if len(cons) == 1 and cons[0].kind == "add":
                targets[id(op)] = cons[0]
    return targets


def lower(graph: Graph, chip: Module, core: str = "AICore") -> list[Instruction]:
    c = chip.find(core)
    mte1, mte2, mte3 = c.find("MTE1"), c.find("MTE2"), c.find("MTE3")
    cube, vector, fixpipe = c.find("Cube"), c.find("Vector"), c.find("FixPipe")
    ub, l1 = c.find("UB"), c.find("L1Buffer")
    l0a, l0b, l0c = c.find("L0A"), c.find("L0B"), c.find("L0C")

    stream_link = _find_stream_link(chip, cube, vector)
    fuse_into = _fusion_targets(graph, stream_link)  # id(matmul op) -> consumer add op

    instrs: list[Instruction] = []
    store_of: dict[int, Instruction] = {}  # id(TensorOp) -> the store that lands its output in GM
    compute_of: dict[int, Instruction] = {}  # id(TensorOp) -> its compute instr (stream producer)

    def src_dep(op: TensorOp, i: int) -> list[Instruction]:
        src = op.in_sources[i] if i < len(op.in_sources) else None
        return [store_of[id(src)]] if src is not None else []

    for op in graph.ops:
        if op.kind == "add":
            loads: list[Instruction] = []
            stream_deps: list[Instruction] = []
            for i, t in enumerate(op.ins):
                src = op.in_sources[i] if i < len(op.in_sources) else None
                if src is not None and fuse_into.get(id(src)) is op:
                    # Matmul result streams Cube->Vector — no GM round-trip, no load.
                    stream_deps.append(compute_of[id(src)])
                else:
                    _check_fits(t, ub)
                    loads.append(
                        Instruction(unit=mte2, tensor=t, deps=src_dep(op, i), label=f"GM->UB in{i}")
                    )
            _check_fits(op.out, ub)
            lat = stream_link.stream_latency if stream_deps else 0.0
            compute = Instruction(
                unit=vector, tensor=op.out, deps=loads,
                stream_deps=stream_deps, stream_latency=lat,
                label="add (fused)" if stream_deps else "add",
            )
            store = Instruction(unit=mte3, tensor=op.out, deps=[compute], label="UB->GM")
            instrs += loads + [compute, store]
            store_of[id(op)] = store
            compute_of[id(op)] = compute

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
            compute_of[id(op)] = mm
            if id(op) in fuse_into:
                # Result streams into the fused consumer: no L0C->GM store, and
                # deliberately no store_of entry (src_dep must never see it).
                instrs += [a_l1, b_l1, a_l0, b_l0, mm]
            else:
                store = Instruction(unit=fixpipe, tensor=op.out, deps=[mm], label="L0C->GM C")
                instrs += [a_l1, b_l1, a_l0, b_l0, mm, store]
                store_of[id(op)] = store

        else:
            raise NotImplementedError(f"lowering for kind {op.kind!r}")

    return instrs
