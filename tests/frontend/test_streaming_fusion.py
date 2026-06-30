"""Cube->Vector streaming fusion in linalg lowering (phase 1.5 item 7).

Built on hand-constructed `ir.Graph`s so the tests need no `iree` parser. The
three unfused numbers (single add = 200, single matmul = 768, chained add->add =
400) mirror the iree-gated tests in test_mlir_parser.py and act as the regression
net for the shared `lower()` loop, which those tests cannot exercise here.
"""
from pathlib import Path

import torch

from arch_sim.arch import Module, load, validate
from arch_sim.ir import Graph, TensorOp
from arch_sim.frontend import lower
from arch_sim.sim import simulate, total_cycles

SPEC = Path(__file__).parents[2] / "specs" / "npu.yaml"
STREAM_SPEC = Path(__file__).parents[2] / "specs" / "npu-stream.yaml"


def meta(*shape, dtype=torch.float32):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _add(ins, out, in_sources):
    return TensorOp(kind="add", ins=ins, out=out, in_sources=in_sources)


def _matmul(ins, out, in_sources):
    return TensorOp(kind="matmul", ins=ins, out=out, in_sources=in_sources)


def _without_stream(chip: Module) -> Module:
    """Drop streaming paths so lowering falls back to the GM round-trip."""
    for unit in chip.walk():
        if isinstance(unit, Module):
            unit.paths = [p for p in unit.paths if not getattr(p, "stream", False)]
    return chip


def _matmul_add_graph():
    """%c = matmul(%a, %b); %y = add(%c, %bias)  — bias is a graph input."""
    a, b, c = meta(64, 64), meta(64, 64), meta(64, 64)
    bias, y = meta(64, 64), meta(64, 64)
    mm = _matmul([a, b], c, [None, None])
    add = _add([c, bias], y, [mm, None])
    g = Graph()
    g.add(mm)
    g.add(add)
    return g, mm, add


# --- regression net: unfused paths must keep their known-good timings ----------

def test_unfused_single_add_is_200():
    g = Graph()
    g.add(_add([meta(1024), meta(1024)], meta(1024), [None, None]))
    assert simulate(lower(g, load(SPEC))) == 200


def test_unfused_single_matmul_is_768():
    g = Graph()
    g.add(_matmul([meta(64, 64), meta(64, 64)], meta(64, 64), [None, None]))
    assert simulate(lower(g, load(SPEC))) == 768


def test_chained_adds_go_through_gm_at_400():
    """add->add is never fused (fusion is matmul->add only): stays serial via GM."""
    op1 = _add([meta(1024), meta(1024)], meta(1024), [None, None])
    op2 = _add([op1.out, meta(1024)], meta(1024), [op1, None])
    g = Graph()
    g.add(op1)
    g.add(op2)
    instrs = lower(g, load(SPEC))
    simulate(instrs)
    assert total_cycles(instrs) == 400


# --- fusion: structural proof, not just a cycle delta --------------------------

def test_matmul_into_add_streams_and_skips_gm():
    g, _, _ = _matmul_add_graph()
    instrs = lower(g, load(SPEC))

    # The matmul result never materializes to GM: FixPipe carries only that store.
    assert not any(i.unit.name == "FixPipe" for i in instrs)

    cube = [i for i in instrs if i.unit.name == "Cube"]
    vector = [i for i in instrs if i.unit.name == "Vector"]
    assert len(cube) == 1 and len(vector) == 1
    # The Vector op streams directly from the Cube matmul.
    assert vector[0].stream_deps == [cube[0]]
    assert vector[0].stream_latency == 0.0  # npu.yaml declares no fill latency
    assert simulate(instrs) == 1056  # observed; pins the fused timing


def test_fusion_beats_unfused_round_trip():
    g_f, _, _ = _matmul_add_graph()
    fused = lower(g_f, load(SPEC))
    simulate(fused)

    g_u, _, _ = _matmul_add_graph()
    unfused = lower(g_u, _without_stream(load(SPEC)))
    simulate(unfused)

    # Without the stream link, the matmul stores to GM via FixPipe and the add
    # reloads it — the round-trip the fusion removes.
    assert any(i.unit.name == "FixPipe" for i in unfused)
    assert not any(i.unit.name == "FixPipe" for i in fused)
    assert total_cycles(fused) < total_cycles(unfused)


def test_stream_latency_propagates_from_link():
    """A non-zero fill latency declared on the link reaches the Vector op."""
    chip = load(SPEC)
    link = next(p for u in chip.walk() if isinstance(u, Module)
                for p in u.paths if getattr(p, "stream", False))
    link.stream_latency = 5
    g, _, _ = _matmul_add_graph()
    instrs = lower(g, chip)
    vector = [i for i in instrs if i.unit.name == "Vector"][0]
    assert vector.stream_deps  # fused
    assert vector.stream_latency == 5


def test_npu_stream_variant_fuses_with_declared_latency():
    """The npu-stream variant declares the Cube->Vector wire with stream_latency
    8; lowering picks it up end-to-end and the matmul->add fuses."""
    chip = load(STREAM_SPEC)
    assert validate(chip).errors == []
    g, _, _ = _matmul_add_graph()
    instrs = lower(g, chip)
    assert not any(i.unit.name == "FixPipe" for i in instrs)  # no GM round-trip
    vector = [i for i in instrs if i.unit.name == "Vector"][0]
    assert vector.stream_deps  # fused
    assert vector.stream_latency == 8  # from the spec's cube_to_vector link
    simulate(instrs)


def test_multi_consumer_matmul_does_not_fuse():
    """A matmul whose result feeds two ops can't stream (one link, one consumer):
    it falls back to the GM round-trip and nothing streams."""
    a, b, c = meta(64, 64), meta(64, 64), meta(64, 64)
    mm = _matmul([a, b], c, [None, None])
    y1 = _add([c, meta(64, 64)], meta(64, 64), [mm, None])
    y2 = _add([c, meta(64, 64)], meta(64, 64), [mm, None])
    g = Graph()
    for op in (mm, y1, y2):
        g.add(op)
    instrs = lower(g, load(SPEC))
    assert any(i.unit.name == "FixPipe" for i in instrs)  # matmul stored to GM
    assert not any(i.stream_deps for i in instrs)  # nothing streamed
    simulate(instrs)
