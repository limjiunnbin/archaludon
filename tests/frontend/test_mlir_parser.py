from pathlib import Path

import pytest
import torch

from arch_sim.arch import load
from arch_sim.frontend import lower, parse, parse_file
from arch_sim.sim import simulate, total_cycles


FIXTURES = Path(__file__).parent / "fixtures"
SPEC = Path(__file__).parents[2] / "specs" / "npu.yaml"


def test_parse_extracts_linalg_add_shape_and_dtype():
    g = parse_file(FIXTURES / "add.mlir")
    assert len(g.ops) == 1
    op = g.ops[0]
    assert op.kind == "add"
    assert tuple(op.ins[0].shape) == (1024,)
    assert tuple(op.ins[1].shape) == (1024,)
    assert op.ins[0].dtype is torch.float32
    assert tuple(op.out.shape) == (1024,)


def test_parse_extracts_linalg_matmul_shape_and_dtype():
    g = parse_file(FIXTURES / "matmul.mlir")
    assert len(g.ops) == 1
    op = g.ops[0]
    assert op.kind == "matmul"
    assert tuple(op.ins[0].shape) == (64, 64)
    assert tuple(op.ins[1].shape) == (64, 64)
    assert tuple(op.out.shape) == (64, 64)


def test_lowered_add_matches_hand_built_at_200_cycles():
    chip = load(SPEC)
    g = parse_file(FIXTURES / "add.mlir")
    instrs = lower(g, chip)
    assert simulate(instrs) == 200


def test_lowered_matmul_matches_hand_built_at_768_cycles():
    chip = load(SPEC)
    g = parse_file(FIXTURES / "matmul.mlir")
    instrs = lower(g, chip)
    assert simulate(instrs) == 768


def test_chained_ops_link_through_gm():
    """Two adds in sequence: %d = (a + b) + e -- the second add's load of %c
    depends on the first add's store."""
    chip = load(SPEC)
    text = '''
    func.func @chain(%a: tensor<1024xf32>, %b: tensor<1024xf32>, %e: tensor<1024xf32>,
                     %i0: tensor<1024xf32>, %i1: tensor<1024xf32>) -> tensor<1024xf32> {
      %c = linalg.add ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>) outs(%i0 : tensor<1024xf32>) -> tensor<1024xf32>
      %d = linalg.add ins(%c, %e : tensor<1024xf32>, tensor<1024xf32>) outs(%i1 : tensor<1024xf32>) -> tensor<1024xf32>
      return %d : tensor<1024xf32>
    }
    '''
    g = parse(text)
    assert len(g.ops) == 2
    # the second op's first input is the first op's output
    assert g.ops[1].in_sources[0] is g.ops[0]
    assert g.ops[1].in_sources[1] is None  # %e is a graph input
    instrs = lower(g, chip)
    simulate(instrs)
    # The two adds run sequentially through GM: 200 + 200 = 400.
    assert total_cycles(instrs) == 400


def test_unsupported_linalg_op_raises():
    text = '''
    func.func @sub(%a: tensor<8xf32>, %b: tensor<8xf32>, %i: tensor<8xf32>) -> tensor<8xf32> {
      %c = linalg.sub ins(%a, %b : tensor<8xf32>, tensor<8xf32>) outs(%i : tensor<8xf32>) -> tensor<8xf32>
      return %c : tensor<8xf32>
    }
    '''
    with pytest.raises(NotImplementedError, match="linalg.sub"):
        parse(text)


def test_oversize_tensor_raises_on_lower():
    chip = load(SPEC)
    # UB capacity is 262144 B; 65537 f32 elements = 262148 B exceeds it.
    text = '''
    func.func @big(%a: tensor<65537xf32>, %b: tensor<65537xf32>, %i: tensor<65537xf32>) -> tensor<65537xf32> {
      %c = linalg.add ins(%a, %b : tensor<65537xf32>, tensor<65537xf32>) outs(%i : tensor<65537xf32>) -> tensor<65537xf32>
      return %c : tensor<65537xf32>
    }
    '''
    g = parse(text)
    with pytest.raises(ValueError, match="exceeds capacity"):
        lower(g, chip)
