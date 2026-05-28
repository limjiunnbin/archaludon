import pytest
import torch

from arch_sim.arch import ComputeUnit, Pipe
from arch_sim.cost import Move, Op


def meta(shape, dtype=torch.float32):
    return torch.empty(*shape, dtype=dtype, device="meta")


def test_op_elementwise_cycles():
    vec = ComputeUnit(name="Vector", throughput_ops_per_cycle=128)
    assert Op(vec, meta((1024,))) == 8


def test_matmul_tile_aligned_no_padding():
    cube = ComputeUnit(name="Cube", operand_shape=(16, 16, 16), throughput_ops_per_cycle=4096)
    # 64 is a multiple of 16: no padding, 64^3 / 4096 = 64
    assert Op(cube, meta((64, 64, 64))) == 64


def test_matmul_pads_odd_dims_up_to_tile():
    cube = ComputeUnit(name="Cube", operand_shape=(16, 16, 16), throughput_ops_per_cycle=4096)
    # 17 rounds up to 32 on every axis -> 32^3 / 4096 = 8 (vs ~1.2 without padding)
    assert Op(cube, meta((17, 17, 17))) == 8
    # a single 1x1x1 still costs one full tile
    assert Op(cube, meta((1, 1, 1))) == 1


def test_op_without_operand_shape_uses_numel():
    # Vector has no operand_shape -> unchanged numel/throughput path
    vec = ComputeUnit(name="Vector", throughput_ops_per_cycle=128)
    assert Op(vec, meta((100,))) == 100 / 128


def test_op_rank_mismatch_for_tiled_unit_raises():
    cube = ComputeUnit(name="Cube", operand_shape=(16, 16, 16), throughput_ops_per_cycle=4096)
    with pytest.raises(ValueError, match="rank"):
        Op(cube, meta((64, 64)))  # 2D tensor, 3D tile


def test_op_dtype_does_not_change_compute_cycles():
    vec = ComputeUnit(name="Vector", throughput_ops_per_cycle=128)
    assert Op(vec, meta((1024,), dtype=torch.float16)) == 8


def test_move_bytes_over_bandwidth():
    mte2 = Pipe(name="MTE2", bandwidth=64)
    assert Move(mte2, meta((1024,), dtype=torch.float32)) == 64


def test_move_dtype_changes_byte_cost():
    mte2 = Pipe(name="MTE2", bandwidth=64)
    assert Move(mte2, meta((1024,), dtype=torch.float16)) == 32


def test_move_fixpipe_at_128_bandwidth():
    fix = Pipe(name="FixPipe", bandwidth=128)
    assert Move(fix, meta((256,), dtype=torch.float32)) == 8


def test_meta_tensors_allocate_no_storage():
    t = meta((1_000_000_000,))
    assert t.device.type == "meta"
    assert t.numel() == 1_000_000_000
    assert t.element_size() == 4
