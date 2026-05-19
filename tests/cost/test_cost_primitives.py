import torch

from arch_sim.arch import ComputeUnit, Pipe
from arch_sim.cost import Move, Op


def meta(shape, dtype=torch.float32):
    return torch.empty(*shape, dtype=dtype, device="meta")


def test_op_elementwise_cycles():
    vec = ComputeUnit(name="Vector", throughput_ops_per_cycle=128)
    assert Op(vec, meta((1024,))) == 8


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
