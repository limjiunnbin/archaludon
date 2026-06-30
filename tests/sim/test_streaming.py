"""Compute->compute streaming: a producer feeds a consumer directly, no writeback.

Costs are sized so producer/consumer land on round cycle counts despite the
units having different throughputs (Cube 4096, Vector 128, Scalar 1 ops/cyc).
"""
from pathlib import Path

import torch

from arch_sim.sim import Instruction, simulate, total_cycles

FIXTURE = Path(__file__).parent.parent / "arch" / "fixtures" / "ascend_like.yaml"


def meta(*shape, dtype=torch.float32):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _chip():
    from arch_sim.arch import load

    return load(FIXTURE)


def test_equal_rate_streaming_overlaps_to_max_not_sum():
    chip = _chip()
    vector = chip.find("AICore.Vector")  # 128 ops/cyc
    scalar = chip.find("AICore.Scalar")  # 1 op/cyc

    prod = Instruction(unit=vector, tensor=meta(1024))  # 1024/128 = 8
    cons = Instruction(unit=scalar, tensor=meta(8), stream_deps=[prod])  # 8/1 = 8

    simulate([prod, cons])

    # Both run 0..8 — fully overlapped. Serialized would have been 16.
    assert (prod.start_time, prod.end_time) == (0, 8)
    assert (cons.start_time, cons.end_time) == (0, 8)
    assert total_cycles([prod, cons]) == 8


def test_faster_consumer_is_producer_bound():
    chip = _chip()
    scalar = chip.find("AICore.Scalar")  # slow producer
    vector = chip.find("AICore.Vector")  # fast consumer

    prod = Instruction(unit=scalar, tensor=meta(64))  # 64/1 = 64
    cons = Instruction(unit=vector, tensor=meta(1024), stream_deps=[prod])  # 1024/128 = 8

    simulate([prod, cons])

    # Consumer starts immediately but cannot finish before the producer streams
    # its last element: it is paced by the producer and ends at 64.
    assert (prod.start_time, prod.end_time) == (0, 64)
    assert cons.start_time == 0
    assert cons.end_time == 64
    assert total_cycles([prod, cons]) == 64


def test_slower_consumer_is_consumer_bound():
    chip = _chip()
    vector = chip.find("AICore.Vector")  # fast producer
    scalar = chip.find("AICore.Scalar")  # slow consumer

    prod = Instruction(unit=vector, tensor=meta(1024))  # 8
    cons = Instruction(unit=scalar, tensor=meta(64), stream_deps=[prod])  # 64

    simulate([prod, cons])

    assert (prod.start_time, prod.end_time) == (0, 8)
    assert cons.start_time == 0
    assert cons.end_time == 64  # consumer's own cost dominates
    assert total_cycles([prod, cons]) == 64


def test_stream_latency_shifts_fill_and_drain():
    chip = _chip()
    vector = chip.find("AICore.Vector")
    scalar = chip.find("AICore.Scalar")

    prod = Instruction(unit=vector, tensor=meta(1024))  # 0..8
    cons = Instruction(unit=scalar, tensor=meta(8), stream_deps=[prod], stream_latency=2)

    simulate([prod, cons])

    # Fill: consumer starts 2 cycles after the producer (first datum arrives).
    # Drain: it can't finish before producer end + latency = 10.
    assert cons.start_time == 2
    assert cons.end_time == 10
    assert total_cycles([prod, cons]) == 10


def test_three_stage_stream_chain_pipelines():
    chip = _chip()
    cube = chip.find("AICore.Cube")  # 4096 ops/cyc, 16^3 tile
    vector = chip.find("AICore.Vector")
    scalar = chip.find("AICore.Scalar")

    a = Instruction(unit=cube, tensor=meta(16, 16, 128))  # 32768/4096 = 8
    b = Instruction(unit=vector, tensor=meta(1024), stream_deps=[a])  # 8
    c = Instruction(unit=scalar, tensor=meta(8), stream_deps=[b])  # 8

    simulate([a, b, c])

    assert (a.start_time, a.end_time) == (0, 8)
    assert (b.start_time, b.end_time) == (0, 8)
    assert (c.start_time, c.end_time) == (0, 8)
    assert total_cycles([a, b, c]) == 8  # three stages, fully overlapped


def test_streamed_producer_needs_no_storage_slot():
    """A streamed producer (dst=None) retires at its execution end, no buffer wait."""
    chip = _chip()
    vector = chip.find("AICore.Vector")
    scalar = chip.find("AICore.Scalar")

    prod = Instruction(unit=vector, tensor=meta(1024), dst=None)
    cons = Instruction(unit=scalar, tensor=meta(8), stream_deps=[prod])

    simulate([prod, cons])

    assert prod.retire_time == prod.end_time  # no destination buffer to block on
