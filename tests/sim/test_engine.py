from pathlib import Path

import pytest
import torch

from arch_sim.arch import load
from arch_sim.sim import Channel, Instruction, Sim, run, total_cycles


FIXTURE = Path(__file__).parent.parent / "arch" / "fixtures" / "ascend_like.yaml"


def meta(n, dtype=torch.float32):
    return torch.empty(n, dtype=dtype, device="meta")


def test_c_equals_a_plus_b_finishes_in_200_cycles():
    chip = load(FIXTURE)
    mte2 = chip.find("AICore.MTE2")
    mte3 = chip.find("AICore.MTE3")
    vector = chip.find("AICore.Vector")

    a, b, c = meta(1024), meta(1024), meta(1024)
    move_a = Instruction(unit=mte2, tensor=a)
    move_b = Instruction(unit=mte2, tensor=b)
    add = Instruction(unit=vector, tensor=c, deps=[move_a, move_b])
    move_c = Instruction(unit=mte3, tensor=c, deps=[add])

    ch_mte2 = Channel(mte2)
    ch_mte2.enqueue(move_a)
    ch_mte2.enqueue(move_b)
    ch_vec = Channel(vector)
    ch_vec.enqueue(add)
    ch_mte3 = Channel(mte3)
    ch_mte3.enqueue(move_c)

    trace = run([ch_mte2, ch_vec, ch_mte3])

    assert total_cycles(trace) == 200
    assert (move_a.start_time, move_a.end_time) == (0, 64)
    assert (move_b.start_time, move_b.end_time) == (64, 128)
    assert (add.start_time, add.end_time) == (128, 136)
    assert (move_c.start_time, move_c.end_time) == (136, 200)


def test_independent_channels_run_in_parallel():
    """Two independent moves on different pipes both end at their own bandwidth cost."""
    chip = load(FIXTURE)
    mte2 = chip.find("AICore.MTE2")
    mte3 = chip.find("AICore.MTE3")

    t = meta(1024)
    m_in = Instruction(unit=mte2, tensor=t)
    m_out = Instruction(unit=mte3, tensor=t)  # no dep — independent

    ch_in = Channel(mte2)
    ch_in.enqueue(m_in)
    ch_out = Channel(mte3)
    ch_out.enqueue(m_out)

    run([ch_in, ch_out])

    assert m_in.end_time == 64
    assert m_out.end_time == 64


def test_sim_groups_instructions_by_unit():
    chip = load(FIXTURE)
    mte2 = chip.find("AICore.MTE2")
    mte3 = chip.find("AICore.MTE3")
    vector = chip.find("AICore.Vector")

    a, b, c = meta(1024), meta(1024), meta(1024)
    move_a = Instruction(unit=mte2, tensor=a)
    move_b = Instruction(unit=mte2, tensor=b)
    add = Instruction(unit=vector, tensor=c, deps=[move_a, move_b])
    move_c = Instruction(unit=mte3, tensor=c, deps=[add])

    sim = Sim(chip, [move_a, move_b, add, move_c])
    sim.run()

    assert sim.total_cycles() == 200
    assert sim.module is chip


def test_deadlock_raises():
    chip = load(FIXTURE)
    vector = chip.find("AICore.Vector")

    t = meta(8)
    sentinel = Instruction(unit=vector, tensor=t)  # never enqueued
    waiter = Instruction(unit=vector, tensor=t, deps=[sentinel])

    ch = Channel(vector)
    ch.enqueue(waiter)

    with pytest.raises(RuntimeError, match="deadlock"):
        run([ch])
