from pathlib import Path

import pytest
import torch

from arch_sim.arch import load
from arch_sim.sim import Instruction, Sim, simulate, total_cycles


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

    instrs = [move_a, move_b, add, move_c]
    simulate(instrs)

    assert total_cycles(instrs) == 200
    assert (move_a.start_time, move_a.end_time) == (0, 64)
    assert (move_b.start_time, move_b.end_time) == (64, 128)
    assert (add.start_time, add.end_time) == (128, 136)
    assert (move_c.start_time, move_c.end_time) == (136, 200)


def test_independent_units_run_in_parallel():
    """Two independent moves on different pipes both end at their own bandwidth cost."""
    chip = load(FIXTURE)
    mte2 = chip.find("AICore.MTE2")
    mte3 = chip.find("AICore.MTE3")

    t = meta(1024)
    m_in = Instruction(unit=mte2, tensor=t)
    m_out = Instruction(unit=mte3, tensor=t)  # no dep — independent

    simulate([m_in, m_out])

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


def test_dispatcher_head_of_line_blocking():
    """Two slow MTE2 moves followed in program order by an independent Vector op.

    With MTE2.queue_depth=1 the in-order dispatcher can't issue the Vector op
    until an MTE2 slot frees (it's blocked behind the second MTE2 move in
    program order), even though Vector is sitting idle. Widening MTE2's queue
    to 2 lets the dispatcher reach the Vector op and start it at t=0.
    """
    chip = load(FIXTURE)
    mte2 = chip.find("AICore.MTE2")
    vector = chip.find("AICore.Vector")

    def workload(mte2_qd):
        mte2.queue_depth = mte2_qd
        t = meta(1024)
        m0 = Instruction(unit=mte2, tensor=t)
        m1 = Instruction(unit=mte2, tensor=t)
        v = Instruction(unit=vector, tensor=t)  # independent of m0/m1
        simulate([m0, m1, v])
        return v.start_time

    assert workload(1) == 64  # blocked: dispatcher can't reach v until m0 retires
    assert workload(2) == 0   # wide queue: v issues at t=0 and starts immediately


def test_deadlock_raises():
    chip = load(FIXTURE)
    vector = chip.find("AICore.Vector")

    t = meta(8)
    sentinel = Instruction(unit=vector, tensor=t)  # never handed to the sim
    waiter = Instruction(unit=vector, tensor=t, deps=[sentinel])

    with pytest.raises(RuntimeError, match="deadlock"):
        simulate([waiter])
