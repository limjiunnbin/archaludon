from pathlib import Path

import pytest
import torch

from arch_sim.arch import load
from arch_sim.sim import Instruction, simulate


FIXTURE = Path(__file__).parent.parent / "arch" / "fixtures" / "ascend_like.yaml"


def meta(n=1024):
    return torch.empty(n, dtype=torch.float32, device="meta")


def _stall_workload(ub_capacity):
    chip = load(FIXTURE)
    vector = chip.find("AICore.Vector")
    mte3 = chip.find("AICore.MTE3")
    ub = chip.find("AICore.UB")
    ub.queue_depth = ub_capacity

    instrs = []
    vs = []
    for _ in range(4):
        v = Instruction(unit=vector, tensor=meta(), dst=ub)
        s = Instruction(unit=mte3, tensor=meta(), deps=[v], dst=None)
        vs.append(v)
        instrs += [v, s]
    return instrs, vs


def test_vector_stall_totals_264_with_full_ub():
    instrs, vs = _stall_workload(ub_capacity=2)
    total = simulate(instrs)

    assert total == 264
    # v3 finishes executing at 32 but can't retire until 72 (UB full) — a 40-cycle stall.
    assert vs[3].end_time == 32
    assert vs[3].retire_time == 72
    # the earlier ops retire as soon as they finish executing
    assert vs[0].end_time == vs[0].retire_time == 8


def test_deeper_ub_buffer_delays_the_stall():
    # With room for all 4 results, no Vector op ever stalls on UB.
    instrs, vs = _stall_workload(ub_capacity=4)
    simulate(instrs)
    for v in vs:
        assert v.retire_time == v.end_time  # retired immediately after execution


def test_unbounded_dst_means_no_backpressure():
    chip = load(FIXTURE)
    vector = chip.find("AICore.Vector")
    # dst=None everywhere → no capacity gate → vector ops retire at exec end, serialized.
    ops = [Instruction(unit=vector, tensor=meta(), dst=None) for _ in range(3)]
    total = simulate(ops)
    assert total == 24  # 3 * 8 cyc, back to back
    assert [o.retire_time for o in ops] == [8, 16, 24]


def test_backpressure_deadlock_raises():
    chip = load(FIXTURE)
    vector = chip.find("AICore.Vector")
    ub = chip.find("AICore.UB")
    ub.queue_depth = 1
    # Two producers into a depth-1 UB whose contents are never drained:
    # the first fills UB, the second can never retire and nothing frees the slot.
    a = Instruction(unit=vector, tensor=meta(), dst=ub)
    b = Instruction(unit=vector, tensor=meta(), dst=ub)
    with pytest.raises(RuntimeError, match="deadlock"):
        simulate([a, b])
