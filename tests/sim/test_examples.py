from pathlib import Path

import torch

from arch_sim.arch import load
from arch_sim.sim import Instruction, Sim


FIXTURE = Path(__file__).parent.parent / "arch" / "fixtures" / "ascend_like.yaml"


def meta(*shape, dtype=torch.float32):
    return torch.empty(*shape, dtype=dtype, device="meta")


def test_matmul_64x64x64_takes_768_cycles():
    chip = load(FIXTURE)
    mte1 = chip.find("AICore.MTE1")
    mte2 = chip.find("AICore.MTE2")
    cube = chip.find("AICore.Cube")

    # Fixture doesn't have FixPipe; route the writeback through MTE3 via UB.
    fix_substitute = chip.find("AICore.MTE3")

    a, b, c = meta(64, 64), meta(64, 64), meta(64, 64)
    work = meta(64, 64, 64)

    load_a_l1 = Instruction(unit=mte2, tensor=a)
    load_b_l1 = Instruction(unit=mte2, tensor=b)
    load_a_l0a = Instruction(unit=mte1, tensor=a, deps=[load_a_l1])
    load_b_l0b = Instruction(unit=mte1, tensor=b, deps=[load_b_l1])
    matmul = Instruction(unit=cube, tensor=work, deps=[load_a_l0a, load_b_l0b])
    store_c = Instruction(unit=fix_substitute, tensor=c, deps=[matmul])

    sim = Sim(chip, [load_a_l1, load_b_l1, load_a_l0a, load_b_l0b, matmul, store_c])
    sim.run()

    # MTE2 16KB at 64 B/cyc: 256 cyc * 2 transfers = 512
    # MTE1 16KB at 256 B/cyc: 64 cyc each, starts when its predecessor MTE2 finishes
    # Cube 64*64*64 / 4096 = 64 cyc, waits for second L0 load (576)
    # MTE3 16KB at 64 B/cyc = 256 cyc, follows cube (ends at 640)
    assert matmul.start_time == 576
    assert matmul.end_time == 640
    assert sim.total_cycles() == 896  # 640 + 256 (MTE3 in fixture, slower than FixPipe)


def test_parallel_add_overlaps_move_with_compute():
    """g = (a+b) + (d+e). With the in-order per-AICore dispatcher and
    queue_depth=1, `add_c` can't be issued until the dispatcher has cleared the
    moves ahead of it in program order — so it starts at 192 (when MTE2's queue
    frees and Vector can be admitted), not at 128. The overlap is now: while
    MTE2 moves `e` (192..256), Vector adds `c` (192..200). The total is still
    bounded by MTE2's four serial loads + the final compute/store chain."""
    chip = load(FIXTURE)
    mte2 = chip.find("AICore.MTE2")
    mte3 = chip.find("AICore.MTE3")
    vector = chip.find("AICore.Vector")

    a, b, d, e, c, f, g = (meta(1024) for _ in range(7))
    move_a = Instruction(unit=mte2, tensor=a)
    move_b = Instruction(unit=mte2, tensor=b)
    move_d = Instruction(unit=mte2, tensor=d)
    move_e = Instruction(unit=mte2, tensor=e)
    add_c = Instruction(unit=vector, tensor=c, deps=[move_a, move_b])
    add_f = Instruction(unit=vector, tensor=f, deps=[move_d, move_e])
    add_g = Instruction(unit=vector, tensor=g, deps=[add_c, add_f])
    store_g = Instruction(unit=mte3, tensor=g, deps=[add_g])

    Sim(chip, [move_a, move_b, move_d, move_e, add_c, add_f, add_g, store_g]).run()

    assert move_d.start_time == 128            # moves serialize on MTE2
    assert add_c.start_time == 192             # delayed by in-order issue
    assert add_c.end_time == 200
    assert store_g.end_time == 336


def test_two_cores_run_independent_adds_in_parallel():
    """Two adds on two AICores finish in 200 cycles (one core's worth), not 400."""
    fixture = Path(__file__).parent.parent / "arch" / "fixtures" / "ascend_2core.yaml"
    chip = load(fixture)
    a0_mte2 = chip.find("AICore0.MTE2")
    a0_mte3 = chip.find("AICore0.MTE3")
    a0_vec = chip.find("AICore0.Vector")
    a1_mte2 = chip.find("AICore1.MTE2")
    a1_mte3 = chip.find("AICore1.MTE3")
    a1_vec = chip.find("AICore1.Vector")

    instrs = []
    for mte2, mte3, vec in [(a0_mte2, a0_mte3, a0_vec), (a1_mte2, a1_mte3, a1_vec)]:
        x, y, z = meta(1024), meta(1024), meta(1024)
        la = Instruction(unit=mte2, tensor=x)
        lb = Instruction(unit=mte2, tensor=y)
        ad = Instruction(unit=vec, tensor=z, deps=[la, lb])
        st = Instruction(unit=mte3, tensor=z, deps=[ad])
        instrs.extend([la, lb, ad, st])

    sim = Sim(chip, instrs)
    sim.run()
    assert sim.total_cycles() == 200


def test_softmax_1024_takes_152_cycles():
    chip = load(FIXTURE)
    mte2 = chip.find("AICore.MTE2")
    mte3 = chip.find("AICore.MTE3")
    vector = chip.find("AICore.Vector")

    x = meta(1024)
    y = meta(1024)

    load_x = Instruction(unit=mte2, tensor=x)
    exp_x = Instruction(unit=vector, tensor=x, deps=[load_x])
    sum_e = Instruction(unit=vector, tensor=x, deps=[exp_x])
    div_y = Instruction(unit=vector, tensor=y, deps=[sum_e])
    store_y = Instruction(unit=mte3, tensor=y, deps=[div_y])

    sim = Sim(chip, [load_x, exp_x, sum_e, div_y, store_y])
    sim.run()

    assert load_x.end_time == 64
    assert div_y.end_time == 88   # 64 + 3*8
    assert store_y.end_time == 152
    assert sim.total_cycles() == 152
