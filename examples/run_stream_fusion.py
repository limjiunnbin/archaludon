"""Cube -> Vector fusion via compute-to-compute streaming.

A 64x64x64 matmul on the Cube, followed by a Vector activation on the result.

  unfused:  Cube -> L0C --FixPipe--> UB -> Vector -> UB --MTE3--> GM
  fused:    Cube ===stream===> Vector -> UB --MTE3--> GM   (no L0C/UB round-trip)

The streamed Vector op starts as soon as the Cube starts producing and is paced
by it, so the activation hides under the matmul and the FixPipe store disappears.
"""
from pathlib import Path

import torch

from arch_sim.arch import load
from arch_sim.sim import Instruction, simulate, total_cycles


def build(chip, fused: bool):
    mte2 = chip.find("AICore.MTE2")
    mte1 = chip.find("AICore.MTE1")
    cube = chip.find("AICore.Cube")
    vector = chip.find("AICore.Vector")
    fixpipe = chip.find("AICore.FixPipe")
    mte3 = chip.find("AICore.MTE3")
    l0c = chip.find("AICore.L0C")
    ub = chip.find("AICore.UB")

    M, N, K = 64, 64, 64
    a = torch.empty(M, K, dtype=torch.float32, device="meta")
    b = torch.empty(K, N, dtype=torch.float32, device="meta")
    c = torch.empty(M, N, dtype=torch.float32, device="meta")
    y = torch.empty(M, N, dtype=torch.float32, device="meta")  # activation output
    work = torch.empty(M, N, K, dtype=torch.float32, device="meta")

    a_l1 = Instruction(unit=mte2, tensor=a, label="GM->L1 A")
    b_l1 = Instruction(unit=mte2, tensor=b, label="GM->L1 B")
    a_l0 = Instruction(unit=mte1, tensor=a, deps=[a_l1], label="L1->L0A A")
    b_l0 = Instruction(unit=mte1, tensor=b, deps=[b_l1], label="L1->L0B B")

    if fused:
        mm = Instruction(unit=cube, tensor=work, deps=[a_l0, b_l0], dst=None, label="Cube C=A@B")
        act = Instruction(unit=vector, tensor=y, stream_deps=[mm], dst=ub, label="Vector act (streamed)")
        store = Instruction(unit=mte3, tensor=y, deps=[act], label="UB->GM Y")
        return [a_l1, b_l1, a_l0, b_l0, mm, act, store]

    mm = Instruction(unit=cube, tensor=work, deps=[a_l0, b_l0], dst=l0c, label="Cube C=A@B")
    c_store = Instruction(unit=fixpipe, tensor=c, deps=[mm], dst=ub, label="L0C->UB C")
    act = Instruction(unit=vector, tensor=y, deps=[c_store], dst=ub, label="Vector act")
    store = Instruction(unit=mte3, tensor=y, deps=[act], label="UB->GM Y")
    return [a_l1, b_l1, a_l0, b_l0, mm, c_store, act, store]


def show(title, instrs):
    print(f"{title}: {total_cycles(instrs):.0f} cycles")
    for i in instrs:
        print(f"  {i.unit.name:<8s} {i.start_time:>4.0f}..{i.end_time:<4.0f}  {i.label}")
    print()


def main() -> None:
    spec = Path(__file__).resolve().parents[1] / "specs" / "npu.yaml"

    unfused = build(load(spec), fused=False)
    simulate(unfused)
    show("unfused (through L0C/UB)", unfused)

    fused = build(load(spec), fused=True)
    simulate(fused)
    show("fused   (Cube->Vector stream)", fused)

    saved = total_cycles(unfused) - total_cycles(fused)
    print(f"streaming fusion saved {saved:.0f} cycles")


if __name__ == "__main__":
    main()
