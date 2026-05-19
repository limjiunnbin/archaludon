"""Simulate softmax(x) over 1024 fp32 elements on NPU.

Decomposed into three Vector ops: exp, sum, div. Data lives in UB; we ignore
that sum produces a scalar (v1 cost is per-element on the workload tensor).
"""
from pathlib import Path

import torch

from arch_sim.arch import load
from arch_sim.sim import Instruction, Sim


def main() -> None:
    chip = load(Path(__file__).resolve().parents[1] / "specs" / "npu.yaml")
    mte2 = chip.find("AICore.MTE2")
    mte3 = chip.find("AICore.MTE3")
    vector = chip.find("AICore.Vector")

    N = 1024
    x = torch.empty(N, dtype=torch.float32, device="meta")
    y = torch.empty(N, dtype=torch.float32, device="meta")

    load_x = Instruction(unit=mte2, tensor=x, label="GM->UB x")
    exp_x = Instruction(unit=vector, tensor=x, deps=[load_x], label="exp(x)")
    sum_e = Instruction(unit=vector, tensor=x, deps=[exp_x], label="sum(exp)")
    div_y = Instruction(unit=vector, tensor=y, deps=[sum_e], label="y = exp / sum")
    store_y = Instruction(unit=mte3, tensor=y, deps=[div_y], label="UB->GM y")

    sim = Sim(chip, [load_x, exp_x, sum_e, div_y, store_y])
    sim.run()

    print(f"softmax over {N} fp32 -> {sim.total_cycles():.0f} cycles")
    for instr in sim.instructions:
        print(f"  {instr.unit.name}: {instr.start_time:.0f}..{instr.end_time:.0f}  {instr.label}")


if __name__ == "__main__":
    main()
