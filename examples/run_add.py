"""Simulate c = a + b on NPU. Hand-built instruction list."""
from pathlib import Path

import torch

from arch_sim.arch import load
from arch_sim.sim import Instruction, Sim


def main() -> None:
    chip = load(Path(__file__).resolve().parents[1] / "specs" / "npu.yaml")
    mte2 = chip.find("AICore.MTE2")
    mte3 = chip.find("AICore.MTE3")
    vector = chip.find("AICore.Vector")

    a = torch.empty(1024, dtype=torch.float32, device="meta")
    b = torch.empty(1024, dtype=torch.float32, device="meta")
    c = torch.empty(1024, dtype=torch.float32, device="meta")

    move_a = Instruction(unit=mte2, tensor=a, label="GM->UB a")
    move_b = Instruction(unit=mte2, tensor=b, label="GM->UB b")
    add = Instruction(unit=vector, tensor=c, deps=[move_a, move_b], label="c = a + b")
    move_c = Instruction(unit=mte3, tensor=c, deps=[add], label="UB->GM c")

    sim = Sim(chip, [move_a, move_b, add, move_c])
    sim.run()

    print(f"total cycles: {sim.total_cycles():.0f}")
    for instr in sim.instructions:
        print(f"  {instr.unit.name}: {instr.start_time:.0f}..{instr.end_time:.0f}  {instr.label}")


if __name__ == "__main__":
    main()
