"""Simulate g = (a + b) + (d + e) on NPU.

The two inner adds (c=a+b, f=d+e) are structurally independent in the DAG,
but they share the single Vector unit so they serialize on its channel.
The parallelism you'll see is between MTE2 (still moving operands) and
Vector (already adding what arrived earlier).
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

    def meta():
        return torch.empty(1024, dtype=torch.float32, device="meta")

    a, b, d, e, c, f, g = (meta() for _ in range(7))

    move_a = Instruction(unit=mte2, tensor=a, label="GM->UB a")
    move_b = Instruction(unit=mte2, tensor=b, label="GM->UB b")
    move_d = Instruction(unit=mte2, tensor=d, label="GM->UB d")
    move_e = Instruction(unit=mte2, tensor=e, label="GM->UB e")

    add_c = Instruction(unit=vector, tensor=c, deps=[move_a, move_b], label="c = a + b")
    add_f = Instruction(unit=vector, tensor=f, deps=[move_d, move_e], label="f = d + e")
    add_g = Instruction(unit=vector, tensor=g, deps=[add_c, add_f], label="g = c + f")

    store_g = Instruction(unit=mte3, tensor=g, deps=[add_g], label="UB->GM g")

    sim = Sim(chip, [move_a, move_b, move_d, move_e, add_c, add_f, add_g, store_g])
    sim.run()

    print(f"total cycles: {sim.total_cycles():.0f}")
    for instr in sorted(sim.instructions, key=lambda i: i.start_time):
        print(f"  {instr.unit.name:>6s}: {instr.start_time:>4.0f}..{instr.end_time:<4.0f}  {instr.label}")


if __name__ == "__main__":
    main()
