"""Two independent adds, one per AICore. c=a+b on AICore0 in parallel with f=d+e on AICore1.

The 2-core spec is `specs/npu.yaml` with `count: 2` on the AICore module
(see `specs/npu-2core.yaml`). Bumping that to `count: 20` would give an
Ascend 910B-shaped chip with no other changes.
"""
from pathlib import Path

import torch

from arch_sim.arch import load
from arch_sim.sim import Instruction, Sim


def main() -> None:
    chip = load(Path(__file__).resolve().parents[1] / "specs" / "npu-2core.yaml")

    def units(core_name):
        core = chip.find(core_name)
        return core.find("MTE2"), core.find("MTE3"), core.find("Vector")

    mte2_0, mte3_0, vec_0 = units("AICore0")
    mte2_1, mte3_1, vec_1 = units("AICore1")

    def meta():
        return torch.empty(1024, dtype=torch.float32, device="meta")

    a, b, c, d, e, f = (meta() for _ in range(6))

    load_a = Instruction(unit=mte2_0, tensor=a, label="A0: GM->UB a")
    load_b = Instruction(unit=mte2_0, tensor=b, label="A0: GM->UB b")
    add_c = Instruction(unit=vec_0, tensor=c, deps=[load_a, load_b], label="A0: c = a + b")
    store_c = Instruction(unit=mte3_0, tensor=c, deps=[add_c], label="A0: UB->GM c")

    load_d = Instruction(unit=mte2_1, tensor=d, label="A1: GM->UB d")
    load_e = Instruction(unit=mte2_1, tensor=e, label="A1: GM->UB e")
    add_f = Instruction(unit=vec_1, tensor=f, deps=[load_d, load_e], label="A1: f = d + e")
    store_f = Instruction(unit=mte3_1, tensor=f, deps=[add_f], label="A1: UB->GM f")

    sim = Sim(chip, [load_a, load_b, add_c, store_c, load_d, load_e, add_f, store_f])
    sim.run()

    print(f"total cycles: {sim.total_cycles():.0f}  (single-core baseline for one add: 200; for two serialized: 400)")
    for instr in sorted(sim.instructions, key=lambda i: i.start_time):
        qn = instr.unit.qualified_name().split(".", 1)[1]  # drop "Chip."
        print(f"  {qn:<20s} {instr.start_time:>4.0f}..{instr.end_time:<4.0f}  {instr.label}")


if __name__ == "__main__":
    main()
