"""Dispatcher head-of-line blocking, illustrated by changing one knob.

Three instructions in program order:
    [m0 (MTE2),  m1 (MTE2),  v  (Vector)   ]    # v is independent of m0, m1

With MTE2.queue_depth = 1, the in-order dispatcher reaches m1 second and stalls
there (MTE2's inflight queue is full of m0). It can't proceed to v until m0
retires, so the Vector unit sits idle for 64 cycles before v even starts.

With MTE2.queue_depth = 2, both moves are admitted at t=0 and the dispatcher
walks straight to v, which starts immediately on Vector.

Total cycles don't change (MTE2 is the bottleneck either way), but the
per-instruction schedule does — and on a denser workload that head-of-line
stall would push v onto the critical path.
"""
from pathlib import Path

import torch

from arch_sim.arch import load
from arch_sim.sim import Instruction, simulate


def run(chip, mte2_qd: int):
    mte2 = chip.find("AICore.MTE2")
    vector = chip.find("AICore.Vector")
    mte2.queue_depth = mte2_qd

    def meta():
        return torch.empty(1024, dtype=torch.float32, device="meta")

    m0 = Instruction(unit=mte2, tensor=meta(), label="m0  (MTE2)")
    m1 = Instruction(unit=mte2, tensor=meta(), label="m1  (MTE2)")
    v = Instruction(unit=vector, tensor=meta(), label="v   (Vector, independent)")
    instrs = [m0, m1, v]
    simulate(instrs)
    return instrs


def main() -> None:
    chip = load(Path(__file__).resolve().parents[1] / "specs" / "npu.yaml")
    for qd in (1, 2):
        print(f"--- MTE2.queue_depth = {qd} ---")
        instrs = run(chip, qd)
        for i in instrs:
            print(f"  {i.start_time:>4.0f}..{i.end_time:<4.0f}  {i.label}")
        v_start = next(i for i in instrs if i.unit.name == "Vector").start_time
        print(f"  Vector started at t={v_start:.0f}")
        print()


if __name__ == "__main__":
    main()
