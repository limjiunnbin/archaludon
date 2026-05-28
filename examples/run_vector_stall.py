"""Vector op stalls because its destination buffer (UB) fills up.

A burst of Vector ops each write a result into UB; MTE3 drains UB to GM 8x slower
than Vector produces. We give Vector and MTE3 deep enough instruction queues so the
dispatcher can issue all ops ahead — that lets Vector outrun the drain, fills UB
(capacity 2 results), and forces a finished Vector op to wait at retirement.

If you instead set Vector/MTE3 queue_depth=1, the same workload still totals 264
cycles but the stall location moves to the dispatcher (it can't issue past a full
MTE3 queue); change the depths to see the difference.

Assumes inputs already resident in UB (no loads modeled here).
"""
from pathlib import Path

import torch

from arch_sim.arch import load
from arch_sim.sim import Instruction, simulate
from arch_sim.sim.report import Report


def main() -> None:
    chip = load(Path(__file__).resolve().parents[1] / "specs" / "npu.yaml")
    vector = chip.find("AICore.Vector")
    mte3 = chip.find("AICore.MTE3")
    ub = chip.find("AICore.UB")
    ub.queue_depth = 2     # UB holds at most 2 pending result tensors
    vector.queue_depth = 8  # let the dispatcher issue ahead
    mte3.queue_depth = 8

    def meta():
        return torch.empty(1024, dtype=torch.float32, device="meta")

    instrs: list[Instruction] = []
    for i in range(4):
        v = Instruction(unit=vector, tensor=meta(), dst=ub, label=f"v{i} = op(x{i})")
        s = Instruction(unit=mte3, tensor=meta(), deps=[v], dst=None, label=f"store v{i}->GM")
        instrs += [v, s]

    total = simulate(instrs)

    print("per-instruction (start .. exec-end .. retire):")
    for instr in instrs:
        stall = instr.retire_time - instr.end_time
        flag = "   <-- STALLED" if stall > 0 else ""
        print(
            f"  {instr.unit.name:<7s} {instr.start_time:>4.0f}..{instr.end_time:<4.0f}"
            f" ret {instr.retire_time:<4.0f}  {instr.label}{flag}"
        )

    # bandwidth utilization = busiest pipe's execution cycles / total
    mte3_busy = sum(i.end_time - i.start_time for i in instrs if i.unit is mte3)
    report = Report(num_cycles=int(total), bandwidth_utilization=mte3_busy / total)
    print()
    report.display()


if __name__ == "__main__":
    main()
