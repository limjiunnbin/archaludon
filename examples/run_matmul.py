"""Simulate C = A @ B (64x64x64 fp32) on NPU.

Path: GM -> L1 -> L0A/L0B (via MTE2 then MTE1), Cube does the matmul into L0C,
FixPipe writes L0C -> GM.
"""
from pathlib import Path

import torch

from arch_sim.arch import load
from arch_sim.sim import Instruction, Sim


def main() -> None:
    chip = load(Path(__file__).resolve().parents[1] / "specs" / "npu.yaml")
    mte1 = chip.find("AICore.MTE1")
    mte2 = chip.find("AICore.MTE2")
    cube = chip.find("AICore.Cube")
    fixpipe = chip.find("AICore.FixPipe")

    M, N, K = 64, 64, 64
    a = torch.empty(M, K, dtype=torch.float32, device="meta")
    b = torch.empty(K, N, dtype=torch.float32, device="meta")
    c = torch.empty(M, N, dtype=torch.float32, device="meta")
    # v1 cost.Op uses numel; for matmul we want M*N*K MACs.
    # Encode that by handing Cube a meta tensor whose numel equals the MAC count.
    work = torch.empty(M, N, K, dtype=torch.float32, device="meta")

    load_a_l1 = Instruction(unit=mte2, tensor=a, label="GM->L1 A")
    load_b_l1 = Instruction(unit=mte2, tensor=b, label="GM->L1 B")
    load_a_l0a = Instruction(unit=mte1, tensor=a, deps=[load_a_l1], label="L1->L0A A")
    load_b_l0b = Instruction(unit=mte1, tensor=b, deps=[load_b_l1], label="L1->L0B B")
    matmul = Instruction(unit=cube, tensor=work, deps=[load_a_l0a, load_b_l0b], label="Cube C = A@B")
    store_c = Instruction(unit=fixpipe, tensor=c, deps=[matmul], label="L0C->GM C")

    sim = Sim(chip, [load_a_l1, load_b_l1, load_a_l0a, load_b_l0b, matmul, store_c])
    sim.run()

    print(f"matmul {M}x{K} @ {K}x{N} fp32 -> {sim.total_cycles():.0f} cycles")
    for instr in sim.instructions:
        print(f"  {instr.unit.name}: {instr.start_time:.0f}..{instr.end_time:.0f}  {instr.label}")


if __name__ == "__main__":
    main()
