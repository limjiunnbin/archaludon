"""Parse MLIR linalg, lower it to Instructions for the NPU spec, and simulate.

Pipeline: .mlir text  ->  arch_sim.frontend.parse  ->  ir.Graph
              ir.Graph  ->  arch_sim.frontend.lower  ->  list[Instruction]
              instructions  ->  arch_sim.sim.simulate  ->  total cycles
"""
from pathlib import Path

from arch_sim.arch import load
from arch_sim.frontend import lower, parse_file
from arch_sim.sim import simulate


def run(chip, mlir_path: Path) -> float:
    graph = parse_file(mlir_path)
    instrs = lower(graph, chip)
    return simulate(instrs)


def main() -> None:
    chip = load(Path(__file__).resolve().parents[1] / "specs" / "npu.yaml")
    kernels = Path(__file__).resolve().parent / "kernels"
    for mlir in sorted(kernels.glob("*.mlir")):
        cycles = run(chip, mlir)
        print(f"{mlir.name:<16s} -> {cycles:.0f} cycles")


if __name__ == "__main__":
    main()
