"""Render the NPU spec via arch.visualize."""
from pathlib import Path

from arch_sim.arch import load, visualize

SPEC = Path(__file__).resolve().parents[1] / "specs" / "npu.yaml"
OUT = Path(__file__).with_name("npu.png")


def main() -> None:
    print(f"wrote {visualize(load(SPEC), OUT)}")


if __name__ == "__main__":
    main()
