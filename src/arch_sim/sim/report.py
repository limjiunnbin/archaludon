from dataclasses import dataclass


@dataclass
class Report:
    num_cycles: int = 0
    bandwidth_utilization: float = 1.0

    def display(self):
        print(f"total cycles: {self.num_cycles}")
        print(f"bandwidth utilization: {self.bandwidth_utilization:.2%}")
