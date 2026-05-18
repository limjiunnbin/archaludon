from arch_sim.arch import ComputeUnit, UnitKind


def test_defaults():
    c = ComputeUnit(name="Vector")
    assert c.operation == "generic"
    assert c.throughput_ops_per_cycle == 1.0
    assert c.operand_shape is None
    assert c.kind is UnitKind.COMPUTE


def test_fixed_shape_unit():
    cube = ComputeUnit(
        name="Cube",
        operation="matmul",
        operand_shape=(16, 16, 16),
        throughput_ops_per_cycle=4096,
    )
    assert cube.operand_shape == (16, 16, 16)
    assert cube.operation == "matmul"
