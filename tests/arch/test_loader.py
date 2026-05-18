from pathlib import Path

import pytest

from arch_sim.arch import (
    ComputeUnit,
    DMAEngine,
    Module,
    StorageUnit,
    dump,
    load,
    loads,
    validate,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ascend_like.yaml"


def test_load_fixture_validates_clean():
    chip = load(FIXTURE)
    assert chip.name == "Chip"
    report = validate(chip)
    assert report.errors == []


def test_load_fixture_structure():
    chip = load(FIXTURE)
    core = chip.find("AICore")
    assert isinstance(core, Module)
    l1 = chip.find("AICore.L1Buffer")
    cube = chip.find("AICore.Cube")
    mte2 = chip.find("AICore.MTE2")
    assert isinstance(l1, StorageUnit)
    assert isinstance(cube, ComputeUnit) and cube.operand_shape == (16, 16, 16)
    assert isinstance(mte2, DMAEngine)


def test_roundtrip_dump_load_is_structurally_equal():
    chip = load(FIXTURE)
    text = dump(chip)
    chip2 = loads(text)
    assert _shape(chip) == _shape(chip2)


def test_loads_rejects_non_mapping():
    with pytest.raises(ValueError):
        loads("- just\n- a\n- list\n")


def _shape(mod: Module):
    """Compare structurally: names + kinds + path endpoints + path engines, recursively."""
    children = []
    for c in mod.children:
        if isinstance(c, Module):
            children.append(_shape(c))
        else:
            children.append((c.name, c.kind.value, _unit_extras(c)))
    paths = [
        (p.src.qualified_name(), p.dst.qualified_name(),
         p.engine.qualified_name() if p.engine else None,
         p.direction.value, p.name)
        for p in mod.paths
    ]
    return (mod.name, mod.kind.value, children, paths)


def _unit_extras(unit):
    if isinstance(unit, StorageUnit):
        return ("storage", unit.capacity_bytes, unit.banks, unit.read_ports, unit.write_ports)
    if isinstance(unit, ComputeUnit):
        return ("compute", unit.operation, unit.throughput_ops_per_cycle, unit.operand_shape)
    if isinstance(unit, DMAEngine):
        return (
            "dma",
            tuple(k.value for k in unit.allowed_src_kinds),
            tuple(k.value for k in unit.allowed_dst_kinds),
            tuple(unit.allowed_src_names) if unit.allowed_src_names else None,
            tuple(unit.allowed_dst_names) if unit.allowed_dst_names else None,
            unit.bandwidth,
        )
    return (unit.kind.value,)
