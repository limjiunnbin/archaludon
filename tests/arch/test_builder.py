"""Tests for the fluent (Module.add_*/connect) and dict (builder.build) APIs."""
from arch_sim.arch import (
    ComputeUnit,
    DMAEngine,
    Module,
    StorageUnit,
    UnitKind,
)
from arch_sim.arch.builder import build


def test_fluent_builds_reference_topology():
    chip = Module("Chip")
    gm = chip.add_storage("GM", capacity_bytes=8 * 1024 * 1024 * 1024)
    core = chip.add_module("AICore")
    l1 = core.add_storage("L1Buffer", capacity_bytes=1024 * 1024)
    l0a = core.add_storage("L0A", capacity_bytes=64 * 1024)
    cube = core.add_compute("Cube", operation="matmul", operand_shape=(16, 16, 16))
    mte1 = core.add_dma(
        "MTE1",
        allowed_src_kinds=[UnitKind.STORAGE],
        allowed_dst_kinds=[UnitKind.STORAGE],
        allowed_src_names=["L1Buffer"],
        allowed_dst_names=["L0A", "L0B"],
    )
    mte2 = core.add_dma(
        "MTE2",
        allowed_src_kinds=[UnitKind.STORAGE],
        allowed_dst_kinds=[UnitKind.STORAGE],
        allowed_src_names=["GM"],
        allowed_dst_names=["L1Buffer"],
    )

    core.connect(l1, l0a, engine=mte1, name="l1_to_l0a")
    core.connect(l0a, cube, name="l0a_to_cube")
    chip.connect(gm, l1, engine=mte2, name="gm_to_l1")

    assert isinstance(l1, StorageUnit)
    assert isinstance(cube, ComputeUnit)
    assert isinstance(mte1, DMAEngine)
    assert len(core.paths) == 2
    assert len(chip.paths) == 1
    assert l1.qualified_name() == "Chip.AICore.L1Buffer"


def test_build_from_dict_matches_fluent():
    spec = {
        "name": "Chip",
        "kind": "module",
        "children": [
            {"name": "GM", "kind": "storage", "capacity_bytes": 1024},
            {
                "name": "AICore",
                "kind": "module",
                "children": [
                    {"name": "L1", "kind": "storage", "capacity_bytes": 512},
                    {"name": "Cube", "kind": "compute", "operation": "matmul",
                     "operand_shape": [16, 16, 16]},
                    {"name": "Mover",
                     "kind": "dma",
                     "allowed_src_kinds": ["storage"],
                     "allowed_dst_kinds": ["storage"],
                     "allowed_src_names": ["GM"],
                     "allowed_dst_names": ["L1"]},
                ],
                "paths": [
                    {"name": "l1_to_cube", "src": "L1", "dst": "Cube"},
                ],
            },
        ],
        "paths": [
            {"name": "gm_to_l1", "src": "GM", "dst": "AICore.L1", "engine": "AICore.Mover"},
        ],
    }
    chip = build(spec)
    assert len(chip.children) == 2
    assert len(chip.paths) == 1
    core = chip.find("AICore")
    assert isinstance(core, Module)
    assert len(core.paths) == 1


def test_build_rejects_unknown_kind():
    spec = {"name": "X", "kind": "module", "children": [{"name": "Y", "kind": "wat"}]}
    try:
        build(spec)
    except ValueError as e:
        assert "wat" in str(e)
    else:
        raise AssertionError("expected ValueError")
