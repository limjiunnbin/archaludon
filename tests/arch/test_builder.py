"""Tests for the fluent (Module.add_*/connect) and dict (builder.build) APIs."""
from arch_sim.arch import (
    ComputeUnit,
    Module,
    Pipe,
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
    mte1 = core.add_pipe(
        "MTE1",
        allowed_src_kinds=[UnitKind.STORAGE],
        allowed_dst_kinds=[UnitKind.STORAGE],
        allowed_src_names=["L1Buffer"],
        allowed_dst_names=["L0A", "L0B"],
    )
    mte2 = core.add_pipe(
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
    assert isinstance(mte1, Pipe)
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
                     "kind": "pipe",
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


def test_count_replicates_module_and_paths():
    spec = {
        "name": "Chip",
        "kind": "module",
        "children": [
            {"name": "GM", "kind": "storage", "capacity_bytes": 1024},
            {
                "name": "AICore",
                "kind": "module",
                "count": 2,
                "children": [
                    {"name": "UB", "kind": "storage", "capacity_bytes": 256},
                    {"name": "MTE2", "kind": "pipe",
                     "allowed_src_kinds": ["storage"],
                     "allowed_dst_kinds": ["storage"],
                     "allowed_src_names": ["GM"],
                     "allowed_dst_names": ["UB"],
                     "bandwidth": 64},
                ],
            },
        ],
        "paths": [
            {"name": "gm_to_ub", "src": "GM", "dst": "AICore.UB", "engine": "AICore.MTE2"},
        ],
    }
    chip = build(spec)

    assert {c.name for c in chip.children if isinstance(c, Module)} == {"AICore0", "AICore1"}
    # Each replica has its own UB and MTE2 (distinct objects).
    a0_ub = chip.find("AICore0.UB")
    a1_ub = chip.find("AICore1.UB")
    assert a0_ub is not a1_ub
    # Top-level path replicated per replica.
    assert len(chip.paths) == 2
    assert {p.name for p in chip.paths} == {"gm_to_ub_AICore0", "gm_to_ub_AICore1"}
    assert {p.dst.qualified_name() for p in chip.paths} == {"Chip.AICore0.UB", "Chip.AICore1.UB"}


def test_count_zero_pads_to_width():
    spec = {
        "name": "X",
        "kind": "module",
        "children": [{"name": "C", "kind": "module", "count": 20, "children": []}],
    }
    chip = build(spec)
    names = sorted(c.name for c in chip.children)
    assert names[0] == "C00"
    assert names[-1] == "C19"


def test_count_internal_paths_independent_per_replica():
    """Each replica should get its own deep-copied internal paths."""
    spec = {
        "name": "Chip",
        "kind": "module",
        "children": [
            {
                "name": "Core",
                "kind": "module",
                "count": 2,
                "children": [
                    {"name": "A", "kind": "storage", "capacity_bytes": 64},
                    {"name": "B", "kind": "storage", "capacity_bytes": 64},
                ],
                "paths": [{"name": "a_to_b", "src": "A", "dst": "B"}],
            },
        ],
    }
    chip = build(spec)
    core0 = chip.find("Core0")
    core1 = chip.find("Core1")
    assert isinstance(core0, Module) and isinstance(core1, Module)
    assert len(core0.paths) == 1 and len(core1.paths) == 1
    # Each replica's path points at its own A/B, not the other replica's.
    assert core0.paths[0].src is chip.find("Core0.A")
    assert core1.paths[0].src is chip.find("Core1.A")
