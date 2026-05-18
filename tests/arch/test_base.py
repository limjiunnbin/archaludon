from arch_sim.arch import Module, StorageUnit, UnitKind


def test_kind_set_from_classvar():
    s = StorageUnit(name="L1", capacity_bytes=1024)
    assert s.kind is UnitKind.STORAGE


def test_qualified_name_walks_parents():
    chip = Module("Chip")
    core = chip.add_module("AICore")
    l1 = core.add_storage("L1", capacity_bytes=1024)
    assert l1.qualified_name() == "Chip.AICore.L1"
    assert core.qualified_name() == "Chip.AICore"
    assert chip.qualified_name() == "Chip"


def test_parent_set_on_add_child():
    chip = Module("Chip")
    core = chip.add_module("AICore")
    assert core.parent is chip
    l1 = core.add_storage("L1")
    assert l1.parent is core


def test_walk_yields_self_and_descendants():
    chip = Module("Chip")
    core = chip.add_module("AICore")
    l1 = core.add_storage("L1")
    seen = set(chip.walk())
    assert {chip, core, l1} == seen


def test_find_resolves_dotted_path():
    chip = Module("Chip")
    core = chip.add_module("AICore")
    l1 = core.add_storage("L1")
    assert chip.find("AICore.L1") is l1
    assert chip.find("AICore") is core


def test_identity_equality_and_hash():
    a = StorageUnit(name="X")
    b = StorageUnit(name="X")
    assert a != b
    assert hash(a) != hash(b)
    assert a == a
    assert {a, b} == {a, b}
