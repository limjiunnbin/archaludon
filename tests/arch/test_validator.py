from pathlib import Path

from arch_sim.arch import (
    DataPath,
    DMAEngine,
    Direction,
    Module,
    StorageUnit,
    UnitKind,
    load,
    validate,
)
from arch_sim.arch.validator import ValidationReport

FIXTURE = Path(__file__).parent / "fixtures" / "ascend_like.yaml"


def test_reference_arch_has_no_errors():
    report = validate(load(FIXTURE))
    assert report.errors == []
    assert report.ok
    assert bool(report)


def test_name_collision_flagged():
    chip = Module("Chip")
    chip.add_storage("L1")
    # Bypass add_storage to force a duplicate name (which builder/connect won't catch).
    dup = StorageUnit(name="L1")
    chip.add_child(dup)
    report = validate(chip)
    assert any("collision" in e and "L1" in e for e in report.errors)


def test_path_referencing_outside_tree_flagged():
    chip = Module("Chip")
    a = chip.add_storage("A")
    stray = StorageUnit(name="Stray")  # never added to any module
    chip.paths.append(DataPath(src=a, dst=stray, direction=Direction.UNI, name="leak"))
    report = validate(chip)
    assert any("not in module tree" in e for e in report.errors)


def test_engine_kind_mismatch_caught_by_validator():
    chip = Module("Chip")
    a = chip.add_storage("A")
    b = chip.add_storage("B")
    eng = chip.add_dma(
        "M",
        allowed_src_kinds=[UnitKind.STORAGE],
        allowed_dst_kinds=[UnitKind.COMPUTE],
    )
    # Path appended directly so we skip the connect-time check.
    chip.paths.append(DataPath(src=a, dst=b, engine=eng, name="badkind"))
    report = validate(chip)
    assert any("badkind" in e and "kind" in e for e in report.errors)


def test_orphan_storage_warns():
    chip = Module("Chip")
    chip.add_storage("Lonely")
    report = validate(chip)
    assert any("Lonely" in w for w in report.warnings)


def test_port_oversubscription_warns():
    chip = Module("Chip")
    src = chip.add_storage("Src", read_ports=1)
    d1 = chip.add_storage("D1", write_ports=1)
    d2 = chip.add_storage("D2", write_ports=1)
    chip.connect(src, d1, name="p1")
    chip.connect(src, d2, name="p2")
    report = validate(chip)
    assert any("read ports" in w for w in report.warnings)


def test_validation_report_ok_false_when_errors_present():
    rep = ValidationReport(errors=["boom"])
    assert not rep.ok
    assert not bool(rep)
