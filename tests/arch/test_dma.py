import pytest

from arch_sim.arch import (
    DataPath,
    DMAEngine,
    Direction,
    Module,
    StorageUnit,
    UnitKind,
)


def test_validate_allows_matching_kinds_and_names():
    src = StorageUnit(name="GM")
    dst = StorageUnit(name="L1Buffer")
    eng = DMAEngine(
        name="MTE2",
        allowed_src_kinds=[UnitKind.STORAGE],
        allowed_dst_kinds=[UnitKind.STORAGE],
        allowed_src_names=["GM"],
        allowed_dst_names=["L1Buffer", "UB"],
    )
    ok, reason = eng.validate(DataPath(src=src, dst=dst, engine=eng))
    assert ok, reason


def test_validate_rejects_disallowed_src_name():
    src = StorageUnit(name="L1Buffer")
    dst = StorageUnit(name="L0A")
    eng = DMAEngine(
        name="MTE1",
        allowed_src_kinds=[UnitKind.STORAGE],
        allowed_dst_kinds=[UnitKind.STORAGE],
        allowed_src_names=["L1Buffer"],
        allowed_dst_names=["L0A", "L0B"],
    )
    bad_src = StorageUnit(name="GM")
    ok, reason = eng.validate(DataPath(src=bad_src, dst=dst, engine=eng))
    assert not ok
    assert "GM" in reason


def test_validate_rejects_wrong_kind():
    src = StorageUnit(name="L1")
    dst = StorageUnit(name="L0A")  # storage, but engine only allows compute dst
    eng = DMAEngine(
        name="X",
        allowed_src_kinds=[UnitKind.STORAGE],
        allowed_dst_kinds=[UnitKind.COMPUTE],
    )
    ok, reason = eng.validate(DataPath(src=src, dst=dst, engine=eng))
    assert not ok
    assert "kind" in reason


def test_connect_raises_on_engine_violation():
    chip = Module("Chip")
    gm = chip.add_storage("GM")
    l1 = chip.add_storage("L1Buffer")
    mte = chip.add_dma(
        "MTE1",
        allowed_src_kinds=[UnitKind.STORAGE],
        allowed_dst_kinds=[UnitKind.STORAGE],
        allowed_src_names=["L1Buffer"],
        allowed_dst_names=["L0A", "L0B"],
    )
    with pytest.raises(ValueError, match="DMA validation failed"):
        chip.connect(gm, l1, engine=mte, name="bad")


def test_datapath_endpoints_helper():
    a = StorageUnit(name="A")
    b = StorageUnit(name="B")
    p = DataPath(src=a, dst=b, direction=Direction.BI)
    assert p.endpoints() == (a, b)
