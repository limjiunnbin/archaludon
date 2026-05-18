import pytest

from arch_sim.arch import StorageUnit, UnitKind


def test_defaults():
    s = StorageUnit(name="L1")
    assert s.capacity_bytes == 0
    assert s.banks == 1
    assert s.read_ports == 1
    assert s.write_ports == 1
    assert s.kind is UnitKind.STORAGE


def test_can_serve_within_budget():
    s = StorageUnit(name="L1", read_ports=2, write_ports=2)
    assert s.can_serve(2, 2)
    assert s.can_serve(1, 0)
    assert not s.can_serve(3, 0)
    assert not s.can_serve(0, 3)


def test_can_serve_rejects_negative():
    s = StorageUnit(name="L1", read_ports=1, write_ports=1)
    with pytest.raises(ValueError):
        s.can_serve(-1, 0)
