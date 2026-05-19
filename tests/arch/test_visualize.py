from pathlib import Path

from arch_sim.arch import load, to_dot

FIXTURE = Path(__file__).parent / "fixtures" / "ascend_like.yaml"


def test_dot_has_digraph_header():
    out = to_dot(load(FIXTURE))
    assert out.startswith("digraph arch {")
    assert out.rstrip().endswith("}")


def test_dot_contains_unit_names_and_engine_labels():
    out = to_dot(load(FIXTURE))
    for name in ("L1Buffer", "Cube", "Vector", "UB", "GM"):
        assert name in out
    # Pipes surface as edge labels.
    for engine in ("MTE1", "MTE2", "MTE3"):
        assert f'label="{engine}"' in out


def test_dot_emits_subgraph_per_module():
    out = to_dot(load(FIXTURE))
    assert "subgraph cluster_" in out
    # Chip + AICore -> two clusters minimum.
    assert out.count("subgraph cluster_") >= 2


def test_dot_has_directed_edges():
    out = to_dot(load(FIXTURE))
    assert " -> " in out
