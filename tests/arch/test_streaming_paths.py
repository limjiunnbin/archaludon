"""Declarative compute->compute streaming links (phase 1.5).

`DataPath.stream` is spec metadata: the builder/loader carry it, the validator
checks it (compute->compute, acyclic), and the visualizer renders it. It does
not feed the simulator yet — that wiring is MLIR lowering, done later.
"""
from pathlib import Path

from arch_sim.arch import (
    Direction,
    Module,
    UnitKind,
    dump,
    load,
    loads,
    to_dot,
    validate,
)
from arch_sim.arch.builder import build

NPU = Path(__file__).parents[2] / "specs" / "npu.yaml"


def _two_compute_chip(stream=True, stream_latency=0.0, fifo_depth=1):
    chip = Module("Chip")
    core = chip.add_module("AICore")
    cube = core.add_compute("Cube", operation="matmul")
    vector = core.add_compute("Vector", operation="vector")
    core.connect(
        cube, vector, name="cube_to_vector",
        stream=stream, stream_latency=stream_latency, fifo_depth=fifo_depth,
    )
    return chip


def test_compute_to_compute_stream_builds_and_validates():
    chip = _two_compute_chip()
    path = chip.find("AICore").paths[0]  # type: ignore[attr-defined]
    assert path.stream is True
    report = validate(chip)
    assert report.errors == []
    # A legitimate compute->compute stream produces no stream warnings.
    assert not any("stream path" in w for w in report.warnings)


def test_stream_fields_default_off():
    chip = Module("Chip")
    a = chip.add_compute("A")
    b = chip.add_compute("B")
    p = chip.connect(a, b, name="plain")
    assert p.stream is False
    assert p.stream_latency == 0.0
    assert p.fifo_depth == 1


def test_stream_path_to_storage_warns():
    chip = Module("Chip")
    core = chip.add_module("AICore")
    cube = core.add_compute("Cube", operation="matmul")
    ub = core.add_storage("UB", capacity_bytes=1024)
    core.connect(cube, ub, name="cube_to_ub", stream=True)
    report = validate(chip)
    # Streaming into storage is a spec mistake — warn, don't error.
    assert report.ok
    assert any("cube_to_ub" in w and "storage" in w for w in report.warnings)


def test_stream_path_with_engine_warns():
    chip = Module("Chip")
    core = chip.add_module("AICore")
    cube = core.add_compute("Cube", operation="matmul")
    vector = core.add_compute("Vector", operation="vector")
    pipe = core.add_pipe(
        "FixPipe",
        allowed_src_kinds=[UnitKind.COMPUTE],
        allowed_dst_kinds=[UnitKind.COMPUTE],
    )
    core.connect(cube, vector, name="streamed_with_engine", stream=True, engine=pipe)
    report = validate(chip)
    assert any("streamed_with_engine" in w and "engine" in w for w in report.warnings)


def test_cyclic_stream_edges_error():
    chip = Module("Chip")
    core = chip.add_module("AICore")
    a = core.add_compute("A")
    b = core.add_compute("B")
    c = core.add_compute("C")
    core.connect(a, b, name="a_b", stream=True)
    core.connect(b, c, name="b_c", stream=True)
    core.connect(c, a, name="c_a", stream=True)  # closes the loop
    report = validate(chip)
    assert not report.ok
    assert any("cyclic streaming" in e for e in report.errors)


def test_acyclic_stream_chain_is_fine():
    chip = Module("Chip")
    core = chip.add_module("AICore")
    a = core.add_compute("A")
    b = core.add_compute("B")
    c = core.add_compute("C")
    core.connect(a, b, name="a_b", stream=True)
    core.connect(b, c, name="b_c", stream=True)
    report = validate(chip)
    assert not any("cyclic" in e for e in report.errors)


def test_build_from_dict_reads_stream_keys():
    spec = {
        "name": "Chip",
        "kind": "module",
        "children": [
            {"name": "Cube", "kind": "compute", "operation": "matmul"},
            {"name": "Vector", "kind": "compute", "operation": "vector"},
        ],
        "paths": [
            {"name": "cube_to_vector", "src": "Cube", "dst": "Vector",
             "stream": True, "stream_latency": 4, "fifo_depth": 2},
        ],
    }
    chip = build(spec)
    p = chip.paths[0]
    assert p.stream is True
    assert p.stream_latency == 4
    assert p.fifo_depth == 2


def test_stream_fields_roundtrip_through_dump_load():
    """The loader must emit and the builder must read the SAME keys — guards
    against a silent key-name mismatch that _shape()-based tests miss."""
    chip = _two_compute_chip(stream=True, stream_latency=3, fifo_depth=4)
    reloaded = loads(dump(chip))
    p = reloaded.find("AICore").paths[0]  # type: ignore[attr-defined]
    assert p.stream is True
    assert p.stream_latency == 3
    assert p.fifo_depth == 4
    # Idempotent: dump->load->dump is a fixed point on the stream keys.
    assert dump(reloaded) == dump(chip)


def test_default_stream_fields_not_emitted():
    """A non-stream path stays clean in YAML — no stream keys leak in."""
    chip = Module("Chip")
    a = chip.add_compute("A")
    b = chip.add_compute("B")
    chip.connect(a, b, name="plain")
    text = dump(chip)
    assert "stream" not in text
    assert "fifo_depth" not in text


def test_visualizer_renders_stream_edge_distinctly():
    chip = _two_compute_chip(stream=True, stream_latency=2)
    out = to_dot(chip)
    assert 'color="firebrick"' in out
    assert "style=bold" in out
    assert 'label="stream L2"' in out


def test_npu_spec_has_clean_stream_link():
    """The shipped spec's cube_to_vector stream path loads and validates clean."""
    chip = load(NPU)
    core = chip.find("AICore")
    assert isinstance(core, Module)
    streamed = [p for p in core.paths if p.stream]
    assert len(streamed) == 1
    sp = streamed[0]
    assert sp.name == "cube_to_vector"
    assert sp.src.name == "Cube" and sp.dst.name == "Vector"
    assert sp.direction == Direction.UNI
    report = validate(chip)
    assert report.errors == []
    assert not any("stream path" in w for w in report.warnings)
