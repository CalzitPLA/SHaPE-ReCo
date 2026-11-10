"""reco_engine/run_path.py: the run's path from synthetic step streams - grid header and voxel -> world boxes on an
asymmetric off-centre grid, operations added / truncated / re-added, candidates, the classic-solution swap, and the
program rebuilt from the path."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reco_engine.result_schema import Operation, Primitive, ReconstructionResult  # noqa: E402
from reco_engine.run_path import PathRecorder, load_path, program_from_path  # noqa: E402


def box(cx, label=None):
    return Primitive("box", center_mm=(cx, 0.5, -1.0), dims_mm=dict(x_half=1.0, y_half=2.0, z_half=0.5), label=label)


def cyl(cx):
    return Primitive("cylinder", quaternion=(0.0, 0.0, 0.38268343, 0.92387953), center_mm=(cx, 1.0, 2.0),
                     dims_mm=dict(radius=0.4, half_length=3.0), u=(0.0, 3.0))


def snap(*ops):
    return ReconstructionResult("part.stl", [Operation(op, p) for op, p in ops])


def step(kind, result, **info):
    return SimpleNamespace(kind=kind, result=result, info=info)


ORIGIN, PITCH = np.array([1.5, -2.0, 0.25]), 0.5


def grid_spec():
    i, j, k = np.meshgrid(np.arange(4), np.arange(5), np.arange(6), indexing="ij")
    return SimpleNamespace(centres=ORIGIN + PITCH * np.stack([i, j, k], axis=-1))


def run(tmp_path, steps):
    rec = PathRecorder(tmp_path / "path.jsonl")
    for i, s in enumerate(steps):
        rec.feed(i, s, 0.1 * i)
    rec.close()
    return load_path(tmp_path / "path.jsonl")


def test_grid_and_world_boxes(tmp_path):
    log = run(tmp_path, [step("input", snap(), spec=grid_spec()),
                         step("rejected", snap(("add", box(0.0))), op="cut", bbox=((1, 2, 3), (2, 3, 4)), reason="no fit")])
    assert np.allclose(log.grid["origin"], ORIGIN) and log.grid["pitch"] == PITCH and log.grid["shape"] == [4, 5, 6]
    rej = next(e for e in log.events if e.get("kind") == "rejected")
    lo, hi = np.array(rej["bbox_world"][0]), np.array(rej["bbox_world"][1])
    assert np.allclose(lo, ORIGIN + (np.array([1, 2, 3]) - 0.5) * PITCH)
    assert np.allclose(hi, ORIGIN + (np.array([2, 3, 4]) + 0.5) * PITCH)


def test_operations_added_truncated_and_readded(tmp_path):
    base, c1, c2 = ("add", box(0.0, "base")), ("cut", cyl(0.5)), ("cut", cyl(-0.5))
    log = run(tmp_path, [
        step("input", snap(), spec=grid_spec()),
        step("add", snap(base), primitive=base[1]),
        step("candidate", snap(base), op="cut", primitive=cyl(9.0), gain=-3.0, ok=False, reason="harms the part"),
        step("cut", snap(base, c1), op="cut", primitive=c1[1], gain=12.0),
        step("feature_confirmed", snap(base), phase="feature", reason="confirmation failed"),   # truncation
        step("cut", snap(base, c2), op="cut", primitive=c2[1], gain=8.0),
        step("done", snap(base, c2), stopped="residual within tolerance"),
    ])
    assert [e["op"] for e in log.final] == ["add", "cut"]
    assert log.final[1]["primitive"]["center_mm"] == [-0.5, 1.0, 2.0]            # the re-added cut, not the withdrawn one
    assert [e["op_index"] for e in log.removed] == [1]
    assert log.removed[0]["kind"] == "feature_confirmed"
    cand = [e for e in log.candidates if e.get("reason") == "harms the part"]
    assert len(cand) == 1 and cand[0]["n_ops"] == 1 and cand[0]["primitive"]["center_mm"][0] == 9.0
    assert log.stopped == "residual within tolerance"
    steps = [e for e in log.events if e.get("event") == "step"]
    assert [e["n_ops"] for e in steps] == [0, 1, 1, 2, 1, 2, 2]


def test_classic_branch_becomes_the_final_program(tmp_path):
    base, c1 = ("add", box(0.0)), ("cut", cyl(0.5))
    other = ("add", box(3.0))
    log = run(tmp_path, [
        step("add", snap(base), primitive=base[1]),
        step("cut", snap(base, c1), op="cut", primitive=c1[1]),
        step("solution", snap(other), solution="classic", kept="classic solution"),
        step("done", snap(other), solution="classic", stopped="classic kept"),
    ])
    assert log.branch == "classic"
    assert [e["primitive"]["center_mm"][0] for e in log.final] == [3.0]


def test_program_from_path_rebuilds_the_operations(tmp_path):
    comp = Primitive("composite", parts=[box(0.0), cyl(0.0)], combine="cut")
    ops = (("add", box(0.0)), ("cut", cyl(0.7)), ("add", comp))
    log = run(tmp_path, [step("add", snap(*ops[:1])), step("cut", snap(*ops[:2])), step("add", snap(*ops))])
    result = program_from_path(log, "part.stl")
    assert [o.op for o in result.operations] == ["add", "cut", "add"]
    assert result.operations[1].primitive.u == (0.0, 3.0)
    assert result.operations[2].primitive.kind == "composite" and len(result.operations[2].primitive.parts) == 2
    for line in (tmp_path / "path.jsonl").read_text().splitlines():
        json.loads(line)                                                      # every line is plain JSON
