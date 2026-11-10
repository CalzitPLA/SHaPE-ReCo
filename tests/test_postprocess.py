"""reco_gui/postprocess.py: the post-processing window's core (sampling, mass shares, benchmark, minimize) and the window
itself (offscreen, no 3D) on a simple part whose program carries parts it does not need."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"

from reco_engine.result_schema import Operation, Primitive, ReconstructionResult  # noqa: E402


def _box(c, h):
    return Primitive(kind="box", center_mm=tuple(float(x) for x in c), dims_mm=dict(x_half=h[0], y_half=h[1], z_half=h[2]))


def _part():
    """The input: a 20 x 20 x 10 box. The program: the box, a redundant add inside it, a boss outside the part (invented
    material) and a cut into the part (missing material) - a simple part with a needlessly complex tree."""
    import trimesh
    m = trimesh.creation.box((20.0, 20.0, 10.0))
    ops = [Operation("add", _box((0, 0, 0), (10, 10, 5))),
           Operation("add", _box((2, 2, 0), (3, 3, 3))),                 # inside the box: nothing of its own
           Operation("add", _box((0, 0, 7), (2, 2, 2))),                 # on top: outside the input
           Operation("cut", _box((-6, -6, 3), (2, 2, 2.5)))]             # into the part: removes input material
    return ReconstructionResult(input_path="box.stl", operations=ops), (np.asarray(m.vertices), np.asarray(m.faces))


def test_shares_fit_and_minimize_on_a_simple_part():
    from reco_gui import postprocess as pp
    result, mesh = _part()
    s = pp.sample_program(result, mesh, n=60_000)
    keep = np.ones(4, bool)
    rows = pp.part_table(s, keep)
    adds = [r for r in rows if r["op"] == "add"]
    assert sum(r["share"] for r in adds) == pytest.approx(1.0, abs=1e-9)          # the adds hold the whole volume
    assert rows[1]["unique"] == 0.0 and rows[0]["unique"] > 0                       # the inner add has nothing of its own
    assert rows[2]["fit"] == pytest.approx(0.0, abs=0.02)                           # the boss lies outside the input
    assert rows[3]["fit"] == pytest.approx(0.0, abs=0.02) and rows[3]["removed"] > 0   # the cut removes input material
    m0 = pp.metrics(s, keep)
    assert m0["iou"] < 0.99 and m0["missing"] > 0 and m0["invented"] > 0
    keep2, log = pp.minimize(s, keep, tol=0.002)
    assert keep2.tolist() == [True, False, False, False] and len(log) == 3          # one box is the program
    assert pp.metrics(s, keep2)["iou"] == pytest.approx(1.0, abs=1e-6)
    assert len(pp.edited_result(result, keep2).operations) == 1
    _, log0 = pp.minimize(s, keep, tol=0.0)                                         # with no tolerance: only what does not hurt
    assert {d["index"] for d in log0} == {1, 2, 3}                                  # (all three of them, on this part)
    from reco_gui.op_tree import csg_tree
    a, b = pp.group_share(csg_tree(s.ops), rows)
    assert a == pytest.approx(1.0) and b == pytest.approx(rows[3]["share"])


def test_minimize_keeps_the_parts_the_shape_needs():
    """Without an input the model itself is the reference: only the parts that do not change it may go."""
    from reco_gui import postprocess as pp
    result, _ = _part()
    s = pp.sample_program(result, None, n=60_000)
    keep2, log = pp.minimize(s, np.ones(4, bool), tol=0.0)
    assert keep2.tolist() == [True, False, True, True]                              # only the redundant add goes
    assert log[0]["index"] == 1 and log[0]["change"] == 0.0


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_the_window_selects_benchmarks_minimizes_decides_and_exports(qapp, tmp_path):
    from reco_gui.postprocess import PostprocessWindow
    result, mesh = _part()
    w = PostprocessWindow(result, input_path=None, mesh=mesh, samples=40_000)
    w.wait()
    assert w.sample is not None and len(w._items) == 4
    assert "%" in w._items[0].text(1)                                               # mass ratios in the tree
    w.select_parts([2])
    assert w.selected_parts() == [2] and "without the selection" in w.report.toHtml()
    w._decide("drop")
    assert w.keep.tolist() == [True, True, False, True] and w._items[2].text(5) == "drop"
    w.tol.setValue(0.002)
    w._minimize()
    w.wait()
    assert w.proposed == {1, 3} and w.b_accept.isEnabled() and "with the proposal" in w.report.toHtml()
    w._accept_proposal()
    assert w.keep.tolist() == [True, False, False, False]
    decided = w.decided()
    assert len(decided.operations) == 1 and decided.operations[0].primitive.dims_mm["x_half"] == 10
    path = w.export_json(str(tmp_path / "post.json"))
    assert len(json.loads(Path(path).read_text())["operations"]) == 1
    run_dir = w.save_run(store=tmp_path / "store")
    rec = json.loads((Path(run_dir) / "run.json").read_text())
    assert rec["label"] == "post-processed" and rec["extra"]["postprocess"]["parts_after"] == 1
    assert rec["extra"]["postprocess"]["dropped"] == [2, 3, 4]
    w._reset()
    assert w.keep.all()
    w.close()


def test_the_loops_prune_step_keeps_one_valid_solid(tmp_path):
    """ResidualConfig.prune_parts: the drops of minimize in their order (the cut, then the boss, then the redundant add);
    a fake FreeCAD build that splits the model when the redundant add is gone: the longest prefix that stays one valid
    solid (the cut and the boss) is pruned, the rest stays. The pruned parts stay in the program, not accepted."""
    from types import SimpleNamespace
    from reco_engine import residual_loop as rl
    result, mesh = _part()
    field = SimpleNamespace(vertices=mesh[0], faces=mesh[1])
    cfg = rl.ResidualConfig(prune_parts=True, prune_samples=60_000, out_dir=str(tmp_path))
    builds = []

    def build(r, out_dir, name, timeout):
        has_inner = any(o.primitive.center_mm == (2.0, 2.0, 0.0) for o in r.operations)
        builds.append(name)
        return dict(valid=True, n_solids=1 if has_inner else 2)
    gen = rl._prune_parts(result, field, cfg, build=build)
    steps = []
    try:
        while True:
            steps.append(next(gen))
    except StopIteration as stop:
        text = stop.value
    assert [s.kind for s in steps] == ["prune"] and steps[0].info["dropped"] == [4, 3]       # the cut and the boss
    assert steps[0].info["proposed"] == 3 and "pruned 2 of 4 parts" in text
    assert [o.accepted for o in result.operations] == [True, True, False, False]
    assert all("pruned" in o.primitive.label for o in result.operations[2:])
    assert len(builds) == 3                                   # all three drops, then the binary search: 1 and 2
    assert steps[0].info["iou_after"] > steps[0].info["iou_before"]
