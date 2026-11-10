"""
Open a stored run, headless (offscreen Qt, 3D view off): the rows of the store newest first, the dialog's filter and
choice, and what the window shows of the chosen run - summary, log, the note of a run without a model.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"

pytest.importorskip("PyQt5")
SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

from PyQt5 import QtWidgets  # noqa: E402

from reco_gui import open_run  # noqa: E402
from reco_gui.app import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def write_run(store, part, name, when, iou=None, model=True, label=None, finished=True):
    d = store / part / name
    d.mkdir(parents=True)
    box = dict(kind="box", dims_mm=dict(x_half=1.0, y_half=2.0, z_half=3.0), label=None)
    run = dict(schema=1, part=part, label=label, timestamp=f"2026-09-2{when}T10:00:00+0200", time=1000.0 + when, name=name,
               input=dict(path=f"/parts/{part}.step", sha256="x"), method="cascade", finished=finished,
               code=dict(short="abc1234", branch="bolt-holes", dirty=False), settings=dict(cascade_top_k=2),
               program=[dict(op="add", primitive=box), dict(op="cut", primitive=dict(box, kind="cylinder", label="hole stack"))],
               metrics=dict(exact_iou=iou, chamfer_mm=0.31, solution_stats=dict(iou=0.9)),
               model=dict(stl="model.stl") if model else {})
    (d / "run.json").write_text(json.dumps(run))
    (d / "steps.jsonl").write_text("\n".join(json.dumps(dict(index=i, stage=s, phase=p, message=m, seconds=1.5))
                                             for i, (s, p, m) in enumerate([("input", "", "6900 input voxels"),
                                                                            ("accepted", "cut", "cut cylinder")])))
    if model:
        (d / "model.stl").write_text("solid x\nendsolid x\n")
    return d


def tetrahedron(path):
    v = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 6.0]])
    return v, np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])


def test_rows_newest_first(tmp_path):
    write_run(tmp_path, "frame", "a", 1, iou=0.9655)
    write_run(tmp_path, "liner", "b", 3, iou=0.9723, label="holes")
    write_run(tmp_path, "liner", "c", 2, model=False, finished=False)
    rows = open_run.list_rows(tmp_path)
    assert [Path(p).name for p, _ in rows] == ["b", "c", "a"]
    assert rows[0][1][:6] == ["2026-09-23 10:00:00", "liner", "holes", "cascade", "2", "0.9723"]
    assert rows[1][1][5] == "" and rows[1][1][7] == "no"


def test_load_stored(tmp_path):
    d = write_run(tmp_path, "liner", "b", 3, iou=0.9723)
    s = open_run.load_stored(d, load_mesh=tetrahedron)
    assert s["mesh"] is not None and s["note"] is None
    assert "exact IoU 0.9723" in s["caption"] and "2 operations" in s["caption"]
    assert "hole stack" in s["html"] and "exact_iou" in s["html"] and "cascade_top_k" in s["html"]
    assert len(s["log"]) == 2 and "cut cylinder" in s["log"][1] and "(1.5 s)" in s["log"][1]
    bare = open_run.load_stored(write_run(tmp_path, "liner", "c", 2, model=False), load_mesh=tetrahedron)
    assert bare["mesh"] is None and "no model" in bare["note"]


def test_a_cut_program_and_its_recovery_are_shown(tmp_path):
    d = write_run(tmp_path, "liner", "b", 3, iou=0.9723)
    run = json.loads((d / "run.json").read_text())
    whole = [dict(o) for o in run["program"]]
    run["program"][1]["primitive"] = dict(run["program"][1]["primitive"], kind="composite",
                                          parts=["{'kind': 'cylinder', 'quaternion': (0.0, 0.0, 0.0, 1.0), 'center_mm'", whole[0]["primitive"]])
    (d / "run.json").write_text(json.dumps(run))
    s = open_run.load_stored(d, load_mesh=tetrahedron)
    assert "cut in 1 node(s)" in s["html"] and "recover_run_program" in s["html"]
    (d / "program_recovered.json").write_text(json.dumps(dict(program=whole)))
    s = open_run.load_stored(d, load_mesh=tetrahedron)
    assert "recovered (program_recovered.json)" in s["html"] and s["run"]["program"] == whole


def test_dialog_filter_and_choice(qapp, tmp_path):
    write_run(tmp_path, "frame", "a", 1, iou=0.9655)
    b = write_run(tmp_path, "liner", "b", 3, iou=0.9723)
    dialog = open_run.OpenRunDialog(store=tmp_path)
    assert dialog.table.rowCount() == 2
    dialog.filter.setText("frame")
    assert dialog.table.isRowHidden(0) and not dialog.table.isRowHidden(1)
    dialog._accept()                                              # the selected row is hidden: nothing is chosen
    assert dialog.chosen is None
    dialog.filter.setText("")
    dialog.table.selectRow(0)
    dialog._accept()
    assert dialog.chosen == str(b)


def test_window_opens_a_run(qapp, tmp_path):
    d = write_run(tmp_path, "liner", "b", 3, iou=0.9723)
    window = MainWindow()
    try:
        stored = window.open_run(d, load_mesh=tetrahedron)
        assert stored is not None
        assert "stored run b" in window.status_label.text() and "0.9723" in window.status_label.text()
        assert "stored run opened" in window.log.toPlainText() and "cut cylinder" in window.log.toPlainText()
        assert "Program (2 operations)" in window.detail.toPlainText()
        assert window.saved_run == str(d)
        assert window.open_run(tmp_path / "missing") is None and "Open failed" in window.status_label.text()
    finally:
        window.close()
