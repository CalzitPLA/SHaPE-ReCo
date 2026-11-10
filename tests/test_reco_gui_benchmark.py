"""
The Benchmark tab, headless (offscreen Qt, 3D view off): the stored runs of the chosen part as ranked rows, the rating
button, what the window shows of a clicked run (model, summary, rating, trees) and the user's choice.
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
sys.path.insert(0, str(SANDBOX_ROOT / "tests"))

from PyQt5 import QtWidgets  # noqa: E402

from reco_gui import benchmark_panel as bp  # noqa: E402
from reco_gui.app import MainWindow  # noqa: E402
from test_benchmark import BARE, CUBES, save, write_input, write_truth  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def tetrahedron(path):
    v = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 6.0]])
    return v, np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])


def test_row_texts_and_rating_html():
    r = dict(saved="2026-10-11 02:00:00", label=None, variant="cubes", ops=2, one_valid_solid=False, exact_iou=0.9876, voxel_iou=None,
             chamfer_mm=0.31, missing_volume=1.5, invented_volume=None, seconds=12.4, rated=True, tier="B", separation="clean", failed=[])
    t = bp.row_texts(3, r)
    assert t[:6] == ["3", "2026-10-11 02:00:00", "", "cubes", "2", "NO"] and t[6] == "0.9876" and t[7] == "" and t[11] == "12"
    assert t[12:15] == ["", "", ""]                                       # no event time, no nephogram yet
    assert t[bp.TIER_COLUMN] == "B" and t[bp.TIER_COLUMN + 1] == "clean"
    assert bp.row_texts(1, dict(r, rated=False, tier=None))[bp.TIER_COLUMN] == "-"
    assert bp.row_texts(1, dict(r, event_us=0.627, p95_missing=1.2e-5, p95_invented=0.42))[12:15] == ["0.627", "1.2e-05", "0.42"]
    assert len(t) == len(bp.COLUMNS)
    html = bp.rating_html(dict(grade=dict(tier="B", failed=[], limiting=["hausdorff"]), n_ops=2,
                               vector=dict(volume=dict(iou=0.97, missing=0.01, invented=0.02), ops_per_truth_leaf=1.0),
                               attribution=dict(separation=dict(cls="clean", culprit=1, cause="x"),
                                                by_op=[dict(op_index=0, op="add", delta_iou=0.7, wrong_writes=0.1, persisting=0.0, redundant=False)])))
    assert "hausdorff" in html and "0.9700" in html and "clean" in html and "0.700" in html
    assert bp.rating_html(None) == ""


def test_panel_in_the_window(qapp, tmp_path):
    store, stl = tmp_path / "store", write_input(tmp_path)
    write_truth(stl)
    d1 = save(store, stl, [("add", "box", dict(x_half=1.0, y_half=2.0, z_half=3.0))], BARE, iou=0.93, seconds=5.0)
    d2 = save(store, stl, [("add", "box", dict(x_half=1.0, y_half=2.0, z_half=3.0)), ("cut", "cylinder", dict(radius=0.5, half_length=4.0))],
              CUBES, iou=0.99, seconds=20.0, label="holes")
    window = MainWindow(stl=str(stl), autorun=False, autosave=False)
    panel = window.benchmark
    panel.store = store
    statuses = []
    panel.status.connect(statuses.append)
    window.show_benchmark()
    assert panel.input_path == stl and panel.table.rowCount() == 2
    assert "2 stored run(s)" in panel.part_label.text() and "part.json" in panel.part_label.text()
    assert panel.table.item(0, 3).text() == "cubes" and panel.table.item(0, 6).text() == "0.9900"   # ranked for accuracy
    assert panel.table.item(0, bp.TIER_COLUMN).text() == "-"                                      # not rated yet
    assert window.bottom_tabs.currentWidget() is panel and window.act_timeline.isChecked()
    # a click: the run's model, summary and trees
    shown = window.show_benchmark_run(str(d2), load_mesh=tetrahedron)
    assert shown is not None and "holes" in window.detail.toPlainText()
    tree = window.op_tree
    assert tree.model_root.text(1).startswith("holes, path cubes: 2 operation(s)")
    assert tree.history_root.text(0) == "Known tree" and tree.history_root.child(0).text(0) == "Subtraction"
    assert tree.history_root.child(0).child(1).text(0) == "cylinder"
    # rating, synchronously
    panel.rate(sync=True)
    assert any(s.startswith("Rated: 2 of 2") for s in statuses)
    assert panel.table.item(0, bp.TIER_COLUMN).text() in "ABCDE" and (d2 / "rating.json").is_file()
    panel.preference.setCurrentText("design")
    assert panel.rows[0]["tier"] is not None
    window.show_benchmark_run(str(d2), load_mesh=tetrahedron)
    assert "aegir rating against the known tree" in window.detail.toPlainText()
    # the choice: labelled "chosen", shown, bold in the table
    panel.table.setCurrentCell(0, 0)
    target = Path(panel.selected_dir())
    assert target == Path(panel.rows[0]["dir"])
    chosen = window.use_benchmark_run(str(target))
    assert chosen.name.endswith("_chosen") and not target.exists() and chosen.exists()
    assert [r for r in panel.rows if r["label"] == "chosen"] and "Chosen:" in window.status_label.text()
    assert panel.table.item(0, 2).font().bold() if panel.rows[0]["label"] == "chosen" else True
    # no known tree: Rate says so
    other = write_input(tmp_path, "other")
    save(store, other, [("add", "sphere", dict(radius=1.0))], BARE, iou=1.0)
    panel.set_input(other)
    assert panel.table.rowCount() == 1 and "known tree none" in panel.part_label.text()
    panel.rate(sync=True)
    assert statuses[-1].startswith("Rate: no known tree")
    window.show_benchmark_input()
    window.close()


def test_alternatives_checked_event_time_nephogram_and_the_event_time_ranking(qapp, tmp_path, monkeypatch):
    """The comparison of alternatives (the user, 2026-10-29): the newest runs checked by default, set_checked replaces
    the checks; 'Event time' scans only the checked runs without one (the scan itself is replaced here) and the column
    and the 'event time' ranking use it; 'Nephogram' emits the stored nephogram and shows its histograms."""
    import json as _json
    from reco_engine import event_time as et
    store, stl = tmp_path / "store", write_input(tmp_path)
    box = [("add", "box", dict(x_half=1.0, y_half=2.0, z_half=3.0))]
    dirs = [save(store, stl, box, BARE, iou=0.95 + 0.01 * i, seconds=5.0 + i) for i in range(4)]
    panel = bp.BenchmarkPanel(store=store)
    panel.set_input(stl)
    assert len(panel.rows) == 4 and len(panel.checked) == bp.ALTERNATIVES          # the newest three by default
    panel.set_checked([str(dirs[0]), str(dirs[3])])
    assert panel.checked == {str(dirs[0]), str(dirs[3])}
    assert sum(panel.table.item(i, 0).checkState() == 2 for i in range(4)) == 2
    scanned = []

    def fake_event_time(run_dir, progress=None, **kw):
        scanned.append(run_dir)
        us = 2.0 if run_dir == str(dirs[3]) else 0.5
        r = dict(navigation_us=us, min_us=us, max_us=us, placement=dict(gun="stack"), rounds_counted=3)
        (Path(run_dir) / et.OUT_DIR).mkdir(exist_ok=True)
        (Path(run_dir) / et.OUT_DIR / et.RESULT_JSON).write_text(_json.dumps(r))
        return r
    monkeypatch.setattr(et, "event_time", fake_event_time)
    statuses = []
    panel.status.connect(statuses.append)
    panel.event_time(sync=True)
    assert sorted(scanned) == sorted([str(dirs[0]), str(dirs[3])]) and any("us per event" in s for s in statuses)
    by_dir = {r["dir"]: r for r in panel.rows}
    assert by_dir[str(dirs[3])]["event_us"] == 2.0 and by_dir[str(dirs[1])]["event_us"] is None
    panel.preference.setCurrentText("event time")                    # within 0.02 of the best IoU: the cheaper first
    assert panel.rows[0]["dir"] == str(dirs[3]) or panel.rows[0]["event_us"] == min(
        r["event_us"] for r in panel.rows if r.get("event_us") is not None and r["exact_iou"] >= 0.96)
    panel.event_time(sync=True)                                      # nothing left to scan
    assert len(scanned) == 2 and "every checked run has one" in statuses[-1]
    # a stored nephogram: emitted, its histograms shown
    from reco_gui.nephogram import write_histogram
    nd = dirs[3] / "nephogram"
    nd.mkdir()
    info = dict(dir=str(nd), name="t", scale=1.0, files=dict(macro=str(nd / "nephogram.FCMacro")),
                stats={k: dict(n=10, mean=0.1, p95=0.3, max=0.5, band_share=[0.5, 0.5] + [0.0] * 6)
                       for k in ("input_to_model", "model_to_input")})
    write_histogram(info, nd / "histogram.png")
    (nd / "stats.json").write_text(_json.dumps(info))
    panel.refresh()
    row = [i for i, r in enumerate(panel.rows) if r["dir"] == str(dirs[3])][0]
    assert panel.rows[row]["p95_invented"] == 0.3 and panel.rows[row]["histogram"]
    got = []
    panel.nephogramReady.connect(got.append)
    panel.table.setCurrentCell(row, 1)
    panel.nephogram()
    assert got and got[0]["dir"] == str(nd) and not panel.histogram.pixmap().isNull()
