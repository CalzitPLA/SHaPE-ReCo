"""reco_gui/decision_bar.py: the non-blocking question (offscreen): shown by ask, its buttons emit, keep hides it, the
export's buttons appear after the export."""

import os

import pytest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_bar_is_non_modal_and_emits(qapp):
    from reco_gui.decision_bar import DecisionBar
    bar = DecisionBar()
    got = []
    bar.tryRequested.connect(lambda v: got.append(("try", v)))
    bar.keepRequested.connect(lambda: got.append(("keep", None)))
    bar.exportRequested.connect(lambda: got.append(("export", None)))
    ranked = [dict(variant="tree", label="the tree", p_good=0.5, n=60, measured=True, by_rescue=False, n_fail=0, p_rescue=None),
              dict(variant="cubes", label="cubes", p_good=0.3, n=60, measured=True, by_rescue=False, n_fail=0, p_rescue=None)]
    assert bar.isHidden()
    bar.ask("IoU 0.71, below 0.95. Try another approach?", ranked)
    assert not bar.isHidden() and not bar.isModal() and bar.b_try.text() == "Try tree" and bar.b_choose.isVisibleTo(bar)
    bar.b_try.click()
    bar.b_export.click()
    bar.b_keep.click()
    assert got == [("try", "tree"), ("export", None), ("keep", None)]
    bar.exporting()
    assert not bar.b_export.isEnabled()
    bar.export_failed("boom")
    assert bar.b_export.isEnabled() and bar.b_export.text() == "Export failed"


def test_main_window_asks_with_the_bar_not_a_dialog(qapp, monkeypatch):
    """_offer_retry shows the bar even when the window is visible; no QMessageBox is created."""
    from reco_gui.app import MainWindow
    from reco_gui import app as app_mod
    monkeypatch.setattr(app_mod.QtWidgets, "QMessageBox", None)            # any modal box would raise here
    w = MainWindow()
    from reco_gui.setup import RunSettings as Settings
    w.settings = Settings(input_stl="x.stl", method="cascade", retry_iou=0.95, retry_max=3, alternatives=0)

    class Last:
        info = dict(stats=dict(iou=0.71))

    class S:
        last = Last()
    w.session = S()
    w._offer_retry()
    assert not w.decision_bar.isHidden() and "0.710" in w.decision_bar.text.text()
    w._retry_keep()
    assert w.decision_bar.isHidden()
    w.close()


def test_compare_action_and_distance_layer_exist(qapp):
    from reco_gui.app import MainWindow
    w = MainWindow()
    assert w.act_compare.text().startswith("Compare with the input")          # the nephogram (Tools menu since 2026-10-29)
    w.compare_export()                                                    # no run yet: a status line, no thread
    assert "no input or model mesh" in w.status_label.text()
    assert "distance" in w.view_cascade._checks and not w.view_cascade._checks["distance"].isChecked()
    w.view_cascade.show_distance(None)                                    # harmless without VTK (RECO_GUI_NO_3D)
    w.close()


def test_alternatives_go_on_by_themselves_and_end_in_the_benchmark(qapp, monkeypatch):
    """alternatives = 3 (the user, 2026-10-29): no question; after each run the next untried path by measured rescue
    rate starts by itself (a path repeating a program does not count) until 3 different reconstructions exist; then
    the Benchmark tab shows the stored runs of those, checked, and the panel gets the user's own path back."""
    from reco_gui.app import MainWindow
    from reco_gui.setup import RunSettings as Settings
    from reco_engine import path_policy
    w = MainWindow()
    w.settings = Settings(input_stl="x.stl", method="cascade", retry_iou=0.95, retry_max=3, alternatives=3)
    w._tried_input = "x.stl"
    w._home_path = {k: v for k, v in path_policy._FIELDS.items()}
    started, shown = [], []
    monkeypatch.setattr(w, "_retry_start", lambda v, alternative=False: started.append(v))
    monkeypatch.setattr(w, "show_benchmark", lambda select=None: shown.append(list(select or [])))
    monkeypatch.setattr(w, "_program_fingerprint", lambda result: result)
    monkeypatch.setattr(path_policy, "variant_of", lambda s: started[-1] if started else "bare")
    monkeypatch.setattr(w.settings_panel, "current_path", lambda: Path("x.stl"))

    class Last:
        def __init__(self, prog, iou):
            self.result, self.info = prog, dict(stats=dict(iou=iou))

    def finish(prog, iou, run_dir):
        class S:
            last = Last(prog, iou)
            states = [1, 2, 3]
        w.session = S()
        w.saved_run, w._saved_states = run_dir, 3
        w._offer_retry()
        qapp.processEvents()

    finish("p1", 0.99, "/runs/a")                  # even a good first run goes on: the user compares, not the threshold
    assert len(started) == 1 and w.decision_bar.isHidden() and not shown
    finish("p1", 0.99, "/runs/b")                  # the same program again: not a new reconstruction
    assert len(started) == 2 and not shown
    finish("p2", 0.90, "/runs/c")
    finish("p3", 0.95, "/runs/d")                  # the third different one: done
    assert len(started) == 3 and shown == [["/runs/a", "/runs/c", "/runs/d"]]
    assert len(set(started)) == 3 and "bare" not in started
    w.close()


def _chain_window(monkeypatch, part="x.stl"):
    from reco_gui.app import MainWindow
    from reco_gui.setup import RunSettings as Settings
    from reco_engine import path_policy
    w = MainWindow()
    w.settings = Settings(input_stl=part, method="cascade", retry_iou=0.95, retry_max=3, alternatives=3)
    w._tried_input = part
    w._home_path = {k: v for k, v in path_policy._FIELDS.items()}
    started, shown = [], []
    monkeypatch.setattr(w, "_retry_start", lambda v, alternative=False: started.append(v))
    monkeypatch.setattr(w, "show_benchmark", lambda select=None: shown.append(list(select or [])))
    monkeypatch.setattr(w, "_program_fingerprint", lambda result: result)
    return w, started, shown


def _finish(w, qapp, prog, iou, run_dir):
    class Last:
        result, info = prog, dict(stats=dict(iou=iou))

    class S:
        last = Last()
        states = [1, 2, 3]
    w.session = S()
    w.saved_run, w._saved_states = run_dir, 3
    w._offer_retry()
    qapp.processEvents()


def test_no_alternatives_after_an_exact_result_nor_on_another_part(qapp, monkeypatch):
    """B2/B3 of the user's session of 2026-10-29: bare reached IoU 1.000 on the ABC plate and two more paths rebuilt its
    program (170 s); later the chain started 'cubes' on the part the user had opened meanwhile."""
    w, started, shown = _chain_window(monkeypatch)
    monkeypatch.setattr(w.settings_panel, "current_path", lambda: Path("x.stl"))
    _finish(w, qapp, "p1", 1.0, "/runs/a")
    assert not started and shown == [["/runs/a"]] and "(exact): no alternatives" in w.log.toPlainText()
    w.close()
    w, started, shown = _chain_window(monkeypatch)
    monkeypatch.setattr(w.settings_panel, "current_path", lambda: Path("other.step"))
    _finish(w, qapp, "p1", 0.9, "/runs/a")
    assert not started and not shown and "another part is chosen (other.step)" in w.log.toPlainText()
    w.close()


def test_an_alternative_that_reaches_a_tried_rough_model_is_stopped(qapp, monkeypatch):
    """B2: a path that acts before the feature phase only and builds a tried path's rough model is stopped at the first
    feature step and counted as the same program; a path that also steers the feature phase (sub-parts) runs on."""
    from reco_engine import path_policy
    w, started, shown = _chain_window(monkeypatch)
    monkeypatch.setattr(w.settings_panel, "current_path", lambda: Path("x.stl"))
    paused = []
    monkeypatch.setattr(w, "pause", lambda: paused.append(1))
    variant = ["bare"]
    monkeypatch.setattr(path_policy, "variant_of", lambda s: variant[0])

    class State:
        result = "rough-1"
    w._rough_pending = True
    w._rough_model_reached(State())                                     # bare: the first rough model of the part
    assert w._rough_prints == {"bare": "rough-1"} and not paused
    w._tried, w._tried_programs = [("bare", 0.9)], [("bare", 0.9, "p1")]
    for v, stopped in (("sub-parts", False), ("cubes", True)):
        variant[0] = v
        w._alternative, w._rough_pending, w._same_as = True, True, None
        w._rough_model_reached(State())
        assert bool(paused) == stopped and (w._same_as == (v, "bare")) == stopped
    w._t_run, w._cpu_run = 0.0, 0.0
    w._on_idle("paused")
    qapp.processEvents()                                                # _next_alternative, then the next path's start
    qapp.processEvents()
    assert w._tried_programs[-1] == ("cubes", 0.9, "p1") and started            # counted as bare's program; the next path
    assert "the same as path bare's" in w.log.toPlainText()
    w.close()


def test_after_the_alternatives_the_result_bar_leads_into_the_compare_mode(qapp, monkeypatch):
    """The user, 2026-10-29: 'when the part is reconstructed enable a method to get into the compare mode' - the bar
    above the view names the result; Compare opens step 4: the Benchmark tab with the runs and the nephogram of the
    best one (computed from the stored run in a thread)."""
    w, started, shown = _chain_window(monkeypatch)
    monkeypatch.setattr(w.settings_panel, "current_path", lambda: Path("x.stl"))
    monkeypatch.setattr(path_policy_mod(), "variant_of", lambda s: started[-1] if started else "bare")
    _finish(w, qapp, "p1", 0.90, "/runs/a")
    _finish(w, qapp, "p2", 0.95, "/runs/b")
    _finish(w, qapp, "p3", 0.97, "/runs/c")
    assert w.result_bar.isVisibleTo(w) and w.result_bar.run_dirs == ["/runs/a", "/runs/b", "/runs/c"]
    assert "0.9700" in w.result_bar.label.text() and "3 different reconstructions" in w.result_bar.label.text()
    calls = []
    monkeypatch.setattr(w, "enter_compare_mode", lambda run_dir=None, select=None: calls.append((run_dir, select)))
    w.result_bar.compare.click()
    assert calls == [("/runs/a", ["/runs/a", "/runs/b", "/runs/c"])]
    w.close()


def path_policy_mod():
    from reco_engine import path_policy
    return path_policy


def test_the_compare_mode_colours_a_stored_run(qapp, monkeypatch, tmp_path):
    from reco_gui.app import MainWindow
    from reco_gui import nephogram
    import numpy as np
    w = MainWindow()
    shown, ready = [], []
    monkeypatch.setattr(w, "show_benchmark", lambda select=None: shown.append(select))
    monkeypatch.setattr(w, "_compare_mode_ready", lambda d, info, mesh, err: ready.append((d, info, err)))
    monkeypatch.setattr(nephogram, "stored_nephogram", lambda run_dir: dict(stats={}, files={}, run=run_dir))

    import json
    import trimesh
    trimesh.creation.box((2, 2, 2)).export(tmp_path / "x.stl")
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "run.json").write_text(json.dumps(dict(name="run", input=dict(path=str(tmp_path / "x.stl")))))
    w.result_bar.show_result("done", [str(tmp_path / "run")])
    w.enter_compare_mode(str(tmp_path / "run"), select=[str(tmp_path / "run")])
    assert w.flow.current == 3 and shown == [[str(tmp_path / "run")]] and not w.result_bar.isVisibleTo(w)
    w._compare_thread.wait(20000)
    qapp.processEvents()
    assert ready and ready[0][0] == str(tmp_path / "run") and ready[0][2] == "" and ready[0][1]["run"] == str(tmp_path / "run")
    w.close()
