"""
The supervision window driven headlessly (offscreen Qt, 3D view off) with a scripted stub session: start, single
step, continue to a breakpoint, look at an earlier step while the run goes on, run to the end, and an engine error.
It checks the wiring between the worker thread, the timeline, the detail panel and the buttons - not the pixels.
The stub walks through cascade stage names with a minimal fold (the real folds are tested in test_reco_gui_cascade /
test_reco_gui_field); the voxel mode's scripted session was removed with route C on 2026-09-30.
"""

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"

pytest.importorskip("PyQt5")
SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

from PyQt5 import QtWidgets  # noqa: E402

from reco_gui.app import MainWindow  # noqa: E402
from reco_gui.session import StepSession  # noqa: E402
from reco_gui.setup import RunSettings  # noqa: E402

SCRIPT = ("input", "member", "add", "residual", "component", "candidate", "accepted", "grade", "done")


def step(stage, message=None):
    return SimpleNamespace(phase="cascade", stage=stage, message=message or f"the {stage} step", iteration=None)


def scripted():
    for stage in SCRIPT:
        yield step(stage)


def fold(prev, s, index):
    return SimpleNamespace(index=index, step=s, finished=s.stage == "done", iteration=None)


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_for(app, condition, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def settings(tmp_path):
    stl = tmp_path / "part.stl"
    stl.write_text("solid x\nendsolid x\n")
    return RunSettings(input_stl=stl)


def make_window(steps_factory):
    def fake_make(s, models=None):
        return StepSession(steps_factory(), fold), None, None
    w = MainWindow(make_session=fake_make)
    w.settings_panel.c_alternatives.setValue(0)        # one run: no further paths after it (alternatives, 2026-10-29)
    return w


def test_step_continue_browse_and_finish(qapp, tmp_path):
    w = make_window(scripted)
    w.pace.setValue(0.0)
    w.requestStart.emit(settings(tmp_path))
    assert wait_for(qapp, lambda: w.session is not None and not w._busy)
    assert w.act_step.isEnabled() and w.act_cont.isEnabled() and not w.act_pause.isEnabled()

    w.step()
    assert wait_for(qapp, lambda: w.timeline.rowCount() == 1 and not w._busy)
    assert "input" in w.detail.toPlainText()

    w.cont()                                                  # the cascade's default breakpoints: member, add, ...
    assert wait_for(qapp, lambda: not w._busy and w.timeline.rowCount() > 1)
    assert w.session.last.step.stage == "member" and w.timeline.rowCount() == 2
    assert "Stopped at a breakpoint" in w.status_label.text()

    w.timeline.selectRow(0)                                   # look at the 'input' step again
    qapp.processEvents()
    assert "the input step" in w.detail.toPlainText()

    for breakpoint_action in w.break_actions.values():
        breakpoint_action.setChecked(False)
    w.cont()
    assert wait_for(qapp, lambda: w.session.finished and not w._busy)
    assert w.timeline.rowCount() == len(SCRIPT) == len(w.session.states)
    assert "finished" in w.status_label.text().lower()
    assert not w.act_step.isEnabled() and not w.act_cont.isEnabled() and w.act_start.isEnabled()
    w.close()


def test_restart_gives_fresh_session(qapp, tmp_path):
    w = make_window(scripted)
    w.requestStart.emit(settings(tmp_path))
    assert wait_for(qapp, lambda: w.session is not None)
    w.step()
    assert wait_for(qapp, lambda: w.timeline.rowCount() == 1 and not w._busy)
    first = w.session
    w.requestStart.emit(settings(tmp_path))
    assert wait_for(qapp, lambda: w.session is not first and w.session is not None and not w._busy)
    w.timeline.setRowCount(0)
    w.step()
    assert wait_for(qapp, lambda: w.timeline.rowCount() == 1)
    w.close()


def test_engine_error_is_reported_not_swallowed(qapp, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: shown.append(a[2]))

    def boom():
        yield step("input")
        raise RuntimeError("freecad exploded")

    w = make_window(boom)
    w.requestStart.emit(settings(tmp_path))
    assert wait_for(qapp, lambda: w.session is not None)
    w.step()
    assert wait_for(qapp, lambda: w.timeline.rowCount() == 1 and not w._busy)
    w.step()
    assert wait_for(qapp, lambda: bool(shown))
    assert "freecad exploded" in shown[0] and "Failed" in w.status_label.text()
    assert "freecad exploded" in w.log.toPlainText()
    w.close()


def test_a_second_click_while_busy_is_ignored(qapp, tmp_path):
    w = make_window(scripted)
    w.requestStart.emit(settings(tmp_path))
    assert wait_for(qapp, lambda: w.session is not None and not w._busy)
    w.step()
    w.step()                                                  # the worker has not answered yet: must not queue twice
    assert wait_for(qapp, lambda: w.timeline.rowCount() >= 1 and not w._busy)
    qapp.processEvents()
    assert w.timeline.rowCount() == 1
    w.close()


def test_only_the_tabs_of_the_chosen_method_are_shown(qapp):
    """Each method fills its own views; the others would stay empty, so they are hidden (and come back on a switch)."""
    w = make_window(lambda: iter(()))
    combo = w.settings_panel.method

    def visible():
        return {w.tabs.tabText(i) for i in range(w.tabs.count()) if w.tabs.isTabVisible(i)}
    for method, expected in (("cascade", {"Cascade"}), ("field", {"Vector field"}), ("cascade", {"Cascade"})):
        combo.setCurrentIndex(combo.findData(method))
        qapp.processEvents()
        assert visible() == expected, method
        assert w.tabs.tabText(w.tabs.currentIndex()) in expected
    assert combo.findData("voxel") == -1
    w.close()
