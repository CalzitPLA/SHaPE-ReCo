"""The vector-field mode of the supervision GUI: the peeling loop as steps (reco_field.peel.peel_steps), their folding into
FieldStates (reco_gui.field_session), the detail report, the settings, and the window driven headlessly with a real part
(offscreen Qt, 3D view off)."""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.peel import peel, peel_steps  # noqa: E402
from reco_gui.field_session import FIELD_STAGES, make_field_session  # noqa: E402
from reco_gui.report import render_state, timeline_row  # noqa: E402
from reco_gui.setup import RunSettings, make_session  # noqa: E402
from src.closest_point_field import load_field  # noqa: E402

PART = ROOT / "dataset_gen" / "examples" / "blind_hole.stl"        # a box with one cylindrical cut: base + one accepted operation
SMALL = dict(n_surface=1500, n_near=800)


@pytest.fixture(scope="module")
def steps():
    return list(peel_steps(load_field(PART), n_vectors=50, grade=True, **SMALL))


def test_steps_come_in_the_documented_order(steps):
    stages = [s.stage for s in steps]
    assert stages[:2] == ["field", "base"] and stages[-2:] == ["grade", "done"]
    assert set(stages) <= set(FIELD_STAGES)
    assert "cluster" in stages and "candidate" in stages and "accepted" in stages
    assert stages.index("cluster") < stages.index("candidate") < stages.index("accepted")
    assert steps[0].vectors.shape == (50, 3) and steps[0].field is not None


def test_peel_is_the_last_step_of_peel_steps(steps):
    result = steps[-1].result
    plain = peel(load_field(PART), **SMALL)
    assert [(o.op, o.fit.kind) for o in result.ops] == [(o.op, o.fit.kind) for o in plain.ops]
    assert result.explained == pytest.approx(plain.explained) and result.stopped in ("target reached", "max operations", "nothing helps")


def test_vectors_point_to_the_surface(steps):
    s0 = steps[0]
    closest = s0.field.query(s0.vector_points).closest
    assert np.allclose(s0.vector_points + s0.vectors, closest)


def test_session_folds_steps_into_states():
    s = make_field_session(PART, n_vectors=30, grade=False, **SMALL)
    while s.advance() is not None:
        pass
    assert s.finished and s.error is None
    first, last = s.states[0], s.states[-1]
    assert first.input_mesh is not None and first.vectors is not None and first.model_mesh is None
    assert last.finished and last.model_mesh is not None and len(last.ops) >= 2 and last.explained > 0.97
    assert last.input_mesh is first.input_mesh and last.grade is None                         # carried along, not rebuilt
    by_stage = {st.step.stage: st for st in s.states}
    assert by_stage["cluster"].cluster is not None and by_stage["cluster"].candidate is None
    cand = by_stage["candidate"]
    assert cand.candidate is not None and cand.candidate_mesh is not None and cand.verdict is None
    assert by_stage["accepted"].verdict == "accepted" and by_stage["accepted"].candidate_mesh is None
    assert len(by_stage["accepted"].ops) == len(by_stage["candidate"].ops) + 1
    meshes = by_stage["accepted"].model_mesh
    assert meshes.vertices.shape[1] == 3 and meshes.triangles.max() < len(meshes.vertices)


def test_report_and_timeline_rows():
    s = make_field_session(PART, n_vectors=10, grade=True, **SMALL)
    while s.advance() is not None:
        pass
    for st in s.states:
        assert len(timeline_row(st)) == 5
        text = render_state(st)
        assert st.step.stage in text
    assert "Chamfer" in render_state(s.states[-1]) and "Result" in render_state(s.states[-1])
    assert "cut:cylinder" in render_state(s.states[-1]) and "Program" in render_state(s.states[-1])
    cand = next(st for st in s.states if st.step.stage == "candidate")
    assert "Candidate" in render_state(cand) and "Cluster" in render_state(cand)


def test_make_session_dispatches_on_the_method():
    session, models, workdir = make_session(RunSettings(input_stl=PART, method="field", field_surface_points=1500,
                                                        field_near_points=800, field_grade=False))
    assert models is None and workdir is None and session.states == []
    with pytest.raises(ValueError):
        make_session(RunSettings(input_stl=PART, method="nonsense"))


# ------------------------------------------------------------------------------------------------ the window

pytest.importorskip("PyQt5")


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_for(app, condition, timeout=120.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_window_runs_the_vector_field_loop(qapp):
    from reco_gui.app import MainWindow
    w = MainWindow(stl=str(PART))
    panel = w.settings_panel
    assert panel.method.currentData() == "cascade"                  # the main line is the default ...
    panel.method.setCurrentIndex(panel.method.findData("field"))    # ... the vector field is chosen here
    assert panel.method.currentData() == "field"
    assert set(w.break_actions) == {"base", "cluster", "candidate", "accepted", "rejected", "grade"}
    assert [s for s, a in w.break_actions.items() if a.isChecked()] == ["base", "accepted", "rejected"]
    panel.f_surface.setValue(1500)
    panel.f_near.setValue(800)
    panel.f_grade.setChecked(False)
    w.pace.setValue(0.0)
    w.start()
    assert wait_for(qapp, lambda: w.session is not None and not w._busy)
    assert w.settings.method == "field" and w.act_step.isEnabled()

    w.step()
    assert wait_for(qapp, lambda: w.timeline.rowCount() == 1 and not w._busy)
    assert "field" in w.detail.toPlainText() and "surface points" in w.detail.toPlainText()
    w.cont()                                                               # to the base
    assert wait_for(qapp, lambda: not w._busy and w.session.last.step.stage == "base")
    assert "Stopped at a breakpoint" in w.status_label.text() and "box" in w.detail.toPlainText()
    w.cont()                                                               # to the first accepted operation
    assert wait_for(qapp, lambda: not w._busy and w.session.last.step.stage == "accepted")
    assert "Program" in w.detail.toPlainText() and "cut:cylinder" in w.detail.toPlainText()

    w.timeline.selectRow(1)                                                # an earlier step, while the run is paused
    qapp.processEvents()
    assert "base" in w.detail.toPlainText()

    for a in w.break_actions.values():
        a.setChecked(False)
    w.cont()
    assert wait_for(qapp, lambda: w.session.finished and not w._busy)
    assert "Result" in w.detail.toPlainText() and w.timeline.rowCount() == len(w.session.states)
    assert not w.act_step.isEnabled() and w.act_start.isEnabled()
    w.close()


def test_panel_shows_only_the_settings_of_the_chosen_method(qapp):
    from reco_gui.settings_panel import SettingsPanel
    panel = SettingsPanel()
    panel.show()
    assert panel.method.currentData() == "cascade"             # the main line is the default
    panel.method.setCurrentIndex(panel.method.findData("field"))
    assert not panel.field_box.isVisible()                    # minimal by default: behind "More settings"
    panel.more.setChecked(True)
    panel._method_changed()
    assert panel.field_box.isVisible() and not panel.cascade_box.isVisible()
    panel.method.setCurrentIndex(panel.method.findData("cascade"))
    assert not panel.field_box.isVisible() and panel.cascade_box.isVisible()
    assert panel.method.findData("voxel") == -1                # route C's voxel mode is gone
    panel.close()


def test_predict_max_ops_button_fills_in_the_spinbox(qapp):
    """The 'Predict from genus' button (reco_field.peel.predict_ops_budget) gives the GUI a value tailored to the loaded
    part, without the user having to run the loop first to find out 24 was too few (or needlessly many)."""
    from reco_gui.settings_panel import SettingsPanel
    panel = SettingsPanel()
    panel.method.setCurrentIndex(panel.method.findData("field"))
    panel.set_input(str(PART))                                                       # blind_hole.stl: genus 0, one cut
    before = panel.f_max_ops.value()
    panel._predict_max_ops()
    assert panel.f_max_ops.value() != before and "ops" in panel.f_predict_label.text()
    assert "genus 0" in panel.f_predict_label.text()

    panel.set_input("/no/such/file.stl")
    panel._predict_max_ops()
    assert "not found" in panel.f_predict_label.text()
    panel.close()


def test_autorun_reconstructs_live_without_any_click(qapp):
    from reco_gui.app import MainWindow
    w = MainWindow(stl=str(PART), autorun=True)
    w.settings_panel.f_surface.setValue(1500)
    w.settings_panel.f_near.setValue(800)
    w.settings_panel.f_grade.setChecked(False)
    w.pace.setValue(0.0)
    assert wait_for(qapp, lambda: w.session is not None and w.session.finished and not w._busy)
    assert w.timeline.rowCount() == len(w.session.states) and "Result" in w.detail.toPlainText()
    w.close()
