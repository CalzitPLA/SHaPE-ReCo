"""The cascade (add-cut) mode of the supervision GUI with a FAKE residual loop (a scripted sequence of ResidualSteps on a real
part): folding into CascadeStates (reco_gui.cascade_session), the detail report and timeline rows, the settings, and the
window driven headlessly (offscreen Qt, 3D view off) - step, continue to breakpoints, look back, run to the end."""
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine.result_schema import Operation, Primitive, ReconstructionResult, StageSource  # noqa: E402
from reco_gui.cascade_session import (CASCADE_STAGES, CascadeFolder, make_cascade_session, primitive_sdf,  # noqa: E402
                                      voxel_mesh)
from reco_gui.report import render_state, timeline_row  # noqa: E402
from reco_gui.setup import RunSettings, make_session  # noqa: E402

PART = ROOT / "dataset_gen" / "examples" / "bs_box_sphere_pocket.stl"      # a 20 mm box with a spherical pocket
SPEC = {"lo": (-10.0, -10.0, -10.0), "hi": (10.0, 10.0, 10.0), "resolution": 16}


@dataclass
class FakeStep:
    kind: str
    result: ReconstructionResult
    info: dict


def src(mode, conf=1.0):
    return [StageSource(stage, mode, conf) for stage in ("type", "rotation", "dimensions", "position")]


# the type stage fused two sources (reco_engine.cascade: StageSource.detail records every vote and the winner's subclass)
VOTES = [dict(id="type_descriptors_v5", mode="descriptor:type_descriptors_v5", weight=0.78, seconds=0.2,
              probs={"box": 0.71, "cylinder": 0.2, "sphere": 0.09}, subclass="full"),
         dict(id="type_v3", mode="cnn:type_v3", weight=0.5, seconds=0.05, probs={"box": 0.97, "sphere": 0.03}, subclass=None)]
TYPE_STAGE = dict(parent="box", subclass="full", subclass_source="descriptor:type_descriptors_v5", mode="fused",
                  probs={"box": 0.81, "cylinder": 0.12, "sphere": 0.07}, entries=VOTES)
BOX = Primitive("box", center_mm=(0, 0, 0), dims_mm={"x_half": 10, "y_half": 10, "z_half": 10}, confidence=0.97,
                sources=[StageSource("type", "fused", 0.81, 0.25, detail=dict(entries=VOTES, subclass="full",
                                                                              subclass_source="descriptor:type_descriptors_v5"))]
                + src("analytic:frame")[1:],
                alternatives=[Primitive("cylinder", dims_mm={"radius": 10, "half_length": 10}, confidence=0.02)])
SPHERE = Primitive("sphere", center_mm=(0, 0, 10), dims_mm={"radius": 6}, confidence=0.9, sources=src("analytic:sphere_fit"))
WRONG = Primitive("cylinder", quaternion=(0.0, 0.7071068, 0.0, 0.7071068), center_mm=(0, 0, 8),
                  dims_mm={"radius": 3, "half_length": 4}, confidence=0.4, sources=src("cnn:type_v3", 0.4))
CUBES = Primitive("box", center_mm=(8, 8, 8), dims_mm={"x_half": 1, "y_half": 1, "z_half": 1}, confidence=0.0,
                  sources=[StageSource("type", "voxel_fallback", 0.0)])


def mask(sl):
    m = np.zeros((16, 16, 16), bool)
    m[sl] = True
    return m


def fake_steps(path, config):
    """What reco_engine.residual_loop.residual_steps yields (its info keys: spec on the input only, primitive, mask, pos/neg,
    gain), scripted: add the box, cut the pocket, reject a wrong candidate, accept a voxel-cube fallback, grade, done."""
    assert config["top_k"] >= 1
    ops = []

    def res(pos, neg, chamfer=None):
        return ReconstructionResult(input_path=str(path), operations=list(ops), residual_voxels_pos=pos,
                                    residual_voxels_neg=neg, chamfer_mm=chamfer)
    yield FakeStep("input", res(4096, 0), {"spec": SPEC, "type_stage": TYPE_STAGE, "hint_subclass": "box/full"})
    ops.append(Operation("add", BOX, component_voxels=4096, net_gain=4000))
    yield FakeStep("add", res(0, 120), {"candidate": BOX, "op": "add", "net_gain": 4000.0})
    pocket = mask((slice(6, 10), slice(6, 10), slice(12, 16)))
    yield FakeStep("residual", res(0, 120), {"pos": np.zeros((16, 16, 16), bool), "neg": pocket})
    yield FakeStep("component", res(0, 120), {"mask": pocket, "op": "cut"})
    yield FakeStep("candidate", res(0, 120), {"primitive": SPHERE, "op": "cut", "gain": 110})
    ops.append(Operation("cut", SPHERE, component_voxels=120, net_gain=110, freecad_valid=True))
    yield FakeStep("accepted", res(0, 10), {"candidate": SPHERE, "op": "cut", "net_gain": 110.0})
    corner = mask((slice(14, 16), slice(14, 16), slice(14, 16)))
    yield FakeStep("residual", res(8, 2), {"pos": corner, "neg": np.zeros((16,) * 3, bool)})
    yield FakeStep("component", res(8, 2), {"mask": corner, "op": "add"})
    yield FakeStep("candidate", res(8, 2), {"primitive": WRONG, "op": "add", "gain": -30})
    ops.append(Operation("add", WRONG, net_gain=-30, accepted=False))
    yield FakeStep("rejected", res(8, 2), {"op": "add", "message": "add: no candidate passes"})
    ops.append(Operation("add", CUBES, net_gain=8))
    yield FakeStep("fallback", res(0, 2), {"candidate": CUBES, "op": "add", "net_gain": 8.0})
    yield FakeStep("grade", res(0, 2, chamfer=0.031), {})
    yield FakeStep("done", res(0, 2, chamfer=0.031), {})


def run_all(session):
    while session.advance() is not None:
        pass
    return session


@pytest.fixture(scope="module")
def session():
    return run_all(make_cascade_session(PART, steps_fn=fake_steps, voxel_meshes=True))


def test_primitive_sdf_matches_the_schema_conventions():
    box = primitive_sdf(BOX)
    assert box.sdf(np.array([[0, 0, 0.0]]))[0] == pytest.approx(-10) and box.sdf(np.array([[12, 0, 0.0]]))[0] == pytest.approx(2)
    cyl = primitive_sdf(WRONG)                              # axis along world x (90 deg about y), centred at its centre
    assert cyl.sdf(np.array([[3.9, 0, 8.0]]))[0] < 0 < cyl.sdf(np.array([[0, 0, 11.5]]))[0]
    assert cyl.sdf(np.array([[-3.9, 0, 8.0]]))[0] < 0
    assert primitive_sdf(Primitive("unknown")) is None


def test_voxel_mesh_draws_the_exposed_faces():
    m = voxel_mesh(mask((slice(0, 2), slice(0, 1), slice(0, 1))), np.zeros(3), np.full(3, 16.0))     # two cubes of 1 mm
    assert len(m.triangles) == 2 * 10 and m.vertices.min() == 0 and m.vertices.max() == 2
    assert voxel_mesh(np.zeros((4, 4, 4), bool), np.zeros(3), np.ones(3)) is None


def test_states_fold_the_steps(session):
    st = session.states
    assert session.finished and session.error is None
    assert [s.step.stage for s in st] == [s.kind for s in fake_steps(PART, {"top_k": 2})]
    assert set(s.step.stage for s in st) <= set(CASCADE_STAGES)
    first = st[0]
    assert first.input_mesh is not None and first.model_mesh is None and first.program == ()
    by = {}
    for s in st:
        by.setdefault(s.step.stage, []).append(s)
    add = by["add"][0]
    assert add.model_mesh is not None and len(add.program) == 1 and add.candidate_mesh is None and add.verdict == "accepted"
    res = by["residual"][0]
    assert res.residual_neg_mesh is not None and res.residual_pos_mesh is None and res.residual_neg == 120
    comp = by["component"][0]
    assert comp.component_voxels == 64 and comp.component_mesh is not None and comp.candidate is None
    cand = by["candidate"][0]
    assert cand.candidate is SPHERE and cand.candidate_op == "cut" and cand.candidate_mesh is not None and cand.verdict is None
    acc = by["accepted"][0]
    assert acc.verdict == "accepted" and len(acc.program) == 2 and acc.model_mesh is not add.model_mesh
    rej = by["rejected"][0]
    assert rej.verdict == "rejected" and len(rej.program) == 2 and rej.candidate is WRONG and rej.net_gain == -30
    fb = by["fallback"][0]
    assert fb.candidate_fallback and fb.fallback_mesh is not None and fb.verdict == "accepted" and len(fb.program) == 3
    last = st[-1]
    assert last.finished and last.input_mesh is first.input_mesh and last.result.chamfer_mm == pytest.approx(0.031)
    assert last.candidate is None and last.component_mesh is None


def test_the_model_mesh_is_cached():
    folder = CascadeFolder(voxel_meshes=True)
    states, prev = [], None
    for i, step in enumerate(fake_steps(PART, {"top_k": 1})):
        prev = folder(prev, step, i)
        states.append(prev)
    same = [s for s in states if len(s.program) == 2 and s.step.stage in ("residual", "component", "candidate", "rejected")]
    assert len(same) >= 3 and all(s.model_mesh is same[0].model_mesh for s in same)


def test_report_and_timeline_rows(session):
    for st in session.states:
        row = timeline_row(st)
        assert len(row) == 5 and row[2] == st.step.stage
        assert st.step.stage in render_state(st)
    assert timeline_row(session.states[0])[3] == ""                              # no iteration on the input
    last = render_state(session.states[-1])
    assert "Program" in last and "add:box" in last and "cut:sphere" in last and "Result" in last and "Chamfer" in last
    assert "cnn:type_v3" in last and "analytic:sphere_fit" in last and "voxel_fallback" in last       # per-stage provenance
    assert "fused" in last and "descriptor:type_descriptors_v5" in last and "w=0.78" in last and "w=0.50" in last
    assert "subclass <b>full</b>" in last
    assert "alternatives" in last and "cylinder" in last and "valid" in last
    rej = render_state(next(s for s in session.states if s.step.stage == "rejected"))
    assert "Candidate" in rej and "rejected" in rej and "net gain" in rej and "Provenance" in rej
    fb = render_state(next(s for s in session.states if s.step.stage == "fallback"))
    assert "voxel-cube fallback" in fb


def test_make_session_dispatches_the_cascade_method(monkeypatch):
    import reco_gui.cascade_session as cs
    seen = {}

    def fake_make(path, **kw):
        seen.update(kw, path=path)
        return "session"
    monkeypatch.setattr(cs, "make_cascade_session", fake_make)
    out = make_session(RunSettings(input_stl=PART, method="cascade", cascade_top_k=3, cascade_op_guard=True,
                                   cascade_max_ops=7, cascade_registry=Path("r.json")))
    assert out == ("session", None, None)
    assert seen["top_k"] == 3 and seen["op_guard"] is True and seen["max_ops"] == 7 and seen["registry_path"] == Path("r.json")
    assert seen["type_source"] == "registry"
    make_session(RunSettings(input_stl=PART, method="cascade", cascade_type_source="descriptors"))
    assert seen["type_source"] == "descriptors"


def test_type_stage_is_folded_and_reported(session):
    first = session.states[0]
    assert first.type_stage == TYPE_STAGE and "type box/full (fused)" in first.step.message
    assert session.states[-1].type_stage == TYPE_STAGE                                   # kept for the whole run
    text = render_state(first)
    assert "Type stage" in text and "<b>box</b>" in text and "<b>full</b>" in text and "sub-loop hint" in text
    assert "descriptor:type_descriptors_v5" in text and "w=0.78" in text and "cnn:type_v3" in text and "w=0.50" in text
    assert "Type stage" not in render_state(next(s for s in session.states if s.step.stage == "accepted"))


def test_type_source_filters_the_registry_in_residual_config():
    from reco_engine.cascade_node_support import residual_config
    ids = lambda cfg: [e.id for e in cfg.registry if e.stage == "type"]  # noqa: E731
    everything = ids(residual_config(registry_path=str(ROOT / "models" / "registry.json")))
    assert "type_descriptors_v5" in everything and any(i.startswith("type_v") for i in everything)
    only_desc = ids(residual_config(registry_path=str(ROOT / "models" / "registry.json"), type_source="descriptors"))
    assert only_desc == ["type_descriptors_v5"]
    only_cnn = ids(residual_config(registry_path=str(ROOT / "models" / "registry.json"), type_source="cnn"))
    assert "type_descriptors_v5" not in only_cnn and only_cnn
    assert residual_config(type_source="registry").registry is None                     # untouched: the loop's default


# ------------------------------------------------------------------------------------------------ the window

pytest.importorskip("PyQt5")


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_for(app, condition, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def fake_make_session(settings, models=None):
    assert settings.method == "cascade"
    return make_cascade_session(settings.input_stl, top_k=settings.cascade_top_k, steps_fn=fake_steps), None, None


def test_panel_settings_for_the_cascade(qapp):
    from reco_gui.settings_panel import SettingsPanel
    panel = SettingsPanel()
    panel.show()
    assert panel.method.currentData() == "cascade"                                    # the main line is the default
    panel.method.setCurrentIndex(panel.method.findData("cascade"))
    assert not panel.cascade_box.isVisible() and not panel.advanced_box.isVisible()     # minimal: "More settings" is closed
    panel.more.setChecked(True)
    assert panel.cascade_box.isVisible() and not panel.field_box.isVisible()
    assert not panel.c_op_guard.isChecked() and "bypassed" in panel.c_op_guard.text()
    assert panel.c_registry.text().endswith(str(Path("models") / "registry.json"))
    assert panel.c_type_source.currentData() == "registry" and panel.c_type_source.count() == 3
    panel.set_input(str(PART))
    panel.c_top_k.setValue(3)
    s = panel.collect()
    assert s.method == "cascade" and s.cascade_top_k == 3 and s.cascade_max_ops is None and not s.cascade_op_guard
    panel.c_op_guard.setChecked(True)
    panel.c_max_ops.setValue(5)
    panel.c_freecad.setChecked(True)
    s = panel.collect()
    assert s.cascade_op_guard and "bypassed" not in panel.c_op_guard.text() and s.cascade_max_ops == 5 and s.cascade_freecad_per_step
    assert s.cascade_type_source == "registry"
    panel.c_type_source.setCurrentIndex(panel.c_type_source.findData("descriptors"))
    assert panel.collect().cascade_type_source == "descriptors"
    panel.c_type_source.setCurrentIndex(panel.c_type_source.findData("cnn"))
    assert panel.collect().cascade_type_source == "cnn"
    panel.method.setCurrentIndex(panel.method.findData("field"))
    assert not panel.cascade_box.isVisible() and panel.field_box.isVisible()
    panel.close()


def test_window_steps_through_the_cascade(qapp):
    from reco_gui.app import MainWindow
    w = MainWindow(make_session=fake_make_session, stl=str(PART))
    panel = w.settings_panel
    panel.method.setCurrentIndex(panel.method.findData("cascade"))
    assert set(w.break_actions) == set(CASCADE_STAGES) - {"input", "done"}
    assert [s for s, a in w.break_actions.items() if a.isChecked()] == []      # none at start (the user, 2026-10-15)
    for stage in ("add", "accepted", "rejected"):                               # this test stops at those stages
        w.break_actions[stage].setChecked(True)
    assert w.tabs.currentWidget() is w.view_cascade
    w.pace.setValue(0.0)
    w.start()
    assert wait_for(qapp, lambda: w.session is not None and not w._busy)
    assert w.settings.method == "cascade" and w.act_step.isEnabled()

    w.step()
    assert wait_for(qapp, lambda: w.timeline.rowCount() == 1 and not w._busy)
    assert "input" in w.detail.toPlainText() and "missing material" in w.detail.toPlainText()
    assert "Type stage" in w.detail.toPlainText() and "descriptor:type_descriptors_v5" in w.detail.toPlainText()
    w.cont()                                                              # to the first addition
    assert wait_for(qapp, lambda: not w._busy and w.session.last.step.stage == "add")
    assert "Stopped at a breakpoint" in w.status_label.text() and "add:box" in w.detail.toPlainText()
    w.cont()                                                              # to the accepted cut
    assert wait_for(qapp, lambda: not w._busy and w.session.last.step.stage == "accepted")
    assert "cut:sphere" in w.detail.toPlainText() and "analytic:sphere_fit" in w.detail.toPlainText()

    w.timeline.selectRow(0)                                               # look back while paused
    qapp.processEvents()
    assert "Program" not in w.detail.toPlainText()

    for a in w.break_actions.values():
        a.setChecked(False)
    w.cont()
    assert wait_for(qapp, lambda: w.session.finished and not w._busy)
    assert "Result" in w.detail.toPlainText() and w.timeline.rowCount() == len(w.session.states)
    assert not w.act_step.isEnabled() and w.act_start.isEnabled()
    w.close()


def test_a_new_run_is_followed_live_again(qapp):
    """Open (and a click on an earlier guess) switch "follow live" off; a new run switches it on again, else its steps
    fill the timeline while the 3D view and the detail panel stay (the user, 2026-10-16: "the live preview is gone")."""
    from reco_gui.app import MainWindow
    w = MainWindow(make_session=fake_make_session, stl=str(PART))
    panel = w.settings_panel
    panel.method.setCurrentIndex(panel.method.findData("cascade"))
    w.follow_box.setChecked(False)                                        # what open_run does
    assert not w._follow
    w.pace.setValue(0.0)
    w.start()
    assert wait_for(qapp, lambda: w.session is not None and not w._busy)
    assert w.follow_box.isChecked() and w._follow
    w.step()
    assert wait_for(qapp, lambda: w.timeline.rowCount() == 1 and not w._busy)
    assert "input" in w.detail.toPlainText()                              # the live step is shown, not the old summary
    w.close()


def test_start_reports_a_missing_residual_loop(qapp, monkeypatch):
    """Without reco_engine/residual_loop.py the GUI starts; Start in the cascade mode fails with a message, not a crash."""
    import builtins
    from PyQt5 import QtWidgets
    from reco_gui.app import MainWindow
    real_import = builtins.__import__

    def no_loop(name, *a, **k):
        if name == "reco_engine.residual_loop":
            raise ImportError("no residual loop")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_loop)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: None)
    w = MainWindow(stl=str(PART))
    w.settings_panel.method.setCurrentIndex(w.settings_panel.method.findData("cascade"))
    w.start()
    assert wait_for(qapp, lambda: "Failed" in w.status_label.text())
    assert "no residual loop" in w.log.toPlainText() and w.session is None
    w.close()


def test_cut_history_becomes_two_layers():
    """residual_loop's cut history (info["cut_history"], int counts per voxel) is drawn as 'cut once' and 'cut again'."""
    history = np.zeros((16, 16, 16), np.int16)
    history[4:8, 4:8, 4:8] = 1
    history[6:8, 6:8, 6:8] = 2                                           # an overlap of two cutters: 8 voxels cut again
    fold = CascadeFolder(voxel_meshes=True)
    res = ReconstructionResult(input_path=str(PART), operations=[Operation("add", BOX)], recut_voxels=8)
    s = fold(None, FakeStep("input", res, {"spec": SPEC}), 0)
    assert s.cut_once_mesh is None and s.recut_mesh is None
    s = fold(s, FakeStep("residual", res, {"pos": np.zeros((16,) * 3, bool), "neg": np.zeros((16,) * 3, bool),
                                           "cut_history": history}), 1)
    assert s.recut_voxels == 8 and s.cut_once_mesh is not None and s.recut_mesh is not None
    assert s.recut_mesh.vertices.min(axis=0) == pytest.approx([-2.5, -2.5, -2.5])       # voxel 6 of 16 on [-10, 10]: -10 + 6 * 1.25
    assert "cut again" in render_state(s)


def test_csg_tree_nests_the_program_like_freecad():
    """Consecutive adds share one Fusion, a cut takes the model so far as its base; a leading cut gets an empty base."""
    from reco_gui.op_tree import csg_tree
    a, b, c, d = (Operation("add", BOX), Operation("add", CUBES), Operation("cut", SPHERE), Operation("cut", WRONG))
    assert csg_tree(()) is None
    tree = csg_tree((a, b, c, d, Operation("add", BOX)))
    assert tree[0] == "fuse" and len(tree[1]) == 2
    cut = tree[1][0]
    assert cut[0] == "cut" and [n[0] for n in cut[1]] == ["fuse", "prim", "prim"]
    assert [n[1] for n in cut[1][0][1]] == [0, 1] and [n[1] for n in cut[1][1:]] == [2, 3]
    assert csg_tree((c,))[1][0] == ("empty",)


def test_window_shows_the_operations_tree_and_the_output(qapp, tmp_path, monkeypatch):
    """The fake run's steps land in the History by component, the accepted program in the Model as Cut(box, sphere) fused
    with the fallback; a click on a model leaf shows the step that accepted it; printed lines reach the Output panel."""
    from reco_gui.app import MainWindow
    monkeypatch.setenv("SHAPE_RECO_TIMINGS_DIR", str(tmp_path))
    monkeypatch.setenv("SHAPE_RECO_WALL_CACHE", str(tmp_path / "wall"))
    w = MainWindow(make_session=fake_make_session, stl=str(PART))
    w.capture_output()
    try:
        w.settings_panel.method.setCurrentIndex(w.settings_panel.method.findData("cascade"))
        assert w.op_dock.isVisibleTo(w) and w.output_dock.isVisibleTo(w)
        w.pace.setValue(0.0)
        w.start()
        assert wait_for(qapp, lambda: w.session is not None and not w._busy)
        for a in w.break_actions.values():
            a.setChecked(False)
        w.cont()
        assert wait_for(qapp, lambda: w.session.finished and not w._busy)
        history = w.op_tree.history_root
        n_items = sum(history.child(k).childCount() or 1 for k in range(history.childCount()))
        assert n_items == len(w.session.states)
        groups = [history.child(k) for k in range(history.childCount()) if history.child(k).text(0).startswith("component")]
        assert [g.text(1)[0] for g in groups] == ["✓", "✓", "✗", "✓"]   # add, cut, rejected, the fallback (its own component)
        model = w.op_tree.model_root
        assert model.text(1) == "3 operation(s): 2 add, 1 cut"
        top = model.child(0)
        assert top.text(0) == "Fusion" and [top.child(k).text(0) for k in range(2)] == ["Cut", "3. add: box"]
        cut = top.child(0)
        assert [cut.child(k).text(0) for k in range(2)] == ["1. add: box", "2. cut: sphere"]
        index = cut.child(1).data(0, 256)
        assert w.session.states[index].step.stage == "accepted"
        w.op_tree.stateActivated.emit(index)
        assert w.timeline.selectionModel().selectedRows()[0].row() == index

        log = w.log.toPlainText()
        assert "--- run finished   [since Start:" in log and "(whole run)" in log            # CPU time per step and in total
        assert all("CPU" in line for line in log.splitlines() if line[:3].strip().isdigit())
        assert all(any(x.path == "gui fold" for x in st.timings) for st in w.session.states)   # each step's sections
        assert len(list(tmp_path.glob("*.json"))) == 1                                       # the run's timing JSON
        assert "CPU" in w.status_label.text()
        st = w.stats                                                                    # the Stats dock
        assert len(st.steps) == len(w.session.states) and st.slider.maximum() == len(w.session.states) - 1
        assert st.steps[-1]["ops"] == 3 and st.steps[-1]["neg"] == 2 and st.steps[-1]["wall"] is not None
        st._sample(); st._sample()                                                      # two samples -> one point
        assert st.samples["t"] and st.samples["cores"][0] >= 0 and st.samples["rss"][0] > 0
        st.live.setChecked(True)
        st.slider.setValue(2)                                                           # moved by hand
        assert not st.live.isChecked() and w.timeline.selectionModel().selectedRows()[0].row() == 2
        assert "3 " in st.readout.text()
        st._dirty = True
        st._redraw()                                                                    # draws without error
        print("[cascade] a line the run prints")
        assert wait_for(qapp, lambda: "[cascade] a line the run prints" in w.log.toPlainText())
    finally:
        w.close()
    assert not hasattr(sys.stdout, "line")                                       # the streams are restored on close


def test_wall_grid_voxels_sit_on_the_part_and_flag_a_coarse_loop_grid():
    """An off-centre, asymmetric thin plate (0.2 thick): tau = 0.2, the tau-pitch voxels cover the plate's own box (world
    axes, no flip) with the right volume, and a loop pitch of 1.0 is flagged while 0.1 is not."""
    import trimesh
    from reco_gui.cascade_session import grid_warning, wall_grid_voxels
    plate = trimesh.creation.box(extents=(4.0, 0.2, 2.5))
    plate.apply_translation((7.0, -3.0, 11.0))
    mesh, info = wall_grid_voxels(plate.vertices, plate.faces)
    assert abs(info["tau_p5"] - 0.2) < 1e-3 and not info["capped"]
    assert abs(info["volume_ratio"] - 1.0) < 0.1
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    assert np.allclose(lo, plate.bounds[0], atol=info["pitch"]) and np.allclose(hi, plate.bounds[1], atol=info["pitch"])
    assert "5.0 x the thin walls" in grid_warning(dict(info, loop_pitch=1.0))
    assert grid_warning(dict(info, loop_pitch=0.1)) == ""


def test_model_mesh_keeps_walls_thinner_than_the_old_grid():
    """Two 0.1-thick plates 60 apart (the frame's case: walls far below a 72^3 cell over the part) and a hole cut through
    one: the exact (manifold3d) model has both plates and the hole, by volume, and the next step reuses the cached prefix."""
    pytest.importorskip("manifold3d")
    from reco_gui.cascade_session import CascadeFolder
    a = Primitive("box", (0, 0, 0, 1), (0.0, 0.0, 0.0), {"x_half": 2.0, "y_half": 0.05, "z_half": 1.5})
    b = Primitive("box", (0, 0, 0, 1), (1.0, 60.0, -2.0), {"x_half": 2.0, "y_half": 0.05, "z_half": 1.5})
    hole = Primitive("cylinder", (0.7071068, 0, 0, 0.7071068), (0.5, 0.0, 0.0), {"radius": 0.4, "half_length": 1.0})
    fold = CascadeFolder(voxel_meshes=True)
    two = fold.model_mesh((Operation("add", a), Operation("add", b)), False)
    assert two is not None
    tri = two.vertices[two.triangles]
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    assert abs(volume - 2 * 4 * 0.1 * 3) < 1e-4
    assert two.vertices[:, 1].min() < 0 and two.vertices[:, 1].max() > 60        # both plates, 60 apart
    cut = fold.model_mesh((Operation("add", a), Operation("add", b), Operation("cut", hole)), False)
    tri = cut.vertices[cut.triangles]
    volume_cut = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    assert 0.9 * np.pi * 0.16 * 0.1 < volume - volume_cut < 1.01 * np.pi * 0.16 * 0.1   # the hole through plate a
    assert len(fold._meshers[False]._models) == 4                                  # (), a, a+b, a+b-hole


def test_heartbeat_says_what_the_computing_step_does(qapp, capfd):
    """While a step computes, the status line carries the elapsed time, the cores in use, the open timing section and
    the running child processes; verbose puts the log (and the heartbeat) on the terminal too."""
    import subprocess
    from types import SimpleNamespace
    from reco_engine.timing import Recorder
    from reco_gui.app import MainWindow, running_children
    w = MainWindow(make_session=fake_make_session, stl=str(PART), verbose=True)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert any(c.startswith("python") for c in running_children())
        rec = Recorder()
        w.session = SimpleNamespace(recorder=rec, finished=False)
        w._set_status("Computing ...")
        w._on_busy(True)
        assert w.heartbeat.isActive()
        w._beat_logged -= w.VERBOSE_EVERY_S                     # as if 10 s had passed
        with rec.section("loop"), rec.section("fit"):
            w._on_heartbeat()
        text = w.status_label.text()
        assert text.startswith("Computing ...   [0 s, now ") and "in loop/fit (0 s)" in text and "running python" in text
        assert "in loop/fit" in w.log.toPlainText().splitlines()[-1]
        assert "in loop/fit" in capfd.readouterr().out
        w._on_busy(False)
        assert not w.heartbeat.isActive()
    finally:
        child.kill()
        child.wait()
        w.session = None
        w.close()


def test_wall_grid_reuses_the_loops_wall_thickness(monkeypatch, tmp_path):
    """When the loop measured the wall thickness (grid "tau": info["wall_tau"]), the GUI's tau layer uses those samples
    and casts no rays of its own; without them it measures once itself."""
    import trimesh
    from src import local_grids
    from reco_gui.cascade_session import wall_grid_voxels
    plate = trimesh.creation.box(extents=(4.0, 0.2, 2.5))
    _, _, tau = local_grids.wall_thickness(plate.vertices, plate.faces)
    from reco_engine import cache
    monkeypatch.setenv("SHAPE_RECO_WALL_CACHE", str(tmp_path))
    monkeypatch.setattr(cache, "_WALL_MEMORY", {})
    calls = []
    real = local_grids.wall_thickness
    monkeypatch.setattr(local_grids, "wall_thickness", lambda *a, **k: calls.append(1) or real(*a, **k))
    mesh, info = wall_grid_voxels(plate.vertices, plate.faces, tau=tau)
    assert calls == [] and info["tau_reused"] and abs(info["tau_p5"] - 0.2) < 1e-3 and mesh is not None
    _, info2 = wall_grid_voxels(plate.vertices, plate.faces)
    assert calls == [1] and not info2["tau_reused"] and info2["tau_p5"] == info["tau_p5"]


def test_wall_thickness_is_measured_once_per_mesh(monkeypatch, tmp_path):
    """reco_engine.cache.wall_thickness_cached: the rays are cast the first time a mesh is seen; later calls - in this
    process or, from disk, in the next one - reuse them. Another mesh is measured; "off" disables the cache."""
    import trimesh
    from src import local_grids
    from reco_engine import cache
    monkeypatch.setenv("SHAPE_RECO_WALL_CACHE", str(tmp_path))
    monkeypatch.setattr(cache, "_WALL_MEMORY", {})
    calls = []
    real = local_grids.wall_thickness
    monkeypatch.setattr(local_grids, "wall_thickness", lambda *a, **k: calls.append(1) or real(*a, **k))
    plate = trimesh.creation.box(extents=(4.0, 0.2, 2.5))
    first = cache.wall_thickness_cached(plate.vertices, plate.faces)
    again = cache.wall_thickness_cached(plate.vertices, plate.faces)
    monkeypatch.setattr(cache, "_WALL_MEMORY", {})                        # a new process: only the disk is left
    from_disk = cache.wall_thickness_cached(plate.vertices, plate.faces)
    assert calls == [1] and np.array_equal(first, again) and np.array_equal(first, from_disk)
    assert len(list(tmp_path.glob("*.npy"))) == 1
    cache.wall_thickness_cached(plate.vertices * 2.0, plate.faces)       # another mesh
    assert calls == [1, 1]
    monkeypatch.setenv("SHAPE_RECO_WALL_CACHE", "off")
    monkeypatch.setattr(cache, "_WALL_MEMORY", {})
    cache.wall_thickness_cached(plate.vertices, plate.faces)
    assert calls == [1, 1, 1]
