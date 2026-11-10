"""
The FreeCAD rebuild on demand, headless (offscreen Qt, 3D view off, the grade replaced by a stub): the run's config has
the final build off unless the setting asks for it, the button is dead without a result, and a rebuild fills a COPY of
the result, reports its numbers in the status line and frees the button again.
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

from reco_engine.result_schema import Operation, Primitive, ReconstructionResult  # noqa: E402
from reco_gui.app import MainWindow  # noqa: E402
from reco_gui.cascade_session import make_cascade_session  # noqa: E402
from reco_gui.rebuild import FreecadRebuild, grade_text, groups, groups_text, rebuild  # noqa: E402


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


def a_result():
    box = Primitive(kind="box", dims_mm=dict(x_half=1.0, y_half=2.0, z_half=3.0))
    return ReconstructionResult(input_path="/somewhere/part.step", operations=[Operation("add", box)])


def stub_grade(result, field, config, name):
    result.chamfer_mm, result.step_path, result.stl_path = 0.25, f"/tmp/{name}.step", f"/tmp/{name}.stl"
    return dict(valid=True, n_solids=1, chamfer=0.25, exact_iou=0.9812, exact_method="occt")


def test_the_run_has_no_final_build_unless_asked():
    seen = []

    def steps(path, config):
        seen.append(config)
        return iter(())
    make_cascade_session("part.stl", steps_fn=steps, freecad_final=False)
    make_cascade_session("part.stl", steps_fn=steps, freecad_final=True)
    make_cascade_session("part.stl", steps_fn=steps, freecad_final=False, freecad_per_step=True)
    assert [c["freecad"] for c in seen] == [False, True, True]


def test_rebuild_fills_a_copy():
    r = a_result()
    built, grade = rebuild(r, object(), stub_grade)
    assert r.chamfer_mm is None and r.step_path is None
    assert built.chamfer_mm == 0.25 and built.stl_path == "/tmp/part.stl"
    text = grade_text(grade, 12.3)
    assert "one valid solid" in text and "chamfer 0.25 mm" in text and "exact IoU 0.9812 (occt)" in text and "12 s" in text
    assert "failed" in grade_text(dict(valid=False, error="freecadcmd: no answer"), 3.0)


def test_button(qapp):
    window = MainWindow()
    try:
        assert not window.act_rebuild.isEnabled()
        window.rebuild_in_freecad()                                   # no session: a status line, nothing started
        assert "no cascade result" in window.status_label.text()
        window.rebuilder = FreecadRebuild(window, grade_fn=stub_grade)
        window.rebuilder.done.connect(window._on_rebuilt)
        window.session = SimpleNamespace(states=[SimpleNamespace(result=None), SimpleNamespace(result=a_result())],
                                         config=object())
        window.rebuild_in_freecad()
        assert wait_for(qapp, lambda: "FreeCAD rebuild (" in window.status_label.text())
        assert "exact IoU 0.9812" in window.status_label.text()
        assert window.act_rebuild.isEnabled() and window.act_rebuild.text() == "FreeCAD rebuild"
    finally:
        window.close()


def stub_groups(ops, out_dir, name, tools="solid"):
    return dict(fcstd=f"{out_dir}/{name}_groups.FCStd", groups=dict(adds=sum(o[0] == "add" for o in ops)),
                operations=[dict(index=i, op=o[0], label=o[1], valid=True) for i, o in enumerate(ops)], seconds=0.1, tools=tools)


def test_groups_take_the_accepted_operations():
    r = a_result()
    r.operations.append(Operation("cut", Primitive(kind="cylinder", dims_mm=dict(radius=1.0, half_length=2.0)), accepted=False))
    out = groups(r, "/tmp/somewhere", "leaves", stub_groups)
    assert [o["op"] for o in out["operations"]] == ["add"] and out["tools"] == "leaves"
    assert out["fcstd"] == "/tmp/somewhere/part_groups.FCStd"
    text = groups_text(out, 0.52)
    assert "1 tool(s)" in text and "1 adds" in text and "no final boolean" in text and "0.5 s" in text


def test_groups_button(qapp):
    window = MainWindow()
    try:
        assert not window.act_groups.isEnabled()
        window.groups_in_freecad()
        assert "no cascade result" in window.status_label.text()
        window.rebuilder = FreecadRebuild(window, groups_fn=stub_groups)
        window.rebuilder.grouped.connect(window._on_grouped)
        window.session = SimpleNamespace(states=[SimpleNamespace(result=a_result())], config=object())
        window.groups_in_freecad()
        assert wait_for(qapp, lambda: "FreeCAD groups (" in window.status_label.text())
        assert window.groups_document.endswith("part_groups.FCStd") and window.act_groups.isEnabled()
    finally:
        window.close()


def test_gdml_button(qapp, tmp_path):
    window = MainWindow()
    try:
        assert not window.act_gdml.isEnabled()
        assert window.export_gdml(tmp_path / "none.gdml") is None
        window.session = SimpleNamespace(states=[SimpleNamespace(result=a_result())], config=object())
        info = window.export_gdml(tmp_path / "part.gdml")
        assert info["solids"] == dict(box=1) and (tmp_path / "part.gdml").is_file()
        assert "GDML written" in window.status_label.text()
    finally:
        window.close()
