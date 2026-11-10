"""The cascade 3D view's Parts panel and final-result switch (reco_gui/viewer_cascade.py), without VTK (RECO_GUI_NO_3D):
one checkable row per accepted operation, All/None/Adds/Cuts, checks kept when the program grows, 'Final result'
switched on once when the run finishes and off again when stepping back in the timeline."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"

QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
from PyQt5 import QtCore  # noqa: E402

from reco_engine.result_schema import Operation, Primitive, StageSource  # noqa: E402
from reco_gui.viewer_cascade import CascadeView3D, box_mesh, part_label  # noqa: E402


@dataclass
class FakeState:
    program: tuple = ()
    finished: bool = False
    result: object = None


def ops():
    box = Primitive("box", dims_mm=dict(x_half=10, y_half=5, z_half=2))
    web = Primitive("box", dims_mm=dict(x_half=1, y_half=5, z_half=2))
    web.label = "HEA 500 web"
    hole = Primitive("cylinder", dims_mm=dict(radius=2, half_length=6))
    cube = Primitive("box", dims_mm=dict(x_half=1, y_half=1, z_half=1), sources=[StageSource("type", "voxel_fallback", 0.0)])
    return [Operation("add", box), Operation("add", web), Operation("cut", hole), Operation("add", cube)]


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def checked(view):
    return [view.parts.item(i).checkState() == QtCore.Qt.Checked for i in range(view.parts.count())]


def test_labels_and_box_mesh():
    o = ops()
    assert part_label(0, o[0]).startswith("#1  add  box (x 10, y 5, z 2")
    assert part_label(1, o[1]) == "#2  add  HEA 500 web"
    assert part_label(3, o[3]).endswith("[fallback]")
    m = box_mesh(o[0].primitive)
    assert len(m.triangles) == 12 and m.vertices.min(axis=0).tolist() == [-10, -5, -2]


def test_parts_panel_checks(app):
    view = CascadeView3D()
    o = ops()
    view.set_state(FakeState(program=tuple(o[:2])))
    assert view.parts.count() == 2 and checked(view) == [False, False]
    view.check_parts("all")
    assert checked(view) == [True, True]
    view.set_state(FakeState(program=tuple(o)))                      # the program grew: old checks are kept
    assert view.parts.count() == 4 and checked(view) == [True, True, False, False]
    view.check_parts("cut")
    assert checked(view) == [False, False, True, False]
    view.check_parts("add")
    assert checked(view) == [True, True, False, True]
    view.check_parts("none")
    assert not any(checked(view))


def test_final_result_switches_on_once_and_off_when_stepping_back(app):
    view = CascadeView3D()
    o = tuple(ops())
    view.set_state(FakeState(program=o))
    assert not view.final_box.isChecked()
    view.set_state(FakeState(program=o, finished=True))
    assert view.final_box.isChecked()
    view.final_box.setChecked(False)                                   # the user may look at the step view again
    view.set_state(FakeState(program=o, finished=True))
    assert not view.final_box.isChecked()                              # ... and it is not forced back on
    view.set_state(FakeState(program=o[:2]))                           # an earlier step of the timeline
    view.set_state(FakeState(program=o, finished=True))
    assert view.final_box.isChecked()
    view.set_state(None)
    assert view.parts.count() == 0 and not view.final_box.isChecked()


def test_input_step_gives_voxels_and_field_arrows():
    """The loop's input step carries the voxelisation (occ_in) and a field sample (field_arrows); the GUI state keeps both
    for the 'Voxels' and 'Vector field' layers, and residual_loop.field_arrows samples only near the surface."""
    import numpy as np
    from reco_engine.residual_loop import field_arrows
    from reco_engine.result_schema import ReconstructionResult
    from reco_gui.cascade_session import CascadeFolder

    n = 8
    ax = (np.arange(n) + 0.5) - n / 2
    centres = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1)
    sdf = 2.0 - np.abs(centres).max(axis=-1)                           # a cube of half size 2, positive inside
    t_in = np.zeros((5, n, n, n), np.float32)
    t_in[0], t_in[1] = sdf > 0, sdf
    t_in[2] = 1.0
    arrows = field_arrows(t_in, centres, pitch=1.0, band=1.0, max_n=50)
    assert len(arrows["points"]) <= 50 and (np.abs(arrows["sdf"]) <= 1.0).all()
    assert arrows["vectors"].shape == arrows["points"].shape

    class Step:
        kind = "input"
        result = ReconstructionResult(input_path="")
        info = {"spec": {"lo": (-4, -4, -4), "hi": (4, 4, 4)}, "occ_in": t_in[0] > 0.5, "field_arrows": arrows}
    s = CascadeFolder()(None, Step(), 0)
    assert s.input_voxel_mesh is not None and s.field_arrows is arrows
    assert s.input_voxel_mesh.vertices.min() == -2 and s.input_voxel_mesh.vertices.max() == 2
