"""The starting part (the user, 2026-10-29: "define a starting part by a basic shape type and the bounding part or
sub-part, or by exact location and size, via a dialog"): reco_engine.user_guess.bounding_shape / exact_shape on
off-centre, tilted shapes, and the dialog (offscreen) that gives the loop its first_guess entry."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine import user_guess as ug  # noqa: E402

T = np.array([40.0, -25.0, 12.0])                       # off-centre on every axis


def _tilt():
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", [20.0, -35.0, 50.0], degrees=True).as_matrix()


def test_the_oriented_box_and_the_oblique_cylinder_around_the_part():
    box = trimesh.creation.box((30.0, 12.0, 6.0))
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = _tilt(), T
    box.apply_transform(M)
    s = ug.bounding_shape("box", box.vertices, box.faces)
    assert sorted(2 * v for v in s.dims_mm.values()) == pytest.approx([6.0, 12.0, 30.0], abs=1e-6)
    assert np.allclose(s.center_mm, T, atol=1e-6) and ug.held_share(s, box.vertices) == 1.0 and s.op == "add"
    aligned = ug.bounding_shape("box", box.vertices, box.faces, axis="z")      # world-aligned: bigger, still holds it
    assert np.prod(list(aligned.dims_mm.values())) > np.prod(list(s.dims_mm.values()))
    cyl = trimesh.creation.cylinder(radius=4.0, height=25.0, sections=64)
    cyl.apply_transform(M)
    c = ug.bounding_shape("cylinder", cyl.vertices, cyl.faces)
    assert c.dims_mm["radius"] == pytest.approx(4.0, rel=2e-3) and 2 * c.dims_mm["half_length"] == pytest.approx(25.0, rel=1e-6)
    assert ug.held_share(c, cyl.vertices) == 1.0
    for kind in ("cone", "sphere", "torus"):                                    # every kind gives a buildable shape
        k = ug.bounding_shape(kind, cyl.vertices, cyl.faces)
        assert k.kind == kind and k.primitive().defect() is None


def test_a_sub_part_gets_its_own_box():
    pytest.importorskip("manifold3d")
    a = trimesh.creation.box((40.0, 8.0, 8.0))
    a.apply_translation(T + [20.0, 4.0, 4.0])                  # the long leg, x 0..40
    b = trimesh.creation.box((8.0, 30.0, 8.0))
    b.apply_translation(T + [4.0, 23.0, 4.0])                  # the short leg, y 8..38
    L = a.union(b)
    s = ug.bounding_shape("box", L.vertices, L.faces, lo=T + [9.0, -1.0, -1.0], hi=T + [41.0, 9.0, 9.0], axis="x")
    assert 2 * s.dims_mm["x_half"] == pytest.approx(31.0, abs=0.5) and 2 * s.dims_mm["y_half"] == pytest.approx(8.0, abs=1e-6)
    assert "sub-part" in s.label
    with pytest.raises(ValueError):
        ug.bounding_shape("box", L.vertices, L.faces, lo=T + [100, 100, 100], hi=T + [101, 101, 101])


def test_the_exact_shape_and_its_sizes_round_trip():
    s = ug.exact_shape("cone", T, (0.0, 90.0, 0.0), (5.0, 2.0, 18.0))
    assert s.dims_mm == {"radius1": 5.0, "radius2": 2.0, "half_length": 9.0} and s.center_mm == tuple(T)
    from reco_engine.result_to_program import rotation_matrix
    assert np.allclose(rotation_matrix(s.primitive())[:, 2], [1.0, 0.0, 0.0], atol=1e-9)   # the axis turned onto x
    assert ug.shape_sizes(s) == [5.0, 2.0, 18.0]
    assert ug.shape_sizes(ug.exact_shape("box", T, sizes=(3, 4, 5))) == [3.0, 4.0, 5.0]
    with pytest.raises(ValueError):
        ug.exact_shape("box", T, sizes=(3, 4))
    with pytest.raises(ValueError):
        ug.exact_shape("sphere", T, sizes=(-1,))


# ---------------------------------------------------------------------------------------------------- the dialog, the flow
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _plate():
    box = trimesh.creation.box((60.0, 20.0, 4.0))
    box.apply_translation(T)
    return box


def test_the_dialog_gives_the_loops_first_guess_entry(qapp):
    from reco_gui.start_part_dialog import StartPartDialog
    plate = _plate()
    d = StartPartDialog(plate.vertices, plate.faces, part_name="plate")
    assert d.entry() is None and d.summary() == "automatic"              # automatic by default
    d.mode_buttons["around"].setChecked(True)
    d.kind.setCurrentIndex(d.kind.findData("box"))
    d.axis.setCurrentIndex(d.axis.findData("z"))
    e = d.entry()
    assert e["op"] == "add" and e["kind"] == "box" and np.allclose(e["center_mm"], T)
    assert e["role"] == "cover" and e["refined"] is True and not d.refit.isVisible()   # a cover is never refitted
    assert "100.0%" in d.hint.text()
    d.sub.setChecked(True)
    d.region[("hi", 0)].setValue(T[0])                                   # the left half of the plate
    e = d.entry()
    assert 2 * e["dims_mm"]["x_half"] == pytest.approx(30.0, abs=0.5) and d.summary() == "box around the sub-part"
    d.mode_buttons["exact"].setChecked(True)
    d.kind.setCurrentIndex(d.kind.findData("sphere"))
    assert d.cover.isChecked() and d.entry()["role"] == "cover"          # filled from 'around': a cover of the sub-part
    assert d.entry()["region_mm"] is not None
    d.cover.setChecked(False)
    d.sizes[0].setValue(7.5)
    d.refit.setChecked(False)
    e = d.entry()
    assert e["kind"] == "sphere" and e["dims_mm"] == {"radius": 7.5} and e["refined"] is True and e["role"] is None
    assert d.summary() == "sphere exact, as given"


def test_the_flow_goes_part_starting_part_reconstruct_and_hands_over_the_starting_part(qapp, tmp_path, monkeypatch):
    from PyQt5 import QtWidgets
    from reco_gui.app import MainWindow
    from reco_gui import start_part_dialog as spd
    stl = tmp_path / "plate.stl"
    _plate().export(stl)
    w = MainWindow(stl=str(stl), autosave=False)
    assert w.flow.current == 1 and 0 in w.flow.done and w.flow.notes[0] == "plate.stl" and w.flow.notes[1] == "automatic"

    def accept(dlg):
        dlg.mode_buttons["around"].setChecked(True)
        dlg.kind.setCurrentIndex(dlg.kind.findData("cylinder"))
        return QtWidgets.QDialog.Accepted
    monkeypatch.setattr(spd.StartPartDialog, "exec_", accept)
    w.flow_goto(1)
    assert w.start_part_entry["kind"] == "cylinder" and w.flow.current == 2 and 1 in w.flow.done
    assert w.flow.notes[1] == "cylinder around the part"
    w.requestStart.disconnect(w.worker.start)                              # no real run in a test
    got = []
    w.requestStart.connect(got.append)
    w.flow_goto(2)
    assert got and got[0].cascade_first_guess == [w.start_part_entry] and w.flow.notes[2] == "running ..."
    assert w._flow_run                                                   # B4: step 3 runs through once the session is ready
    menus = [a.text().replace("&", "") for a in w.menuBar().actions()]
    assert menus == ["File", "View", "Tools"] and w.act_compare is not w.act_compare_runs
    w.close()


# ---------------------------------------------------------------------------------------------------- a cover in the loop
@pytest.mark.slow                                                        # two loop runs, ~30 s
def test_a_cover_around_the_part_or_a_sub_part_is_kept_though_its_net_gain_is_negative():
    """B1 of the user's session of 2026-10-29: the box around the ABC plate "gains 0 voxels" and a sub-part box of 00005438
    "-872" were dropped by the add's gain rule. A cover is judged by the material it holds: here an off-centre ring that
    fills 40 % of its box (the box's net gain is negative) - the box around it and the box around its left half are kept,
    a box shifted off the part is still dropped, and the loop cuts the window afterwards."""
    pytest.importorskip("vtk")
    sys.path.insert(0, str(ROOT / "tests"))
    from test_add_cut_loop import _field, _cfg as loop_cfg
    from reco_engine import residual_loop as rl
    from src import sdf_primitives as sp
    shift = np.array([7.0, -3.0, 2.0])
    ring = sp.Placed(sp.Box([10.0, 8.0, 2.0]) - sp.Box([8.0, 6.0, 3.0]), translation=shift)
    field = _field(ring)
    V, F = field.vertices, field.faces
    whole = ug.bounding_shape("box", V, F, axis="z")
    off = ug.UserShape.from_dict(dict(whole.as_dict(), center_mm=list(np.asarray(whole.center_mm) + [30.0, 0.0, 0.0])))
    assert whole.role == "cover" and whole.region_mm is None and off.role == "cover"
    cfg, _ = loop_cfg(first_guess=[dict(whole.as_dict(), refined=True), dict(off.as_dict(), refined=True)], max_ops=4)
    steps = list(rl.residual_steps(field, cfg))
    user = [s for s in steps if s.info.get("source") == "user"]
    assert [s.kind for s in user] == ["candidate", "accepted", "candidate", "rejected"], [s.info["message"] for s in user]
    assert user[0].info["gain"] < cfg.min_gain                          # dropped by the gain rule before the fix
    assert "cover, holds 100.0%" in user[1].info["message"] and "a cover holds 0.0%" in user[3].info["reason"]
    ops = steps[-1].result.operations
    assert ops[0].op == "add" and any(o.op == "cut" for o in ops[1:])   # the window is cut out of the cover

    lo, hi = V.min(axis=0), V.max(axis=0)
    left = ug.bounding_shape("box", V, F, lo=lo, hi=[shift[0], hi[1], hi[2]], axis="z")
    assert left.region_mm is not None and 2 * left.dims_mm["x_half"] == pytest.approx(10.0, abs=0.3)
    cfg, _ = loop_cfg(first_guess=[dict(left.as_dict(), refined=True)], max_ops=4)
    user = [s for s in rl.residual_steps(field, cfg) if s.info.get("source") == "user"]
    assert [s.kind for s in user] == ["candidate", "accepted"] and "in its region" in user[1].info["message"]
