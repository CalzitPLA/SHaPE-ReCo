"""Super-thin wall suppression of the feature input (feature_stage.piece_thickness, piece_grid, open_thin,
guess_feature): a real pocket piece, a skin-only piece and a pocket with a skin attached - asymmetric, off-centre,
turned out of the axes, built in freecadcmd."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

pytest.importorskip("igl")

from reco_engine import feature_stage as fs  # noqa: E402
from reco_field.export import find_freecadcmd  # noqa: E402

pytestmark = pytest.mark.skipif(find_freecadcmd() is None, reason="freecadcmd not found")

DIAGONAL = 250.0                    # the input's diagonal of this piece set: the threshold is 0.5
PIECES = r"""
import json, os
import FreeCAD as App, Part, MeshPart
from FreeCAD import Vector as V
job = json.load(open(os.environ["PIECES_JOB"]))
pocket = Part.makeBox(30, 20, 12, V(40, -15, 8))
skin = Part.makeBox(80, 50, 0.2, V(-35, 20, 3))
both = Part.makeBox(30, 20, 12, V(-60, 5, -30)).fuse(Part.makeBox(90, 60, 0.2, V(-60, 5, -30))).removeSplitter()
place = App.Placement(V(7, -3, 11), App.Rotation(V(1, 2, 3), 25))
out = {}
for name, s in (("pocket", pocket), ("skin", skin), ("both", both)):
    s.Placement = place
    assert s.isValid() and len(s.Solids) == 1
    MeshPart.meshFromShape(Shape=s, LinearDeflection=0.01, AngularDeflection=0.1).write(os.path.join(job["dir"], name + ".stl"))
    s.exportStep(os.path.join(job["dir"], name + ".step"))
    b = s.BoundBox
    out[name] = dict(volume=s.Volume, bbox=[[b.XMin, b.YMin, b.ZMin], [b.XMax, b.YMax, b.ZMax]])
json.dump(out, open(os.path.join(job["dir"], "pieces.json"), "w"))
"""


@pytest.fixture(scope="module")
def pieces(tmp_path_factory):
    d = tmp_path_factory.mktemp("thin_wall")
    (d / "pieces.py").write_text(PIECES)
    (d / "job.json").write_text(json.dumps(dict(dir=str(d))))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["PIECES_JOB"] = str(d / "job.json")
    proc = subprocess.run([find_freecadcmd(), str(d / "pieces.py")], env=env, capture_output=True, text=True, timeout=300)
    assert (d / "pieces.json").is_file(), (proc.stderr or proc.stdout)[-400:]
    from src.closest_point_field import load_field
    info = json.loads((d / "pieces.json").read_text())
    return {k: dict(v, id=k, sign="minus", faces=[], stl=str(d / f"{k}.stl"), step=str(d / f"{k}.step"),
                    field=load_field(str(d / f"{k}.stl"))) for k, v in info.items()}


def test_the_threshold_is_a_share_of_the_diagonal_or_absolute():
    assert fs.thin_wall_threshold(DIAGONAL) == pytest.approx(fs.THIN_WALL * DIAGONAL)
    assert fs.thin_wall_threshold(DIAGONAL, mm=1.5) == 1.5
    assert fs.thin_wall_threshold(DIAGONAL, 0.0) == 0.0


def test_only_the_skin_is_thin_as_a_whole(pieces):
    thin = fs.thin_wall_threshold(DIAGONAL)
    th = {k: fs.piece_thickness(p["field"], thin) for k, p in pieces.items()}
    assert th["skin"]["suppressed"] and th["skin"]["p95"] == pytest.approx(0.2, abs=1e-3)
    assert not th["pocket"]["suppressed"] and th["pocket"]["thin_share"] == 0.0
    assert th["pocket"]["tau"] == pytest.approx(12.0, abs=1e-3)
    assert not th["both"]["suppressed"] and 0.5 < th["both"]["thin_share"] < 1.0
    assert th["both"]["p5"] == pytest.approx(0.2, abs=1e-3)          # the skin is the thinnest wall
    assert th["both"]["tau"] > 10.0                                   # ... and no longer the one that counts
    off = {k: fs.piece_thickness(p["field"], 0.0) for k, p in pieces.items()}
    assert not any(t["suppressed"] for t in off.values())
    assert off["both"]["tau"] == pytest.approx(0.2, abs=1e-3) and off["both"]["thin_share"] == 0.0


def test_the_skin_no_longer_dictates_the_grid(pieces):
    thin = fs.thin_wall_threshold(DIAGONAL)
    fld = pieces["both"]["field"]
    _, _, before = fs.piece_grid(fld, max_cells=200_000)
    _, ch, after = fs.piece_grid(fld, max_cells=200_000, thin=thin)
    assert before["relaxed"] and before["tau"] == pytest.approx(0.2, abs=1e-3)
    assert not after["relaxed"] and after["tau"] > 10.0 and after["cells"] <= 200_000
    assert after["thin_share"] > 0.5 and after["thin_threshold"] == pytest.approx(thin)
    assert (np.asarray(ch[0]) > 0.5).sum() * after["pitch"] ** 3 == pytest.approx(30 * 20 * 12, rel=0.25)


def test_the_opening_removes_the_skin_and_keeps_the_pocket():
    occ = np.zeros((40, 30, 24), bool)
    occ[5:17, 8:20, 3:15] = True                                     # the pocket, 12 cells thick
    occ[5:38, 4:28, 3:5] = True                                      # a skin of two cells on its floor
    opened = fs.open_thin(occ, pitch=0.1, thin=0.5)                   # a ball of 2.5 cells: 5 cells and thicker stay
    assert not opened[20:, :, :].any() and not opened[:, :8, :].any()  # the skin is gone
    assert opened[8:14, 11:17, 6:12].all()                           # the pocket's core stays
    assert opened.sum() <= occ[5:17, 8:20, 3:15].sum()
    assert fs.open_thin(occ, pitch=1.0, thin=0.5) is occ              # a grid that does not resolve it: untouched
    assert fs.open_thin(occ, pitch=0.1, thin=0.0) is occ              # off
    skin = np.zeros((20, 20, 20), bool)
    skin[2:18, 2:18, 9:11] = True
    assert fs.open_thin(skin, pitch=0.1, thin=0.5) is skin            # nothing would be left: untouched


def test_a_suppressed_piece_stays_in_the_output_with_its_volume(pieces):
    thin = fs.thin_wall_threshold(DIAGONAL)
    f = fs.guess_feature(pieces["skin"], sources=("fit",), thin=thin)
    assert f.suppressed is not None and "super-thin" in f.suppressed and f.primitive is None
    assert f.volume == pytest.approx(80 * 50 * 0.2, rel=1e-6) and f.thickness["suppressed"]
    g = fs.guess_feature(pieces["pocket"], sources=("fit",), thin=thin)
    assert g.suppressed is None and g.primitive is not None and g.kind == "box" and g.score > 0.95
    h = fs.guess_feature(pieces["skin"], sources=("fit",), thin=0.0)   # off: guessed as before
    assert h.suppressed is None and h.thickness is None


def test_the_loop_lists_a_suppressed_piece_and_keeps_its_volume(tmp_path):
    import trimesh
    sys.path.insert(0, str(ROOT / "tests"))
    from test_feature_phase import World, _phase
    from reco_engine.result_schema import Primitive
    pieces = {}
    for name, ext, t in (("pocket", (6.0, 4.0, 3.0), (40.0, 25.0, 30.0)), ("skin", (40.0, 30.0, 0.1), (-50.0, 35.0, 60.0))):
        T = np.eye(4)
        T[:3, 3] = t
        m = trimesh.creation.box(ext, T)
        m.export(tmp_path / f"{name}.stl")
        prim = Primitive(kind="box", center_mm=t, dims_mm=dict(zip(("x_half", "y_half", "z_half"), np.array(ext) / 2)))
        pieces[name] = dict(sign="minus", volume=float(m.volume), centre=t, stl=str(tmp_path / f"{name}.stl"),
                            prim=prim, bbox=[list(m.bounds[0]), list(m.bounds[1])], score=0.95)
    w = World(pieces)
    steps, _, info = _phase(w, feature_min_share=1e-6, feature_thin_wall_mm=0.5, out=tmp_path)
    assert w.guessed == ["pocket"]                                     # the skin is never guessed
    first = steps[0].info
    assert [q["piece"] for q in first["suppressed"]] == ["skin"] and "super-thin" in first["suppressed"][0]["reason"]
    assert first["suppressed_volume"] == pytest.approx(120.0)
    assert info["applied"] == 1 and info["suppressed_pieces"] == 1 and info["suppressed_volume"] == pytest.approx(120.0)
    assert info["invented_volume"] == pytest.approx(120.0)             # it still counts in the exact residual
    w = World(pieces)
    steps, _, info = _phase(w, feature_min_share=1e-6, feature_thin_wall=0.0, out=tmp_path)
    assert sorted(w.guessed) == ["pocket", "skin"] and info["suppressed_pieces"] == 0 and info["applied"] == 2
