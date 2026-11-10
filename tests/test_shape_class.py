"""Shape classes (reco_engine/shape_class.py): the Betti numbers of a STEP part from FreeCAD (exact_check.betti through
exact_facts), the surface family, the through holes that explain handles, a known tree's complexity, and the key."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine import shape_class as sc  # noqa: E402
from reco_field.export import build_in_freecad, find_freecadcmd  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402

needs_freecad = pytest.mark.skipif(find_freecadcmd() is None, reason="freecadcmd not found")

T = np.array([40.0, -25.0, 12.0])                      # off-centre on every axis


def _box():
    return sp.Box((20.0, 15.0, 10.0))


def _bore(x, y, r=4.0):
    return sp.Placed(sp.Cylinder(r, 40.0), translation=np.array([x, y, -20.0]))


SOLIDS = {
    "box": (lambda: _box(), "g0.P.K1-6", (1, 0, 0)),
    "bore": (lambda: _box() - _bore(3.0, 2.0), "g1.M.K7-15", (1, 1, 0)),
    "two_bores": (lambda: _box() - _bore(-10.0, 2.0) - _bore(10.0, -2.0), "g2-3.M.K7-15", (1, 2, 0)),
    "void": (lambda: _box() - sp.Placed(sp.Sphere(5.0), translation=np.array([2.0, 1.0, 0.0])), "g0v.D.K7-15", (1, 0, 1)),
    "torus": (lambda: sp.Torus(20.0, 5.0), "g1.A.K1-6", (1, 1, 0)),
}


@pytest.fixture(autouse=True)
def own_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAPE_RECO_SHAPE_CLASS_CACHE", str(tmp_path / "cache"))


@pytest.fixture(scope="module")
def described(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("solids")
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SHAPE_RECO_SHAPE_CLASS_CACHE", str(tmp / "cache"))
        for name, (make, _, _) in SOLIDS.items():
            ex = build_in_freecad(sp.Placed(make(), translation=T), tmp / name, name)
            out[name] = sc.describe(ex.step)
    return out


@needs_freecad
def test_betti_numbers_and_keys_of_solids_with_known_topology(described):
    for name, (_, key, betti) in SOLIDS.items():
        d = described[name]
        assert "error" not in d, (name, d)
        assert (d["b0"], d["b1"], d["b2"]) == betti, (name, d)
        assert sc.classify(d)["key"] == key, (name, d)
    assert described["void"]["shell_genus"] == [0, 0]                  # outer shell and the void, both spheres
    assert "void" in sc.classify(described["void"])["flags"]


@needs_freecad
def test_the_euler_poincare_fallback_agrees_with_the_tessellation(described, tmp_path):
    """The B-rep counts (used when a tessellation is not closed) give the same Betti numbers."""
    for name, (_, _, betti) in SOLIDS.items():
        ep = sc.euler_poincare(Path(described[name]["path"]), tmp_path / name)
        assert (ep["b0"], ep["b1"], ep["b2"]) == betti, (name, ep)
    assert described["box"]["betti_method"] == "tessellation"


def test_more_tessellation_pieces_than_shells_are_several_bodies():
    assert sc.topology_class(dict(b0=1, b1=-1, b2=0)) == "multi"


@needs_freecad
def test_bores_are_hole_handles_and_a_torus_handle_is_not(described):
    """A through bore is a handle the hole stacks explain (b1_holes); the torus's handle is of the window kind."""
    assert (described["bore"]["b1_holes"], described["bore"]["b1_rest"]) == (1, 0)
    assert (described["two_bores"]["b1_holes"], described["two_bores"]["b1_rest"]) == (2, 0)
    assert (described["torus"]["b1_holes"], described["torus"]["b1_rest"]) == (0, 1)
    assert "holes" in sc.classify(described["bore"])["flags"]
    assert described["torus"]["blends"] == 0                           # a whole torus is no blend


@needs_freecad
def test_a_blind_bore_is_no_handle(tmp_path):
    blind = _box() - sp.Placed(sp.Cylinder(4.0, 12.0), translation=np.array([3.0, 2.0, 2.0]))   # from the top, 8 deep
    d = sc.describe(build_in_freecad(sp.Placed(blind, translation=T), tmp_path, "blind").step)
    assert (d["b1"], d["through_holes"]) == (0, 0)


def test_an_stl_is_the_mesh_fallback_two_bodies_are_multi_and_a_void_counts(tmp_path):
    """Without a B-rep: the mesh's pieces (outer by positive volume, voids by negative), no faces, family '?'.
    (The FreeCAD builder keeps one solid, so the two bodies are a mesh here.)"""
    import trimesh
    a, b = trimesh.creation.box((40.0, 30.0, 20.0)), trimesh.creation.box((6.0, 6.0, 6.0))
    b.apply_translation([60.0, 0.0, 0.0])
    two = trimesh.util.concatenate([a, b])
    two.apply_translation(T)
    two.export(tmp_path / "two.stl")
    d = sc.describe(tmp_path / "two.stl")
    assert (d["source"], d["b0"], d["b1"], d["b2"]) == ("mesh", 2, 0, 0)
    assert sc.classify(d)["key"].startswith("multi.?.")
    inner = trimesh.creation.box((10.0, 10.0, 10.0))
    inner.invert()                                                     # a void: its faces look into it
    hollow = trimesh.util.concatenate([trimesh.creation.box((40.0, 30.0, 20.0)), inner])
    hollow.apply_translation(T)
    hollow.export(tmp_path / "hollow.stl")
    d = sc.describe(tmp_path / "hollow.stl")
    assert (d["b0"], d["b1"], d["b2"]) == (1, 0, 1) and sc.classify(d)["T"] == "g0v"


def _face(kind, area, **kw):
    return dict(type=kind, area=area, **kw)


def test_surface_families_from_faces():
    z, x = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
    planes = [_face("plane", 100.0, normal=n, point=[0.0, 0.0, 0.0]) for n in (z, x, [0.0, 1.0, 0.0])]
    assert sc.surface_facts(planes, 10.0)["family"] == "P"
    shaft = [_face("cylinder", 400.0, axis=z, center=[0.0, 0.0, 0.0], radius=5.0, convex=True, along=[-6.4, 6.4]),
             _face("plane", 80.0, normal=z, point=[0.0, 0.0, 6.4])]
    assert sc.surface_facts(shaft, 10.0)["family"] == "A"
    bores = planes + [_face("cylinder", 10.0, axis=z, center=[c, 0.0, 0.0], radius=1.0, convex=False) for c in (2.0, -2.0)]
    f = sc.surface_facts(bores, 10.0)
    assert (f["family"], f["axis_lines"], f["concave_curved"]) == ("M", 2, 2)
    dome = planes + [_face("sphere", 60.0, center=[0.0, 0.0, 0.0], radius=3.0, convex=True)]
    assert sc.surface_facts(dome, 10.0)["family"] == "D"
    assert sc.surface_facts(planes + [_face("other", 1.0)], 10.0)["family"] == "F"
    fillet = _face("cylinder", 0.25 * 2 * np.pi * 2.0 * 10.0, axis=x, center=[0.0, 0.0, 0.0], radius=2.0, along=[0.0, 10.0])
    assert sc.surface_facts(planes + [fillet], 10.0)["blends"] == 1


def _placed(prim, axis=(0.0, 0.0, 1.0)):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    b = np.cross(a, [1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.cross(a, [0.0, 1.0, 0.0])
    b /= np.linalg.norm(b)
    R = np.column_stack([b, np.cross(a, b), a])                       # local z = axis
    return dict(type="Placed", primitive=prim, rotation=R.tolist(), translation=[0.0, 0.0, 0.0])


def test_tree_facts_of_a_known_tree():
    box = _placed(dict(type="Box", half=[5.0, 5.0, 5.0]))
    bore = _placed(dict(type="Cylinder", radius=1.0, height=20.0, u=[0.0, 2 * np.pi], inner_radius=0.0))
    tube = _placed(dict(type="Cylinder", radius=3.0, height=4.0, u=[0.0, 2 * np.pi], inner_radius=2.0), axis=(1.0, 0.0, 0.0))
    sector = _placed(dict(type="Torus", major=6.0, minor=1.0, u=[0.0, 1.0], v=[0.0, 2.0]), axis=(0.0, 0.0, -1.0))
    tree = dict(type="Union", a=dict(type="Subtraction", a=box, b=bore), b=dict(type="Union", a=tube, b=sector))
    t = sc.tree_facts(tree)
    assert (t["leaves"], t["depth"], t["ops"]) == (4, 2, {"Union": 2, "Subtraction": 1})
    assert t["subtypes"] == {"box/full": 1, "cylinder/full": 1, "cylinder/tube": 1, "torus/sector": 1}
    assert (t["partial"], t["axis_dirs"]) == (2, 2)                    # the torus's -z is the bore's axis


def test_the_c_axis_known_tree_estimate_and_honest_fallback():
    d = dict(b0=1, b1=0, b2=0, family="M", carriers=24, concave_curved=0)
    assert sc.classify(d, dict(leaves=5, partial=0))["C"] == "C4-7"
    assert sc.classify(d)["key"] == "g0.M.K16-40"                       # no calibration: carrier bins, said so
    assert sc.classify(d)["c_source"] == "brep"
    calib = dict(ok=True, a=-1.0, b=1.0, c=0.0)                         # leaves = carriers / 2
    c = sc.classify(d, calib=calib)
    assert (c["C"], c["c_source"], c["leaves"]) == ("C8-15", "estimate", pytest.approx(12.0))
    assert sc.classify(d, calib=dict(calib, ok=False))["c_source"] == "brep"
    assert sc.classify(dict(d, b1=None))["T"] == "g?"
