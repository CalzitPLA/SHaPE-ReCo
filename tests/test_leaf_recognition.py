"""reco_engine/leaf_recognition.py: the primitives inside a compound part recognized from its own surfaces - the angular
coverage of a surface, a union of two crossing cylinders through a face inventory built from its mesh (no FreeCAD), and
a real STEP union of a cylinder and a 270-degree cone sector (FreeCAD, slow)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine import leaf_recognition as lr  # noqa: E402


def test_the_turn_of_a_surface_is_whole_or_a_sector():
    t = np.linspace(0, 1.5 * np.pi, 400)                                # a 270-degree arc around z, radius 4
    P = np.stack([4 * np.cos(t + 0.3), 4 * np.sin(t + 0.3), np.linspace(0, 10, 400)], axis=1)
    R, span = lr._turn(P, np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert span == pytest.approx(1.5 * np.pi, abs=math.radians(1.0))
    assert np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360 == pytest.approx(math.degrees(0.3), abs=1.0)   # x at the start
    full = np.stack([4 * np.cos(np.linspace(0, 2 * np.pi, 400, endpoint=False)), np.zeros(400), np.zeros(400)], axis=1)
    full[:, 1] = 4 * np.sin(np.linspace(0, 2 * np.pi, 400, endpoint=False))
    assert lr._turn(full, np.zeros(3), np.array([0.0, 0.0, 1.0]))[1] == pytest.approx(2 * np.pi)


def _inventory(mesh, cylinders):
    """A step_faces-like inventory from a mesh: triangles on each (centre, axis, radius) cylinder form its face, the rest
    are grouped into planes by normal and offset."""
    cent, nrm, area = mesh.triangles_center, mesh.face_normals, mesh.area_faces
    fid = np.full(len(mesh.faces), -1)
    faces = []
    for c, a, r in cylinders:
        c, a = np.asarray(c, float), np.asarray(a, float)
        rel = cent - c
        rho = np.linalg.norm(rel - np.outer(rel @ a, a), axis=1)
        on = (np.abs(rho - r) < 0.03 * r) & (np.abs(nrm @ a) < 0.2) & (fid < 0)
        fid[on] = len(faces)
        faces.append(dict(id=len(faces), type="Cylinder", axis=a.tolist(), center=c.tolist(), radius=float(r), convex=True,
                          dominating=False, area=float(area[on].sum())))
    groups = {}
    for i in np.flatnonzero(fid < 0):
        groups.setdefault((tuple(np.round(nrm[i], 2)), round(float(nrm[i] @ cent[i]), 1)), []).append(i)
    for idx in groups.values():
        fid[idx] = len(faces)
        faces.append(dict(id=len(faces), type="Plane", normal=nrm[idx[0]].tolist(), point=cent[idx[0]].tolist(),
                          dominating=False, area=float(area[idx].sum())))
    V = np.asarray(mesh.vertices, float)
    return dict(faces=faces, vertices=V, triangles=np.asarray(mesh.faces), face_id=fid,
                diag=float(np.linalg.norm(np.ptp(V, axis=0))))


def test_a_union_of_two_crossing_cylinders_is_two_adds():
    trimesh = pytest.importorskip("trimesh")
    big = trimesh.creation.cylinder(radius=5.0, height=20.0, sections=96)                     # along z
    small = trimesh.creation.cylinder(radius=3.0, height=30.0, sections=96)
    small.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))     # along x
    part = trimesh.boolean.union([big, small], engine="manifold")
    inv = _inventory(part, [((0, 0, 0), (0, 0, 1), 5.0), ((0, 0, 0), (1, 0, 0), 3.0)])
    hyps, surfaces = lr.hypotheses(inv)
    assert [s.kind for s in surfaces if s.kind != "plane"] == ["cylinder", "cylinder"]
    assert all(h.whole for h in hyps if h.surface == "cylinder")                             # both go all the way round
    mesh = (np.asarray(part.vertices), np.asarray(part.faces))
    lr.verify(hyps, mesh)
    adds = [h for h in hyps if h.passed and h.role == "add" and h.surface == "cylinder"]
    assert sorted(round(h.primitive.dims_mm["radius"], 1) for h in adds) == [3.0, 5.0]
    program, info = lr.select(hyps, mesh, n=60_000)
    kinds = sorted((o.op, o.primitive.kind, round(o.primitive.dims_mm["radius"], 1)) for o in program.operations)
    assert kinds == [("add", "cylinder", 3.0), ("add", "cylinder", 5.0)] and info["iou"] > 0.98
    small_add = next(o.primitive for o in program.operations if round(o.primitive.dims_mm["radius"], 1) == 3.0)
    assert small_add.dims_mm["half_length"] == pytest.approx(15.0, rel=0.02)                 # its own extent, not the part's


@pytest.mark.slow
def test_a_step_union_of_a_cylinder_and_a_cone_sector_is_recognized(tmp_path):
    """A real STEP: a cylinder (r 6, along z) and a 270-degree cone sector (r 9 -> 3, along x) joined. The recognition
    recalls both leaves of the program it was built from (validation.match_leaves) with two parts."""
    from reco_field.export import build_in_freecad, find_freecadcmd
    if find_freecadcmd() is None:
        pytest.skip("freecadcmd not found")
    from src import sdf_primitives as sp
    from reco_engine.csg_surface import compile_program, compile_tree
    from reco_engine.result_to_program import primitive_to_sp
    from reco_engine.validation import match_leaves
    cyl = sp.Placed(sp.Cylinder(6.0, 24.0), translation=np.array([0.0, 0.0, -12.0]))
    Ry = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])                    # local z -> world x
    a = math.radians(25.0)                                  # turned about its axis: no sector face in the cylinder's seam
    Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])   # (OCCT's fuse
    cone = sp.Placed(sp.Cone(9.0, 3.0, 30.0, (0.0, 1.5 * np.pi)), rotation=Ry @ Rz,        # returned the cylinder alone,
                     translation=np.array([-10.0, 0.0, 0.0]))                               # valid, without an error)
    truth = sp.Union(cyl, cone)
    ex = build_in_freecad(truth, tmp_path, "union")
    assert ex.volume > 1.2 * math.pi * 36 * 24                       # the union holds the cone, not the cylinder alone
    rec = lr.recognize(ex.step)
    ops = rec["program"].operations
    assert len(ops) == 2 and all(o.op == "add" for o in ops) and rec["info"]["iou"] > 0.98
    model = compile_program([(o.op, primitive_to_sp(o.primitive)) for o in ops])
    lo, hi = (np.asarray(b) for b in truth.bounds())
    m = match_leaves(compile_tree(truth), model, float(np.linalg.norm(hi - lo)))
    assert m["recall"] == 1.0 and m["precision"] == 1.0
    sector = next(o.primitive for o in ops if o.primitive.kind == "cone")
    assert sector.u is not None and sector.u[1] == pytest.approx(1.5 * np.pi, abs=0.05)
