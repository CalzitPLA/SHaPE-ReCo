#!/usr/bin/env python3
"""feature_stage.contact_grade: the tool's surface as contact (on the input's surface), harm (inside the input's
material) and free (in invented material or air), on analytic cases (meshes by reco_field.peel.mesh_from_sdf)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

pytest.importorskip("vtk")
pytest.importorskip("igl")

from reco_engine import feature_stage as fs  # noqa: E402
from reco_engine import tools as T  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402

DOWN = np.diag([1.0, -1.0, -1.0])


def _mesh(s, res=128):
    from reco_field.peel import mesh_from_sdf
    v, f = mesh_from_sdf(s, resolution=res)
    return np.asarray(v, float), np.asarray(f)


def _grade(tool, input_sdf, piece_sdf, op="cut"):
    """The input and the piece meshed by marching cubes (128 / 96 cells): its edges are rounded by about a cell, so the
    tolerance is 1.2 cells of the input's grid (FreeCAD's tessellation in the loop has exact edges)."""
    Vi, Fi = _mesh(input_sdf)
    Vp, Fp = _mesh(piece_sdf, 96)
    cell = 1.1 * float((Vi.max(axis=0) - Vi.min(axis=0)).max()) / 127
    return fs.contact_grade(tool, Vp, Fp, Vi, Fi, lambda q: np.asarray(input_sdf.sdf(q)) <= 0, tol=1.2 * cell, op=op)


PLATE = sp.Placed(sp.Box([20.0, 20.0, 5.0]), translation=np.array([0.0, 0.0, 5.0]))    # z from 0 to 10


def _hole(length, top=10.0, over=0.0):
    return T.TOOLS["through_hole"].build(dict(d=6.0, length=length), DOWN, np.array([2.0, -3.0, top]), over)


def test_the_over_length_of_a_through_hole_is_free():
    """The same bore, cutter 10 (the plate) or 30 long (10 above and below): contact recall and precision do not
    change, only the free area grows."""
    exact, long = _hole(10.0), _hole(10.0, over=10.0)
    inp = PLATE - primitive_to_sp(long)
    piece = primitive_to_sp(exact)
    a, b = _grade(exact, inp, piece), _grade(long, inp, piece)
    assert a["contact_recall"] > 0.97 and b["contact_recall"] > 0.97
    assert a["contact_precision"] > 0.99 and b["contact_precision"] > 0.99
    assert b["free_area"] > a["free_area"] + 2 * np.pi * 3.0 * 15.0      # the two over-lengths, at least


def test_a_blind_hole_of_the_wrong_depth_costs():
    """Blind hole 6 deep: its floor is a contact face. A cutter 4 deep misses the floor (recall drops), one 8 deep
    cuts into the material under it (precision drops)."""
    def blind(depth):
        return T.TOOLS["circular_pocket"].build(dict(d=6.0, depth=depth), DOWN, np.array([2.0, -3.0, 10.0]), 0.0)
    inp = PLATE - primitive_to_sp(T.TOOLS["circular_pocket"].build(dict(d=6.0, depth=6.0), DOWN, np.array([2.0, -3.0, 10.0]), 1.0))
    piece = primitive_to_sp(blind(6.0))
    right, short, deep = (_grade(blind(t), inp, piece) for t in (6.0, 4.0, 8.0))
    assert right["contact_recall"] > 0.97 and right["contact_precision"] > 0.99
    assert short["contact_recall"] < right["contact_recall"] - 0.15
    assert deep["contact_precision"] < right["contact_precision"] - 0.1 and deep["harm_area"] > 0


def test_a_fillet_tools_box_faces_are_free():
    """box minus cylinder in a concave corner (an add): the cylinder face is the contact, the box's two wall faces lie
    against the walls (inside the input after the add: not its surface) and its caps on the ends - all free."""
    fillet = T.TOOLS["fillet"].build(dict(r=4.0, length=20.0))
    L = sp.Union(sp.Placed(sp.Box([10.0, 10.0, 10.0]), translation=np.array([-10.0, 0.0, 10.0])),
                 sp.Placed(sp.Box([10.0, 10.0, 10.0]), translation=np.array([0.0, -10.0, 10.0])))
    inp = sp.Union(L, primitive_to_sp(fillet))
    g = _grade(fillet, inp, primitive_to_sp(fillet), op="add")
    # contact: the quarter cylinder, the two end faces (flush with the part's ends: real surfaces too) and, on each
    # wall, the band that meets the cylinder TANGENTIALLY - within tol of the input's surface for y > 2.5 of 4 at
    # tol 0.31 (sqrt(16 + (4 - y)^2) - 4 <= tol): about 37 % of each wall, measured 2026-09-28
    contact = 0.5 * np.pi * 4.0 * 20.0 + 2 * (1 - np.pi / 4) * 16.0
    assert contact * 0.9 < g["contact_area"] < contact + 2 * 0.45 * 4.0 * 20.0
    assert g["contact_precision"] > 0.95 and g["free_area"] > 2 * 4.0 * 20.0 * 0.5


def test_a_pocket_open_to_one_side():
    """An open pocket (open at +x through the plate's side): its open end is free, its walls and floor contact."""
    op = T.TOOLS["open_pocket"].build(dict(L=12.0, W=8.0, depth=4.0, r=2.0), DOWN, np.array([14.0, 0.0, 10.0]), 0.0)
    inp = PLATE - primitive_to_sp(T.TOOLS["open_pocket"].build(dict(L=12.0, W=8.0, depth=4.0, r=2.0), DOWN,
                                                               np.array([14.0, 0.0, 10.0]), 1.0))
    piece = sp.Intersection(primitive_to_sp(op), PLATE)
    g = _grade(op, inp, piece)
    assert g["contact_recall"] > 0.95 and g["contact_precision"] > 0.99
    assert g["free_area"] > 8.0 * 4.0 * 0.8                               # the open end and the mouth
