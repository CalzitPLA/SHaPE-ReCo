"""scripts/exact_of_run.py (grow, snap_tool) and scripts/count_holes.py (holes_of): the pure-Python parts, no FreeCAD."""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "dataset_gen")]

import count_holes  # noqa: E402
import exact_of_run  # noqa: E402
from reco_engine.result_schema import Primitive  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp  # noqa: E402

# a drilled blind hole along a tilted axis: cylinder R 2.5 fused with a 120 degree tip, minus a box (a cut operand)
Q = (0.1, 0.2, 0.05, math.sqrt(1 - 0.01 - 0.04 - 0.0025))


def hole():
    from scipy.spatial.transform import Rotation
    d = Rotation.from_quat(Q).as_matrix()[:, 2]
    c = np.array([3.0, -1.0, 2.0])
    cyl = Primitive(kind="cylinder", quaternion=Q, center_mm=tuple(c), dims_mm=dict(radius=2.5, half_length=5.0))
    tip_h = 2.5 / math.tan(math.radians(60))
    cone = Primitive(kind="cone", quaternion=Q, center_mm=tuple(c + d * (5.0 + tip_h / 2)),
                     dims_mm=dict(radius1=2.5, radius2=0.0, half_length=tip_h / 2))
    box = Primitive(kind="box", quaternion=Q, center_mm=tuple(c - d * 5.0), dims_mm=dict(x_half=1.0, y_half=1.0, z_half=1.0))
    body = Primitive(kind="composite", quaternion=(0, 0, 0, 1), center_mm=tuple(c), dims_mm={}, parts=[cyl, cone], combine="fuse")
    return Primitive(kind="composite", quaternion=(0, 0, 0, 1), center_mm=tuple(c), dims_mm={}, parts=[body, box],
                     combine="cut"), d, c, tip_h


def test_grow_contains_and_moves_every_face_out():
    T, d, c, _ = hole()
    sdf, grown, shrunk = (primitive_to_sp(p).sdf for p in (T, exact_of_run.grow(T, 1e-2), exact_of_run.grow(T, -1e-2)))
    P = np.random.default_rng(0).uniform(c - 7.0, c + 7.0, size=(200000, 3))
    s, sg, ss = np.asarray(sdf(P)), np.asarray(grown(P)), np.asarray(shrunk(P))
    assert not np.any((s < 0) & (sg > 0))                  # the grown tool contains the tool
    assert not np.any((ss < 0) & (s > 0))                  # the shrunk one lies inside it
    near = np.abs(s) < 2e-2                                # near the tool's surface the grown one is 1e-2 further out
    assert near.sum() > 100 and abs(np.median(sg[near] - s[near]) + 1e-2) < 1e-3


def test_snap_tool_moves_a_copy_onto_the_input_axis_and_apex():
    T, d, c, tip_h = hole()
    apex = c + d * (5.0 + tip_h)
    faces = [dict(type="cylinder", axis=list(d), center=list(c), radius=2.5),
             dict(type="cone", axis=list(d), center=list(apex), apex=list(apex), radius=0.0)]
    off = np.cross(d, [0.0, 0.0, 1.0])
    off = 3e-6 * off / np.linalg.norm(off) + 2e-6 * d       # a pattern copy 3e-6 mm beside and 2e-6 along the axis
    from reco_engine.result_to_program import transform_primitive
    moved, info = exact_of_run.snap_tool(transform_primitive(T, np.eye(3), off), faces, tol=1e-3)
    assert info["moved"] and info["axial"] and info["shift"] < 1e-5
    cyl = moved.parts[0].parts[0]
    assert np.linalg.norm(np.asarray(cyl.center_mm) - c) < 1e-9
    _, info = exact_of_run.snap_tool(T, faces[:1], tol=1e-3)
    assert info == dict(matched=True, axial=False, moved=False)   # a line without an apex: left where it is


def row(face, kind, axis, foot, radius, span, t, concave=True):
    return dict(face=face, kind=kind, axis=axis, foot=foot, radius=radius, span=span, t=t, concave=concave)


def test_holes_of_counts_full_circles_split_by_gaps():
    z, x = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
    rows = [row(0, "cylinder", z, [0, 0, 0], 2.459, math.pi, [0, 10]),       # a hole in two half faces
            row(1, "cylinder", z, [0, 0, 0], 2.459, math.pi, [0, 10]),
            row(2, "cone", z, [0, 0, 0], 2.459, 2 * math.pi, [10, 11.4]),       # its tip
            row(3, "cylinder", z, [0, 0, 0], 2.459, 2 * math.pi, [20, 30]),     # the same line after a gap: another hole
            row(4, "cylinder", x, [0, 5, 5], 5.0, math.pi / 2, [0, 50]),        # a fillet: no hole
            row(5, "cylinder", x, [0, 9, 9], 6.0, 2 * math.pi, [0, 7], concave=False)]   # a boss: no hole
    holes, partial = count_holes.holes_of(rows, 0.01, 0.05)
    assert len(holes) == 2 and partial == 2
    assert holes[0]["cones"] == 1 and holes[0]["radius"] == 2.46     # the radius to the tolerance's grid
