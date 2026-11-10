"""
The rough model of a turned part (reco_engine/feature_stage.py snap_to_faces, reco_engine/turned_contour.py): the
sector's end planes on the input's planes through the axis, and the turned contour r(t) read from the input's coaxial
cylinders and cones. The faces are written by hand as reco_field/exact_check.py's exact_faces gives them, about a
tilted axis off the origin.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

from reco_engine import feature_stage as fs  # noqa: E402
from reco_engine.result_schema import Operation, Primitive, ReconstructionResult  # noqa: E402
from reco_engine.result_to_program import quaternion_of, rotation_matrix  # noqa: E402

AXIS = np.array([2.0, -1.0, 3.0]) / math.sqrt(14.0)
BASE = np.array([40.0, -25.0, 310.0])
X = np.cross(AXIS, [0.0, 0.0, 1.0])
X /= np.linalg.norm(X)
Y = np.cross(AXIS, X)
FRAME = np.stack([X, Y, AXIS], axis=1)


def e(angle):
    return math.cos(angle) * X + math.sin(angle) * Y


def t(angle):
    return -math.sin(angle) * X + math.cos(angle) * Y


def box_of(points):
    points = np.asarray(points, float)
    return [list(points.min(axis=0)), list(points.max(axis=0))]


def sector_faces(radius, bore, t0, t1, u0, u1):
    """A sector of a tube about the axis between the angles u0 and u1: outside, bore, the two end planes square to
    the axis and the two planes through the axis."""
    tm = 0.5 * (t0 + t1)
    um = 0.5 * (u0 + u1)
    corners = [BASE + AXIS * s + r * e(u) for s in (t0, t1) for r in (bore, radius) for u in np.linspace(u0, u1, 9)]
    faces = []
    for r, convex in ((radius, True), (bore, False)):
        faces.append(dict(type="cylinder", area=(u1 - u0) * r * (t1 - t0), axis=list(AXIS), center=list(BASE), radius=r,
                          convex=convex, point=list(BASE + AXIS * tm + r * e(um)),
                          normal=list(e(um) if convex else -e(um)), along=[t0, t1], bbox=box_of(corners)))
    for s, sign in ((t0, -1.0), (t1, 1.0)):
        faces.append(dict(type="plane", area=0.5 * (u1 - u0) * (radius ** 2 - bore ** 2), normal=list(sign * AXIS),
                          point=list(BASE + AXIS * s + 0.5 * (radius + bore) * e(um)), bbox=box_of(corners)))
    for u, sign in ((u0, -1.0), (u1, 1.0)):
        side = [BASE + AXIS * s + r * e(u) for s in (t0, t1) for r in (bore, radius)]
        faces.append(dict(type="plane", area=(radius - bore) * (t1 - t0), normal=list(sign * t(u)),
                          point=list(BASE + AXIS * tm + 0.5 * (radius + bore) * e(u)), bbox=box_of(side)))
    return faces


def fitted(radius, t0, t1, u, shift=(0.0, 0.0), spin=0.0):
    """The rough sector as a fit gives it: the axis `shift` beside the true one (in the frame's x, y), its frame
    turned by `spin` about the axis, the angles `u` counted in that frame."""
    c, s = math.cos(spin), math.sin(spin)
    R = FRAME @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    centre = BASE + AXIS * 0.5 * (t0 + t1) + shift[0] * X + shift[1] * Y
    return Primitive(kind="cylinder", quaternion=quaternion_of(R), center_mm=tuple(centre),
                     dims_mm=dict(radius=radius, half_length=0.5 * (t1 - t0)), u=(u[0] - spin, u[1] - spin),
                     label="bounding cylinder sector")


def world_ends(p):
    R = rotation_matrix(p)
    return [math.cos(a) * R[:, 0] + math.sin(a) * R[:, 1] for a in p.u]


@pytest.mark.parametrize("u0,u1,spin", [(0.0, math.pi, 0.0), (0.0, math.pi, 1.3), (0.4, 0.4 + math.radians(120.0), -2.0)])
def test_a_sector_ends_on_the_planes_through_the_axis(u0, u1, spin):
    """The liner's case (2026-09-29): the axis was fitted 1.26 mm beside the true one, the angles counted about it
    came out 0.58 degrees too wide at both ends (181.16 degrees)."""
    faces = sector_faces(192.0, 125.0, -4.0, 1587.3, u0, u1)
    wide = math.radians(0.58)
    um = 0.5 * (u0 + u1)
    p = fitted(192.3, -4.0, 1587.3, (u0 - wide, u1 + wide), shift=tuple(1.26 * np.array([math.cos(um), math.sin(um)])),
               spin=spin)
    result = ReconstructionResult(input_path="", operations=[Operation("add", p)])
    moved = fs.snap_to_faces(result, faces)
    assert len(moved) == 1
    assert moved[0]["sector_deg_before"] == pytest.approx(math.degrees(u1 - u0) + 1.16, abs=1e-6)
    assert moved[0]["sector_deg"] == pytest.approx(math.degrees(u1 - u0), abs=1e-7)
    assert p.dims_mm["radius"] == pytest.approx(192.0)
    a, b = world_ends(p)
    assert np.allclose(a, e(u0), atol=1e-9) and np.allclose(b, e(u1), atol=1e-9)
    assert np.allclose(rotation_matrix(p)[:, 2], AXIS, atol=1e-12)
    off = np.asarray(p.center_mm) - BASE
    assert np.linalg.norm(off - AXIS * (off @ AXIS)) < 1e-9


def test_a_sector_without_a_plane_through_its_axis_keeps_its_angles():
    """The planes of the input lie 8 degrees off the sector's ends: further than SNAP_SECTOR, nothing turns."""
    faces = sector_faces(192.0, 125.0, 0.0, 400.0, 0.0, math.pi)
    p = fitted(192.0, 0.0, 400.0, (-math.radians(8.0), math.pi + math.radians(8.0)))
    before = tuple(p.u)
    fs.snap_to_faces(ReconstructionResult(input_path="", operations=[Operation("add", p)]), faces)
    assert tuple(p.u) == pytest.approx(before)


def test_a_plane_beside_the_axis_is_no_end_of_the_sector():
    """A plane parallel to the axis but 40 mm beside it (a flat on the part) does not contain the axis."""
    faces = [f for f in sector_faces(192.0, 125.0, 0.0, 400.0, 0.0, math.pi) if f["type"] != "plane"]
    faces.append(dict(type="plane", area=5000.0, normal=list(-t(0.0)), point=list(BASE + AXIS * 200.0 + 150.0 * e(0.0) + 40.0 * t(0.0)),
                      bbox=box_of([BASE - 500.0, BASE + 500.0])))
    p = fitted(192.0, 0.0, 400.0, (-0.01, math.pi + 0.01))
    fs.snap_to_faces(ReconstructionResult(input_path="", operations=[Operation("add", p)]), faces)
    assert tuple(p.u) == pytest.approx((-0.01, math.pi + 0.01))


# ------------------------------------------------------------------------------------------------ the turned contour

from reco_engine import turned_contour as tc  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp  # noqa: E402


def curved(kind, t0, t1, r0, r1, convex=True, u=(0.0, math.pi), along=True, flip=False):
    """A coaxial cylinder or cone between t0 and t1 (radius r0 at t0, r1 at t1), as exact_faces writes it: the
    cone's centre is where its radius is `radius`, its apex apart; `flip`: the surface's axis the other way round."""
    um, tm = 0.5 * (u[0] + u[1]), 0.5 * (t0 + t1)
    ax = -AXIS if flip else AXIS
    pts = [BASE + AXIS * s + r * e(a) for s, r in ((t0, r0), (t1, r1)) for a in np.linspace(u[0], u[1], 9)]
    f = dict(type="cylinder" if kind == "cyl" else "cone", axis=list(ax), convex=convex, bbox=box_of(pts),
             point=list(BASE + AXIS * tm + 0.5 * (r0 + r1) * e(um)))
    if kind == "cyl":
        f.update(center=list(BASE + AXIS * 3.0), radius=r0, area=(u[1] - u[0]) * r0 * (t1 - t0))
        ref = 3.0
    else:
        k = (r1 - r0) / (t1 - t0)
        t_apex = t0 - r0 / k
        f.update(center=list(BASE + AXIS * t0), radius=r0, apex=list(BASE + AXIS * t_apex), semi_angle=math.atan(abs(k)),
                 area=0.5 * (u[1] - u[0]) * abs(r1 ** 2 - r0 ** 2) / math.sin(math.atan(abs(k))))
        ref = t0
    if along:
        f["along"] = sorted((-1.0 if flip else 1.0) * (t - ref) for t in (t0, t1))
    return f


def liner_like(along=True):
    """The liner's blank about the tilted axis: half a tube R 192 / bore R 125 from -4 to 1587.3, chamfer 10 x 45
    degrees at the bottom, shoulder R 175 from 1537.3 with a chamfer 10 x 45 degrees at the top; R 192 in two faces
    (a pocket splits it), a pocket corner R 5 beside the axis and a bolt hole's cylinder across it."""
    faces = [curved("cone", -4.0, 6.0, 182.0, 192.0, along=along, flip=True),
             curved("cyl", 6.0, 800.0, 192.0, 192.0, along=along), curved("cyl", 700.0, 1537.3, 192.0, 192.0, along=along),
             curved("cyl", 1537.3, 1577.3, 175.0, 175.0, along=along, flip=True),
             curved("cone", 1577.3, 1587.3, 175.0, 165.0, along=along),
             curved("cyl", -4.0, 1587.3, 125.0, 125.0, convex=False, along=along)]
    corner = curved("cyl", 200.0, 280.0, 5.0, 5.0, convex=False, along=along)
    corner["center"] = list(np.asarray(corner["center"]) + 170.0 * e(1.0))
    hole = curved("cyl", 0.0, 60.0, 6.0, 6.0, convex=False, along=along)
    hole["axis"] = list(e(0.8))
    return faces + [corner, hole]


def liner_inside(pts):
    """The material of that blank."""
    q = (np.asarray(pts, float) - BASE) @ FRAME
    r, s = np.hypot(q[:, 0], q[:, 1]), q[:, 2]
    outer = np.where(s >= 1537.3, np.minimum(175.0, 175.0 - (s - 1577.3)), np.minimum(192.0, 182.0 + (s + 4.0)))
    return (q[:, 1] >= 0.0) & (s >= -4.0) & (s <= 1587.3) & (r >= 125.0) & (r <= outer)


def rough(bore=125.0, u=(0.0, math.pi)):
    ops = [Operation("add", fitted(192.0, -4.0, 1587.3, u)), Operation("cut", fitted(bore, -50.0, 1650.0, u))]
    ops[1].primitive.u = None
    return ReconstructionResult(input_path="", operations=ops)


def sampled_volume(model, n=400000, seed=1):
    lo, hi = BASE - 1700.0, BASE + 1700.0
    rng = np.random.default_rng(seed)
    q = rng.uniform([-200.0, -200.0, -60.0], [200.0, 200.0, 1660.0], size=(n, 3))
    pts = BASE + q @ FRAME.T
    assert (pts > lo).all() and (pts < hi).all()
    return float(np.mean(model.sdf(pts) <= 0.0)) * 400.0 * 400.0 * 1720.0, pts


def test_the_coaxial_faces_of_a_side_with_their_reach():
    faces = liner_like()
    out = tc.coaxial_segments(faces, AXIS, BASE + AXIS * 791.65, convex=True)
    assert [s["face"] for s in out] == [0, 1, 2, 3, 4]                # not the bore, the corner, the radial hole
    got = [(s["kind"], round(s["t0"] + 791.65, 6), round(s["t1"] + 791.65, 6), round(s["r0"], 6), round(s["r1"], 6)) for s in out]
    assert got == [("cone", -4.0, 6.0, 182.0, 192.0), ("cyl", 6.0, 800.0, 192.0, 192.0), ("cyl", 700.0, 1537.3, 192.0, 192.0),
                   ("cyl", 1537.3, 1577.3, 175.0, 175.0), ("cone", 1577.3, 1587.3, 175.0, 165.0)]
    assert [s["face"] for s in tc.coaxial_segments(faces, -AXIS, BASE, convex=False)] == [5]


def test_without_a_reach_a_tilted_face_is_left_out():
    """The box of a face about a tilted axis says too much about its reach; about a world axis it is exact."""
    assert tc.coaxial_segments(liner_like(along=False), AXIS, BASE) == []
    f = dict(type="cylinder", axis=[0.0, 0.0, -1.0], center=[3.0, 4.0, 50.0], radius=20.0, convex=True, area=1.0,
             point=[23.0, 4.0, 15.0], bbox=[[-17.0, 4.0, 10.0], [23.0, 24.0, 20.0]])
    (s,) = tc.coaxial_segments([f], [0.0, 0.0, 1.0], [3.0, 4.0, 0.0])
    assert (s["t0"], s["t1"]) == (10.0, 20.0)


def test_the_profile_is_the_envelope_of_the_faces():
    segs = tc.coaxial_segments(liner_like(), AXIS, BASE, convex=True)
    segs.append(dict(kind="cyl", t0=300.0, t1=500.0, r0=150.0, r1=150.0, area=1.0, face=99))     # below R 192: hidden
    prof = tc.contour(segs, -4.0, 1587.3)
    assert [(p["kind"], round(p["t0"], 6), round(p["t1"], 6), round(p["r0"], 6), round(p["r1"], 6)) for p in prof] == \
        [("cone", -4.0, 6.0, 182.0, 192.0), ("cyl", 6.0, 1537.3, 192.0, 192.0), ("cyl", 1537.3, 1577.3, 175.0, 175.0),
         ("cone", 1577.3, 1587.3, 175.0, 165.0)]
    assert prof[1]["faces"] == [1, 2]
    gap = tc.contour([s for s in segs if s["face"] in (0, 4)], -4.0, 1587.3)
    assert [(round(p["t0"], 6), round(p["t1"], 6)) for p in gap] == [(-4.0, 6.0), (1577.3, 1587.3)]


@pytest.mark.parametrize("u", [(0.0, math.pi), None])
def test_the_rough_cylinder_takes_the_contour(u):
    """Two cuts, the chamfer at the bottom and the shoulder with its chamfer at the top; the model's volume is the
    blank's (closed formula), and no point of the input is cut away."""
    result = rough()
    result.operations[0].primitive.u = u
    inside = liner_inside if u else (lambda pts: liner_inside(pts) | liner_inside(BASE + ((np.asarray(pts) - BASE) @ FRAME * [1, -1, 1]) @ FRAME.T))
    done = tc.turned_contour(result, liner_like(), inside)
    assert [d["side"] for d in done] == ["outside", "outside"]
    assert [o.op for o in result.operations] == ["add", "cut", "cut", "cut"]
    assert done[0]["label"] == "turned contour, outside: cone R 182 -> 192"
    assert done[1]["label"] == "turned contour, outside: R 175 x 40 | cone R 175 -> 165"
    span = math.pi if u else 2 * math.pi
    bottom = 0.5 * span * (192.0 ** 2 * 10.0 - 10.0 * (182.0 ** 2 + 182.0 * 192.0 + 192.0 ** 2) / 3.0)
    top = 0.5 * span * (192.0 ** 2 * 50.0 - 175.0 ** 2 * 40.0 - 10.0 * (175.0 ** 2 + 175.0 * 165.0 + 165.0 ** 2) / 3.0)
    assert done[0]["volume"] == pytest.approx(bottom, rel=1e-9) and done[1]["volume"] == pytest.approx(top, rel=1e-9)
    assert max(d["harm"] for d in done) == 0.0
    from reco_engine.result_to_program import program_model
    model = program_model(result)
    volume, pts = sampled_volume(model)
    blank = 0.5 * span * (192.0 ** 2 - 125.0 ** 2) * 1591.3 - bottom - top
    assert volume == pytest.approx(blank, rel=5e-3)
    m, i = model.sdf(pts) <= 0.0, inside(pts)
    assert not (i & ~m).any()                                                     # nothing of the input is missing
    assert (m & ~i).mean() * 400.0 * 400.0 * 1720.0 < 2e-4 * blank                # and nothing is too much


def test_a_ring_that_holds_material_is_not_cut():
    """The shoulder's faces are there, but at other angles the part goes on to R 192 (lugs): no cut at the top."""
    def with_lugs(pts):
        q = (np.asarray(pts, float) - BASE) @ FRAME
        r, s, a = np.hypot(q[:, 0], q[:, 1]), q[:, 2], np.arctan2(q[:, 1], q[:, 0])
        lug = (s >= 1537.3) & (s <= 1587.3) & (r >= 125.0) & (r <= 192.0) & (a > 1.0) & (a < 1.2)
        return liner_inside(pts) | lug
    result = rough()
    done = tc.turned_contour(result, liner_like(), with_lugs)
    assert [d["label"] for d in done] == ["turned contour, outside: cone R 182 -> 192"]


def test_a_recess_of_the_bore_is_cut_and_a_narrower_bore_is_reported():
    faces = [f for f in liner_like() if f["convex"]]
    faces += [curved("cyl", -4.0, 900.0, 125.0, 125.0, convex=False), curved("cone", 900.0, 910.0, 125.0, 135.0, convex=False),
              curved("cyl", 910.0, 1587.3, 135.0, 135.0, convex=False)]

    def inside(pts):
        q = (np.asarray(pts, float) - BASE) @ FRAME
        r, s = np.hypot(q[:, 0], q[:, 1]), q[:, 2]
        return liner_inside(pts) & (r >= np.clip(125.0 + (s - 900.0), 125.0, 135.0))
    result = rough()
    done = tc.turned_contour(result, faces, inside)
    assert [d["label"] for d in done if d["side"] == "inside"] == ["turned contour, inside: cone R 125 -> 135 | R 135 x 677"]
    from reco_engine.result_to_program import program_model
    model = program_model(result)
    _, pts = sampled_volume(model, n=200000)
    m, i = model.sdf(pts) <= 0.0, inside(pts)
    assert not (i & ~m).any() and (m & ~i).mean() < 1e-4
    result = rough(bore=130.0)                                           # the rough cut took R 130: too much below 905
    done = tc.turned_contour(result, faces, inside)
    assert [d["op"] for d in done if d["side"] == "inside"] == [None, "cut"]
    assert [d["label"] for d in done if d["op"]][-1] == "turned contour, inside: R 135 x 677"


def test_no_contour_without_a_rough_cylinder_or_with_a_second_add():
    box = Primitive(kind="box", dims_mm=dict(x_half=1.0, y_half=1.0, z_half=1.0))
    assert tc.turned_contour(ReconstructionResult(input_path="", operations=[Operation("add", box)]), liner_like(), liner_inside) == []
    result = rough()
    result.operations.append(Operation("add", box))
    assert tc.turned_contour(result, liner_like(), liner_inside) == []
    assert len(result.operations) == 3
