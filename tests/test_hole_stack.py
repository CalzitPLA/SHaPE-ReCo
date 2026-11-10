"""
Hole stacks (reco_engine/hole_stack.py): a hole's steps read from a piece's faces as they come from
reco_field/exact_check.py - here written by hand for a bolt hole like the liner's (counterbore, shank, cone, thread
core), tilted and off-centre, its faces split in two halves by the seam. Checked: every step's radius and reach, with
and without the faces' `along`; that a pocket's quarter-cylinder corner is no step; which ends are closed; and the
built tool's volume (sampled in its SDF) against the closed formula.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

from reco_engine import hole_stack as hs  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp  # noqa: E402

AXIS = np.array([2.0, -1.0, 3.0]) / math.sqrt(14.0)
BASE = np.array([40.0, -25.0, 310.0])
X = np.cross(AXIS, [0.0, 0.0, 1.0])
X /= np.linalg.norm(X)
Y = np.cross(AXIS, X)


def ring(t, r, n=24, turn=2 * math.pi):
    a = np.linspace(0.0, turn, n, endpoint=False)
    return BASE + AXIS * t + r * (np.outer(np.cos(a), X) + np.outer(np.sin(a), Y))


def cylinder(r, t0, t1, along, origin="input", halves=2, turn=2 * math.pi, centre=BASE):
    faces = []
    for k in range(halves):
        a = (k + 0.5) * turn / halves
        f = dict(type="cylinder", area=turn / halves * r * (t1 - t0), axis=list(AXIS), center=list(centre), radius=r,
                 convex=True, origin=origin,
                 point=list(centre + AXIS * ((t0 + t1) / 2 - (centre - BASE) @ AXIS) + r * (math.cos(a) * X + math.sin(a) * Y)))
        if along:
            f["along"] = [t0 - (centre - BASE) @ AXIS, t1 - (centre - BASE) @ AXIS]
        faces.append(f)
    return faces


def cone(t_apex, semi, t0, t1, along, flip=False):
    """The cone about the axis with its apex at t_apex, between t0 and t1 (both on one side of the apex); `flip`: the
    surface's own axis is stored the other way round, as OCCT may."""
    faces = []
    r0, r1 = abs(t0 - t_apex) * math.tan(semi), abs(t1 - t_apex) * math.tan(semi)
    apex = BASE + AXIS * t_apex
    for k in range(2):
        a = (k + 0.5) * math.pi
        tm, rm = (t0 + t1) / 2, (r0 + r1) / 2
        f = dict(type="cone", area=0.5 * math.pi * abs(r1 ** 2 - r0 ** 2) / math.sin(semi),
                 axis=list(-AXIS if flip else AXIS), center=list(apex), apex=list(apex), radius=0.0, semi_angle=semi,
                 convex=True, origin="input", point=list(BASE + AXIS * tm + rm * (math.cos(a) * X + math.sin(a) * Y)))
        if along:
            f["along"] = sorted(((-1.0 if flip else 1.0) * (t - t_apex)) for t in (t0, t1))
        faces.append(f)
    return faces


def bolt_hole(along):
    """Counterbore R 6 x 7, shank R 3.5 x 50, cone 3.5 -> 2.5 (semi angle 60 deg), core R 2.5 x 9, open at both ends;
    the shoulder between counterbore and shank is a real plane."""
    semi = math.radians(60.0)
    h = (3.5 - 2.5) / math.tan(semi)
    t_apex = 57.0 + 3.5 / math.tan(semi)
    faces = cylinder(6.0, 0.0, 7.0, along) + cylinder(3.5, 7.0, 57.0, along)
    faces += cone(t_apex, semi, 57.0, 57.0 + h, along, flip=True) + cylinder(2.5, 57.0 + h, 66.0 + h, along)
    faces.append(dict(type="plane", area=math.pi * (36.0 - 12.25), normal=list(AXIS), point=list(BASE + AXIS * 7.0 + 4.7 * X),
                      origin="input"))
    faces.append(dict(type="cylinder", area=120.0, axis=[0.0, 0.0, 1.0], center=[0.0, 0.0, 0.0], radius=190.0,
                      convex=True, origin="model", point=list(BASE)))
    V = np.concatenate([ring(0.0, 6.0), ring(7.0, 6.0), ring(7.0, 3.5), ring(57.0, 3.5), ring(57.0 + h, 2.5),
                        ring(66.0 + h, 2.5)])
    volume = math.pi * (36.0 * 7.0 + 12.25 * 50.0 + h * (12.25 + 3.5 * 2.5 + 6.25) / 3.0 + 6.25 * 9.0)
    return faces, V, volume, h


@pytest.mark.parametrize("along", [True, False])
def test_bolt_hole_steps(along):
    faces, V, volume, h = bolt_hole(along)
    stacks = hs.hole_stacks(faces, V)
    assert len(stacks) == 1
    s = stacks[0]
    assert [q["kind"] for q in s["segments"]] == ["cyl", "cyl", "cone", "cyl"]
    sign = 1.0 if s["axis"] @ AXIS > 0 else -1.0
    reach = sorted(sorted(sign * (q[k] + (s["base"] - BASE) @ s["axis"]) for k in ("t0", "t1")) for q in s["segments"])
    assert np.allclose(reach, [[0.0, 7.0], [7.0, 57.0], [57.0, 57.0 + h], [57.0 + h, 66.0 + h]], atol=1e-6)
    assert sorted(round(max(q["r0"], q["r1"]), 6) for q in s["segments"]) == [2.5, 3.5, 3.5, 6.0]
    assert not s["closed_lo"] and not s["closed_hi"]
    assert s["volume"] == pytest.approx(volume, rel=1e-9)
    assert s["faces"] == list(range(8))                       # the model's face and the shoulder are no steps
    assert "R 6 x 7" in hs.stack_label(s) and "cone R" in hs.stack_label(s)


def test_tool_volume_and_open_ends():
    faces, V, volume, h = bolt_hole(True)
    stacks = hs.hole_stacks(faces, V)
    exact = hs.stacks_tool(stacks, 0.0)
    assert len(hs.stack_parts(stacks[0])) == 4
    rng = np.random.default_rng(0)
    lo, hi = V.min(axis=0) - 8.0, V.max(axis=0) + 8.0
    pts = rng.uniform(lo, hi, size=(400000, 3))
    box = float(np.prod(hi - lo))
    inside = np.asarray(primitive_to_sp(exact).sdf(pts)) <= 0
    assert inside.mean() * box == pytest.approx(volume, rel=0.03)
    over = np.asarray(primitive_to_sp(hs.stacks_tool(stacks, 2.0)).sdf(pts)) <= 0
    extra = math.pi * 2.0 * (36.0 + 6.25)                     # both ends open: 2 mm more of R 6 and of R 2.5
    assert (over & ~inside).mean() * box == pytest.approx(extra, rel=0.1)
    assert not (inside & ~over).any()


def test_blind_hole_tip_is_closed_and_a_corner_is_no_step():
    semi = math.radians(60.0)
    tip = 2.459 / math.tan(semi)
    faces = cylinder(2.459, 0.0, 15.0, False) + cone(15.0 + tip, semi, 15.0, 15.0 + tip, False, flip=True)
    corner_at = BASE + 30.0 * X
    faces += cylinder(5.0, -4.0, 0.0, False, halves=1, turn=math.pi / 2, centre=corner_at)   # a pocket's rounded corner
    faces.append(dict(type="plane", area=570.0, normal=list(AXIS), point=list(BASE + 9.0 * X), origin="input"))  # its floor
    a = np.linspace(0.0, math.pi / 2, 7)
    corner = corner_at + np.concatenate([AXIS * t + 5.0 * (np.outer(np.cos(a), X) + np.outer(np.sin(a), Y)) for t in (-4.0, 0.0)])
    V = np.concatenate([ring(0.0, 2.459), ring(15.0, 2.459), [BASE + AXIS * (15.0 + tip)], corner])
    stacks = hs.hole_stacks(faces, V)
    assert len(stacks) == 1
    s = stacks[0]
    assert [q["kind"] for q in s["segments"]] == ["cyl", "cone"]
    sign = 1.0 if s["axis"] @ AXIS > 0 else -1.0
    closed_at_tip, closed_at_entry = (s["closed_hi"], s["closed_lo"]) if sign > 0 else (s["closed_lo"], s["closed_hi"])
    assert closed_at_tip and not closed_at_entry              # the floor is larger than the hole: no cap
    assert "tip 120 deg" in hs.stack_label(s)
    assert hs.hole_stacks([f for f in faces if f["type"] == "plane"], V) == []


def test_flat_bottom_is_closed():
    faces = cylinder(4.0, 0.0, 10.0, True)
    faces.append(dict(type="plane", area=math.pi * 16.0, normal=list(AXIS), point=list(BASE + AXIS * 10.0), origin="input"))
    s = hs.hole_stacks(faces, None)[0]
    sign = 1.0 if s["axis"] @ AXIS > 0 else -1.0
    assert (s["closed_hi"] if sign > 0 else s["closed_lo"]) and not (s["closed_lo"] if sign > 0 else s["closed_hi"])
    part, = hs.stack_parts(s, over=1.0)
    assert part.dims_mm["half_length"] == pytest.approx(5.5)   # 10 mm and 1 mm past the open end only


def test_a_tip_that_ends_a_hair_past_its_apex_is_kept():
    semi = math.radians(60.0)
    tip = 6.9175 / math.tan(semi)
    faces = cylinder(6.9175, 0.0, 50.0, False) + cone(50.0 + tip, semi, 50.0, 50.0 + tip, False, flip=True)
    for f in faces:
        if f["type"] == "cone":
            f["area"] *= 1.0 + 1e-9                           # the area's rounding: the end lies 2e-9 past the apex
    V = np.concatenate([ring(0.0, 6.9175), ring(50.0, 6.9175), [BASE + AXIS * (50.0 + tip)]])
    s, = hs.hole_stacks(faces, V)
    assert [q["kind"] for q in s["segments"]] == ["cyl", "cone"]
    assert s["closed_lo"] != s["closed_hi"]


def test_an_open_end_runs_by_the_stacks_own_length():
    faces = cylinder(2.0, 0.0, 10.0, True)
    s, = hs.hole_stacks(faces, None)
    part, = hs.stack_parts(s, over=None)
    assert part.dims_mm["half_length"] == pytest.approx(0.5 * (10.0 + 2 * 0.05 * 10.0))
