"""Primitive.defect (reco_engine/result_schema.py): the parameter sets that crashed 104 ABC runs on 2026-09-30 are
refused before any SDF, IoU or FreeCAD build - by the schema, by the cascade's admit, by the loop's judge and by
primitive_to_sp."""

import math

import numpy as np
import pytest

from reco_engine.result_schema import Primitive
from reco_engine import result_to_program as r2p


def prim(kind, **dims):
    return Primitive(kind=kind, dims_mm=dims)


GOOD = [prim("box", x_half=1, y_half=2, z_half=3), prim("sphere", radius=2), prim("cylinder", radius=1, half_length=4),
        prim("cone", radius1=0.0, radius2=2, half_length=1), prim("torus", major_radius=5, minor_radius=1),
        Primitive(kind="cylinder", dims_mm=dict(radius=1, half_length=1), u=(0.0, math.pi)),
        Primitive(kind="torus", dims_mm=dict(major_radius=5, minor_radius=1), u=(0.0, 2 * math.pi), v=(-1.0, 1.0)),
        Primitive(kind="halfspace")]

BAD = [
    # the recorded ABC crash 00009690: a torus v range the fit left inverted -> "sector needs u1 > u0"
    (Primitive(kind="torus", dims_mm=dict(major_radius=5, minor_radius=1), v=(4.68592354760375, 4.632641448837973)), "inverted"),
    (Primitive(kind="cylinder", dims_mm=dict(radius=1, half_length=1), u=(1.0, 1.0)), "empty"),
    (Primitive(kind="cylinder", dims_mm=dict(radius=1, half_length=1), u=(0.0, 7.0)), "wider than a turn"),
    (prim("box", x_half=1, y_half=0.0, z_half=3), "too small"),                              # "height of box too small"
    (prim("box", x_half=1, y_half=float("nan"), z_half=3), "not finite"),                   # "Bnd_Box is void"
    (prim("sphere", radius=-1), "too small"),
    (prim("cylinder", radius=1), "missing"),
    (prim("cone", radius1=0, radius2=0, half_length=1), "both radii zero"),
    (prim("cone", radius1=-1, radius2=1, half_length=1), "negative"),
    (Primitive(kind="box", dims_mm=dict(x_half=1, y_half=1, z_half=1), quaternion=(0, 0, 0, 0)), "not a rotation"),  # zero-norm axis
    (Primitive(kind="box", dims_mm=dict(x_half=1, y_half=1, z_half=1), center_mm=(0, float("inf"), 0)), "centre"),
    (Primitive(kind="unknown"), "unknown"),
    (Primitive(kind="composite", parts=[prim("box", x_half=1, y_half=1, z_half=1)], combine="cut"), "two parts"),
    (Primitive(kind="composite", parts=[prim("box", x_half=1, y_half=1, z_half=1), prim("sphere", radius=0)], combine="cut"),
     "composite part"),
]


def test_good_primitives_have_no_defect_and_build():
    for p in GOOD:
        assert p.defect() is None, p
        sdf = r2p.primitive_to_sp(p)
        assert np.isfinite(sdf.sdf(np.zeros((4, 3)))).all()


@pytest.mark.parametrize("p,text", BAD)
def test_bad_primitives_are_named_and_refused_before_any_build(p, text):
    d = p.defect()
    assert d is not None and text in d, (d, text)
    with pytest.raises(ValueError, match="cannot build the primitive"):
        r2p.primitive_to_sp(p)


def test_loop_judge_and_cascade_admit_reject_the_recorded_crash():
    from reco_engine import residual_loop as rl
    bad = BAD[0][0]
    class S:                                           # a stub state: the judge must answer before touching any grid
        s_model = None

        def size(self, *a):
            return 0
    out = rl._evaluate(S(), bad, "cut", 0.9)
    assert out["ok"] is False and "invalid primitive" in out["reason"] and "inverted" in out["reason"]

    from reco_engine import cascade
    ctx = cascade._Context.__new__(cascade._Context)
    ctx.guarded = []
    assert ctx.admit(bad) is None and ctx.guarded and "invalid primitive" in ctx.guarded[0]
    assert ctx.admit(GOOD[0]) is GOOD[0]
