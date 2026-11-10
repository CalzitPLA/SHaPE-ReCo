# -*- coding: utf-8 -*-
"""
Reference data for dataset_gen/src/sdf_primitives.py from FreeCAD's own Part module. Run INSIDE FreeCAD:

    freecadcmd scripts/sdf_reference_freecad.py            # writes tests/data/sdf_reference.json

For each shape (built exactly as the Part API documents; degrees for its angles) it samples points - uniformly in the
bounding box (+15 %) and close to the surface - and records what FreeCAD says: `inside` (Shape.isInside, faces count as inside)
and `dist` (the distance to the surface: Compound(Faces).distToShape). tests/test_sdf_primitives.py compares them with the numpy
SDFs, so the tests need no FreeCAD. Deterministic (fixed seed). Units: the parameters' own.
"""
import json
import math
import os
import random
import sys

import FreeCAD as App
import Part

V = App.Vector
OUT = os.environ.get("SDF_REFERENCE_OUT", os.path.join(os.getcwd(), "tests", "data", "sdf_reference.json"))
N_UNIFORM, N_NEAR = 700, 350


def wedge():
    return Part.makeWedge(0, 0, 0, 0.5, 0.5, 3, 2, 2, 1.5, 2.5)


def feature(kind, **props):
    doc = App.ActiveDocument or App.newDocument("ref")
    obj = doc.addObject(kind, "o%d" % len(doc.Objects))
    for k, v in props.items():
        setattr(obj, k, v)
    doc.recompute()
    return obj.Shape


def tube(sector):
    outer = Part.makeCylinder(1.0, 2.0, V(0, 0, 0), V(0, 0, 1), sector)
    inner = Part.makeCylinder(0.5, 2.0, V(0, 0, 0), V(0, 0, 1), 360)
    return outer.cut(inner)


SHAPES = {
    "box":                lambda: Part.makeBox(2.0, 3.0, 4.0),
    "sphere":             lambda: Part.makeSphere(1.5),
    "sphere_u0_180":      lambda: Part.makeSphere(1.5, V(0, 0, 0), V(0, 0, 1), -90, 90, 180),
    "sphere_lat_plane":   lambda: Part.makeSphere(1.5, V(0, 0, 0), V(0, 0, 1), -30, 60, 360),
    "sphere_lat_plane_u": lambda: Part.makeSphere(1.5, V(0, 0, 0), V(0, 0, 1), 20, 70, 250),
    "cylinder":           lambda: Part.makeCylinder(1.0, 2.0),
    "cylinder_u200":      lambda: Part.makeCylinder(1.0, 2.0, V(0, 0, 0), V(0, 0, 1), 200),
    "tube_full":          lambda: tube(360),
    "tube_u120":          lambda: tube(120),
    "cone":               lambda: Part.makeCone(1.0, 0.4, 2.0),
    "cone_u90":           lambda: Part.makeCone(1.0, 0.4, 2.0, V(0, 0, 0), V(0, 0, 1), 90),
    "cone_point":         lambda: Part.makeCone(1.0, 0.0, 2.0),
    "torus":              lambda: Part.makeTorus(3.0, 1.0),
    "torus_u90":          lambda: Part.makeTorus(3.0, 1.0, V(0, 0, 0), V(0, 0, 1), 0, 360, 90),
    "torus_u250":         lambda: Part.makeTorus(3.0, 1.0, V(0, 0, 0), V(0, 0, 1), 0, 360, 250),
    "wedge":              wedge,
    "ellipsoid":          lambda: feature("Part::Ellipsoid", Radius1=0.5, Radius2=1.2, Radius3=0.8),
    "prism6":             lambda: feature("Part::Prism", Polygon=6, Circumradius=1.2, Height=2.0),
    "prism3":             lambda: feature("Part::Prism", Polygon=3, Circumradius=1.0, Height=1.5),
    # Part::Torus measures the minor-circle angle the other way round: its pie [Angle1, Angle2] is our v = (-Angle2, -Angle1)
    # (v: counter-clockwise from the outer equator towards +z). Measured against FreeCAD 1.0.0, see docs/freecad_part_shapes.md.
    "torus_v":            lambda: feature("Part::Torus", Radius1=3.0, Radius2=1.0, Angle1=-30, Angle2=100, Angle3=360),
    "torus_v_u":          lambda: feature("Part::Torus", Radius1=3.0, Radius2=1.0, Angle1=-60, Angle2=120, Angle3=150),
}


def build_reference(name, make, rng):
    shape = make()
    faces = Part.Compound(shape.Faces)
    bb = shape.BoundBox
    lo = [bb.XMin, bb.YMin, bb.ZMin]
    hi = [bb.XMax, bb.YMax, bb.ZMax]
    pad = [0.15 * (h - l) for l, h in zip(lo, hi)]
    pts, inside, dist = [], [], []

    def record(p):
        v = V(*p)
        pts.append([round(c, 6) for c in p])
        inside.append(bool(shape.isInside(v, 1e-7, True)))
        dist.append(round(faces.distToShape(Part.Vertex(v))[0], 6))

    uniform = [[rng.uniform(l - q, h + q) for l, h, q in zip(lo, hi, pad)] for _ in range(N_UNIFORM)]
    for p in uniform:
        record(p)
    for p in uniform[:N_NEAR]:                                   # points 0.03 from the surface, on the same side as p
        d, sol, _ = faces.distToShape(Part.Vertex(V(*p)))
        s = sol[0][0]
        v = [p[i] - s[i] for i in range(3)]
        n = math.sqrt(sum(c * c for c in v)) or 1.0
        record([s[i] + 0.03 * v[i] / n for i in range(3)])
    return dict(points=pts, inside=inside, dist=dist, volume=shape.Volume, valid=shape.isValid(),
                bbox=[lo, hi])


def main():
    rng = random.Random(1234)
    ref = {}
    for name, make in SHAPES.items():
        ref[name] = build_reference(name, make, rng)
        r = ref[name]
        print("%-20s volume=%9.4f valid=%s points=%d inside=%d" % (name, r["volume"], r["valid"], len(r["points"]), sum(r["inside"])))
        sys.stdout.flush()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(ref, f)
    print("wrote", OUT)


main()
