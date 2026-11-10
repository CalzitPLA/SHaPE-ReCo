"""Census of the rotational faces of a STEP part, grouped by axis line (freecadcmd).

    CENSUS_STEP=part.step CENSUS_OUT=census.json freecadcmd scripts/step_feature_census.py

Every cylindrical, conical and toroidal face is assigned to its axis line; one axis with its stack of faces is one
candidate feature of rotation: a drilled hole (cylinder + cone point), a stepped hole (coaxial cylinders), a pocket
corner (a 90 degree cylinder sector), a turned contour. Equal stacks along the same direction are counted together.
A face is "in" (concave, hole-like) when its outward normal points towards the axis, "out" otherwise. The length of a
face along the axis is taken from its vertices, so it is approximate where a face ends on a curved surface.
"""
import collections
import json
import math
import os

import FreeCAD as App
import Part


def canonical(axis):
    """The axis direction with its first non-zero component positive."""
    axis = App.Vector(axis)
    axis.normalize()
    for c in (axis.x, axis.y, axis.z):
        if abs(c) > 1e-6:
            return axis * -1 if c < 0 else axis
    return axis


def axis_key(axis, center):
    axis = canonical(axis)
    foot = center - axis * center.dot(axis)  # the point of the axis line closest to the origin
    return tuple(round(v, 3) for v in axis) + tuple(round(v, 1) for v in foot)


def is_concave(face, axis, center):
    u0, u1, v0, v1 = face.ParameterRange
    u, v = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    p, n = face.valueAt(u, v), face.normalAt(u, v)
    on_axis = center + axis * (p - center).dot(axis)
    return (on_axis - p).dot(n) > 0


def census(path):
    shape = Part.Shape()
    shape.read(path)
    solid = shape.Solids[0] if shape.Solids else shape
    types = collections.Counter()
    groups = collections.defaultdict(list)
    for face in solid.Faces:
        surface = face.Surface
        kind = type(surface).__name__
        types[kind] += 1
        if kind == "Cylinder":
            tag = ["cylinder", round(surface.Radius, 4)]
        elif kind == "Cone":
            tag = ["cone", round(math.degrees(surface.SemiAngle), 2)]
        elif kind == "Toroid":
            tag = ["torus", round(surface.MajorRadius, 3), round(surface.MinorRadius, 3)]
        else:
            continue
        axis, center = surface.Axis, surface.Center
        along = canonical(axis)
        heights = [v.Point.dot(along) for v in face.Vertexes] or [center.dot(along)]
        u0, u1, _, _ = face.ParameterRange
        groups[axis_key(axis, center)].append(
            dict(tag=tag, concave=is_concave(face, axis, center), sweep=math.degrees(u1 - u0),
                 lo=round(min(heights), 2), hi=round(max(heights), 2)))

    signatures = collections.Counter()
    for key, faces in groups.items():
        merged = {}  # faces of one surface between the same heights are one ring, split by the modeller
        for f in sorted(faces, key=lambda f: (f["lo"], f["hi"])):
            k = (json.dumps(f["tag"]), f["concave"], f["lo"], f["hi"])
            merged[k] = merged.get(k, 0.0) + f["sweep"]
        stack = tuple((tag, "in" if concave else "out", round(hi - lo, 2), round(sweep))
                      for (tag, concave, lo, hi), sweep in merged.items())
        signatures[(key[:3], stack)] += 1

    box = solid.BoundBox
    return dict(
        path=path, faces=len(solid.Faces), bbox=[box.XLength, box.YLength, box.ZLength], volume=solid.Volume,
        surface_types=dict(types), axes=len(groups),
        signatures=[dict(direction=list(direction), count=count,
                         stack=[dict(surface=json.loads(tag), side=side, length=length, sweep_deg=sweep)
                                for tag, side, length, sweep in stack])
                    for (direction, stack), count in sorted(signatures.items(), key=lambda item: -item[1])])


result = census(os.environ["CENSUS_STEP"])
with open(os.environ["CENSUS_OUT"], "w") as handle:
    json.dump(result, handle, indent=1)
