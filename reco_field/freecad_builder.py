# -*- coding: utf-8 -*-
"""
Builds a FreeCAD solid from a program of SDF primitives (a JSON tree written by reco_field/export.py). Run INSIDE FreeCAD:

    PROGRAM_JSON=program.json STEP_OUT=out.step STL_OUT=out.stl RESULT_JSON=result.json freecadcmd reco_field/freecad_builder.py

No imports from this project: it needs only FreeCAD's Part (and MeshPart for the STL). The Part conventions it applies are the ones measured in
docs/freecad_part_shapes.tex:

  * our Box is centred, FreeCAD's has its corner at the origin;
  * a cylinder or cone stands on z = 0, a sphere and a torus are centred (as ours);
  * sweep u: FreeCAD sweeps from +x counter-clockwise by the angle; a sector (u0, u1) is built with the sweep u1 - u0 and turned by u0 about z;
  * sphere latitude: Part.makeSphere(angle1, angle2) is the ball between two PLANES (our latitude="plane"); the cone variant is not buildable
    directly and is rejected;
  * torus minor range v: Part::Torus measures the angle the other way round, its [Angle1, Angle2] is our v = (-Angle2, -Angle1); the
    function Part.makeTorus is only used with its minor angles at 0..360 (its own minor angles are unusable);
  * a half space is a very large box on the -normal side of its plane.
"""
import json
import math
import os

import FreeCAD as App
import Part

V = App.Vector
Z = V(0, 0, 1)
ORIGIN = V(0, 0, 0)
TWO_PI = 2.0 * math.pi
DOC = App.newDocument("reco_export")


def deg(x):
    return math.degrees(x)


def is_full(rng, full):
    return rng[0] <= full[0] + 1e-9 and rng[1] >= full[1] - 1e-9


def rotated_z(shape, u0):
    if abs(u0) > 1e-12:
        shape = shape.copy()
        shape.rotate(ORIGIN, Z, deg(u0))
    return shape


def sweep_of(u):
    """(sweep in degrees or None for a full turn, start angle in radians)."""
    if is_full(u, (0.0, TWO_PI)):
        return None, 0.0
    sweep = deg(u[1] - u[0])
    if sweep <= 1e-9:          # a zero-width sector would become `sweep or 360`, a FULL primitive: refuse it instead
        raise ValueError(f"empty sector: u = ({u[0]}, {u[1]}) has no angular width")
    return sweep, u[0]


def rigid(shape, R, t):
    m = App.Matrix(R[0][0], R[0][1], R[0][2], t[0], R[1][0], R[1][1], R[1][2], t[1], R[2][0], R[2][1], R[2][2], t[2], 0, 0, 0, 1)
    out = shape.copy()
    out.transformShape(m)
    return out


def snap(r):
    return 0.0 if r < 1e-4 else r


def build_torus(d):
    R, r = d["major"], d["minor"]
    sweep, u0 = sweep_of(d["u"])
    v = d["v"]
    if is_full(v, (0.0, TWO_PI)) or v[1] - v[0] >= TWO_PI - 1e-9:
        shape = Part.makeTorus(R, r, ORIGIN, Z, 0, 360, sweep if sweep is not None else 360)
    else:
        # Part::Torus feature: [Angle1, Angle2] = (-v1, -v0), in [-180, 180]; a range across 180 is two pies
        a1 = -deg(v[1])
        width = deg(v[1] - v[0])
        a1 = (a1 + 180.0) % 360.0 - 180.0
        pieces = [(a1, a1 + width)] if a1 + width <= 180.0 + 1e-9 else [(a1, 180.0), (-180.0, a1 + width - 360.0)]
        shape = None
        for lo, hi in pieces:
            obj = DOC.addObject("Part::Torus", "t%d" % len(DOC.Objects))
            obj.Radius1, obj.Radius2, obj.Angle1, obj.Angle2, obj.Angle3 = R, r, lo, min(hi, 180.0), sweep if sweep is not None else 360
            DOC.recompute()
            piece = obj.Shape.copy()
            shape = piece if shape is None else shape.fuse(piece)
    return rotated_z(shape, u0)


def build(d):
    kind = d["type"]
    if kind == "Box":
        h = d["half"]
        return Part.makeBox(2 * h[0], 2 * h[1], 2 * h[2], V(-h[0], -h[1], -h[2]))
    if kind == "Sphere":
        sweep, u0 = sweep_of(d["u"])
        full_v = is_full(d["v"], (-math.pi / 2, math.pi / 2))
        if not full_v and d["latitude"] != "plane":
            raise ValueError("a sphere with a cone-limited latitude cannot be built with Part.makeSphere (only latitude='plane')")
        shape = Part.makeSphere(d["radius"], ORIGIN, Z, -90 if full_v else deg(d["v"][0]), 90 if full_v else deg(d["v"][1]), sweep or 360)
        if d["inner_radius"] > 0:
            shape = shape.cut(Part.makeSphere(d["inner_radius"]))
        return rotated_z(shape, u0)
    if kind == "Cylinder":
        sweep, u0 = sweep_of(d["u"])
        shape = Part.makeCylinder(d["radius"], d["height"], ORIGIN, Z, sweep or 360)
        if d["inner_radius"] > 0:
            shape = shape.cut(Part.makeCylinder(d["inner_radius"], d["height"] + 2.0, V(0, 0, -1.0)))
        return rotated_z(shape, u0)
    if kind == "Cone":
        sweep, u0 = sweep_of(d["u"])
        r1, r2 = snap(d["radius1"]), snap(d["radius2"])
        if abs(r1 - r2) <= 1e-9 * max(1.0, r1):                   # Part.makeCone refuses equal radii ("creation of cone
            return rotated_z(Part.makeCylinder(r1, d["height"], ORIGIN, Z, sweep or 360), u0)   # failed"): a cylinder
        return rotated_z(Part.makeCone(r1, r2, d["height"], ORIGIN, Z, sweep or 360), u0)
    if kind == "Torus":
        return build_torus(d)
    if kind == "HalfSpace":
        big = d["big"]
        n = V(*d["normal"])
        box = Part.makeBox(2 * big, 2 * big, big + d["offset"], V(-big, -big, -big))           # z <= offset
        return rigid_from_rotation(box, App.Rotation(Z, n))
    if kind == "Placed":
        return rigid(build(d["primitive"]), d["rotation"], d["translation"])
    a, b = build(d["a"]), build(d["b"])
    if kind in ("Union", "Intersection", "Subtraction"):
        return boolean(a, b, kind)
    raise ValueError("unknown primitive " + kind)


FUZZY = 1e-4                        # the tolerance of the fuzzy boolean (mm), as reco_field/exact_check.py's
BOOLEAN_POINTS = 256                # points that judge between a plain and a fuzzy boolean that disagree
BOOLEAN_LOG = []                    # one entry per boolean whose plain and fuzzy results disagreed


def boolean(a, b, kind):
    """a | b, a & b, a - b, checked. OCCT's plain booleans fail silently on tangent operands, and consistently: on
    the liner's pocket tools (2026-09-28) box - (square - cylinder) came back as the box, valid, and the
    intersection of the two as empty, so no volume balance shows it (2407 mm^3 of the input cut by a tool that is
    exact). Hence both the plain and the fuzzy (FUZZY) boolean are computed whenever the operands' boxes overlap;
    where their volumes differ by more than 1e-7 of the larger operand, seeded points in the overlap of the boxes,
    classified against a and b themselves, judge: the result that places more of them right is taken (the plain one
    on a tie), and BOOLEAN_LOG gets an entry."""
    op = {"Union": "fuse", "Intersection": "common", "Subtraction": "cut"}[kind]
    plain = getattr(a, op)(b)
    ba, bb = a.BoundBox, b.BoundBox
    if not ba.intersect(bb):
        return plain
    try:
        fuzzy = getattr(a, op)([b], FUZZY)
    except Exception:
        return plain
    if plain.isValid() and abs(plain.Volume - fuzzy.Volume) <= 1e-7 * max(a.Volume, b.Volume):
        return plain
    import random
    rng = random.Random(0)
    lo = [max(ba.XMin, bb.XMin), max(ba.YMin, bb.YMin), max(ba.ZMin, bb.ZMin)]
    hi = [min(ba.XMax, bb.XMax), min(ba.YMax, bb.YMax), min(ba.ZMax, bb.ZMax)]
    wrong = dict(plain=0, fuzzy=0)
    for _ in range(BOOLEAN_POINTS):
        p = V(*[l + rng.random() * (h - l) for l, h in zip(lo, hi)])
        in_a, in_b = a.isInside(p, 1e-6, True), b.isInside(p, 1e-6, True)
        want = (in_a or in_b) if kind == "Union" else (in_a and in_b) if kind == "Intersection" else (in_a and not in_b)
        for name, r in (("plain", plain), ("fuzzy", fuzzy)):
            wrong[name] += r.isInside(p, 1e-6, True) != want
    taken = "fuzzy" if fuzzy.isValid() and (wrong["fuzzy"] < wrong["plain"] or not plain.isValid()) else "plain"
    BOOLEAN_LOG.append(dict(kind=kind, plain=plain.Volume, fuzzy=fuzzy.Volume, wrong=wrong, taken=taken))
    return fuzzy if taken == "fuzzy" else plain


def rigid_from_rotation(shape, rotation):
    out = shape.copy()
    out.transformShape(App.Placement(ORIGIN, rotation).toMatrix())
    return out


def _export_one(shape, step_path, stl_path, deflection):
    shape.exportStep(step_path)        # Part.export([TopoShape]) wrote a STEP with no geometry (1640 bytes, re-reads as null)
    import MeshPart
    mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=deflection, AngularDeflection=0.1, Relative=False)
    mesh.write(stl_path)
    bb = shape.BoundBox
    return dict(step=step_path, stl=stl_path, volume=shape.Volume, valid=bool(shape.isValid()), n_faces=len(shape.Faces),
               bbox=[[bb.XMin, bb.YMin, bb.ZMin], [bb.XMax, bb.YMax, bb.ZMax]])


def main_build():
    program = json.load(open(os.environ["PROGRAM_JSON"]))
    whole = build(program).removeSplitter()
    solids = sorted(whole.Solids, key=lambda s: s.Volume, reverse=True)
    deflection = float(os.environ.get("DEFLECTION", "0.02"))
    if not solids:                     # e.g. a cut that removes everything: an empty compound is not a "valid" result
        raise RuntimeError("the program built no solid (empty result of the booleans)")
    biggest = solids[0]
    out = _export_one(biggest, os.environ["STEP_OUT"], os.environ["STL_OUT"], deflection)
    result = dict(volume=out["volume"], valid=out["valid"], n_solids=len(solids), n_faces=out["n_faces"], bbox=out["bbox"])
    split_dir = os.environ.get("SPLIT_OUT_DIR")                 # opt-in: every solid, not just the biggest (unset: no change)
    if split_dir and len(solids) > 1:
        prefix = os.environ.get("SPLIT_PREFIX", "solid")
        result["solids"] = [out] + [_export_one(s, os.path.join(split_dir, f"{prefix}_{i}.step"),
                                                 os.path.join(split_dir, f"{prefix}_{i}.stl"), deflection)
                                    for i, s in enumerate(solids[1:], start=1)]
    json.dump(result, open(os.environ["RESULT_JSON"], "w"))


# ------------------------------------------------------------------------------------------------ groups
# The user's rule 2026-09-29: "get the freecad rebuild faster by just loading and grouping primitives and leaving out
# the final boolean". GROUPS_JSON names a job dict(operations=[dict(op, label, program)], fcstd, tools): every
# operation's tool becomes one object of the document, in the group of its operation ("adds", "cuts", ...), in the
# order of the program; the operations are NOT folded into one solid. tools "solid": a composite tool is built with
# its own few booleans; "leaves": no boolean at all, the tool is the group of its primitives' shapes (a primitive that
# the tool subtracts is named "minus"); "primitives": as "leaves", but every primitive is a PARAMETRIC Part object
# (Part::Box, Part::Cylinder, Part::Cone, ...) with its placement - the document consists of primitives only.

def leaves(d, minus=False):
    """[(shape, minus)] of a program tree without any boolean."""
    kind = d["type"]
    if kind == "Placed":
        return [(rigid(s, d["rotation"], d["translation"]), m) for s, m in leaves(d["primitive"], minus)]
    if kind in ("Union", "Intersection", "Subtraction"):
        return leaves(d["a"], minus) + leaves(d["b"], (not minus) if kind == "Subtraction" else minus)
    return [(build(d), minus)]


def placement_of(R, t):
    return App.Placement(App.Matrix(R[0][0], R[0][1], R[0][2], t[0], R[1][0], R[1][1], R[1][2], t[1],
                                    R[2][0], R[2][1], R[2][2], t[2], 0, 0, 0, 1))


def placed_leaves(d, P=None, minus=False):
    """[(leaf dict, its placement in the world, minus)] of a program tree."""
    P = App.Placement() if P is None else P
    kind = d["type"]
    if kind == "Placed":
        return placed_leaves(d["primitive"], P.multiply(placement_of(d["rotation"], d["translation"])), minus)
    if kind in ("Union", "Intersection", "Subtraction"):
        return placed_leaves(d["a"], P, minus) + placed_leaves(d["b"], P, (not minus) if kind == "Subtraction" else minus)
    return [(d, P, minus)]


def primitive_object(name, d, P):
    """The leaf as a PARAMETRIC Part primitive (Part::Box, Part::Cylinder, Part::Cone, Part::Sphere, Part::Torus) with
    its placement: dimensions and pose can be read and edited in FreeCAD. None for what has no such object (a hollow
    body, a sphere or torus segment, a half space) - the caller stores the shape instead."""
    kind = d["type"]
    local = App.Placement()
    if kind == "Box":
        h = d["half"]
        o = DOC.addObject("Part::Box", name)
        o.Length, o.Width, o.Height = 2 * h[0], 2 * h[1], 2 * h[2]
        local = App.Placement(V(-h[0], -h[1], -h[2]), App.Rotation())
    elif kind == "Cylinder" and d["inner_radius"] <= 0:
        sweep, u0 = sweep_of(d["u"])
        o = DOC.addObject("Part::Cylinder", name)
        o.Radius, o.Height, o.Angle = d["radius"], d["height"], sweep or 360
        local = App.Placement(ORIGIN, App.Rotation(Z, deg(u0)))
    elif kind == "Cone":
        sweep, u0 = sweep_of(d["u"])
        o = DOC.addObject("Part::Cone", name)
        o.Radius1, o.Radius2, o.Height, o.Angle = snap(d["radius1"]), snap(d["radius2"]), d["height"], sweep or 360
        local = App.Placement(ORIGIN, App.Rotation(Z, deg(u0)))
    elif kind == "Sphere" and d["inner_radius"] <= 0 and is_full(d["u"], (0.0, TWO_PI)) \
            and is_full(d["v"], (-math.pi / 2, math.pi / 2)):
        o = DOC.addObject("Part::Sphere", name)
        o.Radius = d["radius"]
    elif kind == "Torus" and is_full(d["u"], (0.0, TWO_PI)) and is_full(d["v"], (0.0, TWO_PI)):
        o = DOC.addObject("Part::Torus", name)
        o.Radius1, o.Radius2 = d["major"], d["minor"]
    else:
        return None
    o.Placement = P.multiply(local)
    o.recompute()
    want = build(d)
    want.transformShape(P.toMatrix())
    size = max(want.BoundBox.DiagonalLength, 1e-9)
    if abs(o.Shape.Volume - want.Volume) > 1e-6 * max(want.Volume, 1e-9) or \
            o.Shape.CenterOfMass.distanceToPoint(want.CenterOfMass) > 1e-6 * size:
        DOC.removeObject(o.Name)                                     # not the solid the program means: its shape instead
        return None
    return o


def main_groups():
    import time
    job = json.load(open(os.environ["GROUPS_JSON"]))
    t0 = time.time()
    groups, rows = {}, []
    for n, o in enumerate(job["operations"]):
        t1 = time.time()
        name = {"add": "adds", "cut": "cuts"}.get(o["op"], str(o["op"]) + "s")
        if name not in groups:
            groups[name] = DOC.addObject("App::DocumentObjectGroup", name)
        label = "%03d %s %s" % (n, o["op"], o.get("label") or o["program"]["type"])
        row = dict(index=n, op=o["op"], label=label)
        try:
            if job.get("tools") == "primitives":
                parts = placed_leaves(o["program"])
                holder = groups[name]
                if len(parts) > 1:
                    holder = DOC.addObject("App::DocumentObjectGroup", "tool%03d" % n)
                    holder.Label = label
                    groups[name].addObject(holder)
                parametric = 0
                for k, (d, P, minus) in enumerate(parts):
                    f = primitive_object("tool%03d_%02d" % (n, k), d, P)
                    if f is None:
                        f = DOC.addObject("Part::Feature", "tool%03d_%02d" % (n, k))
                        shape = build(d)
                        shape.transformShape(P.toMatrix())
                        f.Shape = shape
                    else:
                        parametric += 1
                    f.Label = label if len(parts) == 1 else "%03d.%02d %s %s" % (n, k, "minus" if minus else "plus", d["type"].lower())
                    holder.addObject(f)
                row.update(parts=len(parts), parametric=parametric, seconds=time.time() - t1)
                rows.append(row)
                continue
            if job.get("tools") == "leaves":
                parts = leaves(o["program"])
                if len(parts) == 1:
                    shape = parts[0][0]
                else:
                    sub = DOC.addObject("App::DocumentObjectGroup", "tool%03d" % n)
                    sub.Label = label
                    groups[name].addObject(sub)
                    for k, (s, minus) in enumerate(parts):
                        f = DOC.addObject("Part::Feature", "tool%03d_%02d" % (n, k))
                        f.Label = "%03d.%02d %s" % (n, k, "minus" if minus else "plus")
                        f.Shape = s
                        sub.addObject(f)
                    row.update(parts=len(parts), seconds=time.time() - t1)
                    rows.append(row)
                    continue
            else:
                shape = build(o["program"])
            f = DOC.addObject("Part::Feature", "tool%03d" % n)
            f.Label = label
            f.Shape = shape
            groups[name].addObject(f)
            row.update(parts=1, volume=float(shape.Volume), valid=bool(shape.isValid()), seconds=time.time() - t1)
        except Exception as exc:                                     # one tool failing leaves the others
            row.update(error=str(exc)[-200:], seconds=time.time() - t1)
        rows.append(row)
    DOC.recompute()
    DOC.saveAs(job["fcstd"])
    json.dump(dict(fcstd=job["fcstd"], operations=rows, groups={k: len(g.Group) for k, g in groups.items()},
                   seconds=time.time() - t0), open(os.environ["RESULT_JSON"], "w"))


def main():
    if os.environ.get("GROUPS_JSON"):
        main_groups()
    else:
        main_build()


# The LAST statement of this file must stay this plain call: reco_field/exact_check.py takes the builder's functions by
# cutting the source at the last "main()" (its _BUILDER_PRELUDE). 2026-09-29: an if / else around the call left a
# dangling "else:" there - the feature checks and the features document failed with a SyntaxError for half a day.
main()
