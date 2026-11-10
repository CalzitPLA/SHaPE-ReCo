#!/usr/bin/env python3
"""
reco_engine.result_schema -> reco_field program: the bridge that lets the cascade / add-cut loop be built and graded in
FreeCAD by the SAME builder the vector-field route uses (reco_field/export.py -> reco_field/freecad_builder.py), instead of
a third copy of the Part conventions.

A schema `Primitive` is a body in its own local frame (centred; the axis of a body of revolution is local z) rotated by
`quaternion` (qx, qy, qz, qw) and moved to `center_mm`. The SDF primitives of dataset_gen/src/sdf_primitives.py stand where
FreeCAD puts them (a cylinder and a cone on z = 0), so the translation is shifted by half the height along the axis - the same
bookkeeping reco_field/fit.py's builders do, so a FitResult converted to the schema and back is the same solid.

    model = program_model(result)                  # sp.Primitive tree of the accepted operations, first one the base
    program = program_dict(result)                 # reco_field.export.program_to_dict of it (JSON-ready)
    exported = build_in_freecad(model, out_dir)    # reco_field.export
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "dataset_gen") not in sys.path:
    sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import sdf_primitives as sp  # noqa: E402

from reco_engine.result_schema import Primitive, ReconstructionResult  # noqa: E402

HALF_PI = np.pi / 2


def rotation_matrix(p: Primitive) -> np.ndarray:
    return Rotation.from_quat(np.asarray(p.quaternion, float)).as_matrix()


def quaternion_of(R: np.ndarray) -> tuple:
    """(qx, qy, qz, qw) of a rotation matrix whose columns are the body's local axes in the world, as python floats (the
    schema's JSON round trip compares equal only for plain floats)."""
    R = np.asarray(R, float)
    if np.linalg.det(R) < 0:                                        # a reflected frame: flip the last axis, same solid for
        R = R.copy(); R[:, 2] *= -1.0                               # every centred / axis-symmetric body used here
    return tuple(float(x) for x in Rotation.from_matrix(R).as_quat())


def primitive_to_sp(p: Primitive) -> sp.Primitive:
    """The placed SDF solid of one schema primitive. Raises ValueError for "unknown" (there is nothing to build).
    "halfspace": {x : n.x <= n.c} with n = the local +z axis (a cut removes the +n side); "composite": its parts joined
    by `combine`."""
    bad = p.defect()
    if bad is not None:                       # a clear refusal here, not a crash deep in the SDF or in freecadcmd
        raise ValueError(f"cannot build the primitive: {bad}")
    if p.kind == "composite":
        a, b = primitive_to_sp(p.parts[0]), primitive_to_sp(p.parts[1])
        return {"common": sp.Intersection, "cut": sp.Subtraction, "fuse": sp.Union}[p.combine](a, b)
    R = rotation_matrix(p)
    if p.kind == "halfspace":
        n = R[:, 2]
        return sp.HalfSpace(n, float(n @ np.asarray(p.center_mm, float)))
    c = np.asarray(p.center_mm, float)
    d = p.dims_mm
    u = tuple(p.u) if p.u else sp.FULL
    if p.kind == "box":
        return sp.Placed(sp.Box([d["x_half"], d["y_half"], d["z_half"]]), rotation=R, translation=c)
    if p.kind == "sphere":
        if p.v:                                                     # Part builds plane-limited latitudes only
            return sp.Placed(sp.Sphere(d["radius"], u, tuple(p.v), latitude="plane"), rotation=R, translation=c)
        return sp.Placed(sp.Sphere(d["radius"], u), rotation=R, translation=c)
    if p.kind == "cylinder":
        h = d["half_length"]
        return sp.Placed(sp.Cylinder(d["radius"], 2.0 * h, u), rotation=R, translation=c - R @ np.array([0.0, 0.0, h]))
    if p.kind == "cone":
        h = d["half_length"]
        return sp.Placed(sp.Cone(d["radius1"], d["radius2"], 2.0 * h, u), rotation=R, translation=c - R @ np.array([0.0, 0.0, h]))
    if p.kind == "torus":
        v = tuple(p.v) if p.v else sp.FULL
        return sp.Placed(sp.Torus(d["major_radius"], d["minor_radius"], u, v), rotation=R, translation=c)
    raise ValueError(f"cannot build a primitive of kind {p.kind!r}")


def transform_primitive(p: Primitive, R: np.ndarray, t=(0.0, 0.0, 0.0)) -> Primitive:
    """The primitive moved rigidly by x -> R x + t (its centre, its frame; a composite's parts too). Sectors (u, v) and
    dims are local and stay. Used to map a candidate found on a rotated copy of the input back to the input's frame."""
    import copy
    R = np.asarray(R, float)
    q = copy.copy(p)
    q.center_mm = tuple(float(x) for x in R @ np.asarray(p.center_mm, float) + np.asarray(t, float))
    q.quaternion = quaternion_of(R @ rotation_matrix(p))
    q.parts = [transform_primitive(part, R, t) for part in (p.parts or [])]
    q.alternatives = []
    return q


def primitive_from_fit(fit, **fields) -> Primitive:
    """A schema Primitive from a reco_field.fit.FitResult (its frame, centre and half vector; see FitResult's field comments)."""
    R = np.eye(3) if fit.frame is None else np.asarray(fit.frame, float)
    half = np.asarray(fit.half, float)
    if fit.kind == "box":
        dims = dict(x_half=half[0], y_half=half[1], z_half=half[2])
    elif fit.kind == "sphere":
        dims = dict(radius=half[0])
    elif fit.kind == "cylinder":
        dims = dict(radius=half[0], half_length=half[2])
    elif fit.kind == "cone":
        dims = dict(radius1=half[0], radius2=half[1], half_length=half[2])
    elif fit.kind == "torus":
        dims = dict(major_radius=half[0], minor_radius=half[1])
    else:
        raise ValueError(f"no schema kind for a {fit.kind} fit")
    return Primitive(kind=fit.kind, quaternion=quaternion_of(R), center_mm=tuple(float(x) for x in fit.centre),
                     dims_mm={k: float(abs(v)) for k, v in dims.items()},
                     u=tuple(float(x) for x in fit.u) if fit.u else None,
                     v=tuple(float(x) for x in fit.v) if fit.v else None, **fields)


def program_model(result: ReconstructionResult, upto: Optional[int] = None) -> Optional[sp.Primitive]:
    """The SDF tree of the ACCEPTED operations (the first `upto` of them when given), folded left to right: the first is
    the base, then Union for "add", Subtraction for "cut" and Intersection for "intersect" - the order the loop
    decided them in."""
    model = None
    ops = [o for o in result.operations if o.accepted]
    for o in ops[:upto] if upto is not None else ops:
        prim = primitive_to_sp(o.primitive)
        if model is None:
            if o.op != "add":
                raise ValueError("the first operation of a program must be an add (the base)")
            model = prim
        elif o.op == "add":
            model = sp.Union(model, prim)
        elif o.op == "cut":
            model = sp.Subtraction(model, prim)
        elif o.op == "intersect":
            model = sp.Intersection(model, prim)
        else:
            raise ValueError(o.op)
    return model


def program_dict(result: ReconstructionResult) -> dict:
    from reco_field.export import program_to_dict
    return program_to_dict(program_model(result))
