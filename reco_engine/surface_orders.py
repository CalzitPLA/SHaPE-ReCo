#!/usr/bin/env python3
"""
Two measures of a model against its input that do not count volume (the user's rules of 2026-09-29):

  containment   "100% surface on or inside the solid": the share of the INPUT's surface that lies on or inside the
                model. A rough model that is to be cut must contain the whole part: 1.0 is the condition, not a grade.
  coverage      "maximal exact surface coverage": how much surface of model and input is the SAME surface - not
                only in position, but in every order a smooth surface has ("for all Sobolev spaces of the primitive"
                is the SEARCH, not this measure: the user, 2026-09-29, "sobolev space is the space of solutions of
                each function" - per primitive type the space of all its instances, in which the instance with the
                most exact coverage is looked for: reco_engine/subpart_cover.py):
                  order 0   the point lies on the other surface                      (position,   H^0)
                  order 1   ... and both surfaces have the same normal there          (tangent,    H^1)
                  order 2   ... and the same curvature in every tangent direction     (curvature,  H^2)
                A cylinder of radius 10 that touches a bore of radius 10.5 along a line agrees there to order 1 and
                fails order 2; a box face on a plane of the part agrees to every order. Order 2 is the exact one:
                two pieces of surface that agree to order 2 on an area are the same plane, cylinder, cone, sphere
                or torus. Each order is a share of area, in both directions: of the model's surface (`model_on_input`:
                what the model has that the part has too) and of the input's surface (`input_on_model`: how much of
                the part's surface the model reproduces).

Both sides are ANALYTIC: the input's faces as FreeCAD gives them (reco_field.exact_check.exact_faces: type, axis,
centre, radius, ...; a free-form face has no analytic form and reaches order 0 only, by the mesh), the model's
surfaces from its primitives. Normals are the gradient of the surface's distance function, the curvature its
shape operator, taken by central differences of the normal along the tangent plane - one rule for every surface
type, and the same code on both sides. The position (order 0) is judged on the input's MESH, which knows where a face
is trimmed; the analytic face is chosen among those whose box holds the point.

Limits (review R-02, 2026-09-30; documented, not changed): the orders do not see which side the material is on - the
normals are compared up to sign (|n_a . n_b|) and the `convex` flag exact_faces writes is not read - so a boss and a
bore of the same axis and radius reach order 2 against each other, and a model that FILLS a bore counts as reproducing
the bore's wall. The cone's distance uses |h|, so both nappes match. The default tolerances are shares of the
diagonal (containment and coverage 1e-3, the curvature step 1e-4): on a large slender part (the frame, diagonal
8.4 m) that is 8.4 mm, the same scale difference_grade.py had to correct for its skin depth.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

ANGLE = math.radians(2.0)           # order 1: the normals agree within this
CURVATURE = 0.01                    # order 2: the shape operators differ by at most this share of the larger one ...
FLAT = 1e-3                         # ... or both are flat: curvature x the part's diagonal below this
PRIMITIVE_TYPES = ("plane", "cylinder", "cone", "sphere", "torus")


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------------------------------------ analytic surfaces

def distance(s: Dict, P: np.ndarray) -> np.ndarray:
    """The signed distance of the points to the (untrimmed) surface `s`."""
    P = np.asarray(P, float)
    k = s["type"]
    if k == "plane":
        return (P - s["point"]) @ s["normal"]
    q = P - s["center"]
    h = q @ s["axis"] if "axis" in s else None
    if k == "sphere":
        return np.linalg.norm(q, axis=1) - s["radius"]
    rho = np.linalg.norm(q - np.outer(h, s["axis"]), axis=1)
    if k == "cylinder":
        return rho - s["radius"]
    if k == "cone":                                           # centre = the apex, the nappe along +axis
        return rho * math.cos(s["semi_angle"]) - np.abs(h) * math.sin(s["semi_angle"])
    if k == "torus":
        return np.hypot(rho - s["major_radius"], h) - s["minor_radius"]
    raise ValueError(f"no analytic surface of type {k!r}")


def normal(s: Dict, P: np.ndarray, h: float) -> np.ndarray:
    """The unit gradient of the distance function at the points, by central differences of step h."""
    g = np.stack([(distance(s, P + h * e) - distance(s, P - h * e)) / (2 * h) for e in np.eye(3)], axis=1)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-300)


def shape_operator(s: Dict, P: np.ndarray, h: float) -> np.ndarray:
    """(n, 3, 3): the derivative of the unit normal along the tangent plane at the points - the curvature in every
    tangent direction, zero along the normal. A plane gives 0, a cylinder of radius r gives 1 / r across its axis."""
    N = normal(s, P, h)
    H = np.zeros((len(P), 3, 3))
    for j, e in enumerate(np.eye(3)):
        H[:, :, j] = (normal(s, P + h * e, h) - normal(s, P - h * e, h)) / (2 * h)
    T = np.eye(3)[None] - N[:, :, None] * N[:, None, :]                  # onto the tangent plane, from both sides
    H = T @ H @ T
    return 0.5 * (H + np.transpose(H, (0, 2, 1)))


def input_surfaces(faces: Sequence[Dict]) -> List[Dict]:
    """The analytic surfaces of the input's faces (exact_faces), each with its face's box; free-form faces are left
    out (they have no analytic form here)."""
    out = []
    for i, f in enumerate(faces):
        k = f.get("type")
        if k not in PRIMITIVE_TYPES:
            continue
        s = dict(type=k, face=i, area=float(f.get("area") or 0.0))
        if k == "plane":
            s.update(point=np.asarray(f["point"], float), normal=_unit(f["normal"]))
        else:
            s["center"] = np.asarray(f.get("apex") if k == "cone" and f.get("apex") is not None else f["center"], float)
            if k != "sphere":
                s["axis"] = _unit(f["axis"])
            if k in ("cylinder", "sphere"):
                s["radius"] = float(f["radius"])
            if k == "cone":
                s["semi_angle"] = abs(float(f["semi_angle"]))
            if k == "torus":
                s.update(major_radius=float(f["major_radius"]), minor_radius=float(f["minor_radius"]))
        if f.get("bbox"):
            s["lo"], s["hi"] = (np.asarray(b, float) for b in f["bbox"])
        out.append(s)
    return out


def primitive_surfaces(p) -> List[Dict]:
    """The analytic surfaces of a primitive or boolean tree (result_schema.Primitive): a box's six planes, a
    cylinder's wall and two ends, a cone's wall and ends, a sphere, a torus; a sector's two planes through the axis."""
    from reco_engine.result_to_program import rotation_matrix
    if p.kind == "composite":
        return [s for q in p.parts for s in primitive_surfaces(q)]
    if p.kind in ("halfspace", "unknown"):
        return []
    R, c, d = rotation_matrix(p), np.asarray(p.center_mm, float), p.dims_mm
    out = []

    def plane(point, n):
        out.append(dict(type="plane", point=np.asarray(point, float), normal=_unit(n)))
    if p.kind == "box":
        for i, key in enumerate(("x_half", "y_half", "z_half")):
            for sgn in (-1.0, 1.0):
                plane(c + sgn * d[key] * R[:, i], sgn * R[:, i])
        return out
    a = R[:, 2]
    if p.kind == "sphere":
        return [dict(type="sphere", center=c, radius=float(d["radius"]))]
    if p.kind == "torus":
        out.append(dict(type="torus", center=c, axis=a, major_radius=float(d["major_radius"]),
                        minor_radius=float(d["minor_radius"])))
    elif p.kind == "cylinder":
        out.append(dict(type="cylinder", center=c, axis=a, radius=float(d["radius"])))
        for sgn in (-1.0, 1.0):
            plane(c + sgn * d["half_length"] * a, sgn * a)
    elif p.kind == "cone":
        r1, r2, hl = float(d["radius1"]), float(d["radius2"]), float(d["half_length"])
        if abs(r1 - r2) < 1e-12:
            out.append(dict(type="cylinder", center=c, axis=a, radius=r1))
        else:
            z_apex = -hl + 2 * hl * r1 / (r1 - r2)                       # where the wall's radius is 0
            out.append(dict(type="cone", center=c + z_apex * a, axis=a, semi_angle=abs(math.atan2(r1 - r2, 2 * hl))))
        for sgn, r in ((-1.0, r1), (1.0, r2)):
            if r > 0:
                plane(c + sgn * hl * a, sgn * a)
    if p.u:                                                              # a sector: the two planes through the axis
        for ang, sgn in ((p.u[0], -1.0), (p.u[1], 1.0)):
            t = -math.sin(ang) * R[:, 0] + math.cos(ang) * R[:, 1]       # the direction of growing angle there
            plane(c, sgn * t)
    return out


# ------------------------------------------------------------------------------------------------ matching

def nearest(surfaces: Sequence[Dict], P: np.ndarray, tol: float, boxes: bool):
    """(index of the surface each point lies on within `tol`, -1 for none; its distance). With `boxes` only a surface
    whose face's box holds the point (by `tol`) counts."""
    best, dist = np.full(len(P), -1), np.full(len(P), np.inf)
    for i, s in enumerate(surfaces):
        d = np.abs(distance(s, P))
        ok = d <= tol
        if boxes and "lo" in s:
            ok &= np.all((P >= s["lo"] - tol) & (P <= s["hi"] + tol), axis=1)
        take = ok & (d < dist)
        best[take], dist[take] = i, d[take]
    return best, dist


def orders(own: Sequence[Dict], other: Sequence[Dict], P: np.ndarray, on: np.ndarray, tol: float, diag: float,
           own_boxes: bool, other_boxes: bool) -> np.ndarray:
    """(n,) the order each point reaches: -1 not on the other surface, 0 position, 1 normal too, 2 curvature too.
    `own`: the surfaces the points were taken from; `other`: the ones they are compared with; `on`: which points lie
    on the other side's real (trimmed) surface."""
    P = np.asarray(P, float)
    reach = np.where(on, 0, -1)
    if not on.any() or not own or not other:
        return reach
    idx = np.flatnonzero(on)
    a, _ = nearest(own, P[idx], tol, own_boxes)
    b, _ = nearest(other, P[idx], tol, other_boxes)
    h = 1e-4 * diag
    for ia in np.unique(a[a >= 0]):
        for ib in np.unique(b[(a == ia) & (b >= 0)]):
            sel = (a == ia) & (b == ib)
            Q = P[idx[sel]]
            na, nb = normal(own[ia], Q, h), normal(other[ib], Q, h)
            dot = np.einsum("ij,ij->i", na, nb)
            first = np.abs(dot) >= math.cos(ANGLE)
            Sa, Sb = shape_operator(own[ia], Q, h), shape_operator(other[ib], Q, h)
            Sb = Sb * np.sign(dot)[:, None, None]                        # the same surface seen from the other side
            diff = np.linalg.norm(Sa - Sb, axis=(1, 2))
            size = np.maximum(np.linalg.norm(Sa, axis=(1, 2)), np.linalg.norm(Sb, axis=(1, 2)))
            second = first & ((diff <= CURVATURE * size) | (size * diag <= FLAT))
            reach[idx[sel]] = np.where(second, 2, np.where(first, 1, 0))
    return reach


def _mesh_samples(V: np.ndarray, F: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(seed)
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    pick = rng.choice(len(F), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    s = np.sqrt(u)
    return tri[pick, 0] * (1 - s)[:, None] + tri[pick, 1] * (s * (1 - v))[:, None] + tri[pick, 2] * (s * v)[:, None]


def containment(model, V: np.ndarray, F: np.ndarray, n: int = 20000, tol: Optional[float] = None, seed: int = 0) -> Dict:
    """The share of the input's surface on or inside the model (an SDF tree, src.sdf_primitives): `inside` - 1.0 is
    "100% surface on or inside the solid" - and how far the farthest point lies outside (`worst`, the file's units).
    `tol` default: 1e-3 of the input's diagonal (its mesh's chords lie that close to its surface)."""
    V, F = np.asarray(V, float), np.asarray(F)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    tol = 1e-3 * diag if tol is None else tol
    P = _mesh_samples(V, F, n, seed)
    d = np.asarray(model.sdf(P))
    return dict(inside=float((d <= tol).mean()), worst=float(max(d.max(), 0.0)), points=int(n), tol=tol,
                outside=P[d > tol])


def coverage(primitive, faces: Sequence[Dict], V: np.ndarray, F: np.ndarray, n: int = 8000, tol: Optional[float] = None,
             seed: int = 0) -> Dict:
    """The surface coverage of a model (a Primitive or boolean tree, or a list of them fused) against the input (its
    exact faces and its mesh V, F), per order and in both directions:
      model_on_input[k]   share of the MODEL's surface that agrees with the input's to order k
      input_on_model[k]   share of the INPUT's surface that the model reproduces to order k
      exact               input_on_model[2]: the part's surface the model has exactly
      free_form           share of the input's surface on free-form faces (it cannot pass order 0)
    `tol` default: 1e-3 of the input's diagonal."""
    import igl
    from reco_engine.feature_stage import tool_surface_samples
    from reco_engine.result_to_program import primitive_to_sp
    from reco_engine.result_schema import Primitive
    V, F = np.ascontiguousarray(V, np.float64), np.ascontiguousarray(F, np.int64)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    tol = 1e-3 * diag if tol is None else tol
    if isinstance(primitive, (list, tuple)):
        tree = primitive[0]
        for q in primitive[1:]:
            tree = Primitive(kind="composite", center_mm=tuple(tree.center_mm), dims_mm={}, parts=[tree, q], combine="fuse")
        primitive = tree
    own, other = primitive_surfaces(primitive), input_surfaces(faces)
    model = primitive_to_sp(primitive)
    out = dict(tol=tol, surfaces_model=len(own), surfaces_input=len(other))
    # the model's surface against the input
    P, W = tool_surface_samples(primitive, n, seed)
    if len(P):
        d2 = igl.point_mesh_squared_distance(np.ascontiguousarray(P, np.float64), V, F)[0]
        on = np.sqrt(np.maximum(d2, 0.0)) <= tol
        r = orders(own, other, P, on, tol, diag, False, True)
        out["model_on_input"] = [float(W[r >= k].sum() / W.sum()) for k in (0, 1, 2)]
        out["model_inside_input"] = None
    else:
        out["model_on_input"] = [0.0, 0.0, 0.0]
    # the input's surface against the model
    Q = _mesh_samples(V, F, n, seed + 1)
    on = np.abs(np.asarray(model.sdf(Q))) <= tol
    r = orders(other, own, Q, on, tol, diag, True, False)
    out["input_on_model"] = [float((r >= k).mean()) for k in (0, 1, 2)]
    out["exact"] = out["input_on_model"][2]
    total = sum(float(f.get("area") or 0.0) for f in faces) or 1.0
    out["free_form"] = sum(float(f.get("area") or 0.0) for f in faces if f.get("type") not in PRIMITIVE_TYPES) / total
    return out
