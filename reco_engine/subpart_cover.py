#!/usr/bin/env python3
"""
The rough model as several sub-parts (the user's rule 2026-09-29: "for one part of the rough guess add, assume we add
more subparts before cutting rough"): instead of ONE bounding primitive that is then cut, the part is covered by a few
primitives that are added to each other - a stepped shaft by one cylinder per step, an angle by two boxes - and only
then the cuts begin. The cover is judged by the two measures of reco_engine/surface_orders.py:

  containment   every sub-part's primitive bounds its sub-part, so the cover contains the whole input by
                construction; it is measured all the same (1.0 is the condition)
  exact         the share of the input's surface that the cover's surfaces reproduce exactly (in position, normal
                and curvature): the cover with the most of it wins, a further sub-part must gain at least `min_gain`

"For all Sobolev spaces of the primitive" (the user, 2026-09-29: "sobolev space is the space of solutions of each
function"): every primitive TYPE is a function with a space of solutions - all boxes, all cylinders, all cones, all
spheres, all tori. The cover is searched in each of these spaces: per type the instance that holds the sub-part and
reproduces the most of the input's surface exactly, and then the best of the types. The spaces are not scanned
blindly: the input's own faces name the instances worth trying - a cylinder face its axis line, a cone face its apex
and angle, a sphere face its centre, a torus face its centre circle, the plane faces the box's frame.

How a cover is found, from the input's exact faces and its mesh:
  directions    the box's frame from the plane faces' normals, the axes from the faces of revolution (both by area),
                the centres, apexes and centre circles of the sphere, cone and torus faces; without any such face
                the principal axes of the mesh
  primitives    of a set of surface points, per type: the box in the frame; per axis line the cylinder that holds
                the points; per cone face the cone of its apex line and angle, moved along its axis until it holds
                them; per sphere face the sphere about its centre; per torus face the torus about its centre circle
  splits        a sub-part is split by a plane of the input (a plane face square to a direction, at its position):
                the splits are ranked by the volume the two halves' best primitives save, the best few are measured
                by their exact coverage
The search is greedy: the best single primitive, then the best split of one of its sub-parts, and so on, up to
`max_parts`, while a split gains `min_gain` of exact coverage or `min_saving` of the cover's volume.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from reco_engine import surface_orders as so
from reco_engine.result_schema import Primitive, StageSource
from reco_engine.result_to_program import primitive_to_sp, quaternion_of

COS = math.cos(math.radians(1.0))


def _src() -> List[StageSource]:
    return [StageSource(s, "subpart_cover", 1.0) for s in ("type", "rotation", "dimensions", "position")]


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _frame_z(z) -> np.ndarray:
    z = _unit(z)
    x = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def directions(faces: Sequence[Dict], V: np.ndarray) -> Dict:
    """dict(frame (3, 3): the box's axes as columns; axes [(direction, point on the line, area)]: the axis lines of the
    input's faces of revolution, the largest area first; planes [(normal, offset, area)]: its plane faces)."""
    normals, axes, planes, instances = [], [], [], []
    for f in faces:
        a = float(f.get("area") or 0.0)
        if f.get("type") == "sphere":
            instances.append(dict(kind="sphere", centre=np.asarray(f["center"], float), area=a))
        elif f.get("type") == "cone" and abs(float(f.get("semi_angle") or 0.0)) > 1e-6:
            instances.append(dict(kind="cone", axis=_unit(f["axis"]), apex=np.asarray(f.get("apex", f["center"]), float),
                                  semi_angle=abs(float(f["semi_angle"])), point=np.asarray(f["point"], float), area=a))
        elif f.get("type") == "torus":
            instances.append(dict(kind="torus", axis=_unit(f["axis"]), centre=np.asarray(f["center"], float),
                                  major=float(f["major_radius"]), area=a))
        if f.get("type") == "plane":
            n = _unit(f["normal"])
            planes.append((n, float(n @ np.asarray(f["point"], float)), a))
            for g in normals:
                if abs(g[0] @ n) >= COS:
                    g[1] += a
                    break
            else:
                normals.append([n, a])
        elif f.get("type") in ("cylinder", "cone", "torus") and "axis" in f:
            d, c = _unit(f["axis"]), np.asarray(f.get("center"), float)
            foot = c - d * (c @ d)
            for g in axes:
                if abs(g[0] @ d) >= COS and np.linalg.norm(foot - g[1]) <= 1e-6 * (1.0 + np.linalg.norm(foot)):
                    g[2] += a
                    break
            else:
                axes.append([d, foot, a])
    normals.sort(key=lambda g: -g[1])
    axes.sort(key=lambda g: -g[2])
    frame = None
    if normals:
        z = normals[0][0]
        second = next((g[0] for g in normals[1:] if abs(g[0] @ z) < 0.1), None)
        if second is None and axes:
            second = next((g[0] for g in axes if abs(g[0] @ z) < 0.1), None)
        if second is not None:
            x = _unit(second - z * (second @ z))
            frame = np.stack([x, np.cross(z, x), z], axis=1)
        else:
            frame = _frame_z(z)
    elif axes:
        frame = _frame_z(axes[0][0])
    else:
        c = np.asarray(V, float) - np.mean(V, axis=0)
        frame = np.linalg.svd(c, full_matrices=False)[2].T
        if np.linalg.det(frame) < 0:
            frame[:, 2] *= -1
    instances.sort(key=lambda g: -g["area"])
    return dict(frame=frame, axes=[(g[0], g[1], g[2]) for g in axes], planes=planes, instances=instances)


def box_of(P: np.ndarray, R: np.ndarray) -> Primitive:
    L = P @ R
    lo, hi = L.min(axis=0), L.max(axis=0)
    h = np.maximum(0.5 * (hi - lo), 1e-9)
    return Primitive(kind="box", quaternion=quaternion_of(R), center_mm=tuple(float(v) for v in R @ (0.5 * (lo + hi))),
                     dims_mm=dict(x_half=float(h[0]), y_half=float(h[1]), z_half=float(h[2])), confidence=1.0, sources=_src(),
                     label="sub-part box")


def cylinder_of(P: np.ndarray, axis: np.ndarray, foot: Optional[np.ndarray] = None) -> Primitive:
    """The cylinder about the axis line (direction `axis` through `foot`; None: through the middle of the points'
    extent across the axis) that holds the points."""
    a = _unit(axis)
    R = _frame_z(a)
    L = P @ R
    if foot is None:
        mid = 0.5 * (L[:, :2].min(axis=0) + L[:, :2].max(axis=0))
    else:
        mid = (np.asarray(foot, float) @ R)[:2]
    r = float(np.linalg.norm(L[:, :2] - mid, axis=1).max())
    z0, z1 = float(L[:, 2].min()), float(L[:, 2].max())
    c = R @ np.array([mid[0], mid[1], 0.5 * (z0 + z1)])
    return Primitive(kind="cylinder", quaternion=quaternion_of(R), center_mm=tuple(float(v) for v in c),
                     dims_mm=dict(radius=max(r, 1e-9), half_length=max(0.5 * (z1 - z0), 1e-9)), confidence=1.0,
                     sources=_src(), label="sub-part cylinder")


def sphere_of(P: np.ndarray, centre: np.ndarray) -> Primitive:
    r = float(np.linalg.norm(P - centre, axis=1).max())
    return Primitive(kind="sphere", center_mm=tuple(float(v) for v in centre), dims_mm=dict(radius=max(r, 1e-9)),
                     confidence=1.0, sources=_src(), label="sub-part sphere")


def cone_of(P: np.ndarray, axis: np.ndarray, apex: np.ndarray, semi_angle: float, point: np.ndarray) -> Optional[Primitive]:
    """The cone frustum with the face's axis line and angle that holds the points: the face's cone moved along its
    axis until the outermost point lies on it. None when the points reach over the apex (both nappes)."""
    a = _unit(axis)
    if (np.asarray(point, float) - apex) @ a < 0:                       # the nappe of the face along +a
        a = -a
    q = P - apex
    h = q @ a
    rho = np.linalg.norm(q - np.outer(h, a), axis=1)
    k = math.tan(semi_angle)
    shift = float((rho / k - h).max())                                   # the apex moves back by this much
    z0, z1 = float(h.min()), float(h.max())
    r0, r1 = (z0 + shift) * k, (z1 + shift) * k
    if r0 < -1e-9 or z1 - z0 <= 1e-9:
        return None
    R = _frame_z(a)
    c = apex + a * 0.5 * (z0 + z1)
    return Primitive(kind="cone", quaternion=quaternion_of(R), center_mm=tuple(float(v) for v in c),
                     dims_mm=dict(radius1=max(float(r0), 0.0), radius2=float(r1), half_length=0.5 * (z1 - z0)),
                     confidence=1.0, sources=_src(), label="sub-part cone")


def torus_of(P: np.ndarray, axis: np.ndarray, centre: np.ndarray, major: float,
             diagonal: Optional[float] = None) -> Optional[Primitive]:
    """The torus about a centre circle (axis, centre, major) that holds the points P, or None when the torus guard
    (reco_engine/torus_guard.py) rejects it: no ring any more (spindle, horn), a pin-hole ring (near-horn), or, with
    `diagonal` (the part's), far larger than the part."""
    from reco_engine.torus_guard import torus_defect
    a = _unit(axis)
    q = P - centre
    h = q @ a
    rho = np.linalg.norm(q - np.outer(h, a), axis=1)
    minor = float(np.hypot(rho - major, h).max())
    if torus_defect(major, max(minor, 1e-9), diagonal) is not None:      # not this type's solid
        return None
    return Primitive(kind="torus", quaternion=quaternion_of(_frame_z(a)), center_mm=tuple(float(v) for v in centre),
                     dims_mm=dict(major_radius=float(major), minor_radius=max(minor, 1e-9)), confidence=1.0,
                     sources=_src(), label="sub-part torus")


def volume(p: Primitive) -> float:
    d = p.dims_mm
    if p.kind == "box":
        return 8.0 * d["x_half"] * d["y_half"] * d["z_half"]
    if p.kind == "sphere":
        return 4.0 / 3.0 * math.pi * d["radius"] ** 3
    if p.kind == "torus":
        return 2.0 * math.pi ** 2 * d["major_radius"] * d["minor_radius"] ** 2
    if p.kind == "cone":
        r0, r1 = d["radius1"], d["radius2"]
        return math.pi * 2.0 * d["half_length"] * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0
    return math.pi * d["radius"] ** 2 * 2.0 * d["half_length"]


def candidates(P: np.ndarray, dirs: Dict, max_axes: int = 3, max_instances: int = 4) -> List[Primitive]:
    """One candidate or more from the space of every primitive type (see the module's text)."""
    out = [box_of(P, dirs["frame"])]
    for a, foot, _ in dirs["axes"][:max_axes]:
        out.append(cylinder_of(P, a, foot))
        out.append(cylinder_of(P, a, None))
    for g in dirs.get("instances", [])[:max_instances]:
        if g["kind"] == "sphere":
            out.append(sphere_of(P, g["centre"]))
        elif g["kind"] == "cone":
            out.append(cone_of(P, g["axis"], g["apex"], g["semi_angle"], g["point"]))
        else:
            out.append(torus_of(P, g["axis"], g["centre"], g["major"], dirs.get("diagonal")))
    return [c for c in out if c is not None]


def split_planes(P: np.ndarray, dirs: Dict, tol: float, min_side: float = 0.02) -> List[tuple]:
    """[(normal, offset)] the input's planes that are square to a direction of the cover and have points of the
    sub-part on both sides (at least `min_side` of them each), one per position."""
    ds = [dirs["frame"][:, i] for i in range(3)] + [a for a, _, _ in dirs["axes"][:3]]
    out = []
    for n, off, _ in dirs["planes"]:
        d = next((d for d in ds if abs(d @ n) >= COS), None)
        if d is None:
            continue
        s, o = (1.0, off) if d @ n > 0 else (-1.0, -off)
        if any(abs(q[0] @ d) >= COS and abs(q[1] - o) <= tol for q in out):
            continue
        t = P @ d - o
        if (t > tol).mean() >= min_side and (t < -tol).mean() >= min_side:
            out.append((d, o))
    return out


def _halves(Q: np.ndarray, d: np.ndarray, o: float, tol: float, dirs: Dict):
    """The two halves of a sub-part's points Q at the plane d . x = o, as point sets that END ON the plane: every
    half gets its own points' shadows on the plane (so its primitive reaches the plane and no gap of a sample's width
    stays), and the points ON the plane - a shoulder's face, the end of a step - go, moved exactly onto it, to ONE
    half: the one whose primitive grows less by them. Counted to both, the shoulder of a stepped shaft gave its
    radius to the thin step as well (2026-09-29)."""
    t = Q @ d - o
    below, above, on = Q[t < -tol], Q[t > tol], Q[np.abs(t) <= tol]
    on = on - np.outer(on @ d - o, d)
    below = np.vstack([below, below - np.outer(below @ d - o, d)])
    above = np.vstack([above, above - np.outer(above @ d - o, d)])
    if not len(on):
        return below, above

    def least(X):
        return min(volume(c) for c in candidates(X, dirs))
    with_below, with_above = np.vstack([below, on]), np.vstack([above, on])
    if least(with_below) - least(below) <= least(with_above) - least(above):
        return with_below, above
    return below, with_above


def measure(prims: Sequence[Primitive], faces, V, F, n: int, tol: Optional[float]) -> Dict:
    g = so.coverage(list(prims), faces, V, F, n=n, tol=tol)
    return dict(exact=g["exact"], input_on_model=g["input_on_model"], model_on_input=g["model_on_input"],
                volume=float(sum(volume(p) for p in prims)))


def cover(V: np.ndarray, F: np.ndarray, faces: Sequence[Dict], max_parts: int = 6, min_gain: float = 0.03,
          min_saving: float = 0.1, tries: int = 4, n: int = 4000, points: int = 30000, tol: Optional[float] = None,
          seed: int = 0, part_diagonal: Optional[float] = None) -> Dict:
    """The cover of the input by sub-parts: dict(primitives [Primitive], exact, input_on_model, model_on_input,
    containment, volume, history [dict(parts, exact, volume, split)]). See the module's text. `part_diagonal`: the
    yardstick of the torus guard (default: the diagonal of V - the part, or the feature piece being covered)."""
    V, F = np.asarray(V, float), np.asarray(F)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    tol = 1e-3 * diag if tol is None else tol
    dirs = directions(faces, V)
    dirs["diagonal"] = float(part_diagonal) if part_diagonal else diag
    P = np.vstack([V, so._mesh_samples(V, F, points, seed)])
    # the best single primitive
    parts = [P]
    graded = [(measure([c], faces, V, F, n, tol), c) for c in candidates(P, dirs)]
    m, first = max(graded, key=lambda t: (round(t[0]["exact"], 2), -t[0]["volume"]))
    prims = [first]
    history = [dict(parts=1, exact=m["exact"], volume=m["volume"], split=None, kind=first.kind)]
    while len(prims) < max_parts:
        options = []
        for i, Q in enumerate(parts):
            for d, o in split_planes(Q, dirs, tol):
                lo, hi = _halves(Q, d, o, tol, dirs)
                halves = [min(candidates(h, dirs), key=volume) for h in (lo, hi)]
                saved = volume(prims[i]) - sum(volume(h) for h in halves)
                options.append((saved, i, (d, o), (lo, hi), halves))
        options.sort(key=lambda t: -t[0])
        best = None
        for saved, i, plane, halves_idx, halves in options[:tries]:
            trial = prims[:i] + prims[i + 1:]
            # each half: the primitive with the most exact coverage among its candidates, the cover around it fixed
            chosen = []
            for h in halves_idx:
                cands = candidates(h, dirs)
                graded = [(measure(trial + chosen + [c], faces, V, F, n, tol), c) for c in cands]
                chosen.append(max(graded, key=lambda t: (round(t[0]["exact"], 2), -t[0]["volume"]))[1])
            mm = measure(trial + chosen, faces, V, F, n, tol)
            if best is None or (round(mm["exact"], 2), -mm["volume"]) > (round(best[0]["exact"], 2), -best[0]["volume"]):
                best = (mm, i, plane, halves_idx, chosen)
        if best is None:
            break
        mm, i, plane, halves_idx, chosen = best
        gain, saving = mm["exact"] - m["exact"], 1.0 - mm["volume"] / m["volume"]
        if gain < min_gain and saving < min_saving:
            break
        prims = prims[:i] + prims[i + 1:] + chosen
        parts = parts[:i] + parts[i + 1:] + list(halves_idx)
        m = mm
        history.append(dict(parts=len(prims), exact=m["exact"], volume=m["volume"], gain=gain, saving=saving,
                            split=dict(normal=[float(v) for v in plane[0]], offset=float(plane[1])),
                            kinds=[c.kind for c in chosen]))
    tree = primitive_to_sp(prims[0])
    from src import sdf_primitives as sp
    for q in prims[1:]:
        tree = sp.Union(tree, primitive_to_sp(q))
    c = so.containment(tree, V, F, tol=tol)
    final = measure(prims, faces, V, F, 2 * n, tol)
    for k, q in enumerate(prims):
        q.label = f"sub-part {k + 1} of {len(prims)}: {q.kind}"
    return dict(primitives=prims, containment=c["inside"], worst_outside=c["worst"], history=history, tol=tol, **final)
