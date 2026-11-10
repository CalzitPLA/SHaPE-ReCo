#!/usr/bin/env python3
"""
The primitives inside a compound part, recognized from the part's own exact surfaces (the user, 2026-10-30: "how could
we get the method of recognizing subshapes - parts or whole primitives - in higher ordered shapes?").

Every B-rep face of a STEP part lies on an analytic surface that names a primitive: a cylinder by its axis and radius, a
cone by its apex and angle, a sphere, a torus; three orthogonal pairs of planes span a box. The start of the loop
(reco_engine/start_shape.py) used only the DOMINATING surfaces - the whole part on their material side - which in a
union no leaf has, so its cover fell back to the oriented box and its excess (1 good of 82 known unions). Here every
surface counts (Shapiro-Vossler: the natural halfspaces of all boundary surfaces; .claude/guru/literature/L01, L02):

    hypotheses   one per curved surface, over the surface's OWN extent (its triangles: axial range, angular coverage):
                 convex -> an add, concave -> a cut; "whole" (the full angle) and, when the surface covers only part of
                 the turn, "part" (the sector); boxes from three orthogonal plane pairs
                 a DOMINATING convex surface (the whole part on its material side) also gives a "cover": the whole
                 primitive over the part, a first operation or an intersection (A & B)
    verify       a cut holds >= CUT_PRECISION of its volume outside the input; an add >= ADD_PRECISION inside it or
                 inside a verified cut (A - B: A's volume in B's bore is explained); a cover holds >= COVER_HOLDS of
                 the input (the winding number of the input mesh, reco_engine/prune.input_inside)
    select       a greedy CSG on one set of sample points: the first operation the add or cover of the best IoU alone,
                 then every round the add, cut or intersection that raises the IoU most, until it gains < MIN_DELTA;
                 then reco_engine/prune.minimize (L03: a global cost instead of accepting greedily)

Measured on the gate's 37 STEP parts, version 1 (adds and cuts only; docs/data/leaf_recognition_study_gate.md): union
leaves recalled 0.54 / 0.32 / 0.19 (2-3 / 4-7 / 8+ leaves) against today's loop 0.00 / 0.03 / 0.00, with 1.5-4 parts
against 10-12; intersections and subtractions 0.00 (their leaves hold the whole part: covers, version 2).

    rec = recognize(step_path)          # dict(hypotheses, passed, program (ReconstructionResult), info)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reco_engine.result_schema import Operation, Primitive, ReconstructionResult, StageSource
from reco_engine.result_to_program import quaternion_of

ADD_PRECISION = 0.98        # an add: this share of its volume inside the input (or inside a verified cut)
CUT_PRECISION = 0.98        # a cut: this share of its volume outside the input
COVER_HOLDS = 0.98          # a cover: this share of the input's vertices inside it
MIN_DELTA = 0.003           # selection: a round adds at least this much IoU
SECTOR_GAP = math.radians(15.0)     # a surface leaving a larger gap around its axis is a sector ("part")
VERIFY_SAMPLES = 3000
SOURCE = "surfaces"


@dataclass
class Surface:
    kind: str                           # plane | cylinder | cone | sphere | torus
    params: Dict
    faces: List[int]
    area: float
    convex: Optional[bool] = None
    dominating: bool = False


@dataclass
class Hypothesis:
    primitive: Primitive
    role: str                           # add | cut | cover (the whole primitive over the part: first or intersect)
    surface: str                        # the surface kind it came from (box: "planes")
    faces: List[int]
    whole: bool                         # the full turn (or a box); False: a sector
    area: float
    precision: Optional[float] = None
    passed: Optional[bool] = None
    note: str = ""


# ------------------------------------------------------------------------------------------------ surfaces

def _frame(z) -> np.ndarray:
    z = np.asarray(z, float) / np.linalg.norm(z)
    u = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(z, u), z], axis=1)


def _unit_axis(a) -> np.ndarray:
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    return a if a[np.argmax(np.abs(a))] > 0 else -a


def surfaces_of(inv: Dict, tol: float) -> List[Surface]:
    """The faces of a STEP inventory (reco_engine.brep.step_faces) grouped by the surface they lie on, with their ids."""
    groups: Dict[tuple, Surface] = {}
    for f in inv["faces"]:
        t = f["type"]
        if t == "Plane":
            n = np.asarray(f["normal"], float)
            key = ("plane", tuple(np.round(n, 3)), round(float(n @ np.asarray(f["point"], float)) / tol))
            kind, params = "plane", dict(normal=n.tolist(), point=f["point"])
        elif t == "Cylinder":
            a = _unit_axis(f["axis"])
            c = np.asarray(f["center"], float)
            foot = c - (c @ a) * a
            key = ("cylinder", tuple(np.round(a, 3)), tuple(np.round(foot / tol)), round(f["radius"] / tol), bool(f.get("convex")))
            kind, params = "cylinder", dict(axis=a.tolist(), center=c.tolist(), radius=float(f["radius"]))
        elif t == "Cone":
            apex = np.asarray(f["apex"], float)
            a = np.asarray(f["center"], float) - apex
            a = a / np.linalg.norm(a)                               # from the apex towards the growing radius
            key = ("cone", tuple(np.round(apex / tol)), tuple(np.round(a, 3)), round(float(f["semi_angle"]), 4), bool(f.get("convex")))
            kind, params = "cone", dict(apex=apex.tolist(), axis=a.tolist(), semi_angle=float(f["semi_angle"]))
        elif t == "Sphere":
            c = np.asarray(f["center"], float)
            key = ("sphere", tuple(np.round(c / tol)), round(f["radius"] / tol), bool(f.get("convex")))
            kind, params = "sphere", dict(center=c.tolist(), radius=float(f["radius"]))
        elif t == "Toroid" and all(k in f for k in ("axis", "center", "major_radius", "minor_radius")):
            a = _unit_axis(f["axis"])
            c = np.asarray(f["center"], float)
            key = ("torus", tuple(np.round(c / tol)), tuple(np.round(a, 3)), round(f["major_radius"] / tol),
                   round(f["minor_radius"] / tol), bool(f.get("convex")))
            kind, params = "torus", dict(axis=a.tolist(), center=c.tolist(), major_radius=float(f["major_radius"]),
                                        minor_radius=float(f["minor_radius"]))
        else:
            continue
        g = groups.get(key)
        if g is None:
            groups[key] = Surface(kind, params, [int(f["id"])], float(f["area"]), f.get("convex"), bool(f.get("dominating")))
        else:
            g.faces.append(int(f["id"]))
            g.area += float(f["area"])
            g.dominating = g.dominating or bool(f.get("dominating"))
    return sorted(groups.values(), key=lambda s: -s.area)


def _face_vertices(inv: Dict, ids: Sequence[int]) -> np.ndarray:
    V = np.asarray(inv["vertices"], float)
    T = np.asarray(inv["triangles"])
    sel = np.isin(np.asarray(inv["face_id"]), list(ids))
    return V[np.unique(T[sel].ravel())] if sel.any() else np.zeros((0, 3))


def _turn(P: np.ndarray, centre, axis) -> Tuple[np.ndarray, float]:
    """(frame with x at the start of the covered turn and z = axis, the covered angle): the largest gap between the
    points' angles around the axis is the uncovered part (2 pi when it is smaller than SECTOR_GAP)."""
    R = _frame(axis)
    rel = P - np.asarray(centre, float)
    th = np.sort(np.mod(np.arctan2(rel @ R[:, 1], rel @ R[:, 0]), 2 * np.pi))
    if len(th) < 3:
        return R, 2 * np.pi
    gaps = np.diff(np.concatenate([th, th[:1] + 2 * np.pi]))
    k = int(np.argmax(gaps))
    if gaps[k] < SECTOR_GAP:
        return R, 2 * np.pi
    start = th[(k + 1) % len(th)]
    x = np.cos(start) * R[:, 0] + np.sin(start) * R[:, 1]
    z = R[:, 2]
    return np.stack([x, np.cross(z, x), z], axis=1), float(2 * np.pi - gaps[k])


def _prim(kind, R, centre, dims, u=None, label="") -> Primitive:
    return Primitive(kind=kind, quaternion=quaternion_of(R), center_mm=tuple(float(x) for x in centre),
                     dims_mm={k: float(v) for k, v in dims.items()}, u=None if u is None else (0.0, float(u)),
                     label=label, sources=[StageSource("type", f"analytic:{SOURCE}", 1.0)])


# ------------------------------------------------------------------------------------------------ hypotheses

def _cover(s: Surface, V: np.ndarray, pad: float) -> Optional[Hypothesis]:
    """The whole primitive of a dominating convex surface over the whole part (start_shape._surface_solid's idea)."""
    if not (s.dominating and s.convex):
        return None
    if s.kind == "cylinder":
        a, c, r = np.asarray(s.params["axis"], float), np.asarray(s.params["center"], float), s.params["radius"]
        t = (V - c) @ a
        t0, t1 = float(t.min()) - pad, float(t.max()) + pad
        p = _prim("cylinder", _frame(a), c + a * (t0 + t1) / 2, dict(radius=r, half_length=(t1 - t0) / 2), None,
                  f"surface cylinder r {r:.4g}, cover")
    elif s.kind == "cone":
        apex, a, al = np.asarray(s.params["apex"], float), np.asarray(s.params["axis"], float), s.params["semi_angle"]
        sv = (V - apex) @ a
        s0, s1 = max(float(sv.min()) - pad, 0.0), float(sv.max()) + pad
        p = _prim("cone", _frame(a), apex + a * (s0 + s1) / 2,
                  dict(radius1=s0 * math.tan(al), radius2=s1 * math.tan(al), half_length=(s1 - s0) / 2), None,
                  f"surface cone {math.degrees(al):.1f} deg, cover")
    elif s.kind == "sphere":
        p = _prim("sphere", np.eye(3), s.params["center"], dict(radius=s.params["radius"]), None,
                  f"surface sphere r {s.params['radius']:.4g}, cover")
    elif s.kind == "torus":
        p = _prim("torus", _frame(s.params["axis"]), s.params["center"], dict(major_radius=s.params["major_radius"],
                  minor_radius=s.params["minor_radius"]), None, f"surface torus R {s.params['major_radius']:.4g}, cover")
    else:
        return None
    return Hypothesis(p, "cover", s.kind, s.faces, True, s.area)


def _curved(s: Surface, inv: Dict, pad: float, planes: List[Surface], tol: float) -> List[Hypothesis]:
    P = _face_vertices(inv, s.faces)
    if len(P) < 3:
        return []
    role = "add" if s.convex else "cut"
    grow = pad if role == "cut" else 0.0                           # a cut reaches past the faces it bores through
    out = []

    def both(kind, R, centre, dims, span, label):
        out.append(Hypothesis(_prim(kind, R, centre, dims, None, f"{label}, whole"), role, s.kind, s.faces, True, s.area))
        if span < 2 * np.pi - 1e-6:
            out.append(Hypothesis(_prim(kind, R, centre, dims, span, f"{label}, sector {math.degrees(span):.0f} deg"),
                                  role, s.kind, s.faces, False, s.area))

    if s.kind == "cylinder":
        a, c, r = np.asarray(s.params["axis"], float), np.asarray(s.params["center"], float), s.params["radius"]
        t = (P - c) @ a
        t0, t1 = float(t.min()) - grow, float(t.max()) + grow
        R, span = _turn(P, c, a)
        both("cylinder", R, c + a * (t0 + t1) / 2, dict(radius=r, half_length=(t1 - t0) / 2), span,
             f"surface cylinder r {r:.4g}")
    elif s.kind == "cone":
        apex, a, al = np.asarray(s.params["apex"], float), np.asarray(s.params["axis"], float), s.params["semi_angle"]
        sv = (P - apex) @ a
        s0, s1 = max(float(sv.min()) - grow, 0.0), float(sv.max()) + grow
        R, span = _turn(P, apex, a)
        both("cone", R, apex + a * (s0 + s1) / 2,
             dict(radius1=s0 * math.tan(al), radius2=s1 * math.tan(al), half_length=(s1 - s0) / 2), span,
             f"surface cone {math.degrees(al):.1f} deg")
    elif s.kind == "sphere":
        c, r = np.asarray(s.params["center"], float), s.params["radius"]
        through = [p for p in planes if abs(np.asarray(p.params["normal"]) @ (c - np.asarray(p.params["point"]))) < tol]
        axis = None
        for i in range(len(through)):
            for j in range(i + 1, len(through)):
                x = np.cross(through[i].params["normal"], through[j].params["normal"])
                if np.linalg.norm(x) > 1e-3:
                    axis = x / np.linalg.norm(x)
                    break
            if axis is not None:
                break
        if axis is None:
            out.append(Hypothesis(_prim("sphere", np.eye(3), c, dict(radius=r), None, f"surface sphere r {r:.4g}, whole"),
                                  role, s.kind, s.faces, True, s.area))
        else:
            R, span = _turn(P, c, axis)
            both("sphere", R, c, dict(radius=r), span, f"surface sphere r {r:.4g}")
    elif s.kind == "torus":
        a, c = np.asarray(s.params["axis"], float), np.asarray(s.params["center"], float)
        R, span = _turn(P, c, a)
        both("torus", R, c, dict(major_radius=s.params["major_radius"], minor_radius=s.params["minor_radius"]), span,
             f"surface torus R {s.params['major_radius']:.4g}")
    return out


def _boxes(planes: List[Surface], inv: Dict, tol: float, limit: int = 400) -> List[Hypothesis]:
    """Boxes spanned by three mutually orthogonal pairs of opposite planes whose faces overlap across the pair."""
    info = []
    for p in planes:
        n = np.asarray(p.params["normal"], float)
        P = _face_vertices(inv, p.faces)
        if len(P) < 3:
            continue
        info.append((p, n, float(n @ np.asarray(p.params["point"], float)), P))
    pairs = []
    for i in range(len(info)):
        for j in range(i + 1, len(info)):
            (pi, ni, di, Pi), (pj, nj, dj, Pj) = info[i], info[j]
            if ni @ nj > -0.999:
                continue
            lo, hi = -dj, di                                        # material between: -dj <= ni.x <= di
            if hi - lo <= tol:
                continue
            pairs.append((ni, lo, hi, pi.faces + pj.faces, pi.area + pj.area, np.vstack([Pi, Pj])))
    out = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if abs(pairs[i][0] @ pairs[j][0]) > 1e-3:
                continue
            for k in range(j + 1, len(pairs)):
                if abs(pairs[i][0] @ pairs[k][0]) > 1e-3 or abs(pairs[j][0] @ pairs[k][0]) > 1e-3:
                    continue
                trio = (pairs[i], pairs[j], pairs[k])
                axes = np.stack([q[0] for q in trio], axis=1)
                if np.linalg.det(axes) < 0:
                    axes[:, 2] *= -1
                    trio = (trio[0], trio[1], (-trio[2][0], -trio[2][2], -trio[2][1]) + trio[2][3:])
                mids = np.array([(q[1] + q[2]) / 2 for q in trio])
                halves = np.array([(q[2] - q[1]) / 2 for q in trio])
                centre = axes @ mids
                pts = np.vstack([q[5] for q in trio])                # every face of the box lies within the box
                loc = (pts - centre) @ axes
                if (np.abs(loc) > halves + 2 * tol).any():
                    continue
                out.append(Hypothesis(_prim("box", axes, centre, dict(x_half=halves[0], y_half=halves[1], z_half=halves[2]),
                                            None, "surface box"), "add", "planes",
                                      sorted(set(f for q in trio for f in q[3])), True, float(sum(q[4] for q in trio))))
                if len(out) >= limit:
                    return out
    return out


def hypotheses(inv: Dict, tol_rel: float = 1e-3) -> Tuple[List[Hypothesis], List[Surface]]:
    diag = float(inv.get("diag") or np.linalg.norm(np.ptp(np.asarray(inv["vertices"], float), axis=0)))
    tol = tol_rel * diag
    surfaces = surfaces_of(inv, tol)
    planes = [s for s in surfaces if s.kind == "plane"]
    V = np.asarray(inv["vertices"], float)
    out: List[Hypothesis] = []
    for s in surfaces:
        if s.kind != "plane":
            out += _curved(s, inv, 0.005 * diag, planes, tol)
            c = _cover(s, V, 0.005 * diag)
            if c is not None:
                out.append(c)
    out += _boxes(planes, inv, tol)
    return out, surfaces


# ------------------------------------------------------------------------------------------------ verify, select

def verify(hyps: List[Hypothesis], mesh, n: int = VERIFY_SAMPLES, seed: int = 0) -> List[Hypothesis]:
    """Cuts first (their share outside the input), then adds (their share inside the input or inside a verified cut),
    then covers (the share of the input's vertices they hold); `passed` by the bounds."""
    from reco_engine.ibeam import primitive_samples
    from reco_engine.prune import input_inside
    from reco_engine.result_to_program import primitive_to_sp

    def samples(i, h):
        try:
            pts = primitive_samples(h.primitive, n, seed + i)
        except Exception as exc:
            h.precision, h.passed, h.note = 0.0, False, f"no samples: {exc}"
            return None
        if len(pts) == 0:
            h.precision, h.passed, h.note = 0.0, False, "no samples"
            return None
        return pts
    for i, h in enumerate(hyps):
        if h.role == "cut" and (pts := samples(i, h)) is not None:
            h.precision = float((~input_inside(mesh, pts)).mean())
            h.passed = h.precision >= CUT_PRECISION
    cut_sdfs = [primitive_to_sp(h.primitive) for h in hyps if h.role == "cut" and h.passed]
    for i, h in enumerate(hyps):
        if h.role == "add" and (pts := samples(i, h)) is not None:
            ok = input_inside(mesh, pts)
            for c in cut_sdfs:
                ok |= np.asarray(c.sdf(pts)) <= 0
            h.precision = float(ok.mean())
            h.passed = h.precision >= ADD_PRECISION
    V = np.asarray(mesh[0], float)
    tol = 1e-3 * float(np.linalg.norm(np.ptp(V, axis=0)))
    for h in hyps:
        if h.role == "cover":
            h.precision = float(np.mean(np.asarray(primitive_to_sp(h.primitive).sdf(V)) <= tol))
            h.passed = h.precision >= COVER_HOLDS
    return hyps


def select(hyps: List[Hypothesis], mesh, n: int = 100_000, min_delta: float = MIN_DELTA, tol: float = 0.002,
           input_path: str = "") -> Tuple[ReconstructionResult, Dict]:
    """A greedy CSG over the passed hypotheses on one set of sample points: the first operation the add or cover of the
    best IoU alone, then every round the add (model | h), cut (model - h) or intersection with a cover (model & h) that
    raises the IoU most, until a round gains < `min_delta`; then prune.minimize at `tol`. Returns (program, info)."""
    from reco_engine import prune as pr
    cand = [h for h in hyps if h.passed]
    if not any(h.role in ("add", "cover") for h in cand):
        return ReconstructionResult(input_path=input_path, operations=[]), dict(reason="no add and no cover passed",
                                                                               candidates=len(cand))
    pool = ReconstructionResult(input_path=input_path, operations=[Operation("add", h.primitive) for h in cand])
    s = pr.sample_program(pool, mesh, n=n)
    inp = s.input

    def iou(m):
        u = np.sum(m | inp)
        return float(np.sum(m & inp) / u) if u else 0.0
    model = np.zeros(len(s.points), bool)
    chosen: List[Tuple[int, str]] = []
    current = 0.0
    while True:
        best = (current + min_delta, None, None, None)
        for k, h in enumerate(cand):
            if any(k == j for j, _ in chosen):
                continue
            m = s.inside[:, k]
            if not chosen:
                moves = [("add", m)] if h.role in ("add", "cover") else []
            elif h.role == "add":
                moves = [("add", model | m)]
            elif h.role == "cut":
                moves = [("cut", model & ~m)]
            else:
                moves = [("intersect", model & m)]
            for op, trial in moves:
                v = iou(trial)
                if v > best[0]:
                    best = (v, k, op, trial)
        if best[1] is None:
            break
        current, model = best[0], best[3]
        chosen.append((best[1], best[2]))
    ops = [Operation(op, cand[k].primitive) for k, op in chosen]
    sub = pr.Sample(s.points, s.cell, s.inside[:, [k for k, _ in chosen]], inp, ops)
    keep, _ = pr.minimize(sub, np.ones(len(ops), bool), tol=tol)
    if not keep[0]:                                              # the program must start with an add
        keep[0] = True
    program = ReconstructionResult(input_path=input_path, operations=[o for o, k in zip(ops, keep) if k])
    if program.operations and program.operations[0].op != "add":
        program.operations[0].op = "add"
    m = pr.metrics(sub, keep)
    by = {}
    for o in program.operations:
        by[o.op] = by.get(o.op, 0) + 1
    return program, dict(candidates=len(cand), operations=len(program.operations), by_op=by, iou=m["iou"],
                         missing=m["missing"], invented=m["invented"])


def recognize(step_path, mesh=None, **kw) -> Dict:
    """Hypotheses, verification and selection for one STEP part; `mesh` (V, F) defaults to the STEP's own tessellation."""
    from reco_engine.brep import step_faces
    inv = step_faces(step_path)
    if mesh is None:
        mesh = (np.asarray(inv["vertices"], float), np.asarray(inv["triangles"]))
    hyps, surfaces = hypotheses(inv)
    verify(hyps, mesh)
    program, info = select(hyps, mesh, input_path=str(step_path), **kw)
    return dict(hypotheses=hyps, passed=[h for h in hyps if h.passed], surfaces=surfaces, program=program, info=info,
                mesh=mesh)
