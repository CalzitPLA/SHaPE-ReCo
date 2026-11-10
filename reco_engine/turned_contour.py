#!/usr/bin/env python3
"""
The turned contour of the rough model. A turned part's blank is no plain cylinder: it has shoulders, chamfers and
grooves, all of them faces of revolution about the one axis - the same thing as a hole stack (reco_engine/hole_stack.py),
on the outside. The input's cylinders and cones that are coaxial with the rough cylinder give a profile r(t) along
its axis; where the profile lies inside the rough cylinder's radius, a ring is cut: a cylinder larger than the rough
one minus the stack of the profile's cylinders and cones (Part primitives and booleans only).

    done = turned_contour(result, exact_faces(path)["faces"], inside)      # appends the cuts to result.operations

Measured 2026-09-29 on the liner: of the material the finished model had too much, the missing contour was 495 328
mm3 (shoulder R 175 from z = 1537.3 with its chamfer 10 x 45 degrees) + 28 681 mm3 (chamfer 10 x 45 degrees at the
bottom of R 192), sampled; with the sector's wedges it joined everything that touches the split plane or the top into
one piece of 1.19e6 mm3 with 676 faces, which no feature tool explains.

The profile is an envelope: between two neighbouring ends of faces the OUTSIDE is the coaxial face of the largest
radius whose material lies inside it (convex), the INSIDE (a bore) the one of the smallest radius whose material lies
outside it. A face says nothing about the other angles (a hexagon's chamfer cone covers its corners only, a flat may
reach further out elsewhere), so every ring is sampled against the input before it is cut: it counts when at most
HARM of its points lie in the input's material. The inside is cut where the bore is wider than the rough model's
coaxial cut (or where there is none); where it is narrower the rough cut took too much, which is reported and left
to the feature phase.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from reco_engine import hole_stack as hs
from reco_engine.result_schema import Operation, ReconstructionResult, StageSource
from reco_engine.tools import AX_COS, _cyl, _fuse_all, _op, _unit

MODE_CONTOUR = "analytic:turned_contour"
HARM = 0.002                        # a ring is cut when at most this share of its points lies in the input's material
SAMPLES = 20000                     # points per ring
OVER = 0.05                         # the cutting cylinder is this share larger than the rough one; an end at the rough
#                                     cylinder's end runs this share of the run's length past it
ON_AXIS = 1e-3                      # a face is coaxial when its axis passes the rough one within this share of its radius


def _reach(f: Dict, a: np.ndarray, base: np.ndarray) -> Optional[tuple]:
    """(t0, t1) of a coaxial face along the axis: by its `along` (exact_check.face_info), else by its box when the
    axis is an axis of the world (the box of a tilted face says too much); None without either."""
    got = hs._along(f, a, base)
    if got is not None:
        return got
    k = int(np.argmax(np.abs(a)))
    if f.get("bbox") is None or abs(a[k]) < 1.0 - 1e-9:
        return None
    b = np.asarray(f["bbox"], float)
    t = sorted(float(a[k]) * (float(b[i][k]) - float(base[k])) for i in (0, 1))
    return t[0], t[1]


def coaxial_segments(faces: Sequence[Dict], axis, base, convex: bool = True, tol: float = ON_AXIS) -> List[Dict]:
    """The input's cylinders and cones on the axis (`axis` through `base`), one entry per face: dict(kind "cyl" |
    "cone", t0, t1, r0, r1, area, face (its index)), a cone also t_apex, k (tan of its semi angle), semi_angle.
    `convex`: the faces whose material lies inside them (the outside of the part), else those of a bore."""
    a, base = _unit(axis), np.asarray(base, float)
    out = []
    for i, f in enumerate(faces):
        if f.get("type") not in ("cylinder", "cone") or "axis" not in f or "center" not in f \
                or bool(f.get("convex", True)) != convex:
            continue
        off = np.asarray(f["center"], float) - base
        r_ref = abs(float(f.get("radius") or 0.0))
        if abs(_unit(f["axis"]) @ a) < AX_COS or np.linalg.norm(off - a * (off @ a)) > tol * max(r_ref, 1.0) + 1e-6:
            continue
        reach = _reach(f, a, base)
        if reach is None or reach[1] - reach[0] <= 1e-9:
            continue
        seg = dict(t0=reach[0], t1=reach[1], area=float(f["area"]), face=i)
        if f["type"] == "cylinder":
            if r_ref <= 0:
                continue
            seg.update(kind="cyl", r0=r_ref, r1=r_ref)
        else:
            semi = abs(float(f["semi_angle"]))
            t_apex = float((np.asarray(f.get("apex", f["center"]), float) - base) @ a)
            seg.update(kind="cone", t_apex=t_apex, k=math.tan(semi), semi_angle=semi)
            if seg["t0"] < t_apex < seg["t1"]:                          # never through the apex: the face's side of it
                if float((np.asarray(f["point"], float) - base) @ a) >= t_apex:
                    seg["t0"] = t_apex
                else:
                    seg["t1"] = t_apex
            seg.update(r0=hs._radius_at(seg, seg["t0"]), r1=hs._radius_at(seg, seg["t1"]))
        out.append(seg)
    return out


def _same(p: Dict, q: Dict, eps: float) -> bool:
    if p["kind"] != q["kind"]:
        return False
    if p["kind"] == "cyl":
        return abs(p["r0"] - q["r0"]) <= eps
    return abs(p["t_apex"] - q["t_apex"]) <= eps and abs(p["k"] - q["k"]) <= 1e-9 \
        and hs.stack_side(p) == hs.stack_side(q)


def contour(segments: Sequence[Dict], t_lo: float, t_hi: float, outside: bool = True) -> List[Dict]:
    """The profile r(t) between t_lo and t_hi as pieces sorted along the axis (the segments' keys, `faces`: the indices
    of the faces behind a piece): between two neighbouring ends of segments the one of the largest radius (`outside`)
    or of the smallest (a bore); neighbours of one surface are one piece. Where no face is there is no piece."""
    eps = 1e-6 * (1.0 + abs(t_hi - t_lo))
    cuts = sorted({min(max(float(t), t_lo), t_hi) for s in segments for t in (s["t0"], s["t1"])} | {t_lo, t_hi})
    marks = []
    for t in cuts:
        if not marks or t - marks[-1] > eps:
            marks.append(t)
    pieces: List[Dict] = []
    for lo, hi in zip(marks, marks[1:]):
        mid = 0.5 * (lo + hi)
        over = [s for s in segments if s["t0"] <= lo + eps and s["t1"] >= hi - eps]
        if not over:
            continue
        best = (max if outside else min)(over, key=lambda s: hs._radius_at(s, mid))
        faces = sorted({s["face"] for s in over if _same(s, best, eps)
                        and abs(hs._radius_at(s, mid) - hs._radius_at(best, mid)) <= eps})
        piece = dict({k: best[k] for k in ("kind", "t_apex", "k", "semi_angle") if k in best}, t0=lo, t1=hi,
                     r0=hs._radius_at(best, lo), r1=hs._radius_at(best, hi), faces=faces)
        last = pieces[-1] if pieces else None
        if last is not None and abs(last["t1"] - lo) <= eps and _same(last, piece, eps):
            last.update(t1=hi, r1=piece["r1"], faces=sorted(set(last["faces"]) | set(faces)))
        else:
            pieces.append(piece)
    return pieces


def _ring_points(piece: Dict, r_lo: Callable, r_hi: Callable, u, n: int, rng) -> np.ndarray:
    """Points (local: x, y about the axis, t along it) spread evenly over the volume between the radii r_lo(t) and
    r_hi(t) of the piece, within the sector `u` (None: all the way round)."""
    lo = min(r_lo(piece["t0"]), r_lo(piece["t1"]))
    hi = max(r_hi(piece["t0"]), r_hi(piece["t1"]))
    out, have = [], 0
    for _ in range(50):
        t = rng.uniform(piece["t0"], piece["t1"], 4 * n)
        r = np.sqrt(rng.uniform(lo ** 2, hi ** 2, 4 * n))
        keep = (r > np.array([r_lo(x) for x in t])) & (r < np.array([r_hi(x) for x in t]))
        ang = rng.uniform(*(u if u else (0.0, 2 * math.pi)), int(keep.sum()))
        out.append(np.stack([r[keep] * np.cos(ang), r[keep] * np.sin(ang), t[keep]], axis=1))
        have += len(out[-1])
        if have >= n:
            break
    return np.concatenate(out)[:n] if out else np.zeros((0, 3))


def _ring_volume(piece: Dict, r_other: float, u) -> float:
    span = (u[1] - u[0]) if u else 2 * math.pi
    h = piece["t1"] - piece["t0"]
    own = h * (piece["r0"] ** 2 + piece["r0"] * piece["r1"] + piece["r1"] ** 2) / 3.0      # integral of r(t)^2
    return 0.5 * span * abs(h * r_other ** 2 - own)


def _runs(pieces: Sequence[Dict], eps: float) -> List[List[Dict]]:
    runs: List[List[Dict]] = []
    for p in pieces:
        if runs and abs(runs[-1][-1]["t1"] - p["t0"]) <= eps:
            runs[-1].append(p)
        else:
            runs.append([p])
    return runs


def _checked(pieces, radius, outside, R, c, u, inside, harm, n, rng, eps) -> List[Dict]:
    """The pieces of the profile that leave a ring against the rough radius, each with the share of its ring's points
    that lie in the input (`harm`), and only those at or below `harm`."""
    out = []
    for p in pieces:
        if outside and min(p["r0"], p["r1"]) >= radius - eps or not outside and max(p["r0"], p["r1"]) <= radius + eps:
            continue
        p = dict(p)
        if outside:                                              # a cone may pass the rough radius: up to there only
            p["r0"], p["r1"] = min(p["r0"], radius), min(p["r1"], radius)
            pts = _ring_points(p, lambda t: min(hs._radius_at(p, t), radius), lambda t: radius, u, n, rng)
        else:
            if min(p["r0"], p["r1"]) < radius - eps:            # narrower than the rough cut in part: not cut here
                continue
            pts = _ring_points(p, lambda t: radius, lambda t: hs._radius_at(p, t), u, n, rng)
        if not len(pts):
            continue
        p["harm"] = float(np.mean(inside(np.asarray(c, float) + pts @ R.T)))
        p["volume"] = _ring_volume(p, radius, u)
        if p["harm"] <= harm:
            out.append(p)
    return out


def _stack(run: Sequence[Dict], R, c, t_lo: float, t_hi: float, eps: float) -> tuple:
    """(the run's cylinders and cones, its ends t0 and t1): an end at the rough cylinder's end runs OVER of the run's
    length past it, a cone with its own slope."""
    open_lo, open_hi = run[0]["t0"] <= t_lo + eps, run[-1]["t1"] >= t_hi - eps
    over = OVER * (run[-1]["t1"] - run[0]["t0"])
    stack = dict(axis=R[:, 2], base=np.asarray(c, float), segments=[dict(p) for p in run], closed_lo=not open_lo,
                 closed_hi=not open_hi)
    return hs.stack_parts(stack, over), run[0]["t0"] - (over if open_lo else 0.0), run[-1]["t1"] + (over if open_hi else 0.0)


def turned_contour(result: ReconstructionResult, faces: Sequence[Dict], inside: Callable, harm: float = HARM,
                   n: int = SAMPLES, seed: int = 0) -> List[Dict]:
    """The rough cylinder (the first accepted operation: an add of a cylinder or a cylinder sector, and the only add)
    takes the input's turned contour: per run of neighbouring profile pieces one cut is appended to
    `result.operations`. `faces`: the input's exact faces (reco_field.exact_check.exact_faces); `inside(points) ->
    bool`: the input's material. Returns what was cut: [dict(label, op, side, volume, harm, pieces)] and, for a bore
    narrower than the rough cut, dict(side "inside", skipped ...) without an operation."""
    from reco_engine.result_to_program import rotation_matrix
    ops = [o for o in result.operations if o.accepted]
    if not ops or ops[0].op != "add" or ops[0].primitive.kind != "cylinder" or ops[0].primitive.v \
            or any(o.op == "add" for o in ops[1:]):
        return []
    p = ops[0].primitive
    R, c = rotation_matrix(p), np.asarray(p.center_mm, float)
    radius, h = float(p.dims_mm["radius"]), float(p.dims_mm["half_length"])
    u = tuple(float(x) for x in p.u) if p.u else None
    eps = 1e-6 * (1.0 + 2 * h)
    rng = np.random.default_rng(seed)
    done: List[Dict] = []

    def source():
        return [StageSource(s, MODE_CONTOUR, 1.0) for s in ("type", "rotation", "dimensions", "position")]

    def label(run):
        return hs.stack_label(dict(segments=run))

    # ---- the outside: rings between the profile and the rough radius
    outer = contour(coaxial_segments(faces, R[:, 2], c, convex=True), -h, h, outside=True)
    for run in _runs(_checked(outer, radius, True, R, c, u, inside, harm, n, rng, eps), eps):
        parts, t0, t1 = _stack(run, R, c, -h, h, eps)
        tool = _op(_cyl(R, c, radius * (1.0 + OVER), t0, t1), _fuse_all(parts), "cut")
        tool.label = f"turned contour, outside: {label(run)}"
        tool.sources = source()
        result.operations.append(Operation(op="cut", primitive=tool))
        done.append(dict(label=tool.label, op="cut", side="outside", volume=float(sum(q["volume"] for q in run)),
                         harm=float(max(q["harm"] for q in run)),
                         pieces=[{k: q[k] for k in ("kind", "t0", "t1", "r0", "r1")} for q in run]))

    # ---- the inside: the bore where it is wider than the rough model's coaxial cut
    bore = 0.0
    for o in ops[1:]:
        q = o.primitive
        if o.op != "cut" or q.kind != "cylinder" or q.u or q.v:
            continue
        off = np.asarray(q.center_mm, float) - c
        if abs(rotation_matrix(q)[:, 2] @ R[:, 2]) >= AX_COS and \
                np.linalg.norm(off - R[:, 2] * (off @ R[:, 2])) <= ON_AXIS * radius and \
                abs(off @ R[:, 2]) + h <= float(q.dims_mm["half_length"]) + eps:
            bore = max(bore, float(q.dims_mm["radius"]))
    inner = contour(coaxial_segments(faces, R[:, 2], c, convex=False), -h, h, outside=False)
    narrow = [q for q in inner if min(q["r0"], q["r1"]) < bore - eps]
    if narrow:
        done.append(dict(label=f"the bore is narrower than the rough cut R {bore:.4g}: {label(narrow)}", op=None,
                         side="inside", skipped="the rough cut took too much; left to the feature phase",
                         pieces=[{k: q[k] for k in ("kind", "t0", "t1", "r0", "r1")} for q in narrow]))
    for run in _runs(_checked(inner, bore, False, R, c, u, inside, harm, n, rng, eps), eps):
        parts, _, _ = _stack(run, R, c, -h, h, eps)
        tool = _fuse_all(parts)
        tool.label = f"turned contour, inside: {label(run)}"
        tool.sources = source()
        result.operations.append(Operation(op="cut", primitive=tool))
        done.append(dict(label=tool.label, op="cut", side="inside", volume=float(sum(q["volume"] for q in run)),
                         harm=float(max(q["harm"] for q in run)),
                         pieces=[{k: q[k] for k in ("kind", "t0", "t1", "r0", "r1")} for q in run]))
    return done
