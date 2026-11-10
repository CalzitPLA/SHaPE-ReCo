#!/usr/bin/env python3
"""
Hole stacks (the user's rule 2026-09-28: "the hole features could be just reconstructed as stacked cylinders and
cones"). A drilled, counterbored, countersunk or tapped hole is, on one axis, a stack of cylinders and cones; the
piece's exact faces name every step: its radius (or apex and semi angle), and where along the axis it begins and ends.

    stacks = hole_stacks(piece["faces"], V)          # one per axis line, the steps sorted along the axis
    tool = stacks_tool(stacks, over)                  # the stacks fused: a composite of cylinders and cones, world mm

Measured on the liner (run 20260928-220854): its 24 bolt holes are R 6 x 7 | R 3.5 x 50 | R 2.459 x 9.4 | drill tip
120 deg - four steps, which no catalogue tool has (reco_engine/tools.py reads at most two), and the single cylinders
read per face were made as long as the whole piece (feature_stage.cylinder_faces), so the R 6 counterbore ran 50 mm
deep: precision 0.93 against the 0.999 asked for. Two more pieces are a pocket with two blind holes: two stacks.

A step counts when its faces go all the way round the axis (their area is at least `full` of the step's lateral
area): a pocket's quarter-cylinder corner is no hole. The reach of a step along the axis comes from the face's `along`
(reco_field/exact_check.py's face_info); without it the length is the faces' area taken all the way round, about the
face's mid point, and counts only when the piece's mesh has both end circles there.

An end of a stack is CLOSED when a cone runs to a point there or a real plane square to the axis caps it (a flat
bottom: the cap is no larger than the step's cross-section and looks out of the stack); every other end is OPEN and runs
a little past the piece (5 % of the stack's length) - into the air or the invented material the model's face stood in. A gap between two steps (a
small chamfer or rounding that is no step of its own) is bridged by the thinner of the two.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from reco_engine.result_schema import Primitive
from reco_engine.tools import AX_COS, REAL, _cone, _cyl, _frame_z, _fuse_all, _unit

FULL = 0.9                          # a step's faces cover at least this share of its lateral area
POINT = 1e-3                        # a cone runs to a point: its end radius below this share of its other radius


def _along(f: Dict, a: np.ndarray, base: np.ndarray) -> Optional[tuple]:
    """(t0, t1) of a face along the axis `a` from `base` by its `along` (its reach from its own centre along its own
    axis); None when the face has none (pieces written before 2026-09-28)."""
    if not f.get("along"):
        return None
    c = np.asarray(f["center"], float)
    s = 1.0 if _unit(f["axis"]) @ a > 0 else -1.0
    t = sorted(float((c - base) @ a) + s * float(x) for x in f["along"])
    return t[0], t[1]


def _ring(V: Optional[np.ndarray], a: np.ndarray, base: np.ndarray, t: float, r: float, tol: float) -> bool:
    """Has the mesh a vertex on the circle of radius r about the axis at t?"""
    if V is None or not len(V):
        return False
    d = np.asarray(V, float) - base
    tv = d @ a
    rho = np.linalg.norm(d - np.outer(tv, a), axis=1)
    return bool(((np.abs(tv - t) <= tol) & (np.abs(rho - r) <= tol)).any())


def _radius_at(seg: Dict, t: float) -> float:
    if seg["kind"] == "cyl":
        return seg["r0"]
    return abs(t - seg["t_apex"]) * seg["k"]


def _lateral(seg: Dict) -> float:
    if seg["kind"] == "cyl":
        return 2 * math.pi * seg["r0"] * (seg["t1"] - seg["t0"])
    return math.pi * abs(seg["r1"] ** 2 - seg["r0"] ** 2) / max(math.sin(seg["semi_angle"]), 1e-12)


def hole_stacks(faces: Sequence[Dict], V: Optional[np.ndarray] = None, full: float = FULL, tol: float = 1e-3) -> List[Dict]:
    """The stacks of a piece, the largest first: [dict(axis, base, segments, closed_lo, closed_hi, faces, volume)].
    `segments`: dict(kind "cyl" | "cone", t0, t1, r0, r1, area, coverage) sorted along the axis, r0 at t0; `faces`:
    the indices (into `faces`) of the faces the stack explains. Only real faces (origin input / both) that are convex
    on the piece (its material inside them) make steps."""
    curved = [(i, f) for i, f in enumerate(faces)
              if f["type"] in ("cylinder", "cone") and "axis" in f and "center" in f and f.get("origin") in REAL
              and f.get("convex", True) and (f["type"] == "cone" or float(f.get("radius") or 0.0) > 0)]
    if not curved:
        return []
    if V is not None and len(V):
        scale = float(np.linalg.norm(np.ptp(np.asarray(V, float), axis=0)))
    else:
        scale = max(float(f.get("radius") or 1.0) for _, f in curved)
    lines: List[Dict] = []
    for i, f in sorted(curved, key=lambda q: -q[1]["area"]):
        a = _unit(f["axis"])
        c = np.asarray(f["center"], float)
        for ln in lines:
            off = c - ln["base"]
            if abs(a @ ln["axis"]) >= AX_COS and \
                    np.linalg.norm(off - ln["axis"] * (off @ ln["axis"])) <= tol * max(float(f.get("radius") or 1.0), 1.0) + 1e-6:
                ln["members"].append((i, f))
                break
        else:
            lines.append(dict(axis=a, base=c, members=[(i, f)]))
    out = []
    for ln in lines:
        a, base = ln["axis"], ln["base"]
        surfaces: List[Dict] = []
        for i, f in ln["members"]:
            pm = np.asarray(f["point"], float) - base
            tm = float(pm @ a)
            if f["type"] == "cylinder":
                key = ("cyl", round(float(f["radius"]), 6))
            else:
                apex = np.asarray(f.get("apex", f["center"]), float)
                key = ("cone", round(float((apex - base) @ a), 5), round(float(f["semi_angle"]), 6),
                       bool((np.asarray(f["point"], float) - apex) @ a > 0))
            reach = _along(f, a, base)
            for s in surfaces:            # one surface: reaches that touch or, without them, one middle - one step
                if s["key"] != key or (reach is None) != (s["reach"] is None):
                    continue
                if (reach is None and abs(s["tm"] - tm) <= 1e-6 * (1 + scale)) or \
                        (reach is not None and reach[0] <= s["reach"][1] + 1e-6 * (1 + scale)
                         and reach[1] >= s["reach"][0] - 1e-6 * (1 + scale)):
                    if reach is not None:
                        s["reach"] = (min(s["reach"][0], reach[0]), max(s["reach"][1], reach[1]))
                    s["area"] += f["area"]
                    s["faces"].append(i)
                    break
            else:
                surfaces.append(dict(key=key, face=f, tm=tm, rho=float(np.linalg.norm(pm - a * tm)), reach=reach,
                                     area=float(f["area"]), faces=[i]))
        segs = []
        for s in surfaces:
            f = s["face"]
            seg = dict(kind=s["key"][0], area=s["area"], faces=s["faces"])
            if seg["kind"] == "cyl":
                seg["r0"] = seg["r1"] = float(f["radius"])
                h = s["area"] / (2 * math.pi * seg["r0"])                # all the way round: checked below
            else:
                apex = np.asarray(f.get("apex", f["center"]), float)
                seg.update(t_apex=float((apex - base) @ a), k=math.tan(float(f["semi_angle"])),
                           semi_angle=float(f["semi_angle"]))
                h = s["area"] * math.cos(seg["semi_angle"]) / (2 * math.pi * max(s["rho"], 1e-12))
            if s["reach"] is not None:
                seg["t0"], seg["t1"] = s["reach"]
            else:
                seg["t0"], seg["t1"] = s["tm"] - h / 2, s["tm"] + h / 2
            if seg["t1"] - seg["t0"] <= 1e-9 * (1 + scale):
                continue
            if seg["kind"] == "cone":
                # a tip ends AT its apex: rounding may put the end a hair past it (2026-09-29: 16 of 42 stacks of the
                # liner's largest piece lost their drill tip to a strict comparison here)
                eps = 1e-6 * (1.0 + scale)
                if seg["t0"] < seg["t_apex"] - eps and seg["t_apex"] + eps < seg["t1"]:
                    continue
                if 0.5 * (seg["t0"] + seg["t1"]) >= seg["t_apex"]:
                    seg["t0"] = max(seg["t0"], seg["t_apex"])
                else:
                    seg["t1"] = min(seg["t1"], seg["t_apex"])
                seg["r0"], seg["r1"] = _radius_at(seg, seg["t0"]), _radius_at(seg, seg["t1"])
            if s["reach"] is not None:
                lat = _lateral(seg)
                seg["coverage"] = seg["area"] / lat if lat > 0 else 0.0
                if seg["coverage"] < full:
                    continue
            else:
                # no reach in the metadata: the length is the area's, taken all the way round - true only when the
                # mesh has both end circles there (a quarter cylinder's would lie at 1/8 of its real length)
                tol_end = 0.05 * (seg["t1"] - seg["t0"]) + 1e-3 * max(seg["r0"], seg["r1"], 1.0) + 1e-4 * scale
                if not (_ring(V, a, base, seg["t0"], seg["r0"], tol_end) and _ring(V, a, base, seg["t1"], seg["r1"], tol_end)):
                    continue
                seg["coverage"] = 1.0
            segs.append(seg)
        if not segs:
            continue
        segs.sort(key=lambda q: (q["t0"], q["t1"]))
        t_lo, t_hi = segs[0]["t0"], max(q["t1"] for q in segs)
        last = max(segs, key=lambda q: q["t1"])
        caps = []
        for f in faces:
            if f["type"] == "plane" and f.get("origin") in REAL and abs(_unit(f["normal"]) @ a) >= AX_COS:
                off = np.asarray(f["point"], float) - base
                caps.append(dict(t=float(off @ a), sign=float(np.sign(_unit(f["normal"]) @ a)), area=float(f["area"])))

        def closed(seg, t_end, sign):
            r_end = _radius_at(seg, t_end)
            if seg["kind"] == "cone" and r_end <= POINT * max(seg["r0"], seg["r1"]):
                return True
            return any(abs(c["t"] - t_end) <= 1e-3 * scale + 1e-6 and c["sign"] == sign
                       and c["area"] <= 1.05 * math.pi * r_end ** 2 for c in caps)
        volume = sum(math.pi * (q["t1"] - q["t0"]) * (q["r0"] ** 2 + q["r0"] * q["r1"] + q["r1"] ** 2) / 3.0 for q in segs)
        out.append(dict(axis=a, base=base, segments=segs, closed_lo=closed(segs[0], t_lo, -1.0),
                        closed_hi=closed(last, t_hi, 1.0), faces=sorted(i for q in segs for i in q["faces"]),
                        volume=float(volume)))
    return sorted(out, key=lambda s: -s["volume"])


OVER = 0.05                         # an open end runs this share of the STACK's length past it


def stack_parts(stack: Dict, over: Optional[float] = 0.0) -> List[Primitive]:
    """The stack's cylinders and cones, in the world: every step from its t0 to its t1, an open end `over` further (a
    cone goes on with its slope, not below radius 0), a gap to the next step bridged by the thinner of the two.
    `over` None: OVER x the stack's own length - a length of the piece would be wrong for a small hole in a large
    piece (2026-09-29: 2 % of the diagonal of the liner's largest piece are 33 mm)."""
    a, base = stack["axis"], np.asarray(stack["base"], float)
    if over is None:
        over = OVER * (max(q["t1"] for q in stack["segments"]) - min(q["t0"] for q in stack["segments"]))
    R = _frame_z(a)
    segs = [dict(q) for q in stack["segments"]]
    for q, n in zip(segs, segs[1:]):
        gap = n["t0"] - q["t1"]
        if gap > 1e-9:
            if _radius_at(q, q["t1"]) <= _radius_at(n, n["t0"]):
                q["t1"] = n["t0"]
            else:
                n["t0"] = q["t1"]
    if not stack["closed_lo"]:
        segs[0]["t0"] -= over
    if not stack["closed_hi"]:
        last = max(segs, key=lambda q: q["t1"])
        last["t1"] += over
    parts = []
    for q in segs:
        if q["kind"] == "cyl":
            parts.append(_cyl(R, base, q["r0"], q["t0"], q["t1"]))
            continue
        lo, hi = q["t0"], q["t1"]
        if lo < q["t_apex"] < hi:                               # never through the apex: the face's side of it
            lo, hi = (q["t_apex"], hi) if stack_side(q) > 0 else (lo, q["t_apex"])
        parts.append(_cone(R, base, _radius_at(q, lo), _radius_at(q, hi), lo, hi))
    return parts


def stack_side(seg: Dict) -> float:
    """+1: the cone's step lies above its apex along the axis, -1 below."""
    return 1.0 if 0.5 * (seg["t0"] + seg["t1"]) >= seg["t_apex"] else -1.0


def stack_label(stack: Dict) -> str:
    names = []
    for q in stack["segments"]:
        if q["kind"] == "cyl":
            names.append(f"R {q['r0']:.4g} x {q['t1'] - q['t0']:.3g}")
        elif min(q["r0"], q["r1"]) <= POINT * max(q["r0"], q["r1"]):
            names.append(f"tip {2 * math.degrees(q['semi_angle']):.0f} deg")
        else:
            names.append(f"cone R {q['r0']:.4g} -> {q['r1']:.4g}")
    return " | ".join(names)


def stacks_tool(stacks: Sequence[Dict], over: Optional[float] = 0.0) -> Optional[Primitive]:
    """All the stacks fused into one tool; None without a stack."""
    parts = [p for s in stacks for p in stack_parts(s, over)]
    if not parts:
        return None
    tool = _fuse_all(parts)
    n = len(stacks)
    tool.label = (f"hole stack: {stack_label(stacks[0])}" if n == 1
                  else f"{n} hole stacks: " + "; ".join(stack_label(s) for s in stacks))
    return tool
