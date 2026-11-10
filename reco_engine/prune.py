#!/usr/bin/env python3
"""
The parts of a reconstruction measured on sample points (Qt-free; the post-processing window reco_gui/postprocess.py and
the loop's pruning step use it - the user, 2026-10-30: "minimize the number of parts: simple trees for simple parts
instead of complex trees for simple parts").

Each accepted part's inside mask is computed once on uniform points in the box of the adds and the input; any keep/drop
combination is then a fold over the masks - no SDF, no FreeCAD - so a benchmark or a minimization takes seconds.

    s = sample_program(result, mesh=(V, F))
    rows = part_table(s, keep)          # mass (share of the final volume), unique volume, fit per part
    m = metrics(s, keep)                # sampled IoU, missing, invented, volume, parts
    keep2, log = minimize(s, keep, tol) # fewer parts while the IoU stays within tol of the program as it came
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

SAMPLES = 150_000           # points of a benchmark: the share of a part of 0.1 % of the volume is good to about 8 %
TOLERANCE = 0.002           # minimize: the IoU a minimization may lose against the program as it came


@dataclass
class Sample:
    points: np.ndarray                  # (n, 3) uniform in the box of the adds and the input
    cell: float                         # mm^3 one point stands for
    inside: np.ndarray                  # (n, k) bool: the point is inside part k's primitive
    input: Optional[np.ndarray]         # (n,) bool: the point is inside the input (None without a mesh)
    ops: list                           # the accepted operations, in program order


def accepted_ops(result) -> list:
    return [o for o in result.operations if getattr(o, "accepted", True)]


def _box_of(ops, mesh) -> Tuple[np.ndarray, np.ndarray]:
    """The box the model can occupy (the adds' bounds: nothing outside an add is material) and the input's."""
    from reco_engine.result_to_program import primitive_to_sp
    los, his = [], []
    for o in ops:
        if o.op != "add":
            continue
        try:
            lo, hi = (np.asarray(b, float) for b in primitive_to_sp(o.primitive).bounds())
        except Exception:
            continue
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            los.append(lo)
            his.append(hi)
    if mesh is not None:
        v = np.asarray(mesh[0], float)
        los.append(v.min(axis=0))
        his.append(v.max(axis=0))
    if not los:
        raise ValueError("no add and no input: nothing to sample")
    lo, hi = np.min(los, axis=0), np.max(his, axis=0)
    pad = 0.02 * float((hi - lo).max())
    return lo - pad, hi + pad


def input_inside(mesh, points: np.ndarray) -> np.ndarray:
    """Inside the closed input mesh: the generalized winding number (libigl), else trimesh's ray test."""
    v = np.ascontiguousarray(mesh[0], dtype=np.float64)
    f = np.ascontiguousarray(mesh[1], dtype=np.int64)
    try:
        import igl
        return np.abs(igl.winding_number(v, f, np.ascontiguousarray(points, dtype=np.float64))) > 0.5
    except ImportError:
        import trimesh
        return np.asarray(trimesh.Trimesh(v, f, process=False).contains(points), bool)


def sample_program(result, mesh=None, n: int = SAMPLES, seed: int = 0,
                   progress: Optional[Callable[[str], None]] = None) -> Sample:
    """The sample points of a program and its input: each accepted part's inside mask and the input's."""
    from reco_engine.result_to_program import primitive_to_sp
    ops = accepted_ops(result)
    lo, hi = _box_of(ops, mesh)
    pts = np.random.default_rng(seed).uniform(lo, hi, size=(n, 3))
    inside = np.zeros((n, len(ops)), bool)
    for k, o in enumerate(ops):
        if progress and k % 10 == 0:
            progress(f"sampling part {k + 1} of {len(ops)}")
        inside[:, k] = np.asarray(primitive_to_sp(o.primitive).sdf(pts)) <= 0
    inp = None
    if mesh is not None:
        if progress:
            progress("sampling the input")
        inp = input_inside(mesh, pts)
    return Sample(pts, float(np.prod(hi - lo)) / n, inside, inp, ops)


def fold(s: Sample, keep: Sequence[bool]):
    """The kept parts applied in order: (the final model's mask, per point the index of the last cut or intersect that
    emptied it (-1: none), {cut index: the points it removed when it was applied})."""
    n = len(s.points)
    model = np.zeros(n, bool)
    last = np.full(n, -1, int)
    removed: Dict[int, np.ndarray] = {}
    for i, o in enumerate(s.ops):
        if not keep[i]:
            continue
        m = s.inside[:, i]
        if o.op == "cut":
            removed[i] = model & m
            model &= ~m
            last[m] = i
        elif o.op == "intersect":
            removed[i] = model & ~m
            model &= m
            last[~m] = i
        else:                                                   # add, and the voxel fallback's boxes
            model |= m
    return model, last, removed


def metrics(s: Sample, keep: Sequence[bool]) -> dict:
    """The program of the kept parts against the input: volume, sampled IoU, missing and invented volume, parts."""
    model = fold(s, keep)[0]
    out = dict(operations=int(np.sum(keep)), volume=float(model.sum()) * s.cell, iou=None, missing=None, invented=None)
    if s.input is not None:
        inter, union = np.sum(model & s.input), np.sum(model | s.input)
        out.update(iou=float(inter / union) if union else None, missing=float(np.sum(s.input & ~model)) * s.cell,
                   invented=float(np.sum(model & ~s.input)) * s.cell)
    return out


def part_table(s: Sample, keep: Sequence[bool]) -> List[dict]:
    """Per part (in program order): mass (mm^3 and share of the final volume, fractional where adds overlap), unique
    volume, fit; a cut: the volume it removes for good and the share of that outside the input. Dropped parts: zeros."""
    model, last, removed = fold(s, keep)
    total = float(model.sum())
    owners = np.zeros(len(s.points), int)
    covers = []
    for i, o in enumerate(s.ops):
        c = None
        if keep[i] and o.op not in ("cut", "intersect"):
            c = s.inside[:, i] & model & (last < i)                # material it holds since the last cut of the point
            owners += c
        covers.append(c)
    w = np.where(owners > 0, 1.0 / np.maximum(owners, 1), 0.0)
    rows = []
    for i, o in enumerate(s.ops):
        r = dict(index=i, op=o.op, kept=bool(keep[i]), mass=0.0, share=0.0, unique=0.0, removed=0.0, fit=None)
        if covers[i] is not None:
            c = covers[i]
            mass = float(w[c].sum())
            r.update(mass=mass * s.cell, share=mass / total if total else 0.0,
                     unique=float(np.sum(c & (owners == 1))) * s.cell)
            if s.input is not None and mass > 0:
                r["fit"] = float(w[c & s.input].sum()) / mass
        elif i in removed:
            gone = removed[i] & ~model                              # what it removed and nothing put back
            r.update(removed=float(gone.sum()) * s.cell, share=float(gone.sum()) / total if total else 0.0)
            if s.input is not None and gone.any():
                r["fit"] = float(np.sum(gone & ~s.input)) / float(gone.sum())
        rows.append(r)
    return rows


def group_share(node, rows) -> Tuple[float, float]:
    """(mass share, removed share) of a tree node (op_tree.csg_tree): the sum over its leaves."""
    if node[0] == "prim":
        r = rows[node[1]]
        return (r["share"], 0.0) if r["op"] not in ("cut", "intersect") else (0.0, r["share"])
    if node[0] == "empty":
        return 0.0, 0.0
    a = b = 0.0
    for child in node[1]:
        x, y = group_share(child, rows)
        a, b = a + x, b + y
    return a, b


def minimize(s: Sample, keep: Sequence[bool], tol: float = TOLERANCE,
             progress: Optional[Callable[[str], None]] = None) -> Tuple[np.ndarray, List[dict]]:
    """Fewer parts for the same shape: drop, one at a time, the kept part whose removal leaves the best program, as long
    as its IoU against the input stays within `tol` of the program's as it came (without an input: the model changes by
    at most `tol` of its volume). At least one add stays. Returns (keep, [dict(index, iou or change) per drop])."""
    keep = np.asarray(keep, bool).copy()
    start = metrics(s, keep)
    base_model = fold(s, keep)[0]
    vol0 = max(float(base_model.sum()), 1.0)

    def score(k):                                               # higher is better
        if s.input is not None:
            return metrics(s, k)["iou"] or 0.0
        return -float(np.sum(fold(s, k)[0] ^ base_model)) / vol0

    floor = (start["iou"] - tol) if s.input is not None and start["iou"] is not None else -tol
    log = []
    while True:
        adds = [i for i in np.flatnonzero(keep) if s.ops[i].op not in ("cut", "intersect")]
        best, best_i = None, None
        for i in np.flatnonzero(keep):
            if s.ops[i].op not in ("cut", "intersect") and len(adds) <= 1:
                continue
            trial = keep.copy()
            trial[i] = False
            v = score(trial)
            if v >= floor and (best is None or v > best):
                best, best_i = v, i
        if best_i is None:
            break
        keep[best_i] = False
        log.append(dict(index=int(best_i), op=s.ops[best_i].op,
                        **({"iou": best} if s.input is not None else {"change": -best})))
        if progress:
            progress(f"dropped part {best_i + 1}: {len(log)} so far")
    return keep, log


def edited_result(result, keep: Sequence[bool]):
    """A copy of the result with only the kept accepted operations (the program as decided)."""
    out = copy.deepcopy(result)
    ops = accepted_ops(out)
    out.operations = [o for o, k in zip(ops, keep) if k]
    return out


