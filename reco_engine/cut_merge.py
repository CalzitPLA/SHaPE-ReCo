#!/usr/bin/env python3
"""
Fewer, better sized cuts (the user, 2026-10-29: "our cut method needs to be more accurate, often two cuts could be done
by only one; check the sizing of cuts"). On a finished program (the loop's operations):

  merge      two cut boxes in one frame whose common box (their hull) holds no input material beyond what they already
             remove, with no add between them that reaches into the hull, are one cut: the hull. Two coaxial cut
             cylinders of one radius the same along their axis. The merged cut takes the earlier one's place.
  redundant  a cut that lies inside another cut, with no add between them reaching into it, removes nothing new.
  slack      how far each face of a cut box could move outwards before it reaches input material (a cut that ends short
             of the open space leaves a skin or needs a second cut).

"Input material" is judged on the input mesh itself (the fast winding number, libigl): samples of the region the hull adds
(hull minus both cuts) must all lie outside the input, up to `tol` of the samples within half a sample spacing of the
surface. Cuts and adds are reco_engine.result_schema.Primitive (kind box / cylinder), the frame is their quaternion.

    report = audit(ops, V, F)                       # pairs, redundant, slack - nothing changed
    ops2, report = merge_cuts(ops, V, F)            # the program with the merges and the redundant cuts applied
"""

from __future__ import annotations

import copy
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

SAMPLES = 6000          # points of the region a merge adds
TOL = 0.0               # share of those points allowed inside the input (0: none)
ANGLE_DEG = 0.5         # frames this close are one frame


def _R(prim) -> np.ndarray:
    from reco_engine.result_to_program import rotation_matrix
    return rotation_matrix(prim)


def _box(prim):
    """(R, lo, hi) of a box in its own frame's coordinates (p_local = R^T (p - 0))."""
    R = _R(prim)
    c = R.T @ np.asarray(prim.center_mm, float)
    h = np.array([prim.dims_mm["x_half"], prim.dims_mm["y_half"], prim.dims_mm["z_half"]], float)
    return R, c - h, c + h


def _same_frame(Ra: np.ndarray, Rb: np.ndarray, deg: float = ANGLE_DEG) -> Optional[np.ndarray]:
    """The permutation-free check that b's axes are a's (up to sign and order): returns the matrix that maps b's local
    half extents onto a's axes, or None."""
    M = Ra.T @ Rb
    P = np.round(M)
    if np.allclose(np.abs(P).sum(axis=0), 1) and np.allclose(np.abs(P).sum(axis=1), 1) and \
            np.max(np.abs(M - P)) < math.sin(math.radians(deg)):
        return P
    return None


def box_in_frame(prim, R) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(lo, hi) of a box primitive in the frame R (None when its axes are not R's)."""
    Rb, lo_b, hi_b = _box(prim)
    P = _same_frame(R, Rb)
    if P is None:
        return None
    c_b = Rb @ ((lo_b + hi_b) / 2)                                   # world centre
    h_b = np.abs(P) @ ((hi_b - lo_b) / 2)
    c = R.T @ c_b
    return c - h_b, c + h_b


def _as_box(R, lo, hi, like):
    from reco_engine.result_to_program import quaternion_of
    q = copy.copy(like)
    c = R @ ((lo + hi) / 2)
    h = (hi - lo) / 2
    q.quaternion = tuple(float(x) for x in quaternion_of(R))
    q.center_mm = tuple(float(x) for x in c)
    q.dims_mm = dict(x_half=float(h[0]), y_half=float(h[1]), z_half=float(h[2]))
    return q


def _prim_bounds(prim) -> Tuple[np.ndarray, np.ndarray]:
    from reco_engine.result_to_program import primitive_to_sp
    lo, hi = primitive_to_sp(prim).bounds()
    return np.asarray(lo, float), np.asarray(hi, float)


def _overlap(a, b) -> bool:
    return bool(np.all(a[0] <= b[1]) and np.all(b[0] <= a[1]))


class Inside:
    """The input as an inside test: the fast winding number of its mesh (> 0.5 inside)."""

    def __init__(self, V, F):
        import igl
        self.V, self.F = np.ascontiguousarray(V, dtype=np.float64), np.ascontiguousarray(F, dtype=np.int64)
        self._igl = igl

    def __call__(self, P: np.ndarray) -> np.ndarray:
        if len(P) == 0:
            return np.zeros(0, bool)
        w = self._igl.fast_winding_number(self.V, self.F, np.ascontiguousarray(P, dtype=np.float64))
        return np.asarray(w) > 0.5


def _added_region_clear(R, hull, parts, inside: Callable, n: int = SAMPLES, tol: float = TOL, seed: int = 0) -> Tuple[bool, dict]:
    """Do the samples of hull minus the parts (boxes in frame R) all lie outside the input? dict(samples, inside, volume)."""
    rng = np.random.default_rng(seed)
    lo, hi = hull
    L = rng.uniform(lo, hi, size=(n, 3))
    keep = np.ones(n, bool)
    for plo, phi in parts:
        keep &= ~np.all((L >= plo - 1e-9) & (L <= phi + 1e-9), axis=1)
    extra = L[keep]
    vol = float(np.prod(hi - lo)) * keep.mean()
    if len(extra) == 0:
        return True, dict(samples=0, inside=0, volume=0.0)
    W = extra @ R.T
    k = int(np.count_nonzero(inside(W)))
    return k <= tol * len(extra), dict(samples=int(len(extra)), inside=k, volume=vol)


def _between_adds(ops, i: int, j: int) -> List:
    return [p for o, p in ops[i + 1:j] if o == "add"]


def merge_candidate(ops, i: int, j: int, inside: Callable, n: int = SAMPLES) -> Optional[dict]:
    """The merge of cuts i < j (both boxes in one frame): dict(prim, i, j, saved_volume, extra) or None."""
    (oi, a), (oj, b) = ops[i], ops[j]
    if oi != "cut" or oj != "cut" or a.kind != "box" or b.kind != "box":
        return None
    R, lo_a, hi_a = _box(a)
    bb = box_in_frame(b, R)
    if bb is None:
        return None
    lo_b, hi_b = bb
    hull = (np.minimum(lo_a, lo_b), np.maximum(hi_a, hi_b))
    hull_world = _prim_bounds(_as_box(R, hull[0], hull[1], a))
    if any(_overlap(hull_world, _prim_bounds(p)) for p in _between_adds(ops, i, j)):
        return None
    ok, extra = _added_region_clear(R, hull, [(lo_a, hi_a), (lo_b, hi_b)], inside, n)
    if not ok:
        return None
    return dict(prim=_as_box(R, hull[0], hull[1], a), i=i, j=j, extra=extra)


def redundant(ops, k: int) -> Optional[int]:
    """The index of a cut that contains cut k (boxes in one frame, no add between reaching into k), or None."""
    ok_, p = ops[k]
    if ok_ != "cut" or p.kind != "box":
        return None
    R, lo, hi = _box(p)
    kb = _prim_bounds(p)
    for m, (om, q) in enumerate(ops):
        if m == k or om != "cut" or q.kind != "box":
            continue
        qb = box_in_frame(q, R)
        if qb is None or not (np.all(qb[0] <= lo + 1e-6) and np.all(qb[1] >= hi - 1e-6)):
            continue
        a, b = sorted((m, k))
        if not any(_overlap(kb, _prim_bounds(x)) for x in _between_adds(ops, a, b)):
            return m
    return None


def face_slack(prim, inside: Callable, step: float, limit: float, n: int = 400, seed: int = 0) -> List[float]:
    """How far each of a cut box's six faces can move outwards (in steps of `step`, up to `limit`) before the slab it
    sweeps holds input material: [-x, +x, -y, +y, -z, +z] in the box's frame."""
    rng = np.random.default_rng(seed)
    R, lo, hi = _box(prim)
    out = []
    for ax in range(3):
        for side in (0, 1):
            moved = 0.0
            while moved < limit:
                slab_lo, slab_hi = lo.copy(), hi.copy()
                if side == 0:
                    slab_lo[ax], slab_hi[ax] = lo[ax] - moved - step, lo[ax] - moved
                else:
                    slab_lo[ax], slab_hi[ax] = hi[ax] + moved, hi[ax] + moved + step
                L = rng.uniform(slab_lo, slab_hi, size=(n, 3))
                if np.any(inside(L @ R.T)):
                    break
                moved += step
            out.append(round(moved, 4))
    return out


def audit(ops: Sequence, V, F, step: Optional[float] = None, n: int = SAMPLES) -> dict:
    """What could be simpler in a program: mergeable cut pairs, redundant cuts, and the slack of every cut box's faces
    (in units of the part's diagonal / 100 by default steps)."""
    ops = list(ops)
    inside = Inside(V, F)
    diag = float(np.linalg.norm(np.ptp(np.asarray(V, float), axis=0)))
    step = step or diag / 200.0
    pairs = []
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            m = merge_candidate(ops, i, j, inside, n)
            if m is not None:
                pairs.append(dict(i=i, j=j, extra_volume=m["extra"]["volume"]))
    red = [dict(k=k, inside_of=m) for k in range(len(ops)) if (m := redundant(ops, k)) is not None]
    slack = []
    for k, (o, p) in enumerate(ops):
        if o == "cut" and p.kind == "box":
            s = face_slack(p, inside, step, limit=diag / 4)
            slack.append(dict(k=k, slack=s, open_faces=sum(x >= diag / 4 - 1e-9 for x in s),
                              short_faces=sum(0 < x < diag / 4 - 1e-9 for x in s)))
    cuts = sum(o == "cut" for o, _ in ops)
    return dict(n_ops=len(ops), cuts=cuts, box_cuts=sum(o == "cut" and p.kind == "box" for o, p in ops),
                mergeable_pairs=pairs, redundant=red, slack=slack, step=step, diag=diag)


def merge_cuts(ops: Sequence, V, F, n: int = SAMPLES, max_rounds: int = 50) -> Tuple[List, dict]:
    """The program with every merge applied (the pair whose hull adds the least volume first, repeatedly) and the
    redundant cuts dropped. Returns (operations, dict(merges, dropped))."""
    ops = list(ops)
    inside = Inside(V, F)
    merges, dropped = [], []
    for _ in range(max_rounds):
        best = None
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                m = merge_candidate(ops, i, j, inside, n)
                if m is not None and (best is None or m["extra"]["volume"] < best["extra"]["volume"]):
                    best = m
        if best is None:
            break
        ops[best["i"]] = ("cut", best["prim"])
        del ops[best["j"]]
        merges.append(dict(i=best["i"], j=best["j"], extra_volume=best["extra"]["volume"]))
    k = 0
    while k < len(ops):
        if redundant(ops, k) is not None:
            dropped.append(k)
            del ops[k]
        else:
            k += 1
    return ops, dict(merges=merges, dropped=dropped)


def _lift_box(R, clo, chi, p, adds, margin: float, tol: float):
    """The box `p` with its faces near a cover face (within `tol`, the box in the cover's frame within ANGLE_DEG) moved
    outwards by `margin` where no add reaches into the swept slab; (box or None, faces moved). The box takes the cover's
    exact frame (a fitted 0.07 deg tilt is a fitting artefact on an axis-aligned part)."""
    bb = box_in_frame(p, R)
    if bb is None:
        return None, 0
    lo, hi = bb[0].copy(), bb[1].copy()
    moved = 0
    for a in range(3):
        for side, cover_face in ((0, clo[a]), (1, chi[a])):
            face = lo[a] if side == 0 else hi[a]
            if abs(face - cover_face) > tol:
                continue
            slab_lo, slab_hi = lo.copy(), hi.copy()
            if side == 0:
                slab_lo[a], slab_hi[a] = cover_face - margin, face
            else:
                slab_lo[a], slab_hi[a] = face, cover_face + margin
            slab = _prim_bounds(_as_box(R, slab_lo, slab_hi, p))
            if any(_overlap(slab, _prim_bounds(q)) for q in adds):
                continue
            if side == 0:
                lo[a] = cover_face - margin
            else:
                hi[a] = cover_face + margin
            moved += 1
    return (_as_box(R, lo, hi, p) if moved else None), moved


def lift_cover_faces(ops: Sequence, margin: float, tol: Optional[float] = None) -> Tuple[List, int]:
    """Cut faces that lie in a face of the cover (the first add, a box) moved outwards by `margin`: a cut ending on the
    cover's own face leaves a skin of the cover there - of zero thickness, or a wedge when the cut's face is tilted. The
    bracket of 2026-10-29: the rough guess's box - cylinder had its box face 0.007 mm off the cover's y = 40 face and
    tilted 0.07 deg - Hausdorff/D 0.1485 at exact IoU 0.9992. A box cut, the box of a cut composite (its first part
    for "cut", every box part for "union") are lifted; `tol` defaults to margin / 6 (half a cell for a margin of three).
    A face moves only when no other add reaches into the slab it sweeps. Returns (operations, faces moved)."""
    ops = list(ops)
    if not ops or ops[0][0] != "add" or ops[0][1].kind != "box":
        return ops, 0
    R, clo, chi = _box(ops[0][1])
    tol = margin / 6.0 if tol is None else tol
    adds = [p for o, p in ops[1:] if o == "add"]
    moved = 0
    for k, (o, p) in enumerate(ops):
        if o != "cut":
            continue
        if p.kind == "box":
            q, m = _lift_box(R, clo, chi, p, adds, margin, tol)
            if q is not None:
                q.label = (p.label or "box") + ", past the cover's face"
                ops[k], moved = (o, q), moved + m
        elif p.kind == "composite" and p.parts:
            idx = [0] if (p.combine or "cut") in ("cut", "subtract", "difference") else range(len(p.parts))
            parts, m_all = list(p.parts), 0
            for i in idx:
                if parts[i].kind == "box":
                    q, m = _lift_box(R, clo, chi, parts[i], adds, margin, tol)
                    if q is not None:
                        parts[i], m_all = q, m_all + m
            if m_all:
                c = copy.copy(p)
                c.parts = parts
                c.label = (p.label or "composite") + ", past the cover's face"
                ops[k], moved = (o, c), moved + m_all
    return ops, moved


def _cut_boxes(ops):
    """[(op index, part index or None, box)] of every box a cut removes: box cuts and the minuend box of a cut composite."""
    out = []
    for k, (o, p) in enumerate(ops):
        if o != "cut":
            continue
        if p.kind == "box":
            out.append((k, None, p))
        elif p.kind == "composite" and p.parts and (p.combine or "cut") in ("cut", "subtract", "difference") \
                and p.parts[0].kind == "box":
            out.append((k, 0, p.parts[0]))
    return out


def _set_box(ops, k, part, box, label):
    o, p = ops[k]
    if part is None:
        box.label = (p.label or "box") + label
        ops[k] = (o, box)
    else:
        c = copy.copy(p)
        c.parts = list(p.parts)
        c.parts[part] = box
        c.label = (p.label or "composite") + label
        ops[k] = (o, c)


SEAM_LABEL = "seam cut (the free part of a seam between two cuts)"


def _covered(cut, boxes, rel: float = 1e-6) -> bool:
    """Does one of `boxes` (cut boxes) already hold the box `cut`, up to `rel` of its size?"""
    R, lo, hi = _box(cut)
    eps = rel * max(1.0, float(np.linalg.norm(hi - lo)))
    for b in boxes:
        bb = box_in_frame(b, R)
        if bb is not None and np.all(bb[0] <= lo + eps) and np.all(bb[1] >= hi - eps):
            return True
    return False


def close_seams(ops: Sequence, V, F, tol: float, overlap: float, n: int = 2000, return_origin: bool = False):
    """Two cuts whose facing faces lie less than `tol` apart (one frame, overlapping cross-sections) leave a wall of that
    thickness between them - the bracket of 2026-10-29: 0.019 mm between a largest-box cut ending at x 6.700 and the
    rough guess's box ending at x 6.681, Hausdorff/D 0.1485 at exact IoU 0.9994. The face of the earlier cut is moved
    `overlap` into the other where the slab it sweeps (its WHOLE cross-section) holds no input material (the input mesh's
    winding number on `n` samples); where the sweep holds material (the
    bracket's rib between the two), the free part of the shared cross-section gets seam cuts (_seam_cuts), inserted
    after the later of the two. Decided on the program's own cuts (a seam cut is no partner); the extensions are applied
    and the seam cuts inserted at the end. Returns (operations, seams closed + seam cuts added) and, with
    `return_origin`, the index of the input operation each output one comes from (None: an inserted seam cut) - the
    caller must not pair the lists by position (the loop's first version did and wrecked the bracket: IoU 0.704).
    A seam cut of an earlier pass is no partner either, and a seam cut that a cut already holds is not added again
    (protocol rough_tidy_v1, DEV: the second pass after the feature phase repeated aegir_n6_I_000121's seam cut twice)."""
    ops = list(ops)
    inside = Inside(V, F)
    held = [b for _, _, b in _cut_boxes(ops)]                         # every box a cut removes, seam cuts included
    boxes = [(k, part, b) for k, part, b in _cut_boxes(ops) if not (ops[k][1].label or "").startswith(SEAM_LABEL)]
    current = {(k, part): b for k, part, b in boxes}                 # a box as extended so far
    changed, inserts = set(), []
    for x in range(len(boxes)):
        for y in range(x + 1, len(boxes)):
            (ki, pi, _), (kj, pj, _) = boxes[x], boxes[y]
            if ki == kj:
                continue
            bi, bj = current[(ki, pi)], current[(kj, pj)]
            R, lo_i, hi_i = _box(bi)
            bb = box_in_frame(bj, R)
            if bb is None:
                continue
            lo_j, hi_j = bb
            for a in range(3):
                o_ = [k for k in range(3) if k != a]
                if not all(max(lo_i[k], lo_j[k]) < min(hi_i[k], hi_j[k]) for k in o_):
                    continue                                       # no shared cross-section
                for side in (1, -1):
                    gap = (lo_j[a] - hi_i[a]) if side > 0 else (lo_i[a] - hi_j[a])
                    if not (-1e-12 <= gap <= tol):                 # touching or a thin wall, not overlapping already
                        continue
                    face = hi_i[a] if side > 0 else lo_i[a]
                    seam_lo, seam_hi = lo_i.copy(), hi_i.copy()
                    for k in o_:
                        seam_lo[k], seam_hi[k] = max(lo_i[k], lo_j[k]), min(hi_i[k], hi_j[k])
                    if side > 0:
                        seam_lo[a], seam_hi[a] = face, face + gap + overlap
                    else:
                        seam_lo[a], seam_hi[a] = face - gap - overlap, face
                    last = max(ki, kj)
                    # no guard against adds between the two: what a seam fix removes lies outside the input (the inside
                    # test below), so it can only take away excess - the bracket's rib piece, put back by an add between
                    # the merged cut and the rough guess, had kept the wall between them standing
                    # what the face really sweeps: box i's WHOLE cross-section (a first version tested only the shared
                    # part and cut 371 and 345 mm^3 of the bracket's boss where box i reached beyond it)
                    sweep_lo, sweep_hi = lo_i.copy(), hi_i.copy()
                    sweep_lo[a], sweep_hi[a] = seam_lo[a], seam_hi[a]
                    L = np.random.default_rng(len(inserts) + len(changed)).uniform(sweep_lo, sweep_hi, size=(n, 3))
                    if np.any(inside(L @ R.T)):
                        for c in _seam_cuts(R, a, seam_lo, seam_hi, bi, inside):
                            if not _covered(c, held):
                                inserts.append((last, c))
                                held.append(c)
                        continue
                    if side > 0:
                        hi_i = hi_i.copy()
                        hi_i[a] = face + gap + overlap
                    else:
                        lo_i = lo_i.copy()
                        lo_i[a] = face - gap - overlap
                    current[(ki, pi)] = bi = _as_box(R, lo_i, hi_i, bi)
                    changed.add((ki, pi))
    for k, part in changed:
        _set_box(ops, k, part, current[(k, part)], ", seam closed")
    origin: List = list(range(len(ops)))
    for after, cut in sorted(inserts, key=lambda t: -t[0]):           # from the back: the indices before stay valid
        ops.insert(after + 1, ("cut", cut))
        origin.insert(after + 1, None)
    if return_origin:
        return ops, len(changed) + len(inserts), origin
    return ops, len(changed) + len(inserts)


def _largest_free_rectangle(free: np.ndarray):
    """(i0, i1, j0, j1) inclusive of the largest all-True rectangle of a 2-D boolean grid (the histogram method), or None."""
    rows, cols = free.shape
    heights = np.zeros(cols, int)
    best, best_area = None, 0
    for i in range(rows):
        heights = np.where(free[i], heights + 1, 0)
        stack = []
        for j in range(cols + 1):
            h = heights[j] if j < cols else 0
            start = j
            while stack and stack[-1][1] >= h:
                s0, hh = stack.pop()
                area = hh * (j - s0)
                if hh and area > best_area:
                    best_area, best = area, (i - hh + 1, i, s0, j - 1)
                start = s0
            stack.append((start, h))
    return best


def _seam_cuts(R, a: int, seam_lo, seam_hi, like, inside, cells: int = 48, max_cuts: int = 3,
               min_share: float = 0.05) -> List:
    """The free part of a seam that holds material somewhere: the seam's cross-section on a cells x cells grid, a cell
    free when samples at its corners through the seam's thickness are all outside the input; the largest free rectangle
    becomes a cut, up to `max_cuts` of them while one covers at least `min_share` of the cross-section. Returns the cuts."""
    o_ = [k for k in range(3) if k != a]
    u = np.linspace(seam_lo[o_[0]], seam_hi[o_[0]], cells + 1)
    v = np.linspace(seam_lo[o_[1]], seam_hi[o_[1]], cells + 1)
    depth = np.linspace(seam_lo[a], seam_hi[a], 3)
    G = np.zeros((cells + 1, cells + 1, 3, 3))
    G[..., o_[0]] = u[:, None, None]
    G[..., o_[1]] = v[None, :, None]
    G[..., a] = depth[None, None, :]
    corner_free = ~inside(G.reshape(-1, 3) @ R.T).reshape(cells + 1, cells + 1, 3).any(axis=2)
    free = corner_free[:-1, :-1] & corner_free[1:, :-1] & corner_free[:-1, 1:] & corner_free[1:, 1:]
    out = []
    for _ in range(max_cuts):
        r = _largest_free_rectangle(free)
        if r is None:
            break
        i0, i1, j0, j1 = r
        if (i1 - i0 + 1) * (j1 - j0 + 1) < min_share * free.size:
            break
        lo, hi = seam_lo.copy(), seam_hi.copy()
        lo[o_[0]], hi[o_[0]] = u[i0], u[i1 + 1]
        lo[o_[1]], hi[o_[1]] = v[j0], v[j1 + 1]
        L = np.random.default_rng(len(out)).uniform(lo, hi, size=(2000, 3))
        if not np.any(inside(L @ R.T)):
            cut = _as_box(R, lo, hi, like)
            cut.label = SEAM_LABEL
            cut.parts, cut.combine = [], None
            out.append(cut)
        free[i0:i1 + 1, j0:j1 + 1] = False
    return out
