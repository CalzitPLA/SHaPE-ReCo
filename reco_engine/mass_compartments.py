#!/usr/bin/env python3
"""
The mass compartments of a part (the user, 2026-10-29: "implement an approach which looks for mass-population
compartments on the whole and then add-cuts until we got good fits"): the occupancy of the whole input split, step by
step, into boxes that hold its mass with little empty space - the start of the add-cut loop (ResidualConfig.start_shape
"compartments", residual_loop._compartment_add): the well-filled compartments are added, the loop then cuts what they
hold too much and adds what they miss, until its fits converge.

    leaves = compartments(occ)       # [dict(lo, hi, voxels, volume, fill)], the most voxels first (index space, inclusive)

The split (a binary space partition by mass): a box is shrunk to the occupied voxels inside it; its waste is its volume
minus its voxels. The split of a box is the plane across one axis that leaves the least waste once both halves are shrunk
to their own mass - every plane of every axis evaluated in linear time from the per-slice mass profile and the occupied
extents across the slice (cumulative minima and maxima). A box is a leaf when it is filled to `fill_good`, when it is
small, or when no plane removes at least `min_gain` of its waste and the mass profile has no jump of `jump_min` of the
slice area either (a window enclosed by the box stays inside both halves of every single cut, so the waste cannot fall
at once; the profile jumps where a compartment ends - the frame's bars against the stretch over the window); the box
with the most waste is split first, at most `max_leaves` leaves. Bits of fewer than `min_voxels` voxels (connected) are
dropped first, leaves filled below `min_fill` after (scattered mass: the loop adds it later).

Unlike the voxel-cube cover (start_shape "cubes": the largest boxes INSIDE the material, an under-cover the loop must
add to), the compartments COVER the mass with bounded waste: an over-cover the loop cuts - add first, then cut.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

FILL_GOOD = 0.85          # a compartment filled at least this much is a leaf
MIN_GAIN = 0.05           # a split must remove at least this share of the box's waste
MAX_LEAVES = 12
MIN_VOXELS = 27           # a box smaller than this is not split further
MIN_FILL = 0.25           # a leaf filled less than this is left out
JUMP_MIN = 0.25           # a jump of the mass profile of at least this share of the slice area splits a box that no
                          # waste split improves (a frame's window stays inside both halves of every single cut)
BIG = 1 << 30


def shrink(occ: np.ndarray, lo, hi) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """(lo, hi, voxels) of the occupied voxels inside the box lo..hi (inclusive), or None when it holds none."""
    lo, hi = np.asarray(lo, int), np.asarray(hi, int)
    sub = occ[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    n = int(sub.sum())
    if n == 0:
        return None
    ext = []
    for a in range(3):
        prof = sub.any(axis=tuple(x for x in range(3) if x != a))
        idx = np.nonzero(prof)[0]
        ext.append((idx[0], idx[-1]))
    return lo + np.array([e[0] for e in ext]), lo + np.array([e[1] for e in ext]), n


def _volume(lo, hi) -> int:
    return int(np.prod(np.asarray(hi) - np.asarray(lo) + 1))


def _extents(S: np.ndarray):
    """Per slice of axis 0 of S (n_a, n_b, n_c): its voxels and the occupied index range along b and c (empty: BIG / -BIG)."""
    count = S.sum(axis=(1, 2)).astype(np.int64)
    out = []
    for proj in (S.any(axis=2), S.any(axis=1)):                  # (n_a, n_b), (n_a, n_c)
        has = proj.any(axis=1)
        first = np.where(has, np.argmax(proj, axis=1), BIG)
        last = np.where(has, proj.shape[1] - 1 - np.argmax(proj[:, ::-1], axis=1), -BIG)
        out.append((first, last))
    return count, out


def best_split(occ: np.ndarray, lo, hi) -> Optional[Dict]:
    """The plane that leaves the least waste in the box lo..hi (already shrunk): dict(axis, at, waste, jump, left, right,
    by_jump) with the children shrunk ((lo, hi, voxels)); `by_jump` the plane at the largest step of the mass profile
    (relative to the slice area). None when the box cannot be split (one voxel thick everywhere)."""
    lo, hi = np.asarray(lo, int), np.asarray(hi, int)
    sub = occ[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    best, jump = None, None
    for a in range(3):
        n_a = sub.shape[a]
        if n_a < 2:
            continue
        others = [x for x in range(3) if x != a]
        S = np.moveaxis(sub, a, 0)                               # (n_a, n_b, n_c), b and c the other axes in order
        count, ((bf, bl), (cf, cl)) = _extents(S)
        idx = np.arange(n_a)
        nonempty = count > 0
        # left part [0 .. s], right part [s + 1 .. n_a - 1]
        cl_count = np.cumsum(count)
        cr_count = cl_count[-1] - cl_count
        l_afirst = np.minimum.accumulate(np.where(nonempty, idx, BIG))
        l_alast = np.maximum.accumulate(np.where(nonempty, idx, -BIG))
        r_afirst = np.minimum.accumulate(np.where(nonempty, idx, BIG)[::-1])[::-1]
        r_alast = np.maximum.accumulate(np.where(nonempty, idx, -BIG)[::-1])[::-1]
        l_bf, l_bl = np.minimum.accumulate(bf), np.maximum.accumulate(bl)
        l_cf, l_cl = np.minimum.accumulate(cf), np.maximum.accumulate(cl)
        r_bf, r_bl = np.minimum.accumulate(bf[::-1])[::-1], np.maximum.accumulate(bl[::-1])[::-1]
        r_cf, r_cl = np.minimum.accumulate(cf[::-1])[::-1], np.maximum.accumulate(cl[::-1])[::-1]
        area = float(S.shape[1] * S.shape[2])

        def box(af, al, b0, b1, c0, c1, n):
            blo, bhi = np.zeros(3, int), np.zeros(3, int)
            blo[a], bhi[a] = af, al
            blo[others[0]], bhi[others[0]] = b0, b1
            blo[others[1]], bhi[others[1]] = c0, c1
            return lo + blo, lo + bhi, int(n)
        for s in range(n_a - 1):
            if cl_count[s] == 0 or cr_count[s] == 0:
                continue
            vl = (l_alast[s] - l_afirst[s] + 1) * (l_bl[s] - l_bf[s] + 1) * (l_cl[s] - l_cf[s] + 1)
            vr = (r_alast[s + 1] - r_afirst[s + 1] + 1) * (r_bl[s + 1] - r_bf[s + 1] + 1) * (r_cl[s + 1] - r_cf[s + 1] + 1)
            waste = int(vl - cl_count[s] + vr - cr_count[s])
            j = abs(int(count[s + 1]) - int(count[s])) / area           # the mass profile's step at this plane
            cand = None
            if best is None or waste < best["waste"] or (jump is None or j > jump["jump"]):
                cand = dict(axis=a, at=int(lo[a] + s), waste=waste, jump=j,
                            left=box(l_afirst[s], l_alast[s], l_bf[s], l_bl[s], l_cf[s], l_cl[s], cl_count[s]),
                            right=box(r_afirst[s + 1], r_alast[s + 1], r_bf[s + 1], r_bl[s + 1], r_cf[s + 1], r_cl[s + 1],
                                      cr_count[s]))
            if cand is not None and (best is None or waste < best["waste"]):
                best = cand
            if cand is not None and (jump is None or j > jump["jump"]):
                jump = cand
    if best is not None:
        best["by_jump"] = jump
    return best


def compartments(occ: np.ndarray, fill_good: float = FILL_GOOD, min_gain: float = MIN_GAIN, max_leaves: int = MAX_LEAVES,
                 min_voxels: int = MIN_VOXELS, min_fill: float = MIN_FILL, jump_min: float = JUMP_MIN) -> List[Dict]:
    """The mass compartments of the occupancy (see the module docstring): [dict(lo, hi, voxels, volume, fill)], the most
    voxels first; indices inclusive."""
    occ = np.asarray(occ, bool)
    if min_voxels > 1 and occ.any():                              # specks first: they would stretch every box
        from scipy import ndimage
        lab, n = ndimage.label(occ)
        sizes = np.bincount(lab.ravel())
        small = np.nonzero(sizes < min_voxels)[0]
        small = small[small > 0]
        if len(small):
            occ = occ & ~np.isin(lab, small)
    root = shrink(occ, (0, 0, 0), np.array(occ.shape) - 1)
    if root is None:
        return []
    leaves: List[Tuple] = []
    heap: List[Tuple] = []
    tick = 0

    def push(node):
        nonlocal tick
        lo, hi, n = node
        heapq.heappush(heap, (-(_volume(lo, hi) - n), tick, node))
        tick += 1

    push(root)
    while heap:
        _neg_waste, _t, (lo, hi, n) = heapq.heappop(heap)
        vol = _volume(lo, hi)
        if n >= fill_good * vol or vol < 2 * min_voxels or len(leaves) + len(heap) + 1 >= max_leaves:
            leaves.append((lo, hi, n))
            continue
        sp = best_split(occ, lo, hi)
        if sp is not None and sp["waste"] > (1.0 - min_gain) * (vol - n):          # no cut lowers the waste enough:
            sp = sp["by_jump"] if sp["by_jump"] is not None and sp["by_jump"]["jump"] >= jump_min else None   # the profile
        if sp is None:
            leaves.append((lo, hi, n))
            continue
        push(sp["left"])
        push(sp["right"])
    out = [dict(lo=lo.tolist(), hi=hi.tolist(), voxels=int(n), volume=_volume(lo, hi), fill=n / _volume(lo, hi))
           for lo, hi, n in leaves if n >= min_fill * _volume(lo, hi)]
    return sorted(out, key=lambda c: -c["voxels"])
