#!/usr/bin/env python3
"""
Fitting a VOID (a subtracted primitive) to the fit grade, not to the over-coverage.

The cut phase used to propose a void the way the cover phase proposes an addition: the CNN looks at the over-coverage
(assembly minus input), names a shape, and the shape is sized to that region's bounding box and nudged to maximise its
recall against it. That is the wrong target. A void's job is to make the ASSEMBLY equal the input: it may protrude into
empty space as far as it likes, and what matters is only what it removes from the assembly. The over-coverage of a sphere cut
flat is a spherical cap; the CNN calls it a sphere, a sphere fits a cap badly, and the void that does the job is a box - a
half-space - that is far bigger than the cap.

`fit_void` therefore optimises the void directly against the reference:

  * the objective is the tolerance-aware MISMATCH of the assembly after subtracting the void: voxels that differ from the
    reference and lie deeper than `tol` from the reference's boundary (a band of one voxel around the boundary is forgiven, as
    in reco_engine.fit_grade). Removing a voxel that is in the assembly changes the mismatch by +1 (real material, deeper
    than the band), -1 (spurious material, deeper than the band) or 0 (inside the band), so the objective is a sum over the
    removed voxels: one inside-test per evaluation, about a millisecond;
  * three model families compete, box (rotation, half extents, centre), sphere (centre, radius) and cylinder (axis, radius, half length,
    centre), each by coordinate descent from
    a bounding-box start with steps that shrink from 4 voxels / 2 degrees to a quarter voxel / 0.5 degrees (a void may need to
    grow a long way, unlike a cover primitive that is tied to what it explains); the lower mismatch wins - the type is chosen by the
    grade, not by the CNN;
  * a void that does not lower the mismatch by at least `min_gain` voxels is not returned.

The result carries what the primitive builder needs (topology, quaternion, macro parameters, centre in array index coordinates).
Voxel-only, no FreeCAD, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

REALIZATION_SCALE_MM = 10.0        # dataset_gen macros: half extent / radius mm = ratio * 10


@dataclass
class VoidFit:
    topology: str                   # "box", "sphere" or "cylinder"
    quaternion: List[float]         # world quaternion for the macro builder
    parameters: Dict[str, float]
    center_vox: np.ndarray          # array index coordinates
    mismatch_before: int
    mismatch_after: int
    scores: Dict[str, int]          # mismatch reached by each family that was tried


def _band(reference: np.ndarray, tol: int) -> np.ndarray:
    """Voxels within `tol` (chessboard) of the reference's boundary: differences there are forgiven."""
    from scipy import ndimage
    ref = reference > 0
    inner = ndimage.distance_transform_cdt(ref, metric="chessboard") > tol          # deeper than tol inside
    outer = ndimage.distance_transform_cdt(~ref, metric="chessboard") > tol         # farther than tol outside
    return ~(inner | outer) if ref.any() else np.zeros_like(ref)


def _rot(axis: int, degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    r = np.eye(3)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    r[i, i], r[i, j], r[j, i], r[j, j] = c, (-s if axis != 1 else s), (s if axis != 1 else -s), c
    return r


class _Objective:
    """mismatch(removed) = mismatch(assembly) + sum of weights over the removed voxels of the assembly."""

    def __init__(self, reference: np.ndarray, state: np.ndarray, tol: int):
        ref, st = reference > 0, state > 0
        band = _band(reference, tol)
        self.points = np.argwhere(st).astype(np.float64)
        idx = np.argwhere(st)
        in_ref = ref[tuple(idx.T)]
        in_band = band[tuple(idx.T)]
        # removing a voxel: +1 if it is reference material beyond the band, -1 if it is spurious beyond the band, else 0
        self.weights = np.where(in_band, 0, np.where(in_ref, 1, -1)).astype(np.int64)
        # what the assembly already gets wrong beyond the band (missing material the assembly lacks + spurious material)
        missing = int(np.count_nonzero(ref & ~st & ~band))
        spurious = int(-self.weights[self.weights < 0].sum())
        self.base = missing + spurious

    def box(self, M, h, c) -> int:
        inside = (np.abs((self.points - c) @ M) <= h).all(axis=1)
        return self.base + int(self.weights[inside].sum())

    def sphere(self, c, r) -> int:
        inside = ((self.points - c) ** 2).sum(axis=1) <= r * r
        return self.base + int(self.weights[inside].sum())

    def cylinder(self, c, axis, r, half_length) -> int:
        from reco_engine.cylinder_fit import cylinder_mask_points
        return self.base + int(self.weights[cylinder_mask_points(self.points, c, axis, r, half_length)].sum())


BOX_STEPS = ((2.0, 4.0), (1.0, 2.0), (0.5, 1.0), (0.5, 0.5), (0.25, 0.25))
POLISH_BOX = ((0.5, 0.5), (0.25, 0.25))
SPHERE_STEPS = (4.0, 2.0, 1.0, 0.5, 0.25)
POLISH_SPHERE = (0.5, 0.25)
CYL_STEPS = ((4.0, 2.0), (2.0, 1.0), (1.0, 0.5), (0.5, 0.5), (0.25, 0.25))
POLISH_CYL = ((0.5, 0.5), (0.25, 0.25))


def _descend_box(obj: _Objective, M, h, c, max_evaluations: int, steps=BOX_STEPS):
    M, h, c = np.array(M, dtype=np.float64), np.array(h, dtype=np.float64), np.array(c, dtype=np.float64)
    best = obj.box(M, h, c)
    evals = 1
    for rot_step, len_step in steps:
        improved = True
        while improved and evals < max_evaluations:
            improved = False
            for axis in range(3):
                for sign in (1.0, -1.0):
                    for kind in ("rot", "half", "center"):
                        M2, h2, c2 = M, h, c
                        if kind == "rot":
                            M2 = M @ _rot(axis, sign * rot_step)
                        elif kind == "half":
                            h2 = h.copy(); h2[axis] = max(h2[axis] + sign * len_step, 1.0)
                        else:
                            c2 = c.copy(); c2[axis] += sign * len_step
                        score = obj.box(M2, h2, c2)
                        evals += 1
                        if score < best:
                            M, h, c, best, improved = M2, h2, c2, score, True
    return M, h, c, best


def _descend_sphere(obj: _Objective, c, r, max_evaluations: int, steps=SPHERE_STEPS):
    c, r = np.array(c, dtype=np.float64), float(r)
    best = obj.sphere(c, r)
    evals = 1
    for step in steps:
        improved = True
        while improved and evals < max_evaluations:
            improved = False
            for axis in range(4):
                for sign in (1.0, -1.0):
                    c2, r2 = c, r
                    if axis < 3:
                        c2 = c.copy(); c2[axis] += sign * step
                    else:
                        r2 = max(r + sign * step, 1.0)
                    score = obj.sphere(c2, r2)
                    evals += 1
                    if score < best:
                        c, r, best, improved = c2, r2, score, True
    return c, r, best


def _descend_cylinder(obj: _Objective, F, c, r, L, max_evaluations: int, steps=CYL_STEPS):
    from reco_engine.cylinder_fit import _rot as tilt
    F, c, r, L = np.array(F, dtype=np.float64), np.array(c, dtype=np.float64), float(r), float(L)
    best = obj.cylinder(c, F[:, 0], r, L)
    evals = 1
    for len_step, rot_step in steps:
        improved = True
        while improved and evals < max_evaluations:
            improved = False
            for kind in ("center", "radius", "length", "tilt"):
                axes = range(3) if kind == "center" else (range(1, 3) if kind == "tilt" else (0,))
                for axis in axes:
                    for sign in (1.0, -1.0):
                        F2, c2, r2, L2 = F, c, r, L
                        if kind == "center":
                            c2 = c.copy(); c2[axis] += sign * len_step
                        elif kind == "radius":
                            r2 = max(r + sign * len_step, 1.0)
                        elif kind == "length":
                            L2 = max(L + sign * len_step, 1.0)
                        else:
                            F2 = F @ tilt(axis, sign * rot_step)
                        score = obj.cylinder(c2, F2[:, 0], r2, L2)
                        evals += 1
                        if score < best:
                            F, c, r, L, best, improved = F2, c2, r2, L2, score, True
    return F, c, r, L, best


def fit_void(reference: np.ndarray, state: np.ndarray, region: np.ndarray, tol: float = 1.0, voxel_mm: float = 0.625,
             families: Sequence[str] = ("box", "sphere", "cylinder"), min_gain: int = 20, max_evaluations: int = 700,
             box_start: Optional[np.ndarray] = None) -> Optional[VoidFit]:
    """The void (box, sphere or cylinder) that lowers the tolerance-aware mismatch of `state` against `reference` the most, started
    from the bounding box of `region` (the part of the over-coverage the loop is currently looking at) and of its largest
    connected piece, whichever ends lower. `box_start` is an
    optional (3, 3) orientation for the box family (columns = axes in the array frame); the default start is the grid axes.
    None if no void gains at least `min_gain` voxels."""
    from training import rotation_symmetry as rs
    pts = np.argwhere(region > 0)
    if len(pts) < 10 or not (state > 0).any():
        return None
    tol_i = int(round(tol))
    obj = _Objective(reference, state, tol_i)
    # start boxes: the whole region, and its largest connected piece (two pockets are two voids, not one big one)
    from scipy import ndimage
    labels, count = ndimage.label(region > 0)
    starts_pts = [pts]
    if count > 1:
        sizes = ndimage.sum(region > 0, labels, index=range(1, count + 1))
        starts_pts.append(np.argwhere(labels == (1 + int(np.argmax(sizes)))))
    scores, results = {}, {}
    if "box" in families:
        orientations = [np.eye(3)] + ([np.asarray(box_start, dtype=np.float64)] if box_start is not None else [])
        best = None
        for sp in starts_pts:
            for M0 in orientations:
                proj = sp.astype(np.float64) @ M0                                # the region seen along M0's axes
                lo, hi = proj.min(axis=0), proj.max(axis=0)
                M, h, c, score = _descend_box(obj, M0, (hi - lo + 1) / 2.0, M0 @ ((lo + hi) / 2.0), max_evaluations)
                if best is None or score < best[3]:
                    best = (M, h, c, score)
        scores["box"], results["box"] = best[3], best
    if "sphere" in families:
        best = None
        for sp in starts_pts:
            center = (sp.min(axis=0) + sp.max(axis=0)) / 2.0
            half = (sp.max(axis=0) - sp.min(axis=0) + 1) / 2.0
            c, r, score = _descend_sphere(obj, center, float(half.mean()), max_evaluations)
            if best is None or score < best[2]:
                best = (c, r, score)
        scores["sphere"], results["sphere"] = best[2], best
    if "cylinder" in families:
        from reco_engine.cylinder_fit import _frame_with_axis
        best = None
        for sp in starts_pts:
            for i in range(3):
                F = _frame_with_axis(np.eye(3)[:, i])
                proj = sp.astype(np.float64) @ F
                lo, hi = proj.min(axis=0), proj.max(axis=0)
                half = (hi - lo + 1) / 2.0
                res = _descend_cylinder(obj, F, F @ ((lo + hi) / 2.0), float(half[1:].mean()), float(half[0]), max_evaluations)
                if best is None or res[4] < best[4]:
                    best = res
        scores["cylinder"], results["cylinder"] = best[4], best
    if not scores:
        return None
    # The band forgives a voxel around the boundary, so a box and a cylinder can both score 0 on a round bore. Polish every family
    # against the strict (band-free) mismatch and break ties with it: the family that really matches wins.
    strict = _Objective(reference, state, 0)
    strict_scores = {}
    if "box" in results:
        M, h, c, score = results["box"]
        M2, h2, c2, s2 = _descend_box(strict, M, h, c, max_evaluations, POLISH_BOX)
        if obj.box(M2, h2, c2) <= score:
            results["box"] = (M2, h2, c2, obj.box(M2, h2, c2))
        strict_scores["box"] = strict.box(*results["box"][:3])
    if "sphere" in results:
        c, r, score = results["sphere"]
        c2, r2, s2 = _descend_sphere(strict, c, r, max_evaluations, POLISH_SPHERE)
        if obj.sphere(c2, r2) <= score:
            results["sphere"] = (c2, r2, obj.sphere(c2, r2))
        strict_scores["sphere"] = strict.sphere(*results["sphere"][:2])
    if "cylinder" in results:
        F, c, r, L, score = results["cylinder"]
        F2, c2, r2, L2, s2 = _descend_cylinder(strict, F, c, r, L, max_evaluations, POLISH_CYL)
        if obj.cylinder(c2, F2[:, 0], r2, L2) <= score:
            results["cylinder"] = (F2, c2, r2, L2, obj.cylinder(c2, F2[:, 0], r2, L2))
        strict_scores["cylinder"] = strict.cylinder(results["cylinder"][1], results["cylinder"][0][:, 0], *results["cylinder"][2:4])
    scores = {k: (results[k][-1]) for k in results}
    family = min(scores, key=lambda k: (scores[k], strict_scores[k]))
    if obj.base - scores[family] < min_gain:
        return None
    if family == "box":
        M, h, c, score = results["box"]
        if np.linalg.det(M) < 0:
            M = M.copy(); M[:, 2] *= -1.0
        return VoidFit("box", rs.array_orientation_to_world_quaternion(M),
                       {"x_half_ratio": float(h[0] * voxel_mm / REALIZATION_SCALE_MM), "y_half_ratio": float(h[1] * voxel_mm / REALIZATION_SCALE_MM),
                        "z_half_ratio": float(h[2] * voxel_mm / REALIZATION_SCALE_MM)}, c, obj.base, score, scores)
    if family == "cylinder":
        from reco_engine.cylinder_fit import axis_quaternion
        F, c, r, L, score = results["cylinder"]
        return VoidFit("cylinder", axis_quaternion(F[:, 0]),
                       {"radius_ratio": float(r * voxel_mm / REALIZATION_SCALE_MM), "half_length_ratio": float(L * voxel_mm / REALIZATION_SCALE_MM)},
                       c, obj.base, score, scores)
    c, r, score = results["sphere"]
    return VoidFit("sphere", [0.0, 0.0, 0.0, 1.0], {"radius_ratio": float(r * voxel_mm / REALIZATION_SCALE_MM)}, c, obj.base, score, scores)
