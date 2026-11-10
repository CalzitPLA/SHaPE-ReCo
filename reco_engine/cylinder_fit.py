#!/usr/bin/env python3
"""
Axis, radius, length and centre of a solid CYLINDER from a voxel grid, by direct fitting - the third analytic primitive next to
reco_engine.box_orientation and reco_engine.sphere_fit, for the in-scale voxel mode (SHAPE_RECO_VOXEL_MODE=fixed).

The dataset macro builds a cylinder along local +Z (radius = radius_ratio * 10 mm, height = 2 * half_length_ratio * 10 mm) and
rotates it about the origin; only the axis direction matters (a cylinder has no roll). The fit therefore has seven degrees of
freedom: centre (3), axis (2), radius, half length. It starts from six axis hypotheses - the three grid axes and the three
principal axes of the material - each with the bounding extents of the material along and across that axis, and moves the
parameters by coordinate descent on the IoU between the analytic cylinder and the target (a voxel centre is inside when its
distance along the axis is within the half length and its distance from the axis within the radius), steps shrinking from
one voxel / 2 degrees to a quarter voxel / 0.25 degrees. The best of the six wins.

    fitter = CylinderFitter(grid_mm=40.0)
    est = fitter.estimate(voxel01)     # est.parameters == {"radius_ratio": ..., "half_length_ratio": ...}, est.quaternion, est.center_vox
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

REALIZATION_SCALE_MM = 10.0        # dataset_gen cylinder macro: radius / half length mm = ratio * 10


def axis_quaternion(axis_array: np.ndarray) -> List[float]:
    """World quaternion [qx, qy, qz, qw] that turns the macro's local +Z into the cylinder axis given in the array frame
    (a = A w: training.rotation_symmetry.A_WORLD_TO_ARRAY). Minimal rotation; the roll about the axis is free."""
    from training import rotation_symmetry as rs
    u = rs.A_WORLD_TO_ARRAY.T @ (np.asarray(axis_array, dtype=np.float64) / np.linalg.norm(axis_array))
    z = np.array([0.0, 0.0, 1.0])
    d = float(np.clip(z @ u, -1.0, 1.0))
    if d > 1.0 - 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    if d < -1.0 + 1e-9:
        return [1.0, 0.0, 0.0, 0.0]                                    # a half turn about x
    k = np.cross(z, u)
    s = np.linalg.norm(k)
    k = k / s
    angle = np.arccos(d)
    return [float(v) for v in (*(k * np.sin(angle / 2.0)), np.cos(angle / 2.0))]


def _rot(axis: int, degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    r = np.eye(3)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    r[i, i], r[i, j], r[j, i], r[j, j] = c, (-s if axis != 1 else s), (s if axis != 1 else -s), c
    return r


def cylinder_mask_points(points: np.ndarray, center: np.ndarray, axis: np.ndarray, radius: float, half_length: float) -> np.ndarray:
    d = points - center
    along = d @ axis
    radial2 = (d ** 2).sum(axis=1) - along ** 2
    return (np.abs(along) <= half_length) & (radial2 <= radius * radius)


@dataclass
class CylinderEstimate:
    quaternion: List[float]
    parameters: Dict[str, float]
    center_vox: np.ndarray
    axis: np.ndarray
    radius_vox: float
    half_length_vox: float
    iou_before: float = 0.0
    iou_after: float = 0.0
    refined: bool = True


def _window(target: np.ndarray):
    occ = target > 0
    pts = np.argwhere(occ)
    lo = np.maximum(pts.min(axis=0) - 4, 0)
    hi = np.minimum(pts.max(axis=0) + 5, target.shape)
    truth = occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].reshape(-1)
    grid = np.stack(np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    return grid, truth, pts


def refine_cylinder(target: np.ndarray, frame: np.ndarray, center, radius: float, half_length: float, max_evaluations: int = 500):
    """Coordinate descent on the IoU. `frame` (3x3, orthonormal): column 0 is the cylinder axis. Returns
    (frame, center, radius, half_length, iou_before, iou_after)."""
    if (target > 0).sum() < 20:
        return np.asarray(frame, float), np.asarray(center, float), float(radius), float(half_length), 0.0, 0.0
    grid, truth, _ = _window(target)
    n_truth = int(truth.sum())

    def iou(F, c, r, L):
        inside = cylinder_mask_points(grid, c, F[:, 0], r, L)
        inter = int(np.count_nonzero(inside & truth))
        return inter / max(int(np.count_nonzero(inside)) + n_truth - inter, 1)

    F, c, r, L = np.array(frame, float), np.array(center, float), float(radius), float(half_length)
    before = best = iou(F, c, r, L)
    evals = 1
    for len_step, rot_step in ((1.0, 2.0), (0.5, 1.0), (0.25, 0.5), (0.25, 0.25)):
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
                            F2 = F @ _rot(axis, sign * rot_step)
                        score = iou(F2, c2, r2, L2)
                        evals += 1
                        if score > best + 1e-4:
                            F, c, r, L, best, improved = F2, c2, r2, L2, score, True
    return F, c, r, L, before, best


def _frame_with_axis(axis: np.ndarray) -> np.ndarray:
    u = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v1 = np.cross(u, helper); v1 /= np.linalg.norm(v1)
    v2 = np.cross(u, v1)
    return np.stack([u, v1, v2], axis=1)


class CylinderFitter:
    def __init__(self, grid_mm: float = 40.0, resolution: int = 64, refine: bool = True):
        self.grid_mm, self.resolution, self.refine = grid_mm, resolution, refine

    @property
    def voxel_mm(self) -> float:
        return self.grid_mm / self.resolution

    def _starts(self, voxel01: np.ndarray):
        from training import rotation_symmetry as rs
        pts = np.argwhere(np.asarray(voxel01) > 0).astype(np.float64)
        axes = [np.eye(3)[:, i] for i in range(3)]
        E = rs.pca_orientation((np.asarray(voxel01) > 0).astype(np.uint8))
        if E is not None:
            axes += [E[:, i] for i in range(3)]
        for u in axes:
            F = _frame_with_axis(u)
            proj = pts @ F                                              # coordinates along the axis and across it
            lo, hi = proj.min(axis=0), proj.max(axis=0)
            half = (hi - lo + 1) / 2.0
            yield F, F @ ((lo + hi) / 2.0), float(half[1:].mean()), float(half[0])

    def estimate(self, voxel01: np.ndarray) -> Optional[CylinderEstimate]:
        """None when there is too little material to measure."""
        if (np.asarray(voxel01) > 0).sum() < 20:
            return None
        if not self.refine:
            F, c, r, L = next(iter(self._starts(voxel01)))
            best = (F, c, r, L, 0.0, 0.0)
        else:
            # coarse pass over all six axis hypotheses (a few dozen evaluations each), then the full descent for the two best only
            coarse = [(refine_cylinder(voxel01, F, c, r, L, max_evaluations=60), (F, c, r, L)) for F, c, r, L in self._starts(voxel01)]
            coarse.sort(key=lambda item: -item[0][5])
            best = None
            for (F1, c1, r1, L1, before, _after), _start in coarse[:2]:
                res = refine_cylinder(voxel01, F1, c1, r1, L1)
                res = (res[0], res[1], res[2], res[3], before, res[5])
                if best is None or res[5] > best[5]:
                    best = res
        F, c, r, L, before, after = best
        return CylinderEstimate(
            quaternion=axis_quaternion(F[:, 0]),
            parameters={"radius_ratio": r * self.voxel_mm / REALIZATION_SCALE_MM, "half_length_ratio": L * self.voxel_mm / REALIZATION_SCALE_MM},
            center_vox=np.asarray(c), axis=F[:, 0], radius_vox=r, half_length_vox=L, iou_before=before, iou_after=after, refined=self.refine)
