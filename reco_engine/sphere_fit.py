#!/usr/bin/env python3
"""
Centre and radius of a SPHERE from a voxel grid: the analytic counterpart of reco_engine.box_orientation for the sphere
primitive, in the in-scale voxel mode (SHAPE_RECO_VOXEL_MODE=fixed).

The old Dimension CNN answers a sphere's radius in the stretched frame, where absolute size does not exist, so the
candidate was built at an arbitrary size and then stretched to the target's bounding box - loose on any target that is
not a whole sphere (a sphere cut flat, a sphere with a bite): measured surface F-score 0.55-0.57 on the two sphere parts
against 0.94-1.0 for the refined boxes. Here the sphere is fitted directly: start from the bounding box (centre, mean
half extent) and move centre and radius by coordinate descent on the IoU between the analytic sphere and the target
(voxel centres within the radius are inside), shrinking steps from 1 voxel to a quarter voxel. A sphere has no orientation, so
the quaternion is the identity. Needs a start within a few voxels: a target that is not roughly a sphere is fitted as
well as a sphere can be, and the loop's grade (reco_engine/fit_grade.py) decides whether that is good enough.

    fitter = SphereFitter(grid_mm=40.0)
    est = fitter.estimate(voxel01)          # est.parameters == {"radius_ratio": ...}, est.center_vox, est.radius_vox
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

REALIZATION_SCALE_MM = 10.0        # dataset_gen sphere macro: radius mm = radius_ratio * 10


@dataclass
class SphereEstimate:
    quaternion: List[float]
    parameters: Dict[str, float]
    center_vox: np.ndarray
    radius_vox: float
    iou_before: float = 0.0
    iou_after: float = 0.0
    refined: bool = True


def refine_sphere(target: np.ndarray, center, radius: float, max_evaluations: int = 400):
    """Coordinate descent on the IoU between the analytic sphere and `target`. Returns (center, radius, iou_before, iou_after)."""
    occ = target > 0
    pts = np.argwhere(occ)
    if len(pts) < 20:
        return np.asarray(center, dtype=np.float64), float(radius), 0.0, 0.0
    lo = np.maximum(pts.min(axis=0) - 4, 0)
    hi = np.minimum(pts.max(axis=0) + 5, target.shape)
    truth = occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].reshape(-1)
    grid = np.stack(np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    n_truth = int(truth.sum())

    def iou(c, r):
        inside = ((grid - c) ** 2).sum(axis=1) <= r * r
        inter = int(np.count_nonzero(inside & truth))
        return inter / max(int(np.count_nonzero(inside)) + n_truth - inter, 1)

    c, r = np.array(center, dtype=np.float64), float(radius)
    before = best = iou(c, r)
    evals = 1
    for step in (1.0, 0.5, 0.25):
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
                    score = iou(c2, r2)
                    evals += 1
                    if score > best + 1e-4:
                        c, r, best, improved = c2, r2, score, True
    return c, r, before, best


class SphereFitter:
    def __init__(self, grid_mm: float = 40.0, resolution: int = 64, refine: bool = True):
        self.grid_mm, self.resolution, self.refine = grid_mm, resolution, refine

    @property
    def voxel_mm(self) -> float:
        return self.grid_mm / self.resolution

    def estimate(self, voxel01: np.ndarray) -> Optional[SphereEstimate]:
        """None when there is too little material to measure."""
        pts = np.argwhere(np.asarray(voxel01) > 0)
        if len(pts) < 20:
            return None
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
        radius = float(((pts.max(axis=0) - pts.min(axis=0) + 1) / 2.0).mean())
        before = after = 0.0
        if self.refine:
            center, radius, before, after = refine_sphere(voxel01, center, radius)
        return SphereEstimate(
            quaternion=[0.0, 0.0, 0.0, 1.0], parameters={"radius_ratio": radius * self.voxel_mm / REALIZATION_SCALE_MM},
            center_vox=np.asarray(center), radius_vox=radius, iou_before=before, iou_after=after, refined=self.refine)
