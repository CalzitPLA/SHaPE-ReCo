#!/usr/bin/env python3
"""
Grading the quality of a fit: a reconstruction (or an assembly under construction) against the input, both voxel grids.

Why not just IoU
----------------
Voxel IoU (reco_engine.compare) is what the loop and the reports used. It has four problems, measured in
scripts/grading_study.py:

  * It cannot reach 1 for a correct answer. A shape re-voxelised after a half-voxel shift or a 1 degree tilt - identical
    geometry, different sampling - scores 0.85-1.0 (median 0.93) on the 45-voxel test parts, while a real defect such
    as an unfilled pocket scores 0.91-0.97: the two ranges overlap, so a 0.03 difference between two runs means nothing.
  * It weighs a one-voxel boundary offset like a missing feature: a box that is 1 voxel too big all round loses
    as much as the same volume of pocket left unexplained.
  * It says HOW MUCH is wrong but not WHAT: missing material, spurious material, where, how big.
  * It is asymmetric in what the loop looks at: acceptance saw only the unexplained fraction of the input (false
    negatives) and the candidate's recall against a local target; spill (false positives) was left to the cut phase
    and to a soft penalty in the local search. A bounding box with 30 % false positives was accepted.

The grade
---------
A tolerance `tol` (in voxels, chessboard metric; 1 = the resolution of the data) defines what counts as the same place:

  fn     reference material farther than `tol` from the prediction ("missing")
  fp     predicted material farther than `tol` from the reference ("spurious")
  error  (fn + fp) / |reference|                     0 = every voxel is within tolerance, comparable across parts
  defects  connected pieces of fn / fp that survive a one-voxel opening: each with its volume, its share of the
           reference and where it is - the list a person or a later stage can act on
  surface  symmetric surface distances (Chamfer, 95th percentile, F-score at `tol`), in mm when `voxel_mm` is known

`raw` keeps the plain IoU / precision / recall for continuity. Everything is voxel-only (scipy distance transforms on
a 64^3 grid: a few ms), needs no FreeCAD, and is deterministic.

`assembly_error` / `Grader` use the same measure INSIDE the loop: a candidate is accepted when the grade of the
assembly that results from adding it (cover) or subtracting it (cut) beats the current one by more than the price of one
more primitive - a global, two-sided criterion in place of the candidate's local recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class GradeConfig:
    tol_voxels: float = 1.0          # distance that still counts as "the same place"
    open_radius: int = 1             # defects thinner than 2 * radius + 1 voxels are boundary noise, not features
    min_defect_voxels: int = 20      # smaller pieces are ignored in the defect list (they still count in fn / fp)
    voxel_mm: Optional[float] = None


@dataclass
class Defect:
    kind: str                        # "missing" (reference not explained) or "spurious" (prediction not in the reference)
    voxels: int
    fraction_of_reference: float
    centroid_vox: List[float]
    bbox_vox: List[List[int]]        # [[min...], [max...]] in array order
    volume_mm3: Optional[float] = None

    def __str__(self) -> str:
        size = f"{self.fraction_of_reference:.1%} of the part"
        if self.volume_mm3 is not None:
            size += f", {self.volume_mm3:.0f} mm3"
        return f"{self.kind} {self.voxels} voxels ({size}) around {[round(c) for c in self.centroid_vox]}"


@dataclass
class FitGrade:
    iou: float
    precision: float
    recall: float
    fn_voxels: int
    fp_voxels: int
    reference_voxels: int
    error: float
    chamfer: float                   # mean symmetric surface distance, voxels (mm when voxel_mm was given: chamfer_mm)
    hd95: float
    fscore: float                    # surface F-score at the tolerance
    defects: List[Defect] = field(default_factory=list)
    chamfer_mm: Optional[float] = None
    hd95_mm: Optional[float] = None

    @property
    def missing(self) -> List[Defect]:
        return [d for d in self.defects if d.kind == "missing"]

    @property
    def spurious(self) -> List[Defect]:
        return [d for d in self.defects if d.kind == "spurious"]

    def summary(self) -> str:
        head = (f"error {self.error:.3f} (missing {self.fn_voxels}, spurious {self.fp_voxels} voxels beyond tolerance), "
                f"IoU {self.iou:.3f}, surface F {self.fscore:.3f}, Chamfer {self.chamfer:.2f} vox")
        return head + (f"; {len(self.defects)} defect(s): " + "; ".join(str(d) for d in self.defects) if self.defects else "; no defects")


# ------------------------------------------------------------------------------------------------ primitives

def _edt_within(mask: np.ndarray, tol: float) -> np.ndarray:
    """Voxels within `tol` of `mask` (mask included), in the chessboard metric: a one-voxel tolerance forgives a one-voxel
    shell all round, corners included (a Euclidean ball would leave the corner voxels of a slightly larger box)."""
    from scipy import ndimage
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return ndimage.distance_transform_cdt(~mask, metric="chessboard") <= tol


def _surface(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


def tolerant_sets(prediction: np.ndarray, reference: np.ndarray, tol: float):
    """(fn, fp): reference voxels beyond `tol` from the prediction, predicted voxels beyond `tol` from the reference."""
    p, r = prediction > 0, reference > 0
    return r & ~_edt_within(p, tol), p & ~_edt_within(r, tol)


def _defects(region: np.ndarray, kind: str, config: GradeConfig, reference_voxels: int) -> List[Defect]:
    from scipy import ndimage
    if config.open_radius > 0:
        side = 2 * config.open_radius + 1
        region = ndimage.binary_opening(region, structure=np.ones((side,) * 3, dtype=bool))
    labels, count = ndimage.label(region)
    out = []
    for k in range(1, count + 1):
        piece = labels == k
        n = int(piece.sum())
        if n < config.min_defect_voxels:
            continue
        idx = np.argwhere(piece)
        out.append(Defect(
            kind=kind, voxels=n, fraction_of_reference=n / max(reference_voxels, 1),
            centroid_vox=[float(v) for v in idx.mean(axis=0)], bbox_vox=[idx.min(axis=0).tolist(), idx.max(axis=0).tolist()],
            volume_mm3=(n * config.voxel_mm ** 3) if config.voxel_mm else None))
    return sorted(out, key=lambda d: -d.voxels)


def surface_distances(prediction: np.ndarray, reference: np.ndarray):
    """(distances from the reference surface to the prediction surface, and the other way round), in voxels."""
    from scipy import ndimage
    ps, rs = _surface(prediction > 0), _surface(reference > 0)
    if not ps.any() or not rs.any():
        return np.array([np.inf]), np.array([np.inf])
    to_pred = ndimage.distance_transform_edt(~ps)
    to_ref = ndimage.distance_transform_edt(~rs)
    return to_pred[rs], to_ref[ps]


def grade(prediction: np.ndarray, reference: np.ndarray, config: GradeConfig = None) -> FitGrade:
    """Grade `prediction` against `reference` (both {0,1} grids of the same shape)."""
    config = config or GradeConfig()
    p, r = prediction > 0, reference > 0
    n_ref, n_pred = int(r.sum()), int(p.sum())
    inter = int((p & r).sum())
    union = n_ref + n_pred - inter
    fn, fp = tolerant_sets(p, r, config.tol_voxels)
    d_ref_to_pred, d_pred_to_ref = surface_distances(p, r)
    tau = config.tol_voxels
    finite = np.isfinite(d_ref_to_pred).all() and np.isfinite(d_pred_to_ref).all()
    if finite:
        chamfer = float((d_ref_to_pred.mean() + d_pred_to_ref.mean()) / 2.0)
        hd95 = float(max(np.percentile(d_ref_to_pred, 95), np.percentile(d_pred_to_ref, 95)))
        rec_s, pre_s = float((d_ref_to_pred <= tau).mean()), float((d_pred_to_ref <= tau).mean())
        fscore = 2 * rec_s * pre_s / (rec_s + pre_s) if rec_s + pre_s > 0 else 0.0
    else:
        chamfer = hd95 = float("inf")
        fscore = 0.0
    g = FitGrade(
        iou=inter / union if union else 0.0, precision=inter / n_pred if n_pred else 0.0, recall=inter / n_ref if n_ref else 0.0,
        fn_voxels=int(fn.sum()), fp_voxels=int(fp.sum()), reference_voxels=n_ref,
        error=(int(fn.sum()) + int(fp.sum())) / max(n_ref, 1), chamfer=chamfer, hd95=hd95, fscore=fscore,
        defects=_defects(fn, "missing", config, n_ref) + _defects(fp, "spurious", config, n_ref))
    if config.voxel_mm and finite:
        g.chamfer_mm, g.hd95_mm = chamfer * config.voxel_mm, hd95 * config.voxel_mm
    return g


def assembly_error(prediction: np.ndarray, reference: np.ndarray, tol: float = 1.0) -> float:
    """Just the tolerance-aware error (fn + fp) / |reference|: the cheap form the loop evaluates per candidate."""
    fn, fp = tolerant_sets(prediction, reference, tol)
    return (int(fn.sum()) + int(fp.sum())) / max(int((reference > 0).sum()), 1)


# ------------------------------------------------------------------------------------------------ in the loop

@dataclass
class Grader:
    """Accept/reject by the grade of the ASSEMBLY, not by a candidate's local recall.

    reference   the input the assembly should equal
    state       the assembly so far (cover phase: starts empty; cut phase: starts as the built union)
    mode        "add" - a candidate is unioned into the state; "subtract" - it is carved out of it
    tol         tolerance in voxels
    shape_cost  what one more primitive must gain, in units of `error` (default 0.005: half a percent of the part); the
                loop also stops when the error itself is below this - no candidate could pay for itself
    """
    reference: np.ndarray
    state: np.ndarray
    mode: str = "add"
    tol: float = 1.0
    shape_cost: float = 0.005

    def __post_init__(self):
        if self.mode not in ("add", "subtract"):
            raise ValueError("mode must be 'add' or 'subtract'")
        self.reference = self.reference > 0
        self.state = self.state > 0
        self.error = assembly_error(self.state, self.reference, self.tol)

    def trial(self, candidate: np.ndarray) -> np.ndarray:
        c = candidate > 0
        return (self.state | c) if self.mode == "add" else (self.state & ~c)

    def evaluate(self, candidate: np.ndarray):
        """(error after the candidate, gain = error before - error after - shape_cost)."""
        error = assembly_error(self.trial(candidate), self.reference, self.tol)
        return error, self.error - error - self.shape_cost

    def accept(self, candidate: np.ndarray) -> None:
        self.state = self.trial(candidate)
        self.error = assembly_error(self.state, self.reference, self.tol)
