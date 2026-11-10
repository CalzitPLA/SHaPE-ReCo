#!/usr/bin/env python3
"""
The statistics vector of a whole solution (docs/thesis/chapters/10b_solution_spaces.tex, def. "statistics of a
solution"): the numbers that compare programs with each other, all functions of the input's shape and the solution's
shape, none of how it was found.

    stats = solution_stats(result, occ_in, occ_model, mesh, h, grade=..., seconds=...)
    -> dict(iou, residual_pos, residual_neg, covered, excess_share, coverage, coverage_soft, position, orientation,
            score, chamfer, chamfer_max, valid, ops, seconds)

iou / covered / excess_share: on the loop's grid (ray-parity voxels, the ground truth). coverage: the share of the
part's surface area on the model's surface (strict, tolerance 2e-3 x diagonal). coverage_soft, position, orientation:
the band terms at scale h - every triangle weighted by (1 - (d/2h)^2)+ x (n_part . n_model)+, and the two terms
separately: position = mean over the part's surface of min(1, (d/2h)^2), orientation = mean of (1 - n_part . n_model)
(0 = every normal agrees). score = covered x coverage (the objective of 2026-09-26).

The vector is measured on the loop's grid and the part's mesh, never by OCCT: its `iou` is a voxel IoU, blind below
the grid's pitch (the frame on the cube grid: residual 0 / 0 at an exact IoU of 0.910, docs/FRAME_FAILURE.md); the exact
IoU is in the grade (residual_loop._exact_grade, ReconstructionResult.exact_iou). When the surface terms fail, the error
is kept in `surface_error` and `score` becomes 0 (coverage missing), with no other sign.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from reco_engine.result_schema import ReconstructionResult
from reco_engine.result_to_program import program_model


def surface_terms(model_sdf, mesh: Dict, h: float, tol_rel: float = 2e-3, max_triangles: int = 60000) -> Dict:
    """coverage, coverage_soft, position and orientation of an SDF model (anything with .sdf(points)) against the
    part's surface triangles (mesh: vertices, triangles)."""
    V, T = np.asarray(mesh["vertices"], float), np.asarray(mesh["triangles"])
    if len(T) > max_triangles:
        T = T[np.random.default_rng(0).choice(len(T), size=max_triangles, replace=False)]
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    tri = V[T]
    raw_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(raw_n, axis=1)
    total = max(float(area.sum()), 1e-12)
    cent = tri.mean(axis=1)
    d = np.asarray(model_sdf.sdf(cent), float)
    eps = 1e-3 * diag
    grad = np.stack([(np.asarray(model_sdf.sdf(cent + eps * e), float) - np.asarray(model_sdf.sdf(cent - eps * e), float)) / (2 * eps)
                     for e in np.eye(3)], axis=1)
    n_m = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
    n_t = raw_n / np.maximum(2.0 * area[:, None], 1e-12)
    agree = np.einsum("ij,ij->i", n_t, n_m)
    w_pos = np.clip(1.0 - (d / (2.0 * h)) ** 2, 0.0, None)
    coverage = float(area[np.abs(d) <= tol_rel * diag].sum() / total)
    return dict(coverage=coverage,
                coverage_soft=float((area * w_pos * np.clip(agree, 0.0, None)).sum() / total),
                position=float((area * np.minimum(1.0, (d / (2.0 * h)) ** 2)).sum() / total),
                orientation=float((area * (1.0 - agree)).sum() / total))


def solution_stats(result: ReconstructionResult, occ_in: np.ndarray, occ_model: np.ndarray, mesh: Optional[Dict],
                   h: float, grade: Optional[Dict] = None, seconds: Optional[float] = None) -> Dict:
    """The statistics vector of `result` (its accepted operations) - see the module docstring."""
    a, b = np.asarray(occ_in, bool), np.asarray(occ_model, bool)
    n_in, n_model = int(a.sum()), int(b.sum())
    common, union = int((a & b).sum()), int((a | b).sum())
    out = dict(iou=(common / union) if union else 0.0,
               residual_pos=int((a & ~b).sum()), residual_neg=int((b & ~a).sum()),
               covered=(common / n_model) if n_model else 0.0,
               excess_share=((n_model - common) / n_in) if n_in else None,
               ops=len([o for o in result.operations if o.accepted]),
               valid=(bool(grade.get("valid")) if grade else None),
               chamfer=(grade.get("chamfer") if grade else None),
               chamfer_max=(grade.get("chamfer_max") if grade else None),
               seconds=seconds)
    if mesh is not None:
        try:
            model = program_model(result)
            if model is not None:
                out.update(surface_terms(model, mesh, h))
        except Exception as exc:                                       # the statistics never break a run
            out["surface_error"] = str(exc)[-200:]
    out["score"] = float(out["covered"] * out.get("coverage", 0.0))
    return out
