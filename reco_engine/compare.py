#!/usr/bin/env python3
"""Voxel-level boolean operations and IoU, used to score a reconstructed
primitive against the (residual) input and to carve it out for the next
iteration of the decomposition loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BooleanMetrics:
    iou: float
    precision: float
    recall: float


def as_bool(voxel: np.ndarray) -> np.ndarray:
    """`voxel` as a bool grid - no copy when it already is one."""
    return voxel if voxel.dtype == np.bool_ else voxel.astype(bool)


def metrics_bool(a: np.ndarray, b: np.ndarray, a_count: int) -> BooleanMetrics:
    """compare() for bool grids when |a| is already known - what an
    inner loop that scores many candidates against ONE target wants
    (reco_engine.local_search does ~64 of these per candidate). The
    union comes from inclusion-exclusion rather than another pass over
    the grid; every count is an exact integer, so the result is
    identical to the straightforward computation."""
    b_count = int(np.count_nonzero(b))
    intersection = int(np.count_nonzero(a & b))
    union = a_count + b_count - intersection
    return BooleanMetrics(
        iou=float(intersection / union) if union > 0 else 0.0,
        precision=float(intersection / b_count) if b_count > 0 else 0.0,
        recall=float(intersection / a_count) if a_count > 0 else 0.0,
    )


def compare(input_voxel: np.ndarray, candidate_voxel: np.ndarray) -> BooleanMetrics:
    """Both arrays are expected as {0, 1} (or bool) occupancy grids of the
    same shape - NOT the {-1, +1}-normalized tensors used for CNN input.
    """
    a = as_bool(input_voxel)
    return metrics_bool(a, as_bool(candidate_voxel), int(np.count_nonzero(a)))


def subtract(residual_voxel: np.ndarray, candidate_voxel: np.ndarray) -> np.ndarray:
    """residual_voxel - candidate_voxel, as {0, 1} uint8."""
    a = residual_voxel.astype(bool)
    b = candidate_voxel.astype(bool)
    return np.logical_and(a, np.logical_not(b)).astype(np.uint8)
