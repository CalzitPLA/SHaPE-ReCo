#!/usr/bin/env python3
"""
Robust alignment of a box-like body to the grid: the orientation of its FACES, not of its second moments.

Why not PCA. The principal axes of the occupied voxels are the box's axes only for an intact, generic box. A planar cut
or a spherical bite moves the centre of mass and mixes the moments, so PCA is off by 10-20 degrees on cut and carved
boxes (measured: median 0.8 deg plain, 18.6 deg cut, 20.6 deg carved). What survives a cut or a bite are the box's
remaining faces: their surface normals still cluster on three orthogonal axes, and the cut plane or the bite adds only
a few normals that agree with no axis.

Method (a "Manhattan frame" fit, robust because it is trimmed):
  1. surface normals from the gradient of the Gaussian-smoothed occupancy, on the boundary shell of the largest piece;
  2. from several starting frames (PCA and random rotations), iterate: assign every normal that lies within `tau` of an
     axis to it (both signs), solve the best rotation for that assignment (Procrustes / SVD), shrink `tau` (35 -> 7 deg);
     normals that fit no axis - cut faces, bites, speckle - stop counting;
  3. keep the start whose final frame explains most of the surface (weight within 8 deg of an axis);
  4. return the frame in the convention of `rotation_symmetry.pca_orientation`: columns = axes, longest extent first,
     right-handed. It is defined up to the box's symmetry group, exactly like the PCA frame.

Pure numpy / scipy; no learning. `frame_orientation(volume)` returns None when there is too little surface.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

TAUS_DEG = (35.0, 25.0, 15.0, 10.0, 7.0)          # inlier tolerance per round
SCORE_DEG = 8.0                                   # a normal within this angle of an axis counts for the frame
SIGMA = 1.2                                       # smoothing of the occupancy before taking normals (voxels)
MIN_SURFACE = 60                                  # fewer boundary voxels than this: nothing to align


def _largest_piece(occ: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    labels, n = ndimage.label(occ)
    if n <= 1:
        return occ
    sizes = ndimage.sum(occ, labels, index=range(1, n + 1))
    return labels == (1 + int(np.argmax(sizes)))


def surface_normals(volume: np.ndarray, sigma: float = SIGMA):
    """(points (n,3), unit outward normals (n,3)) on the boundary shell of the largest piece; array axes."""
    from scipy import ndimage
    occ = _largest_piece(volume > 0)
    if occ.sum() < MIN_SURFACE:
        return None, None
    field = ndimage.gaussian_filter(occ.astype(np.float32), sigma)
    gradient = np.stack(np.gradient(field), axis=-1)                      # (.., .., .., 3), points inward
    magnitude = np.linalg.norm(gradient, axis=-1)
    shell = (field > 0.15) & (field < 0.85) & (magnitude > 1e-3)
    points = np.argwhere(shell).astype(np.float64)
    if len(points) < MIN_SURFACE:
        return None, None
    normals = -gradient[shell] / magnitude[shell][:, None]
    return points, normals


def _procrustes(H: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    return u @ np.diag([1.0, 1.0, d]) @ vt


def fit_frame(normals: np.ndarray, start: np.ndarray, taus: Sequence[float] = TAUS_DEG) -> np.ndarray:
    """Trimmed alternating fit of an orthonormal frame (columns) to the normals, from `start`."""
    R = start
    for tau in taus:
        c = normals @ R                                                   # cosines with the three axes
        k = np.argmax(np.abs(c), axis=1)
        best = c[np.arange(len(c)), k]
        inlier = np.abs(best) >= np.cos(np.radians(tau))
        if inlier.sum() < 12:
            break
        H = np.zeros((3, 3))
        for axis in range(3):
            sel = inlier & (k == axis)
            H[:, axis] = (np.sign(best[sel])[:, None] * normals[sel]).sum(axis=0)
        R = _procrustes(H)
    return R


def frame_score(normals: np.ndarray, R: np.ndarray, deg: float = SCORE_DEG) -> float:
    """Share of the surface whose normal lies within `deg` of one of the frame's axes."""
    return float((np.abs(normals @ R).max(axis=1) >= np.cos(np.radians(deg))).mean())


def _random_rotations(rng: np.random.Generator, n: int):
    for _ in range(n):
        q = rng.normal(size=4)
        x, y, z, w = q / np.linalg.norm(q)
        yield np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _order_and_orient(R: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Longest extent first along the axes of R, right-handed (the pca_orientation convention)."""
    proj = (points - points.mean(axis=0)) @ R
    extent = proj.max(axis=0) - proj.min(axis=0)
    R = R[:, np.argsort(-extent, kind="stable")].copy()
    R[:, 2] = np.cross(R[:, 0], R[:, 1])
    return R


def frame_orientation(volume: np.ndarray, starts: int = 24, seed: int = 0, sigma: float = SIGMA) -> Optional[np.ndarray]:
    """Array-frame orientation of a box-like body from its face normals; None when there is too little surface.
    Deterministic for a given `seed`."""
    from training import rotation_symmetry as rs
    points, normals = surface_normals(volume, sigma)
    if points is None:
        return None
    candidates = list(_random_rotations(np.random.default_rng(seed), starts))
    pca = rs.pca_orientation(volume)
    if pca is not None:
        candidates.insert(0, pca)
    best, best_score = None, -1.0
    for start in candidates:
        R = fit_frame(normals, start)
        score = frame_score(normals, R)
        if score > best_score:
            best, best_score = R, score
    return _order_and_orient(best, points)
