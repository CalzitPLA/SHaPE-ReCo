#!/usr/bin/env python3
"""
Orientation of a BOX in a voxel grid, with its symmetries handled explicitly.

Why this module exists
----------------------
A box is mapped onto itself by rotations, so a voxel grid does not determine "the" quaternion of a
sample. Worse, the sample's dimensions are drawn independently of its rotation, so (rotation, dimensions)
and (rotation * g, dimensions permuted by g) describe the same body for every g in the 24-element cube
rotation group O. A network that regresses the raw label can only learn the mean of these equivalent
answers (which is not a valid answer). What is learnable is:

  canonical frame    body axes ordered by their dimension, longest first (x >= y >= z); this removes
                     the axis permutation. What is left is the symmetry group of the box in that frame:
                     the 4 half-turns {I, Rx(pi), Ry(pi), Rz(pi)} (D2) for a generic box, 8 elements when two
                     dimensions are (nearly) equal, 24 for a cube.
  array frame        the voxel array is indexed [a0, a1, a2] = [z, -y, x] (a1 runs backwards), i.e.
                     a = A @ world with A a proper rotation (det +1). A body whose world orientation is R
                     has, as columns of M = A @ R, its axes as they lie in the array; rotating the array by a
                     grid rotation g (a signed permutation of the axes) turns M into g @ M.
  symmetry-aware     min over the admissible group elements of the geodesic angle between the prediction
  loss and metric    and M @ g.

Everything here is pure numpy / torch and independent of the FreeCAD pipeline.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# world -> array axes: [a0, a1, a2] = [z, -y, x]
A_WORLD_TO_ARRAY = np.array([[0.0, 0.0, 1.0],
                             [0.0, -1.0, 0.0],
                             [1.0, 0.0, 0.0]])
DIM_KEYS = ("x_half_ratio", "y_half_ratio", "z_half_ratio")
TIE_TOLERANCE = 0.10        # two dimensions closer than this (relative) are treated as equal: 3 voxels of 30


def record_dims(record) -> List[float]:
    """The box's three dimensions in local x, y, z order, for ordering and ties only (any common scale): the
    generator's ratio parameters, or the world half extents `dims_mm` of records that carry no ratios (v5 I-beam
    plates, envelopes, negative residuals)."""
    params = getattr(record, "parameters", None) or {}
    if all(k in params for k in DIM_KEYS):
        return [float(params[k]) for k in DIM_KEYS]
    d = (getattr(record, "meta", None) or {}).get("dims_mm") or {}
    return [float(d["x_half"]), float(d["y_half"]), float(d["z_half"])]


def cube_group() -> np.ndarray:
    """The 24 proper signed permutation matrices, shape (24, 3, 3). Index 0 is the identity."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            m = np.zeros((3, 3))
            for col, (row, s) in enumerate(zip(perm, signs)):
                m[row, col] = s
            if np.linalg.det(m) > 0:
                mats.append(m)
    mats.sort(key=lambda m: -np.trace(m))                   # identity (trace 3) first
    return np.stack(mats)


GROUP = cube_group()


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """[qx, qy, qz, qw] -> rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(m: np.ndarray) -> List[float]:
    """Rotation matrix -> [qx, qy, qz, qw] with qw >= 0."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(m).as_quat()
    return [float(v) for v in (q if q[3] >= 0 else -q)]


def canonical_axes(dims: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """(order, sorted dims): body axes by descending dimension (stable: ties keep x, y, z order)."""
    dims = np.asarray(dims, dtype=np.float64)
    order = np.array(sorted(range(3), key=lambda i: (-dims[i], i)))
    return order, dims[order]


def canonical_orientation(world_matrix: np.ndarray, dims: Sequence[float]) -> np.ndarray:
    """World rotation of a box with body-frame dimensions `dims` -> the array-frame orientation M of its
    canonical (longest-first) axes. The third axis' sign is fixed so that det M = +1; the box is mirror
    symmetric, so this describes the same body."""
    order, _ = canonical_axes(dims)
    m = A_WORLD_TO_ARRAY @ world_matrix[:, order]
    if np.linalg.det(m) < 0:
        m[:, 2] *= -1.0
    return m


def array_orientation_to_world_quaternion(m: np.ndarray) -> List[float]:
    """Inverse of `canonical_orientation` on the frame side: array-frame orientation M of the longest-first
    axes -> [qx, qy, qz, qw] of the world rotation whose columns are those axes (build the box with its
    dimensions in longest-first order, x >= y >= z, then apply this rotation)."""
    return matrix_to_quat(A_WORLD_TO_ARRAY.T @ m)


def admissible_group(sorted_dims: Sequence[float], tol: float = TIE_TOLERANCE) -> np.ndarray:
    """Boolean (24,) over `GROUP`: which g map the box onto itself, given its dimensions in canonical order.
    g is admissible when it only exchanges axes of (nearly) equal dimension."""
    d = np.asarray(sorted_dims, dtype=np.float64)
    ok = np.zeros(len(GROUP), dtype=bool)
    for k, g in enumerate(GROUP):
        good = True
        for col in range(3):
            row = int(np.argmax(np.abs(g[:, col])))
            if row != col and abs(d[row] - d[col]) > tol * max(d[row], d[col]):
                good = False
        ok[k] = good
    return ok


def record_target(quaternion: Sequence[float], dims: Sequence[float], tol: float = TIE_TOLERANCE):
    """(M, admissible mask) for one dataset record: `quaternion` = [qx, qy, qz, qw] world rotation of the
    macro-built shape, `dims` = its (x, y, z) half-extent ratios."""
    order, sd = canonical_axes(dims)
    return canonical_orientation(quat_to_matrix(quaternion), dims), admissible_group(sd, tol)


def geodesic_angle(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Angle (radians) of the rotation m1^T m2, broadcasting over leading dims."""
    tr = np.einsum("...ji,...ji->...", m1, m2)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def symmetric_error(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Degrees: min over admissible g of the angle between pred (N,3,3) and target @ g. (N,) result."""
    tg = np.einsum("nij,kjl->nkil", target, GROUP)                      # (N, 24, 3, 3)
    ang = geodesic_angle(pred[:, None], tg)                             # (N, 24)
    ang = np.where(mask, ang, np.inf)
    return np.degrees(ang.min(axis=1))


def symmetric_loss_torch(pred, target, mask, eps: float = 1e-6):
    """Mean over the batch of min over admissible g of the geodesic angle (radians) between pred (B,3,3)
    and target (B,3,3) @ g. `mask` is (B, 24) bool."""
    import torch
    group = torch.as_tensor(GROUP, dtype=pred.dtype, device=pred.device)      # (24, 3, 3)
    tg = torch.einsum("bij,kjl->bkil", target, group)                          # (B, 24, 3, 3)
    tr = torch.einsum("bji,bkji->bk", pred, tg)
    ang = torch.acos(((tr - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps))
    ang = ang.masked_fill(~mask, float("inf"))
    return ang.min(dim=1).values.mean()


def sixd_to_matrix_torch(x):
    """(B, 6) -> (B, 3, 3) by Gram-Schmidt (Zhou et al. 2019): the two vectors are the first two columns."""
    import torch
    import torch.nn.functional as F
    a1, a2 = x[:, 0:3], x[:, 3:6]
    b1 = F.normalize(a1, dim=1, eps=1e-8)
    b2 = F.normalize(a2 - (b1 * a2).sum(1, keepdim=True) * b1, dim=1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=2)


def quat_to_matrix_torch(q):
    """(B, 4) [qx, qy, qz, qw] -> (B, 3, 3)."""
    import torch
    import torch.nn.functional as F
    x, y, z, w = F.normalize(q, dim=1, eps=1e-8).unbind(1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1),
    ], 1)


# ---------------------------------------------------------------------------- grid rotations

def rotate_volume(volume: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Rotate a cubic array about its centre by the signed permutation matrix `g` (acting on the array's
    centred index coordinates): a voxel at centred coordinates c ends up at g @ c. An orientation M in the
    array frame becomes g @ M."""
    perm = [int(np.argmax(np.abs(g[i]))) for i in range(3)]             # new axis i takes old axis perm[i]
    out = np.transpose(volume, perm)
    for i in range(3):
        if g[i, perm[i]] < 0:
            out = np.flip(out, axis=i)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------- classical baseline

def pca_orientation(volume: np.ndarray, keep_largest_component: bool = True) -> Optional[np.ndarray]:
    """Array-frame orientation from the second moments of the occupied voxels: principal axes, longest
    first, right-handed. It is exact for a generic box, undefined for a cube (isotropic moments; any
    result is arbitrary there), and biased by cuts. Isolated noise voxels are dropped by keeping the
    largest connected component. None when there is nothing to measure."""
    occ = volume > 0
    if keep_largest_component and occ.any():
        from scipy import ndimage
        labels, n = ndimage.label(occ)
        if n > 1:
            sizes = ndimage.sum(occ, labels, index=range(1, n + 1))
            occ = labels == (1 + int(np.argmax(sizes)))
    pts = np.argwhere(occ).astype(np.float64)
    if len(pts) < 10:
        return None
    vals, vecs = np.linalg.eigh(np.cov((pts - pts.mean(axis=0)).T))
    vecs = vecs[:, ::-1].copy()                                          # descending variance
    if np.linalg.det(vecs) < 0:
        vecs[:, 2] *= -1.0
    return vecs


# ---------------------------------------------------------------------------- canonicalisation

def _centroid_of_body(volume: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    occ = volume > 0
    labels, n = ndimage.label(occ)
    if n > 1:
        sizes = ndimage.sum(occ, labels, index=range(1, n + 1))
        occ = labels == (1 + int(np.argmax(sizes)))
    return np.argwhere(occ).mean(axis=0)


def resample_rotated(volume: np.ndarray, rotation: np.ndarray, source_centre: Optional[np.ndarray] = None) -> np.ndarray:
    """Resample so that output[o] = input[source_centre + rotation @ (o - c)] with c the grid centre
    (`source_centre` defaults to c). A body with orientation M in `volume` therefore has orientation
    rotation^T @ M in the output, and is centred at c when `source_centre` was its centroid.
    Trilinear on the {0, 1} array, thresholded at 0.5."""
    from scipy import ndimage
    n = volume.shape[0]
    centre = np.full(3, (n - 1) / 2.0)
    src = centre if source_centre is None else np.asarray(source_centre, dtype=np.float64)
    out = ndimage.affine_transform((volume > 0).astype(np.float32), rotation, offset=src - rotation @ centre,
                                   order=1, mode="constant", cval=0.0)
    return (out > 0.5).astype(np.uint8)


def align_volume(volume: np.ndarray, method: str = "pca"):
    """(aligned volume, E) with E the estimated orientation (or the identity when there is nothing to measure): the
    body's axes lie along the grid axes in the output and its centroid at the grid centre. An orientation M in
    `volume` is E^T @ M in the aligned volume - close to a symmetry element for a box the method finds, larger where
    it errs.

    method "pca": second-moment principal axes (`pca_orientation`) - exact for an intact generic box, 15-20 degrees off
        for cut or carved ones. Default, and what the trained residual networks were built on.
    method "frame": the orientation of the box's faces, from the surface normals (training/box_alignment.py) - robust
        to planar cuts, bites and speckle (measured: median 1.3 degrees on plain, cut and carved boxes). A body
        with too little surface for normals falls back to the PCA frame."""
    if method == "pca":
        E = pca_orientation(volume)
    elif method == "frame":
        from training.box_alignment import frame_orientation
        E = frame_orientation(volume)
        if E is None:                                   # too little surface for normals: fall back to the moments
            E = pca_orientation(volume)
    else:
        raise ValueError(f"align method must be 'pca' or 'frame', got {method!r}")
    if E is None:
        return (volume > 0).astype(np.uint8), np.eye(3)
    return resample_rotated(volume, E, _centroid_of_body(volume)), E


def small_rotation(rng: np.random.Generator, max_degrees: float) -> np.ndarray:
    """A random rotation by an angle uniform in [0, max_degrees] about a uniformly random axis."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.uniform(0.0, max_degrees))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def measured_half_extents(volume: np.ndarray) -> np.ndarray:
    """Half extents (voxels), longest first, of the largest connected piece along the grid axes: exact for a box
    already aligned with the grid (plain box), an underestimate when material was cut away, and blurred by about
    a voxel on the thin axis. The classical counterpart of the dimension network."""
    from scipy import ndimage
    occ = volume > 0
    labels, n = ndimage.label(occ)
    if n > 1:
        sizes = ndimage.sum(occ, labels, index=range(1, n + 1))
        occ = labels == (1 + int(np.argmax(sizes)))
    pts = np.argwhere(occ)
    return np.sort((pts.max(axis=0) - pts.min(axis=0) + 1).astype(float))[::-1] / 2.0
