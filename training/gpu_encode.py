#!/usr/bin/env python3
"""
The grid-derived network inputs (occupancy, dirlog, dirlog_inside) computed ON THE GPU for a whole batch, so the loaders
only ship uint8 grids. Same definitions as training/field_input.py (dirlog_channels / dirlog_inside_channels / encode),
whose constants are imported, not copied:

  sdf     exact Euclidean distance transform, both sides (field_input._signed_distance: -(d_in - 0.5) inside,
          d_out - 0.5 outside). Separable exact squared EDT, one axis after the other, each a minimum over the R
          candidates. The squared distances are integers <= 3 R^2, which float32 represents exactly, so the transform is
          exact.
  normal  unit gradient of the Gaussian-smoothed sdf: scipy.ndimage.gaussian_filter semantics (kernel radius
          int(4 sigma + 0.5), half-sample symmetric boundary) and np.gradient's edge rule (central inside, one-sided at
          the faces).
  d_log   field_input.log_distance of the clipped sdf.
  inside  channel 5 of dirlog_inside: the stored base-point bits (dataset_gen/src/inside_bits), else the sample's
          .clean.bin, else the grid itself - the order of training/voxel_dataset.py.

Why: on the CPU, dirlog costs about 374 ms per sample (two scipy EDTs 212 ms) in EVERY epoch; here a batch of 16 takes
about 7-9 ms per sample (shared GPU), and the host->GPU copy shrinks from 5-6 float32 channels to one uint8 grid (+ a packed bit
grid). Differences from the CPU path: material, inside and d_log are identical (d_log to float rounding); the normal
differs by up to about 2e-3 where the smoothed gradient nearly vanishes (float32 here, float64 in scipy).

    x = encode_batch(grids_uint8 (B,R,R,R) on device, "dirlog_inside", inside=bits (B,R,R,R))   # (B,6,R,R,R) float32
    grids, inside, g = rotate_batch(grids, inside, rng)          # the 24 grid rotations, BEFORE encoding (equivariant)
    grid_u8, inside_u8 = load_raw(record, resolution)            # host side, for a DataLoader
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from training.field_input import DEFAULT_LOG_SCALE, DEFAULT_SIGMA

GPU_KINDS = ("occupancy", "dirlog", "dirlog_inside")
CHANNELS = {"occupancy": 1, "dirlog": 5, "dirlog_inside": 6}


def _sq_edt(feature: torch.Tensor) -> torch.Tensor:
    """Exact squared Euclidean distance (in voxels) of every voxel to the nearest True voxel of `feature` (B,R,R,R);
    grids without any feature voxel get a large value (the caller replaces them, as field_input._signed_distance does).
    Per axis and sample one broadcast minimum over the R candidates: R^4 floats (64 MB at R = 64) at a time; measured
    twice as fast as a running minimum (64 small launches per axis), with identical results."""
    big = float(4 * max(feature.shape[1:]) ** 2)
    f = torch.where(feature, torch.zeros((), device=feature.device), torch.full((), big, device=feature.device))
    for axis in (1, 2, 3):
        g = f.movedim(axis, -1)                                   # (B, ..., R): minimise along the last axis
        n = g.shape[-1]
        pos = torch.arange(n, device=f.device, dtype=f.dtype)
        d2 = (pos[:, None] - pos[None, :]) ** 2                   # (x, y)
        out = torch.empty_like(g)
        for b in range(g.shape[0]):
            out[b] = (g[b][..., None, :] + d2).amin(-1)
        f = out.movedim(-1, axis)
    return f


def _gaussian_kernel(sigma: float, truncate: float = 4.0, device=None) -> torch.Tensor:
    r = int(truncate * float(sigma) + 0.5)                       # scipy.ndimage._gaussian_kernel1d
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float64)
    k = torch.exp(-0.5 / float(sigma) ** 2 * x ** 2)
    return (k / k.sum()).float()


def _symmetric_pad(x: torch.Tensor, p: int) -> torch.Tensor:
    """scipy mode 'reflect' (d c b a | a b c d) on the last three dims."""
    for d in (-3, -2, -1):
        n = x.shape[d]
        lo, hi = x.narrow(d, 0, p).flip(d), x.narrow(d, n - p, p).flip(d)
        x = torch.cat([lo, x, hi], dim=d)
    return x


def _gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    k = _gaussian_kernel(sigma, device=x.device)
    r = (len(k) - 1) // 2
    y = _symmetric_pad(x[:, None], r)
    y = F.conv3d(y, k.view(1, 1, -1, 1, 1))
    y = F.conv3d(y, k.view(1, 1, 1, -1, 1))
    y = F.conv3d(y, k.view(1, 1, 1, 1, -1))
    return y[:, 0]


def _np_gradient(x: torch.Tensor, dim: int) -> torch.Tensor:
    n = x.shape[dim]
    g = torch.empty_like(x)
    g.narrow(dim, 1, n - 2).copy_((x.narrow(dim, 2, n - 2) - x.narrow(dim, 0, n - 2)) / 2)
    g.narrow(dim, 0, 1).copy_(x.narrow(dim, 1, 1) - x.narrow(dim, 0, 1))
    g.narrow(dim, n - 1, 1).copy_(x.narrow(dim, n - 1, 1) - x.narrow(dim, n - 2, 1))
    return g


def dirlog_batch(occupied: torch.Tensor, sigma: float = DEFAULT_SIGMA, log_scale: float = DEFAULT_LOG_SCALE) -> torch.Tensor:
    """(B,5,R,R,R) float32 [material, nx, ny, nz, d_log] of a boolean batch (B,R,R,R) - field_input.dirlog_channels."""
    occ = occupied.bool()
    empty, full = ~occ.flatten(1).any(1), occ.flatten(1).all(1)
    d_out = torch.sqrt(_sq_edt(occ))
    d_in = torch.sqrt(_sq_edt(~occ))
    sdf = torch.where(occ, -(d_in - 0.5), d_out - 0.5)
    sdf[empty] = 1e3                                              # field_input._signed_distance: empty +1e3, full -1e3
    sdf[full] = -1e3
    smooth = _gaussian(sdf, sigma)
    g = torch.stack([_np_gradient(smooth, d) for d in (1, 2, 3)], dim=1)
    norm = g.norm(dim=1, keepdim=True)
    n = torch.where(norm > 1e-6, g / norm.clamp_min(1e-6), torch.zeros((), device=g.device))
    d = sdf.clamp(-1e3, 1e3)
    d_log = torch.sign(d) * torch.log1p(d.abs() / log_scale) / math.log1p(occ.shape[1] / log_scale)
    return torch.cat([occ[:, None].float(), n.float(), d_log[:, None].float()], dim=1)


def encode_batch(grids: torch.Tensor, kind: str, inside: Optional[torch.Tensor] = None,
                 sigma: float = DEFAULT_SIGMA) -> torch.Tensor:
    """(B,C,R,R,R) float32 network input of a batch of grids (B,R,R,R), any dtype, > 0 = material, in the .bin array
    order. `inside` (B,R,R,R): the base-point bits of dirlog_inside (default: the grid itself, as field_input.encode)."""
    if kind not in GPU_KINDS:
        raise ValueError(f"GPU encoding supports {GPU_KINDS}, got {kind!r}")
    occ = grids > 0
    if kind == "occupancy":                                       # field_input.encode: the grid in {-1, +1}
        return (occ.float() * 2 - 1)[:, None]
    x = dirlog_batch(occ, sigma)
    if kind == "dirlog":
        return x
    bit = occ if inside is None else inside > 0
    return torch.cat([x, bit[:, None].float()], dim=1)


# ------------------------------------------------------------------------------------------------ augmentation

def rotate_grid(x: torch.Tensor, g: np.ndarray) -> torch.Tensor:
    """rotation_symmetry.rotate_volume for a (..., R, R, R) tensor: new axis i takes old axis perm[i], flipped where the
    signed permutation `g` has a negative entry."""
    lead = x.dim() - 3
    perm = [int(np.argmax(np.abs(g[i]))) for i in range(3)]
    out = x.permute(*range(lead), *[lead + p for p in perm])
    flips = [lead + i for i in range(3) if g[i, perm[i]] < 0]
    return out.flip(flips) if flips else out


def rotate_batch(grids: torch.Tensor, inside: Optional[torch.Tensor], group_idx) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply GROUP[group_idx[b]] to sample b of `grids` (and `inside`) - the grid rotations of the CPU datasets, done on
    the raw grids BEFORE encoding (equivariant by construction). Returns contiguous tensors."""
    from training.rotation_symmetry import GROUP
    g_out, i_out = [], []
    for b, gi in enumerate(group_idx):
        g = GROUP[int(gi)]
        g_out.append(rotate_grid(grids[b], g))
        if inside is not None:
            i_out.append(rotate_grid(inside[b], g))
    return torch.stack(g_out).contiguous(), (torch.stack(i_out).contiguous() if inside is not None else None)


# ------------------------------------------------------------------------------------------------ host side

def load_raw(voxel_path, resolution: int = 64, with_inside: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(uint8 grid (R,R,R), uint8 inside bits or None) of one sample, in the .bin array order - what a DataLoader worker
    ships instead of an encoded float tensor. Inside bits: stored <stem>.inside.npy, else <stem>.clean.bin, else the grid
    (training/voxel_dataset.py's order)."""
    grid = np.fromfile(voxel_path, dtype=np.uint8).reshape((resolution,) * 3)
    if not with_inside:
        return grid, None
    import sys
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dataset_gen.src.inside_bits import read_inside
    inside = read_inside(voxel_path, resolution)
    if inside is None:
        clean = Path(voxel_path).with_name(Path(voxel_path).stem + ".clean.bin")
        inside = np.fromfile(clean, dtype=np.uint8).reshape((resolution,) * 3) > 0 if clean.exists() else grid > 0
    return grid, inside.astype(np.uint8)
