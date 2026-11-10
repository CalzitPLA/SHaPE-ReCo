#!/usr/bin/env python3
"""
Point-cloud backbone shared by the vector-field route's networks (training/train_field_type.py,
train_field_rotation.py, train_field_dimension.py, train_field_position.py): the point-cloud analog of
training/backbone.py's Voxel3DBackbone, playing the same role for a `reco_field.fit.Samples`-shaped input
(surface points + normals, no grid) instead of a voxel grid - the vector-field route's own founding reason
for existing (docs/VECTORFIELD.md: voxel discretization erases exactly the information this route needs)
stays true all the way into its learned components: these networks never see a grid.

    normalize_points(points)                    -> (centered/scaled points, centroid, scale)
    encode_patch(points, normals, n_points)      -> (6, n_points) float32: normalized [xyz, nxyz], fixed size
    PointCloudBackbone()(x)                      -> (B, feature_dim): per-point MLP, symmetric max-pool
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """(normalized points, centroid, scale) - centered on the point cloud's own centroid, scaled by its own
    extent (the largest distance from the centroid to any point), so the network sees shape, not absolute
    size or position - both are recovered afterward from `scale`/`centroid` by the caller (the Position/
    Dimension heads predict values already in these normalized units)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    scale = float(np.linalg.norm(centered, axis=1).max())
    scale = max(scale, 1e-6)
    return centered / scale, centroid, scale


def encode_patch(points: np.ndarray, normals: np.ndarray, n_points: int = 512, seed: int = 0) -> np.ndarray:
    """(6, n_points) float32: normalized [x, y, z, nx, ny, nz] per point, resampled (with replacement if the
    patch has fewer than n_points) to a fixed count so patches batch together."""
    rng = np.random.default_rng(seed)
    normed, _centroid, _scale = normalize_points(np.asarray(points, dtype=np.float64))
    n = len(normed)
    idx = rng.choice(n, size=n_points, replace=n < n_points)
    feat = np.concatenate([normed[idx], np.asarray(normals, dtype=np.float64)[idx]], axis=1)
    return feat.T.astype(np.float32)


class PointCloudBackbone(nn.Module):
    """(B, 6, N) -> (B, feature_dim): a per-point MLP (PointNet-style, no T-Net - the caller's own
    centroid/scale normalization already handles translation/scale; rotation is not made invariant, since
    orientation is exactly what the Rotation/axis heads need to read off) followed by a symmetric (order-
    independent) max-pool, so the network's output does not depend on the arbitrary order points were
    sampled in."""

    def __init__(self, in_channels: int = 6, feature_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, feature_dim, 1), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.mlp(x)                 # (B, feature_dim, N)
        return f.max(dim=2).values      # (B, feature_dim)
