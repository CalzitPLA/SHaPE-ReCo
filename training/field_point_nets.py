#!/usr/bin/env python3
"""
Network classes and per-topology target derivation for the vector-field route's point-cloud networks
(training/train_field_type.py, train_field_rotation.py, train_field_dimension.py,
train_field_position.py) - all built on training.field_point_backbone.PointCloudBackbone, the point-cloud
analog of training.backbone.Voxel3DBackbone.

Topology set: reco_field.fit.WHOLE_KINDS = ("sphere", "box", "cylinder", "cone", "torus") - what
reco_field.fit.fit_best already searches over. Each of Rotation/Dimension/Position is ONE DEDICATED
network PER TOPOLOGY (no topology embedding anywhere - each network only ever sees its own shape), matching
the original train_rotation_cnn.py/train_dimension_cnn.py's topology-conditioned design taken one step
further. Sizes and positions are regressed directly in real mm (the field carries absolute scale; no
grid-normalization step is needed the way the voxel route needs one).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training import rotation_symmetry as rs
from training.field_point_backbone import PointCloudBackbone
from training.train_box_dims import REALIZATION_SCALE_MM

WHOLE_KINDS = ("sphere", "box", "cylinder", "cone", "torus")
KIND_TO_IDX = {k: i for i, k in enumerate(WHOLE_KINDS)}
AXIS_KINDS = ("cylinder", "cone", "torus")            # pose = axis of revolution only
ROTATION_KINDS = ("box",) + AXIS_KINDS                 # sphere has no rotation network at all

# Parameter keys per topology (dataset_gen/src/dataset_utils.PARAMETER_SPECS), in the order the Dimension
# network's head outputs them - all "_ratio" keys, realized at REALIZATION_SCALE_MM mm (the same convention
# training/train_box_dims.py and reco_engine/box_orientation.py already use).
DIMENSION_KEYS: Dict[str, Tuple[str, ...]] = {
    "box": ("x_half_ratio", "y_half_ratio", "z_half_ratio"),
    "sphere": ("radius_ratio",),
    "cylinder": ("radius_ratio", "half_length_ratio"),
    "cone": ("radius1_ratio", "radius2_ratio", "height_ratio"),
    "torus": ("major_radius_ratio", "minor_radius_ratio"),
}
LOCAL_AXIS = np.array([0.0, 0.0, 1.0])     # sdf_primitives.py's convention: bodies of revolution turn about local z


def true_size_mm(topology: str, parameters: Dict[str, float], scale_mm: float = REALIZATION_SCALE_MM) -> np.ndarray:
    """The topology's DIMENSION_KEYS, realized to real mm - the Dimension network's regression target."""
    return np.array([parameters[k] * scale_mm for k in DIMENSION_KEYS[topology]], dtype=np.float64)


def true_axis_world(quaternion) -> np.ndarray:
    """The world-frame axis of revolution (unit vector) of a body built with LOCAL_AXIS then rotated by
    `quaternion` (qx,qy,qz,qw) - the axis Rotation network's regression target for cylinder/cone/torus."""
    M = rs.quat_to_matrix(quaternion)
    axis = M @ LOCAL_AXIS
    return axis / max(np.linalg.norm(axis), 1e-9)


# ---------------------------------------------------------------------------------------------------- nets

class FieldTypeNet(nn.Module):
    """Point cloud -> logits over WHOLE_KINDS (5-way)."""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.backbone = PointCloudBackbone(6, feature_dim)
        self.head = nn.Linear(feature_dim, len(WHOLE_KINDS))
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class FieldBoxPoseNet(nn.Module):
    """Point cloud -> a 3x3 rotation matrix (6D representation, Gram-Schmidt) - box only."""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.backbone = PointCloudBackbone(6, feature_dim)
        self.head = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rs.sixd_to_matrix_torch(self.head(self.backbone(x)))


class FieldAxisNet(nn.Module):
    """Point cloud -> a unit axis-of-revolution vector - cylinder/cone/torus."""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.backbone = PointCloudBackbone(6, feature_dim)
        self.head = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.backbone(x)), dim=1, eps=1e-8)


class FieldDimensionNet(nn.Module):
    """Point cloud -> the topology's own size parameters, in LOG mm (scale-invariant target; the caller
    exponentiates). One network per topology, so `n_params` is fixed at construction - no masking needed."""

    def __init__(self, n_params: int, feature_dim: int = 256):
        super().__init__()
        self.backbone = PointCloudBackbone(6, feature_dim)
        self.head = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, n_params))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class FieldPositionNet(nn.Module):
    """Point cloud -> the primitive's true centre, as an offset from the patch's own centroid, in units of
    the patch's own extent (training/field_point_backbone.normalize_points' own units - the caller
    multiplies back by the patch's extent and adds the centroid to recover a world position). One network
    per topology (like Rotation/Dimension), since how a patch's centroid relates to the true centre differs
    by shape (e.g. a cylinder's flat end patch centres very differently than its curved side does)."""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.backbone = PointCloudBackbone(6, feature_dim)
        self.head = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
