#!/usr/bin/env python3
"""
Training data for the vector-field route's point-cloud networks: samples a whole shape's surface (the Type
network) or a random PARTIAL PATCH of it (the per-topology Rotation/Dimension/Position networks - a real
cluster in the peeling loop is never a whole clean shape, `reco_field.peel._side_clusters`, so training on
whole shapes only would be a train/test mismatch this project already hit once before, docs/CODEMAP.md's
"How the Type CNN recognises a base shape as it is sliced").

Reuses `reco_field.fit.make_samples` for the underlying uniform surface sampling (the same function `fit.py`
itself calls) rather than re-deriving it, and `training.manifest.load_manifest`/`Record` for ground-truth
topology/parameters/rotation - the same `dataset_gen` STLs the voxel route trains on, just never voxelized.

    sample_whole(stl_path, n_points, seed)                    -> Samples-like (points, normals)
    sample_patch(stl_path, n_points, seed, radius_fraction)   -> (points, normals, true_centre_world, patch_extent) or None
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "dataset_gen") not in sys.path:
    sys.path.insert(0, str(ROOT / "dataset_gen"))

from src.closest_point_field import load_field  # noqa: E402

from reco_field.fit import make_samples  # noqa: E402
from training.field_point_backbone import encode_patch  # noqa: E402
from training.manifest import Record, group_split, load_manifest  # noqa: E402

DEFAULT_N_SURFACE = 2000
DEFAULT_RADIUS_FRACTION = (0.15, 0.6)   # a patch's radius, as a fraction of the whole part's own extent
MIN_PATCH_POINTS = 30


def sample_whole(stl_path, n_points: int = 512, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """(points (n_points,3), normals (n_points,3)) uniformly over the whole shape's surface."""
    field = load_field(stl_path)
    s = make_samples(field, n_surface=max(n_points, DEFAULT_N_SURFACE), n_near=1, seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(s.surface), size=n_points, replace=len(s.surface) < n_points)
    return s.surface[idx], s.normals[idx]


def sample_patch(stl_path, n_points: int = 512, seed: int = 0,
                 radius_fraction: Tuple[float, float] = DEFAULT_RADIUS_FRACTION,
                 attempts: int = 20) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """(points, normals, true_centre_world (3,), patch_extent) of a random partial patch of the shape's
    surface - a point on the surface plus everything within a random radius of it (a fraction of the whole
    part's own extent, matching the scale a real `reco_field.peel` cluster grows to) - or None if no draw
    within `attempts` finds enough points (MIN_PATCH_POINTS). `true_centre_world` is the whole part's own
    bounding-box centre (`field.bounds`) - what the Position network's label is defined relative to; not
    exact for every topology's own analytic centre (a cone's is its base, not its bbox centre - see
    `reco_field.fit.FitResult.centre`) but the same reasonable STARTING estimate `placement.estimate_placement`
    already uses for the voxel route, refined afterward by the same adaptive fitting this whole network only
    ever proposes a start for (see the plan's "Integration" section)."""
    field = load_field(stl_path)
    extent = float(np.linalg.norm(field.bounds[1] - field.bounds[0]))
    true_centre = 0.5 * (field.bounds[0] + field.bounds[1])
    s = make_samples(field, n_surface=DEFAULT_N_SURFACE, n_near=1, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for attempt in range(attempts):
        seed_point = s.surface[rng.integers(len(s.surface))]
        radius = rng.uniform(*radius_fraction) * extent
        dist = np.linalg.norm(s.surface - seed_point, axis=1)
        mask = dist <= radius
        if int(mask.sum()) < MIN_PATCH_POINTS:
            continue
        points, normals = s.surface[mask], s.normals[mask]
        idx = rng.choice(len(points), size=n_points, replace=len(points) < n_points)
        patch_extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        return points[idx], normals[idx], true_centre, max(patch_extent, 1e-6)
    return None


def load_field_split(dataset_root, val_fraction: float, seed: int):
    """(train, val) Records - group_split (training/manifest.py), but WITHOUT requiring --voxelize: the
    vector-field networks read the STL's own field directly and never touch a .bin voxel file."""
    records = load_manifest(dataset_root, require_voxel=False)
    if not records:
        raise RuntimeError(f"No samples found under {dataset_root}.")
    train, val = group_split(records, val_fraction, seed)
    if not val:
        raise RuntimeError("Validation split is empty - increase --val-fraction or dataset size.")
    return train, val


def records_of_topology(records, topology: str):
    return [r for r in records if r.topology == topology]


class TypePointDataset:
    """(patch tensor (6, n_points), type index) - the whole shape's surface, uniformly sampled
    (training/field_points.sample_whole)."""

    def __init__(self, records: List[Record], n_points: int = 512, seed: int = 0):
        from training.field_point_nets import KIND_TO_IDX
        self.records, self.n_points, self.seed, self.kind_to_idx = records, n_points, seed, KIND_TO_IDX

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        import torch
        r = self.records[idx]
        points, normals = sample_whole(r.stl_path, self.n_points, seed=self.seed + idx)
        x = torch.from_numpy(encode_patch(points, normals, self.n_points, seed=self.seed + idx))
        return x, self.kind_to_idx[r.topology]


class TopologyPatchDataset:
    """(patch tensor, record) for one topology's records - a random partial patch per sample
    (training/field_points.sample_patch), retried with a different record on a rare failed draw so a
    DataLoader batch is never short. Per-role target derivation (rotation/dimension/position) lives in each
    training/train_field_*.py script, not here, since what a "target" means differs by role."""

    def __init__(self, records: List[Record], n_points: int = 512, seed: int = 0):
        self.records, self.n_points, self.seed = records, n_points, seed

    def __len__(self):
        return len(self.records)

    def _draw(self, idx: int):
        r = self.records[idx]
        draw = sample_patch(r.stl_path, self.n_points, seed=self.seed + idx)
        if draw is None:
            return self._draw((idx + 1) % len(self.records))
        return r, draw

    def __getitem__(self, idx: int):
        import torch
        r, (points, normals, true_centre, patch_extent) = self._draw(idx)
        x = torch.from_numpy(encode_patch(points, normals, self.n_points, seed=self.seed + idx))
        patch_centroid = points.mean(axis=0)
        return dict(x=x, record=r, true_centre=true_centre, patch_centroid=patch_centroid, patch_extent=patch_extent)
