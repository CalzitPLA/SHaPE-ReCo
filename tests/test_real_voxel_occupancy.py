"""reco_field/peel.py's `real_voxel_occupancy`: the ground-truth voxelizer (ray parity at the voxel centres, D-vox) on
the SAME grid as `input_occupancy`. The part is asymmetric and far from the world origin on every axis: the earlier
version centred its cube on the origin and returned 0 voxels for exactly this case (`.claude/guru/council/guru.md`
P2), and an axis swap or a mis-signed axis would show up as a wrong centroid or extent on that axis."""
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src.closest_point_field import SurfaceField  # noqa: E402
from reco_field.peel import _grid_centres, input_occupancy, real_voxel_occupancy  # noqa: E402

HALF = np.array([6.0, 3.0, 9.0])
CENTRE = np.array([100.0, 50.0, 20.0])


def _box_field():
    mesh = trimesh.creation.box(extents=2 * HALF)
    mesh.apply_translation(CENTRE)
    return SurfaceField(mesh.vertices, mesh.faces)


def _grid(field, pad=0.1):
    lo, hi = np.asarray(field.bounds[0], float), np.asarray(field.bounds[1], float)
    p = pad * float((hi - lo).max())
    return lo - p, hi + p


def test_off_origin_asymmetric_box_keeps_its_centre_and_extents():
    field = _box_field()
    lo, hi = _grid(field)
    occ = real_voxel_occupancy(field, lo, hi, 48)
    assert occ.dtype == bool and occ.shape == (48, 48, 48) and occ.any()
    centres = _grid_centres(lo, hi, 48)[occ]
    step = (hi - lo) / 48
    assert np.all(np.abs(centres.mean(axis=0) - CENTRE) < 1.0 * step), centres.mean(axis=0)
    half = (centres.max(axis=0) - centres.min(axis=0)) / 2
    assert np.all(np.abs(half - HALF) < 2.0 * step), half


def test_agrees_with_input_occupancy_on_the_same_grid():
    field = _box_field()
    lo, hi = _grid(field)
    a = real_voxel_occupancy(field, lo, hi, 32)
    b = input_occupancy(field, lo, hi, 32)
    iou = (a & b).sum() / (a | b).sum()
    assert iou >= 0.98, iou
