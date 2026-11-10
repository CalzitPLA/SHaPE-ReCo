"""
Tests for dataset_gen.src.voxel_noise: the "spice up the training set"
noise injected in the voxelization layer (dataset_gen/src/voxelization.py's
assemble_pngs_to_bin, right before a clean {0,1} occupancy grid is
written to disk) - not a training-time augmentation, so a generated
sample's noise is baked in and reproducible the same way its geometry
is.

All synthetic - no FreeCAD, no stltovoxel. Run with:
    pytest tests/test_voxel_noise.py -v
"""

import sys
from pathlib import Path

import numpy as np

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

from src.voxel_noise import VoxelNoiseConfig, apply_voxel_noise  # noqa: E402

R = 32


def _box():
    v = np.zeros((R, R, R), dtype=np.uint8)
    v[10:20, 10:20, 10:20] = 1
    return v


def test_disabled_by_default_is_a_noop():
    config = VoxelNoiseConfig()
    assert not config.enabled
    volume = _box()
    rng = np.random.default_rng(0)
    result = apply_voxel_noise(volume, config, rng)
    assert result is volume, "disabled config must return the identical object, not even a copy"


def test_never_mutates_input():
    volume = _box()
    original = volume.copy()
    config = VoxelNoiseConfig(salt_prob=0.05, blob_count=2)
    apply_voxel_noise(volume, config, np.random.default_rng(0))
    assert np.array_equal(volume, original)


def test_salt_only_adds_never_removes():
    volume = _box()
    config = VoxelNoiseConfig(salt_prob=0.05)
    result = apply_voxel_noise(volume, config, np.random.default_rng(1))
    assert np.array_equal(volume & result, volume), "every originally-occupied voxel must stay occupied"
    assert result.sum() > volume.sum(), "salt must add at least some voxels at this probability/grid size"


def test_pepper_only_removes_never_adds():
    volume = _box()
    config = VoxelNoiseConfig(pepper_prob=0.2)
    result = apply_voxel_noise(volume, config, np.random.default_rng(2))
    assert np.array_equal(result & (1 - volume), np.zeros_like(volume)), "no voxel outside the original box may be set"
    assert result.sum() < volume.sum()


def test_blobs_only_add_and_respect_grid_bounds():
    volume = np.zeros((R, R, R), dtype=np.uint8)  # empty, so every occupied voxel came from a blob
    config = VoxelNoiseConfig(blob_count=5, blob_min_size=3, blob_max_size=8)
    result = apply_voxel_noise(volume, config, np.random.default_rng(3))
    assert result.sum() > 0
    assert result.shape == (R, R, R)


def test_dilate_only_grows_the_shape():
    volume = _box()
    config = VoxelNoiseConfig(dilate_prob=1.0)
    result = apply_voxel_noise(volume, config, np.random.default_rng(4))
    assert np.array_equal(volume & result, volume), "dilation must be a strict superset of the original"
    assert result.sum() > volume.sum()


def test_erode_only_shrinks_the_shape():
    volume = _box()
    config = VoxelNoiseConfig(erode_prob=1.0)
    result = apply_voxel_noise(volume, config, np.random.default_rng(5))
    assert np.array_equal(result & (1 - volume), np.zeros_like(volume)), "erosion must be a strict subset"
    assert result.sum() < volume.sum()


def test_deterministic_given_same_rng_state():
    volume = _box()
    config = VoxelNoiseConfig(salt_prob=0.03, pepper_prob=0.03, blob_count=2, dilate_prob=0.5, erode_prob=0.5)
    r1 = apply_voxel_noise(volume, config, np.random.default_rng(7))
    r2 = apply_voxel_noise(volume, config, np.random.default_rng(7))
    assert np.array_equal(r1, r2)


def test_output_dtype_matches_input():
    volume = _box()
    config = VoxelNoiseConfig(salt_prob=0.05, blob_count=1)
    result = apply_voxel_noise(volume, config, np.random.default_rng(8))
    assert result.dtype == volume.dtype
