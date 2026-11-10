"""
Scale modes of the stltovoxel voxelizer (dataset_gen/src/voxelization.py): "fit" stretches the bounding
box onto the grid per axis (the mode default, unchanged), "uniform" keeps proportions, "fixed" keeps
proportions and absolute size. These grids are stltovoxel's own (the parity default makes only the
in-scale grid, fixed + anchor origin), so the tests ask for it explicitly. Uses the real stltovoxel;
skipped if it is not installed.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

SANDBOX = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX / "dataset_gen"))
from src.voxelization import find_stltovoxel, voxelize_stl_to_bin  # noqa: E402

pytestmark = pytest.mark.skipif(find_stltovoxel() is None, reason="stltovoxel not installed")
N = 32


def voxelize(tmp_path, extents_xyz, **kw):
    """A box with the given (x, y, z) size in mm -> the [z, y, x] grid."""
    stl = tmp_path / "part.stl"
    trimesh.creation.box(extents=extents_xyz).export(stl)
    out = tmp_path / "part.bin"
    assert voxelize_stl_to_bin(str(stl), str(out), resolution=N, voxelizer="stltovoxel", **kw)
    return np.fromfile(out, dtype=np.uint8).reshape(N, N, N).astype(bool)


def bbox_size(grid):
    idx = np.argwhere(grid)
    return (idx.max(0) - idx.min(0) + 1).tolist(), idx.min(0).tolist()


def test_fit_stretches_every_axis_onto_the_grid(tmp_path):
    size, _ = bbox_size(voxelize(tmp_path, (20, 10, 5)))
    # a 4:2:1 box fills the grid on every axis (z loses its two empty pad slices to the subsampling)
    assert all(n >= N - 2 for n in size)


def test_uniform_keeps_proportions_and_centres(tmp_path):
    grid = voxelize(tmp_path, (20, 10, 5), scale_mode="uniform")      # x 20 -> the largest, spans the grid
    size, origin = bbox_size(grid)                                     # array order [z, y, x]
    assert size[2] == pytest.approx(N, abs=1) and size[1] == pytest.approx(N / 2, abs=1) \
        and size[0] == pytest.approx(N / 4, abs=1)
    assert origin[0] == (N - size[0]) // 2 and origin[1] == (N - size[1]) // 2      # centred on the short axes
    assert grid.sum() == pytest.approx(size[0] * size[1] * size[2], rel=0.02)         # still a solid box, not stretched


def test_a_thin_slab_stays_thin_in_uniform_mode(tmp_path):
    size, _ = bbox_size(voxelize(tmp_path, (20, 20, 1), scale_mode="uniform"))
    assert size[0] <= 3 and size[1] == size[2] == pytest.approx(N, abs=1)             # it would be N thick under "fit"


def test_fixed_keeps_absolute_size(tmp_path):
    small, _ = bbox_size(voxelize(tmp_path, (10, 5, 2.5), scale_mode="fixed", scale_mm=40.0))
    big, _ = bbox_size(voxelize(tmp_path, (20, 10, 5), scale_mode="fixed", scale_mm=40.0))
    assert small[2] == pytest.approx(N * 10 / 40, abs=1) and big[2] == pytest.approx(N * 20 / 40, abs=1)
    assert big[2] == pytest.approx(2 * small[2], abs=1)                    # twice the size stays twice the size
    assert big[1] == pytest.approx(N * 10 / 40, abs=1) and big[0] == pytest.approx(N * 5 / 40, abs=1)


def test_fixed_clips_a_part_bigger_than_the_grid(tmp_path):
    size, _ = bbox_size(voxelize(tmp_path, (80, 10, 5), scale_mode="fixed", scale_mm=40.0))
    assert size[2] == N                                                     # centre-cropped to the grid


def test_fixed_needs_a_scale_and_unknown_modes_are_rejected(tmp_path):
    stl = tmp_path / "part.stl"
    trimesh.creation.box(extents=(1, 1, 1)).export(stl)
    with pytest.raises(ValueError, match="scale_mm"):
        voxelize_stl_to_bin(str(stl), str(tmp_path / "o.bin"), resolution=N, scale_mode="fixed", voxelizer="stltovoxel")
    with pytest.raises(ValueError, match="scale_mm"):                     # the parity default on the in-scale grid too
        voxelize_stl_to_bin(str(stl), str(tmp_path / "o.bin"), resolution=N, scale_mode="fixed", anchor="origin")
    with pytest.raises(ValueError, match="scale_mode"):
        voxelize_stl_to_bin(str(stl), str(tmp_path / "o.bin"), resolution=N, scale_mode="stretchy")


def test_array_axis_order_is_the_same_in_every_mode(tmp_path):
    """The long axis of an x-long box lands on array axis 2 whichever mode is used."""
    for kw in ({}, {"scale_mode": "uniform"}, {"scale_mode": "fixed", "scale_mm": 24.0}):
        grid = voxelize(tmp_path, (20, 10, 5), **kw)
        size, _ = bbox_size(grid)
        if kw:      # (the fit mode stretches everything, so only the scale modes can show it)
            assert size[2] > size[1] > size[0]


def test_cache_keys_separate_the_modes(tmp_path):
    from reco_engine.cache import Cache
    stl = tmp_path / "a.stl"
    stl.write_bytes(b"solid x")
    base = Cache.voxel_key(stl, 64)
    assert Cache.voxel_key(stl, 64, "fit") == base                       # the default arguments are "fit"
    assert Cache.voxel_key(stl, 64, "uniform") != base
    assert Cache.voxel_key(stl, 64, "fixed", 24.0) != Cache.voxel_key(stl, 64, "fixed", 30.0)
    # voxelizer and code version in the key: tests/test_cache.py
