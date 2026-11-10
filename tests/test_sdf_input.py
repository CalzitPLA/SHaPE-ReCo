"""SDF network input (training/sdf_input.py): correctness against analytic SDFs, range, edge cases, cache round trip, and that
a dataset yields it."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.sdf_input import cache_path, encode, precompute, sdf_from_occupancy  # noqa: E402

N = 64
GRID = np.stack(np.meshgrid(*[np.arange(N)] * 3, indexing="ij"), -1).astype(float) - (N - 1) / 2


def sphere(r, centre=(0.0, 0.0, 0.0)):
    return np.linalg.norm(GRID - np.asarray(centre), axis=-1) - r


def test_matches_the_analytic_sdf_of_a_sphere_near_the_surface():
    d = sphere(15.0, (2.0, -1.0, 3.0))
    sdf = sdf_from_occupancy(d <= 0, truncation=8.0) * 8.0
    near = np.abs(d) <= 2
    assert np.abs(sdf - d)[near].mean() < 0.25                          # measured: 0.19 voxel
    assert np.array_equal(sdf <= 0, d <= 0) or (np.sign(sdf) != np.sign(d)).mean() < 0.01
    assert sdf[N // 2, N // 2, N // 2] < 0                                # inside is negative


def test_range_and_truncation():
    out = sdf_from_occupancy(sphere(10.0) <= 0, truncation=8.0)
    assert out.dtype == np.float32 and out.min() == -1.0 and out.max() == 1.0
    assert np.abs(sdf_from_occupancy(sphere(10.0) <= 0, truncation=16.0)).max() <= 1.0


def test_empty_and_full_grids_do_not_break():
    assert (sdf_from_occupancy(np.zeros((8, 8, 8), bool)) == 1.0).all()
    assert (sdf_from_occupancy(np.ones((8, 8, 8), bool)) == -1.0).all()


def test_encode_leaves_occupancy_untouched_and_rejects_unknown_kinds():
    v = np.where(sphere(9.0) <= 0, 1.0, -1.0).astype(np.float32)
    assert encode(v, "occupancy") is v
    assert encode(v, "sdf").shape == v.shape and np.array_equal(encode(v, "sdf") < 0, v > 0)
    with pytest.raises(ValueError):
        encode(v, "mesh")


def test_cache_round_trip_and_dataset_yield(tmp_path):
    torch = pytest.importorskip("torch")
    from training.voxel_dataset import VoxelDataset
    paths = []
    for i, r in enumerate((8.0, 14.0)):
        p = tmp_path / f"s{i}.bin"
        (sphere(r) <= 0).astype(np.uint8).tofile(p)
        paths.append(p)
    cache = tmp_path / "cache"
    precompute(paths, N, 8.0, cache, workers=2)
    assert all(cache_path(cache, p, 8.0).is_file() for p in paths)
    ds = VoxelDataset([SimpleNamespace(voxel_path=p) for p in paths], N, input_kind="sdf", sdf_truncation=8.0, sdf_cache=cache)
    t, _ = ds._load(1)
    assert t.shape == (1, N, N, N) and float(t.min()) == -1.0 and float(t.max()) == 1.0
    direct = sdf_from_occupancy(sphere(14.0) <= 0, 8.0)
    assert np.abs(t[0].numpy() - direct).max() < 2e-3                    # float16 cache


def test_cache_misses_when_the_grid_is_rewritten_or_the_code_changes(tmp_path, monkeypatch):
    """T-19: the key (dataset_gen/src/cache_keys.py) holds the .bin's identity, so a grid rewritten in place
    (scripts/revoxelize_parity.py, add_voxel_noise.py) is recomputed, not served stale; and this module's code and
    constants; the pre-T-19 key (path + truncation) is never produced."""
    import hashlib
    import os
    from training import sdf_input
    p = tmp_path / "s.bin"
    (sphere(8.0) <= 0).astype(np.uint8).tofile(p)
    cache = tmp_path / "cache"
    key = cache_path(cache, p, 8.0)
    assert key.stem != hashlib.sha1(f"{p.resolve()}|8.0".encode()).hexdigest()[:20]
    assert cache_path(cache, p, 4.0) != key
    sdf_input.load_sdf_input(p, N, 8.0, cache)
    assert key.is_file()
    (sphere(12.0) <= 0).astype(np.uint8).tofile(p)                              # the grid rewritten in place
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert cache_path(cache, p, 8.0) != key
    fresh = sdf_input.load_sdf_input(p, N, 8.0, cache)
    assert np.abs(fresh - sdf_from_occupancy(sphere(12.0) <= 0, 8.0)).max() < 2e-3   # the new grid's SDF, not the old one
    now = cache_path(cache, p, 8.0)
    monkeypatch.setattr(sdf_input, "DEFAULT_TRUNCATION", 6.0)                   # a constant of the encoder changes
    assert cache_path(cache, p, 8.0) != now
