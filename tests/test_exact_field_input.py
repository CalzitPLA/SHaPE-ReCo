"""training/exact_field_input.py: the mesh's own exact closest-surface-point field as a network input, resolution
independent of the occupancy grid, Nyquist-oversampled by default, cached to disk."""
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training import exact_field_input as efi  # noqa: E402


@pytest.fixture
def box_stl(tmp_path):
    m = trimesh.creation.box(extents=(8.0, 5.0, 3.0))
    path = tmp_path / "box.stl"
    m.export(path)
    return path


def test_field_resolution_defaults_to_nyquist_2x():
    assert efi.field_resolution_for(64) == 128
    assert efi.field_resolution_for(64, oversample=3) == 192
    assert efi.field_resolution_for(32) == 64


def test_compute_exact_field_shape_and_channels(box_stl):
    out = efi.compute_exact_field(box_stl, resolution=16, grid_mm=20.0)
    assert out.shape == (efi.CHANNELS, 16, 16, 16) and out.dtype == np.float32
    assert np.isfinite(out).all()


def test_compute_exact_field_is_just_surfacefield_grid_vectors(box_stl):
    """compute_exact_field is a thin wrapper: it must return exactly what SurfaceField.grid_vectors does (that
    function's own correctness - direction, magnitude - is covered by tests/test_closest_point_field.py against
    the trimesh oracle; this module only adds caching and the resolution/grid_mm keying around it)."""
    sys.path.insert(0, str(ROOT / "dataset_gen"))
    from src.closest_point_field import load_field
    expected = load_field(box_stl).grid_vectors(20, 20.0).astype(np.float32)
    got = efi.compute_exact_field(box_stl, resolution=20, grid_mm=20.0)
    np.testing.assert_array_equal(got, expected)


def test_load_exact_field_input_caches_and_is_reproducible(box_stl, tmp_path):
    cache = tmp_path / "cache"
    a = efi.load_exact_field_input(box_stl, resolution=12, grid_mm=20.0, cache_dir=cache)
    assert len(list(cache.glob("*.npy"))) == 1
    b = efi.load_exact_field_input(box_stl, resolution=12, grid_mm=20.0, cache_dir=cache)
    np.testing.assert_allclose(a, b, atol=1e-2)                              # float16 round trip, not bit-identical
    assert len(list(cache.glob("*.npy"))) == 1                               # second call hit the cache, didn't recompute+overwrite differently


def test_cache_path_depends_on_resolution_and_grid_mm(box_stl, tmp_path):
    cache = tmp_path / "cache"
    p1 = efi.cache_path(cache, box_stl, 12, 20.0)
    p2 = efi.cache_path(cache, box_stl, 16, 20.0)
    p3 = efi.cache_path(cache, box_stl, 12, 40.0)
    assert len({p1, p2, p3}) == 3


def test_cache_path_depends_on_the_mesh_file_the_backend_and_the_code(box_stl, tmp_path, monkeypatch):
    """T-19 (dataset_gen/src/cache_keys.py): an STL rewritten in place, another field backend or a changed constant of
    closest_point_field is a miss; the pre-T-19 key (path + resolution + grid) is never produced."""
    import hashlib
    import os
    sys.path.insert(0, str(ROOT / "dataset_gen"))
    from src import closest_point_field as cpf
    cache = tmp_path / "cache"
    monkeypatch.setenv("SHAPE_RECO_FIELD_BACKEND", "numpy")
    key = efi.cache_path(cache, box_stl, 12, 20.0)
    assert key.stem != hashlib.sha1(f"{box_stl.resolve()}|12|20.0".encode()).hexdigest()[:20]
    monkeypatch.setenv("SHAPE_RECO_FIELD_BACKEND", "igl")                       # exact instead of the one-pass query
    assert efi.cache_path(cache, box_stl, 12, 20.0) != key
    monkeypatch.setenv("SHAPE_RECO_FIELD_BACKEND", "numpy")
    monkeypatch.setattr(cpf, "DENSE_BLOCK", 1024)
    assert efi.cache_path(cache, box_stl, 12, 20.0) != key
    monkeypatch.setattr(cpf, "DENSE_BLOCK", 32768)
    assert efi.cache_path(cache, box_stl, 12, 20.0) == key
    st = box_stl.stat()
    os.utime(box_stl, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))         # the STL rewritten in place
    assert efi.cache_path(cache, box_stl, 12, 20.0) != key


def test_precompute_fills_the_cache_for_every_mesh(tmp_path):
    cache = tmp_path / "cache"
    paths = []
    for i, extents in enumerate([(8, 5, 3), (4, 4, 4), (6, 2, 9)]):
        m = trimesh.creation.box(extents=extents)
        p = tmp_path / f"b{i}.stl"
        m.export(p)
        paths.append(p)
    efi.precompute(paths, resolution=10, cache_dir=cache, grid_mm=20.0, workers=2)
    assert len(list(cache.glob("*.npy"))) == 3
    for p in paths:
        assert efi.cache_path(cache, p, 10, 20.0).is_file()


def test_voxel_dataset_routes_exact_field_to_the_mesh(box_stl, tmp_path):
    import torch
    from training.manifest import Record
    from training.voxel_dataset import VoxelDataset
    rec = Record.__new__(Record)
    rec.stl_path = box_stl
    rec.placement = {"grid_mm": 20.0}
    ds = VoxelDataset([rec], resolution=8, input_kind="exact_field", field_cache=tmp_path / "cache")
    assert ds.field_resolution == 16                                          # Nyquist default: 2x resolution=8
    tensor, _ = ds._load(0)
    assert tuple(tensor.shape) == (3, 16, 16, 16)
    assert isinstance(tensor, torch.Tensor)


def test_type_cnn_takes_the_exact_field_input(box_stl, tmp_path):
    import torch
    from training.manifest import Record
    from training.train_type_cnn import TypeCNN
    from training.voxel_dataset import VoxelDataset
    rec = Record.__new__(Record)
    rec.stl_path = box_stl
    rec.placement = {"grid_mm": 20.0}
    tensor, _ = VoxelDataset([rec], resolution=8, input_kind="exact_field", field_cache=tmp_path / "cache")._load(0)
    out = TypeCNN(num_classes=5, in_channels=3)(tensor[None])
    assert out.shape == (1, 5)
