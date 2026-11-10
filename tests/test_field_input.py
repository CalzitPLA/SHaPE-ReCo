"""training/field_input.py: occupancy + signed distance + the vector to the nearest surface point as one network input, derived from the grid itself."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training import field_input as fi  # noqa: E402


def ball(centre, radius, n=64):
    idx = np.indices((n, n, n)).astype(float)
    return ((idx[0] - centre[0]) ** 2 + (idx[1] - centre[1]) ** 2 + (idx[2] - centre[2]) ** 2) <= radius ** 2


def pm1(occ):
    return occ.astype(np.float32) * 2 - 1


def test_kinds_and_channel_counts():
    vol = pm1(ball((31.5, 31.5, 31.5), 12))
    for kind, c in fi.CHANNELS.items():
        if kind in fi.FROM_NPZ or kind == "dirlog_residual":   # not derivable from a grid alone (.field.npz / the current model)
            with pytest.raises(ValueError):
                fi.encode(vol, kind)
            continue
        out = fi.encode(vol, kind)
        assert out.shape == (c, 64, 64, 64) and out.dtype == np.float32 and np.isfinite(out).all(), kind
    with pytest.raises(ValueError):
        fi.encode(vol, "nonsense")
    np.testing.assert_array_equal(fi.encode(vol, "occupancy")[0], vol)                       # the baseline input is untouched
    fused = fi.encode(vol, "fused")
    np.testing.assert_array_equal(fused[0], vol)
    np.testing.assert_array_equal(fused[1], fi.encode(vol, "sdf")[0])
    np.testing.assert_array_equal(fused[2:], fi.encode(vol, "field"))

    vf = fi.encode(vol, "voxel_field")
    assert vf.shape == (4, 64, 64, 64)
    np.testing.assert_array_equal(vf[0], vol)                                                # occupancy first
    np.testing.assert_array_equal(vf[1:], fi.encode(vol, "field"))                            # the field stacked on top, no sdf channel


def test_ranges_and_the_length_of_v_is_the_distance():
    sdf, field = fi.sdf_and_field(ball((31.5, 31.5, 31.5), 12), truncation=8.0)
    assert sdf.min() >= -1 and sdf.max() <= 1
    length = np.linalg.norm(field, axis=0)
    assert length.max() <= 1.0 + 1e-5
    assert np.abs(length - np.abs(sdf)).max() < 1e-5                                          # |V| = |sdf| (V = -f grad f, unit gradient)


def test_v_points_to_the_surface_from_both_sides_and_along_the_normal():
    c = np.array([31.5, 31.5, 31.5])
    occ = ball(c, 12)
    sdf, field = fi.sdf_and_field(occ, truncation=8.0)
    idx = np.stack(np.indices(occ.shape), axis=-1).astype(float)
    radial = idx - c
    radial /= np.linalg.norm(radial, axis=-1, keepdims=True)
    shell = (np.abs(sdf) * 8 > 0.8) & (np.abs(sdf) * 8 < 4)                                   # 1-4 voxels from the surface
    V = np.moveaxis(field, 0, -1)
    to_surface = -np.sign(sdf)[..., None] * radial                                            # inward from outside, outward from inside
    cos = (V * to_surface).sum(-1) / np.maximum(np.linalg.norm(V, axis=-1), 1e-9)
    assert np.median(cos[shell]) > 0.98 and np.percentile(cos[shell], 10) > 0.9
    assert (cos[shell & (sdf > 0)] > 0.9).mean() > 0.9 and (cos[shell & (sdf < 0)] > 0.9).mean() > 0.9


def test_flipping_an_array_axis_negates_that_component_of_v():
    """The vectors are in the grid's array axes: mirroring axis 0 mirrors the field and negates its component 0 (what the training augmentation relies on)."""
    rng = np.random.default_rng(0)
    occ = ball((28.5, 33.5, 30.5), 11) | ball((40.5, 30.5, 34.5), 6)
    a = fi.sdf_and_field(occ)[1]
    b = fi.sdf_and_field(occ[::-1].copy())[1]
    np.testing.assert_allclose(b[0], -a[0][::-1], atol=1e-4)
    np.testing.assert_allclose(b[1], a[1][::-1], atol=1e-4)
    np.testing.assert_allclose(b[2], a[2][::-1], atol=1e-4)


def test_empty_and_full_grids_do_not_break():
    for occ in (np.zeros((16, 16, 16), bool), np.ones((16, 16, 16), bool)):
        out = fi.encode(pm1(occ), "fused")
        assert np.isfinite(out).all() and out.shape == (5, 16, 16, 16)


def test_dataset_and_networks_take_the_fused_input(tmp_path):
    import torch
    from training.manifest import Record
    from training.train_rotation_sym import SymRotationCNN
    from training.train_type_cnn import TypeCNN
    from training.voxel_dataset import VoxelDataset
    path = tmp_path / "b.bin"
    ball((31.5, 31.5, 31.5), 12).astype(np.uint8).tofile(path)
    rec = Record.__new__(Record)
    rec.voxel_path = path
    for kind, c in fi.CHANNELS.items():
        if kind in fi.FROM_NPZ or kind == "dirlog_residual":   # needs a .field.npz / the current model: tested elsewhere
            continue
        tensor, _ = VoxelDataset([rec], 64, input_kind=kind)._load(0)
        assert tuple(tensor.shape) == (c, 64, 64, 64)
        assert TypeCNN(num_classes=5, in_channels=c)(tensor[None]).shape == (1, 5)
        assert SymRotationCNN("6d", in_channels=c)(tensor[None]).shape == (1, 3, 3)


def test_dirlog_direction_has_full_weight_at_the_surface_and_distance_never_saturates():
    c = np.array([29.3, 35.0, 27.6])                                                       # off-centre, asymmetric
    occ = ball(c, 12)
    x = fi.encode(pm1(occ), "dirlog")
    assert x.shape == (5, 64, 64, 64) and x.dtype == np.float32
    np.testing.assert_array_equal(x[0], occ.astype(np.float32))                                # material in {0, 1}
    length = np.linalg.norm(x[1:4], axis=0)
    assert np.abs(length - 1).max() < 1e-4                                                  # unit direction everywhere
    idx = np.stack(np.indices(occ.shape), axis=-1).astype(float)
    r = np.linalg.norm(idx - c, axis=-1)
    band = np.abs(r - 12) < 1.5
    radial = (idx - c) / np.maximum(r, 1e-9)[..., None]
    cos = (np.moveaxis(x[1:4], 0, -1) * radial).sum(-1)
    assert np.median(cos[band]) > 0.99                                                      # the OUTWARD normal, on both sides
    assert np.abs(x[4]).max() < 0.999                                                        # no clipping anywhere in the grid
    assert (x[4][occ] <= 0).all() and (x[4][~occ] > 0).all()                                # negative inside, as "fused"
    fused = fi.encode(pm1(occ), "fused")
    assert np.linalg.norm(fused[2:5], axis=0)[band].mean() < 0.25                           # the old layout fades there


def test_log_distance_is_odd_monotone_and_resolution_normalised():
    d = np.linspace(-64, 64, 257)
    y = fi.log_distance(d, 64)
    np.testing.assert_allclose(y, -y[::-1], atol=1e-6)
    assert (np.diff(y) > 0).all() and abs(y[-1] - 1) < 1e-6
    assert abs(fi.log_distance(np.array([128.0]), 128)[0] - 1) < 1e-6


def test_exact_kind_cache_misses_on_a_code_constant_or_a_rewritten_field(tmp_path, monkeypatch):
    """T-19: the dirlog_exact/dirlog_curv cache key (dataset_gen/src/cache_keys.py) holds the .field.npz's identity and
    the code of field_input and field_channels with their constants; the pre-T-19 key is never produced."""
    import hashlib
    import os
    trimesh = pytest.importorskip("trimesh")
    sys.path.insert(0, str(ROOT / "dataset_gen"))
    from src import field_channels as fc
    monkeypatch.setenv("SHAPE_RECO_CACHE", str(tmp_path / "cache"))
    box = trimesh.creation.box(extents=(9.0, 5.0, 4.0))
    box.apply_translation((1.3, -0.7, 0.4))
    stl = tmp_path / "b.stl"
    box.export(stl)
    spec = fc.inscale_spec(20.0, 16)
    field = fc.field_of_file(stl)
    channels = fc.mesh_channels(field, spec, inside=fc.parity_occupancy(field, spec))
    bin_path = tmp_path / "b.bin"
    fc.save_field_npz(fi.field_npz_path(bin_path), channels, spec, source="mesh")
    x = fi.load_exact_kind(bin_path, "dirlog_exact")
    key = fi.exact_kind_cache_path(bin_path, "dirlog_exact")
    assert key.is_file() and np.allclose(fi.load_exact_kind(bin_path, "dirlog_exact"), x, atol=2e-3)   # float16 entry
    npz = fi.field_npz_path(bin_path).resolve()
    legacy = hashlib.sha1(f"{fi.EXACT_KIND_CACHE_VERSION}|{npz}|{npz.stat().st_mtime_ns}|dirlog_exact".encode()).hexdigest()[:24]
    assert key.stem != legacy                                                    # old entries are never read again
    assert fi.exact_kind_cache_path(bin_path, "dirlog_curv") != key
    monkeypatch.setattr(fc, "DEFAULT_LOG_SCALE", 2.0)                            # a constant of dirlog_input's module
    assert fi.exact_kind_cache_path(bin_path, "dirlog_exact") != key
    monkeypatch.setattr(fc, "DEFAULT_LOG_SCALE", 1.0)
    assert fi.exact_kind_cache_path(bin_path, "dirlog_exact") == key
    st = npz.stat()
    os.utime(npz, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))              # the field rewritten in place
    assert fi.exact_kind_cache_path(bin_path, "dirlog_exact") != key
