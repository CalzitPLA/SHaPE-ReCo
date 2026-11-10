"""training/inputs.py "voxfield": the voxel grid and the exact surface vector field in one tensor - channel layout and
ranges on a non-cubic grid, train/infer identity (the dataset path and the loop-style direct call give the same tensor bit
for bit), the sparse band round trip, the rotation augmentation turning the vector channels (on an asymmetric, off-centre
L prism, not a cube), the documented voxel-only fallback, and "dirlog_inside" unchanged (workstation trains on it)."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from training import field_input as fi  # noqa: E402
from training import inputs as vi  # noqa: E402
from training.manifest import Record  # noqa: E402

GRID_MM, RES = 20.0, 24


# ------------------------------------------------------------------------------------------------ fixtures

def _sphere_field(shape=(20, 28, 24), centre=(8.3, 15.1, 11.6), radius=6.2):
    """Analytic exact field of an off-centre sphere on a NON-cubic grid (array axes, pitch 1): inside bits, sdf positive
    inside, V = c(x) - x, and the unit outward radial direction."""
    x = np.stack(np.indices(shape), 0).astype(np.float64)
    r = x - np.asarray(centre)[:, None, None, None]
    dist = np.linalg.norm(r, axis=0)
    radial = r / dist
    return dist < radius, radius - dist, radial * (radius - dist), radial


def _l_prism(rotation=np.eye(3), offset=(2.1, -1.7, 1.3), scale=1.6):
    """A watertight L-shaped prism (only a mirror plane of its own; tilted and off-centre it has no grid symmetry)."""
    import trimesh
    poly = np.array([(0, 0), (6, 0), (6, 2), (2, 2), (2, 5), (0, 5)], float) - (2.3, 1.9)
    v = np.concatenate([np.c_[poly, np.full(6, -1.5)], np.c_[poly, np.full(6, 1.5)]]) * scale
    fan = [(3, 4, 5), (3, 5, 0), (3, 0, 1), (3, 1, 2)]                            # from the reflex corner (2, 2)
    faces = [(a, c, b) for a, b, c in fan] + [(a + 6, b + 6, c + 6) for a, b, c in fan]
    for i in range(6):
        j = (i + 1) % 6
        faces += [(i, j, j + 6), (i, j + 6, i + 6)]
    tilt = trimesh.transformations.euler_matrix(0.41, -0.27, 0.63)[:3, :3]
    mesh = trimesh.Trimesh(v @ tilt.T, np.array(faces), process=True)
    mesh.fix_normals()
    assert mesh.is_watertight and mesh.volume > 0
    mesh.vertices = mesh.vertices @ np.asarray(rotation).T + np.asarray(offset) @ np.asarray(rotation).T
    return mesh


@pytest.fixture
def igl_backend(monkeypatch):
    try:
        import igl  # noqa: F401
        monkeypatch.setenv("SHAPE_RECO_FIELD_BACKEND", "igl")                  # exact and fast; the generator's default
    except ImportError:
        pass


def _world_field(mesh_path):
    """What the loop has for an input mesh: parity bits and mesh_channels on the in-scale grid (world axes)."""
    from src.field_channels import field_of_file, inscale_spec, mesh_channels, parity_occupancy
    spec = inscale_spec(GRID_MM, RES)
    field = field_of_file(mesh_path)
    inside = parity_occupancy(field, spec)
    return spec, inside, mesh_channels(field, spec, inside=inside)


def _write_sample(tmp_path, with_field=True):
    """A dataset sample: <stem>.stl, a NOISY .bin (parity + 3 flipped voxels), <stem>.inside.npy, <stem>.field.npz."""
    from src.field_channels import save_field_npz, world_to_bin
    from src.inside_bits import write_inside
    stl = tmp_path / "l.stl"
    _l_prism().export(stl)
    spec, inside_w, ch = _world_field(stl)
    grid = world_to_bin(inside_w).astype(np.uint8)
    for idx in [(0, 0, 0), (12, 11, 10), (23, 5, 17)]:
        grid[idx] ^= 1
    bin_path = tmp_path / "l.bin"
    grid.tofile(bin_path)
    write_inside(stl, bin_path, GRID_MM, RES)
    if with_field:
        save_field_npz(tmp_path / "l.field.npz", ch, spec, source="mesh")
    return stl, bin_path, spec, inside_w, ch


def _record(stl, bin_path):
    return Record(topology="box", mode="type", parameters={}, rotation={}, stl_path=stl, voxel_path=bin_path, group_key="g",
                  placement={"noise": True})


# ------------------------------------------------------------------------------------------------ tests

def test_layout_and_ranges_non_cubic():
    inside, sdf, vec, radial = _sphere_field()
    noisy = inside.copy()
    noisy[0, 0, 0] = True
    x = vi.voxfield_channels(noisy, inside, sdf, vec, pitch=1.0, saturation=5.0)
    assert x.shape == (6, 20, 28, 24) and x.dtype == np.float32 and fi.CHANNELS["voxfield"] == vi.CHANNELS == 6
    assert fi.VECTOR_CHANNELS["voxfield"] == vi.VECTOR_CHANNELS == (1, 4) and "voxfield" in fi.FROM_NPZ
    assert np.array_equal(x, x.astype(np.float16).astype(np.float32))            # float16-representable by definition
    assert np.array_equal(x[0] > 0.5, noisy) and np.array_equal(x[5] > 0.5, inside)   # noise only in channel 0
    d = np.where(inside, -np.abs(sdf), np.abs(sdf))
    band = np.abs(d) < 5.0
    norm = np.linalg.norm(x[1:4], axis=0)
    assert np.all(norm[~band] == 0) and np.allclose(norm[band], 1.0, atol=2e-3)
    assert np.allclose(x[1:4][:, band], radial[:, band], atol=2e-3)              # the OUTWARD normal on both sides
    sat = vi.saturated_log(5.0)
    assert np.all(np.abs(x[4]) <= sat + 1e-3) and np.all(np.sign(x[4][np.abs(d) > 0.01]) == np.sign(d[np.abs(d) > 0.01]))
    ref = fi.log_distance(d, 64)                                                 # the dirlog scale inside the band
    assert np.allclose(x[4][band], ref[band], atol=1e-3)
    assert np.allclose(np.abs(x[4][~band]), sat, atol=1e-3)


def test_fallback_is_explicit_and_documented():
    inside, sdf, vec, radial = _sphere_field()
    fb = vi.voxfield_channels(inside, inside)                                    # no field: the voxel-only fallback
    ex = vi.voxfield_channels(inside, inside, sdf, vec)
    assert fb.shape == ex.shape and np.array_equal(fb[[0, 5]], ex[[0, 5]])
    band = (np.linalg.norm(fb[1:4], axis=0) > 0.5) & (np.linalg.norm(ex[1:4], axis=0) > 0.5)
    cos = np.sum(fb[1:4] * ex[1:4], axis=0)[band]
    assert np.median(np.degrees(np.arccos(np.clip(cos, -1, 1)))) < 10            # an approximation of the exact field
    with pytest.raises(ValueError):
        vi.voxfield_channels(inside, inside, sdf, None)                          # half a field is an error, not a fallback
    with pytest.raises(ValueError):
        fi.encode(np.where(inside, 1.0, -1.0), "voxfield")                      # never silently from a bare grid


def test_dataset_fallback_without_field(tmp_path, monkeypatch, igl_backend):
    monkeypatch.setenv("SHAPE_RECO_CACHE", str(tmp_path / "cache"))
    from src.inside_bits import read_inside
    from training.voxel_dataset import VoxelDataset
    stl, bin_path, *_ = _write_sample(tmp_path, with_field=False)
    assert vi.voxfield_source(bin_path) == "voxels"
    t, _ = VoxelDataset([_record(stl, bin_path)], RES, input_kind="voxfield")._load(0)
    material = np.fromfile(bin_path, np.uint8).reshape((RES,) * 3) > 0
    assert np.array_equal(t.numpy(), vi.voxfield_channels(material, read_inside(bin_path, RES)))


def test_train_infer_identity(tmp_path, monkeypatch, igl_backend):
    """The dataset path (.bin + .inside.npy + float16 .field.npz, sparse cache) and the loop's direct call on the same solid
    (its own float32 mesh_channels and parity, world axes) give the same tensor bit for bit - fresh and from the cache."""
    monkeypatch.setenv("SHAPE_RECO_CACHE", str(tmp_path / "cache"))
    from src.field_channels import bin_to_world
    from training.voxel_dataset import VoxelDataset
    stl, bin_path, spec, inside_w, ch = _write_sample(tmp_path)
    assert vi.voxfield_source(bin_path) == "field"
    ds = VoxelDataset([_record(stl, bin_path)], RES, input_kind="voxfield")
    first = ds._load(0)[0].numpy()
    assert vi.voxfield_cache_path(bin_path).is_file()
    second = ds._load(0)[0].numpy()                                             # from the sparse cache
    material_w = bin_to_world(np.fromfile(bin_path, np.uint8).reshape((RES,) * 3)) > 0
    direct = vi.voxfield_from_world(material_w, inside_w, ch[1], ch[2:5], spec.pitch)
    assert first.shape == (6, RES, RES, RES) and 200 <= (first[5] > 0.5).sum()
    assert np.array_equal(first.view(np.uint32), direct.view(np.uint32))
    assert np.array_equal(second.view(np.uint32), direct.view(np.uint32))
    assert (first[0] != first[5]).sum() == 3                                     # the three noise voxels, material only
    g = np.stack(np.gradient(first[4].astype(np.float64)), 0)                  # the frame: n is the ARRAY-axes gradient
    near = (np.abs(first[4]) > 0.2) & (np.linalg.norm(first[1:4], axis=0) > 0.5)
    cos = np.sum(g * first[1:4], axis=0)[near] / np.maximum(np.linalg.norm(g, axis=0)[near], 1e-12)
    assert near.sum() > 500 and np.median(cos) > 0.95


def test_cache_misses_when_an_encoding_constant_or_an_input_file_changes(tmp_path, monkeypatch, igl_backend):
    """T-19: the voxfield cache key (dataset_gen/src/cache_keys.py) holds the encoder's code and constants and every
    input file's identity. A changed constant is a miss and the tensor is recomputed (not the old one served); a .bin
    rewritten in place (revoxelize_parity.py, add_voxel_noise.py) is a miss; the pre-T-19 key is never produced."""
    import os
    monkeypatch.setenv("SHAPE_RECO_CACHE", str(tmp_path / "cache"))
    stl, bin_path, *_ = _write_sample(tmp_path)
    first = vi.load_voxfield(bin_path)
    key = vi.voxfield_cache_path(bin_path)
    assert key.is_file()
    deps = vi._sample_files(bin_path, RES)[3]
    legacy = hashlib.sha1("|".join([vi.CACHE_VERSION, str(RES), repr(vi.SATURATION_VOXELS), repr(vi.DEFAULT_LOG_SCALE)]
                                   + [f"{q.resolve()}:{q.stat().st_size}:{q.stat().st_mtime_ns}" for q in deps])
                          .encode()).hexdigest()[:24]
    assert key.stem != legacy                                                    # old entries are never read again
    monkeypatch.setattr(vi, "LOG_NORM_VOXELS", 32.0)                             # an encoding constant changes
    assert vi.voxfield_cache_path(bin_path) != key and not vi.voxfield_cache_path(bin_path).is_file()
    changed = vi.load_voxfield(bin_path)
    assert not np.array_equal(changed, first)                                    # recomputed with the new constant
    monkeypatch.setattr(vi, "LOG_NORM_VOXELS", 64.0)
    assert vi.voxfield_cache_path(bin_path) == key                               # back to the old code: the old entry
    assert np.array_equal(vi.load_voxfield(bin_path).view(np.uint32), first.view(np.uint32))
    st = bin_path.stat()
    os.utime(bin_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))         # the .bin rewritten in place
    assert vi.voxfield_cache_path(bin_path) != key


def test_sparse_round_trip(tmp_path):
    inside, sdf, vec, _ = _sphere_field()
    noisy = inside.copy()
    noisy[3, 4, 5] = True
    x = vi.voxfield_channels(noisy, inside, sdf, vec, saturation=4.0)
    path = tmp_path / "s.npz"
    vi.save_voxfield_sparse(path, x, saturation=4.0)
    y = vi.load_voxfield_sparse(path)
    assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    band = int((np.linalg.norm(x[1:4], axis=0) > 0).sum())
    stored = len(np.load(path)["index"])
    assert band <= stored < x[0].size                                           # only the band is stored
    fb = vi.voxfield_channels(inside, inside)                                    # the fallback round-trips too
    assert np.array_equal(vi.voxfield_from_sparse(vi.voxfield_to_sparse(fb)).view(np.uint32), fb.view(np.uint32))


def test_rotation_augmentation_turns_the_vectors(tmp_path, igl_backend):
    """rotate_network_input(voxfield(solid), g) == voxfield(solid rotated in the world by the rotation g stands for): the
    grid moves and the normal's components turn - checked on a tilted, off-centre L prism."""
    from src.field_channels import rotate_network_input
    from training.rotation_symmetry import GROUP, rotate_volume
    P = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], float)                      # world (x, y, z) -> array (z, -y, x)

    def encode(mesh, name):
        path = tmp_path / name
        mesh.export(path)
        spec, inside_w, ch = _world_field(path)
        return vi.voxfield_from_world(inside_w, inside_w, ch[1], ch[2:5], spec.pitch)

    x0 = encode(_l_prism(), "l0.stl")
    picked = [g for g in GROUP if not np.allclose(g, np.eye(3))][::7][:2]
    for n, g in enumerate(picked):
        want = encode(_l_prism(rotation=P @ g @ P), f"l{n + 1}.stl")
        got = rotate_network_input(x0, g, fi.VECTOR_CHANNELS["voxfield"])
        on_surface = np.abs(want[4]) < 1e-3                                       # |d| < 0.004 voxel: parity is a tie
        assert on_surface.sum() <= 2                                              # (measured: one centre 4e-5 voxel off)
        assert np.array_equal(got[[0, 5]][:, ~on_surface], want[[0, 5]][:, ~on_surface])
        band = (np.linalg.norm(want[1:4], axis=0) > 0) & (np.linalg.norm(got[1:4], axis=0) > 0) & ~on_surface
        assert band.sum() > 1000
        assert np.abs(got[1:4][:, band] - want[1:4][:, band]).max() < 5e-3
        assert np.abs(got[4] - want[4])[~on_surface].max() < 2e-3
        wrong = np.stack([rotate_volume(c, g) for c in x0[1:4]])                 # grid moved, components NOT turned
        assert np.abs(wrong[:, band] - want[1:4][:, band]).max() > 0.5


DIRLOG_INSIDE_SHA256 = {                                                          # from the untouched code (main fa651fde)
    "a": "f3ab9bfedda8c3cb5987079869d7df1885ccae387c0a4fa066705bdc32cc44a2",
    "b": "ad9613f371c452da107b27e89eda8a6aac2fbfea358ea33ec397e862bd4eaa9a",
}


def test_dirlog_inside_unchanged():
    occ = np.zeros((40, 40, 40), bool)
    occ[5:30, 8:20, 10:35] = True
    occ[5:15, 20:33, 10:20] = True
    occ[18:24, 10:16, :] = False
    noisy = occ ^ (np.random.default_rng(0).random(occ.shape) < 0.01)
    i, j, k = np.indices((32, 32, 32))
    ball = (i - 12) ** 2 + (j - 18) ** 2 / 2 + (k - 15) ** 2 < 60
    for name, (g, ins) in (("a", (noisy, occ)), ("b", (ball, ball))):
        x = fi.encode(np.where(g, 1.0, -1.0).astype(np.float32), "dirlog_inside", inside=ins)
        assert hashlib.sha256(np.ascontiguousarray(x, np.float32).tobytes()).hexdigest() == DIRLOG_INSIDE_SHA256[name]
