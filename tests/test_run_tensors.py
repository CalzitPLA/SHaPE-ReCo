"""reco_engine.run_tensors: the band-only SparseGrid (round trip, windows, occupancy bits, storage) and the once-per-run
RunTensors, checked against the dense field_channels tensors on an ASYMMETRIC, off-centre solid on a non-cubic grid."""

import sys
import threading
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

pytest.importorskip("igl")

from reco_engine import run_tensors as rt  # noqa: E402
from reco_engine.run_tensors import RunTensors, SparseGrid  # noqa: E402
from src.closest_point_field import SurfaceField  # noqa: E402
from src.field_channels import GridSpec, mesh_channels, parity_occupancy, voxel_channels  # noqa: E402
from src.local_grids import BoxGrid  # noqa: E402

# An L-shaped prism, nothing about it symmetric or centred: the long arm runs along x, the short one along y, the
# extrusion along z; every extent differs. World mm.
L_XY = [(3.0, -7.0), (25.0, -7.0), (25.0, -2.0), (9.0, -2.0), (9.0, 6.0), (3.0, 6.0)]
L_Z = (11.0, 15.0)
BOX = BoxGrid(lo=(0.0, -10.0, 8.0), shape=(28, 19, 11), pitch=1.0)          # X != Y != Z


def l_prism() -> SurfaceField:
    """The closed, outward-oriented triangle mesh of the prism: caps as fans from the reflex corner, two triangles per
    side wall."""
    n = len(L_XY)
    v = np.array([(x, y, z) for z in L_Z for x, y in L_XY], float)            # bottom ring 0..5, top ring 6..11
    faces = []
    for a, b in ((4, 5), (5, 0), (0, 1), (1, 2)):                             # corner 3 (9, -2) sees every other corner
        faces += [(3, b, a), (3 + n, a + n, b + n)]                           # bottom looks down, top looks up
    for a in range(n):
        b = (a + 1) % n
        faces += [(a, b, b + n), (a, b + n, a + n)]
    return SurfaceField(v, np.array(faces))


def l_inside(x, y, z):
    arm_x = (x > 3) & (x < 25) & (y > -7) & (y < -2)
    arm_y = (x > 3) & (x < 9) & (y > -7) & (y < 6)
    return (arm_x | arm_y) & (z > L_Z[0]) & (z < L_Z[1])


class CountingField:
    """A SurfaceField that counts its closest-point queries (under the igl backend the field itself is queried)."""

    def __init__(self, field):
        self._field = field
        self.vertices, self.faces, self.bounds = field.vertices, field.faces, field.bounds
        self.face_normals = field.face_normals
        self.queries = 0
        self.points = 0

    def query(self, points, **kwargs):
        self.queries += 1
        self.points += len(points)
        return self._field.query(points, **kwargs)


def random_field(seed=0, shape=(7, 9, 11), fraction=0.3):
    rng = np.random.default_rng(seed)
    dense = rng.normal(size=(5,) + shape).astype(np.float32)
    dense[0] = rng.random(shape) < 0.5
    return dense, rng.random(shape) < fraction


FILL = np.array([[0, -9, 0, 0, 0], [1, 9, 0, 0, 0]], np.float32)


# ------------------------------------------------------------------------------------------------ SparseGrid

def test_round_trip_in_the_band_and_fill_outside():
    dense, band = random_field()
    grid = SparseGrid.from_dense(dense, band, fill=FILL)
    out = grid.dense()
    assert out.dtype == np.float32 and out.shape == dense.shape
    assert np.array_equal(out[:, band], dense[:, band])                       # exact inside the band
    assert np.array_equal(out[0], dense[0])                                   # occupancy exact everywhere
    occ = dense[0] > 0.5
    assert np.array_equal(grid.occupancy(), occ)
    assert np.all(out[1][~band & occ] == 9) and np.all(out[1][~band & ~occ] == -9)
    assert np.all(out[2:][:, ~band] == 0)
    assert np.array_equal(grid.band_mask(), band)


def test_nothing_outside_the_band_is_stored():
    dense, band = random_field(1)
    grid = SparseGrid.from_dense(dense, band, fill=FILL)
    assert grid.n_band == int(band.sum()) and grid.values.shape == (band.sum(), 4)
    assert grid.index.dtype == np.int32 and np.array_equal(grid.index, np.flatnonzero(band))
    bits = (dense.shape[1] * dense.shape[2] * ((dense.shape[3] + 7) // 8))
    assert grid.nbytes == band.sum() * (4 + 16) + bits + FILL.nbytes
    secret = dense.copy()
    secret[1:, ~band] = 12345.0                                               # a value outside the band cannot come back
    assert not np.any(SparseGrid.from_dense(secret, band, fill=FILL).dense() == 12345.0)


def test_from_indices_sorts_and_rejects_duplicates():
    dense, band = random_field(2)
    ref = SparseGrid.from_dense(dense, band, fill=FILL)
    order = np.random.default_rng(0).permutation(ref.n_band)
    grid = SparseGrid(ref.shape, ref.index[order], ref.values[order], ref.occupancy_bits, fill=FILL)
    assert np.array_equal(grid.dense(), ref.dense())
    with pytest.raises(ValueError):
        SparseGrid(ref.shape, np.r_[ref.index, ref.index[:1]], np.r_[ref.values, ref.values[:1]], ref.occupancy_bits)


def test_window_equals_the_slice_of_the_dense_array():
    dense, band = random_field(3, shape=(13, 10, 21))
    grid = SparseGrid.from_dense(dense, band, fill=FILL)
    full = grid.dense()
    rng = np.random.default_rng(5)
    windows = [((0, 13), (0, 10), (0, 21)), ((2, 3), (4, 5), (6, 7)), ((5, 5), (0, 10), (0, 21)),
               ((0, 13), (3, 9), (7, 17)), ((-4, 99), (2, 4), (15, 40))]
    for _ in range(30):
        lo = [int(rng.integers(0, n)) for n in full.shape[1:]]
        windows.append(tuple((a, int(rng.integers(a, n + 1))) for a, n in zip(lo, full.shape[1:])))
    for w in windows:
        sl = tuple(slice(max(a, 0), b) for a, b in w)
        assert np.array_equal(grid.dense(window=w), full[(slice(None),) + sl]), w
        assert np.array_equal(grid.occupancy(w), dense[0][sl] > 0.5), w
        assert np.array_equal(grid.band_mask(w), band[sl]), w
    w = ((1, 9), (2, 8), (3, 20))
    assert np.array_equal(grid.dense(channels=["vz", "sdf"], window=w), full[[4, 1], 1:9, 2:8, 3:20])
    assert np.array_equal(grid.dense(channels=0, window=(None, slice(2, 8), None)), full[[0], :, 2:8, :])


def test_a_window_does_not_allocate_the_grid():
    shape = (160, 150, 170)                                                   # 4.08 M cells: 81.6 MB dense
    rng = np.random.default_rng(0)
    index = np.sort(rng.choice(int(np.prod(shape)), 200_000, replace=False))
    values = rng.normal(size=(len(index), 4)).astype(np.float32)
    occ = np.zeros(shape, bool)
    occ[40:120, 30:100, 50:150] = True
    grid = SparseGrid(shape, index, values, SparseGrid.pack_occupancy(occ), fill=FILL)
    del occ
    w = ((0, 160), (60, 84), (70, 94))                                        # spans ALL of x: 92 160 cells
    tracemalloc.start()
    out = grid.dense(window=w)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert out.shape == (5, 160, 24, 24)
    assert peak < 4 * out.nbytes < grid.dense_nbytes / 10, (peak, out.nbytes)
    lin = (np.arange(160)[:, None, None] * 150 + np.arange(60, 84)[None, :, None]) * 170 + np.arange(70, 94)[None, None, :]
    hit = np.isin(lin, index)
    assert np.array_equal(out[1][hit], values[np.searchsorted(index, lin[hit]), 0])
    assert int(hit.sum()) == int(grid.band_mask(w).sum()) > 0


def test_save_and_load(tmp_path):
    dense, band = random_field(4, shape=(6, 17, 9))
    grid = SparseGrid.from_dense(dense, band, fill=FILL)
    path = tmp_path / "sub" / "grid.npz"
    grid.save(path, note=np.arange(3))
    assert path.exists() and not (tmp_path / "sub" / "grid.npz.npz").exists()
    back, extra = SparseGrid.load(path, extra=True)
    assert back.values.dtype == np.float32 and back.index.dtype == np.int32
    assert back.shape == grid.shape and back.channels == grid.channels
    assert np.array_equal(back.dense(), grid.dense()) and np.array_equal(extra["note"], np.arange(3))
    grid.save(tmp_path / "half.npz", dtype="float16")
    half = SparseGrid.load(tmp_path / "half.npz")
    assert half.values.dtype == np.float16 and half.nbytes < grid.nbytes
    assert np.array_equal(half.dense()[:, band], dense.astype(np.float16).astype(np.float32)[:, band])
    assert np.array_equal(half.occupancy(), dense[0] > 0.5)


def test_a_field_without_an_occupancy_channel():
    dense, band = random_field(6)
    grid = SparseGrid.from_dense(dense[1:], band, channels=("a", "b", "c", "d"), occupancy_channel=None,
                                 fill=[1, 2, 3, 4])
    out = grid.dense()
    assert np.array_equal(out[:, band], dense[1:][:, band])
    assert all(np.all(out[c][~band] == c + 1) for c in range(4))


# ------------------------------------------------------------------------------------------------ RunTensors

def test_grid_axes_and_parity_match_field_channels():
    spec = GridSpec((-3.0, -11.5, 6.0), (27.0, 18.5, 36.0), 30)
    xs, ys, zs, pitch = rt.grid_axes(spec)
    c = spec.centres()
    assert np.array_equal(xs, c[:, 0, 0, 0]) and np.array_equal(ys, c[0, :, 0, 1]) and np.array_equal(zs, c[0, 0, :, 2])
    assert pitch == spec.pitch
    assert np.array_equal(rt.window_centres(spec, ((3, 9), (0, 30), (11, 12))), c[3:9, :, 11:12])
    field = l_prism()
    assert np.array_equal(rt.parity_grid(field, spec), parity_occupancy(field, spec))


def test_asymmetric_off_centre_shape_on_a_box_grid():
    """The axis check: an L prism whose occupancy is known analytically, on a 28 x 19 x 11 grid."""
    field = l_prism()
    tensors = RunTensors(field, BOX, band_voxels=3, backend="igl")
    x, y, z = np.meshgrid(BOX.axis(0), BOX.axis(1), BOX.axis(2), indexing="ij")
    truth = l_inside(x, y, z)
    assert truth.sum() == (22 * 5 + 6 * 8) * 4                                # asymmetric: no transposed copy matches
    assert np.array_equal(tensors.occupancy(), truth)                         # exact everywhere, band or not
    assert np.array_equal(tensors.parity(), truth)
    band = tensors.band_mask()
    assert 0 < band.sum() < band.size and not band[27, 18, 10]                 # a far corner is not stored
    assert tensors.grid.n_band == band.sum()
    dense = rt.dense_channels(field, BOX, backend="igl")
    assert np.array_equal(tensors.channels()[:, band], dense[:, band])        # max abs difference 0 in the band
    assert np.abs(dense[1][~band]).min() > 2.0                                # nothing nearer than 2 voxels left out
    # a cell 1.5 mm beyond the +x end of the long arm points back along -x; one above the short arm along -z
    i, j, k = 26, 5, 4                                                        # centre (26.5, -4.5, 12.5)
    assert np.allclose(tensors.channels(window=((i, i + 1), (j, j + 1), (k, k + 1)))[:, 0, 0, 0], [0, -1.5, -1.5, 0, 0])
    i, j, k = 5, 14, 8                                                        # centre (5.5, 4.5, 16.5)
    assert np.allclose(tensors.channels(window=((i, i + 1), (j, j + 1), (k, k + 1)))[:, 0, 0, 0], [0, -1.5, 0, 0, -1.5])
    w = ((2, 12), (10, 19), (0, 11))                                          # a window around the short arm's end
    assert np.array_equal(tensors.channels(window=w)[:, band[2:12, 10:19]], dense[:, 2:12, 10:19][:, band[2:12, 10:19]])
    assert np.array_equal(tensors.channels(window=w, complete=True), dense[:, 2:12, 10:19])
    far = tensors.channels()[:, ~band]
    assert np.all(far[1] == -3.0) and np.all(far[2:] == 0)                    # the part is thinner than the band: all outside


def test_the_band_equals_mesh_channels_on_a_cube():
    field = l_prism()
    spec = GridSpec((-3.0, -11.5, -2.0), (27.0, 18.5, 28.0), 30)
    with rt._field_backend("igl"):
        ref = mesh_channels(field, spec)
    assert np.array_equal(rt.dense_channels(field, spec, backend="igl"), ref)
    tensors = RunTensors(field, spec, band_voxels=4, backend="igl")
    band = tensors.band_mask()
    assert np.array_equal(tensors.channels()[:, band], ref[:, band])
    assert np.array_equal(tensors.channels()[0], ref[0])
    assert np.array_equal(tensors.channels(complete=True), ref)


def test_the_default_backend_gives_mesh_channels_in_the_band():
    """The one-pass query on the refined mesh, in blocks: the same numbers as the dense call (CPU here, conftest)."""
    field = l_prism()
    spec = GridSpec((-3.0, -11.5, -2.0), (27.0, 18.5, 28.0), 20)
    ref = mesh_channels(field, spec)
    tensors = RunTensors(field, spec, band_voxels=3)
    band = tensors.band_mask()
    assert np.array_equal(tensors.channels()[:, band], ref[:, band])


def test_computed_once_per_run(monkeypatch):
    field = CountingField(l_prism())
    parity_calls = []
    real = rt.parity_grid
    monkeypatch.setattr(rt, "parity_grid", lambda f, s: parity_calls.append(1) or real(f, s))
    tensors = RunTensors(field, BOX, band_voxels=3, backend="igl")
    assert field.queries == 0 and tensors.builds == 0 and not parity_calls    # lazy: nothing until the first access
    grids = []
    threads = [threading.Thread(target=lambda: grids.append(tensors.grid)) for _ in range(4)]   # worker + readers
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(g is grids[0] for g in grids)
    first = tensors.channels(window=((0, 10), (0, 10), (0, 11)))
    tensors.occupancy(), tensors.parity(), tensors.channels(), tensors.band_mask()
    assert np.array_equal(first, tensors.channels(window=((0, 10), (0, 10), (0, 11))))
    assert tensors.builds == 1 and len(parity_calls) == 1
    assert field.queries == 1 and field.points == tensors.grid.n_band < np.prod(BOX.shape)   # the band's centres only
    tensors.channels(window=((24, 28), (15, 19), (0, 4)), complete=True)     # the one request that queries again
    assert field.queries == 2 and field.points == tensors.grid.n_band + 64 and tensors.builds == 1


def test_disk_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("SHAPE_RECO_TENSOR_CACHE", raising=False)
    monkeypatch.delenv("SHAPE_RECO_CACHE", raising=False)
    assert RunTensors(l_prism(), BOX).cache_dir is None                       # off unless asked for
    first = RunTensors(CountingField(l_prism()), BOX, band_voxels=3, backend="igl", cache=tmp_path)
    ref = first.channels()
    assert not first.from_cache and len(list(tmp_path.glob("*.npz"))) == 1
    field = CountingField(l_prism())
    again = RunTensors(field, BOX, band_voxels=3, backend="igl", cache=tmp_path)
    assert np.array_equal(again.channels(), ref) and np.array_equal(again.parity(), first.parity())
    assert again.from_cache and field.queries == 0 and again.builds == 1
    other = RunTensors(CountingField(l_prism()), BOX, band_voxels=2, backend="igl", cache=tmp_path)
    assert not other.grid is None and not other.from_cache                    # the band is part of the key
    monkeypatch.setenv("SHAPE_RECO_TENSOR_CACHE", "off")
    assert RunTensors(l_prism(), BOX).cache_dir is None
    monkeypatch.setenv("SHAPE_RECO_TENSOR_CACHE", str(tmp_path))
    assert RunTensors(l_prism(), BOX).cache_dir == tmp_path


def test_float16_values():
    field = l_prism()
    exact = RunTensors(field, BOX, band_voxels=3, backend="igl")
    half = RunTensors(field, BOX, band_voxels=3, backend="igl", dtype="float16")
    assert half.grid.values.dtype == np.float16 and half.nbytes < exact.nbytes
    assert np.array_equal(half.occupancy(), exact.occupancy())
    assert np.abs(half.channels() - exact.channels()).max() < 0.01


# ------------------------------------------------------------------------------------------------ pieces

def test_piece_channels_equal_voxel_channels_in_the_window():
    spec = GridSpec((-3.0, -11.5, -2.0), (27.0, 18.5, 28.0), 30)
    c = spec.centres()
    occ = l_inside(c[..., 0], c[..., 1], c[..., 2] - 0.0)
    assert occ.sum() > 100
    ref = voxel_channels(occ, spec)
    window, part = rt.piece_channels(occ, spec, margin=3)
    assert window == rt.window_of(occ, 3) and part.shape[1:] == tuple(b - a for a, b in window)
    assert part.size < ref.size / 3
    sl = (slice(None),) + rt.window_slices(window)
    assert np.array_equal(part[0], ref[sl][0])
    assert np.abs(part - ref[sl]).max() < 1e-5
    with pytest.raises(ValueError):
        rt.piece_channels(occ, spec, window=((0, 5), (0, 5), (0, 5)))
    assert rt.piece_channels(np.zeros_like(occ), spec)[1].size == 0
