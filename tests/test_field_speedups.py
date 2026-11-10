"""The exact-field speed-ups of 2026-09-24 (docs/CODEMAP.md "Training plan v5 pieces", docs/data/field_igl_bench.json):
result-identical ones (cKDTree threads, the parity computed once per generated sample and once per residual-loop grid,
predict_records under --gpu-encode) are checked against the previous code path; the opt-in libigl backend
(SHAPE_RECO_FIELD_BACKEND=igl, a DIFFERENT - exact - answer where the one-pass query misses) is checked against the
exact query and skipped when libigl is not installed."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import field_channels as fc  # noqa: E402
from src.closest_point_field import SurfaceField, kdtree_workers, load_field  # noqa: E402


def _rotated_box(extents=(8.0, 5.0, 3.0), centre=(1.3, -0.7, 0.4)):
    m = trimesh.creation.box(extents=extents)
    m.apply_transform(trimesh.transformations.euler_matrix(0.3, -0.5, 0.9))       # no grid-symmetric ties
    m.apply_translation(centre)
    return m


@pytest.fixture
def backend(monkeypatch):
    def use(name):
        if name is None:
            monkeypatch.delenv("SHAPE_RECO_FIELD_BACKEND", raising=False)
        else:
            monkeypatch.setenv("SHAPE_RECO_FIELD_BACKEND", name)
    return use


# ------------------------------------------------------------------------------------------- identical-result changes

def test_kdtree_workers_env_and_default(monkeypatch):
    monkeypatch.delenv("SHAPE_RECO_KDTREE_WORKERS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)        # a pinned thread count is obeyed (see below)
    assert kdtree_workers() == -1
    monkeypatch.setenv("SHAPE_RECO_KDTREE_WORKERS", "2")
    assert kdtree_workers() == 2


def test_kdtree_threads_give_identical_queries(monkeypatch, backend):
    backend("numpy")
    m = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    field = SurfaceField(m.vertices, m.faces, k=24)
    P = np.random.default_rng(3).uniform(-16, 16, (4000, 3))
    out = {}
    for w in ("1", "-1"):
        monkeypatch.setenv("SHAPE_RECO_KDTREE_WORKERS", w)
        out[w] = [field.query(P, exact=e) for e in (True, False)]
    for a, b in zip(out["1"], out["-1"]):
        assert np.array_equal(a.closest, b.closest) and np.array_equal(a.face, b.face)
        assert np.array_equal(a.distance, b.distance) and np.array_equal(a.signed_distance, b.signed_distance)


def test_mesh_channels_with_precomputed_parity_is_identical(backend):
    backend("numpy")
    m = _rotated_box()
    field = SurfaceField(m.vertices, m.faces)
    spec = fc.grid_spec(-8, 8, 24)
    inside = fc.parity_occupancy(field, spec)
    assert np.array_equal(fc.mesh_channels(field, spec), fc.mesh_channels(field, spec, inside=inside))
    # the residual loop's own parity (kept unchanged) equals field_channels' on the loop's grid
    from reco_engine import residual_loop as rl
    assert np.array_equal(rl.parity_occupancy(field, spec), inside)


def test_residual_loop_channels_compute_parity_once(monkeypatch, backend):
    backend("numpy")
    from reco_engine import residual_loop as rl
    m = _rotated_box()
    field = SurfaceField(m.vertices, m.faces)
    spec = fc.grid_spec(-8, 8, 24)
    expected = fc.mesh_channels(field, spec)                   # the old _channels: mesh_channels, then the parity again
    inside = rl.parity_occupancy(field, spec)
    expected[1] = np.where(inside, 1.0, -1.0) * np.abs(expected[1])
    expected[0] = inside
    calls = []
    real_fc, real_rl = fc.parity_occupancy, rl.parity_occupancy
    monkeypatch.setattr(fc, "parity_occupancy", lambda f, s: (calls.append("fc"), real_fc(f, s))[1])
    monkeypatch.setattr(rl, "parity_occupancy", lambda f, s: (calls.append("rl"), real_rl(f, s))[1])
    got = rl._channels(field, spec, False)
    assert len(calls) == 1
    assert np.array_equal(got, expected)


def test_parity_of_file_is_cached_per_file_and_identical(tmp_path, backend):
    backend("numpy")
    stl = tmp_path / "part.stl"
    _rotated_box().export(stl)
    spec = fc.inscale_spec(20.0, 16)
    fc._FILE_CACHE.clear()
    a = fc.parity_of_file(stl, spec)
    assert np.array_equal(a, fc.parity_occupancy(load_field(stl), spec))
    assert not a.flags.writeable
    assert fc.parity_of_file(stl, spec) is a                   # the same file -> the cached array
    assert fc.field_of_file(stl) is fc.field_of_file(stl)
    assert fc.parity_of_file(stl, fc.inscale_spec(20.0, 12)) is not a   # another grid -> another entry
    # a rewritten file (another size) is recomputed
    trimesh.creation.icosphere(subdivisions=2, radius=4.0).export(stl)
    b = fc.parity_of_file(stl, spec)
    assert b is not a and not np.array_equal(a, b)
    assert len(fc._FILE_CACHE) <= fc._FILE_CACHE_MAX


def test_inside_bits_and_parity_voxelizer_share_the_cached_parity(tmp_path, backend, monkeypatch):
    backend("numpy")
    from src import inside_bits, voxelization
    stl = tmp_path / "s.stl"
    _rotated_box().export(stl)
    spec = fc.inscale_spec(20.0, 16)
    fc._FILE_CACHE.clear()
    calls = []
    real = fc.parity_occupancy
    monkeypatch.setattr(fc, "parity_occupancy", lambda f, s: (calls.append(1), real(f, s))[1])
    assert voxelization._voxelize_parity(str(stl), str(tmp_path / "s.bin"), 16, 20.0)
    bits = inside_bits.compute_inside(stl, 20.0, 16)
    ch = fc.mesh_channels(fc.field_of_file(stl), spec, inside=fc.parity_of_file(stl, spec))
    assert len(calls) == 1                                     # three consumers, one parity
    vol = np.fromfile(tmp_path / "s.bin", np.uint8).reshape(16, 16, 16)
    assert np.array_equal(vol > 0, bits)
    assert np.array_equal(fc.bin_to_world(bits), ch[0] > 0)


def test_predict_records_encodes_raw_grids_under_gpu_encode(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from training import train_common as tc
    from training import train_type_cnn as ttc
    from training.manifest import Record
    rng = np.random.default_rng(0)
    records = []
    for i in range(3):
        occ = np.zeros((16, 16, 16), np.uint8)
        occ[4 + i:11, 3:12, 5:9 + i] = 1
        occ.tofile(tmp_path / f"s{i}.bin")
        records.append(Record(topology="box", mode="type", parameters={}, rotation={}, stl_path=tmp_path / f"s{i}.stl",
                              voxel_path=tmp_path / f"s{i}.bin", group_key="g", placement={}, meta={"subclass": "full"}))
    device = torch.device("cpu")
    monkeypatch.setattr(tc, "SPEED", tc.Speed())
    monkeypatch.setattr(ttc, "SPEED", tc.SPEED)
    for hier in (False, True):                                  # TypeDataset and the hierarchical _PlainDataset paths
        torch.manual_seed(0)
        model = (ttc.HierTypeCNN(in_channels=5) if hier else ttc.TypeCNN(in_channels=5)).eval()
        tc.SPEED.gpu_kind = None
        cpu = ttc.predict_records(model, records, device, 2, resolution=16, input_kind="dirlog")
        tc.SPEED.gpu_kind = "dirlog"                            # --gpu-encode: the loader ships uint8 grids
        fast = ttc.predict_records(model, records, device, 2, resolution=16, input_kind="dirlog")
        for key in cpu:
            assert fast[key].shape == cpu[key].shape and fast[key].shape[0] == 3
            finite = np.isfinite(cpu[key])
            assert np.allclose(fast[key][finite], cpu[key][finite], atol=2e-2)   # gpu_encode's normals differ <= 2e-3
            assert np.array_equal(fast[key].argmax(1), cpu[key].argmax(1))


def test_cudnn_benchmark_flag_is_opt_in():
    import argparse
    from training import train_common as tc
    p = argparse.ArgumentParser()
    tc.add_speed_arguments(p)
    assert p.parse_args([]).cudnn_benchmark is False
    assert p.parse_args(["--cudnn-benchmark"]).cudnn_benchmark is True


# ------------------------------------------------------------------------------------------- the opt-in igl backend

try:
    import igl  # noqa: F401
    HAVE_IGL = True
except ImportError:
    HAVE_IGL = False
needs_igl = pytest.mark.skipif(not HAVE_IGL, reason="SHAPE_RECO_FIELD_BACKEND=igl needs libigl (pip install libigl)")


def test_igl_backend_fails_loudly_without_the_package(backend, monkeypatch):
    monkeypatch.setitem(sys.modules, "igl", None)               # "import igl" then raises ImportError
    backend("igl")
    with pytest.raises(ImportError):
        fc.field_backend()


@needs_igl
def test_igl_backend_is_selected_only_by_env(backend):
    backend("igl")
    assert fc.field_backend() == "igl"
    backend("numpy")
    assert fc.field_backend() == "numpy"


@needs_igl
def test_igl_query_equals_the_exact_query(backend):
    m = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    field = SurfaceField(m.vertices, m.faces)
    P = np.random.default_rng(1).uniform(-16, 16, (3000, 3))
    backend("numpy")
    ref = field.query(P, exact=True)
    backend("igl")
    q = field.query(P, exact=False)                             # exact whatever the flag says
    assert np.abs(q.distance - ref.distance).max() < 1e-9
    assert np.mean(np.linalg.norm(q.closest - ref.closest, axis=1) < 1e-9) > 0.99
    assert np.array_equal(np.sign(q.signed_distance), np.sign(ref.signed_distance))
    assert q.face.shape == (3000,) and (q.face < len(m.faces)).all()


@needs_igl
def test_igl_mesh_channels_skip_refinement_and_match_the_default(backend, monkeypatch):
    m = _rotated_box()
    field = SurfaceField(m.vertices, m.faces)
    spec = fc.grid_spec(-8, 8, 24)
    backend("numpy")
    ref = fc.mesh_channels(field, spec)
    backend("igl")
    refined = []
    real = fc.refined_field
    monkeypatch.setattr(fc, "refined_field", lambda f, e: (refined.append(1), real(f, e))[1])
    got = fc.mesh_channels(field, spec)
    assert not refined                                          # the raw 12-face box is queried directly
    assert np.array_equal(got[0], ref[0])                       # the parity sign is untouched
    assert np.abs(got[1] - ref[1]).max() < 1e-5                 # |d| exact both ways on a box (max diff measured 1e-14)
    assert np.abs(got[2:] - ref[2:]).max() < 1e-5


@needs_igl
def test_igl_never_reports_a_larger_distance_than_the_one_pass_query(backend):
    """The one-pass query returns a real surface point, so its distance is an upper bound; igl's must never exceed it."""
    m = trimesh.creation.cylinder(radius=6.0, height=14.0, sections=96)
    m.apply_transform(trimesh.transformations.euler_matrix(0.4, 0.7, 0.2))
    field = SurfaceField(m.vertices, m.faces)
    spec = fc.grid_spec(-12, 12, 32)
    backend("numpy")
    ref = fc.mesh_channels(field, spec)
    backend("igl")
    got = fc.mesh_channels(field, spec)
    assert (np.abs(got[1]) <= np.abs(ref[1]) + 1e-9).all()
    assert np.abs(got[1] - ref[1]).max() < 0.1                  # the documented one-pass miss size (<= 0.05 mm at 64^3)


def test_blocked_one_pass_query_is_bit_identical_to_one_block(monkeypatch):
    """_dense_one_pass in blocks of DENSE_BLOCK points gives exactly the unblocked answer (memory bound, same result)."""
    pytest.importorskip("torch")
    trimesh = pytest.importorskip("trimesh")
    import src.closest_point_field as cpf
    m = trimesh.creation.cylinder(radius=4.0, height=9.0, sections=40)
    field = cpf.SurfaceField(m.vertices, m.faces)
    pts = np.random.default_rng(1).uniform(-8, 8, (3000, 3))
    monkeypatch.setenv("SHAPE_RECO_FIELD_BACKEND", "torch")
    whole = cpf._dense_one_pass(field, pts)
    monkeypatch.setattr(cpf, "DENSE_BLOCK", 700)
    blocked = cpf._dense_one_pass(field, pts)
    for a, b in zip(whole, blocked):
        assert np.array_equal(a, b)


def test_kdtree_workers_follow_pinned_threads(monkeypatch):
    import src.closest_point_field as cpf
    monkeypatch.delenv("SHAPE_RECO_KDTREE_WORKERS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert cpf.kdtree_workers() == -1
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert cpf.kdtree_workers() == 1
    monkeypatch.setenv("SHAPE_RECO_KDTREE_WORKERS", "4")
    assert cpf.kdtree_workers() == 4


def test_new_data_uses_the_igl_field_backend_by_default_and_records_it(tmp_path, monkeypatch):
    """User decision 2026-09-24: new samples get the exact libigl field unless a backend is set explicitly."""
    pytest.importorskip("igl")
    trimesh = pytest.importorskip("trimesh")
    from src import sample_writer
    box = trimesh.creation.box(extents=(9.0, 5.0, 3.0))
    box.apply_translation((6.0, -4.0, 2.5))
    stl = tmp_path / "b.stl"
    box.export(stl)
    monkeypatch.delenv("SHAPE_RECO_FIELD_BACKEND", raising=False)
    rec = sample_writer.write_field(stl, tmp_path / "b.field.npz", resolution=16, grid_mm=40.0, field_points=32)
    assert rec["backend"] == "igl"
    monkeypatch.setenv("SHAPE_RECO_FIELD_BACKEND", "numpy")                   # an explicit choice is honoured
    rec = sample_writer.write_field(stl, tmp_path / "c.field.npz", resolution=16, grid_mm=40.0, field_points=32)
    assert rec["backend"] == "numpy"
