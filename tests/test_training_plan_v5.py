"""Pieces added for the v5 training plan (docs/training_plan_v5.tex): noise in the shared sample writer, the voxel axis
trainer's label and augmentation, rotation_symmetry.record_dims for records without ratio parameters, and the cost /
observability report helpers. FreeCAD and stltovoxel are stubbed; the real paths run in the generator smoke runs."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))
sys.path.insert(0, str(ROOT / "scripts"))

from src import sample_writer, voxel_noise  # noqa: E402
from training import rotation_symmetry as rs  # noqa: E402


def _stub_voxelize(monkeypatch, volume):
    def fake(stl_path, bin_path, resolution=64, grid_mm=40.0, executable="stltovoxel", voxelizer=None, report=None):
        from src.voxelization import voxelizer_for
        if report is not None:
            report["voxelizer"] = voxelizer_for("fixed", "origin", voxelizer)
        volume.astype(np.uint8).tofile(bin_path)
        return volume.copy()
    monkeypatch.setattr(sample_writer, "voxelize", fake)
    from src import inside_bits                                 # the stub STL is not a mesh: no inside bits here
    monkeypatch.setattr(inside_bits, "write_inside", lambda stl, bin_path, grid_mm=40.0, resolution=64: "stub.inside.npy")


@pytest.fixture
def block():
    v = np.zeros((16, 16, 16), np.uint8)
    v[4:12, 5:10, 3:13] = 1
    return v


def _finish(tmp_path, name, noise):
    stl = tmp_path / "box" / f"{name}.stl"
    stl.parent.mkdir(parents=True, exist_ok=True)
    stl.write_text("solid x\nendsolid x\n")
    return sample_writer.finish_sample(stl, {"topology": "box"}, tmp_path, resolution=16, with_field=False, noise=noise)


def test_finish_sample_without_noise_keeps_the_grid(tmp_path, monkeypatch, block):
    _stub_voxelize(monkeypatch, block)
    rec = _finish(tmp_path, "a", None)
    assert rec["placement"]["noise"] is False and rec["placement"]["voxels"] == int(block.sum())
    assert "voxels_clean" not in rec["placement"]


def test_finish_sample_noise_is_written_recorded_and_reproducible(tmp_path, monkeypatch, block):
    _stub_voxelize(monkeypatch, block)
    spec = dict(config=voxel_noise.VoxelNoiseConfig(salt_prob=0.05, pepper_prob=0.1), prob=1.0)
    rec = _finish(tmp_path, "a", spec)
    stored = np.fromfile(tmp_path / rec["voxel_path"], np.uint8)
    assert rec["placement"]["noise"] is True and rec["placement"]["voxels_clean"] == int(block.sum())
    assert rec["placement"]["voxels"] == int(stored.sum()) != int(block.sum())
    clean = np.fromfile(tmp_path / rec["voxel_path"].replace(".bin", ".clean.bin"), np.uint8)
    assert np.array_equal(clean, block.reshape(-1))            # the clean grid is kept next to the noised one
    again = _finish(tmp_path, "a", spec)                        # same path -> same rng -> same grid
    assert again["placement"]["voxels"] == rec["placement"]["voxels"]


def test_noise_prob_zero_records_a_clean_sample(tmp_path, monkeypatch, block):
    _stub_voxelize(monkeypatch, block)
    rec = _finish(tmp_path, "a", dict(config=voxel_noise.VoxelNoiseConfig(salt_prob=0.05), prob=0.0))
    assert rec["placement"]["noise"] is False and rec["placement"]["voxels"] == int(block.sum())


def test_noise_cli_helpers_mirror_the_generator_flags():
    import argparse
    ap = argparse.ArgumentParser()
    voxel_noise.add_noise_arguments(ap)
    assert voxel_noise.noise_spec_from_args(ap.parse_args([])) is None
    spec = voxel_noise.noise_spec_from_args(ap.parse_args(["--noise-pepper-prob", "0.02", "--noise-prob", "0.5"]))
    assert spec["prob"] == 0.5 and spec["config"].pepper_prob == 0.02 and spec["config"].enabled


# ------------------------------------------------------------------------------------------------ axis trainer

def _record(topology, q, **meta):
    return SimpleNamespace(topology=topology, rotation=dict(qx=q[0], qy=q[1], qz=q[2], qw=q[3]), meta=meta,
                           parameters={}, placement={})


def test_label_axis_is_local_z_in_the_array_frame():
    from training.train_axis import label_axis
    assert np.allclose(label_axis(_record("cylinder", [0, 0, 0, 1])), rs.A_WORLD_TO_ARRAY @ [0, 0, 1])
    s = np.sqrt(0.5)                                            # 90 deg about world x: local z -> world -y
    assert np.allclose(label_axis(_record("cylinder", [s, 0, 0, s])), rs.A_WORLD_TO_ARRAY @ [0, -1, 0], atol=1e-7)


def test_pca_axis_of_an_offcentre_rod_and_the_grid_augmentation_turn_together():
    """An off-centre rod along array axis 2 (asymmetric placement, not a cube): the PCA axis finds it, and a grid
    rotation g moves the volume's PCA axis exactly as it moves the label (g @ axis) - what AxisDataset relies on."""
    from training.train_axis import angle_deg, pca_axis
    v = np.zeros((32, 32, 32), np.float32)
    v[3:9, 20:26, 2:30] = 1
    assert angle_deg(pca_axis(v)[None], np.array([[0, 0, 1.0]]), np.array([0.0]))[0] < 1e-6
    for g in rs.GROUP[[1, 5, 13]]:
        rotated = rs.rotate_volume(v, g)
        assert angle_deg(pca_axis(rotated)[None], (g @ [0, 0, 1.0])[None], np.array([0.0]))[0] < 1e-6


def test_angle_is_sign_free_only_for_cylinder_and_torus():
    from training.train_axis import angle_deg
    a, b = np.array([[0, 0, 1.0]]), np.array([[0, 0, -1.0]])
    assert angle_deg(a, b, np.array([0.0]))[0] == pytest.approx(0.0)
    assert angle_deg(a, b, np.array([1.0]))[0] == pytest.approx(180.0)


def test_record_dims_falls_back_to_dims_mm():
    with_ratios = SimpleNamespace(parameters={k: v for k, v in zip(rs.DIM_KEYS, (3.0, 2.0, 1.0))}, meta={})
    plate = SimpleNamespace(parameters={}, meta={"dims_mm": {"x_half": 9.0, "y_half": 4.0, "z_half": 0.5}})
    assert rs.record_dims(with_ratios) == [3.0, 2.0, 1.0] and rs.record_dims(plate) == [9.0, 4.0, 0.5]


# ------------------------------------------------------------------------------------------------ reports

def test_timing_report_reads_field_and_freecad_seconds(tmp_path):
    import timing_report
    shard = tmp_path / "train" / "shard0" / "box"
    shard.mkdir(parents=True)
    recs = [dict(topology="box", subclass="full", sample_kind="whole", placement=dict(field=dict(seconds=s)))
            for s in (10.0, 20.0, 30.0)]
    recs.append(dict(topology="box", subclass="frame", placement=dict(field=dict(seconds=5.0)),
                     freecad=dict(seconds_build=7.0, seconds_export=1.0)))
    (shard / "dataset_metadata.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    field, build, _ = timing_report.collect(tmp_path)
    assert timing_report.summarize(field["box/full"])["median"] == 20.0
    assert build["box/frame"] == [8.0]


def test_second_moment_half_extents_are_exact_for_an_intact_box():
    """check_observability_v5's estimator h_i = sqrt(3 lambda_i): on a grid-aligned voxel box it reads h sqrt(1 - 1/(4h^2))
    (n voxel centres have variance (n^2 - 1) / 12), i.e. 3 % short on a 4-voxel plate, 0.01 % on a 30-voxel side; the
    whole-voxel extent count it replaces is off by up to a voxel."""
    v = np.zeros((40, 40, 40), np.uint8)
    v[5:35, 10:22, 18:22] = 1                                   # half extents 15, 6, 2 voxels
    lam = np.sort(np.linalg.eigvalsh(np.cov(np.argwhere(v > 0).astype(float).T, bias=True)))[::-1]
    h = np.sqrt(3 * lam)
    assert np.allclose(h, np.sqrt(np.array([15, 6, 2]) ** 2 - 0.25), atol=1e-9)


def test_manifold_report_describes_the_clean_grid_even_when_noised(tmp_path, monkeypatch, block):
    """Salt speckle makes the stored grid many components; the record's manifold report (what --require-manifold and the
    stratum filters read) stays the clean body's, as in dataset_generator.py."""
    _stub_voxelize(monkeypatch, block)
    rec = _finish(tmp_path, "a", dict(config=voxel_noise.VoxelNoiseConfig(salt_prob=0.02), prob=1.0))
    assert rec["placement"]["noise"] is True and rec["placement"]["manifold"]["ok"]


# ------------------------------------------------------------------------------------------------ parity voxelizer (F7)

def test_parity_voxelizer_places_an_offcentre_asymmetric_box_exactly(tmp_path, monkeypatch):
    """The parity voxelizer - the DEFAULT (T-19): no env, no argument - places an asymmetric box, off-centre in the 40 mm
    grid, on exactly the voxels whose centres it contains, in the .bin order [z, -y, x] (checked through bin_to_world,
    the project's inverse)."""
    trimesh = pytest.importorskip("trimesh")
    from src.voxelization import voxelize_stl_to_bin
    from src.field_channels import bin_to_world
    lo, hi = np.array([-12.2, 3.1, -4.9]), np.array([4.1, 9.3, 15.2])          # mm, world
    box = trimesh.creation.box(extents=hi - lo)
    box.apply_translation((lo + hi) / 2)
    stl = tmp_path / "b.stl"
    box.export(stl)
    monkeypatch.delenv("SHAPE_RECO_VOXELIZER", raising=False)
    out, ran = tmp_path / "b.bin", {}
    assert voxelize_stl_to_bin(str(stl), str(out), 64, scale_mode="fixed", scale_mm=40.0, anchor="origin", report=ran)
    assert ran == {"voxelizer": "parity"}
    world = bin_to_world(np.fromfile(out, np.uint8).reshape(64, 64, 64) > 0)
    centres = -20.0 + (np.arange(64) + 0.5) * 40.0 / 64
    for axis in range(3):
        inside = np.nonzero((centres > lo[axis]) & (centres < hi[axis]))[0]
        got = np.nonzero(world.any(axis=tuple(a for a in range(3) if a != axis)))[0]
        assert got.min() == inside.min() and got.max() == inside.max(), axis
    assert world.sum() == np.prod([((centres > lo[a]) & (centres < hi[a])).sum() for a in range(3)])


def test_a_speck_is_recorded_as_degenerate_and_skipped_by_the_trainers(tmp_path, monkeypatch):
    """The canon (CLAUDE.md): a sample under 200 clean voxels is kept in the manifest as placement.degenerate and skipped
    by every trainer (subclass_schema.filter_records, train_axis, train_rotation_sym) - never silently dropped."""
    from training.manifest import load_manifest
    from training.subclass_schema import filter_records
    from training.train_rotation_sym import usable_boxes
    speck = np.zeros((16, 16, 16), np.uint8)
    speck[5:7, 5:7, 5:7] = 1
    monkeypatch.delenv("SHAPE_RECO_VOXELIZER", raising=False)
    _stub_voxelize(monkeypatch, speck)
    rec = _finish(tmp_path, "s", None)
    assert rec["placement"]["degenerate"] is True and rec["placement"]["voxelizer"] == "parity"
    recs = load_manifest(tmp_path, require_voxel=True)
    assert len(recs) == 1 and filter_records(recs)[0] == [] and usable_boxes(recs) == []


def test_finish_sample_stores_the_inside_bits_of_the_placed_mesh(tmp_path, monkeypatch):
    """A real off-centre box through the shared writer with NO voxelizer asked for: the .bin is the parity grid (the
    default, T-19), recorded as placement.voxelizer "parity", and <stem>.inside.npy (the standard base-point bit) is
    written, named in the record, and equals that .bin."""
    trimesh = pytest.importorskip("trimesh")
    from src import inside_bits
    monkeypatch.delenv("SHAPE_RECO_VOXELIZER", raising=False)
    box = trimesh.creation.box(extents=(9.0, 5.0, 3.0))
    box.apply_translation((6.0, -4.0, 2.5))
    (tmp_path / "box").mkdir()
    stl = tmp_path / "box" / "b.stl"
    box.export(stl)
    rec = sample_writer.finish_sample(stl, {"topology": "box"}, tmp_path, resolution=64, with_field=False)
    assert rec["placement"]["inside"] == "b.inside.npy" and rec["placement"]["voxelizer"] == "parity"
    bits = inside_bits.read_inside(tmp_path / rec["voxel_path"])
    grid = np.fromfile(tmp_path / rec["voxel_path"], np.uint8).reshape(64, 64, 64) > 0
    assert bits is not None and bits.sum() > 0 and np.array_equal(bits, grid)


def test_voxelizer_choice_default_explicit_and_legacy_modes(monkeypatch):
    """T-19: parity is the default on the in-scale grid; stltovoxel only when asked for (argument or env); the legacy
    grids (fit, uniform, fixed+center) are stltovoxel's - by default with a LegacyVoxelizerWarning, and an explicit
    parity request there is an error, never a silent fallback."""
    import warnings
    from src.voxelization import LegacyVoxelizerWarning, voxelizer_for
    monkeypatch.delenv("SHAPE_RECO_VOXELIZER", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                        # the in-scale grid: no warning at all
        assert voxelizer_for("fixed", "origin") == "parity"
        assert voxelizer_for("fixed", "origin", "stltovoxel") == "stltovoxel"
        assert voxelizer_for("fit", "center", "stltovoxel") == "stltovoxel"
    for mode, anchor in (("fit", "center"), ("uniform", "center"), ("fixed", "center")):
        with pytest.warns(LegacyVoxelizerWarning):
            assert voxelizer_for(mode, anchor) == "stltovoxel"
        with pytest.raises(ValueError, match="parity"):
            voxelizer_for(mode, anchor, "parity")
    monkeypatch.setenv("SHAPE_RECO_VOXELIZER", "stltovoxel")
    assert voxelizer_for("fixed", "origin") == "stltovoxel"
    assert voxelizer_for("fixed", "origin", "parity") == "parity"             # the argument wins over the env
    monkeypatch.setenv("SHAPE_RECO_VOXELIZER", "parity")
    with pytest.raises(ValueError, match="parity"):                           # the env is an explicit request too
        voxelizer_for("fit", "center")
    monkeypatch.setenv("SHAPE_RECO_VOXELIZER", "stl2vox")
    with pytest.raises(ValueError, match="voxelizer"):
        voxelizer_for("fixed", "origin")


def test_the_recorded_voxelizer_is_the_one_that_ran(tmp_path, monkeypatch):
    """placement.voxelizer comes from voxelize_stl_to_bin's report, not from the env: with env stltovoxel the record says
    stltovoxel (and the .bin is not the parity grid of the off-centre ball: 161 of ~1350 voxels differ), without env it
    says parity. (An axis-aligned box is no test: both voxelizers give it the same voxels.)"""
    if not (shutil.which("stltovoxel") or (Path.home() / ".local/bin/stltovoxel").is_file()):
        pytest.skip("stltovoxel not installed")
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.icosphere(subdivisions=3, radius=4.3)
    box.apply_translation((6.2, -4.1, 2.3))
    records = {}
    for env in ("stltovoxel", None):
        root = tmp_path / str(env)
        (root / "box").mkdir(parents=True)
        stl = root / "box" / "b.stl"
        box.export(stl)
        if env:
            monkeypatch.setenv("SHAPE_RECO_VOXELIZER", env)
        else:
            monkeypatch.delenv("SHAPE_RECO_VOXELIZER", raising=False)
        rec = sample_writer.finish_sample(stl, {"topology": "box"}, root, resolution=64, with_field=False)
        grid = np.fromfile(root / rec["voxel_path"], np.uint8).reshape(64, 64, 64) > 0
        records[env] = (rec["placement"]["voxelizer"], grid)
    assert records["stltovoxel"][0] == "stltovoxel" and records[None][0] == "parity"
    assert not np.array_equal(records["stltovoxel"][1], records[None][1])     # really two different voxelizers ran
