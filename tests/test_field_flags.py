"""dataset_generator.py's --with-field / --segment-prob / --emit-negative-residual / --seed-offset, src/solid_builder.py
(voxel draws re-expressed in world mm), src/sample_writer.py (the shared writer), the vf_exact input kind and the
axis-loss fix of train_field_rotation. FreeCAD is STUBBED here (a blocky voxel mesh of the sdf program stands in for
Part's solid) - the real FreeCAD path is exercised by the smoke run, not by unit tests."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import field_channels as fc  # noqa: E402
from src import inscale, sample_writer, solid_builder  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402


def assert_field_matches_voxels(ch, spec, vox, voxelizer="stltovoxel"):
    """The verifier's criterion: stltovoxel's grid is up to half a voxel off the world grid (voxelization.py rounds its
    anchor offset), so field and .bin may differ only next to the surface. The parity grid (the default since T-19) is
    the field's own occupancy: equal voxel for voxel."""
    if voxelizer == "parity":
        assert np.array_equal(ch[0] > 0, vox)
        return
    off = (ch[0] > 0) != vox
    assert off.mean() < 0.05
    assert not off.any() or np.abs(ch[1][off]).max() <= 1.0 * spec.pitch


def stub_build_stl(program, out_stl, freecad=None, deflection=0.05, pitch=0.2371):
    """Test stand-in for FreeCAD: the program's occupancy on a fine grid, meshed as its outward blocky voxel surface."""
    lo, hi = program.bounds()
    lo, hi = np.maximum(lo, -25.0) - 2.137 * pitch, np.minimum(hi, 25.0) + 2 * pitch     # off the voxel grid planes
    n = int(np.ceil((hi - lo).max() / pitch))
    spec = fc.GridSpec(tuple(lo), tuple(lo + n * pitch), n)
    occ = program.sdf(spec.centres()) <= 0
    if not occ.any():
        raise RuntimeError("empty solid")
    field = fc.voxel_field(occ, spec)
    Path(out_stl).parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(field.vertices, field.faces, process=False).export(str(out_stl))
    return dict(volume=float(occ.sum() * pitch ** 3), valid=True, n_solids=1)


# ---------------------------------------------------------------------------------------------- solid_builder

def test_subclass_mapping():
    from src.dataset_generator import subclass_of
    assert subclass_of(None, {"hollow": {}}) == "hollow"
    assert subclass_of({"segment": {"u": [0, 1]}}, {}) == "sector"
    assert subclass_of({"segment": {"u": [0, 1], "v": [0, 1]}}, {}) == "sector"
    assert subclass_of({"segment": {"v": [0, 1]}}, {}) == "band"
    assert subclass_of({"segment": None}, {"carve": {}}) == "full"
    for t, subs in sample_writer.SUBCLASSES.items():
        assert "full" in subs and "frame" in subs


def _array_occupancy(prim, r=64, g=40.0):
    return (prim.sdf(sp.voxel_centres(r, g)) <= 0).astype(np.uint8)          # .bin order [z, -y, x]


def test_cut_halfspaces_reproduce_apply_cuts():
    import random
    base_prim = sp.Placed(sp.Box([9.0, 6.0, 5.0]), rotation=solid_builder.quat_to_matrix([0.2, 0.1, 0.3, 0.9]),
                          translation=np.array([1.0, -2.0, 0.5]))
    base = _array_occupancy(base_prim)
    rng = random.Random(3)
    cuts = [dict(normal=[0.0, 0.0, -1.0], keep=0.6), dict(normal=list(np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])), keep=0.7)]
    cut, applied = inscale.apply_cuts(base, cuts, min_thickness=0.0)
    assert len(applied) == 2
    prog = base_prim
    for h in solid_builder.cut_halfspaces(base, applied, 40.0):
        prog = sp.Intersection(prog, h)
    exact = _array_occupancy(prog)
    assert (exact != cut).mean() < 0.002                                     # only the odd crumb keep_main_piece dropped
    del rng


@pytest.mark.parametrize("cutter", ["sphere", "box"])
def test_carve_cutter_in_world_mm_matches_the_voxel_bite(cutter):
    import random
    base_prim = sp.Placed(sp.Sphere(12.0) if cutter == "box" else sp.Box([10.0, 7.0, 6.0]),
                          rotation=np.eye(3), translation=np.array([2.0, -1.0, 3.0]))
    base = _array_occupancy(base_prim)
    carved, rec = inscale.carve(base, random.Random(5), cutter)
    assert carved is not None
    bite = (base > 0) & (carved == 0)
    cut = _array_occupancy(solid_builder.carve_cutter(rec, 40.0, 64)) > 0
    assert np.array_equal(bite, (base > 0) & cut)


def test_labels_centre_and_canonical_rotation():
    R = solid_builder.quat_to_matrix([0.3, -0.2, 0.5, 0.78])
    placement = dict(scale_factor=0.8, center_mm=[3.0, -4.0, 1.0])
    bbox_center = np.array([1.0, 2.0, -0.5])
    cyl = solid_builder.local_primitive("cylinder", {"radius_ratio": 0.5, "half_length_ratio": 1.0}, 10.0)
    lab = solid_builder.labels(cyl, R, placement, bbox_center)
    prog = solid_builder.placed_program(cyl, R, placement, bbox_center)
    spec = fc.grid_spec(-20, 20, 96)
    occ = prog.sdf(spec.centres()) <= 0
    centroid = spec.centres()[occ].mean(axis=0)
    assert np.allclose(centroid, lab["center_mm"], atol=0.1)
    assert lab["dims_mm"] == pytest.approx(dict(radius=4.0, half_length=8.0))
    assert lab["rotation_canonical"]["group"] == "Dinf"
    ax = np.array(lab["rotation_canonical"]["axis"])
    assert abs(abs(ax @ R[:, 2]) - 1) < 1e-9 and ax[np.argmax(np.abs(ax))] > 0
    box = sp.Box([3.0, 5.0, 2.0])
    q1 = solid_builder.rotation_canonical(box, R)["quaternion"]
    Rflip = R @ np.diag([-1.0, -1.0, 1.0])                                   # the same box, a symmetric pose
    assert np.allclose(q1, solid_builder.rotation_canonical(box, Rflip)["quaternion"], atol=1e-9)
    cone = solid_builder.local_primitive("cone", {"radius1_ratio": 1.0, "radius2_ratio": 0.3, "height_ratio": 2.0}, 10.0)
    assert solid_builder.rotation_canonical(cone, R)["group"] == "Cinf"
    seg = solid_builder.local_primitive("torus", {"major_radius_ratio": 1.0, "minor_radius_ratio": 0.3}, 10.0,
                                        {"u": [0.5, 2.0]})
    assert solid_builder.rotation_canonical(seg, R)["u"] == pytest.approx([0.0, 1.5])


# ---------------------------------------------------------------------------------------------- sample_writer

def test_sample_writer_registers_a_manifest_record(tmp_path, monkeypatch):
    from training.manifest import load_manifest
    monkeypatch.delenv("SHAPE_RECO_VOXELIZER", raising=False)                  # the default: parity, no stltovoxel needed
    root = tmp_path / "shard0"
    stl = root / "box" / "part.stl"
    prim = sp.Box([6.0, 4.0, 3.0])
    R = solid_builder.quat_to_matrix([0.1, 0.2, 0.3, 0.9])
    stub_build_stl(sp.Placed(prim, rotation=R, translation=np.array([2.0, -3.0, 1.0])), stl)
    rec = sample_writer.finish_sample(stl, {"topology": "box", "subclass": "rhs", "freecad_volume_mm3": 576.0,
                                            "_shape": dict(prim=prim, R=R, t=[2.0, -3.0, 1.0])}, root, resolution=32,
                                      field_points=64)
    assert rec["stl_path"] == "box/part.stl" and rec["voxel_path"] == "box/part.bin"
    assert rec["center_mm"] == pytest.approx([2.0, -3.0, 1.0]) and rec["freecad_volume_mm3"] == 576.0
    assert "_shape" not in rec and rec["voxel_volume_mm3"] > 0
    records = load_manifest(root)
    assert len(records) == 1 and records[0].topology == "box" and records[0].placement["field"]["source"] == "mesh"
    ch, spec, data = fc.load_field_npz(stl.with_name("part.field.npz"))
    assert ch.shape == (5, 32, 32, 32) and data["points"].shape == (64, 6)
    vox = fc.bin_to_world(np.fromfile(root / rec["voxel_path"], np.uint8).reshape(32, 32, 32) > 0)
    assert rec["placement"]["voxelizer"] == "parity"
    assert_field_matches_voxels(ch, spec, vox, rec["placement"]["voxelizer"])
    with pytest.raises(ValueError):
        sample_writer.finish_sample(stl, {"topology": "box", "subclass": "sector"}, root, resolution=32)


def test_append_record_is_one_line_per_writer(tmp_path):
    from multiprocessing import Pool
    path = tmp_path / "m.jsonl"
    with Pool(4) as pool:
        pool.starmap(sample_writer.append_record, [(path, {"i": i, "pad": "x" * 5000}) for i in range(40)])
    lines = path.read_text().splitlines()
    assert sorted(json.loads(l)["i"] for l in lines) == list(range(40))


# ---------------------------------------------------------------------------------------------- the generator

def test_generator_flags_end_to_end_with_stub_freecad(tmp_path, monkeypatch):
    from src import dataset_generator as dg
    from training.manifest import load_manifest
    monkeypatch.setattr(solid_builder, "build_stl", stub_build_stl)
    monkeypatch.delenv("SHAPE_RECO_VOXELIZER", raising=False)                  # the default: parity, no stltovoxel needed
    out = tmp_path / "shard0"
    argv = ["dataset_generator", "--mode", "type", "--kinds", "box", "cylinder", "--count-per-type", "2",
            "--macro-dir", str(ROOT / "dataset_gen" / "FCMacros"), "--output-dir", str(out), "--freecad", sys.executable,
            "--voxelize", "--voxel-scale-mode", "fixed", "--voxel-resolution", "32", "--carve-prob", "1",
            "--segment-prob", "1", "--with-field", "--emit-negative-residual", "--field-points", "32",
            "--size-fraction", "0.5", "0.8", "--seed", "4", "--seed-offset", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    dg.main()
    recs = [json.loads(l) for f in out.rglob("dataset_metadata.jsonl") if f.parent != out for l in f.read_text().splitlines()]
    base = [r for r in recs if r.get("sample_kind") != "negres"]
    neg = [r for r in recs if r.get("sample_kind") == "negres"]
    assert len(base) == 4
    for r in base:
        assert r["subclass"] in sample_writer.SUBCLASSES[r["topology"]] and r["seed_offset"] == 2
        assert {"dims_mm", "center_mm", "rotation_canonical"} <= set(r)
        assert (out / r["stl_path"]).with_name(r["placement"]["field"]["path"]).is_file()
        if r["topology"] == "cylinder":
            assert r["subclass"] == "sector" and "u" in r["segment"]
        if "carve" in r["placement"]:
            assert r["placement"]["solid_revoxelized"]
    carved = [r for r in base if "carve" in r["placement"]]
    assert carved and len(neg) == len(carved)
    for r in neg:
        assert r["topology"] == "sphere" and r["subclass"] == "full"            # a box is carved by a sphere
        assert (out / r["stl_path"]).is_file() and (out / r["voxel_path"]).is_file()
    assert len(load_manifest(out)) == len(recs)
    # the field and the stored voxels describe the same (carved) solid
    for r in carved:
        ch, spec, _ = fc.load_field_npz((out / r["stl_path"]).with_name(r["placement"]["field"]["path"]))
        vox = fc.bin_to_world(np.fromfile(out / r["voxel_path"], np.uint8).reshape(32, 32, 32) > 0)
        assert r["placement"]["voxelizer"] == "parity"
        assert_field_matches_voxels(ch, spec, vox, r["placement"]["voxelizer"])


def test_old_behaviour_without_flags(monkeypatch):
    from src import dataset_generator as dg
    monkeypatch.setattr(sys, "argv", ["dataset_generator", "--mode", "type", "--macro-dir", ".", "--output-dir", "x"])
    args = dg.parse_args()
    assert not dg.uses_programs(args) and args.seed_offset == 0 and args.topologies == ["box", "cylinder", "sphere", "torus"]
    monkeypatch.setattr(sys, "argv", ["dataset_generator", "--mode", "type", "--macro-dir", ".", "--output-dir", "x",
                                      "--with-field"])
    with pytest.raises(SystemExit):                                          # needs the in-scale grid
        dg.parse_args()


# ---------------------------------------------------------------------------------------------- training side

def test_vf_exact_dataset_reads_the_field(tmp_path):
    import torch  # noqa: F401
    from training.field_input import CHANNELS, load_vf_exact
    from training.manifest import Record
    from training.voxel_dataset import VoxelDataset
    spec = fc.inscale_spec(40.0, 16)
    occ = sp.Box([8.0, 5.0, 3.0]).sdf(spec.centres() - np.array([3.0, -2.0, 1.0])) <= 0
    fc.world_to_bin(occ).astype(np.uint8).tofile(tmp_path / "s.bin")
    fc.save_field_npz(tmp_path / "s.field.npz", fc.voxel_channels(occ, spec), spec)
    rec = Record.__new__(Record)
    rec.voxel_path = tmp_path / "s.bin"
    tensor, _ = VoxelDataset([rec], 16, input_kind="vf_exact")._load(0)
    assert tuple(tensor.shape) == (CHANNELS["vf_exact"], 16, 16, 16)
    x = load_vf_exact(tmp_path / "s.bin")
    stored = np.fromfile(tmp_path / "s.bin", np.uint8).reshape(16, 16, 16)
    assert np.array_equal(x[0] > 0, stored > 0)                             # same array frame as the .bin
    assert np.array_equal(x[1] < 0, stored > 0)                             # fused sign: negative inside


def test_axis_loss_is_sign_free_for_cylinder_and_torus_only():
    import torch
    from training.train_field_rotation import axis_loss
    target = torch.nn.functional.normalize(torch.tensor([[0.3, -0.4, 0.8], [1.0, 0.2, 0.1]]), dim=1)
    for kind in ("cylinder", "torus"):
        assert axis_loss(-target, target, kind).item() == pytest.approx(0.0, abs=1e-6)
        assert axis_loss(target, target, kind).item() == pytest.approx(0.0, abs=1e-6)
    assert axis_loss(-target, target, "cone").item() == pytest.approx(2.0, abs=1e-6)
    assert axis_loss(target, target, "cone").item() == pytest.approx(0.0, abs=1e-6)
    ortho = torch.tensor([[0.0, 0.0, 1.0]])
    assert axis_loss(ortho, torch.tensor([[1.0, 0.0, 0.0]]), "cylinder").item() == pytest.approx(1.0)
