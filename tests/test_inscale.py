"""
In-scale training samples (dataset_gen/src/inscale.py + voxelization anchor="origin"): a random size and
position in a partly populated grid, recorded per sample.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

SANDBOX = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX / "dataset_gen"))
from src import inscale  # noqa: E402
from src.voxelization import find_stltovoxel, voxelize_stl_to_bin  # noqa: E402
from src.freecad_path import find_freecadcmd  # noqa: E402

FREECAD = find_freecadcmd()               # $SHAPE_RECO_FREECADCMD, PATH, /usr/bin/freecadcmd (T-04)
needs_generator = pytest.mark.skipif(find_stltovoxel() is None or FREECAD is None,
                                     reason="needs stltovoxel and freecadcmd")

N, G = 64, 40.0


def test_sample_placement_stays_inside_the_grid_and_in_range():
    rng = random.Random(0)
    for _ in range(500):
        extents = [rng.uniform(1, 30) for _ in range(3)]
        p = inscale.sample_placement(rng, extents, G, 0.25, 0.9)
        assert 0.25 <= p["size_fraction"] <= 0.9
        assert max(p["extent_mm"]) == pytest.approx(p["size_fraction"] * G)                # largest extent = fraction of grid
        ratios = np.array(p["extent_mm"]) / np.array(extents)
        assert np.allclose(ratios, ratios[0])                                              # uniform scale: proportions kept
        for c, e in zip(p["center_mm"], p["extent_mm"]):
            assert abs(c) + e / 2 <= G / 2 + 1e-9                                          # the whole box fits


def test_sample_placement_is_reproducible_and_varies():
    a = [inscale.sample_placement(random.Random(7), [10, 20, 5], G, 0.25, 0.9) for _ in range(2)]
    assert a[0] == a[1]
    rng = random.Random(1)
    draws = [inscale.sample_placement(rng, [10, 10, 10], G, 0.25, 0.9)["center_mm"][0] for _ in range(50)]
    assert np.ptp(draws) > 10                                                              # positions really move


def test_centered_position_and_bad_arguments():
    p = inscale.sample_placement(random.Random(0), [1, 2, 3], G, 0.5, 0.5, position="centered")
    assert p["center_mm"] == [0.0, 0.0, 0.0] and p["size_fraction"] == 0.5
    for bad in ((0.0, 0.5), (0.6, 0.5), (0.2, 1.5)):
        with pytest.raises(ValueError):
            inscale.sample_placement(random.Random(0), [1, 1, 1], G, *bad)
    with pytest.raises(ValueError, match="position"):
        inscale.sample_placement(random.Random(0), [1, 1, 1], G, 0.2, 0.5, position="sideways")


def test_voxel_frame_record_uses_the_arrays_convention():
    p = inscale.sample_placement(random.Random(0), [10, 10, 10], G, 0.5, 0.5, position="centered")
    assert p["center_vox"] == [N / 2, N / 2, N / 2]
    p["center_mm"] = [4.0, 8.0, -2.0]                     # recompute by hand: [a0, a1, a2] = [z, -y, x] in voxels
    q = inscale.sample_placement(random.Random(0), [10, 10, 10], G, 0.5, 0.5)
    v = G / N
    assert q["center_vox"] == pytest.approx([N / 2 + q["center_mm"][2] / v, N / 2 - q["center_mm"][1] / v,
                                            N / 2 + q["center_mm"][0] / v])


@pytest.mark.parametrize("voxelizer", ["parity", "stltovoxel"])
@pytest.mark.parametrize("center_mm", [(0, 0, 0), (8, -6, 5), (-9, 7, -4), (12, 12, 12)])
def test_anchor_origin_keeps_the_absolute_position(tmp_path, center_mm, voxelizer):
    """A 6 mm cube at a known world position lands where the conversion says: x -> a2, y -> a1 REVERSED, z -> a0 - by
    the parity default and by the legacy stltovoxel."""
    if voxelizer == "stltovoxel" and find_stltovoxel() is None:
        pytest.skip("stltovoxel not installed")
    cube = trimesh.creation.box(extents=(6, 6, 6))
    cube.apply_translation(center_mm)
    stl, out = tmp_path / "c.stl", tmp_path / "c.bin"
    cube.export(stl)
    assert voxelize_stl_to_bin(str(stl), str(out), resolution=N, scale_mode="fixed", scale_mm=G, anchor="origin",
                               voxelizer=voxelizer)
    grid = np.fromfile(out, dtype=np.uint8).reshape(N, N, N).astype(bool)
    centroid = np.argwhere(grid).mean(0) + 0.5                                             # voxel-edge coordinates
    v = G / N
    x, y, z = center_mm
    expected = [N / 2 + z / v, N / 2 - y / v, N / 2 + x / v]                                # [a0, a1, a2]
    assert centroid == pytest.approx(expected, abs=1.0)
    assert grid.sum() == pytest.approx((6 / v) ** 3, rel=0.2)                              # size kept too


def test_anchor_needs_the_fixed_mode(tmp_path):
    stl = tmp_path / "c.stl"
    trimesh.creation.box(extents=(1, 1, 1)).export(stl)
    with pytest.raises(ValueError, match="anchor"):
        voxelize_stl_to_bin(str(stl), str(tmp_path / "o.bin"), resolution=N, scale_mode="uniform", anchor="origin")


@needs_generator
def test_generator_writes_partly_populated_grids_with_placement_metadata(tmp_path):
    out = tmp_path / "ds"
    cmd = [sys.executable, "-m", "src.dataset_generator", "--mode", "type", "--topologies", "sphere",
           "--count-per-type", "3", "--macro-dir", "./FCMacros", "--output-dir", str(out),
           "--freecad", FREECAD, "--voxelize", "--voxel-resolution", str(N),
           "--voxel-scale-mode", "fixed", "--grid-mm", str(G), "--size-fraction", "0.3", "0.6", "--seed", "5"]
    env = {k: v for k, v in os.environ.items() if k != "SHAPE_RECO_VOXELIZER"}         # the default voxelizer: parity
    subprocess.run(cmd, cwd=SANDBOX / "dataset_gen", check=True, capture_output=True, timeout=300, env=env)
    records = [json.loads(line) for line in (out / "sphere" / "dataset_metadata.jsonl").read_text().splitlines()]
    assert len(records) == 3
    fills = []
    for r in records:
        placement = r["placement"]
        assert 0.3 <= placement["size_fraction"] <= 0.6 and placement["grid_mm"] == G
        assert placement["voxelizer"] == "parity"
        grid = np.fromfile(out / r["voxel_path"], dtype=np.uint8).reshape(N, N, N).astype(bool)
        fills.append(grid.mean())
        # the sphere really sits where the record says: centroid of the occupied voxels vs center_vox
        assert np.argwhere(grid).mean(0) + 0.5 == pytest.approx(placement["center_vox"], abs=2.0)
    assert max(fills) < 0.2                                                                # far from filling the grid
    header = json.loads((out / "dataset_metadata.jsonl").read_text().splitlines()[0])
    assert header["voxel_scale_mode"] == "fixed" and header["size_fraction"] == [0.3, 0.6]
    assert header["voxelizer"] == "parity"


# ------------------------------------------------------------------------ cuts

def solid_cube(n=32, lo=6, hi=26):
    g = np.zeros((n,) * 3, dtype=np.uint8)
    g[lo:hi, lo:hi, lo:hi] = 1
    return g


def solid_sphere(n=48, r=18):
    idx = np.indices((n,) * 3) - n / 2 + 0.5
    return (np.sqrt((idx ** 2).sum(0)) <= r).astype(np.uint8)


def test_axis_aligned_cut_keeps_the_low_side_by_the_kept_fraction():
    cube = solid_cube()
    cut, applied = inscale.apply_cuts(cube, [dict(normal=[1.0, 0.0, 0.0], keep=0.5)])
    assert len(applied) == 1
    assert cut.sum() == pytest.approx(cube.sum() / 2, rel=0.06)
    assert cut[:16].sum() == cut.sum() and cut[16:].sum() == 0                     # the high-index half is gone
    flipped, _ = inscale.apply_cuts(cube, [dict(normal=[-1.0, 0.0, 0.0], keep=0.5)])
    assert flipped[16:].sum() == flipped.sum()                                       # the mirror normal keeps the other end


def test_a_hemisphere_has_half_the_volume_and_a_cap_a_matching_fraction():
    sphere = solid_sphere()
    half, _ = inscale.apply_cuts(sphere, [dict(normal=[0.0, 0.0, 1.0], keep=0.5)])
    assert half.sum() == pytest.approx(sphere.sum() / 2, rel=0.05)
    cap, _ = inscale.apply_cuts(sphere, [dict(normal=[0.0, 0.0, 1.0], keep=0.2)])
    h = 0.2 * 36                                                                     # cap height in voxels (diameter 36)
    assert cap.sum() == pytest.approx(np.pi * h * h * (3 * 18 - h) / 3, rel=0.08)   # closed-form spherical cap


def test_oblique_cuts_work_and_two_opposite_cuts_make_a_slab():
    cube = solid_cube()
    oblique, applied = inscale.apply_cuts(cube, [dict(normal=list(np.ones(3) / np.sqrt(3)), keep=0.5)])
    assert 0.3 * cube.sum() < oblique.sum() < 0.7 * cube.sum() and len(applied) == 1
    slab, applied = inscale.apply_cuts(cube, [dict(normal=[1.0, 0, 0], keep=0.7), dict(normal=[-1.0, 0, 0], keep=0.5)])
    assert len(applied) == 2 and slab.sum() < 0.4 * cube.sum()
    extent = np.argwhere(slab)[:, 0]
    assert extent.max() - extent.min() + 1 == pytest.approx(0.35 * 20, abs=2)         # 0.7 * 0.5 of the 20-voxel cube


def test_keep_one_removes_nothing_and_a_cut_that_would_empty_the_shape_is_skipped():
    cube = solid_cube()
    same, _ = inscale.apply_cuts(cube, [dict(normal=[0.0, 1.0, 0.0], keep=1.0)])
    assert np.array_equal(same, cube)
    tiny, applied = inscale.apply_cuts(cube, [dict(normal=[0.0, 1.0, 0.0], keep=0.02)], min_voxels=1000)   # one 400-voxel layer
    assert applied == [] and np.array_equal(tiny, cube)                                # left alone rather than reduced to a sliver
    assert inscale.apply_cuts(np.zeros((8, 8, 8), np.uint8), [dict(normal=[1.0, 0, 0], keep=0.5)])[1] == []


def test_sample_cuts_ranges_and_reproducibility():
    rng = random.Random(3)
    draws = [inscale.sample_cuts(rng, max_cuts=2, keep_min=0.35, keep_max=0.95) for _ in range(400)]
    assert all(1 <= len(d) <= 2 for d in draws)
    assert 0.2 < np.mean([len(d) == 2 for d in draws]) < 0.5                            # second cut ~35 %
    normals = [c["normal"] for d in draws for c in d]
    assert np.allclose([np.linalg.norm(n) for n in normals], 1.0)
    assert all(0.35 <= c["keep"] <= 0.95 for d in draws for c in d)
    aligned = np.mean([max(abs(x) for x in n) > 0.999 for n in normals])
    assert 0.4 < aligned < 0.65                                                          # axis-aligned about half the time
    assert inscale.sample_cuts(random.Random(9)) == inscale.sample_cuts(random.Random(9))
    assert all(len(inscale.sample_cuts(random.Random(i), max_cuts=1)) == 1 for i in range(20))


@needs_generator
def test_generator_cuts_then_adds_noise(tmp_path):
    def generate(out, extra):
        cmd = [sys.executable, "-m", "src.dataset_generator", "--mode", "type", "--topologies", "sphere",
               "--count-per-type", "4", "--macro-dir", "./FCMacros", "--output-dir", str(out),
               "--freecad", FREECAD, "--voxelize", "--voxel-resolution", str(N),
               "--voxel-scale-mode", "fixed", "--grid-mm", str(G), "--size-fraction", "0.5", "0.7", "--seed", "8"] + extra
        subprocess.run(cmd, cwd=SANDBOX / "dataset_gen", check=True, capture_output=True, timeout=300)
        return [json.loads(l) for l in (out / "sphere" / "dataset_metadata.jsonl").read_text().splitlines()]

    cut = generate(tmp_path / "cut", ["--cut-prob", "1.0", "--noise-salt-prob", "0.001", "--noise-blob-count", "2"])
    assert all("cuts" in r["placement"] and r["placement"]["voxels"] < r["placement"]["voxels_uncut"] for r in cut)
    for r in cut:
        grid = np.fromfile(tmp_path / "cut" / r["voxel_path"], dtype=np.uint8).reshape(N, N, N)
        assert grid.sum() > r["placement"]["voxels"]                                     # noise added voxels AFTER the cut
    header = json.loads((tmp_path / "cut" / "dataset_metadata.jsonl").read_text().splitlines()[0])
    assert header["cut_prob"] == 1.0 and header["noise"]["blob_count"] == 2

    plain = generate(tmp_path / "plain", [])                                             # cut-prob defaults to 0: no cuts
    assert all("cuts" not in r["placement"] for r in plain)


@needs_generator
def test_full_field_boxes_are_the_whole_grid_then_noised(tmp_path):
    def generate(out, extra):
        cmd = [sys.executable, "-m", "src.dataset_generator", "--mode", "type", "--topologies", "box", "sphere",
               "--count-per-type", "4", "--macro-dir", "./FCMacros", "--output-dir", str(out),
               "--freecad", FREECAD, "--voxelize", "--voxel-resolution", str(N),
               "--voxel-scale-mode", "fixed", "--grid-mm", str(G), "--size-fraction", "0.3", "0.5", "--seed", "8"] + extra
        subprocess.run(cmd, cwd=SANDBOX / "dataset_gen", check=True, capture_output=True, timeout=300)
        return {t: [json.loads(l) for l in (out / t / "dataset_metadata.jsonl").read_text().splitlines()[1:]]
                for t in ("box", "sphere")}

    rows = generate(tmp_path / "full", ["--full-field-prob", "1.0", "--noise-pepper-prob", "0.02", "--noise-salt-prob", "0.001"])
    for r in rows["box"]:
        assert r["placement"]["full_field"] and r["placement"]["voxels_uncut"] == N ** 3
        grid = np.fromfile(tmp_path / "full" / r["voxel_path"], dtype=np.uint8).reshape(N, N, N)
        assert 0.9 * N ** 3 < grid.sum() < N ** 3                                        # a full cube with pepper holes: noised
    assert all("full_field" not in r["placement"] and r["placement"]["voxels"] < 0.5 * N ** 3
               for r in rows["sphere"])                                                  # only boxes; spheres stay in scale

    plain = generate(tmp_path / "plain", [])                                             # default 0: no full-field samples
    assert all("full_field" not in r["placement"] for r in plain["box"])


def test_thickness_tells_plates_from_solids_at_any_tilt():
    n = 40
    idx = np.indices((n, n, n)).astype(float)
    cube = np.zeros((n, n, n), np.uint8); cube[10:30, 10:30, 10:30] = 1
    assert inscale.thickness(cube) == pytest.approx(20, abs=1)
    layer = np.zeros((n, n, n), np.uint8); layer[20, 5:35, 5:35] = 1
    assert inscale.thickness(layer) <= 1.5
    # a triangle in a tilted plane, one voxel thick: x + y + z = 60, corners cut by the triangle x, y, z >= 10
    triangle = ((np.abs(idx.sum(axis=0) - 60) < 0.9) & (idx.min(axis=0) >= 10)).astype(np.uint8)
    assert triangle.sum() > 100 and inscale.thickness(triangle) < 3
    tetra = ((idx.sum(axis=0) <= 60) & (idx.min(axis=0) >= 10)).astype(np.uint8)        # the solid corner
    assert inscale.thickness(tetra) > 8


def test_cuts_that_would_flatten_a_box_into_a_plate_are_skipped():
    cube = np.zeros((40, 40, 40), np.uint8); cube[8:32, 8:32, 8:32] = 1
    layer, applied = inscale.apply_cuts(cube, [dict(normal=[1.0, 0, 0], keep=0.04)])        # one 24x24 layer: > 300 voxels
    assert applied == [] and layer.sum() == cube.sum()
    layer, applied = inscale.apply_cuts(cube, [dict(normal=[1.0, 0, 0], keep=0.04)], min_thickness=0)
    assert len(applied) == 1 and layer.sum() == 24 * 24                                     # the guard is switchable
    # two cuts: the second would shave the wedge left by the first to a sliver along the cube's diagonal plane
    n3 = np.ones(3) / np.sqrt(3)
    wedge, applied = inscale.apply_cuts(cube, [dict(normal=list(n3), keep=0.6), dict(normal=list(-n3), keep=0.04)])
    assert len(applied) == 1 and inscale.thickness(wedge) >= 4
    already_thin = np.zeros((40, 40, 40), np.uint8); already_thin[8:32, 8:32, 18:20] = 1    # thickness 2 before the cut
    cut, applied = inscale.apply_cuts(already_thin, [dict(normal=[1.0, 0, 0], keep=0.5)])
    assert len(applied) == 1                                                                # not made worse: still allowed


# ------------------------------------------------------------------ one uniform manifold, box - sphere, sphere - box

def _solid(kind, n=48):
    idx = np.indices((n, n, n)).astype(float) - (n - 1) / 2
    if kind == "box":
        return (np.abs(idx) <= np.array([14.0, 10.0, 7.0]).reshape(3, 1, 1, 1)).all(axis=0).astype(np.uint8)
    return ((idx ** 2).sum(axis=0) <= 16.0 ** 2).astype(np.uint8)


def test_manifold_report_counts_islands_and_cavities():
    body = _solid("box")
    assert inscale.manifold_report(body) == {"components": 1, "cavities": 0, "ok": True}
    island = body.copy(); island[2, 2, 2] = 1
    assert inscale.manifold_report(island)["components"] == 2
    bubble = body.copy(); bubble[22:26, 22:26, 22:26] = 0                    # closed void well inside the box
    report = inscale.manifold_report(bubble)
    assert report["components"] == 1 and report["cavities"] == 1 and not report["ok"]
    assert not inscale.manifold_report(np.zeros((8, 8, 8), np.uint8))["components"]   # empty: zero pieces, not "ok"


@pytest.mark.parametrize("base,cutter", [("box", "sphere"), ("sphere", "box")])
def test_carve_gives_one_uniform_manifold_that_bites_into_the_surface(base, cutter):
    body = _solid(base)
    rng = random.Random(3)
    done = 0
    for _ in range(25):
        out, rec = inscale.carve(body, rng, cutter)
        if out is None:
            continue
        done += 1
        assert inscale.manifold_report(out)["ok"], rec                           # one piece, no bubble
        assert set(np.unique(out)) <= {0, 1} and (out <= body).all()          # only ever removes material
        assert 0.04 <= rec["removed_fraction"] <= 0.5 and rec["cutter"] == cutter
        assert abs((1 - out.sum() / body.sum()) - rec["removed_fraction"]) < 1e-9
        assert inscale.thickness(out) >= 4.0
    assert done >= 15                                                             # most draws are usable


def test_carve_is_reproducible_and_refuses_bad_input():
    body = _solid("box")
    a, ra = inscale.carve(body, random.Random(5), "sphere")
    b, rb = inscale.carve(body, random.Random(5), "sphere")
    assert np.array_equal(a, b) and ra == rb
    with pytest.raises(ValueError):
        inscale.carve(body, random.Random(1), "cone")
    assert inscale.carve(np.zeros_like(body), random.Random(1), "sphere") == (None, None)
    assert inscale.carve(body, random.Random(1), "sphere", removed=(0.9, 0.95)) == (None, None)   # impossible demand


# ------------------------------------------------------------------ hollow bodies: a closed cavity ON PURPOSE

@pytest.mark.parametrize("kind", ["box", "sphere"])
def test_hollow_leaves_one_piece_and_exactly_one_cavity(kind):
    body = _solid(kind)
    rng = random.Random(3)
    done = 0
    for _ in range(20):
        out, rec = inscale.hollow(body, rng)
        if out is None:
            continue
        done += 1
        report = inscale.manifold_report(out)
        assert report["components"] == 1 and report["cavities"] == 1 and not report["ok"]     # the opposite of carve()
        assert set(np.unique(out)) <= {0, 1} and (out <= body).all()                           # only ever removes material
        assert out.sum() < body.sum()
        assert abs(rec["kept_fraction"] - out.sum() / body.sum()) < 1e-9
        assert rec["wall_voxels"] >= 2 and rec["wall_fraction"] > 0    # rounded to whole voxels: not tight to the drawn fraction
    assert done >= 15


def test_hollow_is_reproducible_and_refuses_bad_input():
    body = _solid("box")
    a, ra = inscale.hollow(body, random.Random(5))
    b, rb = inscale.hollow(body, random.Random(5))
    assert np.array_equal(a, b) and ra == rb
    assert inscale.hollow(np.zeros_like(body), random.Random(1)) == (None, None)
    tiny = np.zeros((6, 6, 6), np.uint8); tiny[2:4, 2:4, 2:4] = 1
    assert inscale.hollow(tiny, random.Random(1)) == (None, None)                              # too small to leave a shell


def test_a_thicker_wall_keeps_more_material():
    body = _solid("box")
    thin, _ = inscale.hollow(body, random.Random(7), wall_fraction=(0.08, 0.08))
    thick, _ = inscale.hollow(body, random.Random(7), wall_fraction=(0.28, 0.28))
    assert thin is not None and thick is not None and thick.sum() > thin.sum()


@needs_generator
def test_generator_hollows_boxes_and_spheres_with_a_closed_cavity(tmp_path):
    out = tmp_path / "hollow"
    cmd = [sys.executable, "-m", "src.dataset_generator", "--mode", "type", "--topologies", "box", "sphere",
           "--count-per-type", "6", "--macro-dir", "./FCMacros", "--output-dir", str(out),
           "--freecad", FREECAD, "--voxelize", "--voxel-resolution", str(N),
           "--voxel-scale-mode", "fixed", "--grid-mm", str(G), "--size-fraction", "0.5", "0.7",
           "--hollow-prob", "1.0", "--require-manifold", "--seed", "4"]
    subprocess.run(cmd, cwd=SANDBOX / "dataset_gen", check=True, capture_output=True, timeout=300)
    hollowed = 0
    for topology in ("box", "sphere"):
        for line in (out / topology / "dataset_metadata.jsonl").read_text().splitlines()[1:]:
            rec = json.loads(line)
            pl = rec["placement"]
            if "hollow" in pl:
                hollowed += 1
                assert pl["manifold"]["components"] == 1 and pl["manifold"]["cavities"] == 1 and not pl["manifold"]["ok"]
                assert rec["topology"] == topology and "carve" not in pl and "cuts" not in pl
    assert hollowed >= 8                                                          # --require-manifold does not block hollow


@needs_generator
def test_generator_carves_box_minus_sphere_and_sphere_minus_box_and_records_manifolds(tmp_path):
    out = tmp_path / "carve"
    cmd = [sys.executable, "-m", "src.dataset_generator", "--mode", "type", "--topologies", "box", "sphere",
           "--count-per-type", "6", "--macro-dir", "./FCMacros", "--output-dir", str(out),
           "--freecad", FREECAD, "--voxelize", "--voxel-resolution", str(N),
           "--voxel-scale-mode", "fixed", "--grid-mm", str(G), "--size-fraction", "0.5", "0.7",
           "--carve-prob", "1.0", "--seed", "4"]
    subprocess.run(cmd, cwd=SANDBOX / "dataset_gen", check=True, capture_output=True, timeout=300)
    carved = 0
    for topology, cutter in (("box", "sphere"), ("sphere", "box")):
        for line in (out / topology / "dataset_metadata.jsonl").read_text().splitlines()[1:]:
            rec = json.loads(line)
            pl = rec["placement"]
            assert pl["manifold"]["ok"]                                                # clean part: one piece, no bubble
            if "carve" in pl:
                carved += 1
                assert pl["carve"]["cutter"] == cutter and "cuts" not in pl and rec["topology"] == topology
                assert pl["voxels"] < pl["voxels_uncut"]
    assert carved >= 8                                                                 # nearly every one of the 12 carved


def test_keep_main_piece_drops_crumbs_but_not_a_real_split():
    body = _solid("box")
    crumb = body.copy(); crumb[1, 1, 1] = 1; crumb[1, 1, 2] = 1              # two stray voxels
    cleaned, whole = inscale.keep_main_piece(crumb)
    assert whole and inscale.manifold_report(cleaned)["ok"] and cleaned.sum() == body.sum()
    halves = body.copy(); halves[:, :, 23:25] = 0                            # a real split into two big halves
    _, whole = inscale.keep_main_piece(halves)
    assert not whole
    assert inscale.keep_main_piece(body)[1]


@pytest.mark.parametrize("kind", ["box", "sphere"])
def test_random_planar_cuts_never_leave_more_than_one_piece(kind):
    body = _solid(kind)
    rng = random.Random(11)
    for _ in range(80):
        out, applied = inscale.apply_cuts(body, inscale.sample_cuts(rng))
        assert inscale.manifold_report(out)["ok"], applied


@needs_generator
def test_require_manifold_only_writes_uniform_manifolds(tmp_path):
    out = tmp_path / "man"
    cmd = [sys.executable, "-m", "src.dataset_generator", "--mode", "type", "--topologies", "box",
           "--count-per-type", "8", "--macro-dir", "./FCMacros", "--output-dir", str(out),
           "--freecad", FREECAD, "--voxelize", "--voxel-resolution", str(N),
           "--voxel-scale-mode", "fixed", "--grid-mm", str(G), "--size-fraction", "0.4", "0.8", "--cut-prob", "1.0",
           "--carve-prob", "0.5", "--require-manifold", "--seed", "21"]
    subprocess.run(cmd, cwd=SANDBOX / "dataset_gen", check=True, capture_output=True, timeout=300)
    rows = [json.loads(l) for l in (out / "box" / "dataset_metadata.jsonl").read_text().splitlines()[1:]]
    assert len(rows) >= 6 and all(r["placement"]["manifold"]["ok"] and r["placement"]["draws"] <= 6 for r in rows)


@needs_generator
def test_noise_prob_mixes_noisy_and_clean_samples_in_every_class_and_kind(tmp_path):
    out = tmp_path / "np"
    cmd = [sys.executable, "-m", "src.dataset_generator", "--mode", "type", "--topologies", "box", "sphere",
           "--count-per-type", "40", "--macro-dir", "./FCMacros", "--output-dir", str(out),
           "--freecad", FREECAD, "--voxelize", "--voxel-resolution", str(N),
           "--voxel-scale-mode", "fixed", "--grid-mm", str(G), "--size-fraction", "0.4", "0.8", "--cut-prob", "0.5",
           "--carve-prob", "0.5", "--require-manifold", "--noise-prob", "0.5", "--noise-pepper-prob", "0.02",
           "--noise-salt-prob", "0.002", "--seed", "31"]
    subprocess.run(cmd, cwd=SANDBOX / "dataset_gen", check=True, capture_output=True, timeout=600)
    seen = {}
    for topology in ("box", "sphere"):
        for line in (out / topology / "dataset_metadata.jsonl").read_text().splitlines()[1:]:
            rec = json.loads(line); pl = rec["placement"]
            grid = np.fromfile(out / rec["voxel_path"], dtype=np.uint8).reshape(N, N, N)
            kind = "carved" if "carve" in pl else "cut" if pl.get("cuts") else "plain"
            seen.setdefault((topology, kind), []).append(pl["noise"])
            if pl["noise"] is False:
                assert int(grid.sum()) == pl["voxels"]                                  # exactly the clean part
                assert inscale.manifold_report(grid)["ok"]
    for (topology, kind), flags in seen.items():                                        # both kinds of sample in every group
        if len(flags) >= 8:
            assert 0 < sum(flags) < len(flags), (topology, kind, flags)
    assert len(seen) >= 5


@needs_generator
def test_noise_prob_default_noises_everything(tmp_path):
    out = tmp_path / "all"
    cmd = [sys.executable, "-m", "src.dataset_generator", "--mode", "type", "--topologies", "sphere",
           "--count-per-type", "5", "--macro-dir", "./FCMacros", "--output-dir", str(out),
           "--freecad", FREECAD, "--voxelize", "--voxel-resolution", str(N),
           "--voxel-scale-mode", "fixed", "--grid-mm", str(G), "--size-fraction", "0.4", "0.8",
           "--noise-pepper-prob", "0.02", "--seed", "3"]
    subprocess.run(cmd, cwd=SANDBOX / "dataset_gen", check=True, capture_output=True, timeout=300)
    rows = [json.loads(l) for l in (out / "sphere" / "dataset_metadata.jsonl").read_text().splitlines()[1:]]
    assert all(r["placement"]["noise"] is True for r in rows)
