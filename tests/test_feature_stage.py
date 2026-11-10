"""The feature stage (reco_engine/feature_stage.py): the pieces of the exact difference between an input and a rough
model, and the feature guessed for each. The shapes are off-centre on every axis and the bore runs along y, so a
swapped axis or a sign error in a conversion shows."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine import feature_stage as fs  # noqa: E402
from reco_engine.result_to_program import rotation_matrix  # noqa: E402
from reco_field.exact_check import exact_residual  # noqa: E402
from reco_field.export import build_in_freecad, find_freecadcmd  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402

needs_freecad = pytest.mark.skipif(find_freecadcmd() is None, reason="freecadcmd not found")

T = np.array([40.0, -25.0, 12.0])                      # off-centre on every axis
HALF = (6.0, 4.0, 2.0)
ALONG_Y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])     # local z -> world y
BORE_R, BORE_AT = 0.8, np.array([-3.0, 0.0, 0.5])      # the bore's axis passes through this point (box frame)


def _box():
    return sp.Box(HALF)


def _bore(length=20.0):
    """A cylinder along y through BORE_AT, `length` long, centred on the box in y."""
    return sp.Placed(sp.Cylinder(BORE_R, length), rotation=ALONG_Y, translation=BORE_AT - np.array([0.0, length / 2, 0.0]))


def _build(tmp_path, name, solid):
    return build_in_freecad(sp.Placed(solid, translation=T), tmp_path / name, name)


def _pieces(tmp_path, inp, model):
    r = exact_residual(inp.step, model.step, pieces_dir=tmp_path / "pieces")
    assert "error" not in r, r
    return r


def _axis(primitive: dict) -> np.ndarray:
    from reco_engine.result_schema import Primitive
    return rotation_matrix(Primitive(kind=primitive["kind"], quaternion=tuple(primitive["quaternion"])))[:, 2]


@needs_freecad
def test_pieces_carry_their_exact_faces_and_where_each_face_comes_from(tmp_path):
    r = _pieces(tmp_path, _build(tmp_path, "in", _box() - _bore()), _build(tmp_path, "model", _box()))
    assert r["r_minus_solids"] == 1 and len(r["pieces"]) == 1
    p = r["pieces"][0]
    assert p["sign"] == "minus" and Path(p["step"]).is_file() and Path(p["stl"]).is_file()
    assert p["volume"] == pytest.approx(np.pi * BORE_R ** 2 * 2 * HALF[1], rel=1e-6)
    assert p["centre"] == pytest.approx(list(T + BORE_AT), abs=1e-6)
    wall = [f for f in p["faces"] if f["type"] == "cylinder"]
    caps = [f for f in p["faces"] if f["type"] == "plane"]
    assert len(wall) == 1 and len(caps) == 2
    assert wall[0]["origin"] == "input" and wall[0]["convex"] and wall[0]["radius"] == pytest.approx(BORE_R)
    assert abs(np.asarray(wall[0]["axis"]) @ [0.0, 1.0, 0.0]) == pytest.approx(1.0)
    assert {f["origin"] for f in caps} == {"model"}                    # the box's faces only close the piece
    assert json.loads((tmp_path / "pieces" / "pieces.json").read_text())["pieces"][0]["id"] == p["id"]


@needs_freecad
def test_without_pieces_dir_the_result_is_the_one_it_was(tmp_path):
    inp, model = _build(tmp_path, "in", _box() - _bore()), _build(tmp_path, "model", _box())
    plain = exact_residual(inp.step, model.step)
    with_pieces = exact_residual(inp.step, model.step, pieces_dir=tmp_path / "pieces")
    assert "pieces" not in plain and "left_out_volume" not in plain and "left_out_count" not in plain
    timing = ("method_seconds",)                     # wall time of the general fuse: differs run to run (server_b, 2026-10-01)
    drop = ("pieces", "left_out_volume", "left_out_count", "pieces_method") + timing
    assert {k: v for k, v in with_pieces.items() if k not in drop} == {k: v for k, v in plain.items() if k not in timing}


@needs_freecad
def test_a_through_bore_is_read_from_its_faces(tmp_path):
    r = _pieces(tmp_path, _build(tmp_path, "in", _box() - _bore()), _build(tmp_path, "model", _box()))
    f = fs.guess_feature(r["pieces"][0], sources=("step faces", "fit"))
    assert (f.op, f.kind, f.depth, f.source) == ("cut", "cylinder", "through", "step faces")
    assert f.score > 0.97
    assert f.primitive["dims_mm"]["radius"] == pytest.approx(BORE_R, rel=1e-9)
    assert f.primitive["dims_mm"]["half_length"] == pytest.approx(HALF[1], rel=1e-3)
    assert f.primitive["center_mm"] == pytest.approx(list(T + BORE_AT), abs=1e-3)
    assert abs(_axis(f.primitive) @ [0.0, 1.0, 0.0]) == pytest.approx(1.0)


@needs_freecad
def test_a_blind_bore_is_closed_by_a_face_of_the_input(tmp_path):
    """The bore enters from y = -4 and ends at y = +1 inside the box: its bottom is a surface of the input."""
    blind = sp.Placed(sp.Cylinder(BORE_R, 10.0), rotation=ALONG_Y, translation=BORE_AT + np.array([0.0, -9.0, 0.0]))
    r = _pieces(tmp_path, _build(tmp_path, "in", _box() - blind), _build(tmp_path, "model", _box()))
    f = fs.guess_feature(r["pieces"][0], sources=("step faces",))
    assert (f.op, f.kind, f.depth) == ("cut", "cylinder", "blind")
    assert f.primitive["dims_mm"]["half_length"] == pytest.approx(2.5, rel=1e-3)          # from y = -4 to y = +1
    assert f.primitive["center_mm"] == pytest.approx(list(T + BORE_AT + [0.0, -1.5, 0.0]), abs=1e-3)


@needs_freecad
def test_a_boss_is_an_add(tmp_path):
    """A cylinder standing on the box's top (z = +2), off-centre: missed material, one piece."""
    boss = sp.Placed(sp.Cylinder(1.2, 3.0), translation=np.array([2.5, -1.0, HALF[2]]))
    r = _pieces(tmp_path, _build(tmp_path, "in", sp.Union(_box(), boss)), _build(tmp_path, "model", _box()))
    assert [p["sign"] for p in r["pieces"]] == ["plus"]
    f = fs.guess_feature(r["pieces"][0], sources=("step faces", "fit"))
    assert (f.op, f.kind, f.depth) == ("add", "cylinder", None)
    assert f.score > 0.97
    assert f.primitive["dims_mm"]["radius"] == pytest.approx(1.2, rel=1e-9)
    assert f.primitive["center_mm"] == pytest.approx(list(T + [2.5, -1.0, HALF[2] + 1.5]), abs=1e-3)


@needs_freecad
def test_a_pocket_is_a_box_in_the_frame_of_its_planes(tmp_path):
    pocket = sp.Placed(sp.Box((1.5, 1.0, 0.75)), translation=np.array([3.0, -2.0, HALF[2] - 0.75]))
    r = _pieces(tmp_path, _build(tmp_path, "in", _box() - pocket), _build(tmp_path, "model", _box()))
    f = fs.guess_feature(r["pieces"][0], sources=("step faces",))
    assert (f.op, f.kind) == ("cut", "box") and f.score > 0.97
    assert sorted(f.primitive["dims_mm"].values()) == pytest.approx([0.75, 1.0, 1.5], rel=1e-6)
    assert f.primitive["center_mm"] == pytest.approx(list(T + [3.0, -2.0, HALF[2] - 0.75]), abs=1e-6)
    assert f.faces == {"plane": 5}                                     # five walls of the input, the sixth is open


@needs_freecad
def test_the_whole_stage_from_a_given_rough_model(tmp_path):
    inp, model = _build(tmp_path, "in", _box() - _bore()), _build(tmp_path, "model", _box())
    out = fs.feature_stage(inp.step, tmp_path / "out", rough_step=model.step, sources=("step faces", "fit"))
    s = out["summary"]
    assert s["pieces"] == 1 and s["cuts"] == 1 and s["adds"] == 0 and s["good_features"] == 1
    assert s["good_share"] == pytest.approx(1.0)
    assert s["rough_iou"] == pytest.approx(inp.volume / model.volume, rel=1e-9)
    assert json.loads((tmp_path / "out" / "features.json").read_text())["features"][0]["kind"] == "cylinder"
    assert "cylinder" in fs.format_features(out)
    doc = fs.write_fcstd(out, tmp_path / "out" / "features.FCStd")
    assert doc.is_file() and doc.stat().st_size > 1000


def test_an_stl_input_is_refused(tmp_path):
    with pytest.raises(ValueError, match="STEP"):
        fs.feature_stage(tmp_path / "part.stl", tmp_path / "out")


@needs_freecad
def test_the_exact_faces_win_when_the_piece_is_closed_by_a_curved_model_face(tmp_path):
    """A plate with a R 7 through-hole and a 12 x 8 pocket; the rough model is the plate under a curved top (a
    cylinder R 40 along x - the liner's pieces are closed by its curved bore). The exact STEP-faces primitives run
    to the piece's farthest point, into AIR above the curve: volume IoU punished that and a fit won (measured: the
    hole's fitted cone 0.957 over the exact cylinder's 0.952, the pocket's fitted box 0.791 over the exact 0.765).
    Scored as cuts (precision, recall) the exact faces win, with the STEP's own radius and pocket size."""
    from src.closest_point_field import load_field
    along_x = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    hole = sp.Placed(sp.Cylinder(7.0, 30.0), translation=np.array([-15.0, 5.0, -15.0]))
    pocket = sp.Placed(sp.Box((6.0, 4.0, 2.0)), translation=np.array([12.0, -8.0, 3.5]))
    inp = _build(tmp_path, "in", sp.Box((30.0, 20.0, 5.0)) - hole - pocket)
    top = sp.Placed(sp.Cylinder(40.0, 80.0), rotation=along_x, translation=np.array([-40.0, 0.0, 4.6 - 40.0]))
    model = _build(tmp_path, "model", sp.Intersection(sp.Box((30.0, 20.0, 5.0)), top))
    r = _pieces(tmp_path, inp, model)
    f = load_field(inp.step)
    inside = fs.input_inside(np.asarray(f.vertices, float), np.asarray(f.faces))
    cuts = sorted((p for p in r["pieces"] if p["sign"] == "minus"), key=lambda p: -p["volume"])
    assert len(cuts) == 2
    old = [fs.guess_candidates(p, sources=("step faces", "fit"), stop_at=None)["candidates"][0] for p in cuts]
    assert old[1].source == "fit"                                          # the defect, as it was (the pocket)
    new = [fs.guess_candidates(p, sources=("step faces", "fit"), stop_at=None, op="cut",
                               inside_input=inside)["candidates"][0] for p in cuts]
    assert [c.source for c in new] == ["step faces", "step faces"]
    assert new[0].primitive.kind == "cylinder" and new[0].primitive.dims_mm["radius"] == pytest.approx(7.0, rel=1e-9)
    assert new[0].precision > 0.99 and new[0].recall > 0.99
    assert new[1].primitive.kind == "box"
    assert sorted(new[1].primitive.dims_mm.values())[1:] == pytest.approx([4.0, 6.0], rel=1e-6)
    assert new[1].precision > 0.99 and new[1].recall > 0.99

# ------------------------------------------------------------------------------------------------ the difference grade

def _grade(tmp_path, inp, model, features=None, delta0=None):
    from reco_engine.difference_grade import grade_difference
    from src.closest_point_field import load_field
    r = _pieces(tmp_path, inp, model)
    return r, grade_difference(r, load_field(inp.stl), load_field(model.stl), features, delta0)


@needs_freecad
def test_the_grade_splits_the_defect_exactly_and_tells_a_skin_from_a_bore(tmp_path):
    """The input: a box with a bore. Guess A: the box without the bore (one compact piece of depth = the bore's
    radius). Guess B: the bored box, 0.02 too small on every side (a skin of depth 0.02 around it)."""
    from reco_engine.difference_grade import classify
    inp = _build(tmp_path, "in", _box() - _bore())
    r, g = _grade(tmp_path / "a", inp, _build(tmp_path, "a_model", _box()))
    assert [p.klass for p in g.pieces] == ["open"]                        # no feature guessed for it: open
    assert g.pieces[0].depth == pytest.approx(BORE_R, rel=0.02)
    assert g.e_skin + g.e_feature + g.e_open == pytest.approx(1.0 - g.iou, abs=1e-9)
    assert classify(g.pieces[0].depth, 1.0, g.delta0) == "feature"        # ... a feature once a primitive fits
    small = sp.Box(tuple(h - 0.02 for h in HALF)) - _bore()
    r, g = _grade(tmp_path / "b", inp, _build(tmp_path, "b_model", small), delta0=0.05)
    assert {p.klass for p in g.pieces} == {"skin"} and g.n_skin >= 1
    assert max(p.depth for p in g.pieces) == pytest.approx(0.02 * 3 ** 0.5, rel=0.05)    # the corner of the skin
    assert g.e_skin == pytest.approx(1.0 - g.iou, abs=1e-9) and g.reach == pytest.approx(1.0, abs=1e-9)
    assert g.effort == 1


@needs_freecad
def test_the_predicted_reach_is_what_applying_the_feature_measures(tmp_path):
    from reco_engine.difference_grade import grade_difference
    from src.closest_point_field import load_field
    inp, model = _build(tmp_path, "in", _box() - _bore()), _build(tmp_path, "model", _box())
    r = _pieces(tmp_path, inp, model)
    f = fs.guess_feature(r["pieces"][0], sources=("step faces",))
    g = grade_difference(r, load_field(inp.stl), load_field(model.stl), [dict(id=f.id, score=f.score, kind=f.kind,
                                                                               primitive=f.primitive)])
    assert (g.n_feature, g.effort) == (1, 1) and g.e_open == 0.0
    union, v = r["input_volume"] + r["r_minus"], r["pieces"][0]["volume"]
    assert g.reach == pytest.approx(r["input_volume"] / (union - f.score * v))     # a cut shrinks the union
    cut = sp.Placed(sp.Box(HALF), translation=T) - fs.primitive_to_sp(fs.Primitive(
        kind="cylinder", quaternion=tuple(f.primitive["quaternion"]), center_mm=tuple(f.primitive["center_mm"]),
        dims_mm=f.primitive["dims_mm"]))
    built = build_in_freecad(cut, tmp_path / "applied", "applied")
    measured = exact_residual(inp.step, built.step)
    assert "error" not in measured, measured
    assert measured["iou"] == pytest.approx(g.reach, abs=0.01)             # the score is sampled: 0.01


# ---------------------------------------------------------------------------------------------- the voxel fallback

def _turned(mesh, axis, deg, t):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.radians(deg) * np.asarray(axis, float) / np.linalg.norm(axis)).as_matrix()
    T[:3, 3] = t
    mesh.apply_transform(T)
    return mesh


def _tool_inside(tool, V, F, n=100000):
    """(share of the tool's sampled points inside the mesh, the tool's sampled volume)."""
    from reco_engine.result_to_program import primitive_to_sp
    sdf = primitive_to_sp(tool)
    lo, hi = (np.asarray(b, float) for b in sdf.bounds())
    pts = lo + np.random.default_rng(1).random((n, 3)) * (hi - lo)
    sel = pts[np.asarray(sdf.sdf(pts)) <= 0]
    return float(fs._inside_mesh(V, F, sel).mean()), float(np.prod(hi - lo)) * len(sel) / n


def test_the_voxel_tool_lies_inside_the_piece_and_says_what_it_covers():
    """A bore R 2.5 x 15 (the liner's small holes), turned and off-centre: boxes of cubes at tau / 3 or finer, all of
    them inside the piece (precision first), at most max_boxes; fewer boxes cover less."""
    import trimesh
    m = _turned(trimesh.creation.cylinder(radius=2.5, height=15.0, sections=64), [1, 2, 3], 35.0, [40.0, -25.0, 12.0])
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    axes = np.asarray(m.principal_inertia_vectors, float).T
    r = fs.voxel_tool(V, F, tau=5.0, frames=dict(world=np.eye(3), principal=axes), max_boxes=16)
    assert r["frame"] == "principal" and 1 < r["boxes"] <= 16 and not r["relaxed"]
    assert r["pitch"] <= r["licensed_pitch"] == pytest.approx(min(5.0 / 3.0, 15.0 / 32.0), rel=0.05)
    assert r["primitive"].kind == "composite" and r["primitive"].label.startswith("voxel fallback, ")
    assert len(fs.leaves(r["primitive"])) == r["boxes"]
    inside, volume = _tool_inside(r["primitive"], V, F)
    assert inside == 1.0 and volume == pytest.approx(r["covered_volume"], rel=0.02)
    assert 0.7 * m.volume < r["covered_volume"] <= r["voxel_volume"] < m.volume
    few = fs.voxel_tool(V, F, tau=5.0, frames=dict(principal=axes), max_boxes=2)
    assert few["boxes"] == 2 and few["covered_volume"] < r["covered_volume"]
    assert _tool_inside(few["primitive"], V, F)[0] == 1.0


def test_the_voxel_tool_relaxes_its_pitch_under_the_cell_cap_and_says_so():
    import trimesh
    m = _turned(trimesh.creation.box((30.0, 20.0, 1.0)), [0, 0, 1], 0.0, [5.0, 60.0, -7.0])
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    r = fs.voxel_tool(V, F, tau=1.0, max_cells=500)
    assert r is None or r["relaxed"]                                  # a pitch over the plate's thickness: no cube fits
    r = fs.voxel_tool(V, F, tau=1.0, max_cells=200000)
    assert not r["relaxed"] and r["boxes"] == 1 and r["primitive"].kind == "box"
    assert np.allclose(r["primitive"].center_mm, [5.0, 60.0, -7.0], atol=1e-6)
    assert r["primitive"].dims_mm["z_half"] == pytest.approx(0.5 * 6.0 / 6.04)      # six whole cells across the plate
    assert r["covered_volume"] == pytest.approx(600.0, rel=0.03) and r["covered_volume"] < 600.0
