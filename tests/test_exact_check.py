"""The exact judge (reco_field/exact_check.py) and its use in the cascade loop (reco_engine/residual_loop.py): the OCCT
residual of a built model against its input, and the flag that says when the loop's own grid is too coarse to see the
thinnest wall."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.exact_check import exact_residual  # noqa: E402
from reco_field.export import build_in_freecad, find_freecadcmd  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402
from src.closest_point_field import SurfaceField  # noqa: E402

needs_freecad = pytest.mark.skipif(find_freecadcmd() is None, reason="freecadcmd not found")

T = np.array([40.0, -25.0, 12.0])                      # off-centre on every axis


def _bored_and_plain(tmp_path):
    """Input: a 12 x 8 x 4 box with a through bore of radius 1.5 (off-centre); model: the same box without the bore."""
    through = sp.Placed(sp.Cylinder(1.5, 10.0), translation=np.array([2.0, 1.0, -5.0]))
    bored = build_in_freecad(sp.Placed(sp.Box((6.0, 4.0, 2.0)) - through, translation=T), tmp_path / "in", "in")
    plain = build_in_freecad(sp.Placed(sp.Box((6.0, 4.0, 2.0)), translation=T), tmp_path / "model", "model")
    return bored, plain


@needs_freecad
def test_exact_residual_of_a_model_that_misses_nothing_but_invents_the_bore(tmp_path):
    bored, plain = _bored_and_plain(tmp_path)
    r = exact_residual(bored.step, plain.step)
    assert "error" not in r, r
    bore = plain.volume - bored.volume                       # 1.5^2 pi 4 = 28.27 mm^3
    assert bore == pytest.approx(np.pi * 1.5 ** 2 * 4.0, rel=1e-3)
    assert r["r_plus"] == pytest.approx(0.0, abs=1e-6)
    assert r["r_minus"] == pytest.approx(bore, rel=1e-6)
    assert r["r_minus_solids"] == 1
    assert r["iou"] == pytest.approx(bored.volume / (bored.volume + bore), rel=1e-9)


@needs_freecad
def test_exact_residual_swapped_roles_and_an_stl_input(tmp_path):
    """Input = the plain box, model = the bored one: the bore is now MISSED material (r_plus, one piece). The same
    judged against the input's STL (a facetted solid): equal up to the tessellation of the bore wall."""
    bored, plain = _bored_and_plain(tmp_path)
    r = exact_residual(plain.step, bored.step)
    bore = plain.volume - bored.volume
    assert r["r_plus"] == pytest.approx(bore, rel=1e-6) and r["r_plus_solids"] == 1 and r["r_minus"] < 1e-6
    r_stl = exact_residual(plain.stl, bored.step)
    assert "error" not in r_stl, r_stl
    assert r_stl["input_volume"] == pytest.approx(plain.volume, rel=1e-6)          # a box's facets are exact
    assert r_stl["r_plus"] == pytest.approx(bore, rel=0.02)


def test_exact_residual_reports_a_bad_input_instead_of_raising(tmp_path):
    r = exact_residual(tmp_path / "missing.step", tmp_path / "missing_too.step",
                       freecad=find_freecadcmd() or "/nonexistent/freecadcmd")
    assert "error" in r


def test_grid_blindness_is_pitch_over_the_thinnest_wall():
    """A 0.5 thick plate: a grid of pitch 2 is 4 x too coarse (blind), one of pitch 0.1 is not."""
    from reco_engine.residual_loop import _grid_blindness
    plate = trimesh.creation.box((10.0, 10.0, 0.5))
    plate.apply_translation(T)
    field = SurfaceField(plate.vertices, plate.faces)
    assert _grid_blindness(SimpleNamespace(pitch=2.0), field) == pytest.approx(4.0, rel=1e-3)
    assert _grid_blindness(SimpleNamespace(pitch=0.1), field) == pytest.approx(0.2, rel=1e-3)


def _input_step(mesh):
    from reco_engine import cascade
    from reco_engine import residual_loop as rl
    cfg = rl.ResidualConfig(resolution=32, max_resolution=32, max_ops=1, freecad=False, registry=cascade.DEFAULT_REGISTRY)
    return next(s for s in rl.residual_steps(SurfaceField(mesh.vertices, mesh.faces), cfg) if s.kind == "input")


def test_a_blind_grid_skips_the_type_stage_and_a_resolvable_one_does_not(monkeypatch):
    """The decision, with the blindness forced (the geometry side is test_grid_blindness_is_pitch_over_the_thinnest_wall):
    at pitch / tau > 1 the type stage does not answer and gives no hint; at or below 1 it runs as before."""
    from reco_engine import residual_loop as rl
    box = trimesh.creation.box((10.0, 8.0, 6.0))
    box.apply_translation(T)
    monkeypatch.setattr(rl, "_grid_blindness", lambda state, field: 4.0)
    s = _input_step(box)
    assert "type stage skipped: the grid is blind (pitch 4.0 x the thinnest wall)" in s.info["message"]
    assert s.info["type_stage"] is None and s.info["hint_subclass"] is None
    monkeypatch.setattr(rl, "_grid_blindness", lambda state, field: 0.5)
    s = _input_step(box)
    assert "type stage skipped" not in s.info["message"]


def test_sampled_iou_is_independent_of_occt_and_close_to_the_true_value():
    """Input: a 12 x 8 x 4 box (mesh); model: the same box shifted by 1 along x (CSG signed distance). True IoU =
    11 / 13 = 0.846; the grid sample (pitch = the 4 mm wall) is within a few percent."""
    from reco_field.exact_check import sampled_iou
    box = trimesh.creation.box((12.0, 8.0, 4.0))
    box.apply_translation(T)
    model = sp.Placed(sp.Box((6.0, 4.0, 2.0)), translation=T + np.array([1.0, 0.0, 0.0]))
    r = sampled_iou(SurfaceField(box.vertices, box.faces), model)
    assert r["iou"] == pytest.approx(11.0 / 13.0, abs=0.03), r


@needs_freecad
def test_model_genus_comes_from_the_brep(tmp_path):
    """The genus of a built model from its B-rep tessellation (no STL stitching involved): a box with two through bores
    is genus 2; with `booleans=False` only the model is read."""
    bores = [sp.Placed(sp.Cylinder(1.0, 10.0), translation=np.array([x, 0.0, -5.0])) for x in (-2.5, 2.5)]
    part = sp.Placed(sp.Box((6.0, 4.0, 2.0)) - bores[0] - bores[1], translation=T)
    out = build_in_freecad(part, tmp_path, "two_bores")
    r = exact_residual(out.step, out.step, booleans=False)
    assert r["model_genus"] == 2 and "r_plus" not in r
    r = exact_residual(out.step, out.step)
    assert r["model_genus"] == 2 and r["iou"] == pytest.approx(1.0, abs=1e-9)


def test_the_checks_prelude_takes_the_builders_functions_and_never_runs_it():
    """The builder ends with 'if GROUPS_JSON: main_groups() / else: main()'; the prelude of the check and document
    scripts must compile with whatever the builder ends with and must not call a main (2026-09-29: cut at the text
    'main()' it was a syntax error, every check failed and the loop took that as 'no check')."""
    import ast
    from reco_field.exact_check import _BUILDER_PRELUDE
    from reco_field.export import BUILDER
    filt = _BUILDER_PRELUDE[_BUILDER_PRELUDE.index("_tree.body = "):_BUILDER_PRELUDE.index("exec(")]
    ns = dict(ast=ast, _tree=ast.parse(Path(BUILDER).read_text()))
    exec(filt, ns)
    tree = ns["_tree"]
    compile(tree, "freecad_builder", "exec")
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"build", "boolean", "main"} <= names
    for n in tree.body:
        if not isinstance(n, ast.FunctionDef):
            assert not any(isinstance(c, ast.Call) and getattr(c.func, "id", "").startswith("main") for c in ast.walk(n))


@needs_freecad
def test_feature_checks_balance_for_a_tool_moved_by_a_rigid_motion(tmp_path):
    """The check's columns of a tool on its piece: the piece is the through bore the model misses (pi 1.5^2 4), the
    tool the same cylinder 2 longer on either side and turned by 1e-8 rad (a pattern transfer's rounding): common = the
    piece, left 0, beyond = the two ends, and common + beyond = the tool, common + left = the piece."""
    from reco_field.exact_check import feature_checks
    from reco_field.export import program_to_dict
    bored, plain = _bored_and_plain(tmp_path)
    r = exact_residual(bored.step, plain.step, pieces_dir=tmp_path / "pieces")
    piece = r["pieces"][0]
    a = 1e-8
    R = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]])
    tool = sp.Placed(sp.Placed(sp.Cylinder(1.5, 8.0), translation=np.array([2.0, 1.0, -4.0])), rotation=R, translation=T)
    c = feature_checks(bored.step, [dict(id=7, op="cut", piece_step=piece["step"], program=program_to_dict(tool))])[7]
    assert "error" not in c, c
    v = np.pi * 1.5 ** 2 * 4.0
    assert c["piece_volume"] == pytest.approx(v, rel=1e-6) and c["common_volume"] == pytest.approx(v, rel=1e-4)
    assert c["left_of_piece_volume"] == pytest.approx(0.0, abs=1e-4 * v)
    assert c["beyond_piece_volume"] == pytest.approx(v, rel=1e-4) and c["harm_volume"] == pytest.approx(0.0, abs=1e-4 * v)
    assert c["common_volume"] + c["beyond_piece_volume"] == pytest.approx(c["feature_volume"], rel=1e-4)
    # a tool inside its piece (the voxel fallback's boxes): nothing beyond it, nothing harmed, the rest is left
    inner = sp.Placed(sp.Placed(sp.Box((0.5, 0.5, 1.0)), translation=np.array([2.0, 1.0, 0.0])), translation=T)
    c = feature_checks(bored.step, [dict(id=8, op="cut", piece_step=piece["step"], program=program_to_dict(inner))])[8]
    assert "error" not in c, c
    assert c["common_volume"] == pytest.approx(2.0, rel=1e-6) and c["left_of_piece_volume"] == pytest.approx(v - 2.0, rel=1e-6)
    assert c["beyond_piece_volume"] == pytest.approx(0.0, abs=1e-9) and c["harm_volume"] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------------------------------------ the incremental difference

def _two_bores(tmp_path, radius=1.5):
    """Input: the 12 x 8 x 4 box with two through bores (off-centre); the rough model: the box; the tools: the bores'
    cylinders, longer than the box is thick (their ends in the air), the first of `radius`."""
    from reco_field.export import program_to_dict
    box = sp.Box((6.0, 4.0, 2.0))
    holes = [sp.Placed(sp.Cylinder(1.5, 4.0), translation=np.array([x, 1.0, -2.0])) for x in (-3.0, 2.0)]
    tools = [sp.Placed(sp.Cylinder(r, 10.0), translation=T + np.array([x, 1.0, -5.0]))
             for x, r in ((-3.0, radius), (2.0, 1.5))]
    inp = build_in_freecad(sp.Placed(box - holes[0] - holes[1], translation=T), tmp_path / "in", "in")
    rough = build_in_freecad(sp.Placed(box, translation=T), tmp_path / "rough", "rough")
    return inp, rough, tools, [dict(op="cut", program=program_to_dict(t)) for t in tools]


@needs_freecad
def test_incremental_residual_is_the_global_one_without_the_global_boolean(tmp_path):
    """The rough model's difference kept as a state, one bore cut: the difference of the new model from the state and
    the tool alone is what the boolean of input and model gives - one piece less, the same volumes, and its own state
    serves the next tool."""
    from reco_field.exact_check import incremental_residual
    inp, rough, tools, jobs = _two_bores(tmp_path)
    r0 = exact_residual(inp.step, rough.step, state_dir=tmp_path / "state_0")
    assert "error" not in r0 and r0["r_minus_solids"] == 2 and (tmp_path / "state_0" / "minus.brep").is_file()
    one = build_in_freecad(sp.Placed(sp.Box((6.0, 4.0, 2.0)), translation=T) - tools[0], tmp_path / "one", "one")
    r1 = incremental_residual(inp.step, one.step, tmp_path / "state_0", jobs[:1], state_dir=tmp_path / "state_1",
                              pieces_dir=tmp_path / "pieces_1")
    assert "error" not in r1, r1
    g1 = exact_residual(inp.step, one.step)
    bore = np.pi * 1.5 ** 2 * 4.0
    assert r1["method"] == "incremental" and g1["method"] == "general fuse"      # the global one: the fuse first
    assert r1["r_minus"] == pytest.approx(bore, rel=1e-3) and r1["r_minus"] == pytest.approx(g1["r_minus"], rel=1e-6)
    assert r1["r_plus"] == pytest.approx(0.0, abs=1e-6) and r1["iou"] == pytest.approx(g1["iou"], rel=1e-9)
    assert r1["r_minus_solids"] == 1 and abs(r1["budget_error"]) < 1e-6
    assert [p["sign"] for p in r1["pieces"]] == ["minus"]
    assert {f["origin"] for f in r1["pieces"][0]["faces"]} == {"input", "model"}      # the bore's wall, the box's faces
    both = build_in_freecad(sp.Placed(sp.Box((6.0, 4.0, 2.0)), translation=T) - tools[0] - tools[1], tmp_path / "both", "both")
    r2 = incremental_residual(inp.step, both.step, tmp_path / "state_1", jobs[1:])
    assert "error" not in r2, r2
    assert r2["r_minus"] == pytest.approx(0.0, abs=1e-6) and r2["iou"] == pytest.approx(1.0, abs=1e-9)
    same = incremental_residual(inp.step, one.step, tmp_path / "state_1", [], pieces_dir=tmp_path / "pieces_again")
    assert same["r_minus"] == pytest.approx(r1["r_minus"], rel=1e-12) and same["booleans"] == 0    # no tools: the
    assert [p["volume"] for p in same["pieces"]] == pytest.approx([r1["r_minus"]], rel=1e-9)       # state's pieces
    assert Path(same["pieces"][0]["step"]).is_file()


@needs_freecad
def test_incremental_residual_finds_what_a_cut_harms(tmp_path):
    """A bore cut with radius 1.7 where the input's is 1.5: the ring of input material it takes is missed material
    (r_plus), as in the global difference."""
    from reco_field.exact_check import incremental_residual
    inp, rough, tools, jobs = _two_bores(tmp_path, radius=1.7)
    exact_residual(inp.step, rough.step, state_dir=tmp_path / "state_0")
    one = build_in_freecad(sp.Placed(sp.Box((6.0, 4.0, 2.0)), translation=T) - tools[0], tmp_path / "one", "one")
    r = incremental_residual(inp.step, one.step, tmp_path / "state_0", jobs[:1])
    assert "error" not in r, r
    ring = np.pi * (1.7 ** 2 - 1.5 ** 2) * 4.0
    assert r["r_plus"] == pytest.approx(ring, rel=1e-3) and r["r_plus_solids"] == 1
    assert r["r_minus"] == pytest.approx(np.pi * 1.5 ** 2 * 4.0, rel=1e-3)
    g = exact_residual(inp.step, one.step)
    assert r["iou"] == pytest.approx(g["iou"], rel=1e-6)


@needs_freecad
def test_incremental_residual_refuses_a_model_its_tools_do_not_explain(tmp_path):
    """The model with both bores cut, but only one tool named: the volume budget fails, an error and no number."""
    from reco_field.exact_check import incremental_residual
    inp, rough, tools, jobs = _two_bores(tmp_path)
    exact_residual(inp.step, rough.step, state_dir=tmp_path / "state_0")
    r = incremental_residual(inp.step, inp.step, tmp_path / "state_0", jobs[:1])
    assert "error" in r and "do not explain" in r["error"] and "iou" not in r


# ------------------------------------------------------------------------------------------------ the general fuse

Q = np.array([[0.8138, -0.4698, 0.3420], [0.5438, 0.8232, -0.1632], [-0.2049, 0.3188, 0.9254]])   # a tilted frame
U, _, VT = np.linalg.svd(Q)
Q = U @ VT


def _pair(tmp_path, model_bore: float, tilt: bool = False):
    """Input: a 12 x 8 x 4 box with a through bore R 1.5; model: the same box with a bore of `model_bore` (0: none)."""
    R = Q if tilt else np.eye(3)
    box = sp.Box((6.0, 4.0, 2.0))

    def bore(r):
        return sp.Placed(sp.Cylinder(r, 10.0), translation=np.array([2.0, 1.0, -5.0]))
    inp = build_in_freecad(sp.Placed(box - bore(1.5), R, T), tmp_path / "in", "in")
    model = box if model_bore <= 0 else box - bore(model_bore)
    return inp, build_in_freecad(sp.Placed(model, R, T), tmp_path / "model", "model")


@needs_freecad
@pytest.mark.parametrize("tilt", [False, True])
def test_the_general_fuse_gives_both_differences(tmp_path, tilt):
    """A bore 0.1 larger in the model: the ring between the two bores is missing, nothing is invented; and the other
    way round. Against the closed formulas, in a world frame and tilted."""
    inp, model = _pair(tmp_path, 1.6, tilt)
    ring = np.pi * (1.6 ** 2 - 1.5 ** 2) * 4.0
    r = exact_residual(inp.step, model.step)
    assert "error" not in r, r
    assert r["method"] == "general fuse"
    assert r["r_plus"] == pytest.approx(ring, rel=1e-6) and r["r_minus"] == pytest.approx(0.0, abs=1e-6)
    assert abs(r["balance_error"]) <= 1e-5 * r["input_volume"]
    back = exact_residual(model.step, inp.step)
    assert back["r_minus"] == pytest.approx(ring, rel=1e-6) and back["r_plus"] == pytest.approx(0.0, abs=1e-6)
    assert back["iou"] == pytest.approx(r["iou"], rel=1e-9)


@needs_freecad
def test_the_general_fuse_and_the_cuts_agree_and_a_model_that_is_the_input_has_no_difference(tmp_path):
    inp, model = _pair(tmp_path, 0.0)
    fused = exact_residual(inp.step, model.step)
    cuts = exact_residual(inp.step, model.step, general_fuse=False)
    assert fused["method"] == "general fuse" and cuts["method"] == "occt"
    for key in ("r_plus", "r_minus", "iou"):
        assert fused[key] == pytest.approx(cuts[key], rel=1e-9, abs=1e-9)
    assert fused["r_minus"] == pytest.approx(np.pi * 1.5 ** 2 * 4.0, rel=1e-6)
    same = exact_residual(inp.step, inp.step)
    assert "error" not in same, same
    assert same["r_plus"] == pytest.approx(0.0, abs=1e-6) and same["r_minus"] == pytest.approx(0.0, abs=1e-6)
    assert same["iou"] == pytest.approx(1.0, abs=1e-9)


@needs_freecad
def test_the_general_fuse_writes_the_pieces(tmp_path):
    inp, model = _pair(tmp_path, 0.0, tilt=True)
    r = exact_residual(inp.step, model.step, pieces_dir=tmp_path / "pieces")
    assert r["method"] == "general fuse"
    assert [p["sign"] for p in r["pieces"]] == ["minus"]
    assert r["pieces"][0]["volume"] == pytest.approx(np.pi * 1.5 ** 2 * 4.0, rel=1e-6)
    kinds = sorted(f["type"] for f in r["pieces"][0]["faces"])
    assert kinds.count("cylinder") >= 1 and kinds.count("plane") == 2
    assert (tmp_path / "pieces" / "pieces.json").is_file()
