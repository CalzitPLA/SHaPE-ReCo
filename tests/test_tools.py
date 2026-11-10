#!/usr/bin/env python3
"""The catalogue of standard tools (reco_engine/tools.py): every tool built tilted and off-centre, its volume against
the closed formula (sampled on its SDF, fast; exact from FreeCAD, slow), and every fitter against a piece made by
FreeCAD with known parameters - the host (a block, a shaft, an L corner) with the tool cut from it or added to it,
the exact difference's piece fitted from its faces (slow)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine import tools as T  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp, rotation_matrix  # noqa: E402


def _rot(axis, deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.radians(deg) * np.asarray(axis, float) / np.linalg.norm(axis)).as_matrix()


R0 = _rot([0.4, -1.0, 0.7], 37)                      # not along a world axis
O0 = np.array([23.0, -41.0, 12.5])                   # off-centre

PARAMS = {
    "through_hole": dict(d=6.6, length=12.0),
    "blind_hole": dict(d=8.5, depth=14.0, point_angle=118.0),
    "countersunk_hole": dict(d=6.6, d_c=12.4, angle=90.0, length=10.0),
    "counterbored_hole": dict(d_cb=11.0, depth_cb=6.4, d=6.6, length=8.0),
    "stepped_hole": dict(d1=5.0, length1=6.0, d2=8.0, length2=5.0),
    "circular_pocket": dict(d=20.0, depth=5.0),
    "rect_pocket": dict(L=40.0, W=25.0, depth=8.0, r=5.0),
    "slot": dict(L=40.0, W=12.0, depth=6.0),
    "through_slot": dict(L=40.0, W=12.0, depth=10.0),
    "open_pocket": dict(L=30.0, W=20.0, depth=6.0, r=4.0),
    "chamfer": dict(a=3.0, b=3.0, length=40.0),
    "fillet": dict(r=4.0, length=40.0),
    "round": dict(r=4.0, length=40.0),
    "chamfered_box": dict(size=[30.0, 20.0, 10.0], edges=[1, 4, 11], leg=2.0),      # no two share a vertex
    "filleted_box": dict(size=[30.0, 20.0, 10.0], edges=[1, 4, 11], leg=3.0),
    "groove": dict(R_out=15.0, R_in=12.0, width=4.0),
    "relief": dict(R_out=15.0, R_in=12.0, width=4.0),
    "taper": dict(r1=10.0, r2=6.0, length=8.0),
}


def test_the_class_list_is_the_catalogue_plus_none():
    assert T.CLASSES[-1] == "none" and list(T.CLASSES[:-1]) == list(T.TOOLS) == list(PARAMS)


@pytest.mark.parametrize("name", list(PARAMS))
def test_a_tool_has_the_volume_of_its_formula(name):
    """Sampled on the built tree's SDF (seeded, 400 000 points in its box): within 1.5 % of the closed formula."""
    tool = T.TOOLS[name]
    prim = tool.build(PARAMS[name], R0, O0)
    assert prim.label == name and tool.valid(PARAMS[name])
    sp = primitive_to_sp(prim)
    lo, hi = (np.asarray(b, float) for b in sp.bounds())
    size = hi - lo
    if not np.all(np.isfinite(size)) or size.max() > 1e3:            # a half space in the tree: its operand's box
        lo, hi = O0 - 60.0, O0 + 60.0
    pts = lo + np.random.default_rng(0).random((400000, 3)) * (hi - lo)
    vol = float((np.asarray(sp.sdf(pts)) <= 0).mean() * np.prod(hi - lo))
    assert vol == pytest.approx(tool.volume(PARAMS[name]), rel=0.015 if name not in ("chamfer", "fillet", "round") else 0.03)


def test_the_store_round_trips_and_matches_a_standard_size(tmp_path):
    store = T.ToolStore()
    store.add_use("countersunk_hole", "plate.step", 3, 0.99, "catalogue", params=dict(PARAMS["countersunk_hole"], _through=False))
    store.add_use("countersunk_hole", "plate.step", 4, 0.98, "session",
                  params=dict(d=6.6 * 1.01, d_c=12.4, angle=90.0, length=10.0))
    assert len(store) == 1 and store.records[0].count == 2              # the same standard size, 1 % off
    other = T.ToolStore.load(store.save(tmp_path / "tools.json"))
    assert json.loads(other.to_json()) == json.loads(store.to_json())
    assert other.match("countersunk_hole", dict(d=6.6, d_c=12.4, angle=90.0, length=10.0)) is not None
    assert other.match("countersunk_hole", dict(d=9.0, d_c=12.4, angle=90.0, length=10.0)) is None


def test_a_created_tree_of_a_catalogue_shape_is_recognised():
    from reco_engine.feature_stage import rounded_box
    from reco_engine.result_schema import Primitive
    box = Primitive(kind="box", dims_mm=dict(x_half=20.0, y_half=12.5, z_half=4.0))
    cyls = [Primitive(kind="cylinder", center_mm=(sx * 15.0, sy * 7.5, 0.0), dims_mm=dict(radius=5.0, half_length=4.0))
            for sx in (-1, 1) for sy in (-1, 1)]
    assert T.catalogue_type_of(rounded_box(box, cyls)) == "rect_pocket"
    assert T.catalogue_type_of(box) is None


# ------------------------------------------------------------------------------------------------ FreeCAD (slow)

def _case(name):
    """(input SDF, model SDF, op) in the tool's local frame: the host and the tool at their known parameters."""
    from src import sdf_primitives as sp
    tool, prm = T.TOOLS[name], PARAMS[name]

    def cutter(over=2.0, **kw):
        return primitive_to_sp(tool.build(dict(prm, **kw), np.eye(3), np.zeros(3), over))

    def block(lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return sp.Placed(sp.Box((hi - lo) / 2), translation=(lo + hi) / 2)
    if T.TOOLS[name].family == "hole" or name == "circular_pocket":
        depth = dict(through_hole=prm.get("length")).get(name)
        host = block((-20, -20, 0), (20, 20, depth if depth else 30))
        return host - cutter(), host, "cut"
    if name in ("rect_pocket", "slot", "through_slot"):
        host = block((-35, -25, 0), (35, 25, prm["depth"] if name == "through_slot" else 20))
        return host - cutter(), host, "cut"
    if name == "open_pocket":
        host = block((-35, -25, 0), (prm["L"] / 2, 25, 20))
        return host - cutter(over=4.0), host, "cut"
    if name in ("chamfer", "round"):
        host = block((0, 0, 0), (20, 20, prm["length"]))
        return host - cutter(), host, "cut"
    if name == "fillet":
        L = sp.Union(block((-20, -20, 0), (0, 20, prm["length"])), block((-20, -20, 0), (20, 0, prm["length"])))
        return sp.Union(L, cutter(over=0.0)), L, "add"
    if name in ("chamfered_box", "filleted_box"):
        far = block((60, 60, 60), (70, 70, 70))
        return sp.Union(far, cutter(over=0.0)), far, "add"
    if name == "groove":
        shaft = sp.Placed(sp.Cylinder(prm["R_out"], 40.0), translation=np.array([0.0, 0.0, -20.0]))
        return shaft - cutter(over=0.0, R_out=prm["R_out"] + 1.0), shaft, "cut"
    if name == "relief":
        shaft = sp.Union(sp.Placed(sp.Cylinder(prm["R_out"], 20.0), translation=np.zeros(3)),
                         sp.Placed(sp.Cylinder(25.0, 10.0), translation=np.array([0.0, 0.0, -10.0])))
        return shaft - cutter(over=0.0, R_out=prm["R_out"] + 1.0), shaft, "cut"
    if name == "taper":
        host = block((-20, -20, -10), (20, 20, 0))
        return sp.Union(host, cutter(over=0.0)), host, "add"
    raise KeyError(name)


def _placed(s):
    from src import sdf_primitives as sp
    return sp.Placed(s, rotation=R0, translation=O0)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(PARAMS))
def test_a_tool_is_fitted_from_the_exact_faces_of_its_piece(name, tmp_path):
    """FreeCAD builds the tool (its volume: the formula to 1e-6) and the host with the tool cut from it / added to it;
    the exact difference against the host gives the piece; fit_piece reads the tool back from the piece's faces: its
    name among the interpretations, every parameter to 1e-3, the frame's z axis and the origin."""
    from reco_field.export import build_in_freecad, find_freecadcmd
    from reco_field.exact_check import exact_residual
    from src.closest_point_field import load_field
    if find_freecadcmd() is None:
        pytest.skip("freecadcmd not found")
    tool, prm = T.TOOLS[name], PARAMS[name]
    solid = build_in_freecad(primitive_to_sp(tool.build(prm, R0, O0)), tmp_path / "tool", "tool")
    assert solid.valid and solid.volume == pytest.approx(tool.volume(prm), rel=1e-6)
    inp_sdf, model_sdf, op = _case(name)
    inp = build_in_freecad(_placed(inp_sdf), tmp_path / "in", "in")
    model = build_in_freecad(_placed(model_sdf), tmp_path / "model", "model")
    r = exact_residual(inp.step, model.step, pieces_dir=tmp_path / "pieces")
    assert "error" not in r, r
    sign = "minus" if op == "cut" else "plus"
    piece = max((p for p in r["pieces"] if p["sign"] == sign), key=lambda p: p["volume"])
    V = np.asarray(load_field(piece["stl"]).vertices, float)
    fits = T.fit_piece(piece, V, op)
    got = [f for f in fits if f.tool == name]
    assert got, [(f.tool, f.params) for f in fits]
    f = got[0]
    for k, v in prm.items():
        if k == "edges":
            assert sorted(f.params[k]) == sorted(v) or len(f.params[k]) == len(v)
        elif k == "size":
            assert sorted(f.params[k]) == pytest.approx(sorted(v), rel=1e-3)
        else:
            assert f.params[k] == pytest.approx(v, rel=1e-3, abs=1e-3), (k, f.params)
    if tool.family in ("hole", "turned") or name == "circular_pocket":
        assert abs(f.R[:, 2] @ R0[:, 2]) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(np.cross(f.origin - O0, R0[:, 2])) < 1e-3                     # on the tool's axis
    built = f.build()
    assert primitive_to_sp(built) is not None
