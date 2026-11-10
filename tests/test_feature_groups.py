#!/usr/bin/env python3
"""reco_engine/feature_groups.py: the pieces of one exact difference grouped by tool type. A plate (tilted,
off-centre) with 3 countersunk holes of one size, 2 of another, 4 rounded pockets, 1 slot and 1 spherical blob, cut by
FreeCAD; the difference against the plain plate gives 11 pieces (one FreeCAD run for the module, ~10 s)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine import feature_groups as FG  # noqa: E402
from reco_engine import tools as T  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp  # noqa: E402

R0 = np.array([[0.96, -0.2, 0.2], [0.24, 0.95, -0.2], [-0.15, 0.24, 0.96]])
R0, _ = np.linalg.qr(R0)
R0 = R0 * np.sign(np.diag(R0))
O0 = np.array([17.0, -9.0, 4.0])
DOWN = np.diag([1.0, -1.0, -1.0])                  # a tool's +z into the plate from its top face (z = 20)


def _tool(name, xy, **prm):
    return primitive_to_sp(T.TOOLS[name].build(prm, DOWN, np.array([xy[0], xy[1], 20.0]), over=1.0))


@pytest.fixture(scope="module")
def pieces(tmp_path_factory):
    from reco_field.export import build_in_freecad, find_freecadcmd
    from reco_field.exact_check import exact_residual
    from src import sdf_primitives as sp
    if find_freecadcmd() is None:
        pytest.skip("freecadcmd not found")
    tmp = tmp_path_factory.mktemp("groups")
    plate = sp.Placed(sp.Box([100.0, 70.0, 10.0]), translation=np.array([0.0, 0.0, 10.0]))
    cuts = [_tool("countersunk_hole", (x, -55.0), d=6.6, d_c=12.4, angle=90.0, length=8.0) for x in (-70.0, -50.0, -30.0)]
    cuts += [_tool("countersunk_hole", (x, -55.0), d=9.0, d_c=18.0, angle=90.0, length=8.0) for x in (0.0, 30.0)]
    cuts += [_tool("rect_pocket", (x, y), L=24.0, W=14.0, depth=6.0, r=3.0) for x in (-60.0, -25.0) for y in (0.0, 30.0)]
    cuts += [_tool("slot", (40.0, 25.0), L=30.0, W=10.0, depth=5.0)]
    cuts += [sp.Placed(sp.Sphere(7.0), translation=np.array([70.0, -40.0, 20.0]))]
    part = plate
    for c in cuts:
        part = part - c
    inp = build_in_freecad(sp.Placed(part, rotation=R0, translation=O0), tmp / "in", "plate")
    model = build_in_freecad(sp.Placed(plate, rotation=R0, translation=O0), tmp / "model", "model")
    r = exact_residual(inp.step, model.step, pieces_dir=tmp / "pieces")
    assert "error" not in r, r
    return r["pieces"]


def _entries(pieces):
    from reco_engine.feature_stage import piece_descriptor
    from src.closest_point_field import load_field
    return [dict(id=i + 1, piece=p, V=np.asarray(load_field(p["stl"]).vertices, float), op="cut",
                 descriptor=piece_descriptor(p)) for i, p in enumerate(pieces)]


def test_the_pieces_fall_into_types_groups_and_families(pieces):
    out = FG.group_pieces(_entries(pieces))
    sizes = sorted((g["tool_type"], len(g["members"])) for g in out["groups"])
    assert sizes == [("countersunk_hole", 2), ("countersunk_hole", 3), ("rect_pocket", 4), ("slot", 1)], out["groups"]
    fam = [g["family"] for g in out["groups"] if g["tool_type"] == "countersunk_hole"]
    assert fam == ["countersunk_hole", "countersunk_hole"]              # two sizes, one family
    assert len(out["ungrouped"]) == 1                                   # the spherical blob
    blob = out["ungrouped"][0]
    assert out["per_piece"][blob].tool_type is None
    for g in out["groups"]:
        assert g["source"] == "analytic" and g["confidence"] == 1.0 and g["representative"] == g["members"][0]


class Hallucinating(FG.FeatureNet):
    """Names a geometrically impossible type for every piece with certainty."""

    def classify(self, tensor):
        return [("groove", 0.97), ("taper", 0.03)]


class Reorders(FG.FeatureNet):
    def classify(self, tensor):
        return [("rect_pocket", 0.9), ("slot", 0.1)]


def test_a_merely_admitted_type_stays_behind_a_read_one():
    fused, _ = FG.fuse([("countersunk_hole", 1.0), ("groove", 0.5)], [("groove", 0.99), ("countersunk_hole", 0.01)])
    assert [t for t, _ in fused] == ["countersunk_hole", "groove"]


def test_the_network_only_reranks_what_the_geometry_admits(pieces):
    entries = _entries(pieces)
    tensors = {e["id"]: np.zeros((6, 4, 4, 4), np.float32) for e in entries}
    out = FG.group_pieces(entries, net=Hallucinating(), tensors=tensors)
    assert all(g["tool_type"] != "groove" for g in out["groups"])     # excluded by the geometry: ignored
    assert sorted(g["tool_type"] for g in out["groups"]) == ["countersunk_hole", "countersunk_hole", "rect_pocket", "slot"]
    slot = next(e for e in entries if FG.group_pieces([e])["groups"] and FG.group_pieces([e])["groups"][0]["tool_type"] == "slot")
    ranked, _ = FG.classify_analytic(slot["piece"], slot["V"])
    assert [t for t, _ in ranked][:2] == ["slot", "rect_pocket"]
    fused, source = FG.fuse(ranked, Reorders().classify(None))
    assert source == "fused" and [t for t, _ in fused][:2] == ["rect_pocket", "slot"]


def test_the_stub_has_no_opinion_and_the_class_list_is_shared():
    assert FG.NoOpinion().classify(None) == [] and FG.FeatureNet.classes == T.CLASSES
    assert FG.fuse([("slot", 1.0)], []) == ([("slot", 1.0)], "analytic")
