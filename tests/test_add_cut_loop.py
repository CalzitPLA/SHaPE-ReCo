#!/usr/bin/env python3
"""reco_engine/residual_loop.py on analytic solids with a known program (meshed with reco_field.peel.mesh_from_sdf, no STL
files), with a fake FreeCAD build: the model's SDF and the input's SDF are compared with the right signs (a box against its
own box is no residual), the residual shrinks, adds come before cuts, a leftover no primitive explains goes to the voxel-cube
fallback, and the result survives the JSON round trip."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import sdf_primitives as sp  # noqa: E402
from src.closest_point_field import SurfaceField  # noqa: E402

from reco_engine import cascade, residual_loop as rl  # noqa: E402
from reco_engine.cascade import _Entry  # noqa: E402
from reco_engine.result_schema import Primitive, ReconstructionResult  # noqa: E402

pytest.importorskip("vtk")


def _field(prim) -> SurfaceField:
    from reco_field.peel import mesh_from_sdf
    v, f = mesh_from_sdf(prim, resolution=96)
    return SurfaceField(v, f)


class FakeFreeCAD:
    def __init__(self):
        self.models = []

    def __call__(self, model, out_dir, name, field):
        self.models.append(model)
        return dict(valid=True, n_solids=1, step=f"{name}.step", stl=f"{name}.stl", chamfer=0.01)


def _cfg(**kw):
    fc = FakeFreeCAD()
    base = dict(resolution=32, max_ops=6, builder=fc, registry=cascade.DEFAULT_REGISTRY)     # network-free, no checkpoints
    base.update(kw)
    return rl.ResidualConfig(**base), fc


def _run(prim, **kw):
    cfg, fc = _cfg(**kw)
    steps = list(rl.residual_steps(_field(prim), cfg))
    return steps, steps[-1].result, fc


def test_box_against_its_own_box_is_no_residual():
    """The sign conventions differ (field_channels / Query: positive inside; sdf_primitives: negative inside): the loop's
    model SDF must be negated before it is compared with the input channel."""
    field = _field(sp.Box([8.0, 6.0, 4.0]))
    cfg = rl.ResidualConfig(resolution=32)
    spec = rl._spec_for(field, cfg)
    state = rl._State(rl._channels(field, spec, False), spec)
    assert state.s_in[16, 16, 16] > 0                                    # the centre of the box is inside, positive
    state.s_model = state.prim_sdf(Primitive(kind="box", dims_mm=dict(x_half=8.0, y_half=6.0, z_half=4.0)))
    assert state.s_model[16, 16, 16] > 0
    assert state.size() == 0
    moved = state.prim_sdf(Primitive(kind="box", center_mm=(3.0, 0.0, 0.0), dims_mm=dict(x_half=8.0, y_half=6.0, z_half=4.0)))
    pos, neg = state.residual(moved)
    assert pos.sum() > 0 and neg.sum() > 0


def test_parity_sign_matches_the_solid():
    field = _field(sp.Box([8.0, 6.0, 4.0]) - sp.Placed(sp.Cylinder(3.0, 20.0), translation=np.array([0.0, 0.0, -10.0])))
    spec = rl._spec_for(field, rl.ResidualConfig(resolution=32))
    occ = rl.parity_occupancy(field, spec)
    c = spec.centres()
    truth = (np.abs(c) <= [8, 6, 4]).all(axis=-1) & (np.hypot(c[..., 0], c[..., 1]) > 3)
    assert (occ != truth).mean() < 0.01


def test_box_is_one_add():
    steps, res, fc = _run(sp.Placed(sp.Box([8.0, 6.0, 4.0]), translation=np.array([1.0, -2.0, 0.5])))
    assert [o.op for o in res.operations] == ["add"]
    p = res.operations[0].primitive
    assert p.kind == "box"
    assert sorted(p.dims_mm.values()) == pytest.approx([4, 6, 8], abs=0.15)
    assert np.allclose(p.center_mm, (1.0, -2.0, 0.5), atol=0.15)
    assert res.residual_voxels_pos + res.residual_voxels_neg == 0
    assert steps[0].kind == "input" and steps[-1].kind == "done" and any(s.kind == "grade" for s in steps)
    assert res.chamfer_mm == 0.01 and res.operations[-1].freecad_valid is True and len(fc.models) == 1


def test_box_minus_cylinder_hole_is_add_then_cut():
    solid = sp.Box([8.0, 8.0, 5.0]) - sp.Placed(sp.Cylinder(3.0, 30.0), translation=np.array([0.0, 0.0, -15.0]))
    steps, res, _ = _run(solid)
    ops = [o.op for o in res.operations]
    assert ops[0] == "add" and "cut" in ops
    assert ops == sorted(ops)                                            # every add before every cut ("add" < "cut")
    cut = next(o for o in res.operations if o.op == "cut")
    assert cut.primitive.kind == "cylinder" and cut.primitive.dims_mm["radius"] == pytest.approx(3.0, abs=0.25)
    sizes = [s.info["size"] for s in steps if s.kind == "residual"]
    assert sizes[-1] < sizes[0] and sizes[-1] <= 0.01 * steps[0].info["input_voxels"]
    # the cutter's caps were pushed out of the part: no coplanar faces for FreeCAD
    assert cut.primitive.dims_mm["half_length"] > 5.0


def test_cut_history_counts_every_cut_and_marks_area_cut_again():
    """The cut history: each accepted cut adds 1 wherever its cutter covers the grid, also where an earlier cut had already
    removed the material; an add leaves it alone. Two overlapping cutters: the overlap reads 2 and is reported as cut again."""
    field = _field(sp.Box([8.0, 8.0, 5.0]))
    spec = rl._spec_for(field, rl.ResidualConfig(resolution=32))
    state = rl._State(rl._channels(field, spec, False), spec)
    base = Primitive(kind="box", dims_mm=dict(x_half=8.0, y_half=8.0, z_half=5.0))
    s_base = state.prim_sdf(base)
    assert state.commit(state.apply(state.s_model, s_base, "add"), s_base, "add") == 0
    assert not state.cut_history.any()
    first = Primitive(kind="box", center_mm=(-2.0, 0.0, 0.0), dims_mm=dict(x_half=3.0, y_half=3.0, z_half=10.0))
    second = Primitive(kind="box", center_mm=(2.0, 0.0, 0.0), dims_mm=dict(x_half=3.0, y_half=3.0, z_half=10.0))
    s1, s2 = state.prim_sdf(first), state.prim_sdf(second)
    assert state.commit(state.apply(state.s_model, s1, "cut"), s1, "cut") == 0
    assert state.cut_history.max() == 1
    recut = state.commit(state.apply(state.s_model, s2, "cut"), s2, "cut")
    overlap = (s1 >= 0) & (s2 >= 0)
    assert recut == int(overlap.sum()) > 0
    assert (state.cut_history[overlap] == 2).all()
    assert (state.cut_history[(s1 >= 0) ^ (s2 >= 0)] == 1).all()
    assert (state.cut_history[~((s1 >= 0) | (s2 >= 0))] == 0).all()


def test_cut_history_reaches_the_steps_and_the_result():
    solid = sp.Box([8.0, 8.0, 5.0]) - sp.Placed(sp.Cylinder(3.0, 30.0), translation=np.array([0.0, 0.0, -15.0]))
    steps, res, _ = _run(solid)
    done = steps[-1].info["cut_history"]
    assert done.dtype == np.int16 and done.max() >= 1                     # the hole was cut
    accepted_cuts = [s for s in steps if s.kind == "accepted" and s.info["op"] == "cut"]
    assert accepted_cuts and all("cut_history" in s.info and "recut" in s.info for s in accepted_cuts)
    assert res.recut_voxels == int((done >= 2).sum())
    assert ReconstructionResult.from_dict(json.loads(res.to_json())).recut_voxels == res.recut_voxels


def test_box_plus_box_step_shrinks_the_residual_with_adds_first():
    solid = sp.Box([8.0, 8.0, 3.0]) | sp.Placed(sp.Box([4.0, 4.0, 2.5]), translation=np.array([0.0, 0.0, 5.0]))
    steps, res, _ = _run(solid)
    ops = [o.op for o in res.operations]
    assert ops == sorted(ops)
    sizes = [s.info["size"] for s in steps if s.kind == "residual"]
    final = res.residual_voxels_pos + res.residual_voxels_neg
    assert final < 0.2 * sizes[0]


def test_op_guard_env_switch(monkeypatch):
    """$SHAPE_RECO_OP_GUARD=1: later iterations pick add or cut by net gain; still a valid, shrinking program."""
    monkeypatch.setenv("SHAPE_RECO_OP_GUARD", "1")
    cfg, _ = _cfg()
    assert rl._op_guard(cfg) is True
    assert rl._op_guard(rl.ResidualConfig(op_guard=False)) is False
    solid = sp.Box([8.0, 8.0, 5.0]) - sp.Placed(sp.Cylinder(3.0, 30.0), translation=np.array([0.0, 0.0, -15.0]))
    steps = list(rl.residual_steps(_field(solid), cfg))
    res = steps[-1].result
    assert res.operations[0].op == "add" and res.residual_voxels_pos + res.residual_voxels_neg <= 0.01 * steps[0].info["input_voxels"]


def test_unexplainable_blob_goes_to_voxel_cubes():
    """A registry whose type stage only knows the torus: the box-shaped pocket is explained by nothing the cascade may
    propose, so it is left to voxel boxes (and void_fit is switched off). Since the largest filled boxes of a piece
    are a proposer of the loop itself (2026-09-26), the box arrives as an accepted cut with the source
    "voxel_fallback"; before, it came in a "fallback" step after the loop. Both are the same resolution."""
    reg = [_Entry("torus-only", "type", ("torus",), "prior", cost=0)] + [e for e in cascade.DEFAULT_REGISTRY if e.stage != "type"]
    solid = sp.Box([8.0, 8.0, 5.0]) - sp.Placed(sp.Box([3.0, 2.0, 2.0]), translation=np.array([0.0, 0.0, 4.0]))
    steps, res, _ = _run(solid, registry=reg, use_void_fit=False)
    kinds = [s.kind for s in steps]
    fb = [s.info["primitive"] for s in steps if s.kind in ("fallback", "accepted")
          and any(src.mode == "voxel_fallback" for src in s.info["primitive"].sources)]
    assert fb and all(p.kind == "box" for p in fb)
    first = next(i for i, s in enumerate(steps) if s.kind in ("fallback", "accepted") and s.info["primitive"] is fb[0])
    assert first > kinds.index("add")
    assert not [o for o in res.operations if o.primitive.kind == "torus"]          # the one kind it was offered
    assert res.operations[-1].op == "cut"
    final = res.residual_voxels_pos + res.residual_voxels_neg
    assert final < 0.3 * [s.info["size"] for s in steps if s.kind == "residual"][0]


def test_voxel_boxes_merge_a_blob_into_few_boxes():
    m = np.zeros((12, 12, 12), bool)
    m[1:7, 1:9, 2:5] = True
    m[6:10, 3:5, 2:5] = True                                             # an L
    boxes = rl.voxel_boxes(m, max_boxes=8)
    covered = np.zeros_like(m)
    for lo, hi in boxes:
        covered[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = True
    assert len(boxes) == 2 and (covered == m).all()


def test_result_json_round_trip():
    _, res, _ = _run(sp.Box([8.0, 8.0, 5.0]) - sp.Placed(sp.Cylinder(3.0, 30.0), translation=np.array([0.0, 0.0, -15.0])))
    back = ReconstructionResult.from_dict(json.loads(res.to_json()))
    assert back == res
    from reco_engine.result_to_program import program_dict
    assert program_dict(back)["type"] == "Subtraction"


# ------------------------------------------------------------------------------------------------ subclass routing (B3)

def _fake_type(answer):
    """A fake registry type entry (descriptor mode, answered by `models`) that returns `answer`: [(kind, p)] or
    ([(kind, p)], {"parent/sub": p})."""
    entry = _Entry("fake", "type", cascade.KINDS, "descriptor", "x.joblib", cost=0, metrics={"parent_acc": 0.9})
    return entry, {"fake": (lambda x: answer)}


def _steps(prim, answer=None, **kw):
    if answer is None:
        cfg, fc = _cfg(freecad=False, **kw)
    else:
        entry, models = _fake_type(answer)
        cfg, fc = _cfg(registry=[entry] + cascade.DEFAULT_REGISTRY, models=models, freecad=False, **kw)
    steps = list(rl.residual_steps(_field(prim), cfg))
    return steps, [s.kind for s in steps], steps[-1].result


def test_plan_after_a_hinted_miss_is_the_analytic_plan_minus_what_ran():
    cfg = rl.ResidualConfig()
    member = rl._subloop_plan(cfg, "box/ibeam")
    assert member == dict(members={"ibeam"}, hollow=False, forced=True)
    assert rl._plan_after_miss(member, cfg) == dict(members=None, hollow=True, forced=False)
    frame = rl._subloop_plan(cfg, "cylinder/frame")
    assert frame["members"] is None and rl._plan_after_miss(frame, cfg) == dict(members=False, hollow=True, forced=False)
    hollow = rl._subloop_plan(cfg, "sphere/hollow")
    assert hollow == dict(members=False, hollow=True, forced=True)
    assert rl._plan_after_miss(hollow, cfg) == dict(members=None, hollow=False, forced=False)
    for plain in ("box/full", "cylinder/sector", "torus/band"):          # a plain hint leaves the analytic gates on
        assert rl._subloop_plan(cfg, plain) == dict(members=None, hollow=True, forced=False)
    assert rl._subloop_plan(cfg, None) == dict(members=None, hollow=True, forced=False)


def test_type_stage_subclass_becomes_the_hint_and_the_analytic_detection_is_the_fallback():
    """A fake descriptor entry calls a plain box "box/ibeam": the member sub-loop runs first (forced), finds no I-beam,
    the analytic gates take over (one bar is one primitive: no member phase) and the plain first add follows. The input
    step records the type stage and the hint; the first add reuses the fused type stage."""
    steps, kinds, res = _steps(sp.Box([8.0, 6.0, 4.0]), ([("box", 0.9), ("sphere", 0.1)], {"box/ibeam": 0.8, "box/full": 0.2}))
    first = steps[0]
    assert first.kind == "input" and first.info["hint_subclass"] == "box/ibeam"
    ts = first.info["type_stage"]
    assert ts["parent"] == "box" and ts["subclass"] == "ibeam" and ts["subclass_source"] == "descriptor:fake" and ts["mode"] == "fused"
    assert [e["id"] for e in ts["entries"]] == ["fake", "type-analytic"] and ts["entries"][0]["weight"] == 0.9
    assert "type box/ibeam (fused" in first.info["message"]
    assert "hollow" not in kinds and "ibeam_box" not in kinds and "add" in kinds
    accepted = [o for o in res.operations if o.accepted]
    assert len(accepted) == 1 and accepted[0].op == "add" and accepted[0].primitive.kind == "box"
    src = accepted[0].primitive.sources[0]
    assert src.stage == "type" and src.mode == "fused" and src.detail["subclass"] == "ibeam"
    assert {e["id"] for e in src.detail["entries"]} == {"fake", "type-analytic"}


def test_hollow_hint_runs_the_hollow_sub_loop_and_a_plain_hint_leaves_the_analytic_gate_on():
    hollow_box = sp.Box([10.0, 10.0, 10.0]) - sp.Box([6.0, 6.0, 6.0])
    steps, kinds, res = _steps(hollow_box, ([("box", 0.9), ("cylinder", 0.1)], {"box/hollow": 0.9, "box/full": 0.1}))
    assert steps[0].info["hint_subclass"] == "box/hollow" and "hollow" in kinds
    ops = [(o.op, o.primitive.kind) for o in res.operations if o.accepted]
    assert ops[:2] == [("add", "box"), ("cut", "box")]
    # "box/full" (a wrong hint): it does not veto the analytic gate, which still finds the cavity (gate G3, 2026-09-24:
    # the veto collapsed the I-beam frame when the TypeNet called it "box/full")
    steps, kinds, res = _steps(hollow_box, ([("box", 0.9), ("cylinder", 0.1)], {"box/full": 0.9, "box/hollow": 0.1}))
    assert steps[0].info["hint_subclass"] == "box/full" and "hollow" in kinds
    # a member hint that misses on the hollow box: the analytic fallback still finds the cavity
    steps, kinds, res = _steps(hollow_box, ([("box", 0.9), ("cylinder", 0.1)], {"box/rhs": 0.9, "box/full": 0.1}))
    assert steps[0].info["hint_subclass"] == "box/rhs" and "hollow" in kinds


def test_no_subclass_from_the_type_stage_keeps_the_default_behaviour():
    steps, kinds, res = _steps(sp.Box([8.0, 6.0, 4.0]))                          # the network-free default registry
    ts = steps[0].info["type_stage"]
    assert steps[0].info["hint_subclass"] is None and ts["subclass"] is None and ts["parent"] == "box"
    assert ts["mode"] == "analytic:voxel_fit" and len(ts["entries"]) == 1
    accepted = [o for o in res.operations if o.accepted]
    assert len(accepted) == 1 and accepted[0].primitive.kind == "box" and accepted[0].primitive.sources[0].mode == "analytic:voxel_fit"
    # an explicit hint (argument or config) is not overridden by the type stage
    steps, kinds, res = _steps(sp.Box([8.0, 6.0, 4.0]), hint_subclass="box/hollow")
    assert steps[0].info["hint_subclass"] == "box/hollow" and steps[0].info["type_stage"] is None


def test_a_cavity_that_another_body_reaches_into_is_cut_first_and_the_body_added_back():
    """The gate's unions (2026-10-29): a hollow leaf's cavity that other bodies reach into was refused - the cut of its
    inner sphere also takes their material (63-84 % of what it removes is empty, the bar was 95 %) and the model filled the
    cavity. hollow_cut_first accepts it ("cut first") and the residual phase adds the taken material back; off, the cavity
    stays filled."""
    body = sp.Union(sp.Box([10.0, 10.0, 10.0]) - sp.Sphere(6.0),
                    sp.Placed(sp.Cylinder(3.0, 10.0), translation=np.array([0.0, 0.0, -10.0])))
    steps, kinds, res = _steps(body, max_ops=8, hollow_cut_first=True)        # off by default: measured worse on the gate
    cav = [s.info for s in steps if s.kind == "member_op" and "cavity" in s.info["message"]]
    assert cav and any(c["ok"] and c["reason"].startswith("cut first") for c in cav), [c["message"] for c in cav]
    ops = [(o.op, o.primitive.kind) for o in res.operations if o.accepted]
    assert ops[:2] == [("add", "box"), ("cut", "sphere")] and any(op == "add" for op, _ in ops[2:]), ops   # put back

    def iou(result):                                                    # volume IoU against the body, sampled
        from reco_engine.result_to_program import program_model
        P = np.random.default_rng(0).uniform(-10.5, 10.5, size=(200_000, 3))
        a, b = np.asarray(body.sdf(P)) <= 0, np.asarray(program_model(result).sdf(P)) <= 0
        return (a & b).sum() / (a | b).sum()
    on = iou(res)
    steps, kinds, res = _steps(body, max_ops=8, hollow_cut_first=False)
    assert any(s.kind == "rejected" and "no cutter passes" in s.info["message"] for s in steps)
    assert ("cut", "sphere") not in [(o.op, o.primitive.kind) for o in res.operations if o.accepted][:2]
    off = iou(res)
    assert on > 0.99 and on > off + 0.02, (on, off)                      # measured 0.9956 against 0.9620 (2026-10-29)
