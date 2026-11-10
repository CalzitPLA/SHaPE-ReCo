#!/usr/bin/env python3
"""reco_engine/cascade.py on synthetic voxel targets, with a fake registry and fake type nets (no checkpoints, no mesh):
confidence gating short-circuits the costlier type modes, top-k keeps the branch the downstream fit explains best, and a
target no branch explains comes back as "unknown" with its branches as alternatives."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine import cascade  # noqa: E402
from reco_engine.cascade import _Entry, run_cascade  # noqa: E402
from reco_engine.result_schema import KINDS, Primitive, ReconstructionResult, Operation  # noqa: E402

POSE = [e for e in cascade.DEFAULT_REGISTRY if e.stage != "type"]


def _target(occ_fn, resolution=32, half=12.0):
    from src.field_channels import voxel_channels
    spec = cascade.grid_spec(-half, half, resolution)
    occ = occ_fn(spec.centres())
    return voxel_channels(occ, spec), spec


def _box(c):
    return (np.abs(c[..., 0]) <= 6) & (np.abs(c[..., 1]) <= 4) & (np.abs(c[..., 2]) <= 3)


class FakeNet:
    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def __call__(self, x):
        assert x.shape == (64, 64, 64) and set(np.unique(x)) <= {-1.0, 1.0}      # the in-scale {-1, +1} encoding
        self.calls += 1
        return self.answer


def test_gate_short_circuits_costlier_type_modes():
    ch, spec = _target(_box)
    cheap, costly = FakeNet([("box", 0.95), ("sphere", 0.05)]), FakeNet([("sphere", 1.0)])
    reg = [_Entry("cheap", "type", ("box", "sphere"), "cnn", cost=1, gate=0.9),
           _Entry("costly", "type", ("box", "sphere"), "cnn", cost=5, gate=0.9)] + POSE
    p = run_cascade(ch, spec, reg, models={"cheap": cheap, "costly": costly}, field_refine=False)
    assert costly.calls == 0 and cheap.calls == 1
    assert p.kind == "box" and p.sources[0].mode == "cnn:cheap" and p.sources[0].confidence == pytest.approx(0.95)
    assert [s.stage for s in p.sources] == ["type", "rotation", "dimensions", "position"]
    assert p.sources[1].mode == "analytic:frame"
    d = p.dims_mm
    assert sorted([d["x_half"], d["y_half"], d["z_half"]]) == pytest.approx([3, 4, 6], abs=0.5)
    # unsure -> the next mode is paid for
    cheap2, costly2 = FakeNet([("box", 0.6), ("sphere", 0.4)]), FakeNet([("box", 0.99)])
    run_cascade(ch, spec, reg, models={"cheap": cheap2, "costly": costly2}, field_refine=False)
    assert costly2.calls == 1


def test_top_k_keeps_the_branch_the_fit_explains():
    ch, spec = _target(_box)
    net = FakeNet([("sphere", 0.55), ("box", 0.45)])                      # the type net is wrong on top-1
    reg = [_Entry("net", "type", ("box", "sphere"), "cnn", cost=1, gate=0.99)] + POSE
    p = run_cascade(ch, spec, reg, models={"net": net}, top_k=2, field_refine=False)
    assert p.kind == "box"
    assert [a.kind for a in p.alternatives] == ["sphere"]
    assert p.residual < 0.1
    # top-1 only: the sphere branch alone does not explain the box -> unknown
    q = run_cascade(ch, spec, reg, models={"net": net}, top_k=1, field_refine=False, pass_iou=0.8)
    assert q.is_unknown and [a.kind for a in q.alternatives] == ["sphere"]


def test_unknown_when_no_branch_passes():
    def two_blobs(c):
        a = (np.linalg.norm(c - np.array([-6, 0, 0]), axis=-1) <= 3)
        b = (np.abs(c - np.array([6, 0, 0])) <= 2.5).all(axis=-1)
        return a | b
    ch, spec = _target(two_blobs)
    reg = [_Entry("prior", "type", ("box", "sphere", "cylinder"), "prior", cost=0)] + POSE
    p = run_cascade(ch, spec, reg, top_k=3, field_refine=False, pass_iou=0.8)
    assert p.is_unknown
    assert {a.kind for a in p.alternatives} <= {"box", "sphere", "cylinder"} and p.alternatives
    assert all(s.stage == "type" for s in p.sources)


def test_default_registry_sphere_and_cylinder_analytic():
    ch, spec = _target(lambda c: np.linalg.norm(c - np.array([1.0, -1.0, 0.5]), axis=-1) <= 5)
    p = run_cascade(ch, spec, cascade.DEFAULT_REGISTRY, kinds=("box", "sphere", "cylinder"), field_refine=False)
    assert p.kind == "sphere" and p.dims_mm["radius"] == pytest.approx(5, abs=0.5)
    assert np.allclose(p.center_mm, (1.0, -1.0, 0.5), atol=0.5)

    def cyl(c):                                                            # axis along y
        return (np.hypot(c[..., 0], c[..., 2]) <= 3) & (np.abs(c[..., 1]) <= 7)
    ch, spec = _target(cyl)
    p = run_cascade(ch, spec, cascade.DEFAULT_REGISTRY, kinds=("box", "sphere", "cylinder"), field_refine=False)
    assert p.kind == "cylinder"
    from reco_engine.result_to_program import rotation_matrix
    assert abs(rotation_matrix(p)[:, 2] @ np.array([0, 1, 0])) > 0.99    # local z = the axis
    assert p.dims_mm["radius"] == pytest.approx(3, abs=0.5) and p.dims_mm["half_length"] == pytest.approx(7, abs=0.6)


def test_schema_round_trip_of_a_cascade_result():
    ch, spec = _target(_box)
    reg = [_Entry("net", "type", ("box", "sphere"), "cnn", cost=1, gate=0.9)] + POSE
    p = run_cascade(ch, spec, reg, models={"net": FakeNet([("box", 0.9), ("sphere", 0.1)])}, field_refine=False)
    r = ReconstructionResult("x.stl", [Operation("add", p, component_voxels=10, net_gain=3.0)])
    import json
    assert ReconstructionResult.from_dict(json.loads(r.to_json())) == r
    assert p.kind in KINDS and len(p.quaternion) == 4


# ------------------------------------------------------------------------------------------------ scale-normalised input

def test_inscale_modes_scale_an_oversized_target():
    """A 300 mm box never fitted the 40 mm in-scale grid ("fixed": None); "fit" and "always" resample it so that its largest
    extent is INSCALE_FILL x 64 voxels, centred, and report the scale; a small target is unchanged in "fit" mode."""
    def big_box(c):
        return (np.abs(c[..., 0]) <= 150) & (np.abs(c[..., 1]) <= 100) & (np.abs(c[..., 2]) <= 75)
    ch, spec = _target(big_box, resolution=32, half=200.0)
    ctx = cascade._Context(ch, spec, None, None, None)
    assert cascade.inscale_occupancy(ctx, mode="fixed") is None
    t = cascade.inscale_target(ctx, mode="fit")
    assert t is not None and t.x.shape == (64, 64, 64) and t.x.dtype == np.float32 and set(np.unique(t.x)) == {-1.0, 1.0}
    assert t.inside.dtype == bool and np.array_equal(t.inside, t.x > 0)
    occ = t.x > 0
    extents = [np.ptp(np.argwhere(occ)[:, i]) + 1 for i in range(3)]
    assert max(extents) == pytest.approx(cascade.INSCALE_FILL * 64, abs=3)
    assert sorted(extents) == pytest.approx([22.4, 29.9, 44.8], abs=3)                 # 150 : 100 : 75 kept
    assert t.scale == pytest.approx(t.extent_mm / (cascade.INSCALE_FILL * 40.0)) and t.scale > 1
    centre = np.argwhere(occ).mean(axis=0)
    assert np.allclose(centre, 31.5, atol=1.0)
    assert cascade.inscale_occupancy(ctx, mode="fit") is not None and cascade.inscale_target(ctx, mode="always").scale == t.scale
    # a target that fits: "fit" keeps its own scale (the old grid), "always" normalises it too
    ch, spec = _target(_box)
    ctx = cascade._Context(ch, spec, None, None, None)
    fixed, fit, always = (cascade.inscale_target(ctx, mode=m) for m in ("fixed", "fit", "always"))
    assert np.array_equal(fixed.x, fit.x) and fit.scale == 1.0
    assert np.ptp(np.argwhere(always.x > 0)[:, 2]) + 1 == pytest.approx(cascade.INSCALE_FILL * 64, abs=3) and always.scale < 1
    with pytest.raises(ValueError):
        cascade.inscale_target(ctx, mode="bogus")


def test_type_entries_see_the_target_scale_normalised():
    """The type stage runs its sources on the normalised grid whatever the target's size (the frame, real parts)."""
    def big_box(c):
        return (np.abs(c[..., 0]) <= 150) & (np.abs(c[..., 1]) <= 100) & (np.abs(c[..., 2]) <= 75)
    ch, spec = _target(big_box, resolution=32, half=200.0)
    seen = {}

    def net(x):
        seen["extent"] = max(np.ptp(np.argwhere(x > 0)[:, i]) + 1 for i in range(3))
        return [("box", 0.9), ("sphere", 0.1)]
    reg = [_Entry("net", "type", ("box", "sphere"), "cnn", cost=1)] + POSE
    p = run_cascade(ch, spec, reg, models={"net": net}, field_refine=False)
    assert seen["extent"] == pytest.approx(cascade.INSCALE_FILL * 64, abs=3) and p.kind == "box"
    assert p.dims_mm["x_half"] == pytest.approx(150, abs=8)                              # pose and dims still in mm
    fixed = [_Entry("net", "type", ("box", "sphere"), "cnn", cost=1, metrics={"inscale_mode": "fixed"})] + POSE
    seen.clear()
    run_cascade(ch, spec, fixed, models={"net": net}, field_refine=False)
    assert not seen                                                                       # the old behaviour on request


# ------------------------------------------------------------------------------------------------ fusion, subclass

def _votes(p):
    return {e["id"]: e for e in p.sources[0].detail["entries"]}


def test_type_stage_fuses_entries_by_parent_acc_and_routes_the_subclass():
    ch, spec = _target(_box)
    desc = FakeNet(([("box", 0.4), ("sphere", 0.6)], {"box/ibeam": 0.7, "box/full": 0.3, "sphere/full": 1.0}))
    net = FakeNet([("box", 0.9), ("sphere", 0.1)])
    reg = [_Entry("desc", "type", ("box", "sphere"), "descriptor", "x.joblib", cost=1, metrics={"parent_acc": 0.75}),
           _Entry("net", "type", ("box", "sphere"), "cnn", cost=2)] + POSE          # no parent_acc: weight 0.5
    ts = cascade.type_stage(ch, spec, reg, models={"desc": desc, "net": net})
    w_box = (0.75 * 0.4 + 0.5 * 0.9) / 1.25
    w_sph = (0.75 * 0.6 + 0.5 * 0.1) / 1.25
    assert ts.probs["box"] == pytest.approx(w_box / (w_box + w_sph)) and ts.mode == "fused" and ts.parent == "box"
    assert ts.subclass == "ibeam" and ts.subclass_source == "descriptor:desc" and ts.hint == "box/ibeam"
    assert [v.id for v in ts.votes] == ["desc", "net"] and [v.weight for v in ts.votes] == [0.75, 0.5]
    d = ts.as_dict()
    assert d["parent"] == "box" and d["entries"][0]["subclass"] == "ibeam" and d["entries"][1]["subclass"] is None
    # the same through run_cascade: the type source is "fused" and records every vote
    p = run_cascade(ch, spec, reg, models={"desc": desc, "net": net}, field_refine=False, type_result=ts)
    assert desc.calls == 1 and net.calls == 1                                             # the result was reused, not recomputed
    assert p.kind == "box" and p.sources[0].mode == "fused" and p.sources[0].confidence == pytest.approx(ts.probs["box"])
    votes = _votes(p)
    assert votes["desc"]["mode"] == "descriptor:desc" and votes["desc"]["weight"] == 0.75 and votes["desc"]["subclass"] == "ibeam"
    assert votes["net"]["mode"] == "cnn:net" and votes["net"]["probs"]["box"] == 0.9
    assert p.sources[0].detail["subclass"] == "ibeam"
    # a single answering entry keeps its own numbers and label; a subclass from a hierarchical CNN's TypePrediction shape
    from reco_engine.predict import TypePrediction
    hier = FakeNet(TypePrediction("box", 0.8, [("box", 0.8), ("sphere", 0.2)], subclass="hollow",
                                  subclasses={"box": [("hollow", 0.6), ("full", 0.4)], "sphere": [("full", 1.0)]}))
    ts = cascade.type_stage(ch, spec, [_Entry("hier", "type", ("box", "sphere"), "cnn", cost=1)] + POSE, models={"hier": hier})
    assert ts.mode == "cnn:hier" and ts.probs == {"box": 0.8, "sphere": 0.2} and ts.hint == "box/hollow"
    # no entry knows subclasses: no hint, the sources stay as before
    ts = cascade.type_stage(ch, spec, [_Entry("net", "type", ("box", "sphere"), "cnn", cost=1)] + POSE, models={"net": net})
    assert ts.subclass is None and ts.hint is None and ts.origin["box"].detail["entries"][0]["subclass"] is None
    # an entry that cannot say a kind casts no vote against it (weights renormalised per kind)
    part = FakeNet([("box", 0.99)])
    full = FakeNet([("box", 0.1), ("sphere", 0.05), ("cylinder", 0.85)])
    reg = [_Entry("part", "type", ("box", "sphere"), "cnn", cost=1), _Entry("full", "type", KINDS[:3], "cnn", cost=2)]
    fused = cascade._fuse(cascade.type_stage(ch, spec, reg + POSE, models={"part": part, "full": full}).votes, KINDS[:3])
    assert fused["cylinder"] > fused["box"] and sum(fused.values()) == pytest.approx(1.0)


def test_descriptor_entry_uses_a_descriptor_model_and_skips_a_missing_file(capsys):
    class Model:                                                        # the reco_engine.type_descriptors.DescriptorModel shape
        def predict_proba(self, occ):
            assert occ.dtype == bool and occ.shape == (64, 64, 64)
            return {"box": 0.7, "sphere": 0.2, "cylinder": 0.1, "cone": 0.0, "torus": 0.0}, {"box/rhs": 0.55, "box/full": 0.45}
    ch, spec = _target(_box)
    reg = [_Entry("d", "type", KINDS, "descriptor", "x.joblib", cost=1)] + POSE
    ts = cascade.type_stage(ch, spec, reg, models={"d": Model()})
    assert ts.parent == "box" and ts.subclass == "rhs" and ts.mode == "descriptor:d"
    missing = [_Entry("d2", "type", KINDS, "descriptor", "models/does_not_exist.joblib", cost=1)] + POSE
    ts = cascade.type_stage(ch, spec, missing)
    assert ts.mode == "prior:uniform" and "no descriptor model" in capsys.readouterr().err


def test_filter_type_entries():
    reg = [_Entry("d", "type", KINDS, "descriptor", "x.joblib"), _Entry("c", "type", KINDS, "cnn", "y"),
           _Entry("a", "type", KINDS, "analytic", "voxel_fit")] + POSE
    ids = lambda r: [e.id for e in r if e.stage == "type"]  # noqa: E731
    assert ids(cascade.filter_type_entries(reg)) == ["d", "c", "a"]
    assert ids(cascade.filter_type_entries(reg, "descriptors")) == ["d", "a"]
    assert ids(cascade.filter_type_entries(reg, "cnn")) == ["c", "a"]
    assert len(cascade.filter_type_entries(reg, "cnn")) == 2 + len(POSE)
    with pytest.raises(ValueError):
        cascade.filter_type_entries(reg, "nets")
