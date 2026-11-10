"""reco_engine/validation.py: the rating of a reconstruction against a known tree, on synthetic programs with known
answers - metrics (exact and equivalent trees, a grown sphere), the grade, and the attribution of every defect class to
its operation, the separation point included."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reco_engine.csg_surface import compile_program  # noqa: E402
from reco_engine.result_schema import Operation, Primitive, ReconstructionResult  # noqa: E402
from reco_engine.run_path import PathLog  # noqa: E402
from reco_engine.truth import sp, truth_from_tree  # noqa: E402
from reco_engine.validation import (GRADE_V1, attribute, grade, rate, surface_metrics, volume_history,  # noqa: E402
                                    volume_metrics)

FAST = dict(cells=48, seed=0)
B = sp.Box((2.0, 1.5, 1.0))
C = sp.Placed(sp.Cylinder(0.5, 4.0), np.eye(3), np.array([0.0, 0.0, -2.0]))
TRUTH = truth_from_tree(sp.Subtraction(B, C), "box_with_bore", meta=dict(genus=1))


def placed_box(half, t):
    return sp.Placed(sp.Box(half), np.eye(3), np.asarray(t, float))


def sig(att):
    return [r for r in att["regions"] if r["significant"]]


def test_prefix_fold_is_program_model():
    ops = [("add", B), ("cut", C), ("add", placed_box((0.3, 0.3, 0.3), (2.2, 0.0, 0.0))), ("intersect", sp.HalfSpace((0, 0, 1), 0.5))]
    fold = sp.Intersection(sp.Union(sp.Subtraction(B, C), ops[2][1]), ops[3][1])
    P = np.random.default_rng(0).uniform(-3, 3, (20000, 3))
    assert np.array_equal(compile_program(ops).sdf(P) <= 0, fold.sdf(P) <= 0)


def test_exact_and_equivalent_programs_are_tier_a():
    for ops in ([("add", B), ("cut", C)],
                [("add", placed_box((1.0, 1.5, 1.0), (-1.0, 0, 0))), ("add", placed_box((1.0, 1.5, 1.0), (1.0, 0, 0))), ("cut", C)]):
        h = volume_history(TRUTH, ops, **FAST)
        v = volume_metrics(h)
        s = surface_metrics(TRUTH, compile_program(ops), n=30_000)
        s.pop("_samples")
        assert v["iou"] == 1.0 and v["missing"] == 0.0 and v["invented"] == 0.0
        assert s["hausdorff"] < 1e-6 and s["coverage"]["0.001"]["recall"] > 0.999
        g = grade(dict(volume=v, surface=s), dict(failed=[]))
        assert g["tier"] == "A", g
        assert sig(attribute(TRUTH, h, ops)) == []


def test_grown_sphere_gives_the_analytic_distances():
    eps = 0.01
    truth = truth_from_tree(sp.Sphere(1.0), "ball")
    D = truth.diag
    s = surface_metrics(truth, compile_program([("add", sp.Sphere(1.0 + eps))]), n=30_000)
    s.pop("_samples")
    t2m, m2t = s["truth_to_model"], s["model_to_truth"]
    assert t2m["mean"] == pytest.approx(eps / D, rel=1e-4) and t2m["rms"] == pytest.approx(eps / D, rel=1e-4)
    assert t2m["std"] < 1e-9 and t2m["signed_mean"] == pytest.approx(-eps / D, rel=1e-4)    # the truth lies inside
    assert m2t["signed_mean"] == pytest.approx(eps / D, rel=1e-4)                            # the model lies outside
    assert s["hausdorff"] == pytest.approx(eps / D, rel=1e-4)


def test_grade_tiers_and_hard_failures():
    good = dict(volume=dict(iou=0.995), surface=dict(hausdorff=5e-3, rms=1e-3,
                coverage={f"{t:g}": dict(recall=0.97, precision=0.96) for t in (1e-3, 2e-3, 5e-3, 1e-2)}))
    assert grade(good, dict(failed=[]))["tier"] == "B"
    assert grade(good, dict(failed=["genus 0 != 1"]))["tier"] == "E"
    assert grade(dict(volume=dict(iou=0.5), surface={}), dict(failed=[]))["tier"] == "E"
    assert [t[0] for t in GRADE_V1["tiers"]] == ["A", "B", "C", "D"]


@pytest.mark.parametrize("ops,cls,culprit,time", [
    # an omission is timed at the stop (no path here); its culprit is the add that brought the material (omitted cut)
    ([("add", B)], "omitted cut", 0, 1),
    ([("add", B), ("cut", C), ("add", placed_box((0.4, 0.4, 0.4), (2.3, 0.0, 0.0)))], "over-add", 2, 2),
    ([("add", B), ("cut", C), ("cut", placed_box((0.6, 0.6, 0.6), (2.0, 1.5, 1.0)))], "over-cut", 2, 2),
    ([("add", B), ("intersect", sp.HalfSpace((0.0, 0.0, 1.0), 0.5)), ("cut", C)], "over-clip", 1, 1),
    ([("add", placed_box((2.4, 1.5, 1.0), (0.4, 0.0, 0.0))), ("cut", C)], "uncut excess", 0, 0),
    ([("add", placed_box((1.6, 1.5, 1.0), (-0.4, 0.0, 0.0))), ("cut", C)], "omitted add", -1, 2),
])
def test_each_defect_class_goes_to_its_operation(ops, cls, culprit, time):
    h = volume_history(TRUTH, ops, **FAST)
    att = attribute(TRUTH, h, ops)
    regions = sig(att)
    assert regions, att["classes"]
    top = max(regions, key=lambda r: r["volume_share"])
    assert top["cls"] == cls and top["culprit"] == culprit, (top["cls"], top["culprit"], att["classes"])
    sep = att["separation"]
    assert sep is not None and sep["time"] == time
    if cls.startswith("omitted"):
        assert sep["cause"] in ("never proposed", "budget or time")


def test_repaired_over_cut_is_transient_and_a_redundant_add_is_flagged():
    corner = placed_box((0.6, 0.6, 0.6), (2.0, 1.5, 1.0))
    inner = placed_box((0.5, 0.5, 0.5), (-1.0, 0.5, 0.0))
    ops = [("add", B), ("cut", corner), ("add", sp.Intersection(corner, B)), ("add", inner), ("cut", C)]
    h = volume_history(TRUTH, ops, **FAST)
    att = attribute(TRUTH, h, ops)
    assert volume_metrics(h)["iou"] == 1.0
    assert sig(att) == []
    assert att["by_op"][1]["transient"] > 0 and att["by_op"][1]["persisting"] == 0
    assert att["by_op"][3]["redundant"]
    assert not att["by_op"][0]["redundant"]


def test_a_rejected_candidate_that_would_have_repaired_an_omission_is_the_separation_point():
    ops = [("add", B)]
    h = volume_history(TRUTH, ops, **FAST)
    bore = dict(kind="cylinder", quaternion=[0, 0, 0, 1], center_mm=[0.0, 0.0, 0.0], dims_mm=dict(radius=0.5, half_length=2.0),
                u=None, v=None, label=None, parts=[], combine=None)
    wrong = dict(bore, center_mm=[5.0, 0.0, 0.0])
    log = PathLog(events=[], grid=None, branch="main", final=[dict(op="add", op_index=0, net_gain=10.0)], removed=[],
                  candidates=[dict(event="step", kind="rejected", index=3, n_ops=1, op="cut", primitive=wrong, reason="far"),
                              dict(event="step", kind="rejected", index=5, n_ops=1, op="cut", primitive=bore,
                                   reason="gain below the threshold")],
                  stopped="residual within tolerance")
    att = attribute(TRUTH, h, ops, log)
    sep = att["separation"]
    assert sep["cls"] == "omitted cut" and sep["time"] == 1
    assert sep["cause"] == "judge rejected a good candidate"
    assert sep["event"]["index"] == 5 and sep["event"]["reason"] == "gain below the threshold"


def test_rate_a_schema_program_end_to_end():
    result = ReconstructionResult("box_with_bore.stl", [
        Operation("add", Primitive("box", dims_mm=dict(x_half=2.0, y_half=1.5, z_half=1.0))),
        Operation("cut", Primitive("cylinder", dims_mm=dict(radius=0.5, half_length=2.0)))])
    r = rate(TRUTH, result, None, None, cells=48, n_surface=20_000, n_prefix=4_000)
    assert r["grade"]["tier"] == "A", r["grade"]
    assert r["vector"]["tree"]["recall"] == 1.0 and r["vector"]["tree"]["precision"] == 1.0
    assert r["vector"]["prefix_coverage"][-1] > 0.99 and r["vector"]["prefix_coverage"][0] < r["vector"]["prefix_coverage"][-1]
    assert r["attribution"]["separation"] is None
    assert not r["unmeasured"] and r["hard"]["failed"] == []
    empty = rate(TRUTH, ReconstructionResult("x", []), None, None)
    assert empty["grade"]["tier"] == "E" and empty["attribution"]["separation"]["cause"] == "no program"


def test_a_cone_described_from_the_other_end_matches():
    from reco_engine.csg_surface import compile_tree
    from reco_engine.validation import match_leaves
    truth = compile_tree(sp.Placed(sp.Cone(3.0, 1.0, 4.0), np.eye(3), np.zeros(3)))
    flip = np.diag([1.0, -1.0, -1.0])                                  # turned upside down: the big end at z = 4
    model = compile_tree(sp.Placed(sp.Cone(1.0, 3.0, 4.0), flip, np.array([0.0, 0.0, 4.0])))
    P = np.random.default_rng(2).uniform(-4, 6, (20000, 3))
    assert np.array_equal(truth.sdf(P) <= 0, model.sdf(P) <= 0)        # the same solid
    m = match_leaves(truth, model, 6.0)
    assert m["recall"] == 1.0 and m["precision"] == 1.0


def test_a_uniform_undersize_skin_is_a_fit_deviation_of_its_operation():
    truth = truth_from_tree(sp.Box((2.0, 1.5, 1.0)), "box")
    ops = [("add", sp.Box((1.996, 1.497, 0.998)))]                     # every face 1-4e-3 short: a thin skin missing
    h = volume_history(truth, ops, cells=96, seed=0)
    att = attribute(truth, h, ops, model=compile_program(ops))
    fit = [r for r in att["regions"] if r["cls"] == "fit deviation"]
    assert len(fit) == 1 and fit[0]["culprit"] == 0 and fit[0]["significant"]
    sep = att["separation"]
    assert sep["cls"] == "fit deviation" and sep["time"] == 0 and sep["cause"] == "fit precision of an accepted operation"
    assert all(r["time"] is None for r in att["regions"] if not r["significant"])


def test_the_grade_names_its_limiting_factor():
    v = dict(volume=dict(iou=0.998), surface=dict(hausdorff=1.6e-4, rms=7.7e-5,
             coverage={f"{t:g}": dict(recall=0.999, precision=0.999) for t in (1e-3, 2e-3, 5e-3, 1e-2)}))
    g = grade(v, dict(failed=[]))
    assert g["tier"] == "B"
    assert g["limiting"] == [dict(tier="A", check="iou", value=0.998, op=">=", threshold=0.999)]


def test_a_small_early_fit_deviation_does_not_hide_the_decisive_omission():
    ops = [("add", sp.Box((1.998, 1.498, 0.998)))]                     # a thin skin short (step 0) and no bore at all
    h = volume_history(TRUTH, ops, cells=96, seed=0)
    att = attribute(TRUTH, h, ops, model=compile_program(ops))
    assert att["first_defect"]["cls"] == "fit deviation" and att["first_defect"]["time"] == 0
    assert att["separation"]["cls"] == "omitted cut" and att["separation"]["time"] == 1
    assert att["dominant"]["cls"] == "omitted cut"
