"""reco_engine/residual_tree.py: the decision on the difference, the children, the solid gate; and the tree end to end
on an example part (mode "tree" with the auto start) - no FreeCAD, no feature phase, a coarse grid."""

import os
from pathlib import Path

import numpy as np
import pytest

from reco_engine import residual_loop as rl
from reco_engine import residual_tree as rt
from reco_engine.result_schema import Primitive

ROOT = Path(__file__).resolve().parents[1]


class Stub:
    """A state with a residual: pos / neg masks given; the loop's grid is a 1 mm cube grid of n^3."""

    def __init__(self, pos, neg, occ_in=None):
        self._pos, self._neg = pos, neg
        self.occ_in = occ_in if occ_in is not None else pos | ~neg
        self.occ_model = (self.occ_in & ~pos) | neg                 # the model: the input without the missing, plus the excess
        self.cut_history = np.zeros(pos.shape, np.int16)

    def residual(self, s_model=None):
        return self._pos, self._neg

    def size(self, s_model=None):
        return int(self._pos.sum() + self._neg.sum())


def cfg(**kw):
    base = dict(mode="tree", tolerance=0.005, min_component=6, rough_min_piece_share=0.001, cuts_first=None)
    base.update(kw)
    return rl.ResidualConfig(**base)


def masks(n=20):
    pos = np.zeros((n, n, n), bool)
    neg = np.zeros((n, n, n), bool)
    return pos, neg


def test_classify_accepts_within_tolerance_and_stops_on_the_budget():
    pos, neg = masks()
    st = Stub(pos, neg)
    d = rt.classify(st, cfg(), 8000, 1, 10, lambda op: 0.0, rl._components, 6, set(), rl._signature)
    assert d.branch == "D0" and "tolerance" in d.reason
    pos[2:8, 2:8, 2:8] = True
    d = rt.classify(st, cfg(), 8000, 10, 10, lambda op: 0.0, rl._components, 6, set(), rl._signature)
    assert d.branch == "D6" and "budget" in d.reason


def test_classify_carves_first_then_adds_then_leaves_the_rest_to_the_features():
    pos, neg = masks()
    pos[2:6, 2:6, 2:6] = True                   # 64 missing voxels
    neg[10:18, 10:18, 10:18] = True             # 512 excess voxels
    st = Stub(pos, neg)
    d = rt.classify(st, cfg(), 8000, 1, 10, lambda op: 0.0, rl._components, 6, set(), rl._signature)
    assert d.branch == "D1" and d.components[0][0] == "cut" and len(d.components) == 2
    d = rt.classify(st, cfg(cuts_first=False), 8000, 1, 10, lambda op: 0.0, rl._components, 6, set(), rl._signature)
    assert d.branch == "D2" and d.components[0][0] == "add"
    # the excess component already tried: the missing one is next
    tried = {rl._signature("cut", neg)}
    d = rt.classify(st, cfg(), 8000, 1, 10, lambda op: 0.0, rl._components, 6, tried, rl._signature)
    assert d.branch == "D2"
    # everything below the rough floor: left for the feature phase
    d = rt.classify(st, cfg(), 8000, 1, 10, lambda op: 1000.0, rl._components, 6, set(), rl._signature)
    assert d.branch == "D3" and "feature phase" in d.reason
    # nothing at all above the component floor (and not within tolerance: tolerance 0)
    pos[:], neg[:] = False, False
    neg[0, 0, 0:3] = True
    d = rt.classify(st, cfg(tolerance=0.0), 8000, 1, 10, lambda op: 0.0, rl._components, 6, set(), rl._signature)
    assert d.branch == "D6" and "no component" in d.reason


def test_fuse_chain_is_one_composite_with_tree_sources():
    boxes = [Primitive(kind="box", center_mm=(float(i), 0, 0), dims_mm=dict(x_half=0.5, y_half=1, z_half=1)) for i in range(3)]
    c = rt.fuse_chain(boxes, "three boxes", "tree:carve")
    assert c.kind == "composite" and c.combine == "fuse" and c.label == "three boxes" and rt.is_child(c)
    assert c.parts[1] is boxes[2] and c.parts[0].kind == "composite" and c.parts[0].parts[0] is boxes[0]
    one = rt.fuse_chain(boxes[:1], "one", "tree:add")
    assert one.kind == "box" and rt.is_child(one) and not rt.is_child(boxes[0])
    assert c.defect() is None


def test_a_child_never_sets_the_rough_floor_scale():
    fg = dict(add=None, cut=None)
    assert rt.grade_of(fg, "cut", 700000, child=True) == {} and fg["cut"] is None     # the child's gain is no scale
    g = rt.grade_of(fg, "cut", 5000, child=False)
    assert fg["cut"] == 5000.0 and g["grade"] == 1.0
    g = rt.grade_of(fg, "cut", 60000, child=True)                                    # a later child is graded on it
    assert g["grade"] == 12.0 and g["first_cut_gain"] == 5000.0
    assert rt.grade_of(None, "cut", 1, child=False) == {}


def test_severs_counts_pieces_against_the_input():
    occ = np.zeros((12, 12, 12), bool)
    occ[2:10, 2:10, 2:10] = True
    st = Stub(np.zeros_like(occ), np.zeros_like(occ), occ_in=occ)
    whole = np.where(occ, 1.0, -1.0)
    assert not rl._severs(st, whole)
    split = whole.copy()
    split[2:10, 5:7, 2:10] = -1.0                 # a slot through the middle: two pieces
    assert rl._severs(st, split)
    assert not rl._severs(st, split, before=split)   # no worse than the model before


@pytest.mark.skipif(not (ROOT / "dataset_gen" / "examples" / "bs_box_step.stl").is_file(), reason="example part missing")
def test_tree_runs_the_step_part_to_tolerance_without_fallbacks():
    """bs_box_step: a box with a step - the auto start covers it with its box, the step is one excess component, cut by
    a box. Every round is a decision step; the model stays one piece."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    c = rl.ResidualConfig(mode="tree", start_shape="auto", resolution=48, max_resolution=48, grid="cube", freecad=False,
                          feature_phase=False, voxel_fallback=True, field_backend="igl", polish=False, start_vary=False,
                          exact_solids_gate=False)
    steps = list(rl.residual_steps(str(ROOT / "dataset_gen" / "examples" / "bs_box_step.stl"), c))
    kinds = [s.kind for s in steps]
    assert kinds[-1] == "done" and "decision" in kinds
    done = steps[-1]
    decisions = [s.info["branch"] for s in steps if s.kind == "decision"]
    # either the cover and a cut of the step, or (D1c: the step's excess got no primitive, the child of boxes) the
    # contained base slab and the step added - two primitives both ways
    assert decisions[0] in ("D1", "D2") and decisions[-1] == "D0", decisions
    ops = [o for o in done.result.operations if o.accepted]
    assert 2 <= len(ops) <= 4 and ops[0].op == "add"
    if "D1c" in decisions:
        assert sum(o.op == "add" for o in ops) >= 2 and any("contained start" in (o.primitive.label or "") for o in ops)
    assert not any(s.kind == "fallback" for s in steps)
    assert done.info["stats"]["iou"] >= 0.98


def _box(label="", half=4.0):
    p = Primitive(kind="box", quaternion=(0.0, 0.0, 0.0, 1.0), center_mm=(0.0, 0.0, 0.0),
                  dims_mm=dict(x_half=half, y_half=half, z_half=half))
    p.label = label
    return p


def test_exact_gate_refuses_the_cut_whose_build_splits_the_model(tmp_path):
    """D5 exact (2026-10-16): the chosen cut is built with the program so far; a build in more solids than allowed is
    refused and the next passing candidate by gain is taken; a failed build leaves the grid's answer standing; the gate
    is off without FreeCAD."""
    from reco_engine.result_schema import Operation, ReconstructionResult
    occ = np.zeros((12, 12, 12), bool)
    occ[2:10, 2:10, 2:10] = True
    st = Stub(np.zeros_like(occ), np.zeros_like(occ), occ_in=occ)
    result = ReconstructionResult(input_path="x.stl", operations=[Operation("add", _box("cover"))])
    built = []

    def fake_build(res, out_dir, name, timeout):
        last = res.operations[-1].primitive.label
        built.append(last)
        if last == "crash":
            raise RuntimeError("freecadcmd died")
        return dict(valid=last != "invalid", n_solids=3 if last == "splits" else 1, step=str(out_dir / f"{name}.step"),
                    volume=-5.0 if last == "inverted" else 10.0)

    c = cfg(freecad=False, feature_build=fake_build, out_dir=tmp_path, exact_solids_max_checks=2)
    assert rl._exact_severs(st, result, c, _box("splits"), "cut").startswith("severs the model (D5 exact: 3 solids")
    assert rl._exact_severs(st, result, c, _box("keeps"), "cut") is None
    assert rl._exact_severs(st, result, c, _box("crash"), "cut") is None          # no answer: the grid's verdict stands
    assert "D5 exact" in rl._exact_severs(st, result, c, _box("splits"), "add")   # a floating add is gated too
    assert rl._exact_severs(st, result, c, _box("invalid"), "cut") == "makes the solid invalid (D5 exact)"
    assert "non-positive volume" in rl._exact_severs(st, result, c, _box("inverted"), "cut")
    assert st.exact_gate["checks"] == 6 and st.exact_gate["rejected"] == 4 and st.exact_gate["failed"] == 1
    # without an out_dir (the GUI) the gate builds in a temp dir of its own (2026-10-16 evening: NameError, tempfile
    # was imported inside another function only - every GUI run with freecadcmd found crashed at its first cut)
    st2 = Stub(np.zeros_like(occ), np.zeros_like(occ), occ_in=occ)
    c2 = cfg(freecad=False, feature_build=fake_build, out_dir=None, exact_solids_max_checks=2)
    assert rl._exact_severs(st2, result, c2, _box("keeps"), "cut") is None
    assert Path(st2.exact_gate_dir).is_dir() and Path(st2.exact_gate_dir).name == "solids_gate"
    # the loop's choice: the best by gain splits, the second keeps the solid
    ev = lambda gain: dict(ok=True, reason="", gain=gain, s_new=np.where(occ, 1.0, -1.0), s_prim=None)
    chosen = dict(best=(_box("splits"), ev(100)), tried=[(_box("keeps"), ev(60)), (_box("splits"), ev(100)), (_box("small"), ev(10))], why="")
    chosen["tried"][1] = chosen["best"]
    p, e = rl._unsevered_best(st, result, c, chosen, "cut")
    assert p.label == "keeps" and chosen["tried"][1][1]["ok"] is False and "D5 exact" in chosen["tried"][1][1]["reason"]
    # the budget: two builds at most per choice - the third candidate is taken on the grid's word
    chosen = dict(best=(_box("splits"), ev(100)), tried=[(_box("splits"), ev(100)), (_box("splits"), ev(90)), (_box("splits"), ev(80))], why="")
    chosen["tried"][0] = chosen["best"]
    n0 = len(built)
    p, e = rl._unsevered_best(st, result, c, chosen, "cut")
    assert p is chosen["tried"][2][0] and len(built) - n0 == 2
    # off when switched off, whatever build exists
    off = cfg(feature_build=fake_build, exact_solids_gate=False)
    assert not rl._exact_gate_on(off) and rl._exact_severs(st, result, off, _box("splits"), "cut") is None
    assert rl._exact_gate_on(cfg(feature_build=fake_build))
