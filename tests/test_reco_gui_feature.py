"""The feature phase in the supervision GUI, with a FAKE step stream that follows the step contract of the loop
(kinds feature_residual, feature_piece, feature_candidate, feature_accepted, feature_rejected, feature_done): folding into
CascadeStates with the pieces and their states (reco_gui.cascade_session.FeatureView), the timeline rows, the detail panel, the
operations tree, what the 3D view would draw (its data side, `viewer_cascade.layer_meshes`: the window runs offscreen with the
3D view off), breakpoints, the heartbeat, the settings. No FreeCAD, no networks."""
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine.result_schema import Operation, Primitive, ReconstructionResult, StageSource  # noqa: E402
from reco_gui.cascade_session import (CASCADE_STAGES, FEATURE_STAGES, CascadeFolder, bbox_mesh, feature_config,  # noqa: E402
                                      feature_progress, make_cascade_session)
from reco_gui.report import feature_log_lines, render_state, timeline_row  # noqa: E402
from reco_gui.setup import RunSettings, make_session  # noqa: E402

PART = ROOT / "dataset_gen" / "examples" / "bs_box_sphere_pocket.stl"      # a 20 mm box with a spherical pocket
SPEC = {"lo": (-10.0, -10.0, -10.0), "hi": (10.0, 10.0, 10.0), "resolution": 16}
LIMITS = dict(max_pieces=5, max_iterations=2, time_budget_s=600.0, min_score=0.5)


@dataclass
class FakeStep:
    kind: str
    result: ReconstructionResult
    info: dict


def src(mode, conf=1.0):
    return [StageSource(stage, mode, conf) for stage in ("type", "rotation", "dimensions", "position")]


BOX = Primitive("box", center_mm=(0, 0, 0), dims_mm={"x_half": 10, "y_half": 10, "z_half": 10}, confidence=0.97,
                sources=src("analytic:frame"))
POCKET = Primitive("sphere", center_mm=(0, 0, 10), dims_mm={"radius": 6}, confidence=0.9, sources=src("analytic:sphere_fit"))
HOLE = Primitive("cylinder", center_mm=(6, -5, 0), dims_mm={"radius": 1.5, "half_length": 11}, confidence=0.8,
                 sources=src("faces"))
WRONG = Primitive("cylinder", quaternion=(0.0, 0.7071068, 0.0, 0.7071068), center_mm=(0, 0, 8),
                  dims_mm={"radius": 3, "half_length": 4}, confidence=0.4, sources=src("cnn:type_v3", 0.4))
BOSS = Primitive("box", center_mm=(7, 6, 7), dims_mm={"x_half": 2, "y_half": 1, "z_half": 1}, confidence=0.7,
                 sources=src("fit"))

# five pieces of the first iteration's difference, asymmetric and off-centre: id -> (side, volume, box)
BOXES = {1: ("invented", 452.4, ((-6, -6, 4), (6, 6, 10))), 2: ("invented", 155.5, ((4.5, -6.5, -10), (7.5, -3.5, 10))),
         3: ("invented", 40.0, ((-9, 2, -3), (-7, 5, 1))), 4: ("missing", 16.0, ((5, 5, 6), (9, 7, 8))),
         5: ("invented", 3.0, ((-2, -9, -9), (-1, -8, -6)))}
TOTAL = sum(v for _, v, _ in BOXES.values())


def piece(pid, boxes=BOXES, total=TOTAL):
    side, volume, box = boxes[pid]
    return dict(id=pid, side=side, op="add" if side == "missing" else "cut", volume=volume, share=volume / total,
                bbox=box, stl=None)


def mesh_of(pid, boxes=BOXES):
    m = bbox_mesh(*boxes[pid][2])
    return dict(vertices=m.vertices.astype(np.float32), faces=m.triangles.astype(np.int32))


def validation(valid, before, after, iou, n_solids=1):
    return dict(valid=valid, n_solids=n_solids, residual_before=before, residual_after=after, exact_iou=iou, seconds=1.25)


SECOND = {1: ("invented", 30.0, ((-9, 2, -3), (-7, 5, 1)))}                  # the second iteration numbers its pieces anew


def fake_steps(path, config):
    """The rough model (one box), then the feature phase as the loop yields it: an iteration of 5 pieces - piece 1 accepted,
    2 rejected after two guesses (the FreeCAD build), 3 rejected without a guess (primitive None), 4 accepted (with a key
    this GUI does not know), 5 never looked at -, a second iteration of one piece, rejected, then the summary."""
    assert config["top_k"] >= 1
    ops = []

    def res(pos=0, neg=0, chamfer=None, iou=None):
        return ReconstructionResult(input_path=str(path), operations=list(ops), residual_voxels_pos=pos,
                                    residual_voxels_neg=neg, chamfer_mm=chamfer, exact_iou=iou)

    def f(kind, message, **info):
        return FakeStep(kind, res(iou=info.get("exact_iou")), dict(info, message=message, phase="feature"))

    yield FakeStep("input", res(4096, 0), {"spec": SPEC})
    ops.append(Operation("add", BOX, component_voxels=4096, net_gain=4000))
    yield FakeStep("add", res(0, 120), {"candidate": BOX, "op": "add", "net_gain": 4000.0})
    yield FakeStep("residual", res(0, 120), {"pos": np.zeros((16,) * 3, bool), "neg": np.zeros((16,) * 3, bool)})

    yield f("feature_residual", "feature iteration 1: 5 pieces, missing 16, invented 650.9 mm^3, exact IoU 0.9230",
            iteration=1, missing_volume=16.0, invented_volume=650.9, input_volume=7547.6, exact_iou=0.923,
            pieces=[piece(i) for i in BOXES], dropped_small=3, dropped_limit=0, limits=dict(LIMITS), seconds=4.2)
    # piece 1: one guess, accepted
    yield f("feature_piece", "piece 1 (1 of 5): invented, 452.4 mm^3", iteration=1, piece=piece(1), index=1, of=5,
            mesh=mesh_of(1), mesh_truncated=False)
    yield f("feature_candidate", "piece 1: candidate cut:sphere from void_fit, score 0.912", iteration=1, piece_id=1,
            op="cut", primitive=POCKET, source="void_fit", score=0.912, seconds=0.31)
    ops.append(Operation("cut", POCKET, net_gain=110, freecad_valid=True))
    yield f("feature_accepted", "piece 1: accepted cut:sphere (void_fit, score 0.912)", iteration=1, piece_id=1, op="cut",
            primitive=POCKET, source="void_fit", score=0.912, validation=validation(True, 666.9, 214.5, 0.9712))
    # piece 2: two guesses, the better one fails the build
    yield f("feature_piece", "piece 2 (2 of 5): invented, 155.5 mm^3", iteration=1, piece=piece(2), index=2, of=5,
            mesh=mesh_of(2), mesh_truncated=True)
    yield f("feature_candidate", "piece 2: candidate cut:cylinder from cascade, score 0.41", iteration=1, piece_id=2,
            op="cut", primitive=WRONG, source="cascade", score=0.41, seconds=0.9)
    yield f("feature_candidate", "piece 2: candidate cut:cylinder from faces, score 0.633", iteration=1, piece_id=2,
            op="cut", primitive=HOLE, source="faces", score=0.633, seconds=0.2)
    yield f("feature_rejected", "piece 2: rejected cut:cylinder (the build split the model)", iteration=1, piece_id=2,
            op="cut", primitive=HOLE, source="faces", score=0.633, reason="the build split the model into 2 solids",
            validation=validation(False, 214.5, 214.5, 0.9712, n_solids=2))
    # piece 3: no guess at all
    yield f("feature_piece", "piece 3 (3 of 5): invented, 40 mm^3", iteration=1, piece=piece(3), index=3, of=5,
            mesh=mesh_of(3), mesh_truncated=False)
    yield f("feature_rejected", "piece 3: no guess", iteration=1, piece_id=3, op="cut", primitive=None, source=None,
            score=None, reason="no guess reached the minimum score 0.5", validation=None)
    # piece 4: an add, with a key this GUI does not know
    yield f("feature_piece", "piece 4 (4 of 5): missing, 16 mm^3", iteration=1, piece=piece(4), index=4, of=5,
            mesh=mesh_of(4), mesh_truncated=False, something_new={"a": 1})
    yield f("feature_candidate", "piece 4: candidate add:box from fit, score 0.98", iteration=1, piece_id=4, op="add",
            primitive=BOSS, source="fit", score=0.98, seconds=0.05, something_new=[1, 2, 3])
    ops.append(Operation("add", BOSS, net_gain=4, freecad_valid=True))
    yield f("feature_accepted", "piece 4: accepted add:box (fit, score 0.98)", iteration=1, piece_id=4, op="add",
            primitive=BOSS, source="fit", score=0.98, validation=dict(validation(True, 214.5, 198.5, 0.9731), extra="x"),
            something_new=np.zeros(3))
    # second iteration: one piece, rejected; the phase ends on its iteration limit
    yield f("feature_residual", "feature iteration 2: 1 piece, missing 0, invented 198.5 mm^3, exact IoU 0.9731",
            iteration=2, missing_volume=0.0, invented_volume=198.5, input_volume=7547.6, exact_iou=0.9731,
            pieces=[piece(1, SECOND, 30.0)], dropped_small=4, dropped_limit=0, limits=dict(LIMITS), seconds=3.9)
    yield f("feature_piece", "piece 1 (1 of 1): invented, 30 mm^3", iteration=2, piece=piece(1, SECOND, 30.0), index=1,
            of=1, mesh=mesh_of(1, SECOND), mesh_truncated=False)
    yield f("feature_candidate", "piece 1: candidate cut:cylinder from cascade, score 0.2", iteration=2, piece_id=1,
            op="cut", primitive=WRONG, source="cascade", score=0.2, seconds=0.8)
    yield f("feature_rejected", "piece 1: rejected cut:cylinder (score 0.2 below 0.5)", iteration=2, piece_id=1, op="cut",
            primitive=WRONG, source="cascade", score=0.2, reason="score 0.200 below the minimum 0.5", validation=None)
    yield f("feature_done", "feature phase done: 2 applied, 3 rejected, 1 skipped (max_iterations)", iterations=2,
            applied=2, rejected=3, skipped=1, stopped="max_iterations", seconds=31.4, exact_iou=0.9731, missing_volume=0.0,
            invented_volume=198.5)
    yield FakeStep("grade", res(0, 2, chamfer=0.031, iou=0.9731), {})
    yield FakeStep("done", res(0, 2, chamfer=0.031, iou=0.9731), {})


KINDS = [s.kind for s in fake_steps(PART, {"top_k": 1})]


def run_all(session):
    while session.advance() is not None:
        pass
    return session


@pytest.fixture(scope="module")
def session():
    return run_all(make_cascade_session(PART, steps_fn=fake_steps))


def states_of(session, stage):
    return [s for s in session.states if s.step.stage == stage]


def piece_states(state):
    return {p.id: p.state for p in state.feature.pieces}


# ------------------------------------------------------------------------------------------------ the fold

def test_every_kind_folds_and_is_a_stage(session):
    assert session.finished and session.error is None
    assert [s.step.stage for s in session.states] == KINDS
    assert set(FEATURE_STAGES) - {"feature_split"} <= set(KINDS) and set(KINDS) <= set(CASCADE_STAGES)    # the split: below
    for st in session.states:
        if st.step.stage in FEATURE_STAGES:                      # message, phase and stage as the log reads them
            assert st.step.phase == "feature" and st.step.message == st.info["message"] and st.in_feature_phase
            assert "mesh" not in st.info                         # kept as the piece's mesh, not per state
        else:
            assert not st.in_feature_phase
    assert not states_of(session, "feature_done")[0].finished and session.states[-1].finished
    assert session.states[2].feature is None                     # the rough loop has no feature view
    assert session.states[-1].feature.done["stopped"] == "max_iterations"          # kept for the summary


def test_piece_states_after_each_step(session):
    expected = [                                                 # after each feature step of the stream, in order
        {1: "pending", 2: "pending", 3: "pending", 4: "pending", 5: "pending"},      # feature_residual 1
        {1: "current", 2: "pending", 3: "pending", 4: "pending", 5: "pending"},      # piece 1
        {1: "current", 2: "pending", 3: "pending", 4: "pending", 5: "pending"},      # its guess
        {1: "accepted", 2: "pending", 3: "pending", 4: "pending", 5: "pending"},     # accepted
        {1: "accepted", 2: "current", 3: "pending", 4: "pending", 5: "pending"},     # piece 2
        {1: "accepted", 2: "current", 3: "pending", 4: "pending", 5: "pending"},
        {1: "accepted", 2: "current", 3: "pending", 4: "pending", 5: "pending"},     # its second guess
        {1: "accepted", 2: "rejected", 3: "pending", 4: "pending", 5: "pending"},
        {1: "accepted", 2: "rejected", 3: "current", 4: "pending", 5: "pending"},    # piece 3
        {1: "accepted", 2: "rejected", 3: "rejected", 4: "pending", 5: "pending"},   # rejected without a guess
        {1: "accepted", 2: "rejected", 3: "rejected", 4: "current", 5: "pending"},   # piece 4
        {1: "accepted", 2: "rejected", 3: "rejected", 4: "current", 5: "pending"},
        {1: "accepted", 2: "rejected", 3: "rejected", 4: "accepted", 5: "pending"},
        {1: "pending"},                                                               # feature_residual 2
        {1: "current"}, {1: "current"}, {1: "rejected"},
        {1: "rejected"},                                                              # feature_done
    ]
    feature = [s for s in session.states if s.step.stage in FEATURE_STAGES]
    assert [piece_states(s) for s in feature] == expected
    assert [s.feature.iteration for s in feature] == [1] * 13 + [2] * 5
    assert [(s.feature.index, s.feature.of) for s in states_of(session, "feature_piece")] == [(1, 5), (2, 5), (3, 5), (4, 5), (1, 1)]


def test_volumes_and_exact_iou_per_iteration(session):
    first = states_of(session, "feature_residual")[0].feature
    assert first.missing_volume == 16.0 and first.invented_volume == 650.9 and first.input_volume == 7547.6
    assert first.exact_iou == 0.923 and first.limits == LIMITS and first.dropped_small == 3 and first.history == ()
    acc = states_of(session, "feature_accepted")
    assert [s.feature.exact_iou for s in acc] == [0.9712, 0.9731]            # from the validated build
    assert [s.feature.applied for s in acc] == [1, 2]
    last = session.states[-1].feature
    assert [h["iteration"] for h in last.history] == [1, 2]
    one, two = last.history
    assert (one["pieces"], one["applied"], one["rejected"], one["skipped"]) == (5, 2, 2, 1) and one["exact_iou"] == 0.9731
    assert (two["pieces"], two["applied"], two["rejected"], two["skipped"]) == (1, 0, 1, 0) and two["invented_volume"] == 198.5
    assert last.applied == 2 and last.rejected == 3
    p2 = states_of(session, "feature_rejected")[0].feature.piece(2)
    assert p2.candidates == 2 and p2.primitive is HOLE and p2.source == "faces" and p2.score == 0.633
    assert "split" in p2.reason and p2.validation["n_solids"] == 2 and p2.mesh_truncated
    p3 = states_of(session, "feature_rejected")[1].feature.piece(3)
    assert p3.candidates == 0 and p3.primitive is None and p3.score is None and "no guess" in p3.reason


def test_candidate_and_program_follow_the_feature_steps(session):
    cand = states_of(session, "feature_candidate")
    assert [c.candidate for c in cand] == [POCKET, WRONG, HOLE, BOSS, WRONG]
    assert all(c.candidate_mesh is not None and c.verdict is None for c in cand)
    assert [c.candidate_op for c in cand] == ["cut", "cut", "cut", "add", "cut"]
    acc = states_of(session, "feature_accepted")
    assert [len(a.program) for a in acc] == [2, 3] and all(a.verdict == "accepted" and a.candidate_mesh is None for a in acc)
    assert acc[0].model_mesh is not states_of(session, "feature_piece")[0].model_mesh       # the model was re-meshed
    rej = states_of(session, "feature_rejected")
    assert rej[0].candidate is HOLE and rej[0].verdict == "rejected" and rej[0].candidate_mesh is not None
    assert rej[1].candidate is None and rej[1].candidate_mesh is None                      # primitive None: nothing drawn
    assert all(p.candidate is None for p in states_of(session, "feature_piece"))
    assert session.states[-1].candidate is None


def test_one_timeline_iteration_per_piece_and_the_row_text(session):
    rows = [timeline_row(s) for s in session.states]
    assert len(rows) == len(KINDS) and [r[2] for r in rows] == KINDS
    assert [r[0] for r in rows] == [str(i + 1) for i in range(len(KINDS))]
    by_stage = {}
    for r in rows:
        by_stage.setdefault(r[2], []).append(r)
    assert [r[3] for r in by_stage["feature_piece"]] == ["2", "3", "4", "5", "6"]          # the rough add was iteration 1
    assert [r[3] for r in by_stage["feature_candidate"]] == ["2", "3", "3", "5", "6"]
    assert [r[3] for r in by_stage["feature_rejected"]] == ["3", "4", "6"]
    assert [r[3] for r in by_stage["feature_residual"] + by_stage["feature_done"]] == ["", "", ""]
    assert all(r[1] == "feature" for stage in FEATURE_STAGES for r in by_stage.get(stage, ()))
    assert by_stage["feature_residual"][0][4] == "feature iteration 1: 5 pieces, missing 16, invented 650.9 mm^3, exact IoU 0.9230"
    assert by_stage["feature_accepted"][1][4] == "piece 4: accepted add:box (fit, score 0.98)"
    assert by_stage["feature_done"][0][4].startswith("feature phase done: 2 applied, 3 rejected, 1 skipped")


def test_messages_are_made_when_the_loop_sends_none():
    fold, prev, messages = CascadeFolder(wall_grid=False), None, {}
    for i, step in enumerate(fake_steps(PART, {"top_k": 1})):
        info = {k: v for k, v in step.info.items() if k not in ("message", "phase")}
        prev = fold(prev, FakeStep(step.kind, step.result, info), i)
        messages.setdefault(step.kind, prev.step.message)
        assert prev.step.phase == "feature" or step.kind not in FEATURE_STAGES
    assert messages["feature_residual"] == "feature iteration 1: 5 piece(s), missing 16, invented 650.9 mm^3, exact IoU 0.9230"
    assert messages["feature_piece"] == "piece 1 (1 of 5): invented, 452.4 mm^3 (67.8%)"
    assert messages["feature_candidate"] == "piece 1: candidate cut:sphere from void_fit, score 0.912"
    assert messages["feature_accepted"] == "piece 1: accepted cut:sphere from void_fit, score 0.912"
    assert messages["feature_rejected"].endswith("(the build split the model into 2 solids)")
    assert messages["feature_done"] == "feature phase done: 2 applied, 3 rejected, 1 skipped; stopped: max_iterations"


def test_detail_html_has_score_reason_validation_and_limits(session):
    for st in session.states:
        assert st.step.stage in render_state(st)
    res = render_state(states_of(session, "feature_residual")[0])
    assert "Feature phase" in res and "650.9" in res and "0.9230" in res and "5 pending" in res
    assert "3 too small" in res and "limits in force" in res and "max pieces 5" in res and "time budget s 600" in res
    assert "Pieces of this iteration" in res and "452.4" in res
    cand = render_state(states_of(session, "feature_candidate")[0])
    assert "Piece in work" in cand and "void_fit" in cand and "0.912" in cand and "cut:sphere" in cand
    assert "invented &rarr; cut" in cand and "(1 of 5)" in cand and "67.8%" in cand and "needs &ge; 0.5" in cand
    acc = render_state(states_of(session, "feature_accepted")[0])
    assert "accepted" in acc and "666.9 &rarr; 214.5" in acc and "0.9712" in acc and "1 solid(s)" in acc and "valid" in acc
    rej = render_state(states_of(session, "feature_rejected")[0])
    assert "the build split the model into 2 solids" in rej and "INVALID" in rej and "2 solid(s)" in rej
    assert "0.633" in rej and "faces" in rej and "guesses tried" in rej
    none = render_state(states_of(session, "feature_rejected")[1])
    assert "no guess reached the minimum score 0.5" in none and "none" in none and "FreeCAD build" not in none
    done = render_state(states_of(session, "feature_done")[0])
    assert "Feature phase: summary" in done and "<b>max_iterations</b>" in done and "2 applied" in done
    assert "31.4 s" in done and "Feature iterations" in done and "iteration 2" in done
    assert "Feature phase: summary" in render_state(session.states[-1])                   # still there at the end
    assert "Piece in work" not in render_state(session.states[-1])


def test_verbose_lines_of_the_feature_steps(session):
    lines = feature_log_lines(states_of(session, "feature_residual")[0])
    assert lines[0].startswith("limits: max_pieces 5") and "3 too small" in lines[1]
    assert len([x for x in lines if x.startswith("piece")]) == 5 and "452.4" in lines[2]
    rej = feature_log_lines(states_of(session, "feature_rejected")[0])
    assert "n_solids 2" in rej[0] and rej[1] == "reason: the build split the model into 2 solids"
    assert feature_log_lines(states_of(session, "feature_rejected")[1]) == ["reason: no guess reached the minimum score 0.5"]
    assert len(feature_log_lines(states_of(session, "feature_done")[0])) == 2
    assert feature_log_lines(session.states[0]) == [] and feature_log_lines(session.states[-1]) == []


def test_what_the_viewer_would_draw(session):
    """The data side of the 3D view (the render window itself is not opened offscreen): per step the layers and their
    meshes. The pieces sit at their world boxes (asymmetric, off-centre: no axis swapped or mirrored)."""
    from reco_gui.viewer_cascade import DEBUG_DEFAULT_ON, DEBUG_LAYERS, FEATURE_LAYERS, layer_meshes

    def box_of(mesh):
        return mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    assert set(FEATURE_LAYERS) <= set(DEBUG_LAYERS) and set(FEATURE_LAYERS) <= set(DEBUG_DEFAULT_ON)
    rough = layer_meshes(session.states[2])
    assert not set(rough) & set(FEATURE_LAYERS) and "model" in rough and "input" in rough
    res = layer_meshes(states_of(session, "feature_residual")[0])
    assert set(res) & set(FEATURE_LAYERS) == {"f_pending"} and len(res["f_pending"].triangles) == 5 * 12
    lo, hi = box_of(res["f_pending"])
    assert lo == pytest.approx([-9, -9, -10]) and hi == pytest.approx([9, 7, 10])
    cand = layer_meshes(states_of(session, "feature_candidate")[3])                       # piece 4, the add
    assert set(cand) & set(FEATURE_LAYERS) == {"f_pending", "f_current", "f_accepted", "f_rejected", "f_candidate"}
    assert "candidate" not in cand and "model" in cand and "input" in cand
    lo, hi = box_of(cand["f_current"])
    assert lo == pytest.approx([5, 5, 6]) and hi == pytest.approx([9, 7, 8])
    lo, hi = box_of(cand["f_candidate"])                                                  # the guess lies over its piece
    assert lo == pytest.approx([5, 5, 6], abs=0.2) and hi == pytest.approx([9, 7, 8], abs=0.2)
    assert len(cand["f_rejected"].triangles) == 2 * 12 and len(cand["f_accepted"].triangles) == 12
    lo, hi = box_of(cand["f_accepted"])
    assert lo == pytest.approx([-6, -6, 4]) and hi == pytest.approx([6, 6, 10])
    # a layer whose pieces did not change keeps its mesh object: the view redraws only what changed
    before = layer_meshes(states_of(session, "feature_piece")[3])
    assert before["f_rejected"] is cand["f_rejected"] and before["f_accepted"] is cand["f_accepted"]
    assert before["f_current"] is cand["f_current"] and "f_candidate" not in before
    rej = layer_meshes(states_of(session, "feature_rejected")[0])
    assert "f_candidate" in rej and "f_current" not in rej
    done = layer_meshes(states_of(session, "feature_done")[0])
    assert set(done) & set(FEATURE_LAYERS) == {"f_rejected"}
    end = layer_meshes(session.states[-1])                                                # after the phase: the model only
    assert not set(end) & set(FEATURE_LAYERS)


def test_tolerant_fold_unknown_keys_missing_keys_and_no_primitive():
    fold = CascadeFolder(wall_grid=False)
    res = ReconstructionResult(input_path=str(PART), operations=[Operation("add", BOX)])
    s = fold(None, FakeStep("input", res, {"spec": SPEC}), 0)
    # a piece the iteration never listed, no feature_residual before it, keys this GUI does not know
    s = fold(s, FakeStep("feature_candidate", res, {"piece_id": 9, "op": "cut", "primitive": None, "new_key": object()}), 1)
    assert piece_states(s) == {9: "current"} and s.candidate is None and s.step.phase == "feature"
    assert "piece 9" in s.step.message and s.step.iteration is not None
    s = fold(s, FakeStep("feature_rejected", res, {"piece_id": 9, "primitive": None, "reason": None, "validation": "bad"}), 2)
    assert piece_states(s) == {9: "rejected"} and s.feature.piece(9).validation is None
    s = fold(s, FakeStep("feature_piece", res, {"piece": {"id": 3, "bbox": "nonsense", "volume": None},
                                                "mesh": {"vertices": [[0, 0, 0]], "faces": [[0, 1, 2]]}}), 3)
    assert piece_states(s) == {9: "rejected", 3: "current"} and s.feature.piece(3).mesh is None
    s = fold(s, FakeStep("feature_piece", res, {}), 4)                                     # nothing at all
    assert s.feature.piece(3).state == "current"
    s = fold(s, FakeStep("feature_residual", res, {"pieces": None, "limits": 5, "exact_iou": "n/a"}), 5)
    assert s.feature.pieces == () and s.feature.exact_iou is None
    s = fold(s, FakeStep("feature_done", None, {}), 6)
    assert s.feature.done == {} and not s.finished and len(s.program) == 1
    for state in (s,):
        render_state(state), timeline_row(state), feature_log_lines(state)
    assert feature_progress(s) == ""


def batch_steps(path, config):
    """The order of the loop when it validates an iteration's guesses together (its default): every piece is guessed
    first, the verdicts of the FreeCAD build follow at the end - labelled primitives, as the loop makes them."""
    ops = [Operation("add", BOX)]

    def f(kind, **info):
        return FakeStep(kind, ReconstructionResult(input_path=str(path), operations=list(ops)), dict(info, phase="feature"))
    yield FakeStep("input", ReconstructionResult(input_path=str(path)), {"spec": SPEC})
    yield f("feature_residual", iteration=1, missing_volume=16.0, invented_volume=650.9, input_volume=7547.6,
            exact_iou=0.923, pieces=[piece(i) for i in (1, 2, 3)], dropped_small=0, dropped_limit=2, limits=dict(LIMITS))
    guesses = {1: (POCKET, "fit"), 2: (HOLE, "step faces")}
    for k, pid in enumerate((1, 2, 3)):
        yield f("feature_piece", iteration=1, piece=piece(pid), index=k + 1, of=3, mesh=mesh_of(pid), mesh_truncated=False)
        if pid in guesses:
            yield f("feature_candidate", iteration=1, piece_id=pid, op="cut", primitive=guesses[pid][0],
                    source=guesses[pid][1], score=0.9, seconds=0.1)
        else:
            yield f("feature_rejected", iteration=1, piece_id=pid, op="cut", primitive=None, source=None, score=None,
                    reason="no candidate", validation=None)
    val = dict(validation(True, 666.9, 100.0, 0.98), batch=2)
    for pid in (1, 2):
        prim, source = guesses[pid]
        applied = Primitive(prim.kind, prim.quaternion, prim.center_mm, dict(prim.dims_mm), confidence=prim.confidence,
                            sources=list(prim.sources), label=f"feature ({source})")
        ops.append(Operation("cut", applied, freecad_valid=True))
        yield f("feature_accepted", iteration=1, piece_id=pid, op="cut", primitive=applied, source=source, score=0.9,
                validation=dict(val))
    yield f("feature_done", iterations=1, applied=2, rejected=1, skipped=0, stopped="max iterations", seconds=12.0,
            exact_iou=0.98, missing_volume=0.0, invented_volume=100.0)
    yield FakeStep("done", ReconstructionResult(input_path=str(path), operations=list(ops)), {})


def test_guesses_validated_together_wait_as_guessed():
    session = run_all(make_cascade_session(PART, steps_fn=batch_steps))
    states = [piece_states(s) for s in session.states if s.step.stage in FEATURE_STAGES]
    assert states == [
        {1: "pending", 2: "pending", 3: "pending"},
        {1: "current", 2: "pending", 3: "pending"}, {1: "current", 2: "pending", 3: "pending"},
        {1: "guessed", 2: "current", 3: "pending"}, {1: "guessed", 2: "current", 3: "pending"},      # 1 waits for the build
        {1: "guessed", 2: "guessed", 3: "current"}, {1: "guessed", 2: "guessed", 3: "rejected"},
        {1: "accepted", 2: "guessed", 3: "rejected"}, {1: "accepted", 2: "accepted", 3: "rejected"},
        {1: "accepted", 2: "accepted", 3: "rejected"}]
    # a verdict has the timeline number of its piece, however late it comes
    numbers = [(s.step.stage, s.step.iteration) for s in session.states if s.step.iteration is not None]
    assert numbers == [("feature_piece", 0), ("feature_candidate", 0), ("feature_piece", 1), ("feature_candidate", 1),
                       ("feature_piece", 2), ("feature_rejected", 2), ("feature_accepted", 0), ("feature_accepted", 1)]
    from reco_gui.viewer_cascade import layer_meshes
    waiting = next(s for s in session.states if s.feature is not None
                   and piece_states(s) == {1: "guessed", 2: "guessed", 3: "rejected"})
    assert {"f_guessed", "f_rejected"} <= set(layer_meshes(waiting)) and len(layer_meshes(waiting)["f_guessed"].triangles) == 24
    assert feature_progress(waiting) == "feature phase: piece 3 of 3, iteration 1, validating 2 guess(es)"
    assert "2 guessed" in render_state(waiting)
    assert feature_progress(session.states[3]) == "feature phase: piece 1 of 3, iteration 1"
    done = next(s for s in session.states if s.step.stage == "feature_done")
    assert done.feature.history[-1]["applied"] == 2 and "batch</td><td>2" in render_state(session.states[-3])


def test_operations_tree_groups_a_late_verdict_with_its_piece(qapp):
    from reco_gui.op_tree import OperationTree
    session = run_all(make_cascade_session(PART, steps_fn=batch_steps))
    tree = OperationTree()
    for st in session.states:
        tree.add_step(st)
    tree.show_model(session.states[-1])
    history = tree.history_root
    groups = {history.child(k).text(0): history.child(k) for k in range(history.childCount())}
    first = groups["feature piece 1 (iteration 1)"]
    assert [first.child(k).text(0).split()[1] for k in range(first.childCount())] == [
        "feature_piece", "feature_candidate", "feature_accepted"]
    assert first.text(1) == "\u2713 cut: sphere (fit) \u2014 feature (fit)"
    assert groups["feature piece 3 (iteration 1)"].text(1) == "\u2717 rejected: no candidate"

    def leaves(item):
        return [x for k in range(item.childCount()) for x in (leaves(item.child(k)) or [item.child(k)])]
    names = [x.text(0) for x in leaves(tree.model_root)]
    assert names == ["1. add: box", "2. cut: sphere  \u2014 feature (fit), piece 1",
                     "3. cut: cylinder  \u2014 feature (step faces), piece 2"]


def test_breakpoints_on_the_new_stages_stop_the_run():
    session = make_cascade_session(PART, steps_fn=fake_steps)
    assert session.run(breakpoints={"feature_residual"}) == "breakpoint" and session.last.step.stage == "feature_residual"
    assert session.run(breakpoints={"feature_rejected", "feature_done"}) == "breakpoint"
    assert session.last.step.stage == "feature_rejected" and piece_states(session.last)[2] == "rejected"
    assert session.run(breakpoints={"feature_piece"}) == "breakpoint" and session.last.feature.current == 3
    assert session.run(breakpoints={"feature_candidate"}) == "breakpoint" and session.last.candidate is BOSS
    assert session.run(breakpoints={"feature_accepted"}) == "breakpoint" and len(session.last.program) == 3
    assert session.run(breakpoints={"feature_done"}) == "breakpoint" and not session.finished
    assert session.run(breakpoints={"feature_done"}) == "finished"


def test_progress_line_names_piece_and_iteration(session):
    lines = [feature_progress(s) for s in session.states]
    assert lines[:3] == ["", "", ""] and lines[-3:] == ["", "", ""]                       # rough loop; done, grade, done
    assert lines[3] == "feature phase: 5 piece(s), iteration 1"
    assert lines[4] == lines[5] == "feature phase: piece 1 of 5, iteration 1"
    assert lines[14] == "feature phase: piece 4 of 5, iteration 1"
    assert lines[17] == "feature phase: piece 1 of 1, iteration 2"


# ------------------------------------------------------------------------------------------------ settings

def test_feature_settings_reach_the_config_under_the_names_it_has():
    settings = dict(feature_phase=True, max_pieces=40, max_iterations=3, time_budget_s=None, min_score=0.5)
    kw, missing = feature_config(settings, None)                                 # a plain dict (tests): the first names
    assert kw == dict(feature_phase=True, feature_max_pieces=40, feature_max_iterations=3, feature_min_score=0.5)
    assert missing == ()
    kw, missing = feature_config(settings, {"top_k", "feature_phase", "feature_pieces", "min_feature_score"})
    assert kw == dict(feature_phase=True, feature_pieces=40, min_feature_score=0.5) and missing == ("max_iterations",)
    kw, missing = feature_config(settings, {"top_k"})                            # a loop without the feature phase
    assert kw == {} and missing == ("feature_phase", "max_pieces", "max_iterations", "min_score")
    assert feature_config(dict(feature_phase=False), {"feature_phase"}) == ({"feature_phase": False}, ())


def test_the_real_config_takes_what_the_gui_passes():
    """With the loop as it is in this tree: every keyword the GUI passes is a field of the ResidualConfig."""
    from dataclasses import fields
    from reco_engine.cascade_node_support import residual_config
    from reco_engine.residual_loop import ResidualConfig
    from reco_gui.cascade_session import config_defaults, config_fields
    known = config_fields()
    assert known == {f.name for f in fields(ResidualConfig)}
    kw, missing = feature_config(dict(feature_phase=True, max_pieces=7, max_iterations=2, time_budget_s=60.0,
                                      min_score=0.4), known)
    assert set(kw) <= known and set(config_defaults()) <= {"feature_phase", "max_pieces", "max_total_pieces",
                                                          "max_iterations", "time_budget_s", "min_score",
                                                          "rough_subparts", "feature_subparts", "apply_at_once",
                                                          "max_applied"}
    config = residual_config(top_k=1, **kw)
    assert all(getattr(config, name) == value for name, value in kw.items())


def test_make_session_passes_the_feature_settings(monkeypatch):
    import reco_gui.cascade_session as cs
    seen = {}
    monkeypatch.setattr(cs, "make_cascade_session", lambda path, **kw: seen.update(kw) or "session")
    make_session(RunSettings(input_stl=PART, method="cascade"))
    assert seen["feature_phase"] is True and seen["feature_max_pieces"] is None and seen["feature_min_score"] is None
    make_session(RunSettings(input_stl=PART, method="cascade", cascade_feature_phase=False, cascade_feature_max_pieces=12,
                             cascade_feature_max_iterations=4, cascade_feature_time_budget_s=90.0,
                             cascade_feature_min_score=0.7))
    assert seen["feature_phase"] is False and seen["feature_max_pieces"] == 12 and seen["feature_max_iterations"] == 4
    assert seen["feature_time_budget_s"] == 90.0 and seen["feature_min_score"] == 0.7
    configs = []

    def steps(path, config):
        configs.append(config)
        return iter(())
    make_cascade_session(PART, steps_fn=steps, feature_phase=True, feature_max_pieces=9)
    assert configs[0]["feature_phase"] is True and configs[0]["feature_max_pieces"] == 9
    assert "feature_min_score" not in configs[0]


# ------------------------------------------------------------------------------------------------ the window

pytest.importorskip("PyQt5")


@pytest.fixture(scope="module")
def qapp():
    from PyQt5 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_for(app, condition, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


def fake_make_session(settings, models=None):
    assert settings.method == "cascade" and settings.cascade_feature_phase
    return make_cascade_session(settings.input_stl, top_k=settings.cascade_top_k, steps_fn=fake_steps,
                                feature_phase=settings.cascade_feature_phase), None, None


def test_panel_has_the_feature_phase_group(qapp):
    from reco_gui.settings_panel import SettingsPanel
    panel = SettingsPanel()
    panel.show()
    panel.method.setCurrentIndex(panel.method.findData("cascade"))
    assert not panel.feature_box.isVisible()
    panel.more.setChecked(True)
    assert panel.feature_box.isVisible() and panel.feature_box.title() == "Feature phase"
    assert panel.c_features.isChecked()                                                   # on by default in the GUI
    panel.set_input(str(PART))
    for w in panel._feature_limits:
        w.setValue(0)
    s = panel.collect()
    assert s.cascade_feature_phase and s.cascade_feature_max_pieces is None and s.cascade_feature_max_iterations is None
    assert s.cascade_feature_time_budget_s is None and s.cascade_feature_min_score is None
    panel.c_feature_pieces.setValue(40)
    panel.c_feature_iterations.setValue(3)
    panel.c_feature_seconds.setValue(300)
    panel.c_feature_score.setValue(0.6)
    s = panel.collect()
    assert (s.cascade_feature_max_pieces, s.cascade_feature_max_iterations) == (40, 3)
    assert s.cascade_feature_time_budget_s == 300.0 and s.cascade_feature_min_score == pytest.approx(0.6)
    panel.c_features.setChecked(False)
    assert not panel.c_feature_pieces.isEnabled() and not panel.collect().cascade_feature_phase
    panel.method.setCurrentIndex(panel.method.findData("field"))
    assert not panel.feature_box.isVisible()
    panel.close()


def test_window_runs_the_feature_phase(qapp, tmp_path, monkeypatch):
    """The window on the fake stream: the new stages are breakpoints and stop the run; one timeline row and one log line
    per step; the detail panel, the operations tree (the applied feature operations in the Model with their source, the
    rejected guesses in the History group of their piece) and the view's state follow."""
    from reco_gui.app import MainWindow
    from reco_gui.viewer_cascade import layer_meshes
    monkeypatch.setenv("SHAPE_RECO_TIMINGS_DIR", str(tmp_path))
    w = MainWindow(make_session=fake_make_session, stl=str(PART))
    w.settings_panel.c_alternatives.setValue(0)   # one run: no further paths after it (2026-10-29)
    try:
        panel = w.settings_panel
        panel.method.setCurrentIndex(panel.method.findData("cascade"))
        assert set(FEATURE_STAGES) <= set(w.break_actions)
        for stage, a in w.break_actions.items():
            a.setChecked(stage in ("feature_rejected", "feature_done"))
        w.pace.setValue(0.0)
        w.start()
        assert wait_for(qapp, lambda: w.session is not None and not w._busy)
        assert "feature phase (max pieces" in w.log.toPlainText()
        w.cont()
        assert wait_for(qapp, lambda: not w._busy and w.session.last.step.stage == "feature_rejected")
        assert "Stopped at a breakpoint" in w.status_label.text() and "(feature phase, feature_rejected)" in w.status_label.text()
        assert w.timeline.rowCount() == KINDS.index("feature_rejected") + 1
        text = w.detail.toPlainText()
        assert "the build split the model into 2 solids" in text and "0.633" in text and "INVALID" in text
        assert w.view_cascade._state is w.session.last and "f_candidate" in layer_meshes(w.view_cascade._state)
        assert w.view_cascade.parts.count() == 2                                     # the rough box and the first feature cut

        w.break_actions["feature_rejected"].setChecked(False)
        w.cont()
        assert wait_for(qapp, lambda: not w._busy and w.session.last.step.stage == "feature_done")
        assert not w.session.finished and "max_iterations" in w.detail.toPlainText()
        w.break_actions["feature_done"].setChecked(False)
        w.cont()
        assert wait_for(qapp, lambda: w.session.finished and not w._busy)

        assert w.timeline.rowCount() == len(KINDS)
        for row, kind in enumerate(KINDS):
            assert w.timeline.item(row, 2).text() == kind
            assert w.timeline.item(row, 4).text() == w.session.states[row].step.message
        log = w.log.toPlainText()
        for st in w.session.states:                        # every step's message is in the output, the candidates summed up
            assert (st.step.message in log) == (st.step.stage != "feature_candidate")
        assert "    [9-10] feature 2 candidates: best cut cylinder from faces score 0.633" in log
        assert "    [6] feature 1 candidate: best cut sphere from void_fit score 0.912" in log
        assert "feature feature_accepted piece 4: accepted add:box (fit, score 0.98)" in log
        assert "↳ reason: the build split the model into 2 solids" in log and "↳ limits: max_pieces 5" in log
        assert "↳ piece   1  invented -> cut" in log

        model = w.op_tree.model_root
        assert model.text(1) == "3 operation(s): 2 add, 1 cut"

        def leaves(item):
            out = []
            for k in range(item.childCount()):
                child = item.child(k)
                out += leaves(child) if child.childCount() else [child]
            return out
        names = [x.text(0) for x in leaves(model) if x.text(0)[0].isdigit()]
        assert names == ["1. add: box", "2. cut: sphere  — feature piece 1, void_fit",
                         "3. add: box  — feature piece 4, fit"]
        cut = next(x for x in leaves(model) if x.text(0).startswith("2."))
        assert "score 0.912" in cut.toolTip(0)
        assert w.session.states[cut.data(0, 256)].step.stage == "feature_accepted"
        history = w.op_tree.history_root
        groups = {history.child(k).text(0): history.child(k) for k in range(history.childCount())}
        pieces = [name for name in groups if name.startswith("feature piece")]
        assert pieces == ["feature piece 1 (iteration 1)", "feature piece 2 (iteration 1)", "feature piece 3 (iteration 1)",
                          "feature piece 4 (iteration 1)", "feature piece 1 (iteration 2)"]
        assert [groups[name].text(1)[0] for name in pieces] == ["✓", "✗", "✗", "✓", "✗"]
        second = groups["feature piece 2 (iteration 1)"]
        assert [second.child(k).text(0).split()[1] for k in range(second.childCount())] == [
            "feature_piece", "feature_candidate", "feature_candidate", "feature_rejected"]
        assert "the build split the model into 2 solids" in second.text(1)
        assert groups["feature piece 1 (iteration 1)"].text(1) == "✓ cut: sphere (void_fit)"
        n_items = sum(history.child(k).childCount() or 1 for k in range(history.childCount()))
        assert n_items == len(w.session.states)

        w.timeline.selectRow(KINDS.index("feature_candidate"))                       # look back at the first guess
        qapp.processEvents()
        assert "void_fit" in w.detail.toPlainText() and "0.912" in w.detail.toPlainText()
        assert w.view_cascade._state.step.stage == "feature_candidate"
    finally:
        w.close()


def test_heartbeat_names_the_feature_phase(qapp, session, capfd):
    from reco_engine.timing import Recorder
    from reco_gui.app import MainWindow
    w = MainWindow(make_session=fake_make_session, stl=str(PART), verbose=True)
    w.settings_panel.c_alternatives.setValue(0)   # one run: no further paths after it (2026-10-29)
    try:
        rec = Recorder()
        at = KINDS.index("feature_piece") + 3                                        # piece 2 of 5 is the last folded state
        assert session.states[at].step.stage == "feature_piece"
        w.session = SimpleNamespace(recorder=rec, finished=False, last=session.states[at], states=session.states[:at + 1])
        w._set_status("Computing ...")
        w._on_busy(True)
        w._beat_logged -= w.VERBOSE_EVERY_S
        with rec.section("feature"), rec.section("guess"):
            w._on_heartbeat()
        text = w.status_label.text()
        assert "in feature/guess (0 s)" in text and text.endswith("feature phase: piece 2 of 5, iteration 1]")
        assert "feature phase: piece 2 of 5, iteration 1" in w.log.toPlainText().splitlines()[-1]
        assert "feature phase: piece 2 of 5, iteration 1" in capfd.readouterr().out   # verbose: on the terminal too
        w.session = SimpleNamespace(recorder=rec, finished=False, last=session.states[1])
        assert "feature phase" not in w.heartbeat_text()                             # the rough loop
        w.session = SimpleNamespace(recorder=rec, finished=False, last=session.states[-2])
        assert "feature phase" not in w.heartbeat_text()                             # after the phase
        w._on_busy(False)
    finally:
        w.session = None
        w.close()


def test_command_line_switch(monkeypatch):
    """--features / --no-features set the panel's checkbox (the parser only; no window)."""
    import argparse
    import reco_gui.__main__ as main
    seen = {}
    real = argparse.ArgumentParser.parse_args

    def parse(self, args=None, namespace=None):
        seen["args"] = real(self, args, namespace)
        raise SystemExit(0)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    for argv, expected in ((["--no-features"], False), (["--features"], True), ([], None)):
        monkeypatch.setattr(sys, "argv", ["reco_gui"] + argv)
        with pytest.raises(SystemExit):
            main.main()
        assert seen["args"].features is expected


# ------------------------------------------------------------------------------------------------ Close

def blocking_make_session(block):
    """A session whose second step blocks in `block()` - a step that computes for long."""
    def steps(path, config):
        yield FakeStep("input", ReconstructionResult(input_path=str(path)), {"spec": SPEC})
        block()
        yield FakeStep("done", ReconstructionResult(input_path=str(path)), {})

    def make(settings, models=None):
        return make_cascade_session(settings.input_stl, steps_fn=steps), None, None
    return make


def started_window(qapp, make):
    from reco_gui.app import MainWindow
    w = MainWindow(make_session=make, stl=str(PART))
    w.settings_panel.c_alternatives.setValue(0)   # one run: no further paths after it (2026-10-29)
    w.hard_exit = lambda code: w.__dict__.setdefault("exited", code)
    w.settings_panel.method.setCurrentIndex(w.settings_panel.method.findData("cascade"))
    w.pace.setValue(0.0)
    w.show()
    w.start()
    assert wait_for(qapp, lambda: w.session is not None and not w._busy)
    return w


def test_close_action_closes_the_idle_window(qapp):
    from PyQt5 import QtGui
    w = started_window(qapp, fake_make_session)
    assert w.act_close.text() == "Close" and w.act_close.shortcut() == QtGui.QKeySequence("Ctrl+Q")
    assert not w.act_close.icon().isNull() and w.act_close.isEnabled() and w.isVisible()
    file_menu = w.menuBar().actions()[0].menu()                                           # the File menu (2026-10-29: the
    assert file_menu.actions()[-1] is w.act_close                                          # toolbar became the flow + run bars)
    w.heartbeat.start()
    w.act_close.trigger()
    assert not w.isVisible() and not w.heartbeat.isActive() and w.thread.isFinished() and "exited" not in w.__dict__


def test_close_while_a_child_process_computes(qapp, capfd):
    """The worker waits for a child process that runs for 60 s (a FreeCAD build): Close ends the child, the worker returns,
    the thread stops - nothing is left running and the process is not ended by force."""
    import subprocess
    from reco_gui.app import child_processes
    w = started_window(qapp, blocking_make_session(
        lambda: subprocess.run([sys.executable, "-c", "import time; time.sleep(60)"])))
    w.CLOSE_WAIT_S, w.CLOSE_KILL_WAIT_S = 0.3, 5.0
    before = {pid for pid, _, _ in child_processes()}
    w.cont()
    assert wait_for(qapp, lambda: w._busy and len({pid for pid, _, _ in child_processes()} - before) == 1)
    assert w.act_close.isEnabled() and w.heartbeat.isActive()                             # enabled while busy
    t0 = time.time()
    w.act_close.trigger()
    assert time.time() - t0 < 5.0
    assert not w.isVisible() and not w.heartbeat.isActive() and w.thread.isFinished() and "exited" not in w.__dict__
    assert {pid for pid, _, _ in child_processes()} <= before                            # the child is gone
    err = capfd.readouterr().err
    assert "closing: the worker is still computing" in err and "ended the child processes: python" in err


def test_close_while_the_worker_itself_computes(qapp, capfd):
    """No child to end and the worker does not return: the process is ended (here: the stand-in is called)."""
    import threading
    release = threading.Event()
    w = started_window(qapp, blocking_make_session(lambda: release.wait(30)))
    w.CLOSE_WAIT_S, w.CLOSE_KILL_WAIT_S = 0.2, 0.2
    try:
        w.cont()
        assert wait_for(qapp, lambda: w._busy and len(w.session.states) == 1)
        w.act_close.trigger()
        assert not w.isVisible() and not w.heartbeat.isActive() and w.__dict__.get("exited") == 0
        assert "the worker thread does not return: ending the process" in capfd.readouterr().err
    finally:
        release.set()
        assert w.thread.wait(5000)                                                        # the pause and the quit took effect


def test_dataset_generation_window_has_a_close_action(qapp):
    from reco_gui.dataset_gen_window import DatasetGenWindow
    w = DatasetGenWindow()
    w.show()
    assert w.act_close.text() == "Close" and w.act_close.isEnabled() and w.isVisible()
    w.act_close.trigger()
    assert not w.isVisible()


# ------------------------------------------------------------------------------------------------ Save, the run store

def test_save_action_and_auto_save_at_the_end(qapp, tmp_path, monkeypatch):
    """Save is enabled once the session has a state and the worker is idle; triggering it writes a run directory (a run
    that has not finished says so); 'save every finished run' stores the run at its end; Compare shows the diff against
    the earlier run of the same input in the detail panel."""
    from PyQt5 import QtGui, QtWidgets
    from reco_engine import run_store
    store = tmp_path / "store"
    monkeypatch.setenv("SHAPE_RECO_RESULTS", str(store))
    monkeypatch.setenv("SHAPE_RECO_TIMINGS_DIR", str(tmp_path / "timings"))
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("rough only", True))
    w = started_window(qapp, fake_make_session)
    try:
        assert w.settings_panel.save_runs.isChecked() and w.settings.save_finished_run            # on by default
        assert w.act_save.text() == "Save run ..." and w.act_save.shortcut() == QtGui.QKeySequence("Ctrl+S")
        assert not w.act_save.icon().isNull()
        actions = w.act_save.associatedWidgets()[0].actions()
        assert actions.index(w.act_save) < actions.index(w.act_close)
        assert not w.act_save.isEnabled() and not w.act_compare_runs.isEnabled()                 # no state yet
        for stage, a in w.break_actions.items():
            a.setChecked(stage == "feature_residual")
        w.cont()
        assert wait_for(qapp, lambda: not w._busy and w.session.last.step.stage == "feature_residual")
        assert w.act_save.isEnabled()
        w.act_save.trigger()                                                                    # a partial run
        first = w.saved_run
        assert first is not None and first.parent == store / PART.stem and first.name.endswith("_rough-only")
        run = run_store.load_run(first)
        assert run["finished"] is False and run["steps"] == 4 and run["metrics"]["n_operations"] == 1
        assert run["metrics"]["chamfer_mm"] is None and run["metrics"]["stopped"] is None and run["view"] is None
        assert run["settings"]["gui"]["cascade_feature_phase"] is True and run["settings"]["config"]["feature_phase"] is True
        assert f"run saved: {first}" in w.log.toPlainText() and str(first) in w.status_label.text()
        assert w.act_compare_runs.isEnabled()
        w.compare_with_previous()
        assert "no earlier run" in w.detail.toPlainText()

        w.follow_box.setChecked(True)
        for a in w.break_actions.values():
            a.setChecked(False)
        w.cont()
        assert wait_for(qapp, lambda: w.session.finished and not w._busy and w.saved_run != first)     # saved by itself
        runs = run_store.list_runs(PART.stem)
        assert [r["path"] for r in runs] == [str(w.saved_run), str(first)] and runs[0]["finished"] is True
        done = runs[0]
        assert done["label"] is None and done["steps"] == len(KINDS) and done["metrics"]["n_operations"] == 3
        assert done["metrics"]["feature_phase"]["stopped"] == "max_iterations" and done["metrics"]["exact_iou"] == 0.9731
        assert done["metrics"]["chamfer_mm"] == 0.031 and done["metrics"]["voxel_iou"] is None
        assert done["metrics"]["operations_by_kind"] == {"add:box": 2, "cut:sphere": 1}
        assert done["timings"]["wall"] > 0 and any(t["path"] == "gui fold" for t in done["timings"]["sections"])
        lines = (w.saved_run / "steps.jsonl").read_text().splitlines()
        assert len(lines) == len(KINDS) and '"stage": "feature_accepted"' in lines[KINDS.index("feature_accepted")]
        assert "Saved:" in w.status_label.text() and "Run finished" in w.status_label.text()
        w.act_compare_runs.trigger()
        text = w.detail.toPlainText()
        assert "Compare with previous" in text and first.name in text and w.saved_run.name in text
        assert "n_operations" in text and "only in b" in text and "the same file" in text and w.detail.isVisible()
        assert not w.act_save.isEnabled() or w.act_save.isEnabled() == (not w._busy)
        auto = w.saved_run
        assert w.save_run() == auto and "saved already" in w.log.toPlainText()                  # by hand, after the end:
        labelled = w.save_run("again")                                                          # a label, not a copy
        assert labelled == w.saved_run and labelled.name == auto.name + "_again" and not auto.exists()
        runs = run_store.list_runs(PART.stem)
        assert len(runs) == 2 and runs[0]["label"] == "again" and runs[0]["steps"] == len(KINDS)
    finally:
        w.close()


def test_auto_save_can_be_switched_off_and_tests_do_not_write_into_the_users_store(qapp, tmp_path, monkeypatch):
    from reco_engine import run_store
    monkeypatch.setenv("SHAPE_RECO_TIMINGS_DIR", str(tmp_path / "timings"))
    monkeypatch.setenv("SHAPE_RECO_RESULTS", str(tmp_path / "store"))
    from reco_gui.app import MainWindow
    w = MainWindow(make_session=fake_make_session, stl=str(PART))
    w.settings_panel.c_alternatives.setValue(0)   # one run: no further paths after it (2026-10-29)
    try:
        w.settings_panel.method.setCurrentIndex(w.settings_panel.method.findData("cascade"))
        w.settings_panel.save_runs.setChecked(False)
        w.pace.setValue(0.0)
        w.start()
        assert wait_for(qapp, lambda: w.session is not None and not w._busy)
        for a in w.break_actions.values():
            a.setChecked(False)
        w.cont()
        assert wait_for(qapp, lambda: w.session.finished and not w._busy)
        assert w.saved_run is None and run_store.list_runs() == []
        w.settings.save_finished_run = True
        assert w.autosave is None and w._auto_save_wanted()            # left open: a store is named, so yes
        monkeypatch.delenv("SHAPE_RECO_RESULTS")                       # ... and without a named store: no
        assert not w._auto_save_wanted()
        w.autosave = True                                              # `python3 -m reco_gui`: the default store
        assert w._auto_save_wanted()
        w.settings.save_finished_run = False
        assert not w._auto_save_wanted()
        monkeypatch.setenv("SHAPE_RECO_RESULTS", str(tmp_path / "store"))
        w.settings.save_finished_run, w.autosave = True, False        # MainWindow(autosave=False): never
        assert not w._auto_save_wanted()
    finally:
        w.close()


# ------------------------------------------------------------------------------------------------ pace and log

REASONS = ["failed the void check (purity 0.41 < 0.6)", "failed the void check (purity 0.12 < 0.6)",
           "only 4% of the nearby surface on it (a new boolean needs 20%)",
           "only 11% of the nearby surface on it (a new boolean needs 20%)",
           "only 0% of the nearby surface on it (a new boolean needs 20%)", "sign check"]


def candidate_steps(path, config):
    """A component with eight candidates as the rough loop yields them (ok, reason, gain, surface): two pass."""
    ops = [Operation("add", BOX)]

    def res():
        return ReconstructionResult(input_path=str(path), operations=list(ops))
    yield FakeStep("input", res(), {"spec": SPEC})
    yield FakeStep("add", res(), {"candidate": BOX, "op": "add", "net_gain": 4000.0})
    yield FakeStep("component", res(), {"mask": np.ones((16,) * 3, bool), "op": "cut"})
    for k, reason in enumerate(REASONS):
        yield FakeStep("candidate", res(), {"primitive": WRONG, "op": "cut", "gain": 10 + k, "ok": False, "reason": reason,
                                            "surface": 0.04, "message": f"cut cylinder: gain {10 + k}, {reason}"})
    yield FakeStep("candidate", res(), {"primitive": HOLE, "op": "cut", "gain": 250, "ok": True, "reason": "",
                                        "surface": 0.3, "message": "cut cylinder: gain 250"})
    yield FakeStep("candidate", res(), {"primitive": POCKET, "op": "cut", "gain": 39321, "ok": True, "reason": "",
                                        "surface": 0.47, "message": "cut sphere: gain 39321"})
    ops.append(Operation("cut", POCKET, net_gain=39321))
    yield FakeStep("accepted", res(), {"candidate": POCKET, "op": "cut", "net_gain": 39321.0})
    yield FakeStep("done", res(), {})


def test_the_pace_is_not_paid_on_candidates():
    from reco_gui.cascade_session import CASCADE_PACED_STAGES, UNPACED_STAGES
    assert set(UNPACED_STAGES) == {"candidate", "feature_candidate"}
    assert set(CASCADE_PACED_STAGES) == set(CASCADE_STAGES) - set(UNPACED_STAGES)
    session = make_cascade_session(PART, steps_fn=candidate_steps)
    assert session.paced == CASCADE_PACED_STAGES
    session.advance()                                                   # the input (loading the part) is not timed
    t0 = time.time()
    assert session.run(delay=0.2, paced=session.paced) == "finished"
    paced = time.time() - t0
    kinds = [s.step.stage for s in session.states[1:]]
    assert len(kinds) == 12 and kinds.count("candidate") == 8
    assert 3 * 0.2 <= paced < 3 * 0.2 + 0.8                             # add, component, accepted (done ends the run)
    everything = make_cascade_session(PART, steps_fn=candidate_steps)
    everything.advance()
    t0 = time.time()
    assert everything.run(delay=0.2) == "finished"                      # paced None: after every step, as before
    assert time.time() - t0 >= 11 * 0.2
    # a candidate is still a breakpoint
    again = make_cascade_session(PART, steps_fn=candidate_steps)
    assert again.run(breakpoints={"candidate"}, delay=0.0, paced=again.paced) == "breakpoint"
    assert again.last.step.stage == "candidate"


def test_candidates_summary_counts_the_reasons():
    from reco_gui.report import candidates_summary
    session = run_all(make_cascade_session(PART, steps_fn=candidate_steps))
    candidates = [s for s in session.states if s.step.stage == "candidate"]
    assert candidates_summary(candidates) == (
        "8 candidates: best cut sphere gain 39321 (47% of the nearby surface); 2 pass; "
        "3 x only #% of the nearby surface on it (a new boolean needs #%), 2 x failed the void check (purity # < #), "
        "1 x sign check")
    assert candidates_summary(candidates[:2]) == "2 candidates: none passes; 2 x failed the void check (purity # < #)"
    assert candidates_summary([]) == ""


def test_output_has_one_line_per_component_unless_every_candidate_is_asked_for(qapp, tmp_path, monkeypatch):
    from reco_gui.app import MainWindow
    monkeypatch.setenv("SHAPE_RECO_TIMINGS_DIR", str(tmp_path))

    def make(settings, models=None):
        return make_cascade_session(settings.input_stl, steps_fn=candidate_steps), None, None

    def run(verbose, stop_at=None):
        w = MainWindow(make_session=make, stl=str(PART), verbose=verbose)
        w.settings_panel.c_alternatives.setValue(0)   # one run: no further paths after it (2026-10-29)
        w.settings_panel.method.setCurrentIndex(w.settings_panel.method.findData("cascade"))
        w.pace.setValue(0.0)
        w.start()
        assert wait_for(qapp, lambda: w.session is not None and not w._busy)
        for stage, a in w.break_actions.items():
            a.setChecked(stage == stop_at)
        w.cont()
        assert wait_for(qapp, lambda: not w._busy and (w.session.finished or stop_at is not None))
        return w
    w = run(False)
    try:
        assert not w.log_candidates.isChecked() and w.log_candidates.text() == "log every candidate"
        assert w.output_dock.widget().isAncestorOf(w.log_candidates) and w.output_dock.widget().isAncestorOf(w.log)
        lines = w.log.toPlainText().splitlines()
        summary = [x for x in lines if "candidates:" in x]
        assert len(summary) == 1 and summary[0].startswith("    [4-11] cut 8 candidates: best cut sphere gain 39321")
        assert "3 x only #% of the nearby surface" in summary[0]
        assert not any("cut cylinder: gain" in x for x in lines)
        assert lines.index(summary[0]) + 1 == next(i for i, x in enumerate(lines) if " accepted " in x)   # then the verdict
        assert w.timeline.rowCount() == 13                              # the timeline keeps a row per step
    finally:
        w.close()
    w = run(False, stop_at="candidate")                                 # stopped at a candidate: it is said at once
    try:
        assert w.session.last.step.stage == "candidate" and "    [4] cut 1 candidate: none passes" in w.log.toPlainText()
    finally:
        w.close()
    w = run(True)                                                       # --verbose: a line per candidate
    try:
        assert w.log_candidates.isChecked()
        lines = w.log.toPlainText().splitlines()
        assert len([x for x in lines if "cut cylinder: gain" in x]) == 7 and not any("candidates:" in x for x in lines)
    finally:
        w.close()


def test_heartbeat_counts_own_threads_and_running_children_apart(qapp):
    import subprocess
    from reco_gui.app import MainWindow, children_cpu
    w = MainWindow(make_session=fake_make_session, stl=str(PART))
    w.settings_panel.c_alternatives.setValue(0)   # one run: no further paths after it (2026-10-29)
    child = subprocess.Popen([sys.executable, "-c", "while True: pass"])
    try:
        w.session = SimpleNamespace(recorder=None, finished=False, last=None)
        w._on_busy(True)
        assert "children" in w.heartbeat_text()
        time.sleep(0.6)
        assert child.pid in children_cpu() and children_cpu()[child.pid][0] > 0.2
        text = w.heartbeat_text()
        rate = float(text.split("children ")[1].split(",")[0])
        assert 0.5 <= rate <= 1.3                                       # one busy child: about one core
        child.kill()
        child.wait()                                                    # its CPU time arrives in os.times().children_*
        time.sleep(0.2)
        text = w.heartbeat_text()
        own = float(text.split("now ")[1].split(" cores")[0])
        assert own < 2.0 and "children" not in text                     # no jump, and no running child left
        w._on_busy(False)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        w.session = None
        w.close()


# ------------------------------------------------------------------------------------------------ split pieces

SEGMENTS = {11: (24, 96.0, ((-6, -6, 4), (-2, 6, 10))), 12: (30, 120.0, ((-2, -6, 4), (2, 6, 10))),
            13: (20, 80.0, ((2, -6, 4), (6, 6, 10)))}                      # id -> (voxels, volume, box): piece 1 in three


def segment(sid):
    voxels, volume, box = SEGMENTS[sid]
    return dict(id=sid, parent=1, voxels=voxels, volume=volume, share=volume / TOTAL, bbox=box)


def split_steps(path, config):
    """Piece 1 has no guess that scores and is split by its necks into three segments: 11 accepted, 12 rejected, 13
    accepted; the piece itself then ends accepted without a primitive. Piece 2 is split too and none of its two
    segments passes: the piece ends rejected."""
    ops = [Operation("add", BOX)]

    def f(kind, **info):
        return FakeStep(kind, ReconstructionResult(input_path=str(path), operations=list(ops)), dict(info, phase="feature"))
    yield FakeStep("input", ReconstructionResult(input_path=str(path)), {"spec": SPEC})
    yield f("feature_residual", iteration=1, missing_volume=0.0, invented_volume=650.9, input_volume=7547.6,
            exact_iou=0.923, pieces=[piece(1), piece(2)], dropped_small=0, dropped_limit=0, limits=dict(LIMITS))
    yield f("feature_piece", iteration=1, piece=piece(1), index=1, of=2, mesh=mesh_of(1), mesh_truncated=False)
    yield f("feature_candidate", iteration=1, piece_id=1, op="cut", primitive=WRONG, source="fit", score=0.31,
            precision=0.31, recall=0.9, seconds=0.2)
    yield f("feature_split", iteration=1, piece_id=1, method="necks", pitch=0.5, grid=(24, 24, 12),
            segments=[segment(i) for i in SEGMENTS], seconds=0.4,
            message="piece #1 split by necks into 3 segments (pitch 0.5 mm, 24 x 24 x 12 cells)")
    boxes = {i: ("invented", v[1], v[2]) for i, v in SEGMENTS.items()}
    guesses = {11: (POCKET, 0.95, 0.91), 12: (WRONG, 0.4, 0.8), 13: (HOLE, 0.88, 0.97)}
    for k, sid in enumerate(SEGMENTS):
        prim, precision, recall = guesses[sid]
        yield f("feature_piece", iteration=1, piece=segment(sid), index=k + 1, of=3, mesh=mesh_of(sid, boxes),
                mesh_truncated=False)
        yield f("feature_candidate", iteration=1, piece_id=sid, op="cut", primitive=prim, source="voxel fit",
                score=min(precision, recall), precision=precision, recall=recall, seconds=0.1)
        if sid == 12:
            yield f("feature_rejected", iteration=1, piece_id=sid, op="cut", primitive=prim, source="voxel fit", score=0.4,
                    reason="best score 0.400 < 0.5", validation=None)
        else:
            ops.append(Operation("cut", prim, freecad_valid=True))
            yield f("feature_accepted", iteration=1, piece_id=sid, op="cut", primitive=prim, source="voxel fit",
                    score=min(precision, recall), validation=validation(True, 650.9, 500.0, 0.95))
    yield f("feature_accepted", iteration=1, piece_id=1, op="cut", primitive=None, source=None, score=None,
            segments_accepted=[11, 13], validation=None, message="piece #1: 2 of 3 segments accepted")
    yield f("feature_piece", iteration=1, piece=piece(2), index=2, of=2, mesh=mesh_of(2), mesh_truncated=False)
    two = [dict(id=14, parent=2, voxels=9, volume=70.0, share=0.1, bbox=((4.5, -6.5, -10), (7.5, -3.5, 0))),
           dict(id=15, parent=2, voxels=9, volume=85.5, share=0.12, bbox=((4.5, -6.5, 0), (7.5, -3.5, 10)))]
    yield f("feature_split", iteration=1, piece_id=2, method="axis", pitch=1.0, grid=(3, 3, 20), segments=two, seconds=0.1)
    for k, d in enumerate(two):
        yield f("feature_piece", iteration=1, piece=d, index=k + 1, of=2, mesh=None, mesh_truncated=False)
        yield f("feature_rejected", iteration=1, piece_id=d["id"], op="cut", primitive=None, source=None, score=None,
                reason="no candidate", validation=None)
    yield f("feature_rejected", iteration=1, piece_id=2, op="cut", primitive=None, source=None, score=None,
            reason="split: 0 of 2 segments accepted", validation=None)
    yield f("feature_done", iterations=1, applied=2, rejected=3, skipped=0, stopped="nothing applied", seconds=9.0,
            exact_iou=0.95, missing_volume=0.0, invented_volume=500.0, split_pieces=2, segments=5, segments_accepted=2)
    yield FakeStep("done", ReconstructionResult(input_path=str(path), operations=list(ops)), {})


@pytest.fixture(scope="module")
def split_session():
    return run_all(make_cascade_session(PART, steps_fn=split_steps))


def test_split_is_a_stage_a_breakpoint_and_paced():
    from reco_gui.cascade_session import CASCADE_PACED_STAGES
    assert "feature_split" in FEATURE_STAGES and "feature_split" in CASCADE_STAGES and "feature_split" in CASCADE_PACED_STAGES
    session = make_cascade_session(PART, steps_fn=split_steps)
    assert session.run(breakpoints={"feature_split"}, paced=session.paced) == "breakpoint"
    assert session.last.step.stage == "feature_split" and session.last.step.phase == "feature"
    assert session.last.step.message.startswith("piece #1 split by necks into 3 segments")


def test_segments_fold_under_their_piece(split_session):
    st = split_session.states
    assert split_session.error is None and [s.step.stage for s in st] == [s.kind for s in split_steps(PART, {})]
    split = states_of(split_session, "feature_split")[0].feature
    assert [(p.id, p.parent, p.state) for p in split.pieces] == [
        (1, None, "split"), (11, 1, "pending"), (12, 1, "pending"), (13, 1, "pending"), (2, None, "pending")]
    one = split.piece(1)
    assert one.split == dict(method="necks", pitch=0.5, grid=(24, 24, 12), segments=(11, 12, 13), seconds=0.4)
    assert [(p.side, p.op, p.voxels) for p in split.segments(1)] == [("invented", "cut", 24), ("invented", "cut", 30),
                                                                     ("invented", "cut", 20)]       # inherited
    cand = states_of(split_session, "feature_candidate")[1]
    seg = cand.feature.piece(11)
    assert seg.state == "current" and seg.precision == 0.95 and seg.recall == 0.91 and seg.score == 0.91
    assert cand.feature.piece(1).state == "split" and cand.candidate is POCKET
    parent_done = next(s for s in st if s.step.stage == "feature_accepted" and s.info.get("piece_id") == 1)
    f = parent_done.feature
    assert {p.id: p.state for p in f.pieces} == {1: "accepted", 11: "accepted", 12: "rejected", 13: "accepted", 2: "pending"}
    assert f.piece(1).segments_accepted == (11, 13) and f.piece(1).primitive is WRONG          # its own last guess
    assert parent_done.candidate is None and parent_done.candidate_mesh is None and parent_done.verdict is None
    assert f.applied == 2 and f.rejected == 1                        # the segments count, the split piece does not
    assert parent_done.step.message == "piece #1: 2 of 3 segments accepted" and len(parent_done.program) == 3
    last = st[-1].feature
    assert {p.id: p.state for p in last.pieces} == {1: "accepted", 11: "accepted", 12: "rejected", 13: "accepted",
                                                    2: "rejected", 14: "rejected", 15: "rejected"}
    assert [p.id for p in last.pieces] == [1, 11, 12, 13, 2, 14, 15]                          # nested order
    assert last.applied == 2 and last.rejected == 3 and last.piece(2).reason == "split: 0 of 2 segments accepted"
    assert last.done["split_pieces"] == 2 and last.done["segments"] == 5 and last.done["segments_accepted"] == 2
    # the timeline: a segment has its own number, the split and the piece's verdict have the piece's
    numbers = [(s.step.stage, s.info.get("piece_id", (s.info.get("piece") or {}).get("id")), s.step.iteration)
               for s in st if s.step.iteration is not None]
    assert [n for _, pid, n in numbers if pid == 1] == [0, 0, 0, 0] and [n for _, pid, n in numbers if pid == 12] == [2, 2, 2]
    progress = [feature_progress(s) for s in st]
    assert progress[5] == "feature phase: piece #1, segment 1 of 3, iteration 1"
    assert progress[2] == "feature phase: piece 1 of 2, iteration 1"
    fold, s = CascadeFolder(wall_grid=False), None                   # without messages, and a split of an unknown piece
    res = ReconstructionResult(input_path=str(PART))
    s = fold(s, FakeStep("feature_split", res, {"piece_id": 7, "method": "axis", "segments": [{"id": 7}, None], "grid": 5}), 0)
    assert [(p.id, p.parent, p.state) for p in s.feature.pieces] == [(7, None, "split"), (8, 7, "pending"), (9, 7, "pending")]
    assert s.step.message == "piece 7 split by axis into 2 segment(s)"
    s = fold(s, FakeStep("feature_accepted", res, {"piece_id": 7, "primitive": None}), 1)
    assert s.feature.piece(7).state == "accepted" and s.step.message == "piece 7: 0 segment(s) accepted"
    render_state(s), timeline_row(s), feature_log_lines(s)


def test_split_piece_is_drawn_as_its_segments(split_session):
    from reco_gui.cascade_session import SEGMENT_OPACITY, SEGMENT_PALETTE, STATE_COLOURS
    from reco_gui.viewer_cascade import COLOURED_LAYERS, FEATURE_LAYERS, layer_meshes
    assert set(COLOURED_LAYERS) - {"distance"} <= set(FEATURE_LAYERS)      # "distance" is the nephogram, a main layer
    before = layer_meshes(states_of(split_session, "feature_piece")[0])
    assert "f_segments" not in before and len(before["f_current"].triangles) == 12              # the whole piece
    cand = next(s for s in split_session.states if s.step.stage == "feature_candidate" and s.info["piece_id"] == 12)
    drawn = layer_meshes(cand)
    assert {"f_segments", "f_outline", "f_candidate", "f_pending"} <= set(drawn) and "f_current" not in drawn
    assert len(drawn["f_pending"].triangles) == 12                                              # piece 2 only
    seg, outline = drawn["f_segments"], drawn["f_outline"]
    assert len(seg.triangles) == 36 and seg.colours.shape == (36, 4) and seg.colours.dtype == np.uint8
    assert outline.colours.shape == (36, 4)
    lo, hi = seg.vertices.min(axis=0), seg.vertices.max(axis=0)
    assert lo == pytest.approx([-6, -6, 4]) and hi == pytest.approx([6, 6, 10])                  # together: piece 1's box
    rgba = lambda rgb, a: [round(255 * c) for c in rgb] + [round(255 * a)]  # noqa: E731
    colours = [seg.colours[k * 12].tolist() for k in range(3)]                                  # 11 accepted, 12 in work, 13
    assert colours == [rgba(SEGMENT_PALETTE[0], SEGMENT_OPACITY["accepted"]), rgba(SEGMENT_PALETTE[1], SEGMENT_OPACITY["current"]),
                       rgba(SEGMENT_PALETTE[2], SEGMENT_OPACITY["pending"])]
    assert len({tuple(c[:3]) for c in colours}) == 3                                             # distinguishable
    assert [outline.colours[k * 12].tolist() for k in range(3)] == [rgba(STATE_COLOURS[k], 1.0)
                                                                    for k in ("accepted", "current", "pending")]
    end = layer_meshes(states_of(split_session, "feature_done")[0])
    assert "f_accepted" not in end and "f_rejected" not in end and len(end["f_segments"].triangles) == 60
    assert not set(layer_meshes(split_session.states[-1])) & set(FEATURE_LAYERS)


def test_split_in_the_detail_panel_and_the_log(split_session):
    from reco_gui.report import candidates_summary
    st = split_session.states
    split = render_state(states_of(split_session, "feature_split")[0])
    assert "by <b>necks</b> into 3 segments" in split and "pitch 0.5 mm" in split and "24 x 24 x 12 cells" in split
    assert "&#8627; <span" in split and split.count("&#8627;") == 3 and "1 split" in split     # nested under piece 1
    cand = render_state(next(s for s in st if s.step.stage == "feature_candidate" and s.info["piece_id"] == 11))
    assert "precision" in cand and "0.950" in cand and "recall" in cand and "0.910" in cand and "voxel fit" in cand
    assert "segment" in cand and "of piece <b>1</b>" in cand and "by <b>necks</b>" in cand and "<td>24</td>" in cand
    parent = next(s for s in st if s.step.stage == "feature_accepted" and s.info.get("piece_id") == 1)
    html_text = render_state(parent)
    assert "segments accepted" in html_text and "11, 13" in html_text and "(2 accepted, 1 rejected)" in html_text
    assert "precision 0.400, recall 0.800" in html_text                                       # segment 12 in the list
    assert feature_log_lines(parent) == ["piece #1 split by necks into 3 segments: 2 accepted, 1 rejected"]
    rejected = next(s for s in st if s.step.stage == "feature_rejected" and s.info.get("piece_id") == 2)
    assert feature_log_lines(rejected) == ["piece #2 split by axis into 2 segments: 0 accepted, 2 rejected",
                                           "reason: split: 0 of 2 segments accepted"]
    lines = feature_log_lines(states_of(split_session, "feature_split")[0])
    assert len(lines) == 3 and lines[0].startswith("segment  11") and "24 voxels" in lines[0]
    done = render_state(states_of(split_session, "feature_done")[0])
    assert "2 piece(s) into 5 segments, 2 accepted" in done
    one = candidates_summary([s for s in st if s.step.stage == "feature_candidate" and s.info["piece_id"] == 13])
    assert one == "1 candidate: best cut cylinder from voxel fit score 0.880 (precision 0.880, recall 0.970)"


def test_operations_tree_nests_the_segments(qapp, split_session):
    from reco_gui.op_tree import OperationTree
    tree = OperationTree()
    for st in split_session.states:
        tree.add_step(st)                                          # the accepted piece without a primitive included
    tree.show_model(split_session.states[-1])
    history = tree.history_root
    groups = [history.child(k) for k in range(history.childCount()) if history.child(k).text(0).startswith("feature")]
    assert [g.text(0) for g in groups] == ["feature piece 1 (iteration 1)", "feature piece 2 (iteration 1)"]
    first = groups[0]
    kids = [first.child(k).text(0) for k in range(first.childCount())]
    assert [k.split()[-1] if k[0].isdigit() else k for k in kids] == [
        "feature_piece", "feature_candidate", "feature_split", "segment 11", "segment 12", "segment 13", "feature_accepted"]
    assert first.text(1) == "\u2713 split: 2 segment(s) accepted"
    segs = {first.child(k).text(0): first.child(k) for k in range(first.childCount())}
    assert segs["segment 11"].text(1) == "\u2713 cut: sphere (voxel fit)" and segs["segment 12"].text(1).startswith("\u2717")
    assert [segs["segment 12"].child(k).text(0).split()[1] for k in range(3)] == [
        "feature_piece", "feature_candidate", "feature_rejected"]
    assert groups[1].text(1) == "\u2717 rejected: split: 0 of 2 segments accepted" and groups[1].childCount() == 5
    assert tree.model_root.text(1) == "3 operation(s): 1 add, 2 cut"


def test_window_logs_the_split(qapp, tmp_path, monkeypatch):
    from reco_gui.app import MainWindow
    monkeypatch.setenv("SHAPE_RECO_TIMINGS_DIR", str(tmp_path))

    def make(settings, models=None):
        return make_cascade_session(settings.input_stl, steps_fn=split_steps), None, None
    w = MainWindow(make_session=make, stl=str(PART), autosave=False)
    w.settings_panel.c_alternatives.setValue(0)   # one run: no further paths after it (2026-10-29)
    try:
        w.settings_panel.method.setCurrentIndex(w.settings_panel.method.findData("cascade"))
        assert "feature_split" in w.break_actions
        w.pace.setValue(0.0)
        w.start()
        assert wait_for(qapp, lambda: w.session is not None and not w._busy)
        for a in w.break_actions.values():
            a.setChecked(False)
        w.cont()
        assert wait_for(qapp, lambda: w.session.finished and not w._busy)
        log = w.log.toPlainText()
        assert "\u21b3 piece #1 split by necks into 3 segments: 2 accepted, 1 rejected" in log
        assert "feature_split piece #1 split by necks into 3 segments (pitch 0.5 mm, 24 x 24 x 12 cells)" in log
        assert "candidate: best cut sphere from voxel fit score 0.910 (precision 0.950, recall 0.910)" in log
        assert w.timeline.rowCount() == len(w.session.states) and w.saved_run is None
    finally:
        w.close()
