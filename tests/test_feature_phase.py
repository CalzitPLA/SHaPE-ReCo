#!/usr/bin/env python3
"""The feature phase of the residual loop (reco_engine/residual_loop.py, ResidualConfig.feature_phase): the program's
exact difference to the input, a primitive per piece, applied and validated. The fast tests run the phase with fakes
for its three outside calls (FreeCAD build, OCCT difference, the guess) - a small "world" of known pieces whose
residual drops when an operation with the piece's centre joins the program - and check the ranking and the caps, the
step contract with the GUI (every kind and key), the revert of an operation that fails validation, each limit, and
that one failing piece does not end the phase. One slow end-to-end test (--run-slow, freecadcmd) runs the whole loop
on a STEP box with two bores, one of them below the voxel pitch."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine import residual_loop as rl  # noqa: E402
from reco_engine.feature_stage import Candidate, extend_ends  # noqa: E402
from reco_engine.result_schema import Operation, Primitive, ReconstructionResult  # noqa: E402

V_IN = 1000.0


class World:
    """Known pieces {name: dict(sign, centre, volume, score, poison, useless, fail, destroys)}. The program explains a piece when
    one of its operations sits at the piece's centre (a guess from `guess` does); a "poison" operation makes the build
    invalid, a "useless" one explains nothing, a "fail" piece makes the guess raise; explaining a piece with "destroys"
    also removes that much of the input (it becomes missing) - only by an operation of source "destroys_mode...", if given."""

    def __init__(self, pieces, stl=None, guess_sleep=0.0):
        self.pieces = pieces
        self.models = {}
        self.builds = 0
        self.differences = 0
        self.stl = stl
        self.guess_sleep = guess_sleep
        self.guessed = []
        self.piece_differences = 0

    @staticmethod
    def _at(ops, centre):
        return any(np.linalg.norm(np.subtract(o.primitive.center_mm, centre)) < 1e-3 for o in ops)

    def _explained(self, result):
        ops = [o for o in result.operations if o.accepted]
        return {n for n, p in self.pieces.items() if self._at(ops, p["centre"]) and not p.get("useless")}

    def _left_parts(self, result, p):
        """A compound piece: the volume of its parts no operation sits near (3 units)."""
        ops = [o for o in result.operations if o.accepted]
        return sum(v for c, v in p["parts"] if not any(np.linalg.norm(np.subtract(o.primitive.center_mm, c)) < 3.0
                                                       for o in ops))

    def build(self, result, out_dir, name, timeout):
        self.builds += 1
        step = f"{name}.step"
        self.models[step] = result
        poison = any(self._at(result.operations, p["centre"]) for p in self.pieces.values() if p.get("poison"))
        return dict(valid=not poison, n_solids=1, step=step, stl=None)

    def difference(self, input_path, model_step, pieces_dir, max_pieces, min_solid, timeout):
        self.differences += 1
        model = self.models[model_step]
        done = self._explained(model)
        left = {n: (dict(p, volume=self._left_parts(model, p)) if p.get("parts") else p)
                for n, p in self.pieces.items() if n not in done}
        left = {n: p for n, p in left.items() if p["volume"] > 0}
        plus = sum(p["volume"] for p in left.values() if p["sign"] == "plus")
        ops = [o for o in model.operations if o.accepted]
        plus += sum(p.get("destroys", 0.0) for n, p in self.pieces.items() if n in done and any(   # input material it took
            np.linalg.norm(np.subtract(o.primitive.center_mm, p["centre"])) < 1e-3
            and (p.get("destroys_mode") is None or any(src.mode.startswith(p["destroys_mode"]) for src in o.primitive.sources))
            for o in ops))
        minus = sum(p["volume"] for p in left.values() if p["sign"] == "minus")
        out = dict(input_volume=V_IN, model_volume=V_IN - plus + minus, r_plus=plus, r_minus=minus, fuzzy=None,
                   iou=(V_IN - plus) / (V_IN + minus))
        if pieces_dir is not None:
            self.piece_differences += 1
            ps = sorted(left.items(), key=lambda t: -t[1]["volume"])
            out["pieces"] = [dict(id=n, sign=p["sign"], volume=p["volume"], centre=list(p["centre"]),
                                  bbox=p.get("bbox") or [list(np.subtract(p["centre"], 1)), list(np.add(p["centre"], 1))],
                                  stl=p.get("stl", self.stl), step=None, faces=p.get("faces", [])) for n, p in ps]
            out["left_out_count"] = dict(small=dict(plus=0, minus=0), capped=dict(plus=0, minus=0))
        return out

    def guess(self, piece, sources, deadline, registry, models):
        time.sleep(self.guess_sleep)
        self.guessed.append(piece["id"])
        if piece.get("segment") is not None:                      # a segment: a primitive at its middle
            lo, hi = (np.asarray(b, float) for b in piece["bbox"])
            half = (hi - lo) / 2
            prim = Primitive(kind="box", center_mm=tuple((lo + hi) / 2),
                             dims_mm=dict(x_half=half[0], y_half=half[1], z_half=half[2]))
            c = Candidate("fit", prim, 0.9, 0.01)
            c.precision, c.recall = 0.9995, 0.9
            return dict(candidates=[c], failed={}, timed_out=False)
        p = self.pieces[piece["id"]]
        if p.get("fail"):
            raise RuntimeError(f"the guess of {piece['id']} failed")
        prim = p.get("prim") or Primitive(kind="cylinder", center_mm=tuple(p["centre"]),
                                          dims_mm=dict(radius=0.5, half_length=1.0))
        worse = Primitive(kind="box", center_mm=tuple(np.add(p["centre"], 5.0)),
                          dims_mm=dict(x_half=1.0, y_half=1.0, z_half=1.0))
        best = Candidate("fit", prim, p.get("score", 0.95), 0.01)
        if "precision" in p:                                       # a candidate with its sampled precision and recall
            best.precision, best.recall = p["precision"], p.get("recall", 0.9)
        return dict(candidates=[best, Candidate("cascade", worse, 0.3, 0.02)], failed={}, timed_out=False)


def _world(**pieces):
    return World({n: dict(dict(sign="minus", volume=10.0), **p) for n, p in pieces.items()})


def _cfg(world, **kw):
    base = dict(feature_phase=True, feature_build=world.build, feature_difference=world.difference,
                feature_guess=world.guess, feature_min_share=1e-3)
    base.update(kw)
    return rl.ResidualConfig(**base)


def _rough():
    return ReconstructionResult(input_path="part.step", operations=[
        Operation("add", Primitive(kind="box", dims_mm=dict(x_half=10.0, y_half=10.0, z_half=10.0)))])


def _phase(world, rough=None, **kw):
    result = rough or _rough()
    out = kw.pop("out", None)
    gen = rl._feature_phase(result, _cfg(world, **kw), result.input_path, registry=None, out_dir=out)
    steps = []
    try:
        while True:
            steps.append(next(gen))
    except StopIteration as stop:
        return steps, result, stop.value


def _kinds(steps):
    """The kinds of a step stream without "feature_groups" (the grouping step of 2026-09-28, tested on its own)."""
    return [s.kind for s in steps if s.kind != "feature_groups"]


def _done(steps):
    assert steps[-1].kind == "feature_done"
    return steps[-1].info


# ---------------------------------------------------------------------------------------------- ranking, contract

def test_pieces_are_ranked_by_volume_and_capped(tmp_path):
    w = _world(a=dict(centre=(10, 0, 0), volume=5.0), b=dict(centre=(20, 0, 0), volume=40.0),
               c=dict(centre=(30, 0, 0), volume=20.0), d=dict(centre=(40, 0, 0), volume=0.5))    # d: 5e-4 < 1e-3
    steps, _, _ = _phase(w, feature_max_pieces=2, feature_max_iterations=1, out=tmp_path)
    first = steps[0]
    assert first.kind == "feature_residual"
    # every piece of the difference is taken, the biggest first; feature_max_pieces is the size of one validation batch
    assert [p["volume"] for p in first.info["pieces"]] == [40.0, 20.0, 5.0]
    assert first.info["dropped_small"] == 1 and first.info["dropped_limit"] == 0
    assert [p["id"] for p in first.info["pieces"]] == [1, 2, 3]
    assert [s.info["index"] for s in steps if s.kind == "feature_piece"] == [1, 2, 3]
    assert w.builds == 1 + 2 and w.piece_differences == 1


CONTRACT = {
    "feature_residual": {"iteration", "missing_volume", "invented_volume", "input_volume", "exact_iou", "pieces",
                         "dropped_small", "dropped_limit", "limits", "seconds"},
    "feature_groups": {"iteration", "groups", "ungrouped", "seconds"},
    "feature_piece": {"iteration", "piece", "index", "of", "mesh", "mesh_truncated", "tool_type", "group", "tool_source"},
    "feature_split": {"iteration", "piece_id", "method", "pitch", "grid", "segments", "seconds"},
    "feature_candidate": {"iteration", "piece_id", "op", "primitive", "source", "score", "seconds", "tool_type", "group",
                          "tool_source"},
    "feature_accepted": {"iteration", "piece_id", "op", "primitive", "source", "score", "validation", "tool_type", "group",
                         "tool_source"},
    "feature_rejected": {"iteration", "piece_id", "op", "primitive", "source", "score", "reason", "validation",
                         "tool_type", "group", "tool_source"},
    "feature_done": {"iterations", "applied", "rejected", "skipped", "stopped", "seconds", "exact_iou", "missing_volume",
                     "invented_volume", "split_pieces", "segments", "segments_accepted"},
}
PIECE_KEYS = {"id", "side", "op", "volume", "share", "bbox", "stl", "group", "tool_type"}
VALIDATION_KEYS = {"valid", "n_solids", "residual_before", "residual_after", "exact_iou", "seconds"}


def test_every_step_carries_the_contract(tmp_path):
    import trimesh
    stl = tmp_path / "piece.stl"
    trimesh.creation.box((1.0, 2.0, 3.0)).export(stl)
    w = _world(good=dict(centre=(1, 2, 3), sign="plus"), low=dict(centre=(4, 5, 6), score=0.4))
    w.stl = str(stl)
    steps, result, value = _phase(w, feature_min_score=0.85, feature_split=False, out=tmp_path)
    assert set(s.kind for s in steps) == set(rl.FEATURE_KINDS) - {"feature_split", "advice_asked", "advice_given",
                                                                   "feature_feedback", "feature_confirm",
                                                                   "feature_confirmed"}
    assert set(rl.FEATURE_KINDS) <= set(rl.STEP_KINDS)
    for s in steps:
        assert s.info["phase"] == "feature" and isinstance(s.info["message"], str) and s.info["message"]
        assert CONTRACT[s.kind] <= set(s.info), (s.kind, CONTRACT[s.kind] - set(s.info))
        assert isinstance(s.result, ReconstructionResult)
    res = next(s for s in steps if s.kind == "feature_residual").info
    for p in res["pieces"]:
        assert set(p) == PIECE_KEYS
        assert p["side"] in ("missing", "invented") and p["op"] == {"missing": "add", "invented": "cut"}[p["side"]]
        assert len(p["bbox"]) == 2 and all(len(c) == 3 for c in p["bbox"])
        assert p["share"] == pytest.approx(p["volume"] / V_IN)
    piece = next(s for s in steps if s.kind == "feature_piece").info
    assert piece["mesh"]["vertices"].dtype == np.float32 and piece["mesh"]["faces"].dtype == np.int32
    assert piece["mesh"]["faces"].shape == (12, 3) and piece["mesh_truncated"] is False
    acc = next(s for s in steps if s.kind == "feature_accepted").info
    assert VALIDATION_KEYS <= set(acc["validation"]) and acc["validation"]["valid"] is True
    assert acc["op"] == "add" and isinstance(acc["primitive"], Primitive) and acc["source"] == "fit"
    rej = next(s for s in steps if s.kind == "feature_rejected").info
    assert rej["reason"].startswith("best score 0.400") and rej["validation"] is None
    assert value.items() <= _done(steps).items() and value["applied"] == 1 and value["rejected"] == 1
    # the applied operation is in the program, named by its feature source; the snapshot of the step before is not
    op = result.operations[-1]
    assert op.op == "add" and any(s.mode == "feature:fit" for s in op.primitive.sources)
    assert len(steps[0].result.operations) == 1


def test_a_big_piece_mesh_is_subsampled():
    import tempfile
    import trimesh
    m = trimesh.creation.icosphere(subdivisions=6)                      # 81920 triangles
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "big.stl"
        m.export(path)
        mesh, truncated = rl._feature_piece_mesh(path)
    assert truncated and 1000 < len(mesh["faces"]) <= rl.FEATURE_MESH_TRIANGLES
    assert mesh["faces"].max() < len(mesh["vertices"]) and mesh["vertices"].size < 100_000
    r = np.linalg.norm(mesh["vertices"], axis=1)                        # still the sphere, vertices on it
    assert np.abs(r - 1.0).max() < 0.05


# ---------------------------------------------------------------------------------------------- congruent pieces

def _rotation(axis, deg):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.radians(deg) * np.asarray(axis, float) / np.linalg.norm(axis)).as_matrix()


MOTIONS = [(np.eye(3), np.array([40.0, -10.0, 5.0])), (_rotation([1, 2, 3], 70), np.array([-30.0, 25.0, 12.0])),
           (_rotation([0, 1, -1], 145), np.array([7.0, 60.0, -20.0]))]


def _l_piece(tmp_path, name, R, t, scale=1.0):
    """An asymmetric L (two boxes, off-centre) moved by x -> R x + t, as an STL and a piece dict. The volume is the
    exact one (76.2 at scale 1), as OCCT reports it: the marching-cubes meshes differ by 0.35 %."""
    from reco_field.peel import mesh_from_sdf
    from src import sdf_primitives as sp
    import trimesh
    L = sp.Union(sp.Placed(sp.Box([4.0 * scale, 1.0 * scale, 1.5 * scale]), translation=np.array([1.0, 0.3, 0.0])),
                 sp.Placed(sp.Box([1.0 * scale, 3.0 * scale, 1.5 * scale]), translation=np.array([-2.0, 3.0, 0.0])))
    v, f = mesh_from_sdf(sp.Placed(L, rotation=R, translation=t), resolution=64)
    m = trimesh.Trimesh(v, f)
    path = tmp_path / f"{name}.stl"
    m.export(path)
    exact = 76.2 if scale == 1.0 else float(abs(m.volume))
    return dict(id=name, sign="minus", volume=exact, centre=list(m.center_mass), stl=str(path), faces=[])


def test_congruent_pieces_are_grouped_and_a_different_one_is_not(tmp_path):
    pytest.importorskip("vtk")
    from reco_engine import feature_stage as fs
    pieces = [_l_piece(tmp_path, f"copy{i}", R, t) for i, (R, t) in enumerate(MOTIONS)]
    pieces.insert(2, _l_piece(tmp_path, "other", np.eye(3), np.array([0.0, -40.0, 0.0]), scale=1.1))
    groups = fs.group_congruent([fs.piece_descriptor(p) for p in pieces])
    assert groups == [[0, 1, 3], [2]]


def test_a_transferred_primitive_lands_on_each_copy(tmp_path):
    """A box on the long arm of the L of the first copy, moved onto the others by transfer_candidates: the best
    scoring candidate is the box under the known motion (centre within 0.05, its axes parallel to the moved ones)."""
    pytest.importorskip("vtk")
    from reco_engine import feature_stage as fs
    from reco_engine.result_to_program import quaternion_of, rotation_matrix
    from src.closest_point_field import load_field
    pieces = [_l_piece(tmp_path, f"copy{i}", R, t) for i, (R, t) in enumerate(MOTIONS)]
    descs = [fs.piece_descriptor(p) for p in pieces]
    R0, t0 = MOTIONS[0]
    arm = Primitive(kind="box", quaternion=quaternion_of(R0), center_mm=tuple(R0 @ [1.0, 0.3, 0.0] + t0),
                    dims_mm=dict(x_half=4.0, y_half=1.0, z_half=1.5))
    for (R, t), p, d in zip(MOTIONS[1:], pieces[1:], descs[1:]):
        fld = load_field(p["stl"])
        V, F = np.asarray(fld.vertices, float), np.asarray(fld.faces)
        sc, q = fs.best_transfer(arm, descs[0], d, lambda q2: fs.sampled_score(q2, V, F, n=20000))
        M = R @ R0.T                                                    # copy 0 -> this copy
        want = M @ (np.asarray(arm.center_mm) - t0) + t
        assert np.linalg.norm(np.asarray(q.center_mm) - want) < 0.05, (q.center_mm, want)
        assert np.allclose(np.abs(rotation_matrix(q).T @ (M @ R0)), np.eye(3), atol=0.02)
        assert sc > 0.5


def _box_world(tmp_path, n_groups=3, copies=2, bad_prim=False):
    """Congruent box pieces far apart: `n_groups` shapes, `copies` each, every copy turned and moved; the fake guess
    knows each piece's exact box (or, bad_prim, a box half as large for the representatives)."""
    import trimesh
    from reco_engine.result_to_program import quaternion_of
    pieces = {}
    for g in range(n_groups):
        ext = np.array([3.0 + g, 2.0, 1.0 + 0.5 * g])
        for c in range(copies):
            R = _rotation([1, g + 1, c + 2], 30 * c + 10 * g)
            t = np.array([100.0 * g + 30.0, 60.0 * c + 30.0, 30.0])      # away from the rough box at the origin
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t
            m = trimesh.creation.box(ext, T)
            path = tmp_path / f"g{g}c{c}.stl"
            m.export(path)
            half = ext / 2 * (0.5 if bad_prim and c == 0 else 1.0)
            prim = Primitive(kind="box", quaternion=quaternion_of(R), center_mm=tuple(t),
                             dims_mm=dict(x_half=half[0], y_half=half[1], z_half=half[2]))
            pieces[f"g{g}c{c}"] = dict(sign="minus", volume=float(m.volume), centre=tuple(t), stl=str(path), prim=prim,
                                       bbox=[list(m.bounds[0]), list(m.bounds[1])], score=0.95)
    return World(pieces)


def test_a_group_is_guessed_once_and_counts_once(tmp_path):
    w = _box_world(tmp_path, n_groups=3, copies=2)
    steps, result, info = _phase(w, feature_max_pieces=1, feature_min_share=1e-6, out=tmp_path)
    res = steps[0].info
    assert [len(g) for g in res["groups"]] == [2, 2, 2]
    assert len(w.guessed) == 3                                          # one full guess per group
    pattern = [s.info for s in steps if s.kind == "feature_candidate" and s.info["source"].startswith("pattern:")]
    assert len(pattern) == 3 and all(c["score"] > 0.9 for c in pattern)
    assert info["applied"] == 6 and info["stopped"] == "converged" and info["transferred"] == 3
    # max_pieces = 1 group per batch: three validations, and the pieces of the first difference served all of them
    assert w.piece_differences == 1 and info["differences"] == 1
    assert w.builds == 1 + 3


def test_a_failed_transfer_falls_back_to_a_full_guess(tmp_path):
    w = _box_world(tmp_path, n_groups=1, copies=2, bad_prim=True)
    for p in w.pieces.values():                                         # the representative's box is too small ...
        p["prim"].dims_mm = dict(p["prim"].dims_mm)
    w.pieces["g0c1"]["prim"] = copy_prim = w.pieces["g0c1"]["prim"]     # ... the member's own guess is right
    steps, result, info = _phase(w, feature_min_share=1e-6, out=tmp_path)
    cands = [s.info for s in steps if s.kind == "feature_candidate" and s.info["piece_id"] == 2]
    assert cands[0]["source"] == "pattern:1" and cands[0]["score"] < 0.85
    assert w.guessed == ["g0c0", "g0c1"] and info["transfer_fallbacks"] == 1
    assert copy_prim is w.pieces["g0c1"]["prim"]


# ---------------------------------------------------------------------------------------------- splitting

def _compound_sdf():
    """The synthetic compound excess (off-centre, asymmetric): a box A, a cylinder C standing on A's +x face (no neck:
    it meets A with its full section), and a smaller box B joined to A's -y face through a thin neck N."""
    from src import sdf_primitives as sp
    ALONG_X = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    A = sp.Placed(sp.Box([6.0, 4.0, 3.0]), translation=np.array([10.0, 20.0, -5.0]))
    C = sp.Placed(sp.Cylinder(2.0, 9.0), rotation=ALONG_X, translation=np.array([16.0, 20.5, -5.0]))   # 1 mm from the edges
    N = sp.Placed(sp.Box([0.6, 1.5, 0.6]), translation=np.array([9.0, 14.5, -5.0]))
    B = sp.Placed(sp.Box([2.5, 2.0, 2.0]), translation=np.array([9.0, 11.0, -5.0]))
    parts = dict(A=A, C=C, B=B, N=N)
    return sp.Union(sp.Union(A, C), sp.Union(N, B)), parts


def _compound_grid():
    from reco_field.peel import mesh_from_sdf
    from src.closest_point_field import SurfaceField
    from reco_engine import feature_stage as fs
    solid, parts = _compound_sdf()
    v, f = mesh_from_sdf(solid, resolution=128)
    fld = SurfaceField(v, f)
    spec, ch, _ = fs.piece_grid(fld)
    return fld, spec, ch, parts, (v, f)


def test_the_compound_excess_is_split_into_its_three_parts():
    """necks separate B (behind the thin neck), the surfaces separate the cylinder from the box it stands on: every
    part is >= 80 % one segment, three different segments; every voxel of the piece is in exactly one segment."""
    pytest.importorskip("vtk")
    from reco_engine import feature_stage as fs
    fld, spec, ch, parts, _ = _compound_grid()
    occ = ch[0] > 0.5
    labels, method = fs.segment_piece(occ, ch[1], ch[2:5], spec.pitch)
    assert method == "necks+surfaces"
    assert np.array_equal(labels > 0, occ)                               # each voxel in exactly one segment
    c = spec.centres().reshape(-1, 3)
    neck = np.asarray(parts["N"].sdf(c)).reshape(occ.shape) <= 0
    owner = {}
    for name in ("A", "C", "B"):
        inside = (np.asarray(parts[name].sdf(c)).reshape(occ.shape) <= -spec.pitch) & occ & ~neck
        if name == "A":                                                  # A's voxels that are not C's
            inside &= np.asarray(parts["C"].sdf(c)).reshape(occ.shape) > spec.pitch
        ids, counts = np.unique(labels[inside], return_counts=True)
        owner[name] = int(ids[np.argmax(counts)])
        # measured 2026-09-27 on 36 x 27 x 14 cells: C 100 %, B 100 %, A 87 % (A's voxels next to the junction go to
        # the cylinder: the nearest-seed rule)
        assert counts.max() >= 0.8 * inside.sum(), (name, dict(zip(ids.tolist(), counts.tolist())))
    assert len(set(owner.values())) == 3


def test_small_segments_join_their_largest_neighbour():
    from reco_engine import feature_stage as fs
    labels = np.zeros((20, 6, 6), np.int32)
    labels[1:10, 1:5, 1:5] = 1                                           # 144 voxels
    labels[10:11, 1:5, 1:5] = 2                                          # 16: a sliver between 1 and 3
    labels[11:19, 1:5, 1:5] = 3                                          # 128
    out = fs.merge_small(labels, min_voxels=50, max_segments=6)
    assert out.max() == 2 and np.array_equal(out > 0, labels > 0)
    assert (out[10, 2, 2] == out[5, 2, 2]) or (out[10, 2, 2] == out[15, 2, 2])
    capped = fs.merge_small(labels, min_voxels=1, max_segments=2)
    assert capped.max() == 2


def test_precision_and_recall_of_a_cut_against_analytic_values():
    """Region: the box [0, 2]^3 (volume 8); input material: x > 3. A cutter [1, 4] x [0, 2] x [0, 2] covers half the
    region (recall 0.5) and a third of it is input material (precision 2/3); moved faces inward by `shrink`, it stops
    at x = 3 (precision >= 0.98) and keeps the recall."""
    from reco_engine import feature_stage as fs

    def region(p):
        return np.all((p >= 0.0) & (p <= 2.0), axis=1)

    def inp(p):
        return p[:, 0] > 3.0
    sc = fs.PartScorer("cut", region, (0, 0, 0), (2, 2, 2), 8.0, inp, n=40000)
    cutter = Primitive(kind="box", center_mm=(2.5, 1.0, 1.0), dims_mm=dict(x_half=1.5, y_half=1.0, z_half=1.0))
    pr, rc = sc.score(cutter)
    assert pr == pytest.approx(2.0 / 3.0, abs=0.01) and rc == pytest.approx(0.5, abs=0.01)
    near = Primitive(kind="box", center_mm=(2.1, 1.0, 1.0), dims_mm=dict(x_half=1.1, y_half=1.0, z_half=1.0))
    pr0, _ = sc.score(near)
    assert 0.88 < pr0 < 0.98                                             # removes a little input material ...
    q, pr, rc = sc.shrink(near, 0.98, step=0.05)
    assert pr >= 0.98 and rc == pytest.approx(0.5, abs=0.01)             # ... shrunk until it does not
    assert q.center_mm[0] + q.dims_mm["x_half"] <= 3.06


def _compound_world(tmp_path):
    import trimesh
    fld, spec, ch, parts, (v, f) = _compound_grid()
    path = tmp_path / "compound.stl"
    trimesh.Trimesh(v, f).export(path)
    m = trimesh.load(path)
    # the parts' centres and exact volumes (A 12 x 8 x 6, C R 2 x 9, B 5 x 4 x 4)
    return World({"compound": dict(sign="minus", volume=float(m.volume), centre=tuple(m.center_mass), stl=str(path),
                                   score=0.3, bbox=[list(m.bounds[0]), list(m.bounds[1])],
                                   parts=[((10.0, 20.0, -5.0), 576.0), ((20.5, 20.5, -5.0), float(np.pi * 4 * 9)),
                                          ((9.0, 11.0, -5.0), 80.0)])}), fld


def test_a_split_piece_runs_through_the_contract(tmp_path):
    pytest.importorskip("vtk")
    w, fld = _compound_world(tmp_path)
    result = _rough()
    cfg = _cfg(w, feature_max_iterations=1)
    steps = list(rl._feature_phase(result, cfg, "part.step", None, field=None, out_dir=tmp_path))
    k = _kinds(steps)
    assert k[:3] == ["feature_residual", "feature_piece", "feature_candidate"] and "feature_split" in k
    sp_ = next(s.info for s in steps if s.kind == "feature_split")
    assert CONTRACT["feature_split"] <= set(sp_) and sp_["piece_id"] == 1 and len(sp_["segments"]) == 3
    assert {tuple(sorted(sg)) for sg in sp_["segments"]} == {("bbox", "id", "parent", "share", "volume", "voxels")}
    assert [sg["id"] for sg in sp_["segments"]] == [2, 3, 4] and all(sg["parent"] == 1 for sg in sp_["segments"])
    seg_pieces = [s.info for s in steps if s.kind == "feature_piece" and s.info["piece"].get("parent") == 1]
    assert len(seg_pieces) == 3 and all(sp2["mesh"]["faces"].shape[1] == 3 for sp2 in seg_pieces)
    assert all(sp2["piece"]["op"] == "cut" and sp2["piece"]["side"] == "invented" for sp2 in seg_pieces)
    cands = [s.info for s in steps if s.kind == "feature_candidate" and s.info["piece_id"] in (2, 3, 4)]
    assert all(c["precision"] == 0.9995 and c["recall"] == 0.9 for c in cands)
    parent = next(s.info for s in steps if s.kind == "feature_accepted" and s.info["piece_id"] == 1)
    assert parent["primitive"] is None and sorted(parent["segments_accepted"]) == [2, 3, 4]
    assert VALIDATION_KEYS <= set(parent["validation"])
    done = _done(steps)
    assert CONTRACT["feature_done"] <= set(done)
    assert (done["split_pieces"], done["segments"], done["segments_accepted"], done["applied"]) == (1, 3, 3, 3)
    assert not any(s.kind == "feature_rejected" and s.info["piece_id"] == 1 for s in steps)


def test_without_splitting_a_compound_piece_is_rejected_as_before(tmp_path):
    pytest.importorskip("vtk")
    w, fld = _compound_world(tmp_path)
    steps = list(rl._feature_phase(_rough(), _cfg(w, feature_split=False, feature_voxel_fallback=False), "part.step",
                                   None, field=None, out_dir=tmp_path))
    assert _kinds(steps) == ["feature_residual", "feature_piece", "feature_candidate", "feature_candidate",
                             "feature_rejected", "feature_done"]
    assert steps[-2].info["reason"] == "best score 0.300 < 0.85"


def test_a_piece_too_thin_for_its_licensed_grid_is_split_at_a_coarser_documented_pitch():
    """A cap of 5000 cells: the compound's licensed pitch (tau / 3) needs more, so the pitch is relaxed, flagged, and
    the grid stays under the cap."""
    pytest.importorskip("vtk")
    from reco_engine import feature_stage as fs
    from reco_field.peel import mesh_from_sdf
    from src.closest_point_field import SurfaceField
    solid, _ = _compound_sdf()
    v, f = mesh_from_sdf(solid, resolution=128)
    spec, ch, info = fs.piece_grid(SurfaceField(v, f), max_cells=5000)
    assert info["relaxed"] and info["pitch"] > info["licensed_pitch"] and np.prod(spec.shape) <= 5000
    region = lambda q: np.asarray(solid.sdf(q)) <= 0                    # noqa: E731
    spec2, ch2, _ = fs.piece_grid(SurfaceField(v, f), region=region, bbox=(v.min(axis=0), v.max(axis=0)))
    occ, occ_mesh = ch2[0] > 0.5, fs.piece_grid(SurfaceField(v, f))[1][0] > 0.5
    assert abs(int(occ.sum()) - int(occ_mesh.sum())) <= 0.05 * occ_mesh.sum()   # the predicate's voxels = the mesh's


def test_the_time_budget_stops_the_splitting(tmp_path):
    pytest.importorskip("vtk")
    w, fld = _compound_world(tmp_path)
    cfg = _cfg(w, feature_time_budget=0.0)
    steps = list(rl._feature_phase(_rough(), cfg, "part.step", None, field=None, out_dir=tmp_path))
    assert _done(steps)["stopped"] == "time budget"
    w2, _ = _compound_world(tmp_path)
    w2.guess_sleep = 0.0
    orig = rl._feature_piece_mesh
    cfg2 = _cfg(w2, feature_time_budget=3.0)
    t = time.time()
    steps = list(rl._feature_phase(_rough(), cfg2, "part.step", None, field=None, out_dir=tmp_path))
    assert rl._feature_piece_mesh is orig and time.time() - t < 60


# ---------------------------------------------------------------------------------------------- validation

def test_an_operation_that_breaks_the_solid_is_reverted_and_named(tmp_path):
    """Three guesses validated as one batch: the batch fails (the poison one makes the solid invalid), the halves are
    validated on their own, and only the poison piece is rejected - with the validation that failed."""
    w = _world(a=dict(centre=(1, 0, 0), volume=30.0), bad=dict(centre=(2, 0, 0), volume=20.0, poison=True),
               c=dict(centre=(3, 0, 0), volume=10.0))
    steps, result, info = _phase(w, feature_max_iterations=1, out=tmp_path)
    acc = [s.info["piece_id"] for s in steps if s.kind == "feature_accepted"]
    rej = [s.info for s in steps if s.kind == "feature_rejected"]
    assert sorted(acc) == [1, 3] and [r["piece_id"] for r in rej] == [2]
    assert rej[0]["reason"] == "FreeCAD: the solid is not valid" and rej[0]["validation"]["valid"] is False
    centres = [tuple(o.primitive.center_mm) for o in result.operations]
    assert (2.0, 0.0, 0.0) not in centres and (1.0, 0.0, 0.0) in centres and (3.0, 0.0, 0.0) in centres
    assert info["applied"] == 2 and info["rejected"] == 1


def test_one_build_per_guess_without_batching(tmp_path):
    w = _world(a=dict(centre=(1, 0, 0)), b=dict(centre=(2, 0, 0)))
    _phase(w, feature_batch=False, feature_max_iterations=1, out=tmp_path)
    assert w.builds == 1 + 2                                            # the rough model, then one per guess
    w2 = _world(a=dict(centre=(1, 0, 0)), b=dict(centre=(2, 0, 0)))
    _phase(w2, feature_batch=True, feature_max_iterations=1, out=tmp_path)
    assert w2.builds == 1 + 1


def test_an_operation_that_does_not_lower_the_residual_is_rejected(tmp_path):
    w = _world(a=dict(centre=(1, 0, 0), useless=True))
    steps, result, info = _phase(w, out=tmp_path)
    rej = next(s.info for s in steps if s.kind == "feature_rejected")
    assert rej["reason"].startswith("the exact residual drops by 0") and rej["validation"]["residual_after"] == 10.0
    assert len(result.operations) == 1 and info["stopped"] == "nothing applied"


def test_one_failing_piece_does_not_end_the_phase(tmp_path):
    w = _world(boom=dict(centre=(1, 0, 0), volume=30.0, fail=True), fine=dict(centre=(2, 0, 0), volume=20.0))
    steps, result, info = _phase(w, out=tmp_path)
    rej = next(s.info for s in steps if s.kind == "feature_rejected")
    assert rej["piece_id"] == 1 and "the guess of boom failed" in rej["reason"] and rej["primitive"] is None
    assert [s.info["piece_id"] for s in steps if s.kind == "feature_accepted"] == [2]
    assert info["applied"] == 1


def test_the_ends_of_a_cutter_go_out_where_that_removes_nothing():
    """A cutter along z through a slab (|z| <= 1): the ends lie on the slab's faces and move out by 5 % of the
    length into air; a cutter whose top end is a real floor inside the slab keeps that end."""
    def slab(pts):
        return np.abs(np.asarray(pts)[:, 2]) <= 1.0
    through = Primitive(kind="cylinder", center_mm=(3.0, -2.0, 0.0), dims_mm=dict(radius=0.5, half_length=1.0))
    q = extend_ends(through, "cut", slab)
    assert q.dims_mm["half_length"] == pytest.approx(1.1) and q.center_mm == pytest.approx((3.0, -2.0, 0.0))
    blind = Primitive(kind="cylinder", center_mm=(3.0, -2.0, -0.5), dims_mm=dict(radius=0.5, half_length=0.5))
    q = extend_ends(blind, "cut", slab)
    assert q.dims_mm["half_length"] == pytest.approx(0.525) and q.center_mm[2] == pytest.approx(-0.525)
    assert blind.dims_mm["half_length"] == 0.5                          # the input primitive is not changed


# ---------------------------------------------------------------------------------------------- limits

def test_converged_when_every_piece_is_explained(tmp_path):
    w = _world(a=dict(centre=(1, 0, 0)), b=dict(centre=(2, 0, 0), sign="plus"))
    steps, result, info = _phase(w, out=tmp_path)
    assert info["stopped"] == "converged" and info["applied"] == 2
    assert info["missing_volume"] == 0.0 and info["invented_volume"] == 0.0 and info["exact_iou"] == 1.0
    assert [o.op for o in result.operations] == ["add", "cut", "add"]


def test_max_iterations(tmp_path):
    w = _world(a=dict(centre=(1, 0, 0), volume=30.0), b=dict(centre=(2, 0, 0), volume=20.0),
               c=dict(centre=(3, 0, 0), volume=10.0))
    steps, _, info = _phase(w, feature_max_pieces=1, feature_max_iterations=2, out=tmp_path)
    assert info["stopped"] == "max iterations" and info["iterations"] == 2 and info["applied"] == 2
    assert [s.info["iteration"] for s in steps if s.kind == "feature_residual"] == [1, 2]


def test_max_pieces(tmp_path):
    w = _world(a=dict(centre=(1, 0, 0), volume=30.0), b=dict(centre=(2, 0, 0), volume=20.0))
    steps, _, info = _phase(w, feature_max_total_pieces=1, out=tmp_path)
    assert info["stopped"] == "max pieces" and info["applied"] == 1
    assert sum(s.kind == "feature_piece" for s in steps) == 1


def test_rejections_do_not_use_up_the_applied_cap_and_the_applied_cap_stops(tmp_path):
    w = _world(**{f"low{i}": dict(centre=(10.0 * i + 10.0, 0, 0), volume=30.0 - i, score=0.3) for i in range(3)},
               good=dict(centre=(50.0, 0, 0), volume=5.0))
    _, _, info = _phase(w, feature_max_applied=1, feature_split=False, out=tmp_path)
    assert info["rejected"] == 3 and info["applied"] == 1               # three rejections, the good one still applied
    w2 = _world(a=dict(centre=(1, 0, 0), volume=30.0), b=dict(centre=(2, 0, 0), volume=20.0))
    _, _, info2 = _phase(w2, feature_max_applied=1, out=tmp_path)
    assert info2["stopped"] == "max applied" and info2["applied"] == 1


def test_time_budget(tmp_path):
    w = _world(a=dict(centre=(1, 0, 0), volume=30.0), b=dict(centre=(2, 0, 0), volume=20.0),
               c=dict(centre=(3, 0, 0), volume=10.0))
    w.guess_sleep = 0.3
    steps, _, info = _phase(w, feature_time_budget=0.2, out=tmp_path)
    assert info["stopped"] == "time budget"
    assert sum(s.kind == "feature_piece" for s in steps) == 1 and info["skipped"] == 2
    assert info["applied"] == 1                                         # a guess already made is still validated


def test_nothing_applied(tmp_path):
    w = _world(a=dict(centre=(1, 0, 0), score=0.5))
    _, _, info = _phase(w, out=tmp_path)
    assert info["stopped"] == "nothing applied" and info["rejected"] == 1 and info["applied"] == 0


def test_an_error_of_the_difference_ends_the_phase_with_a_verdict(tmp_path):
    w = _world(a=dict(centre=(1, 0, 0)))
    steps, result, info = _phase(w, feature_difference=lambda *a: dict(error="booleans disagree"), out=tmp_path)
    assert _kinds(steps) == ["feature_done"]
    assert info["stopped"].startswith("error: booleans disagree (occt difference, ") and "(timeout 300 s)" in info["stopped"]
    assert len(result.operations) == 1


def _featured_world(**pieces):
    """A world whose global difference fails once a feature is in the model (the liner's, 2026-09-29), with the
    incremental difference that answers from the state and the tools: (world, the incremental call, its calls)."""
    w = _world(**pieces)
    calls = []

    def global_difference(input_path, model_step, pieces_dir, max_pieces, min_solid, timeout):
        if len(w.models[model_step].operations) > 1:
            return dict(error="exact residual: booleans disagree")
        return w.difference(input_path, model_step, pieces_dir, max_pieces, min_solid, timeout)

    def incremental(input_path, model_step, state, tools, state_dir, pieces_dir, max_pieces, min_solid, timeout):
        calls.append(dict(model=model_step, state=str(state), tools=list(tools), state_dir=str(state_dir),
                          pieces=pieces_dir is not None))
        return dict(w.difference(input_path, model_step, pieces_dir, max_pieces, min_solid, timeout), method="incremental")
    return w, global_difference, incremental, calls


def test_the_second_iteration_gets_its_pieces_from_the_incremental_difference(tmp_path):
    """The global boolean answers for the rough model only: the validation and the second iteration's pieces come
    from the state of the model before and the tools applied since, and every step names its method and timeout."""
    w, glob, inc, calls = _featured_world(a=dict(centre=(1, 0, 0), volume=30.0), b=dict(centre=(2, 0, 0), volume=20.0),
                                          c=dict(centre=(3, 0, 0), volume=10.0))
    steps, _, info = _phase(w, feature_difference=glob, feature_incremental_difference=inc, feature_max_pieces=1,
                            feature_max_iterations=2, feature_call_timeout=77, out=tmp_path)
    assert info["stopped"] == "max iterations" and info["iterations"] == 2 and info["applied"] == 2
    res = [s for s in steps if s.kind == "feature_residual"]
    assert [s.info["difference_method"] for s in res] == ["occt", "incremental"]
    assert "incremental difference" in res[1].info["message"] and "(timeout 77 s)" in res[1].info["message"]
    assert len(res[1].info["pieces"]) == 2 and res[1].info["difference_timeout"] == 77
    acc = [s for s in steps if s.kind == "feature_accepted"]
    assert [s.info["validation"]["method"] for s in acc] == ["incremental", "incremental"]
    assert "[incremental, " in acc[0].info["message"]
    assert info["difference_methods"] == dict(occt=1, incremental=1)
    # validation of a: the rough model's state + one tool; the pieces of iteration 2: that model's own state, no tool
    assert [(len(c["tools"]), c["pieces"]) for c in calls] == [(1, False), (0, True), (1, False)]
    assert calls[0]["state"].endswith("iteration_1") and calls[1]["state"] == calls[0]["state_dir"]
    assert calls[0]["tools"][0]["op"] == "cut" and calls[0]["tools"][0]["program"]["type"]


def test_a_failed_incremental_difference_falls_back_and_says_so(tmp_path):
    """The incremental difference fails: the global one is asked (here it fails on a model with features too), the
    sampled judge is not there without a mesh - the operation is rejected with both reasons in its validation."""
    w, glob, _, _ = _featured_world(a=dict(centre=(1, 0, 0), volume=30.0))
    steps, result, info = _phase(w, feature_difference=glob, out=tmp_path,
                                 feature_incremental_difference=lambda *a: dict(error="incremental difference: no budget"))
    rej = [s for s in steps if s.kind == "feature_rejected"]
    assert len(rej) == 1 and "booleans disagree" in rej[0].info["reason"]
    assert rej[0].info["validation"]["difference_fallback_from"] == "incremental difference: no budget"
    assert info["applied"] == 0 and len(result.operations) == 1
    off = _featured_world(a=dict(centre=(1, 0, 0), volume=30.0))
    steps, _, info = _phase(off[0], feature_difference=off[1], feature_incremental_difference=off[2],
                            feature_incremental=False, out=tmp_path)
    assert off[3] == [] and info["applied"] == 0                        # switched off: the global difference alone


def test_a_big_stl_is_skipped_with_a_message(tmp_path):
    class Field:
        vertices = np.zeros((3, 3))
        faces = np.zeros((20000, 3), int)
    result = ReconstructionResult(input_path="part.stl", operations=_rough().operations)
    cfg = rl.ResidualConfig(feature_phase=True)
    steps = list(rl._feature_phase(result, cfg, "part.stl", None, field=Field(), out_dir=tmp_path))
    assert _kinds(steps) == ["feature_done"]
    assert steps[0].info["stopped"].startswith("error: skipped - an STL of 20000 triangles")


# ---------------------------------------------------------------------------------------------- inside the loop

def _loop_run(**kw):
    """test_add_cut_loop's box with a bore (network-free registry, fake FreeCAD grade)."""
    from test_add_cut_loop import _field, _cfg as loop_cfg
    from src import sdf_primitives as sp
    solid = sp.Box([8.0, 8.0, 5.0]) - sp.Placed(sp.Cylinder(3.0, 30.0), translation=np.array([0.0, 0.0, -15.0]))
    cfg, _ = loop_cfg(**kw)
    return list(rl.residual_steps(_field(solid), cfg))


# measured on HEAD 8c763981 before the feature phase existed (2026-09-27): the steps other than "candidate"
BASELINE = [("input", "12528 input voxels on 32^3, budget 6, type box (analytic:voxel_fit 0.89)"),
            ("residual", "|R| = 1088 (+0 / -1088)"), ("component", "cut component: 1088 voxels"),
            ("accepted", "cut cylinder: |R| -1088"), ("residual", "|R| = 0 (+0 / -0)"),
            ("grade", "FreeCAD: valid True, chamfer 0.01"),
            # the exact solid gate (D5, since 2026-10-16) reports its builds in the final message; the baseline predated it
            ("done", "done: residual within tolerance; D5 exact: 1 build(s), 0 refused, 0 without answer, N s; IoU 1.0000, "
                     "coverage 1.000, score 1.000")]


@pytest.mark.slow                                                        # two loop runs, 25 s
def test_the_loop_without_the_phase_is_unchanged_and_with_it_ends_in_the_phase(tmp_path):
    pytest.importorskip("vtk")
    sys.path.insert(0, str(ROOT / "tests"))
    off = _loop_run()
    import re                                                    # the gate's seconds vary with the machine's load
    assert [(s.kind, re.sub(r"(without answer), \d+ s;", r"\1, N s;", s.info["message"]))
            for s in off if s.kind not in ("candidate", "add")] == BASELINE
    # 57 candidates since the torus guard (T-29, 2026-10-01; 59 before): the cascade's torus of the bore (R 321.5,
    # r 3: 13 x the part's diagonal, oversized) is no branch any more; its place among the singles is taken by a box
    assert sum(s.kind == "candidate" for s in off) == 57 and not any(s.kind in rl.FEATURE_KINDS for s in off)
    w = _world(bore=dict(centre=(100.0, 0.0, 0.0), volume=10.0))
    on = _loop_run(feature_phase=True, feature_build=w.build, feature_difference=w.difference, feature_guess=w.guess,
                   out_dir=tmp_path)                             # not the user's ~/.cache (T-05)
    k = _kinds(on)
    first = k.index("feature_residual")
    # the rough rule (feature phase on): single primitives first, so fewer candidates; the same verdicts
    # the rough rule (default grade 0.1): the first cut is the same single cylinder, the steps are today's kinds
    assert on[0].info["message"] == off[0].info["message"] + "; rough rule: a cut gains >= 0.1 x the first cut's " \
        "gain, an add the same of the first add's, single primitives first"
    assert [s.kind for s in on[1:first] if s.kind != "candidate"] == \
        [s.kind for s in off[1:] if s.kind not in ("candidate", "grade", "done")]
    acc = next(s for s in on if s.kind == "accepted")
    assert acc.info["grade"] == 1.0 and acc.info["message"] == "cut cylinder: |R| -1088, grade 1.000"
    assert k[first:] == ["feature_residual", "feature_piece", "feature_candidate", "feature_candidate",
                         "feature_accepted", "feature_done", "grade", "done"]
    assert on[-1].info["stopped"].endswith("; feature phase: converged, 1 applied")
    assert on[-1].result.operations[-1].primitive.sources[-1].mode == "feature:fit"


@pytest.mark.slow                                                        # one loop run, 15 s
def test_a_small_second_cut_is_graded_against_the_first_and_left_for_the_phase(tmp_path):
    """A box with a R 3 bore and a R 1 bore: the first cut (the big bore) sets the scale; the small bore gains about
    a ninth of it, below rough_cut_grade 0.2 - it is not cut by the voxel loop, the step says why, and the feature
    phase gets it."""
    pytest.importorskip("vtk")
    sys.path.insert(0, str(ROOT / "tests"))
    from test_add_cut_loop import _field, _cfg as loop_cfg
    from src import sdf_primitives as sp
    solid = (sp.Box([8.0, 8.0, 5.0]) - sp.Placed(sp.Cylinder(3.0, 30.0), translation=np.array([-3.0, -2.0, -15.0]))
             - sp.Placed(sp.Cylinder(1.0, 30.0), translation=np.array([4.5, 4.0, -15.0])))
    w = _world(bore=dict(centre=(100.0, 0.0, 0.0), volume=10.0))
    cfg, _ = loop_cfg(feature_phase=True, feature_build=w.build, feature_difference=w.difference, feature_guess=w.guess,
                      rough_cut_grade=0.2, out_dir=tmp_path)
    steps = list(rl.residual_steps(_field(solid), cfg))
    acc = [s for s in steps if s.kind == "accepted"]
    assert len(acc) == 1 and acc[0].info["grade"] == 1.0 and acc[0].info["primitive"].dims_mm["radius"] > 2.5
    rej = next(s for s in steps if s.kind == "rejected" and "grade" in s.info)
    assert 0.05 < rej.info["grade"] < 0.2 and rej.info["first_cut_gain"] == acc[0].info["gain"]
    assert rej.info["reason"].startswith("rough: grade ") and rej.info["reason"].endswith(": left for the feature phase")
    assert any(s.kind == "feature_residual" for s in steps)
    assert steps[-1].info["stopped"].startswith("rough: grade ")
    assert any(src.mode == "rough_grade" for src in acc[0].result.operations[-1].primitive.sources)


# ---------------------------------------------------------------------------------------------- end to end (slow)

@pytest.mark.slow
def test_a_bore_below_the_voxel_pitch_is_found_by_the_phase(tmp_path):
    """A STEP box (40 x 30 x 20, off-centre) with a R 4 bore along z and a R 0.6 bore along x - below the 32^3 grid's
    pitch of ~1.45 - built by freecadcmd; the loop with the phase on must end with both bores as cylinder cuts and one
    valid solid."""
    from reco_field.export import build_in_freecad, find_freecadcmd
    from reco_engine import cascade
    from reco_engine.result_to_program import rotation_matrix
    from src import sdf_primitives as sp
    if find_freecadcmd() is None:
        pytest.skip("freecadcmd not found")
    T = np.array([7.0, -4.0, 3.0])
    ALONG_X = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])        # local z -> world x
    big = sp.Placed(sp.Cylinder(4.0, 40.0), translation=np.array([-8.0, 4.0, -20.0]))
    small = sp.Placed(sp.Cylinder(0.6, 60.0), rotation=ALONG_X, translation=np.array([-30.0, -6.0, 4.0]))
    part = build_in_freecad(sp.Placed(sp.Box([20.0, 15.0, 10.0]) - big - small, translation=T), tmp_path / "in", "part")
    assert part.valid
    cfg = rl.ResidualConfig(resolution=32, max_ops=6, registry=cascade.DEFAULT_REGISTRY, out_dir=tmp_path / "out",
                            feature_phase=True, feature_sources=("step faces", "fit"), exact_grade=False)
    steps = list(rl.residual_steps(part.step, cfg))
    done = steps[-1]
    feats = next(s for s in steps if s.kind == "feature_done").info
    assert done.info["grade"]["valid"] is True, (feats, done.info["grade"])
    cuts = [o.primitive for o in done.result.operations if o.accepted and o.op == "cut" and o.primitive.kind == "cylinder"]

    def has(radius, axis, point):
        for p in cuts:
            a = rotation_matrix(p)[:, 2]
            off = np.asarray(p.center_mm) - point
            off = off - a * (off @ a)
            if abs(p.dims_mm["radius"] - radius) < 0.05 * radius and abs(a @ axis) > 0.999 and np.linalg.norm(off) < 0.1:
                return True
        return False

    assert has(4.0, np.array([0.0, 0.0, 1.0]), T + [-8.0, 4.0, 0.0]), [(p.dims_mm, p.center_mm) for p in cuts]
    assert has(0.6, np.array([1.0, 0.0, 0.0]), T + [0.0, -6.0, 4.0]), (feats, [(p.dims_mm, p.center_mm) for p in cuts])


INSPECT = r"""
import json, os, FreeCAD as App
job = json.load(open(os.environ["INSPECT_JOB"]))
doc = App.openDocument(job["path"])
groups = {o.Label: [c.Label for c in o.Group] for o in doc.Objects if o.TypeId == "App::DocumentObjectGroup"}
pieces = [o.Label for o in doc.Objects if hasattr(o, "Verdict")]
sheet = doc.getObject("Summary")
rows = 0
while sheet.getContents(f"A{4 + rows}"):
    rows += 1
types = sorted({o.TypeId for o in doc.Objects})
json.dump(dict(groups=groups, pieces=pieces, rows=rows, header=sheet.getContents("A1"), types=types), open(job["out"], "w"))
"""


@pytest.mark.slow
def test_the_phase_writes_a_freecad_document_with_the_exploded_difference(tmp_path):
    """The small STEP of the end-to-end test: features.FCStd reopened headless has the groups, one labelled object per
    exploded piece of each difference (as many as the difference's solids above nothing), the applied features with
    their checks, and one Summary row per piece; the exact check agrees with the sampled precision."""
    import json
    import subprocess
    from reco_field.export import build_in_freecad, find_freecadcmd
    from reco_engine import cascade
    from src import sdf_primitives as sp
    if find_freecadcmd() is None:
        pytest.skip("freecadcmd not found")
    T = np.array([7.0, -4.0, 3.0])
    ALONG_X = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    big = sp.Placed(sp.Cylinder(4.0, 40.0), translation=np.array([-8.0, 4.0, -20.0]))
    small = sp.Placed(sp.Cylinder(0.6, 60.0), rotation=ALONG_X, translation=np.array([-30.0, -6.0, 4.0]))
    part = build_in_freecad(sp.Placed(sp.Box([20.0, 15.0, 10.0]) - big - small, translation=T), tmp_path / "in", "part")
    cfg = rl.ResidualConfig(resolution=32, max_ops=6, registry=cascade.DEFAULT_REGISTRY, out_dir=tmp_path / "out",
                            feature_phase=True, feature_sources=("step faces", "fit"), exact_grade=False)
    steps = list(rl.residual_steps(part.step, cfg))
    done = next(s for s in steps if s.kind == "feature_done").info
    assert done["fcstd"] and Path(done["fcstd"]).is_file(), done["message"]
    assert done["message"].endswith(f"FreeCAD document {done['fcstd']}")
    acc = [s.info for s in steps if s.kind == "feature_accepted" and s.info["primitive"] is not None]
    assert acc and all(a["validation"]["common_volume"] > 0 for a in acc)
    for a in acc:                                                      # exact harm vs the sampled precision
        c = next(s.info for s in steps if s.kind == "feature_candidate" and s.info["piece_id"] == a["piece_id"]
                 and s.info["source"] == a["source"])
        exact_precision = 1.0 - a["validation"]["harm_volume"] / max(a["validation"]["common_volume"]
                                                                     + a["validation"]["beyond_piece_volume"], 1e-12)
        assert c["precision"] == pytest.approx(exact_precision, abs=0.02)
    job = tmp_path / "inspect.json"
    job.write_text(json.dumps(dict(path=done["fcstd"], out=str(tmp_path / "inspected.json"))))
    script = tmp_path / "inspect_document.py"            # not inspect.py: it would shadow the stdlib module
    script.write_text(INSPECT)
    env = {k: v for k, v in __import__("os").environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["INSPECT_JOB"] = str(job)
    subprocess.run([find_freecadcmd(), str(script)], env=env, capture_output=True, text=True, timeout=300)
    got = json.loads((tmp_path / "inspected.json").read_text())
    for name in ("Input", "Rough model", "Final model", "Difference, iteration 1", "Features", "Checks"):
        assert name in got["groups"], got["groups"].keys()
    res = next(s.info for s in steps if s.kind == "feature_residual")
    assert len(got["pieces"]) >= len(res["pieces"])                    # every piece exploded, the small ones too
    assert sum(lab.startswith("piece ") for lab in got["pieces"]) >= len(res["pieces"])
    assert len(got["groups"]["Features"]) >= len(acc) and got["header"].lstrip("'") == "exact IoU before"
    assert got["rows"] == len({s.info["piece"]["id"] for s in steps if s.kind == "feature_piece"})


# ---------------------------------------------------------------------------------------------- boolean tools

TILT = _rotation([0.3, 1.0, 0.2], 25)
SHIFT = np.array([31.0, -12.0, 7.5])


def _rounded_pocket(radii, a=8.0, b=4.0, depth=3.0):
    """A pocket a x b (half sizes) with rounded corners `radii` (one per corner, 0 = sharp; a slot: r = b at both
    ends), tilted and off-centre: (solid SDF, piece dict with its exact faces, V)."""
    from src import sdf_primitives as sp
    from reco_field.peel import mesh_from_sdf
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    solid = sp.Box([a, b, depth])
    faces = []
    for (si, sj), r in zip(corners, radii):
        if r <= 0:
            continue
        cx, cy = si * (a - r), sj * (b - r)
        sq = sp.Placed(sp.Box([r / 2 + 0.01, r / 2 + 0.01, depth + 0.1]), translation=np.array([si * (a - r / 2), sj * (b - r / 2), 0.0]))
        cyl = sp.Placed(sp.Cylinder(r, 2 * depth + 0.4), translation=np.array([cx, cy, -depth - 0.2]))
        solid = solid - (sq - cyl)
        faces.append(dict(type="cylinder", origin="input", area=r * depth, axis=list(TILT[:, 2]),
                          center=list(TILT @ [cx, cy, 0.0] + SHIFT), radius=r, convex=True))
    for n, off, area in (((1, 0, 0), a, b * depth), ((-1, 0, 0), a, b * depth), ((0, 1, 0), b, a * depth),
                         ((0, -1, 0), b, a * depth), ((0, 0, -1), depth, 4 * a * b), ((0, 0, 1), depth, 4 * a * b)):
        n = np.asarray(n, float)
        faces.append(dict(type="plane", origin="model" if n[2] > 0 else "input", area=area, normal=list(TILT @ n),
                          point=list(TILT @ (n * off) + SHIFT)))
    placed = sp.Placed(solid, rotation=TILT, translation=SHIFT)
    v, f = mesh_from_sdf(placed, resolution=96)
    return placed, dict(id="pocket", sign="minus", faces=faces, volume=None), np.asarray(v, float)


def _score_setup(placed, V):
    """The piece's region by its SDF; the input: a block around it, minus the pocket (what a cutter must not touch)."""
    lo, hi = V.min(axis=0), V.max(axis=0)

    def region(q):
        return np.asarray(placed.sdf(q)) <= 0

    def inp(q):
        q = np.asarray(q)
        near = np.all((q >= lo - 5.0) & (q <= hi + 5.0), axis=1)
        return near & ~region(q)
    return region, inp, lo, hi


@pytest.mark.parametrize("radii,label", [((4.0, 4.0, 4.0, 4.0), "slot"), ((1.5, 2.5, 1.5, 2.5), "rounded pocket")])
def test_a_rounded_pocket_is_found_as_a_boolean_tool(radii, label):
    """The plain box of the pocket's planes removes input material at the rounded corners (precision < 0.98); the
    rounded box (box minus corner square minus cylinder, per corner) passes with precision and recall >= 0.98."""
    pytest.importorskip("vtk")
    from reco_engine import feature_stage as fs
    placed, piece, V = _rounded_pocket(radii)
    region, inp, lo, hi = _score_setup(placed, V)
    vol = float(np.mean(region(lo + np.random.default_rng(1).random((400000, 3)) * (hi - lo)))) * float(np.prod(hi - lo))
    piece["volume"] = vol
    sc = fs.PartScorer("cut", region, lo, hi, vol, inp, n=60000)
    box = fs._box_of_planes([f for f in piece["faces"] if f["type"] == "plane"], V)
    pr_box, rc_box = sc.score(box)
    assert pr_box < 0.98 and rc_box > 0.98
    tools = fs.tool_candidates(piece, V, [box])
    rb = next(q for q in tools if (q.label or "").startswith("rounded box"))
    assert [len(fs.leaves(q)) for q in tools] == sorted(len(fs.leaves(q)) for q in tools)     # simplest first
    assert rb.label == "rounded box (4 rounded corner(s))" and len(fs.leaves(rb)) == 9
    pr, rc = sc.score(rb)
    assert pr >= 0.98 and rc >= 0.98, (pr, rc)


def test_the_tool_source_prefers_the_simplest_passing_candidate(tmp_path):
    """Through guess_candidates (the loop's path): the plain box fails the precision, the tools are asked, the rounded
    box wins; with a sharp pocket (no cylinder faces) the plain box passes and no tool is built."""
    pytest.importorskip("vtk")
    import trimesh
    from reco_engine import feature_stage as fs
    for radii, want in (((1.5, 2.5, 1.5, 2.5), "tool (rounded box (4 rounded corner(s)))"), ((0, 0, 0, 0), "step faces")):
        placed, piece, V = _rounded_pocket(radii)
        from reco_field.peel import mesh_from_sdf
        v, f = mesh_from_sdf(placed, resolution=96)
        path = tmp_path / f"p{len(radii)}{radii[0]}.stl"
        trimesh.Trimesh(v, f).export(path)
        region, inp, lo, hi = _score_setup(placed, V)
        piece.update(stl=str(path), volume=float(abs(trimesh.Trimesh(v, f).volume)), bbox=[list(lo), list(hi)])
        g = fs.guess_candidates(piece, sources=("step faces",), stop_at=None, op="cut", inside_input=inp,
                                inside_region=region, tools=dict(min_score=0.85, min_precision=0.98))
        best = g["candidates"][0]
        assert best.source == want and best.precision >= 0.98, [(c.source, c.precision, c.recall) for c in g["candidates"]]
        if want == "step faces":
            assert not any(c.source.startswith("tool") for c in g["candidates"])


def test_a_composite_moves_as_a_whole():
    """The pattern transfer of a composite: every operand moved by the same rigid motion - its SDF at R x + t equals
    the original's at x."""
    pytest.importorskip("vtk")
    from reco_engine import feature_stage as fs
    from reco_engine.result_to_program import primitive_to_sp, transform_primitive
    placed, piece, V = _rounded_pocket((1.5, 2.5, 1.5, 2.5))
    box = fs._box_of_planes([f for f in piece["faces"] if f["type"] == "plane"], V)
    rb = next(q for q in fs.tool_candidates(piece, V, [box]) if (q.label or "").startswith("rounded box"))
    R, t = _rotation([1, -2, 0.5], 130), np.array([-40.0, 9.0, 3.0])
    moved = transform_primitive(rb, R, t)
    x = V[:: max(1, len(V) // 2000)] + np.random.default_rng(0).normal(0, 0.5, (len(V[:: max(1, len(V) // 2000)]), 3))
    assert np.allclose(primitive_to_sp(moved).sdf(x @ R.T + t), primitive_to_sp(rb).sdf(x), atol=1e-9)


def test_the_tools_are_capped():
    pytest.importorskip("vtk")
    from reco_engine import feature_stage as fs
    placed, piece, V = _rounded_pocket((1.5, 2.5, 1.5, 2.5))
    box = fs._box_of_planes([f for f in piece["faces"] if f["type"] == "plane"], V)
    assert len(fs.tool_candidates(piece, V, [box], max_candidates=3)) == 3
    assert all(len(fs.leaves(p)) <= 3 for p in fs.tool_candidates(piece, V, [box], max_operands=3))


def _inspect(fcstd, tmp_path):
    import json
    import os
    import subprocess
    from reco_field.export import find_freecadcmd
    job = tmp_path / "inspect.json"
    job.write_text(json.dumps(dict(path=str(fcstd), out=str(tmp_path / "inspected.json"))))
    script = tmp_path / "inspect_document.py"
    script.write_text(INSPECT)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["INSPECT_JOB"] = str(job)
    subprocess.run([find_freecadcmd(), str(script)], env=env, capture_output=True, text=True, timeout=300)
    return json.loads((tmp_path / "inspected.json").read_text())


@pytest.mark.slow
def test_a_slot_and_a_rounded_pocket_end_as_boolean_tools(tmp_path):
    """A STEP plate with a slot (box | two half cylinders) and a pocket with four rounded corners: the coarse rough
    stage cuts boxes, the feature phase finishes the round ends and corners with boolean tools; one valid solid,
    exact IoU >= 0.99 (measured 2026-09-28: 0.9975), and features.FCStd shows a tool as its boolean tree."""
    from reco_field.export import build_in_freecad, find_freecadcmd
    from reco_engine import cascade
    from src import sdf_primitives as sp
    if find_freecadcmd() is None:
        pytest.skip("freecadcmd not found")
    T = np.array([9.0, -5.0, 2.0])

    def slot(c, L, r, depth):
        b = sp.Placed(sp.Box([L / 2, r, depth / 2 + 0.5]), translation=np.array([c[0], c[1], 5.0 - depth / 2 + 0.5]))
        e = [sp.Placed(sp.Cylinder(r, depth + 1.0), translation=np.array([c[0] + s * L / 2, c[1], 5.0 - depth]))
             for s in (-1, 1)]
        return sp.Union(sp.Union(b, e[0]), e[1])

    def rounded(c, a, b, r, depth):
        z = 5.0 - depth / 2 + 0.5
        body = sp.Union(sp.Placed(sp.Box([a - r, b, depth / 2 + 0.5]), translation=np.array([c[0], c[1], z])),
                        sp.Placed(sp.Box([a, b - r, depth / 2 + 0.5]), translation=np.array([c[0], c[1], z])))
        for sx in (-1, 1):
            for sy in (-1, 1):
                body = sp.Union(body, sp.Placed(sp.Cylinder(r, depth + 1.0), translation=np.array(
                    [c[0] + sx * (a - r), c[1] + sy * (b - r), 5.0 - depth])))
        return body
    part = sp.Box([30.0, 20.0, 5.0]) - slot((-12.0, 8.0), 14.0, 3.0, 4.0) - rounded((10.0, -6.0), 9.0, 6.0, 2.5, 3.0)
    inp = build_in_freecad(sp.Placed(part, translation=T), tmp_path / "in", "plate")
    # the share of 1e-5 this test was written with: the default 1e-3 (protocol focus_min_share_v1, 2026-10-30) drops the
    # pocket's rounded corners - wedges of about 4 mm^3 in the 24000 mm^3 plate (1.7e-4) - by design
    cfg = rl.ResidualConfig(resolution=32, max_ops=6, registry=cascade.DEFAULT_REGISTRY, out_dir=tmp_path / "run",
                            feature_phase=True, feature_sources=("step faces", "fit"), exact_grade=True,
                            feature_min_share=1e-5)
    steps = list(rl.residual_steps(inp.step, cfg))
    done = steps[-1]
    assert done.info["grade"]["valid"] is True and done.result.exact_iou >= 0.99, done.info["message"]
    tools = [o for o in done.result.operations if o.primitive.kind == "composite"
             and any(s.mode.startswith("feature:tool") for s in o.primitive.sources)]
    assert tools, [(o.op, o.primitive.kind, o.primitive.label) for o in done.result.operations]
    feats = next(s.info for s in steps if s.kind == "feature_done")
    got = _inspect(feats["fcstd"], tmp_path)
    assert {"Part::Cut", "Part::MultiFuse"} & set(got["types"]), got["types"]


def test_the_rough_rule_never_grades_the_member_sub_loop(monkeypatch):
    """The frame's safeguard (its rough model must not change): the member sub-loop runs in residual_steps BEFORE
    _residual_phase, where the rough rule lives, so its operations - here a fake member phase with a big and a tiny
    add, the tiny one 1 % of the big - are all kept with the rule at 0.1, and none carries a grade; only the
    residual phase's later operations are graded, against the largest add."""
    pytest.importorskip("vtk")
    sys.path.insert(0, str(ROOT / "tests"))
    from test_add_cut_loop import _field, _cfg as loop_cfg
    from src import sdf_primitives as sp
    from reco_engine.result_schema import Operation

    def fake_members(state, field, result, cfg, registry, purity_min, kinds, forced=False):
        for size, gain in ((8.0, 10000.0), (0.5, 100.0)):
            p = Primitive(kind="box", dims_mm=dict(x_half=size, y_half=size, z_half=min(size, 5.0)))
            s_c = state.prim_sdf(p)
            state.commit(state.apply(state.s_model, s_c, "add"), s_c, "add")
            result.operations.append(Operation("add", p, net_gain=gain))
            yield rl.ResidualStep("accepted", rl._snapshot(result), dict(op="add", primitive=p, gain=gain, message="member"))
        return 2
    monkeypatch.setattr(rl, "_member_phase", fake_members)
    w = _world(x=dict(centre=(100.0, 0.0, 0.0), volume=10.0))
    cfg, _ = loop_cfg(feature_phase=True, feature_build=w.build, feature_difference=w.difference, feature_guess=w.guess,
                      rough_cut_grade=0.1, hint_subclass="box/ibeam")
    steps = list(rl.residual_steps(_field(sp.Box([8.0, 8.0, 5.0])), cfg))
    ops = steps[-1].result.operations
    assert [o.net_gain for o in ops[:2]] == [10000.0, 100.0]           # both members kept, the 1 % one too
    assert not any(src.mode == "rough_grade" for o in ops[:2] for src in o.primitive.sources)


def test_splitting_waits_for_every_cheap_guess(tmp_path):
    """A big compound piece (no single primitive) and a small simple one: the small one is guessed and validated
    before the big one is split - the liner lost its whole budget splitting its largest pieces first (2026-09-28)."""
    pytest.importorskip("vtk")
    w, fld = _compound_world(tmp_path)
    w.pieces["small"] = dict(sign="minus", volume=5.0, centre=(200.0, 0.0, 0.0))
    steps = list(rl._feature_phase(_rough(), _cfg(w, feature_max_iterations=1), "part.step", None, field=None,
                                   out_dir=tmp_path))
    k = [s.kind for s in steps]
    small = next(s.info["piece"]["id"] for s in steps if s.kind == "feature_piece" and s.info["piece"]["volume"] == 5.0)
    acc = next(i for i, s in enumerate(steps) if s.kind == "feature_accepted" and s.info["piece_id"] == small)
    assert acc < k.index("feature_split")


# ---------------------------------------------------------------------------------------------- tool types and store

@pytest.fixture(scope="module")
def plate_pieces(tmp_path_factory):
    """A FreeCAD plate with two countersunk holes (one size) and a spherical blob (no catalogue type): the exact
    difference against the plain plate, its pieces with their faces (one FreeCAD run, ~5 s)."""
    from reco_field.export import build_in_freecad, find_freecadcmd
    from reco_field.exact_check import exact_residual
    from reco_engine import tools as TL
    from src import sdf_primitives as sp
    if find_freecadcmd() is None:
        pytest.skip("freecadcmd not found")
    tmp = tmp_path_factory.mktemp("typed")
    down = np.diag([1.0, -1.0, -1.0])
    plate = sp.Placed(sp.Box([40.0, 25.0, 10.0]), translation=np.array([0.0, 0.0, 10.0]))
    part = plate
    for x in (-20.0, 0.0):
        part = part - primitive_to_sp(TL.TOOLS["countersunk_hole"].build(dict(d=6.6, d_c=12.4, angle=90.0, length=8.0),
                                                                      down, np.array([x, 0.0, 20.0]), over=1.0))
    part = part - sp.Placed(sp.Sphere(6.0), translation=np.array([25.0, 8.0, 20.0]))
    inp = build_in_freecad(part, tmp / "in", "plate")
    model = build_in_freecad(plate, tmp / "model", "model")
    r = exact_residual(inp.step, model.step, pieces_dir=tmp / "pieces")
    assert "error" not in r, r
    return sorted(r["pieces"], key=lambda p: -p["volume"]), inp.step


from reco_engine.result_to_program import primitive_to_sp  # noqa: E402


class TypedWorld:
    """The plate's real pieces behind fakes: an accepted operation explains the piece whose contract id it carries
    (its feature source's detail); the guess returns the loop's first typed candidate as passing, else a created
    composite tool."""

    def __init__(self, pieces):
        self.pieces = {p["id"]: p for p in pieces}
        self.order = [p["id"] for p in pieces]
        self.models = {}
        self.typed_seen = []

    def _explained(self, result):
        ids = set()
        for o in result.operations:
            for s in o.primitive.sources:
                if s.mode.startswith("feature:") and s.detail:
                    ids.add(self.order[(int(s.detail["piece"]) - 1) % len(self.order)])
        return ids

    def build(self, result, out_dir, name, timeout):
        self.models[f"{name}.step"] = result
        return dict(valid=True, n_solids=1, step=f"{name}.step")

    def difference(self, input_path, model_step, pieces_dir, max_pieces, min_solid, timeout):
        done = self._explained(self.models[model_step])
        left = [self.pieces[k] for k in self.order if k not in done]
        out = dict(input_volume=V_IN * 100, r_plus=0.0, r_minus=sum(p["volume"] for p in left), iou=0.9)
        if pieces_dir is not None:
            out["pieces"] = left
            out["left_out_count"] = dict(small=dict(plus=0, minus=0), capped=dict(plus=0, minus=0))
        return out

    def guess(self, piece, sources, deadline, registry, models):
        typed = (piece.get("scoring") or {}).get("typed") or []
        self.typed_seen.append([t["source"] for t in typed])
        if typed:
            t = typed[0]
            c = Candidate(t["source"], t["primitive"], 0.99, 0.01)
            c.precision, c.recall = 1.0, 0.99
            c.tool_type, c.tool_source, c.tool_params = t["tool_type"], t["tool_source"], t.get("params")
            return dict(candidates=[c], failed={}, timed_out=False)
        from reco_engine.feature_stage import compose
        lo, hi = (np.asarray(b, float) for b in piece["bbox"])
        box = Primitive(kind="box", center_mm=tuple((lo + hi) / 2), dims_mm=dict(zip(("x_half", "y_half", "z_half"), (hi - lo) / 2)))
        cyl = Primitive(kind="cylinder", center_mm=tuple((lo + hi) / 2), dims_mm=dict(radius=1.0, half_length=1.0))
        c = Candidate("tool (box - cylinder)", compose(box, cyl, "cut", "box - cylinder"), 0.99, 0.01)
        c.precision, c.recall = 1.0, 0.99
        return dict(candidates=[c], failed={}, timed_out=False)


def _typed_run(w, store, tmp_path):
    cfg = rl.ResidualConfig(feature_phase=True, feature_build=w.build, feature_difference=w.difference,
                            feature_guess=w.guess, feature_min_share=1e-9, feature_tool_store=store, feature_check=False)
    return list(rl._feature_phase(_rough(), cfg, "plate.step", None, field=None, out_dir=tmp_path))


def test_recycle_before_catalogue_before_create_and_the_store_carries_over(plate_pieces, tmp_path):
    from reco_engine.tools import ToolStore
    pieces, _ = plate_pieces
    store = ToolStore()
    w = TypedWorld(pieces)
    steps = _typed_run(w, store, tmp_path)
    groups = next(s.info for s in steps if s.kind == "feature_groups")
    assert [(g["tool_type"], len(g["members"])) for g in groups["groups"]] == [("countersunk_hole", 2)]
    assert len(groups["ungrouped"]) == 1                                # the blob
    acc = {s.info["piece_id"]: s.info for s in steps if s.kind == "feature_accepted"}
    csk = groups["groups"][0]["members"]
    assert acc[csk[0]]["source"] == "catalogue:countersunk_hole" and acc[csk[0]]["tool_source"] == "catalogue"
    assert acc[csk[1]]["source"].startswith("pattern:") and acc[csk[1]]["tool_source"] == "catalogue"
    blob = groups["ungrouped"][0]
    assert acc[blob]["tool_source"] == "created" and acc[blob]["tool_type"] == "created:box - cylinder"
    done = _done(steps)
    assert done["tools_by_source"] == {"catalogue": 2, "created": 1} and done["store_size"] == 2
    rec = store.of_type("countersunk_hole")[0]
    assert rec.count == 2 and rec.params["d"] == pytest.approx(6.6, rel=1e-3)
    # the next run of the same part starts with this run's tools: recycled, for the holes and the blob
    path = store.save(tmp_path / "tools.json")
    w2 = TypedWorld(pieces)
    steps2 = _typed_run(w2, str(path), tmp_path / "second")
    acc2 = {s.info["piece_id"]: s.info for s in steps2 if s.kind == "feature_accepted"}
    first = next(s.info for s in steps2 if s.kind == "feature_groups")["groups"][0]["representative"]
    assert acc2[first]["source"].startswith("session:countersunk_hole#1") and acc2[first]["tool_source"] == "session"
    assert any(src.startswith("session:created") for src in sum(w2.typed_seen, []))
    assert _done(steps2)["tools_by_source"].get("session", 0) >= 2
    reloaded = ToolStore.load(path)
    assert reloaded.of_type("countersunk_hole")[0].count == 4          # two runs, two holes each


@pytest.mark.slow
def test_a_plate_of_standard_tools_is_cut_by_catalogue_tools_then_by_session_tools(tmp_path):
    """A STEP plate (tilted, off-centre) with a slot, a rounded pocket and four countersunk holes: the first run cuts
    them with catalogue tools (their representatives), the second run of the same plate with the first run's store
    recycles them (session); one valid solid, exact IoU >= 0.99 (measured 2026-09-28: 0.9962, 53 s per run)."""
    from reco_field.export import build_in_freecad, find_freecadcmd
    from reco_engine import cascade
    from reco_engine import tools as TL
    from src import sdf_primitives as sp
    if find_freecadcmd() is None:
        pytest.skip("freecadcmd not found")
    down = np.diag([1.0, -1.0, -1.0])

    def tool(name, xy, **p):
        return primitive_to_sp(TL.TOOLS[name].build(p, down, np.array([xy[0], xy[1], 10.0]), over=1.0))
    plate = sp.Placed(sp.Box([60.0, 40.0, 5.0]), translation=np.array([0.0, 0.0, 5.0]))
    part = plate - tool("slot", (-25.0, 20.0), L=30.0, W=10.0, depth=4.0) - \
        tool("rect_pocket", (20.0, 15.0), L=26.0, W=16.0, depth=3.0, r=4.0)
    for x in (-45.0, -15.0, 15.0, 45.0):
        part = part - tool("countersunk_hole", (x, -25.0), d=6.6, d_c=12.4, angle=90.0, length=4.0)
    R, _ = np.linalg.qr(np.array([[0.98, -0.17, 0.1], [0.17, 0.98, 0.05], [-0.1, -0.03, 0.99]]))
    R = R * np.sign(np.diag(R))
    inp = build_in_freecad(sp.Placed(part, rotation=R, translation=np.array([11.0, -6.0, 3.0])), tmp_path / "in", "plate")
    store = tmp_path / "tools.json"
    sources = []
    for run in (1, 2):
        cfg = rl.ResidualConfig(resolution=48, max_ops=6, registry=cascade.DEFAULT_REGISTRY, out_dir=tmp_path / f"run{run}",
                                feature_phase=True, feature_sources=("step faces", "fit"), exact_grade=True,
                                feature_tool_store=str(store))
        steps = list(rl.residual_steps(inp.step, cfg))
        groups = next(s.info for s in steps if s.kind == "feature_groups")
        assert sorted((g["tool_type"], len(g["members"])) for g in groups["groups"]) == \
            [("countersunk_hole", 4), ("rect_pocket", 1), ("slot", 1)]
        reps = {g["representative"]: g["tool_type"] for g in groups["groups"]}
        acc = {s.info["piece_id"]: s.info for s in steps if s.kind == "feature_accepted"}
        sources.append({reps[i]: acc[i]["tool_source"] for i in reps if i in acc})
        done = steps[-1]
        assert done.info["grade"]["valid"] is True and done.result.exact_iou >= 0.99
    assert sources[0] == {"countersunk_hole": "catalogue", "rect_pocket": "catalogue", "slot": "catalogue"}
    assert sources[1] == {"countersunk_hole": "session", "rect_pocket": "session", "slot": "session"}


# ---------------------------------------------------------------------------------------------- hooks: advice, first guess

def _advised(w, answers, out_dir, **kw):
    """`out_dir`: the test's own - without one the phase writes below ~/.cache/shape_reco/features, the user's real
    cache, which on the institute servers is an NFS home over its quota (T-05, 2026-10-01)."""
    from reco_engine.advice import AdviceConfig, ScriptedAdviser
    cfg = _cfg(w, adviser=ScriptedAdviser(answers), advice=AdviceConfig(min_share=0.0, close_margin=0.5),
               feature_split=False, **kw)
    return list(rl._feature_phase(_rough(), cfg, "part.step", None, out_dir=out_dir))


def test_a_hard_piece_is_asked_about_and_the_answer_is_acted_on(tmp_path):
    """The piece's best guess scores 0.6 (below 0.85, a "close" miss at margin 0.5): the adviser is asked - the
    advice_asked step comes before the answer - and its answer is what happens: the chosen candidate is applied
    (still validated), "skip" rejects the piece with that reason, "stop" ends the phase."""
    w = _world(a=dict(centre=(10, 0, 0), volume=30.0, score=0.6))
    steps = _advised(w, [dict(choice="candidate", parameters=dict(index=0))], tmp_path / "1", feature_max_iterations=1)
    k = [s.kind for s in steps]
    assert k.index("advice_asked") < k.index("advice_given") < k.index("feature_accepted")
    asked = next(s.info for s in steps if s.kind == "advice_asked")
    assert asked["question"]["kind"] == "close" and asked["piece_id"] == 1 and asked["phase"] == "feature"
    acc = next(s.info for s in steps if s.kind == "feature_accepted")
    assert acc["source"].endswith("(chosen by the user)") and acc["validation"]["valid"]
    w2 = _world(a=dict(centre=(10, 0, 0), volume=30.0, score=0.6))
    steps2 = _advised(w2, [dict(choice="skip")], tmp_path / "2")
    assert next(s.info for s in steps2 if s.kind == "feature_rejected")["reason"] == "skipped by the user"
    w3 = _world(a=dict(centre=(10, 0, 0), volume=30.0, score=0.6), b=dict(centre=(20, 0, 0), volume=20.0, score=0.6))
    steps3 = _advised(w3, [dict(choice="stop")], tmp_path / "3")
    assert _done(steps3)["stopped"] == "stopped by the user"


def test_without_an_adviser_nothing_is_asked(tmp_path):
    w = _world(a=dict(centre=(10, 0, 0), volume=30.0, score=0.6))
    steps = list(rl._feature_phase(_rough(), _cfg(w, feature_split=False), "part.step", None, out_dir=tmp_path))
    assert not any(s.kind.startswith("advice") for s in steps)


@pytest.mark.slow                                                        # one loop run, ~15 s
def test_a_first_guess_is_judged_by_the_loops_own_checks():
    """The user's first guess: a right box add is kept (source "user", the start stage skipped) and a wrong cut -
    through material - is dropped with its reason, never forced."""
    pytest.importorskip("vtk")
    sys.path.insert(0, str(ROOT / "tests"))
    from test_add_cut_loop import _field, _cfg as loop_cfg
    from src import sdf_primitives as sp
    solid = sp.Box([8.0, 8.0, 5.0]) - sp.Placed(sp.Cylinder(3.0, 30.0), translation=np.array([0.0, 0.0, -15.0]))
    guess = [dict(op="add", kind="box", center_mm=[0.2, -0.1, 0.0], quaternion=[0, 0, 0, 1],
                  dims_mm=dict(x_half=8.1, y_half=7.9, z_half=5.0), refined=True),
             dict(op="cut", kind="box", center_mm=[6.0, 6.0, 0.0], quaternion=[0, 0, 0, 1],
                  dims_mm=dict(x_half=1.5, y_half=1.5, z_half=6.0), refined=True)]
    cfg, _ = loop_cfg(first_guess=guess)
    steps = list(rl.residual_steps(_field(solid), cfg))
    user = [s for s in steps if s.info.get("source") == "user"]
    assert [s.kind for s in user] == ["candidate", "accepted", "candidate", "rejected"]
    assert not any(s.kind == "add" for s in steps)                    # no start stage / first add of its own
    ops = steps[-1].result.operations
    assert ops[0].op == "add" and any(src.mode == "user" for src in ops[0].primitive.sources)
    assert any(o.op == "cut" and o.primitive.kind == "cylinder" for o in ops)      # the loop found the bore itself


def test_piece_feedback_of_an_earlier_run_is_used_and_reported(tmp_path):
    """reco_gui's piece_feedback.json beside pieces.json: the pieces of this run matched by sign, volume and centre;
    "leave" and "not_a_feature" skip the piece with that reason, a "feature_feedback" step says where a human decided,
    a piece without feedback goes on as usual."""
    import json
    w = _world(a=dict(centre=(10, 0, 0), volume=30.0), b=dict(centre=(20, 0, 0), volume=20.0),
               c=dict(centre=(30, 0, 0), volume=10.0))
    fb = tmp_path / "run" / "iteration_1" / "piece_feedback.json"
    fb.parent.mkdir(parents=True)
    fb.write_text(json.dumps({"1": dict(verdict="leave", kind=None, note="a casting skin",
                                        piece=dict(number=1, sign="minus", volume=30.0, centre=[10, 0, 0])),
                              "2": dict(verdict="not_a_feature", piece=dict(number=2, sign="minus", volume=20.0000001,
                                                                             centre=[20, 0, 0]))}))
    steps, _, info = _phase(w, piece_feedback=str(tmp_path / "run"), feature_max_iterations=1, out=tmp_path)
    used = [s.info for s in steps if s.kind == "feature_feedback"]
    assert [(u["piece_id"], u["verdict"]) for u in used] == [(1, "leave"), (2, "not_a_feature")]
    assert used[0]["message"] == "piece #1: feedback used - leave (a casting skin)"
    rej = {s.info["piece_id"]: s.info["reason"] for s in steps if s.kind == "feature_rejected"}
    assert rej == {1: "left (feedback)", 2: "not a feature (feedback)"}
    assert [s.info["piece_id"] for s in steps if s.kind == "feature_accepted"] == [3]


# ---------------------------------------------------------------------------------------------- the exact checks

def _with_steps(world):
    """The world's difference with a STEP path per piece: only then the phase checks a feature against its piece."""
    plain = world.difference

    def difference(*a):
        out = plain(*a)
        for p in out.get("pieces") or []:
            p["step"] = f"{p['id']}.step"
        return out
    world.difference = difference
    return world


def test_a_failed_check_call_is_said_and_a_failed_item_has_no_columns(tmp_path):
    """The checks are advisory - but never silent (2026-09-29: a syntax error of the check script failed every call
    and nothing said so): a call that fails as a whole is counted and named in feature_done; an item whose booleans
    disagree keeps its feature without the columns, the others get theirs."""
    w = _with_steps(_world(a=dict(centre=(10, 0, 0)), b=dict(centre=(20, 0, 0), volume=5.0)))
    steps, _, info = _phase(w, feature_checks=lambda path, items, timeout: dict(error="failed: no FreeCAD"), out=tmp_path)
    assert info["applied"] == 2 and info["check_failures"] == 1 and info["check_error"] == "failed: no FreeCAD"
    assert "1 FreeCAD check call(s) failed" in steps[-1].info["message"]

    def checks(path, items, timeout):
        assert [it["piece_step"] for it in items] == ["a.step", "b.step"]
        return {items[0]["id"]: dict(id=items[0]["id"], error="feature check: the booleans disagree"),
                items[1]["id"]: dict(id=items[1]["id"], feature_volume=6.0, piece_volume=5.0, common_volume=5.0,
                                     beyond_piece_volume=1.0, left_of_piece_volume=0.0, harm_volume=0.0)}
    w = _with_steps(_world(a=dict(centre=(10, 0, 0)), b=dict(centre=(20, 0, 0), volume=5.0)))
    steps, _, info = _phase(w, feature_checks=checks, out=tmp_path)
    acc = {s.info["piece_id"]: s.info["validation"] for s in steps if s.kind == "feature_accepted"}
    assert info["applied"] == 2 and info["check_failures"] == 0
    assert acc[1]["common_volume"] is None and acc[1]["left_of_piece_volume"] is None
    assert acc[2]["common_volume"] == 5.0 and acc[2]["left_of_piece_volume"] == 0.0


# ---------------------------------------------------------------------------------------------- the pattern rule

def test_a_failed_representative_does_not_condemn_its_group(tmp_path):
    """The representative's guess fails, the second member's passes: that tool serves the rest of the group by the
    pattern transfer - and the representative, which is not guessed again (liner run 20260928-220854: 35 of 45
    rejections were members of a group whose representative had failed, none of them tried)."""
    w = _box_world(tmp_path, n_groups=1, copies=4)
    guess = w.guess

    def first_fails(piece, *a):                                          # whichever copy leads the group
        w.pieces[piece["id"]]["score"] = 0.3 if not w.guessed else 0.95
        return guess(piece, *a)
    steps, result, info = _phase(w, feature_min_share=1e-6, feature_split=False, feature_guess=first_fails, out=tmp_path)
    assert len(w.guessed) == 2 and info["group_retries"] == 1
    pattern = [(s.info["piece_id"], s.info["source"]) for s in steps
               if s.kind == "feature_candidate" and s.info["source"].startswith("pattern:")]
    assert pattern == [(1, "pattern:2"), (3, "pattern:2"), (4, "pattern:2")]
    assert sorted(s.info["piece_id"] for s in steps if s.kind == "feature_accepted") == [1, 2, 3, 4]
    assert info["applied"] == 4 and info["rejected"] == 0 and info["voxel_fallbacks"] == 0
    assert info["stopped"] == "converged"


def test_a_hopeless_group_costs_k_full_guesses_and_ends_as_voxels(tmp_path):
    """No member finds a tool: feature_group_tries members are guessed in full, the others are not; every piece then
    gets its own voxel boxes - inside the piece, under the limit of boxes, the step naming what was tried before and
    what is left - and the program marks them as the fallback."""
    w = _box_world(tmp_path, n_groups=1, copies=5)
    for p in w.pieces.values():
        p["score"] = 0.3
    steps, result, info = _phase(w, feature_min_share=1e-6, feature_split=False, feature_group_tries=2,
                                 feature_max_iterations=1, out=tmp_path)
    assert len(w.guessed) == 2 and info["group_retries"] == 1
    rej = {s.info["piece_id"]: s.info["reason"] for s in steps if s.kind == "feature_rejected"}
    assert rej[1] == rej[2] == "best score 0.300 < 0.85"
    assert rej[3] == rej[4] == rej[5] == "pattern: 2 congruent piece(s) (#1, #2) were guessed in full and found no tool"
    fb = {s.info["piece_id"]: s.info for s in steps if s.kind == "feature_candidate" and s.info["source"] == "voxel fallback"}
    assert sorted(fb) == [1, 2, 3, 4, 5]
    v = w.pieces["g0c0"]["volume"]                                       # congruent: one volume
    for pid, c in fb.items():
        f = c["fallback"]
        assert 1 <= f["boxes"] <= f["max_boxes"] == 16 and not f["relaxed"] and f["pitch"] <= f["licensed_pitch"]
        assert f["precision"] == 1.0 and 0.9 < f["recall"] < 1.0 and c["tool_source"] == "fallback"
        assert f["covered_volume"] + f["left_volume"] == pytest.approx(v) and f["left_volume"] > 0
        assert "tried before: " in c["message"] and f"left {f['left_volume']:.4g}" in c["message"]
    assert [r["solution"] for r in fb[1]["fallback"]["tried"]] == ["fit", "cascade"]
    assert fb[1]["fallback"]["tried"][0]["score"] == 0.3 and fb[1]["fallback"]["rejected_before"] == rej[1]
    assert fb[4]["fallback"]["tried"] == [dict(solution="the full guesses of its congruence group", members=[1, 2])]
    acc = [s.info for s in steps if s.kind == "feature_accepted"]
    assert sorted(a["piece_id"] for a in acc) == [1, 2, 3, 4, 5]
    assert all(a["fallback"]["left_volume"] > 0 and a["tool_source"] == "fallback" and "FALLBACK" in a["message"] for a in acc)
    assert info["voxel_fallbacks"] == 5 and info["voxel_fallbacks_applied"] == 5 and info["applied"] == 5
    assert info["rejected"] == 0 and info["tools_by_source"] == {"fallback": 5} and info["store_size"] == 0
    for op in result.operations[1:]:
        assert op.op == "cut" and "voxel fallback" in op.primitive.label
        assert any(src.mode == "voxel_fallback" for src in op.primitive.sources)
        assert any(src.mode == "feature:voxel fallback" for src in op.primitive.sources)


def test_without_the_fallback_a_hopeless_group_is_rejected(tmp_path):
    w = _box_world(tmp_path, n_groups=1, copies=5)
    for p in w.pieces.values():
        p["score"] = 0.3
    steps, _, info = _phase(w, feature_min_share=1e-6, feature_split=False, feature_voxel_fallback=False, out=tmp_path)
    assert len(w.guessed) == 3 and info["rejected"] == 5 and info["applied"] == 0 and info["voxel_fallbacks"] == 0
    assert not any(s.info.get("source") == "voxel fallback" for s in steps)


def test_the_split_comes_before_the_voxels_and_a_fallback_that_covers_too_little_is_rejected(tmp_path):
    """The compound piece is split first (its segments pass: no fallback); with a split that finds nothing and a
    fallback that must cover more than it can, the piece is rejected and the reason names the split's segments."""
    pytest.importorskip("vtk")
    w, fld = _compound_world(tmp_path)
    steps = list(rl._feature_phase(_rough(), _cfg(w, feature_max_iterations=1), "part.step", None, field=None,
                                   out_dir=tmp_path))
    assert _done(steps)["voxel_fallbacks"] == 0 and _done(steps)["segments_accepted"] == 3
    w, fld = _compound_world(tmp_path)
    steps = list(rl._feature_phase(_rough(), _cfg(w, feature_max_iterations=1, feature_min_recall=0.95,
                                                  feature_fallback_min_recall=0.999), "part.step", None, field=None,
                                   out_dir=tmp_path))
    k = _kinds(steps)
    assert k.index("feature_split") < len(k) - 2 and _done(steps)["voxel_fallbacks"] == 1
    last = [s.info for s in steps if s.kind == "feature_rejected" and s.info["piece_id"] == 1][-1]
    assert last["reason"].startswith("voxel fallback: ") and "under 0.999; tried before: fit cylinder score 0.300" in last["reason"]
    assert "split (" in last["reason"] and "segment #2" in last["reason"]
    assert _done(steps)["rejected"] == 4 and _done(steps)["applied"] == 0      # the piece once, its three segments


# ---------------------------------------------------------------------------------------------- small pieces

def _wall(radius, origin="input"):
    return dict(type="cylinder", origin=origin, radius=radius, area=1.0)


def test_a_small_piece_with_a_real_wall_is_a_feature_by_its_own_size(tmp_path):
    """Under the volume share (1e-3 of 1000: 1.0) a piece is kept when it has a real curved face of radius r and at
    least pi r^3 of volume (the liner's holes R 2.459 x 15: 293.9 mm3 under a floor of 459, 19.8 x pi r^3): the bore
    of radius 0.3 is, the sliver without a real face, the shallow dent of radius 1 and the bore whose wall is the
    model's are not."""
    from reco_engine.feature_stage import feature_radius, feature_sized
    w = _world(big=dict(centre=(10, 0, 0), volume=40.0),
               bore=dict(centre=(20, 0, 0), volume=0.5, faces=[_wall(0.3), _wall(0.4), dict(type="plane", origin="model")]),
               sliver=dict(centre=(30, 0, 0), volume=0.5, faces=[dict(type="plane", origin="input")]),
               dent=dict(centre=(40, 0, 0), volume=0.5, faces=[_wall(1.0)]),
               models=dict(centre=(50, 0, 0), volume=0.5, faces=[_wall(0.3, "model")]))
    assert feature_radius(w.pieces["bore"]) == 0.3 and feature_radius(w.pieces["models"]) is None
    assert feature_sized(w.pieces["dent"], 1.0) is None and feature_sized(w.pieces["bore"], 0.0) is None
    steps, _, info = _phase(w, feature_max_iterations=1, out=tmp_path)
    first = steps[0].info
    assert [p["volume"] for p in first["pieces"]] == [40.0, 0.5] and first["dropped_small"] == 3
    assert [(q["piece"], q["reason"][:29]) for q in first["kept_small"]] == [("bore", "a real face of radius 0.3 and")]
    assert "1 under the share kept: a real face of radius 0.3" in first["message"]
    assert info["applied"] == 2
    steps, _, info = _phase(w := _world(**{k: dict(v) for k, v in w.pieces.items()}), feature_max_iterations=1,
                            feature_min_radius_depth=0.0, out=tmp_path)
    assert steps[0].info["dropped_small"] == 4 and steps[0].info["kept_small"] == [] and info["applied"] == 1


def test_a_small_piece_known_to_the_session_is_not_dropped(plate_pieces, tmp_path):
    """The plate's pieces under a share no piece reaches: with the store of a run that cut them, the two holes are
    kept for the size of the session's countersunk hole and the blob for its congruence with the session's created
    tool; with an empty store all three are too small."""
    from reco_engine.tools import ToolStore
    pieces, _ = plate_pieces
    store = ToolStore()
    _typed_run(TypedWorld(pieces), store, tmp_path)
    assert len(store) == 2

    def run(st, out):
        w = TypedWorld(pieces)
        cfg = rl.ResidualConfig(feature_phase=True, feature_build=w.build, feature_difference=w.difference,
                                feature_guess=w.guess, feature_min_share=0.5, feature_min_radius_depth=0.0,
                                feature_tool_store=st, feature_check=False, feature_max_iterations=1)
        return list(rl._feature_phase(_rough(), cfg, "plate.step", None, field=None, out_dir=out))
    first = run(store, tmp_path / "known")[0].info
    reasons = sorted(q["reason"] for q in first["kept_small"])
    assert reasons == ["congruent with the session's tool created:box - cylinder#1",
                       "the size of the session's tool countersunk_hole#1",
                       "the size of the session's tool countersunk_hole#1"]
    assert first["dropped_small"] == 0 and len(first["pieces"]) == 3
    first = run(ToolStore(), tmp_path / "unknown")[0].info
    assert first["dropped_small"] == 3 and first["kept_small"] == [] and first["pieces"] == []


def test_a_first_add_taken_from_the_alternatives_does_not_refer_to_itself():
    """The cover may choose one of the cascade's alternatives: its own list of alternatives must not lead back to it
    (the ABC pilot of 2026-09-29: the result could not be written, RecursionError)."""
    import inspect
    src = inspect.getsource(rl._first_add)
    assert "if x is not chosen" in src and "replace(x, alternatives=[])" in src
    from dataclasses import replace
    a, b = Primitive(kind="box"), Primitive(kind="cylinder")
    a.alternatives = [b]
    others = [a] + list(a.alternatives)
    b.alternatives = [replace(x, alternatives=[]) for x in others if x is not b]
    from dataclasses import asdict
    assert [x["kind"] for x in asdict(b)["alternatives"]] == ["box"]


# ------------------------------------------------------------------------------- sub-parts before the voxels

def _plane_faces(mesh):
    """The plane faces of a mesh of flat facets, as the exact faces name them (type, normal, point, area)."""
    return [dict(type="plane", normal=[float(v) for v in mesh.face_normals[f[0]]],
                 point=[float(v) for v in mesh.triangles[f[0]][0]], area=float(a), origin="input")
            for f, a in zip(mesh.facets, mesh.facets_area)]


def test_an_angle_piece_becomes_two_boxes_that_hold_it_and_nothing_else():
    """A piece no single primitive explains - an angle, tilted and off-centre: the sub-part tool covers it with two
    boxes, all of the piece inside the tool and all of the tool inside the piece; the single box holds a third more."""
    import trimesh
    from reco_engine import feature_stage as fs
    ring = np.array([(0, 0), (8, 0), (8, 2), (2, 2), (2, 7), (0, 7), (0, 2)], float)
    flat = [(0, 1, 2), (0, 2, 3), (0, 3, 6), (6, 3, 4), (6, 4, 5)]
    n = len(ring)
    V = np.vstack([np.c_[ring, np.zeros(n)], np.c_[ring, np.full(n, 3.0)]])
    F = [(a, c, b) for a, b, c in flat] + [(a + n, b + n, c + n) for a, b, c in flat]
    for k in range(n):
        m = (k + 1) % n
        F += [(k, m, m + n), (k, m + n, k + n)]
    outline = trimesh.Trimesh(V @ _rotation([1, 2, 3], 25).T + [40.0, -15.0, 7.0], np.array(F), process=False)
    assert outline.is_watertight and outline.volume == pytest.approx(78.0)
    piece = dict(faces=_plane_faces(outline))
    r = fs.subpart_tool(piece, outline.vertices, outline.faces)
    assert r["parts"] == 2 and r["kinds"] == ["box", "box"] and r["primitive"].kind == "composite"
    assert r["precision"] > 0.995 and r["recall"] > 0.995 and r["containment"] > 0.995
    one = fs.subpart_tool(piece, outline.vertices, outline.faces, max_parts=1)
    assert one["parts"] == 1 and one["recall"] > 0.995 and one["precision"] == pytest.approx(26.0 / 56.0, abs=0.02)


def test_a_hopeless_piece_is_covered_by_sub_parts_before_the_voxels(tmp_path):
    """feature_subparts: the pieces no guess explains get the sub-part cover first; it holds them exactly here (boxes),
    so no voxel fallback runs, and the program names the source."""
    w = _box_world(tmp_path, n_groups=1, copies=3)
    for p in w.pieces.values():
        p["score"] = 0.3
    steps, result, info = _phase(w, feature_min_share=1e-6, feature_split=False, feature_group_tries=2,
                                 feature_max_iterations=1, feature_subparts=True, out=tmp_path)
    sub = {s.info["piece_id"]: s.info for s in steps if s.kind == "feature_candidate" and s.info["source"] == "sub-parts"}
    assert sorted(sub) == [1, 2, 3]
    for c in sub.values():
        assert c["subparts"]["parts"] == 1 and c["subparts"]["precision"] > 0.999 and c["tool_source"] == "sub-parts"
    assert info["subparts"] == 3 and info["subparts_applied"] == 3 and info["voxel_fallbacks"] == 0
    assert info["applied"] == 3 and info["tools_by_source"] == {"sub-parts": 3}
    for op in result.operations[1:]:
        assert op.op == "cut" and any(src.mode == "feature:sub-parts" for src in op.primitive.sources)


# ------------------------------------------------------------------------------- applied at once, confirmed later

def _checks_all_good(path, items, timeout):
    return {it["id"]: dict(id=it["id"], feature_volume=6.0, piece_volume=6.0, common_volume=6.0, beyond_piece_volume=0.0,
                           left_of_piece_volume=0.0, harm_volume=0.0) for it in items}


def test_an_accepted_feature_is_applied_at_once_and_confirmed_by_one_build(tmp_path):
    """feature_apply_at_once: every feature is in the program right after its own check, before any build of the
    whole model; one build and one difference confirm them together."""
    w = _with_steps(_world(a=dict(centre=(10, 0, 0)), b=dict(centre=(20, 0, 0), volume=5.0)))
    steps, result, info = _phase(w, feature_checks=_checks_all_good, feature_apply_at_once=True, out=tmp_path)
    k = _kinds(steps)
    acc = [i for i, x in enumerate(k) if x == "feature_accepted"]
    assert len(acc) == 2 and k.count("feature_confirm") == 1 and k.count("feature_confirmed") == 1
    assert acc[-1] < k.index("feature_confirm") < k.index("feature_confirmed")
    first = [s for s in steps if s.kind == "feature_accepted"][0]
    assert first.info["validation"]["method"] == "feature check" and "applied at once" in first.info["message"]
    assert len(first.result.operations) == 2                             # the rough add and this feature, already
    conf = [s for s in steps if s.kind == "feature_confirmed"][0].info
    assert conf["confirmed"] and conf["features"] == 2
    assert w.builds == 2 and info["applied"] == 2 and info["rejected"] == 0   # the rough model, the confirmation
    assert len(result.operations) == 3


def test_a_confirmation_that_fails_takes_the_features_back_and_validates_them_in_halves(tmp_path):
    w = _with_steps(_world(a=dict(centre=(10, 0, 0)), b=dict(centre=(20, 0, 0), volume=5.0)))
    build, calls = w.build, []

    def first_confirmation_fails(result, out_dir, name, timeout):
        calls.append(name)
        b = build(result, out_dir, name, timeout)
        return dict(b, valid=False) if len(calls) == 2 else b             # the build of both features together
    steps, result, info = _phase(w, feature_checks=_checks_all_good, feature_apply_at_once=True,
                                 feature_build=first_confirmation_fails, out=tmp_path)
    conf = [s.info for s in steps if s.kind == "feature_confirmed"]
    assert conf[0]["confirmed"] is False and "not valid" in conf[0]["reason"]
    assert info["applied"] == 2 and len(result.operations) == 3           # each passed on its own
    assert len([s for s in steps if s.kind == "feature_accepted"]) == 2 + 2


def test_a_check_that_disagrees_with_the_sampled_score_is_no_verdict(tmp_path):
    """The check's boolean gives no shared volume for a tool the sample scores 0.95 (the liner's pocket strip,
    2026-09-30): the feature is not rejected there, the batch's build and difference accept it."""
    def checks(path, items, timeout):
        out = _checks_all_good(path, items, timeout)
        out[items[0]["id"]].update(common_volume=0.0, beyond_piece_volume=6.0, left_of_piece_volume=6.0)
        return out
    w = _with_steps(_world(a=dict(centre=(10, 0, 0))))
    steps, result, info = _phase(w, feature_checks=checks, feature_apply_at_once=True, out=tmp_path)
    acc = [s.info for s in steps if s.kind == "feature_accepted"]
    assert len(acc) == 1 and acc[0]["validation"]["method"] != "feature check"
    assert info["applied"] == 1 and info["rejected"] == 0 and "feature_confirm" not in _kinds(steps)


def test_a_detailed_last_piece_gets_its_best_candidate_as_a_rough_guess(tmp_path):
    """The user's rule of 2026-10-29: the last pieces are often too detailed for any tool - rough guess them. A piece
    whose best candidate misses the score (0.8 < 0.85) but is precise enough for a rough guess (0.95 >= 0.9) gets it,
    marked rough, before the voxel fallback; it is applied because the exact residual drops by the whole piece. A piece
    whose best candidate is too imprecise for a rough guess (0.8) goes on to the voxel fallback / is rejected."""
    w = _world(detailed=dict(centre=(10, 0, 0), volume=30.0, score=0.8, precision=0.95, recall=0.85),
               hopeless=dict(centre=(20, 0, 0), volume=20.0, score=0.8, precision=0.8, recall=0.85))
    steps, result, info = _phase(w, feature_split=False, feature_voxel_fallback=False, feature_max_iterations=1, out=tmp_path)
    rough = [s.info for s in steps if s.kind == "feature_candidate" and s.info.get("rough")]
    assert [r["piece_id"] for r in rough] == [1] and "rough guess for piece #1" in rough[0]["message"]
    assert "drops by 50% of the piece (15)" in rough[0]["message"]
    acc = {s.info["piece_id"]: s.info for s in steps if s.kind == "feature_accepted"}
    assert list(acc) == [1] and acc[1]["source"] == "rough:fit"
    assert any(src.mode == "feature:rough:fit" for src in result.operations[-1].primitive.sources)
    rej = {s.info["piece_id"]: s.info["reason"] for s in steps if s.kind == "feature_rejected"}
    assert 2 in rej and info["applied"] == 1
    w2 = _world(detailed=dict(centre=(10, 0, 0), volume=30.0, score=0.8, precision=0.95, recall=0.85))
    steps, _, info = _phase(w2, feature_split=False, feature_voxel_fallback=False, feature_rough=False,
                            feature_max_iterations=1, out=tmp_path)
    assert info["applied"] == 0 and not any(s.info.get("rough") for s in steps)       # the switch


def test_a_rough_guess_may_not_trade_the_part_for_its_piece(tmp_path):
    """Protocol rough_tidy_v1, DEV (2026-10-29): an over-covered piece 5x the part's volume got a box | sphere cut that
    removed all of the part - the residual still dropped by half the piece (5000 -> 1000). The rough guess may remove
    at most feature_rough_max_harm_input (1 %) of the input; the validation says so even without the FreeCAD check's
    numbers. A rough guess that costs 0.5 % of the part is applied."""
    w = _world(giant=dict(centre=(10, 0, 0), volume=5 * V_IN, destroys=V_IN, score=0.8, precision=0.95, recall=0.85),
               small=dict(centre=(20, 0, 0), volume=30.0, destroys=0.005 * V_IN, score=0.8, precision=0.95, recall=0.85))
    steps, result, info = _phase(w, feature_split=False, feature_voxel_fallback=False, feature_max_iterations=1,
                                 out=tmp_path)
    rej = {s.info["piece_id"]: s.info["reason"] for s in steps if s.kind == "feature_rejected"}
    assert "rough guess: removes 1000 mm3 of the part, at most 1% of the input (10)" in rej[1], rej    # the giant
    acc = {s.info["piece_id"]: s.info for s in steps if s.kind == "feature_accepted"}
    assert list(acc) == [2] and acc[2]["source"] == "rough:fit" and info["applied"] == 1             # the small one
    steps, _, info = _phase(w, feature_split=False, feature_voxel_fallback=False, feature_max_iterations=1,
                            feature_rough_max_harm_input=1.5, out=tmp_path)
    assert info["applied"] == 2                                          # the old rule took it


def test_a_refused_rough_guess_leaves_the_piece_its_voxel_fallback(tmp_path):
    """Protocol rough_tidy_v2, DEV (2026-10-29): aegir_n6_I_000121's rough guess was refused (it would have removed the
    part), and the piece then got nothing - the round applied nothing and the phase stopped (exact IoU 0.160 against 0.531
    without the guess). A refused rough guess now leaves the piece its last resort in the same round."""
    w = _box_world(tmp_path, n_groups=1, copies=1)
    p = w.pieces["g0c0"]
    p.update(score=0.8, precision=0.95, recall=0.9, destroys=V_IN, destroys_mode="feature:rough")   # the guess destroys
    steps, result, info = _phase(w, feature_min_share=1e-6, feature_split=False, feature_max_iterations=1, out=tmp_path)
    assert any(s.kind == "feature_candidate" and s.info.get("rough") for s in steps)
    rej = [s.info["reason"] for s in steps if s.kind == "feature_rejected"]
    assert any("exact residual drops by -" in r or r.startswith("rough guess: removes") for r in rej), rej   # refused
    acc = [s.info for s in steps if s.kind == "feature_accepted"]
    assert len(acc) == 1 and acc[0]["source"] == "voxel fallback" and info["voxel_fallbacks_applied"] == 1, (acc, rej)
