"""
The torus guard (reco_engine/torus_guard.py, T-29): its three rules on the tori of the stored programs that broke
them (spindle R 123 / r 1410, horn R = r = 2 of the liner, near-horn R 2753 / r 2750, oversized R 315 / r 2.4 on a
part of diagonal 86) and on tori that pass, and every place that proposes or accepts a torus taking its normal path
for a rejected candidate: the cascade (the branch has no answer), the loop's evaluation (fails, with the reason),
the feature sources (not offered, the reason under `failed`) and the sub-part cover (not this type's solid). The one
repair: a rounded corner's horn face (the liner's slot ends) gets the thinned torus inside it; a spindle corner stays
as the box made it.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine import torus_guard as tg  # noqa: E402
from reco_engine.result_schema import Primitive  # noqa: E402


def torus(R, r, centre=(0.0, 0.0, 0.0)):
    return Primitive(kind="torus", center_mm=tuple(centre), dims_mm=dict(major_radius=float(R), minor_radius=float(r)))


# ------------------------------------------------------------------------------------------------ the rules

@pytest.mark.parametrize("R, r, diag, rule", [
    (123.144, 1410.492, 341.4, "spindle/horn"),          # known_00001964_001
    (2.0, 2.0, 1648.2, "spindle/horn"),                  # the liner's slot ends (horn)
    (2752.968, 2749.982, 36.1, "near-horn"),             # known_00002029_003: near-horn before oversized
    (315.257, 2.420, 86.37, "oversized"),                # known_00003444_002
    (16.45, 5.0, 133.8, None),                           # known_00000424_002, a real torus face
    (104.093, 2.499, 301.6, None),                       # known_00007991_014: a ring part, one torus
    (293.1, 25.13, 283.4, None),                         # the largest real torus face of the ABC census (1.12)
    (6.939, 6.35, 46.0, None),                           # the thinnest-hole real torus face of the census (0.085)
])
def test_rules_on_stored_tori(R, r, diag, rule):
    found = tg.torus_defect(R, r, diag)
    assert (found.rule if found else None) == rule
    if found:
        assert found.text and str(found) == found.text


def test_boundaries_and_degenerate_input():
    R = 10.0
    assert tg.torus_defect(R, R * (1 - 1.001 * tg.NEAR_HORN_GAP)) is None          # just at the gap: admitted
    assert tg.torus_defect(R, R * (1 - 0.999 * tg.NEAR_HORN_GAP)).rule == "near-horn"
    assert tg.torus_defect(R, R + 1e-12).rule == "spindle/horn"
    assert tg.torus_defect(10.0, 2.0, 12.0 / tg.OVERSIZE_RATIO * 1.001) is None      # just at k x diagonal: admitted
    assert tg.torus_defect(10.0, 2.0, 12.0 / tg.OVERSIZE_RATIO * 0.999).rule == "oversized"
    assert tg.torus_defect(1e4, 2.0) is None                                         # no diagonal: no size rule
    assert tg.torus_defect(1e4, 2.0, None) is None and tg.torus_defect(1e4, 2.0, 0.0) is None
    for R_, r_ in ((5.0, 0.0), (float("nan"), 1.0), (5.0, float("inf"))):
        assert tg.torus_defect(R_, r_).rule == "degenerate"
    assert tg.torus_defect(-1.0, 1.0).rule == "spindle/horn"                       # R <= 0 < r: no ring


def test_primitive_defect_finds_a_torus_anywhere_in_a_composite():
    box = Primitive(kind="box", dims_mm=dict(x_half=1.0, y_half=1.0, z_half=1.0))
    corner = Primitive(kind="composite", parts=[box, torus(2.0, 2.0)], combine="cut")
    tool = Primitive(kind="composite", parts=[box, Primitive(kind="composite", parts=[corner, box], combine="cut")],
                     combine="cut")
    assert tg.primitive_defect(tool).rule == "spindle/horn"
    assert len(tg.tori(tool)) == 1 and not tg.admissible(tool) and tg.admissible(box)
    assert tg.primitive_defect(box) is None and tg.primitive_defect(None) is None
    ok = Primitive(kind="composite", parts=[box, torus(20.0, 10.0)], combine="cut")
    assert tg.primitive_defect(ok, 100.0) is None and tg.primitive_defect(ok, 10.0).rule == "oversized"


def test_program_tori_walk_a_program_tree():
    leaf = lambda R, r: dict(type="Placed", primitive=dict(type="Torus", major=R, minor=r), rotation=np.eye(3).tolist(),
                             translation=[0, 0, 0])
    box = dict(type="Placed", primitive=dict(type="Box", half=[1, 1, 1]))
    tree = dict(type="Subtraction", a=dict(type="Union", a=box, b=leaf(20.0, 10.0)), b=leaf(2.0, 2.0))
    assert list(tg.program_tori(tree)) == [(20.0, 10.0, False), (2.0, 2.0, True)]


# ------------------------------------------------------------------------------------------------ the cascade

def _ring_target(resolution=32, half=12.0):
    from reco_engine import cascade
    from src.field_channels import voxel_channels
    spec = cascade.grid_spec(-half, half, resolution)
    c = spec.centres()
    rho = np.hypot(c[..., 0], c[..., 1])
    occ = np.hypot(rho - 6.0, c[..., 2]) <= 2.5
    return voxel_channels(occ, spec), spec


class TypeNet:
    def __init__(self, answer):
        self.answer = answer

    def __call__(self, x):
        return self.answer


def _fake_torus_fit(R, r):
    def fit(samples):
        return SimpleNamespace(kind="torus", frame=np.eye(3), centre=np.zeros(3), half=np.array([R, r]), u=None, v=None)
    return fit


def test_cascade_drops_a_guarded_torus_branch(monkeypatch):
    from reco_engine import cascade
    from reco_engine.cascade import _Entry, run_cascade
    from reco_field import fit as fitmod
    ch, spec = _ring_target()
    reg = [_Entry("net", "type", ("torus", "box"), "cnn", cost=1, gate=0.99)] + \
          [e for e in cascade.DEFAULT_REGISTRY if e.stage != "type"]
    models = {"net": TypeNet([("torus", 0.9), ("box", 0.1)])}
    # a fit that finds the ring: the torus branch wins
    monkeypatch.setitem(fitmod.FITTERS, "torus", _fake_torus_fit(6.0, 2.5))
    p = run_cascade(ch, spec, reg, models=models, top_k=2, pass_iou=0.3)
    assert p.kind == "torus" and p.dims_mm["major_radius"] == pytest.approx(6.0)
    # the same ring fitted as a spindle (r >= R): that mode has no answer, the branch is gone, the box remains
    monkeypatch.setitem(fitmod.FITTERS, "torus", _fake_torus_fit(2.5, 6.0))
    q = run_cascade(ch, spec, reg, models=models, top_k=2, pass_iou=0.3)
    kinds = [q.kind] + [a.kind for a in q.alternatives]
    assert "torus" not in kinds and "box" in kinds
    # oversized against the caller's part diagonal (the grid's own diagonal admits it)
    monkeypatch.setitem(fitmod.FITTERS, "torus", _fake_torus_fit(6.0, 2.5))
    r = run_cascade(ch, spec, reg, models=models, top_k=2, pass_iou=0.3, part_diagonal=5.0)
    assert "torus" not in [r.kind] + [a.kind for a in r.alternatives]


def test_context_admit_records_the_reason():
    from reco_engine import cascade
    ch, spec = _ring_target(resolution=16)
    ctx = cascade._Context(ch, spec, None, None, None, part_diagonal=40.0)
    assert ctx.admit(torus(6.0, 2.5)) is not None and ctx.guarded == []
    assert ctx.admit(torus(2.0, 2.0)) is None and "horn" in ctx.guarded[0]
    assert ctx.admit(torus(100.0, 2.0)) is None and "oversized" in ctx.guarded[1]
    box = Primitive(kind="box", dims_mm=dict(x_half=1.0, y_half=1.0, z_half=1.0))
    assert ctx.admit(box) is box and ctx.admit(None) is None
    # without a caller's diagonal: the grid's (no field)
    ctx2 = cascade._Context(ch, spec, None, None, None)
    assert ctx2.diagonal() == pytest.approx(math.sqrt(3) * 24.0, rel=0.1)


# ------------------------------------------------------------------------------------------------ the loop

def test_loop_evaluation_rejects_a_guarded_torus_with_its_reason():
    from reco_engine import cascade
    from reco_engine import residual_loop as rl
    spec = cascade.box_grid_spec((-10.0, -10.0, -4.0), (10.0, 10.0, 4.0), 0.5)
    c = spec.centres()
    occ = (np.abs(c[..., 0]) <= 8) & (np.abs(c[..., 1]) <= 8) & (np.abs(c[..., 2]) <= 3)
    t_in = np.stack([occ.astype(np.float32), np.where(occ, 1.0, -1.0).astype(np.float32)] +
                    [np.zeros(occ.shape, np.float32)] * 3)
    st = rl._State(t_in, spec)
    assert st.part_diagonal == pytest.approx(math.sqrt(16 ** 2 + 16 ** 2 + 6 ** 2) + 0.0, abs=1.0)
    for bad, rule in ((torus(2.0, 3.0), "spindle/horn"), (torus(5.0, 4.99), "near-horn"),
                      (torus(40.0, 2.0), "oversized")):
        ev = rl._evaluate(st, bad, "add", 0.9)
        assert not ev["ok"] and ev["gain"] == 0 and ev["reason"].startswith("torus guard: ") and ev["guard"] == rule
    ok = rl._evaluate(st, torus(5.0, 2.0), "add", 0.9)
    assert "torus guard" not in ok["reason"]


# ------------------------------------------------------------------------------------------------ the feature phase

def test_thinned_lies_inside_and_keeps_the_outer_equator():
    for R, r in ((2.0, 2.0), (10.0, 9.95), (2753.0, 2750.0)):
        R2, r2 = tg.thinned(R, r)
        assert R2 >= R and r2 <= r and R2 + r2 == pytest.approx(R + r) and tg.torus_defect(R2, r2) is None
        assert (R2 - r2) / R2 == pytest.approx(tg.REPAIR_GAP)
        rng = np.random.default_rng(0)                              # points of the thinned tube (rho, z) ...
        a, d = rng.random(20000) * 2 * np.pi, np.sqrt(rng.random(20000)) * r2
        rho, z = R2 + d * np.cos(a), d * np.sin(a)
        assert (np.hypot(rho - R, z) <= r * (1 + 1e-12)).all()      # ... all in the given one
        assert R2 - R <= tg.REPAIR_GAP / (2 - tg.REPAIR_GAP) * R * (1 + 1e-9)    # the fillet moves by ~0.0101 R at most
    assert tg.thinned(2.0, 2.5) is None                             # a spindle: no repair
    assert tg.thinned(16.45, 5.0) is None                           # admissible as it is
    assert tg.thinned(315.0, 2.4) is None                           # oversized is no shape defect here


def _corner_face(M, m):
    n_out = [1.0 / math.sqrt(3)] * 2 + [-1.0 / math.sqrt(3)]      # the face looks out of the corner (+x, +y, -z)
    return dict(type="torus", axis=[0.0, 0.0, 1.0], major_radius=M, minor_radius=m,
                center=[10.0 - (M + m), 10.0 - (M + m), -5.0 + m], normal=n_out)


def test_rounded_corners_thin_a_horn_corner_and_leave_a_spindle_one():
    from reco_engine import feature_stage as fs
    box = Primitive(kind="box", dims_mm=dict(x_half=10.0, y_half=10.0, z_half=5.0))
    guard = {}
    tool, n = fs.rounded_corners(box, box, [_corner_face(3.0, 2.0)], guard=guard)
    assert n == 1 and [t.dims_mm for t in tg.tori(tool)] == [dict(major_radius=3.0, minor_radius=2.0)] and guard == {}
    tool, n = fs.rounded_corners(box, box, [_corner_face(2.5, 2.5)], guard=guard)      # a horn face: thinned
    (t,) = tg.tori(tool)
    assert n == 1 and tg.primitive_defect(tool) is None and len(guard["thinned"]) == 1
    assert (t.dims_mm["major_radius"], t.dims_mm["minor_radius"]) == pytest.approx(tg.thinned(2.5, 2.5))
    tool, n = fs.rounded_corners(box, box, [_corner_face(2.0, 2.5)], guard=guard)      # a spindle face: no corner
    assert n == 0 and tool is box and len(guard["rejected"]) == 1 and "spindle" in guard["rejected"][0]
    assert fs.guard_note(guard) == ", corner tori by the torus guard: 1 rejected, 1 thinned"


def test_the_liner_slot_keeps_its_rounded_ends_without_a_horn_torus():
    """The slot of tests/test_prismatic_groups.py is the liner's (8 wide, floor R 2, round ends R 4: horn faces
    R = r = 2): its tool holds no guarded torus, is whole, and lies inside the tool with the horn tori."""
    import test_prismatic_groups as tpg
    from reco_engine import prismatic_groups as pg
    from reco_engine.result_to_program import primitive_to_sp
    p = tpg.step_groove_slits(tpg.IDENTITY)
    slot = tpg.by_kind(pg.channels(p.faces))["slot"][0]["tool"]
    assert tg.primitive_defect(slot) is None and "4 corner(s)" in slot.label and "4 thinned" in slot.label
    (_, tool, partial), = pg.prismatic_tools(dict(faces=p.faces), p.vertices())
    assert not partial and tg.primitive_defect(tool) is None
    horn = tg.torus_defect
    try:                                                            # the same slot with the horn tori (no guard)
        tg.torus_defect = lambda R, r, d=None: None
        exact = tpg.by_kind(pg.channels(p.faces))["slot"][0]["tool"]
    finally:
        tg.torus_defect = horn
    lo, hi = np.array([39.0, 1.6, 11.0]), np.array([49.0, 5.5, 195.0])
    pts = lo + np.random.default_rng(0).random((400000, 3)) * (hi - lo)
    a, b = (np.asarray(primitive_to_sp(t).sdf(pts)) <= 0 for t in (slot, exact))
    assert not (a & ~b).any()                                       # nothing outside the exact tool
    assert (b & ~a).sum() * np.prod(hi - lo) / len(pts) < 1.0       # 0.40 mm3 less over 4 corners (4e6 samples)


def test_feature_sources_do_not_offer_a_guarded_torus(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("igl")
    from reco_engine import feature_stage as fs
    failed = {}
    admit = fs.torus_gate(50.0, failed)
    assert admit(torus(10.0, 2.0), "fit") and failed == {}
    assert not admit(torus(2.0, 2.0), "step faces") and not admit(torus(3.0, 3.5), "step faces")
    assert not admit(torus(100.0, 2.0), "cascade")
    assert failed["torus guard (step faces)"].startswith("2 rejected, the first: horn torus")
    assert "oversized" in failed["torus guard (cascade)"]
    # the whole source: a piece whose only real curved face is a horn torus (a slot's end) offers no torus
    ring = trimesh.creation.torus(major_radius=4.0, minor_radius=2.0, major_sections=48, minor_sections=24)
    ring.export(tmp_path / "piece.stl")
    faces = [dict(type="torus", origin="input", area=1.0, axis=[0.0, 0.0, 1.0], center=[0.0, 0.0, 0.0],
                  major_radius=2.0, minor_radius=2.0)]
    g = fs.guess_candidates(dict(stl=str(tmp_path / "piece.stl"), faces=faces), sources=("step faces",), stop_at=None)
    assert all(c.primitive.kind != "torus" for c in g["candidates"])
    assert "horn" in g["failed"]["torus guard (step faces)"]
    faces[0].update(major_radius=4.0, minor_radius=2.0)                  # the ring's own face: offered
    g = fs.guess_candidates(dict(stl=str(tmp_path / "piece.stl"), faces=faces), sources=("step faces",), stop_at=None)
    assert [c.primitive.kind for c in g["candidates"]] == ["torus"] and not g["failed"]


def test_subpart_cover_torus_follows_the_guard():
    from reco_engine import subpart_cover as sc
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    s = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    T, S = np.meshgrid(t, s)
    P = np.stack([(20 + 4 * np.cos(S)) * np.cos(T), (20 + 4 * np.cos(S)) * np.sin(T), 4 * np.sin(S)], -1).reshape(-1, 3)
    axis, centre = np.array([0.0, 0.0, 1.0]), np.zeros(3)
    ring = sc.torus_of(P, axis, centre, 20.0)
    assert ring is not None and ring.dims_mm["minor_radius"] == pytest.approx(4.0, abs=0.05)
    assert sc.torus_of(P, axis, centre, 20.0, diagonal=10.0) is None                 # far larger than the part
    assert sc.torus_of(P, axis, centre, 4.02) is None                                # the points need r >= R: no ring
    near = P[np.abs(P[:, 2]) < 1e-9]                                                  # the equator's points only
    assert sc.torus_of(near, axis, centre, 12.01) is None                            # r = 11.99: near-horn
