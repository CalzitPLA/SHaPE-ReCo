#!/usr/bin/env python3
"""
The two rules of 2026-09-29 ("before going into fall[back] after member phase check other solution, maybe classic
cut-add and the member stored, to check for congruency"):

  B. reco_engine/member_store.py - members grouped by congruency, a source's primitive moved onto the other members,
     a failing copy judged in full by itself, a failing member never rejecting its group, the groups as created tools;
  A. reco_engine/solution_choice.py - the classic solution before the fallback, both judged by the sampled exact IoU,
     the better one kept, the steps the GUI's folder understands.

No FreeCAD here: the members are made by hand or found in a mesh built with trimesh, the loop runs with freecad=False.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine import ibeam as ib  # noqa: E402
from reco_engine import member_store as ms  # noqa: E402
from reco_engine import solution_choice as sc  # noqa: E402
from reco_engine.result_schema import Operation, Primitive, ReconstructionResult  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp  # noqa: E402


# ------------------------------------------------------------------------------------------------ members by hand

def member(kind, params, axis, origin, length, joined=(False, False), turn=0.0):
    """A Member of the given section whose line starts at `origin` and runs `length` along `axis`."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    e_u, e_v = ib._perp_basis(axis)
    c, s = np.cos(turn), np.sin(turn)
    e_u, e_v = c * e_u + s * e_v, -s * e_u + c * e_v
    o = np.asarray(origin, float)
    centre = np.array([o @ e_u, o @ e_v])
    sec = ib.SectionFit(kind, centre, np.eye(2), dict(params), confidence=1.0)
    h = max(params.get("hx", 0), params.get("hy", 0), params.get("r", 0))
    return ib.Member(axis=axis, e_u=e_u, e_v=e_v, loop=np.zeros((0, 2)), holes=[],
                     bbox=(centre[0] - h, centre[1] - h, centre[0] + h, centre[1] + h), area=4 * h * h,
                     start=float(o @ axis), end=float(o @ axis) + length, stations=[], joined=joined, section=sec)


BAR = dict(hx=15.0, hy=10.0)
RHS = dict(hx=40.0, hy=30.0, ix=35.0, iy=25.0)


def same_solid(p, q, lo, hi, n=4000, tol=1e-6):
    pts = np.random.default_rng(0).uniform(lo, hi, (n, 3))
    return np.allclose(primitive_to_sp(p).sdf(pts), primitive_to_sp(q).sdf(pts), atol=tol)


def test_members_group_by_kind_dimensions_and_length():
    m = [member("bar", BAR, (1, 0, 0), (0, 0, 0), 400), member("bar", BAR, (0, 1, 0), (50, 0, 0), 400.2),
         member("bar", BAR, (0, 0, 1), (0, 80, 0), 300), member("bar", dict(hx=15.0, hy=12.0), (1, 0, 0), (0, 0, 90), 400),
         member("rod", dict(r=15.0), (1, 0, 0), (0, 0, 190), 400)]
    d = [ms.member_descriptor(x) for x in m]
    assert ms.group_members(d) == [[0, 1], [2], [3], [4]]              # 400 / 400.2: 5e-4 apart, within 2e-3
    assert ms.group_members(d, tol=1e-4) == [[0], [1], [2], [3], [4]]
    other = member("bar", BAR, (1, 0, 0), (0, 0, 0), 400)
    other.section = ib.SectionFit("other", np.zeros(2), np.eye(2), {})
    assert ms.member_descriptor(other) is None
    assert ms.group_members([None, None]) == [[0], [1]]                # no descriptor: a group of its own


@pytest.mark.parametrize("axis,origin,turn", [((0, 1, 0), (50, -20, 7), 0.0), ((1, 2, 3), (-300, 40, 900), 0.7)])
def test_moved_primitive_is_the_targets_own(axis, origin, turn):
    src, dst = member("bar", BAR, (1, 0, 0), (0, 0, 0), 400), member("bar", BAR, axis, origin, 400, turn=turn)
    (op, p), (_, own) = ib.member_operations(src)[0], ib.member_operations(dst)[0]
    R, t = ms.member_motion(src, dst)
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)
    q = ms.move_primitive(p, R, t, "M1", 0)
    c = np.asarray(own.center_mm)
    assert same_solid(q, own, c - 300, c + 300)
    assert q.sources[-1].mode == ms.MODE_CONGRUENT and q.sources[-1].detail == dict(group="M1", source=0)
    assert len(p.sources) == len(q.sources) - 1                       # the source's primitive is not touched


def test_a_cutter_follows_the_joined_ends():
    src = member("rhs", RHS, (1, 0, 0), (0, 0, 0), 400, joined=(True, False))
    dst = member("rhs", RHS, (0, 1, 0), (10, 20, 30), 400, joined=(False, True))
    none = member("rhs", RHS, (0, 1, 0), (10, 20, 30), 400, joined=(False, False))
    assert ms.cut_transferable(src, dst) and not ms.cut_transferable(src, none)
    cut, own = ib.member_operations(src)[1][1], ib.member_operations(dst)[1][1]
    R, t = ms.member_motion(src, dst)                                  # the half turn: the free end onto the free end
    c = np.asarray(own.center_mm)
    assert same_solid(ms.move_primitive(cut, R, t, "M1", 0), own, c - 300, c + 300)
    store = ms.MemberStore([src, none])
    for k, (op, p) in enumerate(ib.member_operations(src)):
        store.accept(0, k, op, p, False)
    assert store.moved(1, 0, "add") is not None and store.moved(1, 1, "cut") is None


def test_a_failed_member_does_not_reject_its_group():
    m = [member("bar", BAR, (1, 0, 0), (0, 100 * i, 0), 400) for i in range(4)]
    store = ms.MemberStore(m)
    assert [g.members for g in store.groups] == [[0, 1, 2, 3]] and store.info(2)["group"] == "M1"
    assert store.moved(0, 0, "add") is None
    store.reject(0, 0)                                                # the first member's own primitive fails
    assert store.moved(1, 0, "add") is None                           # no source yet: member 2 is judged in full
    own = ib.member_operations(m[1])[0][1]
    store.accept(1, 0, "add", own, False, 0.97)
    assert store.source(2, 0) == 1 and store.source(1, 0) is None
    q = store.moved(2, 0, "add")
    assert same_solid(q, ib.member_operations(m[2])[0][1], np.array([-300, 0, -200.0]), np.array([700, 400, 200.0]))
    store.accept(2, 0, "add", q, True, 0.99)
    store.failed_copy(3)                                              # the copy fails on member 4: its own primitive
    store.accept(3, 0, "add", ib.member_operations(m[3])[0][1], False, 0.96)
    s = store.summary()
    assert s["n_groups"] == 1 and s["operations"] == 3 and s["by_congruency"] == 1 and s["failed_copies"] == 1
    g = s["groups"][0]
    assert g["without_operation"] == [0] and g["copies"] == [2] and g["sources"] == [1] and g["kind"] == "bar"


@pytest.mark.parametrize("kind,params", [("bar", BAR), ("rod", dict(r=12.0)), ("rhs", RHS), ("tube", dict(r=30.0, r_in=24.0))])
def test_piece_descriptor_is_the_meshs(kind, params):
    """The descriptor computed from the section against trimesh's of the same solid (volume, area, moments, centre)."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("manifold3d")
    m = member(kind, params, (0, 0, 1), (20, -30, 100), 500)
    d = ms.member_piece_descriptor(m)
    ops = ib.member_operations(m)

    def mesh_of(p):
        dm = p.dims_mm
        if p.kind == "box":
            return trimesh.creation.box((2 * dm["x_half"], 2 * dm["y_half"], 2 * dm["z_half"]))
        return trimesh.creation.cylinder(dm["radius"], 2 * dm["half_length"], sections=256)
    solid = mesh_of(ops[0][1])
    if len(ops) > 1:
        solid = trimesh.boolean.difference([solid, mesh_of(ops[1][1])], engine="manifold")
    assert d["volume"] == pytest.approx(abs(solid.volume), rel=2e-3)
    assert d["area"] == pytest.approx(solid.area, rel=2e-3)
    assert np.allclose(d["moments"], np.sort(solid.principal_inertia_components), rtol=5e-3)
    assert np.allclose(d["centre"], ops[0][1].center_mm, atol=1e-6)
    assert np.allclose(d["axes"].T @ d["axes"], np.eye(3), atol=1e-9) and np.linalg.det(d["axes"]) > 0
    assert d["sign"] == "plus" and d["faces"] == ms.FACES[kind]


def test_ibeam_descriptor_volume_and_area():
    sec = ib.IBeamSection.__new__(ib.IBeamSection)
    m = member("bar", BAR, (0, 0, 1), (0, 0, 0), 1000)
    m.section = ib.SectionFit("ibeam", np.zeros(2), np.eye(2), {}, confidence=1.0, ibeam=sec)
    sec.h, sec.b, sec.tw, sec.tf = 200.0, 100.0, 6.0, 10.0
    d = ms.member_piece_descriptor(m)
    area = 2 * 100 * 10 + 6 * 180
    assert d["volume"] == pytest.approx(area * 1000)
    assert d["area"] == pytest.approx((4 * 100 + 2 * 200 - 2 * 6) * 1000 + 2 * area)
    assert ms.member_descriptor(m)["dims"] == dict(h=200.0, b=100.0, tw=6.0, tf=10.0)   # by geometry, no profile name


def test_groups_become_created_tools_the_feature_phase_can_recycle(tmp_path):
    from reco_engine import tools as TL
    from reco_engine.feature_stage import congruent
    m = [member("rhs", RHS, (1, 0, 0), (0, 0, 0), 400), member("rhs", RHS, (0, 1, 0), (500, 0, 0), 400),
         member("bar", BAR, (0, 0, 1), (0, 0, 0), 300)]
    store = ms.MemberStore(m)
    for i in (0, 2):
        for k, (op, p) in enumerate(ib.member_operations(m[i])):
            store.accept(i, k, op, p, False, 0.98)
    for k, (op, _) in enumerate(ib.member_operations(m[1])):
        store.accept(1, k, op, store.moved(1, k, op), True, 0.97)
    tools = TL.ToolStore()
    ids = store.to_tools(tools, "part.step")
    assert ids == ["created:member rhs#1", "created:member bar#1"] and len(tools) == 2
    rhs = tools.records[0]
    assert [u["source"] for u in rhs.uses] == ["created", "pattern"] and rhs.uses[1]["score"] == pytest.approx(0.97)
    again = TL.ToolStore.from_json(tools.to_json())                   # the session's file
    tool = TL.construction_primitive(again.records[0])
    assert tool.kind == "composite" and tool.combine == "cut" and [p.kind for p in tool.parts] == ["box", "box"]
    # a piece of the exact difference that IS such a member (a member the loop missed) is congruent with the tool's
    piece = dict(ms.member_piece_descriptor(m[1]))
    assert congruent(TL._descriptor_from_json(again.records[0].descriptor), piece)
    assert not congruent(TL._descriptor_from_json(again.records[1].descriptor), piece)


# ------------------------------------------------------------------------------------------------ the measure

def box_prim(centre, half):
    return Primitive(kind="box", center_mm=tuple(centre), dims_mm=dict(x_half=half[0], y_half=half[1], z_half=half[2]))


def in_box(lo, hi):
    return lambda pts: np.all((np.asarray(pts) >= lo) & (np.asarray(pts) <= hi), axis=1)


def test_sampled_measure_counts_every_volume_once():
    lo, hi = np.array([-10.0, -10, -10]), np.array([10.0, 10, 10])    # the input: a cube of side 20, volume 8000
    v_in, contains = 8000.0, in_box(lo, hi)

    def measure(*ops):
        r = ReconstructionResult("x", [Operation(op, p) for op, p in ops])
        return sc.sampled_measure(r, contains, v_in, (lo, hi), n=20000)
    exact = measure(("add", box_prim((0, 0, 0), (10, 10, 10))))
    assert exact["iou"] == pytest.approx(1.0) and exact["open_share"] == pytest.approx(0.0)
    half = measure(("add", box_prim((-5, 0, 0), (5, 10, 10))))
    assert half["iou"] == pytest.approx(0.5) and half["missing_share"] == pytest.approx(0.5) and half["invented"] == 0
    twice = measure(("add", box_prim((-5, 0, 0), (5, 10, 10))), ("add", box_prim((-2.5, 0, 0), (7.5, 10, 10))))
    assert twice["iou"] == pytest.approx(0.75, abs=0.01)             # the overlap counts for the first add only
    cut = measure(("add", box_prim((0, 0, 0), (10, 10, 10))), ("cut", box_prim((10, 0, 0), (10, 20, 20))))
    assert cut["iou"] == pytest.approx(0.5, abs=0.01) and cut["adds"] == 1 and cut["cuts"] == 1
    big = measure(("add", box_prim((0, 0, 0), (20, 10, 10))))         # twice the input: half of it invented
    assert big["iou"] == pytest.approx(0.5, abs=0.01) and big["invented_share"] == pytest.approx(1.0, abs=0.02)
    ball = measure(("add", Primitive(kind="composite", combine="common", parts=[
        box_prim((0, 0, 0), (10, 10, 10)), box_prim((5, 0, 0), (10, 10, 10))])))       # by rejection: x in [-5, 10]
    assert ball["iou"] == pytest.approx(0.75, abs=0.02)
    assert measure()["iou"] == 0.0 and measure()["missing_share"] == 1.0


def test_open_share_and_better():
    r = ReconstructionResult("x", [Operation("add", box_prim((0, 0, 0), (1, 1, 1)), net_gain=90.0),
                                   Operation("add", box_prim((0, 0, 0), (1, 1, 1)), net_gain=50.0, accepted=False)])
    assert sc.open_share(r, 2.0, 800.0) == pytest.approx(0.1)         # 90 voxel volumes of 8 = 720 of 800
    assert sc.open_share(r, 2.0, 0.0) is None
    hole = box_prim((0, 0, 0), (1, 1, 1))
    hole.sources = ib._sources(ib.MODE_MEMBER, 1.0)                    # a member's own cutter: judged with its add
    r.operations += [Operation("cut", hole, net_gain=40.0), Operation("cut", box_prim((0, 0, 0), (1, 1, 1)), net_gain=5.0)]
    assert sc.open_share(r, 2.0, 800.0) == pytest.approx(0.05)        # the loop's cut counts, the member's does not
    assert sc.better(dict(iou=0.90), dict(iou=0.92), 0.01) and not sc.better(dict(iou=0.90), dict(iou=0.905), 0.01)


def test_when_the_classic_solution_is_tried():
    from reco_engine.residual_loop import ResidualConfig
    cfg = ResidualConfig()
    members = dict(first="members", members=dict(asked=True, detected=40, accepted=40))
    assert sc.reasons_to_try(cfg, None, 0, 1000, 0.5) == []                        # no sub-loop bookkeeping: not the legacy path
    assert sc.reasons_to_try(cfg, members, 0, 1000, 0.0345) == []                  # the frame: within both tolerances
    assert len(sc.reasons_to_try(cfg, members, 0, 1000, 0.114, blind=11.0)) == 1   # r14: the exact account
    assert len(sc.reasons_to_try(cfg, members, 6, 1000, 0.0)) == 1                 # the fallback would run
    assert sc.reasons_to_try(ResidualConfig(solution_exact_tolerance=None), members, 0, 1000, 0.114) == []
    assert sc.reasons_to_try(ResidualConfig(solution_alternatives="never"), members, 6, 1000, 0.5) == []
    assert len(sc.reasons_to_try(ResidualConfig(solution_alternatives="always"), members, 0, 1000, 0.0)) == 1
    missed = dict(first="hollow", members=dict(asked=True, detected=4, accepted=0))   # r08: 4 tubes, 0 operations
    assert "found nothing usable" in sc.reasons_to_try(cfg, missed, 0, 1000, 0.0)[0]
    assert sc.reasons_to_try(cfg, dict(first="classic", members=dict(asked=True, detected=0, accepted=0)), 0, 1000, None)
    assert sc.reasons_to_try(cfg, dict(first="classic", members=None), 50, 1000, None) == []      # one solution only
    assert sc.reasons_to_try(ResidualConfig(solution_alternatives="always"), dict(first="classic", members=None), 0, 1000, None) == []
    with pytest.raises(ValueError):
        sc.reasons_to_try(ResidualConfig(solution_alternatives="sometimes"), members, 0, 1000, 0.0)


# ------------------------------------------------------------------------------------------------ the loop

@pytest.fixture(scope="module")
def bar_frame():
    """A rectangular frame of four bars 30 x 20 (two of 400, two of 300), one body, as a SurfaceField."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("manifold3d")
    from src.closest_point_field import SurfaceField
    a, b, w, h = 400.0, 300.0, 30.0, 20.0
    parts = []
    for y in (-(b - w) / 2, (b - w) / 2):
        parts.append(trimesh.creation.box((a, w, h)).apply_translation((0, y, 0)))
    for x in (-(a - w) / 2, (a - w) / 2):
        parts.append(trimesh.creation.box((w, b - 2 * w, h)).apply_translation((x, 0, 0)))
    u = trimesh.boolean.union(parts, engine="manifold").subdivide_to_size(40.0)
    return SurfaceField(np.asarray(u.vertices), np.asarray(u.faces))


def run(field, **kw):
    from reco_engine import cascade, residual_loop as rl
    base = dict(resolution=48, max_resolution=64, registry=cascade.DEFAULT_REGISTRY, freecad=False, field_backend="igl",
                member_min_length_ratio=1.5, more_booleans=False, boolean_tools=False, vary_on_failure=False)
    base.update(kw)
    return list(rl.residual_steps(field, rl.ResidualConfig(**base)))


def accepted_ops(steps):
    return [s for s in steps if s.kind == "accepted" and not s.info.get("solution")]


def program(step):
    """The accepted operations of a step's result as comparable keys (the dataclasses' own == recurses)."""
    return [(o.op, o.primitive.kind, tuple(np.round(o.primitive.center_mm, 6)), tuple(sorted(o.primitive.dims_mm.items())))
            for o in step.result.operations if o.accepted]


def test_loop_moves_the_source_onto_congruent_members(bar_frame):
    from reco_engine import residual_loop as rl
    steps = run(bar_frame)
    assert {s.kind for s in steps} <= set(rl.STEP_KINDS)
    members = [s for s in steps if s.kind == "member"]
    assert [s.info["group"] for s in members] == ["M1", "M1", "M2", "M2"]
    assert all(s.info["group_size"] == 2 and s.info["descriptor"]["kind"] == "bar" for s in members)
    acc = accepted_ops(steps)
    assert [bool(s.info["by_congruency"]) for s in acc] == [False, True, False, True]
    assert [s.info.get("source_member") for s in acc] == [None, 0, None, 2]
    assert "by congruency: member 1's, group M1" in acc[1].info["message"]
    copy = acc[1].info["primitive"]
    assert copy.sources[-1].mode == ms.MODE_CONGRUENT and copy.dims_mm == acc[0].info["primitive"].dims_mm
    done = steps[-1]
    assert done.kind == "done" and done.info["members"]["by_congruency"] == 2 and done.info["members"]["n_groups"] == 2
    assert done.info["solution"] == "members" and abs(done.info["solution_account"]) < 0.02
    assert "solution" not in [s.kind for s in steps]                  # within the tolerances: nothing more is computed
    # the switch: every member by itself, the same program up to the copies' dimensions
    off = run(bar_frame, member_congruency=False)
    assert len(accepted_ops(off)) == 4 and not any(s.info.get("by_congruency") for s in off)
    assert "group" not in [s for s in off if s.kind == "member"][0].info and off[-1].info["members"] is None


def test_a_copy_that_fails_is_judged_in_full_by_itself(bar_frame, monkeypatch):
    real = ms.member_motion

    def wrong(source, target):                                        # every copy lands 200 beside its member
        R, t = real(source, target)
        return R, t + np.array([0.0, 0.0, 200.0])
    monkeypatch.setattr(ms, "member_motion", wrong)
    steps = run(bar_frame)
    rejected = [s for s in steps if s.kind == "rejected"]
    assert len(rejected) == 2 and all(s.info["by_congruency"] for s in rejected)
    acc = accepted_ops(steps)
    assert len(acc) == 4 and not any(s.info["by_congruency"] for s in acc)
    assert [bool(s.info["after_failed_copy"]) for s in acc] == [False, True, False, True]
    store = steps[-1].info["members"]
    assert store["failed_copies"] == 2 and store["by_congruency"] == 0 and store["operations"] == 4


def test_the_members_reach_the_sessions_tool_store(bar_frame, tmp_path):
    from reco_engine import tools as TL
    tools = TL.ToolStore()
    run(bar_frame, feature_tool_store=tools)
    assert [r.tool for r in tools.records] == ["created:member bar", "created:member bar"]
    assert [[u["source"] for u in r.uses] for r in tools.records] == [["created", "pattern"]] * 2
    path = tmp_path / "tools.json"
    run(bar_frame, feature_tool_store=str(path))
    assert len(TL.ToolStore.load(path)) == 2


def test_classic_solution_is_tried_and_compared(bar_frame):
    from reco_engine import residual_loop as rl
    from reco_gui.cascade_session import CASCADE_STAGES, CascadeFolder
    steps = run(bar_frame, solution_alternatives="always")
    assert {s.kind for s in steps} <= set(rl.STEP_KINDS)
    k = next(i for i, s in enumerate(steps) if s.kind == "solution")
    trial = [s for s in steps if s.info.get("solution") == "classic"]
    assert trial and steps.index(trial[0]) > max(i for i, s in enumerate(steps) if s.kind == "member")
    assert steps.index(trial[-1]) == k - 1 and trial[0].kind == "add" and trial[0].info["message"].startswith("[classic] ")
    assert not any(s.kind in ("done", "grade", "fallback", "solution") for s in trial)
    info = steps[k].info
    assert [r["solution"] for r in info["solutions"]] == ["members", "classic"] and info["measure"] == sc.MEASURE
    assert info["tried"] and info["reasons"] and info["seconds"] > 0
    first, classic = info["solutions"]
    assert first["iou"] > 0.98 and first["operations"] == 4
    assert info["kept"] == ("classic" if classic["iou"] > first["iou"] + info["margin"] else "members")
    kept = program(steps[k])
    assert len(kept) == next(r for r in info["solutions"] if r["solution"] == info["kept"])["operations"]
    assert program(steps[-1])[:len(kept)] == kept
    assert steps[-1].info["solution"] == info["kept"]
    # the GUI's folder: every step a state, the trial's states show the classic program, the choice the kept one
    fold, prev, states = CascadeFolder(wall_grid=False), None, []
    for i, s in enumerate(steps):
        prev = fold(prev, s, i)
        states.append(prev)
    assert {s.step.stage for s in states} <= set(CASCADE_STAGES)
    assert len(states[k - 1].program) == classic["operations"] and len(states[k].program) == len(kept)
    assert states[k].candidate is None and states[k].member is None and states[-1].finished
    from reco_gui.report import render_cascade_state
    html = render_cascade_state(states[k])
    assert "Solutions before the fallback" in html and "(kept)" in html
    assert "computed for the comparison" in render_cascade_state(states[k - 1])


def test_the_better_solution_replaces_the_first_in_place(bar_frame, monkeypatch):
    """The classic solution wins (the measure is scripted): the run goes on with ITS state and program."""
    calls = []
    real = sc.sampled_measure

    def scripted(result, contains, v_in, bounds, n=1500):
        m = real(result, contains, v_in, bounds, n)
        calls.append(m["operations"])
        m["iou"] = 0.5 if len(calls) == 1 else 0.9                    # the members' solution first, then the classic one
        return m
    monkeypatch.setattr(sc, "sampled_measure", scripted)
    steps = run(bar_frame, solution_alternatives="always")
    k = next(i for i, s in enumerate(steps) if s.kind == "solution")
    info = steps[k].info
    assert info["kept"] == "classic" and "replaced the members solution" in steps[-1].info["stopped"]
    classic = program(steps[k - 1])
    assert program(steps[k]) == classic and program(steps[-1])[:len(classic)] == classic
    assert calls == [4, len(classic)] and steps[-1].info["solution"] == "classic"
    assert steps[-1].info["stats"]["ops"] >= len(classic)
    never = run(bar_frame, solution_alternatives="never")
    assert "solution" not in [s.kind for s in never] and len(accepted_ops(never)) == 4


def test_members_asked_for_and_nothing_usable_is_reported(bar_frame, monkeypatch):
    """The member phase finds its members and none of their operations passes (r08 of the frame review): the run's
    solution is the classic one, and a "solution" step says so with its measure."""
    from reco_engine import residual_loop as rl
    real = rl._exact_eval

    def failing(state, judge, result, prim, op, cfg, **kw):
        ev = real(state, judge, result, prim, op, cfg, **kw)
        ev.update(ok=False, reason="scripted: no member operation passes")
        return ev
    monkeypatch.setattr(rl, "_exact_eval", failing)
    steps = run(bar_frame, hollow=False)
    assert not accepted_ops([s for s in steps if s.info.get("member") is not None])
    sol = [s for s in steps if s.kind == "solution"]
    assert len(sol) == 1 and sol[0].info["kept"] == "classic" and not sol[0].info["tried"]
    assert "found nothing usable (4 member(s) detected" in sol[0].info["reasons"][0]
    assert [r["solution"] for r in sol[0].info["solutions"]] == ["classic"]
    assert steps[-1].info["solution"] == "classic" and steps[-1].info["members"]["operations"] == 0
