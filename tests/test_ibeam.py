#!/usr/bin/env python3
"""reco_engine/ibeam.py and the member / hollow sub-loops of reco_engine/residual_loop.py.

Without FreeCAD: fit_ibeam_section and classify_section on synthetic 2D loops (an HEA 500 with tessellated root fillets,
a rotated IPE 200, a T, a U, rectangles, circles, hollow sections), the DIN snap at unit scale 1 and 1/100, the
non-standard web that must not snap, and the world pose of the three boxes.

With FreeCAD (skipped without freecadcmd): parts built by Part booleans in freecadcmd and exported as STL - an HEA 500 x
2000 with fillets, a rotated IPE 200 x 1000, an L-joint of two IPE 200, a box-edge frame of 12 rods, a tube and a hollow
box. detect_members finds the members (count, axis, extent) and the sections (h, b, tw, tf within 0.5 mm, snapped to the
right profile); the sub-loop accepts 3 box adds per I member (a cylinder per rod, add + cut for the tube, add + cut for
the hollow box); the loop's own FreeCAD build is one valid solid with a chamfer below 1 % of the section size.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine import ibeam as ib  # noqa: E402
from reco_engine.result_schema import ReconstructionResult  # noqa: E402
from reco_engine.result_to_program import rotation_matrix  # noqa: E402


# ------------------------------------------------------------------------------------------------ synthetic loops

def i_loop(h, b, tw, tf, r, n_arc=8):
    """Counter-clockwise outline of an I centred at 0, h along y, with tessellated root fillets."""
    pts = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, -h / 2 + tf)]
    y0 = -h / 2 + tf

    def arc(cx, cy, a0, a1):
        return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in np.linspace(a0, a1, n_arc + 1)]
    if r > 0:
        pts += arc(tw / 2 + r, y0 + r, -math.pi / 2, -math.pi)          # bottom right fillet
        pts += arc(tw / 2 + r, h / 2 - tf - r, math.pi, math.pi / 2)    # top right fillet
    else:
        pts += [(tw / 2, y0), (tw / 2, h / 2 - tf)]
    pts += [(b / 2, h / 2 - tf), (b / 2, h / 2), (-b / 2, h / 2), (-b / 2, h / 2 - tf)]
    if r > 0:
        pts += arc(-tw / 2 - r, h / 2 - tf - r, math.pi / 2, 0.0)
        pts += arc(-tw / 2 - r, y0 + r, 0.0, -math.pi / 2)
    else:
        pts += [(-tw / 2, h / 2 - tf), (-tw / 2, y0)]
    pts += [(-b / 2, y0)]
    L = np.array(pts, float)
    keep = np.ones(len(L), bool)
    keep[1:] = np.linalg.norm(np.diff(L, axis=0), axis=1) > 1e-12
    return L[keep]


def rot2(L, deg, shift=(0.0, 0.0)):
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return L @ R.T + np.asarray(shift)


def circle(r, n=48, c=(0.0, 0.0)):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], axis=1)


def rect(w, h, c=(0.0, 0.0)):
    return np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) + np.asarray(c)


def test_i_loop_area_matches_the_formula():
    L = i_loop(490, 300, 12, 23, 27, n_arc=64)
    a = abs(ib.signed_area(L))
    assert a == pytest.approx(2 * 300 * 23 + (490 - 46) * 12 + (4 - math.pi) * 27 ** 2, rel=1e-3)


@pytest.mark.parametrize("deg,shift", [(0, (0, 0)), (90, (5, -3)), (33, (100, 250))])
def test_fit_hea500_recovers_dims_and_snaps(deg, shift):
    L = rot2(i_loop(490, 300, 12, 23, 27), deg, shift)
    s = ib.fit_ibeam_section(L)
    assert s is not None
    for k, v in dict(h=490, b=300, tw=12, tf=23).items():
        assert s.measured[k] == pytest.approx(v, abs=0.5), k
    assert s.measured["r"] == pytest.approx(27, abs=1.5)
    assert s.snapped and s.profile == "HEA 500" and s.unit_scale == 1.0
    assert s.confidence > 0.99
    assert np.allclose(s.centre, shift, atol=0.5)
    assert abs(abs(s.h_dir @ rot2(np.array([[0.0, 1.0]]), deg)[0]) - 1) < 1e-6        # web along the rotated y


def test_fit_at_unit_scale_one_hundredth_snaps_with_scale_100():
    s = ib.fit_ibeam_section(i_loop(490, 300, 12, 23, 27) / 100.0)
    assert s is not None and s.snapped and s.profile == "HEA 500" and s.unit_scale == 100.0
    assert s.h == pytest.approx(4.9, abs=1e-6)


def test_ipe200_without_fillets():
    s = ib.fit_ibeam_section(rot2(i_loop(200, 100, 5.6, 8.5, 0.0), 12))
    assert s is not None and s.profile == "IPE 200" and s.snapped
    assert s.measured["tw"] == pytest.approx(5.6, abs=0.05) and s.measured["r"] == 0.0


def test_non_standard_web_keeps_measured_dims():
    """The frame's posts: 350 x 300 x 12 x 17.5 - an HEA 360 envelope with a 12 mm web (standard 10)."""
    s = ib.fit_ibeam_section(i_loop(350, 300, 12, 17.5, 27))
    assert s is not None and not s.snapped
    assert s.tw == pytest.approx(12, abs=0.1) and s.h == pytest.approx(350, abs=0.1)
    assert s.profile is not None and "envelope" in s.label


@pytest.mark.parametrize("loop", [
    rect(100, 50),                                                                  # bar
    circle(20),                                                                     # rod
    np.array([[-50, -60], [50, -60], [50, -50], [5, -50], [5, 40], [-5, 40], [-5, -50], [-50, -50]], float),  # T
    np.array([[0, 0], [60, 0], [60, 10], [10, 10], [10, 90], [60, 90], [60, 100], [0, 100]], float),        # C / U
])
def test_non_i_loops_are_rejected(loop):
    assert ib.fit_ibeam_section(loop) is None


def test_classify_kinds():
    assert ib.classify_section(circle(20)).kind == "rod"
    c = ib.classify_section(rot2(rect(80, 20), 27))
    assert c.kind == "bar" and sorted([2 * c.params["hx"], 2 * c.params["hy"]]) == pytest.approx([20, 80])
    t = ib.classify_section(circle(30, c=(5, 5)), [circle(24, c=(5, 5))[::-1]])
    assert t.kind == "tube" and t.params["r"] == pytest.approx(30) and t.params["r_in"] == pytest.approx(24)
    h = ib.classify_section(rect(100, 60), [rect(90, 50)[::-1]])
    assert h.kind == "rhs" and (h.params["ix"], h.params["iy"]) == pytest.approx((45, 25))
    assert ib.classify_section(i_loop(200, 100, 5.6, 8.5, 12)).kind == "ibeam"
    assert ib.classify_section(rect(100, 60), [circle(10)]).kind == "other"


def test_ibeam_boxes_pose():
    """A member along world x, h along world z: the three boxes' dims and centres."""
    loop2 = i_loop(200, 100, 5.6, 8.5, 12)                           # (u, v) = (y, z) below
    m = ib.Member(axis=np.array([1.0, 0, 0]), e_u=np.array([0, 1.0, 0]), e_v=np.array([0, 0, 1.0]), loop=loop2 + [0, 500],
                  holes=[], bbox=(-50, 400, 50, 600), area=0.0, start=10.0, end=1010.0, stations=[])
    m.section = ib.classify_section(m.loop)
    boxes = ib.ibeam_boxes(m)
    assert [b.label.split()[-1] for b in boxes] == ["flange", "flange", "web"]
    assert all(b.sources[0].mode == ib.MODE_IBEAM for b in boxes)
    world_half = []
    for b in boxes:
        R = rotation_matrix(b)
        h = np.array([b.dims_mm["x_half"], b.dims_mm["y_half"], b.dims_mm["z_half"]])
        world_half.append(np.abs(R) @ h)
    top, bot, web = boxes
    assert np.allclose(world_half[0], [500, 50, 4.25], atol=1e-6)
    assert np.allclose(world_half[2], [500, 2.8, 100 - 4.25], atol=1e-6)
    assert np.allclose(top.center_mm, [510, 0, 600 - 4.25], atol=1e-6)
    assert np.allclose(bot.center_mm, [510, 0, 400 + 4.25], atol=1e-6)
    assert np.allclose(web.center_mm, [510, 0, 500], atol=1e-6)


def test_label_survives_the_json_round_trip():
    from reco_engine.result_schema import Operation
    m = ib.Member(axis=np.array([0, 0, 1.0]), e_u=np.array([1.0, 0, 0]), e_v=np.array([0, 1.0, 0]),
                  loop=i_loop(490, 300, 12, 23, 27), holes=[], bbox=(-150, -245, 150, 245), area=0.0, start=0.0,
                  end=2000.0, stations=[])
    m.section = ib.classify_section(m.loop)
    r = ReconstructionResult("x", [Operation("add", p) for p in ib.ibeam_boxes(m)])
    back = ReconstructionResult.from_dict(json.loads(r.to_json()))
    assert back.operations[0].primitive.label == "HEA 500 top flange"
    d = json.loads(r.to_json())
    del d["operations"][0]["primitive"]["label"]                    # an old result without the field still loads
    assert ReconstructionResult.from_dict(d).operations[0].primitive.label is None


def test_a_cone_and_a_torus_are_sampled_by_their_own_sdf():
    """The hollow sub-loop's any-kind cavity cutter (2026-10-29) can be a cone; the exact judge's samplers knew only box,
    cylinder and sphere and raised ValueError("cone") on aegir_n16_U_000158 (server_c). Every other kind is sampled by
    rejection on its SDF: the points lie inside, the volume matches the closed form."""
    from reco_engine.result_schema import Primitive
    from reco_engine.result_to_program import primitive_to_sp
    cone = Primitive(kind="cone", quaternion=(0.2, 0.1, 0.0, 0.97), center_mm=(5.0, -3.0, 2.0),
                     dims_mm=dict(radius1=4.0, radius2=1.5, half_length=6.0))
    torus = Primitive(kind="torus", center_mm=(0.0, 0.0, 0.0), dims_mm=dict(major_radius=6.0, minor_radius=1.5))
    for p, exact in ((cone, math.pi * 12.0 / 3 * (16.0 + 6.0 + 2.25)), (torus, 2 * math.pi ** 2 * 6.0 * 1.5 ** 2)):
        pts = ib.primitive_samples(p, 1500, seed=1)
        assert len(pts) == 1500 and (np.asarray(primitive_to_sp(p).sdf(pts)) <= 1e-9).all()
        assert ib.primitive_volume(p) == pytest.approx(exact, rel=0.03)


def test_subloop_plan_hints(monkeypatch):
    from reco_engine import residual_loop as rl
    cfg = rl.ResidualConfig()
    assert rl._subloop_plan(cfg) == dict(members=None, hollow=True, forced=False)
    assert rl._subloop_plan(cfg, "box/ibeam")["members"] == {"ibeam"}
    assert rl._subloop_plan(cfg, "sphere/frame") == dict(members=None, hollow=False, forced=True)
    assert rl._subloop_plan(cfg, "cylinder/tube")["members"] == {"tube"}
    assert rl._subloop_plan(cfg, "box/hollow") == dict(members=False, hollow=True, forced=True)
    # a plain hint does not veto the analytic gates (7e97cc0d, B-139: it had collapsed the frame from 32 operations to 3)
    assert rl._subloop_plan(cfg, "box") == dict(members=None, hollow=True, forced=False)
    monkeypatch.setenv("SHAPE_RECO_IBEAM", "0")
    monkeypatch.setenv("SHAPE_RECO_HOLLOW", "0")
    assert rl._subloop_plan(cfg) == dict(members=False, hollow=False, forced=False)


# ------------------------------------------------------------------------------------------------ FreeCAD-built parts

FREECAD_SCRIPT = r'''
import json, os, math
import FreeCAD as App
import Part, MeshPart

def ibeam(h, b, tw, tf, r, L):
    s = Part.makeBox(b, tf, L, App.Vector(-b / 2, h / 2 - tf, 0))
    s = s.fuse([Part.makeBox(b, tf, L, App.Vector(-b / 2, -h / 2, 0)),
                Part.makeBox(tw, h - 2 * tf, L, App.Vector(-tw / 2, -h / 2 + tf, 0))])
    if r > 0:
        for sx in (-1, 1):
            for sy in (-1, 1):
                x0 = tw / 2 if sx > 0 else -tw / 2 - r
                y0 = h / 2 - tf - r if sy > 0 else -h / 2 + tf
                blk = Part.makeBox(r, r, L, App.Vector(x0, y0, 0))
                cyl = Part.makeCylinder(r, L, App.Vector(sx * (tw / 2 + r), sy * (h / 2 - tf - r), 0))
                s = s.fuse(blk.cut(cyl))
    return s.removeSplitter()

def place(s, axis=(0, 0, 1), angle=0.0, t=(0, 0, 0)):
    s = s.copy()
    s.Placement = App.Placement(App.Vector(*t), App.Rotation(App.Vector(*axis), angle))
    return s

def rods(dx, dy, dz, r):
    parts = []
    for y in (0, dy):
        for z in (0, dz):
            parts.append(Part.makeCylinder(r, dx, App.Vector(0, y, z), App.Vector(1, 0, 0)))
    for x in (0, dx):
        for z in (0, dz):
            parts.append(Part.makeCylinder(r, dy, App.Vector(x, 0, z), App.Vector(0, 1, 0)))
    for x in (0, dx):
        for y in (0, dy):
            parts.append(Part.makeCylinder(r, dz, App.Vector(x, y, 0), App.Vector(0, 0, 1)))
    return parts[0].fuse(parts[1:]).removeSplitter()

out = os.environ["OUT_DIR"]
parts = {}
parts["hea500"] = ibeam(490, 300, 12, 23, 27, 2000)
parts["ipe200_rot"] = place(ibeam(200, 100, 5.6, 8.5, 12, 1000), (1, 2, 3), 37.0, (10, 20, 30))
a = ibeam(200, 100, 5.6, 8.5, 12, 1000)
b = place(ibeam(200, 100, 5.6, 8.5, 12, 1000), (1, 1, 1), 120.0, (-50, 0, 1100))
parts["l_joint"] = a.fuse(b).removeSplitter()
parts["rod_frame"] = rods(400, 300, 200, 10)
tube = Part.makeCylinder(30, 300).cut(Part.makeCylinder(22, 300))
parts["tube"] = place(tube, (1, 0, 0), 30.0, (5, 5, 5))
parts["hollow_box"] = Part.makeBox(100, 80, 60).cut(Part.makeBox(70, 50, 30, App.Vector(15, 15, 15)))
info = {}
for name, s in parts.items():
    m = MeshPart.meshFromShape(Shape=s, LinearDeflection=0.05, AngularDeflection=0.2)
    m.write(os.path.join(out, name + ".stl"))
    info[name] = dict(valid=s.isValid(), solids=len(s.Solids), volume=s.Volume)
json.dump(info, open(os.path.join(out, "parts.json"), "w"))
'''


def _freecad():
    from src.closest_point_field import find_freecadcmd
    return find_freecadcmd()


needs_freecad = pytest.mark.skipif(_freecad() is None, reason="freecadcmd not found")


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    import os
    d = tmp_path_factory.mktemp("fc_parts")
    script = d / "build.py"
    script.write_text(FREECAD_SCRIPT)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["OUT_DIR"] = str(d)
    proc = subprocess.run([_freecad(), str(script)], capture_output=True, text=True, timeout=300, env=env)
    info_path = d / "parts.json"
    assert info_path.is_file(), (proc.stdout + proc.stderr)[-1500:]
    info = json.loads(info_path.read_text())
    for name, i in info.items():
        assert i["valid"] and i["solids"] == 1, (name, i)
    return d


def _field(parts, name):
    from src.closest_point_field import load_field
    return load_field(str(parts / f"{name}.stl"))


def _run(field, tmp_path, **kw):
    from reco_engine import cascade, residual_loop as rl
    base = dict(resolution=48, max_resolution=64, registry=cascade.DEFAULT_REGISTRY, max_ops=4, out_dir=tmp_path,
                voxel_fallback=False)
    base.update(kw)
    steps = list(rl.residual_steps(field, rl.ResidualConfig(**base)))
    return steps


def _check_ibeam(sec, h, b, tw, tf, profile):
    for k, v in dict(h=h, b=b, tw=tw, tf=tf).items():
        assert sec.measured[k] == pytest.approx(v, abs=0.5), (k, sec.measured)
    assert sec.snapped and sec.profile == profile


def _member_ops(steps):
    acc = [s for s in steps if s.kind == "accepted" and s.info.get("member") is not None]
    return acc


def _grade(steps):
    g = [s for s in steps if s.kind == "grade"]
    assert g, "no FreeCAD grade step"
    return g[-1].info["grade"]


@needs_freecad
def test_hea500_member_and_three_boxes(parts, tmp_path):
    f = _field(parts, "hea500")
    ms = ib.detect_members(f)
    assert len(ms) == 1 and ms[0].kind == "ibeam"
    m = ms[0]
    assert abs(abs(m.axis[2]) - 1) < 1e-6 and m.length == pytest.approx(2000, abs=0.5)
    _check_ibeam(m.section.ibeam, 490, 300, 12, 23, "HEA 500")
    steps = _run(f, tmp_path)
    kinds = [s.kind for s in steps]
    assert kinds.count("member") == 1 and kinds.count("ibeam_box") == 3
    acc = _member_ops(steps)
    assert len(acc) == 3 and all(s.info["op"] == "add" and s.info["primitive"].kind == "box" for s in acc)
    g = _grade(steps)
    assert g["valid"] and g["n_solids"] == 1
    assert g["chamfer"] < 0.01 * 490


@needs_freecad
def test_rotated_ipe200(parts, tmp_path):
    f = _field(parts, "ipe200_rot")
    ms = [m for m in ib.detect_members(f) if m.kind == "ibeam"]
    assert len(ms) == 1
    from scipy.spatial.transform import Rotation
    axis = Rotation.from_rotvec(np.radians(37.0) * np.array([1, 2, 3]) / np.sqrt(14)).apply([0, 0, 1])
    assert abs(abs(ms[0].axis @ axis) - 1) < 1e-4
    assert ms[0].length == pytest.approx(1000, abs=0.5)
    _check_ibeam(ms[0].section.ibeam, 200, 100, 5.6, 8.5, "IPE 200")
    steps = _run(f, tmp_path)
    assert len(_member_ops(steps)) == 3
    g = _grade(steps)
    assert g["valid"] and g["n_solids"] == 1 and g["chamfer"] < 0.01 * 200


@needs_freecad
def test_l_joint_two_members(parts, tmp_path):
    f = _field(parts, "l_joint")
    ms = [m for m in ib.detect_members(f) if m.kind == "ibeam"]
    assert len(ms) == 2
    axes = sorted(int(np.argmax(np.abs(m.axis))) for m in ms)
    assert axes == [0, 2]
    for m in ms:
        _check_ibeam(m.section.ibeam, 200, 100, 5.6, 8.5, "IPE 200")
        assert m.length == pytest.approx(1000, abs=0.5)
    steps = _run(f, tmp_path)
    acc = _member_ops(steps)
    assert len(acc) == 6 and all(s.info["op"] == "add" for s in acc)
    g = _grade(steps)
    assert g["valid"] and g["n_solids"] == 1 and g["chamfer"] < 0.01 * 200


@needs_freecad
def test_box_edge_frame_of_rods(parts, tmp_path):
    f = _field(parts, "rod_frame")
    ms = ib.detect_members(f)
    rods = [m for m in ms if m.kind == "rod"]
    assert len(rods) == 12
    assert sorted(round(m.length) for m in rods) == [200] * 4 + [300] * 4 + [400] * 4
    assert all(m.section.params["r"] == pytest.approx(10, abs=0.1) for m in rods)
    steps = _run(f, tmp_path)
    acc = _member_ops(steps)
    assert len(acc) == 12 and all(s.info["primitive"].kind == "cylinder" for s in acc)
    g = _grade(steps)
    assert g["valid"] and g["n_solids"] == 1 and g["chamfer"] < 0.01 * 20 * 10       # 1 % of the frame's size


@needs_freecad
def test_tube_is_add_then_cut(parts, tmp_path):
    f = _field(parts, "tube")
    ms = ib.detect_members(f)
    assert len(ms) == 1 and ms[0].kind == "tube"
    assert ms[0].section.params["r"] == pytest.approx(30, abs=0.1)
    assert ms[0].section.params["r_in"] == pytest.approx(22, abs=0.1)
    steps = _run(f, tmp_path)
    acc = _member_ops(steps)
    assert [(s.info["op"], s.info["primitive"].kind) for s in acc] == [("add", "cylinder"), ("cut", "cylinder")]
    g = _grade(steps)
    assert g["valid"] and g["n_solids"] == 1 and g["chamfer"] < 0.01 * 60


@needs_freecad
def test_hollow_box_is_add_then_cut_of_the_same_kind(parts, tmp_path):
    f = _field(parts, "hollow_box")
    cav = ib.find_cavities(f)
    assert cav is not None and len(cav.inner) == 1
    steps = _run(f, tmp_path)
    kinds = [s.kind for s in steps]
    assert "hollow" in kinds
    res = steps[-1].result
    ops = [(o.op, o.primitive.kind) for o in res.operations if o.accepted]
    assert ops[:2] == [("add", "box"), ("cut", "box")]
    cut = res.operations[1].primitive
    R = rotation_matrix(cut)
    half = np.sort(np.abs(R) @ [cut.dims_mm["x_half"], cut.dims_mm["y_half"], cut.dims_mm["z_half"]])
    assert half == pytest.approx([15, 25, 35], abs=1.0)
    g = _grade(steps)
    assert g["valid"] and g["n_solids"] == 1 and g["chamfer"] < 0.01 * 100


@needs_freecad
def test_hint_disables_the_member_subloop(parts, tmp_path):
    from reco_engine import cascade, residual_loop as rl
    f = _field(parts, "hea500")
    cfg = rl.ResidualConfig(resolution=32, max_resolution=32, registry=cascade.DEFAULT_REGISTRY, max_ops=1, freecad=False)
    steps = list(rl.residual_steps(f, cfg, hint_subclass="box/plain"))
    assert "member" not in [s.kind for s in steps]


def test_gui_folds_member_steps_and_shows_the_profile():
    """reco_gui: a "member" step is the component (yellow) with its profile in the detail panel; "ibeam_box" is a
    candidate; the member stays while its boxes are accepted; the new kinds are stages with a breakpoint at "member"."""
    from reco_engine.residual_loop import ResidualStep
    from reco_engine.result_schema import Operation
    from reco_gui.cascade_session import CASCADE_DEFAULT_BREAKPOINTS, CASCADE_STAGES, CascadeFolder
    from reco_gui.report import render_cascade_state
    assert {"member", "ibeam_box", "member_op", "hollow"} <= set(CASCADE_STAGES) and "member" in CASCADE_DEFAULT_BREAKPOINTS
    m = ib.Member(axis=np.array([0, 0, 1.0]), e_u=np.array([1.0, 0, 0]), e_v=np.array([0, 1.0, 0]),
                  loop=i_loop(490, 300, 12, 23, 27), holes=[], bbox=(-150, -245, 150, 245), area=0.0, start=0.0,
                  end=2000.0, stations=[])
    m.section = ib.classify_section(m.loop)
    ibs = m.section.ibeam
    spec = dict(lo=np.array([-400.0, -400, -100]), hi=np.array([400.0, 400, 2100]))
    mask = np.zeros((8, 8, 8), bool)
    mask[2:6, 1:7, :] = True
    res = ReconstructionResult("hea500.stl")
    boxes = ib.ibeam_boxes(m)
    fold = CascadeFolder(resolution=24)
    st = fold(None, ResidualStep("input", res, dict(spec=spec, message="input")), 0)
    info = dict(member=m, section=m.section, kind="ibeam", axis=(0, 0, 1), extent=(0.0, 2000.0), profile=ibs.label,
                mask=mask, index=0, members=1, ibeam=dict(h=ibs.h, b=ibs.b, tw=ibs.tw, tf=ibs.tf, r=ibs.r,
                                                          measured=ibs.measured, profile=ibs.profile, standard=ibs.standard,
                                                          snapped=True, rel_distance=0.0, unit_scale=1.0, confidence=0.99))
    st = fold(st, ResidualStep("member", res, info), 1)
    assert st.component_mesh is not None and st.member is m and st.step.stage == "member"
    html_text = render_cascade_state(st)
    assert "HEA 500" in html_text and "Member" in html_text
    st = fold(st, ResidualStep("ibeam_box", res, dict(op="add", primitive=boxes[0], gain=150.0, precision=1.0, ok=True,
                                                     reason="", member=m, grid_gain=0)), 2)
    assert st.candidate is boxes[0] and st.candidate_mesh is not None
    assert "HEA 500 top flange" in render_cascade_state(st)
    res2 = ReconstructionResult("hea500.stl", [Operation("add", boxes[0], net_gain=150.0)])
    st = fold(st, ResidualStep("accepted", res2, dict(op="add", primitive=boxes[0], gain=150.0, member=m)), 3)
    assert st.component_mesh is not None and st.model_mesh is not None and st.member is m
    assert "HEA 500 top flange" in render_cascade_state(st)
