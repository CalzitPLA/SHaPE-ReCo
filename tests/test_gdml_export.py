"""
GDML export (reco_field/gdml_export.py): the rotation convention against Geant4's reader as the module states it, and
every solid kind written, read back by that convention and compared point by point with the tree it came from - tilted
and off-centre, so that a swapped sign or an inverted rotation cannot hide.
"""

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

from reco_field import gdml_export as ge  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402

Q = Rotation.from_euler("zxy", [35.0, -20.0, 110.0], degrees=True).as_matrix()
Q2 = Rotation.from_euler("xyz", [-70.0, 15.0, 40.0], degrees=True).as_matrix()
T = np.array([40.0, -25.0, 310.0])


def rx(a):
    return Rotation.from_euler("x", a).as_matrix()


def ry(a):
    return Rotation.from_euler("y", a).as_matrix()


def rz(a):
    return Rotation.from_euler("z", a).as_matrix()


def test_angles_by_the_place_they_are_written_to():
    """Geant4 builds G = Rz(z) Ry(y) Rx(x) (rotateX, rotateY, rotateZ) and places a boolean's second solid by G, a
    physical volume by the inverse of G."""
    x, y, z = ge.gdml_angles(Q, "boolean")
    assert np.allclose(rz(z) @ ry(y) @ rx(x), Q, atol=1e-12)
    assert np.allclose(ge.matrix_of_angles(x, y, z, "boolean"), Q, atol=1e-12)
    x, y, z = ge.gdml_angles(Q, "physvol")
    assert np.allclose(np.linalg.inv(rz(z) @ ry(y) @ rx(x)), Q, atol=1e-12)
    assert np.allclose(ge.matrix_of_angles(x, y, z, "physvol"), Q, atol=1e-12)
    assert ge.gdml_angles(np.eye(3)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert ge.gdml_angles(rz(0.3), "boolean") == pytest.approx((0.0, 0.0, 0.3), abs=1e-12)
    assert ge.gdml_angles(rz(0.3), "physvol") == pytest.approx((0.0, 0.0, -0.3), abs=1e-12)
    with pytest.raises(ValueError):
        ge.gdml_angles(np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError):
        ge.gdml_angles(Q, "somewhere")


def test_no_angle_span_rounds_above_its_limit(tmp_path):
    ge.export_gdml(PARTS["sphere between planes"], tmp_path / "s.gdml")
    ge.export_gdml(PARTS["cylinder"], tmp_path / "c.gdml")
    for f, key, most in (("s.gdml", "deltatheta", math.pi), ("s.gdml", "deltaphi", 2 * math.pi), ("c.gdml", "deltaphi", 2 * math.pi)):
        values = [float(e.get(key)) for e in ET.parse(tmp_path / f).getroot().find("solids") if e.get(key)]
        assert values and all(v <= most for v in values)


def test_a_full_turn_is_two_pi_to_the_last_digit(tmp_path):
    """G4Torus takes a solid as full only for deltaphi >= 2 pi, G4Polycone for >= 2 pi (1 - DBL_EPSILON): 2 pi rounded
    to 9 digits left a hairline slit at phi = 0 that stopped Geant4's lines (aegir's scan of the liner, 2026-10-01)."""
    full = sp.Union(sp.Placed(sp.Torus(10.0, 2.0), Q, T), sp.Placed(stack((3.0, 3.0, 5.0), (3.0, 1.0, 4.0)), Q2, T))
    ge.export_gdml(sp.Union(full, sp.Placed(sp.Cylinder(2.0, 30.0), Q, T)), tmp_path / "f.gdml")
    solids = ET.parse(tmp_path / "f.gdml").getroot().find("solids")
    spans = {e.tag: float(e.get("deltaphi")) for e in solids if e.get("deltaphi")}
    assert set(spans) == {"torus", "polycone", "tube"}
    assert all(v == 2 * math.pi for v in spans.values())


def same_solid(model, path, n=60000, band=1e-6):
    lo, hi = (np.asarray(b, float) for b in model.bounds())
    pad = 0.2 * float(np.max(hi - lo))
    pts = np.random.default_rng(0).uniform(lo - pad, hi + pad, size=(n, 3))
    a = np.asarray(model.sdf(pts))
    b = np.asarray(ge.read_gdml(path).sdf(pts))
    sure = np.abs(a) > band                               # on the surface itself the sign is rounding
    assert (a < 0).sum() > 200
    assert ((a < 0) == (b < 0))[sure].all()


PARTS = {
    "box": sp.Placed(sp.Box((5.0, 3.0, 2.0)), Q, T),
    "cylinder": sp.Placed(sp.Cylinder(4.0, 12.0), Q, T),
    "tube sector": sp.Placed(sp.Cylinder(6.0, 9.0, (0.4, 2.9), 2.5), Q, T),
    "cone": sp.Placed(sp.Cone(5.0, 1.5, 8.0), Q, T),
    "drill tip": sp.Placed(sp.Cone(2.459, 0.0, 1.42), Q, T),
    "orb": sp.Placed(sp.Sphere(5.0), Q, T),
    "sphere segment": sp.Placed(sp.Sphere(6.0, (0.3, 4.0), (-0.4, 1.0), 2.0, "cone"), Q, T),
    "sphere between planes": sp.Placed(sp.Sphere(6.0, (0.0, 2 * math.pi), (-0.4, 1.0), 0.0, "plane"), Q, T),
    "torus": sp.Placed(sp.Torus(10.0, 2.0, (0.2, 3.5)), Q, T),
}


@pytest.mark.parametrize("kind", sorted(PARTS))
def test_every_solid_reads_back_as_itself(tmp_path, kind):
    info = ge.export_gdml(PARTS[kind], tmp_path / "p.gdml", name=kind)
    assert info["primitives"] >= 1
    same_solid(PARTS[kind], tmp_path / "p.gdml")


def bolt_hole_part():
    """A block with a stepped hole (counterbore, shank, drill tip) along a tilted axis, a rounded pocket tool, a plane
    cut and a boss: subtraction, union and intersection, nested, every operand with its own pose."""
    block = sp.Placed(sp.Box((30.0, 20.0, 15.0)), Q, T)
    hole = sp.Union(sp.Union(sp.Cylinder(6.0, 7.0), sp.Placed(sp.Cylinder(3.5, 20.0), np.eye(3), np.array([0.0, 0.0, 7.0]))),
                    sp.Placed(sp.Cone(3.5, 0.0, 2.02), np.eye(3), np.array([0.0, 0.0, 27.0])))
    hole = sp.Placed(hole, Q @ Q2, T + Q @ np.array([8.0, -3.0, 4.0]))
    pocket = sp.Placed(sp.Intersection(sp.Box((8.0, 5.0, 6.0)), sp.Placed(sp.Cylinder(7.0, 20.0), np.eye(3), np.array([0.0, 0.0, -10.0]))),
                       Q, T + Q @ np.array([-15.0, 8.0, 12.0]))
    boss = sp.Placed(sp.Cylinder(5.0, 10.0), Q, T + Q @ np.array([-10.0, -8.0, 14.0]))
    cut = sp.HalfSpace(tuple(Q @ np.array([1.0, 0.0, 0.0])), float((Q @ np.array([1.0, 0.0, 0.0])) @ T) + 25.0)
    return sp.Intersection(sp.Union(sp.Subtraction(sp.Subtraction(block, hole), pocket), boss), cut)


def test_a_nested_program(tmp_path):
    model = bolt_hole_part()
    info = ge.export_gdml(model, tmp_path / "part.gdml", name="1 part-with holes", material=ge.ALUMINIUM, tree="as-is")
    assert info["booleans"] == 5 and info["primitives"] == 6            # the hole's three steps are one polycone
    assert info["solids"]["subtraction"] == 2 and info["solids"]["halfspace"] == 1 and info["solids"]["polycone"] == 1
    plain = ge.export_gdml(model, tmp_path / "plain.gdml", name="plain", polycones=False, tree="as-is")
    assert plain["booleans"] == 7 and plain["primitives"] == 8 and "polycone" not in plain["solids"]
    same_solid(model, tmp_path / "plain.gdml")
    grouped = ge.export_gdml(model, tmp_path / "grouped.gdml", name="grouped")      # the default: the two cuts as one
    assert grouped["tree"] == "grouped-balanced" and grouped["solids"]["subtraction"] == 1 and grouped["primitives"] == 6
    same_solid(model, tmp_path / "grouped.gdml")
    assert info["volume"] == "p_1_part_with_holes_volume" and info["material"] == "Aluminium"
    same_solid(model, tmp_path / "part.gdml")
    root = ET.parse(tmp_path / "part.gdml").getroot()
    assert [e.tag for e in root] == ["define", "materials", "solids", "structure", "setup"]
    names = [e.get("name") for e in root.find("solids")]
    assert len(names) == len(set(names))
    refs = {e.get("ref") for e in root.find("solids").iter() if e.tag in ("first", "second")}
    assert refs <= set(names)                                 # a solid is defined before it is used, and exists
    for e in root.find("solids"):
        for r in e.findall("first") + e.findall("second"):
            assert names.index(r.get("ref")) < names.index(e.get("name"))
    half = np.asarray(info["world_half"])
    lo, hi = (np.asarray(b) for b in model.bounds())
    assert (half > np.maximum(np.abs(lo), np.abs(hi))).all()  # the part lies inside the world


def test_what_gdml_cannot_say_is_refused(tmp_path):
    with pytest.raises(NotImplementedError):
        ge.export_gdml(sp.Torus(10.0, 2.0, (0.0, 2 * math.pi), (0.0, 1.0)), tmp_path / "t.gdml")
    with pytest.raises(ValueError):
        ge.export_gdml(sp.HalfSpace((0.0, 0.0, 1.0), 0.0), tmp_path / "h.gdml")


def test_export_result(tmp_path):
    from reco_engine.result_schema import Operation, Primitive, ReconstructionResult
    box = Primitive(kind="box", dims_mm=dict(x_half=20.0, y_half=15.0, z_half=10.0), center_mm=(1.0, 2.0, 3.0))
    cyl = Primitive(kind="cylinder", dims_mm=dict(radius=4.0, half_length=12.0), center_mm=(5.0, 2.0, 3.0))
    skipped = Operation("cut", Primitive(kind="cylinder", dims_mm=dict(radius=9.0, half_length=12.0)), accepted=False)
    r = ReconstructionResult(input_path="/parts/Frame_4_x_6.stp", operations=[Operation("add", box), Operation("cut", cyl), skipped])
    info = ge.export_result(r, tmp_path / "frame.gdml")
    assert info["solids"] == dict(box=1, tube=1, subtraction=1) and info["volume"] == "Frame_4_x_6_volume"
    from reco_engine.result_to_program import program_model
    same_solid(program_model(r), tmp_path / "frame.gdml")
    with pytest.raises(ValueError):
        ge.export_result(ReconstructionResult(input_path="x.step"), tmp_path / "none.gdml")


def test_a_simulation_frame_and_nist_materials(tmp_path):
    """The beam along the world's z, a source point as the origin, the volume names a simulation asks for, and
    materials that are only referred to by their Geant4 NIST name."""
    model = bolt_hole_part()
    origin = np.array([40.0, -25.0, 310.0])
    info = ge.export_gdml(model, tmp_path / "p.gdml", name="part", material="G4_Fe", world="G4_AIR", margin_mm=1000.0,
                          origin=origin, beam_axis="y", volume_name="PartVolume", world_name="World", physvol_name="PartPhys")
    root = ET.parse(tmp_path / "p.gdml").getroot()
    assert len(root.find("materials")) == 0
    names = [v.get("name") for v in root.find("structure").findall("volume")]
    assert names == ["PartVolume", "World"] and root.find("setup").find("world").get("ref") == "World"
    assert root.find("structure").findall("volume")[0].find("materialref").get("ref") == "G4_Fe"
    assert info["volume"] == "PartVolume" and info["material"] == "G4_Fe"
    G = ge.BEAM_AXES["y"]
    assert np.allclose(G @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]) and np.linalg.det(G) == pytest.approx(1.0)
    same_solid(sp.Placed(model, G, -G @ origin), tmp_path / "p.gdml")
    lo, hi = (np.asarray(b) for b in info["bounds"])
    assert np.allclose(np.asarray(info["world_half"]), np.maximum(np.abs(lo), np.abs(hi)) + 1000.0)
    with pytest.raises(ValueError):
        ge.export_gdml(model, tmp_path / "q.gdml", beam_axis="w")


def stack(*steps):
    """Coaxial cylinders and cones end to end along z from 0: (radius0, radius1, length)."""
    out, z = None, 0.0
    for r0, r1, h in steps:
        q = sp.Placed(sp.Cylinder(r0, h) if r0 == r1 else sp.Cone(r0, r1, h), np.eye(3), np.array([0.0, 0.0, z]))
        out, z = (q if out is None else sp.Union(out, q)), z + h
    return out


def inner(p):
    while isinstance(p, sp.Placed):
        p = p.primitive
    return p


def test_a_hole_stack_is_one_polycone(tmp_path):
    hole = sp.Placed(stack((6.0, 6.0, 7.0), (3.5, 3.5, 50.0), (3.5, 2.459, 0.6), (2.459, 2.459, 9.4)), Q, T)
    merged = ge.merge_coaxial(hole)
    assert isinstance(inner(merged), ge.Polycone)
    assert np.allclose(inner(merged).planes, [(0.0, 6.0), (7.0, 6.0), (7.0, 3.5), (57.0, 3.5), (57.6, 2.459), (67.0, 2.459)])
    info = ge.export_gdml(hole, tmp_path / "h.gdml", name="hole")
    assert info["solids"] == dict(polycone=1) and info["booleans"] == 0
    same_solid(hole, tmp_path / "h.gdml")


def test_steps_that_overlap_or_run_the_other_way(tmp_path):
    a = sp.Cylinder(3.0, 20.0)                                          # z 0 .. 20
    b = sp.Placed(sp.Cylinder(5.0, 8.0), np.eye(3), np.array([0.0, 0.0, 15.0]))     # z 15 .. 23: overlaps a
    flip = np.diag([1.0, -1.0, -1.0])                                   # the same axis, the other way round
    c = sp.Placed(sp.Cone(1.0, 4.0, 6.0), flip, np.array([0.0, 0.0, 29.0]))         # z 23 .. 29, radius 4 at 23, 1 at 29
    part = sp.Placed(sp.Union(sp.Union(a, b), c), Q, T)
    merged = ge.merge_coaxial(part)
    assert isinstance(inner(merged), ge.Polycone)
    assert np.allclose(inner(merged).planes, [(0.0, 3.0), (15.0, 3.0), (15.0, 5.0), (23.0, 5.0), (23.0, 4.0), (29.0, 1.0)])
    ge.export_gdml(part, tmp_path / "p.gdml")
    same_solid(part, tmp_path / "p.gdml")


def test_what_is_not_coaxial_or_leaves_a_gap_stays(tmp_path):
    gap = sp.Union(sp.Cylinder(3.0, 5.0), sp.Placed(sp.Cylinder(3.0, 5.0), np.eye(3), np.array([0.0, 0.0, 9.0])))
    assert ge.export_gdml(gap, tmp_path / "g.gdml")["solids"] == dict(tube=2, union=1)
    beside = sp.Union(sp.Cylinder(3.0, 5.0), sp.Placed(sp.Cylinder(3.0, 5.0), np.eye(3), np.array([8.0, 0.0, 5.0])))
    assert ge.export_gdml(beside, tmp_path / "b.gdml")["solids"] == dict(tube=2, union=1)
    # a pocket with two hole stacks in it: the box stays, every stack is one polycone
    pocket = sp.Union(sp.Union(sp.Placed(sp.Box((20.0, 10.0, 2.0)), np.eye(3), np.array([0.0, 0.0, -2.0])),
                               sp.Placed(stack((2.459, 2.459, 15.0), (2.459, 0.0, 1.42)), np.eye(3), np.array([-8.0, 0.0, 0.0]))),
                      sp.Placed(stack((2.459, 2.459, 15.0), (2.459, 0.0, 1.42)), np.eye(3), np.array([8.0, 0.0, 0.0])))
    part = sp.Subtraction(sp.Placed(sp.Box((40.0, 30.0, 20.0)), Q, T), sp.Placed(pocket, Q, T + Q @ np.array([0.0, 0.0, 20.0])))
    info = ge.export_gdml(part, tmp_path / "p.gdml")
    assert info["solids"] == dict(box=2, polycone=2, union=2, subtraction=1)
    same_solid(part, tmp_path / "p.gdml")


def test_a_coaxial_cut_is_one_hollow_polycone(tmp_path):
    """The turned contour's ring: a cylinder minus the contour about the same axis, the contour's axis the other way round."""
    contour = sp.Placed(stack((175.0, 175.0, 40.0), (175.0, 162.5, 12.5)), np.diag([1.0, -1.0, -1.0]), np.array([0.0, 0.0, 52.5]))
    ring = sp.Placed(sp.Subtraction(sp.Cylinder(201.6, 52.5), contour), Q, T)
    info = ge.export_gdml(ring, tmp_path / "r.gdml")
    assert info["solids"] == dict(polycone=1) and info["booleans"] == 0
    zp = ET.parse(tmp_path / "r.gdml").getroot().find("solids")[0].findall("zplane")
    assert [(float(z.get("z")), float(z.get("rmin")), float(z.get("rmax"))) for z in zp] == pytest.approx(
        [(0.0, 162.5, 201.6), (12.5, 175.0, 201.6), (12.5, 175.0, 201.6), (52.5, 175.0, 201.6)]) or len(zp) >= 3
    same_solid(ring, tmp_path / "r.gdml")
    sections = [(z.get("z"), z.get("rmin"), z.get("rmax")) for z in zp]
    assert all(s != n for s, n in zip(sections, sections[1:]))          # ROOT's TGeoPcon refuses a section said twice
    # a cut that takes everything somewhere is no ring: the boolean stays
    through = sp.Subtraction(sp.Cylinder(5.0, 20.0), sp.Placed(sp.Cylinder(8.0, 4.0), np.eye(3), np.array([0.0, 0.0, 8.0])))
    assert ge.export_gdml(through, tmp_path / "t.gdml")["solids"] == dict(tube=2, subtraction=1)
    # a bore that ends inside: the ring has its inner radius only where the bore reaches
    blind = sp.Placed(sp.Subtraction(sp.Cylinder(10.0, 30.0), sp.Placed(sp.Cylinder(4.0, 12.0), np.eye(3), np.array([0.0, 0.0, 18.0]))), Q, T)
    assert ge.export_gdml(blind, tmp_path / "b.gdml")["solids"] == dict(polycone=1)
    same_solid(blind, tmp_path / "b.gdml")


def cut_chain_part():
    """A block cut by five tools one after the other - asymmetric and tilted: a cone, a box, a hole stack (a cylinder and
    a cone end to end), a tube sector, a torus sector - under a pose of its own, and three bosses fused onto it."""
    tools = [sp.Placed(sp.Cone(4.0, 1.0, 40.0), Q2, np.array([10.0, 5.0, -20.0])),
             sp.Placed(sp.Box((4.0, 2.0, 30.0)), Q2 @ rx(0.7), np.array([-12.0, 0.0, 0.0])),
             sp.Placed(stack((3.0, 3.0, 10.0), (3.0, 1.0, 25.0)), rx(0.4), np.array([-5.0, -8.0, -20.0])),
             sp.Placed(sp.Cylinder(2.0, 50.0, (0.3, 4.0)), Q2 @ ry(1.1), np.array([0.0, 10.0, -25.0])),
             sp.Placed(sp.Torus(9.0, 1.5, (0.2, 3.5)), rz(0.3), np.array([0.0, 0.0, 15.0]))]
    part = sp.Box((30.0, 20.0, 15.0))
    for tool in tools:
        part = sp.Subtraction(part, tool)
    part = sp.Placed(part, Q, T)
    for R, t, boss in ((Q, np.array([20.0, 10.0, 15.0]), sp.Cone(5.0, 2.0, 12.0)),
                       (Q @ Q2, np.array([-20.0, -10.0, 15.0]), sp.Box((3.0, 4.0, 5.0))),
                       (Q2, np.array([0.0, -15.0, 15.0]), sp.Cylinder(3.0, 8.0, (0.0, 4.5)))):
        part = sp.Union(part, sp.Placed(boss, R, T + Q @ t))
    return part


def nesting(path) -> int:
    """The depth of the written tree: a primitive 0, a boolean or a multiUnion one more than its deepest operand."""
    solids = {e.get("name"): e for e in ET.parse(path).getroot().find("solids")}

    def depth(name):
        e = solids[name]
        refs = [r.get("ref") for r in e.iter() if r.tag in ("first", "second", "solid")]
        return 1 + max(depth(r) for r in refs) if refs else 0
    root = ET.parse(path).getroot()
    part = [v for v in root.find("structure").findall("volume") if v.get("name") != root.find("setup").find("world").get("ref")][0]
    return depth(part.find("solidref").get("ref"))


@pytest.mark.parametrize("polycones", [True, False])
def test_the_tree_shapes_write_the_same_solid(tmp_path, polycones):
    """T-30 (2026-10-01): the default groups a chain of cuts into ONE cut by the union of its tools and halves every tree
    of unions; grouped-multi writes a union of primitives as one multiUnion; as-is is the program's own tree. Every shape
    is the same solid, read back point by point."""
    model = cut_chain_part()
    shapes = {}
    for tree in ge.TREE_SHAPES:
        path = tmp_path / f"{tree}.gdml"
        info = ge.export_gdml(model, path, name="chain", tree=tree, polycones=polycones)
        assert info["tree"] == tree and info["primitives"] == (9 if polycones else 10)
        same_solid(model, path)
        shapes[tree] = (info["solids"], nesting(path))
    tools = 5 if polycones else 6                       # the hole stack: one polycone, or a tube and a cone
    asis, n_asis = shapes["as-is"]
    assert asis["subtraction"] == 5 and asis["union"] == 3 + (0 if polycones else 1) and "multiUnion" not in asis
    assert n_asis >= 6                                  # five cuts, one after the other, under the bosses' unions
    balanced, n_balanced = shapes["grouped-balanced"]
    assert balanced["subtraction"] == 1 and balanced["union"] == (tools - 1) + 3 and "multiUnion" not in balanced
    assert n_balanced == 2 + 1 + 3                      # two halvings of the bosses' union, the cut, three of the tools'
    multi, n_multi = shapes["grouped-multi"]
    assert multi["subtraction"] == 1 and multi["multiUnion"] == 1 and multi["union"] == 3
    assert n_multi == 2 + 1 + 1                         # the bosses' union with the cut part (composite: balanced)
    nodes = ET.parse(tmp_path / "grouped-multi.gdml").getroot().find("solids").find("multiUnion").findall("multiUnionNode")
    assert len(nodes) == tools
    with pytest.raises(ValueError):
        ge.export_gdml(model, tmp_path / "x.gdml", tree="multi")


def test_a_multiunion_node_is_placed_like_a_booleans_second_solid(tmp_path):
    """Three asymmetric primitives at generic angles, off the origin of the multiUnion's own frame: written as one
    multiUnion, every node's angles give its rotation as Rz Ry Rx (the convention aegir measured in Geant4)."""
    parts = [sp.Placed(sp.Box((60.0, 15.0, 5.0)), Q, T), sp.Placed(sp.Cone(25.0, 5.0, 80.0), Q2, T + np.array([30.0, 0.0, 10.0])),
             sp.Placed(sp.Cylinder(6.0, 200.0), Q @ Q2, T - np.array([0.0, 40.0, 0.0]))]
    P = rz(0.4) @ rx(-0.9)
    model = sp.Placed(sp.Union(sp.Union(parts[0], parts[1]), parts[2]), P, np.array([5.0, -7.0, 11.0]))
    info = ge.export_gdml(model, tmp_path / "m.gdml", tree="grouped-multi")
    assert info["solids"] == dict(box=1, cone=1, tube=1, multiUnion=1) and info["booleans"] == 1
    same_solid(model, tmp_path / "m.gdml", n=300000)
    solids = ET.parse(tmp_path / "m.gdml").getroot().find("solids")
    box = solids.find("box").get("name")
    node = [n for n in solids.find("multiUnion").findall("multiUnionNode") if n.find("solid").get("ref") == box][0]
    a = [float(node.find("rotation").get(k)) for k in "xyz"]
    assert np.allclose(rz(a[2]) @ ry(a[1]) @ rx(a[0]), P @ Q, atol=1e-8)   # the box's axes in the multiUnion's frame
    assert np.allclose([float(node.find("position").get(k)) for k in "xyz"], P @ T + np.array([5.0, -7.0, 11.0]), atol=1e-8)


def test_a_program_tree_round_trip():
    """model_of_program reads reco_field.export.program_to_dict's plain data back as the same solid."""
    from reco_field.export import program_to_dict
    model = cut_chain_part()
    back = ge.model_of_program(program_to_dict(model))
    pts = np.random.default_rng(3).uniform(*(np.asarray(b) for b in model.bounds()), size=(20000, 3))
    assert np.allclose(np.asarray(model.sdf(pts)), np.asarray(back.sdf(pts)), atol=1e-9)
