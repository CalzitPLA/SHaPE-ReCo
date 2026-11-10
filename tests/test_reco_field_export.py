"""reco_field/export.py: a program of SDF primitives is rebuilt with FreeCAD's Part module (STEP + STL) and agrees with the SDF model and with the
input mesh. Needs freecadcmd for everything but the serialisation tests. The conventions checked here are the ones of docs/freecad_part_shapes.tex."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.export import build_in_freecad, export_check, mesh_chamfer, program_to_dict  # noqa: E402
from reco_field.peel import mesh_from_sdf, peel  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402
from src.closest_point_field import SurfaceField, find_freecadcmd, load_field  # noqa: E402
from src.sdf_primitives import deg  # noqa: E402

needs_freecad = pytest.mark.skipif(find_freecadcmd() is None, reason="freecadcmd not found")
EX = ROOT / "dataset_gen" / "examples"


def rot(axis, angle):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(angle)).as_matrix()


# ---------------------------------------------------------------------------------------------------- serialisation

def test_program_is_plain_data_and_keeps_the_tree():
    import json
    prim = sp.Subtraction(sp.Placed(sp.Box((3, 2, 1)), rot([1, 1, 0], 30), np.array([1.0, 2.0, 3.0])),
                          sp.Intersection(sp.Cone(2, 1, 4, u=deg(10, 200)), sp.HalfSpace((0, 0, 1), 1.5)))
    d = json.loads(json.dumps(program_to_dict(prim)))
    assert d["type"] == "Subtraction" and d["a"]["type"] == "Placed" and d["a"]["primitive"]["half"] == [3, 2, 1]
    assert d["b"]["a"]["type"] == "Cone" and np.allclose(d["b"]["a"]["u"], deg(10, 200))
    assert d["b"]["b"]["type"] == "HalfSpace" and d["b"]["b"]["big"] > 4                  # the box standing for the half space covers the part


def test_program_from_dict_inverts_the_serialisation():
    """The JSON FreeCAD builds reads back as the same solid (a sphere between latitude planes, a torus sector, a hollow
    cylinder sector, a half space, every boolean): the same JSON again, and the same SDF at seeded points."""
    import json
    from reco_field.export import program_from_dict
    prim = sp.Union(
        sp.Subtraction(sp.Placed(sp.Box((3, 2, 1)), rot([1, 1, 0], 30), np.array([1.0, 2.0, 3.0])),
                       sp.Intersection(sp.Cone(2, 1, 4, u=deg(10, 200)), sp.HalfSpace((0, 0, 1), 1.5))),
        sp.Union(sp.Placed(sp.Sphere(2.5, v=deg(-40, 60), latitude="plane"), np.eye(3), np.array([4.0, 0, 0])),
                 sp.Union(sp.Torus(3.0, 0.7, u=deg(0, 270)), sp.Cylinder(1.5, 3.0, u=deg(30, 300), inner_radius=0.5))))
    d = json.loads(json.dumps(program_to_dict(prim)))
    back = program_from_dict(d)
    assert json.loads(json.dumps(program_to_dict(back))) == d
    pts = np.random.default_rng(0).uniform(-8, 8, (4000, 3))
    assert np.array_equal(prim.sdf(pts) <= 0, back.sdf(pts) <= 0)


def test_what_cannot_be_exported_is_refused():
    with pytest.raises(NotImplementedError):
        program_to_dict(sp.Ellipsoid(2, 3, 4))
    with pytest.raises(NotImplementedError):
        program_to_dict(sp.Torus(5, 1, inner_minor=0.5))


# ------------------------------------------------------------------------------------------ analytic primitives in FreeCAD

def built(prim, tmp_path, name="p"):
    out = build_in_freecad(prim, tmp_path, name)
    v, f = mesh_from_sdf(prim, 150)
    return out, SurfaceField(v, f)


CASES = {
    "box_minus_rotated_cylinder": sp.Box((10, 8, 6)) - sp.Placed(sp.Cylinder(2.0, 30.0), rot([1, 2, 0], 25), np.array([1.0, 0.0, -14.0])),
    "cylinder_sector_u0": sp.Cylinder(6.0, 12.0, u=deg(40, 300)),                        # a sector that does not start at 0
    "cone_sector_u0": sp.Cone(8.0, 3.0, 12.0, u=deg(-20, 180)),
    "tube_sector": sp.Cylinder(6.0, 10.0, u=deg(0, 120), inner_radius=3.0),
    "sphere_sector": sp.Sphere(8.0, u=deg(0, 270)),
    "sphere_plane_latitude": sp.Sphere(8.0, v=deg(-20, 60), latitude="plane"),
    "torus_sweep": sp.Torus(9.0, 3.0, u=deg(30, 210)),
    "torus_v_asymmetric": sp.Torus(10.0, 3.0, v=deg(-100, 30)),                          # Part::Torus Angle1=-30, Angle2=100: the mirror
    "torus_v_across_180": sp.Torus(10.0, 3.0, v=deg(100, 250)),                          # a range that wraps: two pies
    "torus_v_and_u": sp.Torus(9.0, 3.0, u=deg(0, 150), v=deg(-120, 60)),
    "sphere_flat": sp.Sphere(9.0) & sp.HalfSpace((0.6, 0.0, 0.8), 4.0),                  # a half space in a slanted direction
}


@needs_freecad
@pytest.mark.parametrize("name", sorted(CASES))
def test_freecad_solid_agrees_with_the_sdf_model(name, tmp_path):
    prim = CASES[name]
    out, ref = built(prim, tmp_path, name)
    assert out.valid and out.n_solids == 1 and out.step.is_file() and out.stl.is_file()
    c = mesh_chamfer(load_field(out.stl), ref)
    assert c["mean"] < 0.03 and c["max"] < 0.2, (name, c)                              # marching-cubes reference: ~0.02 mm
    # the volume agrees with the SDF's (a Monte-Carlo estimate, so a loose bound)
    lo, hi = prim.bounds()
    P = np.random.default_rng(0).uniform(lo, hi, (60000, 3))
    mc = float(np.mean(prim.sdf(P) <= 0)) * float(np.prod(hi - lo))
    assert abs(out.volume - mc) < 0.03 * mc + 2.0, (name, out.volume, mc)


@needs_freecad
def test_torus_v_mirror_gives_the_documented_part_volume(tmp_path):
    """Part::Torus Angle1=-30, Angle2=100 (R=10, r=3) has the volume of a 130 degree pie, 725.49 mm^3; that is our v = (-100, 30) degrees."""
    out = build_in_freecad(sp.Torus(10.0, 3.0, v=deg(-100, 30)), tmp_path, "t")
    assert abs(out.volume - 725.49) < 0.5


@needs_freecad
def test_cone_latitude_sphere_is_refused_clearly(tmp_path):
    with pytest.raises(RuntimeError, match="latitude"):
        build_in_freecad(sp.Sphere(8.0, v=deg(10, 70), latitude="cone"), tmp_path, "s")


# ------------------------------------------------------------------------------------------------- the example parts

PARTS = ["co_countersink", "to_ring_groove", "to_elbow", "cs_sphere_band", "cs_cone_sector", "bs_sphere_flat", "cy_flat", "three_holes"]


@needs_freecad
@pytest.mark.skipif(not (EX / "cs_cone_sector.stl").is_file(), reason="example STLs missing")
@pytest.mark.parametrize("part", PARTS)
def test_reconstruction_built_in_freecad_matches_the_input(part, tmp_path):
    import trimesh
    field = load_field(EX / f"{part}.stl")
    res = peel(field)
    out = build_in_freecad(res.model, tmp_path, part)
    assert out.valid and out.n_solids == 1
    c = export_check(out, field)
    assert c["mean"] < 0.02 and c["max"] < 0.3, (part, c)                              # mm, on parts 20-35 mm across
    truth = trimesh.load(str(EX / f"{part}.stl"), force="mesh").volume
    assert abs(out.volume - truth) < 0.01 * truth, (part, out.volume, truth)
