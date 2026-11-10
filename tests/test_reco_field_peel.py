"""reco_field/peel.py: the peeling loop reconstructs CSG solids. Closed loop on analytically built solids (ground truth known),
the example STLs (program and two-way chamfer), the open ends of cutters, and the geometric grade itself."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.peel import build_model, chamfer, mesh_from_sdf, peel, predict_ops_budget  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402
from src.closest_point_field import SurfaceField, load_field  # noqa: E402


def field_of(prim, resolution=150):
    v, f = mesh_from_sdf(prim, resolution)
    return SurfaceField(v, f)


def kinds(result):
    return [(o.op, o.fit.kind) for o in result.ops]


# ------------------------------------------------------------------------------------ closed loop on known solids

def test_mesh_from_sdf_lies_on_the_surface():
    v, f = mesh_from_sdf(sp.Sphere(7.0), 64)
    assert np.abs(np.linalg.norm(v, axis=1) - 7.0).max() < 1e-3 and len(f) > 1000


def test_chamfer_measures_a_known_offset_both_ways():
    field = field_of(sp.Sphere(8.0), 96)
    same = chamfer(sp.Sphere(8.0), field, n=4000, resolution=96)
    assert same["symmetric_mean"] < 0.01
    bigger = chamfer(sp.Sphere(8.5), field, n=4000, resolution=96)
    assert abs(bigger["input_to_model"]["mean"] - 0.5) < 0.02 and abs(bigger["model_to_input"]["mean"] - 0.5) < 0.02


def test_box_with_a_spherical_pocket():
    truth = sp.Box((10, 8, 6)) - sp.Placed(sp.Sphere(3.0), translation=np.array([1.0, 0.0, 6.0]))
    res = peel(field_of(truth))
    assert kinds(res) == [("base", "box"), ("cut", "sphere")]
    cut = res.ops[1].fit
    assert abs(cut.params["radius"] - 3.0) < 0.1 and np.allclose([cut.params["cx"], cut.params["cy"], cut.params["cz"]], [1, 0, 6], atol=0.1)
    assert chamfer(res.model, field_of(truth), n=6000)["symmetric_mean"] < 0.02


def test_box_with_a_through_hole_and_an_open_cutter():
    truth = sp.Box((10, 8, 6)) - sp.Placed(sp.Cylinder(2.0, 30.0), translation=np.array([2.0, 1.0, -15.0]))
    field = field_of(truth)
    res = peel(field)
    assert kinds(res) == [("base", "box"), ("cut", "cylinder")]
    assert abs(res.ops[1].fit.params["radius"] - 2.0) < 0.1
    lo, hi = res.ops[1].primitive.bounds()
    assert lo[2] < -10 and hi[2] > 10                                        # both ends open: the cutter reaches out of the body
    assert chamfer(res.model, field, n=6000)["symmetric_mean"] < 0.02


def test_union_of_two_boxes():
    truth = sp.Box((10, 8, 3)) | sp.Placed(sp.Box((4, 3, 4)), translation=np.array([1.0, 0.0, 5.0]))
    res = peel(field_of(truth))
    assert kinds(res) == [("base", "box"), ("add", "box")]
    assert np.allclose(sorted(res.ops[0].fit.half), [3, 8, 10], atol=0.1)


def test_sphere_with_a_flat_cut_is_one_sphere_segment_not_a_huge_sphere_or_two_operations():
    """The flat cap is a latitude limit of the sphere (FreeCAD's makeSphere semantic: planes): one primitive, v as a parameter."""
    truth = sp.Sphere(9.0) & sp.HalfSpace((0.0, 0.0, 1.0), 5.0)
    field = field_of(truth)
    res = peel(field)
    assert kinds(res) == [("base", "sphere")]
    fit = res.ops[0].fit
    assert fit.v is not None and fit.latitude == "plane" and abs(fit.params["radius"] - 9.0) < 0.05
    assert chamfer(res.model, field, n=6000)["symmetric_mean"] < 0.02


def test_a_cylinder_with_a_flat_is_a_clip_by_a_plane():
    truth = sp.Cylinder(9.0, 14.0) & sp.HalfSpace((1.0, 0.0, 0.0), 5.0)
    res = peel(field_of(truth))
    assert kinds(res) == [("base", "cylinder"), ("clip", "plane")] and res.ops[0].fit.u is None


def test_tube_is_a_cylinder_minus_a_bore():
    truth = sp.Cylinder(6.0, 14.0) - sp.Placed(sp.Cylinder(3.0, 40.0), translation=np.array([0.0, 0.0, -13.0]))
    res = peel(field_of(truth))
    assert kinds(res) == [("base", "cylinder"), ("cut", "cylinder")]
    assert abs(res.ops[1].fit.params["radius"] - 3.0) < 0.1


# ------------------------------------------------------------------------------------------------ the example parts

EX = ROOT / "dataset_gen" / "examples"
PROGRAMS = {
    "bs_box": [("base", "box")],
    "bs_sphere": [("base", "sphere")],
    "bs_box_step": [("base", "box"), ("add", "box")],
    "bs_box_sphere_pocket": [("base", "box"), ("cut", "sphere")],
    "bs_box_two_pockets": [("base", "box"), ("cut", "sphere"), ("cut", "sphere")],
    "bs_sphere_flat": [("base", "sphere")],
    "blind_hole": [("base", "box"), ("cut", "cylinder")],
    "tube": [("base", "cylinder"), ("cut", "cylinder")],
    "three_holes": [("base", "box"), ("cut", "cylinder"), ("cut", "cylinder"), ("cut", "cylinder")],
    "cy_on_box": [("base", "box"), ("add", "cylinder")],
    "l_bracket": [("base", "box"), ("cut", "box")],
    "side_slot": [("base", "box"), ("cut", "box")],
    # cone, torus and the segments (u, v)
    "co_cone": [("base", "cone")],
    "co_cone_apex": [("base", "cone")],
    "co_countersink": [("base", "box"), ("cut", "cone"), ("cut", "cylinder")],
    "to_torus": [("base", "torus")],
    "to_ring_groove": [("base", "cylinder"), ("cut", "torus")],
    "to_elbow": [("base", "torus")],
    "cs_cyl_sector": [("base", "cylinder")],
    "cs_cone_sector": [("base", "cone")],
    "cs_sphere_sector": [("base", "sphere")],
    "cs_sphere_band": [("base", "sphere")],
    "cs_torus_half": [("base", "torus")],
}

# (u sweep in degrees, has v) of the base primitive of the segment parts: FreeCAD built these with these angles
SEGMENTS = {"to_elbow": (90.0, False), "cs_cyl_sector": (120.0, False), "cs_cone_sector": (200.0, False),
            "cs_sphere_sector": (270.0, False), "cs_torus_half": (180.0, False), "cs_sphere_band": (None, True)}


@pytest.mark.skipif(not (EX / "three_holes.stl").is_file(), reason="example STLs missing")
@pytest.mark.parametrize("part", sorted(PROGRAMS))
def test_example_part_is_reconstructed_as_the_expected_program(part):
    field = load_field(EX / f"{part}.stl")
    res = peel(field)
    assert kinds(res) == PROGRAMS[part]
    assert res.explained > 0.99
    c = chamfer(res.model, field)
    assert c["symmetric_mean"] < 0.02 and c["hausdorff"] < 0.35                    # mm, on parts 20-35 mm across


@pytest.mark.skipif(not (EX / "cs_cyl_sector.stl").is_file(), reason="example STLs missing")
@pytest.mark.parametrize("part", sorted(SEGMENTS))
def test_segment_parts_keep_their_sweep_angle_as_a_parameter(part):
    sweep, has_v = SEGMENTS[part]
    fit = peel(load_field(EX / f"{part}.stl")).ops[0].fit
    if sweep is not None:
        assert fit.u is not None and fit.u[0] == 0.0 and abs(np.degrees(fit.u[1]) - sweep) < 1.5
    else:
        assert fit.u is None
    assert (fit.v is not None) == has_v


@pytest.mark.skipif(not (EX / "co_countersink.stl").is_file(), reason="example STLs missing")
def test_countersink_cone_cutter_is_open_at_the_top_and_reaches_the_apex_at_the_bottom():
    """The cone's wide end leaves the body (open); its narrow end has no cap data either and runs on to the apex, inside the bore."""
    field = load_field(EX / "co_countersink.stl")
    res = peel(field)
    cut = next(o for o in res.ops if o.fit.kind == "cone")
    lo, hi = cut.primitive.bounds()
    assert hi[2] > 20.0                                                        # far above the 20 mm cube's top face
    assert lo[2] > 0.0 and lo[2] < 4.0                                        # starts at (or above) the apex, below the countersink's 4 mm start


@pytest.mark.skipif(not (EX / "blind_hole.stl").is_file(), reason="example STLs missing")
def test_blind_hole_cutter_is_closed_at_the_bottom_and_open_at_the_top():
    res = peel(load_field(EX / "blind_hole.stl"))
    cut = next(o for o in res.ops if o.op == "cut")
    axis = cut.fit.frame[:, 2]
    zs = [float((p @ axis)) for p in (cut.primitive.bounds()[0], cut.primitive.bounds()[1])]
    body = load_field(EX / "blind_hole.stl").bounds
    span = np.abs(body[1] - body[0]).max()
    assert min(abs(np.array(zs))) < span and max(abs(np.array(zs))) > 2 * span     # one end stays at the hole's floor, the other is pushed out


def test_the_model_can_be_rebuilt_from_its_operations():
    res = peel(field_of(sp.Box((10, 8, 6)) - sp.Sphere(3.0)))
    rebuilt = build_model(res.ops)
    p = np.random.default_rng(0).uniform(-8, 8, (500, 3))
    assert np.array_equal(rebuilt.sdf(p) <= 0, res.model.sdf(p) <= 0)


def test_mesh_from_sdf_triangles_face_outward():
    """Flying edges gives no winding guarantee; the fits read the mesh's normals as OUTWARD (a plane's material side depends on it)."""
    for prim in (sp.Sphere(7.0), sp.Box((5, 4, 3)) - sp.Sphere(2.0)):
        field = field_of(prim, 96)
        mid = (field._a + field._b + field._c) / 3
        assert np.mean(prim.sdf(mid + 0.4 * field.face_normals) > prim.sdf(mid - 0.4 * field.face_normals)) > 0.99


# ------------------------------------------------------------------ invented geometry and the side split of clusters

def test_invented_is_zero_for_the_true_model_and_large_for_a_wrong_one():
    from reco_field.peel import invented
    field = field_of(sp.Sphere(8.0), 96)
    ext = float(np.linalg.norm(field.bounds[1] - field.bounds[0]))
    tol = 0.004 * ext
    assert invented(sp.Sphere(8.0), field, tol, ext) < 0.02
    assert invented(sp.Sphere(9.0), field, tol, ext) > 0.98                    # 1 mm off everywhere, tol is 0.06 mm
    slab = sp.Placed(sp.Box((30.0, 30.0, 2.0)))                                # invents most of its surface against a ball
    assert invented(slab, field, tol, ext) > 0.9


def test_invented_only_counts_the_model_surface_that_has_no_input_behind_it():
    from reco_field.peel import invented
    cube = sp.Placed(sp.Box((5.0, 5.0, 5.0)))
    field = field_of(cube, 96)
    ext = float(np.linalg.norm(field.bounds[1] - field.bounds[0]))
    tol = 0.004 * ext
    bigger = sp.Union(cube, sp.Placed(sp.Box((2.0, 2.0, 5.0)), translation=np.array([7.0, 0.0, 0.0])))   # a fin the input lacks
    frac = invented(bigger, field, tol, ext)
    assert 0.05 < frac < 0.5


# --------------------------------------------------------------------------------- predicting a practical max_ops

def _through_hole():
    """A box with ONE hole piercing both of its z faces: genus 1 (the cylinder is centred on the box, taller than it,
    so both ends stick out - sp.Cylinder's own bounds run z in [0, height], not centred, hence the translation)."""
    return sp.Box((10.0, 8.0, 6.0)) - sp.Placed(sp.Cylinder(1.5, 20.0), translation=np.array([0.0, 0.0, -10.0]))


def test_predicted_budget_scales_with_genus_not_a_fixed_number():
    box = field_of(sp.Box((10.0, 8.0, 6.0)), 96)                                            # genus 0
    ring = field_of(_through_hole(), 96)                                                    # one through-hole: genus 1
    b_box, b_ring = predict_ops_budget(box), predict_ops_budget(ring)
    assert b_box.exact_genus == 0 and b_ring.exact_genus == 1
    assert b_ring.max_ops == b_box.max_ops + 1                                              # exactly one more operation budgeted
    assert b_box.reason == "matched the exact genus" and b_ring.reason == "matched the exact genus"


def test_predicted_budget_is_capped_and_falls_back_when_not_closed():
    open_mesh = SurfaceField(*mesh_from_sdf(sp.Sphere(8.0), 64))
    open_mesh = SurfaceField(open_mesh.vertices, open_mesh.faces[:-1])                       # drop one face: no longer closed
    budget = predict_ops_budget(open_mesh)
    assert budget.exact_genus != budget.exact_genus and budget.reason.startswith("not closed")
    assert budget.max_ops == 60                                              # the default upper_cap, not the old MAX_OPS

    # a silly-high genus estimate must not blow the budget past upper_cap
    capped = predict_ops_budget(field_of(_through_hole(), 96), margin=1000, upper_cap=10)
    assert capped.max_ops == 10


def test_peel_defaults_to_the_predicted_budget_and_still_reconstructs():
    """max_ops=None (the new default) must not change the RESULT for parts the fixed default of 24 already solved."""
    field = field_of(_through_hole())
    auto, fixed = peel(field), peel(field, max_ops=24)
    assert kinds(auto) == kinds(fixed) == [("base", "box"), ("cut", "cylinder")]
    assert auto.explained == fixed.explained and auto.budget is not None and fixed.budget is None


def test_side_clusters_do_not_mix_the_inside_and_the_outside_of_the_model():
    from reco_field.peel import _side_clusters
    model = sp.Sphere(10.0)
    k = np.arange(300) + 0.5                                                   # evenly spread points on a unit sphere (Fibonacci)
    phi, theta = np.arccos(1 - 2 * k / 300), np.pi * (1 + 5 ** 0.5) * k
    unit = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    pts = np.vstack([6.0 * unit, 14.0 * unit[:200]])                           # 300 inside the ball, 200 outside
    got = _side_clusters(model, pts, np.arange(len(pts)), radius=6.0)
    assert [(len(idx), inside) for idx, inside in got] == [(300, True), (200, False)]
    assert set(got[0][0]) == set(range(300)) and set(got[1][0]) == set(range(300, 500))
