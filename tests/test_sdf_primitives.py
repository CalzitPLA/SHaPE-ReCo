"""
dataset_gen/src/sdf_primitives.py: the analytic SDFs against FreeCAD's own Part shapes (tests/data/sdf_reference.json, made by
scripts/sdf_reference_freecad.py inside FreeCAD, so these tests need no FreeCAD), exact-math checks of the 2D sector, the
segment parameters, the voxel-grid axis conventions, and - where FreeCAD and stltovoxel exist - the project's own voxelizer.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import sdf_primitives as sp  # noqa: E402
from src.sdf_primitives import deg  # noqa: E402

REF = json.loads((ROOT / "tests" / "data" / "sdf_reference.json").read_text())


def corner_box(l, w, h):
    return sp.Placed(sp.Box((l / 2, w / 2, h / 2)), translation=np.array([l / 2, w / 2, h / 2]))


# FreeCAD shape name -> the SDF primitive that must describe the same solid
PAIRS = {
    "box": lambda: corner_box(2.0, 3.0, 4.0),
    "sphere": lambda: sp.Sphere(1.5),
    "sphere_u0_180": lambda: sp.Sphere(1.5, u=deg(0, 180)),
    "sphere_lat_plane": lambda: sp.Sphere(1.5, v=deg(-30, 60), latitude="plane"),
    "sphere_lat_plane_u": lambda: sp.Sphere(1.5, u=deg(0, 250), v=deg(20, 70), latitude="plane"),
    "cylinder": lambda: sp.Cylinder(1.0, 2.0),
    "cylinder_u200": lambda: sp.Cylinder(1.0, 2.0, u=deg(0, 200)),
    "tube_full": lambda: sp.Cylinder(1.0, 2.0, inner_radius=0.5),
    "tube_u120": lambda: sp.Cylinder(1.0, 2.0, u=deg(0, 120), inner_radius=0.5),
    "cone": lambda: sp.Cone(1.0, 0.4, 2.0),
    "cone_u90": lambda: sp.Cone(1.0, 0.4, 2.0, u=deg(0, 90)),
    "cone_point": lambda: sp.Cone(1.0, 0.0, 2.0),
    "torus": lambda: sp.Torus(3.0, 1.0),
    "torus_u90": lambda: sp.Torus(3.0, 1.0, u=deg(0, 90)),
    "torus_u250": lambda: sp.Torus(3.0, 1.0, u=deg(0, 250)),
    "torus_v": lambda: sp.Torus(3.0, 1.0, v=deg(-100, 30)),         # Part::Torus Angle1=-30, Angle2=100: v = (-Angle2, -Angle1)
    "torus_v_u": lambda: sp.Torus(3.0, 1.0, u=deg(0, 150), v=deg(-120, 60)),
    "wedge": lambda: sp.Wedge(0, 0, 0, 0.5, 0.5, 3, 2, 2, 1.5, 2.5),
    "ellipsoid": lambda: sp.Ellipsoid(1.2, 0.8, 0.5),              # Part::Ellipsoid(Radius1=0.5, Radius2=1.2, Radius3=0.8)
    "prism6": lambda: sp.Prism(6, 1.2, 2.0),
    "prism3": lambda: sp.Prism(3, 1.0, 1.5),
}
EXACT_LOOSE = {"ellipsoid"}                                       # first-order distance: not a bound, only the sign is exact


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_sign_matches_freecad_everywhere_off_the_surface(name):
    r = REF[name]
    P, inside, dist = np.array(r["points"]), np.array(r["inside"]), np.array(r["dist"])
    s = PAIRS[name]().sdf(P)
    off = dist > 0.004                                              # closer than that: on the boundary's tolerance
    assert ((s <= 0) == inside)[off].all(), f"{name}: {(((s <= 0) != inside) & off).sum()} sign disagreements"
    assert off.mean() > 0.95 and inside.any() and (~inside).any()


@pytest.mark.parametrize("name", sorted(set(PAIRS) - EXACT_LOOSE))
def test_distance_is_exact_or_a_tight_lower_bound(name):
    r = REF[name]
    P, dist = np.array(r["points"]), np.array(r["dist"])
    prim = PAIRS[name]()
    s = np.abs(prim.sdf(P))
    assert (s <= dist + 2e-3).all(), f"{name}: |sdf| exceeds FreeCAD's distance by {(s - dist).max():.4f}"       # never overestimates
    if prim.exact:
        assert np.abs(s - dist).max() < 2e-3, f"{name}: exact primitive off by {np.abs(s - dist).max():.4f}"
    else:
        assert (dist - s).max() < 0.5 * dist.max(), name                                                          # a bound, not a guess
        near = dist < 0.05                                                                                          # close to the surface it is tight
        assert (dist - s)[near].max() < 0.031, f"{name}: bound loose by {(dist - s)[near].max():.4f} near the surface"


def test_ellipsoid_distance_is_close_near_the_surface():
    r = REF["ellipsoid"]
    P, dist = np.array(r["points"]), np.array(r["dist"])
    near = dist < 0.05
    assert np.abs(np.abs(PAIRS["ellipsoid"]().sdf(P)) - dist)[near].max() < 0.05


def test_freecad_volumes_agree_with_the_analytic_solids():
    """The solids are the same size: Monte Carlo volume of the SDF against FreeCAD's exact volume."""
    rng = np.random.default_rng(0)
    for name in ("sphere_lat_plane", "torus_u250", "tube_u120", "cone_u90", "wedge", "prism6"):
        prim = PAIRS[name]()
        lo, hi = prim.bounds()
        p = rng.uniform(lo, hi, size=(200000, 3))
        vol = (prim.sdf(p) <= 0).mean() * np.prod(hi - lo)
        assert abs(vol - REF[name]["volume"]) < 0.03 * REF[name]["volume"] + 0.02, (name, vol, REF[name]["volume"])


# ------------------------------------------------------------------------------------------- exact math, no FreeCAD

@pytest.mark.parametrize("u0,u1", [(0.0, deg(60)), (deg(-30), deg(100)), (deg(20), deg(200)), (deg(-100), deg(150)), (0.0, deg(180))])
def test_sector2d_is_the_exact_distance(u0, u1):
    """Against the brute-force distance to the two boundary rays (sampled finely), convex and reflex sectors."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(-3, 3, size=(600, 2))
    s = sp.sector2d(pts[:, 0], pts[:, 1], u0, u1)
    t = np.linspace(0, 8, 4001)
    rays = np.concatenate([np.c_[t * np.cos(a), t * np.sin(a)] for a in (u0, u1)])
    brute = np.sqrt(((pts[:, None, :] - rays[None]) ** 2).sum(-1)).min(1)
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    inside = ((ang - u0) % (2 * np.pi)) <= (u1 - u0)
    assert np.allclose(np.abs(s), brute, atol=5e-3)
    assert ((s <= 0) == inside).all()


def test_segment_parameters_are_the_parameter_region():
    """Sphere and torus segments contain exactly the points whose (u, v, radius) lie in the ranges."""
    rng = np.random.default_rng(2)
    u, v = deg(30, 200), deg(-20, 55)
    sph = sp.Sphere(2.0, u=u, v=v, inner_radius=0.5)
    p = rng.uniform(-2.5, 2.5, (20000, 3))
    r = np.linalg.norm(p, axis=1)
    lon = np.arctan2(p[:, 1], p[:, 0]) % (2 * np.pi)
    lat = np.arcsin(p[:, 2] / np.maximum(r, 1e-12))
    want = (r <= 2.0) & (r >= 0.5) & (lon >= u[0]) & (lon <= u[1]) & (lat >= v[0]) & (lat <= v[1])
    off = np.abs(sph.sdf(p)) > 1e-6
    assert ((sph.sdf(p) <= 0) == want)[off].all()

    tor = sp.Torus(3.0, 1.0, u=deg(-40, 120), v=deg(-60, 100), inner_minor=0.3)
    q = rng.uniform(-4.2, 4.2, (20000, 3))
    q[:, 2] = rng.uniform(-1.2, 1.2, 20000)
    rho = np.hypot(q[:, 0], q[:, 1])
    minor = np.hypot(rho - 3.0, q[:, 2])
    lon = np.arctan2(q[:, 1], q[:, 0])
    va = np.arctan2(q[:, 2], rho - 3.0)
    want = (minor <= 1.0) & (minor >= 0.3) & (lon >= deg(-40)) & (lon <= deg(120)) & (va >= deg(-60)) & (va <= deg(100))
    off = np.abs(tor.sdf(q)) > 1e-6
    assert ((tor.sdf(q) <= 0) == want)[off].all()


def test_latitude_cone_and_plane_differ_and_both_are_valid():
    cone_v = sp.Sphere(1.0, v=deg(20, 70), latitude="cone")
    plane_v = sp.Sphere(1.0, v=deg(20, 70), latitude="plane")
    p = np.array([[0.05, 0.0, 0.5]])                                    # latitude ~84 deg: outside the cone band, inside the planes
    assert cone_v.sdf(p)[0] > 0 and plane_v.sdf(p)[0] <= 0
    with pytest.raises(ValueError):
        sp.Sphere(1.0, v=deg(20, 70), latitude="cube").sdf(p)


def test_placement_and_boolean_operators():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])     # 90 deg about z
    placed = sp.Placed(sp.Box((2.0, 0.5, 0.5)), rotation=R, translation=np.array([1.0, 2.0, 3.0]))
    assert placed.sdf(np.array([[1.0, 4.0, 3.0]]))[0] <= 0 and placed.sdf(np.array([[3.0, 2.0, 3.0]]))[0] > 0     # long axis now along y
    lo, hi = placed.bounds()
    assert np.allclose(lo, [0.5, 0.0, 2.5]) and np.allclose(hi, [1.5, 4.0, 3.5])
    box_minus_sphere = sp.Box((1, 1, 1)) - sp.Sphere(0.8)
    assert box_minus_sphere.sdf(np.array([[0.0, 0.0, 0.0]]))[0] > 0             # the bite
    assert box_minus_sphere.sdf(np.array([[0.95, 0.95, 0.95]]))[0] <= 0          # a corner of the box survives
    assert (sp.Sphere(1.0) | sp.Sphere(1.0)).exact is False and (sp.Sphere(1.0) & sp.Box((0.5, 0.5, 0.5))).sdf(np.array([[0.8, 0, 0]]))[0] > 0      # in the ball, outside the box


def test_voxel_grid_uses_the_projects_array_frame():
    """Array axis 0 is z, axis 1 is y running BACKWARDS, axis 2 is x (reco_engine.placement)."""
    def centroid(offset):
        occ = sp.occupancy_grid(sp.Placed(sp.Sphere(5.0), translation=np.array(offset)), 64, 40.0)
        return np.argwhere(occ).mean(axis=0) - 31.5
    cz, cy, cx = centroid((0, 0, 8.0)), centroid((0, 8.0, 0)), centroid((8.0, 0, 0))
    assert cz[0] > 10 and abs(cz[1]) < 1 and abs(cz[2]) < 1                      # +z -> larger axis-0 index
    assert cy[1] < -10                                                          # +y -> SMALLER axis-1 index
    assert cx[2] > 10                                                           # +x -> larger axis-2 index
    grid = sp.sdf_grid(sp.Sphere(10.0), 64, 40.0)
    assert grid.shape == (64, 64, 64) and grid.min() < -9.0 and abs(grid[31, 31, 31] + 10.0) < 0.6


# --------------------------------------------------------------------------------- against the project's own voxelizer

def _freecad():
    from src.freecad_path import find_freecadcmd  # $SHAPE_RECO_FREECADCMD, PATH, /usr/bin/freecadcmd (T-04)
    return find_freecadcmd()


@pytest.mark.skipif(_freecad() is None, reason="needs FreeCAD")
@pytest.mark.parametrize("voxelizer", ["parity", "stltovoxel"])
def test_sdf_occupancy_matches_the_projects_voxelization_of_the_part_shape(voxelizer):
    """A Part sphere segment and torus segment, exported to STL, voxelized in the in-scale mode (fixed, anchor origin), against
    the SDF rasterized at the voxel centres: same solid, same place - so the frame and the segment definitions agree end to end.
    The parity default (T-19) samples the same centres: no shift, IoU 1.000 / 0.999 (measured 2026-10-01); the legacy
    stltovoxel rounds its anchor, up to one voxel off."""
    if voxelizer == "stltovoxel" and not (shutil.which("stltovoxel") or (Path.home() / ".local/bin/stltovoxel").is_file()):
        pytest.skip("stltovoxel not installed")
    from src.voxelization import voxelize_stl_to_bin
    tmp = Path(tempfile.mkdtemp())
    script = tmp / "build.py"
    script.write_text(f'''
import Part, MeshPart, FreeCAD as App
V = App.Vector
shapes = {{"seg_sphere": Part.makeSphere(12.0, V(0, 0, 0), V(0, 0, 1), -30, 60, 250),
          "seg_torus": Part.makeTorus(14.0, 4.0, V(0, 0, 0), V(0, 0, 1), 0, 360, 200)}}
for name, s in shapes.items():
    mesh = MeshPart.meshFromShape(Shape=s, LinearDeflection=0.02, AngularDeflection=0.05)
    mesh.write(r"{tmp}/" + name + ".stl")
''')
    subprocess.run([_freecad(), str(script)], capture_output=True, timeout=180)
    cases = {"seg_sphere": sp.Sphere(12.0, u=deg(0, 250), v=deg(-30, 60), latitude="plane"),
             "seg_torus": sp.Torus(14.0, 4.0, u=deg(0, 200))}
    for name, prim in cases.items():
        stl = tmp / f"{name}.stl"
        assert stl.is_file(), name
        out = tmp / f"{name}.bin"
        assert voxelize_stl_to_bin(str(stl), str(out), 64, scale_mode="fixed", scale_mm=40.0, anchor="origin",
                                   voxelizer=voxelizer)
        vox = np.fromfile(out, dtype=np.uint8).reshape(64, 64, 64) > 0
        mine = sp.occupancy_grid(prim, 64, 40.0) > 0
        best, best_shift = 0.0, None
        for shift in np.ndindex(3, 3, 3):                                       # the voxelizer places the mesh corner at a ROUNDED voxel
            m = np.roll(mine, [k - 1 for k in shift], axis=(0, 1, 2))            # index (voxelization.py): up to 1 voxel offset per axis
            iou = (vox & m).sum() / max((vox | m).sum(), 1)
            if iou > best:
                best, best_shift = iou, tuple(k - 1 for k in shift)
        aligned = (vox & mine).sum() / max((vox | mine).sum(), 1)
        # the same solid in the same place up to the voxelizer's own rounding; a thin body (the torus tube is 6 voxels across)
        # loses IoU to its boundary rule, hence the thresholds
        assert best > 0.88 and aligned > 0.85 and max(abs(k) for k in best_shift) <= 1, (name, best, aligned, best_shift)
        if voxelizer == "parity":
            assert best_shift == (0, 0, 0) and aligned > 0.99, (name, aligned, best_shift)
