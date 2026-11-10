"""The starting-shape stage (reco_engine/start_shape.py): the most surface, then the least excess, every operand of a
composite justified by surface only it carries; and the STEP surface inventory (reco_engine/brep.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

m3d = pytest.importorskip("manifold3d")
pytest.importorskip("igl")

from reco_engine import residual_loop as rl, start_shape as ss  # noqa: E402
from src.closest_point_field import SurfaceField  # noqa: E402
from src.freecad_path import find_freecadcmd  # noqa: E402

OFF = np.array([30.0, -12.0, 5.0])
M = m3d.Manifold


def field_of(solid):
    mesh = solid.translate(tuple(OFF)).to_mesh()
    return SurfaceField(np.asarray(mesh.vert_properties, float)[:, :3], np.asarray(mesh.tri_verts, np.int64))


def choose(field):
    cfg = rl.ResidualConfig(grid="tau", resolution=48, field_backend="igl")
    spec, _ = rl._tau_spec(field, cfg)
    t_in = rl._channels(field, spec, False, "igl")
    surfaces, mesh = ss.surface_inventory(None, field, channels=t_in, spec=spec)
    graded = [(n, p, ss.grade_start(p, field, mesh)) for n, p in ss.start_candidates(surfaces, field, t_in, spec)]
    return ss.choose_start(graded), graded


def inventory(field):
    cfg = rl.ResidualConfig(grid="tau", resolution=48, field_backend="igl")
    spec, _ = rl._tau_spec(field, cfg)
    return ss.surface_inventory(None, field, channels=rl._channels(field, spec, False, "igl"), spec=spec)[0]


def test_a_stepped_box_starts_with_its_box():
    part = M.cube((20.0, 10.0, 10.0), center=True) - M.cube((8.0, 12.0, 6.0), center=True).translate((6.0, 0.0, 3.0))
    (name, prim, g), graded = choose(field_of(part))
    assert prim.kind == "box", name
    comp = next(gg for n, _, gg in graded if n == "box & cylinder")
    assert "cylinder" in comp["unjustified"]              # its cylinder carries no surface of its own


def test_a_cylinder_with_a_flat_starts_exact():
    part = M.cylinder(20.0, 6.0, 6.0, circular_segments=128, center=True) - M.cube((20, 20, 30), center=True).translate((14.0, 0, 0))
    (name, prim, g), _ = choose(field_of(part))
    assert g["coverage"] > 0.95 and abs(g["excess_share"]) < 0.02, (name, g)


@pytest.mark.skipif(find_freecadcmd() is None, reason="no freecadcmd")
def test_step_inventory_recovers_the_faces(tmp_path, monkeypatch):
    """A half-pipe written as STEP by FreeCAD: the inventory finds its outer cylinder (convex, dominating), its bore
    (concave, not dominating) and the flat side (dominating)."""
    import subprocess
    monkeypatch.setenv("SHAPE_RECO_BREP_CACHE", str(tmp_path / "cache"))
    step = tmp_path / "halfpipe.step"
    script = tmp_path / "make.py"
    script.write_text(
        "import Part, FreeCAD as App\n"
        "t = Part.makeCylinder(19.2, 80).cut(Part.makeCylinder(12.5, 80))\n"
        "b = Part.makeBox(50, 50, 90, App.Vector(-25, 0, -5))\n"
        f"t.cut(b).exportStep({str(step)!r})\n")
    subprocess.run([find_freecadcmd(), str(script)], capture_output=True, timeout=120)
    assert step.is_file()
    from reco_engine.brep import step_faces
    inv = step_faces(step)
    cyl = [f for f in inv["faces"] if f["type"] == "Cylinder"]
    outer = next(f for f in cyl if abs(f["radius"] - 19.2) < 1e-6)
    bore = next(f for f in cyl if abs(f["radius"] - 12.5) < 1e-6)
    assert outer["convex"] and outer["dominating"] and not bore["convex"] and not bore["dominating"]
    assert np.allclose(np.abs(outer["axis"]), [0, 0, 1], atol=1e-9)
    flat = [f for f in inv["faces"] if f["type"] == "Plane" and abs(f["normal"][1] - 1.0) < 1e-9]
    assert flat and all(f["dominating"] for f in flat)
    assert len(inv["face_id"]) == len(inv["triangles"]) and set(inv["face_id"]) <= set(range(len(inv["faces"])))


def test_cuts_first_after_a_starting_shape_only_by_default():
    """D-003: after a superset start the excess is cut before anything is added; the legacy path keeps adds first; the
    flag forces either order."""
    assert rl._cuts_first(rl.ResidualConfig(start_shape="auto"))
    assert rl._cuts_first(rl.ResidualConfig(start_shape="oriented box"))
    assert not rl._cuts_first(rl.ResidualConfig())
    assert rl._cuts_first(rl.ResidualConfig(cuts_first=True))
    assert not rl._cuts_first(rl.ResidualConfig(start_shape="auto", cuts_first=False))


def test_the_stl_inventory_finds_the_cylinder_not_its_facets():
    """A cylinder with a flat: one convex, dominating cylinder Surface that owns the lateral facets; the planes are the
    flat and the caps, not dozens of 0.65 % facet strips."""
    part = M.cylinder(20.0, 6.0, 6.0, circular_segments=128, center=True) - M.cube((20, 20, 30), center=True).translate((14.0, 0, 0))
    surfaces = inventory(field_of(part))
    cyl = [s for s in surfaces if s.kind == "cylinder"]
    assert len(cyl) == 1, [(s.kind, round(s.area_share, 3)) for s in surfaces]
    c = cyl[0]
    assert abs(c.params["radius"] - 6.0) < 0.06 and abs(c.params["axis"][2]) > 0.99
    assert c.params["convex"] and c.dominating and c.area_share > 0.5
    planes = [s for s in surfaces if s.kind == "plane"]
    assert len(planes) <= 4 and sum(s.area_share for s in planes) < 0.45


def test_a_bore_is_a_concave_cylinder():
    part = M.cube((20.0, 14.0, 10.0), center=True) - M.cylinder(12.0, 3.0, 3.0, circular_segments=96, center=True).translate((4.0, -2.0, 0.0))
    surfaces = inventory(field_of(part))
    cyl = [s for s in surfaces if s.kind == "cylinder"]
    assert len(cyl) == 1 and not cyl[0].params["convex"] and not cyl[0].dominating
    assert sum(1 for s in surfaces if s.kind == "plane" and s.dominating) == 6


def test_budget_counts_the_major_surfaces():
    """D-liner 2026-09-26: the genus budget was 5 for a part with 19 major faces; the inventory lifts it."""
    cfg = rl.ResidualConfig()
    major = [ss.Surface("plane", 0.01, False, {}) for _ in range(19)] + [ss.Surface("cylinder", 0.001, False, {})] * 30
    assert rl._ops_budget(5, major, cfg)[0] == 24
    assert rl._ops_budget(5, major[:2], cfg)[0] == 7                  # 1 + margin 4 + 2, above the genus rule
    assert rl._ops_budget(5, [ss.Surface("plane", 0.01, False, {})] * 70, cfg)[0] == 60
    assert rl._ops_budget(5, [ss.Surface("torus", 0.5, False, {})] * 9, cfg)[0] == 5      # tori / cones do not count
    assert rl._ops_budget(5, major, rl.ResidualConfig(budget_from_surfaces=False))[0] == 5


def test_the_score_is_one_for_the_exact_start_and_less_for_a_looser_one():
    """score = covered share of the generated volume x surface coverage: box & cylinder on the cylinder with a flat is
    the part itself (1.0); its bounding box covers 4 faces of 6 and carries excess."""
    part = M.cylinder(20.0, 6.0, 6.0, circular_segments=128, center=True) - M.cube((20, 20, 30), center=True).translate((14.0, 0, 0))
    (name, prim, g), graded = choose(field_of(part))
    assert g["score"] > 0.97, (name, g)
    box = next(gg for n, _, gg in graded if n == "oriented box")
    assert 0.2 < box["score"] < 0.8 and box["covered"] is not None and box["covered"] < 0.9


def test_varying_an_oversized_cylinder_brings_its_radius_back():
    """Coordinate search at scale h on the start's parameters: a cylinder guess 0.6 too wide and 0.3 off-centre on a
    plain cylinder part shrinks and recentres within h/2 (3 units of h), the score rises towards 1."""
    part = M.cylinder(20.0, 6.0, 6.0, circular_segments=128, center=True)
    field = field_of(part)
    surfaces, mesh = ss.surface_inventory(None, field)
    from reco_engine.result_schema import Primitive
    guess = Primitive(kind="cylinder", center_mm=tuple(OFF + np.array([0.3, 0.0, 0.0])),
                      dims_mm=dict(radius=6.6, half_length=10.0))
    g0 = ss.grade_start(guess, field, mesh, h=0.3)
    best, g, steps = ss.vary_start(guess, field, mesh, h=0.3, rounds=8)
    assert g["score_soft"] > g0["score_soft"] + 0.05, (g0["score_soft"], g["score_soft"], steps)
    assert abs(best.dims_mm["radius"] - 6.0) <= 0.15 and abs(best.center_mm[0] - OFF[0]) <= 0.15, (best, steps)
