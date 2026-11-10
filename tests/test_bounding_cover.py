"""The first add's bounding cover (residual_loop._bounding_cover): a round part gets its bounding cylinder - a sector
for a half-pipe like the liner LINER_001 - and a box-like part keeps its oriented bounding box."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

trimesh = pytest.importorskip("trimesh")

from reco_engine import cascade, residual_loop as rl  # noqa: E402
from src.closest_point_field import SurfaceField  # noqa: E402

CENTRE = np.array([30.0, -12.0, 5.0])


def half_pipe(r_out=19.2, r_in=12.5, length=80.0):
    """A half-pipe (the y <= 0 half of a tube along z), moved off the origin."""
    m3d = pytest.importorskip("manifold3d")
    M = m3d.Manifold
    tube = M.cylinder(length, r_out, r_out, circular_segments=128, center=True) - \
        M.cylinder(length + 2, r_in, r_in, circular_segments=128, center=True)
    half = tube - M.cube((3 * r_out, 2 * r_out, length + 4), center=True).translate((0.0, r_out, 0.0))   # keep y <= 0
    mesh = half.translate(tuple(CENTRE)).to_mesh()
    return SurfaceField(np.asarray(mesh.vert_properties, float)[:, :3], np.asarray(mesh.tri_verts, np.int64))


def channels(field, resolution=48):
    cfg = rl.ResidualConfig(grid="tau", resolution=resolution, field_backend="igl")
    spec, _ = rl._tau_spec(field, cfg)
    return rl._channels(field, spec, False, "igl"), spec


def test_half_pipe_gets_a_bounding_half_cylinder():
    pytest.importorskip("igl")
    field = half_pipe()
    t_in, spec = channels(field)
    cyl = cascade.bounding_cylinder(t_in, spec, field=field)
    assert cyl is not None and cyl.kind == "cylinder"
    assert abs(cyl.dims_mm["radius"] - 19.2) < 0.2 and abs(2 * cyl.dims_mm["half_length"] - 80.0) < 0.5
    assert np.linalg.norm(np.asarray(cyl.center_mm)[:2] - CENTRE[:2]) < 0.3
    assert cyl.u is not None and abs((cyl.u[1] - cyl.u[0]) - np.pi) < np.radians(5)      # a 180 deg sector
    prim, how = rl._bounding_cover(t_in, spec, field, spec.pitch)
    assert prim.kind == "cylinder" and how.startswith("bounding cylinder")
    on, inside = cascade.surface_agreement(prim, field, 0.2)
    assert inside >= rl.COVER_MIN_INSIDE


def test_a_box_keeps_its_bounding_box():
    pytest.importorskip("igl")
    box = trimesh.creation.box(extents=(40.0, 10.0, 25.0))
    box.apply_translation(CENTRE)
    field = SurfaceField(np.asarray(box.vertices, float), np.asarray(box.faces))
    t_in, spec = channels(field, resolution=32)
    prim, how = rl._bounding_cover(t_in, spec, field, spec.pitch)
    assert prim.kind == "box" and how.startswith(("oriented box", "world-axis box")) and "100% of its surface" in how


def block_with_bore_and_pocket(r=6.0):
    """A block with a bore along z (radius r, off-centre) and a flat-floored pocket recessed into the bore's wall along
    most of its length - the liner's bore + pockets in small: one leftover component, only part of it cylindrical."""
    m3d = pytest.importorskip("manifold3d")
    M = m3d.Manifold
    block = M.cube((30.0, 30.0, 60.0), center=True)
    bore = M.cylinder(70.0, r, r, circular_segments=128, center=True)
    pocket = M.cube((8.0, 4.0, 44.0), center=True).translate((0.0, -r - 1.0, 0.0))     # floor 3 below the bore
    part = (block - bore - pocket).translate(tuple(CENTRE))
    mesh = part.to_mesh()
    return SurfaceField(np.asarray(mesh.vert_properties, float)[:, :3], np.asarray(mesh.tri_verts, np.int64))


def test_strict_cylinders_find_the_bore_next_to_a_pocket():
    pytest.importorskip("igl")
    from reco_engine.result_schema import Primitive
    field = block_with_bore_and_pocket()
    spec = cascade.box_grid_spec(np.asarray(field.bounds[0]), np.asarray(field.bounds[1]), 1.0)
    # the leftover of a model that is the full block: bore + pocket, as voxels
    c = spec.centres()
    rel = c - CENTRE
    bore = np.hypot(rel[..., 0], rel[..., 1]) <= 6.0
    pocket = (np.abs(rel[..., 0]) <= 4.0) & (np.abs(rel[..., 1] + 7.0) <= 2.0) & (np.abs(rel[..., 2]) <= 22.0)
    comp = (bore | pocket) & (np.abs(rel[..., 2]) <= 30.0)
    ch = rl._voxel_channels(comp, spec)
    cands = cascade.strict_cylinders(ch, spec, field)
    assert cands, "no strict cylinder"
    best = cands[0]
    assert abs(best.dims_mm["radius"] - 6.0) < 0.2
    assert np.linalg.norm(np.asarray(best.center_mm)[:2] - CENTRE[:2]) < 0.2
    # the surface prefers the true bore to a compromise pulled towards the pocket floor
    true = Primitive("cylinder", (0, 0, 0, 1), tuple(CENTRE), {"radius": 6.0, "half_length": 30.0})
    shifted = Primitive("cylinder", (0, 0, 0, 1), tuple(CENTRE + (0.0, -1.5, 0.0)), {"radius": 6.8, "half_length": 30.0})
    assert cascade.near_surface_agreement(true, ch, spec, field) > cascade.near_surface_agreement(shifted, ch, spec, field) + 0.1


def test_half_on_half_far_beats_close_everywhere():
    """The first-add grade (the user's rule): a cover whose surface is half exactly on the input and half far away (one cut
    finishes it) beats one that is close to the input everywhere without matching it (slivers)."""
    pytest.importorskip("igl")
    from reco_engine.result_schema import Primitive
    box = trimesh.creation.box(extents=(20.0, 10.0, 10.0))
    box.apply_translation(CENTRE)
    field = SurfaceField(np.asarray(box.vertices, float), np.asarray(box.faces))
    t_in, spec = channels(field, resolution=48)
    c, occ = cascade.grid_centres(spec), np.asarray(t_in[0]) > 0.5
    # half on / half far: the same box extended by 10 beyond one end (its other faces lie on the input)
    long = Primitive("box", (0, 0, 0, 1), tuple(CENTRE + (5.0, 0, 0)), dict(x_half=15.0, y_half=5.0, z_half=5.0))
    # close everywhere: the box grown by half a voxel on every side
    grown = Primitive("box", (0, 0, 0, 1), tuple(CENTRE), dict(x_half=10.0 + spec.pitch * 0.6, y_half=5.0 + spec.pitch * 0.6,
                                                               z_half=5.0 + spec.pitch * 0.6))
    g_long = rl._grade_cover(long, spec, c, occ, field, spec.pitch)
    g_grown = rl._grade_cover(grown, spec, c, occ, field, spec.pitch)
    assert g_long["on"] > 0.4 and g_long["far"] > 0.2 and g_long["pieces"] == 1
    assert g_grown["near"] > 0.9
    assert g_long["score"] > g_grown["score"]


def test_a_stepped_box_keeps_a_box_as_its_first_fit():
    """A box with a step cut out is all planes: box & bounding cylinder would round off the step's corner (its cylinder
    surface is not the input's) and must not win the first fit (bs_box_step, 2026-09-25)."""
    pytest.importorskip("igl")
    m3d = pytest.importorskip("manifold3d")
    M = m3d.Manifold
    part = (M.cube((20.0, 10.0, 10.0), center=True) - M.cube((8.0, 12.0, 6.0), center=True).translate((6.0, 0.0, 3.0)))
    mesh = part.translate(tuple(CENTRE)).to_mesh()
    field = SurfaceField(np.asarray(mesh.vert_properties, float)[:, :3], np.asarray(mesh.tri_verts, np.int64))
    t_in, spec = channels(field, resolution=48)
    grades = []
    prim, how = rl._bounding_cover(t_in, spec, field, spec.pitch, grades_out=grades)
    assert prim.kind == "box", how
    comp = dict(grades).get("box & cylinder")
    assert comp is None or comp.get("part_on") is None or min(comp["part_on"]) < rl.COMPOSITE_PART_ON
