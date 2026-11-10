#!/usr/bin/env python3
"""reco_engine/polish.py: a curved primitive fitted to the triangles of a tessellated surface (biased by the chords'
sag) is refitted to the mesh's vertices, which lie on the true surface - exact radius, unchanged planes, no harm to a
primitive that is already right. Synthetic meshes (trimesh), no checkpoints, no FreeCAD."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

trimesh = pytest.importorskip("trimesh")

from reco_engine.polish import polish_result, vertex_samples  # noqa: E402
from reco_engine.result_schema import Operation, Primitive, ReconstructionResult  # noqa: E402

CENTRE = np.array([1.5, -2.0, 0.75])


def _result(*ops):
    return ReconstructionResult(input_path="synthetic", operations=[Operation(op, prim) for op, prim in ops])


def _frustum(r1, r2, h, n=96):
    """A closed cone frustum along z, radius r1 at z = -h and r2 at +h, vertices exactly on the cone."""
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ring = lambda r, z: np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z)]   # noqa: E731
    V = np.r_[ring(r1, -h), ring(r2, h), [[0, 0, -h], [0, 0, h]]]
    i, j = np.arange(n), (np.arange(n) + 1) % n
    F = np.r_[np.c_[i, j, n + j], np.c_[i, n + j, n + i], np.c_[np.full(n, 2 * n), j, i], np.c_[np.full(n, 2 * n + 1), n + i, n + j]]
    return trimesh.Trimesh(V, F, process=True)


def test_vertex_samples_merge_a_triangle_soup():
    m = trimesh.creation.icosphere(subdivisions=2, radius=3.0)
    soup = m.vertices[m.faces].reshape(-1, 3)
    P, W, edge = vertex_samples(soup, np.arange(len(soup)).reshape(-1, 3))
    assert len(P) == len(m.vertices) and W.mean() == pytest.approx(1.0) and edge > 0
    assert np.allclose(np.linalg.norm(P, axis=1), 3.0)


def test_sphere_radius_shortened_by_the_sag_is_restored():
    m = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    m.apply_translation(CENTRE)
    biased = Primitive("sphere", center_mm=tuple(CENTRE + [0.002, -0.001, 0.0]), dims_mm={"radius": 10.0 * (1 - 7e-4)})
    res = _result(("add", biased))
    rep = polish_result(res, m.vertices, m.faces)
    p = res.operations[0].primitive
    assert rep.kept == 1 and rep.rms_after < 1e-3 * rep.rms_before
    assert abs(p.dims_mm["radius"] - 10.0) < 1e-6 and np.allclose(p.center_mm, CENTRE, atol=1e-6)


def test_cylinder_radius_restored_and_caps_unchanged():
    m = trimesh.creation.cylinder(radius=5.0, height=20.0, sections=64)
    m.apply_translation(CENTRE)
    res = _result(("add", Primitive("cylinder", center_mm=tuple(CENTRE), dims_mm={"radius": 4.99, "half_length": 10.0})))
    rep = polish_result(res, m.vertices, m.faces)
    d = res.operations[0].primitive.dims_mm
    assert rep.kept == 1 and abs(d["radius"] - 5.0) < 1e-6 and abs(d["half_length"] - 10.0) < 1e-6


def test_cone_frustum_radii_restored():
    m = _frustum(6.0, 3.0, 4.0)
    res = _result(("add", Primitive("cone", dims_mm={"radius1": 5.99, "radius2": 2.995, "half_length": 4.0})))
    rep = polish_result(res, m.vertices, m.faces)
    d = res.operations[0].primitive.dims_mm
    assert rep.kept == 1
    assert abs(d["radius1"] - 6.0) < 1e-6 and abs(d["radius2"] - 3.0) < 1e-6 and abs(d["half_length"] - 4.0) < 1e-6


def test_torus_radii_restored():
    m = trimesh.creation.torus(major_radius=10.0, minor_radius=3.0, major_sections=96, minor_sections=48)
    res = _result(("add", Primitive("torus", dims_mm={"major_radius": 10.01, "minor_radius": 2.99})))
    rep = polish_result(res, m.vertices, m.faces)
    d = res.operations[0].primitive.dims_mm
    assert rep.kept == 1 and abs(d["major_radius"] - 10.0) < 1e-6 and abs(d["minor_radius"] - 3.0) < 1e-6


def test_a_cut_cylinder_is_refitted_from_the_wall_of_its_hole():
    pytest.importorskip("manifold3d")
    box = trimesh.creation.box(extents=(20.0, 20.0, 10.0))
    hole = trimesh.creation.cylinder(radius=4.0, height=14.0, sections=96)
    m = trimesh.boolean.difference([box, hole], engine="manifold")
    res = _result(("add", Primitive("box", dims_mm={"x_half": 10.0, "y_half": 10.0, "z_half": 5.0})),
                  ("cut", Primitive("cylinder", dims_mm={"radius": 4.01, "half_length": 7.0})))
    rep = polish_result(res, m.vertices, m.faces)
    assert [o.kind for o in rep.ops] == ["cylinder"] and rep.kept == 1             # the box is not curved: not touched
    assert abs(res.operations[1].primitive.dims_mm["radius"] - 4.0) < 1e-6
    assert res.operations[0].primitive.dims_mm == {"x_half": 10.0, "y_half": 10.0, "z_half": 5.0}


def test_an_exact_primitive_is_left_where_it_is():
    m = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    exact = Primitive("sphere", dims_mm={"radius": 10.0})
    res = _result(("add", exact))
    polish_result(res, m.vertices, m.faces)
    p = res.operations[0].primitive
    assert abs(p.dims_mm["radius"] - 10.0) < 1e-9 and np.allclose(p.center_mm, 0.0, atol=1e-9)


def test_rejected_operations_are_not_touched():
    m = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    res = _result(("add", Primitive("sphere", dims_mm={"radius": 9.99})))
    res.operations[0].accepted = False
    rep = polish_result(res, m.vertices, m.faces)
    assert rep.ops == [] and res.operations[0].primitive.dims_mm["radius"] == 9.99


def _fake_build(model, out_dir, name, field):
    return dict(valid=True, n_solids=1, step=f"{name}.step", stl=f"{name}.stl", chamfer=0.01)


def test_the_loop_polishes_when_asked():
    from src.closest_point_field import SurfaceField
    from reco_engine import cascade, residual_loop as rl
    m = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    m.apply_translation(CENTRE)
    done = {}
    for polish in (False, True):
        cfg = rl.ResidualConfig(resolution=32, max_ops=3, builder=_fake_build, registry=cascade.DEFAULT_REGISTRY, polish=polish)
        done[polish] = list(rl.residual_steps(SurfaceField(m.vertices, m.faces), cfg))[-1]
    assert done[False].kind == "done" and done[False].info["polish"] is None
    spheres = {k: [o.primitive for o in s.result.operations if o.accepted and o.primitive.kind == "sphere"] for k, s in done.items()}
    assert spheres[False] and spheres[True], "the loop explains a sphere with a sphere"
    assert done[True].info["polish"]["kept"] >= 1 and "polish:" in done[True].info["stopped"]
    err = {k: abs(v[0].dims_mm["radius"] - 10.0) for k, v in spheres.items()}
    assert err[True] < 1e-6 < err[False]


def test_a_primitive_on_another_surface_is_not_dragged():
    """A sphere that sits partly on a box's faces (a failed program): its vertices do not lie on one sphere, so the
    refit improves only a little - it is not kept (CONSISTENT)."""
    m = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    for _ in range(4):
        m = m.subdivide()                                            # enough vertices near the sphere to try a fit
    res = _result(("add", Primitive("sphere", dims_mm={"radius": 10.05})))
    rep = polish_result(res, m.vertices, m.faces)
    assert rep.ops[0].points > 100 and "do not lie on one sphere" in rep.ops[0].reason
    assert rep.kept == 0 and res.operations[0].primitive.dims_mm["radius"] == 10.05
