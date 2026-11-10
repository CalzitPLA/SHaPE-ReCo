"""reco_engine.cascade.BoxGridSpec: a box of cells at one pitch (the wall-thickness grid, D-005). The voxel<->world mapping
is checked on an asymmetric, off-centre tetrahedron (not a cube: a cube hides axis swaps and sign flips)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine import cascade, residual_loop as rl  # noqa: E402
from src.closest_point_field import SurfaceField  # noqa: E402
from src.field_channels import mesh_channels, voxel_channels  # noqa: E402

# off round coordinates: a voxel centre exactly ON a face is a tie (measured: every parity / winding-number disagreement on
# the round version lay at distance 0-9e-15 pitch from the surface), which says nothing about the axis mapping
TET_V = np.array([[3.0, -2.0, 7.0], [9.5, -1.0, 7.5], [4.0, 1.5, 8.0], [5.0, -0.5, 12.0]]) + (0.0123, 0.0371, 0.0529)
TET_F = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])


def tet_field():
    return SurfaceField(TET_V, TET_F)


def test_box_grid_centres_and_shape():
    spec = cascade.box_grid_spec(TET_V.min(0), TET_V.max(0), 0.25)
    c = spec.centres()
    assert c.shape == spec.shape + (3,) and len(set(spec.shape)) == 3                    # not a cube
    assert np.allclose(c[1, 0, 0] - c[0, 0, 0], (0.25, 0, 0)) and np.allclose(c[0, 1, 0] - c[0, 0, 0], (0, 0.25, 0))
    assert np.allclose(c[0, 0, 0], np.asarray(spec.lo) + 0.125)
    assert np.allclose(spec.hi, np.asarray(spec.lo) + np.asarray(spec.shape) * 0.25)
    assert np.allclose(cascade.to_world(spec, (2, 3, 4)), c[2, 3, 4])


def test_parity_and_channels_on_a_box_grid_match_an_independent_inside_test():
    igl = pytest.importorskip("igl")
    spec = cascade.box_grid_spec(TET_V.min(0) - 0.0777, TET_V.max(0), 0.2)       # grid not anchored to a vertex
    field = tet_field()
    occ = rl.parity_occupancy(field, spec)
    ref = (np.abs(igl.winding_number(TET_V, TET_F.astype(np.int64), spec.centres().reshape(-1, 3))) > 0.5).reshape(spec.shape)
    assert occ.shape == spec.shape and occ.sum() > 200
    assert (occ != ref).sum() == 0
    ch = mesh_channels(field, spec, inside=occ)
    assert ch.shape == (5,) + spec.shape and ((ch[0] > 0.5) == occ).all()
    # V points from each centre to its nearest surface point: x + V lies on the tetrahedron's surface (distance ~0)
    x = spec.centres().reshape(-1, 3)
    on = field.query(x + np.moveaxis(ch[2:5], 0, -1).reshape(-1, 3), exact=True).distance
    assert np.abs(on).max() < 1e-6
    vc = voxel_channels(occ, spec)
    assert vc.shape == ch.shape and ((vc[0] > 0.5) == occ).all()


def test_parity_on_a_cube_grid_is_unchanged():
    """The loop's parity now runs through field_channels.parity_axes: on its cube grid the answer is the same as the
    cube-only scanline it replaced (reimplemented here from the old code)."""
    field = tet_field()
    spec = cascade.grid_spec(TET_V.min(0) - 0.5, TET_V.max(0) + 0.5, 40)
    c = cascade.grid_centres(spec)
    pitch = float(c[1, 0, 0, 0] - c[0, 0, 0, 0])
    xs, ys, zs = c[:, 0, 0, 0], c[0, :, 0, 1] + 1.2345e-4 * pitch, c[0, 0, :, 2] + 2.3456e-4 * pitch
    r = len(xs)
    tri = field.vertices[field.faces]
    a, b, cc = tri[:, 0], tri[:, 1], tri[:, 2]
    j0 = np.searchsorted(ys, tri[:, :, 1].min(axis=1)); j1 = np.searchsorted(ys, tri[:, :, 1].max(axis=1))
    k0 = np.searchsorted(zs, tri[:, :, 2].min(axis=1)); k1 = np.searchsorted(zs, tri[:, :, 2].max(axis=1))
    counts = np.zeros((r + 1, r, r), np.int32)
    for f in np.nonzero((j1 > j0) & (k1 > k0))[0]:
        jj, kk = np.meshgrid(np.arange(j0[f], j1[f]), np.arange(k0[f], k1[f]), indexing="ij")
        py, pz = ys[jj], zs[kk]
        d = (b[f, 1] - a[f, 1]) * (cc[f, 2] - a[f, 2]) - (cc[f, 1] - a[f, 1]) * (b[f, 2] - a[f, 2])
        l1 = ((py - a[f, 1]) * (cc[f, 2] - a[f, 2]) - (cc[f, 1] - a[f, 1]) * (pz - a[f, 2])) / d
        l2 = ((b[f, 1] - a[f, 1]) * (pz - a[f, 2]) - (py - a[f, 1]) * (b[f, 2] - a[f, 2])) / d
        hit = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        x = a[f, 0] + l1[hit] * (b[f, 0] - a[f, 0]) + l2[hit] * (cc[f, 0] - a[f, 0])
        np.add.at(counts, (np.searchsorted(xs, x), jj[hit], kk[hit]), 1)
    old = (np.cumsum(counts, axis=0)[:r] % 2) == 1
    assert (rl.parity_occupancy(field, spec) == old).all() and old.sum() > 100


def test_voxel_fitters_answer_in_world_coordinates_on_a_box_grid():
    """The analytic voxel fitters take a cube; on a box grid the target is cropped into one (cascade._fit_cube) and the
    answer shifted back: an off-centre sphere and box are found where they are, with their size."""
    from reco_engine.result_schema import Primitive
    spec = cascade.box_grid_spec((-3.0, 10.0, 4.0), (21.0, 16.0, 40.0), 0.25)       # 104 x 32 x 152 cells
    c = spec.centres()
    sphere = np.linalg.norm(c - (15.3, 12.9, 30.1), axis=-1) <= 2.2
    ctx = cascade._Context(np.stack([sphere.astype(np.float32)] * 5), spec, None, None, None)
    p = cascade._fit_sphere(ctx)
    assert p is not None and np.linalg.norm(np.asarray(p.center_mm) - (15.3, 12.9, 30.1)) < 0.25
    assert abs(p.dims_mm["radius"] - 2.2) < 0.25
    box = Primitive("box", (0.0, 0.0, 0.3826834, 0.9238795), (2.4, 13.1, 20.7), {"x_half": 3.0, "y_half": 1.2, "z_half": 5.0})
    occ = cascade.primitive_occupancy(box, spec)
    ctx = cascade._Context(np.stack([occ.astype(np.float32)] * 5), spec, None, None, None)
    q = cascade._fit_frame_box(ctx, "frame")
    assert q is not None and np.linalg.norm(np.asarray(q.center_mm) - (2.4, 13.1, 20.7)) < 0.3
    assert sorted(round(v, 0) for v in q.dims_mm.values()) == [1.0, 3.0, 5.0]


def test_fit_cube_downsamples_a_big_target():
    occ = np.zeros((400, 60, 30), bool)
    occ[10:390, 5:50, 3:25] = True
    cube, offset, step = cascade._fit_cube(occ)
    assert step == 2 and cube.shape[0] <= cascade.MAX_FIT_CUBE and cube.any()        # 382 voxels -> 2 x 191
    idx = np.argwhere(cube)
    lo, hi = cascade._from_cube(idx.min(0), offset, step), cascade._from_cube(idx.max(0), offset, step)
    assert np.all(np.abs(lo - (10, 5, 3)) <= 2) and np.all(np.abs(hi - (389, 49, 24)) <= 2)


def test_windowed_primitive_sdf_classifies_like_the_exact_one():
    """_State.prim_sdf on a box grid evaluates only around the primitive; every threshold the loop uses (the residual
    band, the model's >= 0) must give the same voxels as the exact SDF, for adds and cuts."""
    from reco_engine.result_schema import Primitive
    spec = cascade.box_grid_spec((-3.0, 10.0, 4.0), (21.0, 16.0, 40.0), 0.25)
    c = spec.centres()
    occ = np.linalg.norm(c - (8.0, 13.0, 20.0), axis=-1) <= 4.0
    t_in = np.stack([occ.astype(np.float32), np.where(occ, 1.0, -1.0).astype(np.float32)] + [np.zeros(occ.shape, np.float32)] * 3)
    st = rl._State(t_in, spec)
    box = Primitive("box", (0.0, 0.0, 0.3826834, 0.9238795), (7.4, 13.1, 20.7), {"x_half": 3.0, "y_half": 1.2, "z_half": 5.0})
    cyl = Primitive("cylinder", (0, 0, 0, 1), (8.0, 13.0, 21.0), {"radius": 1.0, "half_length": 3.0})
    from reco_engine.result_to_program import primitive_to_sp
    exact = {id(p): -primitive_to_sp(p).sdf(c) for p in (box, cyl)}
    s_w = st.apply(st.s_model, st.prim_sdf(box), "add")
    s_e = st.apply(st.s_model, exact[id(box)], "add")
    assert (st.residual(s_w)[0] == st.residual(s_e)[0]).all() and ((s_w >= 0) == (s_e >= 0)).all()
    s_w2, s_e2 = st.apply(s_w, st.prim_sdf(cyl), "cut"), st.apply(s_e, exact[id(cyl)], "cut")
    for a, b in zip(st.residual(s_w2), st.residual(s_e2)):
        assert (a == b).all()
    assert ((s_w2 >= 0) == (s_e2 >= 0)).all() and (st.prim_sdf(cyl) >= 0).sum() == (exact[id(cyl)] >= 0).sum()


def test_window_is_the_component_box_plus_margin_and_maps_back():
    spec = cascade.box_grid_spec((-3.0, 10.0, 4.0), (21.0, 16.0, 40.0), 0.25)
    occ = np.zeros(spec.shape, bool)
    t_in = np.stack([occ.astype(np.float32)] * 5)
    st = rl._State(t_in, spec)
    comp = np.zeros(spec.shape, bool)
    comp[40:45, 3:6, 100:120] = True
    sub, sl = rl._window(st, comp, margin=2)
    assert sub.shape == (9, 7, 24) and np.allclose(sub.centres(), spec.centres()[sl])
    cube = cascade.grid_spec((0, 0, 0), (10, 10, 10), 16)
    st_c = rl._State(np.zeros((5, 16, 16, 16), np.float32), cube)
    assert rl._window(st_c, np.ones((16, 16, 16), bool)) == (cube, (slice(None),) * 3)       # cubes unchanged


def test_an_add_reaching_outside_the_box_grid_is_rejected():
    """Outside the (tight) box grid no voxel counts excess material, so an add that leaves it must be refused there."""
    from reco_engine.result_schema import Primitive
    spec = cascade.box_grid_spec((0.0, 0.0, 0.0), (10.0, 4.0, 6.0), 0.25)
    occ = np.zeros(spec.shape, bool)
    occ[4:36, 4:14, 4:20] = True
    t_in = np.stack([occ.astype(np.float32), np.where(occ, 1.0, -1.0).astype(np.float32)] + [np.zeros(occ.shape, np.float32)] * 3)
    st = rl._State(t_in, spec)
    inside = Primitive("box", (0, 0, 0, 1), (5.0, 2.0, 3.0), {"x_half": 4.0, "y_half": 1.2, "z_half": 2.0})
    long = Primitive("box", (0, 0, 0, 1), (5.0, 2.0, 3.0), {"x_half": 60.0, "y_half": 1.2, "z_half": 2.0})
    assert rl._evaluate(st, inside, "add", 0.9)["ok"]
    ev = rl._evaluate(st, long, "add", 0.9)
    assert not ev["ok"] and "outside the part's box" in ev["reason"]
