"""More boolean operations in the add-cut loop: "intersect" (model & P), half-space cuts and two-primitive composites
(box & cylinder, box - cylinder) - schema, SDF, loop state, candidate generators, exact GUI meshes, the CSG tree."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine import cascade, residual_loop as rl  # noqa: E402
from reco_engine.result_schema import Operation, Primitive, ReconstructionResult  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp, program_model  # noqa: E402

OFF = np.array([7.0, -3.0, 11.0])                     # everything off-centre: no symmetry hides a sign or an axis


def box(c, h, q=(0, 0, 0, 1)):
    return Primitive("box", tuple(q), tuple(np.asarray(c, float)), dict(x_half=h[0], y_half=h[1], z_half=h[2]))


def cyl(c, r, hl, q=(0, 0, 0, 1)):
    return Primitive("cylinder", tuple(q), tuple(np.asarray(c, float)), dict(radius=r, half_length=hl))


def halfspace(point, normal):
    from reco_engine.result_to_program import quaternion_of
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u)
    return Primitive("halfspace", quaternion_of(np.stack([u, np.cross(n, u), n], axis=1)), tuple(np.asarray(point, float)), {})


def test_schema_round_trip_and_sdf():
    b = box(OFF, (2, 1, 1))
    c = cyl(OFF + (2, 0, 0), 1.0, 1.0)
    comp = Primitive("composite", parts=[b, c], combine="common", label="box & cylinder")
    r = ReconstructionResult("x", [Operation("add", b), Operation("intersect", halfspace(OFF + (0, 0, 0.5), (0, 0, 1))),
                                   Operation("cut", comp)])
    back = ReconstructionResult.from_dict(r.to_dict())
    assert [o.op for o in back.operations] == ["add", "intersect", "cut"]
    assert back.operations[2].primitive.combine == "common" and len(back.operations[2].primitive.parts) == 2
    m = program_model(back)
    pts = OFF + np.array([[0, 0, 0], [0, 0, 0.8], [1.8, 0, 0.2], [-1.8, 0, 0]])
    assert np.allclose(m.sdf(pts), [-0.5, 0.3, 0.2, -0.2])        # trimmed above z=0.5, the rounded end removed


def grid_state(occ_fn, pitch=0.25, lo=OFF - 4, hi=OFF + 4):
    spec = cascade.box_grid_spec(lo, hi, pitch)
    c = spec.centres()
    occ = occ_fn(c - OFF)
    t_in = np.stack([occ.astype(np.float32), np.where(occ, 1.0, -1.0).astype(np.float32)] +
                    [np.zeros(occ.shape, np.float32)] * 3)
    return rl._State(t_in, spec), spec


def test_intersect_trims_excess_and_refuses_to_trim_input():
    # input: the lower half (z <= 0) of a 4 x 2 x 2 box; model: the whole box -> excess above z = 0
    st, spec = grid_state(lambda r: (np.abs(r[..., 0]) <= 2) & (np.abs(r[..., 1]) <= 1) & (r[..., 2] <= 0) & (r[..., 2] >= -1))
    whole = box(OFF, (2, 1, 1))
    st.s_model = st.apply(st.s_model, st.prim_sdf(whole), "add")
    assert st.residual()[1].sum() > 0                                          # excess to trim
    # a half-space keeps the side against its normal: normal -z keeps z >= 0 - the excess, trimming away the input
    wrong = rl._evaluate(st, halfspace(OFF, (0, 0, -1)), "intersect", 0.9)
    assert not wrong["ok"] and "check" in wrong["reason"]
    right = rl._evaluate(st, halfspace(OFF, (0, 0, 1)), "intersect", 0.9)     # keeps z <= 0: exactly the input
    assert right["ok"] and right["gain"] == int(st.residual()[1].sum())
    before = int((st.s_model >= 0).sum())
    st.commit(right["s_new"], right["s_prim"], "intersect")
    assert st.residual()[1].sum() == 0 and int((st.s_model >= 0).sum()) < before
    assert st.cut_history.max() == 1                                          # the trimmed voxels are in the history


def test_strict_planes_find_the_faces_of_an_off_centre_box():
    trimesh = pytest.importorskip("trimesh")
    from src.closest_point_field import SurfaceField
    mesh = trimesh.creation.box(extents=(6.0, 3.0, 2.0))
    mesh.apply_translation(OFF)
    field = SurfaceField(np.asarray(mesh.vertices, float), np.asarray(mesh.faces))
    spec = cascade.box_grid_spec(OFF - 4, OFF + 4, 0.25)
    c = spec.centres() - OFF
    occ = (np.abs(c[..., 0]) <= 3) & (np.abs(c[..., 1]) <= 1.5) & (np.abs(c[..., 2]) <= 1)
    ch = rl._voxel_channels(occ, spec)
    planes = cascade.strict_planes(ch, spec, field, top=6)
    assert len(planes) >= 3
    for p in planes:
        hs = primitive_to_sp(p)
        # every plane lies on a face (0 at its centre) and its outward normal points away from the box's material
        assert abs(hs.sdf(np.asarray(p.center_mm)[None])[0]) < 1e-6
        assert hs.sdf(OFF[None])[0] < 0                                       # the box centre is on the kept side


def test_composites_pair_parallel_axes_only():
    b = box(OFF, (4, 1, 1))
    along_x = cyl(OFF + (3, 0, 0), 1.0, 0.5, q=(0, 0.7071068, 0, 0.7071068))    # local z -> world x
    along_z = cyl(OFF, 1.0, 0.5)
    tilted = cyl(OFF, 1.0, 0.5, q=(0.2588190, 0, 0, 0.9659258))                # 30 deg off z
    comps = cascade.composites([b], [along_x, along_z, tilted])
    assert len(comps) == 4 and {c.combine for c in comps} == {"common", "cut"}
    common = next(c for c in comps if c.combine == "common" and abs(c.parts[1].center_mm[1] - OFF[1]) < 1e-9
                  and primitive_to_sp(c.parts[1]).sdf((OFF + (3.9, 0, 0))[None])[0] < 0)
    # the cylinder along x was stretched over the whole box: the composite reaches the box's end at x = +4
    assert primitive_to_sp(common).sdf((OFF + (3.9, 0, 0))[None])[0] < 0


def test_exact_meshes_of_the_new_operations():
    pytest.importorskip("manifold3d")
    from reco_gui.cascade_session import CascadeFolder
    fold = CascadeFolder()
    fold._crop = (OFF - 10, OFF + 10)
    b = box(OFF, (2, 1, 1))
    ops = (Operation("add", b), Operation("intersect", halfspace(OFF, (0, 0, 1))))
    m = fold.model_mesh(ops, False)
    tri = m.vertices[m.triangles]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    assert abs(vol - 4 * 2 * 1) < 1e-3                                        # the box's lower half
    comp = Primitive("composite", parts=[box(OFF, (2, 1, 1)), cyl(OFF + (2, 0, 0), 1.0, 1.0)], combine="cut")
    mc = fold.primitive_mesh(comp)
    tri = mc.vertices[mc.triangles]
    vol_c = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    assert abs(vol_c - (16 - np.pi * 1 * 2 / 2)) < 0.02                       # box minus the half cylinder inside it


def test_csg_tree_shows_intersect_as_common():
    from reco_gui.op_tree import csg_tree
    ops = (Operation("add", box(OFF, (1, 1, 1))), Operation("intersect", halfspace(OFF, (0, 0, 1))),
           Operation("cut", cyl(OFF, 0.5, 2.0)))
    tree = csg_tree(ops)
    assert tree[0] == "cut" and tree[1][0][0] == "common" and [n[1] for n in tree[1][0][1]] == [0, 1]


def test_a_new_boolean_must_explain_the_surface():
    """The gate that keeps a bounding-cylinder intersect with 0 % of the surface on it from being accepted just because
    nothing else passed (bs_box_step, bs_box_dome, 2026-09-25)."""
    cfg = rl.ResidualConfig()
    assert 0.0 < cfg.boolean_min_surface <= 0.5 and cfg.more_booleans


def test_an_axis_aligned_add_box_is_clipped_to_the_part():
    st, spec = grid_state(lambda r: (np.abs(r[..., 0]) <= 2) & (np.abs(r[..., 1]) <= 1) & (np.abs(r[..., 2]) <= 1))
    long = box(OFF, (60.0, 1.0, 1.0), q=(0, 0, 0.7071068, 0.7071068))    # 90 deg about z: runs along y, still
    clipped = rl._clip_to_part(st, long)                                                   # axis-aligned
    assert clipped is not long and "clipped" in clipped.label
    lo_c, hi_c = (np.asarray(b) for b in primitive_to_sp(clipped).bounds())
    assert np.all(lo_c >= np.asarray(spec.lo) - 1e-9) and np.all(hi_c <= np.asarray(spec.hi) + 1e-9)
    assert rl._evaluate(st, clipped, "add", 0.9)["reason"] != "reaches outside the part's box"
    tilted = box(OFF, (60.0, 1.0, 1.0), q=(0, 0, 0.1305262, 0.9914449))                   # 15 deg: left alone
    assert rl._clip_to_part(st, tilted) is tilted


def test_rotated_views_of_a_piece_map_back_to_the_input_frame():
    """vary_on_failure: a box-shaped piece seen through each rotated view, fitted there and mapped back, lands where it
    is (centre within a voxel, the same extents) - the rotation and its inverse agree."""
    from reco_engine.result_to_program import transform_primitive
    trimesh = pytest.importorskip("trimesh")
    from src.closest_point_field import SurfaceField
    mesh = trimesh.creation.box(extents=(6.0, 3.0, 2.0))
    mesh.apply_translation(OFF)
    field = SurfaceField(np.asarray(mesh.vertices, float), np.asarray(mesh.faces))
    st, spec = grid_state(lambda r: (np.abs(r[..., 0]) <= 3) & (np.abs(r[..., 1]) <= 1.5) & (np.abs(r[..., 2]) <= 1))
    comp = st.occ_in.copy()
    sub, sl = rl._window(st, comp)
    views = rl._component_views(st, comp[sl], sub, field, rl.ResidualConfig(vary_rotations=2))
    names = [v[0] for v in views]
    assert "rotation 1" in names and "pitch x0.5" in names
    for name, ch, vspec, vfield, back in views:
        if back is None:
            continue
        ctx = cascade._Context(ch, vspec, vfield, None, None)
        p = cascade._fit_frame_box(ctx, "frame")
        assert p is not None, name
        q = transform_primitive(p, *back)
        assert np.linalg.norm(np.asarray(q.center_mm) - OFF) < 2 * spec.pitch, name
        assert sorted(round(v * 2) for v in q.dims_mm.values()) == [2, 3, 6], (name, q.dims_mm)
