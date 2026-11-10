"""dataset_gen/src/field_channels.dirlog_input: the exact-field "dirlog" input (unit direction, log distance, curvature),
checked on analytic, off-centre, rotated solids against the grid-derived "dirlog" (same array frame) and against 1/r."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import sdf_primitives as sp  # noqa: E402
from src.field_channels import dirlog_input, inscale_spec, rotate_network_input, world_to_bin  # noqa: E402
from training import field_input as fi  # noqa: E402


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def analytic_channels(prim, spec):
    """What mesh_channels stores (world axes, sdf POSITIVE inside, V in mm), from an analytic solid."""
    c = spec.centres().reshape(-1, 3)
    f = prim.sdf(c)
    eps = 1e-5
    g = np.stack([(prim.sdf(c + e) - prim.sdf(c - e)) / (2 * eps) for e in np.eye(3) * eps], axis=1)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    V = -f[:, None] * g
    r = spec.resolution
    out = np.empty((5, r, r, r), np.float32)
    out[0] = (f <= 0).reshape(r, r, r)
    out[1] = -f.reshape(r, r, r)
    out[2:] = V.reshape(r, r, r, 3).transpose(3, 0, 1, 2)
    return out


CYL = sp.Placed(sp.Cylinder(5.0, 16.0), _rot((0.3, 1, 0.2), 38), np.array([-1.1, 0.9, -7.2]))
SPHERE = sp.Placed(sp.Sphere(8.0), np.eye(3), np.array([3.1, -2.2, 1.4]))


def test_exact_dirlog_matches_the_grid_derived_one_in_the_array_frame():
    spec = inscale_spec(40.0, 64)
    ch = analytic_channels(CYL, spec)
    x = dirlog_input(ch, spec)
    assert x.shape == (5, 64, 64, 64)
    occ_bin = world_to_bin(ch[0] > 0.5)
    np.testing.assert_array_equal(x[0], occ_bin.astype(np.float32))
    grid = fi.encode(occ_bin.astype(np.float32) * 2 - 1, "dirlog")
    band = np.abs(sp.sdf_grid(CYL, 64, 40.0)) < 2.0                           # sdf_grid is in the array frame too
    cos = (x[1:4] * grid[1:4]).sum(0)[band]
    assert np.median(cos) > 0.99                                                # same normals, same frame
    assert np.abs(x[4] - grid[4])[band].max() < 0.1                            # same log distance up to the voxel sdf


def test_curvature_channels_recover_one_over_r():
    spec = inscale_spec(40.0, 64)
    x = dirlog_input(analytic_channels(SPHERE, spec), spec, curvature=True)
    assert x.shape == (8, 64, 64, 64)
    d = np.abs(sp.sdf_grid(SPHERE, 64, 40.0))
    band = (d > 0.5) & (d < 2.0)
    kappa = np.tan(x[5:7] * np.pi / 2) / 2.0                                    # CURVATURE_SCALE_MM = 2
    np.testing.assert_allclose(np.median(kappa[:, band], axis=1), 1 / 8.0, rtol=0.05)
    assert np.median(x[7][band]) < 0.1                                           # no edge on a sphere
    xc = dirlog_input(analytic_channels(CYL, spec), spec, curvature=True)
    dc = np.abs(sp.sdf_grid(CYL, 64, 40.0))
    kc = np.tan(xc[5:7] * np.pi / 2) / 2.0
    bandc = (dc > 0.5) & (dc < 2.0) & (xc[7] < 0.3)
    top = np.sort(np.abs(kc[:, bandc]), axis=0)[1]
    assert np.mean(np.abs(top - 0.2) < 0.03) > 0.5                              # the mantle: kappa = 1/5
    assert (xc[7][(dc > 0.5) & (dc < 2.0)] > 0.7).mean() > 0.02                  # the two rims show up as edges


def test_rotation_turns_the_unit_normal_channels():
    from training.rotation_symmetry import GROUP, rotate_volume
    spec = inscale_spec(40.0, 32)
    x = dirlog_input(analytic_channels(CYL, spec), spec, curvature=True)
    g = GROUP[7]
    y = rotate_network_input(x, g, fi.VECTOR_CHANNELS["dirlog_curv"])
    np.testing.assert_array_equal(y[0], rotate_volume(x[0], g))
    np.testing.assert_array_equal(y[5], rotate_volume(x[5], g))                # curvature is a scalar: only moved
    turned = np.einsum("ij,j...->i...", g, np.stack([rotate_volume(c, g) for c in x[1:4]]))
    np.testing.assert_allclose(y[1:4], turned, atol=1e-6)


def test_dirlog_residual_marks_the_cut_and_add_regions_and_uses_the_models_normal():
    from src.field_channels import dirlog_residual, grid_spec, world_to_bin
    spec = grid_spec(-16, 16, 48)
    c = spec.centres().reshape(-1, 3)
    r = spec.resolution
    box = sp.Placed(sp.Box((9.0, 6.0, 5.0)), _rot((1, 0.4, 0.2), 17), np.array([1.2, -0.7, 0.9]))      # off-centre, turned
    bore = sp.Placed(sp.Cylinder(2.5, 30.0), _rot((1, 0.4, 0.2), 17), np.array([2.0, -0.5, -14.0]))
    dome = sp.Sphere(3.0)
    part = (box - bore) | sp.Placed(dome, np.eye(3), np.array([1.5, -0.5, 6.5]))                     # the input: box - bore + dome
    occ_in = (part.sdf(c) <= 0).reshape(r, r, r)
    s_model = -box.sdf(c).reshape(r, r, r)                                                          # the model so far: the box alone
    x = dirlog_residual(s_model, occ_in, spec)
    assert x.shape == (7, r, r, r)
    in_bore = world_to_bin(((box.sdf(c) < 0) & (bore.sdf(c) < 0)).reshape(r, r, r))
    in_dome = world_to_bin(((part.sdf(c) <= 0) & (box.sdf(c) > 0)).reshape(r, r, r))
    np.testing.assert_array_equal(x[6] == -1, in_bore)                                             # excess material: a cut
    np.testing.assert_array_equal(x[6] == 1, in_dome)                                              # missing material: an add
    np.testing.assert_array_equal(x[6], x[0] - x[1])
    grid = fi.encode(world_to_bin(s_model > 0).astype(np.float32) * 2 - 1, "dirlog")               # the model's own dirlog
    near = np.abs(world_to_bin(s_model)) < 2 * spec.pitch
    assert np.median((x[2:5] * grid[1:4]).sum(0)[near]) > 0.98                                      # same normal, same frame
    empty = dirlog_residual(None, occ_in, spec)
    assert (empty[1] == 0).all() and (empty[5] == 1).all() and (empty[6] == empty[0]).all()


def test_dirlog_residual_rotates_with_its_normal_channels():
    from src.field_channels import dirlog_residual, grid_spec
    from training.rotation_symmetry import GROUP, rotate_volume
    spec = grid_spec(-16, 16, 32)
    c = spec.centres().reshape(-1, 3)
    box = sp.Placed(sp.Box((8.0, 5.0, 4.0)), _rot((0.3, 1, 0.2), 25), np.array([1.0, 0.5, -0.8]))
    x = dirlog_residual(-box.sdf(c).reshape(32, 32, 32), (sp.Sphere(7.0).sdf(c) <= 0).reshape(32, 32, 32), spec)
    g = GROUP[5]
    y = rotate_network_input(x, g, fi.VECTOR_CHANNELS["dirlog_residual"])
    np.testing.assert_array_equal(y[6], rotate_volume(x[6], g))
    np.testing.assert_allclose(y[2:5], np.einsum("ij,j...->i...", g, np.stack([rotate_volume(v, g) for v in x[2:5]])), atol=1e-6)
