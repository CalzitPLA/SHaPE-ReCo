"""Box orientation with symmetries: frames, canonical labels, symmetry-aware loss, grid rotations, PCA baseline."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import rotation_symmetry as rs  # noqa: E402


def random_rotation(rng):
    q = rng.normal(size=4)
    return rs.quat_to_matrix(q / np.linalg.norm(q))


def raster_box(M, half, n=64, centre=(31.5, 31.5, 31.5)):
    """Voxel array of a box whose axes (columns of M, in array coordinates) have the given half extents (voxels)."""
    idx = np.indices((n, n, n)).reshape(3, -1).T - np.asarray(centre)
    local = idx @ M                                                       # coordinates along the body axes
    inside = (np.abs(local) <= np.asarray(half) + 1e-9).all(axis=1)
    return inside.reshape(n, n, n).astype(np.uint8)


def test_group_is_the_24_proper_signed_permutations_and_closed():
    g = rs.GROUP
    assert g.shape == (24, 3, 3) and np.allclose(g[0], np.eye(3))
    assert np.allclose(np.linalg.det(g), 1.0)
    keys = {tuple(m.astype(int).ravel()) for m in g}
    assert len(keys) == 24
    for a in g[:6]:
        for b in g[6:12]:
            assert tuple((a @ b).astype(int).ravel()) in keys


def test_frame_matrix_is_a_proper_rotation():
    assert np.linalg.det(rs.A_WORLD_TO_ARRAY) == pytest.approx(1.0)


def test_admissible_group_sizes():
    assert rs.admissible_group([3.0, 2.0, 1.0]).sum() == 4                # generic box: the half-turns D2
    assert rs.admissible_group([2.0, 2.05, 1.0]).sum() == 8               # square prism (within tolerance)
    assert rs.admissible_group([2.0, 1.0, 1.05]).sum() == 8
    assert rs.admissible_group([2.0, 2.0, 2.0]).sum() == 24               # cube
    assert rs.admissible_group([3.0, 2.0, 1.0])[0]                        # the identity is always admissible


def test_equivalent_labels_give_the_same_canonical_orientation():
    """(R, dims) and (R g, dims permuted by g) are the same body: the canonical M must agree up to D2."""
    rng = np.random.default_rng(1)
    for _ in range(30):
        R = random_rotation(rng)
        dims = rng.uniform(0.5, 4.0, 3)
        m0, mask = rs.record_target(rs.matrix_to_quat(R), dims)
        for g in rs.GROUP:
            perm = [int(np.argmax(np.abs(g[:, c]))) for c in range(3)]      # body axis c of the new label = old axis perm[c]
            new_dims = np.empty(3)
            for c in range(3):
                new_dims[c] = dims[perm[c]]
            m1, _ = rs.record_target(rs.matrix_to_quat(R @ g), new_dims)
            err = rs.symmetric_error(m1[None], m0[None], mask[None])[0]
            assert err < 1e-4


def test_symmetric_error_zero_for_group_elements_and_large_for_others():
    rng = np.random.default_rng(2)
    M = random_rotation(rng)
    mask = rs.admissible_group([3.0, 2.0, 1.0])
    for k in np.flatnonzero(mask):
        assert rs.symmetric_error((M @ rs.GROUP[k])[None], M[None], mask[None])[0] < 1e-4
    quarter = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])                # 90 degrees about z: not a symmetry of a generic box
    assert rs.symmetric_error((M @ quarter)[None], M[None], mask[None])[0] == pytest.approx(90.0, abs=1e-3)


def test_torch_loss_matches_numpy_and_is_differentiable():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    pred = np.stack([random_rotation(rng) for _ in range(6)])
    tgt = np.stack([random_rotation(rng) for _ in range(6)])
    mask = np.stack([rs.admissible_group(sorted(rng.uniform(0.5, 4, 3), reverse=True)) for _ in range(6)])
    expect = np.radians(rs.symmetric_error(pred, tgt, mask)).mean()
    p = torch.tensor(pred, requires_grad=True)
    loss = rs.symmetric_loss_torch(p, torch.tensor(tgt), torch.tensor(mask))
    assert float(loss) == pytest.approx(expect, abs=1e-4)
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_sixd_and_quaternion_heads_give_rotations():
    torch = pytest.importorskip("torch")
    x = torch.randn(5, 6, dtype=torch.float64)
    R = rs.sixd_to_matrix_torch(x)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3, dtype=torch.float64).expand(5, 3, 3), atol=1e-9)
    assert torch.allclose(torch.linalg.det(R), torch.ones(5, dtype=torch.float64), atol=1e-9)
    q = torch.randn(5, 4, dtype=torch.float64)
    Rq = rs.quat_to_matrix_torch(q)
    assert np.allclose(Rq[0].numpy(), rs.quat_to_matrix(q[0].numpy()))


def test_rotate_volume_moves_a_marker_by_g_and_orientation_by_g():
    n = 16
    vol = np.zeros((n, n, n), np.uint8)
    vol[3, 5, 11] = 1
    centre = (n - 1) / 2
    for g in rs.GROUP:
        out = rs.rotate_volume(vol, g)
        got = np.array(np.argwhere(out)[0]) - centre
        want = g @ (np.array([3, 5, 11]) - centre)
        assert np.allclose(got, want)


def test_pca_recovers_the_orientation_of_a_rasterised_box_up_to_symmetry():
    rng = np.random.default_rng(4)
    for _ in range(5):
        R = random_rotation(rng)
        dims = np.array([20.0, 12.0, 6.0])
        M = rs.canonical_orientation(np.linalg.inv(rs.A_WORLD_TO_ARRAY) @ R, dims)        # an arbitrary array-frame pose
        vol = raster_box(M, dims)
        E = rs.pca_orientation(vol)
        mask = rs.admissible_group(dims)
        assert rs.symmetric_error(E[None], M[None], mask[None])[0] < 3.0
        # grid rotations act on the estimate as on the label
        g = rs.GROUP[int(rng.integers(24))]
        E2 = rs.pca_orientation(rs.rotate_volume(vol, g))
        assert rs.symmetric_error(E2[None], (g @ M)[None], mask[None])[0] < 3.0


def test_align_volume_puts_the_principal_axes_on_the_grid_and_returns_the_orientation():
    rng = np.random.default_rng(6)
    dims = np.array([22.0, 13.0, 7.0])
    for _ in range(4):
        R = random_rotation(rng)
        M = rs.canonical_orientation(np.linalg.inv(rs.A_WORLD_TO_ARRAY) @ R, dims)
        vol = raster_box(M, dims, centre=(28.0, 33.0, 30.5))                 # off-centre on purpose
        aligned, E = rs.align_volume(vol)
        mask = rs.admissible_group(dims)
        assert rs.symmetric_error(E[None], M[None], mask[None])[0] < 3.0
        # the aligned body is (nearly) axis-aligned, centred and about the same volume
        E2 = rs.pca_orientation(aligned)
        assert rs.symmetric_error(E2[None], np.eye(3)[None], mask[None])[0] < 3.0
        centre = np.argwhere(aligned).mean(axis=0)
        assert np.allclose(centre, 31.5, atol=1.0)
        assert abs(aligned.sum() / vol.sum() - 1) < 0.08


def test_resample_rotated_turns_an_orientation_by_the_transpose():
    rng = np.random.default_rng(7)
    dims = np.array([18.0, 11.0, 6.0])
    M = random_rotation(rng)
    vol = raster_box(M, dims)
    Rj = rs.small_rotation(rng, 25.0)
    out = rs.resample_rotated(vol, Rj.T)                                     # output[o] = input[Rj^T o]: rotates the body by Rj
    mask = rs.admissible_group(dims)
    E = rs.pca_orientation(out)
    assert rs.symmetric_error(E[None], (Rj @ M)[None], mask[None])[0] < 3.0


def test_array_orientation_round_trips_to_the_world_quaternion():
    rng = np.random.default_rng(8)
    for _ in range(10):
        R = random_rotation(rng)
        dims = np.array([3.0, 2.0, 1.0])                                     # already longest-first
        M = rs.canonical_orientation(R, dims)
        R_back = rs.quat_to_matrix(rs.array_orientation_to_world_quaternion(M))
        assert rs.geodesic_angle(R_back, R) < 1e-6 or rs.geodesic_angle(R_back, R @ np.diag([1.0, 1, 1])) < 1e-6
