"""BoxOrienter: output conventions with untrained networks (shape, order, frame conversion), pipeline override hook."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reco_engine.box_orientation import BoxOrienter, REALIZATION_SCALE_MM  # noqa: E402
from training import rotation_symmetry as rs  # noqa: E402
from training.train_box_dims import BoxDimsCNN, half_extents_vox  # noqa: E402
from training.train_rotation_sym import SymRotationCNN  # noqa: E402


def box_volume(half=(14, 9, 5), n=64):
    idx = np.indices((n, n, n)).astype(float) - (n - 1) / 2
    return (np.abs(idx) <= np.array(half, float).reshape(3, 1, 1, 1)).all(axis=0).astype(np.uint8)


def test_dims_net_is_ordered_by_construction():
    net = BoxDimsCNN().eval()
    out = net(torch.randn(4, 1, 32, 32, 32))
    assert out.shape == (4, 3)
    assert (out[:, 0] >= out[:, 1]).all() and (out[:, 1] >= out[:, 2]).all()


def test_orienter_returns_the_macro_convention_for_untrained_nets():
    orienter = BoxOrienter(SymRotationCNN("6d").eval(), BoxDimsCNN().eval(), torch.device("cpu"), grid_mm=40.0)
    est = orienter.estimate(box_volume())
    p = est.parameters
    assert p["x_half_ratio"] >= p["y_half_ratio"] >= p["z_half_ratio"] > 0                # longest first
    assert np.isclose(np.linalg.norm(est.quaternion), 1.0)
    assert np.isclose(np.linalg.det(est.orientation), 1.0) and np.allclose(est.orientation @ est.orientation.T, np.eye(3), atol=1e-5)
    # ratio = half extent in voxels * voxel mm / 10 mm
    assert np.allclose([p["x_half_ratio"], p["y_half_ratio"], p["z_half_ratio"]],
                       est.half_extents_vox * (40.0 / 64) / REALIZATION_SCALE_MM)
    # the quaternion is the world form of the array-frame orientation
    assert np.allclose(rs.quat_to_matrix(est.quaternion), rs.A_WORLD_TO_ARRAY.T @ est.orientation, atol=1e-6)


def test_orienter_gives_none_without_material():
    orienter = BoxOrienter(SymRotationCNN("6d").eval(), BoxDimsCNN().eval(), torch.device("cpu"))
    assert orienter.estimate(np.zeros((64, 64, 64), np.uint8)) is None


def test_half_extent_labels_match_the_voxelised_volume():
    rec = SimpleNamespace(parameters={"x_half_ratio": 1.0, "y_half_ratio": 2.0, "z_half_ratio": 0.5},
                          placement={"scale_factor": 0.8})
    h = half_extents_vox(rec)
    assert np.allclose(h, np.array([2.0, 1.0, 0.5]) * 0.8 * 10.0 / 0.625)                  # longest first, in voxels


def test_orienter_without_a_dimension_net_measures_the_extents():
    orienter = BoxOrienter(SymRotationCNN("6d").eval(), None, torch.device("cpu"))
    est = orienter.estimate(box_volume((14, 9, 5)))
    # an untrained rotation net gives an arbitrary orientation, so only order and positivity are checked here
    assert est.half_extents_vox[0] >= est.half_extents_vox[1] >= est.half_extents_vox[2] > 0
    # the measurement itself is exact for a box that is already on the grid axes
    assert np.allclose(rs.measured_half_extents(box_volume((14, 9, 5))), [14.0 + 0.5, 9.0 + 0.5, 5.0 + 0.5], atol=0.51)


def test_refine_box_recovers_orientation_size_and_centre_from_a_perturbed_start_even_with_a_pocket():
    from reco_engine.box_orientation import _small_rotation, refine_box
    n = 64
    idx = np.indices((n, n, n)).reshape(3, -1).T.astype(float)
    M = rs.small_rotation(np.random.default_rng(3), 25.0)
    h, c = np.array([16.0, 11.5, 7.0]), np.array([31.5, 32.0, 30.5])
    vol = (np.abs((idx - c) @ M) <= h).all(axis=1).reshape(n, n, n)
    pocket = (((idx - (c + M[:, 0] * h[0])) ** 2).sum(axis=1) <= 6.0 ** 2).reshape(n, n, n)
    for target in (vol, vol & ~pocket):
        start = M @ _small_rotation(2, 3.0) @ _small_rotation(0, -2.0)
        M2, h2, c2, before, after = refine_box(target, start, h + [1.5, -1.0, 0.8], c + [1.5, -1.0, 0.5])
        assert after > before + 0.1 and after > 0.9
        assert np.degrees(np.arccos(np.clip((np.trace(M2.T @ M) - 1) / 2, -1, 1))) < 0.5
        assert np.allclose(h2, h, atol=0.5) and np.allclose(c2, c, atol=0.5)


def test_refine_box_leaves_a_perfect_estimate_alone_and_ignores_empty_targets():
    from reco_engine.box_orientation import refine_box
    n = 32
    idx = np.indices((n, n, n)).reshape(3, -1).T.astype(float)
    h, c = np.array([8.0, 6.0, 4.0]), np.array([15.5, 15.5, 15.5])
    vol = (np.abs((idx - c) @ np.eye(3)) <= h).all(axis=1).reshape(n, n, n)
    M2, h2, c2, before, after = refine_box(vol, np.eye(3), h, c)
    assert before == pytest.approx(1.0) and after == pytest.approx(1.0) and np.allclose(h2, h) and np.allclose(c2, c)
    assert refine_box(np.zeros((n, n, n)), np.eye(3), h, c)[4] == 0.0


def test_orienter_refines_by_default_and_keeps_the_longest_first_order():
    volume = box_volume((14, 9, 5))
    est = BoxOrienter(SymRotationCNN("6d").eval(), None, torch.device("cpu")).estimate(volume)   # untrained: an arbitrary start (a local refinement cannot rescue that, only tidy a near miss)
    assert est.refined and est.center_vox is not None
    assert est.half_extents_vox[0] >= est.half_extents_vox[1] >= est.half_extents_vox[2]
    assert np.isclose(np.linalg.det(est.orientation), 1.0)
    plain = BoxOrienter(SymRotationCNN("6d").eval(), None, torch.device("cpu"), refine=False).estimate(volume)
    assert not plain.refined and plain.center_vox is None
