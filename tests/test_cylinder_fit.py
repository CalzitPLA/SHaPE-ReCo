"""Analytic cylinder fit (reco_engine.cylinder_fit): axis, size and centre; macro quaternion; refusal."""
import numpy as np
import pytest

from reco_engine.cylinder_fit import CylinderFitter, axis_quaternion, cylinder_mask_points, refine_cylinder, _frame_with_axis
from training import rotation_symmetry as rs

N = 64
GRID = np.indices((N, N, N)).reshape(3, -1).T.astype(float)


def cyl(center, axis, radius, half_length):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return cylinder_mask_points(GRID, np.asarray(center, float), axis, radius, half_length).reshape(N, N, N).astype(np.uint8)


def test_axis_quaternion_turns_local_z_onto_the_world_axis():
    rng = np.random.default_rng(0)
    for _ in range(8):
        a = rng.normal(size=3)
        q = axis_quaternion(a)
        R = rs.quat_to_matrix(q)
        assert np.isclose(np.linalg.norm(q), 1.0)
        want = rs.A_WORLD_TO_ARRAY.T @ (a / np.linalg.norm(a))
        assert np.allclose(R @ [0, 0, 1], want, atol=1e-6)
    assert axis_quaternion(rs.A_WORLD_TO_ARRAY @ [0, 0, 1]) == [0.0, 0.0, 0.0, 1.0]
    R = rs.quat_to_matrix(axis_quaternion(rs.A_WORLD_TO_ARRAY @ [0, 0, -1]))
    assert np.allclose(R @ [0, 0, 1], [0, 0, -1], atol=1e-6)


def test_refine_recovers_a_tilted_cylinder_from_a_perturbed_start():
    c, u, r, L = np.array([31.3, 32.4, 30.6]), np.array([0.3, 0.9, 0.3]), 9.0, 15.0
    target = cyl(c, u, r, L)
    F = _frame_with_axis(u / np.linalg.norm(u)) @ rs.small_rotation(np.random.default_rng(1), 6.0)
    F2, c2, r2, L2, before, after = refine_cylinder(target, F, c + [1.4, -1.1, 0.8], r + 1.5, L - 2.0)
    assert after > before + 0.1 and after > 0.95
    assert abs(F2[:, 0] @ (u / np.linalg.norm(u))) > 0.998                        # axis within ~3 degrees (up to sign)
    assert abs(r2 - r) < 0.8 and abs(L2 - L) < 1.0 and np.linalg.norm(c2 - c) < 1.2


@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0.2, 0.5, 0.8)])
def test_estimator_finds_axis_radius_length_from_the_bounding_extents(axis):
    c, r, L = (31.5, 31.5, 31.5), 8.0, 14.0
    est = CylinderFitter(grid_mm=40.0).estimate(cyl(c, axis, r, L))
    assert est.iou_after > 0.93
    assert abs(est.axis @ (np.asarray(axis) / np.linalg.norm(axis))) > 0.995
    assert est.parameters["radius_ratio"] == pytest.approx(est.radius_vox * 0.625 / 10.0)
    assert est.parameters["half_length_ratio"] == pytest.approx(est.half_length_vox * 0.625 / 10.0)
    assert abs(est.radius_vox - r) < 1.0 and abs(est.half_length_vox - L) < 1.2
    assert np.isclose(np.linalg.norm(est.quaternion), 1.0)


def test_estimator_handles_a_cut_cylinder_and_refuses_empty_input():
    target = cyl((31.5, 31.5, 31.5), (0, 0, 1), 10.0, 16.0)
    target[:, :, 40:] = 0                                                           # one end cut off
    est = CylinderFitter().estimate(target)
    assert est.iou_after > 0.9 and est.iou_after > est.iou_before - 1e-9
    assert CylinderFitter().estimate(np.zeros((N, N, N), np.uint8)) is None
    assert CylinderFitter(refine=False).estimate(target).iou_after == 0.0

