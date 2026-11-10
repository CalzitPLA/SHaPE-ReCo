"""Analytic sphere fit (reco_engine.sphere_fit) and its hook in the proposal step."""

import numpy as np
import pytest

from reco_engine.sphere_fit import SphereFitter, refine_sphere

N = 64


def ball(c, r, cut_above=None):
    idx = np.indices((N, N, N)).astype(float)
    m = ((idx[0] - c[0]) ** 2 + (idx[1] - c[1]) ** 2 + (idx[2] - c[2]) ** 2) <= r * r
    if cut_above is not None:
        m &= idx[0] <= cut_above
    return m.astype(np.uint8)


def test_refine_recovers_centre_and_radius_from_a_perturbed_start():
    c, r = np.array([30.3, 33.1, 31.6]), 12.4
    center, radius, before, after = refine_sphere(ball(c, r), c + [1.6, -1.2, 0.9], r + 2.0)
    assert after > before + 0.1 and after > 0.97
    assert np.allclose(center, c, atol=0.6) and abs(radius - r) < 0.6


def test_estimator_starts_from_the_bounding_box_and_reports_the_macro_radius_ratio():
    est = SphereFitter(grid_mm=40.0).estimate(ball((31.5, 31.5, 31.5), 14.0))
    assert est.quaternion == [0.0, 0.0, 0.0, 1.0] and est.iou_after > 0.97
    assert est.parameters["radius_ratio"] == pytest.approx(est.radius_vox * (40.0 / 64) / 10.0)
    assert abs(est.radius_vox - 14.0) < 0.6
    assert SphereFitter().estimate(np.zeros((N, N, N), np.uint8)) is None


def test_a_sphere_cut_flat_is_fitted_as_well_as_a_sphere_can_be():
    target = ball((31.5, 31.5, 31.5), 16.0, cut_above=36)               # the top cap removed
    fitted = SphereFitter().estimate(target)
    assert fitted.iou_after > fitted.iou_before + 0.05 and fitted.iou_after > 0.7        # better than the bounding-box start
    assert SphereFitter(refine=False).estimate(target).iou_after == 0.0                  # unrefined: no measurement, start only

