"""Fitting a void to the grade (reco_engine.void_fit): family choice, protrusion, pockets, refusal."""
import numpy as np
import pytest

from reco_engine.void_fit import fit_void

N = 64


def ball(c, r):
    idx = np.indices((N, N, N)).astype(float)
    return ((idx[0] - c[0]) ** 2 + (idx[1] - c[1]) ** 2 + (idx[2] - c[2]) ** 2) <= r * r


def test_a_sphere_cut_flat_gets_a_box_that_protrudes_beyond_the_cap():
    sphere = ball((31.5, 31.5, 31.5), 18.0)
    idx = np.indices((N, N, N))
    reference = sphere & (idx[0] <= 38)                                  # the cap above plane 38 is missing in the input
    region = sphere & ~reference                                         # the over-coverage: a spherical cap
    fit = fit_void(reference.astype(np.uint8), sphere.astype(np.uint8), region.astype(np.uint8))
    assert fit is not None and fit.topology == "box"
    assert fit.scores["box"] < 0.1 * fit.scores["sphere"] + 1 and fit.mismatch_after < 0.1 * fit.mismatch_before
    # the void lies over the cut plane and reaches beyond the sphere's cap
    assert fit.center_vox[0] > 38 - 1 and fit.parameters["x_half_ratio"] > 0


def test_a_hemispherical_pocket_gets_a_sphere_at_the_pockets_centre():
    cube = np.zeros((N, N, N), bool)
    cube[10:54, 10:54, 10:54] = True
    pocket = ball((53.5, 32.0, 32.0), 12.0)
    reference = cube & ~pocket
    region = cube & pocket
    fit = fit_void(reference.astype(np.uint8), cube.astype(np.uint8), region.astype(np.uint8))
    assert fit.topology == "sphere" and fit.mismatch_after < 0.05 * fit.mismatch_before
    assert np.allclose(fit.center_vox, [53.5, 32.0, 32.0], atol=1.5)
    assert fit.parameters["radius_ratio"] == pytest.approx(12.0 * 0.625 / 10.0, rel=0.1)


def test_no_void_is_returned_when_the_assembly_already_matches_or_nothing_would_help():
    cube = np.zeros((N, N, N), np.uint8)
    cube[10:54, 10:54, 10:54] = 1
    assert fit_void(cube, cube, cube) is None                                            # nothing to remove
    smaller = cube.copy(); smaller[10:20] = 0
    assert fit_void(cube, smaller, np.zeros_like(cube)) is None                          # empty region
    assert fit_void(cube, np.zeros_like(cube), cube) is None                             # empty assembly


def test_families_can_be_restricted_and_a_voxel_wide_difference_is_forgiven():
    cube = np.zeros((N, N, N), np.uint8)
    cube[10:54, 10:54, 10:54] = 1
    reference = cube.copy(); reference[53] = 0                                            # one voxel layer: within the band
    assert fit_void(reference, cube, cube - reference) is None
    pocket = ball((53.5, 32.0, 32.0), 12.0)
    ref2 = cube.astype(bool) & ~pocket
    only_box = fit_void(ref2.astype(np.uint8), cube, (cube.astype(bool) & pocket).astype(np.uint8), families=("box",))
    assert only_box.topology == "box" and set(only_box.scores) == {"box"}


# ---------------------------------------------------------------- in the loop (CNN / FreeCAD / stltovoxel stubbed)

def test_two_separate_pockets_are_fitted_one_at_a_time():
    cube = np.zeros((N, N, N), bool)
    cube[10:54, 10:54, 10:54] = True
    p1, p2 = ball((53.5, 22.0, 32.0), 7.0), ball((53.5, 42.0, 32.0), 7.0)
    reference = cube & ~p1 & ~p2
    region = cube & (p1 | p2)
    fit = fit_void(reference.astype(np.uint8), cube.astype(np.uint8), region.astype(np.uint8), families=("sphere",))
    d1, d2 = np.linalg.norm(fit.center_vox - [53.5, 22.0, 32.0]), np.linalg.norm(fit.center_vox - [53.5, 42.0, 32.0])
    assert min(d1, d2) < 2.0 and fit.parameters["radius_ratio"] == pytest.approx(7.0 * 0.625 / 10.0, rel=0.15)   # one pocket, not both


# ---------------------------------------------------------------- cylinder voids

def cyl_mask(center, axis, radius, half_length):
    from reco_engine.cylinder_fit import cylinder_mask_points
    grid = np.indices((N, N, N)).reshape(3, -1).T.astype(float)
    return cylinder_mask_points(grid, np.asarray(center, float), np.asarray(axis, float) / np.linalg.norm(axis), radius, half_length).reshape(N, N, N)


def test_a_through_bore_gets_a_cylinder_that_protrudes_beyond_the_block():
    block = np.zeros((N, N, N), bool)
    block[10:54, 10:54, 10:54] = True
    bore = cyl_mask((31.5, 31.5, 31.5), (1, 0, 0), 7.0, 40.0)                      # runs along array axis 0, longer than the block
    reference = block & ~bore
    region = block & bore
    fit = fit_void(reference.astype(np.uint8), block.astype(np.uint8), region.astype(np.uint8))
    assert fit.topology == "cylinder" and fit.mismatch_after < 0.05 * fit.mismatch_before
    assert fit.parameters["radius_ratio"] == pytest.approx(7.0 * 0.625 / 10.0, rel=0.1)
    assert fit.parameters["half_length_ratio"] * 10.0 / 0.625 >= 21.5                 # at least as long as the block, may protrude
    assert np.allclose(fit.center_vox[1:], [31.5, 31.5], atol=1.0) and np.isclose(np.linalg.norm(fit.quaternion), 1.0)


def test_families_can_exclude_the_cylinder():
    block = np.zeros((N, N, N), bool)
    block[10:54, 10:54, 10:54] = True
    bore = cyl_mask((31.5, 31.5, 31.5), (1, 0, 0), 7.0, 40.0)
    fit = fit_void((block & ~bore).astype(np.uint8), block.astype(np.uint8), (block & bore).astype(np.uint8), families=("box", "sphere"))
    assert fit.topology in ("box", "sphere") and set(fit.scores) == {"box", "sphere"}
