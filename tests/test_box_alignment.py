"""The face-normal frame fit (training/box_alignment.py) on synthetic boxes of known orientation."""

import sys
from pathlib import Path

import numpy as np
import pytest

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))

from training import box_alignment as ba  # noqa: E402
from training import rotation_symmetry as rs  # noqa: E402

N = 64


def random_rotation(rng):
    return next(ba._random_rotations(rng, 1))


def box_volume(R, half=(20.0, 13.0, 7.0), centre=(31.5, 31.5, 31.5)):
    """Occupancy of a box whose axes (columns of R) are given in the array frame."""
    g = np.stack(np.meshgrid(*[np.arange(N)] * 3, indexing="ij"), axis=-1).astype(float) - np.asarray(centre)
    local = g @ R                                                        # coordinates along the box axes
    return (np.abs(local) <= np.asarray(half)).all(axis=-1).astype(np.uint8), g


def error_mod_cube(E, R):
    """Degrees, modulo all 24 grid rotations (the axis ORDER is not part of the question)."""
    return float(rs.symmetric_error(E[None], R[None], np.ones((1, 24), bool))[0])


@pytest.mark.parametrize("seed", range(6))
def test_plain_box_is_found_within_two_degrees(seed):
    R = random_rotation(np.random.default_rng(seed))
    vol, _ = box_volume(R)
    E = ba.frame_orientation(vol)
    assert error_mod_cube(E, R) < 2.0
    assert np.isclose(np.linalg.det(E), 1.0)
    extents = [np.ptp((np.argwhere(vol > 0) - 31.5) @ E[:, k]) for k in range(3)]
    assert extents[0] >= extents[1] >= extents[2]                        # longest first, like pca_orientation


@pytest.mark.parametrize("seed", range(6))
def test_a_planar_cut_that_fools_pca_does_not_fool_the_frame_fit(seed):
    rng = np.random.default_rng(100 + seed)
    R = random_rotation(rng)
    vol, g = box_volume(R)
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)
    proj = g @ normal
    occupied = vol > 0
    cutoff = np.percentile(proj[occupied], 55)                            # remove the 45 % beyond an oblique plane
    vol = (occupied & (proj <= cutoff)).astype(np.uint8)
    assert error_mod_cube(ba.frame_orientation(vol), R) < 4.0
    pca = rs.pca_orientation(vol)
    assert error_mod_cube(pca, R) > error_mod_cube(ba.frame_orientation(vol), R)      # and it is the better estimate


def test_a_spherical_bite_and_speckle_noise_are_ignored():
    rng = np.random.default_rng(7)
    R = random_rotation(rng)
    vol, g = box_volume(R)
    bite = np.linalg.norm(g - R[:, 0] * 20.0 - np.array([0, 0, 2.0]), axis=-1) < 9.0
    vol = (vol > 0) & ~bite
    flips = rng.random(vol.shape) < 0.02                                  # salt and pepper
    vol = (vol ^ flips).astype(np.uint8)
    assert error_mod_cube(ba.frame_orientation(vol), R) < 4.0


def test_deterministic_and_none_for_almost_nothing():
    R = random_rotation(np.random.default_rng(3))
    vol, _ = box_volume(R)
    assert np.allclose(ba.frame_orientation(vol, seed=5), ba.frame_orientation(vol, seed=5))
    tiny = np.zeros((N, N, N), np.uint8)
    tiny[30:32, 30:32, 30:32] = 1
    assert ba.frame_orientation(tiny) is None
    assert ba.frame_orientation(np.zeros((N, N, N), np.uint8)) is None


def test_align_volume_methods_agree_on_the_convention():
    """Both methods return (aligned volume, E) with E orthonormal, det +1, and the aligned body's axes on the grid."""
    R = random_rotation(np.random.default_rng(11))
    vol, _ = box_volume(R)
    for method in ("pca", "frame"):
        aligned, E = rs.align_volume(vol, method)
        assert np.allclose(E.T @ E, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(E), 1.0)
        residual = rs.pca_orientation(aligned)                            # already aligned: what is left is a symmetry element
        assert error_mod_cube(residual, np.eye(3)) < 3.0
    with pytest.raises(ValueError):
        rs.align_volume(vol, "nonsense")
