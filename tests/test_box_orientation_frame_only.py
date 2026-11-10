"""The network-free BoxOrienter (frame fit + measured extents + refinement): accuracy on synthetic boxes, the box
conventions, and that it needs neither checkpoints nor torch."""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from reco_engine.box_orientation import BoxOrienter, REALIZATION_SCALE_MM  # noqa: E402
from test_box_alignment import box_volume, error_mod_cube, random_rotation  # noqa: E402
from training import rotation_symmetry as rs  # noqa: E402

HALF = (20.0, 13.0, 7.0)


def test_frame_only_finds_orientation_and_extents_of_a_rotated_box():
    for seed in range(4):
        R = random_rotation(np.random.default_rng(seed))
        vol, _ = box_volume(R, HALF)
        est = BoxOrienter.frame_only(grid_mm=40.0).estimate(vol)
        assert est is not None and est.refined
        # distinct dimensions: the estimate equals R up to the 4 half-turns of the box (not just the 24 grid rotations)
        admissible = rs.admissible_group(sorted(HALF, reverse=True))[None]
        assert rs.symmetric_error(est.orientation[None], R[None], admissible)[0] < 3.0
        assert np.allclose(est.half_extents_vox, HALF, atol=1.6)
        p = est.parameters
        assert p["x_half_ratio"] >= p["y_half_ratio"] >= p["z_half_ratio"] > 0
        assert np.allclose([p["x_half_ratio"], p["y_half_ratio"], p["z_half_ratio"]],
                           est.half_extents_vox * (40.0 / 64) / REALIZATION_SCALE_MM)
        assert np.isclose(np.linalg.det(est.orientation), 1.0) and np.isclose(np.linalg.norm(est.quaternion), 1.0)


def test_frame_only_survives_a_planar_cut_where_pca_does_not():
    rng = np.random.default_rng(21)
    R = random_rotation(rng)
    vol, g = box_volume(R, HALF)
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)
    proj = g @ normal
    occupied = vol > 0
    cut = (occupied & (proj <= np.percentile(proj[occupied], 55))).astype(np.uint8)
    est = BoxOrienter.frame_only(refine=False).estimate(cut)
    assert error_mod_cube(est.orientation, R) < 4.0
    assert error_mod_cube(BoxOrienter.frame_only(align_method="pca", refine=False).estimate(cut).orientation, R) > 4.0


def test_frame_only_returns_none_without_material():
    assert BoxOrienter.frame_only().estimate(np.zeros((64, 64, 64), np.uint8)) is None


def test_frame_only_runs_without_torch():
    code = ("import sys, numpy as np; sys.path.insert(0, %r); sys.path.insert(0, %r);"
            "from reco_engine.box_orientation import BoxOrienter;"
            "from test_box_alignment import box_volume, random_rotation;"
            "vol, _ = box_volume(random_rotation(np.random.default_rng(0)), (20.0, 13.0, 7.0));"
            "est = BoxOrienter.frame_only().estimate(vol);"
            "assert est is not None and 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)[:3]")
    out = subprocess.run([sys.executable, "-c", code % (str(ROOT), str(ROOT / "tests"))], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-400:]


def test_align_volume_frame_falls_back_to_pca_for_a_tiny_body():
    from training.box_alignment import frame_orientation
    vol = np.zeros((64, 64, 64), np.uint8)
    vol[30:34, 30:33, 20:23] = 1                                   # 36 voxels: too few for surface normals, enough for moments
    assert frame_orientation(vol) is None and rs.pca_orientation(vol) is not None
    aligned, E = rs.align_volume(vol, "frame")
    assert np.allclose(E, rs.pca_orientation(vol)) and aligned.shape == vol.shape
