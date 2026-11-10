"""reco_engine/mass_compartments.py: the mass of a part split into well-filled boxes (the user, 2026-10-29: "look for
mass-population compartments on the whole, then add-cuts until good fits"): an L gives its two boxes, a plate with a
hole stays one compartment (the loop cuts the hole), a frame gives its four bars, scattered mass is left out."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine import mass_compartments as mc  # noqa: E402


def _covers(leaves, occ):
    got = np.zeros_like(occ)
    for c in leaves:
        (x0, y0, z0), (x1, y1, z1) = c["lo"], c["hi"]
        got[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = True
    return got


def test_an_l_shape_gives_its_two_boxes_exactly():
    occ = np.zeros((30, 24, 10), bool)
    occ[2:22, 3:9, 1:7] = True                                   # the long leg, off-centre
    occ[2:8, 9:20, 1:7] = True                                   # the short leg
    leaves = mc.compartments(occ)
    assert len(leaves) == 2 and all(c["fill"] == 1.0 for c in leaves)
    assert sum(c["voxels"] for c in leaves) == int(occ.sum()) and (_covers(leaves, occ) == occ).all()


def test_a_plate_with_a_hole_stays_one_compartment_and_a_frame_splits_into_bars():
    plate = np.zeros((40, 30, 6), bool)
    plate[5:35, 4:26, 1:5] = True
    plate[18:22, 13:17, :] = False                              # a small hole: the loop's cut
    leaves = mc.compartments(plate)
    assert len(leaves) == 1 and leaves[0]["fill"] >= mc.FILL_GOOD
    frame = np.zeros((26, 26, 5), bool)
    frame[3:23, 3:23, 1:4] = True
    frame[6:20, 6:20, :] = False                                # a window: 51 % filled as one box
    leaves = mc.compartments(frame)
    assert len(leaves) == 4 and all(c["fill"] == 1.0 for c in leaves)
    assert sum(c["voxels"] for c in leaves) == int(frame.sum())


def test_scattered_mass_is_left_out_and_the_leaf_count_is_capped():
    occ = np.zeros((40, 40, 40), bool)
    occ[5:30, 5:30, 5:30] = True
    rng = np.random.default_rng(0)
    pts = rng.integers(0, 40, (60, 3))
    occ[pts[:, 0], pts[:, 1], pts[:, 2]] = True                  # noise voxels all over
    leaves = mc.compartments(occ, max_leaves=6)
    assert 1 <= len(leaves) <= 6 and leaves[0]["voxels"] >= 25 ** 3 and all(c["fill"] >= mc.MIN_FILL for c in leaves)
    assert mc.compartments(np.zeros((5, 5, 5), bool)) == []
