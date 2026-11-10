"""residual_loop.largest_boxes: the exact filled boxes of a cut piece (the frame's windows and flange recesses after a box
start), at bounded cost on a frame-sized grid."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine.residual_loop import largest_boxes  # noqa: E402


def test_an_l_shape_gives_back_its_two_boxes():
    m = np.zeros((40, 50, 30), bool)
    m[3:23, 7:44, 5:17] = True                    # the big box: 20 x 37 x 12
    m[23:31, 7:15, 5:27] = True                   # the leg, touching it: 8 x 8 x 22
    (lo1, hi1), (lo2, hi2) = largest_boxes(m, 2, max_side=8)
    assert (lo1.tolist(), hi1.tolist()) == ([3, 7, 5], [22, 43, 16])
    assert (lo2.tolist(), hi2.tolist()) == ([23, 7, 5], [30, 14, 26])


def test_a_frame_sized_excess_is_fast():
    """A 4 x 6 window plate on the frame's tau grid (88 x 562 x 419 cells): the first box is a whole window."""
    m = np.zeros((88, 562, 419), bool)
    for i in range(4):
        for j in range(6):
            m[:, 20 + i * 135:20 + i * 135 + 115, 15 + j * 67:15 + j * 67 + 55] = True
    m[30:58, 10:552, 10:409] |= True              # the flange recess joining them: one connected piece
    t = time.time()
    boxes = largest_boxes(m, 3)
    assert time.time() - t < 20
    assert all(m[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1].all() for lo, hi in boxes)
    vols = [int(np.prod(hi - lo + 1)) for lo, hi in boxes]
    assert vols[0] >= 28 * 542 * 399              # the recess slab is the largest box
    assert vols == sorted(vols, reverse=True)
