"""
reco_engine.void_quality: a subtracted body is good if it removes no input
material, and the best one is the biggest such body.
"""

import numpy as np
import pytest

from reco_engine.local_search import _rescale
from reco_engine.void_quality import assess_void

N = 32


def grid(x0, x1, y0, y1, z0, z1):
    g = np.zeros((N,) * 3, dtype=bool)
    g[x0:x1, y0:y1, z0:z1] = True
    return g


# A solid cube with a square tunnel through it along axis 2.
CUBE = grid(4, 28, 4, 28, 4, 28)
TUNNEL = grid(12, 20, 12, 20, 0, N)
INPUT = CUBE & ~TUNNEL          # what the part really is
ASSEMBLY = CUBE                 # the cover built a solid cube
GROW = lambda c: _rescale(c.astype(np.uint8), 1.25, N)


def test_exact_tunnel_is_good_and_covers_all_excess():
    a = assess_void(TUNNEL, ASSEMBLY, INPUT)
    assert a.verdict == "good" and a.purity == 1.0 and a.removed_material == 0
    assert a.coverage == pytest.approx(1.0)


def test_protruding_into_empty_space_is_harmless():
    longer = grid(12, 20, 12, 20, 0, N)         # already passes through both faces
    beyond = grid(12, 20, 12, 20, 0, N)
    a = assess_void(beyond, ASSEMBLY, INPUT)
    assert a.verdict == "good"
    assert a.volume > a.removes                 # part of it lies outside the assembly, and that is not counted


def test_touching_material_is_bad_and_counted():
    fat = grid(10, 22, 10, 22, 0, N)            # wider than the tunnel: cuts into real material
    a = assess_void(fat, ASSEMBLY, INPUT)
    assert a.verdict == "hits_material"
    assert a.removed_material == int((fat & INPUT).sum()) > 0
    assert a.purity < 0.98


def test_small_pure_void_is_good_but_not_maximal_the_exact_one_is():
    small = grid(14, 18, 14, 18, 8, 24)         # fits inside the tunnel with room to grow
    exact = TUNNEL
    a_small = assess_void(small, ASSEMBLY, INPUT, grow=GROW)
    a_exact = assess_void(exact, ASSEMBLY, INPUT, grow=GROW)
    assert a_small.verdict == "good" and a_small.maximal is False
    assert a_exact.verdict == "good" and a_exact.maximal is True
    assert a_exact.coverage > a_small.coverage       # bigger good void removes more of the excess


def test_void_outside_the_assembly_changes_nothing():
    a = assess_void(grid(0, 3, 0, 3, 0, 3), ASSEMBLY, INPUT)
    assert a.verdict == "empty" and a.removes == 0


def test_purity_threshold_is_adjustable():
    slightly_fat = grid(11, 20, 12, 20, 0, N)   # one voxel column of material hit
    assert assess_void(slightly_fat, ASSEMBLY, INPUT, purity_min=0.99).verdict == "hits_material"
    assert assess_void(slightly_fat, ASSEMBLY, INPUT, purity_min=0.8).verdict == "good"


# -------------------------------------------------- allowed region in the loop
