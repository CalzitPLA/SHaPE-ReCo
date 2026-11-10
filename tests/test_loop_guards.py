"""The box cover of the voxel-cube fallback (reco_engine.loop_guards) and the placement helpers it was tested with."""
import numpy as np
import pytest

from reco_engine.loop_guards import largest_box_in_blocks, max_rectangle

RES = 32


def grid(*blocks):
    g = np.zeros((RES, RES, RES), dtype=np.uint8)
    for (x0, x1), (y0, y1), (z0, z1) in blocks:
        g[x0:x1, y0:y1, z0:z1] = 1
    return g


# ---------------------------------------------------------------- helpers

def test_max_rectangle_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.random((9, 11)) > 0.3
        best = 0
        for r0 in range(9):
            for r1 in range(r0, 9):
                for c0 in range(11):
                    for c1 in range(c0, 11):
                        if m[r0:r1 + 1, c0:c1 + 1].all():
                            best = max(best, (r1 - r0 + 1) * (c1 - c0 + 1))
        assert max_rectangle(m)[0] == best


def test_largest_box_in_blocks_finds_the_planted_box():
    b = np.zeros((10, 10, 10), bool)
    b[2:8, 1:5, 3:9] = True
    b[8, 8, 8] = True
    volume, (i0, i1, r0, r1, c0, c1) = largest_box_in_blocks(b)
    assert volume == 6 * 4 * 6 and (i0, i1, r0, r1, c0, c1) == (2, 7, 1, 4, 3, 8)


# ---------------------------------------------------------------- the loop

def occupied(result):
    out = np.zeros((RES,) * 3, bool)
    for p in result.accepted:
        out |= p.placement.voxel.astype(bool)
    return out


# ---------------------------------------------------------------- bounding-box anchor

def test_bbox_anchor_puts_a_candidate_exactly_on_the_bounding_box_of_a_lopsided_target():
    from reco_engine.placement import estimate_placement, place_candidate
    target = grid(((4, 24), (4, 24), (4, 24)))
    target[4:14, 4:24, 14:24] = 0                                      # a big notch: the centre of gravity moves away from it
    cube = np.ones((RES,) * 3, np.uint8)
    for anchor, exact in (("bbox", True), ("cog", False)):
        placed = place_candidate(cube, estimate_placement(target, RES, 10.0, anchor), RES).astype(bool)
        box = np.argwhere(target)
        lo, hi = box.min(axis=0), box.max(axis=0)
        on_bbox = np.array_equal(np.argwhere(placed).min(axis=0), lo) and np.array_equal(np.argwhere(placed).max(axis=0), hi)
        assert on_bbox == exact
    assert estimate_placement(target, RES, 10.0).anchor == "cog"      # default unchanged
    with pytest.raises(ValueError):
        estimate_placement(target, RES, 10.0, "middle")


def test_bbox_anchor_lands_on_even_and_odd_extents():
    from reco_engine.placement import estimate_placement, place_candidate
    for size in (7, 8):
        for offset in (5, 6):
            t = grid(((offset, offset + size), (offset, offset + size), (offset, offset + size)))
            placed = place_candidate(np.ones((RES,) * 3, np.uint8), estimate_placement(t, RES, 10.0, "bbox"), RES)
            assert np.array_equal(placed, t)


def test_true_scale_placement_translates_without_resizing_and_reports_its_own_footprint():
    from reco_engine.placement import estimate_placement, place_candidate_true_scale
    target = grid(((4, 24), (6, 22), (8, 26)))                            # 20 x 16 x 18
    small = np.zeros((RES,) * 3, np.uint8)
    small[10:16, 10:14, 10:18] = 1                                        # a 6 x 4 x 8 candidate somewhere else
    placed, own = place_candidate_true_scale(small, estimate_placement(target, RES, 10.0, "bbox"), RES, 10.0 / RES)
    assert placed.sum() == small.sum()                                    # same size, not stretched to 20 x 16 x 18
    box = np.argwhere(placed)
    assert list(box.max(axis=0) - box.min(axis=0) + 1) == [6, 4, 8]
    assert np.allclose((box.min(axis=0) + box.max(axis=0)) / 2.0, [13.5 + 0.5 * 0, 13.5, 16.5], atol=0.5)   # bbox centres coincide
    assert list(own.extent_vox) == [6, 4, 8] and own.anchor == "bbox"
    assert np.allclose(own.center_of_gravity_vox, (box.min(axis=0) + box.max(axis=0)) / 2.0)
    empty, same = place_candidate_true_scale(np.zeros_like(small), estimate_placement(target, RES, 10.0, "bbox"), RES, 0.3)
    assert empty.sum() == 0


# ---------------------------------------------------------------- acceptance by the assembly grade

# ---------------------------------------------------------------- inscribed cylinders and balls

def _cyl(center, axis, r, half):
    from reco_engine.cylinder_fit import cylinder_mask_points
    grid = np.indices((64, 64, 64)).reshape(3, -1).T.astype(float)
    return cylinder_mask_points(grid, np.asarray(center, float), np.asarray(axis, float), r, half).reshape(64, 64, 64)

