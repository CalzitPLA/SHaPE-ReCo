"""reco_field/peel.py's `octree` option (docs/FRAME_FAILURE.md, "Debugging the identification schema with the
frame"): an occupancy-octree alternative to `_side_clusters`/`mesh_clusters`, grouping the missing VOLUME by
spatial locality rather than surface proximity or connectivity - ported from `reco_engine/octree_search.py`'s
voxel-CNN strategy (`octree` branch). Off by default - the ordinary peel()/peel_steps() behaviour must not
change at all when it is left False."""
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.peel import _grid_centres, _octree_leaves, input_occupancy, peel  # noqa: E402
from src.closest_point_field import SurfaceField  # noqa: E402


def two_bosses_on_a_slab():
    slab = trimesh.creation.box(extents=(40.0, 20.0, 4.0))
    boss1 = trimesh.creation.box(extents=(3.0, 3.0, 3.0))
    boss1.apply_translation((-15.0, 0.0, 3.5))
    boss2 = trimesh.creation.box(extents=(3.0, 3.0, 3.0))
    boss2.apply_translation((15.0, 0.0, 3.5))
    mesh = trimesh.util.concatenate([slab, boss1, boss2]).process(validate=True)
    return SurfaceField(mesh.vertices, mesh.faces)


def test_input_occupancy_matches_the_mesh_volume_roughly():
    field = two_bosses_on_a_slab()
    lo, hi = field.bounds
    occ = input_occupancy(field, lo, hi, 32)
    cell_volume = np.prod((hi - lo) / 32)
    true_volume = 40.0 * 20.0 * 4.0 + 2 * 3.0 ** 3
    assert abs(occ.sum() * cell_volume - true_volume) / true_volume < 0.15


def test_octree_leaves_prunes_empty_space_and_keeps_real_material():
    occ = np.zeros((16, 16, 16), bool)
    occ[2:6, 2:6, 2:6] = True                    # one small dense block, far from the rest empty
    leaves = _octree_leaves(occ, min_voxels=4, min_fraction=0.05, max_depth=3)
    assert leaves                                 # something survived
    total_leaf_voxels = sum(int(occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].sum()) for lo, hi in leaves)
    assert total_leaf_voxels > 0


def test_octree_leaves_covers_every_voxel_on_an_odd_axis_size():
    """A regression guard for a real bug the ultrareview cloud review caught: `descend`'s upper octant used
    `half` (the SAME width as the lower octant) instead of `size - half` (the remainder), so on an odd axis
    size (e.g. 9 -> half 4) the last voxel along that axis (index 8) fell into neither child and was silently
    dropped from every leaf, however occupied it was."""
    occ = np.ones((9, 9, 9), bool)
    leaves = _octree_leaves(occ, min_voxels=1, min_fraction=0.0, max_depth=1)
    covered = np.zeros_like(occ)
    for lo, hi in leaves:
        covered[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    assert covered.all(), f"{(~covered).sum()} of {occ.size} occupied voxels were never covered by any leaf"


def test_octree_off_by_default_does_not_change_peel_result():
    field = two_bosses_on_a_slab()
    default = peel(field, max_ops=6)
    explicit_off = peel(field, max_ops=6, octree=False)
    assert default.explained == explicit_off.explained
    assert [o.fit.kind for o in default.ops] == [o.fit.kind for o in explicit_off.ops]


def test_octree_mode_finds_and_adds_both_bosses():
    """octree_depth=2 (not the default 1 - a shallow default is deliberately biased toward NOT over-fragmenting a
    simple residual, `docs/FRAME_FAILURE.md`'s frame measurement; here the two bosses are far enough apart that
    a little more depth is what actually tells them apart, which is the property this test demonstrates)."""
    field = two_bosses_on_a_slab()
    result = peel(field, max_ops=6, octree=True, octree_depth=2)
    assert result.explained > 0.95
    assert sum(1 for o in result.ops if o.op == "add") >= 2
