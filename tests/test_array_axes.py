"""
Which way do the voxel array's axes run relative to the world? The array is [z, y, x], but its y axis
runs BACKWARDS (stltovoxel writes image rows top to bottom). The conversion of signed voxel-space
quantities to world coordinates has to account for that, or every position is mirrored in y - which
made the voxel model a mirror image of the assembled geometry.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

from reco_engine.placement import array_to_world, array_to_world_position

SANDBOX = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX / "dataset_gen"))
from src.voxelization import find_stltovoxel, voxelize_stl_to_bin  # noqa: E402


def test_position_conversion_negates_y_and_reverses_order():
    assert array_to_world_position([1.0, 2.0, 3.0]) == [3.0, -2.0, 1.0]       # (x, y, z) = (a2, -a1, a0)
    assert array_to_world_position(v for v in (0, 0, 0)) == [0.0, -0.0, 0.0]


def test_extents_keep_their_sign():
    assert array_to_world([16, 32, 64]) == [64.0, 32.0, 16.0]                # sizes are unsigned: no negation


@pytest.mark.skipif(find_stltovoxel() is None, reason="stltovoxel not installed")
@pytest.mark.parametrize("marker, array_axis, at_high_index", [
    ("+x", 2, True), ("-x", 2, False),
    ("+y", 1, False), ("-y", 1, True),          # y is the reversed one
    ("+z", 0, True), ("-z", 0, False),
])
def test_array_axis_directions_against_the_world(tmp_path, marker, array_axis, at_high_index):
    """A 20 mm cube with a 6 mm marker sticking out of its +/- face on one world axis: the extension lands
    on the expected array axis, at the expected end."""
    axis, sign = "xyz".index(marker[1]), 1 if marker[0] == "+" else -1
    big = trimesh.creation.box(extents=(20, 20, 20))
    small = trimesh.creation.box(extents=(6, 6, 6))
    offset = [0.0, 0.0, 0.0]
    offset[axis] = sign * 12.5
    small.apply_translation(offset)
    stl, out = tmp_path / "m.stl", tmp_path / "m.bin"
    trimesh.util.concatenate([big, small]).export(stl)
    assert voxelize_stl_to_bin(str(stl), str(out), resolution=64, scale_mode="fixed", scale_mm=40.0,
                               voxelizer="stltovoxel")              # its centred grid (anchor center): stltovoxel's own
    grid = np.fromfile(out, dtype=np.uint8).reshape(64, 64, 64).astype(bool)

    extent = np.ptp(np.argwhere(grid), axis=0) + 1
    assert int(np.argmax(extent)) == array_axis                              # the axis that grew
    counts = grid.sum(tuple(a for a in range(3) if a != array_axis))
    nz = np.nonzero(counts)[0]
    low_end, high_end = counts[nz.min():nz.min() + 8].mean(), counts[nz.max() - 7:nz.max() + 1].mean()
    assert (high_end < low_end) == at_high_index                             # the marker end is the sparse one
