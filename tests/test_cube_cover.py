"""The voxel-cube cover (ResidualConfig.start_shape "cubes", 2026-10-10): the largest completely filled boxes of the
input occupancy become the first adds, inside the material; the loop goes on from there."""

from pathlib import Path

import numpy as np
import pytest

from reco_engine import residual_loop as rl

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "dataset_gen" / "examples"


def test_largest_boxes_finds_an_off_centre_asymmetric_block_first():
    mask = np.zeros((40, 30, 50), bool)
    mask[3:8, 10:17, 20:29] = True            # 5 x 7 x 9 = 315 voxels, off centre
    mask[30:33, 2:5, 40:43] = True            # 3 x 3 x 3 = 27 voxels
    boxes = rl.largest_boxes(mask, count=3, max_side=48, min_voxels=8)
    assert len(boxes) == 2
    (lo, hi), (lo2, hi2) = boxes
    assert lo.tolist() == [3, 10, 20] and hi.tolist() == [7, 16, 28]
    assert lo2.tolist() == [30, 2, 40] and hi2.tolist() == [32, 4, 42]


@pytest.mark.skipif(not (EXAMPLES / "l_bracket.stl").is_file(), reason="example part missing")
def test_cube_cover_adds_boxes_inside_the_part_and_the_loop_goes_on():
    trimesh = pytest.importorskip("trimesh")
    stl = EXAMPLES / "l_bracket.stl"
    cfg = rl.ResidualConfig(start_shape="cubes", feature_phase=False, freecad=False, resolution=48, max_resolution=48,
                            grid="cube")
    steps = list(rl.residual_steps(str(stl), cfg))
    kinds = [s.kind for s in steps]
    assert kinds[0] == "input" and "add" in kinds and kinds[-1] == "done"
    add = steps[kinds.index("add")]
    assert add.info["message"].startswith("first add: voxel cube cover")
    cubes = add.info["cubes"]
    assert len(cubes) >= 2 and cubes[0][2] >= cubes[1][2]                      # the largest box first
    res = steps[-1].result
    ops = [o for o in res.operations if o.accepted]
    cover = ops[:len(cubes)]
    assert all(o.op == "add" and o.primitive.kind == "box" for o in cover)
    assert all(np.allclose(o.primitive.quaternion, [0, 0, 0, 1]) for o in cover)   # grown along the grid's axes
    # every cover box lies inside the part: its centre and its corners, the latter pulled in by one pitch
    mesh = trimesh.load(str(stl))
    pitch = float(steps[0].info["spec"].pitch) if hasattr(steps[0].info["spec"], "pitch") else 20.0 / 48
    for o in cover:
        c = np.asarray(o.primitive.center_mm, float)
        h = np.array([o.primitive.dims_mm[k] for k in ("x_half", "y_half", "z_half")], float) - pitch
        corners = c + np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.maximum(h, 0.0)
        assert mesh.contains(np.vstack([c[None], corners])).all(), o.primitive
    # the two legs of the L explain the part: the loop ends near the input
    assert steps[-1].info["stats"]["iou"] >= 0.9, steps[-1].info.get("stopped")


def test_without_a_voxel_cube_the_legacy_start_takes_over():
    """B5 of the user's session of 2026-10-29: the ABC part 00005438 (a 1.6 mm plate, 2-3 voxels thick) has no completely
    filled box of 27 voxels, and the cubes path ended with 0 operations and 'residual +0 / -0'. Now a "fallback" step says
    so and the legacy start reconstructs the plate."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_add_cut_loop import _field
    from src import sdf_primitives as sp
    plate = sp.Placed(sp.Box([20.0, 12.0, 2.0]), translation=np.array([3.0, -2.0, 1.0]))
    cfg = rl.ResidualConfig(start_shape="cubes", feature_phase=False, freecad=False, resolution=32, max_resolution=32,
                            grid="cube", max_ops=3, cubes_min_voxels=10 ** 6)    # no box can be that big: no cube
    steps = list(rl.residual_steps(_field(plate), cfg))
    fb = [s for s in steps if s.kind == "fallback" and s.info.get("reason") == "no voxel cube"]
    assert fb and "the legacy start takes over" in fb[0].info["message"]
    assert steps[-1].kind == "done" and steps[-1].result.operations                # not an empty run
