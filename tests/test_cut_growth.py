"""ResidualConfig.cut_grow: a box cut grows into the open space (residual_loop._grow_cut_box) and the cut audit
(reco_engine/cut_merge.py) - on small synthetic grids and meshes."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine import residual_loop as rl  # noqa: E402
from reco_engine.result_schema import Primitive  # noqa: E402


def _state(material):
    """A grid of 20^3 cells of pitch 1 with centres at 0.5 .. 19.5, the material a boolean array."""
    n = material.shape
    idx = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), axis=-1)
    centres = idx + 0.5
    s_in = np.where(material, 1.0, -5.0)                              # inside +, far outside -
    return SimpleNamespace(centres=centres, pitch=1.0, occ_in=material, s_in=s_in, band=0.5)


def test_a_box_cut_grows_to_the_material_and_through_open_sides():
    m = np.zeros((20, 20, 20), bool)
    m[:, :, :8] = True                                                # a slab z < 8 (the plate)
    m[12:16, 12:16, :] = True                                         # a post x, y in 12..16 (all z)
    st = _state(m)
    cut = Primitive(kind="box", center_mm=(5.0, 5.0, 10.0), dims_mm=dict(x_half=2.0, y_half=2.0, z_half=1.0))
    g = rl._grow_cut_box(st, cut)
    lo = np.subtract(g.center_mm, [g.dims_mm["x_half"], g.dims_mm["y_half"], g.dims_mm["z_half"]])
    hi = np.add(g.center_mm, [g.dims_mm["x_half"], g.dims_mm["y_half"], g.dims_mm["z_half"]])
    assert lo[2] == pytest.approx(8.0) and hi[2] == pytest.approx(23.0)    # down to the plate, up through the open top
    assert lo[0] == pytest.approx(-3.0) and lo[1] == pytest.approx(-3.0)   # open sides: three cells beyond the grid
    assert hi[0] == pytest.approx(23.0)                                     # x first: past the post (y 12..16), open
    assert hi[1] == pytest.approx(12.0)                                     # then y: stops at the post
    turned = Primitive(kind="box", center_mm=(5.0, 5.0, 10.0), quaternion=(0.0, 0.0, 0.3826834, 0.9238795),
                       dims_mm=dict(x_half=2.0, y_half=2.0, z_half=1.0))
    assert rl._grow_cut_box(st, turned) is None                             # world-axis boxes only


def test_a_box_against_material_everywhere_does_not_move():
    m = np.ones((20, 20, 20), bool)
    m[4:7, 4:7, 4:7] = False
    st = _state(m)
    cut = Primitive(kind="box", center_mm=(5.5, 5.5, 5.5), dims_mm=dict(x_half=1.5, y_half=1.5, z_half=1.5))
    assert rl._grow_cut_box(st, cut) is None


def test_the_audit_finds_the_two_stacked_cuts_of_one_box(tmp_path):
    import trimesh
    from reco_engine.cut_merge import audit, merge_cuts
    plate = trimesh.creation.box((40, 40, 8))
    plate.apply_translation((0, 0, 4))                                  # z 0..8
    cover = Primitive(kind="box", center_mm=(0.0, 0.0, 10.0), dims_mm=dict(x_half=20.0, y_half=20.0, z_half=10.0))
    low = Primitive(kind="box", center_mm=(0.0, 0.0, 11.0), dims_mm=dict(x_half=21.0, y_half=21.0, z_half=3.0))   # z 8..14
    high = Primitive(kind="box", center_mm=(0.0, 0.0, 17.0), dims_mm=dict(x_half=21.0, y_half=21.0, z_half=4.0))  # z 13..21
    into = Primitive(kind="box", center_mm=(0.0, 0.0, 6.0), dims_mm=dict(x_half=2.0, y_half=2.0, z_half=3.0))     # cuts material
    ops = [("add", cover), ("cut", low), ("cut", high)]
    r = audit(ops, plate.vertices, plate.faces)
    assert [(p["i"], p["j"]) for p in r["mergeable_pairs"]] == [(1, 2)] and r["cuts"] == 2
    merged, rep = merge_cuts(ops, plate.vertices, plate.faces)
    assert len(merged) == 2 and rep["merges"][0]["i"] == 1
    m = merged[1][1]
    assert m.center_mm[2] - m.dims_mm["z_half"] == pytest.approx(8.0) and m.center_mm[2] + m.dims_mm["z_half"] == pytest.approx(21.0)
    r2 = audit([("add", cover), ("cut", low), ("cut", into)], plate.vertices, plate.faces)
    assert r2["mergeable_pairs"] == []                                  # the hull would cut the plate


def test_open_only_moves_just_the_faces_that_see_free_space_to_the_edge():
    m = np.zeros((20, 20, 20), bool)
    m[:, :, :8] = True
    m[12:16, 12:16, :] = True
    st = _state(m)
    cut = Primitive(kind="box", center_mm=(5.0, 5.0, 10.0), dims_mm=dict(x_half=2.0, y_half=2.0, z_half=1.0))
    g = rl._grow_cut_box(st, cut, open_only=True)
    lo = np.subtract(g.center_mm, [g.dims_mm["x_half"], g.dims_mm["y_half"], g.dims_mm["z_half"]])
    hi = np.add(g.center_mm, [g.dims_mm["x_half"], g.dims_mm["y_half"], g.dims_mm["z_half"]])
    assert lo[2] == pytest.approx(9.0) and hi[2] == pytest.approx(23.0)     # the floor stays (plate below), the top opens
    assert lo[0] == pytest.approx(-3.0) and hi[0] == pytest.approx(23.0)    # x: free to both edges (the post is at y 12+)
    assert lo[1] == pytest.approx(-3.0) and hi[1] == pytest.approx(7.0)     # +y sees the post: stays


def test_a_cut_face_in_the_covers_face_moves_past_it_unless_an_add_is_there():
    from reco_engine.cut_merge import lift_cover_faces
    cover = Primitive(kind="box", center_mm=(0.0, 0.0, 10.0), dims_mm=dict(x_half=20.0, y_half=20.0, z_half=10.0))
    cut = Primitive(kind="box", center_mm=(10.0, 0.0, 15.0), dims_mm=dict(x_half=10.004, y_half=5.0, z_half=5.0))  # +x at 20.004
    ops, moved = lift_cover_faces([("add", cover), ("cut", cut)], margin=3.0)
    c = ops[1][1]
    assert moved == 2 and c.center_mm[0] + c.dims_mm["x_half"] == pytest.approx(23.0)        # +x past the cover
    assert c.center_mm[2] + c.dims_mm["z_half"] == pytest.approx(23.0)                       # the top too (z 20)
    boss = Primitive(kind="box", center_mm=(21.0, 0.0, 15.0), dims_mm=dict(x_half=1.5, y_half=1.0, z_half=1.0))
    ops, moved = lift_cover_faces([("add", cover), ("cut", cut), ("add", boss)], margin=3.0)
    assert moved == 1 and ops[1][1].center_mm[0] + ops[1][1].dims_mm["x_half"] == pytest.approx(20.004)   # +x stays


def test_a_thin_wall_between_two_cuts_is_closed_where_no_material_is():
    import trimesh
    from reco_engine.cut_merge import close_seams
    plate = trimesh.creation.box((40, 40, 8))
    plate.apply_translation((0, 0, 4))                                  # material z 0..8
    a = Primitive(kind="box", center_mm=(-5.0, 0.0, 14.0), dims_mm=dict(x_half=5.0, y_half=10.0, z_half=6.0))   # x -10..0
    b = Primitive(kind="box", center_mm=(5.02, 0.0, 14.0), dims_mm=dict(x_half=5.0, y_half=10.0, z_half=6.0))   # x 0.02..10
    ops, closed = close_seams([("cut", a), ("cut", b)], plate.vertices, plate.faces, tol=0.5, overlap=1.0)
    assert closed == 1 and ops[0][1].center_mm[0] + ops[0][1].dims_mm["x_half"] == pytest.approx(1.02)
    low_a = Primitive(kind="box", center_mm=(-5.0, 0.0, 4.0), dims_mm=dict(x_half=5.0, y_half=10.0, z_half=6.0))  # in the plate
    low_b = Primitive(kind="box", center_mm=(5.02, 0.0, 4.0), dims_mm=dict(x_half=5.0, y_half=10.0, z_half=6.0))
    ops, closed = close_seams([("cut", low_a), ("cut", low_b)], plate.vertices, plate.faces, tol=0.5, overlap=1.0)
    assert ops[0][1] is low_a and ops[1][1] is low_b                    # the wall holds material: no face moves ...
    seam = [p for o, p in ops[2:]]
    assert closed == len(seam) >= 1 and all("seam cut" in p.label for p in seam)
    def outside_plate(p):                                               # ... only its free parts: above or below
        lo, hi = p.center_mm[2] - p.dims_mm["z_half"], p.center_mm[2] + p.dims_mm["z_half"]
        return lo >= 8.0 - 0.3 or hi <= 0.0 + 0.3
    assert all(outside_plate(p) for p in seam)


def test_the_loops_tidy_keeps_every_operation_in_its_place_when_seam_cuts_are_inserted():
    """The first version wrote the tidied primitives back by position: a seam cut inserted before the end shifted every
    later primitive onto the wrong operation (the bracket on server_a: exact IoU 0.704 instead of 0.9994)."""
    import trimesh
    from types import SimpleNamespace
    from reco_engine.result_schema import Operation, ReconstructionResult
    plate = trimesh.creation.box((40, 40, 8))
    plate.apply_translation((0, 0, 4))
    cover = Primitive(kind="box", center_mm=(0.0, 0.0, 6.0), dims_mm=dict(x_half=20.0, y_half=20.0, z_half=6.0))
    low_a = Primitive(kind="box", center_mm=(-5.0, 0.0, 4.0), dims_mm=dict(x_half=5.0, y_half=10.0, z_half=6.0))
    low_b = Primitive(kind="box", center_mm=(5.02, 0.0, 4.0), dims_mm=dict(x_half=5.0, y_half=10.0, z_half=6.0))
    boss = Primitive(kind="cylinder", center_mm=(15.0, 15.0, 10.0), dims_mm=dict(radius=2.0, half_length=2.0))
    result = ReconstructionResult(input_path="x", operations=[Operation("add", cover), Operation("cut", low_a),
                                                              Operation("cut", low_b), Operation("add", boss)])
    field = SimpleNamespace(vertices=plate.vertices, faces=plate.faces)
    steps = list(rl._tidy_cuts(result, field, SimpleNamespace(pitch=1.0)))
    ops = result.operations
    assert steps and len(ops) > 4                                       # seam cuts went in
    assert ops[0].op == "add" and ops[0].primitive.kind == "box" and ops[0].primitive.dims_mm["x_half"] == 20.0
    assert ops[-1].op == "add" and ops[-1].primitive.kind == "cylinder"   # the boss stays the boss, last
    assert [o.op for o in ops[1:-1]] == ["cut"] * (len(ops) - 2)
    assert all("seam cut" in o.primitive.label for o in ops[3:-1])


def test_a_second_tidy_does_not_repeat_the_seam_cuts_of_the_first():
    """Protocol rough_tidy_v1, DEV (2026-10-29): the tidy after the feature phase found the rough phase's seam again (the
    wall still stands in the material) and added its seam cut twice more (aegir_n6_I_000121: three identical cuts)."""
    import trimesh
    from reco_engine.cut_merge import close_seams
    plate = trimesh.creation.box((40, 40, 8))
    plate.apply_translation((0, 0, 4))
    low_a = Primitive(kind="box", center_mm=(-5.0, 0.0, 4.0), dims_mm=dict(x_half=5.0, y_half=10.0, z_half=6.0))
    low_b = Primitive(kind="box", center_mm=(5.02, 0.0, 4.0), dims_mm=dict(x_half=5.0, y_half=10.0, z_half=6.0))
    once, n1 = close_seams([("cut", low_a), ("cut", low_b)], plate.vertices, plate.faces, tol=0.5, overlap=1.0)
    twice, n2 = close_seams(once, plate.vertices, plate.faces, tol=0.5, overlap=1.0)
    assert n1 >= 1 and n2 == 0 and len(twice) == len(once)
