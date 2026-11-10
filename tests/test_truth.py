"""reco_engine/truth.py: the known trees of the validation cycle - program round trip, the v5 record adapter against
src/solid_builder's own labels, the negative-residual exclusion, and the gate that catches a wrong tree."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reco_engine.truth import (program_from_dict, truth_from_tree, truth_gate, v5_exact, v5_leaf)  # noqa: E402
from reco_engine.result_to_program import sp  # noqa: E402
from reco_field.export import program_to_dict  # noqa: E402


def _rot(seed):
    from scipy.spatial.transform import Rotation
    return Rotation.random(random_state=seed).as_matrix()


def test_program_round_trip_keeps_the_sdf():
    tree = sp.Subtraction(
        sp.Union(sp.Placed(sp.Box((1.0, 2.0, 0.5)), _rot(1), np.array([0.3, -0.2, 0.1])),
                 sp.Placed(sp.Sphere(1.2, u=(0.0, 4.0), v=(-0.5, 1.0)), _rot(2), np.array([1.0, 0.5, 0.0]))),
        sp.Intersection(sp.Placed(sp.Cylinder(0.4, 3.0, inner_radius=0.1), _rot(3), np.array([0.0, 0.0, -1.5])),
                        sp.Placed(sp.Torus(1.0, 0.3, u=(0.5, 5.0), v=(0.0, 3.0)), _rot(4), np.zeros(3))))
    back = program_from_dict(program_to_dict(tree))
    P = np.random.default_rng(0).uniform(-3, 3, (5000, 3))
    assert np.allclose(tree.sdf(P), back.sdf(P), atol=1e-12)


@pytest.mark.parametrize("topology,params,segment", [
    ("box", dict(x_half_ratio=1.2, y_half_ratio=0.5, z_half_ratio=0.8), None),
    ("cylinder", dict(radius_ratio=0.6, half_length_ratio=1.1), dict(u=[0.4, 3.9])),
    ("cone", dict(radius1_ratio=0.9, radius2_ratio=0.3, height_ratio=1.7), None),
    ("sphere", dict(radius_ratio=0.9), dict(v=[-0.4, 1.1])),
    ("torus", dict(major_radius_ratio=1.0, minor_radius_ratio=0.3), dict(u=[0.0, 4.0], v=[0.5, 4.5])),
])
def test_v5_leaf_is_solid_builders_shape(topology, params, segment):
    from src import solid_builder
    prim = solid_builder.local_primitive(topology, params, scale_mm=10.0, segment=segment)
    R = _rot(7)
    placement = dict(scale_factor=1.3, center_mm=[2.0, -3.0, 1.5])
    bbox_center = np.array([0.4, -0.7, 0.2])
    want = solid_builder.placed_program(prim, R, placement, bbox_center)
    lab = solid_builder.labels(prim, R, placement, bbox_center)
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()
    record = dict(topology=topology, dims_mm=lab["dims_mm"], center_mm=lab["center_mm"],
                  rotation=dict(qx=q[0], qy=q[1], qz=q[2], qw=q[3]), placement={})
    if "segment" in lab:
        record["segment"] = lab["segment"]
    got = v5_leaf(record)
    P = np.random.default_rng(1).uniform(-30, 30, (20000, 3))
    assert np.array_equal(want.sdf(P) <= 0, got.sdf(P) <= 0)
    assert np.allclose(want.sdf(P), got.sdf(P), atol=1e-9)


def test_v5_exact_refuses_what_cannot_be_rebuilt():
    base = dict(topology="box", subclass="full", placement={})
    assert v5_exact(base) == (True, "plain")
    assert v5_exact(dict(base, placement=dict(carve=dict(cutter="sphere"))))[1] == "carved"
    assert not v5_exact(dict(base, placement=dict(cuts=[dict(normal=[1, 0, 0], keep=0.5)])))[0]
    assert not v5_exact(dict(base, wall_mm=1.0))[0]
    assert not v5_exact(dict(base, members=[{}]))[0]
    assert not v5_exact(dict(base, boxes=[{}]))[0]
    assert not v5_exact(dict(base, sample_kind="negres"))[0]
    assert not v5_exact(dict(base, placement=dict(degenerate=True)))[0]


def test_gate_accepts_the_right_tree_and_catches_a_wrong_one(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("igl")
    R, t = _rot(11), np.array([1.0, -2.0, 0.5])
    mesh = trimesh.creation.box(extents=(4.0, 2.0, 1.0))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    mesh.apply_transform(T)
    path = tmp_path / "box.stl"
    mesh.export(path)
    good = truth_from_tree(sp.Placed(sp.Box((2.0, 1.0, 0.5)), R, t), "box")
    g = truth_gate(good, path, n=60_000)
    assert g["ok"], g["reasons"]
    assert g["genus"] == 0
    wrong = truth_from_tree(sp.Placed(sp.Box((2.0, 1.0, 0.5)), _rot(12), t), "box_wrong_rotation")
    b = truth_gate(wrong, path, n=60_000)
    assert not b["ok"] and any("shape mismatch" in r for r in b["reasons"])
