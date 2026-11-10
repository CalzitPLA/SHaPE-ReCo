"""reco_engine/symmetry.py (the user's rule 2026-10-29: "always check if symmetrical features exist"): the mirror planes
and rotation axes of a part, found on off-centre asymmetric shapes, and the exact mirror of a placed primitive (a sector,
a composite) with a proper frame."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine import symmetry as sy  # noqa: E402
from reco_engine.result_schema import Primitive  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp, quaternion_of, rotation_matrix  # noqa: E402

T = np.array([40.0, -25.0, 12.0])                       # off-centre on every axis


def _names(sym):
    return sorted(sy._axis_name(p["normal"]) for p in sym["planes"])


def test_a_box_has_three_mirror_planes_and_an_off_centre_bore_breaks_one():
    box = trimesh.creation.box((40.0, 20.0, 10.0))
    box.apply_translation(T)
    sym = sy.detect(box.vertices, box.faces)
    assert _names(sym) == ["x", "y", "z"] and np.allclose(sym["centre"], T, atol=1e-6)
    assert {sy._axis_name(a["axis"]): a["order"] for a in sym["axes"]} == {"x": 2, "y": 2, "z": 2}
    pytest.importorskip("manifold3d")
    bore = trimesh.creation.cylinder(radius=3.0, height=30.0)
    bore.apply_translation(T + [12.0, 0.0, 0.0])              # off-centre along x, centred in y
    holed = box.difference(bore)
    sym = sy.detect(holed.vertices, holed.faces)
    assert _names(sym) == ["y", "z"]                         # the x mirror is gone, y and z stay
    assert "mirror planes y, z" in sy.describe(sym) and sy.describe(None) == "none"


def _sector(R=None):
    R = np.eye(3) if R is None else R
    return Primitive(kind="cylinder", quaternion=quaternion_of(R), center_mm=tuple(T + [5.0, 2.0, 0.0]),
                     dims_mm=dict(radius=4.0, half_length=3.0), u=(0.2, 1.9))


def _check_mirror(p, normal, point):
    q = sy.mirror_primitive(p, normal, point)
    M, t = sy.reflection(normal, point)
    a, b = primitive_to_sp(p), primitive_to_sp(q)
    P = np.random.default_rng(0).uniform(-20, 20, (20000, 3)) + T
    assert float(np.abs(np.asarray(b.sdf(P)) - np.asarray(a.sdf(P @ M.T + t))).max()) < 1e-9
    for r in [q] + list(q.parts or []):
        assert abs(float(np.linalg.det(rotation_matrix(r))) - 1.0) < 1e-9     # a proper frame FreeCAD can place
    return q


def test_the_mirror_of_a_sector_and_of_a_composite_is_exact():
    tilt = sy.rotation([1.0, 2.0, 0.5], [0, 0, 0], 0.7)[0]
    q = _check_mirror(_sector(tilt), [1.0, 0.0, 0.0], T)
    assert q.u == pytest.approx((math.pi - 1.9, math.pi - 0.2))
    box = Primitive(kind="box", quaternion=quaternion_of(tilt), center_mm=tuple(T + [6.0, -3.0, 1.0]),
                    dims_mm=dict(x_half=2.0, y_half=1.0, z_half=0.5))
    comp = Primitive(kind="composite", center_mm=box.center_mm, parts=[box, _sector()], combine="cut")
    _check_mirror(comp, [0.3, 1.0, 0.2], T + [0.0, 1.0, 0.0])


def test_partner_transforms_find_the_symmetry_that_maps_one_centre_onto_another():
    box = trimesh.creation.box((40.0, 20.0, 10.0))
    box.apply_translation(T)
    sym = sy.detect(box.vertices, box.faces)
    src = T + [15.0, -7.0, 0.0]
    names = [n for n, _ in sy.partner_transforms(sym, src, T + [-15.0, -7.0, 0.0], 1e-6)]
    assert names and names[0] == "mirror x"
    turned = [n for n, _ in sy.partner_transforms(sym, src, T + [-15.0, 7.0, 0.0], 1e-6)]
    assert any(n.startswith("C2") for n in turned) or "mirror x + mirror y" in turned
    assert sy.partner_transforms(sym, src, T + [3.0, 3.0, 3.0], 1e-6) == [] and sy.partner_transforms(None, src, src, 1) == []
