"""
The rough cover by sub-parts (reco_engine/subpart_cover.py) on parts whose cover is known: a stepped shaft (two
cylinders) and an angle (two boxes), tilted and off-centre, their exact faces written by hand.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("igl")

from reco_engine import subpart_cover as sc  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

R = Rotation.from_euler("zxy", [35.0, -20.0, 110.0], degrees=True).as_matrix()
T = np.array([40.0, -25.0, 310.0])


def placed(mesh, local):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, T + R @ np.asarray(local, float)
    mesh.apply_transform(M)
    return mesh


def joined(meshes):
    V = np.vstack([m.vertices for m in meshes])
    off = np.cumsum([0] + [len(m.vertices) for m in meshes[:-1]])
    return V, np.vstack([m.faces + o for m, o in zip(meshes, off)])


def plane(normal_local, point_local, box):
    return dict(type="plane", area=1.0, normal=list(R @ np.asarray(normal_local, float)),
                point=list(T + R @ np.asarray(point_local, float)), bbox=box)


def shaft():
    """R 10 from z = 0 to 30, R 5 from 30 to 50, about the local z axis. The mesh: the two cylinders as they are
    (their touching discs are inside the part; a surface sample there lies inside the cover, which is what counts)."""
    V, F = joined([placed(trimesh.creation.cylinder(radius=10.0, height=30.0, sections=128), (0, 0, 15.0)),
                   placed(trimesh.creation.cylinder(radius=5.0, height=20.0, sections=128), (0, 0, 40.0))])
    box = [list(V.min(axis=0)), list(V.max(axis=0))]
    faces = [dict(type="cylinder", area=2 * math.pi * 10 * 30, axis=list(R[:, 2]), center=list(T), radius=10.0, bbox=box),
             dict(type="cylinder", area=2 * math.pi * 5 * 20, axis=list(R[:, 2]), center=list(T), radius=5.0, bbox=box)]
    for z, area in ((0.0, 314.0), (30.0, 235.0), (50.0, 78.5)):
        faces.append(dict(plane((0, 0, 1.0 if z else -1.0), (0, 0, z), box), area=area))
    return V, F, faces


def angle():
    """An angle: a plate 60 x 40 x 10 and a wall 10 x 40 x 50 standing on its end."""
    V, F = joined([placed(trimesh.creation.box(extents=(60.0, 40.0, 10.0)), (30.0, 20.0, 5.0)),
                   placed(trimesh.creation.box(extents=(10.0, 40.0, 50.0)), (5.0, 20.0, 35.0))])
    box = [list(V.min(axis=0)), list(V.max(axis=0))]
    faces = [dict(plane(n, p, box), area=a) for n, p, a in (
        ((0, 0, -1), (30, 20, 0), 2400.0), ((0, 0, 1), (35, 20, 10), 2000.0), ((0, 0, 1), (5, 20, 60), 400.0),
        ((-1, 0, 0), (0, 20, 30), 2400.0), ((1, 0, 0), (60, 20, 5), 400.0), ((1, 0, 0), (10, 20, 35), 2000.0),
        ((0, -1, 0), (20, 0, 20), 1100.0), ((0, 1, 0), (20, 40, 20), 1100.0))]
    return V, F, faces


def test_directions_come_from_the_faces():
    V, F, faces = shaft()
    d = sc.directions(faces, V)
    assert len(d["axes"]) == 1 and abs(d["axes"][0][0] @ R[:, 2]) == pytest.approx(1.0)
    assert abs(d["frame"][:, 2] @ R[:, 2]) == pytest.approx(1.0)
    assert len(d["planes"]) == 3


def test_a_stepped_shaft_is_two_cylinders():
    V, F, faces = shaft()
    c = sc.cover(V, F, faces)
    assert [p.kind for p in c["primitives"]] == ["cylinder", "cylinder"]
    assert sorted(round(p.dims_mm["radius"], 3) for p in c["primitives"]) == [5.0, 10.0]
    assert sorted(round(2 * p.dims_mm["half_length"], 3) for p in c["primitives"]) == [20.0, 30.0]
    assert c["containment"] == 1.0
    assert c["history"][0]["parts"] == 1 and c["history"][0]["exact"] < 0.8      # one cylinder R 10 x 50: the step is missing
    assert c["exact"] > 0.9                                                       # the mesh's inner discs are the rest
    assert len(c["history"]) == 2                                                 # no further split pays


def test_an_angle_is_two_boxes():
    V, F, faces = angle()
    c = sc.cover(V, F, faces)
    assert [p.kind for p in c["primitives"]] == ["box", "box"]
    volumes = sorted(round(sc.volume(p)) for p in c["primitives"])
    assert volumes in ([20000, 24000], [16000, 24000], [20000, 20000])           # plate and wall, by either split
    assert c["containment"] == 1.0 and c["exact"] > 0.9
    assert c["volume"] < 0.5 * 60 * 40 * 60                                       # the single box is 144 000


def test_a_plain_box_stays_one_box():
    V, F = joined([placed(trimesh.creation.box(extents=(60.0, 40.0, 10.0)), (30.0, 20.0, 5.0))])
    box = [list(V.min(axis=0)), list(V.max(axis=0))]
    faces = [dict(plane(n, p, box), area=a) for n, p, a in (
        ((0, 0, -1), (30, 20, 0), 2400.0), ((0, 0, 1), (30, 20, 10), 2400.0), ((-1, 0, 0), (0, 20, 5), 400.0),
        ((1, 0, 0), (60, 20, 5), 400.0), ((0, -1, 0), (30, 0, 5), 600.0), ((0, 1, 0), (30, 40, 5), 600.0))]
    c = sc.cover(V, F, faces)
    assert [p.kind for p in c["primitives"]] == ["box"] and c["exact"] == pytest.approx(1.0, abs=0.01)
    assert c["containment"] == 1.0


def test_every_type_has_its_space_a_ball_a_cone_a_ring():
    """A part that is a ball, a cone frustum, a ring: the cover is found in that type's space."""
    ball = placed(trimesh.creation.icosphere(subdivisions=4, radius=12.0), (3.0, -2.0, 5.0))
    V, F = joined([ball])
    box = [list(V.min(axis=0)), list(V.max(axis=0))]
    faces = [dict(type="sphere", area=4 * math.pi * 144, center=list(T + R @ np.array([3.0, -2.0, 5.0])), radius=12.0, bbox=box)]
    c = sc.cover(V, F, faces)
    assert [p.kind for p in c["primitives"]] == ["sphere"] and c["primitives"][0].dims_mm["radius"] == pytest.approx(12.0, abs=1e-6)
    assert c["exact"] > 0.98 and c["containment"] == 1.0

    cone = placed(trimesh.creation.cone(radius=10.0, height=20.0, sections=256), (0.0, 0.0, 0.0))   # base at z 0, apex at 20
    V, F = joined([cone])
    box = [list(V.min(axis=0)), list(V.max(axis=0))]
    semi = math.atan2(10.0, 20.0)
    faces = [dict(type="cone", area=math.pi * 10 * math.hypot(10, 20), axis=list(-R[:, 2]), center=list(T + 20.0 * R[:, 2]),
                  apex=list(T + 20.0 * R[:, 2]), semi_angle=semi, radius=0.0, point=list(T + R @ np.array([5.0, 0.0, 10.0])), bbox=box),
             dict(plane((0, 0, -1), (0, 0, 0), box), area=math.pi * 100)]
    c = sc.cover(V, F, faces)
    assert [p.kind for p in c["primitives"]] == ["cone"]
    d = c["primitives"][0].dims_mm
    assert sorted([round(d["radius1"], 2), round(d["radius2"], 2)]) == [0.0, 10.0] and 2 * d["half_length"] == pytest.approx(20.0, abs=0.01)
    assert c["exact"] > 0.95 and c["containment"] == 1.0

    ring = placed(trimesh.creation.torus(major_radius=20.0, minor_radius=4.0, major_sections=128, minor_sections=64), (0.0, 0.0, 0.0))
    V, F = joined([ring])
    box = [list(V.min(axis=0)), list(V.max(axis=0))]
    faces = [dict(type="torus", area=4 * math.pi ** 2 * 80, axis=list(R[:, 2]), center=list(T), major_radius=20.0,
                  minor_radius=4.0, bbox=box)]
    c = sc.cover(V, F, faces)
    assert [p.kind for p in c["primitives"]] == ["torus"]
    assert c["primitives"][0].dims_mm["minor_radius"] == pytest.approx(4.0, abs=0.01)      # the mesh is a polygon about the axis
    assert c["exact"] > 0.95 and c["containment"] == 1.0
