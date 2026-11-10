"""reco_field.exact_check.mesh_residual: the meshes' exact difference (manifold3d) with the keys and pieces of
exact_residual - no FreeCAD needed; the solids come from manifold3d itself."""

import json

import numpy as np
import pytest

m = pytest.importorskip("manifold3d")
trimesh = pytest.importorskip("trimesh")

from reco_field.exact_check import mesh_residual


def write(solid, path):
    mesh = solid.to_mesh()
    trimesh.Trimesh(np.asarray(mesh.vert_properties, float)[:, :3], np.asarray(mesh.tri_verts, np.int64), process=False).export(str(path))


def test_box_with_bore_against_its_cover(tmp_path):
    # a blind pocket from the side (x) that reaches neither slab: the three excess pieces are disjoint
    pocket = m.Manifold.cylinder(14.0, 4.0, 4.0, circular_segments=64, center=True).rotate((0.0, 90.0, 0.0)).translate((5.0, 0.0, 0.0))
    part = m.Manifold.cube((20.0, 20.0, 20.0), center=True) - pocket
    model = m.Manifold.cube((20.0, 20.0, 22.0), center=True)             # a 1 mm slab too thick on both sides, no pocket
    write(part, tmp_path / "part.stl")
    write(model, tmp_path / "model.stl")
    r = mesh_residual(tmp_path / "part.stl", tmp_path / "model.stl", pieces_dir=tmp_path / "pieces", max_pieces=10)
    assert "error" not in r and r["method"] == "mesh"
    v_bore = np.pi * 16 * 12                                               # the pocket's part inside the cube: x in [-2, 10]
    assert r["input_volume"] == pytest.approx(8000 - v_bore, rel=2e-3) and r["model_volume"] == pytest.approx(8800)
    assert r["r_plus"] == pytest.approx(0.0, abs=1e-6)                     # the model holds the part
    assert r["r_minus"] == pytest.approx(800 + v_bore, rel=2e-3)            # two slabs and the bore
    assert r["iou"] == pytest.approx(r["input_volume"] / (r["input_volume"] + r["r_minus"]))
    assert r["r_plus_solids"] == 0 and r["r_minus_solids"] == 3 and r["model_genus"] == 0
    pieces = r["pieces"]
    assert len(pieces) == 3 and all(p["sign"] == "minus" and p["step"] is None and p["faces"] == [] for p in pieces)
    assert pieces[0]["volume"] >= pieces[1]["volume"] >= pieces[2]["volume"]
    bore = min(pieces, key=lambda p: abs(p["centre"][2]))
    assert bore["volume"] == pytest.approx(v_bore, rel=2e-3) and np.allclose(bore["centre"], [4.0, 0.0, 0.0], atol=1e-6)
    assert (tmp_path / "pieces" / "pieces.json").is_file() and all((tmp_path / "pieces" / f"{p['id']}.stl").is_file() for p in pieces)
    back = trimesh.load(str(tmp_path / "pieces" / (bore["id"] + ".stl")))
    assert back.is_watertight and back.volume == pytest.approx(v_bore, rel=2e-3)


def test_open_mesh_is_an_error_not_a_crash(tmp_path):
    part = m.Manifold.cube((2.0, 2.0, 2.0))
    write(part, tmp_path / "part.stl")
    open_mesh = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], process=False)
    open_mesh.export(str(tmp_path / "open.stl"))
    r = mesh_residual(tmp_path / "part.stl", tmp_path / "open.stl")
    assert "error" in r and "closed solid" in r["error"]
