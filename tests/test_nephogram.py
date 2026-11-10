"""reco_gui/nephogram.py: the comparison export - distances both ways, HSV colours, the PLY scalar field, the files."""

from dataclasses import dataclass

import numpy as np
from pathlib import Path

import pytest

trimesh = pytest.importorskip("trimesh")

from reco_gui.nephogram import cloudcompare_command, hsv_colours, write_nephogram


@dataclass
class M:
    vertices: np.ndarray
    triangles: np.ndarray


def box(size, centre=(0, 0, 0)):
    b = trimesh.creation.box(size)
    b.apply_translation(centre)
    return M(np.asarray(b.vertices, float), np.asarray(b.faces, np.int64))


def test_colours_run_from_blue_to_red():
    c = hsv_colours(np.array([0.0, 0.5, 1.0, 2.0]), scale=1.0)
    assert c.shape == (4, 4) and (c[:, 3] == 255).all()
    assert tuple(c[0, :3]) == (0, 0, 255) and tuple(c[2, :3]) == (255, 0, 0) and tuple(c[3, :3]) == (255, 0, 0)
    assert c[1, 1] == 255 and c[1, 2] == 0                                    # the middle band is a green (band 4 of 8)


def test_export_writes_both_ways_with_the_distance_scalar(tmp_path):
    part = box((10, 10, 10))
    guess = box((10, 10, 12))                                                # 1 mm too thick on both z sides
    info = write_nephogram(part, guess, tmp_path / "cmp", name="t")
    s = info["stats"]
    assert s["input_to_model"]["max"] == pytest.approx(0.0, abs=1e-9)          # the input lies inside the guess's surface? no:
    # the input's z faces are 1 mm inside the guess: their vertices are 1 mm from the guess's surface? No - a vertex of
    # the input's top face (z = 5) is at distance 0 from the guess's SIDE faces (x = +-5): every input vertex lies on a side
    assert s["model_to_input"]["max"] == pytest.approx(1.0, abs=1e-6)         # the guess's z-face corners are 1 mm out
    assert info["scale"] > 0 and info["chamfer"] >= 0
    for f in info["files"]["stl"] + info["files"]["ply"]:
        assert (tmp_path / "cmp" / f.split("/")[-1]).is_file()
    back = trimesh.load(info["files"]["ply"][1], process=False)
    assert "distance" in back.metadata.get("_ply_raw", {}).get("vertex", {}).get("data", {}).dtype.names \
        or "distance" in getattr(back, "vertex_attributes", {})
    assert (tmp_path / "cmp" / "README.txt").read_text().startswith("t:")
    assert cloudcompare_command() is None or isinstance(cloudcompare_command(), list)
    # the histograms always come with the nephogram (the user's rule 2026-10-29): both ways, one PNG
    hist = info["files"]["histogram"]
    assert hist and Path(hist).is_file() and Path(hist).stat().st_size > 1000 and "histogram.png" in (tmp_path / "cmp" / "README.txt").read_text()


def test_freecad_macro_is_written_and_valid_python(tmp_path):
    import ast
    from reco_gui.nephogram import freecad_command
    info = write_nephogram(box((4, 4, 4)), box((4, 4, 5)), tmp_path / "cmp", name="t")
    macro = info["files"]["macro"]
    text = open(macro).read()
    ast.parse(text)                                                            # FreeCAD runs it as Python
    assert "Coloring = True" in text and "Mesh.insert(" in text and "input_to_model.ply" in text and str(tmp_path / "cmp") in text
    assert "nephogram t" in text and "two-way chamfer" in text
    assert "ImageGui.open(" in text and "histogram.png" in text                 # FreeCAD shows the histograms too
    assert freecad_command() is None or isinstance(freecad_command(), list)


def test_triangle_colours_average_the_vertices():
    from reco_gui.nephogram import triangle_colours
    tris = np.array([[0, 1, 2], [1, 2, 3]])
    c = triangle_colours(np.array([0.0, 0.0, 0.0, 3.0]), tris, scale=1.0)
    assert c.shape == (2, 4) and tuple(c[0, :3]) == (0, 0, 255) and tuple(c[1, :3]) == (255, 0, 0)   # mean 1.0 -> red


def test_discrete_bands_and_legend():
    from reco_gui.nephogram import BANDS, band_of, hsv_colours, legend
    d = np.array([0.0, 0.124, 0.126, 0.5, 0.99, 1.0, 7.0])
    k = band_of(d, scale=1.0)
    assert list(k) == [0, 0, 1, 4, 7, 7, 7] and BANDS == 8
    c = hsv_colours(d, 1.0)
    assert tuple(c[0, :3]) == tuple(c[1, :3]) == (0, 0, 255) and tuple(c[-1, :3]) == (255, 0, 0)
    assert tuple(c[2, :3]) != tuple(c[1, :3])                              # the next band is another colour
    leg = legend(1.0)
    assert len(leg) == 8 and leg[0]["text"] == "0 - 0.125 mm" and leg[-1]["text"] == "> 0.875 mm" and leg[-1]["hi"] is None
    assert leg[0]["rgb"] == [0, 0, 255] and leg[-1]["rgb"] == [255, 0, 0]


@pytest.mark.skipif(not __import__("os").path.exists("/usr/bin/freecadcmd"), reason="needs freecadcmd")
def test_exact_distances_against_the_step(tmp_path):
    """The exact B-rep route: a 10 mm box as STEP, points 1 mm off its faces and inside it; then the export with it."""
    from reco_field.exact_check import point_distances
    step = tmp_path / "box.step"
    script = tmp_path / "mk.py"
    script.write_text(f"import Part\nPart.makeBox(10, 10, 10).exportStep(r'{step}')\n")
    import subprocess
    subprocess.run(["/usr/bin/freecadcmd", str(script)], check=True, capture_output=True, timeout=120)
    pts = np.array([[5, 5, 11.0], [5, 5, 5.0], [-2, 5, 5.0], [5, 5, 10.0]])
    r = point_distances(step, pts)
    assert "error" not in r and r["faces"] == 6
    assert np.allclose(r["distance"], [1.0, 0.0, 2.0, 0.0], atol=1e-9) and list(r["inside"]) == [False, True, False, True]   # on the face counts as inside
    part = box((10, 10, 10), centre=(5, 5, 5))
    guess = box((10, 10, 12), centre=(5, 5, 5))
    info = write_nephogram(part, guess, tmp_path / "cmp", name="t", input_path=str(step), model_step=str(step))
    assert info["method"].startswith("exact B-rep")
    assert info["stats"]["model_to_input"]["max"] == pytest.approx(1.0, abs=1e-9)   # the guess's z corners are 1 mm out
    assert info["stats"]["input_to_model"]["max"] == pytest.approx(0.0, abs=1e-9)
    assert "exact B-rep" in (tmp_path / "cmp" / "README.txt").read_text()
