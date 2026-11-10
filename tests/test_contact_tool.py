"""The contact-face guess source (feature_stage.contact_tool): a block with a bore and a filleted pocket reaching out
from the bore, asymmetric, off-centre and turned out of the axes; the model is the block with the bore alone, so the
piece of the exact difference is the pocket, closed towards the bore by the model's cylinder. Built and checked in
freecadcmd (exact)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

pytest.importorskip("igl")

from reco_engine import feature_stage as fs  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp  # noqa: E402
from reco_field import exact_check as ec  # noqa: E402
from reco_field.export import find_freecadcmd, program_to_dict  # noqa: E402

pytestmark = pytest.mark.skipif(find_freecadcmd() is None, reason="freecadcmd not found")

PART = r"""
import json, os
import FreeCAD as App, Part
from FreeCAD import Vector as V
job = json.load(open(os.environ["PART_JOB"]))
block = Part.makeBox(120, 90, 90, V(10, -40, 5))
bore = Part.makeCylinder(25, 120, V(60, 0, -10), V(0, 0, 1))
tool = Part.makeBox(30, 20, 70, V(45, -35, 30))                     # the pocket: floor y = -35, open to the bore and +z
tool = tool.makeFillet(10.0, [e for e in tool.Edges if abs(e.CenterOfMass.y + 35) < 1e-6
                             and abs(abs(e.tangentAt(e.FirstParameter).z) - 1) < 1e-6])
tool = tool.makeFillet(4.0, [e for e in tool.Edges if abs(e.BoundBox.ZMax - 30) < 1e-6 and e.BoundBox.YMin < -15 - 1e-6])
model = block.cut(bore)
part = model.cut(tool)
place = App.Placement(V(7, -3, 11), App.Rotation(V(1, 2, 3), 25))
for s, name in ((part, "input"), (model, "model")):
    s = s.removeSplitter()
    s.Placement = place
    assert s.isValid() and len(s.Solids) == 1
    s.exportStep(os.path.join(job["dir"], name + ".step"))
    if name == "input":
        import Mesh, MeshPart
        m = MeshPart.meshFromShape(Shape=s, LinearDeflection=0.01, AngularDeflection=0.05)
        m.write(os.path.join(job["dir"], "input.stl"))
json.dump(dict(pocket=model.common(tool).Volume, part=part.Volume), open(os.path.join(job["dir"], "volumes.json"), "w"))
"""


@pytest.fixture(scope="module")
def pocket(tmp_path_factory):
    d = tmp_path_factory.mktemp("contact_tool")
    (d / "part.py").write_text(PART)
    (d / "job.json").write_text(json.dumps(dict(dir=str(d))))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["PART_JOB"] = str(d / "job.json")
    proc = subprocess.run([find_freecadcmd(), str(d / "part.py")], env=env, capture_output=True, text=True, timeout=300)
    assert (d / "volumes.json").is_file(), (proc.stderr or proc.stdout)[-400:]
    volumes = json.loads((d / "volumes.json").read_text())
    r = ec.exact_residual(d / "input.step", d / "model.step", pieces_dir=d / "pieces")
    assert "error" not in r, r
    pieces = sorted(r["pieces"], key=lambda p: -float(p["volume"]))
    from src.closest_point_field import load_field
    fld = load_field(pieces[0]["stl"])
    return dict(dir=d, piece=pieces[0], pieces=pieces, volumes=volumes, V=np.asarray(fld.vertices, float),
                F=np.asarray(fld.faces))


def _check(pocket, prims):
    items = [dict(id=k, op="cut", program=program_to_dict(primitive_to_sp(p)), piece_step=pocket["piece"]["step"])
             for k, p in prims.items()]
    r = ec.feature_checks(pocket["dir"] / "input.step", items)
    assert "error" not in r, r
    return r


def test_the_piece_is_the_pocket_with_contact_and_tool_only_faces(pocket):
    p = pocket["piece"]
    assert len(pocket["pieces"]) == 1 and p["sign"] == "minus"
    assert p["volume"] == pytest.approx(pocket["volumes"]["pocket"], rel=1e-6)
    origins = {f["origin"] for f in p["faces"]}
    assert "model" in origins and origins & set(fs.REAL)
    assert {f["type"] for f in p["faces"] if f["origin"] in fs.REAL} >= {"plane", "cylinder", "torus"}


def test_the_contact_tool_cuts_nothing_of_the_input_and_recalls_the_piece(pocket):
    p = pocket["piece"]
    tool = fs.contact_tool(p, pocket["V"])
    assert tool is not None and tool.label.startswith("contact faces")
    assert "2 corner(s)" in tool.label and "open" in tool.label and "nowhere" not in tool.label
    box = fs._box_of_planes([f for f in p["faces"] if f["type"] == "plane"], pocket["V"])
    r = _check(pocket, dict(contact=tool, box=box))
    v = float(p["volume"])
    assert "error" not in r["contact"], r["contact"]
    assert r["contact"]["harm_volume"] <= 1e-6 * v                   # nothing of the input is cut
    assert r["contact"]["left_of_piece_volume"] <= 1e-6 * v          # the whole piece is removed
    # OCCT's common of the tangent surfaces: measured 19654.66 of 19655.30 (3e-5) with piece - tool = 0
    assert r["contact"]["common_volume"] == pytest.approx(v, rel=1e-4)
    assert r["contact"]["beyond_piece_volume"] > 0.01 * v            # open: it runs on past the piece
    assert r["box"]["harm_volume"] > 1e-3 * v                        # the whole piece as a sharp box cuts the fillets


def test_the_open_sides_are_those_the_model_closes(pocket):
    """Closed where a contact face is (the floor, both walls, the rounded end), open towards the bore and the end
    the model's plane closes: the box passes the piece's extent there and nowhere else."""
    from reco_engine.result_to_program import rotation_matrix
    tool = fs.contact_tool(pocket["piece"], pocket["V"])
    box = fs.leaves(tool)[0]
    R, c = rotation_matrix(box), np.asarray(box.center_mm, float)
    h = np.array([box.dims_mm[k] for k in ("x_half", "y_half", "z_half")])
    local = (pocket["V"] - c) @ R
    past_hi, past_lo = h - local.max(axis=0), local.min(axis=0) + h
    past = np.concatenate([past_hi, past_lo])
    assert (past > -1e-4).all()                                   # the STL vertices are float32
    assert int((past > 1e-3).sum()) == 2 and int((np.abs(past) <= 1e-3).sum()) == 4


def test_no_tool_without_tool_only_or_contact_faces(pocket):
    p = pocket["piece"]
    assert fs.contact_tool(dict(p, faces=[dict(f, origin="input") for f in p["faces"]]), pocket["V"]) is None
    assert fs.contact_tool(dict(p, faces=[dict(f, origin="model") for f in p["faces"]]), pocket["V"]) is None
    assert fs.contact_tool(dict(p, faces=[]), pocket["V"]) is None


def test_the_contact_tool_is_offered_first_and_judged_by_the_gate(pocket):
    from src.closest_point_field import load_field
    inp = load_field(str(pocket["dir"] / "input.stl"))
    inside = fs.input_inside(np.asarray(inp.vertices, float), np.asarray(inp.faces))
    tools = dict(min_score=0.9, min_precision=0.999, max_operands=9, max_candidates=12)
    g = fs.guess_candidates(pocket["piece"], sources=("step faces",), op="cut", inside_input=inside, tools=tools,
                            contact_faces=True)
    best = g["candidates"][0]
    assert best.source == "tool (contact faces)" and len(g["candidates"]) == 1     # passing, it ends the search
    assert best.recall >= 0.99 and fs.precise_enough(best, 0.999)
    # a gate it cannot pass: it stays a candidate, the whole-piece sources are asked after it
    g = fs.guess_candidates(pocket["piece"], sources=("step faces",), op="cut", inside_input=inside,
                            tools=dict(tools, min_score=1.1), contact_faces=True)
    sources = [c.source for c in g["candidates"]]
    assert "tool (contact faces)" in sources and "step faces" in sources
    off = fs.guess_candidates(pocket["piece"], sources=("step faces",), op="cut", inside_input=inside, tools=tools)
    assert all(c.source != "tool (contact faces)" for c in off["candidates"])
