"""reco_field/freecad_builder.py writes STEP files that contain the solid (B-002), and refuses results that are not a
solid (B-029: nothing left after the booleans; B-034: a zero-width sector). The STEP is re-read by FreeCAD itself:
checking `is_file()` let an empty STEP (Part.export of a TopoShape, 1640 bytes, no geometry) pass for months."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.export import build_in_freecad, build_program, program_to_dict  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402
from src.closest_point_field import find_freecadcmd  # noqa: E402
from src.sdf_primitives import deg  # noqa: E402

needs_freecad = pytest.mark.skipif(find_freecadcmd() is None, reason="freecadcmd not found")

REREAD = """
import json, os, Part
s = Part.read(os.environ["STEP_IN"])
json.dump(dict(volume=s.Volume, solids=len(s.Solids), valid=bool(s.isValid())), open(os.environ["OUT_JSON"], "w"))
"""


def reread_step(step: Path, tmp_path: Path) -> dict:
    script, out = tmp_path / "reread.py", tmp_path / "reread.json"
    script.write_text(REREAD)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(STEP_IN=str(step), OUT_JSON=str(out))
    subprocess.run([find_freecadcmd(), str(script)], env=env, capture_output=True, timeout=120)
    assert out.is_file(), "freecadcmd could not re-read the STEP"
    return json.loads(out.read_text())


@needs_freecad
def test_exported_step_rereads_to_the_built_solid(tmp_path):
    """An off-centre, asymmetric box minus a bore: the STEP re-reads as one valid solid with the reported volume."""
    part = sp.Placed(sp.Box((6.0, 4.0, 2.5)) - sp.Cylinder(1.2, 10.0), translation=(3.0, -1.5, 0.7))
    out = build_in_freecad(part, tmp_path, "part")
    assert out.valid and out.n_solids == 1 and out.volume > 0
    back = reread_step(out.step, tmp_path)
    assert back["solids"] == 1 and back["valid"]
    assert back["volume"] == pytest.approx(out.volume, rel=1e-6)


@needs_freecad
def test_a_program_that_leaves_no_solid_is_an_error(tmp_path):
    empty = sp.Box((2.0, 2.0, 2.0)) - sp.Box((5.0, 5.0, 5.0))
    with pytest.raises(RuntimeError, match="no solid"):
        build_in_freecad(empty, tmp_path, "empty")


@needs_freecad
def test_a_zero_width_sector_is_an_error_not_a_full_cylinder(tmp_path):
    program = program_to_dict(sp.Cylinder(3.0, 5.0, u=deg(0, 90)))
    program["u"] = [0.5, 0.5]                          # what a degenerate fit could produce; used to become 360 degrees
    with pytest.raises(RuntimeError, match="empty sector"):
        build_program(program, tmp_path, "sector")
