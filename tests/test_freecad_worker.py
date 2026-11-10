"""reco_field/freecad_worker.py: the persistent freecadcmd gives the same result as one process per call (environment,
working directory, exit code), and without SHAPE_RECO_FREECAD_WORKER it is a plain subprocess call."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from reco_field import freecad_worker as fw  # noqa: E402

sys.path.insert(0, str(ROOT / "dataset_gen"))
from src.freecad_path import find_freecadcmd  # noqa: E402

FREECAD = find_freecadcmd()  # $SHAPE_RECO_FREECADCMD, PATH, /usr/bin/freecadcmd (T-04)
pytestmark = pytest.mark.skipif(FREECAD is None, reason="freecadcmd not installed")

SCRIPT = """import os, Part, FreeCAD
FreeCAD.newDocument("x")
open(os.environ["OUT"], "w").write("%.1f %s %d" % (Part.makeBox(float(os.environ["EDGE"]), 1, 1).Volume, os.getcwd(),
                                                  len(FreeCAD.listDocuments())))
if os.environ.get("FAIL"):
    raise SystemExit(3)
"""


@pytest.mark.parametrize("mode", ["0", "1"])
def test_jobs_see_their_own_environment_cwd_and_a_clean_document_list(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("SHAPE_RECO_FREECAD_WORKER", mode)
    script = tmp_path / "job.py"
    script.write_text(SCRIPT)
    for edge in (1, 2, 3):
        out = tmp_path / f"o{edge}.txt"
        rc, _ = fw.run_script(FREECAD, script, {"EDGE": str(edge), "OUT": str(out), "HOME": str(tmp_path)}, cwd=str(tmp_path))
        assert rc == 0 and out.read_text() == f"{edge:.1f} {tmp_path} 1"      # documents of earlier jobs are closed
    rc, _ = fw.run_script(FREECAD, script, {"EDGE": "1", "OUT": str(tmp_path / "f.txt"), "FAIL": "1", "HOME": str(tmp_path)})
    assert rc == 3
