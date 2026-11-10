"""
The grouped build (reco_field.export.build_groups): the tools of a program as objects of a FreeCAD document, grouped by
operation, without the final boolean - in freecadcmd, on a box with a stepped hole and a rounded pocket tool.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

from reco_engine import tools as T  # noqa: E402
from reco_engine.result_schema import Primitive  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp  # noqa: E402
from reco_field.export import build_groups, find_freecadcmd  # noqa: E402

needs_freecad = pytest.mark.skipif(find_freecadcmd() is None, reason="freecadcmd not found")


def operations():
    box = Primitive(kind="box", dims_mm=dict(x_half=20.0, y_half=15.0, z_half=10.0))
    R, c = np.eye(3), np.array([3.0, -2.0, 10.0])
    hole = T._fuse_all([T._cyl(R, c, 6.0, -13.0, -6.0), T._cyl(R, c, 3.5, -20.5, -13.0)])     # 7 deep R 6, then R 3.5
    hole.label = "hole stack"
    return [("add", "box", primitive_to_sp(box)), ("cut", hole.label, primitive_to_sp(hole))]


@needs_freecad
@pytest.mark.parametrize("tools,parts", [("solid", 1), ("leaves", 2), ("primitives", 2)])
def test_groups(tmp_path, tools, parts):
    r = build_groups(operations(), tmp_path, "part", tools=tools)
    assert Path(r["fcstd"]).is_file() and Path(r["fcstd"]).stat().st_size > 1000
    assert r["groups"] == dict(adds=1, cuts=1)
    assert [o["op"] for o in r["operations"]] == ["add", "cut"] and not any("error" in o for o in r["operations"])
    if tools != "primitives":
        assert r["operations"][0]["volume"] == pytest.approx(40.0 * 30.0 * 20.0, rel=1e-9)
    assert r["operations"][1]["parts"] == parts
    if tools == "primitives":
        assert [o["parametric"] for o in r["operations"]] == [1, 2]
    if tools == "solid":
        assert r["operations"][1]["volume"] == pytest.approx(np.pi * (36.0 * 7.0 + 12.25 * 7.5), rel=1e-6)


def test_unknown_tools_mode(tmp_path):
    if find_freecadcmd() is None:
        pytest.skip("freecadcmd not found")
    with pytest.raises(ValueError):
        build_groups(operations(), tmp_path, "part", tools="whatever")


def test_the_builder_source_cut_at_its_main_call_compiles():
    """reco_field/exact_check.py's scripts take the builder's functions by cutting its source at the last "main()"."""
    src = (SANDBOX_ROOT / "reco_field" / "freecad_builder.py").read_text()
    assert src.rstrip().endswith("main()")
    cut = src[:src.rstrip().rfind("main()")]
    compile(cut, "freecad_builder", "exec")
    assert "def build(" in cut and "def main_groups(" in cut
