"""The beam test suite (scripts/make_beam_test_parts.py -> dataset_gen/examples/beams/): every part exists with a
consistent ground-truth JSON, and one real freecadcmd build (the L-corner) is one valid solid of the right volume."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BEAMS = ROOT / "dataset_gen" / "examples" / "beams"
SCRIPT = ROOT / "scripts" / "make_beam_test_parts.py"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dataset_gen"))
from src.freecad_path import find_freecadcmd  # noqa: E402  # $SHAPE_RECO_FREECADCMD, PATH, /usr/bin/freecadcmd (T-04)
os.environ.setdefault("BEAM_PARTS_NO_MAIN", "1")

import make_beam_test_parts as mb  # noqa: E402

OPS = {"ibeam": 3, "rod": 1, "bar": 1, "tube": 2, "rhs": 2}


def test_every_spec_has_its_files():
    for name in mb.spec_parts():
        for ext in ("stl", "step", "json"):
            assert (BEAMS / f"{name}.{ext}").is_file(), f"{name}.{ext} missing: run freecadcmd {SCRIPT}"


@pytest.mark.parametrize("name", sorted(mb.spec_parts()))
def test_ground_truth_consistent(name):
    gt = json.loads((BEAMS / f"{name}.json").read_text())
    assert gt["name"] == name and gt["units"] == "mm"
    assert gt["valid"] and gt["n_solids"] == 1 and gt["volume_mm3"] > 0
    lo, hi = gt["bbox_min"], gt["bbox_max"]
    assert math.isclose(gt["diagonal_mm"], math.dist(lo, hi), rel_tol=1e-6)
    n = len(gt["members"])
    assert n >= 1
    for m in gt["members"]:
        assert m["kind"] in OPS
        assert math.dist(m["start"], m["end"]) > 0
        for p in (m["start"], m["end"]):                     # nominal end points lie in the part's box (1 mm slack)
            assert all(lo[k] - 1 <= p[k] <= hi[k] + 1 for k in range(3)), (name, p)
        if m["kind"] == "ibeam":
            from src import din_profiles as dp
            env = dp.PROFILES[m["envelope"]]
            assert m["section"]["h"] == env.h and m["section"]["b"] == env.b
            if m["profile"]:
                assert m["profile"] == m["envelope"] and m["section"]["tw"] == env.tw
    for j in gt["joints"]:
        assert all(0 <= i < n for i in j["members"])
    exp = gt["expected_program"]
    total = sum(OPS[m["kind"]] for m in gt["members"]) + len(gt["features"])
    assert exp["total"] == total == len(exp["ops"]) == exp["adds"] + exp["cuts"]
    # the solid's volume is at most the sum of the members' prismatic volumes (plus features), overlaps only remove
    assert gt["volume_mm3"] > 0.5 * _nominal_volume(gt)


def _section_area(m):
    s = m["section"]
    if m["kind"] == "ibeam":
        return 2 * s["b"] * s["tf"] + (s["h"] - 2 * s["tf"]) * s["tw"] + (4 - math.pi) * s["r"] ** 2
    if m["kind"] == "rod":
        return math.pi * s["d"] ** 2 / 4
    if m["kind"] == "tube":
        return math.pi * (s["d"] ** 2 - (s["d"] - 2 * s["t"]) ** 2) / 4
    if m["kind"] == "bar":
        return s["w"] * s["t"]
    return s["h"] * s["b"] - (s["h"] - 2 * s["t"]) * (s["b"] - 2 * s["t"])


def _nominal_volume(gt):
    return sum(_section_area(m) * math.dist(m["start"], m["end"]) for m in gt["members"])


def test_single_members_volume_matches_section():
    for name in ("s01_ipe200", "s02_hea500_rot", "s03_rod", "s04_tube", "s05_rhs", "s06_flatbar"):
        gt = json.loads((BEAMS / f"{name}.json").read_text())
        assert gt["volume_mm3"] == pytest.approx(_nominal_volume(gt), rel=0.01), name


@pytest.mark.skipif(find_freecadcmd() is None, reason="freecadcmd not installed")
def test_freecad_build_l_corner(tmp_path):
    env = dict(os.environ, BEAM_PARTS_ONLY="j01_l_butt", BEAM_PARTS_OUT=str(tmp_path))
    env.pop("BEAM_PARTS_NO_MAIN", None)
    subprocess.run([find_freecadcmd(), str(SCRIPT)], env=env, capture_output=True, text=True, timeout=300, check=True)
    status = json.loads((tmp_path / "_build_status.json").read_text())["j01_l_butt"]
    assert status["ok"] and status["valid"] and status["n_solids"] == 1
    gt = json.loads((tmp_path / "j01_l_butt.json").read_text())
    # two HEA 300 (A = 112.5 cm^2): post 2000 (+1 mm into the girder, overlapping), girder 2145 mm
    a = 112.5e2
    assert status["volume_mm3"] == pytest.approx(a * (2000 + 2145), rel=0.01)
    assert gt["volume_mm3"] == pytest.approx(status["volume_mm3"])
    assert (tmp_path / "j01_l_butt.stl").stat().st_size > 0 and (tmp_path / "j01_l_butt.step").stat().st_size > 0
