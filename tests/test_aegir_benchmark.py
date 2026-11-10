"""
The aegir step (scripts/aegir_benchmark.py) without aegir: the geometry it writes has the names and the frame aegir's
workflows expect, the command names the workflow and the geometry's root, and the summary leaves the warm-up out.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))
sys.path.insert(0, str(SANDBOX_ROOT / "scripts"))

import aegir_benchmark as ab  # noqa: E402

OPS = [dict(op="add", primitive=dict(kind="box", quaternion=[0, 0, 0, 1], center_mm=[0.0, -96.0, 808.35],
                                     dims_mm=dict(x_half=20.0, y_half=15.0, z_half=10.0))),
       dict(op="cut", kind="cylinder", quaternion=[0, 0, 0, 1], center_mm=[5.0, -96.0, 808.35],
            dims_mm=dict(radius=4.0, half_length=12.0))]                     # the second in the frame report's flat form


def test_dry_run_writes_the_geometry_aegir_expects(tmp_path, capsys):
    ops = tmp_path / "part.json"
    ops.write_text(json.dumps(OPS))
    assert ab.main([str(ops), "--part", "liner", "--out", str(tmp_path / "out"), "--dry-run", "--events", "1000"]) == 0
    text = capsys.readouterr().out
    gdml = tmp_path / "out" / "geometry" / "sandbox" / "liner_csg_part.gdml"
    assert gdml.is_file() and "1 booleans" in text
    assert f"--ext-str root={tmp_path / 'out'}" in text and "--ext-str part=liner" in text
    assert "--ext-str geometry=liner_csg_part" in text and "gun_st_sandbox_scan_bench.jsonnet" in text and "-j 1" in text
    root = ET.parse(gdml).getroot()
    assert [v.get("name") for v in root.find("structure").findall("volume")] == ["PartVolume", "World"]
    assert root.find("structure").findall("volume")[1].find("physvol").get("name") == "PartPhys"
    box = [e for e in root.find("solids") if e.get("name").endswith("world_box")][0]
    # the box 40 x 30 x 20 about the origin; world x, y, z = source z, x, y: half sizes 10, 20, 15, plus 1000 of air
    assert [float(box.get(k)) for k in "xyz"] == pytest.approx([2 * 1010.0, 2 * 1020.0, 2 * 1015.0])


def test_the_threaded_workflow():
    c = ab.scan_command(Path("/a"), Path("/o"), "frame", "frame_csg", 5000, 4)
    assert "gun_mt_sandbox_scan_bench.jsonnet" in c and "--ext-str concurrency=4" in c and c.endswith("-j 4")


def test_summary_leaves_the_warm_up_out():
    rounds = [dict(seconds=9.0, exit_code=0, loadavg=[1.0]), dict(seconds=2.0, exit_code=0, loadavg=[3.0]),
              dict(seconds=4.0, exit_code=0, loadavg=[2.0]), dict(seconds=50.0, exit_code=1, loadavg=[9.0])]
    s = ab.summary(rounds, 6000)
    assert s["rounds_counted"] == 2 and s["mean_seconds"] == 3.0 and s["events_per_second"] == 2000.0
    assert s["max_load"] == 9.0
    assert ab.summary(rounds[:1], 6000) == dict(rounds_counted=0)


def test_a_run_stored_with_cut_operands_is_refused(tmp_path):
    ops = tmp_path / "old.json"
    ops.write_text(json.dumps([dict(op="add", primitive=dict(kind="composite", quaternion=[0, 0, 0, 1], center_mm=[0, 0, 0],
                                                            dims_mm={}, combine="cut", parts=["Primitive(kind='box' ..."]))]))
    with pytest.raises(ValueError, match=r"cut 1 node\(s\) of this program down to text"):
        ab.load_result(ops)
