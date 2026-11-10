"""scripts/run_part.py ended from outside and from within (2026-10-15): a TERM leaves a record that says "killed" and
the stage; --reconstruction-timeout ends the loop with the model it has and the run still builds and measures it;
--compare-timeout turns a slow sampled comparison into a recorded error. Needs freecadcmd; about a minute."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PART = ROOT / "dataset_gen" / "examples" / "box_with_bore.stl"
pytestmark = pytest.mark.skipif(not (Path("/usr/bin/freecadcmd").exists() or os.environ.get("SHAPE_RECO_FREECADCMD")),
                                reason="needs freecadcmd")


def run(out, *extra, wait=True):
    cmd = [sys.executable, str(ROOT / "scripts" / "run_part.py"), str(PART), "--stl", str(PART), "--out", str(out),
           "--no-features", "--no-polish", "--points", "20000", "--resolution", "48", "--max-resolution", "48", *extra]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", QT_QPA_PLATFORM="offscreen")
    if wait:
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)


def record(out):
    return json.loads((out / "record.json").read_text())


def test_term_from_outside_is_recorded_as_killed_with_its_stage(tmp_path):
    out = tmp_path / "killed"
    p = run(out, wait=False)
    deadline = time.time() + 120
    while time.time() < deadline and not (out / "record.json").is_file():
        time.sleep(0.5)
    while time.time() < deadline and record(out)["stage"] in ("start", "input facts"):
        time.sleep(0.5)
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=60)
    r = record(out)
    assert rc == 128 + signal.SIGTERM and r["state"] == "killed" and "stage" in r["error"] and r["stage"] in r["error"]
    assert r["seconds"].get("all") is not None


def test_reconstruction_timeout_keeps_the_partial_model_and_compare_timeout_is_an_error(tmp_path):
    out = tmp_path / "timeouts"
    proc = run(out, "--reconstruction-timeout", "2", "--compare-timeout", "1")
    r = record(out)
    assert r["state"] == "ended", (r.get("error"), proc.stderr[-800:])
    assert r["run"]["timed_out"] is True and r["run"]["stopped"].startswith("reconstruction timeout (2 s)")
    lines = [json.loads(x) for x in (out / "steps.jsonl").read_text().splitlines()]
    assert lines[-1]["kind"] == "timeout"
    if r["model"].get("operations"):                              # a partial program: built, exact and sampled stages ran
        assert r["stage"] == "done" and "build" in r["seconds"] and "sampled comparison" in r["seconds"]
        sampled = r["comparison"]["sampled"]
        assert sampled.get("iou") is not None or "no answer within 1 s" in sampled.get("error", "")
    else:
        assert r["stage"] == "no model"
