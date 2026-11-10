"""Gate-G3 harness (scripts/g3_run.py, scripts/g3_report.py) on a stub part: no reconstruction, no FreeCAD, fast.

The stub part is an asymmetric, off-centre L-shaped prism (not a cube), so a scale or axis mix-up shows."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import g3_report as rep  # noqa: E402
import g3_run as g3  # noqa: E402


@pytest.fixture
def lpart(tmp_path):
    """An L-shaped prism 30 x 20 x 5 mm, off-centre at (7, -3, 11)."""
    import numpy as np
    import trimesh
    xy = np.array([(0, 0), (30, 0), (30, 8), (8, 8), (8, 20), (0, 20)], float)
    v = np.vstack([np.c_[xy, np.zeros(6)], np.c_[xy, np.full(6, 5.0)]])
    fan = [(3, 4, 5), (3, 5, 0), (3, 0, 1), (3, 1, 2)]                      # star-shaped from the reflex corner
    f = [(a, c, b) for a, b, c in fan] + [(a + 6, b + 6, c + 6) for a, b, c in fan]
    for i in range(6):
        j = (i + 1) % 6
        f += [(i, j, j + 6), (i, j + 6, i + 6)]
    m = trimesh.Trimesh(v, f, process=True)
    trimesh.repair.fix_normals(m)
    assert m.is_watertight and m.volume > 0
    m.apply_translation((7, -3, 11))
    path = tmp_path / "l_stub.stl"
    m.export(path)
    return path


def _fake_result(input_path, chamfer=0.02, iou=0.97, genus=0, in_genus=0.0, rp=12.0, rm=3.0, method="occt"):
    prim = SimpleNamespace(kind="box")
    ops = [SimpleNamespace(op="add", primitive=prim, accepted=True), SimpleNamespace(op="cut", primitive=prim, accepted=True),
           SimpleNamespace(op="cut", primitive=prim, accepted=False)]
    r = SimpleNamespace(operations=ops, genus=genus, input_genus=in_genus, chamfer_mm=chamfer, exact_iou=iou,
                        exact_method=method, exact_r_plus=rp, exact_r_minus=rm, exact_r_plus_solids=1,
                        grid_pitch_over_tau=0.3, residual_voxels_pos=5, residual_voxels_neg=2)
    r.to_dict = lambda: dict(input_path=str(input_path), operations=[], step_path=None, stl_path=None)
    return r


# ------------------------------------------------------------------------------------------------ units (D-008)

def test_units_frame_table_and_assertion():
    frame = "/x/Unbenannt-Frame_4_x_6.stl"
    assert g3.mm_per_unit(frame, 67.0) == 100.0
    assert g3.mm_per_unit(frame, None) == 100.0                          # old records: the table alone
    with pytest.raises(g3.UnitError):                                    # fixed at the source -> stop, don't x100 again
        g3.mm_per_unit(frame, 6700.0)
    assert g3.mm_per_unit("/x/Frame_4_x_6.stp", None) == 1.0             # STEP is mm
    assert g3.mm_per_unit("/x/blind_hole.stl", 20.0) == 1.0
    with pytest.raises(g3.UnitError):
        g3.mm_per_unit("/x/tiny.stl", 0.2)                               # 0.2 mm: implausible, a unit error


def test_default_parts_is_the_g3_set():
    parts = g3.default_parts()
    names = [p.name for p, _ in parts]
    assert names[0] == g3.FRAME_STL.name and "box_i000000.stl" not in names
    beams = sorted((ROOT / "dataset_gen/examples/beams").glob("*.stl"))
    ex = [p for p in (ROOT / "dataset_gen/examples").glob("*.stl") if "box_i000000" not in p.name]
    assert len(parts) == 1 + len(beams) + len(ex)
    assert all(t == 1500 for p, t in parts[:1 + len(beams)]) and all(t == 600 for p, t in parts[1 + len(beams):])
    assert len({g3.part_key(p) for p, _ in parts}) == len(parts)       # record names never collide


# ------------------------------------------------------------------------------------------------ the runner

def test_summarize_converts_units_and_counts_accepted_ops(lpart):
    res = _fake_result(lpart)
    grade = dict(valid=True, n_solids=1, chamfer=0.02, chamfer_max=0.5, exact_error=None, sampled_iou=0.96)
    s = g3.summarize(res, grade, 100.0)
    assert s["n_ops"] == 2 and s["ops"] == ["add:box", "cut:box"]
    assert s["chamfer_mm"] == pytest.approx(2.0) and s["chamfer_units"] == pytest.approx(0.02)
    assert s["chamfer_max_mm"] == pytest.approx(50.0)
    assert s["r_plus_mm3"] == pytest.approx(12.0e6) and s["r_minus_mm3"] == pytest.approx(3.0e6)
    assert s["exact_iou"] == pytest.approx(0.97) and s["genus_match"] is True and s["freecad_valid"] is True
    s = g3.summarize(_fake_result(lpart, genus=None, in_genus=float("nan")), dict(valid=False), 1.0)
    assert s["genus_match"] is None and s["input_genus"] is None and s["freecad_valid"] is False


def test_run_one_writes_a_record_with_commit_and_cpu(lpart, tmp_path):
    seen = {}

    def stub(part, settings, out_dir):
        seen.update(part=part, settings=settings, out_dir=out_dir)
        sum(i * i for i in range(200000))                               # some CPU time
        return _fake_result(part), dict(valid=True, n_solids=1, chamfer=0.01, chamfer_max=0.1)

    st = dict(type_source="registry", max_resolution=128, threads=2, nice=10)
    row = g3.run_one(lpart, "t", tmp_path / "out", st, reconstruct_fn=stub, git=dict(commit="abc123", dirty=False))
    f = tmp_path / "out" / "t" / f"{g3.part_key(lpart)}.json"
    assert f.is_file() and json.loads(f.read_text())["key"] == row["key"]
    assert row["ok"] and row["schema"] == g3.SCHEMA and row["commit"] == "abc123" and row["dirty"] is False
    assert row["mm_per_unit"] == 1.0 and row["extent_mm"] == pytest.approx(30.0, rel=1e-6)
    assert row["cpu_seconds"] is not None and row["cpu_seconds"] >= 0 and row["part_sha256"]
    assert seen["settings"]["max_resolution"] == 128 and seen["out_dir"].parts[-2:] == ("freecad", g3.part_key(lpart))


def test_run_one_failure_is_a_record_not_a_crash(lpart, tmp_path):
    def boom(*_):
        raise RuntimeError("stub failure")
    row = g3.run_one(lpart, "t", tmp_path, dict(type_source="registry", max_resolution=8, threads=1, nice=0),
                     reconstruct_fn=boom, git=dict(commit=None, dirty=None))
    assert row["ok"] is False and "stub failure" in row["error"]


def test_pending_is_resumable(lpart, tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    other = tmp_path / "other.stl"
    other.write_bytes(lpart.read_bytes())
    parts = [(lpart, 600), (other, 600)]
    assert len(g3.pending(parts, run)) == 2
    (run / f"{g3.part_key(lpart)}.json").write_text(json.dumps(dict(ok=True)))
    (run / f"{g3.part_key(other)}.json").write_text(json.dumps(dict(ok=False, timeout=True)))
    assert g3.pending(parts, run) == []                                   # done and failed both skipped ...
    assert g3.pending(parts, run, retry_failed=True) == [(other, 600)]   # ... unless asked to retry the failed


# ------------------------------------------------------------------------------------------------ the report

def _write(d: Path, key: str, rec: dict):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{key}.json").write_text(json.dumps(rec))


def test_verdict_matches_s6_g3_compare():
    import s6_g3_compare as old
    for a in (None, 0.0, 1e-3, 0.02, 0.1, 1.0, 400.0):
        for b in (None, 0.0, 1e-3, 0.02, 0.1, 1.0, 400.0):
            assert rep.verdict(a, b) == old.verdict(a, b)


def test_report_old_and_new_format_units_and_macros(tmp_path):
    frame = "/x/1.0/Unbenannt-Frame_4_x_6.stl"
    base, new = tmp_path / "base", tmp_path / "new"
    # the ad-hoc night format: chamfer_mm is in file units (the frame's 0.033 = 3.3 mm)
    _write(base, "1.0__Unbenannt-Frame_4_x_6", dict(part=frame, ok=True, n_ops=32, chamfer_mm=0.033, freecad_valid=True,
                                                    seconds=90.0, result=dict(input_path=frame)))
    _write(base, "examples__a", dict(part="/x/examples/a.stl", ok=True, n_ops=2, chamfer_mm=0.01, freecad_valid=True,
                                     seconds=10.0, result=dict(exact_iou=0.99, exact_method="occt")))
    _write(base, "examples__gone", dict(part="/x/examples/gone.stl", ok=True, n_ops=1, chamfer_mm=0.01, seconds=1.0))
    # the harness format
    common = dict(schema=g3.SCHEMA, ok=True, freecad_valid=True, commit="c0ffee00aa", dirty=False, timeout=False)
    _write(new, "1.0__Unbenannt-Frame_4_x_6", dict(common, n_ops=40, chamfer_units=0.0033, chamfer_mm=0.33, exact_iou=0.9655,
                                                   exact_method="sampled (pitch 0.161)", genus=None, input_genus=17.0,
                                                   genus_match=None, mm_per_unit=100.0, cpu_seconds=200.0,
                                                   wall_seconds=120.0))
    _write(new, "examples__a", dict(common, n_ops=2, chamfer_units=0.5, chamfer_mm=0.5, exact_iou=0.90, exact_method="occt",
                                    genus=0, input_genus=0.0, genus_match=True, mm_per_unit=1.0, cpu_seconds=20.0,
                                    wall_seconds=12.0))
    _write(new, "examples__t", dict(common, ok=False, timeout=True, error="timeout after 600 s", cpu_seconds=None,
                                    wall_seconds=600.0))
    b, n = rep.load_run(base), rep.load_run(new)
    assert b["1.0__Unbenannt-Frame_4_x_6"]["chamfer_mm"] == pytest.approx(3.3)
    cmp = rep.compare(b, n)
    v = {r["part"]: (r["verdict"], r["iou_verdict"]) for r in cmp["rows"]}
    assert v["Unbenannt-Frame_4_x_6"] == ("better", "n/a")          # 3.3 -> 0.33 mm; sampled IoU gets no verdict
    assert v["a"] == ("worse", "worse") and v["gone"][0] == "missing" and v["t"][0] == "missing"
    raw = rep.compare(b, n, chamfer_unit="raw")                        # the night's rule in file units: +0.05 units
    assert {r["part"]: r["verdict"] for r in raw["rows"]}["Unbenannt-Frame_4_x_6"] == "same"
    sn = rep.summary(n)
    assert sn["timeout"] == 1 and sn["failed"] == 0 and sn["ok"] == 2 and sn["iou_occt_n"] == 1 and sn["iou_sampled_n"] == 1
    out = tmp_path / "out"
    out.mkdir()
    assert rep.main(["--base", str(base), "--new", str(new), "--json", str(out / "r.json"), "--md", str(out / "r.md"),
                     "--tex", str(out / "m.tex")]) == 0
    tex = (out / "m.tex").read_text()
    names = re.findall(r"\\newcommand\{\\([^}]*)\}", tex)
    assert names and all(re.fullmatch(r"Gthree[A-Za-z]+", x) for x in names) and len(names) == len(set(names))
    assert "\\newcommand{\\GthreeFrameIoU}{0.9655}" in tex and "\\newcommand{\\GthreeFrameChamferMm}{0.33}" in tex
    assert "\\newcommand{\\GthreeCommit}{c0ffee00}" in tex and "\\newcommand{\\GthreePartsTimeout}{1}" in tex
    assert json.loads((out / "r.json").read_text())["counts"]["better"] == 1


def test_report_reproduces_the_night_g3_counts():
    """docs/data/night_g3_eval.md: new_fix vs old_cnn {'same': 49, 'n/a': 1, 'better': 7, 'worse': 3} (file units)."""
    d = ROOT / "docs" / "data" / "night_g3"
    if not (d / "old_cnn").is_dir() or not (d / "new_fix").is_dir():
        pytest.skip("night_g3 run dirs not present")
    cmp = rep.compare(rep.load_run(d / "old_cnn"), rep.load_run(d / "new_fix"), chamfer_unit="raw")
    assert cmp["counts"] == {"same": 49, "n/a": 1, "better": 7, "worse": 3}


def test_run_one_records_the_launch_commit_and_drift(lpart, tmp_path, monkeypatch):
    """A 60-part run takes an hour; the tree can move meanwhile. The record keeps the launch commit and flags drift."""
    monkeypatch.setenv("G3_COMMIT", "launch00")
    monkeypatch.setenv("G3_DIRTY", "False")
    stub = lambda part, st, od: (_fake_result(part), dict(valid=True, chamfer=0.01))    # noqa: E731
    st = dict(type_source="registry", max_resolution=8, threads=1, nice=0)
    row = g3.run_one(lpart, "t", tmp_path, st, reconstruct_fn=stub, git=dict(commit="moved000", dirty=False))
    assert row["commit"] == "launch00" and row["commit_at_part"] == "moved000" and row["commit_drift"] is True
    row = g3.run_one(lpart, "t", tmp_path, st, reconstruct_fn=stub, git=dict(commit="launch00", dirty=True))
    assert row["commit_drift"] is False and row["dirty"] is True
