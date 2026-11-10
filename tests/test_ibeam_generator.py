"""dataset_gen/src/ibeam_generator.py (+ ibeam_freecad.py): planning and records with FreeCAD stubbed, and one real
freecadcmd build of one beam (Part volume vs area x length)."""

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import din_profiles as dp  # noqa: E402
from src import ibeam_freecad as ifc  # noqa: E402
from src import ibeam_generator as g  # noqa: E402

from src.freecad_path import find_freecadcmd  # noqa: E402

FREECAD = find_freecadcmd()  # $SHAPE_RECO_FREECADCMD, PATH, /usr/bin/freecadcmd (T-04)


def test_resolve_profiles():
    assert len(g.resolve_profiles("default")) == 20
    assert [p.name for p in g.resolve_profiles("HEA500, IPE 80")] == ["HEA 500", "IPE 80"]
    assert all(p.series == "HEM" for p in g.resolve_profiles("HEM"))
    with pytest.raises(ValueError):
        g.resolve_profiles("HEA 999")


def test_quat_matrix_is_a_rotation():
    rot = g.sample_rotation(random.Random(3), -180, 180)
    R = g.quat_matrix([rot.qx, rot.qy, rot.qz, rot.qw])
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1)
    assert np.allclose(g.quat_matrix([0, 0, 0, 1]), np.eye(3))
    assert np.allclose(g.quat_matrix([0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]) @ [1, 0, 0], [0, 1, 0])


def test_section_points():
    p = dp.PROFILES["HEA 500"]
    pts = ifc.section_points(p)
    assert len(pts) == 12
    assert ifc.polygon_area(pts) == pytest.approx(2 * p.b * p.tf + (p.h - 2 * p.tf) * p.tw)
    q = dp.PROFILES["I 200"]
    t_tip, t_root = ifc.flange_thickness_tip_root(ifc._dims(q))
    assert t_tip < q.tf < t_root and (t_root - t_tip) / ((q.b - q.tw) / 2) == pytest.approx(0.14)
    assert ifc._dims(ifc._dims(q)) == ifc._dims(q)          # idempotent (make_ibeam_solid passes dims through)


def test_world_boxes_identity_and_shift():
    p = dp.PROFILES["IPE 300"]
    boxes = g.world_boxes(p, 400.0, [0, 0, 0, 1], 0.1, [1, 2, 3])
    assert [b["plate"] for b in boxes] == list(g.PLATES)
    assert np.allclose(boxes[0]["center"], [1, 2 + 0.1 * (p.h - p.tf) / 2, 3])
    assert np.allclose(boxes[2]["half_extents"], [0.1 * p.tw / 2, 0.1 * (p.h - 2 * p.tf) / 2, 20])


def test_plan_beam_draws():
    rng = random.Random(0)
    for p in g.resolve_profiles("default"):
        plan = g.plan_beam(rng, p, "b")
        assert 0.5 * p.h <= plan["length"] <= 2.0 * p.h
        assert [s["kind"] for s in plan["samples"]] == ["ibeam", "plate", "plate", "plate", "envelope"]
        for s in plan["samples"]:
            assert 0.5 <= s["size_fraction"] <= 0.95
            if s["thin_ok"]:
                assert s["thin_vox_expected"] >= 1.2 - 1e-9
    plan = g.plan_beam(random.Random(1), dp.PROFILES["HEM 300"], "b", plate_fraction=0, envelope_fraction=0)
    assert [s["kind"] for s in plan["samples"]] == ["ibeam"]


def _stub_run(jobs, freecad):
    out = []
    for j in jobs:
        out.append(dict(id=j["id"], ok=True, volume_mm3=1.0, local_volume_mm3=1000.0, valid=True, n_solids=1, facets=10,
                        scale_factor=0.1, translation_mm=[0.5, 0, 0], center_mm=[0.5, 0, 0], extent_mm=[10, 20, 30]))
    return out


def test_generate_records_with_stubs(tmp_path):
    written = []

    def finish(stl, rec, out_root, resolution=64, with_field=True):
        assert Path(out_root) in Path(stl).parents and with_field
        written.append(rec)

    summary = g.generate(tmp_path, g.resolve_profiles("HEA 500,I 100"), 2, seed=5, shard=3, finish=finish,
                         run=_stub_run, log=lambda *a: None, mitre_prob=1.0, nofillet_prob=0.5,
                         keep_thin_misses=True)
    assert summary["counts"] == {"ibeam": 4, "ibeam_plate": 12, "ibeam_envelope": 4} and not summary["failures"]
    assert all(r["topology"] == "box" for r in written)
    beams = [r for r in written if r["subclass"] == "ibeam"]
    assert len(beams) == 4 and all(isinstance(r["fillet"], bool) for r in beams)
    for r in beams:
        assert len(r["boxes"]) == 3 and r["ibeam"]["profile"] in ("HEA 500", "I 100")
        par = r["parameters"]
        assert par["tw_ratio"] == pytest.approx(par["tw"] / par["h"]) and 0.5 <= par["length_ratio"] <= 2
        assert "tw_vox" in r["placement"] and "_shape" in r
    plates = [r for r in written if r.get("source") == "ibeam_plate"]
    assert sorted({r["role"] for r in plates}) == ["flange", "web"] and all(r["subclass"] == "full" for r in plates)
    env = [r for r in written if r.get("source") == "ibeam_envelope"]
    assert len(env) == 4 and all(r["subclass"] == "full" for r in env)
    # all samples of a beam share its beam_id; indices are unique and carry the shard
    ids = {}
    for r in written:
        ids.setdefault(r["beam_id"], []).append(r)
    assert len(ids) == 4 and all(len(v) == 5 for v in ids.values())
    assert len({r["index"] for r in written}) == 20 and all(r["index"] >= 3_000_000 for r in written)


def test_thin_misses_are_skipped(tmp_path):
    written = []
    summary = g.generate(tmp_path, g.resolve_profiles("IPE 600"), 6, seed=2, finish=lambda s, r, o, **k: written.append(r),
                         run=_stub_run, log=lambda *a: None, min_thin_vox=5.0)       # unreachable for the web
    assert summary["skipped_thin"].get("ibeam") == 6 and summary["skipped_thin"].get("web") == 6
    assert all(r.get("plate") != "web" and r["subclass"] != "ibeam" for r in written)


def test_web_plate_record_ratios():
    p = dp.PROFILES["HEA 500"]
    plan = g.plan_beam(random.Random(2), p, "b")
    jobs = g.build_jobs(plan, Path("/tmp/x"), 0)
    web = next(j for j in jobs if j["_sample"].get("plate") == "web")
    rec = g.make_record(plan, web, _stub_run([web], None)[0], seed=1)
    par = rec["parameters"]
    assert par["x_half_ratio"] == pytest.approx(p.tw / 2 / (p.h / 4))
    assert par["aspect_max_min"] > 30


@pytest.mark.skipif(FREECAD is None, reason="freecadcmd not installed")
def test_freecad_one_beam_volume(tmp_path):
    p = dp.PROFILES["HEA 500"]
    L = 600.0
    job = dict(id=0, kind="ibeam", profile=dict(p.__dict__, toe_radius=0.0), length=L, fillet=True, mitre=None,
               quaternion=[0.1, 0.2, 0.3, 0.927], stl=str(tmp_path / "beam.stl"), deflection=0.01, grid_mm=40.0,
               size_fraction=0.8, center_unit=[0.5, -0.5, 0.0])
    (res,) = g.run_freecad([job], FREECAD)
    assert res["ok"] and res["valid"] and res["n_solids"] == 1, res
    assert res["local_volume_mm3"] == pytest.approx(p.area_cm2 * 100 * L, rel=0.01)
    assert res["volume_mm3"] == pytest.approx(res["local_volume_mm3"] * res["scale_factor"] ** 3, rel=1e-6)
    assert max(res["extent_mm"]) == pytest.approx(0.8 * 40, rel=1e-3)
    assert min(res["bbox_min"]) >= -20 and max(res["bbox_max"]) <= 20
    assert (tmp_path / "beam.stl").stat().st_size > 0
