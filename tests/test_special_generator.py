"""dataset_gen/src/special_generator.py (+ special_freecad.py): the hollow / tube / rhs / frame subclasses. Logic with
FreeCAD and the sample writer stubbed (vocabulary, member-kind balance, record fields, redraws), and real freecadcmd
builds of a hollow box, a tube, an rhs, a box frame of rods and a sphere frame of tubes (one valid solid, volume > 0,
a closed cavity for the hollow one). The real builds are skipped when freecadcmd is missing."""

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import special_generator as sg  # noqa: E402

from src.freecad_path import find_freecadcmd  # noqa: E402

FREECAD = find_freecadcmd()  # $SHAPE_RECO_FREECADCMD, PATH, /usr/bin/freecadcmd (T-04)


# ------------------------------------------------------------------------------------------------ vocabulary

def test_vocabulary_matches_sample_writer():
    from src import sample_writer
    assert {k: tuple(v) for k, v in sg.SUBCLASSES.items()} == {k: tuple(v) for k, v in sample_writer.SUBCLASSES.items()}
    assert sum(len(v) for v in sg.SUBCLASSES.values()) == 24


def test_owned_subclasses():
    assert len(sg.OWNED) == 12
    assert {s.split("/")[1] for s in sg.OWNED} == {"hollow", "tube", "rhs", "frame"}
    for spec in sg.OWNED:
        parent, sub = sg.parse_subclass(spec)
        assert sub in sg.SUBCLASSES[parent]
    for bad in ("box/ibeam", "cylinder/sector", "box/tube", "cone/band", "prism/frame"):
        with pytest.raises(ValueError):
            sg.parse_subclass(bad)


def test_member_kinds_balanced():
    for parent in sg.PARENTS:
        counts = Counter(sg.member_kind_for(parent, i) for i in range(25))
        assert set(counts) == set(sg.MEMBER_KINDS) and set(counts.values()) == {5}
    # one frame per parent (the smoke run) already covers every kind
    assert {sg.member_kind_for(p, 0) for p in sg.PARENTS} == set(sg.MEMBER_KINDS)


def test_jobs_deterministic_and_bounded(tmp_path):
    for spec in sg.OWNED:
        parent, sub = sg.parse_subclass(spec)
        a = sg.make_job(sg.sample_rng(3, 0, spec, 5), parent, sub, 5, tmp_path / "x.stl", 40.0, (0.5, 0.9), 64)
        b = sg.make_job(sg.sample_rng(3, 0, spec, 5), parent, sub, 5, tmp_path / "x.stl", 40.0, (0.5, 0.9), 64)
        c = sg.make_job(sg.sample_rng(4, 0, spec, 5), parent, sub, 5, tmp_path / "x.stl", 40.0, (0.5, 0.9), 64)
        assert a == b and a != c
        R = np.asarray(a["R"])
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)
        assert 0.5 <= a["placement"]["size_fraction"] <= 0.9
        if sub == "frame":
            fr = a["frame"]
            assert fr["thickness_frac"] >= 0.05 and 0.10 <= fr["tess_tol_frac"] <= 0.25
            assert fr["member_kind"] == sg.member_kind_for(parent, 5)
            assert ("ibeam_profile" in fr) == (fr["member_kind"] == "ibeam")
        else:
            assert 0.05 <= a["wall_frac"] <= 0.20
        if sub == "rhs":
            hx, hy, hz = a["dims"]["half"]
            assert hz > max(hx, hy)                     # the open direction is the long axis


# ------------------------------------------------------------------------------------------------ records (stubbed)

def fake_result(job, ok=True):
    info = dict(wall_mm=1.2, wall_frac_effective=job.get("wall_frac"))
    if job["subclass"] == "frame":
        info = dict(member_kind=job["frame"]["member_kind"], tessellation_tol=3.0, tessellation_tol_frac=0.15,
                    tessellation_angular=1.5, n_members=2, n_vertices=3, thickness_mm=1.0, node_spheres=False,
                    node_radius=0.0, ibeam_builder=None,
                    members=[dict(p0=[0, 0, 0], p1=[1, 0, 0], kind=job["frame"]["member_kind"], section=dict(d=1.0)),
                             dict(p0=[1, 0, 0], p1=[1, 1, 0], kind=job["frame"]["member_kind"], section=dict(d=1.0))])
    return dict(id=job["id"], ok=ok, error=None if ok else "invalid shape", attempts=["ok"], n_attempts=1, valid=True,
                n_solids=1, n_shells=2 if job["subclass"] == "hollow" else 1, volume_mm3=123.0, scale_factor=0.8,
                center_mm=[1.0, -2.0, 3.0], extent_mm=[20.0, 10.0, 5.0], parent_center_mm=[1.0, -2.0, 3.0],
                seconds_build=0.1, seconds_export=0.1, info=info)


def stub_finish(calls):
    def finish(stl, record, out_root, resolution=64, with_field=True, field_points=2048):
        calls.append(dict(stl=stl, record=record, out_root=out_root))
        out = {k: v for k, v in record.items() if k != "_shape"}
        out["placement"] = dict(record["placement"], field=dict(seconds=0.5))
        return out
    return finish


@pytest.mark.parametrize("spec", ["box/hollow", "cylinder/tube", "box/rhs", "sphere/frame", "torus/frame", "cone/hollow"])
def test_record_fields(tmp_path, spec):
    calls = []
    out = sg.make_sample(spec, 0, 17, tmp_path, seed=11, shard=0, builder=lambda jobs: [fake_result(j) for j in jobs],
                         finish=stub_finish(calls))
    rec = calls[0]["record"]
    parent, sub = spec.split("/")
    assert rec["topology"] == parent and rec["subclass"] == sub and rec["mode"] == "type"
    assert rec["sample_kind"] == "special" and rec["freecad_volume_mm3"] == 123.0
    assert set(rec["rotation"]) == {"rx", "ry", "rz", "qx", "qy", "qz", "qw"}
    q = [rec["rotation"][k] for k in ("qx", "qy", "qz", "qw")]
    assert np.isclose(np.linalg.norm(q), 1.0)
    pl = rec["placement"]
    assert pl["mode"] == "fixed" and pl["grid_mm"] == 40.0 and pl["stl_placed"] and len(pl["center_vox"]) == 3
    assert calls[0]["stl"] == tmp_path / parent / f"{sub}_000017.stl"
    shape = rec["_shape"]
    assert shape["R"].shape == (3, 3) and len(shape["t"]) == 3
    if sub == "frame":
        assert rec["member_kind"] in sg.MEMBER_KINDS and rec["n_members"] == len(rec["members"]) == 2
        assert rec["tessellation_tol"] > 0
        for m in rec["members"]:
            assert {"p0", "p1", "kind", "section"} <= set(m)
    else:
        assert rec["wall_mm"] > 0 and rec["wall_voxels"] == pytest.approx(1.2 / (40.0 / 64))
    assert out["timing"]["field"] == 0.5 and out["timing"]["spec"] == spec


def test_labels_from_shape_through_sample_writer(tmp_path):
    """The _shape entry yields the parent's labels the way sample_writer derives them (a centred cylinder: the label
    centre is the world position of the parent's centre)."""
    from src import sample_writer
    calls = []
    sg.make_sample("cylinder/tube", 0, 1, tmp_path, seed=2, shard=0, builder=lambda jobs: [fake_result(j) for j in jobs],
                   finish=stub_finish(calls))
    rec = calls[0]["record"]
    lab = sample_writer._labels_from_shape(rec["_shape"])
    assert np.allclose(lab["center_mm"], [1.0, -2.0, 3.0])
    assert lab["dims_mm"]["radius"] == pytest.approx(rec["parameters"]["radius"] * 0.8)
    assert lab["rotation_canonical"]["group"] == "Dinf"


def test_failed_build_is_redrawn(tmp_path):
    seen = []

    def builder(jobs):
        seen.append(jobs[0]["dims"])
        return [fake_result(jobs[0], ok=len(seen) > 1)]
    calls = []
    out = sg.make_sample("box/hollow", 0, 0, tmp_path, seed=1, shard=0, builder=builder, finish=stub_finish(calls))
    assert len(seen) == 2 and seen[0] != seen[1]
    assert out["failures"] == ["invalid shape"] and calls[0]["record"]["freecad"]["host_redraws"] == 1


# ------------------------------------------------------------------------------------------------ real FreeCAD

def _job(tmp_path, spec, i=0, **frame_overrides):
    parent, sub = sg.parse_subclass(spec)
    job = sg.make_job(sg.sample_rng(5, 0, spec, i), parent, sub, i, tmp_path / f"{parent}_{sub}.stl", 40.0, (0.6, 0.8), 64)
    if frame_overrides:
        job["frame"].update(frame_overrides)
    return job


@pytest.mark.skipif(FREECAD is None, reason="freecadcmd not installed")
def test_real_freecad_builds(tmp_path):
    jobs = [_job(tmp_path, "box/hollow"), _job(tmp_path, "cylinder/tube"), _job(tmp_path, "box/rhs"),
            _job(tmp_path, "box/frame", member_kind="rod", node_spheres=True),
            _job(tmp_path, "sphere/frame", member_kind="tube", node_spheres=False, tess_angular=2.2)]
    results = sg.run_freecad(jobs, FREECAD)
    for job, res in zip(jobs, results):
        name = f"{job['parent']}/{job['subclass']}"
        assert res["ok"], (name, res.get("error"), res.get("attempts"))
        assert res["valid"] and res["n_solids"] == 1 and res["volume_mm3"] > 0, name
        assert Path(job["stl_out"]).stat().st_size > 1000, name
        k = res["scale_factor"]
        lo, hi = np.asarray(res["bbox"])
        assert (lo >= -20.0 - 1e-6).all() and (hi <= 20.0 + 1e-6).all(), name          # inside the 40 mm grid
        assert res["volume_local_mm3"] * k ** 3 == pytest.approx(res["volume_mm3"], rel=1e-6)
    hollow, tube, rhs, box_frame, sphere_frame = results
    assert hollow["n_shells"] == 2                                  # the closed cavity
    half = np.asarray(jobs[0]["dims"]["half"]) * hollow["scale_factor"]
    assert hollow["volume_mm3"] < 0.95 * np.prod(2 * half)
    assert tube["n_shells"] == 1 and rhs["n_shells"] == 1           # open ends: no cavity
    assert box_frame["info"]["member_kind"] == "rod" and box_frame["info"]["n_members"] == 18     # 12 edges + 6 diagonals
    assert sphere_frame["info"]["member_kind"] == "tube" and sphere_frame["info"]["n_members"] <= 60
    for m in sphere_frame["info"]["members"]:
        assert m["section"]["d"] >= 0.05 * 2 * jobs[4]["dims"]["radius"] * sphere_frame["scale_factor"] * 0.999
