"""scripts/sci_test.py: the reconstruction protocol on synthetic run directories (pairing, stops, classes, verdict,
waste) - no reconstruction runs."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import sci_test as sci  # noqa: E402


def _run(d: Path, iou, one_solid=True, stopped="", state="ended", seconds=100.0, host="server_b", steps=None, operations=3):
    d.mkdir(parents=True)
    rec = dict(state=state, host=host, commit="81efefb2 2026-10-29", seconds=dict(reconstruction=seconds),
               run=dict(stopped=stopped), model=dict(one_valid_solid=one_solid, operations=operations),
               comparison=dict(exact=dict(iou=iou)) if iou is not None else {})
    (d / "record.json").write_text(json.dumps(rec))
    if steps:
        (d / "path.jsonl").write_text("".join(json.dumps(dict(event="step", **s)) + "\n" for s in steps))
    return d


def _arms(tmp_path, a, b):
    for arm, rows in (("A", a), ("B", b)):
        for part, kw in rows.items():
            _run(tmp_path / f"runs_{arm}" / part, **kw)
    return str(tmp_path / "runs_A" / "*"), str(tmp_path / "runs_B" / "*")


def test_pairs_good_rate_stops_and_the_mcnemar_table(tmp_path):
    a = {f"p{i}": dict(iou=0.99) for i in range(5)} | {f"q{i}": dict(iou=0.5) for i in range(6)}
    b = {f"p{i}": dict(iou=0.99) for i in range(4)} | {"p4": dict(iou=0.99, one_solid=False)} \
        | {f"q{i}": dict(iou=0.97) for i in range(5)} | {"q5": dict(iou=0.3, stopped="feature phase: time budget")}
    b["extra"] = dict(iou=0.99)
    ga, gb = _arms(tmp_path, a, b)
    c = sci.compare(sci.load_arm(ga), sci.load_arm(gb))
    assert c["n"] == 11 and c["only_b"] == ["extra"]
    assert c["table"] == dict(both=4, breaks=1, rescues=5, neither=1, n=11)        # p4 broken by two solids
    assert c["broken"] == ["p4"] and c["rescued"] == [f"q{i}" for i in range(5)]
    assert c["mcnemar_p"] == pytest.approx(0.109375)                                 # 5 of 6 discordant, one-sided
    assert c["b"]["stops"] == {"time budget": 1} and c["without_stops"]["n"] == 10
    assert c["a"]["hosts"] == {"server_b": 11}


def test_the_verdict_needs_every_check(tmp_path):
    a = {f"p{i}": dict(iou=0.5) for i in range(30)}
    b = {f"p{i}": dict(iou=0.99 if i < 20 else 0.5) for i in range(30)}
    ga, gb = _arms(tmp_path, a, b)
    classes = {f"p{i}": dict(cls="g0.P.C1" if i < 15 else "g1.D.C2-3") for i in range(30)}
    c = sci.compare(sci.load_arm(ga), sci.load_arm(gb), classes)
    assert c["classes"]["g0.P.C1"]["n"] == 15 and "p_worse_holm" in c["classes"]["g0.P.C1"]
    v = sci.verdict(c, floor=2.0, anchors={"frame": dict(ok=True)})
    assert v["decision"] == "accept B" and v["net_change"] == 20
    assert sci.verdict(c, floor=25.0, anchors={"frame": dict(ok=True)})["failed"] == ["above_noise"]
    assert sci.verdict(c, floor=2.0, anchors=None)["decision"].startswith("undecided: anchors")
    assert "anchors" in sci.verdict(c, floor=2.0, anchors={"frame": dict(ok=False)})["failed"]


def test_a_time_saving_change_is_judged_by_non_inferiority_and_seconds(tmp_path):
    a = {f"p{i}": dict(iou=0.99 if i % 2 else 0.5, seconds=300.0 + i) for i in range(40)}
    b = {f"p{i}": dict(iou=0.99 if i % 2 else 0.5, seconds=100.0 + i) for i in range(40)}
    ga, gb = _arms(tmp_path, a, b)
    c = sci.compare(sci.load_arm(ga), sci.load_arm(gb))
    v = sci.verdict(c, floor=None, anchors={"liner": dict(ok=True)}, kind="saving")
    assert v["checks"]["noninferior"] and v["checks"]["faster"] and v["decision"] == "accept B"


def test_a_simpler_change_is_judged_by_non_inferiority_and_fewer_parts(tmp_path):
    """The prune (2026-10-30): as good with fewer parts. The same good results and fewer parts: accepted; the same parts:
    not simpler."""
    a = {f"p{i}": dict(iou=0.99 if i % 2 else 0.5, operations=9 + i % 4) for i in range(40)}
    b = {f"p{i}": dict(iou=0.99 if i % 2 else 0.5, operations=4 + i % 3) for i in range(40)}
    ga, gb = _arms(tmp_path, a, b)
    c = sci.compare(sci.load_arm(ga), sci.load_arm(gb))
    v = sci.verdict(c, floor=None, anchors={"liner": dict(ok=True)}, kind="simpler")
    assert c["operations"]["median_diff"] < 0 and v["checks"]["fewer_parts"] and v["decision"] == "accept B"
    v_same = sci.verdict(sci.compare(sci.load_arm(ga), sci.load_arm(ga)), floor=None, anchors=None, kind="simpler")
    assert v_same["checks"]["fewer_parts"] is False


def test_anchor_band_and_the_scene_of_a_known_tree(tmp_path):
    ga, gb = _arms(tmp_path, {"frame": dict(iou=0.9655), "liner": dict(iou=0.9951)},
                   {"frame": dict(iou=0.9600), "liner": dict(iou=0.9700)})
    an = sci.anchor_check(sci.load_arm(ga), sci.load_arm(gb))
    assert an["frame"]["ok"] and not an["liner"]["ok"]
    assert sci.scene_of("aegir_n16_S_000045_T0d4c") == "aegir_n16_S_000045"
    assert sci.scene_of("machined_pocket_07") == "machined_pocket_07"


def test_waste_of_a_run(tmp_path):
    steps = [dict(index=0, seconds=10.0, kind="input"), dict(index=1, seconds=40.0, kind="add"),
             dict(index=2, seconds=50.0, kind="feature_piece", piece_id=1),
             dict(index=3, seconds=80.0, kind="feature_candidate", piece_id=1),
             dict(index=4, seconds=90.0, kind="feature_rejected", piece_id=1),
             dict(index=5, seconds=95.0, kind="feature_piece", piece_id=2),
             dict(index=6, seconds=100.0, kind="feature_accepted", piece_id=2),
             dict(index=7, seconds=160.0, kind="feature_done", stopped="time budget"),
             dict(index=8, seconds=200.0, kind="done", stopped="feature phase: time budget")]
    d = _run(tmp_path / "runs_A" / "p", 0.9, steps=steps, seconds=200.0)
    w = sci.waste_of_run(d)
    assert w["seconds"] == 200.0 and w["rejected_share"] == pytest.approx(50.0 / 200.0)    # piece 1: 40..90
    assert w["after_last_gain"] == pytest.approx(100.0) and w["time_budget"]
    assert w["accepted"] == 2 and w["candidates"] == 1 and w["by_kind"]["feature_candidate"] == pytest.approx(30.0)
    s = sci.waste_summary([w], {"p": dict(cls="g0.P.C1")})
    assert s["time_budget_stops"] == 1 and s["classes"]["g0.P.C1"]["n"] == 1
