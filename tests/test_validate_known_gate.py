#!/usr/bin/env python3
"""scripts/validate_known.py's regression gate: `fork` freezes a stratified subset of a corpus (deterministic, every
stratum before any repeats), `compare` pairs two ratings of the same shapes - tier moves, share of A and median IoU
with paired bootstrap intervals, the runs whose IoU fell. Synthetic corpora, no server."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import validate_known as vk  # noqa: E402


def _corpus(tmp_path, name, strata):
    c = tmp_path / name
    (c / "truths").mkdir(parents=True)
    items, stems = [], []
    for stratum, n in strata.items():
        for i in range(n):
            stem = f"{stratum}_{i}"
            stems.append(stem)
            items.append(dict(stem=stem, source="v5", stratum=stratum, leaves=1, gate=dict(ok=True)))
            (c / "truths" / f"{stem}.json").write_text("{}")
    (c / "selection.json").write_text(json.dumps(dict(name=name, items=items, excluded=[])))
    (c / "list.txt").write_text("".join(s + "\n" for s in stems))
    return c


def _rating(run_dir, tier, iou, name="rating.json"):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / name).write_text(json.dumps(dict(grade=dict(tier=tier), vector=dict(volume=dict(iou=iou)))))


def test_fork_takes_every_stratum_first_and_is_deterministic(tmp_path):
    c = _corpus(tmp_path, "calib", {"box": 10, "cone": 2, "sphere": 5})
    vk.main(["fork", str(c), "--out", str(tmp_path / "gate1"), "--subset", "7"])
    vk.main(["fork", str(c), "--out", str(tmp_path / "gate2"), "--subset", "7"])
    stems = (tmp_path / "gate1" / "list.txt").read_text().split()
    assert stems == (tmp_path / "gate2" / "list.txt").read_text().split() and len(stems) == 7
    strata = [s.rsplit("_", 1)[0] for s in stems]
    assert strata.count("cone") == 2 and strata.count("box") >= 2 and strata.count("sphere") >= 2
    sel = json.loads((tmp_path / "gate1" / "selection.json").read_text())
    assert sel["forked_from"] == "calib" and sorted(i["stem"] for i in sel["items"]) == sorted(stems)
    assert sorted(p.stem for p in (tmp_path / "gate1" / "truths").glob("*.json")) == sorted(stems)
    with pytest.raises(SystemExit):
        vk.main(["fork", str(c), "--out", str(tmp_path / "gate1")])


def test_fork_with_shape_classes_as_strata_also_from_an_unbuilt_pool(tmp_path):
    """--strata-from (shape_atlas.py select --strata-out): only the stems it names, round robin over ITS strata, the
    items relabelled - also when the pool is not built yet (its list.txt empty)."""
    c = _corpus(tmp_path, "pool", {"n3_S": 6, "n5_U": 6})
    (c / "list.txt").write_text("")
    classes = {"n3_S_0": "g0.D.C2-3", "n3_S_1": "g0.D.C2-3", "n3_S_2": "g0.D.C2-3", "n5_U_0": "g1.M.C4-7",
               "n5_U_1": "g1.M.C4-7", "gone": "g0.P.C1"}                # a stem the pool does not have: ignored
    strata = tmp_path / "strata.json"
    strata.write_text(json.dumps(classes))
    vk.main(["fork", str(c), "--out", str(tmp_path / "cls"), "--subset", "4", "--strata-from", str(strata)])
    stems = (tmp_path / "cls" / "list.txt").read_text().split()
    sel = json.loads((tmp_path / "cls" / "selection.json").read_text())
    assert len(stems) == 4 and sum(s.startswith("n5_U") for s in stems) == 2 and "gone" not in stems
    assert {i["stem"]: i["stratum"] for i in sel["items"]} == {s: classes[s] for s in stems}
    assert sel["strata_from"] == str(strata)


def test_compare_pairs_two_ratings_of_the_same_runs(tmp_path):
    c = _corpus(tmp_path, "calib", {"sphere": 4})
    before = [("B", 0.998), ("B", 0.998), ("A", 1.0), ("C", 0.99)]
    after = [("A", 1.0), ("A", 1.0), ("A", 1.0), ("C", 0.985)]
    for i, (b, a) in enumerate(zip(before, after)):
        _rating(c / "runs" / f"sphere_{i}", *b)
        _rating(c / "runs" / f"sphere_{i}", *a, name="rating_polished.json")
    _rating(c / "runs" / "unpaired", "A", 1.0)                       # no second rating: not paired
    out = tmp_path / "cmp.json"
    vk.main(["compare", str(c), str(c), "--rating-b", "rating_polished.json", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["n"] == 4 and r["better"] == 2 and r["worse"] == 0 and r["moves"]["B->A"] == 2
    assert r["tiers_a"]["A"] == 1 and r["tiers_b"]["A"] == 3
    d, lo, hi = r["share_A"]
    assert d == pytest.approx(0.5) and lo <= d <= hi
    assert [x[0] for x in r["iou_fell"]] == ["sphere_3"]
