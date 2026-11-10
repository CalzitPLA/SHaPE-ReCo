"""
The benchmark of the framework's own results (reco_engine.benchmark): the stored runs of one input by content, their
rows, the known tree beside the input, the aegir rating of a stored run (cached), the ranking per preference and the
user's choice.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

from reco_engine import benchmark as bm  # noqa: E402
from reco_engine import run_store  # noqa: E402
from reco_engine.result_schema import Operation, Primitive, ReconstructionResult  # noqa: E402


def write_input(root: Path, name="part") -> Path:
    stl = root / f"{name}.stl"
    stl.write_text(f"solid {name}\nendsolid {name}\n")                  # the store matches runs by content
    return stl


def write_truth(stl: Path):
    box = dict(type="Placed", primitive=dict(type="Box", half=[1.0, 2.0, 3.0]), rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
               translation=[0.0, 0.0, 0.0])
    cyl = dict(type="Placed", primitive=dict(type="Cylinder", radius=0.5, height=8.0, u=[0.0, 6.283185307179586], inner_radius=0.0),
               rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], translation=[0.0, 0.0, 0.0])
    p = stl.with_suffix(".json")
    p.write_text(json.dumps(dict(name=stl.stem, source="test", meta=dict(leaves=2, genus=0),
                                 program=dict(type="Subtraction", a=box, b=cyl))))
    return p


def result(ops):
    """ops: [(op, kind, dims)]"""
    return ReconstructionResult(input_path="x", operations=[
        Operation(op=o, primitive=Primitive(kind=k, dims_mm=d, confidence=0.9), net_gain=10.0) for o, k, d in ops])


def save(store, stl, ops, settings, iou=None, valid=True, seconds=10.0, label=None, finished=True):
    class Session:
        def __init__(self):
            self.states, self.finished, self.config = [], finished, None
    res = result(ops)
    s = Session()

    class State:
        pass
    st = State()
    st.result, st.program, st.info = res, res.operations, {}
    s.states.append(st)
    d = run_store.save_run(s, settings, stl, label=label, store=store)
    run = json.loads((d / "run.json").read_text())
    run["metrics"].update(exact_iou=iou, freecad_valid=valid)
    run["timings"]["wall"] = seconds
    (d / "run.json").write_text(json.dumps(run))
    return d


BARE = dict(method="cascade", cascade_start_shape="legacy", cascade_type_source="registry", cascade_subparts=False, cascade_registry=None)
CUBES = dict(BARE, cascade_start_shape="cubes")


def test_truth_for(tmp_path):
    stl = write_input(tmp_path)
    assert bm.truth_for(stl) is None
    (tmp_path / "truths").mkdir()
    (tmp_path / "truths" / "part.json").write_text(json.dumps(dict(program=dict(type="Box", half=[1, 1, 1]))))
    assert bm.truth_for(stl) == tmp_path / "truths" / "part.json"
    beside = write_truth(stl)
    assert bm.truth_for(stl) == beside                                   # beside the input wins


def test_rows_rank_and_choice(tmp_path):
    store, stl = tmp_path / "store", write_input(tmp_path)
    d1 = save(store, stl, [("add", "box", dict(x_half=1, y_half=2, z_half=3))], BARE, iou=0.93, seconds=5.0)
    d2 = save(store, stl, [("add", "box", dict(x_half=1, y_half=2, z_half=3)), ("cut", "cylinder", dict(radius=0.5, half_length=4.0))], CUBES, iou=0.99, seconds=20.0)
    d3 = save(store, stl, [("add", "box", dict(x_half=1, y_half=2, z_half=3))] * 3, BARE, iou=0.985, valid=False, seconds=2.0)
    other = write_input(tmp_path, "other")
    save(store, other, [("add", "sphere", dict(radius=1))], BARE, iou=1.0)
    b = bm.benchmark(stl, store=store)
    assert b["truth"] is None and len(b["rows"]) == 3                   # the other input's run is not in
    by = {r["dir"]: r for r in b["rows"]}
    assert by[str(d2)]["variant"] == "cubes" and by[str(d1)]["variant"] == "bare"
    assert by[str(d2)]["ops"] == 2 and by[str(d2)]["exact_iou"] == 0.99 and by[str(d2)]["seconds"] == 20.0
    assert by[str(d3)]["one_valid_solid"] is False and by[str(d2)]["rated"] is False and by[str(d2)]["tier"] is None
    assert [r["dir"] for r in bm.rank(b["rows"], "accuracy")] == [str(d2), str(d1), str(d3)]   # valid first, then IoU
    assert [r["dir"] for r in bm.rank(b["rows"], "simplicity")][0] == str(d2)   # d1 is not within 0.02 of the best
    assert [r["dir"] for r in bm.rank(b["rows"], "speed")][0] == str(d2)
    with pytest.raises(ValueError):
        bm.rank(b["rows"], "beauty")
    chosen = bm.choose(d2)
    assert chosen.name.endswith("_chosen") and run_store.load_run(chosen)["label"] == "chosen"
    assert [r for r in bm.benchmark(stl, store=store)["rows"] if r["label"] == "chosen"][0]["dir"] == str(chosen)


def test_rating_against_the_known_tree(tmp_path):
    store, stl = tmp_path / "store", write_input(tmp_path)
    write_truth(stl)
    d = save(store, stl, [("add", "box", dict(x_half=1.0, y_half=2.0, z_half=3.0)), ("cut", "cylinder", dict(radius=0.5, half_length=4.0))],
             BARE, iou=0.99)
    b = bm.benchmark(stl, store=store, rate=True, cells=24, n_surface=2000, n_prefix=500)
    row = b["rows"][0]
    assert b["truth"] == str(stl.with_suffix(".json")) and (d / "rating.json").is_file()
    assert row["rated"] and row["tier"] in "ABCDE" and row["rating_iou"] is not None
    messages = []
    again = bm.benchmark(stl, store=store, rate=True, progress=messages.append)
    assert again["rows"][0]["tier"] == row["tier"] and messages == []   # cached: nothing rated again
    assert bm.rank(again["rows"], "design")[0]["dir"] == str(d)
    # a run without a readable program is not rated and does not end the benchmark
    e = save(store, stl, [], BARE, iou=0.5)
    b2 = bm.benchmark(stl, store=store, rate=True, cells=24, n_surface=2000, n_prefix=500)
    assert {r["dir"]: r["rated"] for r in b2["rows"]}[str(e)] is False
