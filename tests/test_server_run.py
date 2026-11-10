"""reco_gui/server_run.py and run_store.import_run_part: server runs started from the window, stored for the
Benchmark tab (offscreen; no server, no ssh)."""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_server_args_follow_the_path_fields():
    from reco_gui.server_run import server_args, with_variant
    from reco_gui.setup import RunSettings
    s = RunSettings(input_stl=Path("x.step"), method="cascade")
    assert server_args(s) == ""
    t = with_variant(s, "tree")
    assert server_args(t) == "--start 'auto' --mode tree"
    a = with_variant(s, "analytic")
    assert server_args(a) == "--registry models/registry_analytic.json"
    u = replace(with_variant(s, "sub-parts"), cascade_feature_phase=False, cascade_feature_time_budget_s=300.0)
    assert server_args(u) == "--subparts --feature-subparts --no-features --feature-time-budget 300"


def _run_part_dir(d: Path, iou=0.9292):
    d.mkdir(parents=True)
    (d / "record.json").write_text(json.dumps(dict(
        state="ended", host="server_a", commit="a686537f 2026-10-29", started="2026-10-29T10:15:20",
        settings=dict(start="auto", mode="tree", subparts=False, type_source="registry", registry=None, features=True),
        run=dict(stopped="rough: grade <= 0.095"), seconds=dict(reconstruction=258.2, all=356.2),
        model=dict(operations=2, one_valid_solid=True, solids=1, valid=True, by_source={"loop": 1, "feature": 1}),
        comparison=dict(exact=dict(iou=iou, missing=52.0, invented=8681.9), sampled=dict(iou=0.9287)))))
    box = dict(kind="box", quaternion=[0, 0, 0, 1], center_mm=[0, 0, 6], dims_mm=dict(x_half=60, y_half=40, z_half=6))
    (d / "program.json").write_text(json.dumps(dict(operations=[dict(op="add", primitive=box), dict(op="cut", primitive=box)])))
    (d / "steps.jsonl").write_text('{"index": 0, "kind": "input", "seconds": 4.9, "message": "input"}\n'
                                   '{"index": 1, "kind": "add", "seconds": 23.5, "message": "first add"}\n')
    (d / "rating.json").write_text(json.dumps(dict(grade=dict(tier="E"), attribution=dict(separation=dict(cls="uncut excess")))))
    (d / "model.stl").write_text("solid m\nendsolid m\n")
    return d


def test_a_run_part_directory_becomes_a_run_of_the_store(tmp_path):
    from reco_engine import benchmark, run_store
    part = tmp_path / "parts" / "bracket_T1.step"
    part.parent.mkdir()
    part.write_text("ISO-10303-21; a step")
    run_dir = _run_part_dir(tmp_path / "remote" / "run")
    d = run_store.import_run_part(run_dir, part, store=tmp_path / "store")
    run = run_store.load_run(d)
    assert run["part"] == "bracket_T1" and run["metrics"]["exact_iou"] == pytest.approx(0.9292) and run["finished"]
    assert run["code"]["short"] == "a686537f" and run["model"]["stl"] == "model.stl" and run["timings"]["wall"] == 258.2
    steps = [json.loads(x) for x in (d / "steps.jsonl").read_text().splitlines()]
    assert [s["stage"] for s in steps] == ["input", "add"] and steps[1]["seconds"] == pytest.approx(18.6)
    rows = benchmark.benchmark(part, store=tmp_path / "store")["rows"]
    assert len(rows) == 1 and rows[0]["variant"] == "tree" and rows[0]["tier"] == "E" and rows[0]["separation"] == "uncut excess"


def test_the_window_starts_on_a_server_and_the_alternatives_start_together(qapp, tmp_path, monkeypatch):
    from reco_engine import path_policy
    from reco_gui.app import MainWindow
    part = tmp_path / "bracket_T1.stl"
    import trimesh
    trimesh.creation.box((60, 20, 4)).export(part)
    w = MainWindow(stl=str(part), autosave=False)
    w.settings_panel.c_alternatives.setValue(3)
    calls, shown = [], []
    monkeypatch.setattr(w.server_runs, "start", lambda host, settings, entry=None, variant="", name=None:
                        calls.append((host, variant, path_policy.variant_of(settings))))
    monkeypatch.setattr(w, "show_server_result", lambda mirror, load_mesh=None: None)
    monkeypatch.setattr(w, "show_benchmark", lambda select=None: shown.append(list(select or [])))
    monkeypatch.setattr(w, "follow_server_run", lambda mirror, load_mesh=None: None)
    w.requestStart.disconnect(w.worker.start)
    local = []
    w.requestStart.connect(local.append)
    w.run_on.setCurrentIndex(w.run_on.findData("server_a"))
    w.start()
    assert calls == [("server_a", "bare", "bare")] and not local                     # on the server, not here
    from reco_engine import run_store
    stored = {}
    for i, v in enumerate(["bare"]):
        d = run_store.import_run_part(_run_part_dir(tmp_path / f"r{i}", iou=0.90), part, store=tmp_path / "store")
        stored[v] = d
        w._server_stored(tmp_path / f"m{i}", d, v)
    alternatives = [c[1] for c in calls[1:]]
    assert len(alternatives) == 2 and "bare" not in alternatives                    # started together
    assert all(c[1] == c[2] for c in calls)                                         # each with its own path fields
    for i, v in enumerate(alternatives):
        d = run_store.import_run_part(_run_part_dir(tmp_path / f"a{i}"), part, store=tmp_path / "store")
        w._server_stored(tmp_path / f"n{i}", d, v)
    assert shown and len(shown[-1]) == 3 and w._server_chain is None                # the Benchmark tab, all three checked
    calls.clear()
    w.start()
    d = run_store.import_run_part(_run_part_dir(tmp_path / "x", iou=1.0), part, store=tmp_path / "store")
    w._server_stored(tmp_path / "mx", d, "bare")
    assert len(calls) == 1 and "(exact): no alternatives" in w.log.toPlainText()    # an exact first run: nothing more
    w.close()


def test_auto_picks_the_least_loaded_server_and_window_runs_come_first(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("remote_run", ROOT / "scripts" / "remote_run.py")
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    monkeypatch.setattr(rr, "host_loads", lambda hosts=rr.HOSTS: {"server_a": (1.7, 54.4, 32), "server_b": (0.45, 9.0, 20),
                                                                  "server_c": (0.44, 7.0, 16)})
    said = []
    assert rr.pick_host(say=said.append) == "server_b"                 # idle and fast; server_c idle but 3x slower
    assert "server_a 54/32" in said[0]
    monkeypatch.setattr(rr, "host_loads", lambda hosts=rr.HOSTS: {"server_a": (1.19, 38.0, 32), "server_b": (1.65, 33.0, 20),
                                                                  "server_d": (0.0, 0.0, 16)})
    assert rr.pick_host(say=said.append) == "server_a"                  # 1.19 against server_d's 3 x 1
    assert "nice -n 0 python" in rr.remote_command("/r", "/c", "", False, nice=0)
    from reco_gui import server_run
    assert server_run.INTERACTIVE_NICE == 0 and dict(server_run.HOSTS)["auto"].startswith("a server")
