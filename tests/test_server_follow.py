"""reco_gui/server_follow.py and scripts/remote_run.py: a server run followed from its mirrored path.jsonl (offscreen,
no server, no ssh)."""

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen"), str(ROOT / "scripts")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


BOX = dict(kind="box", quaternion=[0, 0, 0, 1], center_mm=[0.0, 0.0, 6.0], dims_mm=dict(x_half=60.0, y_half=40.0, z_half=6.0))
HOLE = dict(kind="cylinder", quaternion=[0, 0, 0, 1], center_mm=[-30.0, 0.0, 6.0], dims_mm=dict(radius=8.0, half_length=7.0))


def _events():
    return [dict(event="grid", origin=[0, 0, 0], pitch=1.0, shape=[4, 4, 4]),
            dict(event="step", index=0, seconds=1.0, kind="input", phase="loop", branch="main", n_ops=0, message="1182 input voxels"),
            dict(event="step", index=1, seconds=5.0, kind="add", phase="loop", branch="main", n_ops=1, message="first add: box"),
            dict(event="op_added", op_index=0, op="add", primitive=BOX, fingerprint="b", net_gain=10.0, step=1),
            dict(event="step", index=2, seconds=9.0, kind="feature_accepted", phase="feature", branch="main", n_ops=2,
                 message="cut cylinder"),
            dict(event="op_added", op_index=1, op="cut", primitive=HOLE, fingerprint="h", net_gain=3.0, step=2),
            dict(event="step", index=3, seconds=12.0, kind="done", phase="loop", branch="main", n_ops=2, message="done")]


def test_model_mesh_of_a_path_so_far(tmp_path):
    from reco_gui.server_follow import model_mesh, step_line
    p = tmp_path / "path.jsonl"
    p.write_text("".join(json.dumps(e) + "\n" for e in _events()[:4]))
    v, f, n = model_mesh(p, resolution=32)
    assert n == 1 and len(f) > 0 and v[:, 0].min() == pytest.approx(-60.0, abs=2.5)
    assert step_line(_events()[2]).startswith("[server]   2 loop  add") and "(5 s)" in step_line(_events()[2])


def test_the_follower_reads_new_lines_only_and_ends_at_done(qapp, tmp_path):
    from reco_gui.server_follow import ServerFollow
    (tmp_path / "link.json").write_text(json.dumps(dict(id="bracket_T1_x", host="server_a")))
    p = tmp_path / "path.jsonl"
    ev = [json.dumps(e) + "\n" for e in _events()]
    p.write_text("".join(ev[:3]) + ev[3][:20])                         # the last line half written
    f = ServerFollow(tmp_path, interval_ms=10_000)
    lines, models, ends = [], [], []
    f.stepLine.connect(lines.append)
    f.modelReady.connect(lambda v, t, cap: models.append(cap))
    f.finished.connect(ends.append)
    f.poll()
    assert len(lines) == 2 and not ends                                  # input, add; the half line waits
    with open(p, "a") as fh:
        fh.write(ev[3][20:] + "".join(ev[4:]))
    f.poll()
    assert len(lines) == 4 and "done" in lines[-1]
    for _ in range(200):                                                 # the mesh thread, then the end
        qapp.processEvents()
        if f._thread is not None:
            f._thread.wait(50)
        f.poll()
        if ends:
            break
    assert ends == ["done"] and models and "2 operations" in models[-1] and "on server_a" in models[-1]


def test_the_remote_command_is_niced_backgrounded_and_rated():
    import remote_run as rr
    c = rr.remote_command("/nfs/x/remote/r1", "/data/user/shape_reco/wt/protocol_e6aa19ed", "--mode tree", truth=True)
    assert "setsid nohup nice -n 10 python scripts/run_part.py /nfs/x/remote/r1/input.step" in c
    assert "--truth /nfs/x/remote/r1/truth.json" in c and c.rstrip().endswith("cat /nfs/x/remote/r1/pid")
    assert "--mode tree" in c and "CUDA_VISIBLE_DEVICES=" in c and "QT_QPA_PLATFORM=offscreen" in c


def _fetched_mirror(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "link.json").write_text(json.dumps(dict(id="bracket_T1_x", host="server_a", step="bracket_T1.step")))
    (run / "record.json").write_text(json.dumps(dict(
        host="server_a", commit="e6aa19ed", run=dict(stopped="rough: grade <= 0.095"), seconds=dict(reconstruction=258.2),
        model=dict(operations=2, one_valid_solid=True, solids=1, valid=True),
        comparison=dict(exact=dict(iou=0.9292, missing=52.0, invented=8681.9), sampled=dict(iou=0.9287)))))
    (run / "program.json").write_text(json.dumps(dict(operations=[dict(op="add", primitive=BOX), dict(op="cut", primitive=HOLE)])))
    (run / "rating.json").write_text(json.dumps(dict(
        grade=dict(tier="E"), vector=dict(trajectory=[dict(iou=0.35), dict(iou=0.42)]),
        attribution=dict(classes={"uncut excess": 0.0748}, separation=dict(
            cls="uncut excess", volume_share=0.0748, culprit=0, cause="the cover's excess was never cut", event=dict(index=196))))))
    return tmp_path


def test_the_result_panel_of_a_fetched_server_run(tmp_path):
    from reco_gui.server_result import result_html
    html_text, info = result_html(_fetched_mirror(tmp_path))
    assert "exact IoU 0.9292, tier E, one valid solid" in info["caption"] and info["model_stl"] is None
    assert "uncut excess" in html_text and "7.48 % of the volume" in html_text and "+0.0700" in html_text
    assert any("separation point: uncut excess" in line for line in info["lines"])


def test_the_window_shows_the_result_when_the_run_has_ended(qapp, tmp_path):
    from reco_gui.app import MainWindow
    m = _fetched_mirror(tmp_path)
    (m / "path.jsonl").write_text("".join(json.dumps(e) + "\n" for e in _events()))
    (m / "run" / "model.stl").write_text("solid x\nendsolid x\n")
    w = MainWindow(autosave=False)
    w.show_server_result = lambda mirror, load_mesh=None: shown.append(str(mirror))   # the fetched run is shown
    shown = []
    w._server_run_ended(m, "done")
    assert shown == [str(m)]
    w.close()


def test_the_follower_reports_what_is_tried_and_where(qapp, tmp_path):
    """The user, 2026-10-29: the preview showed only the shape, not what happens - now the newest primitive tried
    (candidateReady, with what happened to it) and the leftover's box (regionReady)."""
    from reco_gui.server_follow import ServerFollow, shape_caption
    (tmp_path / "link.json").write_text(json.dumps(dict(id="r", host="server_a")))
    ev = _events()[:2] + [
        dict(event="step", index=2, seconds=6.0, kind="component", phase="loop", branch="main", n_ops=1, op="cut",
             voxels=3234, bbox_world=[[-60.0, -15.0, 13.0], [-20.0, 15.0, 34.0]], message="cut component"),
        dict(event="step", index=3, seconds=7.0, kind="candidate", phase="loop", branch="main", n_ops=1, op="cut",
             primitive=HOLE, gain=-1963, ok=False, reason="void check: hits_material", message="cut cylinder")]
    (tmp_path / "path.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
    f = ServerFollow(tmp_path, interval_ms=10_000)
    shapes, regions = [], []
    f.candidateReady.connect(lambda v, t, cap, op: shapes.append((len(t), cap, op)))
    f.regionReady.connect(lambda lo, hi, cap: regions.append((lo, hi, cap)))
    f.poll()
    for _ in range(100):
        qapp.processEvents()
        if f._shape_thread is not None:
            f._shape_thread.wait(50)
        if shapes:
            break
    assert regions and regions[0][0] == [-60.0, -15.0, 13.0] and "3234 voxels" in regions[0][2]
    assert shapes and shapes[0][0] > 0 and shapes[0][2] == "cut"
    assert shapes[0][1] == "step 4: refused cut cylinder (gain -1963) - void check: hits_material"
    assert shape_caption(dict(index=81, kind="accepted", op="cut", primitive=BOX, gain=9242)) == "step 82: accepted cut box (gain 9242)"


def test_the_part_library_lists_groups_with_step_first(tmp_path):
    from reco_gui import setup
    (tmp_path / "step").mkdir()
    (tmp_path / "stl").mkdir()
    for name in ("a.step", "b.step"):
        (tmp_path / "step" / name).write_text("x")
    for name in ("a.stl", "c.stl"):
        (tmp_path / "stl" / name).write_text("x")
    lib = setup.discover_library((("corpus", [tmp_path / "step" / "*.step", tmp_path / "stl" / "*.stl"]),
                                  ("missing", tmp_path / "nothing" / "*.step")))
    assert [g for g, _ in lib] == ["corpus"] and [p.name for p in lib[0][1]] == ["a.step", "b.step", "c.stl"]
