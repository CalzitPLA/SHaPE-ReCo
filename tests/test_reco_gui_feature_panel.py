"""The feature side window (reco_gui.feature_panel) on the fake step stream of test_reco_gui_feature: one record per
piece with its guesses and verdict, FreeCAD's face metadata as lines (from the piece or the pieces.json beside its STL),
and the panel in the window (offscreen, 3D view off): shown by the first feature step, cleared by a restart."""
import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))
sys.path.insert(0, str(ROOT / "tests"))

pytest.importorskip("PyQt5")
from PyQt5 import QtWidgets  # noqa: E402

from reco_gui.feature_panel import FeaturePanel, FeatureTrace, metadata_lines, piece_faces  # noqa: E402
from test_reco_gui_feature import session  # noqa: E402,F401  (the module's fixture: the folded fake run)

FACES = [dict(type="plane", area=100.0, origin="input"), dict(type="plane", area=50.0, origin="both"),
         dict(type="cylinder", area=30.0, radius=5.0, origin="input"), dict(type="plane", area=80.0, origin="model")]


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_metadata_lines_split_contact_and_model_faces():
    lines = metadata_lines(dict(volume=1234.5, bbox=[[1, 2, 3], [11, 4, 9]], valid=True, faces=FACES))
    assert lines[0] == "solid: volume 1234 mm3, box 10 x 2 x 6 mm, valid"
    assert lines[1] == "contact faces (the part's): 2 plane, 1 cylinder; 180 mm2; radii 5"
    assert lines[2] == "model-only faces: 1 plane; 80 mm2"
    assert metadata_lines(dict(volume=1.0))[-1] == "faces: no metadata"
    assert metadata_lines(None) == ["faces: no metadata"]


def test_faces_are_read_from_the_pieces_json_beside_the_stl(tmp_path):
    (tmp_path / "pieces").mkdir()
    stl = tmp_path / "pieces" / "piece_minus_003.stl"
    stl.write_text("")
    (tmp_path / "pieces.json").write_text(json.dumps(dict(pieces=[
        dict(id="piece_minus_002", stl=str(stl.with_name("piece_minus_002.stl")), faces=FACES[:1]),
        dict(id="piece_minus_003", stl=str(stl), faces=FACES)])))
    assert piece_faces(dict(id=3, stl=str(stl))) == FACES
    assert piece_faces(dict(id=3, stl=str(tmp_path / "elsewhere" / "x.stl"))) == []
    (tmp_path / "pieces.json").write_text("{ not json")
    assert piece_faces(dict(id=3, stl=str(stl))) == []


def test_trace_has_one_record_per_piece_with_guesses_and_verdict(session):  # noqa: F811
    trace = FeatureTrace()
    for state in session.states:
        trace.add(state)
    candidates = [s for s in session.states if s.step.stage == "feature_candidate"]
    verdicts = [s for s in session.states if s.step.stage in ("feature_accepted", "feature_rejected")]
    assert candidates and verdicts
    assert sum(len(r.guesses) for r in trace.records.values()) == len(candidates) == trace.totals["guesses"]
    assert trace.totals["accepted"] + trace.totals["rejected"] == len(verdicts)
    ids = {(s.info or {}).get("piece_id") for s in candidates}
    assert ids <= set(trace.records)
    last = {s.info["piece_id"]: s.step.stage[len("feature_"):] for s in verdicts}     # a later verdict replaces an earlier
    for pid, verdict in last.items():
        rec = trace.records[pid]
        assert rec.verdict == verdict
        assert all(g["verdict"] in ("accepted", "rejected") for g in rec.guesses) or verdict == "accepted"
    before = dict(trace.totals)
    for state in session.states:                          # a state seen before changes nothing
        trace.add(state)
    assert dict(trace.totals) == before
    assert "guesses" in trace.summary()


def test_pattern_rejections_are_counted_apart():
    from types import SimpleNamespace as N
    trace = FeatureTrace()
    for i, (pid, reason) in enumerate(((1, "precision 0.879 < 0.999"), (2, "pattern: the congruent piece #1 found no primitive"),
                                        (3, "pattern: the congruent piece #1 found no primitive"))):
        trace.add(N(index=i, step=N(stage="feature_rejected"), info=dict(piece_id=pid, reason=reason), feature=None,
                    candidate=None, candidate_mesh=None))
    assert trace.totals["rejected"] == 3 and trace.totals["rejected by pattern"] == 2
    assert "2 of them only by the pattern" in trace.summary()


def test_panel_shows_the_piece_in_work(qapp, session):  # noqa: F811
    panel = FeaturePanel()
    shown = None
    for state in session.states:
        if state.in_feature_phase:
            panel.set_state(state)
            if state.step.stage == "feature_candidate":
                shown = state
                assert panel.table.rowCount() == len(panel.trace.records[state.info["piece_id"]].guesses)
                assert panel.table.item(panel.table.rowCount() - 1, 0).text() == str(state.info.get("source") or "")
    assert shown is not None and panel.title.text().startswith(("piece #", "segment #"))
    assert panel.total.text().startswith(f"{len(panel.trace.order)} pieces")
    panel.set_state(None)
    assert panel.table.rowCount() == 0 and not panel.trace.records and panel.title.text() == "No feature in work."
