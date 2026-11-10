"""
The Random button and the preview of a picked part, headless (offscreen Qt, 3D view off): which files count as test
parts (a manifest's "ok" rows only), that Random draws every part once before one comes again and announces it, and
that the window loads the announced part and hands it to the 3D views - only the last of several picks.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"

pytest.importorskip("PyQt5")
SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

from PyQt5 import QtWidgets  # noqa: E402

from reco_gui import setup  # noqa: E402
from reco_gui.app import MainWindow  # noqa: E402
from reco_gui.preview import PreviewLoader, load_preview  # noqa: E402
from reco_gui.settings_panel import SettingsPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_for(app, condition, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def parts(tmp_path, monkeypatch):
    """abc/ with a manifest (one ok, one not a single solid), nist/ without one, and a file that is no STEP."""
    (tmp_path / "abc" / "step").mkdir(parents=True)
    (tmp_path / "nist").mkdir()
    for name in ("abc/step/a.step", "abc/step/b.step", "nist/c.stp", "nist/D.STEP", "nist/e.stl"):
        (tmp_path / name).write_text("x")
    (tmp_path / "abc" / "manifest.csv").write_text(
        "stem,step,stl,solids,volume,bbox,faces,status\na,step/a.step,,1,1.0,1 x 1 x 1,6,ok\n"
        "b,step/b.step,,2,,1 x 1 x 1,12,not_single_solid\n")
    monkeypatch.setattr(setup, "TEST_PARTS", tmp_path)
    return tmp_path


def tetrahedron(path):
    v = np.array([[1.0, 2.0, 3.0], [5.0, 2.0, 3.0], [1.0, 4.0, 3.0], [1.0, 2.0, 9.0]])
    return v, np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])


def test_discover_test_parts(parts):
    found = [p.relative_to(parts).as_posix() for p in setup.discover_test_parts()]
    assert found == ["abc/step/a.step", "nist/D.STEP", "nist/c.stp"]
    assert setup.discover_test_parts(parts / "missing") == []


def test_random_draws_every_part_once(qapp, parts):
    panel = SettingsPanel()
    chosen = []
    panel.inputChosen.connect(chosen.append)
    first = [panel.pick_random() for _ in range(3)]
    assert sorted(p.name for p in first) == ["D.STEP", "a.step", "c.stp"]
    assert chosen == [str(p) for p in first]
    assert panel.stl.currentText() == str(first[-1])
    assert panel.pick_random() in first                       # used up: the list is read again


def test_load_preview_caption():
    v, f, caption = load_preview("/somewhere/nist/part.stp", load_mesh=tetrahedron)
    assert v.shape == (4, 3) and f.shape == (4, 3)
    assert "nist/part.stp" in caption and "4 x 2 x 6" in caption and "4 triangles" in caption
    with pytest.raises(ValueError):
        load_preview("x.stp", load_mesh=lambda p: (np.zeros((0, 3)), np.zeros((0, 3), int)))


def test_window_previews_the_picked_part(qapp, parts):
    window = MainWindow()
    window.preview = PreviewLoader(window, load_mesh=tetrahedron)
    shown, failed = [], []
    window.preview.loaded.connect(lambda path, v, f, caption: shown.append((path, caption)))
    window.preview.loaded.connect(window._on_preview)
    window.preview.failed.connect(lambda path, why: failed.append(why))
    window.settings_panel.inputChosen.disconnect()
    window.settings_panel.inputChosen.connect(window.preview_input)
    try:
        window.settings_panel.random.click()
        assert wait_for(qapp, lambda: shown)
        assert shown[0][0] == window.settings_panel.stl.currentText()
        assert "preview:" in window.status_label.text() and "4 triangles" in window.status_label.text()

        def broken(path):
            raise RuntimeError("freecadcmd could not read the file")
        window.preview = PreviewLoader(window, load_mesh=broken)
        window.preview.failed.connect(lambda path, why: failed.append(why))
        window.preview.request("bad.step")
        assert wait_for(qapp, lambda: failed)
        assert "could not read" in failed[0]
    finally:
        window.close()


def test_only_the_last_pick_is_shown(qapp):
    def slow(path):
        time.sleep(0.3 if "first" in str(path) else 0.0)
        return tetrahedron(path)
    loader = PreviewLoader(load_mesh=slow)
    shown = []
    loader.loaded.connect(lambda path, v, f, caption: shown.append(path))
    loader.request("first.step")
    loader.request("second.step")
    assert wait_for(qapp, lambda: shown)
    time.sleep(0.5)
    qapp.processEvents()
    assert shown == ["second.step"]
