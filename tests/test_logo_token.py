"""reco_gui/logo_token.py: the logo turning like a token in the empty 3D views (offscreen Qt, 3D view off).

The frames: the face square to the eye at 0, edge-on at 90 degrees (only the rim), the back face at 180 degrees reads
the right way round (not mirrored); the timer runs only while the token is visible. The views: the token stands in
while nothing is on screen and no run computes, and gives way to a preview, a state, a followed run's meshes and a
nephogram (RECO_GUI_NO_3D: the switching is the same without VTK)."""

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"

QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
from PyQt5 import QtGui  # noqa: E402

from reco_gui.logo_token import LogoToken, badge_path  # noqa: E402
from reco_gui.viewer_cascade import CascadeView3D  # noqa: E402
from reco_gui.viewer_field import FieldView3D  # noqa: E402

W, H = 320, 240


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def pixels(img: QtGui.QImage) -> np.ndarray:
    img = img.convertToFormat(QtGui.QImage.Format_RGB32)
    buf = img.constBits()
    buf.setsize(img.byteCount())
    return np.frombuffer(buf, np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :img.width(), :3].astype(float)


def background_mask(a: np.ndarray) -> np.ndarray:
    bg = np.array([LogoToken.BACKGROUND.blue(), LogoToken.BACKGROUND.green(), LogoToken.BACKGROUND.red()], float)
    return np.abs(a - bg).sum(axis=2) < 6


@pytest.mark.skipif(badge_path() is None, reason="no logo image in packaging/")
def test_face_on_edge_on_and_the_back_reads_right(qapp):
    t = LogoToken()
    front, edge, back = (pixels(t.render_frame(W, H, a)) for a in (0.0, math.pi / 2, math.pi))
    R = LogoToken.SHARE * min(W, H) / 2
    cy = H // 2 - int(0.06 * R)
    # face-on: the token covers about a disc of radius R; edge-on: a narrow band of rim, background beside it
    covered = (~background_mask(front)).sum()
    assert 0.8 * math.pi * R * R < covered < 1.3 * math.pi * R * R
    assert not background_mask(edge)[cy, W // 2] and background_mask(edge)[cy, W // 2 + int(0.4 * R)]
    # the back face at 180 degrees is the front again, not its mirror image (the badge is not symmetric)
    box = (slice(cy - int(0.6 * R), cy + int(0.6 * R)), slice(W // 2 - int(0.6 * R), W // 2 + int(0.6 * R)))
    same = np.abs(back[box] - front[box]).mean()
    mirrored = np.abs(back[box][:, ::-1] - front[box]).mean()
    assert same < 0.5 * mirrored


def test_spins_only_while_visible(qapp):
    t = LogoToken()
    t.resize(W, H)
    assert not t.spinning
    t.show()
    qapp.processEvents()
    assert t.spinning
    a0 = t._angle
    end = time.time() + 2.0
    while time.time() < end and t._angle == a0:
        qapp.processEvents()
        time.sleep(0.01)
    assert t._angle != a0                                  # it turns
    t.hide()
    assert not t.spinning                                  # hidden: no timer, no CPU
    t.close()


@pytest.mark.parametrize("cls", [FieldView3D, CascadeView3D])
def test_the_view_shows_the_token_only_while_empty_and_idle(qapp, cls):
    v = cls()
    assert v.token_shown                                   # nothing loaded
    v.set_running(True)
    assert not v.token_shown                               # a run loads: no token, even before its first state
    v.set_running(False)
    assert v.token_shown
    tri = (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float), np.array([[0, 1, 2]]))
    v.show_input(*tri, "part")
    assert not v.token_shown                               # the preview of a picked part
    v.set_state(None)
    assert v.token_shown                                   # cleared
    v.show_pair(tri, None, "server run")
    assert not v.token_shown                               # a followed server run
    v.set_state(None)
    v.show_layer("candidate", None)
    assert v.token_shown                                   # hiding a layer puts nothing on screen
    if cls is CascadeView3D:
        v.show_distance(object())
        assert not v.token_shown                           # the nephogram
    v.close()


def test_the_window_switches_the_token_with_the_busy_signal_and_the_preview(qapp, tmp_path):
    from reco_gui.app import MainWindow
    w = MainWindow(make_session=lambda s, models=None: (None, None, None))
    views = (w.view_field, w.view_cascade)
    assert all(v.token_shown for v in views)
    w._on_busy(True)
    assert not any(v.token_shown for v in views)
    w._on_busy(False)
    assert all(v.token_shown for v in views)               # idle, nothing on screen
    part = tmp_path / "part.stl"
    part.write_text("solid x\nendsolid x\n")
    w._on_preview(str(part), np.zeros((3, 3)), np.array([[0, 1, 2]]), "part")
    assert not any(v.token_shown for v in views)
    w.close()
