#!/usr/bin/env python3
"""
Screenshots of the vector-field GUI for docs/slides_vectorfield_gui.tex -> docs/figures/gui_*.png.

Runs the real window (needs a display; X11 Qt platform) on co_countersink and on the picture-frame STL, steps it to chosen stages and
saves (a) VTK's own render of the 3D view and (b) the whole window. Qt cannot grab an embedded OpenGL widget (the area comes out
garbled or black), so the window image is composed: the ordinary widgets from QWidget.grab(), the 3D view from VTK's render pasted
into its rectangle.

    python3 scripts/make_gui_slide_figures.py [frame.stl]
"""
import os
import sys
import time
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "xcb"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

from reco_gui.app import MainWindow  # noqa: E402

OUT = ROOT / "docs" / "figures"
FRAME = Path.home() / "shape_reco_parts/benchmark_parts/1.0/Unbenannt-Frame_4_x_6.stl"


def pump(app, seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


def wait(app, cond, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    raise TimeoutError


def snapshot(app, w, name, window=True):
    pump(app)
    tmp = OUT / f"{name}_3d.png"
    w.view_field.grab_image(str(tmp))
    if window:
        img = w.grab().toImage()
        vtk_img = QtGui.QImage(str(tmp))
        origin = w.view_field.widget.mapTo(w, QtCore.QPoint(0, 0))
        target = QtCore.QRect(origin, w.view_field.widget.size())
        p = QtGui.QPainter(img)
        p.drawImage(target, vtk_img)
        p.end()
        img.save(str(OUT / f"{name}_window.png"))
    print("saved", name, flush=True)


def run_to(app, w, wanted):
    """Step the window until the states (stage, occurrence) in `wanted` (a dict {(stage, n): name}) have been photographed."""
    seen = {}
    while not w.session.finished:
        w.step()
        wait(app, lambda: not w._busy)
        st = w.session.last
        stage = st.step.stage
        seen[stage] = seen.get(stage, 0) + 1
        key = (stage, seen[stage])
        if key in wanted:
            snapshot(app, w, wanted[key], window=wanted[key].endswith(("_window", "_win")) or True)
    pump(app, 0.5)


def session(app, part, **panel):
    w = MainWindow(stl=str(part))
    w.resize(1500, 950)
    w.show()
    w.settings_panel.f_grade.setChecked(panel.get("grade", True))
    pump(app, 0.5)
    w.start()
    wait(app, lambda: w.session is not None and not w._busy)
    return w


def main():
    frame = Path(sys.argv[1]) if len(sys.argv) > 1 else FRAME
    app = QtWidgets.QApplication(sys.argv)
    OUT.mkdir(exist_ok=True)

    w = session(app, ROOT / "dataset_gen/examples/co_countersink.stl")
    run_to(app, w, {("field", 1): "gui_cs_field", ("base", 1): "gui_cs_base", ("cluster", 1): "gui_cs_cluster",
                    ("candidate", 1): "gui_cs_candidate", ("accepted", 1): "gui_cs_accepted", ("done", 1): "gui_cs_done"})
    w.close()

    if frame.is_file():
        w = session(app, frame)
        run_to(app, w, {("base", 1): "gui_frame_base", ("accepted", 1): "gui_frame_window", ("rejected", 1): "gui_frame_rejected",
                        ("done", 1): "gui_frame_done"})
        w.close()


if __name__ == "__main__":
    main()
