#!/usr/bin/env python3
"""
The preview of a picked part, before any run: the file is loaded (a STEP file is tessellated by freecadcmd, which takes
seconds) in a thread of its own and handed to the window as a mesh with a caption. Of several picks in a row only the
last one is shown.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PyQt5 import QtCore


def load_preview(path, load_mesh=None):
    """(vertices, triangles, caption) of an STL or STEP file; the caption names the file, its size and its triangles."""
    if load_mesh is None:
        from src.closest_point_field import load_mesh
    path = Path(path)
    t0 = time.time()
    vertices, triangles = load_mesh(path)
    vertices, triangles = np.asarray(vertices, float), np.asarray(triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("no triangles")
    size = vertices.max(axis=0) - vertices.min(axis=0)
    from reco_gui import setup
    try:
        name = path.resolve().relative_to(Path(setup.TEST_PARTS).resolve()).as_posix()      # "abc/step/<file>"
    except ValueError:
        name = f"{path.parent.name}/{path.name}"
    caption = (f"preview: {name}\n"
               f"{size[0]:.4g} x {size[1]:.4g} x {size[2]:.4g} (file units), {len(triangles)} triangles, "
               f"loaded in {time.time() - t0:.1f} s")
    return vertices, triangles, caption


class PreviewLoader(QtCore.QObject):
    loaded = QtCore.pyqtSignal(str, object, object, str)      # path, vertices, triangles, caption
    failed = QtCore.pyqtSignal(str, str)                      # path, reason

    def __init__(self, parent=None, load_mesh=None):
        super().__init__(parent)
        self._load_mesh = load_mesh                # injectable for tests
        self._latest = 0
        self.thread = None

    def request(self, path: str):
        self._latest += 1
        self.thread = threading.Thread(target=self._run, args=(str(path), self._latest), daemon=True)
        self.thread.start()

    def cancel(self):
        self._latest += 1

    def _run(self, path: str, ticket: int):
        try:
            vertices, triangles, caption = load_preview(path, self._load_mesh)
        except Exception as exc:                   # a file FreeCAD cannot read, no freecadcmd, an empty shape
            if ticket == self._latest:
                self.failed.emit(path, f"{type(exc).__name__}: {exc}")
            return
        if ticket == self._latest:
            self.loaded.emit(path, vertices, triangles, caption)
