#!/usr/bin/env python3
"""
3D view of the vector-field reconstruction, in the geometry's own units (mm): the input part, the closest-point field as short
lines from sample points to their nearest surface point, the model built so far, the input surface it does not explain, the
cluster being fitted and the candidate primitive under inspection. Like the voxel view it degrades to a text note without
VTK/OpenGL (or with RECO_GUI_NO_3D set).
"""

from __future__ import annotations

import os
import textwrap
from typing import Dict, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

# name: (label, colour, opacity, tooltip)
LAYERS = {
    "input":       ("Input part",      (0.55, 0.55, 0.53), 0.16, "the STL/STEP model as given (grey)"),
    "field":       ("Field V(x)",      (0.10, 0.55, 0.60), 0.70, "lines from sample points x to their closest surface point c(x): V = c - x"),
    "model":       ("Model",           (0.165, 0.471, 0.839), 0.42, "the CSG program accepted so far (blue)"),
    "unexplained": ("Unexplained",     (0.890, 0.286, 0.282), 1.0, "input surface points the model does not explain (red)"),
    "cluster":     ("Cluster",         (0.980, 0.780, 0.100), 1.0, "the unexplained points being fitted now (yellow)"),
    "candidate":   ("Candidate",       (0.922, 0.408, 0.204), 0.60, "the primitive being tried; an open cutter is cropped to the part's box (orange)"),
}
REJECTED = (0.55, 0.20, 0.20)
CUTTER = (0.910, 0.482, 0.643)


def polydata_mesh(vertices: np.ndarray, triangles: np.ndarray):
    from vtkmodules.util import numpy_support
    from vtkmodules.vtkCommonCore import vtkIdTypeArray
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkCommonCore import vtkPoints
    pts = vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(vertices, dtype=float), deep=True))
    tri = np.asarray(triangles, dtype=np.int64)
    cells = np.hstack([np.full((len(tri), 1), 3, dtype=np.int64), tri]).ravel()
    ca = vtkCellArray()
    ids = numpy_support.numpy_to_vtkIdTypeArray(cells, deep=True)
    ca.SetCells(len(tri), ids)
    poly = vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(ca)
    return poly


def polydata_points(points: np.ndarray):
    from vtkmodules.util import numpy_support
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    n = len(points)
    pts = vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(points, dtype=float).reshape(-1, 3), deep=True))
    ca = vtkCellArray()
    ca.SetCells(n, numpy_support.numpy_to_vtkIdTypeArray(
        np.stack([np.ones(n, dtype=np.int64), np.arange(n, dtype=np.int64)], axis=1).ravel(), deep=True))
    poly = vtkPolyData()
    poly.SetPoints(pts)
    poly.SetVerts(ca)
    return poly


def polydata_lines(a: np.ndarray, b: np.ndarray):
    """Segments a[i] - b[i]."""
    from vtkmodules.util import numpy_support
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    n = len(a)
    pts = vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(np.ascontiguousarray(np.vstack([a, b]), dtype=float), deep=True))
    i = np.arange(n, dtype=np.int64)
    ca = vtkCellArray()
    ca.SetCells(n, numpy_support.numpy_to_vtkIdTypeArray(np.stack([np.full(n, 2, dtype=np.int64), i, i + n], axis=1).ravel(), deep=True))
    poly = vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(ca)
    return poly


class _Layer:
    def __init__(self, renderer, colour, opacity, point_size: float = 0.0, line_width: float = 0.0):
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        self.mapper = vtkPolyDataMapper()
        self.mapper.ScalarVisibilityOff()
        self.actor = vtkActor()
        self.actor.SetMapper(self.mapper)
        prop = self.actor.GetProperty()
        prop.SetColor(*colour)
        prop.SetOpacity(opacity)
        if point_size:
            prop.SetPointSize(point_size)
            prop.SetRenderPointsAsSpheres(1)
        if line_width:
            prop.SetLineWidth(line_width)
        self.actor.VisibilityOff()
        renderer.AddActor(self.actor)
        self.key = None
        self.has = False
        self.wanted = True

    def set(self, key, build=None):
        """Show what `build()` returns, unless `key` is what is shown already; key None (or no builder) shows nothing."""
        if key == self.key and (key is not None or not self.has):
            return
        self.key = key
        poly = build() if (key is not None and build is not None) else None
        self.has = poly is not None and poly.GetNumberOfPoints() > 0
        if self.has:
            self.mapper.SetInputData(poly)


class FlowLayout(QtWidgets.QLayout):
    """A layout that places its widgets left to right and wraps to the next line when the width runs out (Qt's flow
    layout example): a row of switches that never widens the window and never scrolls (the user, 2026-10-29: "the
    application fits the screen")."""

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self._items = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._place(QtCore.QRect(0, 0, width, 0), test=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._place(rect, test=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _place(self, rect, test: bool) -> int:
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line = r.x(), r.y(), 0
        for it in self._items:
            w = it.widget()
            if w is not None and w.isHidden():
                continue
            hint = it.sizeHint()
            if x > r.x() and x + hint.width() > r.right() + 1:          # wrap
                x, y, line = r.x(), y + line + self.spacing(), 0
            if not test:
                it.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self.spacing()
            line = max(line, hint.height())
        return y + line - rect.y() + m.bottom()


def bar_row(bar: "QtWidgets.QHBoxLayout") -> "QtWidgets.QWidget":
    """A row of switches and buttons that does not set the window's least width: narrower than its content it WRAPS to
    a second line (FlowLayout; it scrolled sideways before 2026-10-29). The bar's stretches are dropped."""
    host = QtWidgets.QWidget()
    flow = FlowLayout(host, spacing=max(4, bar.spacing() if bar.spacing() >= 0 else 6))
    while bar.count():
        item = bar.takeAt(0)
        if item.widget() is not None:
            flow.addWidget(item.widget())
    sp = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
    sp.setHeightForWidth(True)
    host.setSizePolicy(sp)
    host.setMinimumWidth(120)
    return host


class FieldView3D(QtWidgets.QWidget):
    """Layered view; `set_state(state)` redraws from a reco_gui.field_session.FieldState (None clears)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layers: Dict[str, _Layer] = {}
        self._checks: Dict[str, QtWidgets.QCheckBox] = {}
        self._bounds = None
        self._camera_set = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        for name, (label, colour, _, tip) in LAYERS.items():
            box = QtWidgets.QCheckBox(label)
            box.setChecked(True)
            box.setToolTip(tip)
            swatch = "#%02x%02x%02x" % tuple(int(255 * c) for c in colour)
            box.setStyleSheet(f"QCheckBox {{ padding-left: 2px; }} QCheckBox::indicator:checked {{ background: {swatch}; }}")
            box.toggled.connect(lambda on, n=name: self._toggle(n, on))
            self._checks[name] = box
            bar.addWidget(box)
        bar.addStretch(1)
        for text, view in (("Iso", "iso"), ("Front", "front"), ("Top", "top"), ("Side", "side")):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.clicked.connect(lambda _=False, v=view: self.set_view(v))
            bar.addWidget(b)
        layout.addWidget(bar_row(bar))

        self.vtk_ok = False
        if os.environ.get("RECO_GUI_NO_3D"):
            layout.addWidget(self._stacked(QtWidgets.QLabel("3D view disabled (RECO_GUI_NO_3D).")), 1)
            return
        try:
            from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
            from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
            from vtkmodules.vtkRenderingCore import vtkRenderer
            import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
            import vtkmodules.vtkRenderingFreeType  # noqa: F401

            self.widget = QVTKRenderWindowInteractor(self)
            layout.addWidget(self._stacked(self.widget), 1)
            self.renderer = vtkRenderer()
            self.renderer.SetBackground(0.985, 0.985, 0.98)
            window = self.widget.GetRenderWindow()
            window.SetAlphaBitPlanes(1)
            window.SetMultiSamples(0)
            self.renderer.SetUseDepthPeeling(1)
            self.renderer.SetMaximumNumberOfPeels(10)
            self.renderer.SetOcclusionRatio(0.05)
            window.AddRenderer(self.renderer)
            self.interactor = window.GetInteractor()
            self.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
            self.interactor.Initialize()
            sizes = {"unexplained": dict(point_size=5.0), "cluster": dict(point_size=8.0), "field": dict(line_width=1.2)}
            for name, (_, colour, opacity, _) in LAYERS.items():
                self._layers[name] = _Layer(self.renderer, colour, opacity, **sizes.get(name, {}))
            self._add_overlay()
            self.vtk_ok = True
        except Exception as exc:
            layout.addWidget(QtWidgets.QLabel(f"3D view unavailable: {exc}"))

    # ------------------------------------------------------------------------------------------------------ token
    def _stacked(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """`widget` (the 3D view) behind the slowly turning logo token (reco_gui/logo_token.py): the token shows while
        the view is empty - no part loaded, no state, no followed run - and no run computes."""
        from reco_gui.logo_token import token_stack
        self._token_stack, self.token = token_stack(widget)
        self._sync_token()
        return self._token_stack

    def _content(self, shown: bool):
        self._empty = not shown
        self._sync_token()

    def set_running(self, running: bool):
        """A run computes (reco_gui/app.py: Start, the worker's busy signal): no token, even before its first state."""
        self._running = bool(running)
        self._sync_token()

    @property
    def token_shown(self) -> bool:
        stack = getattr(self, "_token_stack", None)
        return stack is not None and stack.currentWidget() is self.token

    def _sync_token(self):
        stack = getattr(self, "_token_stack", None)
        if stack is None:
            return
        idle = getattr(self, "_empty", True) and not getattr(self, "_running", False)
        stack.setCurrentIndex(0 if idle else 1)

    # ------------------------------------------------------------------------------------------------------ scene
    def _add_overlay(self):
        from vtkmodules.vtkRenderingCore import vtkTextActor
        self._text = vtkTextActor()
        self._text.GetTextProperty().SetFontSize(13)
        self._text.GetTextProperty().SetColor(0.1, 0.1, 0.1)
        self._text.GetTextProperty().SetVerticalJustificationToTop()
        self._text.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self._text.GetPositionCoordinate().SetValue(0.01, 0.99)
        self.renderer.AddViewProp(self._text)

    def _toggle(self, name: str, on: bool):
        if not self.vtk_ok:
            return
        self._layers[name].wanted = on
        self._apply_visibility()

    def _apply_visibility(self):
        for layer in self._layers.values():
            layer.actor.SetVisibility(1 if (layer.wanted and layer.has) else 0)
        self.widget.GetRenderWindow().Render()

    def set_view(self, view: str):
        if not self.vtk_ok or self._bounds is None:
            return
        lo, hi = self._bounds
        c = (lo + hi) / 2
        d = 2.6 * float(np.linalg.norm(hi - lo))
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*c)
        camera.SetViewUp(0, 0, 1)
        offset = {"iso": (0.62, -0.62, 0.5), "front": (0, -1, 0), "side": (1, 0, 0), "top": (0, 0, 1)}[view]
        camera.SetPosition(*(c + d * np.array(offset)))
        if view == "top":
            camera.SetViewUp(0, 1, 0)
        self.renderer.ResetCameraClippingRange()
        self.widget.GetRenderWindow().Render()

    def _overlay(self, state):
        step = state.step
        lines = textwrap.wrap(f"{step.stage}: {step.message}", 95)
        if state.ops:
            lines.append("model: " + " ".join(f"{o.op}:{o.kind}" for o in state.ops) + f"   explained {state.explained:.1%}")
        if state.candidate is not None and state.verdict != "accepted" and state.candidate.op != "base":
            c = state.candidate
            lines.append("candidate " + f"{c.op}:{c.kind}" + (f" ({state.verdict.upper()})" if state.verdict else ""))
        self._text.SetInput("\n".join(lines))

    def set_state(self, state):
        self._content(state is not None)
        if not self.vtk_ok:
            return
        if state is None:
            for layer in self._layers.values():
                layer.key, layer.has = None, False
                layer.actor.VisibilityOff()
            self._text.SetInput("")
            self._bounds, self._camera_set = None, False
            self.widget.GetRenderWindow().Render()
            return
        L = self._layers
        m = state.input_mesh
        if m is not None:
            L["input"].set(id(m), lambda: polydata_mesh(m.vertices, m.triangles))
            if self._bounds is None:
                self._bounds = (m.vertices.min(axis=0), m.vertices.max(axis=0))
        if state.vectors is not None:
            L["field"].set(id(state.vectors), lambda: polydata_lines(state.vector_points, state.vector_points + state.vectors))
        mm = state.model_mesh
        L["model"].set(id(mm) if mm is not None else None, lambda: polydata_mesh(mm.vertices, mm.triangles))
        surf = state.surface
        for name, idx in (("unexplained", state.unexplained), ("cluster", state.cluster)):
            L[name].set(id(idx) if (surf is not None and idx is not None) else None, lambda idx=idx: polydata_points(surf[idx]))
        cm = state.candidate_mesh
        L["candidate"].set(id(cm) if cm is not None else None, lambda: polydata_mesh(cm.vertices, cm.triangles))
        if cm is not None:
            colour = REJECTED if state.verdict == "rejected" else (CUTTER if state.inside else LAYERS["candidate"][1])
            L["candidate"].actor.GetProperty().SetColor(*colour)
        for layer in L.values():
            layer.actor.SetVisibility(1 if (layer.wanted and layer.has) else 0)
        self._overlay(state)
        if not self._camera_set and self._bounds is not None:
            self._camera_set = True
            self.set_view("iso")
        else:
            self.widget.GetRenderWindow().Render()

    def show_input(self, vertices: np.ndarray, triangles: np.ndarray, caption: str = ""):
        """The part alone, before a run (the preview of a picked file): everything else is cleared."""
        if not self.vtk_ok:
            self._content(True)
            return
        self.set_state(None)
        self._content(True)                                     # after the clearing: the part is on screen
        layer = self._layers["input"]
        self._shown_inputs = getattr(self, "_shown_inputs", 0) + 1      # never the array's id: Python reuses it after a
        layer.set(("preview", self._shown_inputs), lambda: polydata_mesh(vertices, triangles))   # free, the view stayed stale
        layer.actor.SetVisibility(1 if (layer.wanted and layer.has) else 0)
        self._text.SetInput(caption)
        self._bounds = (vertices.min(axis=0), vertices.max(axis=0))
        self._camera_set = True
        self.set_view("iso")

    def show_layer(self, name: str, mesh, colour=None, caption: Optional[str] = None):
        """One layer's mesh (None hides it), in `colour` if given - what a followed server run is trying now
        (reco_gui/server_follow.py: the candidate, the piece in work)."""
        if mesh is not None:
            self._content(True)
        if not self.vtk_ok or name not in self._layers:
            return
        layer = self._layers[name]
        if mesh is None:
            layer.actor.SetVisibility(0)
        else:
            v, t = np.asarray(mesh[0], float), np.asarray(mesh[1])
            self._shown_layers = getattr(self, "_shown_layers", 0) + 1
            layer.set((name, "live", self._shown_layers), lambda v=v, t=t: polydata_mesh(v, t))
            if colour is not None:
                layer.actor.GetProperty().SetColor(*colour)
            layer.actor.SetVisibility(1 if (layer.wanted and layer.has) else 0)
        if caption is not None:
            self._text.SetInput(caption)
        self.widget.GetRenderWindow().Render()

    def show_pair(self, input_mesh, model_mesh, caption: str = ""):
        """The input part (grey) and a model over it (the "model" layer where the view has one): a server run followed
        live (reco_gui/server_follow.py). Either mesh may be None (kept as it is)."""
        if input_mesh is not None or model_mesh is not None:
            self._content(True)
        if not self.vtk_ok:
            return
        self._shown_pairs = getattr(self, "_shown_pairs", 0) + 1
        first = getattr(self, "_pair_camera", None) is None
        for name, mesh in (("input", input_mesh), ("model", model_mesh)):
            if mesh is None or name not in self._layers:
                continue
            v, t = np.asarray(mesh[0], float), np.asarray(mesh[1])
            layer = self._layers[name]
            layer.set((name, "pair", self._shown_pairs), lambda v=v, t=t: polydata_mesh(v, t))
            layer.actor.SetVisibility(1 if (layer.wanted and layer.has) else 0)
            if name == "input" or self._bounds is None:
                self._bounds = (v.min(axis=0), v.max(axis=0))
        self._text.SetInput(caption)
        if first:
            self._pair_camera = True
            self._camera_set = True
            self.set_view("iso")
        else:
            self.widget.GetRenderWindow().Render()

    def grab_image(self, path: str) -> bool:
        if not self.vtk_ok:
            return False
        from vtkmodules.vtkIOImage import vtkPNGWriter
        from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
        window = self.widget.GetRenderWindow()
        window.Render()
        f = vtkWindowToImageFilter()
        f.SetInput(window)
        f.ReadFrontBufferOff()
        f.Update()
        w = vtkPNGWriter()
        w.SetFileName(path)
        w.SetInputConnection(f.GetOutputPort())
        w.Write()
        return True
