#!/usr/bin/env python3
"""
3D view of the cascade (add-cut) reconstruction, in mm.

Main layers (always in the bar): the input part (grey), the model of the accepted operations (blue) and, once the run is
graded, the FreeCAD result - the real CSG solid FreeCAD built from the program, read back from its STL (steel blue,
opaque). The step-by-step layers are in the "Debug layers" menu: the residual as voxels - material still missing (green)
and material the model has in excess (red) -, the cut history (cut once / cut again), the leftover component being fitted
(yellow), the candidate primitive (orange; a cutter pink) and the voxel-cube fallbacks (purple).

The feature phase (live preview): every piece of the exact difference of the iteration as a small translucent body coloured by
its state - pending grey, the piece in work yellow (its real mesh; the others are drawn as their bounding boxes until they
have one), guessed light blue (its guess waits for the FreeCAD build), accepted green, rejected orange, skipped dark grey; a piece
that was split is drawn as its segments, each in its own colour with its state as the opacity and as an outline - and the guessed primitive as a wireframe over the piece in
work. `layer_meshes(state)` is the data side (what would be drawn), without VTK. The camera is set once per run; a step
never moves it (Iso / Front / Top / Side fit the part again).

"Final result" (checked automatically when the run finishes) hides the debug layers and shows the FreeCAD solid over the
input; unchecking it goes back to the step view. The Parts panel on the right lists every accepted operation with a
checkbox: a checked part is drawn on its own (adds in a cycling palette, cuts pink and translucent, fallbacks purple), and
the selected row is highlighted - so a program of many operations (the frame's I-beam boxes) can be taken apart.

Degrades to a text note without VTK/OpenGL (or with RECO_GUI_NO_3D set).
"""

from __future__ import annotations

import os
import textwrap
from typing import Dict, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from reco_gui.viewer_field import FieldView3D, _Layer, bar_row, polydata_lines, polydata_mesh

# name: (label, colour, opacity, tooltip)
MAIN_LAYERS = {
    "input":   ("Input part",     (0.55, 0.55, 0.53), 0.16, "the STL/STEP model as given (grey)"),
    "field":   ("Distance field", (0.20, 0.20, 0.20), 1.00, "the closest-point field V = c(x) - x near the surface: a line from "
                                                            "each sample to its nearest surface point, coloured by the signed "
                                                            "distance - the material flag: blue inside (material), red outside"),
    "model":   ("Model",          (0.165, 0.471, 0.839), 0.42, "the accepted analytic operations (blue)"),
    "freecad": ("FreeCAD result", (0.30, 0.40, 0.52), 1.00, "the solid FreeCAD built from the program (after the grade step)"),
    "distance": ("Distance",      (0.90, 0.90, 0.90), 1.00, "the input coloured by its exact distance to the model (the nephogram: "
                                                            "blue 0 -> green -> yellow -> red at the scale); from the toolbar's "
                                                            "Compare or the decision bar's export (2026-10-15)"),
}
DEBUG_LAYERS = {
    # the voxel cubes (the user's rule 2026-10-15: not on screen unless asked for; the settings' "voxel layers" builds them)
    "voxels":  ("Voxels",         (0.40, 0.52, 0.66), 0.30, "the input's voxelisation on the loop's grid (inside by ray parity) - "
                                                            "what the loop computes on; built only with the settings' 'voxel layers'"),
    "tau":     ("Voxels at \u03c4",  (0.10, 0.62, 0.55), 0.35, "the input's parity voxels at a pitch licensed by its own walls "
                                                            "(the 5th-percentile wall thickness tau, D-005) over the part's box; "
                                                            "built only with the settings' 'voxel layers'"),
    "fallback":  ("Voxel fallback", (0.482, 0.247, 0.702), 0.55, "accepted voxel-cube fallbacks (purple)"),
    "pos":       ("Missing (+)", (0.106, 0.686, 0.478), 0.55, "positive residual: material the input has and the model lacks (green voxels)"),
    "neg":       ("Excess (-)",  (0.890, 0.286, 0.282), 0.55, "negative residual: material the model has and the input lacks (red voxels)"),
    "cut1":      ("Cut once",    (0.400, 0.745, 0.953), 0.20, "cut history: voxels emptied by exactly one cut (light blue)"),
    "cut2":      ("Cut again",   (0.050, 0.180, 0.560), 0.70, "cut history: voxels emptied by two or more cuts (dark blue): "
                                                              "overlapping or redundant cutters"),
    "component": ("Component",   (0.980, 0.780, 0.100), 0.75, "the leftover component being fitted now (yellow voxels)"),
    "candidate": ("Candidate",   (0.922, 0.408, 0.204), 0.60, "the primitive being tried: orange for an addition, pink for a cut, "
                                                              "purple for a voxel-cube fallback"),
}
# the feature phase: the pieces of the exact difference by state (cascade_session.PIECE_STATES), and the guess (wireframe)
FEATURE_LAYERS = {
    "f_pending":   ("Pieces: pending",  (0.55, 0.55, 0.58), 0.30, "feature phase: pieces of the exact difference not looked at "
                                                                 "yet (grey; their bounding boxes)"),
    "f_current":   ("Piece in work",    (1.000, 0.800, 0.000), 0.80, "feature phase: the piece being guessed now (yellow; its "
                                                                 "real mesh)"),
    "f_guessed":   ("Pieces: guessed",  (0.400, 0.700, 0.950), 0.45, "feature phase: pieces with a guess that waits for its "
                                                                 "FreeCAD build (light blue)"),
    "f_accepted":  ("Pieces: accepted", (0.106, 0.686, 0.478), 0.45, "feature phase: pieces whose guess was applied and passed "
                                                                 "the FreeCAD build (green)"),
    "f_rejected":  ("Pieces: rejected", (0.950, 0.520, 0.100), 0.45, "feature phase: pieces whose guesses were rejected (orange)"),
    "f_skipped":   ("Pieces: skipped",  (0.30, 0.30, 0.33), 0.25, "feature phase: pieces left without a verdict (a limit ended "
                                                              "the phase, or no guess; dark grey)"),
    "f_segments":  ("Segments",         (0.60, 0.60, 0.60), 1.00, "feature phase: the segments a piece was split into, each in "
                                                                 "its own colour; the opacity is the state (the segment "
                                                                 "in work opaque, with its voxel surface; rejected faint)"),
    "f_outline":   ("Segments: state",  (0.30, 0.30, 0.30), 1.00, "feature phase: the segments' boxes as outlines in the "
                                                                 "colour of their state (grey pending, yellow in work, "
                                                                 "light blue guessed, green accepted, orange rejected)"),
    "f_candidate": ("Feature guess",    (0.922, 0.408, 0.204), 1.00, "feature phase: the guessed primitive as a wireframe over "
                                                                 "the piece in work (orange: an addition, pink: a cut, dark "
                                                                 "red: rejected)"),
}
COLOURED_LAYERS = ("f_segments", "f_outline", "distance")   # a colour per triangle (cascade_session.ColouredMesh)
PIECE_LAYER = {"segments": "f_segments", "segment_outline": "f_outline", "pending": "f_pending", "current": "f_current", "guessed": "f_guessed", "accepted": "f_accepted", "rejected": "f_rejected",
               "skipped": "f_skipped"}
DEBUG_LAYERS.update(FEATURE_LAYERS)                 # toggled in the "Debug layers" menu, hidden by "Final result"
LAYERS = {**MAIN_LAYERS, **DEBUG_LAYERS}
DEBUG_DEFAULT_ON = ("fallback", "candidate") + tuple(FEATURE_LAYERS)   # the cube layers (pos, neg, component, cuts) off
MAIN_DEFAULT_OFF = ("distance",)                # all main layers start visible (voxels and field included), the nephogram waits for its export
FINAL_HIDES = ("field",)                               # the final-result view shows the solid, not the input's field
CUTTER = (0.910, 0.482, 0.643)
FALLBACK = DEBUG_LAYERS["fallback"][1]
REJECTED = (0.55, 0.20, 0.20)
HIGHLIGHT = (1.0, 0.55, 0.0)
PART_PALETTE = [(0.20, 0.47, 0.84), (0.10, 0.62, 0.47), (0.85, 0.60, 0.10), (0.42, 0.35, 0.80),
                (0.12, 0.60, 0.72), (0.60, 0.45, 0.25), (0.35, 0.55, 0.20), (0.70, 0.30, 0.55)]


def is_fallback(p) -> bool:
    return any(getattr(s, "mode", "") == "voxel_fallback" for s in (getattr(p, "sources", None) or ()))


def box_mesh(p):
    """Exact triangles of a schema box (centre, quaternion, half extents): 8 corners, 12 triangles - no level set, so the
    many thin plates of an I-beam program draw instantly and with sharp edges."""
    from reco_gui.cascade_session import MeshData, quaternion_matrix
    d = p.dims_mm or {}
    h = np.array([d.get("x_half", 0.0), d.get("y_half", 0.0), d.get("z_half", 0.0)], float)
    if not (h > 0).all():
        return None
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * h
    v = corners @ quaternion_matrix(p.quaternion).T + np.asarray(p.center_mm, float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    f = np.array([t for a, b, c, e in quads for t in ((a, b, c), (a, c, e))], np.int64)
    return MeshData(v, f)


def layer_meshes(state) -> Dict[str, object]:
    """{layer: MeshData} of everything `CascadeView3D.set_state(state)` draws as a mesh (the field lines and the FreeCAD
    result are not meshes of the state) - the data side of the view, Qt- and VTK-free. In the feature phase the pieces are
    in their state's layer and the guess is in "f_candidate" (a wireframe) instead of "candidate"."""
    out = {name: mesh for name, mesh in (
        ("input", state.input_mesh), ("voxels", getattr(state, "input_voxel_mesh", None)),
        ("tau", getattr(state, "tau_voxel_mesh", None)), ("model", state.model_mesh), ("fallback", state.fallback_mesh),
        ("pos", state.residual_pos_mesh), ("neg", state.residual_neg_mesh), ("cut1", state.cut_once_mesh),
        ("cut2", state.recut_mesh), ("component", state.component_mesh), ("candidate", state.candidate_mesh))
        if mesh is not None}
    if getattr(state, "in_feature_phase", False):
        guess = out.pop("candidate", None)
        if guess is not None:
            out["f_candidate"] = guess
        for piece_state, mesh in (state.feature.layers or {}).items():
            if piece_state in PIECE_LAYER and mesh is not None:
                out[PIECE_LAYER[piece_state]] = mesh
    return out


def polydata_of(mesh):
    """The vtkPolyData of a MeshData; a ColouredMesh brings its colour per triangle (red, green, blue, opacity)."""
    poly = polydata_mesh(mesh.vertices, mesh.triangles)
    colours = getattr(mesh, "colours", None)
    if colours is not None:
        from vtkmodules.util.numpy_support import numpy_to_vtk
        from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR
        scalars = numpy_to_vtk(np.ascontiguousarray(colours, dtype=np.uint8), deep=True, array_type=VTK_UNSIGNED_CHAR)
        scalars.SetName("colours")
        poly.GetCellData().SetScalars(scalars)
    return poly


def candidate_colour(state) -> tuple:
    return (FALLBACK if state.candidate_fallback else REJECTED if state.verdict == "rejected"
            else CUTTER if state.candidate_op == "cut" else DEBUG_LAYERS["candidate"][1])


def part_label(index: int, op) -> str:
    p = op.primitive
    name = getattr(p, "label", None)
    if not name:
        dims = ", ".join(f"{k.replace('_half', '').replace('_', ' ')} {v:.3g}" for k, v in (p.dims_mm or {}).items())
        name = f"{p.kind} ({dims})" if dims else p.kind
    return f"#{index + 1}  {op.op}  {name}" + (" [fallback]" if is_fallback(p) else "")


class CascadeView3D(FieldView3D):
    """`set_state(state)` redraws from a reco_gui.cascade_session.CascadeState (None clears)."""

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self._layers: Dict[str, _Layer] = {}
        self._checks: Dict[str, QtWidgets.QCheckBox] = {}
        self._debug_actions: Dict[str, QtWidgets.QAction] = {}
        self._bounds = None
        self._camera_set = False
        self._state = None
        self._program: tuple = ()
        self._part_actors: Dict[int, object] = {}
        self._part_mesher = None
        self._freecad_path: Optional[str] = None
        self._final_auto_done = False

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        for name, (label, colour, _, tip) in MAIN_LAYERS.items():
            box = QtWidgets.QCheckBox(label)
            box.setChecked(name not in MAIN_DEFAULT_OFF)
            box.setToolTip(tip)
            swatch = "#%02x%02x%02x" % tuple(int(255 * c) for c in colour)
            box.setStyleSheet(f"QCheckBox::indicator:checked {{ background: {swatch}; }}")
            box.toggled.connect(lambda on, n=name: self._toggle(n, on))
            self._checks[name] = box
            bar.addWidget(box)
        debug = QtWidgets.QToolButton()
        debug.setText("Debug layers")
        debug.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(debug)
        for name, (label, _, _, tip) in DEBUG_LAYERS.items():
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(name in DEBUG_DEFAULT_ON)
            act.setToolTip(tip)
            act.toggled.connect(lambda on, n=name: self._toggle(n, on))
            self._debug_actions[name] = act
        debug.setMenu(menu)
        bar.addWidget(debug)
        self.final_box = QtWidgets.QCheckBox("Final result")
        self.final_box.setToolTip("hide the step-by-step layers and show the FreeCAD solid over the input "
                                  "(checked automatically when the run finishes)")
        self.final_box.toggled.connect(self._final_toggled)
        bar.addWidget(self.final_box)
        bar.addStretch(1)
        for text, view in (("Iso", "iso"), ("Front", "front"), ("Top", "top"), ("Side", "side")):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.clicked.connect(lambda _=False, v=view: self.set_view(v))
            bar.addWidget(b)
        outer.addWidget(bar_row(bar))

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(split, 1)
        view_host = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(view_host)
        layout.setContentsMargins(0, 0, 0, 0)
        split.addWidget(view_host)
        split.addWidget(self._build_parts_panel())
        split.setStretchFactor(0, 1)
        split.setSizes([900, 260])

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

            self.widget = QVTKRenderWindowInteractor(view_host)
            layout.addWidget(self._stacked(self.widget), 1)        # the logo token while the view is empty
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
            for name, (_, colour, opacity, _) in LAYERS.items():
                self._layers[name] = _Layer(self.renderer, colour, opacity)
            for name in DEBUG_LAYERS:
                self._layers[name].wanted = name in DEBUG_DEFAULT_ON
            for name in MAIN_DEFAULT_OFF:
                self._layers[name].wanted = False
            self._setup_field_layer()
            for name in ("f_candidate", "f_outline"):
                wire = self._layers[name].actor.GetProperty()
                wire.SetRepresentationToWireframe()
                wire.SetLineWidth(2.0)
                wire.LightingOff()
            for name in COLOURED_LAYERS:                                  # the colours come with the mesh
                mapper = self._layers[name].mapper
                mapper.SetScalarModeToUseCellData()
                mapper.SetColorModeToDirectScalars()
                mapper.ScalarVisibilityOn()
            self._layers["f_segments"].actor.ForceTranslucentOn()
            self._add_overlay()
            self.vtk_ok = True
        except Exception as exc:
            layout.addWidget(QtWidgets.QLabel(f"3D view unavailable: {exc}"))

    def _setup_field_layer(self):
        """The field layer colours each line by its signed distance: a diverging lookup table, blue (inside) to red."""
        from vtkmodules.vtkCommonCore import vtkLookupTable
        layer = self._layers["field"]
        layer.actor.GetProperty().SetLineWidth(1.3)
        lut = vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        for i in range(256):
            t = i / 255.0
            r, g, b = (0.15 + 0.85 * t, 0.35 * (1 - abs(2 * t - 1)) + 0.25, 1.0 - 0.85 * t)
            lut.SetTableValue(i, r, g, b, 1.0)
        lut.Build()
        layer.mapper.SetLookupTable(lut)
        layer.mapper.SetScalarModeToUseCellData()
        layer.mapper.ScalarVisibilityOn()

    def _field_polydata(self, arrows):
        from vtkmodules.util.numpy_support import numpy_to_vtk
        pts = np.asarray(arrows["points"], float)
        poly = polydata_lines(pts, pts + np.asarray(arrows["vectors"], float))
        d = np.asarray(arrows["sdf"], float)
        # outside is negative in the stored channel (positive inside): red = outside, blue = inside
        scalars = numpy_to_vtk(np.ascontiguousarray(-d), deep=True)
        poly.GetCellData().SetScalars(scalars)
        lim = float(np.abs(d).max()) if len(d) else 1.0
        self._layers["field"].mapper.SetScalarRange(-lim, lim)
        return poly

    # ------------------------------------------------------------------------------------------------ parts panel
    def _build_parts_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(4, 0, 0, 0)
        v.addWidget(QtWidgets.QLabel("<b>Parts</b> (accepted operations)"))
        row = QtWidgets.QHBoxLayout()
        for text, which in (("All", "all"), ("None", "none"), ("Adds", "add"), ("Cuts", "cut")):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.clicked.connect(lambda _=False, w=which: self.check_parts(w))
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.parts = QtWidgets.QListWidget()
        self.parts.setToolTip("check a part to draw it on its own; the selected part is highlighted (orange)")
        self.parts.itemChanged.connect(lambda _item: self._update_parts())
        self.parts.currentRowChanged.connect(lambda _row: self._update_parts())
        v.addWidget(self.parts, 1)
        return panel

    def _sync_parts(self, program):
        """Rebuild the list only when the accepted program changed; keep the user's checks for parts still there."""
        if tuple(id(o) for o in program) == tuple(id(o) for o in self._program):
            return
        checked = {self.parts.item(i).text() for i in range(self.parts.count())
                   if self.parts.item(i).checkState() == QtCore.Qt.Checked}
        for actor in self._part_actors.values():                      # indices may shift: rebuild the actors lazily
            if self.vtk_ok:
                self.renderer.RemoveActor(actor)
        self._part_actors.clear()
        self._program = tuple(program)
        self.parts.blockSignals(True)
        self.parts.clear()
        for i, op in enumerate(program):
            item = QtWidgets.QListWidgetItem(part_label(i, op))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if item.text() in checked else QtCore.Qt.Unchecked)
            self.parts.addItem(item)
        self.parts.blockSignals(False)
        self._update_parts()

    def check_parts(self, which: str):
        """Check all parts, none, only the adds or only the cuts."""
        self.parts.blockSignals(True)
        for i in range(self.parts.count()):
            on = which == "all" or (which in ("add", "cut") and self._program[i].op == which)
            self.parts.item(i).setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)
        self.parts.blockSignals(False)
        self._update_parts()

    def _part_mesh(self, op):
        p = op.primitive
        if p.kind == "box":
            return box_mesh(p)
        if self._part_mesher is None:
            from reco_gui.cascade_session import CascadeFolder
            self._part_mesher = CascadeFolder()
            if self._bounds is not None:
                lo, hi = self._bounds
                pad = 0.08 * float((hi - lo).max())
                self._part_mesher._crop = (lo - pad, hi + pad)      # a cutter reaching far out is cropped to the part
        return self._part_mesher.primitive_mesh(p)

    def _update_parts(self):
        if not self.vtk_ok:
            return
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        current = self.parts.currentRow()
        wanted = {i for i in range(self.parts.count()) if self.parts.item(i).checkState() == QtCore.Qt.Checked}
        if 0 <= current < self.parts.count():
            wanted.add(current)
        for i in list(self._part_actors):
            if i not in wanted:
                self.renderer.RemoveActor(self._part_actors.pop(i))
        n_add = 0
        for i, op in enumerate(self._program):
            fb = is_fallback(op.primitive)
            colour = FALLBACK if fb else CUTTER if op.op == "cut" else PART_PALETTE[n_add % len(PART_PALETTE)]
            if op.op == "add" and not fb:
                n_add += 1
            if i not in wanted:
                continue
            actor = self._part_actors.get(i)
            if actor is None:
                mesh = self._part_mesh(op)
                if mesh is None:
                    continue
                mapper = vtkPolyDataMapper()
                mapper.SetInputData(polydata_mesh(mesh.vertices, mesh.triangles))
                mapper.ScalarVisibilityOff()
                actor = vtkActor()
                actor.SetMapper(mapper)
                self.renderer.AddActor(actor)
                self._part_actors[i] = actor
            prop = actor.GetProperty()
            prop.SetColor(*(HIGHLIGHT if i == current else colour))
            prop.SetOpacity(1.0 if i == current else (0.45 if op.op == "cut" else 0.85))
        self.widget.GetRenderWindow().Render()

    # ------------------------------------------------------------------------------------------------ final result
    def _final_toggled(self, on: bool):
        if not self.vtk_ok:
            return
        for name, act in self._debug_actions.items():
            self._layers[name].wanted = (not on) and act.isChecked()
        for name in FINAL_HIDES:
            self._layers[name].wanted = self._checks[name].isChecked() and not on
        # the FreeCAD solid replaces the (level-set) model in the final view; without it the model stays
        self._layers["model"].wanted = self._checks["model"].isChecked() and not (on and self._layers["freecad"].has)
        self._apply_visibility()
        if self._state is not None:
            self._overlay(self._state)
            self.widget.GetRenderWindow().Render()

    def _toggle(self, name: str, on: bool):
        if not self.vtk_ok:
            return
        if name in DEBUG_LAYERS and self.final_box.isChecked():
            return                                                    # the final view keeps the debug layers hidden
        self._layers[name].wanted = on
        if name == "model" and self.final_box.isChecked() and self._layers["freecad"].has:
            self._layers[name].wanted = False
        self._apply_visibility()

    def _load_freecad(self, result):
        path = getattr(result, "stl_path", None) if result is not None else None
        if not path or path == self._freecad_path or not os.path.isfile(path):
            return
        self._freecad_path = path
        try:
            from src.closest_point_field import load_mesh
            v, f = load_mesh(path, 0.02)
            self._layers["freecad"].set(("freecad", path),
                                        lambda: polydata_mesh(np.asarray(v, float), np.asarray(f, np.int64)))
        except Exception:
            self._layers["freecad"].set(None)

    # ------------------------------------------------------------------------------------------------ drawing
    def _overlay(self, state):
        step = state.step
        if self.final_box.isChecked() and state.result is not None:
            r = state.result
            ops = [o for o in r.operations if o.accepted]
            valid = ops[-1].freecad_valid if ops else None
            lines = [f"final result: {len(ops)} operations "
                     f"({sum(o.op == 'add' for o in ops)} adds, {sum(o.op == 'cut' for o in ops)} cuts)",
                     "FreeCAD: " + ("valid solid" if valid else "INVALID" if valid is False else "not built yet")
                     + (f", chamfer {r.chamfer_mm:.4g} mm" if r.chamfer_mm is not None else ""),
                     f"residual +{r.residual_voxels_pos} / -{r.residual_voxels_neg} voxels"
                     + (f", {r.recut_voxels} cut again" if getattr(r, "recut_voxels", 0) else "")]
            self._text.SetInput("\n".join(lines))
            return
        lines = textwrap.wrap(f"{step.stage}: {step.message}", 95)
        if state.program:
            lines.append(f"{len(state.program)} operations   residual +{state.residual_pos} / -{state.residual_neg} voxels"
                         + (f", {state.recut_voxels} cut again" if state.recut_voxels else ""))
        if getattr(state, "in_feature_phase", False):
            from reco_gui.cascade_session import feature_progress
            f = state.feature
            counts = f.counts()
            lines.append((feature_progress(state) or f"feature phase, iteration {f.iteration}")
                         + "   pieces: " + ", ".join(f"{n} {k}" for k, n in counts.items() if n)
                         + (f"   exact IoU {f.exact_iou:.4f}" if f.exact_iou is not None else ""))
        if state.candidate is not None and state.verdict != "accepted":
            lines.append(f"candidate {state.candidate_op}:{getattr(state.candidate, 'label', None) or state.candidate.kind}"
                         + (" (fallback)" if state.candidate_fallback else "")
                         + (f" ({state.verdict.upper()})" if state.verdict else ""))
        self._text.SetInput("\n".join(lines))

    def show_distance(self, mesh, caption: str = "", legend=None, unit: str = "mm") -> None:
        """The nephogram layer: `mesh` a ColouredMesh of the input with a colour per triangle (reco_gui/nephogram.py);
        None clears it. Switched on when set; it stays until the next run clears the view. `legend`: the discrete
        bands [{lo, hi, text, rgb}] drawn as a colour bar at the right edge, one block per band with its value range."""
        if mesh is not None:
            self._content(True)
        if not self.vtk_ok:
            return
        layer = self._layers["distance"]
        layer.set(id(mesh) if mesh is not None else None, lambda m=mesh: polydata_of(m))
        layer.wanted = mesh is not None
        self._show_legend(legend if mesh is not None else None, unit)
        box = self._checks.get("distance")
        if box is not None:
            box.blockSignals(True)
            box.setChecked(mesh is not None)
            box.blockSignals(False)
        if caption:
            self._text.SetInput(caption)
        self._apply_visibility()
        self.widget.GetRenderWindow().Render()

    def _show_legend(self, legend, unit: str = "mm") -> None:
        """A vtkScalarBarActor with one discrete colour per band, labelled by the band's lower value; removed with None."""
        bar = getattr(self, "_legend_actor", None)
        if bar is not None:
            self.renderer.RemoveViewProp(bar)
            self._legend_actor = None
        if not legend:
            return
        from vtkmodules.vtkCommonCore import vtkLookupTable
        from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
        n = len(legend)
        lut = vtkLookupTable()
        lut.SetNumberOfTableValues(n)
        lut.SetRange(0.0, float(n))
        for b in legend:
            r, g, bl = (c / 255.0 for c in b["rgb"])
            lut.SetTableValue(b["band"], r, g, bl, 1.0)
        lut.Build()
        bar = vtkScalarBarActor()
        bar.SetLookupTable(lut)
        bar.SetTitle(f"distance ({unit})")
        bar.SetNumberOfLabels(0)
        bar.SetMaximumNumberOfColors(n)
        # one annotation per band at its centre: the band's value range, so every colour is tied to its numbers
        try:
            lut.ResetAnnotations()
            for b in legend:
                lut.SetAnnotation(float(b["band"]) + 0.5, b["text"])
            bar.DrawAnnotationsOn()
        except AttributeError:
            pass
        bar.SetOrientationToVertical()
        bar.SetPosition(0.86, 0.12)
        bar.SetWidth(0.08)
        bar.SetHeight(0.75)
        bar.GetTitleTextProperty().SetColor(0.1, 0.1, 0.1)
        bar.GetLabelTextProperty().SetColor(0.1, 0.1, 0.1)
        bar.GetAnnotationTextProperty().SetColor(0.1, 0.1, 0.1)
        bar.GetAnnotationTextProperty().SetFontSize(11)
        self.renderer.AddViewProp(bar)
        self._legend_actor = bar

    def set_state(self, state):
        self._state = state
        self._content(state is not None)
        if state is None:
            self._program = ()
            self.parts.clear()
            self._final_auto_done = False
            self.final_box.blockSignals(True)
            self.final_box.setChecked(False)
            self.final_box.blockSignals(False)
        else:
            self._sync_parts(state.program or ())
            self.final_box.blockSignals(True)
            if state.finished and not self._final_auto_done:          # the run ended: show the final result once
                self._final_auto_done = True
                self.final_box.setChecked(True)
            elif not state.finished and self._final_auto_done:        # stepping back in the timeline: the step view
                self._final_auto_done = False
                self.final_box.setChecked(False)
            self.final_box.blockSignals(False)
        if not self.vtk_ok:
            return
        if state is None:
            for layer in self._layers.values():
                layer.key, layer.has = None, False
                layer.actor.VisibilityOff()
            for actor in self._part_actors.values():
                self.renderer.RemoveActor(actor)
            self._part_actors.clear()
            self._part_mesher = None
            self._freecad_path = None
            self._final_toggled(False)
            self._text.SetInput("")
            self._show_legend(None)
            self._last_input = None
            self._bounds, self._camera_set = None, False
            self.widget.GetRenderWindow().Render()
            return
        L = self._layers
        if state.input_mesh is not None and self._bounds is None:
            self._bounds = (state.input_mesh.vertices.min(axis=0), state.input_mesh.vertices.max(axis=0))
        arrows = getattr(state, "field_arrows", None)
        L["field"].set(id(arrows) if arrows is not None and len(arrows.get("points", ())) else None,
                       lambda a=arrows: self._field_polydata(a))
        meshes = layer_meshes(state)
        # the input part stays on screen through every step (the user's rule 2026-10-15): a state without its own
        # input mesh shows the last one seen; only the user's checkbox hides it
        if meshes.get("input") is not None:
            self._last_input = meshes["input"]
        elif getattr(self, "_last_input", None) is not None:
            meshes["input"] = self._last_input
        for name in LAYERS:
            if name in ("field", "freecad", "distance"):               # the nephogram layer is set by show_distance
                continue
            mesh = meshes.get(name)
            L[name].set(id(mesh) if mesh is not None else None, lambda m=mesh: polydata_of(m))
        self._load_freecad(state.result)
        if self._bounds is None:
            for mesh in (state.model_mesh, state.component_mesh, state.residual_pos_mesh):
                if mesh is not None:
                    self._bounds = (mesh.vertices.min(axis=0), mesh.vertices.max(axis=0))
                    break
        if state.candidate_mesh is not None:
            for name in ("candidate", "f_candidate"):
                L[name].actor.GetProperty().SetColor(*candidate_colour(state))
        self._update_parts()                                          # the actors need the bounds set above
        self._final_toggled(self.final_box.isChecked())
        for layer in L.values():
            layer.actor.SetVisibility(1 if (layer.wanted and layer.has) else 0)
        self._overlay(state)
        if not self._camera_set and self._bounds is not None:
            self._camera_set = True
            self.set_view("iso")
        else:
            self.widget.GetRenderWindow().Render()
