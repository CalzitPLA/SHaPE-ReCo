#!/usr/bin/env python3
"""
The feature phase live, in a small side window: the piece in work alone (not the whole part), what FreeCAD says about
its solid and its faces, and every guess of the algorithm as it arrives - source, tool, precision, recall, score and
the verdict - with the guess drawn over the piece.

`FeatureTrace` is Qt-free: it folds the cascade states into one record per piece (tested without a window);
`FeaturePanel` draws the record of the piece in work, or of the piece of a step looked back at.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

PIECE_COLOUR = (1.0, 0.8, 0.0)
GUESS_COLOURS = {None: (0.2, 0.55, 0.95), "accepted": (0.106, 0.686, 0.478), "rejected": (0.86, 0.2, 0.18)}
REAL = ("input", "both")             # origins of a face that is a surface of the part itself (a contact face)
COLUMNS = ("source", "tool", "precision", "recall", "score", "verdict")


def _num(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _fmt(x, digits: int = 3) -> str:
    x = _num(x)
    return "" if x is None else f"{x:.{digits}f}"


def piece_faces(piece: Optional[dict]) -> List[dict]:
    """The faces FreeCAD wrote for a piece: the piece's own "faces", else its entry in the pieces.json next to its STL
    (the loop's descriptor of a piece need not carry them)."""
    piece = piece or {}
    faces = piece.get("faces")
    if isinstance(faces, list) and faces:
        return [f for f in faces if isinstance(f, dict)]
    stl = piece.get("stl")
    if not stl:
        return []
    for folder in (Path(stl).parent, Path(stl).parent.parent):
        path = folder / "pieces.json"
        if path.is_file():
            try:
                listed = json.loads(path.read_text()).get("pieces") or []
            except (OSError, ValueError, AttributeError):
                return []
            for p in listed:
                if isinstance(p, dict) and p.get("stl") and os.path.basename(p["stl"]) == os.path.basename(stl):
                    return [f for f in (p.get("faces") or []) if isinstance(f, dict)]
    return []


def metadata_lines(piece: Optional[dict], faces: Optional[List[dict]] = None) -> List[str]:
    """What FreeCAD knows about the piece, as short lines: the solid (volume, size) and its faces by surface type,
    apart for the contact faces (surfaces of the part itself) and the faces only the model gives."""
    piece = piece or {}
    faces = piece_faces(piece) if faces is None else faces
    lines = []
    volume, bbox = _num(piece.get("volume")), piece.get("bbox")
    solid = []
    if volume is not None:
        solid.append(f"volume {volume:.4g} mm3")
    if bbox is not None:
        try:
            size = np.asarray(bbox[1], float) - np.asarray(bbox[0], float)
            solid.append("box " + " x ".join(f"{s:.4g}" for s in size) + " mm")
        except (TypeError, ValueError, IndexError):
            pass
    if piece.get("valid") is not None:
        solid.append("valid" if piece["valid"] else "NOT VALID")
    if solid:
        lines.append("solid: " + ", ".join(solid))
    if not faces:
        lines.append("faces: no metadata")
        return lines
    for label, chosen in (("contact faces (the part's)", [f for f in faces if f.get("origin") in REAL]),
                          ("model-only faces", [f for f in faces if f.get("origin") not in REAL])):
        if not chosen:
            lines.append(f"{label}: none")
            continue
        kinds = Counter(str(f.get("type") or "?") for f in chosen)
        area = sum(_num(f.get("area")) or 0.0 for f in chosen)
        text = ", ".join(f"{n} {k}" for k, n in kinds.most_common())
        radii = sorted({round(r, 2) for r in (_num(f.get("radius")) for f in chosen) if r is not None})
        lines.append(f"{label}: {text}; {area:.4g} mm2" + (f"; radii {', '.join(f'{r:g}' for r in radii[:6])}" if radii else ""))
    return lines


@dataclass
class PieceRecord:
    id: object
    title: str = ""
    piece: dict = field(default_factory=dict)
    metadata: List[str] = field(default_factory=list)
    mesh: object = None                          # MeshData of the piece
    guesses: List[dict] = field(default_factory=list)      # COLUMNS -> text, plus "mesh", "verdict"
    verdict: str = ""                            # "accepted" | "rejected" | ""
    reason: str = ""


class FeatureTrace:
    """One PieceRecord per piece, from the cascade states in the order they come; a state seen before changes nothing."""

    def __init__(self):
        self.records: Dict[object, PieceRecord] = {}
        self.order: List[object] = []
        self._seen = set()
        self.totals = Counter()

    def clear(self):
        self.__init__()

    def record_of(self, state) -> Optional[PieceRecord]:
        """The record of the piece the state is about (None outside the feature phase)."""
        info = getattr(state, "info", None) or {}
        pid = info.get("piece_id")
        if pid is None and isinstance(info.get("piece"), dict):
            pid = info["piece"].get("id")
        if pid is None and getattr(state, "feature", None) is not None and getattr(state, "in_feature_phase", False):
            pid = state.feature.current
        return self.records.get(pid)

    def add(self, state) -> Optional[PieceRecord]:
        stage = getattr(getattr(state, "step", None), "stage", "")
        if not stage.startswith("feature_") or stage in ("feature_residual", "feature_done", "feature_groups", "feature_confirm", "feature_confirmed"):
            return None
        info = getattr(state, "info", None) or {}
        given = info.get("piece") if isinstance(info.get("piece"), dict) else None
        pid = info.get("piece_id", (given or {}).get("id"))
        if pid is None:
            return None
        rec = self.records.get(pid)
        if rec is None:
            rec = self.records[pid] = PieceRecord(id=pid, title=f"piece #{pid}")
            self.order.append(pid)
        key = (getattr(state, "index", None), stage, pid)
        if key in self._seen:
            return rec
        self._seen.add(key)
        view = getattr(state, "feature", None)
        shown = next((p for p in (view.pieces if view is not None else ()) if p.id == pid), None)
        if shown is not None and shown.mesh is not None:
            rec.mesh = shown.mesh
        if stage == "feature_piece":
            rec.piece = dict(given or {})
            rec.metadata = metadata_lines(rec.piece)
            k, n = info.get("index"), info.get("of")
            what = "segment" if rec.piece.get("parent") is not None else "piece"
            rec.title = (f"{what} #{pid}" + (f" ({k} of {n})" if k and n else "")
                         + (f": {rec.piece.get('side', '')} -> {rec.piece.get('op', '')}" if rec.piece.get("op") else "")
                         + (f", congruent with #{rec.piece['group']}" if rec.piece.get("group") not in (None, pid) else ""))
        elif stage == "feature_candidate":
            cand = getattr(state, "candidate", None)
            rec.guesses.append(dict(source=str(info.get("source") or ""),
                                    tool=f"{info.get('op') or ''} {getattr(cand, 'kind', '') or ''}".strip(),
                                    precision=_fmt(info.get("precision")), recall=_fmt(info.get("recall")),
                                    score=_fmt(info.get("score")), verdict="", mesh=getattr(state, "candidate_mesh", None)))
            self.totals["guesses"] += 1
        elif stage in ("feature_accepted", "feature_rejected"):
            rec.verdict = stage[len("feature_"):]
            rec.reason = str(info.get("reason") or "")
            self.totals[rec.verdict] += 1
            if rec.reason.startswith("pattern"):
                self.totals["rejected by pattern"] += 1
            source = str(info.get("source") or "")
            best = [g for g in rec.guesses if g["source"] == source] or rec.guesses
            if stage == "feature_accepted" and best:
                best[-1]["verdict"] = "accepted"
            elif stage == "feature_rejected":
                for g in rec.guesses:
                    g["verdict"] = g["verdict"] or "rejected"
        return rec

    def summary(self) -> str:
        t = self.totals
        return (f"{len(self.order)} pieces, {t['guesses']} guesses: {t['accepted']} accepted, {t['rejected']} rejected"
                + (f" ({t['rejected by pattern']} of them only by the pattern of a failed piece)" if t["rejected by pattern"] else ""))


class FeaturePanel(QtWidgets.QWidget):
    """The side window: `set_state(state)` after every step (None clears)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trace = FeatureTrace()
        self._shown = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.title = QtWidgets.QLabel("No feature in work.")
        self.title.setStyleSheet("font-weight: bold;")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        self.meta = QtWidgets.QLabel("")
        self.meta.setWordWrap(True)
        self.meta.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.meta.setToolTip("FreeCAD's metadata of the piece: its solid, and its faces by surface type - the contact faces "
                             "are surfaces of the part itself, the others exist only because the model ends there")
        layout.addWidget(self.meta)
        self.vtk_ok = False
        self._actors = {}
        self._build_view(layout)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)   # all columns in view, no scrolling
        self.table.setToolTip("The guesses for this piece in the order they came; click one to draw it over the piece")
        self.table.currentCellChanged.connect(lambda row, *_: self._draw_guess(row))
        layout.addWidget(self.table, 1)
        self.verdict = QtWidgets.QLabel("")
        self.verdict.setWordWrap(True)
        layout.addWidget(self.verdict)
        self.total = QtWidgets.QLabel("")
        self.total.setWordWrap(True)
        self.total.setStyleSheet("color: #555;")
        layout.addWidget(self.total)

    # ------------------------------------------------------------------------------------------------------ 3D
    def _build_view(self, layout):
        if os.environ.get("RECO_GUI_NO_3D"):
            return
        try:
            from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
            from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
            from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer
            import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

            self.widget = QVTKRenderWindowInteractor(self)
            self.widget.setMinimumSize(200, 160)
            layout.addWidget(self.widget, 2)
            self.renderer = vtkRenderer()
            self.renderer.SetBackground(0.985, 0.985, 0.98)
            window = self.widget.GetRenderWindow()
            window.SetAlphaBitPlanes(1)
            window.SetMultiSamples(0)
            self.renderer.SetUseDepthPeeling(1)
            self.renderer.SetMaximumNumberOfPeels(8)
            window.AddRenderer(self.renderer)
            window.GetInteractor().SetInteractorStyle(vtkInteractorStyleTrackballCamera())
            window.GetInteractor().Initialize()
            for name, colour, opacity in (("piece", PIECE_COLOUR, 0.55), ("guess", GUESS_COLOURS[None], 0.35)):
                mapper = vtkPolyDataMapper()
                mapper.ScalarVisibilityOff()
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*colour)
                actor.GetProperty().SetOpacity(opacity)
                actor.VisibilityOff()
                self.renderer.AddActor(actor)
                self._actors[name] = (actor, mapper)
            self._actors["guess"][0].GetProperty().EdgeVisibilityOn()
            self._actors["guess"][0].GetProperty().SetLineWidth(0.5)
            self.vtk_ok = True
        except Exception as exc:
            layout.addWidget(QtWidgets.QLabel(f"3D view unavailable: {exc}"))

    def _set_mesh(self, name: str, mesh, colour=None):
        if not self.vtk_ok:
            return
        from reco_gui.viewer_field import polydata_mesh
        actor, mapper = self._actors[name]
        if mesh is None or len(mesh.vertices) == 0:
            actor.VisibilityOff()
            return
        mapper.SetInputData(polydata_mesh(np.asarray(mesh.vertices, float), np.asarray(mesh.triangles)))
        if colour is not None:
            actor.GetProperty().SetColor(*colour)
        actor.VisibilityOn()

    def _frame(self, mesh):
        """The camera on the piece (the guess may be far larger: a tool is open where the model closes the piece)."""
        if not self.vtk_ok or mesh is None or len(mesh.vertices) == 0:
            return
        v = np.asarray(mesh.vertices, float)
        lo, hi = v.min(axis=0), v.max(axis=0)
        c, d = (lo + hi) / 2, 2.4 * max(float(np.linalg.norm(hi - lo)), 1e-6)
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*c)
        camera.SetViewUp(0, 0, 1)
        camera.SetPosition(*(c + d * np.array((0.62, -0.62, 0.5))))
        self.renderer.ResetCameraClippingRange()

    def _draw_guess(self, row: int):
        rec = self.trace.records.get(self._shown)
        if rec is None or not (0 <= row < len(rec.guesses)):
            self._set_mesh("guess", None)
        else:
            g = rec.guesses[row]
            self._set_mesh("guess", g.get("mesh"), GUESS_COLOURS.get(g.get("verdict") or None, GUESS_COLOURS[None]))
        if self.vtk_ok:
            self.renderer.ResetCameraClippingRange()
            self.widget.GetRenderWindow().Render()

    # ------------------------------------------------------------------------------------------------------ state
    def set_state(self, state):
        if state is None:
            self.trace.clear()
            self._shown = None
            self.title.setText("No feature in work.")
            self.meta.setText("")
            self.verdict.setText("")
            self.total.setText("")
            self.table.setRowCount(0)
            for name in self._actors:
                self._set_mesh(name, None)
            if self.vtk_ok:
                self.widget.GetRenderWindow().Render()
            return
        rec = self.trace.add(state) or self.trace.record_of(state)
        self.total.setText(self.trace.summary() if self.trace.order else "")
        if rec is None:
            return
        new_piece = rec.id != self._shown
        self._shown = rec.id
        self.title.setText(rec.title)
        self.meta.setText("\n".join(rec.metadata))
        self.table.blockSignals(True)
        self.table.setRowCount(len(rec.guesses))
        for r, g in enumerate(rec.guesses):
            for c, name in enumerate(COLUMNS):
                item = QtWidgets.QTableWidgetItem(g.get(name, ""))
                if g.get("verdict"):
                    item.setForeground(QtGui.QBrush(QtGui.QColor(*(int(255 * x) for x in GUESS_COLOURS[g["verdict"]]))))
                self.table.setItem(r, c, item)
        if rec.guesses:
            self.table.setCurrentCell(len(rec.guesses) - 1, 0)
            self.table.scrollToBottom()
        self.table.blockSignals(False)
        text = {"accepted": "ACCEPTED", "rejected": "REJECTED"}.get(rec.verdict, "in work ...")
        self.verdict.setText(text + (f": {rec.reason}" if rec.reason else ""))
        colour = GUESS_COLOURS.get(rec.verdict or None)
        self.verdict.setStyleSheet("font-weight: bold; color: #%02x%02x%02x;" % tuple(int(255 * x) for x in colour))
        self._set_mesh("piece", rec.mesh)
        if new_piece:
            self._frame(rec.mesh)
        self._draw_guess(len(rec.guesses) - 1)
