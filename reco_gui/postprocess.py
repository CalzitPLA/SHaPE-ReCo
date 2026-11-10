#!/usr/bin/env python3
"""
The post-processing window (the user, 2026-10-30: "a postprocessing window which shows the user the tree of the
reconstruction, the mass ratio of the subparts; interactively the user is able to select parts and then benchmark,
decide and export"; and "minimize the number of parts: simple trees for simple parts instead of complex trees").

A finished reconstruction - the window's newest result or a stored run - is shown as its boolean tree (op_tree.csg_tree:
Fusion / Cut / Common over the primitives) with, per part:
    mass    the share of the final model's volume the part holds (a point held by several adds counts for each by an
            equal fraction, so the shares add up to 100 %); a cut shows the volume it removes for good
    unique  the volume only this part provides: what disappears when it is dropped
    fit     an add: the share of its mass inside the input; a cut: the share of what it removes that lies outside it
Selected parts are benchmarked against the input: the program as it is, without the selection, and as decided (sampled
IoU, missing and invented volume, operations). Decide drops or keeps parts; "Minimize parts" proposes the drops that cost
at most a tolerance of IoU (greedy, the cheapest first). The decided program is checked exactly in FreeCAD on demand and
exported: STEP, FreeCAD groups, GDML, the program as JSON, a run of the run store.

Everything is measured on one set of sample points (reco_engine/prune.py, Qt-free: the loop's pruning step uses it too).

    s = sample_program(result, mesh=(V, F))                 # Qt-free core
    rows = part_table(s, keep); m = metrics(s, keep); keep2, log = minimize(s, keep, tol=0.002)
    w = PostprocessWindow(result, input_path=path)           # the window
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from reco_engine.prune import (SAMPLES, TOLERANCE, Sample, accepted_ops, edited_result, fold,  # noqa: E402,F401
                               group_share, input_inside, metrics, minimize, part_table, sample_program)


def part_name(i: int, o) -> str:
    p = o.primitive
    label = (getattr(p, "label", None) or "").split(",")[0]
    dims = ", ".join(f"{k.replace('_half', '').replace('_', ' ')} {v:.3g}" for k, v in (p.dims_mm or {}).items())
    return f"#{i + 1} {o.op} {p.kind}" + (f" ({dims})" if dims and not label else "") + (f" - {label}" if label else "")


def source_of(o) -> str:
    srcs = getattr(o.primitive, "sources", None) or []
    modes = [getattr(s, "mode", "") for s in srcs]
    feat = [m for m in modes if m.startswith("feature:")]
    return (feat[-1][len("feature:"):] if feat else modes[-1] if modes else "")


# ------------------------------------------------------------------------------------------------ the window

from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

DROP, KEEP = "drop", "keep"
COL_PART, COL_MASS, COL_UNIQUE, COL_FIT, COL_SOURCE, COL_DECISION = range(6)
ADD_COLOUR, CUT_COLOUR, SELECT_COLOUR, DROP_COLOUR = (0.20, 0.47, 0.84), (0.85, 0.30, 0.25), (1.0, 0.55, 0.0), (0.55, 0.55, 0.55)


class _BarDelegate(QtWidgets.QStyledItemDelegate):
    """The mass column: a bar as long as the share, the number on it."""

    def paint(self, painter, option, index):
        share = index.data(QtCore.Qt.UserRole)
        if share is None:
            return super().paint(painter, option, index)
        painter.save()
        r = option.rect.adjusted(2, 3, -2, -3)
        cut = bool(index.data(QtCore.Qt.UserRole + 1))
        painter.fillRect(r, QtGui.QColor("#eef0f2"))
        w = int(r.width() * max(0.0, min(1.0, float(share))))
        painter.fillRect(QtCore.QRect(r.left(), r.top(), w, r.height()), QtGui.QColor("#e2a29b" if cut else "#9db9e6"))
        painter.setPen(QtGui.QColor("#1b1f24"))
        painter.drawText(r.adjusted(4, 0, 0, 0), QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft,
                         index.data(QtCore.Qt.DisplayRole) or "")
        painter.restore()


class _Worker(QtCore.QThread):
    done = QtCore.pyqtSignal(object, str)          # result (or None), error
    said = QtCore.pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(self.said.emit), "")
        except Exception as exc:
            self.done.emit(None, f"{type(exc).__name__}: {exc}"[-400:])


class PostView3D(QtWidgets.QWidget):
    """The input (translucent) and the parts: kept ones in their colour, the selection orange, dropped ones grey."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.show_input_box = QtWidgets.QCheckBox("input")
        self.show_input_box.setChecked(True)
        self.show_dropped_box = QtWidgets.QCheckBox("dropped parts")
        self.show_dropped_box.setChecked(False)
        for b in (self.show_input_box, self.show_dropped_box):
            b.toggled.connect(lambda _on: self.refresh())
            bar.addWidget(b)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.vtk_ok, self._actors, self._input_actor, self._bounds = False, {}, None, None
        self._mesher, self._meshes, self.state = None, {}, ([], [], set())
        if os.environ.get("RECO_GUI_NO_3D"):
            lay.addWidget(QtWidgets.QLabel("3D view disabled (RECO_GUI_NO_3D)."), 1)
            return
        try:
            from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
            from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
            from vtkmodules.vtkRenderingCore import vtkRenderer
            import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
            self.widget = QVTKRenderWindowInteractor(self)
            lay.addWidget(self.widget, 1)
            self.renderer = vtkRenderer()
            self.renderer.SetBackground(0.985, 0.985, 0.98)
            window = self.widget.GetRenderWindow()
            window.AddRenderer(self.renderer)
            self.renderer.SetUseDepthPeeling(1)
            interactor = window.GetInteractor()
            interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
            interactor.Initialize()
            self.vtk_ok = True
        except Exception as exc:
            lay.addWidget(QtWidgets.QLabel(f"3D view unavailable: {exc}"), 1)

    def _actor(self, v, f, colour, opacity):
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        from reco_gui.viewer_field import polydata_mesh
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata_mesh(np.asarray(v, float), np.asarray(f)))
        mapper.ScalarVisibilityOff()
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*colour)
        actor.GetProperty().SetOpacity(opacity)
        self.renderer.AddActor(actor)
        return actor

    def set_input(self, mesh):
        if not self.vtk_ok or mesh is None:
            return
        if self._input_actor is not None:
            self.renderer.RemoveActor(self._input_actor)
        v = np.asarray(mesh[0], float)
        self._bounds = (v.min(axis=0), v.max(axis=0))
        self._input_actor = self._actor(v, mesh[1], (0.6, 0.6, 0.6), 0.18)
        self.renderer.ResetCamera()
        self.widget.GetRenderWindow().Render()

    def _mesh(self, op):
        key = id(op)
        if key not in self._meshes:
            p = op.primitive
            from reco_gui.viewer_cascade import box_mesh
            m = box_mesh(p) if p.kind == "box" else None
            if m is None:
                if self._mesher is None:
                    from reco_gui.cascade_session import CascadeFolder
                    self._mesher = CascadeFolder()
                    if self._bounds is not None:
                        lo, hi = self._bounds
                        pad = 0.08 * float((hi - lo).max())
                        self._mesher._crop = (lo - pad, hi + pad)      # a cutter reaching far out: cropped to the part
                m = self._mesher.primitive_mesh(p)
            self._meshes[key] = m
        return self._meshes[key]

    def set_parts(self, ops, keep, selected):
        self.state = (list(ops), list(keep), set(selected))
        self.refresh()

    def refresh(self):
        if not self.vtk_ok:
            return
        ops, keep, selected = self.state
        for a in self._actors.values():
            self.renderer.RemoveActor(a)
        self._actors.clear()
        if self._input_actor is not None:
            self._input_actor.SetVisibility(1 if self.show_input_box.isChecked() else 0)
        for i, o in enumerate(ops):
            sel, kept = i in selected, keep[i] if i < len(keep) else True
            if not kept and not sel and not self.show_dropped_box.isChecked():
                continue
            m = self._mesh(o)
            if m is None:
                continue
            colour = SELECT_COLOUR if sel else DROP_COLOUR if not kept else CUT_COLOUR if o.op == "cut" else ADD_COLOUR
            opacity = 0.95 if sel else 0.25 if not kept else 0.45 if o.op == "cut" else 0.8
            self._actors[i] = self._actor(m.vertices, m.triangles, colour, opacity)
        self.widget.GetRenderWindow().Render()


class PostprocessWindow(QtWidgets.QMainWindow):
    """The post-processing window of one result: tree with mass ratios, selection, benchmark, decision, export."""

    def __init__(self, result, input_path=None, mesh=None, label: str = "", parent=None, samples: int = SAMPLES,
                 settings=None):
        super().__init__(parent)
        self.result, self.input_path, self.mesh, self.label = result, input_path, mesh, label
        self.settings, self.samples = settings, samples
        self.ops = accepted_ops(result)
        self.keep = np.ones(len(self.ops), bool)
        self.proposed: set = set()
        self.sample: Optional[Sample] = None
        self.rows: List[dict] = []
        self._items: Dict[int, QtWidgets.QTreeWidgetItem] = {}
        self._threads = []
        name = Path(str(input_path or getattr(result, "input_path", "") or "part")).name
        self.setWindowTitle(f"Post-processing - {name}" + (f" ({label})" if label else ""))
        self.resize(1400, 860)
        self._build()
        self._fill_tree()
        self._start(self._prepare, self._prepared, "sampling the program and the input ...")

    # -------------------------------------------------------------------------------------------- layout
    def _build(self):
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(QtWidgets.QLabel("<b>The reconstruction's tree</b> - select parts (Ctrl / Shift for several; a group "
                                      "selects its parts)"))
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["Part", "Mass", "Unique mm³", "Fit", "Source", "Decision"])
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree.setItemDelegateForColumn(COL_MASS, _BarDelegate(self.tree))
        self.tree.header().setSectionResizeMode(COL_PART, QtWidgets.QHeaderView.Stretch)
        self.tree.setColumnWidth(COL_MASS, 150)
        self.tree.headerItem().setToolTip(COL_MASS, "an add: its share of the final volume (shared material split equally); "
                                                    "a cut: the volume it removes for good, as a share of the final volume")
        self.tree.headerItem().setToolTip(COL_UNIQUE, "the volume only this part provides: what disappears when it is dropped")
        self.tree.headerItem().setToolTip(COL_FIT, "an add: the share of its mass inside the input; a cut: the share of what "
                                                   "it removes that lies outside the input")
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        lv.addWidget(self.tree, 1)
        row = QtWidgets.QHBoxLayout()
        self.b_drop = QtWidgets.QPushButton("Drop selected")
        self.b_keep = QtWidgets.QPushButton("Keep selected")
        self.b_reset = QtWidgets.QPushButton("Keep all")
        self.b_min = QtWidgets.QPushButton("Minimize parts")
        self.b_min.setToolTip("drop the parts the shape does not need: one at a time, the cheapest first, while the IoU stays "
                              "within the tolerance of the program as it came; the drops are proposed, then decided by you")
        self.tol = QtWidgets.QDoubleSpinBox()
        self.tol.setDecimals(4)
        self.tol.setRange(0.0, 0.05)
        self.tol.setSingleStep(0.0005)
        self.tol.setValue(TOLERANCE)
        self.tol.setPrefix("IoU tolerance ")
        self.b_accept = QtWidgets.QPushButton("Accept proposal")
        self.b_accept.setEnabled(False)
        for b, fn in ((self.b_drop, lambda: self._decide(DROP)), (self.b_keep, lambda: self._decide(KEEP)),
                      (self.b_reset, self._reset), (self.b_min, self._minimize), (self.b_accept, self._accept_proposal)):
            b.clicked.connect(fn)
        for wdg in (self.b_drop, self.b_keep, self.b_reset):
            row.addWidget(wdg)
        row.addStretch(1)
        for wdg in (self.b_min, self.tol, self.b_accept):
            row.addWidget(wdg)
        lv.addLayout(row)
        split.addWidget(left)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.view = PostView3D()
        right.addWidget(self.view)
        bench = QtWidgets.QWidget()
        bv = QtWidgets.QVBoxLayout(bench)
        bv.setContentsMargins(4, 4, 4, 4)
        bv.addWidget(QtWidgets.QLabel("<b>Benchmark</b> against the input (sampled; the exact check builds it in FreeCAD)"))
        self.report = QtWidgets.QTextBrowser()
        bv.addWidget(self.report, 1)
        ex = QtWidgets.QHBoxLayout()
        self.b_exact = QtWidgets.QPushButton("Exact check")
        self.b_exact.setToolTip("the decided program built in FreeCAD: one valid solid? its volume and faces")
        self.b_exact.clicked.connect(self._exact)
        ex.addWidget(self.b_exact)
        ex.addStretch(1)
        ex.addWidget(QtWidgets.QLabel("Export the decided program:"))
        self.b_step = QtWidgets.QPushButton("STEP ...")
        self.b_groups = QtWidgets.QPushButton("FreeCAD groups ...")
        self.b_gdml = QtWidgets.QPushButton("GDML ...")
        self.b_json = QtWidgets.QPushButton("Program JSON ...")
        self.b_save = QtWidgets.QPushButton("Save as run")
        self.b_save.setToolTip("into the run store as a run labelled 'post-processed': the Benchmark tab and Compare see it")
        for b, fn in ((self.b_step, self.export_step), (self.b_groups, self.export_groups), (self.b_gdml, self.export_gdml),
                      (self.b_json, self.export_json), (self.b_save, self.save_run)):
            b.clicked.connect(lambda _=False, f=fn: f())
            ex.addWidget(b)
        bv.addLayout(ex)
        right.addWidget(bench)
        right.setSizes([520, 340])
        split.addWidget(right)
        split.setSizes([700, 700])
        self.setCentralWidget(split)
        self.status = self.statusBar()

    # -------------------------------------------------------------------------------------------- the tree
    def _fill_tree(self):
        from reco_gui.op_tree import csg_tree
        self.tree.clear()
        self._items.clear()
        root = csg_tree(self.ops)
        if root is None:
            self.tree.addTopLevelItem(QtWidgets.QTreeWidgetItem(["(no operations)"]))
            return

        def add(node, parent):
            if node[0] == "prim":
                i, o = node[1], node[2]
                item = QtWidgets.QTreeWidgetItem([part_name(i, o), "", "", "", source_of(o), KEEP])
                item.setData(0, QtCore.Qt.UserRole, ("prim", i))
                self._items[i] = item
            elif node[0] == "empty":
                item = QtWidgets.QTreeWidgetItem(["(nothing to cut from)"])
                item.setData(0, QtCore.Qt.UserRole, ("empty",))
            else:
                title = {"fuse": "Fusion", "cut": "Cut", "common": "Common"}[node[0]]
                item = QtWidgets.QTreeWidgetItem([f"{title} ({len(node[1])})"])
                item.setData(0, QtCore.Qt.UserRole, ("group", node))
                f = item.font(0)
                f.setBold(True)
                item.setFont(0, f)
                for child in node[1]:
                    add(child, item)
            (parent.addChild(item) if parent is not None else self.tree.addTopLevelItem(item))
            return item

        add(root, None)
        self.tree.expandAll()

    def _refresh_tree(self):
        if not self.rows:
            return
        for i, item in self._items.items():
            r = self.rows[i]
            cut = r["op"] in ("cut", "intersect")
            vol = r["removed"] if cut else r["mass"]
            item.setText(COL_MASS, f"{100 * r['share']:.2f} %  ({'−' if cut else ''}{vol:.4g} mm³)" if r["kept"] else "-")
            item.setData(COL_MASS, QtCore.Qt.UserRole, r["share"] if r["kept"] else None)
            item.setData(COL_MASS, QtCore.Qt.UserRole + 1, cut)
            item.setText(COL_UNIQUE, "" if cut or not r["kept"] else f"{r['unique']:.4g}")
            item.setText(COL_FIT, "" if r["fit"] is None else f"{100 * r['fit']:.1f} %")
            decision = KEEP if self.keep[i] else DROP
            if i in self.proposed:
                decision = "drop (proposed)"
            item.setText(COL_DECISION, decision)
            grey = QtGui.QColor("#9aa0a6") if not self.keep[i] or i in self.proposed else QtGui.QColor("#1b1f24")
            f = item.font(COL_PART)
            f.setStrikeOut(not self.keep[i])
            item.setFont(COL_PART, f)
            for c in range(6):
                item.setForeground(c, grey)
        it = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while it.value():
            item = it.value()
            d = item.data(0, QtCore.Qt.UserRole)
            if d and d[0] == "group":
                a, b = group_share(d[1], self.rows)
                item.setText(COL_MASS, f"{100 * a:.2f} %" + (f"  (cuts −{100 * b:.2f} %)" if b else ""))
                item.setData(COL_MASS, QtCore.Qt.UserRole, a)
            it += 1

    def selected_parts(self) -> List[int]:
        out = set()
        for item in self.tree.selectedItems():
            d = item.data(0, QtCore.Qt.UserRole)
            if not d:
                continue
            if d[0] == "prim":
                out.add(d[1])
            elif d[0] == "group":
                stack = [d[1]]
                while stack:
                    n = stack.pop()
                    if n[0] == "prim":
                        out.add(n[1])
                    elif n[0] in ("fuse", "cut", "common"):
                        stack.extend(n[1])
        return sorted(out)

    def select_parts(self, indices):
        self.tree.clearSelection()
        for i in indices:
            if i in self._items:
                self._items[i].setSelected(True)

    # -------------------------------------------------------------------------------------------- computing
    def _start(self, fn, then, text):
        self.status.showMessage(text)
        t = _Worker(fn, self)
        t.said.connect(self.status.showMessage)
        t.done.connect(then)
        self._threads = [x for x in self._threads if x.isRunning()] + [t]
        t.start()

    def wait(self, ms: int = 120_000):
        """Wait for the running computations (tests, closing)."""
        for t in list(self._threads):
            t.wait(ms)
        QtWidgets.QApplication.processEvents()

    def _prepare(self, say):
        mesh = self.mesh
        if mesh is None and self.input_path and Path(str(self.input_path)).is_file():
            say("loading the input ...")
            from src.closest_point_field import load_field
            field = load_field(str(self.input_path))
            mesh = (np.asarray(field.vertices, float), np.asarray(field.faces))
        return mesh, sample_program(self.result, mesh, n=self.samples, progress=say)

    def _prepared(self, out, err):
        if err:
            self.status.showMessage(f"sampling failed: {err}")
            self.report.setHtml(f"<p>The program could not be sampled: {err}</p>")
            return
        self.mesh, self.sample = out
        self.view.set_input(self.mesh)
        self.view.set_parts(self.ops, self.keep, [])
        self.start_metrics = metrics(self.sample, np.ones(len(self.ops), bool))
        self._recompute()
        self.status.showMessage(f"{len(self.ops)} parts, {len(self.sample.points):,} sample points"
                                + ("" if self.sample.input is not None else " - no input mesh: no IoU, no fit"))

    def _recompute(self):
        if self.sample is None:
            return
        self.rows = part_table(self.sample, self.keep)
        self._refresh_tree()
        self._update_report()

    def _selection_changed(self):
        sel = self.selected_parts()
        self.view.set_parts(self.ops, self.keep, sel)
        self._update_report()

    def _update_report(self):
        if self.sample is None:
            return
        sel = self.selected_parts()
        current = metrics(self.sample, self.keep)
        cols = [("as it came", self.start_metrics), ("as decided", current)]
        if sel:
            without = self.keep.copy()
            without[sel] = False
            cols.insert(1, ("without the selection", metrics(self.sample, without)))
        if self.proposed:
            prop = self.keep.copy()
            prop[sorted(self.proposed)] = False
            cols.append(("with the proposal", metrics(self.sample, prop)))

        def f(x, fmt):
            return "-" if x is None else fmt.format(x)
        h = ["<table border='1' cellspacing='0' cellpadding='3'><tr><th></th>" + "".join(f"<th>{t}</th>" for t, _ in cols)
             + "</tr>"]
        for key, title, fmt in (("operations", "parts", "{:d}"), ("iou", "IoU (sampled)", "{:.4f}"),
                                ("missing", "missing mm³", "{:.4g}"), ("invented", "invented mm³", "{:.4g}"),
                                ("volume", "volume mm³", "{:.4g}")):
            h.append(f"<tr><td>{title}</td>" + "".join(f"<td>{f(m.get(key), fmt)}</td>" for _, m in cols) + "</tr>")
        h.append("</table>")
        if sel:
            rs = [self.rows[i] for i in sel]
            adds = [r for r in rs if r["op"] not in ("cut", "intersect")]
            cuts = [r for r in rs if r["op"] in ("cut", "intersect")]
            h.append(f"<p><b>Selection:</b> {len(sel)} part(s) - {len(adds)} add(s) holding "
                     f"{100 * sum(r['share'] for r in adds):.2f} % of the volume (unique {sum(r['unique'] for r in adds):.4g} mm³)"
                     + (f", {len(cuts)} cut(s) removing {sum(r['removed'] for r in cuts):.4g} mm³" if cuts else "") + ".</p>")
        if getattr(self, "exact_text", None):
            h.append(f"<p><b>Exact check:</b> {self.exact_text}</p>")
        self.report.setHtml("".join(h))

    # -------------------------------------------------------------------------------------------- deciding
    def _decide(self, what):
        sel = self.selected_parts()
        if not sel:
            self.status.showMessage("select parts first")
            return
        if what == DROP:
            trial = self.keep.copy()
            trial[sel] = False
            if not any(trial[i] and self.ops[i].op not in ("cut", "intersect") for i in range(len(self.ops))):
                self.status.showMessage("not every add can go: the program needs one")
                return
            self.keep = trial
        else:
            self.keep[sel] = True
        self.proposed -= set(sel)
        self.exact_text = None
        self._recompute()
        self.view.set_parts(self.ops, self.keep, sel)

    def _reset(self):
        self.keep[:] = True
        self.proposed.clear()
        self.b_accept.setEnabled(False)
        self.exact_text = None
        self._recompute()
        self.view.set_parts(self.ops, self.keep, self.selected_parts())

    def _minimize(self):
        if self.sample is None:
            return
        tol, keep, s = self.tol.value(), self.keep.copy(), self.sample
        self.b_min.setEnabled(False)
        self._start(lambda say: minimize(s, keep, tol, progress=say), self._minimized,
                    f"minimizing the parts (IoU tolerance {tol:g}) ...")

    def _minimized(self, out, err):
        self.b_min.setEnabled(True)
        if err:
            self.status.showMessage(f"minimize failed: {err}")
            return
        keep, log = out
        self.proposed = {d["index"] for d in log}
        self.b_accept.setEnabled(bool(self.proposed))
        self._recompute()
        self.select_parts(sorted(self.proposed))
        self.status.showMessage(f"Minimize parts: {len(self.proposed)} of {int(self.keep.sum())} parts can go "
                                f"(IoU tolerance {self.tol.value():g}); they are selected - Accept proposal drops them")

    def _accept_proposal(self):
        if self.proposed:
            self.keep[sorted(self.proposed)] = False
            self.proposed.clear()
        self.b_accept.setEnabled(False)
        self.exact_text = None
        self._recompute()
        self.view.set_parts(self.ops, self.keep, self.selected_parts())

    def decided(self):
        """The decided program as a ReconstructionResult."""
        return edited_result(self.result, self.keep)

    # -------------------------------------------------------------------------------------------- exact check, export
    def _exact(self):
        result = self.decided()
        self.b_exact.setEnabled(False)

        def run(say):
            import tempfile
            from reco_field.export import build_program
            from reco_engine.result_to_program import program_dict
            say("building the decided program in FreeCAD ...")
            t = time.time()
            ex = build_program(program_dict(result), tempfile.mkdtemp(prefix="postprocess_"), "decided")
            return ex, time.time() - t
        self._start(run, self._exact_done, "exact check: FreeCAD builds the decided program ...")

    def _exact_done(self, out, err):
        self.b_exact.setEnabled(True)
        if err:
            self.exact_text = f"failed: {err}"
        else:
            ex, secs = out
            solid = "one valid solid" if ex.valid and ex.n_solids == 1 else f"{ex.n_solids} solid(s), valid {ex.valid}"
            self.exact_text = f"{solid}, volume {ex.volume:.6g} mm³, {ex.n_faces} faces ({secs:.0f} s)"
            self.exact_step = ex.step
        self._update_report()

    def _save_path(self, title, suffix, filt):
        stem = Path(str(self.input_path or getattr(self.result, "input_path", "") or "part")).stem
        start = str(Path.home() / f"{stem}_post{suffix}")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, start, filt)
        return path

    def export_json(self, path=None):
        path = path or self._save_path("Program as JSON", ".json", "JSON (*.json)")
        if not path:
            return None
        Path(path).write_text(json.dumps(self.decided().to_dict(), indent=1))
        self.status.showMessage(f"written: {path}")
        return path

    def export_gdml(self, path=None):
        path = path or self._save_path("Export GDML", ".gdml", "GDML (*.gdml)")
        if not path:
            return None
        from reco_field.gdml_export import export_result
        info = export_result(self.decided(), path)
        self.status.showMessage(f"GDML written: {info['path']} ({info['primitives']} primitives)")
        return path

    def export_step(self, path=None):
        path = path or self._save_path("Export STEP", ".step", "STEP (*.step *.stp)")
        if not path:
            return None
        result = self.decided()

        def run(say):
            import shutil
            import tempfile
            from reco_field.export import build_program
            from reco_engine.result_to_program import program_dict
            ex = build_program(program_dict(result), tempfile.mkdtemp(prefix="postprocess_"), "decided")
            shutil.copy(ex.step, path)
            return ex
        self._start(run, lambda ex, err: self.status.showMessage(
            f"STEP export failed: {err}" if err else f"STEP written: {path} ({ex.n_solids} solid(s), valid {ex.valid})"),
            "STEP: FreeCAD builds the decided program ...")
        return path

    def export_groups(self, out_dir=None):
        out_dir = out_dir or QtWidgets.QFileDialog.getExistingDirectory(self, "FreeCAD groups into", str(Path.home()))
        if not out_dir:
            return None
        result = self.decided()

        def run(say):
            from reco_gui.rebuild import groups
            return groups(result, out_dir)
        self._start(run, lambda r, err: self.status.showMessage(
            f"FreeCAD groups failed: {err}" if err else f"FreeCAD document: {(r or {}).get('fcstd')}"),
            "FreeCAD groups: writing the document ...")
        return out_dir

    def save_run(self, store=None):
        from reco_engine import run_store
        decided = self.decided()
        dropped = [i + 1 for i, k in enumerate(self.keep) if not k]
        extra = dict(postprocess=dict(source=self.label or None, dropped=dropped, parts_before=len(self.ops),
                                      parts_after=int(self.keep.sum()),
                                      sampled=metrics(self.sample, self.keep) if self.sample is not None else None))
        kw = {} if store is None else dict(store=store)
        path = run_store.save_run(decided, self.settings, self.input_path or decided.input_path, label="post-processed",
                                  extra=extra, **kw)
        self.status.showMessage(f"saved as a run: {path}")
        return path


def _load(run_dir, input_path=None):
    """(result, input path, label) of a run directory: a run of scripts/run_part.py (program.json, record.json) or of
    the run store (run.json)."""
    from reco_engine.result_schema import ReconstructionResult
    d = Path(run_dir).expanduser()
    if (d / "run.json").is_file():
        from reco_engine import benchmark as bm, run_store
        run = run_store.load_run(d)
        return bm.result_of(run), input_path or (run.get("input") or {}).get("path"), d.name
    result = ReconstructionResult.from_dict(json.loads((d / "program.json").read_text()))
    rec = json.loads((d / "record.json").read_text()) if (d / "record.json").is_file() else {}
    return result, input_path or rec.get("stl") or rec.get("input") or result.input_path, d.name


def main(argv=None):
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="the post-processing window of a stored run")
    ap.add_argument("run", help="a run directory: scripts/run_part.py's (program.json, record.json) or the run store's")
    ap.add_argument("--input", default=None, help="the input mesh (STL or STEP), when the run's record does not name it")
    a = ap.parse_args(argv)
    if os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"        # VTK's OpenGL widget needs X11 (as reco_gui/__main__.py; else BadWindow)
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "dataset_gen")]
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    result, inp, label = _load(a.run, a.input)
    w = PostprocessWindow(result, input_path=inp, label=label)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
