#!/usr/bin/env python3
"""
Open a stored run (the user's rule 2026-09-29: "create an open option to load old runs"): the runs of the run store
(reco_engine.run_store; $SHAPE_RECO_RESULTS or ~/shape_reco_results) in a table, newest first, and what a chosen one
shows in the window - its model (model.stl), its summary (part, code, settings that differ from nothing: all of them,
metrics, program) and its log (steps.jsonl). A stored run is looked at, not continued: it has no states to step
through.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5 import QtCore, QtWidgets

COLUMNS = ("saved", "part", "label", "method", "operations", "exact IoU", "chamfer mm", "finished", "commit", "branch")


def _number(v, fmt: str) -> str:
    return "" if v is None else format(float(v), fmt)


def run_row(run: Dict) -> List[str]:
    m, code = run.get("metrics") or {}, run.get("code") or {}
    return [str(run.get("timestamp") or "")[:19].replace("T", " "), str(run.get("part") or ""), str(run.get("label") or ""),
            str(run.get("method") or ""), str(len(run.get("program") or [])), _number(m.get("exact_iou"), ".4f"),
            _number(m.get("chamfer_mm"), ".3g"), "yes" if run.get("finished") else "no",
            str(code.get("short") or "") + ("*" if code.get("dirty") else ""), str(code.get("branch") or "")]


def list_rows(store=None) -> List[tuple]:
    """[(run directory, the row's texts)] of every run of the store, newest first."""
    from reco_engine import run_store
    return [(run["path"], run_row(run)) for run in run_store.list_runs(store=store)]


def log_lines(run_dir) -> List[str]:
    """The run's steps as log lines, from steps.jsonl (a run without the file: from run.json's steps)."""
    path = Path(run_dir) / "steps.jsonl"
    steps = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                steps.append(json.loads(line))
            except ValueError:
                continue
    out = []
    for s in steps:
        took = f" ({s['seconds']:.1f} s)" if isinstance(s.get("seconds"), (int, float)) else ""
        out.append(f"{int(s.get('index', len(out))) + 1:3d} {str(s.get('phase') or ''):7s} {str(s.get('stage') or ''):16s} "
                   f"{s.get('message') or ''}{took}")
    return out


def summary_html(run: Dict) -> str:
    e = html.escape
    code, inp, m = run.get("code") or {}, run.get("input") or {}, run.get("metrics") or {}
    rows = [("run", run.get("name")), ("saved", run.get("timestamp")), ("label", run.get("label")),
            ("input", inp.get("path")), ("method", run.get("method")), ("finished", run.get("finished")),
            ("code", f"{code.get('short')} on {code.get('branch')}" + (" (modified)" if code.get("dirty") else "")),
            ("directory", run.get("path"))]
    if run.get("program_truncated"):                  # cut to text when it was saved (run store before 2026-10-01)
        rows.append(("program", f"recovered ({run['program_source']}); run.json's was cut in {run['program_truncated']} node(s)"
                     if run.get("program_source") != "run.json" else
                     f"cut in {run['program_truncated']} node(s) when it was saved: it cannot be built "
                     "(scripts/recover_run_program.py)"))
    parts = ["<h3>Stored run</h3><table>"] + [f"<tr><td><b>{e(str(k))}</b>&nbsp;</td><td>{e(str(v))}</td></tr>"
                                              for k, v in rows if v is not None] + ["</table>"]
    flat = [(k, v) for k, v in m.items() if not isinstance(v, (dict, list)) and v is not None]
    if flat:
        parts += ["<h3>Metrics</h3><table>"] + [f"<tr><td>{e(k)}&nbsp;</td><td>{e(f'{v:.6g}' if isinstance(v, float) else str(v))}</td></tr>"
                                                for k, v in flat] + ["</table>"]
    program = run.get("program") or []
    if program:
        parts.append(f"<h3>Program ({len(program)} operations)</h3><ol>")
        for o in program:
            prim = o.get("primitive") or {}
            name = prim.get("label") or prim.get("kind") or o.get("kind") or ""
            dims = ", ".join(f"{k} {v:.4g}" for k, v in (prim.get("dims_mm") or {}).items() if isinstance(v, (int, float)))
            parts.append(f"<li>{e(str(o.get('op') or ''))} {e(str(name))}" + (f" <span style='color:#666'>({e(dims)})</span>" if dims else "") + "</li>")
        parts.append("</ol>")
    settings = run.get("settings") or {}
    if settings:
        parts += ["<h3>Settings</h3><table>"] + [f"<tr><td>{e(str(k))}&nbsp;</td><td>{e(str(v))}</td></tr>"
                                                 for k, v in sorted(settings.items())
                                                 if not isinstance(v, (dict, list))] + ["</table>"]
    return "".join(parts)


def load_stored(run_dir, load_mesh=None) -> Dict:
    """Everything the window shows of a stored run: dict(run, html, log, mesh (vertices, triangles) or None, caption,
    note). `load_mesh(path)` replaces src.closest_point_field.load_mesh (tests)."""
    from reco_engine import run_store
    run = run_store.load_run(run_dir)
    directory = Path(run["path"])
    stl = directory / str((run.get("model") or {}).get("stl") or "model.stl")
    mesh, note = None, None
    if stl.is_file():
        if load_mesh is None:
            from src.closest_point_field import load_mesh
        try:
            v, f = load_mesh(str(stl))
            mesh = (v, f)
        except Exception as exc:
            note = f"the model could not be read: {exc}"
    else:
        note = "the run has no model (it was saved without the FreeCAD rebuild)"
    m = run.get("metrics") or {}
    caption = (f"stored run {run.get('name')}: {run.get('part')}\n{len(run.get('program') or [])} operations"
               + (f", exact IoU {m['exact_iou']:.4f}" if m.get("exact_iou") is not None else "")
               + (f", chamfer {m['chamfer_mm']:.3g} mm" if m.get("chamfer_mm") is not None else "")
               + ("" if run.get("finished") else ", not finished"))
    return dict(run=run, html=summary_html(run), log=log_lines(directory), mesh=mesh, caption=caption, note=note)


class OpenRunDialog(QtWidgets.QDialog):
    """The store's runs in a table; `chosen` is the run directory after Open (or a double click)."""

    def __init__(self, parent=None, store=None, rows=None):
        super().__init__(parent)
        self.setWindowTitle("Open a stored run")
        self.chosen: Optional[str] = None
        self._rows = list_rows(store) if rows is None else list(rows)
        layout = QtWidgets.QVBoxLayout(self)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("filter: part, label, branch ...")
        self.filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter)
        self.table = QtWidgets.QTableWidget(len(self._rows), len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        for r, (path, texts) in enumerate(self._rows):
            for c, text in enumerate(texts):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(path)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.doubleClicked.connect(lambda index: self._accept())
        layout.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        self.count = QtWidgets.QLabel(f"{len(self._rows)} run(s)")
        row.addWidget(self.count)
        row.addStretch(1)
        other = QtWidgets.QPushButton("Other directory ...")
        other.clicked.connect(self._browse)
        row.addWidget(other)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Open | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        layout.addLayout(row)
        if self._rows:
            self.table.selectRow(0)
        self.resize(980, 460)

    def _apply_filter(self, text: str):
        words = text.lower().split()
        shown = 0
        for r, (path, texts) in enumerate(self._rows):
            hay = " ".join(texts).lower()
            hide = not all(w in hay for w in words)
            self.table.setRowHidden(r, hide)
            shown += not hide
        self.count.setText(f"{shown} of {len(self._rows)} run(s)")

    def _browse(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "A run directory (with run.json)")
        if path and (Path(path) / "run.json").is_file():
            self.chosen = path
            self.accept()
        elif path:
            QtWidgets.QMessageBox.warning(self, "No run", f"{path} has no run.json")

    def _accept(self):
        r = self.table.currentRow()
        if r < 0 or self.table.isRowHidden(r):
            return
        self.chosen = self._rows[r][0]
        self.accept()
