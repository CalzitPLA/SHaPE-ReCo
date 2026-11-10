#!/usr/bin/env python3
"""
The benchmark in the window (the user, 2026-10-11: benchmarking runs with aegir on the framework's own results after
the run, not on files from the servers, and lets the user choose the most suitable result for his application): the
stored runs of the current part (reco_engine.benchmark: by the input's content), one row per run with what it measured
against the input, its path variant and - where a known tree of the part exists - its aegir rating, ranked for the
chosen preference. A click on a row shows that run (model, summary, trees); "Rate with aegir" rates the unrated runs
in the background; "Use this result" labels the selected run "chosen" and shows it.

The choice between alternatives (the user, 2026-10-29: "show iteratively different reconstructions, start with 3 to
select by default; benchmark their event time with aegir; select via nephogram / histogram and benchmark, like the
presentation"): every row has a check box - the newest ALTERNATIVES runs of the part are checked by default -;
"Event time (aegir)" scans the checked runs without one (reco_engine/event_time.py: navigation us per event, the
deck's method) in the background; "Nephogram" opens the selected run's nephogram (made after the run, or now for an
older run: reco_gui/nephogram.stored_nephogram); below the table the selected run's histograms (always with the
nephogram) and the checked runs' accuracy against their event time.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from reco_engine import benchmark as bm

COLUMNS = ("rank", "saved", "label", "path", "ops", "valid", "exact IoU", "voxel IoU", "chamfer mm", "missing", "invented",
           "seconds", "us/event", "p95 missing", "p95 invented", "tier", "separation", "failed")
RIGHT = (0, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14)               # the number columns, right-aligned
TIER_COLUMN = 15
ALTERNATIVES = 3                                              # runs checked by default: the newest of the part
TIER_COLOURS = dict(zip("ABCDE", ("#1c5cab", "#3987e5", "#86b6ef", "#eda100", "#e34948")))


def _num(v, fmt: str) -> str:
    return "" if v is None else format(float(v), fmt)


def row_texts(i: int, r: dict) -> List[str]:
    """The table's texts of one benchmark row (reco_engine.benchmark.row_of), rank i."""
    return [str(i), r.get("saved") or "", r.get("label") or "", r.get("variant") or "", str(r.get("ops") or 0),
            "" if r.get("one_valid_solid") is None else ("yes" if r["one_valid_solid"] else "NO"),
            _num(r.get("exact_iou"), ".4f"), _num(r.get("voxel_iou"), ".4f"), _num(r.get("chamfer_mm"), ".3g"),
            _num(r.get("missing_volume"), ".3g"), _num(r.get("invented_volume"), ".3g"), _num(r.get("seconds"), ".0f"),
            _num(r.get("event_us"), ".3g"), _num(r.get("p95_missing"), ".3g"), _num(r.get("p95_invented"), ".3g"),
            r.get("tier") or ("" if r.get("rated") else "-"), r.get("separation") or "", ", ".join(r.get("failed") or [])]


def rating_html(rating: Optional[dict]) -> str:
    """The aegir rating of a run for the detail panel (empty string without one)."""
    from reco_gui.report import _kv, _section, _table
    if not rating:
        return ""
    g, att, vec = rating.get("grade") or {}, rating.get("attribution") or {}, rating.get("vector") or {}
    vol = vec.get("volume") or {}
    sep = att.get("separation") if isinstance(att.get("separation"), dict) else {}
    out = [_section("aegir rating against the known tree", _kv([
        ("tier", g.get("tier")), ("failed", html.escape(", ".join(bm._text(x) for x in g.get("failed") or [])) or "-"),
        ("limiting", html.escape("; ".join(bm._text(x) for x in g.get("limiting") or [])) or "-"),
        ("IoU / missing / invented", f"{_num(vol.get('iou'), '.4f')} / {_num(vol.get('missing'), '.3g')} / {_num(vol.get('invented'), '.3g')}"),
        ("operations / truth leaves", f"{rating.get('n_ops')} / {_num(vec.get('ops_per_truth_leaf'), '.2f')} per leaf"),
        ("separation", f"{sep.get('cls')} (op {sep.get('culprit')}, {sep.get('cause')})" if sep else "-"),
        ("unmeasured", rating.get("unmeasured")),
    ]))]
    by_op = att.get("by_op") or []
    if by_op:
        out.append(_section("attribution per operation", _table(
            ("#", "op", "delta IoU", "wrong writes", "persisting", "redundant"),
            [(o.get("op_index"), o.get("op"), _num(o.get("delta_iou"), ".3f"), _num(o.get("wrong_writes"), ".3f"),
              _num(o.get("persisting"), ".3f"), "yes" if o.get("redundant") else "") for o in by_op])))
    return "".join(out)


class _Rater(QtCore.QThread):
    """Rates the unrated runs of the input in the background (reco_engine.benchmark.benchmark(rate=True))."""
    progressed = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, input_path, store=None, parent=None):
        super().__init__(parent)
        self.input_path, self.store = input_path, store

    def run(self):
        try:
            self.done.emit(bm.benchmark(self.input_path, store=self.store, rate=True, progress=self.progressed.emit))
        except Exception as exc:
            self.failed.emit(str(exc))


class _EventTimer(QtCore.QThread):
    """The aegir event time of the given runs, one after the other (reco_engine.event_time.event_time)."""
    progressed = QtCore.pyqtSignal(str)
    finishedOne = QtCore.pyqtSignal(str, dict)
    failedOne = QtCore.pyqtSignal(str, str)

    def __init__(self, run_dirs: List[str], parent=None, **kw):
        super().__init__(parent)
        self.run_dirs, self.kw = list(run_dirs), kw

    def run(self):
        from reco_engine.event_time import event_time
        for d in self.run_dirs:
            try:
                self.finishedOne.emit(d, event_time(d, progress=self.progressed.emit, **self.kw))
            except Exception as exc:                                     # one run's scan must not end the others
                self.failedOne.emit(d, f"{type(exc).__name__}: {exc}")


class _Nephogrammer(QtCore.QThread):
    """The nephogram of a stored run that has none (reco_gui.nephogram.stored_nephogram)."""
    done = QtCore.pyqtSignal(str, dict)
    failed = QtCore.pyqtSignal(str, str)

    def __init__(self, run_dir: str, parent=None):
        super().__init__(parent)
        self.run_dir = run_dir

    def run(self):
        try:
            from reco_gui.nephogram import stored_nephogram
            info = stored_nephogram(self.run_dir)
            info.pop("distance", None)
            self.done.emit(self.run_dir, info)
        except Exception as exc:
            self.failed.emit(self.run_dir, f"{type(exc).__name__}: {exc}")


def scatter_png(rows: List[dict], path) -> Optional[str]:
    """The checked runs' accuracy (exact IoU, else voxel IoU) against their event time (us), labelled by path - the
    deck's 'two numbers decide'; None without two numbers or matplotlib."""
    pts = [(r["event_us"], bm._iou(r), r.get("variant") or "?") for r in rows
           if r.get("event_us") is not None and bm._iou(r) is not None]
    if not pts:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=36, color="#2a78d6", zorder=3)
    for x, y, lab in pts:
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8, color="#333333")
    ax.set_xlabel("navigation per event (us, aegir)", fontsize=8)
    ax.set_ylabel("IoU", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#dcdad4", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("accuracy against cost (checked runs)", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
    return str(path)


class BenchmarkPanel(QtWidgets.QWidget):
    """The runs of one part, ranked. `runChosen(dir)` on a row click, `useRequested(dir)` on "Use this result",
    `inputRequested()` on "Input"; the owner shows them."""

    runChosen = QtCore.pyqtSignal(str)
    useRequested = QtCore.pyqtSignal(str)
    inputRequested = QtCore.pyqtSignal()
    status = QtCore.pyqtSignal(str)
    nephogramReady = QtCore.pyqtSignal(dict)                    # a run's nephogram info: the owner opens it (its tool)

    def __init__(self, store=None, parent=None):
        super().__init__(parent)
        self.store = store
        self.input_path: Optional[Path] = None
        self.result: dict = dict(input=None, truth=None, rows=[])
        self.rows: List[dict] = []
        self._rater: Optional[_Rater] = None
        self._timer: Optional[_EventTimer] = None
        self._neph: Optional[_Nephogrammer] = None
        self.checked: set = set()                                # run directories checked for the comparison
        self._checked_for = None                                 # the input the checks belong to
        top = QtWidgets.QHBoxLayout()
        self.part_label = QtWidgets.QLabel("no part")
        self.preference = QtWidgets.QComboBox()
        for name, text in bm.PREFERENCES.items():
            self.preference.addItem(name, name)
            self.preference.setItemData(self.preference.count() - 1, text, QtCore.Qt.ToolTipRole)
        self.preference.setToolTip("what the ranking is for: " + "; ".join(f"{k}: {v}" for k, v in bm.PREFERENCES.items()))
        self.preference.currentIndexChanged.connect(lambda _i: self._fill())
        self.btn_rate = QtWidgets.QPushButton("Rate with aegir")
        self.btn_rate.setToolTip("rate every unrated run against the part's known tree (<stem>.json beside the input, or "
                                 "truths/<stem>.json): tier A..E, failed criteria, separation, attribution per operation. "
                                 "Runs in the background; needs a known tree")
        self.btn_rate.clicked.connect(self.rate)
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.setToolTip("read the stored runs of this part again")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_use = QtWidgets.QPushButton("Use this result")
        self.btn_use.setToolTip("the selected run is the one for your application: it is labelled 'chosen' in the run store "
                                "and shown (model, summary, trees); FreeCAD rebuild / groups work on it")
        self.btn_use.clicked.connect(self._use)
        self.btn_event = QtWidgets.QPushButton("Event time (aegir)")
        self.btn_event.setToolTip("scan the checked runs with aegir (Geant4: straight tracks through the program as GDML) and "
                                  "measure the navigation time per event - the difference quotient of two event counts "
                                  "minus an empty baseline, the median of 3 rounds (reco_engine/event_time.py). About a "
                                  "minute per run, in the background; stored with the run")
        self.btn_event.clicked.connect(self.event_time)
        self.btn_neph = QtWidgets.QPushButton("Nephogram")
        self.btn_neph.setToolTip("the selected run's nephogram (the surface coloured by its distance to the input) and its "
                                 "histograms; made now for a run that has none")
        self.btn_neph.clicked.connect(self.nephogram)
        self.btn_input = QtWidgets.QPushButton("Input")
        self.btn_input.setToolTip("show the part itself again")
        self.btn_input.clicked.connect(self.inputRequested.emit)
        for w in (self.part_label, QtWidgets.QLabel("rank for"), self.preference, self.btn_rate, self.btn_event, self.btn_neph,
                  self.btn_refresh, self.btn_use, self.btn_input):
            top.addWidget(w)
        top.addStretch(1)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.table.cellClicked.connect(self._clicked)
        self.table.cellDoubleClicked.connect(lambda r, c: self._use())
        self.table.itemChanged.connect(self._item_changed)
        self.histogram = QtWidgets.QLabel("histograms of the selected run: none yet")
        self.histogram.setAlignment(QtCore.Qt.AlignCenter)
        self.histogram.setMinimumHeight(160)
        self.scatter = QtWidgets.QLabel("accuracy against event time: no event time yet")
        self.scatter.setAlignment(QtCore.Qt.AlignCenter)
        below = QtWidgets.QHBoxLayout()
        below.addWidget(self.histogram, 1)
        below.addWidget(self.scatter, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addLayout(top)
        layout.addWidget(self.table, 1)
        layout.addLayout(below)
        self._update_buttons()

    # ---------------------------------------------------------------- data
    def set_input(self, path):
        """The part whose runs are benchmarked (the settings panel's input); reads its runs."""
        self.input_path = Path(str(path)) if path else None
        self.refresh()

    def refresh(self):
        if self.input_path is None or not self.input_path.is_file():
            self.result = dict(input=None, truth=None, rows=[])
        else:
            self.result = bm.benchmark(self.input_path, store=self.store, rate=False)
        self._fill()

    def rate(self, sync: bool = False):
        """Rate the unrated runs against the known tree, in the background (`sync`: here and now, for tests)."""
        if self.input_path is None or self._rater is not None:
            return
        if self.result.get("truth") is None:
            self.status.emit(f"Rate: no known tree of {self.input_path.name} (looked for {self.input_path.stem}.json beside it "
                             f"and in truths/)")
            return
        if sync:
            self._rated(bm.benchmark(self.input_path, store=self.store, rate=True, progress=self.status.emit))
            return
        self._rater = _Rater(self.input_path, self.store, self)
        self._rater.progressed.connect(self.status.emit)
        self._rater.done.connect(self._rated)
        self._rater.failed.connect(lambda why: (self.status.emit(f"Rating failed: {why}"), self._rated(None)))
        self._update_buttons()
        self._rater.start()

    def _rated(self, result):
        if self._rater is not None:
            self._rater.wait(5000)
            self._rater = None
        if result:
            self.result = result
            n = sum(1 for r in result["rows"] if r.get("rated"))
            self.status.emit(f"Rated: {n} of {len(result['rows'])} runs against {Path(result['truth']).name}")
        self._fill()

    def set_checked(self, run_dirs) -> None:
        """Check these runs (the alternatives just made) and only these."""
        self.checked = {str(d) for d in run_dirs if d}
        self._checked_for = str(self.input_path) if self.input_path is not None else None
        self._fill()

    def checked_rows(self) -> List[dict]:
        return [r for r in self.rows if r["dir"] in self.checked]

    def _default_checks(self) -> None:
        """A part seen for the first time: its newest ALTERNATIVES runs are checked."""
        key = str(self.input_path) if self.input_path is not None else None
        if key == self._checked_for and self.checked:
            return
        newest = sorted(self.result.get("rows") or [], key=lambda r: r.get("saved") or "", reverse=True)[:ALTERNATIVES]
        self.checked = {r["dir"] for r in newest}
        self._checked_for = key

    def _item_changed(self, item) -> None:
        if item.column() != 0:
            return
        d = item.data(QtCore.Qt.UserRole)
        if item.checkState() == QtCore.Qt.Checked:
            self.checked.add(d)
        else:
            self.checked.discard(d)
        self._update_scatter()
        self._update_buttons()

    def _fill(self):
        self._default_checks()
        self.rows = bm.rank(self.result.get("rows") or [], self.preference.currentData() or "accuracy")
        t = self.table
        t.blockSignals(True)
        t.setRowCount(len(self.rows))
        for i, r in enumerate(self.rows, start=1):
            for j, text in enumerate(row_texts(i, r)):
                item = QtWidgets.QTableWidgetItem(text)
                if j == 0:                                       # the check box: compare this run
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                    item.setCheckState(QtCore.Qt.Checked if r["dir"] in self.checked else QtCore.Qt.Unchecked)
                if j in RIGHT:
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                if j == TIER_COLUMN and r.get("tier") in TIER_COLOURS:
                    item.setBackground(QtGui.QColor(TIER_COLOURS[r["tier"]]))
                    item.setForeground(QtGui.QColor("#ffffff" if r["tier"] in "AB" else "#1b1b1b"))
                if j == 5 and r.get("one_valid_solid") is False:
                    item.setForeground(QtGui.QColor("#b3261e"))
                if r.get("label") == "chosen":
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                item.setData(QtCore.Qt.UserRole, r["dir"])
                item.setToolTip(f"{r['name']}\n{r['dir']}\ncommit {r.get('commit')}, {r.get('method')}, "
                                + ("finished" if r.get("finished") else "NOT finished")
                                + (", limiting: " + ", ".join(r["limiting"]) if r.get("limiting") else ""))
                t.setItem(i - 1, j, item)
        t.blockSignals(False)
        t.resizeColumnsToContents()
        self._update_scatter()
        name = self.input_path.name if self.input_path is not None else "no part"
        truth = self.result.get("truth")
        self.part_label.setText(f"<b>{html.escape(name)}</b>: {len(self.rows)} stored run(s), known tree "
                                + (f"{html.escape(Path(truth).name)}" if truth else "none"))
        self._update_buttons()

    def _update_buttons(self):
        rating = self._rater is not None
        self.btn_rate.setEnabled(not rating and bool(self.rows) and self.result.get("truth") is not None)
        self.btn_rate.setText("rating ..." if rating else "Rate with aegir")
        timing = self._timer is not None
        self.btn_event.setEnabled(not timing and any(r.get("has_model") or r.get("ops") for r in self.checked_rows()))
        self.btn_event.setText("event time ..." if timing else "Event time (aegir)")
        self.btn_neph.setEnabled(self._neph is None and self.selected_dir() is not None)
        self.btn_neph.setText("nephogram ..." if self._neph is not None else "Nephogram")
        self.btn_use.setEnabled(self.selected_dir() is not None)
        self.btn_input.setEnabled(self.input_path is not None)

    # ---------------------------------------------------------------- event time, nephogram, histograms
    def event_time(self, sync: bool = False, **kw):
        """aegir's event time of the checked runs that have none, in the background (`sync`: here and now, for tests;
        `kw` goes to reco_engine.event_time.event_time)."""
        if self._timer is not None:
            return
        todo = [r["dir"] for r in self.checked_rows() if r.get("event_us") is None]
        if not todo:
            self.status.emit("Event time: every checked run has one (check others, or none is checked)")
            return
        self._timer = _EventTimer(todo, self, **kw)
        self._timer.progressed.connect(self.status.emit)
        self._timer.finishedOne.connect(self._timed)
        self._timer.failedOne.connect(lambda d, why: self.status.emit(f"Event time of {Path(d).name} failed: {why}"))
        self._timer.finished.connect(self._timing_done)
        self._update_buttons()
        if sync:
            self._timer.run()
            self._timing_done()
        else:
            self._timer.start()

    def _timed(self, run_dir: str, r: dict) -> None:
        if r.get("navigation_us") is not None:
            self.status.emit(f"Event time of {Path(run_dir).name}: {r['navigation_us']:.3g} us per event "
                             f"(range {r['min_us']:.3g}-{r['max_us']:.3g}, gun {r['placement']['gun']})")
        self.refresh()

    def _timing_done(self) -> None:
        if self._timer is not None:
            if self._timer.isRunning():
                self._timer.wait(5000)
            self._timer = None
        self._update_buttons()

    def nephogram(self, sync: bool = False):
        """The selected run's nephogram: shown when it exists, else made first (in the background)."""
        d = self.selected_dir()
        if d is None or self._neph is not None:
            return
        info = bm.read_nephogram(d)
        if info is not None:
            self.nephogramReady.emit(info)
            self._show_histogram(d)
            return
        self.status.emit(f"Nephogram of {Path(d).name}: building the model and measuring the distances ...")
        self._neph = _Nephogrammer(d, self)
        self._neph.done.connect(self._nephogram_done)
        self._neph.failed.connect(lambda run, why: (self.status.emit(f"Nephogram of {Path(run).name} failed: {why}"),
                                                    self._nephogram_done(run, None)))
        self._update_buttons()
        if sync:
            self._neph.run()
        else:
            self._neph.start()

    def _nephogram_done(self, run_dir: str, info) -> None:
        if self._neph is not None:
            if self._neph.isRunning():
                self._neph.wait(5000)
            self._neph = None
        self.refresh()
        if info:
            self.nephogramReady.emit(info)
            self._show_histogram(run_dir)
        self._update_buttons()

    def _show_histogram(self, run_dir: Optional[str]) -> None:
        row = next((r for r in self.rows if r["dir"] == run_dir), None)
        path = (row or {}).get("histogram")
        if path and Path(path).is_file():
            pix = QtGui.QPixmap(path)
            self.histogram.setPixmap(pix.scaledToHeight(min(260, pix.height()), QtCore.Qt.SmoothTransformation))
            self.histogram.setToolTip(path)
        else:
            self.histogram.setPixmap(QtGui.QPixmap())
            self.histogram.setText("histograms of the selected run: no nephogram yet (press Nephogram)")

    def _update_scatter(self) -> None:
        import tempfile
        rows = self.checked_rows()
        path = scatter_png(rows, Path(tempfile.gettempdir()) / f"shape_reco_scatter_{id(self)}.png") if rows else None
        if path:
            pix = QtGui.QPixmap(path)
            self.scatter.setPixmap(pix.scaledToHeight(min(260, pix.height()), QtCore.Qt.SmoothTransformation))
        else:
            self.scatter.setPixmap(QtGui.QPixmap())
            self.scatter.setText(f"accuracy against event time: {len(rows)} run(s) checked, "
                                 f"{sum(r.get('event_us') is not None for r in rows)} with an event time")

    # ---------------------------------------------------------------- choice
    def selected_dir(self) -> Optional[str]:
        item = self.table.currentItem()
        return item.data(QtCore.Qt.UserRole) if item is not None else None

    def _clicked(self, row: int, _col: int):
        item = self.table.item(row, 0)
        self._update_buttons()
        if item is not None:
            self._show_histogram(item.data(QtCore.Qt.UserRole))
            self.runChosen.emit(item.data(QtCore.Qt.UserRole))

    def _use(self):
        d = self.selected_dir()
        if d:
            self.useRequested.emit(d)
