#!/usr/bin/env python3
"""
The supervision window: run the reconstruction framework one stage at a time and watch what it does.

    python3 -m reco_gui [--stl part.stl]

Layout: settings dock (left); toolbar (Start, Step, Continue, Pause, pace, breakpoints); the 3D view of the cascade
(add-cut) model in mm (the default) or of the vector field, the model and the unexplained surface (vector-field mode;
the voxel mode of route C was removed 2026-09-30); a detail panel with the numbers behind the current step; the operations tree (right: the accepted program as
Fusion/Cut nodes, and every step grouped by component); the output (bottom: the log and whatever the run prints);
the timeline of all steps (click one to look at it again while the run goes on). "Close" (Ctrl+Q, the right end of the
toolbar) closes the window also while a step computes: the child processes are ended and, if the worker still does not
return, the process.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from reco_engine import path_policy
from reco_gui import report
from reco_gui.feature_panel import FeaturePanel
from reco_gui.benchmark_panel import BenchmarkPanel
from reco_gui.op_tree import OperationTree
from reco_gui.preview import PreviewLoader
from reco_gui.stats_panel import StatsPanel
from reco_gui.cascade_session import CASCADE_DEFAULT_BREAKPOINTS, CASCADE_STAGES
from reco_gui.field_session import FIELD_DEFAULT_BREAKPOINTS, FIELD_STAGES
from reco_gui.settings_panel import SettingsPanel
from reco_gui.viewer_cascade import CascadeView3D
from reco_gui.viewer_field import FieldView3D
from reco_gui.worker import Worker



def own_cpu_seconds() -> float:
    """CPU time of this process's own threads (user + system), without any child process."""
    t = os.times()
    return t.user + t.system


def children_cpu(pid: Optional[int] = None) -> dict:
    """{pid: (CPU seconds used so far, seconds it has run)} of the child processes that run right now and theirs
    (/proc/<pid>/stat: utime + stime). The heartbeat takes the children's rate from these: os.times().children_* counts a
    child only when it has ended, all of its CPU time in one jump ('now 43.1 cores' on a 16-thread machine)."""
    out, queue = {}, [pid or os.getpid()]
    try:
        ticks = os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError):
        return out
    while queue:
        for child, _, age in child_processes(queue.pop()):
            if child in out:
                continue
            try:
                stat = Path(f"/proc/{child}/stat").read_text()
                rest = stat[stat.rindex(")") + 2:].split()
                out[child] = ((int(rest[11]) + int(rest[12])) / ticks, age)
                queue.append(child)
            except (OSError, ValueError, IndexError):
                continue
    return out


def cpu_seconds() -> float:
    """CPU time used so far: user + system of every thread of this process, plus the child processes that have
    finished (freecadcmd, the voxelizer)."""
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def timing_text(wall: float, cpu: float) -> str:
    """'3.2 s, CPU 11.5 s = 3.6 cores' (the core count only when the wall time is long enough to mean something)."""
    cores = f" = {cpu / wall:.1f} cores" if wall >= 0.2 else ""
    return f"{wall:.1f} s, CPU {cpu:.1f} s{cores}"


def child_processes(pid: Optional[int] = None) -> List[tuple]:
    """(pid, name, seconds it has run) of every child process that runs right now (Linux /proc; empty elsewhere).
    Zombies (ended, not yet collected) are left out."""
    pid = pid or os.getpid()
    out = []
    try:
        ticks, now = os.sysconf("SC_CLK_TCK"), float(Path("/proc/uptime").read_text().split()[0])
        for task in Path(f"/proc/{pid}/task").iterdir():
            for child in (task / "children").read_text().split():
                try:
                    stat = Path(f"/proc/{child}/stat").read_text()
                    name, rest = stat[stat.index("(") + 1:stat.rindex(")")], stat[stat.rindex(")") + 2:].split()
                    if rest[0] != "Z":
                        out.append((int(child), name, now - int(rest[19]) / ticks))
                except (OSError, ValueError, IndexError):
                    continue                                    # it ended meanwhile
    except (OSError, ValueError, IndexError):
        pass
    return out


def running_children(pid: Optional[int] = None) -> List[str]:
    """'freecadcmd 35 s' for every child process that runs right now (Linux /proc; empty elsewhere)."""
    return [f"{name} {seconds:.0f} s" for _, name, seconds in child_processes(pid)]


def terminate_children(pid: Optional[int] = None, grace: float = 2.0) -> List[str]:
    """End the child processes of this process and theirs (freecadcmd, the voxelizer; a shell or `timeout` in between):
    SIGTERM, then SIGKILL for what still runs after `grace` seconds. Returns their names."""
    found, queue = {}, [pid or os.getpid()]
    while queue:
        for child, name, _ in child_processes(queue.pop()):
            if child not in found:
                found[child] = name
                queue.append(child)
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 0.5)):
        alive = [c for c in found if _alive(c)]
        for c in alive:
            try:
                os.kill(c, sig)
            except OSError:
                pass
        end = time.time() + wait
        while time.time() < end and any(_alive(c) for c in alive):
            time.sleep(0.05)
    return list(found.values())


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rindex(")") + 2:].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False


class _OutputTee(QtCore.QObject):
    """Stands in for sys.stdout / sys.stderr: writes through to the original stream and hands every complete line to
    the Output panel. Safe from the worker thread (the signal is queued to the GUI thread)."""
    line = QtCore.pyqtSignal(str)

    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self._buffer = ""

    def write(self, text):
        if self.stream is not None:
            self.stream.write(text)
        self._buffer += text
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            self.line.emit(line)
        return len(text)

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.stream.fileno()


class MainWindow(QtWidgets.QMainWindow):
    requestStart = QtCore.pyqtSignal(object)
    requestStep = QtCore.pyqtSignal()
    requestContinue = QtCore.pyqtSignal(object, float)

    HEARTBEAT_S = 2.0                                           # the status line's refresh while a step computes
    CLOSE_WAIT_S = 5.0                                          # closing: the worker gets this long to return ...
    CLOSE_KILL_WAIT_S = 3.0                                     # ... and this long once its child processes were ended
    VERBOSE_EVERY_S = 10.0                                      # verbose: a heartbeat line in the output this often

    def __init__(self, make_session=None, stl: Optional[str] = None, autorun: bool = False, verbose: bool = False,
                 autosave: Optional[bool] = None):
        """`autosave`: may a finished run be stored by itself (the settings' 'save every finished run' decides then)?
        True: yes, into the run store (`python3 -m reco_gui` passes it). False: never. None (the default, e.g. a window
        made in a test): only into a store that is named by $SHAPE_RECO_RESULTS."""
        super().__init__()
        self.autosave = autosave
        self.verbose = verbose                                  # the log also goes to the terminal, with heartbeat lines
        self.hard_exit = os._exit                               # closing while the worker cannot be stopped (tests replace it)
        self.setWindowTitle("SHaPe ReCo - step-by-step supervision")
        self.resize(1500, 950)                                  # fitted to the screen at the end of __init__
        self.session = None
        self.settings = None
        self._busy = False
        self._t_step = None
        self._cpu_step = None
        self._t_run = None                                      # wall and CPU time at Start, for the run's total
        self._cpu_run = None
        self._follow = True
        self._autorun = autorun
        self._autorun_go = False
        self._autorun_wanted = autorun                          # a retried path runs through as the first run did
        self._tried: List[tuple] = []                           # (path variant, IoU) of the runs on the current input
        self._tried_input = None
        self._alternative_runs: List[str] = []                  # the stored runs of this part's alternatives
        self._run_exports: List = []                            # nephogram exports of finished runs (threads kept alive)
        self._home_path = None                                  # the path fields before the alternatives changed them
        self._rough_prints: dict = {}                           # path variant -> fingerprint of its rough model (this part)
        self._rough_pending = False                             # the run's rough model not fingerprinted yet
        self._alternative = False                               # the running run is an alternative (started by the chain)
        self._same_as = None                                    # (variant, tried variant): stopped, the same rough model
        self._flow_run = False                                  # flow step 3: the run goes through once its session is ready
        self.start_part_entry = None                            # the user's starting part (step 2), None: automatic
        self._start_dialog = self._start_dialog_for = None
        self._input_mesh = None
        self._dataset_gen_window = None
        self.saved_run = None                                   # the directory of the run saved last (run store)
        self._saved_states = None                               # ... and how many states it had

        # ---- worker thread
        self.thread = QtCore.QThread(self)
        self.worker = Worker(make_session)
        self.worker.moveToThread(self.thread)
        self.requestStart.connect(self.worker.start)
        self.requestStep.connect(self.worker.step)
        self.requestContinue.connect(self.worker.cont)
        self.worker.stateReady.connect(self._on_state)
        self.worker.status.connect(self._set_status)
        self.worker.started.connect(self._on_started)
        self.worker.idle.connect(self._on_idle)
        self.worker.failed.connect(self._on_failed)
        self.worker.busy.connect(self._on_busy)
        self.thread.start()

        # ---- widgets
        self.settings_panel = SettingsPanel()
        dock = QtWidgets.QDockWidget("Settings", self)
        dock.setWidget(self.settings_panel)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self.settings_panel.setMinimumWidth(280)
        if stl:
            self.settings_panel.set_input(stl)

        self.view_field = FieldView3D()
        self.view_cascade = CascadeView3D()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.view_field, "Vector field")
        self.tabs.addTab(self.view_cascade, "Cascade")

        self.detail = QtWidgets.QTextBrowser()
        self.detail.setOpenLinks(False)
        self.detail.setMinimumWidth(340)

        self.timeline = QtWidgets.QTableWidget(0, 5)
        self.timeline.setHorizontalHeaderLabels(["#", "phase", "stage", "cand.", "summary"])
        self.timeline.horizontalHeader().setStretchLastSection(True)
        self.timeline.verticalHeader().setVisible(False)
        self.timeline.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.timeline.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.timeline.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.timeline.itemSelectionChanged.connect(self._on_select)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.op_tree = OperationTree()
        self.op_tree.stateActivated.connect(self._on_state_activated)
        self._tees = []

        self.log_candidates = QtWidgets.QCheckBox("log every candidate")
        self.log_candidates.setChecked(verbose)
        self.log_candidates.setToolTip("Off: the candidates of a component / a feature piece are one line of the log (how "
                                       "many, the best, the reasons of the others counted). On (--verbose): a line per "
                                       "candidate. The timeline has a row per candidate either way.")
        self._collapsed = []                                    # the candidate states not logged yet (one line later)
        output = QtWidgets.QWidget()
        output_layout = QtWidgets.QVBoxLayout(output)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(0)
        output_layout.addWidget(self.log_candidates)
        output_layout.addWidget(self.log, 1)

        self.op_dock = QtWidgets.QDockWidget("Operations", self)
        self.op_dock.setObjectName("operations")
        self.op_dock.setWidget(self.op_tree)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.op_dock)
        self.op_tree.setMinimumWidth(240)
        self.output_dock = QtWidgets.QDockWidget("Output", self)
        self.output_dock.setObjectName("output")
        self.output_dock.setWidget(output)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.output_dock)
        self.resizeDocks([self.output_dock], [170], QtCore.Qt.Vertical)
        self.stats = StatsPanel()
        self.stats.stepSelected.connect(self._on_state_activated)
        self.stats_dock = QtWidgets.QDockWidget("Stats", self)
        self.stats_dock.setObjectName("stats")
        stats_area = QtWidgets.QScrollArea()                    # the plots keep their size; the dock does not set the window's
        stats_area.setWidget(self.stats)
        stats_area.setWidgetResizable(True)
        stats_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.stats_dock.setWidget(stats_area)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.stats_dock)
        self.tabifyDockWidget(self.op_dock, self.stats_dock)
        self.op_dock.raise_()
        self.feature_panel = FeaturePanel()
        self.feature_dock = QtWidgets.QDockWidget("Feature (live)", self)
        self.feature_dock.setObjectName("feature")
        self.feature_dock.setWidget(self.feature_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.feature_dock)
        self.tabifyDockWidget(self.stats_dock, self.feature_dock)      # a tab of the side column: it adds no height
        self.feature_dock.hide()                             # shown by the first feature step (or the toolbar)
        self.op_dock.raise_()

        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.detail.setVisible(False)                           # minimal by default: toolbar "Details" / "Timeline"
        top.addWidget(self.tabs)
        top.addWidget(self.detail)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 1)
        bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.bottom_tabs = QtWidgets.QTabWidget()
        self.bottom_tabs.addTab(self.timeline, "Timeline")
        self.benchmark = BenchmarkPanel()
        self.benchmark.runChosen.connect(self.show_benchmark_run)
        self.benchmark.useRequested.connect(self.use_benchmark_run)
        self.benchmark.inputRequested.connect(self.show_benchmark_input)
        self.benchmark.status.connect(self._set_status)
        self.benchmark.nephogramReady.connect(self._open_nephogram)
        self.bottom_tabs.addTab(self.benchmark, "Benchmark")
        bottom.addWidget(self.bottom_tabs)
        main = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main.addWidget(top)
        main.addWidget(bottom)
        self.bottom = bottom
        bottom.setVisible(False)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 1)
        # the decision bar above the viewport (reco_gui/decision_bar.py): the "keep or try another path?" question
        # without a modal box, so the viewport stays usable while the user judges (the user's rule of 2026-10-15)
        from reco_gui.decision_bar import DecisionBar
        self.decision_bar = DecisionBar()
        self.decision_bar.tryRequested.connect(self._retry_try)
        self.decision_bar.chooseRequested.connect(self._retry_choose)
        self.decision_bar.keepRequested.connect(self._retry_keep)
        self.decision_bar.exportRequested.connect(self._retry_export)
        self._export_thread = None
        central = QtWidgets.QWidget()
        central_lay = QtWidgets.QVBoxLayout(central)
        central_lay.setContentsMargins(0, 0, 0, 0)
        central_lay.setSpacing(0)
        central_lay.addWidget(self.decision_bar)
        from reco_gui.result_bar import ResultBar                  # the way into the compare mode after a run
        self.result_bar = ResultBar()
        self.result_bar.compareRequested.connect(lambda: self.enter_compare_mode(
            self.result_bar.run_dirs[0] if self.result_bar.run_dirs else None, select=self.result_bar.run_dirs))
        self.result_bar.benchmarkRequested.connect(lambda: self.show_benchmark(select=self.result_bar.run_dirs or None))
        self.result_bar.postprocessRequested.connect(lambda: self.open_postprocess(
            None if self._newest_result() is not None else (self.result_bar.run_dirs[0] if self.result_bar.run_dirs else None)))
        central_lay.addWidget(self.result_bar)
        central_lay.addWidget(main, 1)
        self.setCentralWidget(central)

        self._build_toolbar()
        from reco_gui.server_run import ServerRuns                  # server runs started here (run bar "on")
        self.server_runs = ServerRuns(self)
        self.server_runs.started.connect(self._server_started)
        self.server_runs.stored.connect(self._server_stored)
        self.server_runs.failed.connect(lambda text: self._log(f"--- {text}"))
        self.server_runs.said.connect(lambda text: self._log(f"    {text}"))
        self.status_label = QtWidgets.QLabel("Ready.")
        self.statusBar().addWidget(self.status_label, 1)
        self.detail.setHtml(report.render_state(None))
        self._update_buttons()
        self._status_base = "Ready."
        self._t_busy = self._beat_cpu = self._beat_t = self._beat_logged = None
        self._beat_children = {}
        self.heartbeat = QtCore.QTimer(self)
        self.heartbeat.setInterval(int(self.HEARTBEAT_S * 1000))
        self.heartbeat.timeout.connect(self._on_heartbeat)
        from reco_engine.memory_guard import MemoryGuard
        self.memory_guard = MemoryGuard()                          # $SHAPE_RECO_MIN_FREE_MB or 1200 MB; the settings may lower it
        self.memory_guard.releases.append(self._drop_caches)
        self.fit_to_screen()
        QtCore.QTimer.singleShot(0, self.fit_docks)
        self.flow.set_note(1, "automatic")
        if self.settings_panel.current_path().is_file():
            self._flow_part_done()
        self.preview = PreviewLoader(self)
        self.preview.loaded.connect(self._on_preview)
        self.preview.failed.connect(lambda path, why: self._set_status(f"No preview of {Path(path).name}: {why}"))
        self.settings_panel.inputChosen.connect(self.preview_input)
        self.settings_panel.inputChosen.connect(self.benchmark.set_input)
        if autorun:
            QtCore.QTimer.singleShot(0, self.start)
        elif stl:
            QtCore.QTimer.singleShot(0, lambda: self.preview_input(stl))

    def fit_to_screen(self, share: float = 0.92):
        """The window at 1500 x 950 or at `share` of the screen's free area, whichever is smaller, centred on it."""
        screen = self.screen() if hasattr(self, "screen") else None
        screen = screen or QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        free = screen.availableGeometry()
        w, h = min(1500, int(share * free.width())), min(950, int(share * free.height()))
        self.resize(w, h)
        self.move(free.x() + (free.width() - w) // 2, free.y() + (free.height() - h) // 2)
        if hasattr(self, "flow"):
            QtCore.QTimer.singleShot(0, self.fit_docks)

    # ---------------------------------------------------------------------------------------- toolbar
    def _build_toolbar(self):
        """The window's controls (the user, 2026-10-29: "make the flow linear; the application fits the screen in full
        screen"): the FLOW bar on top (reco_gui/flow_bar.py: Part > Starting part > Reconstruct > Compare > Choose &
        export, with Next), one compact RUN bar under it (start, step, continue, pause, pace, breakpoints, follow live),
        and everything else in the menus File / View / Tools - one row of 31 actions was 2292 px wide and hid Open, Save,
        GDML and the rebuild behind the overflow arrow on a 1920 px screen."""
        from reco_gui.flow_bar import FlowBar
        flow_tb = self.addToolBar("Steps")
        flow_tb.setObjectName("steps")
        flow_tb.setMovable(False)
        self.flow = FlowBar()
        self.flow.stepClicked.connect(self.flow_goto)
        self.flow.nextClicked.connect(lambda: self.flow_goto(self.flow.current))
        flow_tb.addWidget(self.flow)
        self.addToolBarBreak()
        tb = self.addToolBar("Run")
        tb.setObjectName("run")
        tb.setMovable(False)
        self.act_start = tb.addAction("Start / restart")
        self.act_start.setShortcut("Ctrl+R")
        self.act_step = tb.addAction("Step")
        self.act_step.setShortcut("Space")
        self.act_step.setToolTip("Advance one stage (Space)")
        self.act_cont = tb.addAction("Continue")
        self.act_cont.setShortcut("F5")
        self.act_cont.setToolTip("Run until a breakpoint stage (F5)")
        self.act_pause = tb.addAction("Pause")
        self.act_pause.setShortcut("Esc")
        self.act_pause.setToolTip("Stop after the current computation (Esc)")
        from reco_gui.server_run import HOSTS
        tb.addWidget(QtWidgets.QLabel(" on "))
        self.run_on = QtWidgets.QComboBox()
        for key, text in HOSTS:
            self.run_on.addItem(text, key)
        self.run_on.setToolTip("where Start computes: this computer, or a server (scripts/remote_run.py: the laptop's commit, "
                               "niced; followed live here, stored in the run store and the Benchmark tab when it ends; "
                               "the alternatives start together on the server)")
        tb.addWidget(self.run_on)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" pace "))
        self.pace = QtWidgets.QDoubleSpinBox()
        self.pace.setRange(0.0, 10.0)
        self.pace.setSingleStep(0.25)
        self.pace.setValue(0.5)
        self.pace.setSuffix(" s")
        self.pace.setToolTip("Wait this long after each stage while continuing, so the run can be followed")
        tb.addWidget(self.pace)
        self.break_menu = QtWidgets.QMenu(self)
        self.break_actions = {}
        self._break_method = None
        self._set_breakpoint_menu(self.settings_panel.method.currentData())
        self.settings_panel.method.currentIndexChanged.connect(
            lambda _=0: self._set_breakpoint_menu(self.settings_panel.method.currentData()))
        button = QtWidgets.QToolButton()
        button.setText("Break at")
        button.setMenu(self.break_menu)
        button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        button.setToolTip("Continue stops when one of these stages has been reached")
        tb.addWidget(button)
        self.follow_box = QtWidgets.QCheckBox("follow live")
        self.follow_box.setChecked(True)
        self.follow_box.toggled.connect(lambda on: setattr(self, "_follow", on))
        tb.addWidget(self.follow_box)

        # ---- the actions of the menus
        self.act_details = QtWidgets.QAction("Details", self, checkable=True)
        self.act_details.setToolTip("the detail panel: program, per-stage provenance, the current step")
        self.act_details.toggled.connect(self.detail.setVisible)
        self.act_timeline = QtWidgets.QAction("Timeline and Benchmark", self, checkable=True)
        self.act_timeline.setToolTip("the bottom panel: the timeline of all steps (click one to look back) and the Benchmark tab")
        self.act_timeline.toggled.connect(lambda on: self.bottom.setVisible(on))
        self.act_benchmark = QtWidgets.QAction("Benchmark", self)
        self.act_benchmark.setToolTip("the benchmark of this part's stored runs: what each measured against the input, "
                                      "its path, nephogram figures and event time, ranked; choose the result")
        self.act_benchmark.triggered.connect(self.show_benchmark)
        self.act_fullscreen = QtWidgets.QAction("Full screen", self, checkable=True)
        self.act_fullscreen.setShortcut("F11")
        self.act_fullscreen.toggled.connect(self.set_full_screen)
        act_dataset_gen = QtWidgets.QAction("Dataset generation ...", self)
        act_dataset_gen.setToolTip("Open the window that runs src.dataset_generator with a settings form "
                                   "(mode, topologies, voxelization, hollow/carve/cut, noise)")
        act_dataset_gen.triggered.connect(self._open_dataset_gen)
        self.act_start_part = QtWidgets.QAction("Starting part ...", self)
        self.act_start_part.setToolTip("Define the starting part: automatic, a basic shape around the part or a sub-part, "
                                       "or an exact shape (step 2 of the flow)")
        self.act_start_part.triggered.connect(self.open_start_part_dialog)
        self.act_groups = QtWidgets.QAction("FreeCAD groups", self)
        self.act_groups.setToolTip("The fast way into FreeCAD: every operation's tool as an object of a document, grouped "
                                   "into adds and cuts, WITHOUT the final boolean (seconds). The document's path is in "
                                   "the status line and the log; open it in FreeCAD")
        self.act_groups.setEnabled(False)
        self.act_groups.triggered.connect(self.groups_in_freecad)
        self.act_compare = QtWidgets.QAction("Compare with the input (nephogram)", self)
        self.act_compare.setToolTip("The guess and the input compared by their surface distance (reco_gui/nephogram.py): the "
                                    "input coloured by its distance to the model as the 3D view's layer 'Distance', and the "
                                    "export (STL, coloured PLY, histograms, stats, FreeCAD macro) opened in the tool of the settings")
        self.act_compare.triggered.connect(self.compare_export)
        self.act_post = QtWidgets.QAction("Post-process ...", self)
        self.act_post.setToolTip("The reconstruction's tree with the mass ratio of every part: select parts, benchmark them "
                                 "against the input, minimize the number of parts, decide and export (reco_gui/postprocess.py)")
        self.act_post.triggered.connect(lambda: self.open_postprocess())
        self.act_gdml = QtWidgets.QAction("GDML export ...", self)
        self.act_gdml.setToolTip("Export the program on screen as a GDML file for Geant4 / ROOT: the primitives as GDML "
                                 "solids, the operations as boolean solids (reco_field/gdml_export.py)")
        self.act_gdml.setEnabled(False)
        self.act_gdml.triggered.connect(self.export_gdml_dialog)
        self.act_rebuild = QtWidgets.QAction("FreeCAD rebuild", self)
        self.act_rebuild.setToolTip("Build the program on screen in FreeCAD as ONE solid (all booleans) and measure its "
                                    "chamfer distance to the input, in the background")
        self.act_rebuild.setEnabled(False)
        self.act_rebuild.triggered.connect(self.rebuild_in_freecad)
        from reco_gui.rebuild import FreecadRebuild
        self.rebuilder = FreecadRebuild(self)
        self.rebuilder.done.connect(self._on_rebuilt)
        self.rebuilder.grouped.connect(self._on_grouped)
        self.rebuilder.failed.connect(lambda why: self._rebuild_ended(f"FreeCAD rebuild failed: {why}"))
        self.act_open = QtWidgets.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton), "Open run ...", self)
        self.act_open.setShortcut("Ctrl+O")
        self.act_open.setToolTip("Open a stored run (Ctrl+O): its model, its summary (metrics, program, settings) and its "
                                 "log. A stored run is looked at, not continued")
        self.act_open.triggered.connect(self.open_dialog)
        self.act_save = QtWidgets.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton), "Save run ...", self)
        self.act_save.setShortcut("Ctrl+S")
        self.act_save.setToolTip("Save this run to the run store (Ctrl+S; $SHAPE_RECO_RESULTS or ~/shape_reco_results)")
        self.act_save.triggered.connect(self.save_dialog)
        self.act_compare_runs = QtWidgets.QAction("Compare with the previous run", self)
        self.act_compare_runs.setToolTip("Compare the run just saved with the newest earlier run of the same part (the same "
                                         "input file by content): settings, metrics, program, in the detail panel")
        self.act_compare_runs.triggered.connect(self.compare_with_previous)
        self.act_close = QtWidgets.QAction(self.style().standardIcon(QtWidgets.QStyle.SP_DialogCloseButton), "Close", self)
        self.act_close.setShortcut("Ctrl+Q")
        self.act_close.setToolTip("Close the window (Ctrl+Q), also while a step computes: the run is stopped, its child "
                                  "processes (freecadcmd) are ended")
        self.act_close.triggered.connect(self.close)

        bar = self.menuBar()
        m = bar.addMenu("&File")
        for a in (self.act_open, self.act_save, self.act_compare_runs):
            m.addAction(a)
        self.act_follow = m.addAction("Follow a server run ...", self.follow_dialog)
        self.act_follow.setToolTip("a reconstruction that runs on server_a or server_b (scripts/remote_run.py start): its "
                                   "steps and its model live, from the mirror under ~/shape_reco_results/remote")
        m.addSeparator()
        m.addAction(act_dataset_gen)
        m.addSeparator()
        m.addAction(self.act_close)
        m = bar.addMenu("&View")
        for a in (self.act_details, self.act_timeline, self.act_benchmark):
            m.addAction(a)
        m.addSeparator()
        for dock, tip in ((self.op_dock, "the operations tree: the accepted program as Fusion/Cut nodes, every step by component"),
                          (self.output_dock, "the output: the log and whatever the run prints"),
                          (self.stats_dock, "live statistics: CPU, memory, step times, residual, operations; a step slider"),
                          (self.feature_dock, "the feature in work alone, FreeCAD's metadata of it, and every guess for it")):
            act = dock.toggleViewAction()
            act.setToolTip(tip)
            m.addAction(act)
        m.addSeparator()
        m.addAction(self.act_fullscreen)
        m = bar.addMenu("&Tools")
        for a in (self.act_start_part, self.act_compare):
            m.addAction(a)
        m.addSeparator()
        for a in (self.act_post, self.act_rebuild, self.act_groups, self.act_gdml):
            m.addAction(a)
        m.addSeparator()
        self.act_stop_server = m.addAction("Stop server runs", self.stop_server_runs)
        self.act_stop_server.setToolTip("the server runs started from this window that have not ended are stopped (by "
                                        "their process ids); what they stored stays in the run store")
        m.setToolTipsVisible(True)
        self.export_menu = QtWidgets.QMenu(self)                      # step 5 of the flow
        self.act_use = QtWidgets.QAction("Use the selected result", self)
        self.act_use.setToolTip("the run selected in the Benchmark tab is the one for your application: labelled 'chosen'")
        self.act_use.triggered.connect(lambda: self.benchmark._use())
        for a in (self.act_use, self.act_post, None, self.act_rebuild, self.act_groups, self.act_gdml, None, self.act_compare,
                  self.act_save):
            self.export_menu.addSeparator() if a is None else self.export_menu.addAction(a)

        self.act_start.triggered.connect(self.start)
        self.act_step.triggered.connect(self.step)
        self.act_cont.triggered.connect(self.cont)
        self.act_pause.triggered.connect(self.pause)

    # ---------------------------------------------------------------------------------------- the flow
    def flow_goto(self, i: int) -> None:
        """Do step i of the flow (reco_gui/flow_bar.py): 0 the part (the Settings panel; Browse when there is none),
        1 the starting part (its dialog), 2 the reconstruction (Start), 3 the comparison (the Benchmark tab), 4 the
        choice and the export (a menu: use the result, FreeCAD, GDML, save)."""
        self.flow.set_current(i)
        if i == 0:
            dock = self.settings_panel.parentWidget()
            if dock is not None:
                dock.show()
                dock.raise_()
            if not self.settings_panel.current_path().is_file():
                self.settings_panel._browse()
            else:
                self.settings_panel.stl.setFocus()
                self._flow_part_done()
        elif i == 1:
            self.open_start_part_dialog()
        elif i == 2:                                          # Reconstruct: the run goes to the end (pace and breakpoints
            if self._busy:                                    # apply); a session of this part waiting at "Ready" goes on
                return
            same_part = self.settings is not None and str(getattr(self.settings, "input_stl", "")) == str(
                self.settings_panel.current_path())
            if self.session is not None and not self.session.finished and same_part:
                self.cont()
            else:
                self._flow_run = True
                self.start()
        elif i == 3:                                          # Compare: the compare mode on the newest result
            dirs = self.result_bar.run_dirs if getattr(self, "result_bar", None) is not None else []
            self.enter_compare_mode(dirs[0] if dirs else None, select=dirs)
        else:
            b = self.flow.buttons[4]
            self.export_menu.exec_(b.mapToGlobal(QtCore.QPoint(0, b.height())))

    def _flow_part_done(self, path=None) -> None:
        p = Path(str(path)) if path else self.settings_panel.current_path()
        if p.is_file():
            if self.flow.notes[0] != p.name:                     # another part: its starting part starts afresh
                self.start_part_entry, self._start_dialog = None, None
                self.flow.set_note(1, "automatic")
                self.flow.set_done(1, False)
                for k in (2, 3):
                    self.flow.set_done(k, False)
                    self.flow.set_note(k, "")
            self.flow.set_note(0, p.name)
            self.flow.set_done(0)
            if self.flow.current == 0:
                self.flow.set_current(1)

    def open_start_part_dialog(self) -> None:
        """Step 2: the starting part (reco_gui/start_part_dialog.py), on the part's mesh (the preview's, else loaded)."""
        path = self.settings_panel.current_path()
        if not path.is_file():
            self._set_status("Starting part: choose the part first (step 1)")
            self.flow_goto(0)
            return
        if getattr(self, "_start_dialog", None) is None or getattr(self, "_start_dialog_for", None) != str(path):
            mesh = getattr(self, "_input_mesh", None)
            if mesh is None or mesh[0] != str(path):
                try:
                    from src.closest_point_field import load_mesh
                    v, f = load_mesh(str(path))
                except Exception as exc:
                    self._set_status(f"Starting part: the part cannot be read: {exc}")
                    return
                mesh = (str(path), v, f)
            from reco_gui.start_part_dialog import StartPartDialog
            self._start_dialog = StartPartDialog(mesh[1], mesh[2], self, part_name=path.name)
            self._start_dialog_for = str(path)
        dlg = self._start_dialog
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        try:
            self.start_part_entry = dlg.entry()
        except ValueError as exc:
            self._set_status(f"Starting part not taken: {exc}")
            return
        self.flow.set_note(1, dlg.summary())
        self.flow.set_done(1)
        self.flow.set_current(2)
        self._log(f"--- starting part: {dlg.summary()}" + (f" ({self.start_part_entry['kind']} at "
                  f"{', '.join(f'{x:.4g}' for x in self.start_part_entry['center_mm'])} mm)" if self.start_part_entry else ""))

    def set_full_screen(self, on: bool) -> None:
        """F11: full screen and back; the docks are sized to the screen either way."""
        self.showFullScreen() if on else self.showNormal()
        QtCore.QTimer.singleShot(50, self.fit_docks)

    def fit_docks(self) -> None:
        """The side and bottom docks at shares of the window (the 3D view keeps the most): Settings 21 % (280 - 460 px),
        the right column 19 % (240 - 420 px), the output 16 % of the height (100 - 240 px)."""
        W, H = max(1, self.width()), max(1, self.height())
        clamp = lambda v, lo, hi: int(max(lo, min(hi, v)))           # noqa: E731
        sdock = self.settings_panel.parentWidget()
        docks, widths = [], []
        if isinstance(sdock, QtWidgets.QDockWidget) and sdock.isVisible():
            docks.append(sdock)
            widths.append(clamp(0.21 * W, 280, 460))
        if self.op_dock.isVisible():
            docks.append(self.op_dock)
            widths.append(clamp(0.19 * W, 240, 420))
        if docks:
            self.resizeDocks(docks, widths, QtCore.Qt.Horizontal)
        if self.output_dock.isVisible():
            self.resizeDocks([self.output_dock], [clamp(0.16 * H, 100, 240)], QtCore.Qt.Vertical)

    def _set_breakpoint_menu(self, method: str):
        """The stages of the chosen method (they differ) with that method's default breakpoints."""
        if method == self._break_method:
            return
        self._break_method = method
        stages, defaults = {"field": (FIELD_STAGES, FIELD_DEFAULT_BREAKPOINTS)}.get(
            method, (CASCADE_STAGES, CASCADE_DEFAULT_BREAKPOINTS))
        self.break_menu.clear()
        self.break_actions = {}
        for stage in stages:
            if stage in ("seed", "done", "field", "input"):
                continue
            a = self.break_menu.addAction(stage)
            a.setCheckable(True)
            a.setChecked(stage in defaults)
            self.break_actions[stage] = a
        # only the views the chosen method fills: the others would stay empty
        tabs_of = {"field": (self.view_field,), "cascade": (self.view_cascade,)}
        shown = tabs_of.get(method, tabs_of["cascade"])
        for i in range(self.tabs.count()):
            self.tabs.setTabVisible(i, self.tabs.widget(i) in shown)
        self.tabs.setCurrentWidget(shown[0])

    def _breakpoints(self):
        return tuple(s for s, a in self.break_actions.items() if a.isChecked())

    # ---------------------------------------------------------------------------------------- actions
    def preview_input(self, path: str):
        """Show the picked part in the 3D views before any run (not while one computes)."""
        if self._busy:
            return
        self._set_status(f"Loading the preview of {Path(path).name} ...")
        self.preview.request(path)

    def _on_preview(self, path: str, vertices, triangles, caption: str):
        self._input_mesh = (str(path), vertices, triangles)          # the starting-part dialog measures on it
        self._flow_part_done(path)
        if self._busy:
            return
        for view in (self.view_field, self.view_cascade):
            view.show_input(vertices, triangles, caption)
        self._set_status(caption.replace("\n", "  |  "))
        self._log(caption.replace("\n", ": "))

    def start(self):
        self.preview.cancel()
        try:
            settings = self.settings_panel.collect()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot start", str(exc))
            return
        if str(getattr(settings, "input_stl", "")) != self._tried_input:    # another part: its paths start afresh
            self._tried, self._tried_input = [], str(getattr(settings, "input_stl", ""))
            self._tried_programs = []
            self._alternative_runs = []
            self._home_path = {k: getattr(settings, k, None) for k in path_policy._FIELDS}
        if str(getattr(settings, "input_stl", "")) != str(getattr(self, "_rough_part", "")):
            self._rough_prints, self._rough_part = {}, str(getattr(settings, "input_stl", ""))
        self._alternative, self._chain_start = bool(getattr(self, "_chain_start", False)), False
        self._rough_pending, self._same_as = True, None
        if settings.method == "cascade":                      # the user's starting part (step 2), or the path's own start
            settings.cascade_first_guess = [self.start_part_entry] if self.start_part_entry else None
        host = self.run_on.currentData() if hasattr(self, "run_on") else "local"
        if host != "local" and settings.method == "cascade":   # on a server (reco_gui/server_run.py)
            self._start_on_server(host, settings)
            return
        self.flow.set_current(2)
        self.flow.set_note(2, "running ...")
        self.decision_bar.hide()
        self.follow_box.setChecked(True)                    # a new run is followed live: Open and a click on an earlier
        self._follow = True                                 # guess had switched the following off (the user, 2026-10-16:
        self.memory_guard.min_free_mb = float(getattr(settings, "memory_min_free_mb", self.memory_guard.min_free_mb) or 0.0)   # "the live preview is gone")
        self._busy = True                                   # optimistic: the worker's signal arrives later
        self._update_buttons()
        self.timeline.setRowCount(0)
        self.detail.setHtml(report.render_state(None))
        for view in (self.view_field, self.view_cascade):
            view.set_running(True)                          # the logo token stays away while the run loads
            view.set_state(None)
        self.feature_panel.set_state(None)
        self._feature_dock_shown = False
        self.op_tree.clear_all()
        self.stats.reset()
        self.stats.start_sampling()
        self.session = None
        self.saved_run = self._saved_states = None
        self._collapsed = []
        if settings.method == "cascade":
            self._log(f"--- start: {settings.input_stl.name}, cascade (add-cut): registry {settings.cascade_registry}, "
                      f"top-k {settings.cascade_top_k}, op guard {'on' if settings.cascade_op_guard else 'bypassed'}, "
                      f"max ops {settings.cascade_max_ops or 'loop default'}"
                      + (", feature phase (" + ", ".join(
                          f"{name} {value if value is not None else 'loop default'}" for name, value in (
                              ("max pieces", settings.cascade_feature_max_pieces),
                              ("in all", settings.cascade_feature_max_total_pieces),
                              ("max iterations", settings.cascade_feature_max_iterations),
                              ("time budget s", settings.cascade_feature_time_budget_s),
                              ("min score", settings.cascade_feature_min_score))) + ")"
                         if getattr(settings, "cascade_feature_phase", False) else ", no feature phase")
                      + (", FreeCAD build every step" if settings.cascade_freecad_per_step else ""))
        elif settings.method == "field":
            self._log(f"--- start: {settings.input_stl.name}, vector field: up to {settings.field_max_ops} operations, "
                      f"stop at {settings.field_target:.1%} explained, {settings.field_surface_points} surface / "
                      f"{settings.field_near_points} field points")
        self._t_run, self._cpu_run = time.time(), cpu_seconds()
        self._t_step, self._cpu_step = self._t_run, self._cpu_run     # the first step's time includes the loading
        self.requestStart.emit(settings)

    def step(self):
        if self._can_advance():
            self._busy = True
            self._update_buttons()
            self._t_step, self._cpu_step = time.time(), cpu_seconds()
            self.requestStep.emit()

    def cont(self):
        if self._can_advance():
            self._busy = True
            self._update_buttons()
            self._t_step, self._cpu_step = time.time(), cpu_seconds()
            self.requestContinue.emit(self._breakpoints(), float(self.pace.value()))

    def pause(self):
        self.worker.pause()
        self._set_status("Pausing after the current computation ...")

    def _can_advance(self) -> bool:
        return self.session is not None and not self.session.finished and not self._busy

    # ---------------------------------------------------------------------------------------- signals
    def _on_started(self, session):
        self.session, self.settings = session, self.worker.settings
        self._set_breakpoint_menu(self.settings.method)         # the session's method decides, whatever the panel shows now
        self._log("    field session ready" if self.settings.method == "field" else "    cascade session ready")

    def _newest_result(self):
        states = getattr(self.session, "states", None) or []
        return next((s.result for s in reversed(states) if getattr(s, "result", None) is not None
                     and getattr(s.result, "operations", None)), None)

    def open_postprocess(self, run_dir=None):
        """The post-processing window (reco_gui/postprocess.py) of the newest result, or of a stored run (`run_dir`, or
        the result bar's run when there is no result of this session)."""
        from reco_gui.postprocess import PostprocessWindow
        result, label, input_path = None, "", None
        if run_dir is None:
            result = self._newest_result()
            if result is None and getattr(self, "result_bar", None) is not None and self.result_bar.run_dirs:
                run_dir = self.result_bar.run_dirs[0]
            input_path = getattr(self.settings, "input_stl", None) if result is not None else None
        if result is None and run_dir is not None:
            from reco_engine import benchmark as bm, run_store
            try:
                run = run_store.load_run(run_dir)
                result = bm.result_of(run)
            except Exception as exc:
                self._set_status(f"Post-process: the run could not be read ({exc})")
                return None
            label = Path(str(run_dir)).name
            input_path = ((run.get("input") or {}).get("path")) or getattr(result, "input_path", None)
        if result is None or not getattr(result, "operations", None):
            self._set_status("Post-process: no reconstruction with operations yet")
            return None
        input_path = input_path or getattr(result, "input_path", None) or self.settings_panel.current_path()
        w = PostprocessWindow(result, input_path=input_path, label=label, parent=self, settings=self.settings)
        w.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        w.show()
        self._post_windows = [x for x in getattr(self, "_post_windows", []) if x.isVisible()] + [w]
        self._log(f"--- post-processing window: {len(w.ops)} parts of {Path(str(input_path)).name}"
                  + (f" ({label})" if label else ""))
        return w

    def groups_in_freecad(self):
        """The toolbar's button: the newest result's tools as a grouped FreeCAD document, no final boolean."""
        result = self._newest_result()
        if result is None:
            self._set_status("FreeCAD groups: no cascade result with operations yet")
            return
        if not self.rebuilder.request_groups(result):
            self._set_status("FreeCAD groups: a build is running already")
            return
        self.act_groups.setEnabled(False)
        self._set_status("FreeCAD groups started, in the background")

    def _on_grouped(self, r, seconds: float):
        from reco_gui.rebuild import groups_text
        self.groups_document = r.get("fcstd")
        self.act_groups.setEnabled(True)
        text = groups_text(r, seconds)
        self._set_status(text)
        self._log(f"--- {text}")

    def export_gdml_dialog(self):
        result = self._newest_result()
        if result is None:
            self._set_status("GDML: no cascade result with operations yet")
            return
        start = str(Path.home() / f"{Path(str(result.input_path)).stem}.gdml")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export GDML", start, "GDML (*.gdml)")
        if path:
            self.export_gdml(path)

    def export_gdml(self, path):
        """The newest result's program as a GDML file; the info dict, or None when it failed."""
        result = self._newest_result()
        if result is None:
            self._set_status("GDML: no cascade result with operations yet")
            return None
        try:
            from reco_field.gdml_export import export_result
            info = export_result(result, path)
        except Exception as exc:
            self._set_status(f"GDML export failed: {exc}")
            self._log(f"--- GDML export failed: {exc}")
            return None
        text = f"GDML written: {info['path']} ({info['primitives']} primitives, {info['booleans']} booleans)"
        self._set_status(text)
        self._log(f"--- {text}")
        return info

    def rebuild_in_freecad(self):
        """The toolbar's button: the newest result of the cascade run, built and graded in FreeCAD in the background."""
        states = getattr(self.session, "states", None) or []
        result = next((s.result for s in reversed(states) if getattr(s, "result", None) is not None
                       and getattr(s.result, "operations", None)), None)
        config = getattr(self.session, "config", None)
        if result is None or config is None or isinstance(config, dict):
            self._set_status("FreeCAD rebuild: no cascade result with operations yet")
            return
        if not self.rebuilder.request(result, config):
            self._set_status("FreeCAD rebuild: one is running already")
            return
        self.act_rebuild.setEnabled(False)
        self.act_rebuild.setText("FreeCAD rebuild ...")
        n = sum(1 for o in result.operations if o.accepted)
        self._set_status(f"FreeCAD rebuild started: {n} operation(s), in the background")
        print(f"[gui] FreeCAD rebuild started: {n} operation(s)")

    def _rebuild_ended(self, text: str):
        self.act_rebuild.setText("FreeCAD rebuild")
        self.act_rebuild.setEnabled(True)
        self.act_groups.setEnabled(True)
        self._set_status(text)
        print(f"[gui] {text}")

    def _on_rebuilt(self, result, grade, seconds: float):
        from reco_gui.rebuild import grade_text
        self._rebuild_ended(grade_text(grade, seconds))
        try:
            self.view_cascade._load_freecad(result)
            self.view_cascade.final_box.setChecked(True)
        except Exception as exc:                                     # the numbers stand, the picture is an extra
            print(f"[gui] FreeCAD rebuild: the solid is not shown ({exc})")

    def _on_state(self, index: int):
        state = self.session.states[index]
        if self._rough_pending and state.step.stage == "feature_residual":
            self._rough_model_reached(state)
        if getattr(state, "result", None) is not None and hasattr(self, "act_rebuild") and not self.rebuilder.running:
            self.act_rebuild.setEnabled(bool(getattr(state.result, "operations", None)))
            self.act_groups.setEnabled(bool(getattr(state.result, "operations", None)))
            self.act_gdml.setEnabled(bool(getattr(state.result, "operations", None)))
        row = self.timeline.rowCount()
        self.timeline.insertRow(row)
        colour = report.STAGE_COLOURS.get(state.step.stage)
        for col, text in enumerate(report.timeline_row(state)):
            item = QtWidgets.QTableWidgetItem(text)
            if colour:
                item.setBackground(QtGui.QColor(colour))
            self.timeline.setItem(row, col, item)
        if getattr(state, "in_feature_phase", False):           # every feature step, shown or not: the trace is complete
            if not getattr(self, "_feature_dock_shown", False):
                self._feature_dock_shown = True
                self.feature_dock.show()
                self.feature_dock.raise_()
            if self._follow:
                self.feature_panel.set_state(state)
            else:
                self.feature_panel.trace.add(state)
        self.op_tree.add_step(state, follow=self._follow)
        now, cpu = time.time(), cpu_seconds()
        took = f" ({timing_text(now - self._t_step, cpu - self._cpu_step)})" if self._t_step else ""
        self.stats.add_step(state, now - self._t_step if self._t_step else None,
                            cpu - self._cpu_step if self._t_step else None)
        collapse = (hasattr(state, "program") and state.step.stage in report.COLLAPSED_STAGES
                    and not self.log_candidates.isChecked())
        if collapse:                                            # summed up in one line when the candidates are over
            self._collapsed.append(state)
        else:
            self._flush_candidates()
            self._log(f"{state.index + 1:3d} {state.step.phase:5s} {state.step.stage:10s} {state.step.message}{took}")
        for line in report.feature_log_lines(state):            # the feature phase, verbose: pieces, limits, build numbers
            self._log(f"      \u21b3 {line}")
        sections = getattr(state, "timings", ())
        if sections and not collapse:
            from reco_engine.timing import format_sections
            text = format_sections([x for x in sections if x.depth >= 1] or list(sections))
            if text:
                self._log(f"      \u21b3 {text}")
        self._t_step, self._cpu_step = now, cpu
        if self._follow:
            self._show(index)
            self.timeline.blockSignals(True)
            self.timeline.selectRow(row)
            self.timeline.blockSignals(False)
            self.timeline.scrollToBottom()

    def _on_select(self):
        rows = self.timeline.selectionModel().selectedRows()
        if rows and self.session is not None:
            self._show(rows[0].row())

    def _on_state_activated(self, state_index: int):
        if self.session is None or state_index >= len(self.session.states):
            return
        self.follow_box.setChecked(False)                       # looking at an earlier guess: stop following live
        self.timeline.blockSignals(True)
        self.timeline.selectRow(state_index)
        self.timeline.blockSignals(False)
        self._show(state_index)

    def _show(self, index: int):
        state = self.session.states[index]
        self.detail.setHtml(report.render_state(state, self.settings))
        self.op_tree.show_model(state)
        self.stats.select_step(index)
        if hasattr(state, "vectors"):                           # a vector-field state
            self.view_field.set_state(state)
        elif hasattr(state, "program"):                         # a cascade state
            self.view_cascade.set_state(state)
            if getattr(state, "in_feature_phase", False) and not self._follow:
                self.feature_panel.set_state(state)            # looking back: the piece of that step

    def _flush_candidates(self):
        """The one log line of the candidates collected since the last other step."""
        if self._collapsed:
            states, self._collapsed = self._collapsed, []
            first, last = states[0].index + 1, states[-1].index + 1
            where = f"{first}-{last}" if last != first else f"{first}"
            self._log(f"    [{where}] {states[-1].step.phase} {report.candidates_summary(states)}")

    def _on_idle(self, reason: str):
        self._flush_candidates()                                # stopped at a candidate: say what there was so far
        texts = {"ready": "Ready: press Step or Continue.", "breakpoint": "Stopped at a breakpoint.",
                 "paused": "Paused.", "limit": "Ready for the next step.", "finished": "Run finished."}
        last = self.session.last if self.session is not None else None
        where = f" ({last.step.phase} phase, {last.step.stage})" if last is not None and reason in ("breakpoint", "paused", "limit") else ""
        total = ""
        if self._t_run is not None:
            total = f"   [since Start: {timing_text(time.time() - self._t_run, cpu_seconds() - self._cpu_run)}]"
        self._set_status(texts.get(reason, reason) + where + total)
        if reason == "finished":
            QtCore.QTimer.singleShot(2000, self.stats.stop_sampling)       # a last sample or two after the end
        if reason == "finished" and total:
            self._log("--- run finished" + total)
            self._log_timing_summary()
        if reason == "finished" and self._auto_save_wanted():
            self.save_run()
        if reason == "finished" and self.saved_run is not None and self._saved_states == len(getattr(self.session, "states", []) or []):
            self._nephogram_of_run(self.saved_run)              # every stored run gets its nephogram and histograms
        self._update_buttons()
        if reason == "finished":
            self.benchmark.refresh()                              # the run just saved joins the part's benchmark
            self.flow.set_note(2, f"{max(1, len(self._tried))} reconstruction(s)" if self._tried else "finished")
            self._offer_retry()
            if int(getattr(self.settings, "alternatives", 0) or 0) == 0 or getattr(self.settings, "method", "") != "cascade":
                self.flow.set_done(2)                             # no alternatives: the run is the reconstruction
                self.flow.set_current(3)
                last = self.session.last if self.session is not None else None
                result = getattr(last, "result", None)
                n_ops = len([o for o in getattr(result, "operations", []) or [] if getattr(o, "accepted", True)])
                self.result_bar.show_result(
                    self._result_text(getattr(self.settings, "input_stl", "part"),
                                      path_policy.iou_of_done(getattr(last, "info", None)), n_ops),
                    [str(self.saved_run)] if self.saved_run is not None else [])
        if reason == "ready" and self._autorun:                  # live: go once the worker reports it is idle (see _on_busy)
            self._autorun, self._autorun_go = False, True
        elif reason == "ready" and self._flow_run:               # flow step 3 or an alternative: through, breakpoints kept
            self._flow_run = False
            QtCore.QTimer.singleShot(0, self.cont)
        if reason == "paused" and self._same_as is not None:     # an alternative stopped at a tried path's rough model
            variant, same = self._same_as
            self._same_as = None
            iou = dict(self._tried).get(same)
            prog = next((fp for v, _x, fp in self._tried_programs if v == same), None)
            self._tried_programs.append((variant, iou, prog))
            self._log(f"--- path {variant}: stopped after its rough model - the same as path {same}'s, so the same program "
                      f"from here (the feature phase is the same; {timing_text(time.time() - self._t_run, cpu_seconds() - self._cpu_run)} "
                      f"spent instead of a whole run)")
            QtCore.QTimer.singleShot(0, lambda: self._next_alternative(int(getattr(self.settings, "alternatives", 0) or 0)))

    def _offer_retry(self):
        """The user's rule of 2026-10-10: the first run takes the framework as it is; when a finished cascade run's IoU
        is below the settings' retry_iou, ask whether to try another path - the untried variants ranked by their
        measured share of good runs (reco_engine/path_policy.py, docs/data/path_weights.json). A chosen variant is set
        on the settings panel and started; the paths tried on this part and their IoU are logged."""
        s = self.settings
        if s is None or getattr(s, "method", "") != "cascade" or not (getattr(s, "retry_iou", 0.0) or 0.0) > 0:
            return
        last = self.session.last if self.session is not None else None
        iou = path_policy.iou_of_done(getattr(last, "info", None))
        variant = path_policy.variant_of(s)
        # the program's fingerprint: a path that reproduces an earlier path's program (the stepped shaft: analytic and
        # cnn gave the tree's exact program) is said so and not counted as a new try
        prog = self._program_fingerprint(getattr(last, "result", None))
        same = next((v for v, x, fp in getattr(self, "_tried_programs", []) if fp and fp == prog), None)
        self._tried_programs = getattr(self, "_tried_programs", []) + [(variant, iou, prog)]
        if same is not None:
            self._log(f"--- path {variant}: the same program as path {same} (identical operations) - not a new result")
        else:
            self._tried.append((variant, iou))
        tried_text = ", ".join(f"{v} {x:.3f}" if x is not None else f"{v} ?" for v, x in self._tried)
        self._log(f"--- paths tried on this part: {tried_text}")
        states = getattr(self.session, "states", None) or []
        if same is None and self.saved_run is not None and self._saved_states == len(states):   # stored as it is now
            self._alternative_runs.append(str(self.saved_run))
        if int(getattr(s, "alternatives", 0) or 0) > 0:          # the alternatives: no question, the next path by itself
            self._next_alternative(int(s.alternatives))
            return
        if iou is None or iou >= s.retry_iou:
            if len(self._tried) > 1:
                self.show_benchmark()                             # several paths ran: compare them, choose
            return
        if len(self._tried) > int(getattr(s, "retry_max", 3) or 3):
            self._log(f"    IoU {iou:.3f} below {s.retry_iou:.2f}, but {len(self._tried) - 1} further paths were tried already: stop")
            self.show_benchmark()
            return
        ranked = path_policy.rank(tried=[v for v, _ in self._tried])
        if not ranked:
            self._log("    every path variant was tried")
            self.show_benchmark()
            return
        best = ranked[0]
        self._log(f"    IoU {iou:.3f} below {s.retry_iou:.2f}; next path by measured weight: {path_policy.describe(best)}")
        # the question as a bar above the viewport, never a modal box: the user can turn the model, open the docks,
        # export the guess and the input for a comparison (reco_gui/nephogram.py) and decide afterwards
        self.decision_bar.ask(f"IoU {iou:.3f}, below {s.retry_iou:.2f} (paths tried: {tried_text}). Try another approach? "
                              f"Suggested: {path_policy.describe(best)}", ranked)

    def _rough_model_reached(self, state) -> None:
        """The rough model is done (the first feature step): its fingerprint per path; an alternative whose path acts
        before the feature phase only (path_policy.rough_only) and that reached a tried path's rough model is stopped -
        the feature phase is the same, so is the program (on the ABC plate 00009283 sub-parts and analytic rebuilt bare's
        program in full, 104.6 s and 65.7 s; the user's session of 2026-10-29)."""
        self._rough_pending = False
        fp = self._program_fingerprint(getattr(state, "result", None))
        variant = path_policy.variant_of(self.settings)
        same = next((v for v, f in self._rough_prints.items() if f and f == fp and v != variant), None)
        self._rough_prints.setdefault(variant, fp)
        if self._alternative and same is not None and path_policy.rough_only(variant) and path_policy.rough_only(same):
            self._same_as = (variant, same)
            self._log(f"--- path {variant}: its rough model is path {same}'s - stopping the run")
            self.pause()

    def _next_alternative(self, n: int) -> None:
        """The user's rule of 2026-10-29: after a run, go on with the next path until n different reconstructions of the
        part exist (a path that repeats an earlier program does not count), then show them in the Benchmark tab, checked
        - the nephograms and histograms are made after every run, the event time on request. The paths are the untried
        variants by measured rescue rate (reco_engine/path_policy.py); the settings panel gets its own path back at the end."""
        done, attempts = len(self._tried), len(getattr(self, "_tried_programs", []))
        part = str(getattr(self.settings, "input_stl", "") or "")
        best = max((x for _v, x in self._tried if x is not None), default=None)
        if part and str(self.settings_panel.current_path()) != part:      # the user went on to another part
            self._log(f"--- the alternatives of {Path(part).name} stop: another part is chosen "
                      f"({self.settings_panel.current_path().name})")
            if self._home_path:
                self.settings_panel.apply_path({k: v for k, v in self._home_path.items()
                                                if v != path_policy._FIELDS[k] and v is not None})
            return
        elif best is not None and best >= path_policy.EXACT_IOU and done < n:
            self._log(f"--- IoU {best:.4f} (exact): no alternatives - none can do better")
            n = done
        if done < n and attempts < len(path_policy.VARIANTS):
            ranked = path_policy.rank(tried=[v for v, _x, _fp in self._tried_programs])
            if ranked:
                nxt = ranked[0]
                self._log(f"--- alternative {done + 1} of {n}: {path_policy.describe(nxt)}")
                QtCore.QTimer.singleShot(0, lambda v=nxt["variant"]: self._retry_start(v, alternative=True))
                return
            self._log("    every path variant was tried")
        if self._home_path:                                      # the panel shows the user's own path again
            self.settings_panel.apply_path({k: v for k, v in self._home_path.items()
                                            if v != path_policy._FIELDS[k] and v is not None})
        self.flow.set_note(2, f"{done} different reconstruction(s)")
        self.flow.set_done(2)
        self.flow.set_current(3)
        self._log(f"--- {done} different reconstruction(s) of this part: compare them in the Benchmark tab "
                  f"(nephogram and histograms per run; 'Event time (aegir)' for the cost; 'Use this result' to choose)")
        self.show_benchmark(select=self._alternative_runs[-n:])
        best = max(self._tried, key=lambda t: -1.0 if t[1] is None else t[1], default=(None, None))
        self.result_bar.show_result(self._result_text(part or "part", best[1], len(self._alternative_runs[-n:]))
                                    .replace(" operations.", " different reconstructions (the best above)."),
                                    list(self._alternative_runs[-n:]))

    def _nephogram_of_run(self, run_dir) -> None:
        """The nephogram and histograms of the run just stored, into <run dir>/nephogram/ (in a thread; no tool opens):
        the Benchmark tab shows them, 'Nephogram' opens them."""
        from reco_gui.decision_bar import ExportThread
        from reco_gui.nephogram import RUN_DIR
        input_mesh, model_mesh = self._comparison_meshes()
        if input_mesh is None or model_mesh is None:
            return
        name = Path(str(run_dir)).name
        input_path = str(getattr(self.settings, "input_stl", "") or "") or None
        th = ExportThread(input_mesh, model_mesh, Path(run_dir) / RUN_DIR, name, input_path=input_path, result=self._newest_result())
        th.done.connect(lambda info, d=str(run_dir): self._run_nephogram_done(d, info))
        th.failed.connect(lambda why, d=str(run_dir): self._log(f"--- nephogram of {Path(d).name} NOT made: {why}"))
        self._run_exports = [t for t in self._run_exports if t.isRunning()] + [th]
        th.start()

    def _run_nephogram_done(self, run_dir: str, info: dict) -> None:
        s = info.get("stats") or {}
        self._log(f"--- nephogram of {Path(run_dir).name}: input -> model p95 {s.get('input_to_model', {}).get('p95', float('nan')):.4g}, "
                  f"model -> input p95 {s.get('model_to_input', {}).get('p95', float('nan')):.4g} (histograms: "
                  f"{info.get('files', {}).get('histogram') or 'none'})")
        self.benchmark.refresh()

    def _open_nephogram(self, info: dict) -> None:
        """A stored run's nephogram from the Benchmark tab, in the tool of the settings (FreeCAD shows the histograms too)."""
        tool = str(getattr(self.settings, "compare_tool", "freecad") or "none") if self.settings is not None else "freecad"
        files = info.get("files") or {}
        if tool == "freecad" and files.get("macro"):
            from reco_gui.nephogram import open_in_freecad
            opened = open_in_freecad(files["macro"])
            self._log("--- FreeCAD started on " + files["macro"] if opened else "--- FreeCAD not found: open the macro by hand")
        elif tool == "cloudcompare" and files.get("ply"):
            from reco_gui.nephogram import open_in_cloudcompare
            open_in_cloudcompare(files["ply"])
        self._set_status(f"Nephogram: {info.get('dir')}")

    @staticmethod
    def _program_fingerprint(result) -> str:
        """The GEOMETRY of the accepted operations (kind, centre, sizes, frame, sweeps, a composite's parts; rounded to
        1e-6 mm / 1e-4), joined; "" without a program. The provenance (sources, confidence, alternatives) is left out:
        on ABC part 00009283_..._step_001 the paths bare, sub-parts and analytic built the same six operations and
        differed only in the first one's sources, so a full-record fingerprint counted three copies as three
        reconstructions (2026-10-29)."""
        def geom(p):
            if p is None:
                return None
            q = [float(x) for x in (p.quaternion or (0.0, 0.0, 0.0, 1.0))]
            if q[3] < 0 or (q[3] == 0 and q[0] < 0):
                q = [-x for x in q]                              # q and -q are the same frame
            return (p.kind, tuple(round(float(x), 6) for x in p.center_mm), tuple(round(x, 4) for x in q),
                    tuple(sorted((k, round(float(v), 6)) for k, v in (p.dims_mm or {}).items())),
                    tuple(round(float(x), 4) for x in p.u) if p.u else None,
                    tuple(round(float(x), 4) for x in p.v) if p.v else None, p.combine,
                    tuple(geom(x) for x in (p.parts or [])))
        try:
            ops = [o for o in getattr(result, "operations", []) if getattr(o, "accepted", False)]
            return "|".join(repr((o.op, geom(o.primitive))) for o in ops)
        except Exception:
            return ""

    def _retry_start(self, choice: str, alternative: bool = False) -> None:
        self.decision_bar.hide()
        self.settings_panel.apply_path(path_policy.VARIANTS[choice]["overrides"])
        self._log(f"--- path {len(self._tried) + 1}: {choice} ({path_policy.VARIANTS[choice]['label']})")
        self._autorun = self._autorun_wanted
        self._chain_start = alternative
        self._flow_run = alternative or self._flow_run              # an alternative runs through as the first run did
        self.start()

    def _retry_try(self, choice: str) -> None:
        if choice:
            self._retry_start(choice)

    def _retry_choose(self) -> None:
        ranked = self.decision_bar.ranked
        labels = [path_policy.describe(r) for r in ranked]
        text, ok = QtWidgets.QInputDialog.getItem(self, "Which path?", "path variant", labels, 0, False)
        if ok and text in labels:
            self._retry_start(ranked[labels.index(text)]["variant"])

    def _retry_keep(self) -> None:
        self.decision_bar.hide()
        self._log("--- result kept")
        if len(self._tried) > 1:
            self.show_benchmark()

    def _comparison_meshes(self):
        """(input mesh, model mesh) of the newest cascade state, or (None, None)."""
        states = getattr(self.session, "states", None) or []
        state = next((st for st in reversed(states) if getattr(st, "model_mesh", None) is not None), None)
        if state is None:
            return None, None
        return getattr(state, "input_mesh", None), state.model_mesh

    def _retry_export(self) -> None:
        self.compare_export()

    def compare_export(self) -> None:
        """The toolbar's Compare and the decision bar's export: the guess and the input as STL and distance-coloured
        PLY (reco_gui/nephogram.py) in a thread; then the 3D view's layer 'Distance' and the tool of the settings."""
        from reco_gui.decision_bar import ExportThread
        input_mesh, model_mesh = self._comparison_meshes()
        if input_mesh is None or model_mesh is None:
            self._set_status("comparison: no input or model mesh in this run yet")
            return
        if self._export_thread is not None and self._export_thread.isRunning():
            return
        name = Path(str(getattr(self.settings, "input_stl", "") or "part")).stem
        out = Path.home() / ".cache" / "shape_reco" / "compare" / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.decision_bar.exporting()
        self.act_compare.setEnabled(False)
        self._compare_input = input_mesh
        result = self._newest_result()
        input_path = str(getattr(self.settings, "input_stl", "") or "") or None
        self._export_thread = ExportThread(input_mesh, model_mesh, out, name, input_path=input_path, result=result)
        self._export_thread.done.connect(self._on_exported)
        self._export_thread.failed.connect(self._on_export_failed)
        self._export_thread.start()
        self._set_status(f"comparison started: {out}")

    def _show_distance_layer(self, info: dict) -> None:
        """The input coloured by its distance to the model in the 3D view (the 'inside the framework' option, shown in
        every case: the layer 'Distance' can be switched off)."""
        from reco_gui.cascade_session import ColouredMesh
        from reco_gui.nephogram import triangle_colours
        mesh = getattr(self, "_compare_input", None)
        d = (info.get("distance") or {}).get("input_to_model")
        if mesh is None or d is None or len(d) != len(mesh.vertices):
            return
        colours = triangle_colours(d, mesh.triangles, info["scale"])
        s = info["stats"]["input_to_model"]
        bands = info.get("legend") or []
        self.view_cascade.show_distance(ColouredMesh(mesh.vertices, mesh.triangles, colours),
                                        caption=f"distance input -> model, {len(bands)} bands of {info['scale'] / max(1, len(bands)):.4g}: "
                                                f"mean {s['mean']:.4g}, p95 {s['p95']:.4g}, max {s['max']:.4g}",
                                        legend=bands)
        self._log("--- distance bands (input -> model): " + "; ".join(f"{b['text']} {s['band_share'][b['band']]:.0%}" for b in bands))
        self.tabs.setCurrentWidget(self.view_cascade)

    def _on_exported(self, info: dict) -> None:
        self.act_compare.setEnabled(True)
        self.decision_bar.exported(info)
        try:
            self._show_distance_layer(info)
        except Exception as exc:                                       # the layer is a convenience
            self._log(f"--- distance layer not shown: {exc}")
        s = info["stats"]
        text = (f"comparison exported to {info['dir']} ({info.get('method', '')}): input -> model mean {s['input_to_model']['mean']:.4g} / max "
                f"{s['input_to_model']['max']:.4g}, model -> input mean {s['model_to_input']['mean']:.4g} / max "
                f"{s['model_to_input']['max']:.4g}, colour scale {info['scale']:.4g} ({info['seconds']:.1f} s)")
        self._set_status(text)
        self._log("--- " + text)
        tool = str(getattr(self.settings, "compare_tool", "freecad") or "none")   # opened at once, the user's choice
        if tool == "freecad":
            from reco_gui.nephogram import open_in_freecad
            opened = open_in_freecad(info["files"]["macro"]) if info["files"].get("macro") else None
            self._log("--- FreeCAD started on nephogram.FCMacro" if opened else "--- FreeCAD not found: open nephogram.FCMacro by hand")
        elif tool == "cloudcompare":
            from reco_gui.nephogram import open_in_cloudcompare
            opened = open_in_cloudcompare(info["files"]["ply"])
            self._log("--- CloudCompare started on the PLYs" if opened else "--- CloudCompare not found: open the PLYs by hand")

    def _on_export_failed(self, text: str) -> None:
        self.act_compare.setEnabled(True)
        self.decision_bar.export_failed(text)
        self._set_status("comparison export failed: " + text)
        self._log("--- comparison export failed: " + text)

    def _on_failed(self, trace: str):
        self.stats.stop_sampling()
        self._log("ERROR\n" + trace)
        self._set_status("Failed: see the log.")
        QtWidgets.QMessageBox.critical(self, "The run failed", trace.strip().splitlines()[-1] + "\n\nFull traceback in the log.")
        self._update_buttons()

    def _on_busy(self, busy: bool):
        self._busy = busy
        self._update_buttons()
        for view in (self.view_field, self.view_cascade):
            view.set_running(busy)                          # idle and nothing on screen: the logo token again
        if busy:
            self._t_busy = self._beat_t = self._beat_logged = time.time()
            self._beat_cpu = own_cpu_seconds()
            self._beat_children = {pid: seconds for pid, (seconds, _) in children_cpu().items()}
            self.heartbeat.start()
        else:
            self.heartbeat.stop()
        if not busy and self._autorun_go:                        # live: no breakpoints, straight to the end at the toolbar's pace
            self._autorun_go = False
            for a in self.break_actions.values():
                a.setChecked(False)
            self.cont()

    def _set_status(self, text: str):
        self._status_base = text
        self.status_label.setText(text)

    def heartbeat_text(self) -> str:
        """'41 s, now 3.2 cores, in residual/fit (12 s), running freecadcmd 8 s, feature phase: piece 7 of 40, iteration
        2': what the computing step does right now (the feature phase from the last folded state). 'now 0.0 cores' with
        no child process for a long time is a step that hangs."""
        now, cpu, kids = time.time(), own_cpu_seconds(), children_cpu()
        interval = max(now - self._beat_t, 1e-9)
        cores = (cpu - self._beat_cpu) / interval
        used = 0.0
        for pid, (seconds, age) in kids.items():
            before = self._beat_children.get(pid)
            if before is not None:
                used += max(0.0, seconds - before)
            else:                                   # seen for the first time: its share of the interval
                used += seconds * min(1.0, interval / max(age, interval))
        self._beat_t, self._beat_cpu = now, cpu
        self._beat_children = {pid: seconds for pid, (seconds, _) in kids.items()}
        parts = [f"{now - self._t_busy:.0f} s", f"now {cores:.1f} cores"
                 + (f" + children {used / interval:.1f}" if kids else "")]
        rec = getattr(self.session, "recorder", None)
        current = rec.current() if rec is not None else None
        if current:
            parts.append(f"in {current[0]} ({current[1]:.0f} s)")
        children = running_children()
        if children:
            parts.append("running " + ", ".join(children))
        last = getattr(self.session, "last", None)
        if last is not None and getattr(last, "feature", None) is not None:
            from reco_gui.cascade_session import feature_progress
            progress = feature_progress(last)
            if progress:
                parts.append(progress)
        return ", ".join(parts)

    def _on_heartbeat(self):
        if not self._busy or self._t_busy is None:
            return
        text = self.heartbeat_text()
        low = self.memory_guard.check()                          # the memory watcher (reco_engine/memory_guard.py)
        if low:
            self.worker.pause()
            self._status_base = f"PAUSED, {low}. Close other programs or lower the grid, then Continue."
            self._log(f"--- {low}: the run is paused at its next step (the mesh caches were dropped)")
        self.status_label.setText(f"{self._status_base}   [{text}]")
        if self.verbose and time.time() - self._beat_logged >= self.VERBOSE_EVERY_S:
            self._beat_logged = time.time()
            self._log(f"      ... {text}")

    def _drop_caches(self) -> None:
        """What the window can give back when memory is low: the session's model-mesh caches."""
        session = self.session
        for name in ("_cache",):
            cache = getattr(getattr(session, "folder", None), name, None) or getattr(session, name, None)
            if isinstance(cache, dict):
                cache.clear()
        folder = getattr(session, "folder", None)
        for mesher in (getattr(folder, "_meshers", None) or {}).values():
            try:
                mesher.clear()
            except Exception:
                pass

    def _update_buttons(self):
        can = self._can_advance()
        self.act_step.setEnabled(can)
        self.act_cont.setEnabled(can)
        self.act_pause.setEnabled(self._busy)
        self.act_start.setEnabled(not self._busy)
        self.act_save.setEnabled(not self._busy and self.session is not None and len(getattr(self.session, "states", ())) > 0)
        self.act_compare_runs.setEnabled(self.saved_run is not None)

    def _log(self, text: str):
        self.log.appendPlainText(text)
        if self.verbose and sys.__stdout__ is not None:
            print(time.strftime("%H:%M:%S"), text, file=sys.__stdout__, flush=True)

    # ---------------------------------------------------------------------------------------- run store
    def _auto_save_wanted(self) -> bool:
        """'save every finished run' is checked, this window may save by itself (`autosave`; when that was left open,
        a store must be named by $SHAPE_RECO_RESULTS), the run has states and was not saved as it is."""
        allowed = bool(os.environ.get("SHAPE_RECO_RESULTS")) if self.autosave is None else bool(self.autosave)
        if not allowed or not getattr(self.settings, "save_finished_run", False):
            return False
        if self.session is None or not self.session.states:
            return False
        return self._saved_states != len(self.session.states)

    def _grab_view(self, path: str) -> bool:
        """A screenshot of the 3D view in use (GUI thread); False when there is none (offscreen, no VTK)."""
        view = self.tabs.currentWidget()
        grab = getattr(view, "grab_image", None)
        return bool(grab(path)) if grab is not None else False

    def open_dialog(self):
        if self._busy:
            self._set_status("Open: a run is computing - pause or wait first")
            return
        from reco_gui.open_run import OpenRunDialog
        dialog = OpenRunDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted and dialog.chosen:
            self.open_run(dialog.chosen)

    def follow_dialog(self):
        """File > Follow a server run: a mirror of scripts/remote_run.py (~/shape_reco_results/remote/<id>)."""
        from reco_gui.server_follow import ServerFollow  # noqa: F401  (fails early without the module)
        start = str(Path.home() / "shape_reco_results" / "remote")
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Follow a server run (a mirror of remote_run.py)", start)
        if d:
            self.follow_server_run(d)

    def follow_server_run(self, mirror, load_mesh=None):
        """Show a reconstruction that runs on a server, live: its steps in the log, the model over the part
        (reco_gui/server_follow.py; started by scripts/remote_run.py)."""
        from reco_gui.server_follow import ServerFollow
        mirror = Path(mirror).expanduser()
        if getattr(self, "_follow_run", None) is not None:
            self._follow_run.stop()
        f = ServerFollow(mirror, self)
        self._follow_run = f
        self._follow_input = None
        step = mirror / "input.step"
        if step.is_file():
            try:
                if load_mesh is None:
                    from src.closest_point_field import load_mesh
                self._follow_input = load_mesh(str(step))
                for view in (self.view_cascade,):
                    view.show_pair(self._follow_input, None, f.caption)
            except Exception as exc:
                self._log(f"    the input of the server run could not be read: {exc}")
        self.follow_box.setChecked(False)
        self._log(f"--- following {f.caption} (mirror {mirror}); the exact model and its grade: "
                  f"python3 scripts/remote_run.py fetch {mirror}")
        f.stepLine.connect(self._log)
        f.modelReady.connect(lambda v, t, cap: self.view_cascade.show_pair(self._follow_input, (v, t), cap))
        f.candidateReady.connect(self._follow_candidate)
        f.regionReady.connect(self._follow_region)
        f.modelReady.connect(lambda v, t, cap: self._set_status(cap.replace("\n", "  |  ")))
        f.finished.connect(lambda why: self._log(f"--- server run: {why}"))
        f.finished.connect(lambda why, m=mirror: self._server_run_ended(m, why))
        f.start()
        return f

    def enter_compare_mode(self, run_dir=None, select=None) -> None:
        """The compare mode (the user, 2026-10-29: "when the part is reconstructed enable a method to get into the compare
        mode"): step 4 of the flow; the Benchmark tab with the run(s) checked and the histograms of the selected one; the
        input coloured by its distance to the model in the 3D view - a stored run's nephogram (made in a thread when it
        has none: local and server runs alike), else the comparison of the run in the window (compare_export)."""
        self.result_bar.hide()
        self.flow.set_current(3)
        self.show_benchmark(select=[str(d) for d in (select or ([run_dir] if run_dir else []))] or None)
        if run_dir is None:
            if self._comparison_meshes()[1] is not None:
                self.compare_export()
            else:
                self._set_status("compare: choose a run in the Benchmark tab (its Nephogram button colours the input)")
            return

        class Neph(QtCore.QThread):
            done = QtCore.pyqtSignal(object, object, str)

            def run(self_t):
                try:
                    from reco_engine import run_store
                    from reco_gui.nephogram import _Mesh, stored_nephogram
                    from src.closest_point_field import load_mesh
                    run = run_store.load_run(run_dir)
                    v, f = load_mesh(str((run.get("input") or {}).get("path")))
                    self_t.done.emit(stored_nephogram(run_dir), _Mesh(v, f), "")
                except Exception as exc:
                    self_t.done.emit(None, None, str(exc)[-300:])
        self._log(f"--- compare mode: {Path(str(run_dir)).name} - the nephogram and its histograms ...")
        self._set_status("compare mode: computing the distances of the input to the model ...")
        t = Neph(self)
        t.done.connect(lambda info, mesh, err, d=str(run_dir): self._compare_mode_ready(d, info, mesh, err))
        self._compare_thread = t
        t.start()

    def _compare_mode_ready(self, run_dir: str, info, mesh, err: str) -> None:
        if err or info is None:
            self._log(f"--- compare mode: no nephogram for {Path(run_dir).name}: {err}")
            return
        self._compare_input = mesh
        try:
            self._show_distance_layer(info)
        except Exception as exc:
            self._log(f"--- the distance layer is not shown: {exc}")
        self._run_nephogram_done(run_dir, info)                     # the log line, the Benchmark tab's histograms
        self.benchmark.set_checked([run_dir])

    def _result_text(self, part, iou, n_ops, host: str = "") -> str:
        return (f"<b>{Path(str(part)).name}</b> reconstructed{(' on ' + host) if host else ''}: exact IoU "
                f"{'-' if iou is None else f'{iou:.4f}'}, {n_ops} operations. Compare it with the input, or with the "
                f"part's other runs in the Benchmark tab.")

    def _start_on_server(self, host: str, settings):
        """Start the part on a server with the window's settings and follow it; the alternatives come after it. A second
        Start while a server run of the part is starting or running only says so (the user's session of 2026-10-29: the
        deploy took 20 s without a sign, Start was pressed again and the same run went twice to server_a)."""
        from reco_gui.server_run import server_args
        chain = getattr(self, "_server_chain", None)
        if chain is not None:
            self._log(f"--- a server run of {Path(str(chain['part'])).name} is {'starting' if not chain.get('following') else 'running'} "
                      f"on {chain['host']}: Start again after it has ended, or Tools > Stop server runs")
            self._set_status(f"a server run is already on {chain['host']}")
            return
        variant = path_policy.variant_of(settings)
        self._server_chain = dict(part=str(settings.input_stl), host=host, settings=settings, first=variant,
                                  n=int(getattr(settings, "alternatives", 0) or 0), started=[variant], stored={}, done=0)
        self.flow.set_current(2)
        self.flow.set_note(2, f"on {host} ...")
        self.decision_bar.hide()
        self._log(f"--- start on {host}: {Path(str(settings.input_stl)).name}, path {variant}, run_part "
                  f"{server_args(settings) or '(the bare settings)'}"
                  + (", with the starting part" if self.start_part_entry else "") + " - deploying and starting ...")
        self._set_status(f"starting on {host}: deploying the commit when needed, copying the part ...")
        self.server_runs.start(host, settings, self.start_part_entry, variant)

    def stop_server_runs(self):
        """Tools > Stop server runs: every server run started here that has not ended is stopped (by its pid)."""
        from reco_gui.server_run import _remote_run
        rr = _remote_run()
        pending = list(self.server_runs.pending)
        for mirror in pending:
            try:
                rr.cmd_stop(type("A", (), {"mirror": mirror})())
                self._log(f"--- stopped the server run {Path(mirror).name}")
            except Exception as exc:
                self._log(f"--- {Path(mirror).name} could not be stopped: {exc}")
            self.server_runs.pending.pop(mirror, None)
        if getattr(self, "_follow_run", None) is not None:
            self._follow_run.stop()
        self._server_chain = None
        self._set_status(f"{len(pending)} server run(s) stopped" if pending else "no server run to stop")

    def _server_started(self, mirror, variant: str):
        chain = getattr(self, "_server_chain", None) or {}
        if variant == chain.get("first") and chain.get("done", 0) == 0 and len(chain.get("stored", {})) == 0:
            chain["following"] = str(mirror)
            self.follow_server_run(mirror)                      # the first run live; the alternatives in the background
        else:
            host = self.server_runs.hosts.get(str(mirror)) or chain.get("host")
            self._log(f"--- alternative {variant} runs on {host}: {Path(str(mirror)).name}")

    def _server_stored(self, mirror, store_dir, variant: str):
        from reco_engine import run_store
        from reco_gui.server_run import with_variant
        chain = getattr(self, "_server_chain", None)
        iou = None
        if store_dir is not None:
            run = run_store.load_run(store_dir)
            iou = (run.get("metrics") or {}).get("exact_iou")
            self._log(f"--- server run {variant or Path(str(mirror)).name} stored: exact IoU "
                      f"{'-' if iou is None else f'{iou:.4f}'} -> {store_dir} (Benchmark tab)")
        self.benchmark.refresh()
        if chain is None:
            return
        if str(chain["part"]) != str(self.settings_panel.current_path()):
            self._log(f"    (another part is chosen now: the run is in the store of {Path(chain['part']).name})")
        chain["stored"][variant] = (str(store_dir) if store_dir else None, iou)
        chain["done"] += 1
        if variant == chain["first"]:
            self.show_server_result(mirror)
            if chain["n"] > 1 and str(chain["part"]) == str(self.settings_panel.current_path()):
                if iou is not None and iou >= path_policy.EXACT_IOU:
                    self._log(f"--- IoU {iou:.4f} (exact): no alternatives - none can do better")
                else:
                    ranked = [r["variant"] for r in path_policy.rank(tried=[chain["first"]])][:chain["n"] - 1]
                    self._log(f"--- {len(ranked)} alternative(s) start together on {chain['host']}: {', '.join(ranked)}")
                    for v in ranked:
                        chain["started"].append(v)
                        self.server_runs.start(chain["host"], with_variant(chain["settings"], v), self.start_part_entry, v)
        if chain["done"] >= len(chain["started"]):
            dirs = [d for d, _x in chain["stored"].values() if d]
            self._log(f"--- {len(dirs)} server run(s) of {Path(chain['part']).name} stored: "
                      + ", ".join(f"{v} {'-' if x is None else f'{x:.4f}'}" for v, (_d, x) in chain["stored"].items()))
            self.flow.set_note(2, f"{len(dirs)} on {chain['host']}")
            self.flow.set_done(2)
            self.flow.set_current(3)
            self._server_chain = None
            first = chain["stored"].get(chain["first"], (None, None))
            best = max(((x, d) for d, x in chain["stored"].values() if d and x is not None), default=(first[1], first[0]))
            from reco_engine import run_store as _rs
            n_ops = len((_rs.load_run(best[1]).get("program") or [])) if best[1] else 0
            self.result_bar.show_result(self._result_text(chain["part"], best[0], n_ops, chain["host"])
                                        + (f" ({len(dirs)} runs)" if len(dirs) > 1 else ""),
                                        [best[1]] + [d for d in dirs if d != best[1]] if best[1] else dirs)
            if dirs and str(chain["part"]) == str(self.settings_panel.current_path()):
                self.show_benchmark(select=dirs)

    def _follow_candidate(self, v, t, caption: str, op: str):
        """A followed server run's latest try: the primitive in the candidate layer (pink a cut, orange an add)."""
        from reco_gui.viewer_cascade import CUTTER
        colour = CUTTER if op == "cut" else (0.922, 0.408, 0.204)
        f = getattr(self, "_follow_run", None)
        self.view_cascade.show_layer("candidate", (v, t), colour, f"{f.caption if f else ''}\n{caption}")
        self._set_status(caption)

    def _follow_region(self, lo, hi, caption: str):
        """The leftover a followed server run works on: its box in the 'piece in work' layer (yellow)."""
        import numpy as np
        import trimesh
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        box = trimesh.creation.box(extents=np.maximum(hi - lo, 1e-6))
        box.apply_translation((lo + hi) / 2)
        self.view_cascade.show_layer("f_current", (box.vertices, box.faces))

    def _server_run_ended(self, mirror, why: str = "done"):
        """A followed server run ended: its files are fetched (a thread) and shown (reco_gui/server_result.py) - the
        exact model over the part, the record, the rating and the detail study in the detail panel."""
        mirror = Path(mirror)
        if str(mirror) in getattr(getattr(self, "server_runs", None), "pending", {}):
            return                                              # started here: ServerRuns fetches, stores and shows it
        if (mirror / "run" / "record.json").is_file() and (mirror / "run" / "model.stl").is_file():
            self.show_server_result(mirror)
            return
        if why != "done":
            return

        class Fetch(QtCore.QThread):
            done = QtCore.pyqtSignal(str)

            def run(self_t):
                try:
                    from reco_gui.server_result import fetch
                    fetch(mirror)
                    self_t.done.emit("")
                except Exception as exc:
                    self_t.done.emit(str(exc))
        self._log(f"--- fetching the server run's files (record, program, rating, model) ...")
        t = Fetch(self)
        t.done.connect(lambda err, m=mirror: self._log(f"--- fetch failed: {err}") if err else self.show_server_result(m))
        self._fetch_thread = t
        t.start()

    def show_server_result(self, mirror, load_mesh=None):
        """The results of a server run (fetched into <mirror>/run): the exact model over the part, the summary in the
        detail panel (record, program with the IoU after every operation, the rating, the detail study)."""
        from reco_gui.server_result import result_html
        html_text, info = result_html(mirror)
        if info["model_stl"]:
            try:
                if load_mesh is None:
                    from src.closest_point_field import load_mesh
                model = load_mesh(info["model_stl"])
                for layer in ("candidate", "f_current"):                # the run is over: no try, no piece in work
                    self.view_cascade.show_layer(layer, None)
                self.view_cascade.show_pair(getattr(self, "_follow_input", None), model, info["caption"])
            except Exception as exc:
                self._log(f"    the server run's model could not be read: {exc}")
        for line in info["lines"]:
            self._log(line)
        try:                                                    # in the run store once: Benchmark, Open, Compare
            from reco_gui.server_run import import_mirror
            stored = import_mirror(mirror)
            if stored is not None:
                info["stored"] = str(stored)
                self._log(f"    in the run store: {stored} (the Benchmark tab of its part)")
                self.benchmark.refresh()
                if getattr(self, "_server_chain", None) is None:      # a chain shows its bar at its end
                    self.result_bar.show_result(info["caption"] + " - compare it with the input or the part's other runs.",
                                                [str(stored)])
        except Exception as exc:
            self._log(f"    not stored in the run store: {exc}")
        self.follow_box.setChecked(False)
        self.act_details.setChecked(True)
        self.detail.setHtml(html_text)
        self._set_status(info["caption"])
        return info

    def open_run(self, run_dir, load_mesh=None):
        """Show a stored run: the model in the 3D views, the summary in the detail panel, its steps in the log."""
        from reco_gui.open_run import load_stored
        try:
            stored = load_stored(run_dir, load_mesh)
        except Exception as exc:
            self._set_status(f"Open failed: {exc}")
            return None
        self.preview.cancel()
        self._log(f"--- stored run opened: {stored['run']['path']}")
        for line in stored["log"]:
            self._log(line)
        if stored["note"]:
            self._log(f"    {stored['note']}")
        if stored["mesh"] is not None:
            import numpy as np
            v, f = np.asarray(stored["mesh"][0], float), np.asarray(stored["mesh"][1])
            for view in (self.view_field, self.view_cascade):
                view.show_input(v, f, stored["caption"])
        self.follow_box.setChecked(False)                       # a later step would replace the summary
        self.act_details.setChecked(True)
        self.detail.setHtml(stored["html"])
        self.saved_run = stored["run"]["path"]                  # Compare works on the opened run
        self._saved_states = None
        self._set_status(stored["caption"].replace("\n", "  |  ") + (f"  |  {stored['note']}" if stored["note"] else ""))
        self._update_buttons()
        return stored

    def show_benchmark(self, select=None):
        """The Benchmark tab of the bottom panel, on the current input, shown; `select`: these runs checked."""
        path = self.settings_panel.current_path()
        if str(path) and (self.benchmark.input_path is None or str(self.benchmark.input_path) != str(path)):
            self.benchmark.set_input(path)
        else:
            self.benchmark.refresh()
        if select:
            self.benchmark.set_checked(select)
        self.act_timeline.setChecked(True)
        self.bottom_tabs.setCurrentWidget(self.benchmark)

    def show_benchmark_run(self, run_dir, load_mesh=None):
        """A run of the benchmark (a stored run): its model, summary and log as Open shows them, its aegir rating in
        the detail panel, its program and the part's known tree in the Operations dock. Looked at, not continued."""
        from reco_engine import benchmark as bm
        from reco_gui.benchmark_panel import rating_html
        if self._busy:
            self._set_status("Benchmark: a run is computing - pause or wait first")
            return None
        stored = self.open_run(run_dir, load_mesh)
        if stored is None:
            return None
        run = stored["run"]
        rating = bm.read_rating(run["path"])
        if rating:
            self.detail.setHtml(stored["html"] + rating_html(rating))
        result = bm.result_of(run)
        truth = None
        tp = bm.truth_for((run.get("input") or {}).get("path") or "")
        if tp is not None:
            try:
                truth = json.loads(tp.read_text())
            except (OSError, ValueError):
                truth = None
        self.op_tree.show_benchmark(result.operations if result is not None else [], truth,
                                    f"{run.get('label') or run.get('name')}, path {bm.row_of(run)['variant']}")
        self.op_dock.show()
        self.op_dock.raise_()
        return stored

    def use_benchmark_run(self, run_dir):
        """The user's choice: the run is labelled "chosen" in the store and shown."""
        from reco_engine import benchmark as bm
        try:
            new_dir = bm.choose(run_dir)
        except Exception as exc:
            self._set_status(f"Benchmark: the choice could not be labelled: {exc}")
            new_dir = Path(run_dir)
        self._log(f"--- chosen result: {new_dir}")
        self.flow.set_note(3, f"chosen: {Path(new_dir).name}")
        self.flow.set_done(3)
        self.flow.set_current(4)
        self.benchmark.refresh()
        self.show_benchmark_run(new_dir)
        self._set_status(f"Chosen: {new_dir.name}  |  " + self.status_label.text())
        return new_dir

    def show_benchmark_input(self):
        """The part itself again, in the 3D views."""
        path = self.benchmark.input_path
        if path is not None and path.is_file():
            self.preview_input(str(path))

    def save_dialog(self):
        label, ok = QtWidgets.QInputDialog.getText(self, "Save the run", "Label (optional, short):")
        if ok:
            self.save_run(label.strip() or None)

    def save_run(self, label: Optional[str] = None):
        """Store the run as it is now (reco_engine.run_store.save_run); the directory, or None when it failed."""
        if self.session is None or not self.session.states or self.settings is None:
            return None
        from reco_engine import run_store
        if self.saved_run is not None and self._saved_states == len(self.session.states) and Path(self.saved_run).is_dir():
            if not label:                                        # saved as it is: a second copy says nothing new
                self._log(f"--- the run is saved already: {self.saved_run}")
                return self.saved_run
            try:
                self.saved_run = run_store.relabel(self.saved_run, label)
            except Exception as exc:
                self._log(f"--- the label was NOT set: {exc}")
                return None
            self._log(f"--- saved run labelled: {self.saved_run}")
            self._set_status(f"Saved: {self.saved_run}")
            return self.saved_run
        try:
            path = run_store.save_run(self.session, self.settings, self.settings.input_stl, label=label,
                                      screenshot=self._grab_view)
        except Exception as exc:
            self._log(f"--- the run was NOT saved: {exc}")
            self._set_status(f"Saving failed: {exc}")
            return None
        self.saved_run, self._saved_states = path, len(self.session.states)
        self._log(f"--- run saved: {path}" + ("" if self.session.finished else "  (not finished)"))
        self._set_status(f"{self._status_base}   Saved: {path}" if self.session.finished else f"Saved (not finished): {path}")
        self._update_buttons()
        return path

    def compare_with_previous(self):
        """The run saved last against the newest earlier run of the same input, in the detail panel."""
        if self.saved_run is None:
            return
        from reco_engine import run_store
        try:
            run = run_store.load_run(self.saved_run)
            before = run_store.previous_run(run)
            html = report.render_run_diff(run_store.diff_runs(before, run) if before is not None else None, run)
        except Exception as exc:
            html = f"<p style='color:#b3261e'>The comparison failed: {exc}</p>"
        self.follow_box.setChecked(False)                       # the next step would replace the comparison
        self.act_details.setChecked(True)
        self.detail.setHtml(html)

    def _log_timing_summary(self):
        """The run's time by section (reco_engine.timing), biggest first, into the log and a JSON file."""
        rec = getattr(self.session, "recorder", None)
        if rec is None or not rec.sections:
            return
        import json
        from reco_engine.timing import format_summary
        totals, elapsed = rec.summary(), rec.elapsed()
        self._log("--- where the time went (wall incl. sub-sections; self = without them; CPU of the whole process):")
        self._log(format_summary(totals, elapsed))
        try:
            out = Path(os.environ.get("SHAPE_RECO_TIMINGS_DIR") or Path.home() / ".cache" / "shape_reco" / "timings")
            out.mkdir(parents=True, exist_ok=True)
            stem = Path(str(getattr(self.settings, "input_stl", "run"))).stem
            path = out / f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}.json"
            path.write_text(json.dumps(dict(input=str(getattr(self.settings, "input_stl", "")), elapsed=elapsed,
                                            totals=[t.as_dict() for t in totals]), indent=1))
            self._log(f"    written {path}")
        except OSError as exc:
            self._log(f"    timing JSON not written: {exc}")

    def capture_output(self):
        """Show everything the run prints (stdout and stderr, e.g. the cascade's type-entry warnings) in the Output panel
        too; the terminal still gets it. Undone when the window closes."""
        if self._tees:
            return
        for name in ("stdout", "stderr"):
            tee = _OutputTee(getattr(sys, name))
            tee.line.connect(self.log.appendPlainText)      # not _log: the terminal has these lines already
            setattr(sys, name, tee)
            self._tees.append((name, tee))

    def _open_dataset_gen(self):
        if getattr(self, "_dataset_gen_window", None) is None:
            from reco_gui.dataset_gen_window import DatasetGenWindow
            self._dataset_gen_window = DatasetGenWindow()
        self._dataset_gen_window.show()
        self._dataset_gen_window.raise_()
        self._dataset_gen_window.activateWindow()

    def _say(self, text: str):
        """Into the log and, whatever `verbose` is, onto the terminal (closing: the window is about to go)."""
        self.log.appendPlainText(text)
        stream = sys.__stderr__
        if stream is not None:
            print(time.strftime("%H:%M:%S"), text, file=stream, flush=True)

    def closeEvent(self, event):
        for sig in (self.worker.stateReady, self.worker.status, self.worker.started, self.worker.idle, self.worker.failed,
                    self.worker.busy):
            try:
                sig.disconnect()                    # a closed window does not act on what the worker still reports
            except TypeError:
                pass
        self.heartbeat.stop()
        self.stats.stop_sampling()
        if getattr(self, "server_runs", None) is not None:      # the server runs go on there; only the polling stops
            self.server_runs.close()
        if getattr(self, "_follow_run", None) is not None:
            self._follow_run.stop()
        self.worker.pause()
        self.thread.quit()
        stuck = False
        if not self.thread.wait(int(self.CLOSE_WAIT_S * 1000)):
            # the worker is inside a long computation (a FreeCAD build): it sees the pause only at its next step
            self.hide()
            self._say(f"--- closing: the worker is still computing after {self.CLOSE_WAIT_S:.0f} s"
                      + (f" ({self.session.last.step.stage})" if getattr(self.session, "last", None) is not None else ""))
            ended = terminate_children()
            if ended:
                self._say(f"    ended the child processes: {', '.join(ended)}")
            stuck = not self.thread.wait(int(self.CLOSE_KILL_WAIT_S * 1000))
        for name, tee in self._tees:
            setattr(sys, name, tee.stream)
        self._tees = []
        if getattr(self, "_dataset_gen_window", None) is not None:
            self._dataset_gen_window.close()
        super().closeEvent(event)
        if stuck:                       # a computation of its own (no child to end): a process without a window otherwise
            self._say("    the worker thread does not return: ending the process")
            for stream in (sys.stdout, sys.stderr):
                try:
                    stream.flush()
                except Exception:
                    pass
            self.hard_exit(0)
