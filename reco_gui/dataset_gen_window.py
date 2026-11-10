#!/usr/bin/env python3
"""
The dataset-generation window: fill in the settings dock, press Generate, watch `src.dataset_generator` run (one
QProcess per shard, in parallel - see reco_gui/dataset_gen_setup.py's shard_jobs), and see the per-topology,
hollow/carve/cut counts once it is done (or at any point while it is still running).

    python3 -m reco_gui.dataset_gen_main
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from reco_gui.dataset_gen_panel import DatasetGenSettingsPanel
from reco_gui.dataset_gen_setup import ROOT, ShardJob, shard_jobs, summarize_output


class DatasetGenWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SHaPe ReCo - dataset generation")
        self.resize(1100, 800)
        self._processes: list = []          # (QProcess, ShardJob)
        self._output_dir: Path = None
        self._cancelled = False

        self.settings_panel = DatasetGenSettingsPanel()
        dock = QtWidgets.QDockWidget("Settings", self)
        dock.setWidget(self.settings_panel)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self.settings_panel.setMinimumWidth(420)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.progress = QtWidgets.QLabel("Idle.")
        layout.addWidget(self.progress)
        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(140)
        self.summary.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        layout.addWidget(self.summary)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        layout.addWidget(self.log, 1)
        self.setCentralWidget(central)

        tb = self.addToolBar("Run")
        tb.setMovable(False)
        self.act_go = tb.addAction("Generate")
        self.act_go.triggered.connect(self.generate)
        self.act_cancel = tb.addAction("Cancel")
        self.act_cancel.triggered.connect(self.cancel)
        self.act_cancel.setEnabled(False)
        self.act_refresh = tb.addAction("Refresh summary")
        self.act_refresh.triggered.connect(self._refresh_summary)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.act_close = tb.addAction(self.style().standardIcon(QtWidgets.QStyle.SP_DialogCloseButton), "Close")
        self.act_close.setShortcut("Ctrl+Q")
        self.act_close.setToolTip("Close the window (Ctrl+Q); running shards are cancelled")
        self.act_close.triggered.connect(self.close)

        self._poll = QtCore.QTimer(self)
        self._poll.setInterval(2000)
        self._poll.timeout.connect(self._refresh_summary)

    # ------------------------------------------------------------------------------------------ actions
    def generate(self):
        if self._processes:
            return
        try:
            settings = self.settings_panel.collect()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot start", str(exc))
            return
        jobs = shard_jobs(settings)
        self._output_dir = Path(settings.output_dir)
        self._cancelled = False
        self.log.clear()
        self.summary.clear()
        self._log(f"--- generating: mode={settings.mode} topologies={', '.join(settings.topologies)} "
                  f"shards={len(jobs)} output={self._output_dir}")
        python = sys.executable
        for job in jobs:
            proc = QtCore.QProcess(self)
            proc.setWorkingDirectory(str(ROOT / "dataset_gen"))
            proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            proc.readyReadStandardOutput.connect(lambda p=proc, label=job.label: self._on_output(p, label))
            proc.finished.connect(lambda code, status, label=job.label: self._on_finished(label, code, status))
            proc.errorOccurred.connect(lambda err, label=job.label: self._log(f"[{label}] process error: {err}"))
            self._log(f"[{job.label}] {python} -m src.dataset_generator {' '.join(job.args)}")
            proc.start(python, ["-m", "src.dataset_generator", *job.args])
            self._processes.append((proc, job))
        self.act_go.setEnabled(False)
        self.act_cancel.setEnabled(True)
        self.progress.setText(f"Running: 0 / {len(jobs)} shards done.")
        self._poll.start()

    def cancel(self):
        self._cancelled = True
        for proc, job in self._processes:
            if proc.state() != QtCore.QProcess.NotRunning:
                proc.kill()
        self.progress.setText("Cancelled.")
        self.act_go.setEnabled(True)
        self.act_cancel.setEnabled(False)

    # ------------------------------------------------------------------------------------------ signals
    def _on_output(self, proc, label):
        text = bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.splitlines():
            self._log(f"[{label}] {line}")

    def _on_finished(self, label, code, status):
        self._log(f"[{label}] finished (exit {code})")
        done = sum(1 for p, _ in self._processes if p.state() == QtCore.QProcess.NotRunning)
        if not self._cancelled:
            self.progress.setText(f"Running: {done} / {len(self._processes)} shards done.")
        self._refresh_summary()
        if done == len(self._processes):
            self._poll.stop()
            if self._cancelled:
                self._log("--- cancelled")
                return
            self.act_go.setEnabled(True)
            self.act_cancel.setEnabled(False)
            self.progress.setText(f"Done: {len(self._processes)} shard(s).")
            self._log("--- all shards finished")

    def _refresh_summary(self):
        if self._output_dir is None:
            return
        s = summarize_output(self._output_dir)
        lines = [f"total samples so far: {s['total']}", f"per topology: {s['per_topology']}"]
        if s["hollow"] or s["carve"] or s["cut"]:
            lines.append(f"hollow: {s['hollow']}   carve: {s['carve']}   cut: {s['cut']}")
        self.summary.setPlainText("\n".join(lines))

    def _log(self, text: str):
        self.log.appendPlainText(text)

    def closeEvent(self, event):
        self.cancel()
        super().closeEvent(event)
